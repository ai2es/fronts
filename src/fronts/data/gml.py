"""Parse TWC global surface-analysis GML files and rasterize their fronts onto a lat/lon grid.

Each file holds ``CONTOUR_LINE`` (isobars), ``PRESSURE_CENTER`` and ``FRONT`` features. Only
``FRONT`` features are used. ``FRONT`` coordinates are ``lon,lat`` (the isobars are ``lat,lon``),
so the order is validated on read rather than assumed.

The raster holds the *original* front codes consumed by ``constants.FRONT_CLASS_MAP`` (so the
eval pipeline's ``remap_fronts`` applies unchanged), with 0 for no front.
"""

import datetime
import logging
import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

# GML FRONT_TYPE string -> original front code (see constants.FRONT_CLASS_MAP). Types not listed
# here are counted and skipped, never guessed.
GML_FRONT_TYPE_CODES: dict[str, int] = {
    "Cold Front": 1,
    "Warm Front": 2,
    "Stationary Front": 3,
    "Occluded Front": 4,
    "Trough": 14,
    "Dry Line": 16,
}

# Where two features land on the same pixel the later write wins; writing in this order lets the
# sharper front types overwrite the broader trough-like ones. Codes absent here sort first.
_DRAW_ORDER = (15, 13, 14, 16, 3, 4, 2, 1)

# Analysts' polar troughs are occasionally extrapolated a degree or so past the pole (seen: 90.92). Up to
# this many degrees is clipped to the pole; beyond it the file is taken to be lat,lon ordered and rejected.
_POLE_TOLERANCE = 95.0

_TIME_RE = re.compile(r"\.(\d{8}T\d{6})Z\.")

LonLatLine = tuple[np.ndarray, np.ndarray]


def valid_time_from_path(path: str | Path) -> np.datetime64:
    """Reads the valid time out of a ``rec.sfcanalysis.YYYYMMDDTHHMMSSZ...`` file name.

    Args:
        path: GML file path.

    Returns:
        Valid time as ``datetime64[ns]``.

    Raises:
        ValueError: If the name carries no timestamp.
    """
    m = _TIME_RE.search(Path(path).name)
    if m is None:
        raise ValueError(f"No valid time in file name {Path(path).name!r}")
    return np.datetime64(datetime.datetime.strptime(m.group(1), "%Y%m%dT%H%M%S"), "ns")


def parse_fronts(path: str | Path) -> tuple[list[tuple[int, LonLatLine]], Counter]:
    """Extracts FRONT polylines from one GML file.

    Args:
        path: GML file path.

    Returns:
        Tuple of (fronts, unmapped). ``fronts`` is a list of ``(code, (lon, lat))`` with 1-D
        float arrays; ``unmapped`` counts FRONT_TYPE strings absent from ``GML_FRONT_TYPE_CODES``.

    Raises:
        ValueError: If a FRONT coordinate pair has a second value outside [-90, 90], which means the
            file is not ``lon,lat`` ordered as assumed.
    """
    root = ET.parse(path).getroot()
    fronts: list[tuple[int, LonLatLine]] = []
    unmapped: Counter = Counter()
    for feat in root.iter("FRONT"):
        ftype = (feat.findtext("FRONT_TYPE") or "").strip()
        code = GML_FRONT_TYPE_CODES.get(ftype)
        if code is None:
            unmapped[ftype] += 1
            continue
        for coords in feat.iter("{http://www.opengis.net/gml}coordinates"):
            pairs = np.array([p.split(",") for p in (coords.text or "").split()], dtype=np.float64)
            if pairs.ndim != 2 or len(pairs) < 2:
                continue
            if np.abs(pairs[:, 1]).max() > _POLE_TOLERANCE:
                raise ValueError(f"{path}: FRONT coordinates are not lon,lat ordered (|lat| > {_POLE_TOLERANCE})")
            fronts.append((code, (pairs[:, 0], np.clip(pairs[:, 1], -90.0, 90.0))))
    return fronts, unmapped


def _densify(lon: np.ndarray, lat: np.ndarray, step: float) -> LonLatLine:
    """Inserts points so consecutive points are at most ``step`` degrees apart; unwraps the antimeridian."""
    lon = np.rad2deg(np.unwrap(np.deg2rad(lon)))
    out_lon, out_lat = [lon[:1]], [lat[:1]]
    for i in range(len(lon) - 1):
        n = max(int(np.ceil(max(abs(lon[i + 1] - lon[i]), abs(lat[i + 1] - lat[i])) / step)), 1)
        t = np.linspace(0.0, 1.0, n + 1)[1:]
        out_lon.append(lon[i] + t * (lon[i + 1] - lon[i]))
        out_lat.append(lat[i] + t * (lat[i + 1] - lat[i]))
    return np.concatenate(out_lon), np.concatenate(out_lat)


def rasterize_fronts(fronts: Sequence[tuple[int, LonLatLine]], lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Draws front polylines onto a regular grid as original front codes.

    Args:
        fronts: ``(code, (lon, lat))`` polylines in degrees, lon in either -180..180 or 0..360.
        lats: Ascending, evenly spaced latitudes.
        lons: Ascending, evenly spaced longitudes covering 0..360 (e.g. 0..359.75).

    Returns:
        int8 array of shape (len(lats), len(lons)); 0 where no front passes.
    """
    res = float(lons[1] - lons[0])
    grid = np.zeros((len(lats), len(lons)), dtype=np.int8)
    for code, (lon, lat) in sorted(fronts, key=lambda f: _DRAW_ORDER.index(f[0]) if f[0] in _DRAW_ORDER else -1):
        d_lon, d_lat = _densify(lon, lat, res / 2)
        i_lat = np.rint((d_lat - lats[0]) / res).astype(int)
        i_lon = np.rint((d_lon % 360.0 - lons[0]) / res).astype(int) % len(lons)
        keep = (i_lat >= 0) & (i_lat < len(lats))
        grid[i_lat[keep], i_lon[keep]] = code
    return grid


def rasterize_file(path: str | Path, lats: np.ndarray, lons: np.ndarray) -> tuple[np.datetime64, np.ndarray, Counter]:
    """Parses and rasterizes one GML file.

    Returns:
        Tuple of (valid_time, grid, unmapped) as produced by ``valid_time_from_path``,
        ``rasterize_fronts`` and ``parse_fronts``.
    """
    fronts, unmapped = parse_fronts(path)
    return valid_time_from_path(path), rasterize_fronts(fronts, lats, lons), unmapped


def make_front_dataset(times: Sequence[np.datetime64], grids: Sequence[np.ndarray], lats, lons) -> xr.Dataset:
    """Packs rasterized grids into a Dataset with the ``identifier`` variable the eval stores use."""
    return xr.Dataset(
        {"identifier": (("time", "latitude", "longitude"), np.stack(grids))},
        coords={"time": np.asarray(times, dtype="datetime64[ns]"), "latitude": lats, "longitude": lons},
    )
