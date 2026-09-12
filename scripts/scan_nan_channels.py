"""Scan an icechunk ERA5 store for NaN values, per variable, level, and timestep.

Diagnostic for NaN normalization statistics during training: locates which
variables contain NaNs, at which pressure levels and timesteps, and how many
grid points are affected, to distinguish a rare data corruption from a
systematic issue (e.g. derived variables computed from zero or negative
specific humidity).

Usage (on the machine holding the store):
    pixi run python scripts/scan_nan_channels.py \
        --store-path /ourdisk/hpc/ai2es/tman/data/fronts/train \
        --branch main --group era5

    Optionally restrict to suspect variables:
        --variables dewpoint_temperature equivalent_potential_temperature

Also reports, for specific_humidity, how many grid points are <= 0 — the
suspected root cause of NaNs in the humidity-derived variables.
"""

import argparse
import logging

import dask
import xarray as xr

from fronts import utils

logger = logging.getLogger(__name__)


def scan_variable_nans(da: xr.DataArray) -> xr.DataArray:
    """Count NaN grid points per timestep (and level, if present).

    Args:
        da: DataArray with a time dimension and spatial dims latitude/longitude.

    Returns:
        Integer DataArray of NaN counts with dims (time,) or (time, level).
    """
    return da.isnull().sum(dim=["latitude", "longitude"]).compute()


def report_variable(name: str, nan_counts: xr.DataArray, n_spatial: int) -> None:
    """Log a summary of where a variable's NaNs occur.

    Args:
        name: Variable name for log lines.
        nan_counts: Output of scan_variable_nans.
        n_spatial: Number of grid points per (time, level) slice.
    """
    total = int(nan_counts.sum())
    if total == 0:
        logger.info("%s: clean", name)
        return
    affected = nan_counts.where(nan_counts > 0, drop=True)
    n_times = int((nan_counts > 0).any([d for d in nan_counts.dims if d != "time"]).sum())
    n_total_times = nan_counts.sizes["time"]
    logger.warning(
        "%s: %d NaN points across %d/%d timesteps (max %d/%d points in one slice)",
        name,
        total,
        n_times,
        n_total_times,
        int(nan_counts.max()),
        n_spatial,
    )
    if "level" in nan_counts.dims:
        per_level = nan_counts.sum(dim="time")
        for level, count in zip(per_level.level.values, per_level.values, strict=True):
            if count > 0:
                logger.warning("  level %d: %d NaN points", int(level), int(count))
    first_times = affected.time.values[:5]
    logger.warning("  first affected timesteps: %s", list(first_times))


def main():
    """Open the store, scan requested variables, and log NaN locations."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Scan icechunk ERA5 store for NaNs")
    parser.add_argument("--store-path", required=True)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--group", default="era5")
    parser.add_argument("--variables", nargs="+", default=None, help="Subset of variables to scan (default: all)")
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    ds = utils.open_readonly_icechunk_store(store_path=args.store_path, branch=args.branch, group=args.group)
    variables = args.variables or list(ds.data_vars)
    n_spatial = ds.sizes["latitude"] * ds.sizes["longitude"]

    with dask.config.set(scheduler="threads", num_workers=args.num_workers):
        for name in variables:
            report_variable(name, scan_variable_nans(ds[name]), n_spatial)

        if "specific_humidity" in ds.data_vars:
            q = ds["specific_humidity"]
            nonpositive = (q <= 0).sum(dim=["latitude", "longitude"]).compute()
            total = int(nonpositive.sum())
            n_times = int((nonpositive > 0).any("level").sum()) if "level" in q.dims else int((nonpositive > 0).sum())
            logger.info("specific_humidity <= 0: %d points across %d timesteps", total, n_times)
            if "level" in q.dims and total > 0:
                per_level = nonpositive.sum(dim="time")
                for level, count in zip(per_level.level.values, per_level.values, strict=True):
                    logger.info("  level %d: %d non-positive points", int(level), int(count))
            negative = int((q < 0).sum().compute())
            logger.info("specific_humidity < 0 (strictly negative): %d points", negative)


if __name__ == "__main__":
    main()
