"""Rasterize the extracted TWC global-analysis GML files into one zarr of front codes.

Usage:
    python scripts/gml_to_zarr.py --gml_dir /ourdisk/hpc/ai2es/tman/twc_fronts \
        --out /ourdisk/hpc/ai2es/tman/twc_fronts_global.zarr --workers 16
"""

import argparse
import logging
import multiprocessing
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np

from fronts.data import gml

log = logging.getLogger(__name__)

BATCH = 256


def _rasterize(path: Path, lats: np.ndarray, lons: np.ndarray):
    try:
        return gml.rasterize_file(path, lats, lons)
    except Exception as exc:  # truncated/corrupt files exist at the tail of the archive
        log.warning("Skipping %s: %s", path, exc)
        return None


def main() -> None:
    """Rasterizes every GML file under ``--gml_dir`` into one zarr, appending in batches."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gml_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lat_min", type=float, default=-80.0)
    parser.add_argument("--lat_max", type=float, default=80.0)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    lats = np.arange(args.lat_min, args.lat_max + args.resolution / 2, args.resolution)
    lons = np.arange(0.0, 360.0, args.resolution)
    paths = sorted(args.gml_dir.glob("*/*/*.gml"))
    log.info("%d GML files → %d x %d grid", len(paths), len(lats), len(lons))

    unmapped: Counter = Counter()
    skipped, first = 0, True
    with multiprocessing.Pool(args.workers) as pool:
        for start in range(0, len(paths), BATCH):
            results = pool.map(partial(_rasterize, lats=lats, lons=lons), paths[start : start + BATCH])
            ok = [r for r in results if r is not None]
            skipped += len(results) - len(ok)
            for _, _, u in ok:
                unmapped.update(u)
            if not ok:
                continue
            ds = gml.make_front_dataset([r[0] for r in ok], [r[1] for r in ok], lats, lons)
            ds = ds.chunk({"time": 1})  # skipped files make the length arbitrary, so chunks must not span timesteps
            if first:
                ds.to_zarr(args.out, mode="w")
                first = False
            else:
                ds.to_zarr(args.out, append_dim="time")
            log.info("%d / %d files written", start + len(results), len(paths))
    log.info("Skipped %d unreadable files; unmapped FRONT_TYPEs: %s", skipped, dict(unmapped))
    log.info("Done → %s", args.out)


if __name__ == "__main__":
    main()
