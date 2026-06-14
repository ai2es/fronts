"""Rewrite the ERA5 icechunk store with uniform float32 chunking plus land_sea_mask.

One-off remediation. The original store mixed two incompatible chunkings
(``(125, 1, 40, 240)`` small-tile vs ``(24, 6, 320, 960)`` full-spatial) and
stored four derived variables as float64. When ``era5_to_dataarray`` stacks the
variables together, dask cannot find a common chunking and falls back to a
pathological rechunk, yielding ~17 MB/s reads. It was also missing the
``land_sea_mask`` channel the training config expects (77 channels instead of 78).

This reads every variable from the existing store, casts data variables to
float32, sources the static ``land_sea_mask`` field from the original ERA5
source with the same spatial subsetting, applies one uniform
``(time, level, latitude, longitude)`` chunking, and overwrites the era5 group
in place in a single commit (dask streams the chunks, so peak memory stays
bounded). Icechunk versioning keeps the previous data reachable in the prior
snapshot, so the in-place overwrite is reversible.

Usage (on the machine holding the store):
    pixi run python scripts/rechunk_era5_store.py \
        --config configs/generate_icechunk.yaml \
        --time-chunk 32

Afterwards, verify layout with scripts/diagnose_read_throughput.py.
"""

import argparse
import dataclasses
import logging

import dask
import icechunk as ic
import icechunk.xarray
import numpy as np
import xarray as xr

from fronts import utils
from fronts.data import config, generate

logger = logging.getLogger(__name__)


def load_land_sea_mask(era5_config: config.ERA5DataLoaderConfig, template: xr.Dataset) -> xr.DataArray:
    """Return the static land_sea_mask as a float32 (latitude, longitude) field.

    Sources the mask from the configured ERA5 source with the same spatial
    subsetting used to build the store, collapses any time/level dims to a
    single static field, and aligns its spatial coordinates to ``template``.

    Args:
        era5_config: Generation config providing the ERA5 source and bounding box.
        template: Existing store dataset whose latitude/longitude the mask aligns to.

    Returns:
        DataArray of dims (latitude, longitude), dtype float32.
    """
    mask_config = dataclasses.replace(era5_config, variables=["land_sea_mask"])
    ds = generate.generate_era5_download_data(mask_config)
    lsm = ds["land_sea_mask"]
    if "time" in lsm.dims:
        lsm = lsm.isel(time=0, drop=True)
    if "level" in lsm.dims:
        lsm = lsm.isel(level=0, drop=True)
    lsm = lsm.reindex(
        latitude=template["latitude"],
        longitude=template["longitude"],
        method="nearest",
        tolerance=1e-4,
    )
    if lsm.isnull().any().compute().item():
        raise ValueError("land_sea_mask has NaNs after aligning to the store grid; check coordinate overlap.")
    return lsm.astype(np.float32)


def rechunk_store(
    storage_config: config.IcechunkStorageConfig,
    era5_config: config.ERA5DataLoaderConfig,
    time_chunk: int,
    num_workers: int,
) -> None:
    """Cast the store's variables to float32, add land_sea_mask, and overwrite in place.

    Reads the era5 group from a snapshot-pinned read-only session while writing
    the rechunked group through a separate writable session, so source chunks
    remain available throughout the streamed overwrite.

    Args:
        storage_config: Storage config for the store to read from and overwrite.
        era5_config: Generation config used to source land_sea_mask.
        time_chunk: Chunk size along the time dimension; level/lat/lon are whole.
        num_workers: Dask threads for the streaming write.
    """
    ds = utils.open_readonly_icechunk_store(
        storage_config.store_path,
        storage_config.branch_name,
        group=storage_config.group_name,
        zarr_format=storage_config.zarr_format,
    )
    logger.info("Source store: %d variables, sizes %s", len(ds.data_vars), dict(ds.sizes))

    cast = {name: ds[name].astype(np.float32) for name in ds.data_vars}
    out = xr.Dataset(cast, coords=ds.coords)
    out["land_sea_mask"] = load_land_sea_mask(era5_config, ds)

    chunking = {
        "time": time_chunk,
        "level": ds.sizes["level"],
        "latitude": ds.sizes["latitude"],
        "longitude": ds.sizes["longitude"],
    }
    out = out.drop_encoding().chunk({dim: size for dim, size in chunking.items() if dim in out.dims})
    logger.info(
        "Overwriting era5 group in %s in place with chunking %s (%d variables)",
        storage_config.store_path,
        chunking,
        len(out.data_vars),
    )

    storage = ic.local_filesystem_storage(storage_config.store_path)
    repo = ic.Repository.open(storage)
    session = repo.writable_session(storage_config.branch_name)
    with dask.config.set(scheduler="threads", num_workers=num_workers):
        icechunk.xarray.to_icechunk(out, session, group=storage_config.group_name, mode="w", safe_chunks=False)
    snapshot_id = session.commit(storage_config.commit_message)
    logger.info("Committed rechunked store in place as snapshot %s", snapshot_id)


def main():
    """Parse arguments and rewrite the ERA5 store with uniform float32 chunking."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Rechunk the ERA5 icechunk store to uniform float32 layout")
    parser.add_argument("--config", required=True, help="Path to the generation YAML config")
    parser.add_argument("--time-chunk", type=int, default=32, help="Chunk size along time (default: 32)")
    parser.add_argument("--num-workers", type=int, default=16, help="Dask threads for the write (default: 16)")
    args = parser.parse_args()

    era5_config = utils.open_config_yaml_as_dataclass(
        args.config, config.ERA5DataLoaderConfig, config_key="era5_config"
    )
    storage_config = utils.open_config_yaml_as_dataclass(
        args.config, config.IcechunkStorageConfig, config_key="icechunk_storage_config"
    )
    storage_config = dataclasses.replace(
        storage_config,
        commit_message="Rechunk ERA5 to uniform float32 (time, level, lat, lon) with land_sea_mask",
    )

    rechunk_store(storage_config, era5_config, args.time_chunk, args.num_workers)
    logger.info("Done. Verify throughput with scripts/diagnose_read_throughput.py before training.")


if __name__ == "__main__":
    main()
