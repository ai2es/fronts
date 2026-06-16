"""Rewrite the ERA5 icechunk store with uniform float32 chunking plus land_sea_mask.

One-off remediation. The original store mixed two incompatible chunkings
(``(125, 1, 40, 240)`` small-tile vs ``(24, 6, 320, 960)`` full-spatial) and
stored four derived variables as float64. When ``era5_to_dataarray`` stacks the
variables together, dask cannot find a common chunking and falls back to a
pathological rechunk, yielding ~17 MB/s reads. It was also missing the
``land_sea_mask`` channel the training config expects (77 channels instead of
78), because the mask is stored as a coordinate rather than a data variable.

This reads every variable from the existing store, casts data variables to
float32, promotes the static ``land_sea_mask`` coordinate to a data variable so
it becomes the 78th channel, applies one uniform
``(time, level, latitude, longitude)`` chunking, and overwrites the era5 group
in place in a single commit (dask streams the chunks, so peak memory stays
bounded). Icechunk versioning keeps the previous data reachable in the prior
snapshot, so the in-place overwrite is reversible.

Usage (on the machine holding the store):
    pixi run python scripts/rechunk_era5_store.py \
        --config configs/generate_icechunk.yaml \
        --time-chunk 1

Add ``--slurm`` to drive the write from a dask-jobqueue SLURM cluster (uses the
``slurm_config`` block in the YAML) and expose a Dask dashboard:
    pixi run python scripts/rechunk_era5_store.py \
        --config configs/generate_icechunk.yaml \
        --time-chunk 1 \
        --slurm

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
from fronts.data import config
from fronts.data.generate import create_dask_client

logger = logging.getLogger(__name__)

_MASK_NAME = "land_sea_mask"


def extract_land_sea_mask(ds: xr.Dataset) -> xr.DataArray:
    """Return the store's land_sea_mask coordinate as a static float32 field.

    The mask is stored as a coordinate rather than a data variable, so it is
    absent from the channel stack. This reads it from the store, collapses any
    time/level dims to a single static field, and returns it as a data variable.

    Args:
        ds: Existing store dataset containing a ``land_sea_mask`` coordinate.

    Returns:
        DataArray named ``land_sea_mask`` with dims (latitude, longitude), float32.

    Raises:
        KeyError: If the store has no ``land_sea_mask`` coordinate or variable.
    """
    if _MASK_NAME not in ds.variables:
        raise KeyError(f"'{_MASK_NAME}' not found in the store's variables or coordinates: {list(ds.variables)}")
    lsm = ds[_MASK_NAME]
    if "time" in lsm.dims:
        lsm = lsm.isel(time=0, drop=True)
    if "level" in lsm.dims:
        lsm = lsm.isel(level=0, drop=True)
    return lsm.astype(np.float32)


def rechunk_store(
    storage_config: config.IcechunkStorageConfig,
    time_chunk: int,
    num_workers: int,
    distributed: bool = False,
) -> None:
    """Cast the store's variables to float32, promote land_sea_mask, and overwrite in place.

    Reads the era5 group from a snapshot-pinned read-only session while writing
    the rechunked group through a separate writable session, so source chunks
    remain available throughout the streamed overwrite.

    Args:
        storage_config: Storage config for the store to read from and overwrite.
        time_chunk: Chunk size along the time dimension; level/lat/lon are whole.
        num_workers: Dask threads for the local streaming write. Ignored when
            ``distributed`` is True, where the active distributed client drives
            the write instead.
        distributed: If True, write under the active dask.distributed client
            (a SLURM cluster) rather than the local threaded scheduler. The
            distributed session fork/merge is handled inside ``to_icechunk``.
    """
    ds = utils.open_readonly_icechunk_store(
        storage_config.store_path,
        storage_config.branch_name,
        group=storage_config.group_name,
        zarr_format=storage_config.zarr_format,
    )
    logger.info("Source store: %d variables, sizes %s", len(ds.data_vars), dict(ds.sizes))

    mask = extract_land_sea_mask(ds)
    cast = {name: ds[name].astype(np.float32) for name in ds.data_vars}
    coords = {name: ds.coords[name] for name in ds.coords if name != _MASK_NAME}
    out = xr.Dataset(cast, coords=coords)
    out[_MASK_NAME] = mask

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
    if distributed:
        icechunk.xarray.to_icechunk(out, session, group=storage_config.group_name, mode="w", safe_chunks=False)
    else:
        with dask.config.set(scheduler="threads", num_workers=num_workers):
            icechunk.xarray.to_icechunk(out, session, group=storage_config.group_name, mode="w", safe_chunks=False)
    snapshot_id = session.commit(storage_config.commit_message)
    logger.info("Committed rechunked store in place as snapshot %s", snapshot_id)


def main():
    """Parse arguments and rewrite the ERA5 store with uniform float32 chunking."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Rechunk the ERA5 icechunk store to uniform float32 layout")
    parser.add_argument("--config", required=True, help="Path to the generation YAML config")
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=1,
        help=(
            "Chunk size along time (default: 1). Training gathers scattered "
            "timesteps, so a large time chunk forces dask to copy the whole "
            "chunk per touched timestep; keep this at 1 unless reads are "
            "dominated by per-chunk overhead."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=16, help="Dask threads for the local write (default: 16)")
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Launch a dask-jobqueue SLURM cluster (with dashboard) to drive the write",
    )
    args = parser.parse_args()

    storage_config = utils.open_config_yaml_as_dataclass(
        args.config, config.IcechunkStorageConfig, config_key="icechunk_storage_config"
    )
    storage_config = dataclasses.replace(
        storage_config,
        commit_message="Rechunk ERA5 to uniform float32 (time, level, lat, lon) with land_sea_mask",
    )

    client = None
    if args.slurm:
        slurm_config = utils.open_config_yaml_as_dataclass(args.config, config.SlurmConfig, config_key="slurm_config")
        logger.info(f"SLURM config loaded: {slurm_config}")
        client = create_dask_client(slurm_config)
        logger.info(f"Dask client started: {client.dashboard_link}")

    try:
        rechunk_store(storage_config, args.time_chunk, args.num_workers, distributed=args.slurm)
    finally:
        if client is not None:
            client.close()
    logger.info("Done. Verify throughput with scripts/diagnose_read_throughput.py before training.")


if __name__ == "__main__":
    main()
