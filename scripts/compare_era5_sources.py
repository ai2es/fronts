"""Measure memory usage when writing a raw ERA5 subset to icechunk with chunks=None.

This is a throwaway diagnostic for the "non-dask, write_eager" path discussed for
Phase 1 of the ERA5 generation pipeline: open the source zarr store with
chunks=None (no dask graph), subset by variable/time/space, then write directly
via icechunk.xarray.to_icechunk. A background thread samples RSS memory while the
write runs so you can see how it grows relative to the subset's uncompressed size.

Usage:
    pixi run -e data python scripts/memory_check_raw_write.py \
        --era5-uri gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3 \
        --time-start 2019-01-01 --time-end 2019-01-31 \
        --variables temperature u_component_of_wind

For a quick local run without network access, point --era5-uri at one of the
small synthetic zarr stores under tests/data/ (see tests/data/conftest.py).
"""

import argparse
import asyncio
import contextlib
import shutil
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Iterable

import icechunk as ic
import icechunk.xarray
import pandas as pd
import psutil
import xarray as xr
import zarr
import zarr.core.codec_pipeline
from arraylake import Client as ArraylakeClient
from tqdm.asyncio import tqdm as tqdm_asyncio

from fronts import utils

ARCO_ARRAYLAKE_MAPPING = {
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
}

ARCO_ARRAYLAKE_COORD_MAPPING = {
    "time": "valid_time",
    "level": "pressure_level",
}


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2


class _MemorySampler:
    """Polls RSS in a background thread and tracks the peak."""

    def __init__(self, interval_seconds: float = 0.25) -> None:
        self._interval = interval_seconds
        self._peak_mb = _rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_MemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._peak_mb = max(self._peak_mb, _rss_mb())
            time.sleep(self._interval)

    @property
    def peak_mb(self) -> float:
        return self._peak_mb


@contextlib.contextmanager
def chunk_progress_bar():
    """Show a tqdm bar over zarr's async chunk read/write batches.

    zarr's codec pipeline gates concurrent chunk I/O behind
    zarr.core.common.concurrent_map (an asyncio.gather bounded by
    async.concurrency). This temporarily replaces the binding used by
    zarr.core.codec_pipeline with an equivalent that reports progress via
    tqdm.asyncio, so a long single-call write_eager() write shows live
    per-chunk-batch progress instead of running silently.
    """
    original = zarr.core.codec_pipeline.concurrent_map

    async def patched(items: Iterable[tuple], func: Callable[..., Awaitable], limit: int | None = None) -> list:
        items = list(items)
        if limit is None:
            return await tqdm_asyncio.gather(*(func(*item) for item in items), desc="chunk batches", leave=False)
        sem = asyncio.Semaphore(limit)

        async def run(item: tuple):
            async with sem:
                return await func(*item)

        return await tqdm_asyncio.gather(*(run(item) for item in items), desc="chunk batches", leave=False)

    zarr.core.codec_pipeline.concurrent_map = patched
    try:
        yield
    finally:
        zarr.core.codec_pipeline.concurrent_map = original


def open_arco_subset(variables: list[str], time_range, pressure_levels) -> xr.Dataset:
    """Open and subset the Google ARCO ERA5 Zarr store.

    Open and subset the store to the specified variables, time range, pressure levels, and spatial domain.

    Returns:
        An xarray Dataset containing the subset of ARCO ERA5 data.
    """
    era5_uri = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
    ds = xr.open_zarr(era5_uri, chunks=None, storage_options={"token": "anon"})
    return ds[variables].sel(time=time_range, level=pressure_levels)


def open_arraylake_subset(variables: list[str], time_range, pressure_levels) -> xr.Dataset:
    """Open and subset the Earthmover Arraylake ERA5 Icechunk store.

    Open and subset the store to the specified variables, time range, pressure levels, and spatial domain.

    Returns:
        An xarray Dataset containing the subset of Arraylake ERA5 data.
    """
    # Earthmover's Arraylake ERA5 Zarr store, which is publicly accessible with an anonymous token.
    arraylake_repo = "earthmover-public/era5"

    # The group within the Arraylake repo where the pressure level ERA5 data is stored chunked pancake-style
    group = "pressure/spatial"

    client = ArraylakeClient()
    repo = client.get_repo(arraylake_repo)
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, group=group, chunks=None)
    return ds[variables].sel(valid_time=time_range, pressure_level=pressure_levels)


def main() -> None:
    """Run the memory check described in this module's docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--era5-uri",
        default="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    )
    parser.add_argument(
        "--arraylake", action="store_true", help="Use the Earthmover Arraylake ERA5 store instead of ARCO"
    )
    parser.add_argument("--time-start", default="2019-01-01")
    parser.add_argument("--time-end", default="2019-01-02")
    parser.add_argument("--time-resolution", default="6h")
    parser.add_argument("--variables", nargs="+", default=["temperature"])
    parser.add_argument("--pressure-levels", nargs="+", type=int, default=[1000, 850, 500])
    parser.add_argument("--lat-min", type=float, default=25.0)
    parser.add_argument("--lat-max", type=float, default=45.0)
    parser.add_argument("--lon-min", type=float, default=-110.0)
    parser.add_argument("--lon-max", type=float, default=-70.0)
    parser.add_argument("--store-path", default=None, help="Reuse an existing dir instead of a temp dir")
    parser.add_argument("--keep-store", action="store_true", help="Don't delete the icechunk store afterward")
    parser.add_argument(
        "--zarr-async-concurrency",
        type=int,
        default=None,
        help="Override zarr's async.concurrency (max in-flight chunk requests per store). "
        "Default is zarr's built-in default (10).",
    )
    parser.add_argument(
        "--chunk-progress",
        action="store_true",
        help="Show a tqdm bar over zarr's async chunk read/write batches during the write.",
    )
    args = parser.parse_args()

    if args.zarr_async_concurrency is not None:
        zarr.config.set({"async.concurrency": args.zarr_async_concurrency})
        print(f"Set zarr async.concurrency to {args.zarr_async_concurrency}")

    storage_options = {"token": "anon"} if args.era5_uri.startswith("gs://") else None

    print(f"RSS at start:        {_rss_mb():.1f} MB")

    open_kwargs: dict = {"chunks": None}
    if storage_options:
        open_kwargs["storage_options"] = storage_options
    if args.arraylake:
        args.variables = [ARCO_ARRAYLAKE_MAPPING.get(var, var) for var in args.variables]
        ds = open_arraylake_subset(
            args.variables,
            pd.date_range(args.time_start, args.time_end, freq=args.time_resolution),
            args.pressure_levels,
        )
    else:
        ds = open_arco_subset(
            args.variables,
            pd.date_range(args.time_start, args.time_end, freq=args.time_resolution),
            args.pressure_levels,
        )
    print(f"RSS after open:      {_rss_mb():.1f} MB  (dask-backed: {bool(ds.chunks)})")

    times = pd.date_range(args.time_start, args.time_end, freq=args.time_resolution)
    coords = {
        "latitude": slice(args.lat_max, args.lat_min),
        "longitude": slice(args.lon_min, args.lon_max),
        "time": times,
        "level": args.pressure_levels,
    }
    if args.arraylake:
        coords = {ARCO_ARRAYLAKE_COORD_MAPPING.get(coord, coord): sel for coord, sel in coords.items()}
    ds_subset = utils.attach_periodic_lon_index(ds[args.variables]).sel(coords)
    print(f"RSS after subset:    {_rss_mb():.1f} MB  (dask-backed: {bool(ds_subset.chunks)})")
    print(f"Subset shape:        {dict(ds_subset.sizes)}")
    print(f"Subset uncompressed: {ds_subset.nbytes / 1024**2:.1f} MB")

    store_path = args.store_path or tempfile.mkdtemp(prefix="icechunk_memtest_")
    print(f"Writing to store:    {store_path}")

    storage = ic.local_filesystem_storage(store_path)
    repo = ic.Repository.open_or_create(storage)
    session = repo.writable_session("main")

    ds_subset = ds_subset.drop_encoding()
    progress_ctx = chunk_progress_bar() if args.chunk_progress else contextlib.nullcontext()
    with _MemorySampler() as sampler, progress_ctx:
        start = time.monotonic()
        icechunk.xarray.to_icechunk(ds_subset, session, safe_chunks=False)
        elapsed = time.monotonic() - start

    session.commit("memory check write")
    print(f"RSS after write:     {_rss_mb():.1f} MB")
    print(f"Peak RSS during write: {sampler.peak_mb:.1f} MB")
    print(f"Write took {elapsed:.1f}s")

    if not args.keep_store and not args.store_path:
        shutil.rmtree(store_path, ignore_errors=True)
    else:
        print(f"Store left at {store_path}")


if __name__ == "__main__":
    main()
