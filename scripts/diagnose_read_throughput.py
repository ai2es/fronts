"""Diagnose ERA5 and fronts icechunk store read throughput.

Run on the machine where the stores are colocated:

    pixi run python scripts/diagnose_read_throughput.py [config_path]

``config_path`` defaults to ``configs/schooner_train.yaml`` (the ``/ourdisk`` stores);
pass ``configs/schooner_train_scratch.yaml`` to profile the ``/scratch``-staged copy instead,
so the two runs are directly comparable.

Prints per-variable dtype/chunking/compression, then times reads at the raw
store level and through ``inputs_ds_to_dataarray`` so the bottleneck (chunk layout,
dtype, or the stack/transpose graph) can be localized. Each store is timed both with
sequential timesteps and with randomly scattered timesteps (matching ``FrontsPyDataset``'s
shuffled access pattern during training), since the fronts store is virtual — every
timestep read resolves to opening a netcdf file via ``virtual_chunk_local_path`` — and
scattered reads are where virtual-chunk overhead is expected to show up.
"""

import sys
import time

import numpy as np
import xarray as xr

from fronts import utils
from fronts.data import datasets, inputs


def _fmt_bytes(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def _time_read(sub: xr.DataArray, label: str) -> None:
    nbytes = sub.nbytes
    t0 = time.time()
    sub.compute()
    dt = time.time() - t0
    print(f"  {label:26s}: {_fmt_bytes(nbytes)} in {dt:6.1f} s -> {nbytes / 1e6 / max(dt, 1e-9):7.1f} MB/s")


def _time_sequential_and_random(da: xr.DataArray, n_time: int, sizes: tuple[int, ...], seed: int = 0) -> None:
    for k in sizes:
        _time_read(da.isel(time=slice(0, k)), f"{k:3d} steps sequential")
    rng = np.random.default_rng(seed)
    for k in sizes:
        idx = rng.choice(n_time, size=min(k, n_time), replace=False)
        _time_read(da.isel(time=idx), f"{k:3d} steps random")


def _profile_inputs(data_cfg: datasets.DatasetConfig, sizes: tuple[int, ...]) -> None:
    era5_cfg = data_cfg.inputs_icechunk_config
    ds = utils.open_readonly_icechunk_store(
        store_path=era5_cfg.store_path,
        branch=era5_cfg.branch_name,
        group=era5_cfg.group_name,
        zarr_format=era5_cfg.zarr_format,
        virtual_chunk_local_path=era5_cfg.virtual_chunk_local_path,
    )

    print("=== Per-variable layout (ERA5 inputs) ===")
    for name, var in ds.data_vars.items():
        enc = var.encoding
        print(
            f"{name:35s} dtype={var.dtype!s:8s} shape={var.shape} "
            f"chunks={enc.get('chunks')} compressor={enc.get('compressors', enc.get('compressor'))}"
        )

    n_time = ds.sizes["time"]
    print(f"\nTotal timesteps in inputs store: {n_time}")

    sample_var = next(iter(ds.data_vars))
    print(f"\n=== Raw single-variable read timing ('{sample_var}') ===")
    _time_sequential_and_random(ds[sample_var], n_time, sizes)

    print("\n=== Full-stack read timing (inputs_ds_to_dataarray) ===")
    da = inputs.inputs_ds_to_dataarray(ds, data_cfg.variables)
    print(f"  assembled dtype={da.dtype} shape={da.shape} chunks={da.chunks}")
    _time_sequential_and_random(da, n_time, sizes)


def _profile_targets(fronts_cfg: utils.IcechunkStorageConfig, sizes: tuple[int, ...]) -> None:
    ds = utils.open_readonly_icechunk_store(
        store_path=fronts_cfg.store_path,
        branch=fronts_cfg.branch_name,
        group=fronts_cfg.group_name,
        zarr_format=fronts_cfg.zarr_format,
        virtual_chunk_local_path=fronts_cfg.virtual_chunk_local_path,
    )
    target_da = ds["identifier"]

    print("\n\n=== Per-variable layout (fronts targets, virtual chunks) ===")
    enc = target_da.encoding
    print(
        f"{'identifier':35s} dtype={target_da.dtype!s:8s} shape={target_da.shape} "
        f"chunks={enc.get('chunks')} compressor={enc.get('compressors', enc.get('compressor'))}"
    )

    n_time = ds.sizes["time"]
    print(f"\nTotal timesteps in targets store: {n_time}")

    print("\n=== Raw target read timing ('identifier') ===")
    _time_sequential_and_random(target_da, n_time, sizes)


def main() -> None:
    """Print store layout and time reads to localize the throughput bottleneck."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/schooner_train.yaml"
    print(f"Config: {config_path}\n")
    yaml_data = utils.load_yaml(config_path)
    data_cfg = utils.parse_config_section(yaml_data, datasets.DatasetConfig, "data_config")
    sizes = (1, 10, 50)

    _profile_inputs(data_cfg, sizes)
    _profile_targets(data_cfg.targets_icechunk_config, sizes)


if __name__ == "__main__":
    main()
