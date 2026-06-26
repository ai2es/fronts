"""Diagnose ERA5 icechunk store read throughput.

Run on the machine where the store is colocated:

    pixi run python scripts/diagnose_read_throughput.py

Prints per-variable dtype/chunking/compression, then times reads at the raw
store level and through ``inputs_ds_to_dataarray`` so the bottleneck (chunk layout,
dtype, or the stack/transpose graph) can be localized.
"""

import time

from fronts import utils
from fronts.data import datasets, inputs


def _fmt_bytes(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def main() -> None:
    """Print store layout and time reads to localize the throughput bottleneck."""
    yaml_data = utils.load_yaml("configs/schooner_train.yaml")
    data_cfg = utils.parse_config_section(yaml_data, datasets.DatasetConfig, "data_config")
    era5_cfg = data_cfg.inputs_icechunk_config

    ds = utils.open_readonly_icechunk_store(
        store_path=era5_cfg.store_path,
        branch=era5_cfg.branch_name,
        group=era5_cfg.group_name,
        zarr_format=era5_cfg.zarr_format,
        virtual_chunk_local_path=era5_cfg.virtual_chunk_local_path,
    )

    print("=== Per-variable layout ===")
    for name, var in ds.data_vars.items():
        enc = var.encoding
        print(
            f"{name:35s} dtype={var.dtype!s:8s} shape={var.shape} "
            f"chunks={enc.get('chunks')} compressor={enc.get('compressors', enc.get('compressor'))}"
        )

    n_time = ds.sizes["time"]
    print(f"\nTotal timesteps in store: {n_time}")

    sample_var = next(iter(ds.data_vars))
    print(f"\n=== Raw single-variable read timing ('{sample_var}') ===")
    for k in (1, 10, 50):
        sub = ds[sample_var].isel(time=slice(0, k))
        nbytes = sub.nbytes
        t0 = time.time()
        sub.compute()
        dt = time.time() - t0
        print(f"  {k:3d} steps: {_fmt_bytes(nbytes)} in {dt:6.1f} s -> {nbytes / 1e6 / max(dt, 1e-9):7.1f} MB/s")

    print("\n=== Full-stack read timing (inputs_ds_to_dataarray) ===")
    da = inputs.inputs_ds_to_dataarray(ds, data_cfg.variables)
    print(f"  assembled dtype={da.dtype} shape={da.shape} chunks={da.chunks}")
    for k in (1, 10, 50):
        sub = da.isel(time=slice(0, k))
        nbytes = sub.nbytes
        t0 = time.time()
        sub.compute()
        dt = time.time() - t0
        print(f"  {k:3d} steps: {_fmt_bytes(nbytes)} in {dt:6.1f} s -> {nbytes / 1e6 / max(dt, 1e-9):7.1f} MB/s")


if __name__ == "__main__":
    main()
