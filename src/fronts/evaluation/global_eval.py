"""Global evaluation of a 2.0 checkpoint against the TWC global surface analysis.

The checkpoint was trained on northern-hemisphere fronts, so each timestep is run twice:

* ``direct``  — inputs as they are.
* ``flipped`` — inputs mirrored about the equator with the meridional wind negated, so southern-
  hemisphere fronts look like northern-hemisphere ones. Predictions are mirrored back before scoring.

Stats are accumulated separately for the northern (lat >= 0) and southern (lat < 0) hemispheres and
the global stats are their sum, so every region is exact up to a <=250 km neighborhood-expansion edge
effect at the equator. A third mode, ``composite``, takes the northern hemisphere from ``direct`` and
the southern from ``flipped``: the number to read as the model's global skill.

Time is processed in chunks so predictions and targets never exceed memory; TP/FP/TN/FN are additive
over time, so chunk stats are summed and the derived metrics computed once at the end.

Usage:
    python -m fronts.evaluation.global_eval --config_path configs/sooner_eval_global.yaml
"""

import argparse
import dataclasses
import datetime
import logging
import os

import numpy as np
import tensorflow as tf
import xarray as xr

from fronts import evaluate, utils
from fronts.data import datasets
from fronts.model import SharedTargetModel
from fronts.model_1702 import adapter

log = logging.getLogger(__name__)

MODES = ("direct", "flipped", "composite")
HEMISPHERES = ("NH", "SH")
REGIONS = ("global", *HEMISPHERES)
_V_WIND = "v_component_of_wind"


@dataclasses.dataclass
class GlobalEvalConfig:
    """Configuration for a global evaluation run.

    Attributes:
        model_path: Path to the ``.keras`` checkpoint.
        targets_zarr: Zarr written by ``scripts/gml_to_zarr.py`` holding the ``identifier`` variable.
        outdir: Directory for the per-mode stats NetCDF files.
        coordinates: Bounding box [lat_min, lat_max, lon_min, lon_max]; latitudes must be symmetric about 0.
        front_types: Front types to score, in class order.
        front_dilation: Dilation iterations applied to truth labels.
        batch_size: Inference batch size.
        pad_multiple: Latitude is reflect-padded up to a multiple of this (the model's total downsampling factor).
        chunk_size: Timesteps held in memory at once.
        time_start: Keep timesteps on or after this date. None keeps all.
        time_end: Keep timesteps before this date. None keeps all.
        time_resolution: Optional pandas offset string to subsample timesteps (e.g. "6h").
        write_spatial: Also write the (huge) per-pixel spatial stats files.
        gpu_device: GPU index. None runs on CPU.
    """

    model_path: str
    targets_zarr: str
    outdir: str
    coordinates: utils.BoundingBox
    front_types: list[str]
    front_dilation: int
    batch_size: int
    pad_multiple: int
    chunk_size: int
    time_start: datetime.datetime | None
    time_end: datetime.datetime | None
    time_resolution: str | None
    write_spatial: bool
    gpu_device: int | None


class PadToMultiple:
    """Pads the latitude axis up to a multiple before calling a downsampling model, then crops the output.

    The U-Net halves the grid at each level, so an odd or non-divisible row count comes back short
    (641 rows -> 640 out) and no longer lines up with the targets. The symmetric ±80 global grid has
    641 rows (equator included), so it is padded by reflection and the padding rows are cropped away.
    """

    def __init__(self, model, multiple: int):
        self.model = model
        self.multiple = multiple

    def __call__(self, x: tf.Tensor, training: bool = False):
        """Runs the model on latitude-padded inputs and returns outputs cropped to the original rows."""
        n_lat = x.shape[1]
        extra = -n_lat % self.multiple
        top = extra // 2
        x = tf.pad(x, [[0, 0], [top, extra - top], [0, 0], [0, 0]], mode="REFLECT")
        preds = self.model(x, training=training)
        crop = lambda p: p[:, top : top + n_lat]  # noqa: E731
        return [crop(p) for p in preds] if isinstance(preds, (list, tuple)) else crop(preds)


def flip_inputs(input_ds: xr.Dataset) -> xr.Dataset:
    """Mirrors inputs about the equator: latitude order reversed and every meridional wind negated.

    The latitude coordinate is left as is, so row i of the result holds the data of row -i; this is
    why the grid must be symmetric about the equator. Done on raw inputs, before the model's normalization.

    Args:
        input_ds: Dataset with monotonic ``latitude`` (either order) symmetric about 0.

    Returns:
        Dataset on the same coordinates with flipped fields.

    Raises:
        ValueError: If latitudes are not monotonic and symmetric about the equator.
    """
    lats = input_ds["latitude"].values
    steps = np.diff(lats)
    if not ((np.all(steps > 0) or np.all(steps < 0)) and np.allclose(lats, -lats[::-1])):
        raise ValueError("flip_inputs needs monotonic latitudes symmetric about the equator")
    flipped = input_ds.copy()
    for name in flipped.data_vars:
        if _V_WIND in name:
            flipped[name] = -flipped[name]
    flipped = flipped.isel(latitude=slice(None, None, -1))
    return flipped.assign_coords(latitude=lats)


def hemisphere_slices(lats: np.ndarray) -> dict[str, slice]:
    """Row slices for the northern (lat >= 0) and southern (lat < 0) hemispheres, for either axis order."""
    north = np.flatnonzero(lats >= 0.0)
    south = np.flatnonzero(lats < 0.0)
    return {
        "NH": slice(int(north.min()), int(north.max()) + 1),
        "SH": slice(int(south.min()), int(south.max()) + 1),
    }


def combine_global(parts: dict[str, tuple[xr.Dataset, xr.Dataset]]) -> tuple[xr.Dataset, xr.Dataset]:
    """Adds hemispheric (spatial, aggregate) counts into global ones; spatial maps come back with ascending latitude."""
    spatial = xr.concat([parts["SH"][0], parts["NH"][0]], dim="latitude").sortby("latitude")
    aggregate = parts["SH"][1] + parts["NH"][1]
    return spatial, aggregate


def _add_in_place(total: xr.Dataset, chunk: xr.Dataset) -> None:
    for name in total.data_vars:
        total[name].values += chunk[name].values


def _open_data(cfg: GlobalEvalConfig, data_cfg: datasets.DatasetConfig):
    ic = data_cfg.inputs_icechunk_config
    input_ds = utils.open_readonly_icechunk_store(
        ic.store_path,
        ic.branch_name,
        group=ic.group_name,
        zarr_format=ic.zarr_format,
        virtual_chunk_local_path=ic.virtual_chunk_local_path,
        chunks=None,
    )
    fronts_ds = xr.open_zarr(cfg.targets_zarr, chunks=None)
    input_ds = utils.select_spatial_domain(input_ds, cfg.coordinates)
    fronts_ds = utils.select_spatial_domain(fronts_ds, cfg.coordinates)
    fronts_raw = utils.drop_duplicate_times(fronts_ds["identifier"])
    # The loader pairs inputs and targets by row position, not by label, so put the targets in the
    # inputs' latitude order (ERA5 is descending, the rasterized fronts ascending). sel raises
    # KeyError rather than silently misaligning if the grids differ.
    if not np.array_equal(fronts_raw["longitude"].values, input_ds["longitude"].values):
        raise ValueError("Input and target longitudes differ")
    fronts_raw = fronts_raw.sel(latitude=input_ds["latitude"].values)

    times = np.intersect1d(input_ds["time"].values, fronts_raw["time"].values)
    if cfg.time_start:
        times = times[times >= np.datetime64(cfg.time_start)]
    if cfg.time_end:
        times = times[times < np.datetime64(cfg.time_end)]
    if cfg.time_resolution:
        times = utils.apply_time_resolution(times, cfg.time_resolution)
    log.info("Evaluation timesteps: %d", len(times))
    return input_ds.sel(time=times), fronts_raw.sel(time=times)


def run(cfg: GlobalEvalConfig, data_cfg: datasets.DatasetConfig) -> None:
    """Runs direct and flipped inference over the global domain and writes stats for every mode and region."""
    utils.configure_gpu(cfg.gpu_device)
    input_ds, fronts_raw = _open_data(cfg, data_cfg)
    lats = input_ds["latitude"].values
    lons = input_ds["longitude"].values
    rows = hemisphere_slices(lats)

    keras_model = tf.keras.models.load_model(
        cfg.model_path, compile=False, custom_objects={"SharedTargetModel": SharedTargetModel}
    )
    model = adapter.ClassPaddingAdapter(PadToMultiple(keras_model, cfg.pad_multiple))
    data_cfg = dataclasses.replace(data_cfg, front_dilation=cfg.front_dilation)

    totals: dict[tuple[str, str], tuple[xr.Dataset, xr.Dataset]] = {}
    n_times = input_ds.sizes["time"]
    for start in range(0, n_times, cfg.chunk_size):
        sel = slice(start, min(start + cfg.chunk_size, n_times))
        chunk_in, chunk_tg = input_ds.isel(time=sel), fronts_raw.isel(time=sel)
        log.info("Chunk %d-%d of %d", sel.start, sel.stop, n_times)
        for mode, inputs in (("direct", chunk_in), ("flipped", flip_inputs(chunk_in))):
            preds, targets = evaluate.predict_batches(
                model=model,
                input_ds=inputs,
                target_da=chunk_tg,
                data_config=data_cfg,
                batch_size=cfg.batch_size,
                class_weights=data_cfg.class_weights,
            )
            if mode == "flipped":
                preds = preds[:, ::-1]
            for hemi in HEMISPHERES:
                spatial, aggregate, _ = evaluate.accumulate_stats(
                    all_preds=preds[:, rows[hemi]],
                    all_targets=targets[:, rows[hemi]],
                    front_types=cfg.front_types,
                    lats=lats[rows[hemi]],
                    lons=lons,
                    spatial_mask=None,
                )
                if (mode, hemi) in totals:
                    _add_in_place(totals[(mode, hemi)][0], spatial)
                    _add_in_place(totals[(mode, hemi)][1], aggregate)
                else:
                    totals[(mode, hemi)] = (spatial, aggregate)
            del preds, targets

    totals[("composite", "NH")] = totals[("direct", "NH")]
    totals[("composite", "SH")] = totals[("flipped", "SH")]
    for mode in MODES:
        parts = {h: totals[(mode, h)] for h in HEMISPHERES}
        parts_by_region = {**parts, "global": combine_global(parts)}
        outdir = os.path.join(cfg.outdir, mode)
        os.makedirs(outdir, exist_ok=True)
        for region, (spatial, aggregate) in parts_by_region.items():
            derived = evaluate.compute_derived_stats(aggregate, spatial, cfg.front_types)
            suffix = "" if region == "global" else f"_{region}"
            outputs = {"aggregate": aggregate, "derived": derived}
            if cfg.write_spatial:
                outputs["spatial"] = spatial
            for name, ds in outputs.items():
                path = os.path.join(outdir, f"stats_{name}{suffix}.nc")
                ds.to_netcdf(path)
                log.info("%s/%s → %s", mode, region, path)


def main() -> None:
    """Parses arguments, loads the config, and runs the global evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Global evaluation with direct and hemisphere-flipped inference.")
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    yaml_data = utils.load_yaml(args.config_path)
    cfg = utils.parse_config_section(yaml_data, GlobalEvalConfig, "global_eval_config", utils.YAML_TYPE_HOOKS)
    data_cfg = utils.parse_config_section(yaml_data, datasets.DatasetConfig, "data_config", utils.YAML_TYPE_HOOKS)
    run(cfg, data_cfg)


if __name__ == "__main__":
    main()
