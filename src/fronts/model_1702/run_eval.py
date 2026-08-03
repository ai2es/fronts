"""Evaluation driver producing 2.0-format stats for model_1702 and baseline checkpoints.

Runs either the legacy model_1702 (via the config-patch loader and legacy-normalization
adapter) or a standard 2.0 ``.keras`` checkpoint through ``fronts.evaluate.compute_stats``
over a configurable set of regions — full domain, land, ocean, and the five forecast-office
regions — writing ``stats_{spatial,aggregate,derived}{_region}.nc`` files with the same
naming scheme as ``fronts.evaluate`` so the standard plotting code consumes them unchanged.

Usage:
    python -m fronts.model_1702.run_eval --config_path configs/model_1702/eval_1702_conus.yaml
    python -m fronts.model_1702.run_eval --config_path configs/model_1702/eval_1702_full.yaml --region WPC
"""

import argparse
import dataclasses
import datetime
import logging
import os

import numpy as np
import tensorflow as tf

from fronts import evaluate, utils
from fronts.data import datasets
from fronts.model import SharedTargetModel
from fronts.model_1702 import adapter, loader

log = logging.getLogger(__name__)

MODEL_KIND_1702 = "model_1702"
MODEL_KIND_KERAS = "keras"

# Copied verbatim from fronts.callbacks.OFFICE_REGIONS (Unified Surface Analysis / WPC manual
# p.25 approximations). Copied rather than imported because fronts.callbacks imports wandb at
# module scope, which is not installed in every pixi environment this driver runs in.
OFFICE_REGIONS: dict[str, utils.BoundingBox] = {
    "OPC_west": utils.BoundingBox(lat_min=30.0, lat_max=80.0, lon_min=130.0, lon_max=220.0),
    "WPC": utils.BoundingBox(lat_min=30.0, lat_max=80.0, lon_min=220.0, lon_max=300.0),
    "OPC_east": utils.BoundingBox(lat_min=30.0, lat_max=80.0, lon_min=300.0, lon_max=369.75),
    "HFO": utils.BoundingBox(lat_min=0.25, lat_max=30.0, lon_min=130.0, lon_max=220.0),
    "NHC": utils.BoundingBox(lat_min=0.25, lat_max=30.0, lon_min=220.0, lon_max=369.75),
}
REGION_FULL = "full"
REGION_CHOICES = [REGION_FULL, "land", "ocean", *OFFICE_REGIONS]


@dataclasses.dataclass
class HarnessEvalConfig:
    """Configuration for a harness evaluation run.

    Attributes:
        model_path: Path to the checkpoint — ``model_1702.h5`` for kind "model_1702", a
            ``.keras`` file for kind "keras".
        model_kind: "model_1702" (legacy loader + normalization adapter) or "keras"
            (standard 2.0 checkpoint + class-padding adapter).
        outdir: Directory to write the stats NetCDF files into.
        coordinates: Spatial bounding box as [lat_min, lat_max, lon_min, lon_max].
        regions: Regions to evaluate; each produces its own stats file set. Any of
            ``REGION_CHOICES``.
        front_types: Front type labels to score, in class order.
        front_dilation: Binary dilation iterations applied to truth labels.
        batch_size: Batch size for inference.
        time_start: Restrict evaluation to timesteps on or after this date. None keeps all.
        time_end: Restrict evaluation to timesteps before this date. None keeps all.
        time_resolution: Optional pandas offset string to subsample eval timesteps (e.g.
            "6h" to run the fair head-to-head on the 6-hourly intersection of a 3-hourly
            store). None keeps every common timestep.
        gpu_device: GPU index to use. None runs on CPU.
    """

    model_path: str
    model_kind: str
    outdir: str
    coordinates: utils.BoundingBox
    regions: list[str]
    front_types: list[str]
    front_dilation: int
    batch_size: int
    time_start: datetime.datetime | None
    time_end: datetime.datetime | None
    time_resolution: str | None
    gpu_device: int | None


def unwrap_longitudes(lons: np.ndarray) -> np.ndarray:
    """Restores monotonically increasing longitudes for a wrap-crossing domain.

    ``utils.select_spatial_domain`` returns wrap-crossing domains with longitudes like
    ``[130 … 359.75, 0 … 9.75]``; plain range tests against office bounding boxes would drop
    the columns past the wrap. Adding 360 to every longitude smaller than the first restores
    monotonicity without reordering, so masks stay index-aligned.

    Args:
        lons: Longitude values in store order.

    Returns:
        Monotonically increasing longitudes covering the same columns.
    """
    return np.where(lons < lons[0], lons + 360.0, lons)


def build_region_mask(region: str, lats: np.ndarray, lons: np.ndarray) -> np.ndarray | None:
    """Builds the (n_lat, n_lon) inclusion mask for a named region.

    Args:
        region: One of ``REGION_CHOICES``.
        lats: Latitude values in store order.
        lons: Longitude values in store order (may be wrap-crossing).

    Returns:
        Boolean mask, or None for the full domain (no masking).

    Raises:
        ValueError: If the region name is unknown.
    """
    if region == REGION_FULL:
        return None
    if region in ("land", "ocean"):
        return evaluate._build_spatial_mask(lats, lons, region)
    if region in OFFICE_REGIONS:
        box = OFFICE_REGIONS[region]
        unwrapped = unwrap_longitudes(lons)
        lat_in = (lats >= box.lat_min) & (lats <= box.lat_max)
        lon_in = (unwrapped >= box.lon_min) & (unwrapped <= box.lon_max)
        return lat_in[:, np.newaxis] & lon_in[np.newaxis, :]
    raise ValueError(f"Unknown region '{region}'; expected one of {REGION_CHOICES}")


def build_model_adapter(harness_cfg: HarnessEvalConfig, lat_ascending: bool):
    """Loads the configured checkpoint and wraps it for the compute_stats calling convention.

    Args:
        harness_cfg: Harness configuration naming the checkpoint and model kind.
        lat_ascending: Whether the eval data's latitude axis is ascending (model_1702 expects
            descending; the adapter flips when needed).

    Returns:
        A callable ``model(x, training=False)`` emitting (batch, lat, lon, 6) predictions.

    Raises:
        ValueError: If the model kind is unknown.
    """
    if harness_cfg.model_kind == MODEL_KIND_1702:
        log.info("Loading legacy model_1702 from %s …", harness_cfg.model_path)
        model = loader.load_model_1702(harness_cfg.model_path)
        return adapter.FrontFinder1702Adapter(model, lat_ascending=lat_ascending)
    if harness_cfg.model_kind == MODEL_KIND_KERAS:
        log.info("Loading 2.0 checkpoint from %s …", harness_cfg.model_path)
        model = tf.keras.models.load_model(
            harness_cfg.model_path, compile=False, custom_objects={"SharedTargetModel": SharedTargetModel}
        )
        return adapter.ClassPaddingAdapter(model)
    raise ValueError(f"Unknown model_kind '{harness_cfg.model_kind}'")


def _open_eval_data(harness_cfg: HarnessEvalConfig, data_cfg: datasets.DatasetConfig):
    """Opens and aligns the input and target stores; mirrors fronts.evaluate.run's glue."""
    ic_inputs = data_cfg.inputs_icechunk_config
    ic_fronts = data_cfg.targets_icechunk_config

    input_ds = utils.open_readonly_icechunk_store(
        ic_inputs.store_path,
        ic_inputs.branch_name,
        group=ic_inputs.group_name,
        zarr_format=ic_inputs.zarr_format,
        virtual_chunk_local_path=ic_inputs.virtual_chunk_local_path,
        chunks=None,
    )
    fronts_ds = utils.open_readonly_icechunk_store(
        ic_fronts.store_path,
        ic_fronts.branch_name,
        group=ic_fronts.group_name,
        zarr_format=ic_fronts.zarr_format,
        virtual_chunk_local_path=ic_fronts.virtual_chunk_local_path,
        chunks=None,
    )

    input_ds = utils.select_spatial_domain(input_ds, harness_cfg.coordinates)
    fronts_ds = utils.select_spatial_domain(fronts_ds, harness_cfg.coordinates)
    fronts_raw = utils.drop_duplicate_times(fronts_ds["identifier"])

    common_times = np.intersect1d(input_ds["time"].values, fronts_raw["time"].values)
    if harness_cfg.time_start:
        common_times = common_times[common_times >= np.datetime64(harness_cfg.time_start)]
    if harness_cfg.time_end:
        common_times = common_times[common_times < np.datetime64(harness_cfg.time_end)]
    if harness_cfg.time_resolution:
        common_times = utils.apply_time_resolution(common_times, harness_cfg.time_resolution)
    log.info("Evaluation timesteps: %d", len(common_times))
    return input_ds.sel(time=common_times), fronts_raw.sel(time=common_times)


def run(harness_cfg: HarnessEvalConfig, data_cfg: datasets.DatasetConfig) -> None:
    """Evaluates the configured model over every configured region and writes stats files.

    Args:
        harness_cfg: Harness evaluation configuration.
        data_cfg: Dataset configuration naming the icechunk stores and input variables.
    """
    utils.configure_gpu(harness_cfg.gpu_device)
    input_ds, fronts_raw = _open_eval_data(harness_cfg, data_cfg)

    lats = input_ds["latitude"].values
    lons = input_ds["longitude"].values
    lat_ascending = bool(lats[0] < lats[-1])
    model = build_model_adapter(harness_cfg, lat_ascending)

    effective_data_cfg = dataclasses.replace(data_cfg, front_dilation=harness_cfg.front_dilation)
    os.makedirs(harness_cfg.outdir, exist_ok=True)

    all_preds, all_targets = evaluate.predict_batches(
        model=model,
        input_ds=input_ds,
        target_da=fronts_raw,
        data_config=effective_data_cfg,
        batch_size=harness_cfg.batch_size,
        class_weights=data_cfg.class_weights,
    )

    for region in harness_cfg.regions:
        log.info("Evaluating region '%s' …", region)
        spatial_mask = build_region_mask(region, lats, lons)
        spatial_ds, aggregate_ds, derived_ds = evaluate.accumulate_stats(
            all_preds=all_preds,
            all_targets=all_targets,
            front_types=harness_cfg.front_types,
            lats=lats,
            lons=lons,
            spatial_mask=spatial_mask,
        )
        suffix = "" if region == REGION_FULL else f"_{region}"
        for name, ds in (("spatial", spatial_ds), ("aggregate", aggregate_ds), ("derived", derived_ds)):
            path = os.path.join(harness_cfg.outdir, f"stats_{name}{suffix}.nc")
            ds.to_netcdf(path)
            log.info("%s stats → %s", name.capitalize(), path)


def main() -> None:
    """Parses arguments, loads configs, and runs the harness evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate model_1702 or a 2.0 checkpoint over harness regions.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the harness eval YAML.")
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        choices=REGION_CHOICES,
        help="Evaluate a single region instead of the configured list (for per-region SLURM jobs).",
    )
    parser.add_argument("--outdir", type=str, default=None, help="Override output directory from the config.")
    args = parser.parse_args()

    yaml_data = utils.load_yaml(args.config_path)
    harness_cfg: HarnessEvalConfig = utils.parse_config_section(
        yaml_data, HarnessEvalConfig, "harness_eval_config", utils.YAML_TYPE_HOOKS
    )
    data_cfg: datasets.DatasetConfig = utils.parse_config_section(
        yaml_data, datasets.DatasetConfig, "data_config", utils.YAML_TYPE_HOOKS
    )
    if args.region:
        harness_cfg = dataclasses.replace(harness_cfg, regions=[args.region])
    if args.outdir:
        harness_cfg = dataclasses.replace(harness_cfg, outdir=args.outdir)
    run(harness_cfg, data_cfg)


if __name__ == "__main__":
    main()
