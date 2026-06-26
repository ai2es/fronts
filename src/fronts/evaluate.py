r"""Compute TP/FP/TN/FN performance statistics from an icechunk-backed model and data pipeline.

Iterates over all aligned ERA5/fronts timesteps in the icechunk stores, runs model
inference, and accumulates statistics over 5 neighbourhood radii (50-250 km) and
100 probability thresholds (0.01-1.0) with optional land/ocean masking.

Outputs two NetCDF files compatible with the ``performance-diagrams`` subcommand of
``src/fronts/plot/plot.py``.

Usage:
    pixi run -e schooner python src/fronts/evaluation/compute_stats.py \
        --config_path configs/schooner_train.yaml --mask land

    pixi run -e schooner python src/fronts/evaluation/compute_stats.py \
        --config_path configs/schooner_train.yaml --mask ocean --outdir ~/models/fronts/stats
"""

import argparse
import dataclasses
import datetime
import logging
import os
from typing import Any

import numpy as np
import regionmask
import tensorflow as tf
import xarray as xr
from scipy.ndimage import maximum_filter1d
from tqdm import tqdm

from fronts import utils
from fronts.data import datasets, inputs, targets
from fronts.model import SharedTargetModel

log = logging.getLogger(__name__)

# Note that FRONT_TYPE_CLASS_INDEX is not the number from the fronts data, but the index
# in the one-hot encoded array (0=background, 1=CF, 2=WF, 3=SF, 4=OF, 5=DL)
FRONT_TYPE_CLASS_INDEX: dict[str, int] = {"CF": 1, "WF": 2, "SF": 3, "OF": 4, "DL": 5}
NEIGHBORHOODS_KM = np.array([50, 100, 150, 200, 250])
N_THRESHOLDS = 100
THRESHOLDS = np.linspace(0.01, 1.0, N_THRESHOLDS, dtype=np.float32)


@dataclasses.dataclass
class EvalConfig:
    """Configuration for running performance statistics evaluation.

    Attributes:
        model_path: Path to the saved .keras model checkpoint.
        outdir: Directory to write stats_aggregate_{mask}.nc and stats_spatial_{mask}.nc.
        coordinates: Spatial bounding box as [lat_min, lat_max, lon_min, lon_max].
            Defaults to full USAD extent.
        front_types: Front type labels in class order (excluding background class 0).
        mask: Restrict statistics to "land" or "ocean" grid points. None means all points.
        front_dilation: Binary dilation iterations applied to truth labels. None uses
            the value from the paired DatasetConfig.
        gpu_device: GPU index to use. None runs on CPU.
        time_start: Restrict evaluation to timesteps on or after this date. None means no lower bound.
        time_end: Restrict evaluation to timesteps before this date. None means no upper bound.
    """

    model_path: str
    outdir: str
    coordinates: utils.BoundingBox = dataclasses.field(
        default_factory=lambda: utils.BoundingBox(lat_min=0.25, lat_max=80.0, lon_min=130.0, lon_max=369.75)
    )
    front_types: list[str] = dataclasses.field(default_factory=lambda: ["CF", "WF", "SF", "OF", "DL"])
    mask: str | None = None
    front_dilation: int | None = None
    gpu_device: int | None = None
    time_start: datetime.datetime | None = None
    time_end: datetime.datetime | None = None


def _build_spatial_mask(lats: np.ndarray, lons: np.ndarray, mask: str) -> np.ndarray:
    """Return a (n_lat, n_lon) bool array — True where points are included."""
    land_regions = regionmask.defined_regions.natural_earth_v5_1_2.land_110
    # regionmask requires monotonically increasing lons; wrap-crossing domains produce
    # non-monotonic arrays, so sort before masking and inverse-permute columns back.
    sort_idx = np.argsort(lons)
    raw = land_regions.mask(lons[sort_idx], lats)
    inv_idx = np.argsort(sort_idx)
    is_land = ~np.isnan(raw.values[:, inv_idx])
    spatial_mask = is_land if mask == "land" else ~is_land
    log.info("Spatial mask (%s): %d / %d grid points included.", mask, spatial_mask.sum(), spatial_mask.size)
    return spatial_mask


def _expand_all_neighborhoods(
    truth_fronts: np.ndarray,
    n_nbhd: int,
    lat_pixels_per_step: int,
    lon_pixels_per_lat: np.ndarray,
) -> np.ndarray:
    """Return cumulative maximum-filter expansions for all neighbourhood radii.

    Two 1-D passes per step: uniform lat expansion then per-latitude lon expansion,
    which correctly represents a fixed km radius across all latitudes.

    Args:
        truth_fronts: Boolean truth labels of shape (lat, lon, n_fronts).
        n_nbhd: Number of neighbourhood radii.
        lat_pixels_per_step: Pixel radius in the latitude direction, uniform for all rows.
        lon_pixels_per_lat: Per-row pixel radius in the longitude direction, shape (n_lat,).

    Returns:
        Array of shape (n_nbhd, lat, lon, n_fronts) where slice [ni] is the result
        after ni+1 cumulative applications of the filter.
    """
    n_lat, n_lon, n_fronts = truth_fronts.shape
    expanded_stack = np.empty((n_nbhd, n_lat, n_lon, n_fronts), dtype=bool)
    expanded = truth_fronts.copy()
    unique_lon_px = np.unique(lon_pixels_per_lat)
    for ni in range(n_nbhd):
        inter = maximum_filter1d(expanded.astype(np.uint8), size=2 * lat_pixels_per_step + 1, axis=0)
        for lon_px in unique_lon_px:
            rows = np.where(lon_pixels_per_lat == lon_px)[0]
            inter[rows] = maximum_filter1d(inter[rows], size=2 * int(lon_px) + 1, axis=1)
        expanded = inter.astype(bool)
        expanded_stack[ni] = expanded
    return expanded_stack


def compute_stats(
    model: Any,
    era5_da: xr.DataArray,
    targets_da: xr.DataArray,
    front_types: list[str],
    lats: np.ndarray,
    lons: np.ndarray,
    spatial_mask: np.ndarray | None,
    batch_size: int = 32,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Run inference then accumulate TP/FP/TN/FN over all timesteps.

    Args:
        model: Loaded Keras model with baked-in normalization.
        era5_da: ERA5 inputs of shape (time, lat, lon, channel).
        targets_da: One-hot truth labels of shape (time, lat, lon, class).
        front_types: Front type labels in class order excluding background (class 0).
        lats: 1-D latitude array.
        lons: 1-D longitude array.
        spatial_mask: Boolean (n_lat, n_lon) mask — True for included points. None = all.
        batch_size: Batch size for model.predict().

    Returns:
        Tuple of (spatial_ds, aggregate_ds) xr.Datasets with TP/FP/TN/FN variables.
        spatial_ds variables have dims (latitude, longitude, neighborhood, threshold).
        aggregate_ds variables have dims (neighborhood, threshold).
    """
    n_fronts = len(front_types)
    n_nbhd = len(NEIGHBORHOODS_KM)
    n_lat, n_lon = len(lats), len(lons)

    lat_res_km = float(np.abs(np.diff(lats)).mean()) * np.pi * 6371.0 / 180.0
    nbhd_step_km = float(np.unique(np.diff(NEIGHBORHOODS_KM)).item())
    lat_pixels_per_step = round(nbhd_step_km / lat_res_km)
    lon_pixels_per_lat = np.round(nbhd_step_km / (lat_res_km * np.cos(np.deg2rad(lats)))).astype(int)
    log.info(
        "Grid resolution %.2f km → lat_pixels_per_step=%d, lon range=[%d, %d]",
        lat_res_km,
        lat_pixels_per_step,
        lon_pixels_per_lat.min(),
        lon_pixels_per_lat.max(),
    )
    lat_weights = np.cos(np.deg2rad(lats))[:, np.newaxis] * np.ones((1, n_lon), dtype=np.float32)
    if spatial_mask is not None:
        lat_weights = lat_weights * spatial_mask.astype(np.float32)

    spatial_shape = (n_fronts, n_lat, n_lon, n_nbhd, N_THRESHOLDS)
    aggregate_shape = (n_fronts, n_nbhd, N_THRESHOLDS)
    spatial = {k: np.zeros(spatial_shape, dtype=np.float32) for k in ("tp", "fp", "tn", "fn")}
    aggregate = {k: np.zeros(aggregate_shape, dtype=np.float32) for k in ("tp", "fp", "tn", "fn")}

    n_times = era5_da.sizes["time"]
    class_indices = [FRONT_TYPE_CLASS_INDEX[ft] for ft in front_types]

    log.info("Running model.predict() over %d timesteps …", n_times)

    data_workers = utils.limit_workers_for_slurm(max_workers=16)
    eval_dataset = datasets.EvalDataset(era5_da, batch_size=batch_size, workers=data_workers)
    all_preds = model.predict(eval_dataset)
    if isinstance(all_preds, (list, tuple)):
        all_preds = all_preds[0]
    # (n_times, lat, lon, n_classes)

    _above = np.empty((n_fronts, n_lat, n_lon, N_THRESHOLDS), dtype=bool)
    _bool_buf = np.empty_like(_above)
    _float_buf = np.empty((n_fronts, n_lat, n_lon, N_THRESHOLDS), dtype=np.float32)
    _above_f32 = np.empty_like(_float_buf)
    _expanded = np.empty((n_nbhd, n_lat, n_lon, n_fronts), dtype=bool)
    w_4d = lat_weights[np.newaxis, :, :, np.newaxis]
    w_flat = lat_weights.ravel()

    log.info("Accumulating statistics …")
    for t in tqdm(range(n_times), unit="timestep"):
        pred_np = all_preds[t]
        y_np = targets_da.isel(time=t).values

        pred_fronts = pred_np[:, :, class_indices]  # (lat, lon, F)
        truth_fronts = y_np[:, :, class_indices] > 0.5  # (lat, lon, F) bool

        pred_f = pred_fronts.transpose(2, 0, 1)  # (F, lat, lon)
        truth_f = truth_fronts.transpose(2, 0, 1)  # (F, lat, lon)

        np.greater_equal(pred_f[:, :, :, np.newaxis], THRESHOLDS, out=_above)

        truth_4d = truth_f[:, :, :, np.newaxis]  # (F, lat, lon, 1) — tiny view

        # TN: ~above & ~truth, weighted.
        np.logical_not(_above, out=_bool_buf)
        np.logical_and(_bool_buf, ~truth_4d, out=_bool_buf)
        np.multiply(_bool_buf, w_4d, out=_float_buf)
        for ni in range(n_nbhd):
            spatial["tn"][:, :, :, ni, :] += _float_buf
        aggregate["tn"] += _float_buf.sum(axis=(1, 2))[:, np.newaxis, :]

        # FN: ~above & truth, weighted.
        np.logical_not(_above, out=_bool_buf)
        np.logical_and(_bool_buf, truth_4d, out=_bool_buf)
        np.multiply(_bool_buf, w_4d, out=_float_buf)
        for ni in range(n_nbhd):
            spatial["fn"][:, :, :, ni, :] += _float_buf
        aggregate["fn"] += _float_buf.sum(axis=(1, 2))[:, np.newaxis, :]

        _expanded[:] = _expand_all_neighborhoods(truth_fronts, n_nbhd, lat_pixels_per_step, lon_pixels_per_lat)
        exp_f = _expanded.transpose(3, 0, 1, 2)  # (F, n_nbhd, lat, lon)

        _above_f32[:] = _above
        above_flat = _above_f32.reshape(n_fronts, n_lat * n_lon, N_THRESHOLDS)
        exp_flat = exp_f.reshape(n_fronts, n_nbhd, n_lat * n_lon)
        aggregate["tp"] += np.matmul(exp_flat * w_flat, above_flat)
        aggregate["fp"] += np.matmul((~exp_flat) * w_flat, above_flat)

        for ni in range(n_nbhd):
            exp_ni = exp_f[:, ni, :, :, np.newaxis]  # (F, lat, lon, 1) — tiny view
            np.logical_and(_above, exp_ni, out=_bool_buf)
            np.multiply(_bool_buf, w_4d, out=_float_buf)
            spatial["tp"][:, :, :, ni, :] += _float_buf
            np.logical_and(_above, ~exp_ni, out=_bool_buf)
            np.multiply(_bool_buf, w_4d, out=_float_buf)
            spatial["fp"][:, :, :, ni, :] += _float_buf

    dims_sp = ("latitude", "longitude", "neighborhood", "threshold")
    dims_ag = ("neighborhood", "threshold")
    spatial_ds = xr.Dataset(
        coords={"latitude": lats, "longitude": lons, "neighborhood": NEIGHBORHOODS_KM, "threshold": THRESHOLDS},
        data_vars={
            f"{k}_spatial_{ft}": (dims_sp, spatial[k][fi])
            for fi, ft in enumerate(front_types)
            for k in ("tp", "fp", "tn", "fn")
        },
    )
    aggregate_ds = xr.Dataset(
        coords={"neighborhood": NEIGHBORHOODS_KM, "threshold": THRESHOLDS},
        data_vars={
            f"{k}_{ft}": (dims_ag, aggregate[k][fi])
            for fi, ft in enumerate(front_types)
            for k in ("tp", "fp", "tn", "fn")
        },
    )

    return spatial_ds, aggregate_ds


def run(eval_cfg: EvalConfig, data_cfg: datasets.DatasetConfig) -> None:
    """Run stats computation from pre-loaded config objects.

    Args:
        eval_cfg: Evaluation configuration specifying model path, output directory, etc.
        data_cfg: Dataset configuration specifying icechunk store paths and variables.
    """
    utils.configure_gpu(eval_cfg.gpu_device)
    log.info("Loading model from %s …", eval_cfg.model_path)
    keras_model = tf.keras.models.load_model(
        eval_cfg.model_path, compile=False, custom_objects={"SharedTargetModel": SharedTargetModel}
    )
    log.info("Model loaded. Output count: %d.", len(keras_model.outputs))

    ic_era5 = data_cfg.inputs_icechunk_config
    ic_fronts = data_cfg.targets_icechunk_config

    log.info("Opening ERA5 icechunk store …")
    era5_ds = utils.open_readonly_icechunk_store(
        ic_era5.store_path,
        ic_era5.branch_name,
        group=ic_era5.group_name,
        zarr_format=ic_era5.zarr_format,
        virtual_chunk_local_path=ic_era5.virtual_chunk_local_path,
    )
    log.info("Opening fronts icechunk store …")
    fronts_ds = utils.open_readonly_icechunk_store(
        ic_fronts.store_path,
        ic_fronts.branch_name,
        group=ic_fronts.group_name,
        zarr_format=ic_fronts.zarr_format,
        virtual_chunk_local_path=ic_fronts.virtual_chunk_local_path,
    )

    bb = eval_cfg.coordinates
    era5_ds = utils.select_spatial_domain(era5_ds, bb)
    fronts_ds = utils.select_spatial_domain(fronts_ds, bb)

    era5_da = inputs.inputs_ds_to_dataarray(era5_ds, data_cfg.variables)
    fronts_remapped = targets.remap_fronts(fronts_ds["identifier"])
    targets_da = utils.drop_duplicate_times(targets.one_hot_encode_to_dataarray(fronts_remapped))

    dilation = eval_cfg.front_dilation if eval_cfg.front_dilation is not None else data_cfg.front_dilation
    if dilation > 0:
        log.info("Applying front dilation: %d iterations …", dilation)
        targets_da = targets.dilate_fronts(targets_da, dilation)

    common_times = np.intersect1d(era5_da["time"].values, targets_da["time"].values)
    if eval_cfg.time_start:
        common_times = common_times[common_times >= np.datetime64(eval_cfg.time_start)]
    if eval_cfg.time_end:
        common_times = common_times[common_times < np.datetime64(eval_cfg.time_end)]
    log.info("Common timesteps: %d", len(common_times))
    era5_da = era5_da.sel(time=common_times)
    targets_da = targets_da.sel(time=common_times)

    lats = era5_da["latitude"].values
    lons = era5_da["longitude"].values

    spatial_mask = _build_spatial_mask(lats, lons, eval_cfg.mask) if eval_cfg.mask else None

    log.info("Computing statistics …")
    spatial_ds, aggregate_ds = compute_stats(
        model=keras_model,
        era5_da=era5_da,
        targets_da=targets_da,
        front_types=eval_cfg.front_types,
        lats=lats,
        lons=lons,
        spatial_mask=spatial_mask,
        batch_size=data_cfg.batch_size,
    )

    os.makedirs(eval_cfg.outdir, exist_ok=True)
    mask_suffix = f"_{eval_cfg.mask}" if eval_cfg.mask else ""
    spatial_path = os.path.join(eval_cfg.outdir, f"stats_spatial{mask_suffix}.nc")
    aggregate_path = os.path.join(eval_cfg.outdir, f"stats_aggregate{mask_suffix}.nc")

    spatial_ds.astype("float32").to_netcdf(spatial_path)
    aggregate_ds.astype("float32").to_netcdf(aggregate_path)
    log.info("Spatial stats   → %s", spatial_path)
    log.info("Aggregate stats → %s", aggregate_path)


def main() -> None:
    """Parse arguments, load configs, and run stats computation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Compute performance statistics from icechunk stores.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to training config YAML.")
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        choices=["land", "ocean"],
        help="Restrict stats to land or ocean grid points.",
    )
    parser.add_argument("--outdir", type=str, default=None, help="Override output directory from eval_config.")
    args = parser.parse_args()

    yaml_data = utils.load_yaml(args.config_path)
    eval_cfg: EvalConfig = utils.parse_config_section(yaml_data, EvalConfig, "eval_config", utils.YAML_TYPE_HOOKS)
    data_cfg: datasets.DatasetConfig = utils.parse_config_section(yaml_data, datasets.DatasetConfig, "data_config")
    eval_cfg = dataclasses.replace(
        eval_cfg,
        mask=args.mask if args.mask is not None else eval_cfg.mask,
        outdir=args.outdir if args.outdir is not None else eval_cfg.outdir,
    )
    run(eval_cfg, data_cfg)


if __name__ == "__main__":
    main()
