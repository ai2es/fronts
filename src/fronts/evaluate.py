r"""Compute TP/FP/TN/FN performance statistics from an icechunk-backed model and data pipeline.

Iterates over all aligned ERA5/fronts timesteps in the icechunk stores, runs model
inference, and accumulates statistics over 5 neighbourhood radii (50-250 km) and
100 probability thresholds (0.01-1.0) with optional land/ocean masking.

Outputs three NetCDF files compatible with the ``performance-diagrams`` subcommand of
``src/fronts/plot/plot.py``.

Usage:
    pixi run -e schooner python src/fronts/evaluate.py \
        --config_path configs/schooner_eval.yaml --mask land

    pixi run -e schooner python src/fronts/evaluate.py \
        --config_path configs/schooner_eval.yaml --mask ocean --outdir ~/models/fronts/stats
"""

import argparse
import dataclasses
import datetime
import logging
import os

import numpy as np
import regionmask
import tensorflow as tf
import xarray as xr
from scipy.ndimage import maximum_filter1d
from tqdm import tqdm

from fronts import utils
from fronts.data import datasets
from fronts.layers import losses, metrics
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
        outdir: Directory to write stats_spatial_{mask}.nc, stats_aggregate_{mask}.nc, and stats_derived_{mask}.nc.
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


def compute_derived_stats(
    aggregate_ds: xr.Dataset,
    spatial_ds: xr.Dataset,
    front_types: list[str],
) -> xr.Dataset:
    """Derive POD, SR, CSI, FB, HSS, and reliability metrics from raw TP/FP/TN/FN datasets.

    Args:
        aggregate_ds: Per-front-type TP/FP/TN/FN with dims (neighborhood, threshold).
        spatial_ds: Per-front-type spatial TP/FP/TN/FN with dims (lat, lon, neighborhood, threshold).
        front_types: Front type labels in class order excluding background.

    Returns:
        xr.Dataset with derived metric variables per front type.
    """
    data_vars: dict = {}
    for ft in front_types:
        tp = aggregate_ds[f"tp_{ft}"]
        fp = aggregate_ds[f"fp_{ft}"]
        fn = aggregate_ds[f"fn_{ft}"]
        tn = aggregate_ds[f"tn_{ft}"]

        data_vars[f"pod_{ft}"] = xr.where((tp + fn) > 0, tp / (tp + fn), 0.0).astype(np.float32)
        data_vars[f"sr_{ft}"] = xr.where((tp + fp) > 0, tp / (tp + fp), 0.0).astype(np.float32)
        data_vars[f"csi_{ft}"] = xr.where((tp + fp + fn) > 0, tp / (tp + fp + fn), 0.0).astype(np.float32)
        data_vars[f"fb_{ft}"] = xr.where((tp + fn) > 0, (tp + fp) / (tp + fn), 0.0).astype(np.float32)

        hss_denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
        data_vars[f"hss_{ft}"] = xr.where(hss_denom > 0, 2 * (tp * tn - fp * fn) / hss_denom, 0.0).astype(np.float32)

        tp_diff = abs(tp.diff("threshold")).rename({"threshold": "threshold_diff"})
        fp_diff = abs(fp.diff("threshold")).rename({"threshold": "threshold_diff"})
        denom_rf = tp_diff + fp_diff
        data_vars[f"obs_rel_freq_{ft}"] = xr.where(denom_rf > 0, tp_diff / denom_rf, 0.0).astype(np.float32)

        total_0 = (tp + fp + tn + fn).isel(neighborhood=0, threshold=0)
        data_vars[f"rel_forecast_frac_{ft}"] = xr.where(
            total_0 > 0, 100 * (tp + fp).isel(neighborhood=0) / total_0, 0.0
        ).astype(np.float32)

        tp_sp = spatial_ds[f"tp_spatial_{ft}"]
        fp_sp = spatial_ds[f"fp_spatial_{ft}"]
        fn_sp = spatial_ds[f"fn_spatial_{ft}"]
        sp_denom = tp_sp + fp_sp + fn_sp
        data_vars[f"spatial_csi_{ft}"] = xr.where(sp_denom > 0, tp_sp / sp_denom, 0.0).astype(np.float32)

    return xr.Dataset(data_vars)


def predict_batches(
    model: tf.keras.Model,
    input_ds: xr.Dataset,
    target_da: xr.DataArray,
    data_config: datasets.DatasetConfig,
    batch_size: int = 32,
    class_weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model inference over every timestep and concatenate predictions and targets.

    Args:
        model: Loaded Keras model with baked-in normalization, callable as model(x, training=False).
        input_ds: ERA5 input Dataset with dims (time, latitude, longitude) per variable.
        target_da: Raw integer front-code DataArray with dims (time, latitude, longitude).
        data_config: DatasetConfig providing variables list and front_dilation.
        batch_size: Batch size for inference.
        class_weights: Per-class loss weights passed to the FSS loss and HSS metric (logged only;
            does not affect the returned predictions/targets).

    Returns:
        Tuple of (all_preds, all_targets), each shaped (time, latitude, longitude, n_classes).
    """
    loss_fn = losses.fractions_skill_score(mask_size=(3, 3), class_weights=class_weights)
    hss_fn = metrics.heidke_skill_score(class_weights=class_weights)

    @tf.function
    def eval_step(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        preds = model(x, training=False)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        return preds, loss_fn(y, preds), hss_fn(y, preds)

    eval_dataset = datasets.FrontsPyDataset(
        input_ds=input_ds,
        target_da=target_da,
        data_config=data_config,
        batch_size=batch_size,
        shuffle=False,
        workers=0,
    )
    all_preds_list: list[np.ndarray] = []
    all_targets_list: list[np.ndarray] = []
    total_loss, total_hss = 0.0, 0.0
    log.info("Running eval_step over %d timesteps …", input_ds.sizes["time"])
    for i in tqdm(range(len(eval_dataset)), unit="batch"):
        x_batch, y_batch = eval_dataset[i]
        pred_batch, batch_loss, batch_hss = eval_step(tf.constant(x_batch), tf.constant(y_batch))
        all_preds_list.append(pred_batch.numpy())
        all_targets_list.append(y_batch)
        total_loss += float(tf.reduce_mean(batch_loss))
        total_hss += float(batch_hss)
    n_batches = len(eval_dataset)
    all_preds = np.concatenate(all_preds_list, axis=0)
    all_targets = np.concatenate(all_targets_list, axis=0)
    log.info("Eval — loss: %.4f | HSS: %.4f", total_loss / n_batches, total_hss / n_batches)
    return all_preds, all_targets


def accumulate_stats(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    front_types: list[str],
    lats: np.ndarray,
    lons: np.ndarray,
    spatial_mask: np.ndarray | None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Accumulate TP/FP/TN/FN over all timesteps from precomputed predictions and targets.

    Args:
        all_preds: Predictions shaped (time, latitude, longitude, n_classes), as returned by
            predict_batches.
        all_targets: One-hot targets shaped (time, latitude, longitude, n_classes), as returned
            by predict_batches.
        front_types: Front type labels in class order excluding background (class 0).
        lats: 1-D latitude array.
        lons: 1-D longitude array.
        spatial_mask: Boolean (n_lat, n_lon) mask — True for included points. None = all.

    Returns:
        Tuple of (spatial_ds, aggregate_ds, derived_ds) xr.Datasets.
        spatial_ds variables have dims (latitude, longitude, neighborhood, threshold).
        aggregate_ds variables have dims (neighborhood, threshold).
        derived_ds contains POD, SR, CSI, FB, HSS, reliability, and spatial CSI.
    """
    n_fronts = len(front_types)
    n_nbhd = len(NEIGHBORHOODS_KM)
    n_lat, n_lon = len(lats), len(lons)
    n_times = all_preds.shape[0]

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

    class_indices = [FRONT_TYPE_CLASS_INDEX[ft] for ft in front_types]

    _above = np.empty((n_fronts, n_lat, n_lon, N_THRESHOLDS), dtype=bool)
    _bool_buf = np.empty_like(_above)
    _float_buf = np.empty((n_fronts, n_lat, n_lon, N_THRESHOLDS), dtype=np.float32)
    _float_buf2 = np.empty_like(_float_buf)
    _above_f32 = np.empty_like(_float_buf)
    _above_weighted = np.empty_like(_float_buf)
    w_4d = lat_weights[np.newaxis, :, :, np.newaxis]
    w_flat = lat_weights.ravel()

    log.info("Accumulating statistics …")
    for t in tqdm(range(n_times), unit="timestep"):
        pred_np = all_preds[t]
        y_np = all_targets[t]

        pred_fronts = pred_np[:, :, class_indices]  # (lat, lon, F)
        truth_fronts = y_np[:, :, class_indices] > 0.5  # (lat, lon, F) bool

        pred_f = pred_fronts.transpose(2, 0, 1)  # (F, lat, lon)
        truth_f = truth_fronts.transpose(2, 0, 1)  # (F, lat, lon)

        np.greater_equal(pred_f[:, :, :, np.newaxis], THRESHOLDS, out=_above)
        _above_f32[:] = _above
        np.multiply(_above_f32, w_4d, out=_above_weighted)

        truth_4d = truth_f[:, :, :, np.newaxis]  # (F, lat, lon, 1) — tiny view

        # TN: ~above & ~truth, weighted. Broadcast into every neighborhood slot at once —
        # TN/FN don't depend on the neighborhood radius.
        np.logical_not(_above, out=_bool_buf)
        np.logical_and(_bool_buf, ~truth_4d, out=_bool_buf)
        np.multiply(_bool_buf, w_4d, out=_float_buf)
        spatial["tn"] += _float_buf[:, :, :, np.newaxis, :]
        aggregate["tn"] += _float_buf.sum(axis=(1, 2))[:, np.newaxis, :]

        # FN: ~above & truth, weighted.
        np.logical_not(_above, out=_bool_buf)
        np.logical_and(_bool_buf, truth_4d, out=_bool_buf)
        np.multiply(_bool_buf, w_4d, out=_float_buf)
        spatial["fn"] += _float_buf[:, :, :, np.newaxis, :]
        aggregate["fn"] += _float_buf.sum(axis=(1, 2))[:, np.newaxis, :]

        # Force contiguity: matmul below falls back to a slow non-BLAS path on the
        # transpose-then-reshape view (~96x slower than on a contiguous copy).
        expanded = _expand_all_neighborhoods(truth_fronts, n_nbhd, lat_pixels_per_step, lon_pixels_per_lat)
        exp_f = np.ascontiguousarray(expanded.transpose(3, 0, 1, 2))  # (F, n_nbhd, lat, lon)

        above_flat = _above_f32.reshape(n_fronts, n_lat * n_lon, N_THRESHOLDS)
        exp_flat = exp_f.reshape(n_fronts, n_nbhd, n_lat * n_lon)
        aggregate["tp"] += np.matmul(exp_flat * w_flat, above_flat)
        aggregate["fp"] += np.matmul((~exp_flat) * w_flat, above_flat)

        # TP + FP == above_weighted exactly for a 0/1 mask, so FP is derived by subtraction
        # instead of a second logical_and + multiply pass.
        for ni in range(n_nbhd):
            exp_ni = exp_f[:, ni, :, :, np.newaxis]  # (F, lat, lon, 1) — tiny view
            np.multiply(_above_weighted, exp_ni, out=_float_buf)
            spatial["tp"][:, :, :, ni, :] += _float_buf
            np.subtract(_above_weighted, _float_buf, out=_float_buf2)
            spatial["fp"][:, :, :, ni, :] += _float_buf2

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
    derived_ds = compute_derived_stats(aggregate_ds, spatial_ds, front_types)

    return spatial_ds, aggregate_ds, derived_ds


def compute_stats(
    model: tf.keras.Model,
    input_ds: xr.Dataset,
    target_da: xr.DataArray,
    data_config: datasets.DatasetConfig,
    front_types: list[str],
    lats: np.ndarray,
    lons: np.ndarray,
    spatial_mask: np.ndarray | None,
    batch_size: int = 32,
    class_weights: list[float] | None = None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Run inference then accumulate TP/FP/TN/FN over all timesteps.

    Thin wrapper around predict_batches + accumulate_stats, kept for callers that only need one
    region/mask per model load; callers evaluating multiple masks over the same model and data
    should call predict_batches once and accumulate_stats per mask instead.

    Args:
        model: Loaded Keras model with baked-in normalization, callable as model(x, training=False).
        input_ds: ERA5 input Dataset with dims (time, latitude, longitude) per variable.
        target_da: Raw integer front-code DataArray with dims (time, latitude, longitude).
        data_config: DatasetConfig providing variables list and front_dilation.
        front_types: Front type labels in class order excluding background (class 0).
        lats: 1-D latitude array.
        lons: 1-D longitude array.
        spatial_mask: Boolean (n_lat, n_lon) mask — True for included points. None = all.
        batch_size: Batch size for inference.
        class_weights: Per-class loss weights passed to the FSS loss and HSS metric.

    Returns:
        Tuple of (spatial_ds, aggregate_ds, derived_ds) xr.Datasets.
        spatial_ds variables have dims (latitude, longitude, neighborhood, threshold).
        aggregate_ds variables have dims (neighborhood, threshold).
        derived_ds contains POD, SR, CSI, FB, HSS, reliability, and spatial CSI.
    """
    all_preds, all_targets = predict_batches(model, input_ds, target_da, data_config, batch_size, class_weights)
    return accumulate_stats(all_preds, all_targets, front_types, lats, lons, spatial_mask)


def load_eval_arrays(
    eval_cfg: EvalConfig, data_cfg: datasets.DatasetConfig
) -> tuple[xr.Dataset, xr.DataArray, np.ndarray, np.ndarray, np.ndarray | None, datasets.DatasetConfig]:
    """Open icechunk stores, align to a common time/space grid, and resolve dilation/mask.

    Args:
        eval_cfg: Evaluation configuration specifying coordinates, mask, dilation, and time range.
        data_cfg: Dataset configuration specifying icechunk store paths and variables.

    Returns:
        Tuple of (era5_ds, fronts_raw, lats, lons, spatial_mask, effective_data_cfg).
    """
    ic_era5 = data_cfg.inputs_icechunk_config
    ic_fronts = data_cfg.targets_icechunk_config

    log.info("Opening ERA5 icechunk store …")
    era5_ds = utils.open_readonly_icechunk_store(
        ic_era5.store_path,
        ic_era5.branch_name,
        group=ic_era5.group_name,
        zarr_format=ic_era5.zarr_format,
        virtual_chunk_local_path=ic_era5.virtual_chunk_local_path,
        chunks=None,
    )
    log.info("Opening fronts icechunk store …")
    fronts_ds = utils.open_readonly_icechunk_store(
        ic_fronts.store_path,
        ic_fronts.branch_name,
        group=ic_fronts.group_name,
        zarr_format=ic_fronts.zarr_format,
        virtual_chunk_local_path=ic_fronts.virtual_chunk_local_path,
        chunks=None,
    )

    bb = eval_cfg.coordinates
    era5_ds = utils.select_spatial_domain(era5_ds, bb)
    fronts_ds = utils.select_spatial_domain(fronts_ds, bb)
    era5_ds = utils.select_pressure_levels(era5_ds, data_cfg.pressure_levels)

    fronts_raw = utils.drop_duplicate_times(fronts_ds["identifier"])

    era5_times = era5_ds["time"].values
    common_times = np.intersect1d(era5_times, fronts_raw["time"].values)
    if eval_cfg.time_start:
        common_times = common_times[common_times >= np.datetime64(eval_cfg.time_start)]
    if eval_cfg.time_end:
        common_times = common_times[common_times < np.datetime64(eval_cfg.time_end)]
    log.info("Common timesteps: %d", len(common_times))
    era5_ds = era5_ds.sel(time=common_times)
    fronts_raw = fronts_raw.sel(time=common_times)

    dilation = eval_cfg.front_dilation if eval_cfg.front_dilation is not None else data_cfg.front_dilation
    effective_data_cfg = dataclasses.replace(data_cfg, front_dilation=dilation)

    lats = era5_ds["latitude"].values
    lons = era5_ds["longitude"].values

    spatial_mask = _build_spatial_mask(lats, lons, eval_cfg.mask) if eval_cfg.mask else None

    return era5_ds, fronts_raw, lats, lons, spatial_mask, effective_data_cfg


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

    era5_ds, fronts_raw, lats, lons, spatial_mask, effective_data_cfg = load_eval_arrays(eval_cfg, data_cfg)

    log.info("Computing statistics …")
    spatial_ds, aggregate_ds, derived_ds = compute_stats(
        model=keras_model,
        input_ds=era5_ds,
        target_da=fronts_raw,
        data_config=effective_data_cfg,
        front_types=eval_cfg.front_types,
        lats=lats,
        lons=lons,
        spatial_mask=spatial_mask,
        batch_size=data_cfg.batch_size,
        class_weights=data_cfg.class_weights,
    )

    os.makedirs(eval_cfg.outdir, exist_ok=True)
    mask_suffix = f"_{eval_cfg.mask}" if eval_cfg.mask else ""
    spatial_path = os.path.join(eval_cfg.outdir, f"stats_spatial{mask_suffix}.nc")
    aggregate_path = os.path.join(eval_cfg.outdir, f"stats_aggregate{mask_suffix}.nc")
    derived_path = os.path.join(eval_cfg.outdir, f"stats_derived{mask_suffix}.nc")

    spatial_ds.to_netcdf(spatial_path)
    aggregate_ds.to_netcdf(aggregate_path)
    derived_ds.to_netcdf(derived_path)
    log.info("Spatial stats   → %s", spatial_path)
    log.info("Aggregate stats → %s", aggregate_path)
    log.info("Derived stats   → %s", derived_path)


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
