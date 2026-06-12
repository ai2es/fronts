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
import time
from typing import Any

import numpy as np
import regionmask
import xarray as xr
from scipy.ndimage import maximum_filter
from tqdm import tqdm

from fronts import utils
from fronts.data import config, inputs, targets

log = logging.getLogger(__name__)

FRONT_TYPE_CLASS_INDEX: dict[str, int] = {"CF": 1, "WF": 2, "SF": 3, "OF": 4, "DL": 5}
NEIGHBORHOODS_KM = np.array([50, 100, 150, 200, 250])
EXPAND_ITERS_PER_STEP = 2
_EXPAND_SIZE = 2 * EXPAND_ITERS_PER_STEP + 1
N_THRESHOLDS = 100
THRESHOLDS = np.linspace(0.01, 1.0, N_THRESHOLDS, dtype=np.float32)


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
    expand_size: int,
) -> np.ndarray:
    """Return cumulative maximum-filter expansions for all neighbourhood radii.

    Args:
        truth_fronts: Boolean truth labels of shape (lat, lon, n_fronts).
        n_nbhd: Number of neighbourhood radii.
        expand_size: Size of the square maximum-filter kernel per expansion step.

    Returns:
        Array of shape (n_nbhd, lat, lon, n_fronts) where slice [ni] is the result
        after ni+1 cumulative applications of the filter.
    """
    n_lat, n_lon, n_fronts = truth_fronts.shape
    expanded_stack = np.empty((n_nbhd, n_lat, n_lon, n_fronts), dtype=bool)
    expanded = truth_fronts.copy()
    for ni in range(n_nbhd):
        expanded = maximum_filter(expanded.astype(np.uint8), size=(expand_size, expand_size, 1)).astype(bool)
        expanded_stack[ni] = expanded
    return expanded_stack


def _accumulate_timestep(
    pred_fronts: np.ndarray,
    truth_fronts: np.ndarray,
    weights: np.ndarray,
    thresholds: np.ndarray,
    n_nbhd: int,
    expand_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute weighted TP/FP/TN/FN deltas for one timestep.

    Args:
        pred_fronts: Predicted probabilities of shape (lat, lon, n_fronts).
        truth_fronts: Binary truth labels of shape (lat, lon, n_fronts).
        weights: Per-pixel weights of shape (lat, lon), e.g. cosine latitude weights
            with an optional land/ocean mask applied.
        thresholds: 1-D array of probability thresholds, shape (T,).
        n_nbhd: Number of neighbourhood radii.
        expand_size: Kernel size for each cumulative neighbourhood expansion step.

    Returns:
        8-tuple ``(tp_sp, fp_sp, tn_sp, fn_sp, tp_ag, fp_ag, tn_ag, fn_ag)``.
        Spatial shapes — tp/fp: ``(n_fronts, lat, lon, n_nbhd, T)``;
                         tn/fn: ``(n_fronts, lat, lon, 1, T)`` (identical across
                         neighbourhoods; the caller's ``+=`` broadcasts over n_nbhd).
        Aggregate shapes — tp/fp: ``(n_fronts, n_nbhd, T)``;
                           tn/fn: ``(n_fronts, 1, T)`` (broadcasts over n_nbhd).
    """
    n_lat, n_lon, n_fronts = pred_fronts.shape
    n_thresh = len(thresholds)

    # Front-first layout (n_fronts, lat, lon) throughout avoids repeated moveaxis copies.
    pred_f = pred_fronts.transpose(2, 0, 1)  # (F, lat, lon)
    truth_f = truth_fronts.transpose(2, 0, 1)  # (F, lat, lon)

    # Threshold comparison computed once and reused across all neighbourhoods.
    above = pred_f[:, :, :, np.newaxis] >= thresholds  # (F, lat, lon, T) bool
    below = ~above

    w = weights[np.newaxis, :, :, np.newaxis]  # (1, lat, lon, 1)
    truth_4d = truth_f[:, :, :, np.newaxis]  # (F, lat, lon, 1)

    # TN/FN are neighbourhood-independent; bool * float32 promotes automatically.
    tn_weighted = (below & ~truth_4d) * w  # (F, lat, lon, T) float32
    fn_weighted = (below & truth_4d) * w
    tn_sp = tn_weighted[:, :, :, np.newaxis, :]  # (F, lat, lon, 1, T)
    fn_sp = fn_weighted[:, :, :, np.newaxis, :]
    tn_ag = tn_weighted.sum(axis=(1, 2))[:, np.newaxis, :]  # (F, 1, T)
    fn_ag = fn_weighted.sum(axis=(1, 2))[:, np.newaxis, :]

    # Precompute all neighbourhood expansions: (n_nbhd, lat, lon, F) → (F, n_nbhd, lat, lon).
    expanded_stack = _expand_all_neighborhoods(truth_fronts, n_nbhd, expand_size)
    exp_f = expanded_stack.transpose(3, 0, 1, 2)  # (F, n_nbhd, lat, lon)

    # Aggregate TP/FP via batched matmul — avoids materialising (F, n_nbhd, lat, lon, T) float32.
    # Flatten spatial dims: above → (F, lat*lon, T); exp → (F, n_nbhd, lat*lon).
    above_flat = above.reshape(n_fronts, n_lat * n_lon, n_thresh).astype(np.float32)
    w_flat = weights.ravel()  # (lat*lon,)
    exp_flat = exp_f.reshape(n_fronts, n_nbhd, n_lat * n_lon)
    exp_weighted = exp_flat * w_flat  # (F, n_nbhd, lat*lon) float32
    tp_ag = np.matmul(exp_weighted, above_flat)  # (F, n_nbhd, T)
    not_exp_weighted = (~exp_flat) * w_flat
    fp_ag = np.matmul(not_exp_weighted, above_flat)

    # Spatial TP/FP: per-pixel accumulation; one neighbourhood at a time to bound peak memory.
    tp_sp = np.empty((n_fronts, n_lat, n_lon, n_nbhd, n_thresh), dtype=np.float32)
    fp_sp = np.empty_like(tp_sp)
    for ni in range(n_nbhd):
        exp_ni = exp_f[:, ni, :, :, np.newaxis]  # (F, lat, lon, 1)
        tp_sp[:, :, :, ni, :] = (above & exp_ni) * w  # (F, lat, lon, T)
        fp_sp[:, :, :, ni, :] = (above & ~exp_ni) * w

    return tp_sp, fp_sp, tn_sp, fn_sp, tp_ag, fp_ag, tn_ag, fn_ag


def compute_stats(
    model: Any,
    era5_da: xr.DataArray,
    targets_da: xr.DataArray,
    front_types: list[str],
    lats: np.ndarray,
    lons: np.ndarray,
    spatial_mask: np.ndarray | None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Iterate timesteps, run inference, and accumulate TP/FP/TN/FN.

    Args:
        model: Loaded Keras model with baked-in normalization.
        era5_da: ERA5 inputs of shape (time, lat, lon, channel).
        targets_da: One-hot truth labels of shape (time, lat, lon, class).
        front_types: Front type labels in class order excluding background (class 0).
        lats: 1-D latitude array.
        lons: 1-D longitude array.
        spatial_mask: Boolean (n_lat, n_lon) mask — True for included points. None = all.

    Returns:
        Tuple of (spatial_ds, aggregate_ds) xr.Datasets with TP/FP/TN/FN variables.
        spatial_ds variables have dims (latitude, longitude, neighborhood, threshold).
        aggregate_ds variables have dims (neighborhood, threshold).
    """
    n_fronts = len(front_types)
    n_nbhd = len(NEIGHBORHOODS_KM)
    n_lat, n_lon = len(lats), len(lons)

    lat_weights = np.cos(np.deg2rad(lats))[:, np.newaxis] * np.ones((1, n_lon), dtype=np.float32)
    if spatial_mask is not None:
        lat_weights = lat_weights * spatial_mask.astype(np.float32)

    tp_sp = np.zeros((n_fronts, n_lat, n_lon, n_nbhd, N_THRESHOLDS), dtype=np.float32)
    fp_sp = np.zeros_like(tp_sp)
    tn_sp = np.zeros_like(tp_sp)
    fn_sp = np.zeros_like(tp_sp)
    tp_ag = np.zeros((n_fronts, n_nbhd, N_THRESHOLDS), dtype=np.float32)
    fp_ag = np.zeros_like(tp_ag)
    tn_ag = np.zeros_like(tp_ag)
    fn_ag = np.zeros_like(tp_ag)

    n_times = era5_da.sizes["time"]
    class_indices = [FRONT_TYPE_CLASS_INDEX[ft] for ft in front_types]

    log.info("Loading ERA5 and targets into memory …")
    era5_np = era5_da.values.astype(np.float32)  # (time, lat, lon, channel)
    targets_np = targets_da.values.astype(np.float32)  # (time, lat, lon, class)

    # Pre-allocate all large work buffers once to eliminate per-timestep page faults.
    # Each fresh np.empty/astype on a >100 MB array triggers OS page faults (~5 µs/page)
    # that dominate runtime when called every iteration.
    _above = np.empty((n_fronts, n_lat, n_lon, N_THRESHOLDS), dtype=bool)
    _bool_buf = np.empty_like(_above)
    _float_buf = np.empty((n_fronts, n_lat, n_lon, N_THRESHOLDS), dtype=np.float32)
    _above_f32 = np.empty_like(_float_buf)
    _expanded = np.empty((n_nbhd, n_lat, n_lon, n_fronts), dtype=bool)
    w_4d = lat_weights[np.newaxis, :, :, np.newaxis]
    w_flat = lat_weights.ravel()

    for t in tqdm(range(n_times), unit="timestep"):
        t0 = time.perf_counter()

        x_np = era5_np[t]
        y_np = targets_np[t]
        t_load = time.perf_counter()

        pred = model(x_np[np.newaxis], training=False)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred_np = (pred.numpy() if hasattr(pred, "numpy") else np.asarray(pred))[0].astype(np.float32)
        t_infer = time.perf_counter()

        pred_fronts = pred_np[:, :, class_indices]  # (lat, lon, F)
        truth_fronts = y_np[:, :, class_indices] > 0.5  # (lat, lon, F) bool

        pred_f = pred_fronts.transpose(2, 0, 1)  # (F, lat, lon)
        truth_f = truth_fronts.transpose(2, 0, 1)  # (F, lat, lon)

        # Threshold comparison written into pre-allocated buffer — no allocation.
        np.greater_equal(pred_f[:, :, :, np.newaxis], THRESHOLDS, out=_above)

        truth_4d = truth_f[:, :, :, np.newaxis]  # (F, lat, lon, 1) — tiny view

        # TN: ~above & ~truth, weighted.
        np.logical_not(_above, out=_bool_buf)
        np.logical_and(_bool_buf, ~truth_4d, out=_bool_buf)
        np.multiply(_bool_buf, w_4d, out=_float_buf)
        for ni in range(n_nbhd):
            tn_sp[:, :, :, ni, :] += _float_buf
        tn_ag += _float_buf.sum(axis=(1, 2))[:, np.newaxis, :]

        # FN: ~above & truth, weighted.
        np.logical_not(_above, out=_bool_buf)
        np.logical_and(_bool_buf, truth_4d, out=_bool_buf)
        np.multiply(_bool_buf, w_4d, out=_float_buf)
        for ni in range(n_nbhd):
            fn_sp[:, :, :, ni, :] += _float_buf
        fn_ag += _float_buf.sum(axis=(1, 2))[:, np.newaxis, :]

        # Precompute neighbourhood expansions into pre-allocated buffer.
        expanded = truth_fronts.copy()
        for ni in range(n_nbhd):
            expanded = maximum_filter(expanded.astype(np.uint8), size=(_EXPAND_SIZE, _EXPAND_SIZE, 1)).astype(bool)
            _expanded[ni] = expanded
        exp_f = _expanded.transpose(3, 0, 1, 2)  # (F, n_nbhd, lat, lon)

        # Float32 copy of _above for matmul — written into pre-allocated buffer.
        _above_f32[:] = _above
        above_flat = _above_f32.reshape(n_fronts, n_lat * n_lon, N_THRESHOLDS)
        exp_flat = exp_f.reshape(n_fronts, n_nbhd, n_lat * n_lon)
        tp_ag += np.matmul(exp_flat * w_flat, above_flat)
        fp_ag += np.matmul((~exp_flat) * w_flat, above_flat)

        # Spatial TP/FP: per-neighbourhood in-place accumulation.
        for ni in range(n_nbhd):
            exp_ni = exp_f[:, ni, :, :, np.newaxis]  # (F, lat, lon, 1) — tiny view
            np.logical_and(_above, exp_ni, out=_bool_buf)
            np.multiply(_bool_buf, w_4d, out=_float_buf)
            tp_sp[:, :, :, ni, :] += _float_buf
            np.logical_and(_above, ~exp_ni, out=_bool_buf)
            np.multiply(_bool_buf, w_4d, out=_float_buf)
            fp_sp[:, :, :, ni, :] += _float_buf

        t_stats = time.perf_counter()

        if t < 3:
            log.info(
                "t=%d: load=%.3fs  infer=%.3fs  stats=%.3fs  total=%.3fs",
                t,
                t_load - t0,
                t_infer - t_load,
                t_stats - t_infer,
                t_stats - t0,
            )

    spatial_ds = xr.Dataset(
        coords={
            "latitude": lats,
            "longitude": lons,
            "neighborhood": NEIGHBORHOODS_KM,
            "threshold": THRESHOLDS,
        }
    )
    aggregate_ds = xr.Dataset(
        coords={
            "neighborhood": NEIGHBORHOODS_KM,
            "threshold": THRESHOLDS,
        }
    )
    dims_sp = ("latitude", "longitude", "neighborhood", "threshold")
    dims_ag = ("neighborhood", "threshold")
    for fi, ft in enumerate(front_types):
        spatial_ds[f"tp_spatial_{ft}"] = (dims_sp, tp_sp[fi])
        spatial_ds[f"fp_spatial_{ft}"] = (dims_sp, fp_sp[fi])
        spatial_ds[f"tn_spatial_{ft}"] = (dims_sp, tn_sp[fi])
        spatial_ds[f"fn_spatial_{ft}"] = (dims_sp, fn_sp[fi])
        aggregate_ds[f"tp_{ft}"] = (dims_ag, tp_ag[fi])
        aggregate_ds[f"fp_{ft}"] = (dims_ag, fp_ag[fi])
        aggregate_ds[f"tn_{ft}"] = (dims_ag, tn_ag[fi])
        aggregate_ds[f"fn_{ft}"] = (dims_ag, fn_ag[fi])

    return spatial_ds, aggregate_ds


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

    eval_cfg: config.EvalConfig = utils.open_config_yaml_as_dataclass(
        args.config_path,
        config.EvalConfig,
        config_key="eval_config",
        type_hooks={
            utils.BoundingBox: lambda d: utils.BoundingBox(*d),
            datetime.datetime: lambda d: datetime.datetime.fromisoformat(str(d)),
        },
    )
    data_cfg: config.DataConfig = utils.open_config_yaml_as_dataclass(
        args.config_path, config.DataConfig, config_key="data_config"
    )

    eval_cfg = dataclasses.replace(
        eval_cfg,
        mask=args.mask if args.mask is not None else eval_cfg.mask,
        outdir=args.outdir if args.outdir is not None else eval_cfg.outdir,
    )

    import tensorflow as tf

    utils.configure_gpu(eval_cfg.gpu_device)
    log.info("Loading model from %s …", eval_cfg.model_path)
    model = tf.keras.models.load_model(eval_cfg.model_path, compile=False)
    log.info("Model loaded. Output count: %d.", len(model.outputs))

    ic_era5 = data_cfg.era5_icechunk_config
    ic_fronts = data_cfg.fronts_icechunk_config

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

    era5_da = inputs.era5_to_dataarray(era5_ds, data_cfg.variables)
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
        model=model,
        era5_da=era5_da,
        targets_da=targets_da,
        front_types=eval_cfg.front_types,
        lats=lats,
        lons=lons,
        spatial_mask=spatial_mask,
    )

    os.makedirs(eval_cfg.outdir, exist_ok=True)
    mask_suffix = f"_{eval_cfg.mask}" if eval_cfg.mask else ""
    spatial_path = os.path.join(eval_cfg.outdir, f"stats_spatial{mask_suffix}.nc")
    aggregate_path = os.path.join(eval_cfg.outdir, f"stats_aggregate{mask_suffix}.nc")

    spatial_ds.astype("float32").to_netcdf(spatial_path)
    aggregate_ds.astype("float32").to_netcdf(aggregate_path)
    log.info("Spatial stats   → %s", spatial_path)
    log.info("Aggregate stats → %s", aggregate_path)


if __name__ == "__main__":
    main()
