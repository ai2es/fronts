"""Train UNet3Plus on ERA5 to detect weather fronts.

Raw data is fed to the model; a Keras Normalization layer (adapted on training
patches) is prepended and its mean/std are baked into the saved weights.

Supports distributed training across multiple GPUs via TensorFlow's MirroredStrategy.
Metrics are logged to Weights & Biases.
"""

import argparse
import dataclasses
import logging
import os
import random
import time
from typing import Literal

import dask
import numpy as np
import tensorflow as tf
import wandb
import xarray as xr

from fronts import callbacks as fronts_callbacks
from fronts import constants, model, utils
from fronts.data import datasets, inputs, targets
from fronts.layers import losses, metrics
from fronts.utils import apply_time_resolution

logger = logging.getLogger(__name__)


def _get_distribution_strategy() -> tf.distribute.Strategy:
    """Get TensorFlow distribution strategy based on available GPUs.

    Uses MirroredStrategy directly — it auto-detects all available GPUs and
    falls back gracefully to a single device when only one is present.

    Returns:
        MirroredStrategy across all visible GPUs.
    """
    strategy = tf.distribute.MirroredStrategy()
    num_gpus = strategy.num_replicas_in_sync
    logger.info(f"Detected {num_gpus} GPU(s). Using MirroredStrategy.")
    return strategy


@dataclasses.dataclass
class WandBConfig:
    """W&B project and run naming configuration."""

    # Passed straight to wandb.keras.WandbMetricsLogger(log_freq=...): "epoch" logs once per
    # epoch, "batch" logs every batch, or an int N logs every N batches. With long epochs
    # (thousands of steps), "epoch" can leave W&B silent for a very long time.
    log_freq: str | int
    project_name: str = "fronts"
    run_name: str | None = None


@dataclasses.dataclass
class TrainConfig:
    """Training-specific hyperparameters.

    Attributes:
        loss_class_weights: Class weights applied inside the training loss, or None to supervise all
            classes (including background) equally. The reference FrontFinder model trained with
            None — zero-weighting background in the loss leaves ~95% of pixels unsupervised and
            lets predicted probabilities drift toward uniform. Metrics use
            ``data_config.class_weights`` independently of this value.
        gradient_clip_norm: Per-gradient L2-norm clip passed to the Adam optimizer, or None to
            leave gradients unclipped. Caps the damage a single outlier batch can do — one
            unclipped spike can wipe learned features and knock training into a worse basin
            it never escapes.
        loss_name: Which training loss to compile with. "fractions_skill_score" uses a
            fixed pixel-neighborhood mask (see ``fss_mask_size``); "neighborhood_brier_score"
            uses a physical-distance tolerance instead (see ``nbs_tolerance_km`` and related
            fields) and does not require label dilation to express positional slack.
        fss_mask_size: ``mask_size`` passed to ``losses.fractions_skill_score``. Only used
            when ``loss_name == "fractions_skill_score"``.
        nbs_tolerance_km: ``tolerance_km`` passed to ``losses.neighborhood_brier_score``.
            Only used when ``loss_name == "neighborhood_brier_score"``.
        nbs_periodic_lon: ``periodic_lon`` passed to ``losses.neighborhood_brier_score``.
            Only used when ``loss_name == "neighborhood_brier_score"``.
        nbs_lat_dependent_pool: ``lat_dependent_pool`` passed to
            ``losses.neighborhood_brier_score``. Only used when
            ``loss_name == "neighborhood_brier_score"``.
        nbs_include_pixel: ``include_pixel`` passed to ``losses.neighborhood_brier_score`` — adds
            an un-pooled, per-pixel Brier term alongside the neighborhood-pooled one. The pooled
            term alone only constrains the *neighborhood-averaged* forecast fraction, so it gives
            very little gradient pressure to sharpen any single pixel's probability once it's
            already moderately high (the marginal squared-error gain of 0.8 -> 0.99 is tiny and
            further diluted by averaging over the pooling window) — this is why pointwise
            reliability curves (see ``evaluate.obs_rel_freq_pointwise``) tend to plateau well
            below 100% forecast probability. The pixelwise term scores each pixel directly,
            undiluted by pooling, to counteract that. Only used when
            ``loss_name == "neighborhood_brier_score"``.
        nbs_pixel_weight: ``pixel_weight`` passed to ``losses.neighborhood_brier_score`` — the
            pixelwise term's weight relative to the pooled term (which is fixed at 1). Only used
            when ``nbs_include_pixel`` is True.
        use_ema: Whether the optimizer tracks an exponential moving average (Polyak averaging) of
            the model's trainable weights, swapped into the model for validation, checkpointing,
            and early-stopping's best-weight snapshot via a ``SwapEMAWeights(swap_on_epoch=True)``
            callback (see ``_run``). Averaging over recent weight updates trades a small amount of
            fit-to-the-latest-batch for a flatter, less noise-sensitive optimum — cheap to enable
            since it adds no extra forward/backward passes.
        ema_momentum: Decay rate for the weight EMA (Keras default: 0.99, an ~100-step effective
            averaging window). Only used when ``use_ema`` is True.
        per_front_type_metrics: Whether to attach the per-front-type metric set (soft/hard HSS,
            CSI, POD from ``metrics.per_front_type_metrics``, plus the per-front-type loss
            decomposition from ``_per_front_type_loss_metrics``) to the finest model output. See
            ``_compile``. Reporting only — never affects ``loss=`` or gradients.
    """

    loss_class_weights: list[float] | None
    epochs: int = 50
    seed: int = 42
    learning_rate: float = 1e-4
    shuffle: bool = False
    gradient_clip_norm: float | None = None
    loss_name: Literal["fractions_skill_score", "neighborhood_brier_score", "multiclass_wbce"] = (
        "neighborhood_brier_score"
    )
    fss_mask_size: tuple[int, ...] = (3, 3)
    nbs_tolerance_km: float = 25.0
    nbs_periodic_lon: bool = False
    nbs_lat_dependent_pool: bool = False
    nbs_include_pixel: bool = False
    nbs_pixel_weight: float = 0.1
    use_ema: bool = False
    ema_momentum: float = 0.99
    # Defaulted (contrary to the usual no-defaults rule for dataclasses) so the 17 existing
    # YAML configs keep parsing: dacite raises on a missing required field.
    per_front_type_metrics: bool = True


def load_data_into_dataloader(
    data_config: datasets.DatasetConfig,
    split: Literal["train", "val", "test"],
    seed: int = 0,
    shuffle: bool = False,
    drop_remainder: bool = False,
) -> datasets.FrontsPyDataset:
    """Load, align, and encode ERA5 input and fronts data for training.

    Opens the ERA5 and fronts icechunk stores once each with ``chunks=None`` so
    TrainingDataset's per-batch ``isel(...).values`` reads go straight through the
    zarr store with no dask graph, deduplicates time indexes, aligns both to the
    intersection of available timestamps, and returns lazy DataArrays ready for
    batching. The dask-backed arrays needed for the full-training-set
    normalization-stats reduction (which needs dask to chunk that reduction
    instead of materializing everything in RAM at once) are derived from the same
    arrays via a cheap, metadata-only ``.chunk("auto")`` rather than a second
    store open.

    Args:
        data_config: DatasetConfig specifying store paths, branch names, and splits.
        split: Type of dataset to load ("train", "val", "test").
        seed: Integer seed for the RNG used when subsampling timesteps.
        shuffle: If True, reshuffles the sample order at the end of every epoch.
        drop_remainder: If True, drop the final under-sized batch each epoch so every
            batch has exactly ``data_config.batch_size`` samples — see
            ``datasets.FrontsPyDataset`` for why this matters under multi-GPU training.
        workers: Number of ``PyDataset`` prefetch threads. 1 (the ``PyDataset``
            default) fetches each batch synchronously on the main thread, serializing
            every batch's icechunk read with the GPU training step.

    Returns:
        FrontsPyDataset yielding batches of (input, target) pairs for training.
    """

    def _open(icechunk_config: utils.IcechunkStorageConfig) -> xr.Dataset:
        ds = utils.open_readonly_icechunk_store(
            store_path=icechunk_config.store_path,
            branch=icechunk_config.branch_name,
            group=icechunk_config.group_name,
            zarr_format=icechunk_config.zarr_format,
            virtual_chunk_local_path=icechunk_config.virtual_chunk_local_path,
            chunks=None,
        )
        # A wrap-crossing bounding box (lon_max > 360) leaves longitude non-monotonic
        # on disk (e.g. [130, ..., 359.75, 0, ..., 9.75]); downstream plotting
        # (TestVisualizationCallback) and region masking assume it's monotonic.
        return utils.unwrap_longitude(ds)

    logger.info("Loading %s inputs...", split)
    inputs_ds = _open(data_config.inputs_icechunk_config)
    if data_config.pressure_levels is not None:
        logger.info("Restricting to pressure levels: %s", data_config.pressure_levels)
        inputs_ds = utils.select_pressure_levels(inputs_ds, data_config.pressure_levels)

    logger.info("Loading %s targets...", split)
    targets_da = _open(data_config.targets_icechunk_config)["identifier"]

    if data_config.coordinates is not None:
        logger.info("Restricting to spatial domain: %s", data_config.coordinates)
        inputs_ds = utils.select_spatial_domain(inputs_ds, data_config.coordinates)
        targets_da = utils.select_spatial_domain(targets_da, data_config.coordinates)

    # The time indexes aren't identical between the two datasets
    common_times = np.intersect1d(targets_da.time.values, inputs_ds.time.values)

    # Subset to the time resolution; defaults to 6 hourly to match full USAD domain fronts data frequency
    common_times = apply_time_resolution(common_times, data_config.time_resolution)
    logger.info("After time_resolution=%s filter: %d steps", data_config.time_resolution, len(common_times))

    # Class-balancing subsample (drop ~50% of cases without all fronts in the domain) applies
    # to train/val, which both feed model selection; test must stay untouched for honest,
    # unbiased evaluation (see _build_test_visualization_callback).
    if split != "test":
        rng = np.random.default_rng(seed)
        keep = targets.filter_timesteps(targets_da.sel(time=common_times), rng)
        common_times = common_times[keep]
    logger.info(f"Matched time steps: {len(common_times)}")
    inputs_ds_matched = inputs_ds.sel(time=common_times)
    targets_da_matched = targets_da.sel(time=common_times)

    # Get years for splitting data
    train_mask, val_mask, test_mask = utils.split_by_year(
        times=inputs_ds_matched.time.values, test_years=data_config.test_years, val_years=data_config.val_years
    )
    split_mask = {"train": train_mask, "val": val_mask, "test": test_mask}[split]
    split_indices = sorted(np.where(split_mask)[0].tolist())
    logger.info("Split indices: %d timesteps for %s", len(split_indices), split)
    inputs_ds = inputs_ds_matched.isel(time=split_indices)
    targets_da = targets_da_matched.isel(time=split_indices)
    logger.info(
        "%s split: %d timesteps, %d inputs, %d targets",
        split,
        len(split_indices),
        len(inputs_ds.time),
        len(targets_da.time),
    )
    # Get the number of threads to use for PyDataset prefetching from max_pydataset_workers in the DatasetConfig,
    # which is set to 16 by default. This allows for parallel loading of batches without overwhelming ourdisk I/O.
    data_workers = utils.limit_workers_for_slurm(max_workers=data_config.max_pydataset_workers)
    return datasets.FrontsPyDataset(
        input_ds=inputs_ds,
        target_da=targets_da,
        data_config=data_config,
        seed=seed,
        batch_size=data_config.batch_size,
        shuffle=shuffle,
        workers=data_workers,
        max_queue_size=data_config.max_queue_size,
        drop_remainder=drop_remainder,
    )


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _show_input_sample(label: str, inputs: np.ndarray | xr.DataArray, n_show: int = 5) -> None:
    patch = np.asarray(inputs[0, :, :, :])
    n_channels = patch.shape[-1]
    logger.info(f"\n  {label} — first patch stats (first {n_show} of {n_channels} channels):")
    logger.info(f"  {'channel':<10} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    logger.info(f"  {'-' * 52}")
    for i in range(n_show):
        ch = patch[..., i]
        logger.info(f"  {i:<10} {ch.mean():>10.3f} {ch.std():>10.3f} {ch.min():>10.3f} {ch.max():>10.3f}")


def _build_loss(
    loss_name: Literal["fractions_skill_score", "neighborhood_brier_score", "multiclass_wbce"],
    loss_class_weights: list[float] | None,
    latitudes: np.ndarray,
    fss_mask_size: tuple[int, ...],
    nbs_tolerance_km: float,
    nbs_periodic_lon: bool,
    nbs_lat_dependent_pool: bool,
    nbs_include_pixel: bool = False,
    nbs_pixel_weight: float = 0.1,
):
    """Build the configured training loss.

    Args:
        loss_name: "fractions_skill_score" or "neighborhood_brier_score" — see
            ``TrainConfig`` for what each option means.
        loss_class_weights: Per-class loss weights, or None to supervise all classes equally.
        latitudes: 1-D grid latitudes. Only used by "neighborhood_brier_score".
        fss_mask_size: Pooling mask size. Only used by "fractions_skill_score".
        nbs_tolerance_km: Positional tolerance in km. Only used by "neighborhood_brier_score".
        nbs_periodic_lon: Whether longitude wraps. Only used by "neighborhood_brier_score".
        nbs_lat_dependent_pool: Whether to use latitude-dependent pooling. Only used by
            "neighborhood_brier_score".
        nbs_include_pixel: Whether to add the un-pooled pixelwise Brier term. Only used by
            "neighborhood_brier_score".
        nbs_pixel_weight: Relative weight of the pixelwise term when ``nbs_include_pixel`` is
            True. Only used by "neighborhood_brier_score".

    Returns:
        A callable loss function suitable for ``model.compile(loss=...)``.

    Raises:
        ValueError: If ``loss_name`` is not a recognized loss.
    """
    if loss_name == "fractions_skill_score":
        return losses.fractions_skill_score(mask_size=fss_mask_size, class_weights=loss_class_weights)
    if loss_name == "neighborhood_brier_score":
        return losses.neighborhood_brier_score(
            latitudes=latitudes,
            tolerance_km=nbs_tolerance_km,
            class_weights=loss_class_weights,
            periodic_lon=nbs_periodic_lon,
            lat_dependent_pool=nbs_lat_dependent_pool,
            include_pixel=nbs_include_pixel,
            pixel_weight=nbs_pixel_weight,
        )
    if loss_name == "multiclass_wbce":
        return losses.multiclass_wbce_loss(class_weights=loss_class_weights)
    raise ValueError(
        f"Unrecognized loss_name {loss_name!r}; expected 'fractions_skill_score' or 'neighborhood_brier_score'."
    )


def _per_front_type_loss_metrics(
    loss_name: Literal["fractions_skill_score", "neighborhood_brier_score", "multiclass_wbce"],
    loss_class_weights: list[float] | None,
    latitudes: np.ndarray,
    fss_mask_size: tuple[int, ...],
    nbs_tolerance_km: float,
    nbs_periodic_lon: bool,
    nbs_lat_dependent_pool: bool,
    nbs_include_pixel: bool,
    nbs_pixel_weight: float,
    n_classes: int,
) -> list[tf.keras.metrics.Metric]:
    """Build one loss-valued metric per front type (plus background) for reporting.

    Cost note: this builds ``n_classes`` independent ``_build_loss`` calls (one per front type
    plus background), each of which — for both ``"fractions_skill_score"`` and
    ``"neighborhood_brier_score"`` — pools the *entire* ``(batch, H, W, n_classes)`` tensor at
    full resolution before zeroing out every class but one (class weights are applied AFTER
    pooling in both losses; see ``losses.neighborhood_brier_score``'s docstring on pooling
    dominating backward-pass memory). Combined with the compiled loss's own pass, that's
    ``n_classes + 1`` full-resolution poolings per batch on the finest (largest) model output.
    This is forward-only reporting cost — it adds no backward pass and does not touch gradients
    — but it is real, avoidable compute. Sharing one pooling across these independent
    ``tf.keras.metrics.Metric`` objects would require threading pooled intermediates through the
    constraint-sensitive loss path (see the REPORTING ONLY constraint in ``TrainConfig``), so
    this was deliberately deferred rather than fixed; a future maintainer profiling the training
    loop should look here first, not rediscover it via a profiler.

    Each metric rebuilds the configured training loss (the same ``_build_loss`` call used
    for the compiled ``loss=`` argument) with a one-hot ``loss_class_weights`` vector that
    isolates a single class, then wraps it in ``tf.keras.metrics.MeanMetricWrapper`` for a
    stable ``.name`` (``hss_hard`` in ``_compile`` is the existing precedent for this
    pattern). These metrics are REPORTING ONLY — none of them is the compiled ``loss=``
    argument, so none contributes a gradient or changes the optimized scalar.

    Whether the six reported values (five front types plus background) sum exactly to the
    total configured loss depends on how that loss normalizes ``class_weights``, which
    differs between the two options this project supports:

    ``neighborhood_brier_score`` (``losses.neighborhood_brier_score``) applies
    ``class_weights`` directly and unnormalized (``se *= cw`` inside its inner ``_brier``
    helper), and every term of the loss — the pooled Brier term and, when
    ``include_pixel`` is set, the pixelwise term — is a mean of ``se * cw`` over the
    spatial+class axes. That mean is linear in ``cw``: for any two weight vectors u, v and
    scalars p, q, ``loss(p*u + q*v) == p*loss(u) + q*loss(v)``, because ``cw`` only ever
    enters as a multiplicative factor before a sum. Building each per-class metric's
    one-hot vector as ``cw[c] * one_hot(c)`` (using 1.0 in place of ``cw[c]`` when
    ``loss_class_weights`` is None, matching that loss's "supervise all classes equally"
    behavior) therefore makes ``sum_c loss(cw[c] * one_hot(c)) == loss(sum_c cw[c] *
    one_hot(c)) == loss(cw)``, i.e. the per-class values sum EXACTLY to the total loss, on
    every input — including when some ``cw[c]`` is 0 (that class's reported loss is then
    exactly 0, its true contribution). This is pinned by
    ``TestPerFrontTypeLossMetrics.test_nbs_per_class_losses_sum_to_total``.

    ``fractions_skill_score`` (``losses.fractions_skill_score``) renormalizes internally
    (``relative_cw = cw / sum(cw)``, an unguarded division — a zero-sum ``cw`` divides by
    zero) and returns a RATIO of two weighted sums (``1 - MSE_n / MSE_ref``), not a
    weighted sum itself. A ratio of sums is not, in general, a sum of ratios: writing
    ``a_c`` and ``b_c`` for class c's contributions to the numerator and denominator sums,
    ``sum_c cw[c] * (a_c / b_c) != (sum_c cw[c] * a_c) / (sum_c cw[c] * b_c)`` unless every
    ``b_c`` happens to be equal. So no choice of one-hot scaling can make per-class FSS
    losses sum to the total without changing the loss itself, which the reporting-only
    constraint forbids. Every per-class FSS metric here therefore uses a plain, UNSCALED
    one-hot vector (weight 1.0, regardless of that class's configured weight): this avoids
    the zero-sum-``cw`` NaN (a one-hot vector always sums to 1, never 0) and yields each
    class's own, self-normalized FSS loss ``a_c / b_c``, i.e. the FSS loss computed as if
    only that class existed. The weaker property that DOES hold: writing the total as
    ``sum_c lambda_c * (a_c / b_c)`` with ``lambda_c = (relative_cw[c] * b_c) / sum(relative_cw
    * b)`` (a valid convex combination — ``lambda_c >= 0`` and ``sum_c lambda_c == 1``,
    since both ``relative_cw`` and each ``b_c`` are non-negative), the total FSS loss is
    always a data-dependent weighted average of the reported per-class FSS losses, and so
    always lies between their min and max: ``min_c loss_c <= total_loss <= max_c loss_c``.
    Pinned by ``TestPerFrontTypeLossMetrics.test_fss_per_class_losses_bound_total``.

    Args:
        loss_name: Which training loss to rebuild per class — see ``TrainConfig.loss_name``.
        loss_class_weights: The whole-vector class weights passed to the compiled loss, or
            None to supervise all classes equally (weight 1.0 each).
        latitudes: 1-D grid latitudes, forwarded to ``_build_loss``.
        fss_mask_size: Forwarded to ``_build_loss``. Only used by "fractions_skill_score".
        nbs_tolerance_km: Forwarded to ``_build_loss``. Only used by "neighborhood_brier_score".
        nbs_periodic_lon: Forwarded to ``_build_loss``. Only used by "neighborhood_brier_score".
        nbs_lat_dependent_pool: Forwarded to ``_build_loss``. Only used by
            "neighborhood_brier_score".
        nbs_include_pixel: Forwarded to ``_build_loss``. Only used by
            "neighborhood_brier_score".
        nbs_pixel_weight: Forwarded to ``_build_loss``. Only used when ``nbs_include_pixel``
            is True.
        n_classes: The model's configured class count (``model.ModelConfig.n_classes``). Must
            equal ``max(constants.FRONT_TYPE_CLASS_INDEX.values()) + 1`` (background plus every
            mapped front type); a mismatch raises immediately rather than building a mis-shaped
            weight vector that would fail with a confusing shape error deep inside
            ``_build_loss``.

    Returns:
        One ``MeanMetricWrapper`` per front type (``loss_{ft}``, e.g. ``loss_CF``) plus one
        for the background class (``loss_{BACKGROUND_CLASS_KEY}``, i.e. ``loss_none``).

    Raises:
        ValueError: If ``n_classes`` does not match the class count implied by
            ``constants.FRONT_TYPE_CLASS_INDEX``.
    """
    expected_n_classes = max(constants.FRONT_TYPE_CLASS_INDEX.values()) + 1
    if n_classes != expected_n_classes:
        raise ValueError(
            f"n_classes={n_classes} (from model.ModelConfig.n_classes) does not match the "
            f"{expected_n_classes} classes implied by constants.FRONT_TYPE_CLASS_INDEX "
            f"(background plus {sorted(constants.FRONT_TYPE_CLASS_INDEX)}). Per-front-type loss "
            "metrics require these to agree — update constants.FRONT_TYPE_CLASS_INDEX or the "
            "model's n_classes so they match."
        )
    effective_weights = (
        np.ones(n_classes, dtype=np.float32)
        if loss_class_weights is None
        else np.asarray(loss_class_weights, dtype=np.float32)
    )
    class_index_by_key = {constants.BACKGROUND_CLASS_KEY: 0, **constants.FRONT_TYPE_CLASS_INDEX}

    metrics_list: list[tf.keras.metrics.Metric] = []
    for key, class_index in class_index_by_key.items():
        one_hot_weights = np.zeros(n_classes, dtype=np.float32)
        one_hot_weights[class_index] = 1.0 if loss_name == "fractions_skill_score" else effective_weights[class_index]
        loss_fn = _build_loss(
            loss_name=loss_name,
            loss_class_weights=one_hot_weights.tolist(),
            latitudes=latitudes,
            fss_mask_size=fss_mask_size,
            nbs_tolerance_km=nbs_tolerance_km,
            nbs_periodic_lon=nbs_periodic_lon,
            nbs_lat_dependent_pool=nbs_lat_dependent_pool,
            nbs_include_pixel=nbs_include_pixel,
            nbs_pixel_weight=nbs_pixel_weight,
        )
        metrics_list.append(tf.keras.metrics.MeanMetricWrapper(fn=loss_fn, name=f"loss_{key}"))
    return metrics_list


def _load_pretrained_weights(
    unet: tf.keras.Model,
    pretrained_weights_path: str,
    normalization_stat_a: np.ndarray,
    normalization_stat_b: np.ndarray,
) -> None:
    """Warm-start unet's weights from a prior checkpoint, keeping this run's normalization stats.

    Mismatched layers are skipped rather than raising, so a checkpoint whose supervision heads differ
    in shape or count (e.g. a different `levels`) still transfers its shared encoder/decoder weights.
    The `input_normalization` layer is then reset to `normalization_stat_a`/`normalization_stat_b` —
    the stats already computed for this run's domain in `train()` — since the checkpoint's own
    normalization weights reflect its own (possibly different) training domain.

    Args:
        unet: Freshly built model to load weights into, in place.
        pretrained_weights_path: Path to a saved ``.keras`` checkpoint.
        normalization_stat_a: This run's normalization stat A (mean or min), as passed to
            ``model.UNet3Plus``.
        normalization_stat_b: This run's normalization stat B (variance or max).
    """
    logger.info("Warm-starting model weights from %s", pretrained_weights_path)
    unet.load_weights(pretrained_weights_path, skip_mismatch=True)
    norm_layer = unet.get_layer("input_normalization")
    if isinstance(norm_layer, tf.keras.layers.Normalization):
        norm_layer.mean.assign(normalization_stat_a)
        norm_layer.variance.assign(normalization_stat_b)
    elif isinstance(norm_layer, tf.keras.layers.Rescaling):
        channel_range = normalization_stat_b - normalization_stat_a
        scale = 1.0 / np.where(channel_range == 0, 1.0, channel_range)
        norm_layer.scale = scale.astype(np.float32)
        norm_layer.offset = (-normalization_stat_a * scale).astype(np.float32)
    logger.info("Reset input_normalization to this run's domain-specific stats.")


def _freeze_layers(unet: tf.keras.Model, freeze_layer_prefixes: list[str]) -> None:
    """Set layer.trainable = False for every layer whose name starts with any given prefix.

    Args:
        unet: Model whose layers will be frozen in place.
        freeze_layer_prefixes: Layer name prefixes to match via ``layer.name.startswith(prefix)``.
    """
    frozen = 0
    for layer in unet.layers:
        if any(layer.name.startswith(prefix) for prefix in freeze_layer_prefixes):
            layer.trainable = False
            frozen += 1
    logger.info("Froze %d/%d layers matching prefixes %s", frozen, len(unet.layers), freeze_layer_prefixes)


def _compile(
    model: tf.keras.Model,
    learning_rate: float,
    metric_class_weights: list[float] | None,
    train_cfg: "TrainConfig",
    latitudes: np.ndarray,
    n_classes: int,
    gradient_clip_norm: float | None = None,
) -> int:
    """Compile ``model`` with the configured loss, HSS metrics, and optional per-front-type metrics.

    Args:
        model: The (possibly strategy-scoped) model to compile in place.
        learning_rate: Adam learning rate.
        metric_class_weights: Class weights for the reported HSS metrics (independent of the
            training loss's own class weights).
        train_cfg: Training hyperparameters — see ``TrainConfig``.
        latitudes: 1-D grid latitudes, forwarded to ``_build_loss``.
        n_classes: The model's configured class count (``model.ModelConfig.n_classes``),
            forwarded to ``_per_front_type_loss_metrics`` for validation against
            ``constants.FRONT_TYPE_CLASS_INDEX``. Only used when
            ``train_cfg.per_front_type_metrics`` is True.
        gradient_clip_norm: Per-gradient L2-norm clip, or None to leave gradients unclipped.

    Returns:
        The number of model outputs (``len(model.outputs)``).
    """
    n_out = len(model.outputs)
    loss_fn = _build_loss(
        loss_name=train_cfg.loss_name,
        loss_class_weights=train_cfg.loss_class_weights,
        latitudes=latitudes,
        fss_mask_size=train_cfg.fss_mask_size,
        nbs_tolerance_km=train_cfg.nbs_tolerance_km,
        nbs_periodic_lon=train_cfg.nbs_periodic_lon,
        nbs_lat_dependent_pool=train_cfg.nbs_lat_dependent_pool,
        nbs_include_pixel=train_cfg.nbs_include_pixel,
        nbs_pixel_weight=train_cfg.nbs_pixel_weight,
    )
    hss_fn = metrics.heidke_skill_score(class_weights=metric_class_weights)
    hss_hard_fn = tf.keras.metrics.MeanMetricWrapper(
        fn=metrics.heidke_skill_score(class_weights=metric_class_weights, threshold=0.5),
        name="hss_hard",
    )
    per_class: list[tf.keras.metrics.Metric] = []
    if train_cfg.per_front_type_metrics:
        per_class = metrics.per_front_type_metrics(constants.FRONT_TYPE_CLASS_INDEX) + _per_front_type_loss_metrics(
            loss_name=train_cfg.loss_name,
            loss_class_weights=train_cfg.loss_class_weights,
            latitudes=latitudes,
            fss_mask_size=train_cfg.fss_mask_size,
            nbs_tolerance_km=train_cfg.nbs_tolerance_km,
            nbs_periodic_lon=train_cfg.nbs_periodic_lon,
            nbs_lat_dependent_pool=train_cfg.nbs_lat_dependent_pool,
            nbs_include_pixel=train_cfg.nbs_include_pixel,
            nbs_pixel_weight=train_cfg.nbs_pixel_weight,
            n_classes=n_classes,
        )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=gradient_clip_norm,
        use_ema=train_cfg.use_ema,
        ema_momentum=train_cfg.ema_momentum,
    )
    if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[[hss_fn, hss_hard_fn, *per_class]] + [[hss_fn, hss_hard_fn]] * (n_out - 1),
    )
    return n_out


def _build_monitor_callbacks(
    monitor: str,
    patience: int,
    learning_rate_decay_factor: float | None,
    learning_rate_minimum: float | None,
    min_delta: float = 0.0,
    early_stopping_patience: int | None = None,
) -> list[tf.keras.callbacks.Callback]:
    """Builds the plateau-monitoring callbacks: LR decay, early stopping, or both.

    With both LR-decay params set, a ReduceLROnPlateau (patience=``patience``) is
    returned, plus an EarlyStopping when ``early_stopping_patience`` is also set —
    the stopping patience should exceed the decay patience by a few multiples so
    LR reductions get a chance to rescue a plateau before the run ends. Without
    the LR-decay params, a single EarlyStopping with ``patience`` is returned.

    Args:
        monitor: Metric name to monitor.
        patience: Number of epochs with no improvement before decaying the learning
            rate (or, when LR decay is disabled, before stopping).
        learning_rate_decay_factor: Factor to multiply the current learning rate by
            on plateau, or None to disable LR decay.
        learning_rate_minimum: Lower bound on the learning rate when decaying, or
            None to disable LR decay.
        min_delta: Smallest monitored-value change that counts as an improvement. Must be
            explicit rather than Keras's default: ReduceLROnPlateau defaults to an ABSOLUTE
            1e-4, which dwarfs real improvements when the loss itself is only ~1e-3 (as the
            neighborhood Brier loss is), decaying the learning rate to its floor within a
            dozen epochs and silently freezing training.
        early_stopping_patience: Number of epochs with no improvement before ending the
            run when LR decay is active. None disables stopping in that mode, in which
            case the run continues until ``epochs`` or the job walltime.

    Returns:
        List of one or two callbacks.
    """
    if learning_rate_decay_factor is not None and learning_rate_minimum is not None:
        monitor_callbacks: list[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=learning_rate_decay_factor,
                patience=patience,
                min_lr=learning_rate_minimum,
                min_delta=min_delta,
            )
        ]
        if early_stopping_patience is not None:
            monitor_callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=monitor,
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                    min_delta=min_delta,
                )
            )
        return monitor_callbacks
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True, min_delta=min_delta
        )
    ]


def _optimizer_uses_ema(optimizer: tf.keras.optimizers.Optimizer) -> bool:
    """Whether optimizer has weight EMA enabled, unwrapping a mixed-precision LossScaleOptimizer.

    Args:
        optimizer: The model's (possibly LossScaleOptimizer-wrapped) optimizer.

    Returns:
        True if the underlying optimizer was built with ``use_ema=True``.
    """
    inner_optimizer = getattr(optimizer, "inner_optimizer", optimizer)
    return bool(getattr(inner_optimizer, "use_ema", False))


def _resolve_metrics_csv_path(metrics_csv_path: str | None, model_checkpoint_path: str | None) -> str | None:
    """Resolves the path the per-epoch metrics CSV should be written to.

    Args:
        metrics_csv_path: Explicit path from ``CallbacksConfig.metrics_csv_path``, or None to
            derive one from ``model_checkpoint_path``.
        model_checkpoint_path: Path prefix for the best-loss checkpoint, or None.

    Returns:
        ``metrics_csv_path`` if given; otherwise ``metrics_epoch.csv`` next to
        ``model_checkpoint_path``; otherwise None (CSV logging skipped) when both are None.
    """
    if metrics_csv_path is not None:
        return metrics_csv_path
    if model_checkpoint_path is not None:
        return os.path.join(os.path.dirname(model_checkpoint_path), "metrics_epoch.csv")
    return None


def _csv_logger_fieldnames(logs: dict) -> list[str]:
    """Replicates ``tf.keras.callbacks.CSVLogger``'s column-selection logic for one epoch's logs.

    Mirrors ``CSVLogger.on_epoch_end``'s ``self.keys`` computation exactly (sorted log keys, with
    a ``val_``-prefixed mirror appended when no ``val_`` key is already present), so the header a
    real ``CSVLogger`` would write for this run can be computed and compared against an existing
    file's header before deciding whether to append to it. See ``_ResumeSafeCSVLogger``.

    Args:
        logs: The epoch logs dict, as passed to a callback's ``on_epoch_end``.

    Returns:
        The full CSV fieldnames list (``["epoch", *keys]``) ``CSVLogger`` would use.
    """
    keys = sorted(logs.keys())
    if keys and not any(key.startswith("val_") for key in keys):
        keys = keys + [f"val_{key}" for key in keys]
    return ["epoch", *keys]


class _ResumeSafeCSVLogger(tf.keras.callbacks.Callback):
    """Wraps ``CSVLogger`` to avoid silently appending rows under a mismatched header.

    ``CSVLogger(path, append=True)`` fixes its column set from the FIRST epoch's ``logs`` and
    skips writing a header whenever the target file already exists and is non-empty. Resuming a
    run whose metric set changed since that file was last written (e.g.
    ``TrainConfig.per_front_type_metrics`` toggled, or ``TrainConfig.loss_name`` changed) then
    silently writes new rows under the OLD header, misaligning ``metrics_epoch.csv`` — a file
    that exists specifically to be a durable, independently-plottable record, so silent
    misalignment defeats its entire purpose.

    The exact column set a run will produce is only known once its first epoch's logs are
    available (``CSVLogger`` itself has no earlier knowledge of it either — it derives ``self.keys``
    from ``logs.keys()`` on the first ``on_epoch_end`` call), so this wrapper defers building the
    real ``CSVLogger`` until that first call. At that point, if the target file already exists, is
    non-empty, and its header row does not match this run's computed fieldnames (see
    ``_csv_logger_fieldnames``), the old file is rotated aside under a numeric ``.stale-N`` suffix
    and a warning is logged explaining what happened and where the old file went, before a fresh
    file (and header) is started. When the header matches — the genuine same-shape resume case —
    or no file exists yet, behavior is identical to ``CSVLogger(path, append=True)``.

    Attributes:
        path: The CSV file path passed to the underlying ``CSVLogger``.
    """

    def __init__(self, path: str):
        """Initializes the wrapper without touching the filesystem.

        Args:
            path: Path the per-epoch metrics CSV should be written to.
        """
        super().__init__()
        self.path = path
        self._inner: tf.keras.callbacks.CSVLogger | None = None

    def _rotate_stale_file(self) -> None:
        """Moves the existing, mismatched-header file at ``self.path`` aside and logs a warning."""
        base, ext = os.path.splitext(self.path)
        suffix = 1
        stale_path = f"{base}.stale-{suffix}{ext}"
        while os.path.exists(stale_path):
            suffix += 1
            stale_path = f"{base}.stale-{suffix}{ext}"
        os.rename(self.path, stale_path)
        logger.warning(
            "Existing metrics CSV %s has a different column set than this run will produce "
            "(per_front_type_metrics or loss_name likely changed since it was last written). "
            "To avoid silently misaligning it, moved the old file to %s and starting a fresh %s.",
            self.path,
            stale_path,
            self.path,
        )

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Lazily builds (and, if needed, rotates ahead of) the real CSVLogger, then forwards to it."""
        logs = logs or {}
        if self._inner is None:
            if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
                with open(self.path, encoding="utf-8") as f:
                    existing_header = f.readline().rstrip("\n").split(",")
                if existing_header != _csv_logger_fieldnames(logs):
                    self._rotate_stale_file()
            self._inner = tf.keras.callbacks.CSVLogger(self.path, append=True)
            self._inner.set_model(self.model)
            self._inner.set_params(self.params)
            self._inner.on_train_begin()
        self._inner.on_epoch_end(epoch, logs)

    def on_train_end(self, logs: dict | None = None) -> None:
        """Forwards to the real CSVLogger, if any epoch ever ran and built one."""
        if self._inner is not None:
            self._inner.on_train_end(logs)


def _build_run_callbacks(
    uses_ema: bool,
    monitor: str,
    patience: int,
    learning_rate_decay_factor: float | None,
    learning_rate_minimum: float | None,
    monitor_min_delta: float,
    early_stopping_patience: int | None,
    extra_callbacks: list[tf.keras.callbacks.Callback] | None,
    wandb_project: str | None,
    wandb_log_freq: str | int,
    model_checkpoint_path: str | None,
    metrics_csv_path: str | None = None,
    compact_progress_every_n_batches: int | None = None,
) -> list[tf.keras.callbacks.Callback]:
    """Build the ordered callback list passed to model.fit().

    Ordering is load-bearing when ``uses_ema`` is True: SwapEMAWeights must be listed first so
    it swaps the EMA-averaged weights into the model at ``on_epoch_end`` before EarlyStopping /
    ModelCheckpoint (listed after it) capture their epoch's "best" snapshot — otherwise both
    would checkpoint raw, non-averaged weights even though the val_loss they selected on was
    already computed against the EMA weights (SwapEMAWeights always swaps EMA in for every
    internal validation pass, independent of this ordering). See ``TrainConfig.use_ema``.

    Args:
        uses_ema: Whether the compiled model's optimizer has weight EMA enabled.
        monitor: Metric name to monitor for LR decay / early stopping.
        patience: Epochs with no improvement before decaying LR (or stopping, without decay).
        learning_rate_decay_factor: Passed through to ``_build_monitor_callbacks``.
        learning_rate_minimum: Passed through to ``_build_monitor_callbacks``.
        monitor_min_delta: Passed through to ``_build_monitor_callbacks``.
        early_stopping_patience: Passed through to ``_build_monitor_callbacks``.
        extra_callbacks: Additional callbacks (e.g. periodic test-set visualization) to append.
        wandb_project: W&B project name, or None to skip W&B logging/checkpointing.
        wandb_log_freq: Passed to ``wandb.keras.WandbMetricsLogger``.
        model_checkpoint_path: Path prefix for the best-loss checkpoint, or None to skip
            checkpointing.
        metrics_csv_path: Explicit path to append per-epoch metrics to as CSV. None derives
            ``metrics_epoch.csv`` next to ``model_checkpoint_path``; if that is also None, CSV
            logging is skipped entirely. See ``_resolve_metrics_csv_path``.
        compact_progress_every_n_batches: ``CallbacksConfig.compact_progress_every_n_batches``.
            None omits ``CompactProgressCallback`` entirely, leaving Keras's default progress
            bar alone; an int appends it, throttled to that many batches. See
            ``_resolve_fit_verbose``, which silences Keras's own progress bar to match.

    Returns:
        Ordered list of callbacks for ``model.fit()``.
    """
    # Use the W&B Keras callback if logging to W&B, otherwise the standard ModelCheckpoint.
    ckpt_cls = wandb.keras.WandbModelCheckpoint if wandb_project else tf.keras.callbacks.ModelCheckpoint
    callbacks: list[tf.keras.callbacks.Callback] = []
    if uses_ema:
        callbacks.append(tf.keras.callbacks.SwapEMAWeights(swap_on_epoch=True))
    callbacks.extend(
        _build_monitor_callbacks(
            monitor,
            patience,
            learning_rate_decay_factor,
            learning_rate_minimum,
            min_delta=monitor_min_delta,
            early_stopping_patience=early_stopping_patience,
        )
    )
    callbacks.append(fronts_callbacks.GcCallback())
    # Must run before WandbMetricsLogger, _ResumeSafeCSVLogger, and CompactProgressCallback: it
    # mutates the shared `logs` dict that all three read, collapsing per-deep-supervision-output
    # keys into single aggregate hss/val_hss (and stripping the per-output loss keys) and
    # renaming per-front-type keys into slash-delimited form. _ResumeSafeCSVLogger is listed
    # immediately after it for the same reason — writing the consolidated, renamed keys rather
    # than raw sup{N}_{activation}_{metric} ones. CompactProgressCallback is listed after it too,
    # for the same reason again — it reads front/{front_type}/hss and front/{front_type}/csi,
    # which only exist in that form once this callback has run.
    callbacks.append(fronts_callbacks.MetricsConsolidationCallback())
    csv_path = _resolve_metrics_csv_path(metrics_csv_path, model_checkpoint_path)
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        callbacks.append(_ResumeSafeCSVLogger(csv_path))
    if compact_progress_every_n_batches is not None:
        callbacks.append(fronts_callbacks.CompactProgressCallback(compact_progress_every_n_batches))
    callbacks.extend(extra_callbacks or [])
    if wandb_project:
        callbacks.append(wandb.keras.WandbMetricsLogger(log_freq=wandb_log_freq))
    if model_checkpoint_path:
        callbacks.append(
            ckpt_cls(
                f"{model_checkpoint_path}_best_loss.keras",
                monitor="val_loss",
                save_best_only=True,
                mode="min",
            )
        )
    return callbacks


def _resolve_fit_verbose(compact_progress_every_n_batches: int | None) -> str | int:
    """Return the ``verbose`` value for ``model.fit`` given the compact-progress setting.

    ``CompactProgressCallback`` renders its own terminal-width-bounded progress line, so
    Keras's default ``ProgbarLogger`` must be silenced whenever it is active — otherwise both
    would print, defeating the reason ``CompactProgressCallback`` exists.

    Args:
        compact_progress_every_n_batches: ``CallbacksConfig.compact_progress_every_n_batches``.

    Returns:
        ``0`` if the compact callback is active, else Keras's own default ``"auto"``.
    """
    return 0 if compact_progress_every_n_batches is not None else "auto"


def _run(
    model: tf.keras.Model,
    train_data: datasets.FrontsPyDataset,
    val_data: datasets.FrontsPyDataset,
    epochs: int,
    monitor: str,
    patience: int,
    shuffle: bool,
    learning_rate_decay_factor: float | None = None,
    learning_rate_minimum: float | None = None,
    monitor_min_delta: float = 0.0,
    early_stopping_patience: int | None = None,
    model_checkpoint_path: str | None = None,
    wandb_project: str | None = None,
    run_name: str | None = None,
    wandb_log_freq: str | int = "epoch",
    steps_per_epoch: int | None = None,
    validation_steps: int | None = None,
    run_config: dict | None = None,
    extra_callbacks: list[tf.keras.callbacks.Callback] | None = None,
    metrics_csv_path: str | None = None,
    compact_progress_every_n_batches: int | None = None,
) -> tuple[tf.keras.callbacks.History, float]:
    if wandb_project:
        wandb.init(
            project=wandb_project,
            name=run_name,
            reinit=True,
            config=run_config or {},
        )

    callbacks = _build_run_callbacks(
        uses_ema=_optimizer_uses_ema(model.optimizer),
        monitor=monitor,
        patience=patience,
        learning_rate_decay_factor=learning_rate_decay_factor,
        learning_rate_minimum=learning_rate_minimum,
        monitor_min_delta=monitor_min_delta,
        early_stopping_patience=early_stopping_patience,
        extra_callbacks=extra_callbacks,
        wandb_project=wandb_project,
        wandb_log_freq=wandb_log_freq,
        model_checkpoint_path=model_checkpoint_path,
        metrics_csv_path=metrics_csv_path,
        compact_progress_every_n_batches=compact_progress_every_n_batches,
    )
    t0 = time.time()
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        shuffle=shuffle,
        verbose=_resolve_fit_verbose(compact_progress_every_n_batches),
    )
    elapsed = time.time() - t0
    if model_checkpoint_path:
        final_path = f"{model_checkpoint_path}_final.keras"
        model.save(final_path)
        logger.info("Saved final model to %s", final_path)
    if wandb_project:
        wandb.finish()
    return history, elapsed


def _build_dataset_summary(
    split: str, dataset: datasets.FrontsPyDataset, data_cfg: datasets.DatasetConfig
) -> fronts_callbacks.DatasetShapeSummary:
    """Build a DatasetShapeSummary for one already-loaded split, for provenance logging.

    Args:
        split: Split name ("train", "val", or "test").
        dataset: The split's already-loaded FrontsPyDataset.
        data_cfg: Dataset configuration (selects the input-stacking function and variables).

    Returns:
        A DatasetShapeSummary describing the split's model-input shape, target shape, and
        date range. ``dataset.input_ds`` is opened with ``chunks=None`` so per-batch
        training reads go straight through zarr; stacking it directly would force
        ``to_array``/``stack`` to materialize the whole split in RAM (as with the
        norm-stats computation above), so it's re-chunked to dask first to keep this
        metadata-only.

    Raises:
        ValueError: If the split has 0 timesteps.
    """
    stack_inputs = inputs.inputs_ds_to_volume_dataarray if data_cfg.volume_inputs else inputs.inputs_ds_to_dataarray
    input_da = stack_inputs(dataset.input_ds.chunk("auto"), data_cfg.variables)
    return fronts_callbacks.build_dataset_shape_summary(
        split=split,
        input_shape=input_da.shape,
        target_shape=dataset.target_da.shape,
        times=dataset.input_ds["time"].values,
    )


def _build_test_visualization_callback(
    test_dataset: datasets.FrontsPyDataset,
    data_config: datasets.DatasetConfig,
    callbacks_config: fronts_callbacks.CallbacksConfig,
    seed: int,
) -> fronts_callbacks.TestVisualizationCallback:
    """Build the periodic W&B visualization callback from an already-loaded test split.

    The test split is used read-only here purely for visualization: one active
    (front-containing) day for the prediction map, plus a bounded random subsample for
    the periodic performance diagram. Neither is used for fitting or model selection.

    Args:
        test_dataset: The sequestered test split, already loaded via load_data_into_dataloader.
        data_config: DatasetConfig specifying store paths and the test_years split.
        callbacks_config: Provides test_viz_sample_size and every_n_epochs.
        seed: Seed for the subsample RNG.

    Returns:
        A configured TestVisualizationCallback.
    """
    assert callbacks_config.test_viz_every_n_epochs is not None
    active_idx = fronts_callbacks.select_active_test_timestep(test_dataset.target_da)
    active_x, active_y = test_dataset.get_at_indices(np.array([active_idx]))
    active_label = str(test_dataset.input_ds.time.values[active_idx])

    subsample_idxs = fronts_callbacks.select_test_subsample(
        test_dataset.n_samples, callbacks_config.test_viz_sample_size, seed
    )
    subsample_x, subsample_y = test_dataset.get_at_indices(subsample_idxs)

    return fronts_callbacks.TestVisualizationCallback(
        active_day_x=active_x[0],
        active_day_y=active_y[0],
        active_day_label=active_label,
        subsample_x=subsample_x,
        subsample_y=subsample_y,
        lats=test_dataset.input_ds["latitude"].values,
        lons=test_dataset.input_ds["longitude"].values,
        front_types=list(constants.FRONT_TYPE_CLASS_INDEX),
        predict_batch_size=data_config.batch_size,
        every_n_epochs=callbacks_config.test_viz_every_n_epochs,
    )


def _collect_run_metadata(data_config: datasets.DatasetConfig, config_path: str) -> dict[str, str]:
    """Collect provenance metadata for logging: git commit, icechunk snapshots, SLURM vars.

    Args:
        data_config: DatasetConfig containing icechunk store configurations.
        config_path: Path to the YAML config file used for this run.

    Returns:
        Dict suitable for passing to wandb.init(config=...) and logger.info.
    """
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
        "SLURM_ARRAY_TASK_ID",
    )
    meta = {
        "config_name": os.path.basename(config_path),
        "config_path": config_path,
        "git_commit": utils.get_git_commit(),
        "era5_snapshot_id": utils.get_icechunk_snapshot_id(
            data_config.inputs_icechunk_config.store_path,
            data_config.inputs_icechunk_config.branch_name,
            data_config.inputs_icechunk_config.virtual_chunk_local_path,
        ),
        "fronts_snapshot_id": utils.get_icechunk_snapshot_id(
            data_config.targets_icechunk_config.store_path,
            data_config.targets_icechunk_config.branch_name,
            data_config.targets_icechunk_config.virtual_chunk_local_path,
        ),
    }
    for key in slurm_keys:
        value = os.environ.get(key)
        if value is not None:
            meta[key.lower()] = value
    return meta


def _build_wandb_config(
    data_cfg: datasets.DatasetConfig,
    model_cfg: model.ModelConfig,
    callbacks_cfg: fronts_callbacks.CallbacksConfig,
    train_cfg: TrainConfig,
    wandb_cfg: WandBConfig,
    run_meta: dict[str, str],
) -> dict:
    """Assemble the full ``wandb.init(config=...)`` payload from every YAML config section.

    Args:
        data_cfg: Parsed ``data_config`` YAML section.
        model_cfg: Parsed ``model_config`` YAML section.
        callbacks_cfg: Parsed ``callbacks_config`` YAML section.
        train_cfg: Parsed ``train_config`` YAML section.
        wandb_cfg: Parsed ``wandb_config`` YAML section.
        run_meta: Provenance metadata from ``_collect_run_metadata`` (git commit, icechunk
            snapshot ids, SLURM job info).

    Returns:
        Dict nesting each config section (as a plain dict, via ``dataclasses.asdict``) under
        its YAML key, plus the flat ``run_meta`` keys.
    """
    return {
        "data_config": dataclasses.asdict(data_cfg),
        "model_config": dataclasses.asdict(model_cfg),
        "callbacks_config": dataclasses.asdict(callbacks_cfg),
        "train_config": dataclasses.asdict(train_cfg),
        "wandb_config": dataclasses.asdict(wandb_cfg),
        **run_meta,
    }


def train(
    data_cfg: datasets.DatasetConfig,
    model_cfg: model.ModelConfig,
    callbacks_cfg: fronts_callbacks.CallbacksConfig,
    wandb_cfg: WandBConfig | None,
    train_cfg: TrainConfig,
    config_path: str,
) -> None:
    """Run the full training pipeline from pre-loaded config objects.

    Args:
        data_cfg: Dataset configuration specifying store paths and splits.
        model_cfg: Model architecture hyperparameters.
        callbacks_cfg: Early-stopping, checkpoint, and visualization callback config.
        wandb_cfg: W&B logging config, or None to disable W&B.
        train_cfg: Training hyperparameters (epochs, seed, learning rate, shuffle).
        config_path: Path to the YAML config file this run was loaded from, logged as
            provenance (``config_name``/``config_path`` in W&B and the run logs).
    """
    run_meta = _collect_run_metadata(data_cfg, config_path)
    for key, value in run_meta.items():
        logger.info("run_meta %s=%s", key, value)

    train_dataset = load_data_into_dataloader(
        data_cfg, split="train", seed=train_cfg.seed, shuffle=train_cfg.shuffle, drop_remainder=True
    )
    val_dataset = load_data_into_dataloader(data_cfg, split="val", seed=train_cfg.seed)

    logger.info(f"Total batches in training set: {len(train_dataset)}")
    logger.info(f"Total batches in validation set: {len(val_dataset)}")

    t0 = time.time()
    # The split is fully determined by these config values, so they make a reproducible
    # cache key without needing to serialize the (large) list of selected time indices.
    norm_cache_key_parts = (
        run_meta["era5_snapshot_id"],
        "test_years=" + ",".join(str(y) for y in data_cfg.test_years),
        "val_years=" + ",".join(str(y) for y in data_cfg.val_years),
        f"time_resolution={data_cfg.time_resolution}",
        f"seed={train_cfg.seed}",
    )
    # Volume-mode stats have a different shape ((level, variable) vs (channel,)), so give
    # them their own cache key; the extra part is omitted for 2D runs to keep existing
    # cache files valid.
    if data_cfg.volume_inputs:
        norm_cache_key_parts += ("volume_inputs=True",)
    # Re-chunk (metadata-only) the train split's already-small, already-sliced input
    # Dataset so the variable-stacking and min/max reduction build a dask graph
    # instead of materializing the whole split eagerly.
    stack_inputs = inputs.inputs_ds_to_volume_dataarray if data_cfg.volume_inputs else inputs.inputs_ds_to_dataarray
    train_inputs_da = stack_inputs(train_dataset.input_ds.chunk("auto"), data_cfg.variables)

    # Get the number of cpus allocated in the SLURM job
    cpu_count = utils.slurm_cpu_count()
    with dask.config.set(scheduler="threads", num_workers=cpu_count):
        norm_stat_a, norm_stat_b = inputs.load_or_compute_norm_stats(
            train_inputs_da, data_cfg.norm_stats_cache_dir, norm_cache_key_parts, method=data_cfg.normalization_method
        )
    logger.info(f"Normalization stats computed over full training set  ({time.time() - t0:.1f} s)")

    _set_seed(train_cfg.seed)

    # mixed_float16 overflowed forward activations to inf/NaN with this model/loss;
    # keep float32 until the precision is made numerically safe. Switching this back
    # to "mixed_float16" re-enables the float32 output cast and LossScaleOptimizer
    # paths below (both gated on the active policy).
    tf.keras.mixed_precision.set_global_policy("float32")
    logger.info("Mixed precision policy: %s", tf.keras.mixed_precision.global_policy().name)

    strategy = _get_distribution_strategy()

    # Derived from the actual loaded data rather than a hand-maintained config value, so
    # it automatically reflects data_cfg.variables and data_cfg.pressure_levels: (channel,)
    # for 2D models, (level, variable) for 3D/volume models.
    input_shape = (None, None, *train_inputs_da.shape[3:])
    logger.info("Model input shape: %s", input_shape)

    logger.info("Building and compiling model...")
    with strategy.scope():
        unet = model.UNet3Plus(
            input_shape=input_shape,
            num_classes=model_cfg.n_classes,
            levels=model_cfg.levels,
            filter_num=model_cfg.filter_num,
            pool_size=model_cfg.pool_size,
            upsample_size=model_cfg.upsample_size,
            kernel_size=model_cfg.kernel_size,
            squeeze_axes=model_cfg.squeeze_axes,
            first_encoder_connections=model_cfg.first_encoder_connections,
            deep_supervision=model_cfg.deep_supervision,
            batch_normalization=model_cfg.batch_normalization,
            activation=model_cfg.activation,
            output_activation=model_cfg.output_activation,
            modules_per_node=model_cfg.modules_per_node,
            normalization_method=data_cfg.normalization_method,
            normalization_stat_a=norm_stat_a,
            normalization_stat_b=norm_stat_b,
        ).build()
        if model_cfg.pretrained_weights_path is not None:
            _load_pretrained_weights(unet, model_cfg.pretrained_weights_path, norm_stat_a, norm_stat_b)
        if model_cfg.freeze_layer_prefixes is not None:
            _freeze_layers(unet, model_cfg.freeze_layer_prefixes)
        if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
            float32_outputs = [
                tf.keras.layers.Activation("linear", dtype="float32", name=f"output_{i}_float32")(out)
                for i, out in enumerate(unet.outputs)
            ]
            unet = model.SharedTargetModel(unet.inputs, float32_outputs, name=unet.name)
        _compile(
            unet,
            train_cfg.learning_rate,
            metric_class_weights=data_cfg.class_weights,
            train_cfg=train_cfg,
            latitudes=train_dataset.input_ds["latitude"].values,
            n_classes=model_cfg.n_classes,
            gradient_clip_norm=train_cfg.gradient_clip_norm,
        )
    logger.info("Model built and compiled.")

    unet.summary()

    train_steps = len(train_dataset)
    val_steps = len(val_dataset)

    full_pass_epochs = utils.epochs_per_full_pass(train_dataset.n_samples, data_cfg.batch_size, train_steps)
    effective_patience = max(callbacks_cfg.patience, full_pass_epochs)
    if effective_patience != callbacks_cfg.patience:
        logger.info(
            "Raising early-stopping patience %d -> %d so one full training pass (%d epochs of "
            "%d steps) completes without improvement before stopping.",
            callbacks_cfg.patience,
            effective_patience,
            full_pass_epochs,
            train_steps,
        )
    effective_stopping_patience = callbacks_cfg.early_stopping_patience
    if effective_stopping_patience is not None:
        effective_stopping_patience = max(effective_stopping_patience, full_pass_epochs)
    if train_cfg.epochs < full_pass_epochs:
        logger.warning(
            "epochs=%d is fewer than the %d epochs needed for one full training pass; "
            "the model will not see all training data.",
            train_cfg.epochs,
            full_pass_epochs,
        )

    passes_covered = effective_patience / full_pass_epochs if full_pass_epochs else 0.0
    logger.info(
        "Epoch = %d images (subset); full training pass every %d epochs; patience %d covers ~%.1f "
        "passes; validation covers all %d images in %d steps.",
        train_steps * data_cfg.batch_size,
        full_pass_epochs,
        effective_patience,
        passes_covered,
        val_dataset.n_samples,
        val_steps,
    )

    x_sample, _ = train_dataset[0]
    _show_input_sample("builtin-norm (raw)", x_sample)

    wandb_project = wandb_cfg.project_name if wandb_cfg is not None else None
    run_name = wandb_cfg.run_name if wandb_cfg is not None else None
    run_config = (
        _build_wandb_config(data_cfg, model_cfg, callbacks_cfg, train_cfg, wandb_cfg, run_meta)
        if wandb_cfg is not None
        else run_meta
    )

    dataset_summaries = [
        _build_dataset_summary("train", train_dataset, data_cfg),
        _build_dataset_summary("val", val_dataset, data_cfg),
    ]

    extra_callbacks = []
    test_dataset = None
    try:
        logger.info("Loading test split for shape/date-range logging and periodic visualization...")
        test_dataset = load_data_into_dataloader(data_cfg, split="test", seed=train_cfg.seed)
        logger.info("Test split loaded: %d timesteps available.", test_dataset.n_samples)
        dataset_summaries.append(_build_dataset_summary("test", test_dataset, data_cfg))
    except ValueError:
        logger.warning(
            "Skipping test-split shape/date-range logging and periodic visualization: could not "
            "load or summarize the test split (see preceding error).",
            exc_info=True,
        )

    extra_callbacks.append(fronts_callbacks.DatasetSummaryCallback(dataset_summaries))

    if wandb_project and callbacks_cfg.test_viz_every_n_epochs and test_dataset is not None:
        try:
            extra_callbacks.append(
                _build_test_visualization_callback(test_dataset, data_cfg, callbacks_cfg, train_cfg.seed)
            )
        except ValueError:
            logger.warning(
                "Skipping periodic test-set visualization: could not build the callback "
                "(see preceding error). Training will continue without it.",
                exc_info=True,
            )

    logger.info(
        "Starting training: %d epochs, %d train steps/epoch, %d val steps/epoch "
        "(first step traces the tf.function graph and may take a minute)...",
        train_cfg.epochs,
        train_steps,
        val_steps,
    )
    history, elapsed = _run(
        unet,
        train_dataset,
        val_dataset,
        epochs=train_cfg.epochs,
        shuffle=train_cfg.shuffle,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        monitor=callbacks_cfg.monitor,
        patience=effective_patience,
        learning_rate_decay_factor=callbacks_cfg.learning_rate_decay_factor,
        learning_rate_minimum=callbacks_cfg.learning_rate_minimum,
        monitor_min_delta=callbacks_cfg.min_delta,
        early_stopping_patience=effective_stopping_patience,
        model_checkpoint_path=callbacks_cfg.model_checkpoint_path,
        metrics_csv_path=callbacks_cfg.metrics_csv_path,
        compact_progress_every_n_batches=callbacks_cfg.compact_progress_every_n_batches,
        wandb_project=wandb_project,
        run_name=run_name,
        wandb_log_freq=wandb_cfg.log_freq if wandb_cfg is not None else "epoch",
        run_config=run_config,
        extra_callbacks=extra_callbacks,
    )

    best_val = min(history.history.get("val_loss", [float("nan")]))
    logger.info(f"\nBest val_loss: {best_val:.4f}  |  Training time: {elapsed:.1f} s")


def main() -> None:
    """Entry point: load config, build dataset and model, run training."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train UNet3Plus on ERA5 using NOAA fronts data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training config")
    args = parser.parse_args()

    yaml_data = utils.load_yaml(args.config)
    data_cfg = utils.parse_config_section(
        yaml_data, datasets.DatasetConfig, "data_config", type_hooks=utils.YAML_TYPE_HOOKS
    )
    model_cfg = utils.parse_config_section(yaml_data, model.ModelConfig, "model_config")
    callbacks_cfg = utils.parse_config_section(yaml_data, fronts_callbacks.CallbacksConfig, "callbacks_config")
    wandb_cfg = (
        utils.parse_config_section(yaml_data, WandBConfig, "wandb_config") if "wandb_config" in yaml_data else None
    )
    train_cfg = utils.parse_config_section(yaml_data, TrainConfig, "train_config")
    train(data_cfg, model_cfg, callbacks_cfg, wandb_cfg, train_cfg, config_path=args.config)


if __name__ == "__main__":
    main()
