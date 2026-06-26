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
from fronts import model, utils
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
    """Training-specific hyperparameters."""

    epochs: int = 50
    seed: int = 42
    learning_rate: float = 1e-4
    shuffle: bool = False


def load_data_into_dataloader(
    data_config: datasets.DatasetConfig,
    split: Literal["train", "val", "test"],
    seed: int = 0,
    shuffle: bool = False,
    workers: int = 1,
) -> datasets.TrainingDataset:
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
        workers: Number of ``PyDataset`` prefetch threads. 1 (the ``PyDataset``
            default) fetches each batch synchronously on the main thread, serializing
            every batch's icechunk read with the GPU training step.

    Returns:
        TrainingDataset yielding batches of (input, target) pairs for training.
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

    logger.info("Loading %s targets...", split)
    targets_da = _open(data_config.targets_icechunk_config)["identifier"]

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
    inputs_ds = inputs_ds_matched.isel(time=split_indices)
    targets_da = targets_da_matched.isel(time=split_indices)

    # Get the number of threads to use for PyDataset prefetching from max_pydataset_workers in the DatasetConfig,
    # which is set to 16 by default. This allows for parallel loading of batches without overwhelming ourdisk I/O.
    data_workers = utils.limit_workers_for_slurm(max_workers=data_config.max_pydataset_workers)
    return datasets.TrainingDataset(
        input_ds=inputs_ds,
        target_da=targets_da,
        data_config=data_config,
        seed=seed,
        batch_size=data_config.batch_size,
        shuffle=shuffle,
        workers=data_workers,
        max_queue_size=data_config.max_queue_size,
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


def _compile(model: tf.keras.Model, learning_rate: float, class_weights: list[float] | None) -> int:
    n_out = len(model.outputs)
    loss_fn = losses.fractions_skill_score(mask_size=(3, 3), class_weights=class_weights)
    hss_fn = metrics.heidke_skill_score(class_weights=class_weights)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[[hss_fn]] * n_out,
    )
    return n_out


def _run(
    model: tf.keras.Model,
    train_ds,
    val_ds,
    epochs: int,
    monitor: str,
    patience: int,
    shuffle: bool,
    model_checkpoint_path: str | None = None,
    wandb_project: str | None = None,
    run_name: str | None = None,
    wandb_log_freq: str | int = "epoch",
    steps_per_epoch: int | None = None,
    validation_steps: int | None = None,
    run_config: dict | None = None,
    extra_callbacks: list | None = None,
) -> tuple:
    if wandb_project:
        wandb.init(
            project=wandb_project,
            name=run_name,
            reinit=True,
            config=run_config or {},
        )

    # Use the W&B Keras callback if logging to W&B, otherwise the standard ModelCheckpoint.
    ckpt_cls = wandb.keras.WandbModelCheckpoint if wandb_project else tf.keras.callbacks.ModelCheckpoint
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True),
        fronts_callbacks.GcCallback(),
        # Must run before WandbMetricsLogger: it mutates the shared `logs` dict that
        # WandbMetricsLogger reads, collapsing per-deep-supervision-output keys into
        # single aggregate hss/val_hss (and stripping the per-output loss keys).
        fronts_callbacks.MetricsConsolidationCallback(),
    ]
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
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        shuffle=shuffle,
    )
    elapsed = time.time() - t0
    if model_checkpoint_path:
        final_path = f"{model_checkpoint_path}_final.keras"
        model.save(final_path)
        logger.info("Saved final model to %s", final_path)
    if wandb_project:
        wandb.finish()
    return history, elapsed


def _build_test_visualization_callback(
    data_config: datasets.DatasetConfig,
    callbacks_config: fronts_callbacks.CallbacksConfig,
    seed: int,
) -> fronts_callbacks.TestVisualizationCallback:
    """Load the sequestered test split and build the periodic W&B visualization callback.

    The test split is loaded read-only here purely for visualization: one active
    (front-containing) day for the prediction map, plus a bounded random subsample for
    the periodic performance diagram. Neither is used for fitting or model selection.

    Args:
        data_config: DatasetConfig specifying store paths and the test_years split.
        callbacks_config: Provides test_viz_sample_size and every_n_epochs.
        seed: Seed for the subsample RNG.

    Returns:
        A configured TestVisualizationCallback.
    """
    assert callbacks_config.test_viz_every_n_epochs is not None
    logger.info("Loading test split for periodic visualization...")
    test_dataset = load_data_into_dataloader(data_config, split="test", seed=seed)
    logger.info("Test split loaded: %d timesteps available for visualization.", test_dataset.n_samples)

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
        front_types=list(fronts_callbacks.FRONT_TYPE_CLASS_INDEX),
        predict_batch_size=data_config.batch_size,
        every_n_epochs=callbacks_config.test_viz_every_n_epochs,
    )


def _collect_run_metadata(data_config: datasets.DatasetConfig) -> dict:
    """Collect provenance metadata for logging: git commit, icechunk snapshots, SLURM vars.

    Args:
        data_config: DatasetConfig containing icechunk store configurations.

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


def train(
    data_cfg: datasets.DatasetConfig,
    model_cfg: model.ModelConfig,
    callbacks_cfg: fronts_callbacks.CallbacksConfig,
    wandb_cfg: WandBConfig | None,
    train_cfg: TrainConfig,
) -> None:
    """Run the full training pipeline from pre-loaded config objects.

    Args:
        data_cfg: Dataset configuration specifying store paths and splits.
        model_cfg: Model architecture hyperparameters.
        callbacks_cfg: Early-stopping, checkpoint, and visualization callback config.
        wandb_cfg: W&B logging config, or None to disable W&B.
        train_cfg: Training hyperparameters (epochs, seed, learning rate, shuffle).
    """
    run_meta = _collect_run_metadata(data_cfg)
    for key, value in run_meta.items():
        logger.info("run_meta %s=%s", key, value)

    train_dataset = load_data_into_dataloader(data_cfg, split="train", seed=train_cfg.seed, shuffle=train_cfg.shuffle)
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
    # Re-chunk (metadata-only) the train split's already-small, already-sliced input
    # Dataset so the variable-stacking and mean/variance reduction build a dask graph
    # instead of materializing the whole split eagerly.
    train_inputs_da = inputs.inputs_ds_to_dataarray(train_dataset.input_ds.chunk("auto"), data_cfg.variables)

    # Get the number of cpus allocated in the SLURM job
    cpu_count = utils.slurm_cpu_count()
    with dask.config.set(scheduler="threads", num_workers=cpu_count):
        norm_mean, norm_variance = inputs.load_or_compute_norm_stats(
            train_inputs_da, data_cfg.norm_stats_cache_dir, norm_cache_key_parts
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

    logger.info("Building and compiling model...")
    with strategy.scope():
        unet = model.UNet3Plus(
            input_shape=(None, None, model_cfg.n_channels),
            num_classes=model_cfg.n_classes,
            levels=model_cfg.levels,
            filter_num=model_cfg.filter_num,
            pool_size=model_cfg.pool_size,
            upsample_size=model_cfg.upsample_size,
            kernel_size=model_cfg.kernel_size,
            first_encoder_connections=model_cfg.first_encoder_connections,
            deep_supervision=model_cfg.deep_supervision,
            batch_normalization=model_cfg.batch_normalization,
            activation=model_cfg.activation,
            output_activation=model_cfg.output_activation,
            modules_per_node=model_cfg.modules_per_node,
            normalization_mean=norm_mean,
            normalization_variance=norm_variance,
        ).build()
        if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
            float32_outputs = [
                tf.keras.layers.Activation("linear", dtype="float32", name=f"output_{i}_float32")(out)
                for i, out in enumerate(unet.outputs)
            ]
            unet = model.SharedTargetModel(unet.inputs, float32_outputs, name=unet.name)
        _compile(unet, train_cfg.learning_rate, data_cfg.class_weights)
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

    extra_callbacks = []
    if wandb_project and callbacks_cfg.test_viz_every_n_epochs:
        try:
            extra_callbacks.append(_build_test_visualization_callback(data_cfg, callbacks_cfg, train_cfg.seed))
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
        model_checkpoint_path=callbacks_cfg.model_checkpoint_path,
        wandb_project=wandb_project,
        run_name=run_name,
        wandb_log_freq=wandb_cfg.log_freq if wandb_cfg is not None else "epoch",
        run_config=run_meta,
        extra_callbacks=extra_callbacks,
    )

    best_val = min(history.history.get("val_loss", [float("nan")]))
    logger.info(f"\nBest val_loss: {best_val:.4f}  |  Training time: {elapsed:.1f} s")


def main():
    """Entry point: load config, build dataset and model, run training."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train UNet3Plus on ERA5 using NOAA fronts data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training config")
    args = parser.parse_args()

    yaml_data = utils.load_yaml(args.config)
    data_cfg = utils.parse_config_section(yaml_data, datasets.DatasetConfig, "data_config")
    model_cfg = utils.parse_config_section(yaml_data, model.ModelConfig, "model_config")
    callbacks_cfg = utils.parse_config_section(yaml_data, fronts_callbacks.CallbacksConfig, "callbacks_config")
    wandb_cfg = (
        utils.parse_config_section(yaml_data, WandBConfig, "wandb_config") if "wandb_config" in yaml_data else None
    )
    train_cfg = utils.parse_config_section(yaml_data, TrainConfig, "train_config")
    train(data_cfg, model_cfg, callbacks_cfg, wandb_cfg, train_cfg)


if __name__ == "__main__":
    main()
