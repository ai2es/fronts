"""Train UNet3Plus on ERA5 to detect weather fronts.

Raw data is fed to the model; a Keras Normalization layer (adapted on training
patches) is prepended and its mean/std are baked into the saved weights.

Supports distributed training across multiple GPUs via TensorFlow's MirroredStrategy.
Metrics are logged to Weights & Biases.
"""

import argparse
import dataclasses
import logging
import math
import os
import queue
import random
import threading
import time
from typing import Any

import dask
import numpy as np
import tensorflow as tf
import wandb
import xarray as xr
import zarr

from fronts import model, utils
from fronts.data import config, inputs, targets
from fronts.layers import losses, metrics

logger = logging.getLogger(__name__)


def make_batch_dataset(
    inputs: xr.DataArray,
    targets: xr.DataArray,
    n_supervision_outputs: int,
    batch_size: int = 4,
    shuffle: bool = False,
    preload: bool = False,
    epoch_steps: int | None = None,
    load_chunk_steps: int | None = None,
    prefetch_chunks: int = 2,
) -> Any:
    """Create a batched tf.data.Dataset from ERA5 and fronts DataArrays.

    Loads data in chunks via a single parallel dask compute per chunk rather
    than one sample at a time. With ``preload=True`` the entire dataset is
    materialised into RAM once at creation time (recommended for validation).

    The load chunk size is ``load_chunk_steps * batch_size`` samples. Setting
    this independently of ``epoch_steps`` lets you control peak RAM usage: a
    full epoch's worth of data may be too large to allocate as one contiguous
    array, while a smaller chunk still lets the background thread overlap I/O
    with GPU training. Falls back to ``epoch_steps`` when ``load_chunk_steps``
    is not set.

    Thread safety: zarr's global ``ThreadPoolExecutor`` must be bounded before
    any zarr I/O. Call ``zarr.config.update({"threading.max_workers": N})`` in
    the training process before dataset creation (done in ``train.main()``).

    Args:
        inputs: ERA5 DataArray of shape (time, latitude, longitude, channel).
        targets: Front DataArray of shape (time, latitude, longitude, class).
        n_supervision_outputs: Number of deep supervision outputs; the target
            tuple is replicated this many times.
        batch_size: Number of timesteps per batch.
        shuffle: If True, iterates timesteps in a random order each epoch.
        preload: If True, loads the entire dataset into RAM at creation time
            via a single parallel dask compute. Eliminates all I/O during
            iteration. Recommended for validation.
        epoch_steps: Number of batches per epoch passed to model.fit. Controls
            how many steps TF counts before advancing the epoch counter.
        load_chunk_steps: Number of steps' worth of samples to load per
            background prefetch. Defaults to ``epoch_steps`` when not set.
            Set this smaller than ``epoch_steps`` to cap peak RAM per chunk.
        prefetch_chunks: Number of chunks to keep loaded in RAM ahead of the
            generator. With the default of 2, chunk N+1 loads in parallel while
            the GPU trains on chunk N, so the GPU never waits for a new chunk
            as long as one chunk's load time is less than one chunk's train time.
            Increase if disk I/O is slower than GPU throughput.

    Returns:
        Tuple of (tf.data.Dataset, steps_per_epoch).
    """

    def _make_output_signature(
        n_lat: int,
        n_lon: int,
        n_channels: int,
        n_classes: int,
        n_supervision_outputs: int,
    ) -> tuple:
        target_spec = tf.TensorSpec(shape=(n_lat, n_lon, n_classes), dtype=tf.float32)
        return (
            tf.TensorSpec(shape=(n_lat, n_lon, n_channels), dtype=tf.float32),
            tuple(target_spec for _ in range(n_supervision_outputs)),
        )

    def _load(idxs: list) -> None:
        with dask.config.set(scheduler="threads", num_workers=16):
            cx = inputs.isel(time=idxs).compute()
            cy = targets.isel(time=idxs).compute()
        prefetch_q.put((cx, cy))

    def _iter_chunk(chunk_x: xr.DataArray, chunk_y: xr.DataArray):
        for pos in range(chunk_x.sizes["time"]):
            x = np.ascontiguousarray(chunk_x.isel(time=pos).values)
            y = np.ascontiguousarray(chunk_y.isel(time=pos).values)
            yield x, tuple(y for _ in range(n_supervision_outputs))

    assert inputs.sizes["time"] == targets.sizes["time"], (
        f"Input and target time lengths differ: {inputs.sizes['time']} vs {targets.sizes['time']}"
    )

    n_lat = inputs.sizes["latitude"]
    n_lon = inputs.sizes["longitude"]
    n_channels = inputs.sizes["channel"]
    n_classes = targets.sizes["class"]
    total = inputs.sizes["time"]

    if preload:
        logger.info("Pre-loading %d timesteps into RAM...", total)
        with dask.config.set(scheduler="threads", num_workers=16):
            inputs = inputs.compute()
            targets = targets.compute()
        logger.info("Pre-load complete.")

    effective_chunk_steps = load_chunk_steps if load_chunk_steps is not None else epoch_steps
    chunk_size = (effective_chunk_steps * batch_size) if effective_chunk_steps is not None else total

    # Persists across _gen() calls so the last prefetch of epoch N is already
    # in the queue when _gen() restarts for epoch N+1.
    prefetch_q: queue.Queue = queue.Queue(maxsize=prefetch_chunks)

    def _gen():
        order = np.random.permutation(total) if shuffle else np.arange(total)
        chunk_starts = list(range(0, total, chunk_size))
        for k in range(min(prefetch_chunks, len(chunk_starts))):
            nxt = chunk_starts[k]
            threading.Thread(
                target=_load,
                args=(order[nxt : nxt + chunk_size].tolist(),),
                daemon=True,
            ).start()

        for i, _chunk_start in enumerate(chunk_starts):
            chunk_x, chunk_y = prefetch_q.get()
            next_k = i + prefetch_chunks
            if next_k < len(chunk_starts):
                nxt = chunk_starts[next_k]
                threading.Thread(
                    target=_load,
                    args=(order[nxt : nxt + chunk_size].tolist(),),
                    daemon=True,
                ).start()
            yield from _iter_chunk(chunk_x, chunk_y)

    output_signature = _make_output_signature(n_lat, n_lon, n_channels, n_classes, n_supervision_outputs)

    steps_per_epoch = math.ceil(total / batch_size)
    ds = (
        tf.data.Dataset.from_generator(_gen, output_signature=output_signature)
        .batch(batch_size)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds, steps_per_epoch


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

    project_name: str = "fronts"
    run_name: str | None = None


@dataclasses.dataclass
class CallbacksConfig:
    """Early-stopping and checkpoint callback configuration."""

    monitor: str = "val_loss"
    patience: int = 8
    model_checkpoint_path: str | None = None


@dataclasses.dataclass
class TrainConfig:
    """Top-level training configuration assembling all sub-configs."""

    data_config: config.DataConfig
    model_config: model.ModelConfig
    callbacks_config: CallbacksConfig
    wandb_config: WandBConfig | None
    epochs: int = 50
    seed: int = 42
    learning_rate: float = 1e-4


def load_training_data(
    data_config: config.DataConfig,
    seed: int = 0,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Load, align, and encode ERA5 and fronts data for training.

    Opens both icechunk stores, deduplicates the fronts time index, aligns to
    the intersection of available timestamps, and returns lazy DataArrays ready
    for xbatcher.

    Args:
        data_config: DataConfig specifying store paths, branch names, and splits.
        seed: Integer seed for the RNG used when subsampling timesteps.

    Returns:
        Tuple of (era5_da, front_da) with dims (time, latitude, longitude, channel)
        and (time, latitude, longitude, class) respectively.
    """
    logger.info("Loading ERA5...")
    era5_ds = utils.open_readonly_icechunk_store(
        store_path=data_config.era5_icechunk_config.store_path,
        branch=data_config.era5_icechunk_config.branch_name,
        group=data_config.era5_icechunk_config.group_name,
        zarr_format=data_config.era5_icechunk_config.zarr_format,
        virtual_chunk_local_path=data_config.era5_icechunk_config.virtual_chunk_local_path,
    )
    logger.info("Loading fronts...")
    fronts_da = utils.open_readonly_icechunk_store(
        store_path=data_config.fronts_icechunk_config.store_path,
        branch=data_config.fronts_icechunk_config.branch_name,
        group=data_config.fronts_icechunk_config.group_name,
        zarr_format=data_config.fronts_icechunk_config.zarr_format,
        virtual_chunk_local_path=data_config.fronts_icechunk_config.virtual_chunk_local_path,
    )["identifier"]
    fronts_da = fronts_da.isel(time=~fronts_da.indexes["time"].duplicated(keep="first"))
    common_times = np.intersect1d(era5_ds.time.values, fronts_da.time.values)
    rng = np.random.default_rng(seed)
    keep = targets.filter_timesteps(fronts_da.sel(time=common_times), rng)
    common_times = common_times[keep]
    era5_ds = era5_ds.sel(time=common_times)
    fronts_da = fronts_da.sel(time=common_times)
    logger.info(f"Matched time steps: {len(common_times)}")

    logger.info("Building ERA5 DataArray (lazy)...")
    era5_da = inputs.era5_to_dataarray(era5_ds, data_config.variables)

    logger.info("Encoding targets (lazy)...")
    front_da = targets.one_hot_encode_to_dataarray(targets.remap_fronts(fronts_da))
    if data_config.front_dilation > 0:
        front_da = targets.dilate_fronts(front_da, data_config.front_dilation)

    return era5_da, front_da


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
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=[loss_fn] * n_out,
        metrics=[[hss_fn]] * n_out,
    )
    return n_out


class _WandbLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        wandb.log(logs or {}, step=epoch)


def _run(
    model: tf.keras.Model,
    train_ds,
    val_ds,
    epochs: int,
    monitor: str,
    patience: int,
    wandb_project: str | None,
    run_name: str | None = None,
    steps_per_epoch: int | None = None,
    validation_steps: int | None = None,
    run_config: dict | None = None,
) -> tuple:
    use_wandb = wandb_project is not None
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=run_name,
            reinit=True,
            config=run_config or {},
        )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True),
        *([_WandbLogger()] if use_wandb else []),
    ]
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )
    elapsed = time.time() - t0
    if use_wandb:
        wandb.finish()
    return history, elapsed


def _collect_run_metadata(data_config: config.DataConfig) -> dict:
    """Collect provenance metadata for logging: git commit, icechunk snapshots, SLURM vars.

    Args:
        data_config: DataConfig containing icechunk store configurations.

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
            data_config.era5_icechunk_config.store_path,
            data_config.era5_icechunk_config.branch_name,
            data_config.era5_icechunk_config.virtual_chunk_local_path,
        ),
        "fronts_snapshot_id": utils.get_icechunk_snapshot_id(
            data_config.fronts_icechunk_config.store_path,
            data_config.fronts_icechunk_config.branch_name,
            data_config.fronts_icechunk_config.virtual_chunk_local_path,
        ),
    }
    for key in slurm_keys:
        value = os.environ.get(key)
        if value is not None:
            meta[key.lower()] = value
    return meta


def main():
    """Entry point: load config, build dataset and model, run training."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train UNet3Plus on ERA5 using NOAA fronts data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training config")
    args = parser.parse_args()

    cfg = utils.open_config_yaml_as_dataclass(args.config, TrainConfig)

    run_meta = _collect_run_metadata(cfg.data_config)
    for key, value in run_meta.items():
        logger.info("run_meta %s=%s", key, value)

    zarr.config.update({"threading.max_workers": 16})

    era5_da, front_da = load_training_data(cfg.data_config, seed=cfg.seed)

    rng = np.random.default_rng(cfg.seed)
    all_indices = np.arange(era5_da.sizes["time"])
    shuffled = rng.permutation(all_indices)
    n_train = int(len(shuffled) * cfg.data_config.train_split)
    train_indices = sorted(shuffled[:n_train].tolist())
    val_indices = sorted(shuffled[n_train:].tolist())
    train_era5 = era5_da.isel(time=train_indices)
    val_era5 = era5_da.isel(time=val_indices)
    train_front = front_da.isel(time=train_indices)
    val_front = front_da.isel(time=val_indices)

    logger.info(f"Train timesteps: {train_era5.sizes['time']}, Val timesteps: {val_era5.sizes['time']}")

    n_lat = train_era5.sizes["latitude"]
    n_lon = train_era5.sizes["longitude"]

    t0 = time.time()
    norm_mean, norm_variance = inputs.compute_norm_stats(train_era5)
    logger.info(f"Normalization stats computed over full training set  ({time.time() - t0:.1f} s)")

    with dask.config.set(scheduler="threads", num_workers=16):
        val_channel_means = val_era5.mean(dim=["latitude", "longitude"], skipna=False).compute().values
    nan_timesteps = np.asarray(np.isnan(val_channel_means).any(axis=1))
    if nan_timesteps.any():
        n_total = val_era5.sizes["time"]
        logger.warning("Dropping %d/%d val timesteps with NaN ERA5 values", int(nan_timesteps.sum()), n_total)
        keep = ~nan_timesteps
        val_era5 = val_era5.isel(time=keep)
        val_front = val_front.isel(time=keep)

    _set_seed(cfg.seed)

    strategy = _get_distribution_strategy()

    with strategy.scope():
        unet = model.UNet3Plus(
            input_shape=(n_lat, n_lon, cfg.model_config.n_channels),
            num_classes=cfg.model_config.n_classes,
            levels=cfg.model_config.levels,
            filter_num=cfg.model_config.filter_num,
            pool_size=cfg.model_config.pool_size,
            upsample_size=cfg.model_config.upsample_size,
            kernel_size=cfg.model_config.kernel_size,
            first_encoder_connections=cfg.model_config.first_encoder_connections,
            deep_supervision=cfg.model_config.deep_supervision,
            batch_normalization=cfg.model_config.batch_normalization,
            activation=cfg.model_config.activation,
            output_activation=cfg.model_config.output_activation,
            modules_per_node=cfg.model_config.modules_per_node,
            normalization_mean=norm_mean,
            normalization_variance=norm_variance,
        ).build()
        n_out = _compile(unet, cfg.learning_rate, cfg.data_config.class_weights)

    unet.summary()

    if cfg.data_config.front_dilation > 0:
        logger.info("Pre-computing dilated training targets into RAM...")
        t0 = time.time()
        with dask.config.set(scheduler="threads", num_workers=16):
            train_front = train_front.compute()
        logger.info(f"Training targets pre-computed ({time.time() - t0:.1f} s)")

    train_ds, train_steps = make_batch_dataset(
        train_era5,
        train_front,
        n_out,
        cfg.data_config.batch_size,
        shuffle=True,
        epoch_steps=cfg.data_config.steps_per_epoch,
        load_chunk_steps=cfg.data_config.load_chunk_steps,
        prefetch_chunks=cfg.data_config.prefetch_chunks,
    )
    logger.info("Pre-loading validation set into RAM...")
    val_ds, val_steps = make_batch_dataset(
        val_era5,
        val_front,
        n_out,
        cfg.data_config.batch_size,
        preload=True,
    )
    if cfg.data_config.steps_per_epoch is not None:
        train_steps = cfg.data_config.steps_per_epoch

    _show_input_sample("builtin-norm (raw)", train_era5)

    wandb_project = cfg.wandb_config.project_name if cfg.wandb_config is not None else None
    run_name = cfg.wandb_config.run_name if cfg.wandb_config is not None else None
    history, elapsed = _run(
        unet,
        train_ds,
        val_ds,
        epochs=cfg.epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        monitor=cfg.callbacks_config.monitor,
        patience=cfg.callbacks_config.patience,
        wandb_project=wandb_project,
        run_name=run_name,
        run_config=run_meta,
    )

    best_val = min(history.history.get("val_loss", [float("nan")]))
    logger.info(f"\nBest val_loss: {best_val:.4f}  |  Training time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
