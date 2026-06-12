"""Normalization experiment comparing pre-norm, built-in model norm, and no normalization on ERA5 front segmentation.

Experiment variants:
    A) prenorm: z-score normalized in the pipeline; model has no normalization layer.
    B) builtin-norm: raw data with a Keras Normalization layer adapted on training patches.
    C) no-norm: raw data, no normalization (baseline control).

Metrics are logged to Weights & Biases. Final val losses are printed side by side.
"""

import argparse
import logging
import time

import dask
import numpy as np
import xarray as xr

from fronts import utils
from fronts.data import inputs
from fronts.model import UNet3Plus
from fronts.train import (
    TrainConfig,
    _compile,
    _get_distribution_strategy,
    _run,
    _set_seed,
    _show_input_sample,
    load_training_data,
    make_batch_dataset,
)

logger = logging.getLogger(__name__)

WANDB_PROJECT = "fronts-norm-experiment"


def normalize(inputs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize inputs channel-wise using precomputed mean and std."""
    return ((inputs - mean) / std).astype(np.float32)


def compute_norm_stats(inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std over the (time, lat, lon) axes.

    Returns:
        A tuple (mean, std), each of shape (n_channels,). std is clamped to 1.0 for any
        constant channel to avoid division by zero.
    """
    axes = (0, 1, 2)
    mean = inputs.mean(axis=axes)
    std = inputs.std(axis=axes)
    std = np.where(std == 0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _best_val_loss(hist):
    return min(hist.history.get("val_loss", [float("nan")]))


def _parse_args():
    parser = argparse.ArgumentParser(description="UNet3Plus normalization experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training config")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["prenorm", "builtin-norm", "no-norm"],
        default=["prenorm", "builtin-norm", "no-norm"],
        metavar="EXP",
        help="Which experiments to run (default: all three). Choices: prenorm, builtin-norm, no-norm",
    )
    return parser.parse_args()


def main():
    """Run the normalization comparison experiments and log results to W&B."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    experiments = set(args.experiments)

    cfg = utils.open_config_yaml_as_dataclass(args.config, TrainConfig)

    data_config = cfg.data_config
    model_config = cfg.model_config
    callbacks_config = cfg.callbacks_config

    era5_da, front_da = load_training_data(data_config)

    train_era5 = era5_da.isel(time=slice(None, data_config.train_split))
    val_era5 = era5_da.isel(time=slice(data_config.train_split, None))
    train_front = front_da.isel(time=slice(None, data_config.train_split))
    val_front = front_da.isel(time=slice(data_config.train_split, None))

    logger.info(f"Train timesteps: {train_era5.sizes['time']}, Val timesteps: {val_era5.sizes['time']}")

    model_kwargs = {
        "input_shape": (None, None, model_config.n_channels),
        "num_classes": model_config.n_classes,
        "levels": model_config.levels,
        "filter_num": model_config.filter_num,
        "pool_size": model_config.pool_size,
        "upsample_size": model_config.upsample_size,
        "kernel_size": model_config.kernel_size,
        "first_encoder_connections": model_config.first_encoder_connections,
        "deep_supervision": model_config.deep_supervision,
        "batch_normalization": model_config.batch_normalization,
        "activation": model_config.activation,
        "output_activation": model_config.output_activation,
    }

    mean, std = compute_norm_stats(train_era5.values)

    strategy = _get_distribution_strategy()
    results = {}
    total_times = {}

    def _make_batch_datasets(train_in: xr.DataArray, val_in: xr.DataArray, n_out: int):
        train_ds, train_steps = make_batch_dataset(train_in, train_front, n_out, data_config.batch_size, shuffle=True)
        val_ds, val_steps = make_batch_dataset(val_in, val_front, n_out, data_config.batch_size)
        return train_ds, val_ds, train_steps, val_steps

    # ── Experiment A: pre-normalization ───────────────────────────────────────
    if "prenorm" in experiments:
        t_start = time.time()
        logger.info("\n=== Experiment A: Pre-normalization ===")
        t0 = time.time()
        train_norm = xr.DataArray(
            normalize(train_era5.values, mean, std),
            dims=train_era5.dims,
            coords=train_era5.coords,
        )
        val_norm = xr.DataArray(
            normalize(val_era5.values, mean, std),
            dims=val_era5.dims,
            coords=val_era5.coords,
        )
        logger.info(f"Time to normalize data: {time.time() - t0:.1f} s")

        _set_seed(cfg.seed)
        with strategy.scope():
            model_a = UNet3Plus(**model_kwargs).build()
        n_out = _compile(model_a, cfg.learning_rate, data_config.class_weights)
        model_a.summary()

        train_ds_a, val_ds_a, train_steps_a, val_steps_a = _make_batch_datasets(train_norm, val_norm, n_out)
        _show_input_sample("prenorm (normalized)", train_norm)
        results["prenorm"] = _run(
            model_a,
            train_ds_a,
            val_ds_a,
            epochs=cfg.epochs,
            monitor=callbacks_config.monitor,
            patience=callbacks_config.patience,
            wandb_project=WANDB_PROJECT,
            run_name="prenorm",
            steps_per_epoch=train_steps_a,
            validation_steps=val_steps_a,
        )
        total_times["prenorm"] = time.time() - t_start
        logger.info(f"Time to train with pre-normalization: {total_times['prenorm']:.1f} s")

    # ── Experiment B: built-in normalization ──────────────────────────────────
    if "builtin-norm" in experiments:
        t_start = time.time()
        logger.info("\n=== Experiment B: Built-in normalization ===")
        t0 = time.time()
        with dask.config.set(scheduler="threads", num_workers=16):
            norm_mean, norm_variance = inputs.compute_norm_stats(train_era5)
        logger.info(f"Normalization stats computed over full training set  ({time.time() - t0:.1f} s)")

        _set_seed(cfg.seed)
        with strategy.scope():
            model_b = UNet3Plus(
                **model_kwargs,
                normalization_mean=norm_mean,
                normalization_variance=norm_variance,
            ).build()
        n_out = _compile(model_b, cfg.learning_rate, data_config.class_weights)

        train_ds_b, val_ds_b, train_steps_b, val_steps_b = _make_batch_datasets(train_era5, val_era5, n_out)
        _show_input_sample("builtin-norm (raw)", train_era5)
        results["builtin-norm"] = _run(
            model_b,
            train_ds_b,
            val_ds_b,
            epochs=cfg.epochs,
            monitor=callbacks_config.monitor,
            patience=callbacks_config.patience,
            wandb_project=WANDB_PROJECT,
            run_name="builtin-norm",
            steps_per_epoch=train_steps_b,
            validation_steps=val_steps_b,
        )
        total_times["builtin-norm"] = time.time() - t_start
        logger.info(f"Time to train with built-in normalization: {total_times['builtin-norm']:.1f} s")

    # ── Experiment C: no normalization ────────────────────────────────────────
    if "no-norm" in experiments:
        t_start = time.time()
        logger.info("\n=== Experiment C: No normalization ===")
        _set_seed(cfg.seed)
        with strategy.scope():
            model_c = UNet3Plus(**model_kwargs).build()
        n_out = _compile(model_c, cfg.learning_rate, data_config.class_weights)

        train_ds_c, val_ds_c, train_steps_c, val_steps_c = _make_batch_datasets(train_era5, val_era5, n_out)
        _show_input_sample("no-norm (raw)", train_era5)
        results["no-norm"] = _run(
            model_c,
            train_ds_c,
            val_ds_c,
            epochs=cfg.epochs,
            monitor=callbacks_config.monitor,
            patience=callbacks_config.patience,
            wandb_project=WANDB_PROJECT,
            run_name="no-norm",
            steps_per_epoch=train_steps_c,
            validation_steps=val_steps_c,
        )
        total_times["no-norm"] = time.time() - t_start
        logger.info(f"Time to train with no normalization: {total_times['no-norm']:.1f} s")

    logger.info("\n=== Results ===")
    logger.info(f"{'Experiment':<20} | {'Best val_loss':>14} | {'Time (s)':>10} | {'Total time (s)':>14}")
    logger.info("-" * 68)
    for name, (hist, elapsed) in results.items():
        logger.info(f"{name:<20} | {_best_val_loss(hist):>14.4f} | {elapsed:>10.1f} | {total_times[name]:>14.1f}")


if __name__ == "__main__":
    main()
