r"""Post-hoc temperature scaling calibration for a trained UNet3Plus model.

Fits a single scalar temperature T on the validation set by minimising
negative log-likelihood (NLL), then saves a TemperatureScaledModel that
is a drop-in replacement for the original.

Temperature scaling reference:
    Guo et al. (2017): https://arxiv.org/abs/1706.04599

By default, both the input and output model paths are derived from the same
``callbacks_config.model_checkpoint_path`` (interpolated from ``run_name``) that
training writes to: the input defaults to ``{model_checkpoint_path}_best_loss.keras``
and the output to ``{model_checkpoint_path}_calibrated.keras``. Pass
``--model_path``/``--output_path`` to override either.

Usage:
    pixi run -e schooner python src/fronts/calibrate.py \\
        --config_path configs/schooner_train.yaml
"""

import argparse
import dataclasses
import logging

import numpy as np
import scipy.optimize
import tensorflow as tf

from fronts import callbacks as fronts_callbacks
from fronts import utils
from fronts.data import datasets
from fronts.model import SharedTargetModel, TemperatureScaledModel
from fronts.train import load_data_into_dataloader

log = logging.getLogger(__name__)


@dataclasses.dataclass
class CalibrationConfig:
    """Configuration for temperature scaling calibration.

    Attributes:
        model_path: Path to the saved .keras model to calibrate. None derives
            ``{callbacks_config.model_checkpoint_path}_best_loss.keras`` from the
            training config, matching where ``train.py`` writes its best checkpoint.
        output_path: Where to save the calibrated TemperatureScaledModel. None derives
            ``{callbacks_config.model_checkpoint_path}_calibrated.keras``.
        gpu_device: GPU index to use. None runs on CPU.
        max_pixels: Maximum number of pixels to collect for T optimisation.
            Subsamples randomly when the val set exceeds this count.
    """

    model_path: str | None = None
    output_path: str | None = None
    gpu_device: int | None = None
    max_pixels: int = 500_000


def extract_logit_model(model: tf.keras.Model, output_activation: str = "softmax") -> tf.keras.Model:
    """Return a new model with the same inputs but pre-softmax logits as outputs.

    Exploits the stable layer naming convention in UNet3Plus: each deep
    supervision output has an activation layer named ``sup{N}_{activation}``.
    The logits are the inputs to those activation layers.

    Args:
        model: A compiled / loaded UNet3Plus model whose activation layers
            follow the ``sup{N}_{activation}`` naming scheme.
        output_activation: Name of the output activation used during training.

    Returns:
        A ``tf.keras.Model`` with the same inputs as ``model`` and one logit
        tensor output per deep supervision head.
    """
    logit_tensors = [
        model.get_layer(f"sup{i + 1}_{output_activation}").input for i in range(len(model.outputs))
    ]
    return tf.keras.Model(inputs=model.inputs, outputs=logit_tensors)


def fit_temperature(
    logit_model: tf.keras.Model,
    val_dataset: datasets.FrontsPyDataset,
    max_pixels: int = 500_000,
) -> float:
    """Fit temperature T by minimising NLL on the validation set.

    Collects the primary output (index 0) logits and targets across all
    validation batches, randomly subsamples to at most ``max_pixels`` pixels
    to stay memory-bounded, then uses scipy's bounded scalar minimiser to
    find T minimising the cross-entropy.

    Args:
        logit_model: Model returning pre-softmax logits (from ``extract_logit_model``).
        val_dataset: Validation ``FrontsPyDataset``.
        max_pixels: Cap on pixels used for the optimisation.

    Returns:
        Optimal temperature as a Python float.
    """
    logit_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []

    for i in range(len(val_dataset)):
        x, y = val_dataset[i]
        logits = logit_model(x, training=False)
        primary_logits = logits[0] if isinstance(logits, (list, tuple)) else logits
        logit_batches.append(primary_logits.numpy().reshape(-1, primary_logits.shape[-1]))
        target_batches.append(np.asarray(y).reshape(-1, np.asarray(y).shape[-1]))

    all_logits = np.concatenate(logit_batches, axis=0).astype(np.float64)
    all_targets = np.concatenate(target_batches, axis=0).astype(np.float64)

    if all_logits.shape[0] > max_pixels:
        rng = np.random.default_rng(0)
        idx = rng.choice(all_logits.shape[0], size=max_pixels, replace=False)
        all_logits = all_logits[idx]
        all_targets = all_targets[idx]

    def nll(t: float) -> float:
        scaled = all_logits / t
        max_l = scaled.max(axis=1, keepdims=True)
        log_z = np.log(np.exp(scaled - max_l).sum(axis=1, keepdims=True)) + max_l
        log_probs = scaled - log_z
        return -float(np.mean((all_targets * log_probs).sum(axis=1)))

    baseline = nll(1.0)
    result = scipy.optimize.minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
    t_opt = float(result.x)
    log.info(f"Temperature scaling: T={t_opt:.4f}  NLL before={baseline:.4f}  after={result.fun:.4f}")
    return t_opt


def resolve_calibration_config(yaml_data: dict) -> CalibrationConfig:
    """Build a ``CalibrationConfig`` from an optional ``calibration_config`` section.

    Any field left unset (None) is filled in from ``callbacks_config.model_checkpoint_path``
    — the same path ``train.py`` writes its checkpoints to — so a bare training config
    (e.g. ``schooner_train.yaml``) is enough to run calibration with no extra section.

    Args:
        yaml_data: Pre-loaded and interpolated YAML dict (from ``utils.load_yaml``).

    Returns:
        A fully-resolved ``CalibrationConfig`` with concrete ``model_path``/``output_path``.

    Raises:
        ValueError: If ``model_path``/``output_path`` aren't set and
            ``callbacks_config.model_checkpoint_path`` isn't available to derive them from.
    """
    calib_cfg = (
        utils.parse_config_section(yaml_data, CalibrationConfig, "calibration_config")
        if "calibration_config" in yaml_data
        else CalibrationConfig()
    )

    if calib_cfg.model_path is not None and calib_cfg.output_path is not None:
        return calib_cfg

    callbacks_cfg = utils.parse_config_section(yaml_data, fronts_callbacks.CallbacksConfig, "callbacks_config")
    checkpoint_path = callbacks_cfg.model_checkpoint_path
    if checkpoint_path is None:
        raise ValueError(
            "calibration_config.model_path/output_path aren't set and callbacks_config."
            "model_checkpoint_path is None, so the model paths can't be derived. Set either."
        )

    return dataclasses.replace(
        calib_cfg,
        model_path=calib_cfg.model_path or f"{checkpoint_path}_best_loss.keras",
        output_path=calib_cfg.output_path or f"{checkpoint_path}_calibrated.keras",
    )


def main() -> None:
    """CLI entry point for temperature scaling calibration."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Post-hoc temperature scaling calibration for a UNet3Plus model")
    parser.add_argument("--config_path", required=True, help="Path to training YAML config")
    parser.add_argument("--model_path", default=None, help="Override: path to the trained .keras model")
    parser.add_argument("--output_path", default=None, help="Override: where to save the calibrated model")
    parser.add_argument("--gpu_device", type=int, default=None, help="Override: GPU index (None = CPU)")
    parser.add_argument("--max_pixels", type=int, default=None, help="Override: max pixels for T optimisation")
    args = parser.parse_args()

    yaml_data = utils.load_yaml(args.config_path)
    calib_cfg = resolve_calibration_config(yaml_data)
    if args.model_path is not None:
        calib_cfg.model_path = args.model_path
    if args.output_path is not None:
        calib_cfg.output_path = args.output_path
    if args.gpu_device is not None:
        calib_cfg.gpu_device = args.gpu_device
    if args.max_pixels is not None:
        calib_cfg.max_pixels = args.max_pixels

    utils.configure_gpu(calib_cfg.gpu_device)

    data_cfg: datasets.DatasetConfig = utils.parse_config_section(yaml_data, datasets.DatasetConfig, "data_config")

    keras_model = tf.keras.models.load_model(
        calib_cfg.model_path,
        compile=False,
        custom_objects={"SharedTargetModel": SharedTargetModel},
    )

    val_dataset = load_data_into_dataloader(data_cfg, split="val")

    logit_model = extract_logit_model(keras_model)
    t_opt = fit_temperature(logit_model, val_dataset, max_pixels=calib_cfg.max_pixels)

    calibrated = TemperatureScaledModel(logit_model=logit_model, temperature=t_opt, name="temperature_scaled_model")
    calibrated.save(calib_cfg.output_path)
    log.info(f"Saved calibrated model to {calib_cfg.output_path}")


if __name__ == "__main__":
    main()
