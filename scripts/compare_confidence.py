"""Compare predicted-probability distributions between models on identical timesteps.

Answers one question: does a model actually emit lower peak probabilities than
another, measured on the prediction arrays rather than on rendered figures. Figure
brightness is not evidence — two ``plot_test_prediction`` renders at different dpi
produce very different-looking bands from identical probabilities.

Accepts either eval config shape, so each model is described by the same YAML that
already evaluates it:

  * ``eval_config``          — e.g. ``configs/sooner_eval.yaml`` (a 2.0 ``.keras``
    checkpoint, resolved through ``${run_name}``; set ``run_name`` at the top of the
    file to point at a different run)
  * ``harness_eval_config``  — e.g. ``configs/model_1702/eval_1702_conus_6h.yaml``
    (legacy model_1702 or a 2.0 checkpoint, via the model_1702 harness)

Each config brings its own stores and variables, which must match what that model was
trained on. For the comparison to mean anything, point both at the same spatial domain
and time window; the resolved window and grid are logged per model so a mismatch is
visible.

Usage:
    python scripts/compare_confidence.py \
        --config configs/model_1702/eval_1702_conus_6h.yaml \
        --config configs/sooner_eval.yaml \
        --max-timesteps 8
"""

import argparse
import dataclasses
import logging

import numpy as np
import tensorflow as tf

from fronts import constants, evaluate, utils
from fronts.constants import BoundingBox
from fronts.data import datasets
from fronts.model import SharedTargetModel, TemperatureScaledModel
from fronts.model_1702 import run_eval

log = logging.getLogger(__name__)

PERCENTILES = (50, 90, 99)


def summarize(preds: np.ndarray, targets: np.ndarray, front_types: list[str]) -> dict[str, dict[str, float]]:
    """Per-front-type predicted-probability stats on that front's truth pixels.

    Args:
        preds: Predictions shaped (time, latitude, longitude, n_classes).
        targets: One-hot targets shaped like ``preds``.
        front_types: Front type keys to summarize (e.g. ["CF", "WF"]).

    Returns:
        Mapping of front type to its stats. ``p99`` is the headline number: the
        probability the model reaches on the pixels it is supposed to fire on.
        ``max_anywhere`` is the ceiling over every pixel, which exposes a model that
        never emits a high probability at all.
    """
    out: dict[str, dict[str, float]] = {}
    for ft in front_types:
        ci = constants.FRONT_TYPE_CLASS_INDEX[ft]
        channel = preds[..., ci]
        on_front = channel[targets[..., ci] > 0.5]
        if on_front.size == 0:
            log.warning("No truth pixels for %s in this window; skipping.", ft)
            continue
        pcts = np.percentile(on_front, PERCENTILES)
        out[ft] = {
            "n": float(on_front.size),
            "mean": float(on_front.mean()),
            **{f"p{p}": float(v) for p, v in zip(PERCENTILES, pcts)},
            "frac_ge_0.5": float((on_front >= 0.5).mean()),
            "frac_ge_0.8": float((on_front >= 0.8).mean()),
            "max_anywhere": float(channel.max()),
        }
    return out


def _truncate(input_ds, target_da, max_timesteps: int | None):
    """Limit both arrays to the first ``max_timesteps`` timesteps, and log the window."""
    if max_timesteps is not None:
        input_ds = input_ds.isel(time=slice(0, max_timesteps))
        target_da = target_da.isel(time=slice(0, max_timesteps))
    times = input_ds["time"].values
    log.info(
        "  %d timesteps (%s → %s), grid %d x %d",
        len(times),
        times[0],
        times[-1],
        input_ds.sizes["latitude"],
        input_ds.sizes["longitude"],
    )
    return input_ds, target_da


def _run_harness_config(yaml_data: dict, max_timesteps: int | None, coordinates: BoundingBox | None):
    """Predict via the model_1702 harness path (``harness_eval_config``)."""
    harness_cfg: run_eval.HarnessEvalConfig = utils.parse_config_section(
        yaml_data, run_eval.HarnessEvalConfig, "harness_eval_config", utils.YAML_TYPE_HOOKS
    )
    data_cfg: datasets.DatasetConfig = utils.parse_config_section(
        yaml_data, datasets.DatasetConfig, "data_config", utils.YAML_TYPE_HOOKS
    )
    if coordinates is not None:
        harness_cfg = dataclasses.replace(harness_cfg, coordinates=coordinates)
    input_ds, fronts_raw = run_eval._open_eval_data(harness_cfg, data_cfg)
    input_ds, fronts_raw = _truncate(input_ds, fronts_raw, max_timesteps)

    lats = input_ds["latitude"].values
    model = run_eval.build_model_adapter(harness_cfg, lat_ascending=bool(lats[0] < lats[-1]))
    preds, targets = evaluate.predict_batches(
        model=model,
        input_ds=input_ds,
        target_da=fronts_raw,
        data_config=dataclasses.replace(data_cfg, front_dilation=harness_cfg.front_dilation),
        batch_size=harness_cfg.batch_size,
        class_weights=data_cfg.class_weights,
    )
    return preds, targets, harness_cfg.front_types, harness_cfg.model_path


def _run_eval_config(yaml_data: dict, max_timesteps: int | None, coordinates: BoundingBox | None):
    """Predict via the standard 2.0 eval path (``eval_config``)."""
    eval_cfg: evaluate.EvalConfig = utils.parse_config_section(
        yaml_data, evaluate.EvalConfig, "eval_config", utils.YAML_TYPE_HOOKS
    )
    data_cfg: datasets.DatasetConfig = utils.parse_config_section(
        yaml_data, datasets.DatasetConfig, "data_config", utils.YAML_TYPE_HOOKS
    )
    if coordinates is not None:
        eval_cfg = dataclasses.replace(eval_cfg, coordinates=coordinates)
    log.info("  loading %s", eval_cfg.model_path)
    model = tf.keras.models.load_model(
        eval_cfg.model_path,
        compile=False,
        custom_objects={"SharedTargetModel": SharedTargetModel, "TemperatureScaledModel": TemperatureScaledModel},
    )
    era5_ds, fronts_raw, _lats, _lons, _mask, effective_data_cfg = evaluate.load_eval_arrays(eval_cfg, data_cfg)
    era5_ds, fronts_raw = _truncate(era5_ds, fronts_raw, max_timesteps)

    preds, targets = evaluate.predict_batches(
        model=model,
        input_ds=era5_ds,
        target_da=fronts_raw,
        data_config=effective_data_cfg,
        batch_size=data_cfg.batch_size,
        class_weights=data_cfg.class_weights,
    )
    return preds, targets, eval_cfg.front_types, eval_cfg.model_path


def evaluate_config(
    config_path: str, max_timesteps: int | None, coordinates: BoundingBox | None
) -> tuple[str, dict[str, dict[str, float]]]:
    """Load one eval config of either shape, run its model, and summarize its probabilities.

    Args:
        config_path: Path to a YAML holding ``eval_config`` or ``harness_eval_config``.
        max_timesteps: Cap on timesteps to evaluate, or None for the whole window.
        coordinates: Bounding box overriding the config's own, so every model scores the
            same domain. None keeps each config's setting.

    Returns:
        Tuple of (label, per-front-type stats).

    Raises:
        KeyError: If the YAML holds neither config section.
    """
    yaml_data = utils.load_yaml(config_path)
    log.info("=== %s", config_path)

    if "harness_eval_config" in yaml_data:
        preds, targets, front_types, model_path = _run_harness_config(yaml_data, max_timesteps, coordinates)
    elif "eval_config" in yaml_data:
        preds, targets, front_types, model_path = _run_eval_config(yaml_data, max_timesteps, coordinates)
    else:
        raise KeyError(f"{config_path} has neither an 'eval_config' nor a 'harness_eval_config' section.")

    label = str(yaml_data.get("run_name") or model_path.rstrip("/").split("/")[-1])
    return label, summarize(preds, targets, front_types)


def main() -> None:
    """CLI entry point: evaluate each config and print a side-by-side comparison."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", action="append", required=True, help="Eval YAML. Repeat for each model.")
    parser.add_argument("--max-timesteps", type=int, default=8, help="Timesteps to evaluate; 0 means all.")
    parser.add_argument("--gpu-device", type=int, default=0, help="GPU index to configure.")
    parser.add_argument(
        "--coordinates",
        type=float,
        nargs=4,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        default=None,
        help="Override every config's bounding box so all models score the same domain.",
    )
    args = parser.parse_args()

    utils.configure_gpu(args.gpu_device)
    cap = args.max_timesteps if args.max_timesteps > 0 else None
    bbox = BoundingBox(*args.coordinates) if args.coordinates else None
    if bbox is not None:
        log.info("Forcing all models onto domain %s", bbox)

    results = [evaluate_config(path, cap, bbox) for path in args.config]

    front_types = sorted({ft for _, stats in results for ft in stats})
    header = (
        f"{'front':<6} {'model':<38} {'n':>9} {'mean':>7} {'p50':>7} "
        f"{'p90':>7} {'p99':>7} {'>=0.5':>7} {'>=0.8':>7} {'max':>7}"
    )
    print("\n" + header)
    print("-" * len(header))
    for ft in front_types:
        for label, stats in results:
            s = stats.get(ft)
            if s is None:
                continue
            print(
                f"{ft:<6} {label[:38]:<38} {s['n']:>9.0f} {s['mean']:>7.3f} {s['p50']:>7.3f} "
                f"{s['p90']:>7.3f} {s['p99']:>7.3f} {s['frac_ge_0.5']:>7.1%} {s['frac_ge_0.8']:>7.1%} "
                f"{s['max_anywhere']:>7.3f}"
            )
        print()

    print("Headline — p99 on-front probability (the number to compare):")
    for ft in front_types:
        vals = [f"{label[:28]}={stats[ft]['p99']:.3f}" for label, stats in results if ft in stats]
        print(f"  {ft}: " + "   ".join(vals))
    print(
        "\nIf these are close, the figures differ by renderer dpi, not by model confidence.\n"
        "If they differ widely, the effect is real and p99 is the number to move."
    )


if __name__ == "__main__":
    main()
