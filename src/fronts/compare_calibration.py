r"""Compare an uncalibrated model against its temperature-scaled counterpart.

Runs ``evaluate.py``'s stats computation for both the ``_best_loss.keras`` checkpoint
and its ``_calibrated.keras`` sibling (produced by ``calibrate.py``), then renders the
same 4-panel performance diagrams (``plot.py``'s ``performance-diagrams``) for each, so
the two can be flipped between side by side. The reliability diagram panel (b) is where
calibration differences should show up most directly; CSI/HSS (panels a, c, d) should be
close to unchanged since temperature scaling preserves the argmax.

Each model's evaluation runs in its own subprocess — TensorFlow pins GPU visibility on
first use, so a second ``configure_gpu`` call in the same process would raise once the
first model has already touched the device.

Usage:
    pixi run -e schooner python src/fronts/compare_calibration.py \
        --config_path configs/schooner_eval.yaml
"""

import argparse
import dataclasses
import logging
import os
import subprocess
import sys

import xarray as xr

from fronts import evaluate, utils
from fronts.plot.plot import _parse_front_types, plot_performance_diagrams

log = logging.getLogger(__name__)

_BEST_LOSS_SUFFIX = "_best_loss.keras"
_CALIBRATED_SUFFIX = "_calibrated.keras"


@dataclasses.dataclass
class CompareCalibrationConfig:
    """Configuration for comparing an uncalibrated model against its calibrated version.

    Attributes:
        best_model_path: Path to the uncalibrated .keras model. None derives it from
            ``eval_config.model_path`` (expected to already point at ``_best_loss.keras``).
        calibrated_model_path: Path to the temperature-scaled .keras model. None derives
            ``{best_model_path with _best_loss.keras replaced by _calibrated.keras}``,
            matching where ``calibrate.py`` writes its output by default.
        outdir: Base directory to write stats/plots into, under ``best_loss/`` and
            ``calibrated/`` subdirectories. None uses ``eval_config.outdir``.
        map_neighborhood: Neighbourhood radius (km) for the spatial CSI map panel.
        output_type: Output image format (e.g. "png").
        force: Recompute stats even if the output NetCDFs already exist.
    """

    best_model_path: str | None = None
    calibrated_model_path: str | None = None
    outdir: str | None = None
    map_neighborhood: int = 250
    output_type: str = "png"
    force: bool = False


def resolve_compare_config(yaml_data: dict) -> tuple[CompareCalibrationConfig, evaluate.EvalConfig]:
    """Fill in a ``CompareCalibrationConfig`` from an optional ``compare_calibration_config`` section.

    Args:
        yaml_data: Pre-loaded and interpolated YAML dict (from ``utils.load_yaml``).

    Returns:
        The resolved comparison config and the underlying ``EvalConfig`` (for data_config-
        adjacent fields like ``front_types``, ``coordinates``, ``mask``).

    Raises:
        ValueError: If ``best_model_path`` isn't set and can't be derived, or if
            ``calibrated_model_path`` isn't set and ``best_model_path`` doesn't follow the
            ``_best_loss.keras`` naming convention.
    """
    compare_cfg = (
        utils.parse_config_section(yaml_data, CompareCalibrationConfig, "compare_calibration_config")
        if "compare_calibration_config" in yaml_data
        else CompareCalibrationConfig()
    )
    eval_cfg: evaluate.EvalConfig = utils.parse_config_section(
        yaml_data, evaluate.EvalConfig, "eval_config", utils.YAML_TYPE_HOOKS
    )

    best_model_path = compare_cfg.best_model_path or eval_cfg.model_path
    if best_model_path is None:
        raise ValueError(
            "compare_calibration_config.best_model_path isn't set and eval_config.model_path is None."
        )

    calibrated_model_path = compare_cfg.calibrated_model_path
    if calibrated_model_path is None:
        if not best_model_path.endswith(_BEST_LOSS_SUFFIX):
            raise ValueError(
                f"Can't derive calibrated_model_path: best_model_path={best_model_path!r} doesn't end with "
                f"{_BEST_LOSS_SUFFIX!r}. Set compare_calibration_config.calibrated_model_path explicitly."
            )
        calibrated_model_path = best_model_path[: -len(_BEST_LOSS_SUFFIX)] + _CALIBRATED_SUFFIX

    return (
        dataclasses.replace(
            compare_cfg,
            best_model_path=best_model_path,
            calibrated_model_path=calibrated_model_path,
            outdir=compare_cfg.outdir or eval_cfg.outdir,
        ),
        eval_cfg,
    )


def run_eval_subprocess(config_path: str, model_path: str, outdir: str, mask: str | None) -> None:
    """Invoke ``evaluate.py`` in a fresh subprocess to compute stats for one model.

    A subprocess per model sidesteps TensorFlow's GPU-visibility-pinned-on-first-use
    restriction (see module docstring), and reuses ``evaluate.py``'s own stats pipeline
    and NetCDF-writing logic rather than duplicating it.

    Args:
        config_path: Path to the YAML config (forwarded to ``evaluate.py``).
        model_path: Path to the .keras model to evaluate.
        outdir: Directory to write ``stats_{spatial,aggregate,derived}*.nc`` into.
        mask: "land", "ocean", or None.

    Raises:
        subprocess.CalledProcessError: If the ``evaluate.py`` subprocess exits non-zero.
    """
    evaluate_script = os.path.join(os.path.dirname(__file__), "evaluate.py")
    cmd = [
        sys.executable,
        evaluate_script,
        "--config_path",
        config_path,
        "--model_path",
        model_path,
        "--outdir",
        outdir,
    ]
    if mask:
        cmd.extend(["--mask", mask])
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    """CLI entry point: evaluate + plot both models, side by side by output directory."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Compare an uncalibrated model against its temperature-scaled counterpart."
    )
    parser.add_argument("--config_path", required=True, help="Path to eval YAML config (e.g. schooner_eval.yaml)")
    parser.add_argument("--best_model_path", default=None, help="Override: path to the uncalibrated .keras model")
    parser.add_argument("--calibrated_model_path", default=None, help="Override: path to the calibrated .keras model")
    parser.add_argument("--outdir", default=None, help="Override: base output directory")
    parser.add_argument("--mask", default=None, choices=["land", "ocean"], help="Override eval_config.mask")
    parser.add_argument("--front_types", nargs="+", default=None, help="Override eval_config.front_types")
    parser.add_argument("--map_neighborhood", type=int, default=None, help="Override: neighbourhood (km) for CSI map")
    parser.add_argument("--output_type", default=None, help="Override: image format (png, pdf, etc.)")
    parser.add_argument("--force", action="store_true", help="Recompute stats even if output NetCDFs already exist")
    args = parser.parse_args()

    yaml_data = utils.load_yaml(args.config_path)
    compare_cfg, eval_cfg = resolve_compare_config(yaml_data)
    if args.best_model_path is not None:
        compare_cfg.best_model_path = args.best_model_path
    if args.calibrated_model_path is not None:
        compare_cfg.calibrated_model_path = args.calibrated_model_path
    if args.outdir is not None:
        compare_cfg.outdir = args.outdir
    if args.map_neighborhood is not None:
        compare_cfg.map_neighborhood = args.map_neighborhood
    if args.output_type is not None:
        compare_cfg.output_type = args.output_type
    if args.force:
        compare_cfg.force = True

    mask = args.mask if args.mask is not None else eval_cfg.mask
    front_types = args.front_types or eval_cfg.front_types
    mask_suffix = f"_{mask}" if mask else ""

    variants = [
        ("best_loss", compare_cfg.best_model_path),
        ("calibrated", compare_cfg.calibrated_model_path),
    ]

    for label, model_path in variants:
        subdir = os.path.join(compare_cfg.outdir, label)
        derived_path = os.path.join(subdir, f"stats_derived{mask_suffix}.nc")
        if not compare_cfg.force and os.path.exists(derived_path):
            log.info("Eval[%s]: skipping — stats already exist at %s.", label, derived_path)
        else:
            log.info("Eval[%s]: computing stats for %s → %s", label, model_path, subdir)
            run_eval_subprocess(args.config_path, model_path, subdir, mask)

    for label, _ in variants:
        subdir = os.path.join(compare_cfg.outdir, label)
        derived_path = os.path.join(subdir, f"stats_derived{mask_suffix}.nc")
        derived_ds = xr.open_dataset(derived_path)
        types = front_types or _parse_front_types(derived_ds)
        for ft in types:
            log.info("Plotting[%s] %s …", label, ft)
            plot_performance_diagrams(
                front_type=ft,
                derived_ds=derived_ds,
                mask=mask,
                coordinates=eval_cfg.coordinates,
                map_neighborhood=compare_cfg.map_neighborhood,
                output_type=compare_cfg.output_type,
                outdir=subdir,
            )
        derived_ds.close()

    log.info("Done. Compare:")
    for label, model_path in variants:
        log.info("  %-10s %s → %s", label, model_path, os.path.join(compare_cfg.outdir, label))


if __name__ == "__main__":
    main()
