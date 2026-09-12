r"""Rank input variables by permutation importance against a trained model.

For each candidate variable, shuffles its values along the time axis in the raw ERA5
input Dataset (breaking its correlation with the target and every other variable at a
given timestep, while keeping its own spatial/vertical structure intact since all levels
of a variable move together under one permutation), reruns the existing evaluation
pipeline, and compares CSI/HSS/POD against a no-permutation baseline. A variable whose
permutation leaves skill unchanged or improves it is a removal candidate.

Usage:
    pixi run -e schooner python src/fronts/importance.py \
        --config_path configs/schooner_importance.yaml
"""

import argparse
import dataclasses
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from tqdm import tqdm

from fronts import evaluate, utils
from fronts.data import datasets
from fronts.model import SharedTargetModel

log = logging.getLogger(__name__)


@dataclasses.dataclass
class PermutationConfig:
    """Configuration for permutation-importance runs.

    Attributes:
        n_repeats: Number of independent random shuffles per variable to average over.
        seed: Base seed. Each (variable, repeat) draws its own child seed derived from
            this value, so results are reproducible.
        variables: Variable names to permute. None means every entry in data_cfg.variables.
        metrics: Derived-stat metric name prefixes to score, e.g. ("csi", "hss", "pod").
    """

    n_repeats: int
    seed: int
    variables: list[str] | None
    metrics: tuple[str, ...]


def permute_variable(input_ds: xr.Dataset, variable: str, rng: np.random.Generator) -> xr.Dataset:
    """Return a copy of input_ds with `variable`'s time axis independently shuffled.

    All levels of `variable` (if any) move together under one permutation, preserving
    its own spatial/vertical structure while breaking its correlation with time, the
    target, and every other variable. Other variables and all coords are untouched.

    Args:
        input_ds: Raw ERA5 input Dataset with a time dimension.
        variable: Name of the data variable to permute.
        rng: Random generator controlling the shuffle draw.

    Returns:
        A new xr.Dataset with only `variable`'s values reordered along time.
    """
    perm = rng.permutation(input_ds.sizes["time"])
    shuffled = input_ds[variable].isel(time=perm).assign_coords(time=input_ds["time"])
    return input_ds.assign({variable: shuffled})


def reduce_derived_metrics(
    derived_ds: xr.Dataset, front_types: list[str], metrics: tuple[str, ...]
) -> dict[tuple[str, str], float]:
    """Reduce (neighborhood, threshold) derived metrics to one scalar per (metric, front_type).

    Uses max over threshold then mean over neighborhood as an operating-point-free summary.

    Args:
        derived_ds: Output of evaluate.compute_derived_stats.
        front_types: Front type labels in class order excluding background.
        metrics: Derived-stat metric name prefixes to score, e.g. ("csi", "hss", "pod").

    Returns:
        Mapping from (metric, front_type) to a scalar score.
    """
    return {
        (metric, ft): derived_ds[f"{metric}_{ft}"].max("threshold").mean("neighborhood").item()
        for ft in front_types
        for metric in metrics
    }


def build_importance_tables(
    baseline_metrics: dict[tuple[str, str], float],
    permuted_metrics: dict[str, list[dict[tuple[str, str], float]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-repeat and ranked summary tables from baseline and permuted metric scores.

    Args:
        baseline_metrics: (metric, front_type) -> scalar score with no permutation applied.
        permuted_metrics: variable -> list of (metric, front_type) -> scalar score, one
            entry per repeat.

    Returns:
        Tuple of (summary_df, ranking_df). summary_df has one row per
        (variable, repeat, metric, front_type). ranking_df aggregates over repeats and is
        sorted ascending by mean_delta, so removal candidates (near-zero or negative delta)
        sort first.
    """
    summary_rows = [
        {
            "variable": variable,
            "repeat": repeat,
            "metric": metric,
            "front_type": front_type,
            "baseline_value": baseline_metrics[(metric, front_type)],
            "permuted_value": permuted_value,
            "delta": permuted_value - baseline_metrics[(metric, front_type)],
        }
        for variable, repeats in permuted_metrics.items()
        for repeat, repeat_metrics in enumerate(repeats)
        for (metric, front_type), permuted_value in repeat_metrics.items()
    ]
    summary_df = pd.DataFrame(summary_rows)

    ranking_df = (
        summary_df.groupby(["variable", "metric", "front_type"])["delta"]
        .agg(mean_delta="mean", std_delta="std")
        .reset_index()
        .sort_values("mean_delta", ascending=True)
        .reset_index(drop=True)
    )
    return summary_df, ranking_df


def run_permutation_importance(
    model: tf.keras.Model,
    era5_ds: xr.Dataset,
    target_da: xr.DataArray,
    data_config: datasets.DatasetConfig,
    front_types: list[str],
    lats: np.ndarray,
    lons: np.ndarray,
    spatial_mask: np.ndarray | None,
    perm_cfg: PermutationConfig,
    outdir: str,
    batch_size: int,
    class_weights: list[float] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute baseline and permuted statistics for each variable/repeat.

    Writes baseline spatial/aggregate/derived netcdfs under `outdir/baseline/` and, per
    permuted run, only the derived-stats netcdf under `outdir/permuted/{variable}/repeat_{r}/`.

    Args:
        model: Loaded Keras model, callable as model(x, training=False).
        era5_ds: ERA5 input Dataset with dims (time, latitude, longitude) per variable.
        target_da: Raw integer front-code DataArray with dims (time, latitude, longitude).
        data_config: DatasetConfig providing variables list and front_dilation.
        front_types: Front type labels in class order excluding background.
        lats: 1-D latitude array.
        lons: 1-D longitude array.
        spatial_mask: Boolean (n_lat, n_lon) mask — True for included points. None = all.
        perm_cfg: Permutation-importance configuration (variables, n_repeats, seed, metrics).
        outdir: Directory to write baseline and per-variable permuted netcdfs under.
        batch_size: Batch size for inference.
        class_weights: Per-class loss weights passed to the FSS loss and HSS metric.

    Returns:
        Tuple of (summary_df, ranking_df) — see build_importance_tables.
    """
    variables = perm_cfg.variables if perm_cfg.variables is not None else data_config.variables

    log.info("Computing baseline statistics …")
    _spatial_ds, _aggregate_ds, baseline_derived_ds = evaluate.compute_stats(
        model=model,
        input_ds=era5_ds,
        target_da=target_da,
        data_config=data_config,
        front_types=front_types,
        lats=lats,
        lons=lons,
        spatial_mask=spatial_mask,
        batch_size=batch_size,
        class_weights=class_weights,
    )
    baseline_dir = os.path.join(outdir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    _spatial_ds.to_netcdf(os.path.join(baseline_dir, "stats_spatial.nc"))
    _aggregate_ds.to_netcdf(os.path.join(baseline_dir, "stats_aggregate.nc"))
    baseline_derived_ds.to_netcdf(os.path.join(baseline_dir, "stats_derived.nc"))
    baseline_metrics = reduce_derived_metrics(baseline_derived_ds, front_types, perm_cfg.metrics)

    permuted_metrics: dict[str, list[dict[tuple[str, str], float]]] = {}
    for variable_idx, variable in enumerate(tqdm(variables, unit="variable")):
        permuted_metrics[variable] = []
        for repeat in range(perm_cfg.n_repeats):
            rng = np.random.default_rng((perm_cfg.seed, variable_idx, repeat))
            permuted_ds = permute_variable(era5_ds, variable, rng)
            _, _, permuted_derived_ds = evaluate.compute_stats(
                model=model,
                input_ds=permuted_ds,
                target_da=target_da,
                data_config=data_config,
                front_types=front_types,
                lats=lats,
                lons=lons,
                spatial_mask=spatial_mask,
                batch_size=batch_size,
                class_weights=class_weights,
            )
            repeat_dir = os.path.join(outdir, "permuted", variable, f"repeat_{repeat}")
            os.makedirs(repeat_dir, exist_ok=True)
            permuted_derived_ds.to_netcdf(os.path.join(repeat_dir, "stats_derived.nc"))
            variable_metrics = reduce_derived_metrics(permuted_derived_ds, front_types, perm_cfg.metrics)
            permuted_metrics[variable].append(variable_metrics)

    summary_df, ranking_df = build_importance_tables(baseline_metrics, permuted_metrics)
    summary_df.to_csv(os.path.join(outdir, "importance_summary.csv"), index=False)
    ranking_df.to_csv(os.path.join(outdir, "importance_ranking.csv"), index=False)
    return summary_df, ranking_df


def run(eval_cfg: evaluate.EvalConfig, perm_cfg: PermutationConfig, data_cfg: datasets.DatasetConfig) -> None:
    """Run permutation importance from pre-loaded config objects.

    Args:
        eval_cfg: Evaluation configuration specifying model path, output directory, etc.
        perm_cfg: Permutation-importance configuration.
        data_cfg: Dataset configuration specifying icechunk store paths and variables.
    """
    utils.configure_gpu(eval_cfg.gpu_device)
    log.info("Loading model from %s …", eval_cfg.model_path)
    keras_model = tf.keras.models.load_model(
        eval_cfg.model_path, compile=False, custom_objects={"SharedTargetModel": SharedTargetModel}
    )
    log.info("Model loaded. Output count: %d.", len(keras_model.outputs))

    era5_ds, fronts_raw, lats, lons, spatial_mask, effective_data_cfg = evaluate.load_eval_arrays(eval_cfg, data_cfg)

    os.makedirs(eval_cfg.outdir, exist_ok=True)
    summary_df, ranking_df = run_permutation_importance(
        model=keras_model,
        era5_ds=era5_ds,
        target_da=fronts_raw,
        data_config=effective_data_cfg,
        front_types=eval_cfg.front_types,
        lats=lats,
        lons=lons,
        spatial_mask=spatial_mask,
        perm_cfg=perm_cfg,
        outdir=eval_cfg.outdir,
        batch_size=data_cfg.batch_size,
        class_weights=data_cfg.class_weights,
    )
    log.info("Wrote importance csvs (%d ranking rows) to %s", len(ranking_df), eval_cfg.outdir)
    log.info("Ranking (most negative delta first):\n%s", ranking_df)
    del summary_df


def main() -> None:
    """Parse arguments, load configs, and run permutation importance."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Rank input variables by permutation importance.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to permutation-importance config YAML.")
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        choices=["land", "ocean"],
        help="Restrict stats to land or ocean grid points.",
    )
    args = parser.parse_args()

    yaml_data = utils.load_yaml(args.config_path)
    eval_cfg: evaluate.EvalConfig = utils.parse_config_section(
        yaml_data, evaluate.EvalConfig, "eval_config", utils.YAML_TYPE_HOOKS
    )
    if args.mask is not None:
        eval_cfg = dataclasses.replace(eval_cfg, mask=args.mask)
    data_cfg: datasets.DatasetConfig = utils.parse_config_section(yaml_data, datasets.DatasetConfig, "data_config")
    perm_cfg: PermutationConfig = utils.parse_config_section(yaml_data, PermutationConfig, "permutation_config")
    run(eval_cfg, perm_cfg, data_cfg)


if __name__ == "__main__":
    main()
