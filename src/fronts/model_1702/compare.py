"""Cross-model comparison summary for harness evaluation outputs.

Walks a stats root whose subdirectories each hold one evaluation run's ``stats_derived*.nc``
files (e.g. ``model_1702_conus/``, ``baseline_conus/``), extracts the max-CSI operating point
per (run, region, front type, neighborhood), and emits a long-format CSV plus a printed
markdown table with deltas against a reference run. Runs whose directory name contains
``out_of_domain`` are footnoted.

Usage:
    python -m fronts.model_1702.compare \
        --stats_root /ourdisk/hpc/ai2es/tman/models/stats/model_1702/ \
        --reference model_1702_conus --out comparison.csv
"""

import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

from fronts.model_1702 import figures

log = logging.getLogger(__name__)

OUT_OF_DOMAIN_TAG = "out_of_domain"
SUMMARY_NEIGHBORHOODS_KM = [100, 250]


def collect_metrics(stats_root: str) -> pd.DataFrame:
    """Extracts max-CSI operating points from every run directory under a stats root.

    Args:
        stats_root: Directory whose immediate subdirectories each contain one run's
            ``stats_derived*.nc`` files.

    Returns:
        Long-format DataFrame with columns run, region, front_type, neighborhood_km,
        max_csi, threshold, pod, far, fb, hss, out_of_domain.

    Raises:
        FileNotFoundError: If no derived-stats files are found anywhere under the root.
    """
    rows = []
    for run_dir in sorted(p for p in glob.glob(os.path.join(stats_root, "*")) if os.path.isdir(p)):
        run = os.path.basename(run_dir.rstrip("/"))
        for stats_path in sorted(glob.glob(os.path.join(run_dir, f"{figures.DERIVED_STATS_PREFIX}*.nc"))):
            region = figures.region_from_filename(stats_path) or "full"
            with xr.open_dataset(stats_path) as derived_ds:
                front_types = [
                    str(name).removeprefix("csi_") for name in derived_ds.data_vars if str(name).startswith("csi_")
                ]
                for front_type in front_types:
                    csi = derived_ds[f"csi_{front_type}"]
                    best_idx = csi.argmax("threshold")
                    for neighborhood in csi["neighborhood"].values:
                        idx = int(best_idx.sel(neighborhood=neighborhood))
                        point = {"neighborhood": neighborhood, "threshold": csi["threshold"].values[idx]}
                        sr = float(derived_ds[f"sr_{front_type}"].sel(neighborhood=neighborhood).values[idx])
                        rows.append(
                            {
                                "run": run,
                                "region": region,
                                "front_type": front_type,
                                "neighborhood_km": int(neighborhood),
                                "max_csi": float(csi.sel(neighborhood=neighborhood).values[idx]),
                                "threshold": float(point["threshold"]),
                                "pod": float(
                                    derived_ds[f"pod_{front_type}"].sel(neighborhood=neighborhood).values[idx]
                                ),
                                "far": 1.0 - sr,
                                "fb": float(derived_ds[f"fb_{front_type}"].sel(neighborhood=neighborhood).values[idx]),
                                "hss": float(
                                    derived_ds[f"hss_{front_type}"].sel(neighborhood=neighborhood).values[idx]
                                ),
                                "out_of_domain": OUT_OF_DOMAIN_TAG in run,
                            }
                        )
    if not rows:
        raise FileNotFoundError(f"No {figures.DERIVED_STATS_PREFIX}*.nc files found under {stats_root}")
    return pd.DataFrame(rows)


def build_markdown_summary(metrics: pd.DataFrame, reference_run: str | None) -> str:
    """Builds a markdown max-CSI comparison table across runs.

    Args:
        metrics: Long-format DataFrame from :func:`collect_metrics`.
        reference_run: Run name to compute deltas against; every other run gets a
            ``Δ vs {reference}`` column. None emits no delta columns.

    Returns:
        Markdown text with one table per summary neighborhood plus footnotes.
    """
    lines = []
    runs = sorted(metrics["run"].unique())
    out_of_domain_runs = sorted(metrics.loc[metrics["out_of_domain"], "run"].unique())
    for neighborhood in SUMMARY_NEIGHBORHOODS_KM:
        subset = metrics[metrics["neighborhood_km"] == neighborhood]
        if subset.empty:
            continue
        pivot = subset.pivot_table(index=["region", "front_type"], columns="run", values="max_csi")
        lines.append(f"### Max CSI, {neighborhood} km neighborhood")
        header = ["region", "front type"] + [_run_label(run, out_of_domain_runs) for run in runs]
        if reference_run is not None:
            header += [f"delta {run} vs {reference_run}" for run in runs if run != reference_run]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for (region, front_type), row in pivot.iterrows():
            cells = [region, front_type] + [_format_csi(row.get(run)) for run in runs]
            if reference_run is not None:
                reference_value = row.get(reference_run)
                for run in runs:
                    if run == reference_run:
                        continue
                    cells.append(_format_delta(row.get(run), reference_value))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    if out_of_domain_runs:
        footnote_runs = ", ".join(out_of_domain_runs)
        lines.append(f"\\* out-of-domain evaluation (model trained on a different domain): {footnote_runs}")
    return "\n".join(lines)


def _run_label(run: str, out_of_domain_runs: list[str]) -> str:
    return f"{run}\\*" if run in out_of_domain_runs else run


def _format_csi(value: float | None) -> str:
    return "—" if value is None or np.isnan(value) else f"{value:.3f}"


def _format_delta(value: float | None, reference: float | None) -> str:
    if value is None or reference is None or np.isnan(value) or np.isnan(reference):
        return "—"
    return f"{value - reference:+.3f}"


def main() -> None:
    """Parses arguments, collects metrics, writes the CSV, and prints the markdown summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Summarize harness eval stats across models.")
    parser.add_argument("--stats_root", type=str, required=True, help="Directory of per-run stats subdirectories.")
    parser.add_argument("--reference", type=str, default=None, help="Run name to compute deltas against.")
    parser.add_argument("--out", type=str, default="comparison.csv", help="Output CSV path.")
    args = parser.parse_args()

    metrics = collect_metrics(args.stats_root)
    metrics.to_csv(args.out, index=False)
    log.info("Wrote %d rows to %s", len(metrics), args.out)
    print(build_markdown_summary(metrics, args.reference))


if __name__ == "__main__":
    main()
