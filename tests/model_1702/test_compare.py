"""Tests for the figures glob helpers and the comparison summary."""

import os

import numpy as np
import pytest
import xarray as xr

from fronts.model_1702 import compare, figures

NEIGHBORHOODS = [50, 100, 150, 200, 250]
THRESHOLDS = np.linspace(0.01, 1.0, 100)


def _synthetic_derived_ds(peak_csi: float) -> xr.Dataset:
    shape = (len(NEIGHBORHOODS), len(THRESHOLDS))
    csi = np.linspace(0, peak_csi, shape[1])[np.newaxis, :].repeat(shape[0], axis=0)
    flat = np.full(shape, 0.5)
    data_vars = {}
    for front_type in ("CF", "WF"):
        data_vars[f"csi_{front_type}"] = (("neighborhood", "threshold"), csi)
        for name in ("pod", "sr", "fb", "hss"):
            data_vars[f"{name}_{front_type}"] = (("neighborhood", "threshold"), flat)
    return xr.Dataset(data_vars, coords={"neighborhood": NEIGHBORHOODS, "threshold": THRESHOLDS})


@pytest.fixture
def stats_root(tmp_path):
    for run, peak in (("model_1702_conus", 0.4), ("baseline_conus", 0.5), ("model_1702_full_out_of_domain", 0.3)):
        run_dir = tmp_path / run
        run_dir.mkdir()
        _synthetic_derived_ds(peak).to_netcdf(run_dir / "stats_derived.nc")
        _synthetic_derived_ds(peak / 2).to_netcdf(run_dir / "stats_derived_land.nc")
    return str(tmp_path)


class TestRegionFromFilename:
    def test_full_domain_file(self):
        assert figures.region_from_filename("/x/stats_derived.nc") is None

    def test_mask_suffix(self):
        assert figures.region_from_filename("/x/stats_derived_land.nc") == "land"

    def test_office_suffix(self):
        assert figures.region_from_filename("/x/stats_derived_OPC_west.nc") == "OPC_west"


def test_render_stats_dir_requires_stats_files(tmp_path):
    from fronts import utils

    with pytest.raises(FileNotFoundError):
        figures.render_stats_dir(str(tmp_path), utils.BoundingBox(25.0, 56.75, 228.0, 299.75), 250, "png", None)


class TestCollectMetrics:
    def test_rows_and_values(self, stats_root):
        metrics = compare.collect_metrics(stats_root)
        assert set(metrics["run"]) == {"model_1702_conus", "baseline_conus", "model_1702_full_out_of_domain"}
        assert set(metrics["region"]) == {"full", "land"}
        assert set(metrics["front_type"]) == {"CF", "WF"}
        assert len(metrics) == 3 * 2 * 2 * len(NEIGHBORHOODS)
        baseline_full = metrics[
            (metrics["run"] == "baseline_conus")
            & (metrics["region"] == "full")
            & (metrics["front_type"] == "CF")
            & (metrics["neighborhood_km"] == 250)
        ]
        assert baseline_full["max_csi"].item() == pytest.approx(0.5)
        assert baseline_full["far"].item() == pytest.approx(0.5)
        assert not baseline_full["out_of_domain"].item()

    def test_out_of_domain_flagged(self, stats_root):
        metrics = compare.collect_metrics(stats_root)
        flagged = metrics[metrics["run"] == "model_1702_full_out_of_domain"]
        assert flagged["out_of_domain"].all()

    def test_empty_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compare.collect_metrics(str(tmp_path))


class TestMarkdownSummary:
    def test_contains_tables_deltas_and_footnote(self, stats_root):
        metrics = compare.collect_metrics(stats_root)
        text = compare.build_markdown_summary(metrics, reference_run="model_1702_conus")
        assert "### Max CSI, 100 km neighborhood" in text
        assert "### Max CSI, 250 km neighborhood" in text
        assert "delta baseline_conus vs model_1702_conus" in text
        assert "+0.100" in text
        assert "model_1702_full_out_of_domain\\*" in text
        assert "out-of-domain evaluation" in text

    def test_no_reference_omits_deltas(self, stats_root):
        metrics = compare.collect_metrics(stats_root)
        text = compare.build_markdown_summary(metrics, reference_run=None)
        assert "delta" not in text


def test_csv_round_trip(stats_root, tmp_path):
    import pandas as pd

    metrics = compare.collect_metrics(stats_root)
    out_path = os.path.join(tmp_path, "comparison.csv")
    metrics.to_csv(out_path, index=False)
    loaded = pd.read_csv(out_path)
    assert len(loaded) == len(metrics)
    assert set(loaded.columns) == set(metrics.columns)
