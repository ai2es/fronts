"""Tests for fronts.neighborhood_calibration."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from fronts import neighborhood_calibration as nc

THRESHOLDS = np.array([0.25, 0.5, 0.75])
NEIGHBORHOODS = np.array([50, 100])
# Bin 0 holds p < 0.25, bins 1-2 hold [0.25, 0.5) and [0.5, 0.75), bin 3 holds p >= 0.75.
CENTERS = np.array([0.125, 0.375, 0.625, 0.75])


def aggregate_from_bins(counts: np.ndarray, hits: np.ndarray, with_truth: bool = True) -> xr.Dataset:
    """Build fronts.evaluate-style aggregate stats for CF from per-bin pixel and hit counts.

    Args:
        counts: Pixel counts, shape (neighborhood, bin).
        hits: Pixels with a front within the radius, same shape.
        with_truth: Include ``nbhd_truth_CF``.

    Returns:
        Aggregate Dataset with tp/fp/tn/fn_CF (neighborhood, threshold) and nbhd_truth_CF (neighborhood,).
    """
    at_or_above = np.cumsum(counts[:, ::-1], axis=1)[:, ::-1][:, 1:]
    hits_at_or_above = np.cumsum(hits[:, ::-1], axis=1)[:, ::-1][:, 1:]
    dims = ("neighborhood", "threshold")
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "tp_CF": (dims, hits_at_or_above),
        "fp_CF": (dims, at_or_above - hits_at_or_above),
        "tn_CF": (dims, counts.sum(axis=1, keepdims=True) - at_or_above),
        "fn_CF": (dims, np.zeros_like(at_or_above)),
    }
    if with_truth:
        data_vars["nbhd_truth_CF"] = (("neighborhood",), hits.sum(axis=1))
    return xr.Dataset(data_vars, coords={"neighborhood": NEIGHBORHOODS, "threshold": THRESHOLDS})


@pytest.fixture()
def monotone_stats() -> xr.Dataset:
    counts = np.array([[1000.0, 50.0, 30.0, 20.0], [1000.0, 50.0, 30.0, 20.0]])
    hits = np.array([[20.0, 10.0, 15.0, 16.0], [50.0, 30.0, 24.0, 19.0]])
    return aggregate_from_bins(counts, hits)


class TestIsotonicFit:
    def test_keeps_values_that_already_rise(self) -> None:
        np.testing.assert_allclose(nc.isotonic_fit(np.array([0.1, 0.2, 0.5]), np.ones(3)), [0.1, 0.2, 0.5])

    def test_pools_a_drop_by_weight(self) -> None:
        np.testing.assert_allclose(nc.isotonic_fit(np.array([0.3, 0.1]), np.array([1.0, 3.0])), [0.15, 0.15])

    def test_pools_only_the_violating_run(self) -> None:
        fitted = nc.isotonic_fit(np.array([0.1, 0.5, 0.2, 0.6]), np.ones(4))
        np.testing.assert_allclose(fitted, [0.1, 0.35, 0.35, 0.6])


class TestNeighborhoodReliability:
    def test_recovers_per_bin_counts_including_the_bin_below_the_lowest_threshold(
        self, monotone_stats: xr.Dataset
    ) -> None:
        hits, counts = nc.neighborhood_reliability(monotone_stats, "CF")
        np.testing.assert_allclose(counts[0], [1000.0, 50.0, 30.0, 20.0])
        np.testing.assert_allclose(hits[1], [50.0, 30.0, 24.0, 19.0])

    def test_stats_without_truth_counts_raise(self) -> None:
        stats = aggregate_from_bins(np.ones((2, 4)), np.zeros((2, 4)), with_truth=False)
        with pytest.raises(KeyError, match=r"rerun fronts\.evaluate"):
            nc.neighborhood_reliability(stats, "CF")


class TestFitNeighborhoodCalibration:
    def test_rising_rates_are_kept(self, monotone_stats: xr.Dataset) -> None:
        table = nc.fit_neighborhood_calibration(monotone_stats, ["CF"])
        np.testing.assert_allclose(table["calibrated_CF"].sel(neighborhood=50), [0.02, 0.2, 0.5, 0.8], rtol=1e-6)
        np.testing.assert_allclose(table["probability"], CENTERS)

    def test_a_falling_tail_is_pooled(self, monotone_stats: xr.Dataset) -> None:
        # At 100 km the rates are 0.05, 0.6, 0.8, 0.95, so make the top bin fall to 0.5.
        counts = np.array([[1000.0, 50.0, 30.0, 20.0], [1000.0, 50.0, 30.0, 20.0]])
        hits = np.array([[20.0, 10.0, 15.0, 16.0], [50.0, 30.0, 24.0, 10.0]])
        table = nc.fit_neighborhood_calibration(aggregate_from_bins(counts, hits), ["CF"])
        pooled = (24.0 + 10.0) / (30.0 + 20.0)
        np.testing.assert_allclose(table["calibrated_CF"].sel(neighborhood=100), [0.05, 0.6, pooled, pooled], rtol=1e-6)

    def test_empty_bins_are_interpolated(self) -> None:
        counts = np.array([[1000.0, 0.0, 30.0, 20.0], [1000.0, 50.0, 30.0, 20.0]])
        hits = np.array([[20.0, 0.0, 15.0, 16.0], [50.0, 30.0, 24.0, 19.0]])
        table = nc.fit_neighborhood_calibration(aggregate_from_bins(counts, hits), ["CF"])
        expected_gap = np.interp(0.375, [0.125, 0.625], [0.02, 0.5])
        assert float(table["calibrated_CF"].sel(neighborhood=50)[1]) == pytest.approx(expected_gap)


class TestCalibrate:
    def test_maps_probabilities_through_the_table(self, monotone_stats: xr.Dataset) -> None:
        table = nc.fit_neighborhood_calibration(monotone_stats, ["CF"])
        probs = xr.Dataset({"CF": (("latitude", "longitude"), np.array([[0.0, 0.375], [0.5, 1.0]]))})
        out = nc.calibrate(probs, table, neighborhood_km=50)
        np.testing.assert_allclose(out["CF"], [[0.02, 0.2], [0.35, 0.8]], rtol=1e-6)
        assert out["CF"].dims == ("latitude", "longitude")


def test_main_writes_a_table_for_every_front_type(tmp_path: Path, monotone_stats: xr.Dataset) -> None:
    stats_path, out_path = tmp_path / "stats_aggregate.nc", tmp_path / "calibration.nc"
    monotone_stats.to_netcdf(stats_path)
    nc.main(["--stats_path", str(stats_path), "--output_path", str(out_path)])
    with xr.open_dataset(out_path) as table:
        assert list(table.data_vars) == ["calibrated_CF"]
        np.testing.assert_allclose(table["calibrated_CF"].sel(neighborhood=100), [0.05, 0.6, 0.8, 0.95], rtol=1e-6)
        assert table.attrs["source_stats"] == str(stats_path)
