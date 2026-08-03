"""Tests for fronts.importance permutation-importance helpers and driver."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fronts.evaluate import NEIGHBORHOODS_KM, THRESHOLDS
from fronts.importance import (
    PermutationConfig,
    build_importance_tables,
    permute_variable,
    reduce_derived_metrics,
    run_permutation_importance,
)

try:
    import tensorflow as tf

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

_N_TIMES = 6
_N_LAT = 4
_N_LON = 5
_N_LEVELS = 3
_FRONT_TYPES = ["CF", "WF"]
_METRICS = ("csi", "hss")


@pytest.fixture()
def single_level_ds() -> xr.Dataset:
    """Two single-level variables, distinct values so shuffles are distinguishable."""
    rng = np.random.default_rng(0)
    times = np.arange(_N_TIMES)
    return xr.Dataset(
        {
            "temperature": xr.DataArray(
                rng.standard_normal((_N_TIMES, _N_LAT, _N_LON)).astype(np.float32),
                dims=["time", "latitude", "longitude"],
                coords={"time": times},
            ),
            "geopotential": xr.DataArray(
                rng.standard_normal((_N_TIMES, _N_LAT, _N_LON)).astype(np.float32),
                dims=["time", "latitude", "longitude"],
                coords={"time": times},
            ),
        }
    )


@pytest.fixture()
def multi_level_ds() -> xr.Dataset:
    """One pressure-level variable with a level dim, plus one single-level variable."""
    rng = np.random.default_rng(1)
    times = np.arange(_N_TIMES)
    return xr.Dataset(
        {
            "temperature": xr.DataArray(
                rng.standard_normal((_N_TIMES, _N_LEVELS, _N_LAT, _N_LON)).astype(np.float32),
                dims=["time", "level", "latitude", "longitude"],
                coords={"time": times, "level": np.arange(_N_LEVELS)},
            ),
            "2m_temperature": xr.DataArray(
                rng.standard_normal((_N_TIMES, _N_LAT, _N_LON)).astype(np.float32),
                dims=["time", "latitude", "longitude"],
                coords={"time": times},
            ),
        }
    )


class TestPermuteVariable:
    def test_deterministic_given_seed(self, single_level_ds):
        result_a = permute_variable(single_level_ds, "temperature", np.random.default_rng(42))
        result_b = permute_variable(single_level_ds, "temperature", np.random.default_rng(42))
        xr.testing.assert_identical(result_a, result_b)

    def test_different_seeds_differ(self, single_level_ds):
        result_a = permute_variable(single_level_ds, "temperature", np.random.default_rng(1))
        result_b = permute_variable(single_level_ds, "temperature", np.random.default_rng(2))
        assert not result_a["temperature"].equals(result_b["temperature"])

    def test_only_target_variable_changes(self, single_level_ds):
        result = permute_variable(single_level_ds, "temperature", np.random.default_rng(0))
        xr.testing.assert_identical(result["geopotential"], single_level_ds["geopotential"])

    def test_time_coord_preserved(self, single_level_ds):
        result = permute_variable(single_level_ds, "temperature", np.random.default_rng(0))
        xr.testing.assert_identical(result["time"], single_level_ds["time"])

    def test_values_actually_reordered(self, single_level_ds):
        result = permute_variable(single_level_ds, "temperature", np.random.default_rng(0))
        assert not np.array_equal(result["temperature"].values, single_level_ds["temperature"].values)
        original_sorted = np.sort(single_level_ds["temperature"].values, axis=0)
        permuted_sorted = np.sort(result["temperature"].values, axis=0)
        np.testing.assert_allclose(original_sorted, permuted_sorted)

    def test_multi_level_variable_shuffles_all_levels_together(self, multi_level_ds):
        result = permute_variable(multi_level_ds, "temperature", np.random.default_rng(7))
        original = multi_level_ds["temperature"]
        permuted = result["temperature"]

        def _recover_permutation(level: int) -> np.ndarray:
            orig_slice = original.isel(level=level).values.reshape(_N_TIMES, -1)
            perm_slice = permuted.isel(level=level).values.reshape(_N_TIMES, -1)
            recovered = np.empty(_N_TIMES, dtype=int)
            for new_t in range(_N_TIMES):
                matches = np.where(np.all(np.isclose(orig_slice, perm_slice[new_t]), axis=1))[0]
                recovered[new_t] = matches[0]
            return recovered

        perm_level_0 = _recover_permutation(0)
        perm_level_1 = _recover_permutation(1)
        np.testing.assert_array_equal(perm_level_0, perm_level_1)

    def test_multi_level_other_variable_untouched(self, multi_level_ds):
        result = permute_variable(multi_level_ds, "temperature", np.random.default_rng(3))
        xr.testing.assert_identical(result["2m_temperature"], multi_level_ds["2m_temperature"])

    def test_single_timestep_is_identity(self):
        ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": np.array([0])},
                )
            }
        )
        result = permute_variable(ds, "temperature", np.random.default_rng(5))
        xr.testing.assert_identical(result, ds)


class TestReduceDerivedMetrics:
    def _make_derived_ds(self, csi_values: np.ndarray, hss_values: np.ndarray) -> xr.Dataset:
        coords = {"neighborhood": NEIGHBORHOODS_KM, "threshold": THRESHOLDS}
        return xr.Dataset(
            {
                "csi_CF": (["neighborhood", "threshold"], csi_values),
                "hss_CF": (["neighborhood", "threshold"], hss_values),
            },
            coords=coords,
        )

    def test_matches_independent_reduction(self):
        rng = np.random.default_rng(11)
        n_nbhd, n_thresh = len(NEIGHBORHOODS_KM), len(THRESHOLDS)
        csi_values = rng.random((n_nbhd, n_thresh)).astype(np.float32)
        hss_values = rng.random((n_nbhd, n_thresh)).astype(np.float32)
        derived_ds = self._make_derived_ds(csi_values, hss_values)

        result = reduce_derived_metrics(derived_ds, ["CF"], _METRICS)

        expected_csi = derived_ds["csi_CF"].max("threshold").mean("neighborhood").item()
        expected_hss = derived_ds["hss_CF"].max("threshold").mean("neighborhood").item()
        assert result[("csi", "CF")] == pytest.approx(expected_csi)
        assert result[("hss", "CF")] == pytest.approx(expected_hss)

    def test_returns_one_entry_per_metric_and_front_type(self):
        n_nbhd, n_thresh = len(NEIGHBORHOODS_KM), len(THRESHOLDS)
        zeros = np.zeros((n_nbhd, n_thresh), dtype=np.float32)
        derived_ds = self._make_derived_ds(zeros, zeros)
        result = reduce_derived_metrics(derived_ds, ["CF"], _METRICS)
        assert set(result.keys()) == {("csi", "CF"), ("hss", "CF")}


class TestBuildImportanceTables:
    def test_summary_and_ranking_shapes(self):
        baseline = {("csi", "CF"): 0.5, ("csi", "WF"): 0.6}
        permuted = {
            "temperature": [
                {("csi", "CF"): 0.3, ("csi", "WF"): 0.6},
                {("csi", "CF"): 0.4, ("csi", "WF"): 0.65},
            ],
            "geopotential": [
                {("csi", "CF"): 0.55, ("csi", "WF"): 0.62},
                {("csi", "CF"): 0.5, ("csi", "WF"): 0.58},
            ],
        }
        summary_df, ranking_df = build_importance_tables(baseline, permuted)

        assert len(summary_df) == 2 * 2 * 2  # variables x repeats x (metric, front_type) pairs
        assert set(summary_df.columns) == {
            "variable",
            "repeat",
            "front_type",
            "metric",
            "baseline_value",
            "permuted_value",
            "delta",
        }
        assert len(ranking_df) == 2 * 2  # variables x (metric, front_type) pairs
        assert set(ranking_df.columns) == {"variable", "front_type", "metric", "mean_delta", "std_delta"}

    def test_delta_is_permuted_minus_baseline(self):
        baseline = {("csi", "CF"): 0.5}
        permuted = {"temperature": [{("csi", "CF"): 0.3}]}
        summary_df, _ranking_df = build_importance_tables(baseline, permuted)
        row = summary_df.iloc[0]
        assert row["delta"] == pytest.approx(row["permuted_value"] - row["baseline_value"])
        assert row["delta"] == pytest.approx(-0.2)

    def test_ranking_mean_and_std_match_numpy(self):
        baseline = {("csi", "CF"): 0.5}
        permuted = {"temperature": [{("csi", "CF"): 0.3}, {("csi", "CF"): 0.4}, {("csi", "CF"): 0.35}]}
        _summary_df, ranking_df = build_importance_tables(baseline, permuted)
        deltas = np.array([0.3, 0.4, 0.35]) - 0.5
        row = ranking_df.iloc[0]
        assert row["mean_delta"] == pytest.approx(np.mean(deltas))
        assert row["std_delta"] == pytest.approx(np.std(deltas, ddof=1))

    def test_ranking_sorted_ascending_by_mean_delta(self):
        baseline = {("csi", "CF"): 0.5}
        permuted = {
            "hurts_a_lot": [{("csi", "CF"): 0.1}],
            "no_effect": [{("csi", "CF"): 0.5}],
            "hurts_a_little": [{("csi", "CF"): 0.45}],
        }
        _summary_df, ranking_df = build_importance_tables(baseline, permuted)
        assert list(ranking_df["variable"]) == ["hurts_a_lot", "hurts_a_little", "no_effect"]


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestRunPermutationImportance:
    """End-to-end driver tests using synthetic TF-compatible fake models."""

    def _make_input_ds(self, n_times: int, n_lat: int, n_lon: int) -> xr.Dataset:
        rng = np.random.default_rng(21)
        times = np.arange(n_times)
        lats = np.linspace(30.0, 50.0, n_lat, dtype=np.float32)
        lons = np.linspace(200.0, 260.0, n_lon, dtype=np.float32)
        return xr.Dataset(
            {
                "signal_var": xr.DataArray(
                    rng.standard_normal((n_times, n_lat, n_lon)).astype(np.float32),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": times, "latitude": lats, "longitude": lons},
                ),
                "noise_var": xr.DataArray(
                    rng.standard_normal((n_times, n_lat, n_lon)).astype(np.float32),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": times, "latitude": lats, "longitude": lons},
                ),
            }
        )

    def _make_target_da(self, input_ds: xr.Dataset) -> xr.DataArray:
        """CF present wherever signal_var > 0, so the model can key off it exactly."""
        truth = (input_ds["signal_var"].values > 0).astype(np.int32)
        return xr.DataArray(
            truth,
            dims=["time", "latitude", "longitude"],
            coords={"time": input_ds["time"], "latitude": input_ds["latitude"], "longitude": input_ds["longitude"]},
        )

    def _make_signal_following_model(self, n_lat: int, n_lon: int, signal_channel: int, n_classes: int):
        class _SignalModel:
            def __call__(self, x, training=False):
                signal = x[..., signal_channel]
                background = tf.nn.relu(-tf.sign(signal))
                cf = tf.nn.relu(tf.sign(signal))
                other_classes = tf.zeros_like(signal)
                stacked = [background, cf] + [other_classes] * (n_classes - 2)
                return tf.stack(stacked, axis=-1)

        return _SignalModel()

    def test_signal_variable_permutation_hurts_more_than_noise(self, tmp_path):
        from fronts.data import datasets

        n_times, n_lat, n_lon = 8, 4, 5
        input_ds = self._make_input_ds(n_times, n_lat, n_lon)
        target_da = self._make_target_da(input_ds)
        variables = ["signal_var", "noise_var"]
        signal_channel = variables.index("signal_var")

        dummy_store = _dummy_icechunk_config()
        data_config = datasets.DatasetConfig(
            inputs_icechunk_config=dummy_store,
            targets_icechunk_config=dummy_store,
            variables=variables,
            test_years=[],
            val_years=[],
            front_dilation=0,
        )
        model = self._make_signal_following_model(n_lat, n_lon, signal_channel, n_classes=6)
        perm_cfg = PermutationConfig(
            n_repeats=2,
            seed=123,
            variables=variables,
            metrics=("csi",),
        )

        summary_df, ranking_df = run_permutation_importance(
            model=model,
            era5_ds=input_ds,
            target_da=target_da,
            data_config=data_config,
            front_types=["CF"],
            lats=input_ds["latitude"].values,
            lons=input_ds["longitude"].values,
            spatial_mask=None,
            perm_cfg=perm_cfg,
            outdir=str(tmp_path),
            batch_size=8,
            class_weights=None,
        )

        signal_delta = ranking_df.loc[ranking_df["variable"] == "signal_var", "mean_delta"].item()
        noise_delta = ranking_df.loc[ranking_df["variable"] == "noise_var", "mean_delta"].item()
        assert signal_delta < noise_delta
        assert signal_delta < 0
        assert (tmp_path / "baseline" / "stats_derived.nc").exists()
        assert (tmp_path / "permuted" / "signal_var" / "repeat_0" / "stats_derived.nc").exists()
        assert isinstance(summary_df, pd.DataFrame)


def _dummy_icechunk_config():
    from fronts import utils

    return utils.IcechunkStorageConfig(store_path="unused", branch_name="main")
