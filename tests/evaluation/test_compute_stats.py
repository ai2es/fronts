"""Tests for fronts.evaluation.compute_stats._accumulate_timestep and helpers."""

import numpy as np
import pytest
import xarray as xr

from fronts.evaluation.compute_stats import (
    N_THRESHOLDS,
    THRESHOLDS,
    _EXPAND_SIZE,
    _accumulate_timestep,
    _expand_all_neighborhoods,
    compute_stats,
)

N_NBHD = 5
RNG = np.random.default_rng(42)


@pytest.fixture()
def tiny_grid():
    """6x8 grid, 2 front types, uniform unit weights."""
    n_lat, n_lon, n_fronts = 6, 8, 2
    weights = np.ones((n_lat, n_lon), dtype=np.float32)
    return n_lat, n_lon, n_fronts, weights


@pytest.fixture()
def small_thresholds():
    """Three evenly-spaced thresholds for fast hand-verifiable tests."""
    return np.array([0.25, 0.5, 0.75], dtype=np.float32)


class TestExpandAllNeighborhoods:
    def test_shape(self, tiny_grid):
        n_lat, n_lon, n_fronts, _ = tiny_grid
        truth = np.zeros((n_lat, n_lon, n_fronts), dtype=bool)
        truth[3, 4, 0] = True
        result = _expand_all_neighborhoods(truth, n_nbhd=N_NBHD, expand_size=_EXPAND_SIZE)
        assert result.shape == (N_NBHD, n_lat, n_lon, n_fronts)

    def test_monotone_expansion(self, tiny_grid):
        """Each successive neighbourhood must be a superset of the previous."""
        n_lat, n_lon, n_fronts, _ = tiny_grid
        truth = np.zeros((n_lat, n_lon, n_fronts), dtype=bool)
        truth[3, 4, 0] = True
        stack = _expand_all_neighborhoods(truth, n_nbhd=N_NBHD, expand_size=_EXPAND_SIZE)
        for ni in range(1, N_NBHD):
            assert np.all(stack[ni] >= stack[ni - 1]), f"Neighbourhood {ni} not superset of {ni - 1}"

    def test_original_truth_included(self, tiny_grid):
        """Every true pixel in the input must appear in all expanded masks."""
        n_lat, n_lon, n_fronts, _ = tiny_grid
        truth = np.zeros((n_lat, n_lon, n_fronts), dtype=bool)
        truth[1, 2, 1] = True
        stack = _expand_all_neighborhoods(truth, n_nbhd=N_NBHD, expand_size=_EXPAND_SIZE)
        for ni in range(N_NBHD):
            assert stack[ni, 1, 2, 1], f"Original pixel missing at neighbourhood {ni}"

    def test_all_false_stays_false(self, tiny_grid):
        n_lat, n_lon, n_fronts, _ = tiny_grid
        truth = np.zeros((n_lat, n_lon, n_fronts), dtype=bool)
        stack = _expand_all_neighborhoods(truth, n_nbhd=N_NBHD, expand_size=_EXPAND_SIZE)
        assert not stack.any()


class TestAccumulateZeroPredictions:
    """All predictions are 0: no pixel exceeds any threshold > 0."""

    def test_no_tp_fp(self, tiny_grid, small_thresholds):
        n_lat, n_lon, n_fronts, weights = tiny_grid
        pred = np.zeros((n_lat, n_lon, n_fronts), dtype=np.float32)
        truth = RNG.integers(0, 2, (n_lat, n_lon, n_fronts)).astype(bool)

        tp_sp, fp_sp, _tn_sp, _fn_sp, tp_ag, fp_ag, _tn_ag, _fn_ag = _accumulate_timestep(
            pred, truth, weights, small_thresholds, N_NBHD, _EXPAND_SIZE
        )
        assert np.all(tp_sp == 0), "Expected zero TP with zero predictions"
        assert np.all(fp_sp == 0), "Expected zero FP with zero predictions"
        assert np.all(tp_ag == 0)
        assert np.all(fp_ag == 0)

    def test_fn_equals_weighted_truth(self, tiny_grid, small_thresholds):
        """FN should equal the weighted count of true pixels at every threshold."""
        n_lat, n_lon, n_fronts, weights = tiny_grid
        pred = np.zeros((n_lat, n_lon, n_fronts), dtype=np.float32)
        truth = np.zeros((n_lat, n_lon, n_fronts), dtype=bool)
        truth[1, 2, 0] = True
        truth[3, 5, 1] = True

        *_, fn_ag = _accumulate_timestep(pred, truth, weights, small_thresholds, N_NBHD, _EXPAND_SIZE)
        # fn_ag has shape (n_fronts, 1, T); should be n_true_pixels at every threshold
        expected_fn = np.array([1.0, 1.0], dtype=np.float32)  # one true pixel per front
        assert np.allclose(fn_ag[:, 0, :], expected_fn[:, np.newaxis])


class TestAccumulatePerfectPredictions:
    """Pred=1 exactly where truth=1, pred=0 elsewhere."""

    def test_no_fp(self, tiny_grid, small_thresholds):
        n_lat, n_lon, n_fronts, weights = tiny_grid
        truth = np.zeros((n_lat, n_lon, n_fronts), dtype=bool)
        truth[2, 3, 0] = True
        truth[4, 6, 1] = True
        pred = truth.astype(np.float32)

        _tp_sp, fp_sp, *_rest, fp_ag, _tn_ag, _fn_ag = _accumulate_timestep(
            pred, truth, weights, small_thresholds, N_NBHD, _EXPAND_SIZE
        )
        # Predicted positives are inside expanded truth for any neighbourhood, so FP = 0.
        assert np.all(fp_ag == 0), "Expected zero FP for perfect predictions"
        assert np.all(fp_sp == 0)

    def test_no_fn(self, tiny_grid, small_thresholds):
        n_lat, n_lon, n_fronts, weights = tiny_grid
        truth = np.zeros((n_lat, n_lon, n_fronts), dtype=bool)
        truth[2, 3, 0] = True
        pred = truth.astype(np.float32)

        *_, fn_ag = _accumulate_timestep(pred, truth, weights, small_thresholds, N_NBHD, _EXPAND_SIZE)
        # All thresholds <= 1.0; pred=1 exceeds every threshold, so FN = 0.
        assert np.all(fn_ag == 0), "Expected zero FN for perfect predictions"


class TestAccumulateKnownValues:
    """2x2 grid, 1 front type, uniform weights=1, 1 neighbourhood."""

    def _run(self, pred_vals, truth_vals, thresholds):
        n_lat, n_lon, n_fronts = 2, 2, 1
        pred = np.array(pred_vals, dtype=np.float32).reshape(n_lat, n_lon, n_fronts)
        truth = np.array(truth_vals, dtype=bool).reshape(n_lat, n_lon, n_fronts)
        weights = np.ones((n_lat, n_lon), dtype=np.float32)
        return _accumulate_timestep(pred, truth, weights, thresholds, n_nbhd=1, expand_size=3)

    def test_tn_count(self, small_thresholds):
        # pred all 0, truth all False -> everything is TN
        _tp_sp, _fp_sp, _tn_sp, _fn_sp, tp_ag, fp_ag, tn_ag, fn_ag = self._run(
            [0, 0, 0, 0], [False, False, False, False], small_thresholds
        )
        n_pixels = 4
        assert np.allclose(tn_ag[0, 0, :], n_pixels)  # all 4 pixels TN at every threshold
        assert np.all(tp_ag == 0)
        assert np.all(fp_ag == 0)
        assert np.all(fn_ag == 0)

    def test_threshold_splits(self, small_thresholds):
        # pred = [0.6, 0.6, 0.6, 0.6], truth = all False
        # threshold 0.25: all above -> all FP=4, TN=0
        # threshold 0.75: none above -> all TN=4, FP=0
        _tp_sp, _fp_sp, _tn_sp, _fn_sp, tp_ag, fp_ag, tn_ag, fn_ag = self._run(
            [0.6, 0.6, 0.6, 0.6], [False, False, False, False], small_thresholds
        )
        assert fp_ag[0, 0, 0] == pytest.approx(4.0)  # threshold 0.25: all above
        assert tn_ag[0, 0, 0] == pytest.approx(0.0)
        assert fp_ag[0, 0, 2] == pytest.approx(0.0)  # threshold 0.75: none above
        assert tn_ag[0, 0, 2] == pytest.approx(4.0)
        assert np.all(tp_ag == 0)
        assert np.all(fn_ag == 0)

    def test_consistency(self, small_thresholds):
        """TP + FP + TN + FN == weighted pixel count at every threshold/neighbourhood."""
        pred = RNG.random((2, 2, 1)).astype(np.float32)
        truth = RNG.integers(0, 2, (2, 2, 1)).astype(bool)
        weights = np.ones((2, 2), dtype=np.float32)
        _tp_sp, _fp_sp, _tn_sp, _fn_sp, tp_ag, fp_ag, tn_ag, fn_ag = _accumulate_timestep(
            pred, truth, weights, small_thresholds, n_nbhd=1, expand_size=3
        )
        total = tp_ag + fp_ag + tn_ag + fn_ag  # (1, 1, T) after broadcasting
        # Total weighted pixels = 4 (2x2 grid, uniform weight 1)
        assert np.allclose(total, 4.0, atol=1e-5)


class TestAccumulateSpatialMask:
    def test_masked_pixels_contribute_nothing(self, small_thresholds):
        n_lat, n_lon, n_fronts = 4, 4, 1
        pred = np.ones((n_lat, n_lon, n_fronts), dtype=np.float32)
        truth = np.ones((n_lat, n_lon, n_fronts), dtype=bool)
        weights = np.zeros((n_lat, n_lon), dtype=np.float32)  # mask out everything
        _tp_sp, _fp_sp, _tn_sp, _fn_sp, tp_ag, fp_ag, tn_ag, fn_ag = _accumulate_timestep(
            pred, truth, weights, small_thresholds, N_NBHD, _EXPAND_SIZE
        )
        for arr in (tp_ag, fp_ag, tn_ag, fn_ag):
            assert np.all(arr == 0), "Fully masked grid should produce zero stats"


class TestNeighbourhoodExpansion:
    def test_tp_nondecreasing_with_neighbourhood(self):
        """TP can only grow as the neighbourhood radius increases."""
        n_lat, n_lon, n_fronts = 9, 9, 1
        weights = np.ones((n_lat, n_lon), dtype=np.float32)

        # Single truth pixel in the centre; prediction at the same pixel.
        truth = np.zeros((n_lat, n_lon, n_fronts), dtype=bool)
        truth[4, 4, 0] = True
        pred = np.zeros((n_lat, n_lon, n_fronts), dtype=np.float32)
        pred[4, 4, 0] = 0.6

        thresholds = np.array([0.5], dtype=np.float32)
        *_, tp_ag, _fp_ag, _tn_ag, _fn_ag = _accumulate_timestep(
            pred, truth, weights, thresholds, n_nbhd=N_NBHD, expand_size=_EXPAND_SIZE
        )
        for ni in range(1, N_NBHD):
            assert tp_ag[0, ni, 0] >= tp_ag[0, ni - 1, 0], f"TP decreased from neighbourhood {ni - 1} to {ni}"


class TestOutputShapes:
    def test_shapes(self, tiny_grid):
        n_lat, n_lon, n_fronts, weights = tiny_grid
        pred = RNG.random((n_lat, n_lon, n_fronts)).astype(np.float32)
        truth = pred > 0.5
        tp_sp, fp_sp, tn_sp, fn_sp, tp_ag, fp_ag, tn_ag, fn_ag = _accumulate_timestep(
            pred, truth, weights, THRESHOLDS, N_NBHD, _EXPAND_SIZE
        )
        assert tp_sp.shape == (n_fronts, n_lat, n_lon, N_NBHD, N_THRESHOLDS)
        assert fp_sp.shape == (n_fronts, n_lat, n_lon, N_NBHD, N_THRESHOLDS)
        assert tn_sp.shape == (n_fronts, n_lat, n_lon, 1, N_THRESHOLDS)
        assert fn_sp.shape == (n_fronts, n_lat, n_lon, 1, N_THRESHOLDS)
        assert tp_ag.shape == (n_fronts, N_NBHD, N_THRESHOLDS)
        assert fp_ag.shape == (n_fronts, N_NBHD, N_THRESHOLDS)
        assert tn_ag.shape == (n_fronts, 1, N_THRESHOLDS)
        assert fn_ag.shape == (n_fronts, 1, N_THRESHOLDS)


class TestComputeStatsShapes:
    def test_output_shapes_and_variables(self):
        """Run compute_stats end-to-end with a synthetic callable and check outputs."""
        n_lat, n_lon, n_times, n_ch, n_classes = 6, 8, 3, 4, 6
        front_types = ["CF", "WF"]

        lats = np.linspace(30.0, 50.0, n_lat, dtype=np.float32)
        lons = np.linspace(200.0, 260.0, n_lon, dtype=np.float32)
        times = np.arange(n_times)

        era5_da = xr.DataArray(
            RNG.random((n_times, n_lat, n_lon, n_ch)).astype(np.float32),
            dims=["time", "latitude", "longitude", "channel"],
            coords={"time": times, "latitude": lats, "longitude": lons},
        )
        targets_da = xr.DataArray(
            (RNG.random((n_times, n_lat, n_lon, n_classes)) > 0.8).astype(np.float32),
            dims=["time", "latitude", "longitude", "class"],
            coords={"time": times, "latitude": lats, "longitude": lons},
        )

        def fixed_model(x, training=False):
            return np.full((x.shape[0], n_lat, n_lon, n_classes), 0.4, dtype=np.float32)

        spatial_ds, aggregate_ds = compute_stats(
            model=fixed_model,
            era5_da=era5_da,
            targets_da=targets_da,
            front_types=front_types,
            lats=lats,
            lons=lons,
            spatial_mask=None,
        )

        for ft in front_types:
            for metric in ("tp", "fp", "tn", "fn"):
                sp_var = f"{metric}_spatial_{ft}"
                ag_var = f"{metric}_{ft}"
                assert sp_var in spatial_ds, f"Missing variable {sp_var}"
                assert ag_var in aggregate_ds, f"Missing variable {ag_var}"
                assert spatial_ds[sp_var].shape == (n_lat, n_lon, N_NBHD, N_THRESHOLDS)
                assert aggregate_ds[ag_var].shape == (N_NBHD, N_THRESHOLDS)

    def test_nonnegative_outputs(self):
        """All stats values must be non-negative."""
        n_lat, n_lon, n_times, n_ch, n_classes = 4, 4, 2, 2, 6
        lats = np.linspace(20.0, 40.0, n_lat, dtype=np.float32)
        lons = np.linspace(100.0, 120.0, n_lon, dtype=np.float32)
        times = np.arange(n_times)

        era5_da = xr.DataArray(
            RNG.random((n_times, n_lat, n_lon, n_ch)).astype(np.float32),
            dims=["time", "latitude", "longitude", "channel"],
            coords={"time": times, "latitude": lats, "longitude": lons},
        )
        targets_da = xr.DataArray(
            np.zeros((n_times, n_lat, n_lon, n_classes), dtype=np.float32),
            dims=["time", "latitude", "longitude", "class"],
            coords={"time": times, "latitude": lats, "longitude": lons},
        )

        def zero_model(x, training=False):
            return np.zeros((x.shape[0], n_lat, n_lon, n_classes), dtype=np.float32)

        _, aggregate_ds = compute_stats(
            model=zero_model,
            era5_da=era5_da,
            targets_da=targets_da,
            front_types=["CF"],
            lats=lats,
            lons=lons,
            spatial_mask=None,
        )
        for var in aggregate_ds.data_vars:
            assert np.all(aggregate_ds[var].values >= 0), f"{var} contains negative values"
