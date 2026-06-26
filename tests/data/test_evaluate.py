"""Tests for fronts.evaluation.compute_stats helpers and end-to-end stats computation."""

import numpy as np
import pytest
import xarray as xr

from fronts.evaluate import (
    NEIGHBORHOODS_KM,
    N_THRESHOLDS,
    THRESHOLDS,
    _expand_all_neighborhoods,
    compute_stats,
)

_NBH_STEP_KM = float(np.unique(np.diff(NEIGHBORHOODS_KM)).item())
_LAT_RES_KM = 0.25 * np.pi * 6371.0 / 180.0
_LAT_PIXELS = round(_NBH_STEP_KM / _LAT_RES_KM)

N_NBHD = 5
RNG = np.random.default_rng(42)


def _lon_pixels(lats: np.ndarray) -> np.ndarray:
    return np.round(_NBH_STEP_KM / (_LAT_RES_KM * np.cos(np.deg2rad(lats)))).astype(int)


@pytest.fixture()
def tiny_lats():
    """6-row latitude array spanning 30-50N, matching tiny_grid."""
    return np.linspace(30.0, 50.0, 6, dtype=np.float32)


@pytest.fixture()
def tiny_grid():
    """6x8 grid, 2 front types, uniform unit weights."""
    n_lat, n_lon, n_fronts = 6, 8, 2
    weights = np.ones((n_lat, n_lon), dtype=np.float32)
    return n_lat, n_lon, n_fronts, weights


class TestExpandAllNeighborhoods:
    def test_shape(self, tiny_grid, tiny_lats):
        n_lat, n_lon, n_fronts, _ = tiny_grid
        truth = np.zeros((n_lat, n_lon, n_fronts), dtype=bool)
        truth[3, 4, 0] = True
        result = _expand_all_neighborhoods(
            truth, n_nbhd=N_NBHD, lat_pixels_per_step=_LAT_PIXELS, lon_pixels_per_lat=_lon_pixels(tiny_lats)
        )
        assert result.shape == (N_NBHD, n_lat, n_lon, n_fronts)

    def test_monotone_expansion(self, tiny_grid, tiny_lats):
        """Each successive neighbourhood must be a superset of the previous."""
        n_lat, n_lon, n_fronts, _ = tiny_grid
        truth = np.zeros((n_lat, n_lon, n_fronts), dtype=bool)
        truth[3, 4, 0] = True
        stack = _expand_all_neighborhoods(
            truth, n_nbhd=N_NBHD, lat_pixels_per_step=_LAT_PIXELS, lon_pixels_per_lat=_lon_pixels(tiny_lats)
        )
        for ni in range(1, N_NBHD):
            assert np.all(stack[ni] >= stack[ni - 1]), f"Neighbourhood {ni} not superset of {ni - 1}"

    def test_original_truth_included(self, tiny_grid, tiny_lats):
        """Every true pixel in the input must appear in all expanded masks."""
        n_lat, n_lon, n_fronts, _ = tiny_grid
        truth = np.zeros((n_lat, n_lon, n_fronts), dtype=bool)
        truth[1, 2, 1] = True
        stack = _expand_all_neighborhoods(
            truth, n_nbhd=N_NBHD, lat_pixels_per_step=_LAT_PIXELS, lon_pixels_per_lat=_lon_pixels(tiny_lats)
        )
        for ni in range(N_NBHD):
            assert stack[ni, 1, 2, 1], f"Original pixel missing at neighbourhood {ni}"

    def test_all_false_stays_false(self, tiny_grid, tiny_lats):
        n_lat, n_lon, n_fronts, _ = tiny_grid
        truth = np.zeros((n_lat, n_lon, n_fronts), dtype=bool)
        stack = _expand_all_neighborhoods(
            truth, n_nbhd=N_NBHD, lat_pixels_per_step=_LAT_PIXELS, lon_pixels_per_lat=_lon_pixels(tiny_lats)
        )
        assert not stack.any()


def _make_compute_stats_inputs(n_lat, n_lon, n_times, n_classes, lats, lons):
    times = np.arange(n_times)
    era5_da = xr.DataArray(
        RNG.random((n_times, n_lat, n_lon, 2)).astype(np.float32),
        dims=["time", "latitude", "longitude", "channel"],
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    targets_da = xr.DataArray(
        np.zeros((n_times, n_lat, n_lon, n_classes), dtype=np.float32),
        dims=["time", "latitude", "longitude", "class"],
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    return era5_da, targets_da


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

        class _FixedModel:
            def predict(self, dataset):
                return np.full((n_times, n_lat, n_lon, n_classes), 0.4, dtype=np.float32)

        spatial_ds, aggregate_ds = compute_stats(
            model=_FixedModel(),
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
        n_lat, n_lon, n_times, n_classes = 4, 4, 2, 6
        lats = np.linspace(20.0, 40.0, n_lat, dtype=np.float32)
        lons = np.linspace(100.0, 120.0, n_lon, dtype=np.float32)
        era5_da, targets_da = _make_compute_stats_inputs(n_lat, n_lon, n_times, n_classes, lats, lons)

        class _ZeroModel:
            def predict(self, dataset):
                return np.zeros((n_times, n_lat, n_lon, n_classes), dtype=np.float32)

        _, aggregate_ds = compute_stats(
            model=_ZeroModel(),
            era5_da=era5_da,
            targets_da=targets_da,
            front_types=["CF"],
            lats=lats,
            lons=lons,
            spatial_mask=None,
        )
        for var in aggregate_ds.data_vars:
            assert np.all(aggregate_ds[var].values >= 0), f"{var} contains negative values"

    def test_zero_predictions_no_tp_fp(self):
        """Zero predictions must produce zero TP and FP at every threshold."""
        n_lat, n_lon, n_times, n_classes = 4, 4, 2, 6
        lats = np.linspace(20.0, 40.0, n_lat, dtype=np.float32)
        lons = np.linspace(100.0, 120.0, n_lon, dtype=np.float32)
        era5_da, targets_da = _make_compute_stats_inputs(n_lat, n_lon, n_times, n_classes, lats, lons)

        class _ZeroModel:
            def predict(self, dataset):
                return np.zeros((n_times, n_lat, n_lon, n_classes), dtype=np.float32)

        _, aggregate_ds = compute_stats(
            model=_ZeroModel(),
            era5_da=era5_da,
            targets_da=targets_da,
            front_types=["CF"],
            lats=lats,
            lons=lons,
            spatial_mask=None,
        )
        assert np.all(aggregate_ds["tp_CF"].values == 0)
        assert np.all(aggregate_ds["fp_CF"].values == 0)

    def test_tp_nondecreasing_with_neighbourhood(self):
        """TP can only grow as the neighbourhood radius increases."""
        n_lat, n_lon, n_times, n_classes = 9, 9, 1, 6
        lats = np.linspace(30.0, 50.0, n_lat, dtype=np.float32)
        lons = np.linspace(200.0, 240.0, n_lon, dtype=np.float32)
        times = np.arange(n_times)

        targets_np = np.zeros((n_times, n_lat, n_lon, n_classes), dtype=np.float32)
        targets_np[0, 4, 4, 1] = 1.0  # single CF pixel at centre
        era5_da = xr.DataArray(
            RNG.random((n_times, n_lat, n_lon, 2)).astype(np.float32),
            dims=["time", "latitude", "longitude", "channel"],
            coords={"time": times, "latitude": lats, "longitude": lons},
        )
        targets_da = xr.DataArray(
            targets_np,
            dims=["time", "latitude", "longitude", "class"],
            coords={"time": times, "latitude": lats, "longitude": lons},
        )

        pred_np = np.zeros((n_times, n_lat, n_lon, n_classes), dtype=np.float32)
        pred_np[0, 4, 4, 1] = 0.6  # predict at truth pixel

        class _PointModel:
            def predict(self, dataset):
                return pred_np

        _, aggregate_ds = compute_stats(
            model=_PointModel(),
            era5_da=era5_da,
            targets_da=targets_da,
            front_types=["CF"],
            lats=lats,
            lons=lons,
            spatial_mask=None,
        )
        tp = aggregate_ds["tp_CF"].values  # (N_NBHD, N_THRESHOLDS)
        threshold_idx = np.searchsorted(THRESHOLDS, 0.5)
        for ni in range(1, N_NBHD):
            assert tp[ni, threshold_idx] >= tp[ni - 1, threshold_idx], f"TP decreased at neighbourhood {ni}"
