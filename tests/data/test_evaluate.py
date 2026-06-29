"""Tests for fronts.evaluate helpers and end-to-end stats computation."""

import numpy as np
import pytest
import xarray as xr

from fronts.evaluate import (
    NEIGHBORHOODS_KM,
    N_THRESHOLDS,
    THRESHOLDS,
    _expand_all_neighborhoods,
    compute_derived_stats,
    compute_stats,
)

try:
    import tensorflow as tf

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

_NBH_STEP_KM = float(np.unique(np.diff(NEIGHBORHOODS_KM)).item())
_LAT_RES_KM = 0.25 * np.pi * 6371.0 / 180.0
_LAT_PIXELS = round(_NBH_STEP_KM / _LAT_RES_KM)

N_NBHD = 5
RNG = np.random.default_rng(42)

_N_LAT = 6
_N_LON = 8
_N_TIMES = 3
_N_CLASSES = 6
_FRONT_TYPES = ["CF", "WF"]
_VARIABLES = ["geopotential", "temperature"]


def _lon_pixels(lats: np.ndarray) -> np.ndarray:
    return np.round(_NBH_STEP_KM / (_LAT_RES_KM * np.cos(np.deg2rad(lats)))).astype(int)


@pytest.fixture()
def tiny_lats() -> np.ndarray:
    """6-row latitude array spanning 30-50N."""
    return np.linspace(30.0, 50.0, _N_LAT, dtype=np.float32)


@pytest.fixture()
def tiny_lons() -> np.ndarray:
    """8-column longitude array."""
    return np.linspace(200.0, 260.0, _N_LON, dtype=np.float32)


@pytest.fixture()
def small_input_ds(tiny_lats, tiny_lons) -> xr.Dataset:
    """Raw ERA5 Dataset with two variables, small spatial extent."""
    times = np.arange(_N_TIMES)
    return xr.Dataset(
        {
            var: xr.DataArray(
                RNG.standard_normal((_N_TIMES, _N_LAT, _N_LON)).astype(np.float32),
                dims=["time", "latitude", "longitude"],
                coords={"time": times, "latitude": tiny_lats, "longitude": tiny_lons},
            )
            for var in _VARIABLES
        }
    )


@pytest.fixture()
def small_target_da(tiny_lats, tiny_lons) -> xr.DataArray:
    """Raw integer front-code DataArray (0=background, 1=CF)."""
    times = np.arange(_N_TIMES)
    data = RNG.integers(0, 2, size=(_N_TIMES, _N_LAT, _N_LON)).astype(np.int32)
    return xr.DataArray(
        data,
        dims=["time", "latitude", "longitude"],
        coords={"time": times, "latitude": tiny_lats, "longitude": tiny_lons},
    )


@pytest.fixture()
def small_data_config():
    from fronts import utils
    from fronts.data import datasets

    dummy_store = utils.IcechunkStorageConfig(store_path="unused", branch_name="main")
    return datasets.DatasetConfig(
        inputs_icechunk_config=dummy_store,
        targets_icechunk_config=dummy_store,
        variables=_VARIABLES,
        test_years=[],
        val_years=[],
        front_dilation=0,
    )


@pytest.fixture()
def aggregate_ds_fixture(tiny_lats, tiny_lons) -> xr.Dataset:
    """Synthetic aggregate TP/FP/TN/FN Dataset for derive tests."""
    n_nbhd = len(NEIGHBORHOODS_KM)
    coords = {"neighborhood": NEIGHBORHOODS_KM, "threshold": THRESHOLDS}
    rng = np.random.default_rng(7)
    return xr.Dataset(
        {
            f"{k}_{ft}": (["neighborhood", "threshold"], rng.random((n_nbhd, N_THRESHOLDS)).astype(np.float32))
            for ft in _FRONT_TYPES
            for k in ("tp", "fp", "tn", "fn")
        },
        coords=coords,
    )


@pytest.fixture()
def spatial_ds_fixture(tiny_lats, tiny_lons) -> xr.Dataset:
    """Synthetic spatial TP/FP/TN/FN Dataset for derive tests."""
    n_nbhd = len(NEIGHBORHOODS_KM)
    coords = {
        "latitude": tiny_lats,
        "longitude": tiny_lons,
        "neighborhood": NEIGHBORHOODS_KM,
        "threshold": THRESHOLDS,
    }
    rng = np.random.default_rng(8)
    return xr.Dataset(
        {
            f"{k}_spatial_{ft}": (
                ["latitude", "longitude", "neighborhood", "threshold"],
                rng.random((_N_LAT, _N_LON, n_nbhd, N_THRESHOLDS)).astype(np.float32),
            )
            for ft in _FRONT_TYPES
            for k in ("tp", "fp", "tn", "fn")
        },
        coords=coords,
    )


class TestExpandAllNeighborhoods:
    def test_shape(self, tiny_lats):
        truth = np.zeros((_N_LAT, _N_LON, 2), dtype=bool)
        truth[3, 4, 0] = True
        result = _expand_all_neighborhoods(
            truth, n_nbhd=N_NBHD, lat_pixels_per_step=_LAT_PIXELS, lon_pixels_per_lat=_lon_pixels(tiny_lats)
        )
        assert result.shape == (N_NBHD, _N_LAT, _N_LON, 2)

    def test_monotone_expansion(self, tiny_lats):
        """Each successive neighbourhood must be a superset of the previous."""
        truth = np.zeros((_N_LAT, _N_LON, 2), dtype=bool)
        truth[3, 4, 0] = True
        stack = _expand_all_neighborhoods(
            truth, n_nbhd=N_NBHD, lat_pixels_per_step=_LAT_PIXELS, lon_pixels_per_lat=_lon_pixels(tiny_lats)
        )
        for ni in range(1, N_NBHD):
            assert np.all(stack[ni] >= stack[ni - 1]), f"Neighbourhood {ni} not superset of {ni - 1}"

    def test_original_truth_included(self, tiny_lats):
        """Every true pixel in the input must appear in all expanded masks."""
        truth = np.zeros((_N_LAT, _N_LON, 2), dtype=bool)
        truth[1, 2, 1] = True
        stack = _expand_all_neighborhoods(
            truth, n_nbhd=N_NBHD, lat_pixels_per_step=_LAT_PIXELS, lon_pixels_per_lat=_lon_pixels(tiny_lats)
        )
        for ni in range(N_NBHD):
            assert stack[ni, 1, 2, 1], f"Original pixel missing at neighbourhood {ni}"

    def test_all_false_stays_false(self, tiny_lats):
        truth = np.zeros((_N_LAT, _N_LON, 2), dtype=bool)
        stack = _expand_all_neighborhoods(
            truth, n_nbhd=N_NBHD, lat_pixels_per_step=_LAT_PIXELS, lon_pixels_per_lat=_lon_pixels(tiny_lats)
        )
        assert not stack.any()


class TestComputeDerivedStats:
    def test_output_variables_present(self, aggregate_ds_fixture, spatial_ds_fixture):
        derived = compute_derived_stats(aggregate_ds_fixture, spatial_ds_fixture, _FRONT_TYPES)
        for ft in _FRONT_TYPES:
            for var in ("pod", "sr", "csi", "fb", "hss", "obs_rel_freq", "rel_forecast_frac", "spatial_csi"):
                assert f"{var}_{ft}" in derived, f"Missing {var}_{ft}"

    def test_pod_sr_csi_in_unit_range(self, aggregate_ds_fixture, spatial_ds_fixture):
        derived = compute_derived_stats(aggregate_ds_fixture, spatial_ds_fixture, _FRONT_TYPES)
        for ft in _FRONT_TYPES:
            for var in ("pod", "sr", "csi"):
                vals = derived[f"{var}_{ft}"].values
                assert np.all(vals >= 0), f"{var}_{ft} has negative values"
                assert np.all(vals <= 1), f"{var}_{ft} exceeds 1"

    def test_obs_rel_freq_shape(self, aggregate_ds_fixture, spatial_ds_fixture):
        derived = compute_derived_stats(aggregate_ds_fixture, spatial_ds_fixture, _FRONT_TYPES)
        for ft in _FRONT_TYPES:
            shape = derived[f"obs_rel_freq_{ft}"].shape
            assert shape == (N_NBHD, N_THRESHOLDS - 1), f"obs_rel_freq_{ft} shape {shape}"

    def test_spatial_csi_shape(self, aggregate_ds_fixture, spatial_ds_fixture, tiny_lats, tiny_lons):
        derived = compute_derived_stats(aggregate_ds_fixture, spatial_ds_fixture, _FRONT_TYPES)
        for ft in _FRONT_TYPES:
            shape = derived[f"spatial_csi_{ft}"].shape
            assert shape == (_N_LAT, _N_LON, N_NBHD, N_THRESHOLDS), f"spatial_csi_{ft} shape {shape}"

    def test_rel_forecast_frac_shape(self, aggregate_ds_fixture, spatial_ds_fixture):
        derived = compute_derived_stats(aggregate_ds_fixture, spatial_ds_fixture, _FRONT_TYPES)
        for ft in _FRONT_TYPES:
            shape = derived[f"rel_forecast_frac_{ft}"].shape
            assert shape == (N_THRESHOLDS,), f"rel_forecast_frac_{ft} shape {shape}"

    def test_zero_tp_fp_gives_zero_pod_sr(self):
        coords = {"neighborhood": NEIGHBORHOODS_KM, "threshold": THRESHOLDS}
        zero = xr.DataArray(np.zeros((N_NBHD, N_THRESHOLDS), dtype=np.float32), dims=["neighborhood", "threshold"])
        agg = xr.Dataset({"tp_CF": zero, "fp_CF": zero, "tn_CF": zero, "fn_CF": zero}, coords=coords)
        sp_coords = {
            "latitude": np.array([30.0]),
            "longitude": np.array([200.0]),
            "neighborhood": NEIGHBORHOODS_KM,
            "threshold": THRESHOLDS,
        }
        sp_zero = xr.DataArray(
            np.zeros((1, 1, N_NBHD, N_THRESHOLDS), dtype=np.float32),
            dims=["latitude", "longitude", "neighborhood", "threshold"],
        )
        sp = xr.Dataset(
            {"tp_spatial_CF": sp_zero, "fp_spatial_CF": sp_zero, "fn_spatial_CF": sp_zero},
            coords=sp_coords,
        )
        derived = compute_derived_stats(agg, sp, ["CF"])
        assert np.all(derived["pod_CF"].values == 0)
        assert np.all(derived["sr_CF"].values == 0)

    def test_perfect_forecast_pod_sr_one(self):
        """When TP = N and FP = FN = 0, POD = SR = CSI = 1."""
        n = 10.0
        coords = {"neighborhood": NEIGHBORHOODS_KM, "threshold": THRESHOLDS}
        ones = xr.DataArray(np.full((N_NBHD, N_THRESHOLDS), n, dtype=np.float32), dims=["neighborhood", "threshold"])
        zeros = xr.DataArray(np.zeros((N_NBHD, N_THRESHOLDS), dtype=np.float32), dims=["neighborhood", "threshold"])
        agg = xr.Dataset({"tp_CF": ones, "fp_CF": zeros, "tn_CF": zeros, "fn_CF": zeros}, coords=coords)
        sp_coords = {
            "latitude": np.array([30.0]),
            "longitude": np.array([200.0]),
            "neighborhood": NEIGHBORHOODS_KM,
            "threshold": THRESHOLDS,
        }
        sp_ones = xr.DataArray(
            np.full((1, 1, N_NBHD, N_THRESHOLDS), n, dtype=np.float32),
            dims=["latitude", "longitude", "neighborhood", "threshold"],
        )
        sp_zeros = xr.DataArray(
            np.zeros((1, 1, N_NBHD, N_THRESHOLDS), dtype=np.float32),
            dims=["latitude", "longitude", "neighborhood", "threshold"],
        )
        sp = xr.Dataset(
            {"tp_spatial_CF": sp_ones, "fp_spatial_CF": sp_zeros, "fn_spatial_CF": sp_zeros}, coords=sp_coords
        )
        derived = compute_derived_stats(agg, sp, ["CF"])
        np.testing.assert_allclose(derived["pod_CF"].values, 1.0)
        np.testing.assert_allclose(derived["sr_CF"].values, 1.0)
        np.testing.assert_allclose(derived["csi_CF"].values, 1.0)


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestComputeStatsShapes:
    """End-to-end compute_stats tests using a synthetic TF-compatible model."""

    def _make_model(self, n_lat: int, n_lon: int, n_classes: int):
        class _FixedModel:
            def __init__(self, n_lat, n_lon, n_classes):
                self._shape = tf.constant([n_lat, n_lon, n_classes], dtype=tf.int32)

            def __call__(self, x, training=False):
                b = tf.shape(x)[0:1]
                shape = tf.concat([b, self._shape], axis=0)
                return tf.fill(shape, tf.constant(0.4, dtype=tf.float32))

        return _FixedModel(n_lat, n_lon, n_classes)

    def test_output_shapes_and_variables(
        self, small_input_ds, small_target_da, small_data_config, tiny_lats, tiny_lons
    ):
        model = self._make_model(_N_LAT, _N_LON, _N_CLASSES)
        spatial_ds, aggregate_ds, _derived = compute_stats(
            model=model,
            input_ds=small_input_ds,
            target_da=small_target_da,
            data_config=small_data_config,
            front_types=_FRONT_TYPES,
            lats=tiny_lats,
            lons=tiny_lons,
            spatial_mask=None,
        )
        for ft in _FRONT_TYPES:
            for metric in ("tp", "fp", "tn", "fn"):
                assert f"{metric}_spatial_{ft}" in spatial_ds
                assert f"{metric}_{ft}" in aggregate_ds
                assert spatial_ds[f"{metric}_spatial_{ft}"].shape == (_N_LAT, _N_LON, N_NBHD, N_THRESHOLDS)
                assert aggregate_ds[f"{metric}_{ft}"].shape == (N_NBHD, N_THRESHOLDS)

    def test_derived_ds_variables_present(
        self, small_input_ds, small_target_da, small_data_config, tiny_lats, tiny_lons
    ):
        model = self._make_model(_N_LAT, _N_LON, _N_CLASSES)
        _, _, derived_ds = compute_stats(
            model=model,
            input_ds=small_input_ds,
            target_da=small_target_da,
            data_config=small_data_config,
            front_types=_FRONT_TYPES,
            lats=tiny_lats,
            lons=tiny_lons,
            spatial_mask=None,
        )
        for ft in _FRONT_TYPES:
            for var in ("pod", "sr", "csi", "fb", "hss", "obs_rel_freq", "rel_forecast_frac", "spatial_csi"):
                assert f"{var}_{ft}" in derived_ds, f"Missing {var}_{ft} in derived_ds"

    def test_nonnegative_outputs(self, small_input_ds, small_target_da, small_data_config, tiny_lats, tiny_lons):
        """All TP/FP/TN/FN stats values must be non-negative."""
        model = self._make_model(_N_LAT, _N_LON, _N_CLASSES)
        _, aggregate_ds, _ = compute_stats(
            model=model,
            input_ds=small_input_ds,
            target_da=small_target_da,
            data_config=small_data_config,
            front_types=["CF"],
            lats=tiny_lats,
            lons=tiny_lons,
            spatial_mask=None,
        )
        for var in aggregate_ds.data_vars:
            assert np.all(aggregate_ds[var].values >= 0), f"{var} contains negative values"

    def test_zero_predictions_no_tp_fp(self, small_input_ds, small_target_da, small_data_config, tiny_lats, tiny_lons):
        """Zero predictions must produce zero TP and FP at every threshold."""

        class _ZeroModel:
            def __init__(self, n_lat, n_lon, n_classes):
                self._shape = tf.constant([n_lat, n_lon, n_classes], dtype=tf.int32)

            def __call__(self, x, training=False):
                b = tf.shape(x)[0:1]
                return tf.zeros(tf.concat([b, self._shape], axis=0), dtype=tf.float32)

        model = _ZeroModel(_N_LAT, _N_LON, _N_CLASSES)
        _, aggregate_ds, _ = compute_stats(
            model=model,
            input_ds=small_input_ds,
            target_da=small_target_da,
            data_config=small_data_config,
            front_types=["CF"],
            lats=tiny_lats,
            lons=tiny_lons,
            spatial_mask=None,
        )
        assert np.all(aggregate_ds["tp_CF"].values == 0)
        assert np.all(aggregate_ds["fp_CF"].values == 0)

    def test_tp_nondecreasing_with_neighbourhood(self, small_data_config, tiny_lats, tiny_lons):
        """TP can only grow as the neighbourhood radius increases."""
        n_lat, n_lon = 9, 9
        times = np.arange(1)
        lats = np.linspace(30.0, 50.0, n_lat, dtype=np.float32)
        lons = np.linspace(200.0, 240.0, n_lon, dtype=np.float32)

        input_ds = xr.Dataset(
            {
                var: xr.DataArray(
                    RNG.standard_normal((1, n_lat, n_lon)).astype(np.float32),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": times, "latitude": lats, "longitude": lons},
                )
                for var in _VARIABLES
            }
        )
        target_np = np.zeros((1, n_lat, n_lon), dtype=np.int32)
        target_np[0, 4, 4] = 1  # single CF pixel at centre
        target_da = xr.DataArray(
            target_np,
            dims=["time", "latitude", "longitude"],
            coords={"time": times, "latitude": lats, "longitude": lons},
        )

        class _PointModel:
            def __init__(self, n_lat, n_lon, n_classes, row, col, class_idx, value):
                self._n_lat = n_lat
                self._n_lon = n_lon
                self._n_classes = n_classes
                self._row = row
                self._col = col
                self._class_idx = class_idx
                self._value = value

            def __call__(self, x, training=False):
                b = tf.shape(x)[0:1]
                shape = tf.concat([b, tf.constant([self._n_lat, self._n_lon, self._n_classes])], axis=0)
                preds = tf.zeros(shape, dtype=tf.float32)
                indices = tf.constant([[0, self._row, self._col, self._class_idx]])
                updates = tf.constant([self._value])
                return tf.tensor_scatter_nd_update(preds, indices, updates)

        model = _PointModel(n_lat, n_lon, _N_CLASSES, row=4, col=4, class_idx=1, value=0.6)
        _, aggregate_ds, _ = compute_stats(
            model=model,
            input_ds=input_ds,
            target_da=target_da,
            data_config=small_data_config,
            front_types=["CF"],
            lats=lats,
            lons=lons,
            spatial_mask=None,
        )
        tp = aggregate_ds["tp_CF"].values  # (N_NBHD, N_THRESHOLDS)
        threshold_idx = int(np.searchsorted(THRESHOLDS, 0.5))
        for ni in range(1, N_NBHD):
            assert tp[ni, threshold_idx] >= tp[ni - 1, threshold_idx], f"TP decreased at neighbourhood {ni}"
