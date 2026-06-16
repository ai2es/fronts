import numpy as np
import pytest
import xarray as xr

from fronts import utils


def _make_da(lons: list[float]) -> xr.DataArray:
    data = np.zeros(len(lons))
    return xr.DataArray(data, dims=["longitude"], coords={"longitude": lons})


def _make_ds(lons: list[float]) -> xr.Dataset:
    data = np.zeros(len(lons))
    return xr.Dataset({"var": xr.DataArray(data, dims=["longitude"], coords={"longitude": lons})})


class TestUnwrapLongitude:
    def test_already_monotonic_0_to_360(self):
        lons = [0.0, 90.0, 180.0, 270.0, 359.75]
        da = _make_da(lons)
        result = utils.unwrap_longitude(da)
        np.testing.assert_array_equal(result["longitude"].values, lons)

    def test_already_monotonic_negative_180_to_180(self):
        lons = [-180.0, -90.0, 0.0, 90.0, 180.0]
        da = _make_da(lons)
        result = utils.unwrap_longitude(da)
        np.testing.assert_array_equal(result["longitude"].values, lons)

    def test_wrap_crossing_0_to_360(self):
        lons = [130.0, 200.0, 359.75, 0.0, 9.75]
        da = _make_da(lons)
        result = utils.unwrap_longitude(da)
        expected = [130.0, 200.0, 359.75, 360.0, 369.75]
        np.testing.assert_array_almost_equal(result["longitude"].values, expected)

    def test_wrap_crossing_negative_180_to_180(self):
        lons = [90.0, 150.0, 179.75, -179.75, -90.0]
        da = _make_da(lons)
        result = utils.unwrap_longitude(da)
        expected = [90.0, 150.0, 179.75, 180.25, 270.0]
        np.testing.assert_array_almost_equal(result["longitude"].values, expected)

    def test_wrap_at_first_step(self):
        lons = [359.75, 0.0, 0.25, 0.5]
        da = _make_da(lons)
        result = utils.unwrap_longitude(da)
        expected = [359.75, 360.0, 360.25, 360.5]
        np.testing.assert_array_almost_equal(result["longitude"].values, expected)

    def test_single_element_unchanged(self):
        lons = [45.0]
        da = _make_da(lons)
        result = utils.unwrap_longitude(da)
        np.testing.assert_array_equal(result["longitude"].values, lons)

    def test_two_element_wrap(self):
        lons = [359.75, 0.0]
        da = _make_da(lons)
        result = utils.unwrap_longitude(da)
        expected = [359.75, 360.0]
        np.testing.assert_array_almost_equal(result["longitude"].values, expected)

    def test_two_element_no_wrap(self):
        lons = [0.0, 90.0]
        da = _make_da(lons)
        result = utils.unwrap_longitude(da)
        np.testing.assert_array_equal(result["longitude"].values, lons)

    def test_data_values_unchanged(self):
        lons = [350.0, 355.0, 0.0, 5.0]
        data = np.array([1.0, 2.0, 3.0, 4.0])
        da = xr.DataArray(data, dims=["longitude"], coords={"longitude": lons})
        result = utils.unwrap_longitude(da)
        np.testing.assert_array_equal(result.values, data)

    def test_dataset_wrap_crossing(self):
        lons = [130.0, 200.0, 359.75, 0.0, 9.75]
        ds = _make_ds(lons)
        result = utils.unwrap_longitude(ds)
        expected = [130.0, 200.0, 359.75, 360.0, 369.75]
        np.testing.assert_array_almost_equal(result["longitude"].values, expected)

    def test_dataset_already_monotonic(self):
        lons = [0.0, 90.0, 180.0, 270.0]
        ds = _make_ds(lons)
        result = utils.unwrap_longitude(ds)
        np.testing.assert_array_equal(result["longitude"].values, lons)


class TestEpochsPerFullPass:
    def test_paper_example(self):
        assert utils.epochs_per_full_pass(35_200, 64, 10) == 55

    def test_steps_exceed_full_pass_returns_one(self):
        assert utils.epochs_per_full_pass(100, 4, 999) == 1

    def test_full_pass_per_epoch_returns_one(self):
        assert utils.epochs_per_full_pass(800, 4, 200) == 1

    def test_exact_division(self):
        assert utils.epochs_per_full_pass(800, 4, 50) == 4

    def test_non_divisible_samples_round_up(self):
        assert utils.epochs_per_full_pass(101, 4, 5) == 6

    def test_no_samples_returns_zero(self):
        assert utils.epochs_per_full_pass(0, 4, 10) == 0

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError):
            utils.epochs_per_full_pass(100, 0, 10)

    def test_invalid_steps_per_epoch_raises(self):
        with pytest.raises(ValueError):
            utils.epochs_per_full_pass(100, 4, 0)

    def test_negative_samples_raises(self):
        with pytest.raises(ValueError):
            utils.epochs_per_full_pass(-1, 4, 10)
