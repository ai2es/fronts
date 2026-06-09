import numpy as np
import pytest
import xarray as xr

from fronts.data.targets import dilate_fronts, one_hot_encode_to_dataarray, seasonal_test_split

N_TIME = 5
N_LAT = 80
N_LON = 80
N_CLASSES = 6

_SMALL_LAT = 4
_SMALL_LON = 4


@pytest.fixture
def label_da() -> xr.DataArray:
    rng = np.random.default_rng(42)
    data = rng.integers(0, N_CLASSES, size=(N_TIME, N_LAT, N_LON)).astype(np.int32)
    return xr.DataArray(
        data,
        dims=["time", "latitude", "longitude"],
        coords={"time": np.arange(N_TIME)},
    )


class TestOneHotEncodeToDataarray:
    def test_dims(self, label_da):
        result = one_hot_encode_to_dataarray(label_da)
        assert list(result.dims) == ["time", "latitude", "longitude", "class"]

    def test_shape(self, label_da):
        result = one_hot_encode_to_dataarray(label_da)
        assert result.shape == (N_TIME, N_LAT, N_LON, N_CLASSES)

    def test_dtype(self, label_da):
        result = one_hot_encode_to_dataarray(label_da)
        assert result.dtype == np.float32

    def test_values_correct(self, label_da):
        result = one_hot_encode_to_dataarray(label_da).values
        expected = (label_da.values[..., np.newaxis] == np.arange(N_CLASSES)).astype(np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_time_coord_preserved(self, label_da):
        result = one_hot_encode_to_dataarray(label_da)
        np.testing.assert_array_equal(result.coords["time"].values, label_da.time.values)

    def test_one_hot_property(self, label_da):
        result = one_hot_encode_to_dataarray(label_da)
        row_sums = result.values.sum(axis=-1)
        np.testing.assert_array_almost_equal(row_sums, np.ones((N_TIME, N_LAT, N_LON)))


def _make_one_hot_da(data: np.ndarray) -> xr.DataArray:
    n_time, _n_lat, _n_lon, _n_classes = data.shape
    return xr.DataArray(
        data.astype(np.float32),
        dims=["time", "latitude", "longitude", "class"],
        coords={"time": np.arange(n_time)},
    )


def _single_front_pixel_da(cls: int = 1, n_classes: int = N_CLASSES) -> xr.DataArray:
    data = np.zeros((1, _SMALL_LAT, _SMALL_LON, n_classes), dtype=np.float32)
    center_lat = _SMALL_LAT // 2
    center_lon = _SMALL_LON // 2
    data[0, center_lat, center_lon, cls] = 1.0
    data[0, :, :, 0] = 1.0 - data[0, :, :, 1:].any(axis=-1)
    return _make_one_hot_da(data)


class TestDilateFronts:
    def test_no_dilation_returns_unchanged(self):
        da = _single_front_pixel_da()
        result = dilate_fronts(da, dilation=0)
        assert result is da

    def test_shape_preserved(self):
        da = _single_front_pixel_da()
        result = dilate_fronts(da, dilation=1)
        assert result.shape == da.shape

    def test_background_is_complement(self):
        da = _single_front_pixel_da()
        result = dilate_fronts(da, dilation=1).values
        any_front = result[..., 1:].any(axis=-1)
        np.testing.assert_array_equal(result[..., 0], (~any_front).astype(np.float32))

    def test_front_pixel_expands(self):
        da = _single_front_pixel_da(cls=1)
        result = dilate_fronts(da, dilation=1).values
        center_lat = _SMALL_LAT // 2
        center_lon = _SMALL_LON // 2
        front_class = result[0, :, :, 1]
        neighbor_count = (
            front_class[center_lat - 1, center_lon]
            + front_class[center_lat + 1, center_lon]
            + front_class[center_lat, center_lon - 1]
            + front_class[center_lat, center_lon + 1]
        )
        assert neighbor_count >= 4


def _make_times(start: str = "2018-01-01", periods: int = 400, freq_hours: int = 24) -> np.ndarray:
    """Return a datetime64 array at regular hourly intervals spanning multiple seasons."""
    base = np.datetime64(start, "h")
    return base + np.arange(periods) * np.timedelta64(freq_hours, "h")


class TestSeasonalTestSplit:
    @pytest.fixture
    def times(self) -> np.ndarray:
        return _make_times(periods=400)

    def test_fraction_per_season_approximate(self, times):
        rng = np.random.default_rng(0)
        mask = seasonal_test_split(times, 0.1, rng)
        months = times.astype("datetime64[M]").astype(int) % 12 + 1
        season_map = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0], dtype=np.int32)
        seasons = season_map[months]
        for s in range(4):
            season_idx = np.where(seasons == s)[0]
            if len(season_idx) == 0:
                continue
            expected = round(len(season_idx) * 0.1)
            actual = int(mask[season_idx].sum())
            assert abs(actual - expected) <= 1

    def test_no_overlap(self, times):
        rng = np.random.default_rng(0)
        mask = seasonal_test_split(times, 0.1, rng)
        assert not (mask & ~mask).any()
        assert (mask | ~mask).all()

    def test_all_seasons_represented(self, times):
        rng = np.random.default_rng(0)
        mask = seasonal_test_split(times, 0.1, rng)
        months = times[mask].astype("datetime64[M]").astype(int) % 12 + 1
        season_map = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0], dtype=np.int32)
        seasons_present = set(season_map[months].tolist())
        assert seasons_present == {0, 1, 2, 3}

    def test_deterministic_with_same_seed(self, times):
        mask_a = seasonal_test_split(times, 0.1, np.random.default_rng(7))
        mask_b = seasonal_test_split(times, 0.1, np.random.default_rng(7))
        np.testing.assert_array_equal(mask_a, mask_b)

    def test_different_seeds_differ(self, times):
        mask_a = seasonal_test_split(times, 0.1, np.random.default_rng(1))
        mask_b = seasonal_test_split(times, 0.1, np.random.default_rng(2))
        assert not np.array_equal(mask_a, mask_b)

    def test_empty_season_handled(self):
        times = _make_times(start="2020-06-01", periods=92)
        rng = np.random.default_rng(0)
        mask = seasonal_test_split(times, 0.1, rng)
        assert mask.shape == (92,)
        assert mask.any()
