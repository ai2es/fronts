import numpy as np
import pytest
import xarray as xr

from fronts.constants import FRONT_CLASS_MAP
from fronts.data.targets import dilate_fronts, one_hot_encode_to_dataarray, remap_fronts

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
        result = one_hot_encode_to_dataarray(label_da, num_classes=N_CLASSES)
        assert list(result.dims) == ["time", "latitude", "longitude", "class"]

    def test_shape(self, label_da):
        result = one_hot_encode_to_dataarray(label_da, num_classes=N_CLASSES)
        assert result.shape == (N_TIME, N_LAT, N_LON, N_CLASSES)

    def test_dtype(self, label_da):
        result = one_hot_encode_to_dataarray(label_da, num_classes=N_CLASSES)
        assert result.dtype == np.float32

    def test_values_correct(self, label_da):
        result = one_hot_encode_to_dataarray(label_da, num_classes=N_CLASSES).values
        expected = (label_da.values[..., np.newaxis] == np.arange(N_CLASSES)).astype(np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_time_coord_preserved(self, label_da):
        result = one_hot_encode_to_dataarray(label_da, num_classes=N_CLASSES)
        np.testing.assert_array_equal(result.coords["time"].values, label_da.time.values)

    def test_default_num_classes_matches_nine_class_experiment(self, label_da):
        result = one_hot_encode_to_dataarray(label_da)
        assert result.shape == (N_TIME, N_LAT, N_LON, 9)

    def test_one_hot_property(self, label_da):
        result = one_hot_encode_to_dataarray(label_da, num_classes=N_CLASSES)
        row_sums = result.values.sum(axis=-1)
        np.testing.assert_array_almost_equal(row_sums, np.ones((N_TIME, N_LAT, N_LON)))


class TestRemapFronts:
    @pytest.mark.parametrize(
        ("raw_code", "expected_class"),
        [
            (1, 1),  # CF
            (5, 1),  # CF-F (forming)
            (9, 1),  # CF-D (dissipating)
            (2, 2),  # WF
            (6, 2),  # WF-F
            (10, 2),  # WF-D
            (3, 3),  # SF
            (7, 3),  # SF-F
            (11, 3),  # SF-D
            (4, 4),  # OF
            (8, 4),  # OF-F
            (12, 4),  # OF-D
            (16, 5),  # dryline
            (14, 6),  # trough
            (15, 7),  # tropical trough
            (13, 8),  # instability axis
            (0, 0),  # no front
            (17, 0),  # unrecognized code
        ],
    )
    def test_code_mapped_to_expected_class(self, raw_code, expected_class):
        da = xr.DataArray(np.full((1, 2, 2), raw_code, dtype=np.int32), dims=["time", "latitude", "longitude"])
        result = remap_fronts(da).values
        assert (result == expected_class).all()

    def test_forming_and_dissipating_collapse_to_same_class_as_parent(self):
        for parent_code, cls in ((1, 1), (2, 2), (3, 3), (4, 4)):
            forming_code = parent_code + 4
            dissipating_code = parent_code + 8
            da = xr.DataArray(
                np.array([[[parent_code, forming_code, dissipating_code]]], dtype=np.int32),
                dims=["time", "latitude", "longitude"],
            )
            result = remap_fronts(da).values
            assert (result == cls).all()

    def test_dtype_is_int32(self):
        da = xr.DataArray(np.zeros((1, 2, 2), dtype=np.int32), dims=["time", "latitude", "longitude"])
        assert remap_fronts(da).dtype == np.int32

    def test_all_output_classes_zero_through_eight_reachable(self):
        assert set(FRONT_CLASS_MAP.values()) == set(range(1, 9))


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


def _two_adjacent_fronts_da() -> xr.DataArray:
    data = np.zeros((1, 8, 8, N_CLASSES), dtype=np.float32)
    data[0, :, 2, 1] = 1.0
    data[0, :, 4, 3] = 1.0
    data[0, :, :, 0] = 1.0 - data[0, :, :, 1:].any(axis=-1)
    return _make_one_hot_da(data)


class TestDilateFrontsOverlap:
    def test_dilated_targets_stay_one_hot(self):
        result = dilate_fronts(_two_adjacent_fronts_da(), dilation=1).values
        class_sums = result.sum(axis=-1)
        np.testing.assert_array_equal(class_sums, np.ones_like(class_sums))

    def test_original_labels_not_overwritten(self):
        data = np.zeros((1, 8, 8, N_CLASSES), dtype=np.float32)
        data[0, :, 2, 1] = 1.0
        data[0, :, 3, 3] = 1.0
        data[0, :, :, 0] = 1.0 - data[0, :, :, 1:].any(axis=-1)
        result = dilate_fronts(_make_one_hot_da(data), dilation=1).values
        np.testing.assert_array_equal(result[0, :, 3, 3], np.ones(8))
        np.testing.assert_array_equal(result[0, :, 3, 1], np.zeros(8))

    def test_equidistant_collision_resolves_to_lower_class(self):
        result = dilate_fronts(_two_adjacent_fronts_da(), dilation=1).values
        np.testing.assert_array_equal(result[0, :, 3, 1], np.ones(8))
        np.testing.assert_array_equal(result[0, :, 3, 3], np.zeros(8))
