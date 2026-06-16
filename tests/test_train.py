import itertools

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fronts.data.inputs import LazyTimeSource, native_positions
from fronts.data.targets import FRONT_CLASS_MAP, filter_timesteps
from fronts.utils import apply_time_resolution

try:
    from fronts.train import make_batch_dataset

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

_ALL_CODES = list(FRONT_CLASS_MAP.keys())  # [1, 2, 3, 4, 15]


def _make_fronts(time_codes: list[list[int]], lat: int = 4, lon: int = 8) -> xr.DataArray:
    """Build a (time, lat, lon) fronts DataArray where each timestep gets exactly the front codes listed.

    Codes are placed at the first pixels of the first row.
    """
    n_time = len(time_codes)
    data = np.zeros((n_time, lat, lon), dtype=np.int32)
    for t, codes in enumerate(time_codes):
        for i, code in enumerate(codes):
            data[t, 0, i] = code
    return xr.DataArray(data, dims=["time", "latitude", "longitude"])


N_TIME = 5
N_LAT = 32
N_LON = 64
N_CHANNELS = 30
N_CLASSES = 6


class TestFilterTimesteps:
    def test_all_types_present_always_kept(self):
        da = _make_fronts([_ALL_CODES, _ALL_CODES])
        rng = np.random.default_rng(0)
        mask = filter_timesteps(da, rng)
        assert mask.all()

    def test_incomplete_timestep_dropped_by_rng(self):
        # One code missing — outcome is purely the RNG 50% draw.
        # Seed 0: first draw ~0.64 (>= 0.5), so dropped.
        da = _make_fronts([_ALL_CODES[:-1]])
        rng = np.random.default_rng(0)
        mask = filter_timesteps(da, rng)
        assert not mask[0]

    def test_incomplete_timestep_kept_by_rng(self):
        # Seed 2: first draw ~0.26 (< 0.5), so kept.
        da = _make_fronts([_ALL_CODES[:-1]])
        rng = np.random.default_rng(2)
        mask = filter_timesteps(da, rng)
        assert mask[0]

    def test_background_only_uses_rng(self):
        # Pure background (0) has no front types; result is a 50% draw.
        da = _make_fronts([[0]])
        kept = sum(filter_timesteps(da, np.random.default_rng(s))[0] for s in range(200))
        assert 70 < kept < 130  # expect ~100 with reasonable variance

    def test_mixed_timesteps(self):
        # First timestep complete (always kept), second incomplete (RNG-dependent).
        da = _make_fronts([_ALL_CODES, _ALL_CODES[:2]])
        rng = np.random.default_rng(0)
        mask = filter_timesteps(da, rng)
        assert mask[0]  # guaranteed

    def test_return_shape(self):
        da = _make_fronts([_ALL_CODES] * 7)
        mask = filter_timesteps(da, np.random.default_rng(0))
        assert mask.shape == (7,)
        assert mask.dtype == bool


class TestApplyTimeResolution:
    def _make_times(self, freq: str, periods: int, start: str = "2020-01-01") -> np.ndarray:
        return pd.date_range(start, periods=periods, freq=freq).values

    def test_6h_from_3h_keeps_half(self):
        times = self._make_times("3h", 8)  # 00, 03, 06, 09, 12, 15, 18, 21
        result = apply_time_resolution(times, "6h")
        assert len(result) == 4  # 00, 06, 12, 18

    def test_6h_from_3h_correct_hours(self):
        times = self._make_times("3h", 8)
        result = apply_time_resolution(times, "6h")
        hours = pd.DatetimeIndex(result).hour.tolist()
        assert hours == [0, 6, 12, 18]

    def test_already_aligned_unchanged(self):
        times = self._make_times("6h", 4)
        result = apply_time_resolution(times, "6h")
        np.testing.assert_array_equal(result, times)

    def test_12h_from_3h_correct_hours(self):
        times = self._make_times("3h", 8)
        result = apply_time_resolution(times, "12h")
        hours = pd.DatetimeIndex(result).hour.tolist()
        assert hours == [0, 12]

    def test_empty_input(self):
        times = np.array([], dtype="datetime64[ns]")
        result = apply_time_resolution(times, "6h")
        assert len(result) == 0

    def test_no_aligned_timestamps(self):
        times = pd.date_range("2020-01-01 01:00", periods=4, freq="3h").values  # 01, 04, 07, 10
        result = apply_time_resolution(times, "6h")
        assert len(result) == 0

    def test_multi_day_span(self):
        times = self._make_times("3h", 16)  # 2 days of 3h data
        result = apply_time_resolution(times, "6h")
        assert len(result) == 8
        assert all(h in (0, 6, 12, 18) for h in pd.DatetimeIndex(result).hour)


class TestNativePositions:
    def test_identity(self):
        times = np.arange("2020-01-01", "2020-01-06", dtype="datetime64[D]")
        positions = native_positions(times, times)
        np.testing.assert_array_equal(positions, np.arange(5))

    def test_subset_and_order_preserved(self):
        times = np.arange("2020-01-01", "2020-01-06", dtype="datetime64[D]")
        wanted = times[[3, 1]]
        positions = native_positions(times, wanted)
        np.testing.assert_array_equal(positions, [3, 1])

    def test_duplicates_keep_first(self):
        times = np.array(["2020-01-01", "2020-01-01", "2020-01-02"], dtype="datetime64[D]")
        positions = native_positions(times, np.array(["2020-01-02", "2020-01-01"], dtype="datetime64[D]"))
        np.testing.assert_array_equal(positions, [2, 0])

    def test_missing_time_raises(self):
        times = np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]")
        with pytest.raises(ValueError, match="absent"):
            native_positions(times, np.array(["2020-06-01"], dtype="datetime64[D]"))


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestMakeBatchDatasetLazyTimeSource:
    def test_positions_select_and_order_samples(self, era5_da, front_da):
        # positions remap logical sample 0->native 4, 1->native 2; the loader must
        # yield exactly those native timesteps in that order.
        positions = np.array([4, 2])
        ds, _ = make_batch_dataset(
            LazyTimeSource(era5_da, positions),
            LazyTimeSource(front_da, positions),
            1,
            batch_size=1,
        )
        loaded = [x.numpy()[0] for x, _ in itertools.islice(ds, 2)]
        np.testing.assert_allclose(loaded[0], era5_da.isel(time=4).values)
        np.testing.assert_allclose(loaded[1], era5_da.isel(time=2).values)

    def test_subblock_gather_preserves_order_and_values(self, era5_da, front_da):
        # load_subblock smaller than the chunk forces multi-block gathering;
        # the concatenated result must match a single contiguous selection.
        positions = np.array([4, 0, 3, 1, 2])
        ds, _ = make_batch_dataset(
            LazyTimeSource(era5_da, positions),
            LazyTimeSource(front_da, positions),
            1,
            batch_size=1,
            load_subblock=2,
        )
        loaded = [x.numpy()[0] for x, _ in itertools.islice(ds, len(positions))]
        for i, native in enumerate(positions):
            np.testing.assert_allclose(loaded[i], era5_da.isel(time=native).values)

    def test_loader_failure_propagates_instead_of_hanging(self, era5_da, front_da):
        # A position past the array's time length makes the background loader's
        # isel(...).compute() raise. The consumer must surface that exception
        # rather than block forever on an empty prefetch queue.
        bad_positions = np.array([era5_da.sizes["time"] + 100])
        ds, _ = make_batch_dataset(
            LazyTimeSource(era5_da, bad_positions),
            LazyTimeSource(front_da, np.array([0])),
            1,
            batch_size=1,
        )
        with pytest.raises(Exception):
            next(iter(ds))

    def test_multi_source_concat_on_channel(self, era5_da, front_da):
        positions = np.arange(era5_da.sizes["time"])
        ds, _ = make_batch_dataset(
            [LazyTimeSource(era5_da, positions), LazyTimeSource(era5_da, positions)],
            LazyTimeSource(front_da, positions),
            1,
            batch_size=1,
        )
        x_batch, _ = next(iter(ds))
        assert x_batch.shape == (1, N_LAT, N_LON, 2 * N_CHANNELS)


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestMakeBatchDataset:
    def test_input_batch_shape(self, era5_da, front_da):
        batch_size = 2
        ds, _ = make_batch_dataset(era5_da, front_da, 1, batch_size)
        x_batch, _ = next(iter(ds))
        assert x_batch.shape == (batch_size, N_LAT, N_LON, N_CHANNELS)

    def test_target_batch_shape(self, era5_da, front_da):
        batch_size = 2
        ds, _ = make_batch_dataset(era5_da, front_da, 1, batch_size)
        _, y_batch = next(iter(ds))
        assert y_batch[0].shape == (batch_size, N_LAT, N_LON, N_CLASSES)

    def test_n_supervision_outputs(self, era5_da, front_da):
        for n_out in [1, 3, 5]:
            ds, _ = make_batch_dataset(era5_da, front_da, n_out, batch_size=2)
            _, y_batch = next(iter(ds))
            assert len(y_batch) == n_out

    def test_covers_all_timesteps(self, era5_da, front_da):
        batch_size = 2
        ds, _ = make_batch_dataset(era5_da, front_da, 1, batch_size)
        total_samples = sum(x.shape[0] for x, _ in ds)
        assert total_samples == N_TIME

    def test_dtypes_are_float32(self, era5_da, front_da):
        import tensorflow as tf

        ds, _ = make_batch_dataset(era5_da, front_da, 1, batch_size=2)
        x_batch, y_batch = next(iter(ds))
        assert x_batch.dtype == tf.float32
        assert y_batch[0].dtype == tf.float32
