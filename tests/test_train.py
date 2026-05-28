import numpy as np
import pytest
import xarray as xr

from fronts.data.targets import FRONT_CLASS_MAP, filter_timesteps

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
