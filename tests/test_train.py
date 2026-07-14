import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fronts.data.targets import FRONT_CLASS_MAP, filter_timesteps
from fronts.utils import IcechunkStorageConfig, apply_time_resolution

try:
    import tensorflow as tf

    from fronts.data.datasets import DatasetConfig, FrontsPyDataset
    from fronts.data.generate import write_or_append_icechunk_store
    from fronts.data.inputs import inputs_ds_to_dataarray
    from fronts.train import _build_monitor_callback, load_data_into_dataloader

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


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestBuildMonitorCallback:
    def test_both_decay_params_set_returns_reduce_lr_on_plateau(self):
        callback = _build_monitor_callback(
            monitor="val_loss", patience=5, learning_rate_decay_factor=0.2, learning_rate_minimum=1e-6
        )
        assert isinstance(callback, tf.keras.callbacks.ReduceLROnPlateau)
        assert callback.monitor == "val_loss"
        assert callback.factor == 0.2
        assert callback.patience == 5
        assert callback.min_lr == 1e-6

    def test_only_decay_factor_set_returns_early_stopping(self):
        callback = _build_monitor_callback(
            monitor="val_loss", patience=5, learning_rate_decay_factor=0.2, learning_rate_minimum=None
        )
        assert isinstance(callback, tf.keras.callbacks.EarlyStopping)

    def test_only_decay_minimum_set_returns_early_stopping(self):
        callback = _build_monitor_callback(
            monitor="val_loss", patience=5, learning_rate_decay_factor=None, learning_rate_minimum=1e-6
        )
        assert isinstance(callback, tf.keras.callbacks.EarlyStopping)

    def test_neither_set_returns_early_stopping(self):
        callback = _build_monitor_callback(
            monitor="val_loss", patience=5, learning_rate_decay_factor=None, learning_rate_minimum=None
        )
        assert isinstance(callback, tf.keras.callbacks.EarlyStopping)
        assert callback.monitor == "val_loss"
        assert callback.patience == 5
        assert callback.restore_best_weights is True


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestFrontsPyDatasetGather:
    def test_select_and_order_samples(self, era5_ds, front_da, data_config):
        # A non-contiguous time selection must yield exactly those timesteps in order.
        sub_era5 = era5_ds.isel(time=[4, 2])
        sub_front = front_da.isel(time=[4, 2])
        ds = FrontsPyDataset(sub_era5, sub_front, data_config, batch_size=1)
        x0, _ = ds[0]
        x1, _ = ds[1]
        expected = inputs_ds_to_dataarray(era5_ds, data_config.variables).values
        np.testing.assert_allclose(x0[0], expected[4])
        np.testing.assert_allclose(x1[0], expected[2])

    def test_gather_preserves_order_and_values(self, era5_ds, front_da, data_config):
        order = [4, 0, 3, 1, 2]
        sub_era5 = era5_ds.isel(time=order)
        sub_front = front_da.isel(time=order)
        ds = FrontsPyDataset(sub_era5, sub_front, data_config, batch_size=1)
        expected = inputs_ds_to_dataarray(era5_ds, data_config.variables).values
        for i, native in enumerate(order):
            x, _ = ds[i]
            np.testing.assert_allclose(x[0], expected[native])

    def test_input_target_length_mismatch_raises(self, era5_ds, front_da, data_config):
        with pytest.raises(ValueError, match="differ"):
            FrontsPyDataset(
                era5_ds.isel(time=[0, 1]),
                front_da.isel(time=[0]),
                data_config,
                batch_size=1,
            )


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestFrontsPyDataset:
    def _make_ds(self, era5_ds, front_da, data_config, batch_size=2, **kwargs):
        return FrontsPyDataset(era5_ds, front_da, data_config, batch_size=batch_size, **kwargs)

    def test_input_batch_shape(self, era5_ds, front_da, data_config):
        batch_size = 2
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=batch_size)
        x_batch, _ = ds[0]
        assert x_batch.shape == (batch_size, N_LAT, N_LON, len(data_config.variables))

    def test_target_batch_shape(self, era5_ds, front_da, data_config):
        batch_size = 2
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=batch_size)
        _, y_batch = ds[0]
        assert y_batch.shape == (batch_size, N_LAT, N_LON, N_CLASSES)

    def test_covers_all_timesteps(self, era5_ds, front_da, data_config):
        batch_size = 2
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=batch_size)
        total_samples = sum(ds[i][0].shape[0] for i in range(len(ds)))
        assert total_samples == N_TIME

    def test_dtypes_are_float32(self, era5_ds, front_da, data_config):
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=2)
        x_batch, y_batch = ds[0]
        assert x_batch.dtype == np.float32
        assert y_batch.dtype == np.float32

    def test_shuffle_reshuffles_on_epoch_end(self, era5_ds, front_da, data_config):
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=1, shuffle=True, seed=0)
        order_before = ds._order.copy()
        ds.on_epoch_end()
        assert not np.array_equal(order_before, ds._order)

    def test_no_shuffle_preserves_order(self, era5_ds, front_da, data_config):
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=1, shuffle=False)
        np.testing.assert_array_equal(ds._order, np.arange(N_TIME))
        ds.on_epoch_end()
        np.testing.assert_array_equal(ds._order, np.arange(N_TIME))


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestLoadDataIntoDataloaderLongitude:
    """A wrap-crossing bounding box (lon_max > 360, e.g. configs/generate_icechunk.yaml's.

    130-369.75) leaves the longitude coordinate non-monotonic on disk, e.g.
    [330, 350, 0, 20]. xarray's pcolormesh (used by TestVisualizationCallback's
    truth-overlay plot) raises ValueError on such a coordinate, so
    load_data_into_dataloader must return data with longitude unwrapped.
    """

    _TIMES = pd.date_range("2020-01-01", periods=4, freq="6h")
    _LAT = np.array([10.0, 20.0, 30.0, 40.0])
    _LON_WRAP = np.array([330.0, 350.0, 0.0, 20.0])  # physical domain 330 -> 380

    def _write_store(self, tmp_path, name: str, var_name: str) -> IcechunkStorageConfig:
        storage_config = IcechunkStorageConfig(store_path=str(tmp_path / name), branch_name="main")
        ds = xr.Dataset(
            {
                var_name: xr.DataArray(
                    np.zeros((len(self._TIMES), len(self._LAT), len(self._LON_WRAP)), dtype=np.float32),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": self._TIMES, "latitude": self._LAT, "longitude": self._LON_WRAP},
                )
            }
        )
        write_or_append_icechunk_store(storage_config, ds)
        return storage_config

    def test_longitude_is_monotonic_after_loading(self, tmp_path):
        data_config = DatasetConfig(
            inputs_icechunk_config=self._write_store(tmp_path, "inputs", "temperature"),
            targets_icechunk_config=self._write_store(tmp_path, "targets", "identifier"),
            variables=["temperature"],
            test_years=[2020],
            val_years=[],
        )
        test_dataset = load_data_into_dataloader(data_config, split="test", seed=0)
        lons = test_dataset.input_ds["longitude"].values
        assert np.all(np.diff(lons) >= 0), f"longitude not monotonic: {lons}"
