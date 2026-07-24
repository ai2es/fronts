import math

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
    from fronts.model import ModelConfig, UNet3Plus
    from fronts.train import (
        _build_loss,
        _build_monitor_callbacks,
        _freeze_layers,
        _load_pretrained_weights,
        load_data_into_dataloader,
    )

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
class TestBuildMonitorCallbacks:
    def test_both_decay_params_set_returns_reduce_lr_on_plateau(self):
        callbacks = _build_monitor_callbacks(
            monitor="val_loss", patience=5, learning_rate_decay_factor=0.2, learning_rate_minimum=1e-6
        )
        assert len(callbacks) == 1
        callback = callbacks[0]
        assert isinstance(callback, tf.keras.callbacks.ReduceLROnPlateau)
        assert callback.monitor == "val_loss"
        assert callback.factor == 0.2
        assert callback.patience == 5
        assert callback.min_lr == 1e-6

    def test_only_decay_factor_set_returns_early_stopping(self):
        callbacks = _build_monitor_callbacks(
            monitor="val_loss", patience=5, learning_rate_decay_factor=0.2, learning_rate_minimum=None
        )
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], tf.keras.callbacks.EarlyStopping)

    def test_only_decay_minimum_set_returns_early_stopping(self):
        callbacks = _build_monitor_callbacks(
            monitor="val_loss", patience=5, learning_rate_decay_factor=None, learning_rate_minimum=1e-6
        )
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], tf.keras.callbacks.EarlyStopping)

    def test_neither_set_returns_early_stopping(self):
        callbacks = _build_monitor_callbacks(
            monitor="val_loss", patience=5, learning_rate_decay_factor=None, learning_rate_minimum=None
        )
        assert len(callbacks) == 1
        callback = callbacks[0]
        assert isinstance(callback, tf.keras.callbacks.EarlyStopping)
        assert callback.monitor == "val_loss"
        assert callback.patience == 5
        assert callback.restore_best_weights is True

    def test_min_delta_defaults_to_zero_not_keras_absolute_1e4(self):
        """min_delta must default to 0, not Keras's absolute 1e-4.

        At a loss magnitude of ~1e-3, an absolute min_delta of 1e-4 reads every epoch as
        a plateau and decays the LR to its floor within a dozen epochs.
        """
        callbacks = _build_monitor_callbacks(
            monitor="val_loss", patience=3, learning_rate_decay_factor=0.2, learning_rate_minimum=1e-6
        )
        assert callbacks[0].min_delta == 0.0

    def test_min_delta_passed_through_to_both_callback_types(self):
        reduce_lr = _build_monitor_callbacks(
            monitor="val_loss",
            patience=3,
            learning_rate_decay_factor=0.2,
            learning_rate_minimum=1e-6,
            min_delta=1e-5,
        )[0]
        assert reduce_lr.min_delta == 1e-5
        early_stop = _build_monitor_callbacks(
            monitor="val_loss",
            patience=3,
            learning_rate_decay_factor=None,
            learning_rate_minimum=None,
            min_delta=1e-5,
        )[0]
        assert early_stop.min_delta == 1e-5

    def test_lr_decay_with_early_stopping_returns_both(self):
        """LR-decay mode alone has no stop condition; early_stopping_patience adds one."""
        callbacks = _build_monitor_callbacks(
            monitor="val_loss",
            patience=3,
            learning_rate_decay_factor=0.2,
            learning_rate_minimum=1e-6,
            min_delta=1e-5,
            early_stopping_patience=12,
        )
        assert len(callbacks) == 2
        reduce_lr, early_stop = callbacks
        assert isinstance(reduce_lr, tf.keras.callbacks.ReduceLROnPlateau)
        assert reduce_lr.patience == 3
        assert isinstance(early_stop, tf.keras.callbacks.EarlyStopping)
        assert early_stop.patience == 12
        assert early_stop.restore_best_weights is True
        assert early_stop.min_delta == 1e-5

    def test_early_stopping_patience_ignored_without_lr_decay(self):
        callbacks = _build_monitor_callbacks(
            monitor="val_loss",
            patience=5,
            learning_rate_decay_factor=None,
            learning_rate_minimum=None,
            early_stopping_patience=12,
        )
        assert len(callbacks) == 1
        assert callbacks[0].patience == 5


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

    def test_drop_remainder_drops_undersized_final_batch(self, era5_ds, front_da, data_config):
        """N_TIME=5 with batch_size=2 has a 1-sample remainder batch that must be dropped.

        A trailing batch smaller than batch_size splits unevenly across replicas under
        MirroredStrategy, which triggers CUDNN_STATUS_BAD_PARAM in Conv3DBackpropFilterV2
        (https://github.com/tensorflow/tensorflow/issues/60935).
        """
        batch_size = 2
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=batch_size, drop_remainder=True)
        assert len(ds) == N_TIME // batch_size
        for i in range(len(ds)):
            x_batch, y_batch = ds[i]
            assert x_batch.shape[0] == batch_size
            assert y_batch.shape[0] == batch_size

    def test_drop_remainder_false_keeps_undersized_final_batch(self, era5_ds, front_da, data_config):
        batch_size = 2
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=batch_size, drop_remainder=False)
        assert len(ds) == math.ceil(N_TIME / batch_size)
        total_samples = sum(ds[i][0].shape[0] for i in range(len(ds)))
        assert total_samples == N_TIME


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


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestLoadDataIntoDataloaderCoordinates:
    """data_config.coordinates must actually restrict the loaded domain.

    Regression test for a branch-divergence bug where load_data_into_dataloader silently
    ignored data_config.coordinates and always loaded the full domain regardless of the
    bounding box set in the config.
    """

    _TIMES = pd.date_range("2020-01-01", periods=4, freq="6h")
    _LAT = np.array([10.0, 20.0, 30.0, 40.0])
    _LON = np.array([100.0, 110.0, 120.0, 130.0])

    def _write_store(self, tmp_path, name: str, var_name: str) -> IcechunkStorageConfig:
        storage_config = IcechunkStorageConfig(store_path=str(tmp_path / name), branch_name="main")
        ds = xr.Dataset(
            {
                var_name: xr.DataArray(
                    np.zeros((len(self._TIMES), len(self._LAT), len(self._LON)), dtype=np.float32),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": self._TIMES, "latitude": self._LAT, "longitude": self._LON},
                )
            }
        )
        write_or_append_icechunk_store(storage_config, ds)
        return storage_config

    def test_coordinates_restrict_loaded_domain(self, tmp_path):
        from fronts.utils import BoundingBox

        data_config = DatasetConfig(
            inputs_icechunk_config=self._write_store(tmp_path, "inputs", "temperature"),
            targets_icechunk_config=self._write_store(tmp_path, "targets", "identifier"),
            variables=["temperature"],
            test_years=[2020],
            val_years=[],
            coordinates=BoundingBox(lat_min=15.0, lat_max=25.0, lon_min=105.0, lon_max=115.0),
        )
        test_dataset = load_data_into_dataloader(data_config, split="test", seed=0)
        lats = test_dataset.input_ds["latitude"].values
        lons = test_dataset.input_ds["longitude"].values
        assert lats.min() >= 15.0 and lats.max() <= 25.0, f"latitude not restricted: {lats}"
        assert lons.min() >= 105.0 and lons.max() <= 115.0, f"longitude not restricted: {lons}"


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestBuildLoss:
    _LATITUDES = np.linspace(25.0, 56.75, 8)

    def test_fss_returns_callable(self):
        loss_fn = _build_loss(
            loss_name="fractions_skill_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
        )
        assert callable(loss_fn)

    def test_neighborhood_brier_score_returns_callable(self):
        loss_fn = _build_loss(
            loss_name="neighborhood_brier_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
        )
        assert callable(loss_fn)

    def test_unrecognized_loss_name_raises(self):
        with pytest.raises(ValueError, match="Unrecognized loss_name"):
            _build_loss(
                loss_name="bogus",  # type: ignore[arg-type]
                loss_class_weights=None,
                latitudes=self._LATITUDES,
                fss_mask_size=(3, 3),
                nbs_tolerance_km=25.0,
                nbs_periodic_lon=False,
                nbs_lat_dependent_pool=False,
            )

    def test_fss_and_nbs_produce_different_losses_on_same_inputs(self):
        rng = np.random.default_rng(0)
        n_classes = 3
        y_true = tf.one_hot(rng.integers(0, n_classes, size=(2, 8, 8)), n_classes)
        y_pred = tf.nn.softmax(rng.standard_normal((2, 8, 8, n_classes)).astype(np.float32), axis=-1)

        fss_loss = _build_loss(
            loss_name="fractions_skill_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
        )
        nbs_loss = _build_loss(
            loss_name="neighborhood_brier_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
        )
        fss_value = float(tf.reduce_mean(fss_loss(y_true, y_pred)))
        nbs_value = float(tf.reduce_mean(nbs_loss(y_true, y_pred)))
        assert np.isfinite(fss_value)
        assert np.isfinite(nbs_value)


class TestTrainConfigLossClassWeights:
    @pytest.fixture
    def train_config_cls(self):
        return pytest.importorskip("fronts.train").TrainConfig

    def test_null_yaml_value_parses_to_none(self, train_config_cls):
        from fronts import utils

        yaml_data = {"train_config": {"loss_class_weights": None, "epochs": 1}}
        cfg = utils.parse_config_section(yaml_data, train_config_cls, "train_config")
        assert cfg.loss_class_weights is None

    def test_explicit_weights_parse_to_list(self, train_config_cls):
        from fronts import utils

        weights = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        yaml_data = {"train_config": {"loss_class_weights": weights, "epochs": 1}}
        cfg = utils.parse_config_section(yaml_data, train_config_cls, "train_config")
        assert cfg.loss_class_weights == weights

    def test_schooner_configs_parse(self, train_config_cls):
        from fronts import utils

        for path in [
            "configs/schooner_train.yaml",
            "configs/schooner_pipeline.yaml",
            "configs/schooner_train_3d.yaml",
        ]:
            yaml_data = utils.load_yaml(path)
            cfg = utils.parse_config_section(yaml_data, train_config_cls, "train_config")
            assert cfg.loss_class_weights is None
            assert cfg.loss_name == "neighborhood_brier_score"
            assert cfg.nbs_tolerance_km == 25.0

    def test_3d_config_parses(self):
        from fronts import utils
        from fronts.callbacks import CallbacksConfig
        from fronts.data.datasets import DatasetConfig
        from fronts.model import ModelConfig
        from fronts.train import TrainConfig

        yaml_data = utils.load_yaml("configs/schooner_train_3d.yaml")
        data_cfg = utils.parse_config_section(yaml_data, DatasetConfig, "data_config")
        model_cfg = utils.parse_config_section(yaml_data, ModelConfig, "model_config")
        train_cfg = utils.parse_config_section(yaml_data, TrainConfig, "train_config")
        callbacks_cfg = utils.parse_config_section(yaml_data, CallbacksConfig, "callbacks_config")

        assert data_cfg.volume_inputs is True
        assert len(data_cfg.variables) == 10
        assert model_cfg.squeeze_axes == 3
        assert list(model_cfg.pool_size) == [2, 2, 1]
        assert list(model_cfg.upsample_size) == [2, 2, 1]
        assert model_cfg.kernel_size == 5
        assert train_cfg.learning_rate == 1e-4
        assert train_cfg.gradient_clip_norm == 1.0
        assert callbacks_cfg.min_delta == 0.0
        assert callbacks_cfg.early_stopping_patience == 12


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestFrontsPyDatasetVolume:
    """volume_inputs=True must yield (batch, lat, lon, level, variable) batches for a 3D model."""

    _N_TIME = 4
    _N_LAT = 6
    _N_LON = 8
    _LEVELS = (1000, 950)

    def _make_volume_inputs(self):
        rng = np.random.default_rng(11)
        times = np.arange(self._N_TIME)
        input_ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    rng.standard_normal((self._N_TIME, len(self._LEVELS), self._N_LAT, self._N_LON)).astype(np.float32),
                    dims=["time", "level", "latitude", "longitude"],
                    coords={"time": times, "level": list(self._LEVELS)},
                ),
                "mean_sea_level_pressure": xr.DataArray(
                    rng.standard_normal((self._N_TIME, self._N_LAT, self._N_LON)).astype(np.float32),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": times},
                ),
            }
        )
        target_da = xr.DataArray(
            rng.integers(0, 2, size=(self._N_TIME, self._N_LAT, self._N_LON)).astype(np.int32),
            dims=["time", "latitude", "longitude"],
            coords={"time": times},
        )
        dummy_store = IcechunkStorageConfig(store_path="unused", branch_name="main")
        config = DatasetConfig(
            inputs_icechunk_config=dummy_store,
            targets_icechunk_config=dummy_store,
            variables=["temperature", "mean_sea_level_pressure"],
            test_years=[],
            val_years=[],
            volume_inputs=True,
        )
        return input_ds, target_da, config

    def test_batch_is_5d(self):
        input_ds, target_da, config = self._make_volume_inputs()
        ds = FrontsPyDataset(input_ds, target_da, config, batch_size=2)
        x_batch, y_batch = ds[0]
        assert x_batch.shape == (2, self._N_LAT, self._N_LON, len(self._LEVELS), 2)
        assert y_batch.shape == (2, self._N_LAT, self._N_LON, N_CLASSES)

    def test_single_level_variable_broadcast_in_batch(self):
        input_ds, target_da, config = self._make_volume_inputs()
        ds = FrontsPyDataset(input_ds, target_da, config, batch_size=2)
        x_batch, _ = ds[0]
        np.testing.assert_array_equal(x_batch[..., 0, 1], x_batch[..., 1, 1])


def _build_small_unet(
    levels: int = 3,
    deep_supervision: bool = False,
    normalization_stat_a: np.ndarray | None = None,
    normalization_stat_b: np.ndarray | None = None,
) -> "tf.keras.Model":
    filter_num = [8, 16, 32, 64][:levels]
    return UNet3Plus(
        input_shape=(None, None, 4),
        num_classes=6,
        pool_size=(2, 2),
        upsample_size=(2, 2),
        levels=levels,
        filter_num=filter_num,
        deep_supervision=deep_supervision,
        output_activation="softmax",
        normalization_method="minmax",
        normalization_stat_a=normalization_stat_a,
        normalization_stat_b=normalization_stat_b,
    ).build()


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestLoadPretrainedWeights:
    def test_encoder_decoder_weights_transferred(self, tmp_path):
        min_val = np.zeros(4, dtype=np.float32)
        max_val = np.ones(4, dtype=np.float32)
        pretrained = _build_small_unet(normalization_stat_a=min_val, normalization_stat_b=max_val)
        checkpoint_path = str(tmp_path / "pretrained.keras")
        pretrained.save(checkpoint_path)

        fresh = _build_small_unet(normalization_stat_a=min_val, normalization_stat_b=max_val)
        pretrained_kernel = pretrained.get_layer("En1_Conv2D_1").get_weights()[0]
        fresh_kernel_before = fresh.get_layer("En1_Conv2D_1").get_weights()[0]
        assert not np.allclose(pretrained_kernel, fresh_kernel_before)

        _load_pretrained_weights(fresh, checkpoint_path, min_val, max_val)

        fresh_kernel_after = fresh.get_layer("En1_Conv2D_1").get_weights()[0]
        np.testing.assert_allclose(fresh_kernel_after, pretrained_kernel)

    def test_normalization_reset_to_new_stats_not_checkpoint_stats(self, tmp_path):
        checkpoint_min = np.array([0.0, -10.0, 100.0, 0.0], dtype=np.float32)
        checkpoint_max = np.array([1.0, 10.0, 200.0, 1.0], dtype=np.float32)
        pretrained = _build_small_unet(normalization_stat_a=checkpoint_min, normalization_stat_b=checkpoint_max)
        checkpoint_path = str(tmp_path / "pretrained.keras")
        pretrained.save(checkpoint_path)

        full_domain_min = np.array([-50.0, -80.0, 0.0, 0.0], dtype=np.float32)
        full_domain_max = np.array([50.0, 80.0, 1000.0, 1.0], dtype=np.float32)
        fresh = _build_small_unet(normalization_stat_a=full_domain_min, normalization_stat_b=full_domain_max)

        _load_pretrained_weights(fresh, checkpoint_path, full_domain_min, full_domain_max)

        norm_layer = fresh.get_layer("input_normalization")
        expected_scale = 1.0 / (full_domain_max - full_domain_min)
        expected_offset = -full_domain_min * expected_scale
        np.testing.assert_allclose(norm_layer.scale, expected_scale, atol=1e-5)
        np.testing.assert_allclose(norm_layer.offset, expected_offset, atol=1e-5)

    def test_mismatched_supervision_head_shape_skipped_not_fatal(self, tmp_path):
        min_val = np.zeros(4, dtype=np.float32)
        max_val = np.ones(4, dtype=np.float32)
        pretrained = _build_small_unet(
            levels=3, deep_supervision=True, normalization_stat_a=min_val, normalization_stat_b=max_val
        )
        checkpoint_path = str(tmp_path / "pretrained.keras")
        pretrained.save(checkpoint_path)

        fresh = _build_small_unet(
            levels=4, deep_supervision=True, normalization_stat_a=min_val, normalization_stat_b=max_val
        )
        pretrained_kernel = pretrained.get_layer("En1_Conv2D_1").get_weights()[0]

        _load_pretrained_weights(fresh, checkpoint_path, min_val, max_val)

        fresh_kernel_after = fresh.get_layer("En1_Conv2D_1").get_weights()[0]
        np.testing.assert_allclose(fresh_kernel_after, pretrained_kernel)


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestFreezeLayers:
    def test_prefix_match_freezes_expected_layers(self):
        unet = _build_small_unet(levels=3, deep_supervision=True)
        _freeze_layers(unet, ["En"])
        for layer in unet.layers:
            if layer.name.startswith("En"):
                assert layer.trainable is False
            elif layer.name.startswith("De") or layer.name.startswith("sup"):
                assert layer.trainable is True

    def test_no_prefixes_frozen_when_prefix_absent(self):
        unet = _build_small_unet(levels=3, deep_supervision=True)
        _freeze_layers(unet, ["NonexistentPrefix"])
        assert all(layer.trainable for layer in unet.layers)


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestModelConfigFreezeValidation:
    def test_freeze_prefixes_without_pretrained_path_raises(self):
        with pytest.raises(ValueError):
            ModelConfig(freeze_layer_prefixes=["En"], pretrained_weights_path=None)
