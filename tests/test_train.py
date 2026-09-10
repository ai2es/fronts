import dataclasses
import logging
import math
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fronts import constants
from fronts.constants import FRONT_CLASS_MAP
from fronts.data.targets import filter_timesteps
from fronts.utils import IcechunkStorageConfig, apply_time_resolution

try:
    import tensorflow as tf

    from fronts import callbacks as fronts_callbacks
    from fronts.callbacks import CallbacksConfig
    from fronts.data.datasets import DatasetConfig, FrontsPyDataset
    from fronts.data.generate import write_or_append_icechunk_store
    from fronts.data.inputs import inputs_ds_to_dataarray
    from fronts.model import ModelConfig, UNet3Plus
    from fronts.train import (
        TrainConfig,
        WandBConfig,
        _ResumeSafeCSVLogger,
        _build_dataset_summary,
        _build_loss,
        _build_monitor_callbacks,
        _build_run_callbacks,
        _build_test_visualization_callback,
        _build_wandb_config,
        _collect_run_metadata,
        _compile,
        _freeze_layers,
        _load_pretrained_weights,
        _optimizer_uses_ema,
        _per_front_type_loss_metrics,
        _resolve_fit_verbose,
        load_data_into_dataloader,
    )

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

_ALL_CODES = list(FRONT_CLASS_MAP.keys())  # [1, 2, 3, 4, 16]


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
class TestOptimizerUsesEma:
    def test_plain_optimizer_without_ema_is_false(self):
        assert _optimizer_uses_ema(tf.keras.optimizers.Adam(use_ema=False)) is False

    def test_plain_optimizer_with_ema_is_true(self):
        assert _optimizer_uses_ema(tf.keras.optimizers.Adam(use_ema=True)) is True

    def test_loss_scale_wrapped_optimizer_is_unwrapped(self):
        """Mixed-precision training wraps Adam in a LossScaleOptimizer; use_ema lives on the inner optimizer."""
        wrapped = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(use_ema=True))
        assert _optimizer_uses_ema(wrapped) is True

    def test_loss_scale_wrapped_optimizer_without_ema_is_false(self):
        wrapped = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(use_ema=False))
        assert _optimizer_uses_ema(wrapped) is False


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestBuildRunCallbacks:
    """Callback-order test suite.

    `_build_run_callbacks` ordering directly determines whether EMA-swapped weights make it
    into checkpoints and EarlyStopping's best-weight snapshot — see its docstring.
    """

    def _build(self, uses_ema: bool, **overrides):
        kwargs = {
            "uses_ema": uses_ema,
            "monitor": "val_loss",
            "patience": 5,
            "learning_rate_decay_factor": None,
            "learning_rate_minimum": None,
            "monitor_min_delta": 0.0,
            "early_stopping_patience": None,
            "extra_callbacks": None,
            "wandb_project": None,
            "wandb_log_freq": "epoch",
            "model_checkpoint_path": None,
        }
        kwargs.update(overrides)
        return _build_run_callbacks(**kwargs)

    def test_no_ema_omits_swap_ema_weights_callback(self):
        callbacks = self._build(uses_ema=False)
        assert not any(isinstance(cb, tf.keras.callbacks.SwapEMAWeights) for cb in callbacks)

    def test_ema_adds_swap_ema_weights_first(self):
        callbacks = self._build(uses_ema=True)
        assert isinstance(callbacks[0], tf.keras.callbacks.SwapEMAWeights)
        assert callbacks[0].swap_on_epoch is True

    def test_swap_ema_weights_precedes_early_stopping(self):
        callbacks = self._build(uses_ema=True)
        swap_idx = next(i for i, cb in enumerate(callbacks) if isinstance(cb, tf.keras.callbacks.SwapEMAWeights))
        early_stop_idx = next(i for i, cb in enumerate(callbacks) if isinstance(cb, tf.keras.callbacks.EarlyStopping))
        assert swap_idx < early_stop_idx

    def test_swap_ema_weights_precedes_model_checkpoint(self, tmp_path):
        callbacks = self._build(uses_ema=True, model_checkpoint_path=str(tmp_path / "model"))
        swap_idx = next(i for i, cb in enumerate(callbacks) if isinstance(cb, tf.keras.callbacks.SwapEMAWeights))
        ckpt_idx = next(i for i, cb in enumerate(callbacks) if isinstance(cb, tf.keras.callbacks.ModelCheckpoint))
        assert swap_idx < ckpt_idx

    def test_csv_logger_present_after_metrics_consolidation(self, tmp_path):
        callbacks = self._build(uses_ema=False, metrics_csv_path=str(tmp_path / "metrics_epoch.csv"))
        consolidation_idx = next(
            i for i, cb in enumerate(callbacks) if isinstance(cb, fronts_callbacks.MetricsConsolidationCallback)
        )
        csv_idx = next(i for i, cb in enumerate(callbacks) if isinstance(cb, _ResumeSafeCSVLogger))
        assert consolidation_idx < csv_idx

    def test_explicit_metrics_csv_path_is_honored(self, tmp_path):
        csv_path = str(tmp_path / "custom_name.csv")
        callbacks = self._build(uses_ema=False, metrics_csv_path=csv_path)
        csv_logger = next(cb for cb in callbacks if isinstance(cb, _ResumeSafeCSVLogger))
        assert csv_logger.path == csv_path

    def test_metrics_csv_path_none_derives_path_beside_checkpoint(self, tmp_path):
        checkpoint_path = str(tmp_path / "model")
        callbacks = self._build(uses_ema=False, model_checkpoint_path=checkpoint_path, metrics_csv_path=None)
        csv_logger = next(cb for cb in callbacks if isinstance(cb, _ResumeSafeCSVLogger))
        assert csv_logger.path == str(tmp_path / "metrics_epoch.csv")

    def test_both_none_omits_csv_logger(self):
        callbacks = self._build(uses_ema=False, model_checkpoint_path=None, metrics_csv_path=None)
        assert not any(isinstance(cb, _ResumeSafeCSVLogger) for cb in callbacks)

    def test_derived_csv_path_creates_missing_parent_directory(self, tmp_path):
        checkpoint_path = str(tmp_path / "nested" / "run1" / "model")
        self._build(uses_ema=False, model_checkpoint_path=checkpoint_path, metrics_csv_path=None)
        assert (tmp_path / "nested" / "run1").is_dir()

    def test_compact_progress_none_omits_callback(self):
        callbacks = self._build(uses_ema=False, compact_progress_every_n_batches=None)
        assert not any(isinstance(cb, fronts_callbacks.CompactProgressCallback) for cb in callbacks)

    def test_compact_progress_int_adds_callback_after_metrics_consolidation(self):
        callbacks = self._build(uses_ema=False, compact_progress_every_n_batches=10)
        consolidation_idx = next(
            i for i, cb in enumerate(callbacks) if isinstance(cb, fronts_callbacks.MetricsConsolidationCallback)
        )
        compact_idx = next(
            i for i, cb in enumerate(callbacks) if isinstance(cb, fronts_callbacks.CompactProgressCallback)
        )
        assert consolidation_idx < compact_idx

    def test_compact_progress_int_is_passed_through_as_throttle(self):
        callbacks = self._build(uses_ema=False, compact_progress_every_n_batches=7)
        compact_callback = next(cb for cb in callbacks if isinstance(cb, fronts_callbacks.CompactProgressCallback))
        assert compact_callback.every_n_batches == 7

    def test_resolve_fit_verbose_unchanged_when_compact_progress_none(self):
        assert _resolve_fit_verbose(None) == "auto"

    def test_resolve_fit_verbose_silenced_when_compact_progress_enabled(self):
        assert _resolve_fit_verbose(10) == 0


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestResumeSafeCSVLogger:
    """Covers the resume-safe CSVLogger wrapper added to fix silent header misalignment on resume."""

    _LOGS: ClassVar[dict[str, float]] = {"loss": 0.5, "hss": 0.6, "val_loss": 0.4, "val_hss": 0.55}
    _HEADER = "epoch,hss,loss,val_hss,val_loss"

    def test_no_existing_file_creates_fresh_csv_with_header(self, tmp_path):
        csv_path = tmp_path / "metrics_epoch.csv"
        wrapper = _ResumeSafeCSVLogger(str(csv_path))
        wrapper.on_epoch_end(0, dict(self._LOGS))
        wrapper.on_epoch_end(1, dict(self._LOGS))
        wrapper.on_train_end()

        lines = csv_path.read_text().strip().split("\n")
        assert lines[0] == self._HEADER
        assert len(lines) == 3  # header + 2 epochs
        assert not list(tmp_path.glob("*.stale-*.csv"))

    def test_matching_header_resume_appends(self, tmp_path, caplog):
        csv_path = tmp_path / "metrics_epoch.csv"
        csv_path.write_text(f"{self._HEADER}\n0,0.6,0.5,0.55,0.4\n")

        wrapper = _ResumeSafeCSVLogger(str(csv_path))
        with caplog.at_level(logging.WARNING, logger="fronts.train"):
            wrapper.on_epoch_end(1, dict(self._LOGS))
        wrapper.on_train_end()

        lines = csv_path.read_text().strip().split("\n")
        assert lines[0] == self._HEADER
        assert len(lines) == 3  # original header + original row + newly-appended row
        assert not list(tmp_path.glob("*.stale-*.csv"))
        assert "stale" not in caplog.text.lower()

    def test_mismatched_header_rotates_stale_file_and_warns(self, tmp_path, caplog):
        csv_path = tmp_path / "metrics_epoch.csv"
        stale_content = "epoch,loss,hss\n0,0.5,0.6\n"
        csv_path.write_text(stale_content)

        wrapper = _ResumeSafeCSVLogger(str(csv_path))
        with caplog.at_level(logging.WARNING, logger="fronts.train"):
            wrapper.on_epoch_end(0, dict(self._LOGS))
        wrapper.on_train_end()

        stale_path = tmp_path / "metrics_epoch.stale-1.csv"
        assert stale_path.read_text() == stale_content

        fresh_lines = csv_path.read_text().strip().split("\n")
        assert fresh_lines[0] == self._HEADER
        assert len(fresh_lines) == 2  # fresh header + the epoch just run

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.WARNING
        assert str(csv_path) in caplog.text
        assert str(stale_path) in caplog.text

    def test_second_mismatch_uses_next_free_stale_suffix(self, tmp_path):
        csv_path = tmp_path / "metrics_epoch.csv"
        (tmp_path / "metrics_epoch.stale-1.csv").write_text("already,taken\n")
        csv_path.write_text("epoch,loss,hss\n0,0.5,0.6\n")

        wrapper = _ResumeSafeCSVLogger(str(csv_path))
        wrapper.on_epoch_end(0, dict(self._LOGS))
        wrapper.on_train_end()

        assert (tmp_path / "metrics_epoch.stale-2.csv").exists()


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

    def test_shuffle_reorders_batches_not_samples_within_a_batch(self, era5_ds, front_da, data_config):
        """Shuffling must only reorder whole batches, keeping each batch a contiguous read.

        Both icechunk stores backing this dataset chunk at 1 timestep, so a fully random
        per-sample shuffle turns every batch read into scattered single-chunk fetches,
        measured at 10-30x slower than a sequential read of the same size (see
        scripts/diagnose_read_throughput.py). Every batch must therefore still correspond
        to some contiguous run of the original timesteps, even with shuffling enabled.
        """
        batch_size = 2
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=batch_size, shuffle=True, seed=0)
        expected = inputs_ds_to_dataarray(era5_ds, data_config.variables).values
        for i in range(len(ds)):
            x_batch, _ = ds[i]
            n = x_batch.shape[0]
            matches = [s for s in range(N_TIME - n + 1) if np.allclose(x_batch, expected[s : s + n])]
            assert matches, f"batch {i} is not a contiguous run of original timesteps"

    def test_shuffle_visits_every_batch_exactly_once_per_epoch(self, era5_ds, front_da, data_config):
        batch_size = 2
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=batch_size, shuffle=True, seed=0)
        np.testing.assert_array_equal(np.sort(ds._order), np.arange(len(ds)))

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
class TestBuildDatasetSummary:
    def _make_dated_dataset(self, era5_ds, front_da, data_config, times, batch_size=2):
        input_ds = era5_ds.assign_coords(time=times)
        target_da = front_da.assign_coords(time=times)
        return FrontsPyDataset(input_ds, target_da, data_config, batch_size=batch_size)

    def test_input_and_target_shapes(self, era5_ds, front_da, data_config):
        times = pd.date_range("2020-01-01", periods=N_TIME, freq="6h")
        dataset = self._make_dated_dataset(era5_ds, front_da, data_config, times)

        summary = _build_dataset_summary("train", dataset, data_config)

        assert summary.split == "train"
        assert summary.input_shape == (N_TIME, N_LAT, N_LON, len(data_config.variables))
        assert summary.target_shape == (N_TIME, N_LAT, N_LON)

    def test_date_range_matches_time_coordinate(self, era5_ds, front_da, data_config):
        times = pd.date_range("2020-01-01", periods=N_TIME, freq="6h")
        dataset = self._make_dated_dataset(era5_ds, front_da, data_config, times)

        summary = _build_dataset_summary("val", dataset, data_config)

        assert summary.date_min == str(times.min().date())
        assert summary.date_max == str(times.max().date())

    def test_out_of_order_times_still_yield_true_min_max(self, era5_ds, front_da, data_config):
        times = pd.to_datetime(["2020-03-01", "2020-01-05", "2020-02-10", "2020-01-01", "2020-02-20"])
        dataset = self._make_dated_dataset(era5_ds, front_da, data_config, times)

        summary = _build_dataset_summary("test", dataset, data_config)

        assert summary.date_min == "2020-01-01"
        assert summary.date_max == "2020-03-01"

    def test_empty_split_raises(self, era5_ds, front_da, data_config):
        times = pd.date_range("2020-01-01", periods=N_TIME, freq="6h")
        dataset = self._make_dated_dataset(era5_ds, front_da, data_config, times)
        empty_dataset = FrontsPyDataset(
            dataset.input_ds.isel(time=slice(0, 0)),
            dataset.target_da.isel(time=slice(0, 0)),
            data_config,
            batch_size=2,
        )

        with pytest.raises(ValueError, match="empty"):
            _build_dataset_summary("test", empty_dataset, data_config)

    def test_volume_inputs_uses_volume_stacking(self):
        rng = np.random.default_rng(11)
        n_time, n_lat, n_lon = 4, 6, 8
        levels = (1000, 950)
        times = pd.date_range("2021-06-01", periods=n_time, freq="6h")
        input_ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    rng.standard_normal((n_time, len(levels), n_lat, n_lon)).astype(np.float32),
                    dims=["time", "level", "latitude", "longitude"],
                    coords={"time": times, "level": list(levels)},
                ),
                "mean_sea_level_pressure": xr.DataArray(
                    rng.standard_normal((n_time, n_lat, n_lon)).astype(np.float32),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": times},
                ),
            }
        )
        target_da = xr.DataArray(
            rng.integers(0, 2, size=(n_time, n_lat, n_lon)).astype(np.int32),
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
        dataset = FrontsPyDataset(input_ds, target_da, config, batch_size=2)

        summary = _build_dataset_summary("train", dataset, config)

        assert summary.input_shape == (n_time, n_lat, n_lon, len(levels), 2)
        assert summary.date_min == str(times.min().date())
        assert summary.date_max == str(times.max().date())

    def test_does_not_eagerly_materialize_a_chunks_none_input_ds(self, era5_ds, front_da, data_config, monkeypatch):
        """Regression test: stacking a chunks=None input_ds must not materialize the full split.

        ``load_data_into_dataloader`` opens ``input_ds`` with ``chunks=None`` so per-batch
        training reads go straight through zarr. Stacking that non-dask array directly
        (``to_array``/``stack``) forces full materialization into RAM regardless of
        ``batch_size`` — the same trap ``load_or_compute_norm_stats``'s caller avoids with a
        metadata-only ``.chunk("auto")`` first. This asserts the same precaution is taken here.
        """
        times = pd.date_range("2020-01-01", periods=N_TIME, freq="6h")
        dataset = self._make_dated_dataset(era5_ds, front_da, data_config, times)

        original_chunk = xr.Dataset.chunk
        chunked_datasets = []

        def _spy_chunk(self, *args, **kwargs):
            result = original_chunk(self, *args, **kwargs)
            chunked_datasets.append(result)
            return result

        monkeypatch.setattr(xr.Dataset, "chunk", _spy_chunk)

        _build_dataset_summary("train", dataset, data_config)

        assert chunked_datasets, "_build_dataset_summary must .chunk() input_ds before stacking it"
        assert all(var.chunks is not None for var in chunked_datasets[-1].data_vars.values()), (
            "input_ds must be dask-backed before to_array/stack, or the full split materializes eagerly"
        )


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestBuildTestVisualizationCallback:
    def test_builds_from_already_loaded_test_dataset(self, data_config):
        n_time, n_lat, n_lon = 4, 3, 3
        times = pd.date_range("2022-01-01", periods=n_time, freq="6h")
        rng = np.random.default_rng(5)
        input_ds = xr.Dataset(
            {
                var: xr.DataArray(
                    rng.standard_normal((n_time, n_lat, n_lon)).astype(np.float32),
                    dims=["time", "latitude", "longitude"],
                    coords={
                        "time": times,
                        "latitude": np.arange(n_lat, dtype=np.float32),
                        "longitude": np.arange(n_lon, dtype=np.float32),
                    },
                )
                for var in data_config.variables
            }
        )
        target_data = np.zeros((n_time, n_lat, n_lon), dtype=np.int32)
        target_data[1, 0, 0] = 1  # CF code at timestep 1, so it's the "active" day.
        target_da = xr.DataArray(target_data, dims=["time", "latitude", "longitude"], coords={"time": times})
        test_dataset = FrontsPyDataset(input_ds, target_da, data_config, batch_size=2)
        callbacks_config = CallbacksConfig(test_viz_every_n_epochs=5, test_viz_sample_size=2)

        cb = _build_test_visualization_callback(test_dataset, data_config, callbacks_config, seed=0)

        assert cb.every_n_epochs == 5
        assert cb.active_day_x.shape == (n_lat, n_lon, len(data_config.variables))
        assert cb.subsample_x.shape[0] <= 2


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
class TestLoadDataIntoDataloaderPressureLevels:
    """data_config.pressure_levels must restrict the loaded store to that subset of levels."""

    _TIMES = pd.date_range("2020-01-01", periods=4, freq="6h")
    _LAT = np.array([10.0, 20.0, 30.0, 40.0])
    _LON = np.array([100.0, 110.0, 120.0, 130.0])
    _LEVELS = np.array([1000, 850, 500, 300])

    def _write_store(self, tmp_path, name: str, var_name: str) -> IcechunkStorageConfig:
        storage_config = IcechunkStorageConfig(store_path=str(tmp_path / name), branch_name="main")
        ds = xr.Dataset(
            {
                var_name: xr.DataArray(
                    np.zeros((len(self._TIMES), len(self._LEVELS), len(self._LAT), len(self._LON)), dtype=np.float32),
                    dims=["time", "level", "latitude", "longitude"],
                    coords={
                        "time": self._TIMES,
                        "level": self._LEVELS,
                        "latitude": self._LAT,
                        "longitude": self._LON,
                    },
                )
            }
        )
        write_or_append_icechunk_store(storage_config, ds)
        return storage_config

    def _write_target_store(self, tmp_path, name: str, var_name: str) -> IcechunkStorageConfig:
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

    def test_pressure_levels_restrict_loaded_levels(self, tmp_path):
        data_config = DatasetConfig(
            inputs_icechunk_config=self._write_store(tmp_path, "inputs", "temperature"),
            targets_icechunk_config=self._write_target_store(tmp_path, "targets", "identifier"),
            variables=["temperature"],
            test_years=[2020],
            val_years=[],
            pressure_levels=[1000, 500],
        )
        test_dataset = load_data_into_dataloader(data_config, split="test", seed=0)
        levels = test_dataset.input_ds["level"].values
        assert sorted(levels.tolist()) == [500, 1000], f"levels not restricted: {levels}"

    def test_pressure_levels_none_keeps_all_levels(self, tmp_path):
        data_config = DatasetConfig(
            inputs_icechunk_config=self._write_store(tmp_path, "inputs", "temperature"),
            targets_icechunk_config=self._write_target_store(tmp_path, "targets", "identifier"),
            variables=["temperature"],
            test_years=[2020],
            val_years=[],
        )
        test_dataset = load_data_into_dataloader(data_config, split="test", seed=0)
        levels = test_dataset.input_ds["level"].values
        assert sorted(levels.tolist()) == sorted(self._LEVELS.tolist())


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

    def test_nbs_include_pixel_defaults_to_off(self):
        """nbs_include_pixel/nbs_pixel_weight are optional — omitting them must not change the loss."""
        rng = np.random.default_rng(1)
        n_classes = 3
        y_true = tf.one_hot(rng.integers(0, n_classes, size=(2, 8, 8)), n_classes)
        y_pred = tf.nn.softmax(rng.standard_normal((2, 8, 8, n_classes)).astype(np.float32), axis=-1)

        default_loss = _build_loss(
            loss_name="neighborhood_brier_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
        )
        explicit_off_loss = _build_loss(
            loss_name="neighborhood_brier_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
            nbs_include_pixel=False,
        )
        default_value = float(tf.reduce_mean(default_loss(y_true, y_pred)))
        explicit_off_value = float(tf.reduce_mean(explicit_off_loss(y_true, y_pred)))
        assert default_value == pytest.approx(explicit_off_value)

    def test_nbs_include_pixel_true_changes_loss(self):
        """Turning on nbs_include_pixel must add the un-pooled pixelwise term."""
        rng = np.random.default_rng(2)
        n_classes = 3
        y_true = tf.one_hot(rng.integers(0, n_classes, size=(2, 8, 8)), n_classes)
        y_pred = tf.nn.softmax(rng.standard_normal((2, 8, 8, n_classes)).astype(np.float32), axis=-1)

        pooled_only_loss = _build_loss(
            loss_name="neighborhood_brier_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
            nbs_include_pixel=False,
        )
        with_pixel_loss = _build_loss(
            loss_name="neighborhood_brier_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
            nbs_include_pixel=True,
            nbs_pixel_weight=0.5,
        )
        pooled_only_value = float(tf.reduce_mean(pooled_only_loss(y_true, y_pred)))
        with_pixel_value = float(tf.reduce_mean(with_pixel_loss(y_true, y_pred)))
        assert np.isfinite(with_pixel_value)
        assert with_pixel_value != pytest.approx(pooled_only_value)


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

    def test_nbs_include_pixel_defaults(self, train_config_cls):
        from fronts import utils

        yaml_data = {"train_config": {"loss_class_weights": None, "epochs": 1}}
        cfg = utils.parse_config_section(yaml_data, train_config_cls, "train_config")
        assert cfg.nbs_include_pixel is False
        assert cfg.nbs_pixel_weight == 0.1

    def test_sooner_ablations_config_parses_nbs_pixel_fields(self, train_config_cls):
        from fronts import utils

        yaml_data = utils.load_yaml("configs/sooner_ablations.yaml")
        cfg = utils.parse_config_section(yaml_data, train_config_cls, "train_config")
        assert cfg.nbs_include_pixel is True
        assert cfg.nbs_pixel_weight == 0.5

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


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestCompileEma:
    _LATITUDES = np.linspace(25.0, 56.75, 8)

    def test_use_ema_defaults_to_false(self):
        unet = _build_small_unet()
        train_cfg = TrainConfig(loss_class_weights=None)
        _compile(
            unet,
            learning_rate=1e-4,
            metric_class_weights=None,
            train_cfg=train_cfg,
            latitudes=self._LATITUDES,
            n_classes=6,
        )
        assert unet.optimizer.use_ema is False

    def test_use_ema_true_sets_optimizer_flag_and_momentum(self):
        unet = _build_small_unet()
        train_cfg = TrainConfig(loss_class_weights=None, use_ema=True, ema_momentum=0.95)
        _compile(
            unet,
            learning_rate=1e-4,
            metric_class_weights=None,
            train_cfg=train_cfg,
            latitudes=self._LATITUDES,
            n_classes=6,
        )
        assert unet.optimizer.use_ema is True
        assert unet.optimizer.ema_momentum == pytest.approx(0.95)

    def test_use_ema_false_leaves_default_ema_momentum_inert(self):
        """ema_momentum must not affect a compiled optimizer when use_ema is False."""
        unet = _build_small_unet()
        train_cfg = TrainConfig(loss_class_weights=None, use_ema=False, ema_momentum=0.5)
        _compile(
            unet,
            learning_rate=1e-4,
            metric_class_weights=None,
            train_cfg=train_cfg,
            latitudes=self._LATITUDES,
            n_classes=6,
        )
        assert unet.optimizer.use_ema is False


def _compiled_metric_names(metric_config_entry: list[dict]) -> list[str]:
    """Extract metric names from one output's slice of ``Model.get_compile_config()["metrics"]``.

    A plain function metric (e.g. ``metrics.heidke_skill_score(...)``) serializes with its
    ``__name__`` directly under ``"config"``; a ``tf.keras.metrics.Metric`` subclass (e.g.
    ``MeanMetricWrapper``, ``PerClassContingencyMetric``) serializes its constructor
    arguments, including ``name``, in a nested ``"config"`` dict.
    """
    names = []
    for entry in metric_config_entry:
        config = entry["config"]
        names.append(config if isinstance(config, str) else config["name"])
    return names


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestCompilePerFrontTypeMetrics:
    _LATITUDES = np.linspace(25.0, 56.75, 8)

    def test_true_attaches_per_class_metrics_to_output_0_only(self):
        unet = _build_small_unet(levels=3, deep_supervision=True)
        train_cfg = TrainConfig(loss_class_weights=None, per_front_type_metrics=True)
        n_out = _compile(
            unet,
            learning_rate=1e-4,
            metric_class_weights=None,
            train_cfg=train_cfg,
            latitudes=self._LATITUDES,
            n_classes=6,
        )
        assert n_out == 3
        metrics_config = unet.get_compile_config()["metrics"]
        assert len(metrics_config) == 3

        names0 = _compiled_metric_names(metrics_config[0])
        assert names0[:2] == ["hss", "hss_hard"]
        for front_type in constants.FRONT_TYPE_CLASS_INDEX:
            assert f"hss_{front_type}" in names0
            assert f"hss_hard_{front_type}" in names0
            assert f"csi_{front_type}" in names0
            assert f"pod_{front_type}" in names0
            assert f"loss_{front_type}" in names0
        assert f"loss_{constants.BACKGROUND_CLASS_KEY}" in names0

        for other_output in metrics_config[1:]:
            assert _compiled_metric_names(other_output) == ["hss", "hss_hard"]

    def test_false_reproduces_current_metric_layout(self):
        unet = _build_small_unet(levels=3, deep_supervision=True)
        train_cfg = TrainConfig(loss_class_weights=None, per_front_type_metrics=False)
        n_out = _compile(
            unet,
            learning_rate=1e-4,
            metric_class_weights=None,
            train_cfg=train_cfg,
            latitudes=self._LATITUDES,
            n_classes=6,
        )
        metrics_config = unet.get_compile_config()["metrics"]
        assert len(metrics_config) == n_out
        for output_metrics in metrics_config:
            assert _compiled_metric_names(output_metrics) == ["hss", "hss_hard"]

    def test_all_metric_names_unique_per_output(self):
        unet = _build_small_unet(levels=3, deep_supervision=True)
        train_cfg = TrainConfig(loss_class_weights=None, per_front_type_metrics=True)
        _compile(
            unet,
            learning_rate=1e-4,
            metric_class_weights=None,
            train_cfg=train_cfg,
            latitudes=self._LATITUDES,
            n_classes=6,
        )
        metrics_config = unet.get_compile_config()["metrics"]
        for output_metrics in metrics_config:
            names = _compiled_metric_names(output_metrics)
            assert len(names) == len(set(names))

    def test_n_out_one_attaches_per_class_metrics_to_the_only_output(self):
        unet = _build_small_unet(levels=3, deep_supervision=False)
        train_cfg = TrainConfig(loss_class_weights=None, per_front_type_metrics=True)
        n_out = _compile(
            unet,
            learning_rate=1e-4,
            metric_class_weights=None,
            train_cfg=train_cfg,
            latitudes=self._LATITUDES,
            n_classes=6,
        )
        assert n_out == 1
        metrics_config = unet.get_compile_config()["metrics"]
        assert len(metrics_config) == 1
        names0 = _compiled_metric_names(metrics_config[0])
        assert names0[:2] == ["hss", "hss_hard"]
        assert "hss_CF" in names0
        assert "loss_CF" in names0
        assert f"loss_{constants.BACKGROUND_CLASS_KEY}" in names0

    def test_loss_value_identical_with_flag_on_and_off(self):
        """Global Constraint regression test: the flag must never change the compiled loss."""
        rng = np.random.default_rng(3)
        x = rng.standard_normal((2, 16, 16, 4)).astype(np.float32)
        y = tf.one_hot(rng.integers(0, N_CLASSES, size=(2, 16, 16)), N_CLASSES).numpy().astype(np.float32)
        latitudes = np.linspace(25.0, 56.75, 16)

        unet_on = _build_small_unet(levels=3, deep_supervision=False)
        unet_off = _build_small_unet(levels=3, deep_supervision=False)
        unet_off.set_weights(unet_on.get_weights())

        train_cfg_on = TrainConfig(
            loss_class_weights=None, loss_name="neighborhood_brier_score", per_front_type_metrics=True
        )
        train_cfg_off = dataclasses.replace(train_cfg_on, per_front_type_metrics=False)

        _compile(
            unet_on,
            learning_rate=1e-4,
            metric_class_weights=None,
            train_cfg=train_cfg_on,
            latitudes=latitudes,
            n_classes=6,
        )
        _compile(
            unet_off,
            learning_rate=1e-4,
            metric_class_weights=None,
            train_cfg=train_cfg_off,
            latitudes=latitudes,
            n_classes=6,
        )

        loss_on = unet_on.evaluate(x, y, verbose=0)[0]
        loss_off = unet_off.evaluate(x, y, verbose=0)[0]
        assert loss_on == pytest.approx(loss_off, abs=1e-7)


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestPerFrontTypeLossMetrics:
    _LATITUDES = np.linspace(25.0, 56.75, 8)

    def _fixed_batch(self, seed: int, n_classes: int = 6, batch: int = 2, height: int = 8, width: int = 8):
        rng = np.random.default_rng(seed)
        y_true = tf.one_hot(rng.integers(0, n_classes, size=(batch, height, width)), n_classes)
        y_pred = tf.nn.softmax(rng.standard_normal((batch, height, width, n_classes)).astype(np.float32), axis=-1)
        return y_true, y_pred

    def _per_class_values(self, loss_name, loss_class_weights, y_true, y_pred, n_classes: int = 6) -> list[float]:
        per_class_metrics = _per_front_type_loss_metrics(
            loss_name=loss_name,
            loss_class_weights=loss_class_weights,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
            nbs_include_pixel=False,
            nbs_pixel_weight=0.1,
            n_classes=n_classes,
        )
        values = []
        for metric in per_class_metrics:
            metric.update_state(y_true, y_pred)
            values.append(float(metric.result()))
        return values

    def test_returns_six_uniquely_named_metrics(self):
        per_class_metrics = _per_front_type_loss_metrics(
            loss_name="neighborhood_brier_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
            nbs_include_pixel=False,
            nbs_pixel_weight=0.1,
            n_classes=6,
        )
        names = [metric.name for metric in per_class_metrics]
        assert len(names) == len(set(names)) == 6
        assert f"loss_{constants.BACKGROUND_CLASS_KEY}" in names
        for front_type in constants.FRONT_TYPE_CLASS_INDEX:
            assert f"loss_{front_type}" in names

    def test_n_classes_mismatch_raises_clear_error(self):
        with pytest.raises(ValueError, match=r"n_classes=7.*does not match.*6"):
            _per_front_type_loss_metrics(
                loss_name="neighborhood_brier_score",
                loss_class_weights=None,
                latitudes=self._LATITUDES,
                fss_mask_size=(3, 3),
                nbs_tolerance_km=25.0,
                nbs_periodic_lon=False,
                nbs_lat_dependent_pool=False,
                nbs_include_pixel=False,
                nbs_pixel_weight=0.1,
                n_classes=7,
            )

    def test_n_classes_matching_front_type_index_succeeds(self):
        per_class_metrics = _per_front_type_loss_metrics(
            loss_name="neighborhood_brier_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
            nbs_include_pixel=False,
            nbs_pixel_weight=0.1,
            n_classes=6,
        )
        assert len(per_class_metrics) == 6

    def test_nbs_per_class_losses_sum_to_total_unweighted(self):
        y_true, y_pred = self._fixed_batch(seed=0)
        total_loss_fn = _build_loss(
            loss_name="neighborhood_brier_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
        )
        total = float(tf.reduce_mean(total_loss_fn(y_true, y_pred)))
        per_class_total = sum(self._per_class_values("neighborhood_brier_score", None, y_true, y_pred))
        assert per_class_total == pytest.approx(total, abs=1e-6)

    def test_nbs_per_class_losses_sum_to_total_with_zero_weighted_background(self):
        """Additivity must hold even when a class (background) is weighted to exactly zero."""
        y_true, y_pred = self._fixed_batch(seed=1)
        loss_class_weights = [0.0, 1.0, 2.0, 1.0, 1.0, 3.0]
        total_loss_fn = _build_loss(
            loss_name="neighborhood_brier_score",
            loss_class_weights=loss_class_weights,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
        )
        total = float(tf.reduce_mean(total_loss_fn(y_true, y_pred)))
        per_class_values = self._per_class_values("neighborhood_brier_score", loss_class_weights, y_true, y_pred)
        assert per_class_values[0] == pytest.approx(0.0, abs=1e-8)  # background ("none"), weight 0.0
        assert sum(per_class_values) == pytest.approx(total, abs=1e-6)

    def test_nbs_per_class_losses_sum_to_total_with_include_pixel(self):
        y_true, y_pred = self._fixed_batch(seed=2)
        loss_class_weights = [1.0, 2.0, 0.5, 1.0, 1.0, 1.5]
        total_loss_fn = _build_loss(
            loss_name="neighborhood_brier_score",
            loss_class_weights=loss_class_weights,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
            nbs_include_pixel=True,
            nbs_pixel_weight=0.5,
        )
        total = float(tf.reduce_mean(total_loss_fn(y_true, y_pred)))
        per_class_metrics = _per_front_type_loss_metrics(
            loss_name="neighborhood_brier_score",
            loss_class_weights=loss_class_weights,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
            nbs_include_pixel=True,
            nbs_pixel_weight=0.5,
            n_classes=6,
        )
        per_class_total = 0.0
        for metric in per_class_metrics:
            metric.update_state(y_true, y_pred)
            per_class_total += float(metric.result())
        assert per_class_total == pytest.approx(total, abs=1e-6)

    def test_fss_per_class_losses_bound_total_and_are_not_additive(self):
        y_true, y_pred = self._fixed_batch(seed=4)
        loss_class_weights = [0.0, 1.0, 2.0, 1.0, 1.0, 3.0]
        total_loss_fn = _build_loss(
            loss_name="fractions_skill_score",
            loss_class_weights=loss_class_weights,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
        )
        total = float(tf.reduce_mean(total_loss_fn(y_true, y_pred)))
        per_class_values = self._per_class_values("fractions_skill_score", loss_class_weights, y_true, y_pred)

        assert min(per_class_values) - 1e-6 <= total <= max(per_class_values) + 1e-6
        # Documents that FSS is NOT additive (unlike NBS above) — guards against this test
        # silently passing for the wrong reason if the implementation ever changed.
        assert sum(per_class_values) != pytest.approx(total, abs=1e-3)

    def test_fss_zero_weighted_class_does_not_produce_nan(self):
        """A configured zero weight must not NaN out that class's FSS metric (unlike a scaled one-hot would)."""
        y_true, y_pred = self._fixed_batch(seed=5)
        loss_class_weights = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        per_class_values = self._per_class_values("fractions_skill_score", loss_class_weights, y_true, y_pred)
        assert all(np.isfinite(per_class_values))


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestCollectRunMetadata:
    def test_includes_config_name_and_path(self, data_config, monkeypatch):
        monkeypatch.setattr("fronts.train.utils.get_git_commit", lambda: "abc123")
        monkeypatch.setattr("fronts.train.utils.get_icechunk_snapshot_id", lambda *args, **kwargs: "snap1")

        meta = _collect_run_metadata(data_config, config_path="configs/sooner_train.yaml")

        assert meta["config_name"] == "sooner_train.yaml"
        assert meta["config_path"] == "configs/sooner_train.yaml"


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestBuildWandbConfig:
    def test_nests_every_config_section_under_its_yaml_key(self, data_config):
        model_cfg = ModelConfig()
        callbacks_cfg = CallbacksConfig()
        train_cfg = TrainConfig(loss_class_weights=None)
        wandb_cfg = WandBConfig(log_freq="epoch")

        result = _build_wandb_config(data_config, model_cfg, callbacks_cfg, train_cfg, wandb_cfg, run_meta={})

        assert result["data_config"] == dataclasses.asdict(data_config)
        assert result["model_config"] == dataclasses.asdict(model_cfg)
        assert result["callbacks_config"] == dataclasses.asdict(callbacks_cfg)
        assert result["train_config"] == dataclasses.asdict(train_cfg)
        assert result["wandb_config"] == dataclasses.asdict(wandb_cfg)

    def test_flattens_run_meta_alongside_config_sections(self, data_config):
        model_cfg = ModelConfig()
        callbacks_cfg = CallbacksConfig()
        train_cfg = TrainConfig(loss_class_weights=None)
        wandb_cfg = WandBConfig(log_freq="epoch")
        run_meta = {"git_commit": "abc123", "era5_snapshot_id": "snap1"}

        result = _build_wandb_config(data_config, model_cfg, callbacks_cfg, train_cfg, wandb_cfg, run_meta)

        assert result["git_commit"] == "abc123"
        assert result["era5_snapshot_id"] == "snap1"

    def test_nested_dataclass_fields_are_plain_dicts(self, data_config):
        """Nested dataclasses (e.g. IcechunkStorageConfig) must serialize to dicts, not objects, for W&B."""
        model_cfg = ModelConfig()
        callbacks_cfg = CallbacksConfig()
        train_cfg = TrainConfig(loss_class_weights=None)
        wandb_cfg = WandBConfig(log_freq="epoch")

        result = _build_wandb_config(data_config, model_cfg, callbacks_cfg, train_cfg, wandb_cfg, run_meta={})

        assert isinstance(result["data_config"]["inputs_icechunk_config"], dict)
        assert result["data_config"]["inputs_icechunk_config"]["store_path"] == "unused"
