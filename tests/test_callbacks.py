"""Tests for fronts.callbacks: W&B metric consolidation and test-set visualization helpers."""

import numpy as np
import pytest
import xarray as xr

fc = pytest.importorskip("fronts.callbacks")


class TestMetricsConsolidationCallback:
    def test_aggregates_hss_and_strips_per_output_keys(self):
        logs = {
            "loss": 1.0,
            "sup1_softmax_hss": 0.1,
            "sup1_softmax_loss": 0.4,
            "sup2_softmax_hss": 0.3,
            "sup2_softmax_loss": 0.2,
            "val_loss": 1.5,
            "val_sup1_softmax_hss": 0.2,
            "val_sup1_softmax_loss": 0.5,
            "val_sup2_softmax_hss": 0.4,
            "val_sup2_softmax_loss": 0.3,
        }
        fc.MetricsConsolidationCallback().on_epoch_end(0, logs)

        assert logs == {
            "loss": 1.0,
            "val_loss": 1.5,
            "hss": pytest.approx(0.2),
            "val_hss": pytest.approx(0.3),
        }

    def test_aggregates_multiple_custom_metrics_independently(self):
        """A second custom metric (e.g. hss_hard) must aggregate to its own key, not hss's."""
        logs = {
            "loss": 1.0,
            "sup1_softmax_hss": 0.1,
            "sup1_softmax_hss_hard": 0.6,
            "sup1_softmax_loss": 0.4,
            "sup2_softmax_hss": 0.3,
            "sup2_softmax_hss_hard": 0.8,
            "sup2_softmax_loss": 0.2,
        }
        fc.MetricsConsolidationCallback().on_epoch_end(0, logs)

        assert logs == {
            "loss": 1.0,
            "hss": pytest.approx(0.2),
            "hss_hard": pytest.approx(0.7),
        }

    def test_noop_on_empty_logs(self):
        logs = {}
        fc.MetricsConsolidationCallback().on_epoch_end(0, logs)
        assert logs == {}

    def test_noop_on_none_logs(self):
        # Should not raise even though Keras can call on_epoch_end with logs=None.
        fc.MetricsConsolidationCallback().on_epoch_end(0, None)


class TestSelectActiveTestTimestep:
    def test_returns_first_timestep_with_a_front(self):
        data = np.zeros((4, 3, 3), dtype=np.int32)
        data[2, 1, 1] = 1  # CF code at time index 2
        target_da = xr.DataArray(data, dims=["time", "latitude", "longitude"])
        assert fc.select_active_test_timestep(target_da) == 2

    def test_raises_when_no_front_present(self):
        data = np.zeros((4, 3, 3), dtype=np.int32)
        target_da = xr.DataArray(data, dims=["time", "latitude", "longitude"])
        with pytest.raises(ValueError):
            fc.select_active_test_timestep(target_da)


class TestSelectTestSubsample:
    def test_bounded_and_sorted(self):
        idxs = fc.select_test_subsample(n_total=100, sample_size=10, seed=0)
        assert len(idxs) == 10
        assert (np.diff(idxs) > 0).all()
        assert idxs.min() >= 0
        assert idxs.max() < 100

    def test_clamps_to_n_total(self):
        idxs = fc.select_test_subsample(n_total=5, sample_size=200, seed=0)
        assert len(idxs) == 5

    def test_deterministic_for_fixed_seed(self):
        a = fc.select_test_subsample(n_total=50, sample_size=10, seed=42)
        b = fc.select_test_subsample(n_total=50, sample_size=10, seed=42)
        np.testing.assert_array_equal(a, b)


class TestRegionMask:
    def test_whole_domain_is_all_true(self):
        lats = np.array([10.0, 20.0, 30.0])
        lons = np.array([100.0, 200.0])
        mask = fc.region_mask(lats, lons, None)
        assert mask.shape == (3, 2)
        assert mask.all()

    def test_box_restricts_lat_and_lon(self):
        lats = np.array([10.0, 20.0, 30.0, 40.0])
        lons = np.array([100.0, 150.0, 200.0, 250.0])
        region = fc.utils.BoundingBox(lat_min=20.0, lat_max=40.0, lon_min=150.0, lon_max=250.0)
        mask = fc.region_mask(lats, lons, region)
        expected = np.array(
            [
                [False, False, False, False],
                [False, True, True, True],
                [False, True, True, True],
                [False, True, True, True],
            ]
        )
        np.testing.assert_array_equal(mask, expected)


class TestVisualizationCallbackPredict:
    """Tests the viz for predictions in callbacks.

    CallbackPredict must chunk by predict_batch_size rather than calling the model on the full array at once:
    a single unbatched call on e.g. 200 full-resolution test timesteps allocates one huge activation buffer on top
    of training's already resident GPU memory and reliably OOMs (see callbacks.py:on_epoch_end).
    """

    def _make_callback(self, n_samples: int, predict_batch_size: int) -> "fc.TestVisualizationCallback":
        inputs = fc.tf.keras.Input(shape=(2, 2, 1))
        model = fc.tf.keras.Model(inputs, inputs)  # identity: output == input
        cb = fc.TestVisualizationCallback(
            active_day_x=np.zeros((2, 2, 1), dtype=np.float32),
            active_day_y=np.zeros((2, 2, 1), dtype=np.float32),
            active_day_label="active day",
            subsample_x=np.arange(n_samples * 4, dtype=np.float32).reshape(n_samples, 2, 2, 1),
            subsample_y=np.zeros((n_samples, 2, 2, 1), dtype=np.float32),
            lats=np.array([0.0, 1.0]),
            lons=np.array([0.0, 1.0]),
            front_types=["CF"],
            predict_batch_size=predict_batch_size,
        )
        cb.set_model(model)
        return cb

    def test_chunked_prediction_matches_unbatched_input(self):
        # 5 samples with batch_size=2 forces a ragged last chunk (2, 2, 1 samples).
        cb = self._make_callback(n_samples=5, predict_batch_size=2)
        result = cb._predict(cb.subsample_x)
        np.testing.assert_allclose(result, cb.subsample_x)

    def test_never_calls_batch_accumulating_predict(self, monkeypatch):
        # model.predict() accumulates every batch's output into one GPU-resident tensor
        # before returning, which is exactly what OOMs on large full-domain subsamples;
        # _predict must go through predict_on_batch instead (see callbacks.py:_predict).
        cb = self._make_callback(n_samples=5, predict_batch_size=2)
        monkeypatch.setattr(
            cb.model,
            "predict",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("model.predict() must not be called")),
        )
        result = cb._predict(cb.subsample_x)
        np.testing.assert_allclose(result, cb.subsample_x)

    def test_predict_batch_size_field_is_required(self):
        with pytest.raises(TypeError):
            fc.TestVisualizationCallback(
                active_day_x=np.zeros((2, 2, 1), dtype=np.float32),
                active_day_y=np.zeros((2, 2, 1), dtype=np.float32),
                active_day_label="active day",
                subsample_x=np.zeros((1, 2, 2, 1), dtype=np.float32),
                subsample_y=np.zeros((1, 2, 2, 1), dtype=np.float32),
                lats=np.array([0.0, 1.0]),
                lons=np.array([0.0, 1.0]),
                front_types=["CF"],
            )


class TestVisualizationCallbackOnEpochEnd:
    """On_epoch_end must not pass an explicit `step` to wandb.log: WandbMetricsLogger's.

    Step is the cumulative training batch count, not the epoch number, so a `step=epoch`
    call is always behind the run's current step and gets silently dropped by wandb
    (see callbacks.py:on_epoch_end).
    """

    def _make_callback(self, monkeypatch, every_n_epochs: int) -> "fc.TestVisualizationCallback":
        # CF is class index 1, so 2 channels is the minimum needed to exercise the
        # class-index slicing in on_epoch_end.
        inputs = fc.tf.keras.Input(shape=(2, 2, 2))
        model = fc.tf.keras.Model(inputs, inputs)  # identity
        cb = fc.TestVisualizationCallback(
            active_day_x=np.zeros((2, 2, 2), dtype=np.float32),
            active_day_y=np.zeros((2, 2, 2), dtype=np.float32),
            active_day_label="active day",
            subsample_x=np.zeros((3, 2, 2, 2), dtype=np.float32),
            subsample_y=np.zeros((3, 2, 2, 2), dtype=np.float32),
            lats=np.array([0.0, 1.0]),
            lons=np.array([0.0, 1.0]),
            front_types=["CF"],
            predict_batch_size=2,
            every_n_epochs=every_n_epochs,
        )
        cb.set_model(model)
        # Plotting (cartopy map + table figure) is unrelated to the wandb step bug and
        # would otherwise drag in real map rendering; substitute cheap bare figures.
        monkeypatch.setattr(fc.plot_module, "plot_test_prediction", lambda **_: fc.plot_module.plt.figure())
        monkeypatch.setattr(fc.plot_module, "plot_performance_diagram_lite", lambda **_: fc.plot_module.plt.figure())
        return cb

    def test_logs_one_payload_with_no_explicit_step(self, monkeypatch):
        cb = self._make_callback(monkeypatch, every_n_epochs=1)
        calls = []
        monkeypatch.setattr(fc.wandb, "log", lambda payload, **kwargs: calls.append((payload, kwargs)))

        cb.on_epoch_end(epoch=0)

        assert len(calls) == 1
        payload, kwargs = calls[0]
        assert "step" not in kwargs
        assert "test/prediction" in payload
        assert any(k.startswith("test/performance_diagram/") for k in payload)

    def test_skips_logging_outside_cadence(self, monkeypatch):
        cb = self._make_callback(monkeypatch, every_n_epochs=10)
        calls = []
        monkeypatch.setattr(fc.wandb, "log", lambda payload, **kwargs: calls.append((payload, kwargs)))

        cb.on_epoch_end(epoch=0)

        assert calls == []


class TestAccumulateLiteStats:
    def test_matches_hand_computed_counts(self):
        # (time=1, lat=2, lon=2, n_fronts=1)
        pred = np.array([[[0.9], [0.1]], [[0.4], [0.6]]], dtype=np.float32).reshape(1, 2, 2, 1)
        truth = np.array([[1, 0], [0, 1]], dtype=np.float32).reshape(1, 2, 2, 1)
        weights = np.ones((2, 2), dtype=np.float32)
        thresholds = np.array([0.5], dtype=np.float32)

        tp, fp, tn, fn = fc.accumulate_lite_stats(pred, truth, weights, thresholds)

        assert tp[0, 0] == pytest.approx(2.0)
        assert fp[0, 0] == pytest.approx(0.0)
        assert tn[0, 0] == pytest.approx(2.0)
        assert fn[0, 0] == pytest.approx(0.0)

    def test_zero_weight_excludes_pixel(self):
        pred = np.array([[[0.9], [0.9]], [[0.9], [0.9]]], dtype=np.float32).reshape(1, 2, 2, 1)
        truth = np.ones((1, 2, 2, 1), dtype=np.float32)
        weights = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        thresholds = np.array([0.5], dtype=np.float32)

        tp, fp, tn, fn = fc.accumulate_lite_stats(pred, truth, weights, thresholds)

        # Only the two weight=1 pixels (both true positives) should count.
        assert tp[0, 0] == pytest.approx(2.0)
        assert fp[0, 0] == pytest.approx(0.0)
        assert tn[0, 0] == pytest.approx(0.0)
        assert fn[0, 0] == pytest.approx(0.0)
