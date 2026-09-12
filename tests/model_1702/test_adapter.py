"""Tests for the model_1702 and 6-class model adapters."""

import numpy as np
import pytest
import tensorflow as tf

from fronts.model_1702 import adapter, normalization

N_LAT = 4
N_LON = 6


class OrientationProbeModel:
    """Fake legacy model: returns the first six variables at the surface level, untouched.

    Mimics model_1702's interface — expects (batch, lon, lat, level, variable), returns a list
    of heads shaped (batch, lon, lat, 6) — so the adapter's transposes and flips are observable
    in the output values.
    """

    def __call__(self, x, training=False):
        return [x[:, :, :, 0, :6]]


@pytest.fixture
def raw_batch():
    rng = np.random.default_rng(5)
    return rng.random((2, N_LAT, N_LON, 5, 10)).astype(np.float32)


class TestFrontFinder1702Adapter:
    def test_output_contract(self, raw_batch):
        wrapped = adapter.FrontFinder1702Adapter(OrientationProbeModel(), lat_ascending=False)
        out = np.asarray(wrapped(tf.constant(raw_batch)))
        assert out.shape == (2, N_LAT, N_LON, 6)

    def test_normalization_and_orientation_round_trip(self, raw_batch):
        wrapped = adapter.FrontFinder1702Adapter(OrientationProbeModel(), lat_ascending=False)
        out = np.asarray(wrapped(tf.constant(raw_batch)))
        expected = adapter.normalize_volume(raw_batch)[:, :, :, 0, :6]
        np.testing.assert_allclose(out[..., :6], expected, rtol=1e-6)

    def test_lat_ascending_flip_round_trips(self, raw_batch):
        descending = adapter.FrontFinder1702Adapter(OrientationProbeModel(), lat_ascending=False)
        ascending = adapter.FrontFinder1702Adapter(OrientationProbeModel(), lat_ascending=True)
        out_descending = np.asarray(descending(tf.constant(raw_batch)))
        flipped_batch = raw_batch[:, ::-1].copy()
        out_ascending = np.asarray(ascending(tf.constant(flipped_batch)))
        np.testing.assert_allclose(out_ascending, out_descending[:, ::-1], rtol=1e-6)

    def test_non_finite_inputs_normalize_to_zero(self, raw_batch):
        raw_batch[0, 0, 0, 0, 0] = np.nan
        raw_batch[1, 2, 3, 1, 4] = np.inf
        wrapped = adapter.FrontFinder1702Adapter(OrientationProbeModel(), lat_ascending=False)
        out = np.asarray(wrapped(tf.constant(raw_batch)))
        assert out[0, 0, 0, 0] == 0.0
        assert np.isfinite(out).all()

    def test_works_inside_tf_function(self, raw_batch):
        wrapped = adapter.FrontFinder1702Adapter(OrientationProbeModel(), lat_ascending=False)

        @tf.function
        def eval_step(x):
            return wrapped(x, training=False)

        out = np.asarray(eval_step(tf.constant(raw_batch)))
        assert out.shape == (2, N_LAT, N_LON, 6)


class TestClassPaddingAdapter:
    def test_six_class_output_passes_through(self, raw_batch):
        def six_class_model(x, training=False):
            return [tf.ones((tf.shape(x)[0], N_LAT, N_LON, 6)) / 6.0]

        wrapped = adapter.ClassPaddingAdapter(six_class_model)
        out = np.asarray(wrapped(tf.constant(raw_batch)))
        assert out.shape == (2, N_LAT, N_LON, 6)
        np.testing.assert_allclose(out, 1.0 / 6.0)

    def test_too_many_classes_raises(self, raw_batch):
        def ten_class_model(x, training=False):
            return tf.ones((tf.shape(x)[0], N_LAT, N_LON, 10))

        wrapped = adapter.ClassPaddingAdapter(ten_class_model)
        with pytest.raises(ValueError, match="more than"):
            wrapped(tf.constant(raw_batch))


def test_normalize_volume_matches_legacy_formula():
    x = np.full((1, 1, 1, 5, 10), 250.0, dtype=np.float32)
    normalized = adapter.normalize_volume(x)
    maxs, mins = normalization.norm_min_max_tables()
    np.testing.assert_allclose(normalized[0, 0, 0], (250.0 - mins) / (maxs - mins), rtol=1e-6)
