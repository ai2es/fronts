"""Tests for UNet3Plus's min-max input normalization layer."""

import numpy as np
import pytest

try:
    import tensorflow as tf

    from fronts import model as fronts_model

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TF_AVAILABLE, reason="TensorFlow not available")

N_CHANNELS = 4
_INPUT_SHAPE = (None, None, N_CHANNELS)


def _build_model(
    normalization_stat_a=None, normalization_stat_b=None, normalization_method="minmax"
) -> "tf.keras.Model":
    return fronts_model.UNet3Plus(
        input_shape=_INPUT_SHAPE,
        num_classes=6,
        pool_size=(2, 2),
        upsample_size=(2, 2),
        levels=3,
        filter_num=[8, 16, 32],
        deep_supervision=False,
        output_activation="softmax",
        normalization_method=normalization_method,
        normalization_stat_a=normalization_stat_a,
        normalization_stat_b=normalization_stat_b,
    ).build()


class TestMinMaxNormalization:
    def test_no_stats_skips_normalization_layer(self):
        built = _build_model()
        assert not any(layer.name == "input_normalization" for layer in built.layers)

    def test_min_maps_to_zero_and_max_to_one(self):
        min_val = np.array([0.0, -10.0, 100.0, 1e-6], dtype=np.float32)
        max_val = np.array([1.0, 10.0, 200.0, 2e-6], dtype=np.float32)
        built = _build_model(min_val, max_val)

        norm_layer = built.get_layer("input_normalization")
        at_min = norm_layer(min_val.reshape(1, 1, 1, N_CHANNELS)).numpy().reshape(-1)
        at_max = norm_layer(max_val.reshape(1, 1, 1, N_CHANNELS)).numpy().reshape(-1)
        np.testing.assert_allclose(at_min, 0.0, atol=1e-5)
        np.testing.assert_allclose(at_max, 1.0, atol=1e-5)

    def test_midpoint_maps_to_half(self):
        min_val = np.array([0.0, -10.0, 100.0, 1e-6], dtype=np.float32)
        max_val = np.array([1.0, 10.0, 200.0, 2e-6], dtype=np.float32)
        built = _build_model(min_val, max_val)

        norm_layer = built.get_layer("input_normalization")
        midpoint = (min_val + max_val) / 2
        out = norm_layer(midpoint.reshape(1, 1, 1, N_CHANNELS)).numpy().reshape(-1)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_tiny_variance_channel_does_not_overflow_or_explode(self):
        """Test tiny variance to make sure overflow.

        A near-zero-variance channel (like specific_humidity/potential_vorticity in
        native units) must not produce huge normalized values the way z-score would.
        """
        min_val = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        max_val = np.array([1.0, 1.0, 1.0, 1e-12], dtype=np.float32)
        built = _build_model(min_val, max_val)

        norm_layer = built.get_layer("input_normalization")
        sample = np.array([0.5, 0.5, 0.5, 5e-13], dtype=np.float32)
        out = norm_layer(sample.reshape(1, 1, 1, N_CHANNELS)).numpy().reshape(-1)
        assert np.isfinite(out).all()
        assert np.abs(out).max() < 10.0

    def test_zero_range_channel_does_not_produce_nan(self):
        """A perfectly constant channel (max == min) must not divide by zero."""
        min_val = np.array([0.0, 5.0, -3.0, 0.0], dtype=np.float32)
        max_val = np.array([1.0, 5.0, -3.0, 1.0], dtype=np.float32)
        built = _build_model(min_val, max_val)

        norm_layer = built.get_layer("input_normalization")
        sample = np.array([0.5, 5.0, -3.0, 0.5], dtype=np.float32)
        out = norm_layer(sample.reshape(1, 1, 1, N_CHANNELS)).numpy().reshape(-1)
        assert np.isfinite(out).all()

    def test_out_of_range_sample_extrapolates_linearly_without_clipping(self):
        """Matches the AIES reference's (value - min) / (max - min), which does not clip."""
        min_val = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        max_val = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        built = _build_model(min_val, max_val)

        norm_layer = built.get_layer("input_normalization")
        sample = np.array([2.0, -1.0, 0.5, 1.5], dtype=np.float32)
        out = norm_layer(sample.reshape(1, 1, 1, N_CHANNELS)).numpy().reshape(-1)
        np.testing.assert_allclose(out, sample, atol=1e-5)

    def test_forward_pass_end_to_end_with_extreme_channel_range(self):
        """Regression check for the fp16 overflow this replaces.

        A physically tiny-scale channel (like potential_vorticity, range ~1e-12) must
        not blow up the full model's forward pass, unlike the old z-score Normalization layer.
        """
        min_val = np.array([0.0, -50.0, 900.0, -1e-12], dtype=np.float32)
        max_val = np.array([1.0, 50.0, 1100.0, 1e-12], dtype=np.float32)
        built = _build_model(min_val, max_val)

        rng = np.random.default_rng(0)
        x = rng.uniform(low=min_val, high=max_val, size=(2, 16, 16, N_CHANNELS)).astype(np.float32)
        out = built(x, training=False)
        out_np = out.numpy() if not isinstance(out, (list, tuple)) else out[0].numpy()
        assert np.isfinite(out_np).all()

    def test_uses_rescaling_layer(self):
        built = _build_model(np.zeros(N_CHANNELS, np.float32), np.ones(N_CHANNELS, np.float32))
        assert isinstance(built.get_layer("input_normalization"), tf.keras.layers.Rescaling)


class TestStandardizationNormalization:
    def test_no_stats_skips_normalization_layer(self):
        built = _build_model(normalization_method="standardization")
        assert not any(layer.name == "input_normalization" for layer in built.layers)

    def test_uses_normalization_layer(self):
        mean = np.zeros(N_CHANNELS, np.float32)
        variance = np.ones(N_CHANNELS, np.float32)
        built = _build_model(mean, variance, normalization_method="standardization")
        assert isinstance(built.get_layer("input_normalization"), tf.keras.layers.Normalization)

    def test_mean_maps_to_zero(self):
        mean = np.array([0.0, -10.0, 100.0, 1e-6], dtype=np.float32)
        variance = np.array([1.0, 4.0, 25.0, 1e-12], dtype=np.float32)
        built = _build_model(mean, variance, normalization_method="standardization")

        norm_layer = built.get_layer("input_normalization")
        out = norm_layer(mean.reshape(1, 1, 1, N_CHANNELS)).numpy().reshape(-1)
        np.testing.assert_allclose(out, 0.0, atol=1e-4)

    def test_forward_pass_finite(self):
        mean = np.array([0.0, -50.0, 1000.0, 0.0], dtype=np.float32)
        variance = np.array([1.0, 100.0, 400.0, 1.0], dtype=np.float32)
        built = _build_model(mean, variance, normalization_method="standardization")

        rng = np.random.default_rng(0)
        x = rng.standard_normal((2, 16, 16, N_CHANNELS)).astype(np.float32)
        out = built(x, training=False)
        out_np = out.numpy() if not isinstance(out, (list, tuple)) else out[0].numpy()
        assert np.isfinite(out_np).all()


N_LEVELS_3D = 4
N_VARS_3D = 3
_UNET_LEVELS_3D = 3


def _build_3d_model(
    normalization_stat_a=None, normalization_stat_b=None, normalization_method="minmax"
) -> "tf.keras.Model":
    return fronts_model.UNet3Plus(
        input_shape=(None, None, N_LEVELS_3D, N_VARS_3D),
        num_classes=6,
        pool_size=(2, 2, 1),
        upsample_size=(2, 2, 1),
        levels=_UNET_LEVELS_3D,
        filter_num=[4, 8, 16],
        kernel_size=3,
        squeeze_axes=3,
        first_encoder_connections=True,
        deep_supervision=True,
        batch_normalization=True,
        activation="gelu",
        output_activation="softmax",
        modules_per_node=1,
        normalization_method=normalization_method,
        normalization_stat_a=normalization_stat_a,
        normalization_stat_b=normalization_stat_b,
    ).build()


@pytest.fixture(scope="module")
def built_3d_model() -> "tf.keras.Model":
    return _build_3d_model()


class Test3DBuild:
    def test_model_name_marks_3d(self, built_3d_model):
        assert built_3d_model.name == "unet_3plus_3D"

    def test_all_convolutions_are_3d(self, built_3d_model):
        conv_classes = {type(layer).__name__ for layer in built_3d_model.layers if "Conv" in type(layer).__name__}
        assert "Conv3D" in conv_classes
        assert "Conv2D" not in conv_classes

    def test_one_output_per_supervised_level(self, built_3d_model):
        assert len(built_3d_model.outputs) == _UNET_LEVELS_3D

    def test_outputs_are_2d_class_maps(self, built_3d_model):
        """The squeeze collapse must drop the level axis: (batch, lat, lon, classes)."""
        for out in built_3d_model.outputs:
            assert len(out.shape) == 4
            assert out.shape[-1] == 6


class Test3DForwardPass:
    def test_forward_pass_shapes_and_softmax(self, built_3d_model):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2, 16, 16, N_LEVELS_3D, N_VARS_3D)).astype(np.float32)
        outs = built_3d_model(x, training=False)
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        for out in outs:
            values = out.numpy()
            assert values.shape == (2, 16, 16, 6)
            assert np.isfinite(values).all()
            np.testing.assert_allclose(values.sum(axis=-1), 1.0, atol=1e-4)

    def test_dynamic_spatial_dims(self, built_3d_model):
        """The squeeze path must handle spatial sizes unknown at build time."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal((1, 24, 32, N_LEVELS_3D, N_VARS_3D)).astype(np.float32)
        outs = built_3d_model(x, training=False)
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        assert outs[0].shape == (1, 24, 32, 6)


class Test3DMinMaxNormalization:
    def test_level_variable_stats_map_min_to_zero_and_max_to_one(self):
        rng = np.random.default_rng(2)
        min_val = rng.standard_normal((N_LEVELS_3D, N_VARS_3D)).astype(np.float32)
        max_val = min_val + (np.abs(rng.standard_normal((N_LEVELS_3D, N_VARS_3D))) + 0.1).astype(np.float32)
        built = _build_3d_model(min_val, max_val)

        norm_layer = built.get_layer("input_normalization")
        at_min = norm_layer(min_val.reshape(1, 1, 1, N_LEVELS_3D, N_VARS_3D)).numpy().reshape(-1)
        at_max = norm_layer(max_val.reshape(1, 1, 1, N_LEVELS_3D, N_VARS_3D)).numpy().reshape(-1)
        np.testing.assert_allclose(at_min, 0.0, atol=1e-5)
        np.testing.assert_allclose(at_max, 1.0, atol=1e-5)

    def test_uses_rescaling_layer(self):
        min_val = np.zeros((N_LEVELS_3D, N_VARS_3D), np.float32)
        max_val = np.ones((N_LEVELS_3D, N_VARS_3D), np.float32)
        built = _build_3d_model(min_val, max_val)
        assert isinstance(built.get_layer("input_normalization"), tf.keras.layers.Rescaling)


class Test3DStandardizationNormalization:
    def test_uses_normalization_layer(self):
        mean = np.zeros((N_LEVELS_3D, N_VARS_3D), np.float32)
        variance = np.ones((N_LEVELS_3D, N_VARS_3D), np.float32)
        built = _build_3d_model(mean, variance, normalization_method="standardization")
        assert isinstance(built.get_layer("input_normalization"), tf.keras.layers.Normalization)

    def test_forward_pass_finite(self):
        rng = np.random.default_rng(3)
        mean = rng.standard_normal((N_LEVELS_3D, N_VARS_3D)).astype(np.float32)
        variance = (np.abs(rng.standard_normal((N_LEVELS_3D, N_VARS_3D))) + 0.1).astype(np.float32)
        built = _build_3d_model(mean, variance, normalization_method="standardization")

        x = rng.standard_normal((2, 16, 16, N_LEVELS_3D, N_VARS_3D)).astype(np.float32)
        outs = built(x, training=False)
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        for out in outs:
            assert np.isfinite(out.numpy()).all()


class Test3DSerialization:
    def test_save_load_round_trip(self, built_3d_model, tmp_path):
        """The ops.squeeze collapse must survive .keras serialization — checkpoints get reloaded."""
        rng = np.random.default_rng(4)
        x = rng.standard_normal((1, 16, 16, N_LEVELS_3D, N_VARS_3D)).astype(np.float32)
        original = built_3d_model(x, training=False)
        original = original if isinstance(original, (list, tuple)) else [original]

        path = str(tmp_path / "model_3d.keras")
        built_3d_model.save(path)
        reloaded = tf.keras.models.load_model(path, compile=False)
        again = reloaded(x, training=False)
        again = again if isinstance(again, (list, tuple)) else [again]

        for a, b in zip(original, again, strict=True):
            np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-5)
