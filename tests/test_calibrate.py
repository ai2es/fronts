"""Tests for temperature scaling calibration: extract_logit_model, TemperatureScaledModel, fit_temperature."""

import numpy as np
import pytest

try:
    import tensorflow as tf

    from fronts import model as fronts_model
    from fronts.calibrate import extract_logit_model, fit_temperature
    from fronts.model import TemperatureScaledModel

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TF_AVAILABLE, reason="TensorFlow not available")

N_CLASSES = 6
_INPUT_SHAPE = (None, None, 5)


@pytest.fixture(scope="module")
def small_model() -> "tf.keras.Model":
    """Tiny UNet3Plus with deep supervision for testing."""
    return fronts_model.UNet3Plus(
        input_shape=_INPUT_SHAPE,
        num_classes=N_CLASSES,
        pool_size=(2, 2),
        upsample_size=(2, 2),
        levels=3,
        filter_num=[8, 16, 32],
        deep_supervision=True,
        output_activation="softmax",
    ).build()


@pytest.fixture(scope="module")
def logit_model(small_model) -> "tf.keras.Model":
    return extract_logit_model(small_model)


@pytest.fixture
def random_input() -> "np.ndarray":
    rng = np.random.default_rng(0)
    return rng.standard_normal((2, 16, 16, 5)).astype(np.float32)


class TestExtractLogitModel:
    def test_output_count_matches_original(self, small_model, logit_model):
        assert len(logit_model.outputs) == len(small_model.outputs)

    def test_logits_do_not_sum_to_one(self, logit_model, random_input):
        logits = logit_model(random_input, training=False)
        primary = logits[0] if isinstance(logits, (list, tuple)) else logits
        class_sums = tf.reduce_sum(primary, axis=-1).numpy()
        assert not np.allclose(class_sums, 1.0, atol=1e-3), "Logits should not sum to 1 across class dim"

    def test_corresponding_softmax_sums_to_one(self, logit_model, random_input):
        logits = logit_model(random_input, training=False)
        primary = logits[0] if isinstance(logits, (list, tuple)) else logits
        probs = tf.nn.softmax(primary, axis=-1).numpy()
        class_sums = probs.sum(axis=-1)
        np.testing.assert_allclose(class_sums, np.ones_like(class_sums), atol=1e-5)

    def test_input_shape_preserved(self, small_model, logit_model):
        assert logit_model.input_shape == small_model.input_shape


class TestTemperatureScaledModel:
    def test_outputs_sum_to_one(self, logit_model, random_input):
        calibrated = TemperatureScaledModel(logit_model=logit_model, temperature=1.0)
        outputs = calibrated(random_input, training=False)
        primary = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        class_sums = primary.numpy().sum(axis=-1)
        np.testing.assert_allclose(class_sums, np.ones_like(class_sums), atol=1e-5)

    def test_temperature_one_matches_softmax_of_logits(self, logit_model, random_input):
        calibrated = TemperatureScaledModel(logit_model=logit_model, temperature=1.0)
        logits = logit_model(random_input, training=False)
        primary_logits = logits[0] if isinstance(logits, (list, tuple)) else logits
        expected = tf.nn.softmax(primary_logits).numpy()

        outputs = calibrated(random_input, training=False)
        primary_out = outputs[0].numpy() if isinstance(outputs, (list, tuple)) else outputs.numpy()

        np.testing.assert_allclose(primary_out, expected, atol=1e-5)

    def test_low_temperature_sharpens_distribution(self, logit_model, random_input):
        """T < 1 should reduce entropy (make distribution more peaked)."""
        out_t1 = TemperatureScaledModel(logit_model=logit_model, temperature=1.0)(random_input)
        out_low = TemperatureScaledModel(logit_model=logit_model, temperature=0.1)(random_input)

        primary_t1 = out_t1[0].numpy() if isinstance(out_t1, (list, tuple)) else out_t1.numpy()
        primary_low = out_low[0].numpy() if isinstance(out_low, (list, tuple)) else out_low.numpy()

        entropy_t1 = -np.sum(primary_t1 * np.log(primary_t1 + 1e-12), axis=-1).mean()
        entropy_low = -np.sum(primary_low * np.log(primary_low + 1e-12), axis=-1).mean()

        assert entropy_low < entropy_t1, (
            f"Low T should reduce entropy: T=0.1 gave {entropy_low:.4f}, T=1 gave {entropy_t1:.4f}"
        )

    def test_argmax_preserved_across_temperatures(self, logit_model, random_input):
        """The predicted class (argmax) should not change when temperature changes."""
        out_t1 = TemperatureScaledModel(logit_model=logit_model, temperature=1.0)(random_input)
        out_low = TemperatureScaledModel(logit_model=logit_model, temperature=0.5)(random_input)

        p1 = out_t1[0].numpy() if isinstance(out_t1, (list, tuple)) else out_t1.numpy()
        p2 = out_low[0].numpy() if isinstance(out_low, (list, tuple)) else out_low.numpy()

        np.testing.assert_array_equal(p1.argmax(axis=-1), p2.argmax(axis=-1))

    def test_output_count_matches_logit_model(self, logit_model, random_input):
        calibrated = TemperatureScaledModel(logit_model=logit_model, temperature=1.0)
        outputs = calibrated(random_input, training=False)
        assert len(outputs) == len(logit_model.outputs)


class TestFitTemperature:
    def test_underconfident_model_gets_t_less_than_one(self, logit_model):
        """A logit model whose logits are scaled down (underconfident) should get T < 1."""
        rng = np.random.default_rng(1)
        n_samples = 20
        batch_size = 4
        h, w = 8, 8

        class _SyntheticDataset:
            """Mimics the FrontsPyDataset interface for fit_temperature."""

            def __init__(self, inputs, targets):
                self._inputs = inputs
                self._targets = targets

            def __len__(self):
                return len(self._inputs)

            def __getitem__(self, idx):
                return self._inputs[idx], self._targets[idx]

        inputs = [rng.standard_normal((batch_size, h, w, 5)).astype(np.float32) for _ in range(n_samples)]

        raw_logits = [logit_model(x, training=False) for x in inputs]
        primary_raw = [logit[0].numpy() if isinstance(logit, (list, tuple)) else logit.numpy() for logit in raw_logits]

        targets = []
        for logit in primary_raw:
            cls = logit.argmax(axis=-1)
            one_hot = np.eye(N_CLASSES, dtype=np.float32)[cls]
            targets.append(one_hot)

        squeezed_logits = [logit * 0.1 for logit in primary_raw]

        class _SqueezedLogitModel:
            def __call__(self, x, training=False):
                for i, inp in enumerate(inputs):
                    if np.allclose(x, inp):
                        return [squeezed_logits[i]]
                raise ValueError("Input not found in synthetic dataset")

            @property
            def outputs(self):
                return [None]

        dataset = _SyntheticDataset(inputs, targets)

        t_opt = fit_temperature(_SqueezedLogitModel(), dataset, max_pixels=10_000)

        assert t_opt < 1.0, f"Underconfident model (logits * 0.1) should produce T < 1; got T={t_opt:.4f}"
        assert t_opt > 0.01, f"T should be within valid bounds; got T={t_opt:.4f}"
