"""Tests for custom loss functions: BSS, CSI, FSS, POD."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
fractions_skill_score = pytest.importorskip("fronts.layers.losses").fractions_skill_score

N_BATCH = 2
N_H = 8
N_W = 8
N_CLASSES = 6


@pytest.fixture
def all_cf_batch() -> tuple[np.ndarray, np.ndarray]:
    """Truth is CF everywhere; prediction is background everywhere (completely wrong)."""
    y_true = np.zeros((N_BATCH, N_H, N_W, N_CLASSES), dtype=np.float32)
    y_true[..., 1] = 1.0
    y_pred = np.zeros((N_BATCH, N_H, N_W, N_CLASSES), dtype=np.float32)
    y_pred[..., 0] = 1.0
    return y_true, y_pred


@pytest.fixture
def all_bg_batch() -> tuple[np.ndarray, np.ndarray]:
    """Truth is background everywhere; prediction is CF everywhere (completely wrong)."""
    y_true = np.zeros((N_BATCH, N_H, N_W, N_CLASSES), dtype=np.float32)
    y_true[..., 0] = 1.0
    y_pred = np.zeros((N_BATCH, N_H, N_W, N_CLASSES), dtype=np.float32)
    y_pred[..., 1] = 1.0
    return y_true, y_pred


class TestFSSLossPerfectPrediction:
    def test_loss_is_zero_for_perfect_prediction_no_weights(self):
        y = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y[..., 1] = 1.0
        loss = fractions_skill_score(mask_size=(3, 3))(y, y).numpy()
        assert loss == pytest.approx(0.0, abs=1e-5)

    def test_loss_is_zero_for_perfect_prediction_with_weights(self):
        y = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y[..., 1] = 1.0
        class_weights = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        loss = fractions_skill_score(mask_size=(3, 3), class_weights=class_weights)(y, y).numpy()
        assert loss == pytest.approx(0.0, abs=1e-5)

    def test_loss_is_zero_for_all_background_perfect_prediction_with_weights(self):
        y = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y[..., 0] = 1.0
        class_weights = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        loss = fractions_skill_score(mask_size=(3, 3), class_weights=class_weights)(y, y).numpy()
        assert loss == pytest.approx(0.0, abs=1e-5)


class TestFSSLossWrongPrediction:
    def test_wrong_cf_prediction_penalised(self, all_cf_batch):
        """With background weight=0 and wrong CF prediction, loss should be significant."""
        y_true, y_pred = all_cf_batch
        class_weights = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        loss = fractions_skill_score(mask_size=(3, 3), class_weights=class_weights)(y_true, y_pred).numpy().mean()
        assert loss > 0.05, f"Expected significant loss for wrong CF prediction, got {loss:.4f}"

    def test_wrong_cf_prediction_is_symmetric(self, all_cf_batch, all_bg_batch):
        """FSS of CF-truth/BG-pred should equal FSS of BG-truth/CF-pred (symmetric MSE)."""
        class_weights = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        loss_fn = fractions_skill_score(mask_size=(3, 3), class_weights=class_weights)
        loss_a = loss_fn(*all_cf_batch).numpy().mean()
        loss_b = loss_fn(*all_bg_batch).numpy().mean()
        assert loss_a == pytest.approx(loss_b, abs=1e-5)

    def test_loss_increases_as_predictions_worsen(self):
        """Loss should increase monotonically as CF prediction probability drops from 1 to 0."""
        y_true = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_true[..., 1] = 1.0

        class_weights = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        loss_fn = fractions_skill_score(mask_size=(3, 3), class_weights=class_weights)

        cf_probs = [1.0, 0.75, 0.5, 0.25, 0.0]
        losses = []
        for p in cf_probs:
            y_pred = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
            y_pred[..., 0] = 1.0 - p
            y_pred[..., 1] = p
            losses.append(loss_fn(y_true, y_pred).numpy().mean())

        assert losses == sorted(losses), f"Loss should be monotonically non-decreasing: {losses}"


class TestFSSLossClassWeights:
    def test_zero_weight_class_excluded_from_loss(self):
        """With background_weight=0, background-only errors should not change loss vs no-error case."""
        y_true_cf = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_true_cf[..., 1] = 1.0

        class_weights = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        loss_fn = fractions_skill_score(mask_size=(3, 3), class_weights=class_weights)

        loss_perfect = loss_fn(y_true_cf, y_true_cf).numpy().mean()
        assert loss_perfect == pytest.approx(0.0, abs=1e-5)

        loss_nonzero = loss_fn(y_true_cf, np.zeros_like(y_true_cf)).numpy().mean()
        assert loss_nonzero > 0.0, "Loss should be non-zero when CF is predicted as all-zero"

    def test_class_weights_affect_relative_contribution(self):
        """Higher weight on a class should increase its contribution to the total loss."""
        y_true = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_true[..., 1] = 1.0
        y_pred = np.zeros_like(y_true)
        y_pred[..., 0] = 1.0

        loss_equal_weights = fractions_skill_score(
            mask_size=(3, 3), class_weights=[1.0] * N_CLASSES
        )(y_true, y_pred).numpy()

        loss_cf_only = fractions_skill_score(
            mask_size=(3, 3), class_weights=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        )(y_true, y_pred).numpy()

        assert loss_cf_only >= loss_equal_weights - 1e-5, (
            f"CF-only weights should produce >= loss vs equal weights when only CF differs: "
            f"cf_only={loss_cf_only:.4f}, equal={loss_equal_weights:.4f}"
        )


class TestFSSLossBackgroundSupervision:
    def test_unweighted_loss_penalises_background_errors(self):
        """With class_weights=None the background channel is supervised, anchoring softmax mass.

        The reference FrontFinder model trained with no loss class weights; a background-only error
        must register in the unweighted loss but vanish under [0, 1, 1, 1, 1, 1] weights.
        """
        y_true = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_true[..., 0] = 1.0

        y_pred = np.zeros_like(y_true)
        y_pred[..., 0] = 0.5

        loss_unweighted = float(fractions_skill_score(mask_size=(3, 3))(y_true, y_pred).numpy().mean())
        loss_bg_zeroed = float(
            fractions_skill_score(mask_size=(3, 3), class_weights=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0])(
                y_true, y_pred
            ).numpy().mean()
        )

        assert loss_unweighted > 0.0
        assert loss_bg_zeroed == pytest.approx(0.0, abs=1e-5)
