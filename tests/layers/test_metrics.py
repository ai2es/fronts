"""Tests for custom metrics: HSS, FSS, and supporting helpers."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
_metrics = pytest.importorskip("fronts.layers.metrics")
heidke_skill_score = _metrics.heidke_skill_score
fractions_skill_score = _metrics.fractions_skill_score

N_BATCH = 2
N_H = 8
N_W = 8
N_CLASSES = 6


@pytest.fixture
def perfect_pred():
    """y_true and y_pred are identical one-hot arrays."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, N_CLASSES, size=(N_BATCH, N_H, N_W))
    one_hot = (labels[..., np.newaxis] == np.arange(N_CLASSES)).astype(np.float32)
    return one_hot, one_hot.copy()


@pytest.fixture
def front_pred():
    """Realistic scenario: background dominates, front is a small band.

    y_true: class 1 (CF) in rows 3-4 of batch 0; rest is background (class 0).
    y_pred: softmax-like probabilities — CF pixels have p(CF)=0.35, which is below
            the 0.5 hard threshold but clearly above zero. Tests that soft HSS
            (no threshold) responds to this signal while hard threshold=0.5 does not.
    """
    y_true = np.zeros((N_BATCH, N_H, N_W, N_CLASSES), dtype=np.float32)
    y_pred = np.zeros((N_BATCH, N_H, N_W, N_CLASSES), dtype=np.float32)

    y_true[..., 0] = 1.0
    y_pred[..., 0] = 0.90
    y_pred[..., 1] = 0.02
    y_pred[..., 2] = 0.02
    y_pred[..., 3] = 0.02
    y_pred[..., 4] = 0.02
    y_pred[..., 5] = 0.02

    y_true[0, 3:5, :, 0] = 0.0
    y_true[0, 3:5, :, 1] = 1.0

    y_pred[0, 3:5, :, 0] = 0.50
    y_pred[0, 3:5, :, 1] = 0.35
    y_pred[0, 3:5, :, 2] = 0.05
    y_pred[0, 3:5, :, 3] = 0.05
    y_pred[0, 3:5, :, 4] = 0.03
    y_pred[0, 3:5, :, 5] = 0.02

    return y_true, y_pred


class TestHeidkeSkillScorePerfect:
    def test_perfect_no_threshold(self, perfect_pred):
        y_true, y_pred = perfect_pred
        result = heidke_skill_score()(y_true, y_pred).numpy()
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_perfect_with_threshold(self, perfect_pred):
        y_true, y_pred = perfect_pred
        result = heidke_skill_score(threshold=0.5)(y_true, y_pred).numpy()
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_perfect_with_window(self, perfect_pred):
        y_true, y_pred = perfect_pred
        result = heidke_skill_score(threshold=0.5, window_size=(3, 3))(y_true, y_pred).numpy()
        assert result == pytest.approx(1.0, abs=1e-5)


class TestHeidkeSkillScoreSoftMode:
    def test_soft_detects_front_where_hard_threshold_does_not(self, front_pred):
        """Soft HSS (no threshold) should be positive when p(CF)=0.35; hard threshold=0.5 should not fire."""
        y_true, y_pred = front_pred
        class_weights = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        score_soft = heidke_skill_score(class_weights=class_weights)(y_true, y_pred).numpy()
        score_hard = heidke_skill_score(threshold=0.5, class_weights=class_weights)(y_true, y_pred).numpy()
        assert score_soft > 0.0, f"Soft HSS should be positive for a detected front; got {score_soft:.4f}"
        assert score_soft > score_hard, (
            f"Soft HSS ({score_soft:.4f}) should exceed hard-threshold HSS ({score_hard:.4f}) "
            "when softmax front probs sit below 0.5"
        )

    def test_soft_improves_as_predictions_sharpen(self):
        """Soft HSS should increase as the model assigns more probability mass to the correct class."""
        y_true = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_true[..., 0] = 1.0
        y_true[0, 4, 4, 0] = 0.0
        y_true[0, 4, 4, 1] = 1.0

        class_weights = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        hss = heidke_skill_score(class_weights=class_weights)

        def make_pred(cf_prob):
            y_pred = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
            y_pred[..., 0] = 1.0
            y_pred[0, 4, 4, 0] = 1.0 - cf_prob
            y_pred[0, 4, 4, 1] = cf_prob
            return y_pred

        score_low = hss(y_true, make_pred(0.1)).numpy()
        score_mid = hss(y_true, make_pred(0.3)).numpy()
        score_high = hss(y_true, make_pred(0.7)).numpy()

        assert score_low < score_mid < score_high, (
            f"Soft HSS should increase monotonically as CF probability increases: "
            f"{score_low:.4f} < {score_mid:.4f} < {score_high:.4f}"
        )


class TestHeidkeSkillScoreWindow:
    def test_window_helps_near_miss(self):
        """A prediction shifted by 1 pixel should score at least as well with a window."""
        y_true = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_pred = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_true[..., 0] = 1.0
        y_pred[..., 0] = 1.0

        y_true[0, 4, :, 0] = 0.0
        y_true[0, 4, :, 1] = 1.0
        y_pred[0, 5, :, 0] = 0.0
        y_pred[0, 5, :, 1] = 1.0

        class_weights = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        score_exact = heidke_skill_score(threshold=0.5, class_weights=class_weights)(y_true, y_pred).numpy()
        score_window = heidke_skill_score(threshold=0.5, window_size=(3, 3), class_weights=class_weights)(
            y_true, y_pred
        ).numpy()

        assert score_window >= score_exact, (
            f"Window score ({score_window:.4f}) should be >= exact score ({score_exact:.4f}) for a 1-pixel offset"
        )


class TestHeidkeSkillScoreClassWeights:
    def test_background_suppressed(self):
        """With background weight=0, missing a front pixel should reduce HSS below 1."""
        y_true = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_pred = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_true[..., 0] = 1.0
        y_pred[..., 0] = 1.0
        y_true[0, 4, 4, 0] = 0.0
        y_true[0, 4, 4, 2] = 1.0

        result = heidke_skill_score(threshold=0.5, class_weights=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0])(y_true, y_pred).numpy()
        assert result < 1.0, "HSS should be less than 1 when a front pixel is missed"

    def test_equal_weights_matches_unweighted(self, perfect_pred):
        """All-equal non-zero weights should give the same result as no weights for perfect predictions."""
        y_true, y_pred = perfect_pred
        score_no_weights = heidke_skill_score(threshold=0.5)(y_true, y_pred).numpy()
        score_equal_weights = heidke_skill_score(threshold=0.5, class_weights=[1.0] * N_CLASSES)(y_true, y_pred).numpy()
        assert score_no_weights == pytest.approx(score_equal_weights, abs=1e-5)


class TestFSSMetricPerfectPrediction:
    def test_perfect_prediction_is_one(self):
        y = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y[..., 1] = 1.0
        score = fractions_skill_score(mask_size=(3, 3))(y, y).numpy()
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_perfect_prediction_with_weights_is_one(self):
        y = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y[..., 1] = 1.0
        score = fractions_skill_score(mask_size=(3, 3), class_weights=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0])(y, y).numpy()
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_all_background_perfect_prediction_is_one_with_weights(self):
        y = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y[..., 0] = 1.0
        score = fractions_skill_score(mask_size=(3, 3), class_weights=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0])(y, y).numpy()
        assert score == pytest.approx(1.0, abs=1e-5)


class TestFSSMetricWrongPrediction:
    def test_completely_wrong_cf_prediction_penalised(self):
        """CF truth but background prediction should produce FSS < 1."""
        y_true = np.zeros((N_BATCH, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_true[..., 1] = 1.0
        y_pred = np.zeros((N_BATCH, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_pred[..., 0] = 1.0

        class_weights = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        score = fractions_skill_score(mask_size=(3, 3), class_weights=class_weights)(y_true, y_pred).numpy()
        assert score < 0.95, f"Expected low FSS for completely wrong prediction, got {score:.4f}"

    def test_fss_increases_as_predictions_improve(self):
        y_true = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_true[..., 1] = 1.0

        class_weights = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        fss = fractions_skill_score(mask_size=(3, 3), class_weights=class_weights)

        scores = []
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y_pred = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
            y_pred[..., 0] = 1.0 - p
            y_pred[..., 1] = p
            scores.append(float(fss(y_true, y_pred).numpy()))

        assert scores == sorted(scores), f"FSS should increase monotonically as CF probability increases: {scores}"
