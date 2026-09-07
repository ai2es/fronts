"""Tests for custom metrics: HSS, FSS, and supporting helpers."""

from typing import ClassVar

import numpy as np
import pytest

from fronts import constants

tf = pytest.importorskip("tensorflow")
_metrics = pytest.importorskip("fronts.layers.metrics")
heidke_skill_score = _metrics.heidke_skill_score
fractions_skill_score = _metrics.fractions_skill_score
PerClassContingencyMetric = _metrics.PerClassContingencyMetric
per_front_type_metrics = _metrics.per_front_type_metrics

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


CF_CLASS_INDEX = constants.FRONT_TYPE_CLASS_INDEX["CF"]


def _one_class_tensor(true_values, pred_values, class_index=CF_CLASS_INDEX, n_classes=N_CLASSES):
    """Build (1, 1, len(true_values), n_classes) y_true/y_pred tensors with only one class set.

    All classes other than ``class_index`` are left at zero, which is irrelevant to
    ``PerClassContingencyMetric`` since it gathers only ``class_index`` before accumulating.
    """
    n_pixels = len(true_values)
    y_true = np.zeros((1, 1, n_pixels, n_classes), dtype=np.float32)
    y_pred = np.zeros((1, 1, n_pixels, n_classes), dtype=np.float32)
    y_true[0, 0, :, class_index] = true_values
    y_pred[0, 0, :, class_index] = pred_values
    return y_true, y_pred


class TestPerClassContingencyMetricHandComputed:
    """Values below were computed by hand from the contingency table.

    true=[1, 0, 1, 0], pred=[0.9, 0.6, 0.3, 0.1].
    Soft: TP=1.2, FP=0.7, FN=0.8, TN=1.3 -> CSI=1.2/2.7, POD=1.2/2.0, HSS=2.0/8.0.
    Hard (threshold=0.5): pred binarizes to [1, 1, 0, 0] -> TP=1, FP=1, FN=1, TN=1
        -> CSI=1/3, POD=0.5, HSS=0.0.
    """

    TRUE_VALUES: ClassVar[list[float]] = [1.0, 0.0, 1.0, 0.0]
    PRED_VALUES: ClassVar[list[float]] = [0.9, 0.6, 0.3, 0.1]

    @pytest.mark.parametrize(
        "statistic,threshold,expected",
        [
            ("csi", None, 1.2 / 2.7),
            ("pod", None, 0.6),
            ("hss", None, 0.25),
            ("csi", 0.5, 1 / 3),
            ("pod", 0.5, 0.5),
            ("hss", 0.5, 0.0),
        ],
    )
    def test_hand_computed_value(self, statistic, threshold, expected):
        y_true, y_pred = _one_class_tensor(self.TRUE_VALUES, self.PRED_VALUES)
        metric = PerClassContingencyMetric(class_index=CF_CLASS_INDEX, statistic=statistic, threshold=threshold)
        metric.update_state(y_true, y_pred)
        assert metric.result().numpy() == pytest.approx(expected, abs=1e-5)


class TestPerClassContingencyMetricPerfectPrediction:
    @pytest.fixture
    def perfect_pred_all_classes(self):
        """One-hot labels covering every class at least once, predicted perfectly."""
        labels = (np.arange(N_BATCH * N_H * N_W) % N_CLASSES).reshape(N_BATCH, N_H, N_W)
        one_hot = (labels[..., np.newaxis] == np.arange(N_CLASSES)).astype(np.float32)
        return one_hot, one_hot.copy()

    @pytest.mark.parametrize("statistic", ["hss", "csi", "pod"])
    @pytest.mark.parametrize("threshold", [None, 0.5])
    def test_perfect_prediction_is_one_for_every_front_type(self, perfect_pred_all_classes, statistic, threshold):
        y_true, y_pred = perfect_pred_all_classes
        for front_type, class_index in constants.FRONT_TYPE_CLASS_INDEX.items():
            metric = PerClassContingencyMetric(class_index=class_index, statistic=statistic, threshold=threshold)
            metric.update_state(y_true, y_pred)
            assert metric.result().numpy() == pytest.approx(1.0, abs=1e-5), (
                f"{statistic} with threshold={threshold} should be 1.0 for front type {front_type}"
            )


class TestPerClassContingencyMetricBias:
    """The regression test for the stateful design: see task-2 brief 'Why stateful'."""

    @pytest.mark.parametrize("statistic", ["hss", "csi", "pod"])
    def test_accumulated_across_empty_and_perfect_batch_is_one_not_half(self, statistic):
        # Batch 1: no CF pixels at all, background predicted everywhere.
        y_true_1 = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_pred_1 = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_true_1[..., 0] = 1.0
        y_pred_1[..., 0] = 1.0

        # Batch 2: CF pixels predicted perfectly.
        y_true_2 = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_pred_2 = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_true_2[..., 0] = 1.0
        y_pred_2[..., 0] = 1.0
        y_true_2[0, 3:5, :, 0] = 0.0
        y_true_2[0, 3:5, :, 1] = 1.0
        y_pred_2[0, 3:5, :, 0] = 0.0
        y_pred_2[0, 3:5, :, 1] = 1.0

        metric = PerClassContingencyMetric(class_index=CF_CLASS_INDEX, statistic=statistic, threshold=0.5)
        metric.update_state(y_true_1, y_pred_1)
        metric.update_state(y_true_2, y_pred_2)

        result = metric.result().numpy()
        assert result == pytest.approx(1.0, abs=1e-5), (
            f"Accumulated {statistic} for CF across an empty batch then a perfect batch must be 1.0, got {result}. "
            "A batch-mean implementation would give 0.5 here (0.0 from the empty batch, 1.0 from the perfect one)."
        )

    def test_bias_test_genuinely_discriminates_from_batch_mean(self):
        """Prove the two batches above actually distinguish stateful accumulation from batch-mean.

        Computes the naive batch-mean equivalent directly using divide_no_nan semantics, without
        going through PerClassContingencyMetric, to show it yields 0.5 for CF's CSI -- the failure
        mode the stateful design in this module avoids.
        """

        def batch_csi(y_true, y_pred, class_index, threshold):
            y_true_class = y_true[..., class_index]
            y_pred_class = y_pred[..., class_index]
            y_pred_class = np.where(y_pred_class >= threshold, 1.0, 0.0)
            tp = np.sum(y_true_class * y_pred_class)
            fp = np.sum((1 - y_true_class) * y_pred_class)
            fn = np.sum(y_true_class * (1 - y_pred_class))
            denom = tp + fp + fn
            return 0.0 if denom == 0 else tp / denom

        y_true_1 = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_pred_1 = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_true_1[..., 0] = 1.0
        y_pred_1[..., 0] = 1.0

        y_true_2 = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_pred_2 = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_true_2[..., 0] = 1.0
        y_pred_2[..., 0] = 1.0
        y_true_2[0, 3:5, :, 0] = 0.0
        y_true_2[0, 3:5, :, 1] = 1.0
        y_pred_2[0, 3:5, :, 0] = 0.0
        y_pred_2[0, 3:5, :, 1] = 1.0

        batch_mean_csi = np.mean(
            [
                batch_csi(y_true_1, y_pred_1, CF_CLASS_INDEX, 0.5),
                batch_csi(y_true_2, y_pred_2, CF_CLASS_INDEX, 0.5),
            ]
        )
        assert batch_mean_csi == pytest.approx(0.5, abs=1e-9)

        metric = PerClassContingencyMetric(class_index=CF_CLASS_INDEX, statistic="csi", threshold=0.5)
        metric.update_state(y_true_1, y_pred_1)
        metric.update_state(y_true_2, y_pred_2)
        stateful_csi = metric.result().numpy()

        assert stateful_csi == pytest.approx(1.0, abs=1e-5)
        assert stateful_csi != pytest.approx(batch_mean_csi, abs=1e-3)


class TestPerClassContingencyMetricAccumulation:
    @pytest.mark.parametrize("statistic", ["hss", "csi", "pod"])
    def test_two_updates_equal_one_call_on_concatenated_batch(self, front_pred, statistic):
        y_true, y_pred = front_pred
        y_true_a, y_true_b = y_true[:1], y_true[1:]
        y_pred_a, y_pred_b = y_pred[:1], y_pred[1:]

        two_update_metric = PerClassContingencyMetric(class_index=CF_CLASS_INDEX, statistic=statistic, threshold=0.5)
        two_update_metric.update_state(y_true_a, y_pred_a)
        two_update_metric.update_state(y_true_b, y_pred_b)

        concatenated_metric = PerClassContingencyMetric(class_index=CF_CLASS_INDEX, statistic=statistic, threshold=0.5)
        concatenated_metric.update_state(y_true, y_pred)

        assert two_update_metric.result().numpy() == pytest.approx(concatenated_metric.result().numpy(), abs=1e-5)


class TestPerClassContingencyMetricResetState:
    def test_reset_state_returns_to_initial_value(self, front_pred):
        # threshold=None (soft): front_pred's CF probability of 0.35 contributes a nonzero
        # true-positive count, so the metric moves away from its zero initial value.
        y_true, y_pred = front_pred
        metric = PerClassContingencyMetric(class_index=CF_CLASS_INDEX, statistic="csi", threshold=None)
        initial_value = metric.result().numpy()

        metric.update_state(y_true, y_pred)
        assert metric.result().numpy() != pytest.approx(initial_value, abs=1e-5)

        metric.reset_state()
        assert metric.result().numpy() == pytest.approx(initial_value, abs=1e-5)


class TestPerClassContingencyMetric5D:
    def test_works_on_5d_batch_level_lat_lon_class_input(self):
        # Two identical pressure levels stacked; hand-computed values double but ratios match
        # the 4D hand-computed case in TestPerClassContingencyMetricHandComputed.
        n_pixels = 4
        n_levels = 2
        y_true = np.zeros((1, n_levels, 1, n_pixels, N_CLASSES), dtype=np.float32)
        y_pred = np.zeros((1, n_levels, 1, n_pixels, N_CLASSES), dtype=np.float32)
        true_values = [1.0, 0.0, 1.0, 0.0]
        pred_values = [0.9, 0.6, 0.3, 0.1]
        for level in range(n_levels):
            y_true[0, level, 0, :, CF_CLASS_INDEX] = true_values
            y_pred[0, level, 0, :, CF_CLASS_INDEX] = pred_values

        metric = PerClassContingencyMetric(class_index=CF_CLASS_INDEX, statistic="csi", threshold=0.5)
        metric.update_state(y_true, y_pred)
        assert metric.result().numpy() == pytest.approx(1 / 3, abs=1e-5)


class TestPerClassContingencyMetricConfig:
    def test_get_config_round_trips_through_from_config(self):
        metric = PerClassContingencyMetric(
            class_index=CF_CLASS_INDEX, statistic="hss", threshold=0.5, name="hss_hard_CF"
        )
        config = metric.get_config()
        restored = PerClassContingencyMetric.from_config(config)

        assert restored.class_index == metric.class_index
        assert restored.statistic == metric.statistic
        assert restored.threshold == metric.threshold
        assert restored.name == metric.name

        y_true, y_pred = _one_class_tensor(
            TestPerClassContingencyMetricHandComputed.TRUE_VALUES,
            TestPerClassContingencyMetricHandComputed.PRED_VALUES,
        )
        metric.update_state(y_true, y_pred)
        restored.update_state(y_true, y_pred)
        assert restored.result().numpy() == pytest.approx(metric.result().numpy(), abs=1e-5)


class TestPerFrontTypeMetrics:
    def test_returns_four_metrics_per_front_type_with_unique_names(self):
        metrics_list = per_front_type_metrics(constants.FRONT_TYPE_CLASS_INDEX)
        names = [metric.name for metric in metrics_list]

        assert len(names) == len(set(names)), f"Metric names must be unique, got {names}"
        assert len(metrics_list) == 4 * len(constants.FRONT_TYPE_CLASS_INDEX)

        for front_type in constants.FRONT_TYPE_CLASS_INDEX:
            assert f"hss_{front_type}" in names
            assert f"hss_hard_{front_type}" in names
            assert f"csi_{front_type}" in names
            assert f"pod_{front_type}" in names

    def test_hard_threshold_is_configurable(self):
        metrics_list = per_front_type_metrics(constants.FRONT_TYPE_CLASS_INDEX, hard_threshold=0.7)
        hard_metrics = [
            metric
            for metric in metrics_list
            if metric.name.startswith("hss_hard_") or metric.name.startswith("csi_") or metric.name.startswith("pod_")
        ]
        assert all(metric.threshold == 0.7 for metric in hard_metrics)

        soft_hss_metrics = [
            metric
            for metric in metrics_list
            if metric.name.startswith("hss_") and not metric.name.startswith("hss_hard_")
        ]
        assert all(metric.threshold is None for metric in soft_hss_metrics)
