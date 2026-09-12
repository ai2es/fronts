"""Tests for custom loss functions: BSS, CSI, FSS, POD, neighborhood Brier."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
losses = pytest.importorskip("fronts.layers.losses")
fractions_skill_score = pytest.importorskip("fronts.layers.losses").fractions_skill_score

N_BATCH = 2
N_H = 8
N_W = 8
N_CLASSES = 6
RESOLUTION_DEG = 0.25

EQUATOR_LATITUDES = np.arange(N_H * RESOLUTION_DEG, 0.0, -RESOLUTION_DEG)
POLAR_LATITUDES = np.arange(80.0, 80.0 - N_H * RESOLUTION_DEG, -RESOLUTION_DEG)


def _one_hot_background(n_batch: int = N_BATCH, n_h: int = N_H, n_w: int = N_W) -> np.ndarray:
    y = np.zeros((n_batch, n_h, n_w, N_CLASSES), dtype=np.float32)
    y[..., 0] = 1.0
    return y


def _with_front_row(y: np.ndarray, row: int, cls: int = 1) -> np.ndarray:
    y = y.copy()
    y[:, row, :, 0] = 0.0
    y[:, row, :, cls] = 1.0
    return y


class TestPlanWindows:
    def test_meridional_half_width_constant_with_latitude(self):
        plans_eq = losses._plan_windows(EQUATOR_LATITUDES, RESOLUTION_DEG, (25.0,), max_half_x=128)
        plans_polar = losses._plan_windows(POLAR_LATITUDES, RESOLUTION_DEG, (25.0,), max_half_x=128)
        assert plans_eq[0]["half_y"] == plans_polar[0]["half_y"] == 1

    def test_zonal_half_width_widens_with_latitude(self):
        plans_eq = losses._plan_windows(EQUATOR_LATITUDES, RESOLUTION_DEG, (25.0,), max_half_x=128)
        plans_polar = losses._plan_windows(POLAR_LATITUDES, RESOLUTION_DEG, (25.0,), max_half_x=128)
        assert plans_eq[0]["half_x"].max() == 1
        assert plans_polar[0]["half_x"][0] == 5, "1/cos(80 deg) is ~5.8, so the zonal window must be ~5x wider"
        assert plans_polar[0]["half_x"].min() > plans_eq[0]["half_x"].max()

    def test_zonal_half_width_clipped_at_max(self):
        plans = losses._plan_windows(POLAR_LATITUDES, RESOLUTION_DEG, (250.0,), max_half_x=8)
        assert plans[0]["half_x"].max() == 8

    def test_one_plan_per_tolerance(self):
        plans = losses._plan_windows(EQUATOR_LATITUDES, RESOLUTION_DEG, (25.0, 100.0, 250.0), max_half_x=128)
        assert len(plans) == 3
        assert plans[0]["half_y"] < plans[1]["half_y"] < plans[2]["half_y"]

    def test_max_distinct_widths_bounds_unique_half_x_values(self):
        wide_latitudes = np.arange(89.0, 0.0, -RESOLUTION_DEG)
        plans = losses._plan_windows(
            wide_latitudes, RESOLUTION_DEG, (250.0,), max_half_x=128, max_distinct_widths=4
        )
        assert len(np.unique(plans[0]["half_x"])) <= 4

    def test_max_distinct_widths_none_leaves_full_precision(self):
        wide_latitudes = np.arange(89.0, 0.0, -RESOLUTION_DEG)
        plans = losses._plan_windows(
            wide_latitudes, RESOLUTION_DEG, (250.0,), max_half_x=128, max_distinct_widths=None
        )
        assert len(np.unique(plans[0]["half_x"])) > 4


class TestBucketWidths:
    def test_no_op_when_already_within_limit(self):
        half_x = np.array([0, 1, 1, 3, 2])
        np.testing.assert_array_equal(losses._bucket_widths(half_x, max_distinct=8), half_x)

    def test_reduces_distinct_count_to_at_most_max_distinct(self):
        half_x = np.arange(20)
        bucketed = losses._bucket_widths(half_x, max_distinct=4)
        assert len(np.unique(bucketed)) <= 4

    def test_never_shrinks_below_original_value(self):
        """Bucketing must never make a window narrower than what was actually requested."""
        half_x = np.arange(20)
        bucketed = losses._bucket_widths(half_x, max_distinct=4)
        assert np.all(bucketed >= half_x)


class TestLatDependentPool:
    def test_pool_of_ones_is_one_everywhere(self):
        field = tf.ones((1, N_H, N_W, 1), tf.float32)
        pooled = losses._lat_dependent_pool(field, half_y=2, half_x_per_row=np.full(N_H, 2), periodic_lon=False)
        np.testing.assert_allclose(pooled.numpy(), 1.0, atol=1e-6)

    def test_edges_use_valid_cell_normalization_not_reflection(self):
        """A positive column next to the boundary must be averaged over in-domain cells only.

        With half_x=1 the window at column 0 covers valid columns {0, 1}: mean 1/2.
        Reflection would give 2/3; plain zero-padding without count normalization, 1/3.
        """
        field = np.zeros((1, N_H, N_W, 1), dtype=np.float32)
        field[:, :, 1, :] = 1.0
        pooled = losses._lat_dependent_pool(
            tf.constant(field), half_y=0, half_x_per_row=np.full(N_H, 1), periodic_lon=False
        )
        np.testing.assert_allclose(pooled.numpy()[:, :, 0, :], 0.5, atol=1e-6)

    def test_periodic_longitude_wraps(self):
        field = np.zeros((1, N_H, N_W, 1), dtype=np.float32)
        field[:, :, -1, :] = 1.0
        pooled = losses._lat_dependent_pool(
            tf.constant(field), half_y=0, half_x_per_row=np.full(N_H, 1), periodic_lon=True
        )
        np.testing.assert_allclose(pooled.numpy()[:, :, 0, :], 1.0 / 3.0, atol=1e-6)

    @staticmethod
    def _brute_force_pool(field: np.ndarray, half_y: int, half_x_per_row: np.ndarray, periodic: bool) -> np.ndarray:
        """Direct per-pixel window mean over valid cells; the definition the fast version must match."""
        _, n_h, n_w, _ = field.shape
        out = np.zeros_like(field)
        for i in range(n_h):
            half_x = int(half_x_per_row[i])
            rows = list(range(max(0, i - half_y), min(n_h, i + half_y + 1)))
            for j in range(n_w):
                if periodic:
                    cols = [(j + dj) % n_w for dj in range(-half_x, half_x + 1)]
                else:
                    cols = list(range(max(0, j - half_x), min(n_w, j + half_x + 1)))
                out[:, i, j, :] = field[:, rows][:, :, cols].mean(axis=(1, 2))
        return out

    @pytest.mark.parametrize("periodic", [False, True])
    def test_matches_brute_force_with_heterogeneous_widths(self, periodic):
        """Per-row half-widths (incl. repeats and zero) must reproduce the direct window mean exactly."""
        rng = np.random.default_rng(0)
        field = rng.random((2, N_H, 12, 3)).astype(np.float32)
        half_x_per_row = np.array([0, 1, 1, 3, 2, 2, 3, 0])
        pooled = losses._lat_dependent_pool(tf.constant(field), 1, half_x_per_row, periodic)
        expected = self._brute_force_pool(field, 1, half_x_per_row, periodic)
        np.testing.assert_allclose(pooled.numpy(), expected, atol=1e-5)

    def test_rows_stay_in_original_order(self):
        """Rows grouped by width for pooling must come back in latitude order.

        Row i carries the constant value i and every window is fully in-domain along x
        (periodic), so with half_y=0 the pooled row values must equal the row index.
        """
        field = np.tile(np.arange(N_H, dtype=np.float32).reshape(1, N_H, 1, 1), (1, 1, N_W, 1))
        half_x_per_row = np.array([3, 1, 2, 1, 3, 0, 2, 0])
        pooled = losses._lat_dependent_pool(tf.constant(field), 0, half_x_per_row, periodic_lon=True)
        np.testing.assert_allclose(pooled.numpy()[0, :, 0, 0], np.arange(N_H), atol=1e-5)


class TestNeighborhoodBrierScore:
    def test_perfect_prediction_is_zero(self):
        y_true = _with_front_row(_one_hot_background(), row=4)
        loss_fn = losses.neighborhood_brier_score(latitudes=EQUATOR_LATITUDES, tolerance_km=25.0)
        result = loss_fn(y_true, y_true.copy()).numpy()
        assert result.shape == (N_BATCH,)
        np.testing.assert_allclose(result, 0.0, atol=1e-7)

    def test_wrong_prediction_is_positive(self):
        y_true = _with_front_row(_one_hot_background(), row=4)
        y_pred = _one_hot_background()
        loss_fn = losses.neighborhood_brier_score(latitudes=EQUATOR_LATITUDES, tolerance_km=25.0)
        assert loss_fn(y_true, y_pred).numpy().min() > 0.0

    def test_batch_elements_scored_independently(self):
        y_true = _with_front_row(_one_hot_background(), row=4)
        y_pred = y_true.copy()
        y_pred[1] = _one_hot_background(n_batch=1)[0]
        loss_fn = losses.neighborhood_brier_score(latitudes=EQUATOR_LATITUDES, tolerance_km=25.0)
        result = loss_fn(y_true, y_pred).numpy()
        assert result[0] == pytest.approx(0.0, abs=1e-7)
        assert result[1] > 0.0

    def test_displacement_tolerance(self):
        """A 1-pixel near miss must cost far less than a distant miss, and an exact hit costs nothing."""
        n_h = 16
        latitudes = np.arange(n_h * RESOLUTION_DEG, 0.0, -RESOLUTION_DEG)
        y_true = _with_front_row(_one_hot_background(n_h=n_h), row=4)
        loss_fn = losses.neighborhood_brier_score(latitudes=latitudes, tolerance_km=25.0)
        exact = loss_fn(y_true, _with_front_row(_one_hot_background(n_h=n_h), row=4)).numpy().mean()
        near = loss_fn(y_true, _with_front_row(_one_hot_background(n_h=n_h), row=5)).numpy().mean()
        far = loss_fn(y_true, _with_front_row(_one_hot_background(n_h=n_h), row=12)).numpy().mean()
        assert exact == pytest.approx(0.0, abs=1e-7)
        assert 0.0 < near < far
        assert near < 0.75 * far, f"Near miss ({near:.5f}) should be well below a far miss ({far:.5f})"

    def test_zero_class_weight_excludes_class(self):
        y_true = _one_hot_background()
        y_pred = np.zeros_like(y_true)
        class_weights = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        weighted = losses.neighborhood_brier_score(
            latitudes=EQUATOR_LATITUDES, tolerance_km=25.0, class_weights=class_weights
        )
        unweighted = losses.neighborhood_brier_score(latitudes=EQUATOR_LATITUDES, tolerance_km=25.0)
        assert weighted(y_true, y_pred).numpy().max() == pytest.approx(0.0, abs=1e-7)
        assert unweighted(y_true, y_pred).numpy().min() > 0.0

    def test_resolution_inferred_from_descending_latitudes(self):
        loss_fn = losses.neighborhood_brier_score(latitudes=POLAR_LATITUDES, tolerance_km=25.0)
        y_true = _with_front_row(_one_hot_background(), row=4)
        result = loss_fn(y_true, _one_hot_background()).numpy()
        assert np.all(np.isfinite(result)) and result.min() > 0.0

    def test_include_pixel_sharpens_near_miss_penalty(self):
        y_true = _with_front_row(_one_hot_background(), row=4)
        y_pred = _with_front_row(_one_hot_background(), row=5)
        base = losses.neighborhood_brier_score(latitudes=EQUATOR_LATITUDES, tolerance_km=25.0)
        with_pixel = losses.neighborhood_brier_score(
            latitudes=EQUATOR_LATITUDES, tolerance_km=25.0, include_pixel=True, pixel_weight=1.0
        )
        assert with_pixel(y_true, y_pred).numpy().mean() > base(y_true, y_pred).numpy().mean()

    def test_gradient_flows_to_predictions(self):
        y_true = tf.constant(_with_front_row(_one_hot_background(), row=4))
        y_pred = tf.Variable(np.full((N_BATCH, N_H, N_W, N_CLASSES), 1.0 / N_CLASSES, dtype=np.float32))
        loss_fn = losses.neighborhood_brier_score(latitudes=EQUATOR_LATITUDES, tolerance_km=25.0)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(loss_fn(y_true, y_pred))
        grad = tape.gradient(loss, y_pred)
        assert grad is not None
        assert float(tf.reduce_max(tf.abs(grad))) > 0.0

    def test_lat_dependent_pool_defaults_to_false(self):
        """The lat-aware pooling is opt-in; the default must use the cheap isotropic pool."""
        y_true = _with_front_row(_one_hot_background(n_h=16), row=4)
        loss_fn = losses.neighborhood_brier_score(
            latitudes=np.arange(16 * RESOLUTION_DEG, 0.0, -RESOLUTION_DEG), tolerance_km=25.0
        )
        explicit_false = losses.neighborhood_brier_score(
            latitudes=np.arange(16 * RESOLUTION_DEG, 0.0, -RESOLUTION_DEG),
            tolerance_km=25.0,
            lat_dependent_pool=False,
        )
        y_pred = _with_front_row(_one_hot_background(n_h=16), row=6)
        np.testing.assert_allclose(loss_fn(y_true, y_pred).numpy(), explicit_false(y_true, y_pred).numpy())

    def test_isotropic_pool_ignores_zonal_widening_at_high_latitude(self):
        """At high latitude the lat-dependent pool widens zonally; the isotropic pool must not."""
        n_h = 16
        n_w = 16
        latitudes = np.arange(80.0, 80.0 - n_h * RESOLUTION_DEG, -RESOLUTION_DEG)
        y_true = np.zeros((1, n_h, n_w, N_CLASSES), dtype=np.float32)
        y_true[:, 4, 4, 0] = 0.0
        y_true[:, 4, 4, 1] = 1.0
        y_true[:, 4, :, 0] = 1.0
        y_true[:, 4, 4, 0] = 0.0
        y_pred = y_true.copy()
        y_pred[:, 4, 4, :] = 0.0
        y_pred[:, 4, 7, 0] = 0.0
        y_pred[:, 4, 7, 1] = 1.0
        y_pred[:, 4, 4, 0] = 1.0

        lat_dependent = losses.neighborhood_brier_score(
            latitudes=latitudes, tolerance_km=25.0, lat_dependent_pool=True, max_half_x=16
        )
        isotropic = losses.neighborhood_brier_score(
            latitudes=latitudes, tolerance_km=25.0, lat_dependent_pool=False, max_half_x=16
        )
        lat_dependent_loss = float(lat_dependent(y_true, y_pred).numpy().mean())
        isotropic_loss = float(isotropic(y_true, y_pred).numpy().mean())
        assert lat_dependent_loss < isotropic_loss, (
            f"lat-dependent pool (half_x~5 at 80deg) should absorb the 3-column zonal miss "
            f"better than the isotropic pool (half_y-only window): "
            f"lat_dependent={lat_dependent_loss:.5f}, isotropic={isotropic_loss:.5f}"
        )

    def test_traces_under_tf_function(self):
        y_true = _with_front_row(_one_hot_background(), row=4)
        loss_fn = losses.neighborhood_brier_score(latitudes=EQUATOR_LATITUDES, tolerance_km=25.0)
        traced = tf.function(loss_fn)
        result = traced(tf.constant(y_true), tf.constant(_one_hot_background())).numpy()
        assert result.shape == (N_BATCH,)
        assert np.all(np.isfinite(result))


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

        loss_equal_weights = fractions_skill_score(mask_size=(3, 3), class_weights=[1.0] * N_CLASSES)(
            y_true, y_pred
        ).numpy()

        loss_cf_only = fractions_skill_score(mask_size=(3, 3), class_weights=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0])(
            y_true, y_pred
        ).numpy()

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
            fractions_skill_score(mask_size=(3, 3), class_weights=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0])(y_true, y_pred)
            .numpy()
            .mean()
        )

        assert loss_unweighted > 0.0
        assert loss_bg_zeroed == pytest.approx(0.0, abs=1e-5)
