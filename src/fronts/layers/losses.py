"""Custom loss functions for U-Net models: BSS, CSI, FSS, neighborhood Brier, and POD."""

from collections.abc import Callable, Sequence

import numpy as np
import tensorflow as tf

KM_PER_DEG = 111.32


def brier_skill_score(
    alpha: int | float = 1.0,
    beta: int | float = 0.5,
    class_weights: list[int | float] | None = None,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Brier skill score (BSS) loss function.

    Args:
        alpha: Controls how steep the sigmoid function is for discretization. Higher values make it steeper and
            can help prevent training from stalling. Values greater than 4 are not recommended.
        beta: Controls some behaviors of the sigmoid discretization function. Default and recommended value is 0.5.
        class_weights: Weights to apply to each class. Length must equal the number of classes in y_pred and y_true.
    """
    cw: tf.Tensor | None = tf.cast(class_weights, tf.float32) if class_weights is not None else None

    @tf.function
    def bss_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute BSS loss for a batch.

        Args:
            y_true: One-hot encoded tensor containing labels.
            y_pred: Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # discretize model predictions and labels
        y_true = tf.math.sigmoid(alpha * (y_true - beta))
        y_pred = tf.math.sigmoid(alpha * (y_pred - beta))

        losses = tf.math.square(tf.subtract(y_true, y_pred))

        if cw is not None:
            losses *= cw

        spatial_axes = list(range(1, len(losses.shape)))
        return tf.reduce_mean(losses, axis=spatial_axes)

    return bss_loss


def critical_success_index(
    alpha: int | float = 1.0,
    beta: int | float = 0.5,
    class_weights: list[int | float] | None = None,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Critical Success Index (CSI) loss function.

    Args:
        alpha: Controls how steep the sigmoid function is for discretization. Higher values make it steeper and
            can help prevent training from stalling. Values greater than 4 are not recommended.
        beta: Controls some behaviors of the sigmoid discretization function. Default and recommended value is 0.5.
        class_weights: Weights to apply to each class. Length must equal the number of classes in y_pred and y_true.
    """
    cw: tf.Tensor | None = tf.cast(class_weights, tf.float32) if class_weights is not None else None

    @tf.function
    def csi_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute CSI loss for a batch.

        Args:
            y_true: One-hot encoded tensor containing labels.
            y_pred: Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # discretize model predictions and labels
        y_true = tf.math.sigmoid(alpha * (y_true - beta))
        y_pred = tf.math.sigmoid(alpha * (y_pred - beta))

        y_pred_neg = 1 - y_pred
        y_true_neg = 1 - y_true

        sum_over_axes = tf.range(
            tf.rank(y_pred) - 1
        )  # Indices for axes to sum over. Excludes the final (class) dimension.

        true_positives = tf.math.reduce_sum(y_pred * y_true, axis=sum_over_axes)
        false_negatives = tf.math.reduce_sum(y_pred_neg * y_true, axis=sum_over_axes)
        false_positives = tf.math.reduce_sum(y_pred * y_true_neg, axis=sum_over_axes)

        if cw is not None:
            true_positives *= cw
            false_positives *= cw
            false_negatives *= cw

        csi = tf.math.divide(
            tf.math.reduce_sum(true_positives),
            tf.math.reduce_sum(true_positives)
            + tf.math.reduce_sum(false_positives)
            + tf.math.reduce_sum(false_negatives),
        )

        return 1 - csi

    return csi_loss


def fractions_skill_score(
    mask_size: int | tuple[int, ...] | list[int] = (3, 3),
    alpha: int | float = 1.0,
    beta: int | float = 0.5,
    class_weights: list[int | float] | None = None,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Fractions skill score loss function (returns 1 - FSS).

    Args:
        mask_size: Size of the mask/pool in the AveragePooling layers.
        alpha: Controls how steep the sigmoid function is for discretization. Higher values make it steeper and
            can help prevent training from stalling. Values greater than 4 are not recommended.
        beta: Controls some behaviors of the sigmoid discretization function. Default and recommended value is 0.5.
        class_weights: Weights to apply to each class. Length must equal the number of classes in y_pred and y_true.

    References:
        Roberts & Lean (2008): https://doi.org/10.1175/2007MWR2123.1
    """
    # keyword arguments for the AveragePooling layer
    pool_args = {"pool_size": mask_size, "strides": 1, "padding": "same"}

    if isinstance(mask_size, int):
        mask_size = (mask_size,)
    elif isinstance(mask_size, list):
        mask_size = tuple(mask_size)

    assert 1 <= len(mask_size) <= 3, f"mask_size must have length between 1 and 3, received length {len(mask_size)}"

    pool = getattr(tf.keras.layers, f"AveragePooling{len(mask_size)}D")(**pool_args)

    cw: tf.Tensor | None = tf.cast(class_weights, tf.float32) if class_weights is not None else None
    relative_cw: tf.Tensor | None = (cw / tf.reduce_sum(cw)) if cw is not None else None

    @tf.function
    def fss_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute FSS loss for a batch.

        Args:
            y_true: One-hot encoded tensor containing labels.
            y_pred: Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # discretize model predictions and labels
        y_true = tf.math.sigmoid(alpha * (y_true - beta))
        y_pred = tf.math.sigmoid(alpha * (y_pred - beta))

        O_n = pool(y_true)  # observed fractions (Eq. 2 in RL2008)
        M_n = pool(y_pred)  # model forecast fractions (Eq. 3 in RL2008)

        # Reduce over spatial + class axes; keep batch dim so Keras can aggregate across replicas.
        # Class weights applied after pooling so sigmoid targets remain correct for zero-weight classes.
        spatial_axes = list(range(1, len(O_n.shape)))
        if relative_cw is not None:
            MSE_n = tf.reduce_mean(tf.square(O_n - M_n) * relative_cw, axis=spatial_axes)
            MSE_ref = tf.reduce_mean(tf.square(O_n) * relative_cw, axis=spatial_axes) + tf.reduce_mean(
                tf.square(M_n) * relative_cw, axis=spatial_axes
            )
        else:
            MSE_n = tf.reduce_mean(tf.square(O_n - M_n), axis=spatial_axes)
            MSE_ref = tf.reduce_mean(tf.square(O_n), axis=spatial_axes) + tf.reduce_mean(
                tf.square(M_n), axis=spatial_axes
            )

        FSS = 1 - tf.math.divide_no_nan(MSE_n, MSE_ref)  # fractions skill score (Eq. 6 in RL2008)

        return 1 - FSS

    return fss_loss


def _bucket_widths(half_x: np.ndarray, max_distinct: int) -> np.ndarray:
    """Quantize per-row zonal half-widths to at most ``max_distinct`` distinct values.

    Sorted unique widths are split into ``max_distinct`` contiguous groups, and every row
    takes its group's maximum width, so no neighborhood ever shrinks below what was
    requested. This bounds the row-group fan-out in ``_lat_dependent_pool``, which
    otherwise scales with the number of distinct zonal widths across the latitude range
    and dominates backward-pass memory in ``neighborhood_brier_score``.
    """
    unique_vals = np.unique(half_x)
    if len(unique_vals) <= max_distinct:
        return half_x
    representative = {int(v): int(group.max()) for group in np.array_split(unique_vals, max_distinct) for v in group}
    return np.vectorize(representative.get)(half_x)


def _plan_windows(
    latitudes: np.ndarray | Sequence[float],
    resolution_deg: float,
    tolerances_km: Sequence[float],
    max_half_x: int,
    pole_clip_deg: float = 85.0,
    max_distinct_widths: int | None = None,
) -> list[dict[str, int | np.ndarray]]:
    """Precompute window half-widths in pixels for each tolerance scale.

    Args:
        latitudes: Grid latitudes in degrees, one per row.
        resolution_deg: Grid spacing in degrees.
        tolerances_km: Neighborhood tolerance scales in kilometers.
        max_half_x: Upper bound on the zonal half-width in pixels.
        pole_clip_deg: Latitude at which cos(lat) is clipped to keep zonal windows finite near the poles.
        max_distinct_widths: Upper bound on distinct zonal half-widths per tolerance, via
            ``_bucket_widths``. None keeps full per-row precision.

    Returns:
        One dict per tolerance with ``half_y`` (scalar meridional half-width) and ``half_x``
        (per-row zonal half-widths, widened by 1/cos(lat) to stay isotropic in km).
    """
    lat = np.asarray(latitudes, dtype=np.float64)
    coslat = np.clip(np.cos(np.deg2rad(lat)), np.cos(np.deg2rad(pole_clip_deg)), 1.0)
    dy_km = resolution_deg * KM_PER_DEG
    plans = []
    for tol in tolerances_km:
        half_y = round(tol / dy_km)
        half_x = np.round(tol / (resolution_deg * KM_PER_DEG * coslat)).astype(int)
        half_x = np.clip(half_x, 0, max_half_x)
        if max_distinct_widths is not None:
            half_x = _bucket_widths(half_x, max_distinct_widths)
        plans.append({"half_y": half_y, "half_x": half_x})
    return plans


def _boxsum_y(field: tf.Tensor, half: int) -> tf.Tensor:
    """Zero-padded centered box sum along latitude (axis 1) via cumulative sums. No mirroring or wrapping."""
    if half == 0:
        return field
    padded = tf.pad(field, [[0, 0], [half, half], [0, 0], [0, 0]])
    c = tf.cumsum(padded, axis=1)
    c = tf.pad(c, [[0, 0], [1, 0], [0, 0], [0, 0]])
    k = 2 * half + 1
    return c[:, k:, :, :] - c[:, :-k, :, :]


def _boxsum_x(field: tf.Tensor, half: int, periodic: bool) -> tf.Tensor:
    """Centered box sum along longitude (axis 2); zero-padded unless ``periodic``."""
    if half == 0:
        return field
    if periodic:
        padded = tf.concat([field[:, :, -half:, :], field, field[:, :, :half, :]], axis=2)
    else:
        padded = tf.pad(field, [[0, 0], [0, 0], [half, half], [0, 0]])
    c = tf.cumsum(padded, axis=2)
    c = tf.pad(c, [[0, 0], [0, 0], [1, 0], [0, 0]])
    k = 2 * half + 1
    return c[:, :, k:, :] - c[:, :, :-k, :]


def _lat_dependent_pool(field: tf.Tensor, half_y: int, half_x_per_row: np.ndarray, periodic_lon: bool) -> tf.Tensor:
    """Valid-cell-normalized rectangular mean: constant half_y in latitude, per-row half_x in longitude.

    Each neighborhood fraction is (sum of in-domain cells) / (count of in-domain cells), so the
    window simply shrinks at domain edges rather than reflecting or wrapping.

    Each distinct zonal half-width's box sum runs on only the rows that use that width
    (gather, box-sum, reassemble) rather than on the full field with row masking. Zonal box
    sums are row-independent, so this is exact — and it keeps the total work at ~one field
    regardless of how many distinct widths the latitude range produces (66 on the full
    domain), where the full-field-per-width version materialized that many full-size
    intermediates and made this loss ~9x more memory-hungry than a single-scale pooling
    loss, forcing smaller training batches.
    """
    dtype = field.dtype
    hx = np.asarray(half_x_per_row)
    width = tf.shape(field)[2]
    ones_col = tf.ones((1, len(hx), 1, 1), dtype)
    ones_row = tf.ones((1, 1, width, 1), dtype)

    sum_y = _boxsum_y(field, half_y)
    count_y = _boxsum_y(ones_col, half_y)

    pieces = []
    row_groups = []
    for v in sorted({int(u) for u in hx}):
        rows = np.flatnonzero(hx == v)
        sum_xy = _boxsum_x(tf.gather(sum_y, rows, axis=1), v, periodic_lon)
        count_x = _boxsum_x(ones_row, v, periodic_lon)
        pieces.append(sum_xy / (tf.gather(count_y, rows, axis=1) * count_x))
        row_groups.append(rows)

    original_row_order = np.argsort(np.concatenate(row_groups))
    return tf.gather(tf.concat(pieces, axis=1), original_row_order, axis=1)


def neighborhood_brier_score(
    latitudes: np.ndarray | Sequence[float],
    resolution_deg: float | None = None,
    tolerance_km: float = 25.0,
    include_pixel: bool = False,
    pixel_weight: float = 0.1,
    class_weights: list[int | float] | None = None,
    periodic_lon: bool = False,
    max_half_x: int = 128,
    max_distinct_widths: int | None = 8,
    lat_dependent_pool: bool = False,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Latitude-aware neighborhood Brier loss: a proper alternative to FSS-as-loss.

    Single-neighborhood, single-pool design — one ``tolerance_km`` window, one pooling
    call per tensor, mirroring ``fractions_skill_score``'s single ``mask_size``. Returns
    the fractions Brier score (MSE of observed vs forecast neighborhood fractions).
    Proper; rewards calibrated probabilities. NO sigmoid discretization (would break
    propriety). Tolerance is isotropic in km per latitude; domain edges use valid-cell
    normalization. Set the tolerance to the label positional slack (replaces label
    dilation).

    Args:
        latitudes: Grid latitudes in degrees, one per output row. Must match the model's
            output height; every deep-supervision head must emit at full resolution.
        resolution_deg: Grid spacing in degrees. None infers it from ``latitudes``.
        tolerance_km: Neighborhood tolerance in kilometers.
        include_pixel: If True, adds a pixelwise (un-pooled) Brier term for sharpness.
        pixel_weight: Relative weight of the pixelwise term when ``include_pixel`` is True.
        class_weights: Weights to apply to each class. Length must equal the number of classes
            in y_pred and y_true. Applied post-pooling on the squared error.
        periodic_lon: If True, wraps the zonal window across the longitude edges. Only set for
            a genuine global 360-degree band; domain strips with non-adjacent ends must use False.
        max_half_x: Upper bound on the zonal half-width in pixels.
        max_distinct_widths: Upper bound on distinct zonal half-widths. Bounds the row-group
            fan-out in ``_lat_dependent_pool``, which otherwise scales with the number of
            distinct widths across the latitude range and dominates backward-pass memory.
            None keeps full per-row precision. Ignored when ``lat_dependent_pool`` is False.
        lat_dependent_pool: If True, widen the zonal window with 1/cos(lat) via
            ``_lat_dependent_pool``. If False (default), pool with a single symmetric
            ``AveragePooling2D`` window (meridional half-width in both directions, like
            ``fractions_skill_score``) — cheaper in backward-pass memory, at the cost of
            under-widening the zonal tolerance away from the equator.

    References:
        Roberts & Lean (2008): https://doi.org/10.1175/2007MWR2123.1
        Stein & Stoop (2024): https://doi.org/10.1175/MWR-D-22-0235.1
        Gneiting & Raftery (2007): https://doi.org/10.1198/016214506000001437
    """
    lat = np.asarray(latitudes, dtype=np.float64)
    if resolution_deg is None:
        resolution_deg = float(np.median(np.abs(np.diff(lat))))
    plan = _plan_windows(lat, resolution_deg, (tolerance_km,), max_half_x, max_distinct_widths=max_distinct_widths)[0]
    isotropic_pool = (
        None
        if lat_dependent_pool
        else tf.keras.layers.AveragePooling2D(
            pool_size=(2 * plan["half_y"] + 1, 2 * plan["half_y"] + 1), strides=1, padding="same"
        )
    )

    weight_sum = 1.0 + (pixel_weight if include_pixel else 0.0)

    cw: tf.Tensor | None = tf.cast(class_weights, tf.float32) if class_weights is not None else None

    @tf.function
    def nbs_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the neighborhood Brier loss for a batch.

        Args:
            y_true: One-hot encoded tensor containing labels.
            y_pred: Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Reduce over spatial + class axes; keep batch dim so Keras can aggregate across replicas.
        spatial_axes = list(range(1, len(y_pred.shape)))

        def _brier(obs: tf.Tensor, mod: tf.Tensor) -> tf.Tensor:
            se = tf.math.square(obs - mod)
            if cw is not None:
                se *= cw
            return tf.reduce_mean(se, axis=spatial_axes)

        if lat_dependent_pool:
            O_n = _lat_dependent_pool(y_true, plan["half_y"], plan["half_x"], periodic_lon)
            M_n = _lat_dependent_pool(y_pred, plan["half_y"], plan["half_x"], periodic_lon)
        else:
            O_n = isotropic_pool(y_true)
            M_n = isotropic_pool(y_pred)
        total = _brier(O_n, M_n)
        if include_pixel:
            total += float(pixel_weight) * _brier(y_true, y_pred)
        return total / weight_sum

    return nbs_loss


def probability_of_detection(
    class_weights: list[int | float] | None = None,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Probability of Detection (POD) as a loss function (returns miss rate = 1 - POD).

    Intended only for permutation studies; do not use to train models.

    Args:
        class_weights: Weights to apply to each class. Length must equal the number of classes in y_pred and y_true.
    """

    @tf.function
    def pod_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute POD loss for a batch.

        Args:
            y_true: One-hot encoded tensor containing labels.
            y_pred: Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_pred_neg = 1 - y_pred

        sum_over_axes = tf.range(
            tf.rank(y_pred) - 1
        )  # Indices for axes to sum over. Excludes the final (class) dimension.

        true_positives = tf.math.reduce_sum(y_pred * y_true, axis=sum_over_axes)
        false_negatives = tf.math.reduce_sum(y_pred_neg * y_true, axis=sum_over_axes)

        if class_weights is not None:
            relative_class_weights = tf.cast(class_weights / tf.math.reduce_sum(class_weights), tf.float32)
            pod = tf.math.reduce_sum(
                tf.math.divide_no_nan(true_positives, true_positives + false_negatives) * relative_class_weights
            )
        else:
            pod = tf.math.reduce_sum(tf.math.divide_no_nan(true_positives, true_positives + false_negatives))

        return 1 - pod

    return pod_loss
