"""Custom metrics for U-Net models: BSS, CSI, FSS, HSS, and POD."""

import tensorflow as tf


def brier_skill_score(class_weights: list[int | float] | None = None):
    """Brier skill score (BSS).

    Args:
        class_weights: Weights to apply to each class. Length must equal the number of classes in y_pred and y_true.
    """

    @tf.function
    def bss(y_true, y_pred):
        """Compute BSS for a batch.

        Args:
            y_true: One-hot encoded tensor containing labels.
            y_pred: Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        squared_errors = tf.math.square(tf.subtract(y_true, y_pred))

        if class_weights is not None:
            relative_class_weights = tf.cast(class_weights / tf.math.reduce_sum(class_weights), tf.float32)
            squared_errors *= relative_class_weights

        bss = 1 - tf.math.reduce_sum(squared_errors) / tf.size(squared_errors)

        return bss

    return bss


def critical_success_index(
    threshold: float | None = None,
    window_size: tuple[int, ...] | list[int] | None = None,
    class_weights: list[int | float] | None = None,
):
    """Critical success index (CSI).

    Args:
        threshold: Optional probability threshold that binarizes y_pred. Values >= threshold are set to 1, others to 0.
            Must be in (0, 1) if provided.
        window_size: Pool/kernel size of the max-pooling window for neighborhood statistics. Experimental; may return
            unexpected results.
        class_weights: Weights to apply to each class. Length must equal the number of classes in y_pred and y_true.
    """

    @tf.function
    def csi(y_true, y_pred):
        """Compute CSI for a batch.

        Args:
            y_true: One-hot encoded tensor containing labels.
            y_pred: Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if window_size is not None:
            y_pred = tf.nn.max_pool(y_pred, ksize=window_size, strides=1, padding="VALID")
            y_true = tf.nn.max_pool(y_true, ksize=window_size, strides=1, padding="VALID")

        if threshold is not None:
            y_pred = tf.where(y_pred >= threshold, 1.0, 0.0)

        y_pred_neg = 1 - y_pred
        y_true_neg = 1 - y_true

        sum_over_axes = tf.range(
            tf.rank(y_pred) - 1
        )  # Indices for axes to sum over. Excludes the final (class) dimension.

        true_positives = tf.math.reduce_sum(y_pred * y_true, axis=sum_over_axes)
        false_negatives = tf.math.reduce_sum(y_pred_neg * y_true, axis=sum_over_axes)
        false_positives = tf.math.reduce_sum(y_pred * y_true_neg, axis=sum_over_axes)

        if class_weights is not None:
            relative_class_weights = tf.cast(class_weights / tf.math.reduce_sum(class_weights), tf.float32)
            csi = tf.math.reduce_sum(
                tf.math.divide_no_nan(true_positives, true_positives + false_positives + false_negatives)
                * relative_class_weights
            )
        else:
            csi = tf.math.divide(
                tf.math.reduce_sum(true_positives),
                tf.math.reduce_sum(true_positives)
                + tf.math.reduce_sum(false_negatives)
                + tf.math.reduce_sum(false_positives),
            )

        return csi

    return csi


def fractions_skill_score(
    mask_size: int | tuple[int, ...] | list[int] = (3, 3),
    alpha: int | float = 1.0,
    beta: int | float = 0.5,
    class_weights: list[int | float] | None = None,
):
    """Fractions skill score (FSS) metric.

    Args:
        mask_size: Size of the mask/pool in the AveragePooling layers.
        alpha: Controls how steep the sigmoid function is for discretization. Higher values make it steeper.
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

    if class_weights is not None:
        class_weights = tf.cast(class_weights, tf.float32)

    @tf.function
    def fss(y_true, y_pred):
        """Compute FSS for a batch.

        Args:
            y_true: One-hot encoded tensor containing labels.
            y_pred: Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if class_weights is not None:
            y_true *= class_weights
            y_pred *= class_weights

        # discretize model predictions and labels
        y_true = tf.math.sigmoid(alpha * (y_true - beta))
        y_pred = tf.math.sigmoid(alpha * (y_pred - beta))

        O_n = pool(y_true)  # observed fractions (Eq. 2 in RL2008)
        M_n = pool(y_pred)  # model forecast fractions (Eq. 3 in RL2008)

        MSE_n = tf.reduce_mean(
            tf.square(O_n * class_weights - M_n * class_weights)
        )  # MSE for model forecast fractions (Eq. 5 in RL2008)
        MSE_ref = tf.reduce_mean(tf.square(O_n * class_weights)) + tf.reduce_mean(
            tf.square(M_n * class_weights)
        )  # reference forecast (Eq. 7 in RL2008)

        FSS = 1 - MSE_n / (MSE_ref + 1e-10)  # fractions skill score (Eq. 6 in RL2008)

        return FSS

    return fss


def heidke_skill_score(
    threshold: float | None = None,
    window_size: tuple[int, ...] | list[int] | None = None,
    class_weights: list[int | float] | None = None,
):
    """Heidke Skill Score (HSS).

    Args:
        threshold: Optional probability threshold that binarizes y_pred. Values >= threshold are set to 1, others to 0.
            Must be in (0, 1) if provided.
        window_size: Pool/kernel size of the max-pooling window for neighborhood statistics. Experimental; may return
            unexpected results.
        class_weights: Weights to apply to each class. Length must equal the number of classes in y_pred and y_true.
    """

    @tf.function
    def hss(y_true, y_pred):
        """Compute HSS for a batch.

        Args:
            y_true: One-hot encoded tensor containing labels.
            y_pred: Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if window_size is not None:
            y_pred = tf.nn.max_pool(y_pred, ksize=window_size, strides=1, padding="VALID")
            y_true = tf.nn.max_pool(y_true, ksize=window_size, strides=1, padding="VALID")

        if threshold is not None:
            y_pred = tf.where(y_pred >= threshold, 1.0, 0.0)

        sum_over_axes = tf.range(
            tf.rank(y_pred) - 1
        )  # Indices for axes to sum over. Excludes the final (class) dimension.

        true_positives = tf.math.reduce_sum(y_true * y_pred, axis=sum_over_axes)
        false_positives = tf.math.reduce_sum((1 - y_true) * y_pred, axis=sum_over_axes)
        false_negatives = tf.math.reduce_sum(y_true * (1 - y_pred), axis=sum_over_axes)
        true_negatives = tf.math.reduce_sum((1 - y_true) * (1 - y_pred), axis=sum_over_axes)

        if class_weights is not None:
            relative_class_weights = tf.cast(class_weights / tf.math.reduce_sum(class_weights), tf.float32)
            true_positives *= relative_class_weights
            true_negatives *= relative_class_weights
            false_positives *= relative_class_weights
            false_negatives *= relative_class_weights

        a = tf.math.reduce_sum(true_positives)
        b = tf.math.reduce_sum(false_positives)
        c = tf.math.reduce_sum(false_negatives)
        d = tf.math.reduce_sum(true_negatives)

        hss = 2 * tf.math.divide((a * d) - (b * c), ((a + c) * (c + d)) + ((a + b) * (b + d)))

        return hss

    return hss


def probability_of_detection(
    threshold: float | None = None,
    window_size: tuple[int, ...] | list[int] | None = None,
    class_weights: list[int | float] | None = None,
):
    """Probability of Detection (POD).

    Args:
        threshold: Optional probability threshold that binarizes y_pred. Values >= threshold are set to 1, others to 0.
            Must be in (0, 1) if provided.
        window_size: Pool/kernel size of the max-pooling window for neighborhood statistics. Experimental; may return
            unexpected results.
        class_weights: Weights to apply to each class. Length must equal the number of classes in y_pred and y_true.
    """

    @tf.function
    def pod(y_true, y_pred):
        """Compute POD for a batch.

        Args:
            y_true: One-hot encoded tensor containing labels.
            y_pred: Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if window_size is not None:
            y_pred = tf.nn.max_pool(y_pred, ksize=window_size, strides=1, padding="VALID")
            y_true = tf.nn.max_pool(y_true, ksize=window_size, strides=1, padding="VALID")

        y_pred = tf.where(y_pred >= threshold, 1.0, 0.0) if threshold is not None else y_pred
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

        return pod

    return pod
