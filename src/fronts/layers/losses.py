"""Custom loss functions for U-Net models: BSS, CSI, FSS, and POD."""

import tensorflow as tf


def brier_skill_score(
    alpha: int | float = 1.0,
    beta: int | float = 0.5,
    class_weights: list[int | float] | None = None,
):
    """Brier skill score (BSS) loss function.

    Args:
        alpha: Controls how steep the sigmoid function is for discretization. Higher values make it steeper and
            can help prevent training from stalling. Values greater than 4 are not recommended.
        beta: Controls some behaviors of the sigmoid discretization function. Default and recommended value is 0.5.
        class_weights: Weights to apply to each class. Length must equal the number of classes in y_pred and y_true.
    """
    if class_weights is not None:
        class_weights = tf.cast(class_weights, tf.float32)

    @tf.function
    def bss_loss(y_true, y_pred):
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

        if class_weights is not None:
            losses *= class_weights

        brier_score_loss = tf.math.reduce_sum(losses) / tf.size(losses)
        return brier_score_loss

    return bss_loss


def critical_success_index(
    alpha: int | float = 1.0,
    beta: int | float = 0.5,
    class_weights: list[int | float] | None = None,
):
    """Critical Success Index (CSI) loss function.

    Args:
        alpha: Controls how steep the sigmoid function is for discretization. Higher values make it steeper and
            can help prevent training from stalling. Values greater than 4 are not recommended.
        beta: Controls some behaviors of the sigmoid discretization function. Default and recommended value is 0.5.
        class_weights: Weights to apply to each class. Length must equal the number of classes in y_pred and y_true.
    """
    if class_weights is not None:
        class_weights = tf.cast(class_weights, tf.float32)

    @tf.function
    def csi_loss(y_true, y_pred):
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

        if class_weights is not None:
            true_positives *= class_weights
            false_positives *= class_weights
            false_negatives *= class_weights

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
):
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

    if class_weights is not None:
        class_weights = tf.cast(class_weights, tf.float32)

    @tf.function
    def fss_loss(y_true, y_pred):
        """Compute FSS loss for a batch.

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

        MSE_n = tf.reduce_mean(tf.square(O_n - M_n))  # MSE for model forecast fractions (Eq. 5 in RL2008)
        MSE_ref = tf.reduce_mean(tf.square(O_n)) + tf.reduce_mean(
            tf.square(M_n)
        )  # reference forecast (Eq. 7 in RL2008)

        FSS = 1 - MSE_n / (MSE_ref + 1e-10)  # fractions skill score (Eq. 6 in RL2008)

        return 1 - FSS

    return fss_loss


def probability_of_detection(class_weights: list[int | float] | None = None):
    """Probability of Detection (POD) as a loss function (returns miss rate = 1 - POD).

    Intended only for permutation studies; do not use to train models.

    Args:
        class_weights: Weights to apply to each class. Length must equal the number of classes in y_pred and y_true.
    """

    @tf.function
    def pod_loss(y_true, y_pred):
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
