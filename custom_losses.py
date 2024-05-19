"""
Custom loss functions for U-Net models.
    - Brier Skill Score (BSS)
    - Critical Success Index (CSI)
    - Fractions Skill Score (FSS)
    - Probability of Detection (POD)

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.5.18
"""
import tensorflow as tf


def brier_skill_score(class_weights: list[int | float, ...] = None):
    """
    Brier skill score (BSS) loss function.

    class_weights: list of values or None
        List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.
    """

    @tf.function
    def bss_loss(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """

        losses = tf.math.square(tf.subtract(y_true, y_pred))

        if class_weights is not None:
            relative_class_weights = tf.cast(class_weights / tf.math.reduce_sum(class_weights), tf.float32)
            losses *= relative_class_weights

        brier_score_loss = tf.math.reduce_sum(losses) / tf.size(losses)
        return brier_score_loss

    return bss_loss


def critical_success_index(threshold: float = None,
                           window_size: tuple[int, ...] | list[int, ...] = None,
                           class_weights: list[int | float, ...] = None):
    """
    Critical Success Index (CSI) loss function.

    y_true: tf.Tensor
        One-hot encoded tensor containing labels.
    y_pred: tf.Tensor
        Tensor containing model predictions.
    threshold: float or None
        Optional probability threshold that binarizes y_pred. Values in y_pred greater than or equal to the threshold are
            set to 1, and 0 otherwise.
        If the threshold is set, it must be greater than 0 and less than 1.
    window_size: tuple or list of ints or None
        Pool/kernel size of the max-pooling window for neighborhood statistics. (e.g. if calculating the loss with a 4-pixel
            window, this should be set to 4).
        Note that this parameter is experimental and may return unexpected results.
    class_weights: list of values or None
        List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.
    """

    @tf.function
    def csi_loss(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """

        if window_size is not None:
            y_pred = tf.nn.max_pool(y_pred, ksize=window_size, strides=1, padding="VALID")
            y_true = tf.nn.max_pool(y_true, ksize=window_size, strides=1, padding="VALID")

        if threshold is not None:
            y_pred = tf.where(y_pred >= threshold, 1, 0)

        y_pred_neg = 1 - y_pred
        y_true_neg = 1 - y_true

        sum_over_axes = tf.range(tf.rank(y_pred) - 1)  # Indices for axes to sum over. Excludes the final (class) dimension.

        true_positives = tf.math.reduce_sum(y_pred * y_true, axis=sum_over_axes)
        false_negatives = tf.math.reduce_sum(y_pred_neg * y_true, axis=sum_over_axes)
        false_positives = tf.math.reduce_sum(y_pred * y_true_neg, axis=sum_over_axes)

        if class_weights is not None:
            relative_class_weights = tf.cast(class_weights / tf.math.reduce_sum(class_weights), tf.float32)
            true_positives *= relative_class_weights
            false_positives *= relative_class_weights
            false_negatives *= relative_class_weights

        csi = tf.math.divide(tf.math.reduce_sum(true_positives),
            tf.math.reduce_sum(true_positives) + tf.math.reduce_sum(false_positives) + tf.math.reduce_sum(false_negatives))

        return 1 - csi

    return csi_loss


def fractions_skill_score(mask_size: int | tuple[int, ...] | list[int, ...] = (3, 3),
                          threshold: float = None,
                          c: float = 1.0):
    """
    Fractions skill score loss function.

    Parameters
    ----------
    mask_size: int or tuple
        Size of the mask/pool in the AveragePooling layers.
    threshold: float or None
        Threshold for discretization.
    c: int or float
        C parameter in the sigmoid function for soft discretization. Has no effect if 'threshold' parameter is not set.

    Returns
    -------
    fss_loss: float
        Fractions skill score.

    References
    ----------
    (RL2008) Roberts, N. M., and H. W. Lean, 2008: Scale-Selective Verification of Rainfall Accumulations from High-Resolution
        Forecasts of Convective Events. Mon. Wea. Rev., 136, 78–97, https://doi.org/10.1175/2007MWR2123.1.
    """

    # keyword arguments for the AveragePooling layer
    pool_args = dict(pool_size=mask_size, strides=1, padding="same")

    # if mask_size is an int, convert to a tuple. This allows us to check the length of the tuple and pull the correct AveragePooling layer
    if isinstance(mask_size, int):
        mask_size = (mask_size, )

    # if mask_size is an list, convert to a tuple
    elif isinstance(mask_size, list):
        mask_size = tuple(mask_size)

    # make sure the mask size is between 1 and 3
    assert 1 <= len(mask_size) <= 3, "mask_size must have length between 1 and 3, received length %d" % len(mask_size)

    # get the pooling layer based off the length of the mask_size tuple
    pool = getattr(tf.keras.layers, "AveragePooling%dD" % len(mask_size))(**pool_args)

    @tf.function
    def fss_loss(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """

        # discretize model predictions and labels
        if threshold is not None:
            y_true = tf.math.sigmoid(c * (y_true - threshold))
            y_pred = tf.math.sigmoid(c * (y_pred - threshold))

        O_n = pool(y_true)  # observed fractions (Eq. 2 in RL2008)
        M_n = pool(y_pred)  # model forecast fractions (Eq. 3 in RL2008)

        MSE_n = tf.keras.metrics.mean_squared_error(O_n, M_n)  # MSE for model forecast fractions (Eq. 5 in RL2008)
        MSE_ref = tf.reduce_mean(tf.square(O_n)) + tf.reduce_mean(tf.square(M_n))  # reference forecast (Eq. 7 in RL2008)

        FSS = 1 - MSE_n / (MSE_ref + 1e-10)  # fractions skill score (Eq. 6 in RL2008)

        return 1 - FSS

    return fss_loss


def probability_of_detection(threshold: float = None,
                             window_size: tuple[int, ...] | list[int, ...] = None,
                             class_weights: list[int | float, ...] = None):
    """
    Probability of Detection (POD) as a loss function. This turns the function into the miss rate.

    threshold: float or None
        Optional probability threshold that binarizes y_pred. Values in y_pred greater than or equal to the threshold are
            set to 1, and 0 otherwise.
        If the threshold is set, it must be greater than 0 and less than 1.
    window_size: tuple or list of ints or None
        Pool/kernel size of the max-pooling window for neighborhood statistics. (e.g. if calculating the loss with a 5-pixel
            window, this should be set to 5).
        Note that this parameter is experimental and may return unexpected results.
    class_weights: list of values or None
        List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.
    """

    @tf.function
    def pod_loss(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """

        if window_size is not None:
            y_pred = tf.nn.max_pool(y_pred, ksize=window_size, strides=1, padding="VALID")
            y_true = tf.nn.max_pool(y_true, ksize=window_size, strides=1, padding="VALID")

        y_pred = tf.where(y_pred >= threshold, 1.0, 0.0) if threshold is not None else y_pred

        y_pred_neg = 1 - y_pred

        sum_over_axes = tf.range(tf.rank(y_pred) - 1)  # Indices for axes to sum over. Excludes the final (class) dimension.

        true_positives = tf.math.reduce_sum(y_pred * y_true, axis=sum_over_axes)
        false_negatives = tf.math.reduce_sum(y_pred_neg * y_true, axis=sum_over_axes)

        if class_weights is not None:
            relative_class_weights = tf.cast(class_weights / tf.math.reduce_sum(class_weights), tf.float32)
            pod = tf.math.reduce_sum(tf.math.divide_no_nan(true_positives, true_positives + false_negatives) * relative_class_weights)
        else:
            pod = tf.math.reduce_sum(tf.math.divide_no_nan(true_positives, true_positives + false_negatives))

        return 1 - pod

    return pod_loss