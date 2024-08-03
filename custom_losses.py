"""
Custom loss functions for U-Net models.
    - Brier Skill Score (BSS)
    - Critical Success Index (CSI)
    - Fractions Skill Score (FSS)
    - Probability of Detection (POD)

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.8.3
"""
import tensorflow as tf


def brier_skill_score(alpha: int | float = 1.0,
                      beta: int | float = 0.5,
                      class_weights: list[int | float, ...] = None):
    """
    Brier skill score (BSS) loss function.

    alpha: int or float
        Parameter that controls how steep the sigmoid function is for discretization. Higher alpha makes the sigmoid function
            steeper and can help prevent the training process from stalling. Default value is 1. Values greater than 4 are
            not recommended.
    beta: int or float
        Parameter used to control some behaviors of the sigmoid discretization function. Default and recommended value is 0.5.
    class_weights: list of values or None
        List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.
    """

    if class_weights is not None:
        class_weights = tf.cast(class_weights, tf.float32)

    @tf.function
    def bss_loss(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """

        # discretize model predictions and labels
        y_true = tf.math.sigmoid(alpha * (y_true - beta))
        y_pred = tf.math.sigmoid(alpha * (y_pred - beta))

        losses = tf.math.square(tf.subtract(y_true, y_pred))

        if class_weights is not None:
            losses *= class_weights

        brier_score_loss = tf.math.reduce_sum(losses) / tf.size(losses)
        return brier_score_loss

    return bss_loss


def critical_success_index(alpha: int | float = 1.0,
                           beta: int | float = 0.5,
                           class_weights: list[int | float, ...] = None):
    """
    Critical Success Index (CSI) loss function.

    alpha: int or float
        Parameter that controls how steep the sigmoid function is for discretization. Higher alpha makes the sigmoid function
            steeper and can help prevent the training process from stalling. Default value is 1. Values greater than 4 are
            not recommended.
    beta: int or float
        Parameter used to control some behaviors of the sigmoid discretization function. Default and recommended value is 0.5.
    class_weights: list of values or None
        List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.
    """

    if class_weights is not None:
        class_weights = tf.cast(class_weights, tf.float32)

    @tf.function
    def csi_loss(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """

        # discretize model predictions and labels
        y_true = tf.math.sigmoid(alpha * (y_true - beta))
        y_pred = tf.math.sigmoid(alpha * (y_pred - beta))

        y_pred_neg = 1 - y_pred
        y_true_neg = 1 - y_true

        sum_over_axes = tf.range(tf.rank(y_pred) - 1)  # Indices for axes to sum over. Excludes the final (class) dimension.

        true_positives = tf.math.reduce_sum(y_pred * y_true, axis=sum_over_axes)
        false_negatives = tf.math.reduce_sum(y_pred_neg * y_true, axis=sum_over_axes)
        false_positives = tf.math.reduce_sum(y_pred * y_true_neg, axis=sum_over_axes)

        if class_weights is not None:
            true_positives *= class_weights
            false_positives *= class_weights
            false_negatives *= class_weights

        csi = tf.math.divide(tf.math.reduce_sum(true_positives),
            tf.math.reduce_sum(true_positives) + tf.math.reduce_sum(false_positives) + tf.math.reduce_sum(false_negatives))

        return 1 - csi

    return csi_loss


def fractions_skill_score(mask_size: int | tuple[int, ...] | list[int, ...] = (3, 3),
                          alpha: int | float = 1.0,
                          beta: int | float = 0.5,
                          class_weights: list[int | float, ...] = None):
    """
    Fractions skill score loss function.

    Parameters
    ----------
    mask_size: int or tuple
        Size of the mask/pool in the AveragePooling layers.
    alpha: int or float
        Parameter that controls how steep the sigmoid function is for discretization. Higher alpha makes the sigmoid function
            steeper and can help prevent the training process from stalling. Default value is 1. Values greater than 4 are
            not recommended.
    beta: int or float
        Parameter used to control some behaviors of the sigmoid discretization function. Default and recommended value is 0.5.
    class_weights: list of values or None
            List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.

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

    if class_weights is not None:
        class_weights = tf.cast(class_weights, tf.float32)

    @tf.function
    def fss_loss(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """

        # discretize model predictions and labels
        y_true = tf.math.sigmoid(alpha * (y_true - beta))
        y_pred = tf.math.sigmoid(alpha * (y_pred - beta))

        if class_weights is not None:
            y_true *= class_weights
            y_pred *= class_weights

        O_n = pool(y_true)  # observed fractions (Eq. 2 in RL2008)
        M_n = pool(y_pred)  # model forecast fractions (Eq. 3 in RL2008)

        MSE_n = tf.keras.metrics.mean_squared_error(O_n, M_n)  # MSE for model forecast fractions (Eq. 5 in RL2008)
        MSE_ref = tf.reduce_mean(tf.square(O_n)) + tf.reduce_mean(tf.square(M_n))  # reference forecast (Eq. 7 in RL2008)

        FSS = 1 - MSE_n / (MSE_ref + 1e-10)  # fractions skill score (Eq. 6 in RL2008)

        return 1 - FSS

    return fss_loss


def probability_of_detection(class_weights: list[int | float, ...] = None):
    """
    Probability of Detection (POD) as a loss function. This turns the function into the miss rate.

    class_weights: list of values or None
        List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.

    NOTE: This function is only intended for use in permutation studies and should NOT be used to train models.
    """

    @tf.function
    def pod_loss(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """

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