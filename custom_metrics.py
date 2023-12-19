"""
Custom metrics for U-Net models.
    - Brier Skill Score (BSS)
    - Critical Success Index (CSI)
    - Fractions Skill Score (FSS)
    - Probability of Detection (POD)

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.12.19
"""
import tensorflow as tf


def brier_skill_score(class_weights: list[int | float] = None):
    """
    Brier skill score (BSS).

    class_weights: list of values or None
        List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.
    """

    @tf.function
    def bss(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """

        squared_errors = tf.math.square(tf.subtract(y_true, y_pred))

        if class_weights is not None:
            relative_class_weights = tf.cast(class_weights / tf.math.reduce_sum(class_weights), tf.float32)
            squared_errors *= relative_class_weights

        bss = 1 - tf.math.reduce_sum(squared_errors) / tf.size(squared_errors)

        return bss

    return bss


def critical_success_index(threshold: float = None,
                           class_weights: list[int | float] = None,
                           window_size: int = None):
    """
    Critical success index (CSI).

    threshold: float or None
        Optional probability threshold that binarizes y_pred. Values in y_pred greater than or equal to the threshold are
            set to 1, and 0 otherwise.
        If the threshold is set, it must be greater than 0 and less than 1.
    class_weights: list of values or None
        List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.
    window_size: int or None
        Pool/kernel size of the max-pooling window for neighborhood statistics. (e.g. if calculating the CSI with a 4-pixel
            window, this should be set to 4).
        Note that this parameter is experimental and may return unexpected results.
    """

    @tf.function
    def csi(y_true, y_pred):
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
            y_pred = tf.where(y_pred >= threshold, 1., 0.)

        y_pred_neg = 1 - y_pred
        y_true_neg = 1 - y_true

        sum_over_axes = tf.range(tf.rank(y_pred) - 1)  # Indices for axes to sum over. Excludes the final (class) dimension.

        true_positives = tf.math.reduce_sum(y_pred * y_true, axis=sum_over_axes)
        false_negatives = tf.math.reduce_sum(y_pred_neg * y_true, axis=sum_over_axes)
        false_positives = tf.math.reduce_sum(y_pred * y_true_neg, axis=sum_over_axes)

        if class_weights is not None:
            relative_class_weights = tf.cast(class_weights / tf.math.reduce_sum(class_weights), tf.float32)
            csi = tf.math.reduce_sum(tf.math.divide_no_nan(true_positives, true_positives + false_positives + false_negatives) * relative_class_weights)
        else:
            csi = tf.math.divide(tf.math.reduce_sum(true_positives), tf.math.reduce_sum(true_positives) + tf.math.reduce_sum(false_negatives) + tf.math.reduce_sum(false_positives))

        return csi

    return csi


def fractions_skill_score(
    num_dims: int,
    mask_size: int = 3,
    c: float = 1.,
    binary: bool = False,
    threshold: float = 0.5,
    class_weights: list[int | float] = None):
    """
    Fractions skill score loss function. Visit https://github.com/CIRA-ML/custom_loss_functions for documentation.

    Parameters
    ----------
    num_dims: int
        Number of dimensions for the mask.
    mask_size: int or tuple
        Size of the mask/pool in the AveragePooling layers.
    c: int or float
        C parameter in the sigmoid function. This will only be used if 'binary' is False.
    binary: bool
        Convert y_pred to binary values (0/1).
    threshold: float
        If binary is False, this threshold is used in the sigmoid function.
        If binary is True, this is the threshold used to convert y_pred to binary values (0/1).
    class_weights: list of values or None
        List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.

    Returns
    -------
    fractions_skill_score: float
        Fractions skill score.
    """

    pool_kwargs = {'pool_size': (mask_size, ) * num_dims,
                   'strides': (1, ) * num_dims,
                   'padding': 'valid'}

    if num_dims == 2:
        pool1 = tf.keras.layers.AveragePooling2D(**pool_kwargs)
        pool2 = tf.keras.layers.AveragePooling2D(**pool_kwargs)
    else:
        pool1 = tf.keras.layers.AveragePooling3D(**pool_kwargs)
        pool2 = tf.keras.layers.AveragePooling3D(**pool_kwargs)

    @tf.function
    def fss(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """

        if binary:
            y_true = tf.where(y_true > threshold, 1., 0.)
            y_pred = tf.where(y_pred > threshold, 1., 0.)
        else:
            y_true = tf.math.sigmoid(c * (y_true - threshold))
            y_pred = tf.math.sigmoid(c * (y_pred - threshold))

        y_true_density = pool1(y_true)
        n_density_pixels = tf.cast((tf.shape(y_true_density)[1] * tf.shape(y_true_density)[2]), tf.float32)

        y_pred_density = pool2(y_pred)

        if class_weights is None:
            MSE_n = tf.keras.metrics.mean_squared_error(y_true_density, y_pred_density)
        else:
            relative_class_weights = tf.cast(class_weights / tf.math.reduce_sum(class_weights), tf.float32)
            MSE_n = tf.reduce_mean(tf.math.square(y_true_density - y_pred_density) * relative_class_weights, axis=-1)

        O_n_squared_image = tf.keras.layers.Multiply()([y_true_density, y_true_density])
        O_n_squared_vector = tf.keras.layers.Flatten()(O_n_squared_image)
        O_n_squared_sum = tf.reduce_sum(O_n_squared_vector)

        M_n_squared_image = tf.keras.layers.Multiply()([y_pred_density, y_pred_density])
        M_n_squared_vector = tf.keras.layers.Flatten()(M_n_squared_image)
        M_n_squared_sum = tf.reduce_sum(M_n_squared_vector)

        MSE_n_ref = (O_n_squared_sum + M_n_squared_sum) / n_density_pixels

        epsilon = tf.keras.backend.epsilon()  # 1e-7, constant for numeric stability

        if binary:
            if MSE_n_ref == 0:
                return 1 - MSE_n
            else:
                return 1 - (MSE_n / MSE_n_ref)
        else:
            return 1 - (MSE_n / (MSE_n_ref + epsilon))

    return fss


def probability_of_detection(threshold: float = None,
                             class_weights: list[int | float] = None,
                             window_size: int = None):
    """
    Probability of Detection (POD).

    threshold: float or None
        Optional probability threshold that binarizes y_pred. Values in y_pred greater than or equal to the threshold are
            set to 1, and 0 otherwise.
        If the threshold is set, it must be greater than 0 and less than 1.
    class_weights: list of values or None
        List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.
    window_size: int or None
        Pool/kernel size of the max-pooling window for neighborhood statistics. (e.g. if calculating the POD with a 4-pixel
            window, this should be set to 4).
        Note that this parameter is experimental and may return unexpected results.
    """

    @tf.function
    def pod(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """

        if window_size is not None:
            y_pred = tf.nn.max_pool(y_pred, ksize=window_size, strides=1, padding="VALID")
            y_true = tf.nn.max_pool(y_true, ksize=window_size, strides=1, padding="VALID")

        y_pred = tf.where(y_pred >= threshold, 1., 0.) if threshold is not None else y_pred
        y_pred_neg = 1 - y_pred

        sum_over_axes = tf.range(tf.rank(y_pred) - 1)  # Indices for axes to sum over. Excludes the final (class) dimension.

        true_positives = tf.math.reduce_sum(y_pred * y_true, axis=sum_over_axes)
        false_negatives = tf.math.reduce_sum(y_pred_neg * y_true, axis=sum_over_axes)

        if class_weights is not None:
            relative_class_weights = tf.cast(class_weights / tf.math.reduce_sum(class_weights), tf.float32)
            pod = tf.math.reduce_sum(tf.math.divide_no_nan(true_positives, true_positives + false_negatives) * relative_class_weights)
        else:
            pod = tf.math.reduce_sum(tf.math.divide_no_nan(true_positives, true_positives + false_negatives))

        return pod

    return pod