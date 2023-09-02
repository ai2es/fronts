"""
Custom loss functions for U-Net models.
    - Brier Skill Score (BSS)
    - Critical Success Index (CSI)
    - Fractions Skill Score (FSS)

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.5.20
"""
import tensorflow as tf


def brier_skill_score(class_weights=None):
    """
    Brier skill score (BSS) loss function.

    class_weights: list of values or None
        - List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.
    """

    @tf.function
    def bss_loss(y_true, y_pred):
        """
        y_true: tf.Tensor
            - One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            - Tensor containing model predictions.
        """

        losses = tf.math.square(tf.subtract(y_true, y_pred))

        if class_weights is not None:
            relative_class_weights = tf.cast(class_weights / tf.math.reduce_sum(class_weights), tf.float32)
            losses *= relative_class_weights

        brier_score_loss = tf.math.reduce_sum(losses) / tf.size(losses)
        return brier_score_loss

    return bss_loss


def critical_success_index(threshold=None, class_weights=None):
    """
    Critical Success Index (CSI) loss function.

    y_true: tf.Tensor
        - One-hot encoded tensor containing labels.
    y_pred: tf.Tensor
        - Tensor containing model predictions.
    threshold: float or None
        - Optional probability threshold that binarizes y_pred. Values in y_pred greater than or equal to the threshold are
            set to 1, and 0 otherwise.
        - If the threshold is set, it must be greater than 0 and less than 1.
    class_weights: list of values or None
        - List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.
    """

    @tf.function
    def csi_loss(y_true, y_pred):
        """
        y_true: tf.Tensor
            - One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            - Tensor containing model predictions.
        """

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
            csi = tf.math.reduce_sum(tf.math.divide_no_nan(true_positives, true_positives + false_positives + false_negatives) * relative_class_weights)
        else:
            csi = tf.math.divide(tf.math.reduce_sum(true_positives), tf.math.reduce_sum(true_positives) + tf.math.reduce_sum(false_negatives) + tf.math.reduce_sum(false_positives))

        return 1 - csi

    return csi_loss


def fractions_skill_score(num_dims, mask_size=3, c=1.0, cutoff=0.5, want_hard_discretization=False, class_weights=None):
    """
    Fractions skill score loss function. Visit https://github.com/CIRA-ML/custom_loss_functions for documentation.

    Parameters
    ----------
    num_dims: int
        - Number of dimensions for the mask.
    mask_size: int or tuple
        - Size of the mask/pool in the AveragePooling layers.
    c: int or float
        - C parameter in the sigmoid function. This will only be used if 'want_hard_discretization' is False.
    cutoff: float
        - If 'want_hard_discretization' is True, y_true and y_pred will be discretized to only have binary values (0/1)
    want_hard_discretization: bool
        - If True, y_true and y_pred will be discretized to only have binary values (0/1).
        - If False, y_true and y_pred will be discretized using a sigmoid function.
    class_weights: list of values or None
        - List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.

    Returns
    -------
    fractions_skill_score: float
        - Fractions skill score.
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
    def fss_loss(y_true, y_pred):
        """
        y_true: tf.Tensor
            - One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            - Tensor containing model predictions.
        """

        if want_hard_discretization:
            y_true_binary = tf.where(y_true > cutoff, 1.0, 0.0)
            y_pred_binary = tf.where(y_pred > cutoff, 1.0, 0.0)
        else:
            y_true_binary = tf.math.sigmoid(c * (y_true - cutoff))
            y_pred_binary = tf.math.sigmoid(c * (y_pred - cutoff))

        y_true_density = pool1(y_true_binary)
        n_density_pixels = tf.cast((tf.shape(y_true_density)[1] * tf.shape(y_true_density)[2]), tf.float32)

        y_pred_density = pool2(y_pred_binary)

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

        my_epsilon = tf.keras.backend.epsilon()  # this is 10^(-7)

        if want_hard_discretization:
            if MSE_n_ref == 0:
                return MSE_n
            else:
                return MSE_n / MSE_n_ref
        else:
            return MSE_n / (MSE_n_ref + my_epsilon)

    return fss_loss
