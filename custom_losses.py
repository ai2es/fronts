"""
Custom loss functions for U-Net models.

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 4/10/2022 2:48 PM CST

Known bugs:
- none

Please report any bugs to Andrew Justin: andrewjustinwx@gmail.com

NOTE: Throughout this script, common parameters for loss functions include 'y_true' and 'y_pred'.
    - 'y_true' is the target that is trying to be predicted
    - 'y_pred' is the actual prediction made
"""

import tensorflow as tf


def brier_skill_score(y_true, y_pred):
    """ Brier skill score """
    losses = tf.subtract(y_true, y_pred)**2
    brier_score = tf.math.reduce_sum(losses)/tf.cast(tf.size(losses), tf.float32)
    return brier_score


# Fraction Skill Score (FSS) Loss Function - code taken from: https://github.com/CIRA-ML/custom_loss_functions
# Fraction Skill Score original paper: N.M. Roberts and H.W. Lean, "Scale-Selective Verification of Rainfall
#     Accumulation from High-Resolution Forecasts of Convective Events", Monthly Weather Review, 2008.
def make_fractions_skill_score(mask_size, num_dimensions, c=1.0, cutoff=0.5, strides=1, want_hard_discretization=False):
    """
    Make fractions skill score loss function. Visit https://github.com/CIRA-ML/custom_loss_functions for documentation.

    Parameters
    ----------
    mask_size: int or tuple
        - Size of the mask/pool in the AveragePooling layers.
    num_dimensions: int
        - Number of dimensions in the mask/pool in the AveragePooling layers.
    c: int or float
        - C parameter in the sigmoid function. This will only be used if 'want_hard_discretization' is False.
    cutoff: float
        - If 'want_hard_discretization' is True, y_true and y_pred will be discretized to only have binary values (0/1)
    strides: int or tuple
        - Strides in the AveragePooling layers.
    want_hard_discretization: bool
        - If True, y_true and y_pred will be discretized to only have binary values (0/1).
        - If False, y_true and y_pred will be discretized using a sigmoid function.

    Returns
    -------
    fractions_skill_score: float
        Fractions skill score.
    """

    pool_kwargs = {'pool_size': mask_size, 'strides': strides}
    if num_dimensions == 2:
        pool1 = tf.keras.layers.AveragePooling2D(**pool_kwargs)
        pool2 = tf.keras.layers.AveragePooling2D(**pool_kwargs)
    elif num_dimensions == 3:
        pool1 = tf.keras.layers.AveragePooling3D(**pool_kwargs)
        pool2 = tf.keras.layers.AveragePooling3D(**pool_kwargs)
    else:
        raise ValueError("Number of dimensions can only be 2 or 3")

    @tf.function()
    def fractions_skill_score(y_true, y_pred):
        """ Fractions skill score loss function """
        if want_hard_discretization:
            y_true_binary = tf.where(y_true > cutoff, 1.0, 0.0)
            y_pred_binary = tf.where(y_pred > cutoff, 1.0, 0.0)
        else:
            y_true_binary = tf.math.sigmoid(c * (y_true - cutoff))
            y_pred_binary = tf.math.sigmoid(c * (y_pred - cutoff))

        y_true_density = pool1(y_true_binary)
        n_density_pixels = tf.cast((tf.shape(y_true_density)[1] * tf.shape(y_true_density)[2]), tf.float32)

        y_pred_density = pool2(y_pred_binary)

        """
        Multi-GPU support
        
        This line was removed from the code:
        MSE_n = tf.keras.losses.MeanSquaredError()(y_true_density, y_pred_density)
        
        The line above was replaced with:
        MSE_n = tf.keras.metrics.mean_squared_error(y_true_density, y_pred_density)
        
        This replacement prevents a ValueError raised by TensorFlow, which is caused by a loss function being declared outside
        the 'strategy.scope()' loop (see train_model.py).
        """
        MSE_n = tf.keras.metrics.mean_squared_error(y_true_density, y_pred_density)

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

    return fractions_skill_score
