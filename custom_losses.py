"""
Custom loss functions for U-Net models.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 8/13/2021 6:20 PM CDT
"""

import tensorflow as tf
from keras_unet_collection import losses


def dice():
    return losses.dice


# Fraction Skill Score (FSS) Loss Function - code taken from: https://github.com/CIRA-ML/custom_loss_functions
# Fraction Skill Score original paper: N.M. Roberts and H.W. Lean, "Scale-Selective Verification of Rainfall
#     Accumulation from High-Resolution Forecasts of Convective Events", Monthly Weather Review, 2008.

def make_FSS_loss_2D(mask_size):  # choose any mask size for calculating densities

    @tf.function()
    def FSS_loss_2D(y_true, y_pred):

        want_hard_discretization = False

        cutoff = 0.5

        if want_hard_discretization:
            y_true_binary = tf.where(y_true > cutoff, 1.0, 0.0)
            y_pred_binary = tf.where(y_pred > cutoff, 1.0, 0.0)

        else:
            c = 1
            y_true_binary = tf.math.sigmoid(c * (y_true - cutoff))
            y_pred_binary = tf.math.sigmoid(c * (y_pred - cutoff))

        pool1 = tf.keras.layers.AveragePooling2D(pool_size=(mask_size, mask_size), strides=(1, 1),
                                                 padding='valid')
        y_true_density = pool1(y_true_binary)
        n_density_pixels = tf.cast((tf.shape(y_true_density)[1] * tf.shape(y_true_density)[2]),
                                   tf.float32)

        pool2 = tf.keras.layers.AveragePooling2D(pool_size=(mask_size, mask_size),
                                                 strides=(1, 1), padding='valid')
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

    return FSS_loss_2D

def make_FSS_loss_3D(mask_size):  # choose any mask size for calculating densities

    @tf.function()
    def FSS_loss_3D(y_true, y_pred):

        want_hard_discretization = False

        cutoff = 0.5

        if want_hard_discretization:
            y_true_binary = tf.where(y_true > cutoff, 1.0, 0.0)
            y_pred_binary = tf.where(y_pred > cutoff, 1.0, 0.0)

        else:
            c = 1
            y_true_binary = tf.math.sigmoid(c * (y_true - cutoff))
            y_pred_binary = tf.math.sigmoid(c * (y_pred - cutoff))

        pool1 = tf.keras.layers.AveragePooling3D(pool_size=(mask_size, mask_size, mask_size), strides=(1, 1, 1),
                                                 padding='valid')
        y_true_density = pool1(y_true_binary)
        n_density_pixels = tf.cast((tf.shape(y_true_density)[1] * tf.shape(y_true_density)[2]),
                                   tf.float32)

        pool2 = tf.keras.layers.AveragePooling3D(pool_size=(mask_size, mask_size, mask_size),
                                                 strides=(1, 1, 1), padding='valid')
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

    return FSS_loss_3D

def tversky():
    return losses.tversky

def brier_skill_score(y_true, y_pred):
    """
    Computes brier skill score
    """
    losses = tf.subtract(y_true, y_pred)**2
    brier_score = tf.math.reduce_sum(losses)/tf.cast(tf.size(losses), tf.float32)
    return brier_score
