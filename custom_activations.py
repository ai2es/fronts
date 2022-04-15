"""
Custom activation functions

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 4/15/2022 12:26 PM CDT

Known bugs:
- none

Please report any bugs to Andrew Justin: andrewjustinwx@gmail.com
"""
import tensorflow as tf
tf.config.run_functions_eagerly(True)


@tf.function()
def smelu(x):
    """
    SmeLU (Smooth ReLU) activation function for deep learning models.
    https://arxiv.org/pdf/2202.06499.pdf
    """

    beta = tf.Variable(1.0, trainable=True)  # Learnable parameter (see Eq. (7) in the linked paper above)

    y = tf.where(x <= -beta, 0.0,  # Condition 1
        tf.where(tf.abs(x) <= beta, tf.math.divide(tf.math.pow(x + beta, 2.0), tf.math.multiply(4.0, beta)),  # Condition 2
        x))  # Condition 3 (if x >= beta)

    return y

