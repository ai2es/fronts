"""
Custom activation functions

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 4/16/2022 6:53 PM CDT

Known bugs:
- none

Please report any bugs to Andrew Justin: andrewjustinwx@gmail.com
"""
from tensorflow.keras.layers import Layer
import tensorflow as tf


class SmeLU(Layer):
    """
    SmeLU (Smooth ReLU) activation function layer for deep learning models.
    https://arxiv.org/pdf/2202.06499.pdf
    """
    def __init__(self, name=None):
        super(SmeLU, self).__init__(name=name)

    def build(self, input_shape):
        """ Build the SmeLU layer """
        self.beta = self.add_weight(name='beta', dtype='float32', shape=input_shape[1:])  # Learnable parameter (see Eq. 7 in the linked paper above)

    def call(self, inputs, **kwargs):
        """ Call the SmeLU activation function """
        inputs = tf.cast(inputs, 'float32')
        y = tf.where(inputs <= -self.beta, 0.0,  # Condition 1
            tf.where(tf.abs(inputs) <= self.beta, tf.math.divide(tf.math.pow(inputs + self.beta, 2.0), tf.math.multiply(4.0, self.beta)),  # Condition 2
            inputs))  # Condition 3 (if x >= beta)

        return y
