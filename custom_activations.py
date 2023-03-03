"""
Custom activation functions:
    - Gaussian
    - GCU (Growing Cosine Unit)
    - SmeLU (Smooth ReLU)
    - Snake

Code written by: Andrew Justin (andrewjustinwx@gmail.com)

Last updated: 3/3/2023 12:51 PM CT
"""
from tensorflow.keras.layers import Layer
import tensorflow as tf


class Gaussian(Layer):
    """
    Gaussian function activation layer.
    """
    def __init__(self, name=None):
        super(Gaussian, self).__init__(name=name)

    def build(self, input_shape):
        """ Build the Gaussian layer """

    def call(self, inputs):
        """ Call the Gaussian activation function """
        inputs = tf.cast(inputs, 'float32')
        square_tensor = tf.constant(2.0, shape=inputs.shape[1:])
        y = tf.math.exp(tf.math.negative(tf.math.pow(inputs, square_tensor)))

        return y


class GCU(Layer):
    """
    Growing Cosine Unit (GCU) activation layer.
    """
    def __init__(self, name=None):
        super(GCU, self).__init__(name=name)

    def build(self, input_shape):
        """ Build the GCU layer """

    def call(self, inputs):
        """ Call the GCU activation function """
        inputs = tf.cast(inputs, 'float32')
        y = tf.multiply(inputs, tf.math.cos(inputs))

        return y


class SmeLU(Layer):
    """
    SmeLU (Smooth ReLU) activation function layer for deep learning models.

    References
    ----------
    https://arxiv.org/pdf/2202.06499.pdf
    """
    def __init__(self, name=None):
        super(SmeLU, self).__init__(name=name)

    def build(self, input_shape):
        """ Build the SmeLU layer """
        self.beta = self.add_weight(name='beta', dtype='float32', shape=input_shape[1:])  # Learnable parameter (see Eq. 7 in the linked paper above)

    def call(self, inputs):
        """ Call the SmeLU activation function """
        inputs = tf.cast(inputs, 'float32')
        y = tf.where(inputs <= -self.beta, 0.0,  # Condition 1
            tf.where(tf.abs(inputs) <= self.beta, tf.math.divide(tf.math.pow(inputs + self.beta, 2.0), tf.math.multiply(4.0, self.beta)),  # Condition 2
            inputs))  # Condition 3 (if x >= beta)

        return y


class Snake(Layer):
    """
    Snake activation function layer for deep learning models.

    References
    ----------
    https://arxiv.org/pdf/2006.08195.pdf
    """
    def __init__(self, name=None):
        super(Snake, self).__init__(name=name)

    def build(self, input_shape):
        """ Build the Snake layer """
        self.alpha = self.add_weight(name='alpha', dtype='float32', shape=input_shape[1:])  # Learnable parameter (see Eq. 3 in the linked paper above)
        self.square_tensor = tf.constant(2.0, shape=input_shape[1:])

    def call(self, inputs):
        """ Call the Snake activation function """
        inputs = tf.cast(inputs, 'float32')
        y = inputs + tf.multiply(tf.divide(tf.constant(1.0, shape=inputs.shape[1:]), self.alpha), tf.math.pow(tf.math.sin(tf.multiply(self.alpha, inputs)), self.square_tensor))

        return y

