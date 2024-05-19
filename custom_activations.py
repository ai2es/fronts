"""
Custom activation function layers:
    * Elliott
    * Gaussian
    * Growing cosine unit (GCU)
    * Hexpo
    * Improved sigmoid units (ISigmoid)
    * Linearly-scaled hyperbolic tangent (LiSHT)
    * Parametric sigmoid (PSigmoid)
    * Parametric hyperbolic tangent (PTanh)
    * Parametric tangent hyperbolic linear unit (PTELU)
    * Rectified hyperbolic secant (ReSech)
    * Smooth rectified linear unit (SmeLU)
    * Snake
    * Soft-root-sign (SRS)
    * Scaled hyperbolic tangent (STanh)

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.5.18

TODO: thoroughly test all activation functions
"""
from tensorflow.keras.layers import Layer
import tensorflow as tf


class Elliott(Layer):
    """
    Elliott activation layer.

    References
    ----------
    https://link.springer.com/article/10.1007/s00521-017-3210-6
    """
    def __init__(self, name=None, **kwargs):
        super(Elliott, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        """ Build the Elliott activation layer """

    def call(self, inputs):
        inputs = tf.cast(inputs, 'float32')
        y = 0.5 * inputs / (1. + tf.abs(inputs)) + 0.5

        return y


class Gaussian(Layer):
    """
    Gaussian function activation layer.
    """
    def __init__(self, name=None, **kwargs):
        super(Gaussian, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        """ Build the Gaussian layer """

    def call(self, inputs):
        """ Call the Gaussian activation function """
        inputs = tf.cast(inputs, 'float32')
        y = tf.exp(-tf.square(inputs))

        return y


class GCU(Layer):
    """
    Growing Cosine Unit (GCU) activation layer.
    """
    def __init__(self, name=None, **kwargs):
        super(GCU, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        """ Build the GCU layer """

    def call(self, inputs):
        """ Call the GCU activation function """
        inputs = tf.cast(inputs, 'float32')
        y = inputs * tf.cos(inputs)

        return y


class Hexpo(Layer):
    """
    Hexpo activation layer.

    References
    ----------
    https://ieeexplore.ieee.org/document/7966168

    Notes
    -----
    When referencing the above paper, we name the parameters the following (paper -> our name):
        a -> alpha
        b -> beta
        c -> gamma
        d -> delta


    """
    def __init__(self,
                 name=None,
                 alpha_initializer=None,
                 alpha_regularizer=None,
                 alpha_constraint=None,
                 beta_initializer="Ones",
                 beta_regularizer=None,
                 beta_constraint="NonNeg",
                 gamma_initializer=None,
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 delta_initializer="Ones",
                 delta_regularizer=None,
                 delta_constraint="NonNeg",
                 shared_axes=None,
                 **kwargs):
        super(Hexpo, self).__init__(name=name, **kwargs)
        self._name = name
        self.alpha_initializer = alpha_initializer
        self.alpha_regularizer = alpha_regularizer
        self.alpha_constraint = alpha_constraint
        self.beta_initializer = beta_initializer
        self.beta_regularizer = beta_regularizer
        self.beta_constraint = beta_constraint
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint
        self.delta_initializer = delta_initializer
        self.delta_regularizer = delta_regularizer
        self.delta_constraint = delta_constraint
        self.shared_axes = shared_axes

    def build(self, input_shape):
        """ Build the PSigmoid layer """
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for ax in self.shared_axes:
                param_shape[ax - 1] = 1
        else:
            # Turn all arbitrary dimensions (denoted by None) into size 1
            for ax in range(len(param_shape)):
                param_shape[ax] = 1 if param_shape[ax] is None else param_shape[ax]

        # learnable parameter
        self.alpha = self.add_weight(name="alpha",
                                     shape=param_shape,
                                     initializer=self.alpha_initializer,
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)
        self.beta = self.add_weight(name="beta",
                                    shape=param_shape,
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint)
        self.gamma = self.add_weight(name="gamma",
                                     shape=param_shape,
                                     initializer=self.gamma_initializer,
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)
        self.delta = self.add_weight(name="delta",
                                     shape=param_shape,
                                     initializer=self.delta_initializer,
                                     regularizer=self.delta_regularizer,
                                     constraint=self.delta_constraint)

    def call(self, inputs):
        inputs = tf.cast(inputs, 'float32')
        y = tf.where(inputs >= 0., -self.alpha * (tf.exp(-inputs / (self.beta + 1e-7)) - 1.),
                     self.gamma * (tf.exp(inputs / (self.delta + 1e-7)) - 1.))

        return y

    def get_config(self):
        config = super().get_config()
        config.update({"name": self._name,
                       "alpha_initializer": self.alpha_initializer,
                       "alpha_regularizer": self.alpha_regularizer,
                       "alpha_constraint": self.alpha_constraint,
                       "beta_initializer": self.beta_initializer,
                       "beta_regularizer": self.beta_regularizer,
                       "beta_constraint": self.beta_constraint,
                       "gamma_initializer": self.gamma_initializer,
                       "gamma_regularizer": self.gamma_regularizer,
                       "gamma_constraint": self.gamma_constraint,
                       "delta_initializer": self.delta_initializer,
                       "delta_regularizer": self.delta_regularizer,
                       "delta_constraint": self.delta_constraint,
                       "shared_axes": self.shared_axes})

        return config


class ISigmoid(Layer):
    """
    Trainable version of the ISigmoid activation function layer.

    References
    ----------
    https://ieeexplore.ieee.org/document/8415753

    Notes
    -----
    Parameter 'a' in the paper referenced above will be called 'beta' in this layer.
    """
    def __init__(self,
                 name=None,
                 alpha_initializer="zeros",
                 alpha_regularizer=None,
                 alpha_constraint=None,
                 beta_initializer="zeros",
                 beta_regularizer=None,
                 beta_constraint=None,
                 shared_axes=None,
                 **kwargs):
        super(ISigmoid, self).__init__(name=name, **kwargs)
        self.alpha_initializer = alpha_initializer
        self.alpha_regularizer = alpha_regularizer
        self.alpha_constraint = alpha_constraint
        self.beta_initializer = beta_initializer
        self.beta_regularizer = beta_regularizer
        self.beta_constraint = beta_constraint
        self.shared_axes = shared_axes

    def build(self, input_shape):
        """ Build the ISigmoid layer """
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for ax in self.shared_axes:
                param_shape[ax - 1] = 1
        else:
            # Turn all arbitrary dimensions (denoted by None) into size 1
            for ax in range(len(param_shape)):
                param_shape[ax] = 1 if param_shape[ax] is None else param_shape[ax]

        self.alpha = self.add_weight(name="alpha",
                                     shape=param_shape,
                                     initializer=self.alpha_initializer,
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)
        self.beta = self.add_weight(name="beta",
                                    shape=param_shape,
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint)

    def call(self, inputs):
        inputs = tf.cast(inputs, 'float32')
        y = tf.where(inputs >= self.beta, (self.alpha * (inputs - self.beta)) + tf.sigmoid(self.beta),
            tf.where(inputs <= -self.beta, (self.alpha * (inputs + self.beta)) + tf.sigmoid(self.beta),
            tf.sigmoid(inputs)))

        return y

    def get_config(self):
        config = super().get_config()
        config.update({"name": self._name,
                       "alpha_initializer": self.alpha_initializer,
                       "alpha_regularizer": self.alpha_regularizer,
                       "alpha_constraint": self.alpha_constraint,
                       "beta_initializer": self.beta_initializer,
                       "beta_regularizer": self.beta_regularizer,
                       "beta_constraint": self.beta_constraint,
                       "shared_axes": self.shared_axes})

        return config


class LiSHT(Layer):
    """
    Linearly-scaled hyperbolic tangent activation layer.
    
    References
    ----------
    https://arxiv.org/abs/1901.05894
    """
    def __init__(self, name=None, **kwargs):
        super(LiSHT, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        """ Build the LiSHT layer """

    def call(self, inputs):
        inputs = tf.cast(inputs, 'float32')
        y = inputs * tf.tanh(inputs)

        return y


class PSigmoid(Layer):
    """
    Parametric sigmoid activation layer.
    """
    def __init__(self,
                 name=None,
                 alpha_initializer="zeros",
                 alpha_regularizer=None,
                 alpha_constraint=None,
                 shared_axes=None,
                 **kwargs):
        super(PSigmoid, self).__init__(name=name, **kwargs)
        self.alpha_initializer = alpha_initializer
        self.alpha_regularizer = alpha_regularizer
        self.alpha_constraint = alpha_constraint
        self.shared_axes = shared_axes

    def build(self, input_shape):
        """ Build the PSigmoid layer """
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for ax in self.shared_axes:
                param_shape[ax - 1] = 1
        else:
            # Turn all arbitrary dimensions (denoted by None) into size 1
            for ax in range(len(param_shape)):
                param_shape[ax] = 1 if param_shape[ax] is None else param_shape[ax]

        self.alpha = self.add_weight(name="alpha",
                                     shape=param_shape,
                                     initializer=self.alpha_initializer,
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)  # learnable parameter

    def call(self, inputs):
        inputs = tf.cast(inputs, 'float32')
        y = tf.sigmoid(inputs) ** self.alpha

        return y

    def get_config(self):
        config = super().get_config()
        config.update({"name": self._name,
                       "alpha_initializer": self.alpha_initializer,
                       "alpha_regularizer": self.alpha_regularizer,
                       "alpha_constraint": self.alpha_constraint,
                       "shared_axes": self.shared_axes})

        return config


class PTanh(Layer):
    """
    Penalized hyperbolic tangent (PTanh) activation layer.
    """
    def __init__(self,
                 name=None,
                 alpha_initializer="zeros",
                 alpha_regularizer=None,
                 alpha_constraint="MinMaxNorm",  # by default, alpha is restricted to the (0, 1) range
                 shared_axes=None,
                 **kwargs):
        super(PTanh, self).__init__(name=name, **kwargs)
        self.alpha_initializer = alpha_initializer
        self.alpha_regularizer = alpha_regularizer
        self.alpha_constraint = alpha_constraint
        self.shared_axes = shared_axes

    def build(self, input_shape):
        """ Build the PTanh layer """
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for ax in self.shared_axes:
                param_shape[ax - 1] = 1
        else:
            # Turn all arbitrary dimensions (denoted by None) into size 1
            for ax in range(len(param_shape)):
                param_shape[ax] = 1 if param_shape[ax] is None else param_shape[ax]

        self.alpha = self.add_weight(name="alpha",
                                     shape=param_shape,
                                     initializer=self.alpha_initializer,
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)  # learnable parameter

    def call(self, inputs):
        inputs = tf.cast(inputs, 'float32')
        y = tf.where(inputs >= 0., tf.tanh(inputs), self.alpha * tf.tanh(inputs))

        return y

    def get_config(self):
        config = super().get_config()
        config.update({"name": self._name,
                       "alpha_initializer": self.alpha_initializer,
                       "alpha_regularizer": self.alpha_regularizer,
                       "alpha_constraint": self.alpha_constraint,
                       "shared_axes": self.shared_axes})

        return config


class PTELU(Layer):
    """
    Parametric tangent hyperbolic linear unit activation layer.

    References
    ----------
    https://ieeexplore.ieee.org/document/8265328
    """
    def __init__(self,
                 name=None,
                 alpha_initializer="zeros",
                 alpha_regularizer=None,
                 alpha_constraint="NonNeg",
                 beta_initializer="zeros",
                 beta_regularizer=None,
                 beta_constraint="NonNeg",
                 shared_axes=None,
                 **kwargs):
        super(PTELU, self).__init__(name=name, **kwargs)
        self.alpha_initializer = alpha_initializer
        self.alpha_regularizer = alpha_regularizer
        self.alpha_constraint = alpha_constraint
        self.beta_initializer = beta_initializer
        self.beta_regularizer = beta_regularizer
        self.beta_constraint = beta_constraint
        self.shared_axes = shared_axes

    def build(self, input_shape):
        """ Build the PTELU layer """
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for ax in self.shared_axes:
                param_shape[ax - 1] = 1
        else:
            # Turn all arbitrary dimensions (denoted by None) into size 1
            for ax in range(len(param_shape)):
                param_shape[ax] = 1 if param_shape[ax] is None else param_shape[ax]

        # learnable parameter
        self.alpha = self.add_weight(name="alpha",
                                     shape=param_shape,
                                     initializer=self.alpha_initializer,
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)
        self.beta = self.add_weight(name="beta",
                                    shape=param_shape,
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint)

    def call(self, inputs):
        inputs = tf.cast(inputs, 'float32')
        y = tf.where(inputs >= 0., inputs, self.alpha * tf.tanh(self.beta * inputs))

        return y

    def get_config(self):
        config = super().get_config()
        config.update({"name": self._name,
                       "alpha_initializer": self.alpha_initializer,
                       "alpha_regularizer": self.alpha_regularizer,
                       "alpha_constraint": self.alpha_constraint,
                       "beta_initializer": self.beta_initializer,
                       "beta_regularizer": self.beta_regularizer,
                       "beta_constraint": self.beta_constraint,
                       "shared_axes": self.shared_axes})

        return config


class ReSech(Layer):
    """
    Rectified hyperbolic secant (ReSech) activation layer.
    """
    def __init__(self, name=None, **kwargs):
        super(ReSech, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        """ Build the ReSech layer """

    def call(self, inputs):
        inputs = tf.cast(inputs, 'float32')
        y = inputs / tf.cosh(inputs)

        return y


class SmeLU(Layer):
    """
    SmeLU (Smooth ReLU) activation function layer for deep learning models.

    References
    ----------
    https://arxiv.org/pdf/2202.06499.pdf
    """
    def __init__(self,
                 name=None,
                 beta_initializer="ones",
                 beta_regularizer=None,
                 beta_constraint="NonNeg",
                 shared_axes=None,
                 **kwargs):
        super(SmeLU, self).__init__(name=name, **kwargs)
        self._name = name
        self.beta_initializer = beta_initializer
        self.beta_regularizer = beta_regularizer
        self.beta_constraint = beta_constraint
        self.shared_axes = shared_axes

    def build(self, input_shape):
        """ Build the SmeLU layer """
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for ax in self.shared_axes:
                param_shape[ax - 1] = 1
        else:
            # Turn all abritrary dimensions (denoted by None) into size 1
            for ax in range(len(param_shape)):
                param_shape[ax] = 1 if param_shape[ax] is None else param_shape[ax]

        self.beta = self.add_weight(name="beta",
                                    shape=param_shape,
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint)  # Learnable parameter (see Eq. 7 in the linked paper above)

    def call(self, inputs):
        """ Call the SmeLU activation function """
        inputs = tf.cast(inputs, 'float32')
        y = tf.where(inputs <= -self.beta, 0.,  # Condition 1
            tf.where(tf.abs(inputs) <= self.beta, tf.square(inputs + self.beta) / (4. * self.beta),  # Condition 2
            inputs))  # Condition 3 (if x >= beta)

        return y

    def get_config(self):
        config = super().get_config()
        config.update({"name": self._name,
                       "beta_initializer": self.beta_initializer,
                       "beta_regularizer": self.beta_regularizer,
                       "beta_constraint": self.beta_constraint,
                       "shared_axes": self.shared_axes})

        return config


class Snake(Layer):
    """
    Snake activation function layer for deep learning models.

    References
    ----------
    https://arxiv.org/pdf/2006.08195.pdf
    """
    def __init__(self,
                 name=None,
                 alpha_initializer="Ones",
                 alpha_regularizer=None,
                 alpha_constraint="NonNeg",
                 shared_axes=None,
                 **kwargs):
        super(Snake, self).__init__(name=name, **kwargs)
        self.alpha_initializer = alpha_initializer
        self.alpha_regularizer = alpha_regularizer
        self.alpha_constraint = alpha_constraint
        self.shared_axes = shared_axes

    def build(self, input_shape):
        """ Build the Snake layer """
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for ax in self.shared_axes:
                param_shape[ax - 1] = 1
        else:
            # Turn all arbitrary dimensions (denoted by None) into size 1
            for ax in range(len(param_shape)):
                param_shape[ax] = 1 if param_shape[ax] is None else param_shape[ax]

        # Learnable parameter (see Eq. 3 in the linked paper above)
        self.alpha = self.add_weight(name='alpha',
                                     shape=param_shape,
                                     initializer=self.alpha_initializer,
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)

    def call(self, inputs):
        """ Call the Snake activation function """
        inputs = tf.cast(inputs, 'float32')
        y = inputs + ((1. / (self.alpha + 1e-7)) * tf.square(tf.sin(self.alpha * inputs)))

        return y

    def get_config(self):
        config = super().get_config()
        config.update({"name": self._name,
                       "alpha_initializer": self.alpha_initializer,
                       "alpha_regularizer": self.alpha_regularizer,
                       "alpha_constraint": self.alpha_constraint,
                       "shared_axes": self.shared_axes})

        return config


class SRS(Layer):
    """
    Soft-Root-Sign activation layer.

    References
    ----------
    https://arxiv.org/abs/2003.00547
    """
    def __init__(self,
                 name=None,
                 alpha_initializer="ones",
                 alpha_regularizer=None,
                 alpha_constraint="NonNeg",
                 beta_initializer="ones",
                 beta_regularizer=None,
                 beta_constraint="NonNeg",
                 shared_axes=None,
                 **kwargs):
        super(SRS, self).__init__(name=name, **kwargs)
        self.alpha_initializer = alpha_initializer
        self.alpha_regularizer = alpha_regularizer
        self.alpha_constraint = alpha_constraint
        self.beta_initializer = beta_initializer
        self.beta_regularizer = beta_regularizer
        self.beta_constraint = beta_constraint
        self.shared_axes = shared_axes

    def build(self, input_shape):
        """ Build the SRS layer """
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for ax in self.shared_axes:
                param_shape[ax - 1] = 1
        else:
            # Turn all arbitrary dimensions (denoted by None) into size 1
            for ax in range(len(param_shape)):
                param_shape[ax] = 1 if param_shape[ax] is None else param_shape[ax]

        # learnable parameter
        self.alpha = self.add_weight(name="alpha",
                                     shape=param_shape,
                                     initializer=self.alpha_initializer,
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)
        self.beta = self.add_weight(name="beta",
                                    shape=param_shape,
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint)

    def call(self, inputs):
        inputs = tf.cast(inputs, 'float32')
        y = inputs / ((inputs / self.alpha) + tf.exp(-inputs / self.beta))

        return y

    def get_config(self):
        config = super().get_config()
        config.update({"name": self._name,
                       "alpha_initializer": self.alpha_initializer,
                       "alpha_regularizer": self.alpha_regularizer,
                       "alpha_constraint": self.alpha_constraint,
                       "beta_initializer": self.beta_initializer,
                       "beta_regularizer": self.beta_regularizer,
                       "beta_constraint": self.beta_constraint,
                       "shared_axes": self.shared_axes})

        return config


class STanh(Layer):
    """
    Scaled hyperbolic tangent function.

    References
    ----------
    https://ieeexplore.ieee.org/document/726791
    """
    def __init__(self, name=None, **kwargs):
        super(STanh, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        """ Build the STanh layer """

    def call(self, inputs):
        """ Call the STanh activation function """
        inputs = tf.cast(inputs, 'float32')
        y = 1.7159 * tf.tanh((2/3) * inputs)

        return y