"""
Functions for building U-Net models:
- U-Net
- U-Net 3+

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 4/26/2022 10:16 PM CDT

Known bugs:
- none

Please report any bugs to Andrew Justin: andrewjustinwx@gmail.com
"""

import numpy as np
from tensorflow.keras.layers import Conv2D, Conv3D, BatchNormalization, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Softmax
from tensorflow.keras import layers
import custom_activations


def convolution_module(tensor, filters, kernel_size, num_modules=1, padding='same', use_bias=False, batch_normalization=True,
    activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
    activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name=None):
    """
    Insert modules into an encoder or decoder node.

    Parameters
    ----------
    tensor: tf.Tensor
        - Input tensor for the convolution module(s).
    filters: int
        - Number of filters in the Conv2D/Conv3D layer(s).
    kernel_size: int or tuple
        - Size of the kernel in the Conv2D/Conv3D layer(s).
    num_modules: int
        - Number of convolution modules to insert. Must be greater than 0, otherwise a ValueError exception is raised.
    padding: str
        - Padding in the Conv2D/Conv3D layer(s). 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        - If True, a bias vector will be used in the Conv2D/Conv3D layers.
    batch_normalization: bool
        - If True, a BatchNormalization layer will follow every Conv2D/Conv3D layer.
    activation: str
        - Activation function to use after every Conv2D/Conv3D layer (BatchNormalization layer, if batch_normalization is True).
        - Can be any of tf.keras.activations, 'prelu', 'leaky_relu', or 'smelu' (case-insensitive).
    kernel_initializer: str or tf.keras.initializers object
        - Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        - Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        - Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        - Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        - Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        - Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        - Constraint function applied to the bias vector in the Conv2D/Conv3D layers.
    name: str or None
        - Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        - Output tensor.
    """

    tensor_dims = len(tensor.shape)  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if num_modules < 1:
        raise ValueError("num_modules must be greater than 0, at least one module must be added")

    if tensor_dims == 4:  # A 2D image tensor has 4 dimensions: (None [for batch size], image_size_x, image_size_y, n_channels)
        conv_layer = Conv2D
    elif tensor_dims == 5:  # A 3D image tensor has 5 dimensions: (None [for batch size], image_size_x, image_size_y, image_size_z, n_channels)
        conv_layer = Conv3D
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions")

    # Arguments for the Conv2D/Conv3D layers.
    conv_kwargs = dict({})
    conv_kwargs['filters'] = filters
    conv_kwargs['kernel_size'] = kernel_size
    conv_kwargs['padding'] = padding
    conv_kwargs['use_bias'] = use_bias
    conv_kwargs['kernel_initializer'] = kernel_initializer
    conv_kwargs['bias_initializer'] = bias_initializer
    conv_kwargs['kernel_regularizer'] = kernel_regularizer
    conv_kwargs['bias_regularizer'] = bias_regularizer
    conv_kwargs['activity_regularizer'] = activity_regularizer
    conv_kwargs['kernel_constraint'] = kernel_constraint
    conv_kwargs['bias_constraint'] = bias_constraint

    activation_layer = choose_activation_layer(activation)  # Choose activation layer for the convolution modules.
    activation_str = activation

    """ 
    Arguments for the Activation layer(s), if applicable. 
    - If activation is 'prelu' or 'leaky_relu', a PReLU layer or a LeakyReLU layer will be used instead and this dictionary
      will have no effect.
    """
    activation_kwargs = dict({})

    if activation_layer == layers.Activation:
        if activation_str == 'smelu':
            activation_kwargs['activation'] = custom_activations.SmeLU  # SmeLU is a custom activation function, so the function itself must be passed into the Activation layer
        else:
            activation_kwargs['activation'] = activation

    # Insert convolution modules
    for module in range(num_modules):

        # Create names for the Conv2D/Conv3D layers and the activation layer.
        conv_kwargs['name'] = f'{name}_Conv{tensor_dims - 2}D_{module+1}'
        activation_kwargs['name'] = f'{name}_{activation_str}_{module+1}'

        conv_tensor = conv_layer(**conv_kwargs)(tensor)  # Perform convolution on the input tensor

        if batch_normalization is True:
            batch_norm_tensor = BatchNormalization(name=f'{name}_BatchNorm_{module+1}')(conv_tensor)  # Insert layer for batch normalization
            activation_tensor = activation_layer(**activation_kwargs)(batch_norm_tensor)  # Pass output tensor from BatchNormalization into the activation layer
        else:
            activation_tensor = activation_layer(**activation_kwargs)(conv_tensor)  # Pass output tensor from the convolution layer into the activation layer.

        tensor = activation_tensor

    return tensor


def aggregated_feature_map(tensor, filters, kernel_size, level1, level2, upsample_size, padding='same', use_bias=False,
    batch_normalization=True, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name=None):
    """
    Connect two nodes in the U-Net 3+ with an aggregated feature map (AFM).

    Parameters
    ----------
    tensor: tf.Tensor
        - Input tensor for the convolution module.
    filters: int
        - Number of filters in the Conv2D/Conv3D layer.
    kernel_size: int or tuple
        - Size of the kernel in the Conv2D/Conv3D layer.
    level1: int
        - Level of the first node that is connected to the AFM. This node will provide the input tensor to the AFM. Must be
        greater than level2 (i.e. the first node must be on a lower level in the U-Net 3+ since we are up-sampling), otherwise
        a ValueError exception is raised.
    level2: int
        - Level of the second node that is connected to the AFM. This node will receive the output of the AFM. Must be smaller
        than level1 (i.e. the second node must be on a higher level in the U-Net 3+ since we are up-sampling), otherwise
        a ValueError exception is raised.
    upsample_size: tuple or list
        - Upsampling size for rows and columns in the UpSampling2D/UpSampling3D layer.
    padding: str
        - Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        - If True, a bias vector will be used in the Conv2D/Conv3D layer.
    batch_normalization: bool
        - If True, a BatchNormalization layer will follow the Conv2D/Conv3D layer.
    activation: str
        - Activation function to use after the Conv2D/Conv3D layer (BatchNormalization layer, if batch_normalization is True).
        - Can be any of tf.keras.activations, 'prelu', 'leaky_relu', or 'smelu' (case-insensitive).
    kernel_initializer: str or tf.keras.initializers object
        - Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        - Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        - Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        - Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        - Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        - Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        - Constraint function applied to the bias vector in the Conv2D/Conv3D layers.
    name: str or None
        - Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        - Output tensor.
    """

    tensor_dims = len(tensor.shape)  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if level1 <= level2:
        raise ValueError("level2 must be smaller than level1 in aggregated feature maps")

    # Arguments for the convolution module.
    module_kwargs = dict({})
    module_kwargs['filters'] = filters
    module_kwargs['kernel_size'] = kernel_size
    module_kwargs['num_modules'] = 1
    module_kwargs['padding'] = padding
    module_kwargs['use_bias'] = use_bias
    module_kwargs['batch_normalization'] = batch_normalization
    module_kwargs['activation'] = activation
    module_kwargs['kernel_initializer'] = kernel_initializer
    module_kwargs['bias_initializer'] = bias_initializer
    module_kwargs['kernel_regularizer'] = kernel_regularizer
    module_kwargs['bias_regularizer'] = bias_regularizer
    module_kwargs['activity_regularizer'] = activity_regularizer
    module_kwargs['kernel_constraint'] = kernel_constraint
    module_kwargs['bias_constraint'] = bias_constraint
    module_kwargs['name'] = name

    # Keyword arguments for the UpSampling2D/UpSampling3D layers
    upsample_kwargs = dict({})
    upsample_kwargs['name'] = f'{name}_UpSampling{tensor_dims - 2}D'
    upsample_kwargs['size'] = np.power(upsample_size, abs(level1 - level2))

    if tensor_dims == 4:  # If the image is 2D
        upsample_layer = UpSampling2D
        if len(upsample_size) != 2:
            raise TypeError(f"For 2D up-sampling, the pool size must be a tuple or list with 2 integers. Received shape: {np.shape(upsample_size)}")
    elif tensor_dims == 5:  # If the image is 3D
        upsample_layer = UpSampling3D
        if len(upsample_size) != 3:
            raise TypeError(f"For 3D up-sampling, the pool size must be a tuple or list with 3 integers. Received shape: {np.shape(upsample_size)}")
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions")

    tensor = upsample_layer(**upsample_kwargs)(tensor)  # Pass the tensor through the UpSample2D/UpSample3D layer

    tensor = convolution_module(tensor, **module_kwargs)  # Pass input tensor through convolution module

    return tensor


def full_scale_skip_connection(tensor, filters, kernel_size, level1, level2, pool_size=None, padding='same', use_bias=False,
    batch_normalization=True, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name=None):
    """
    Connect two nodes in the U-Net 3+ with a full-scale skip connection (FSC).

    Parameters
    ----------
    tensor: tf.Tensor
        - Input tensor for the convolution module.
    filters: int
        - Number of filters in the Conv2D/Conv3D layer.
    kernel_size: int or tuple
        - Size of the kernel in the Conv2D/Conv3D layer.
    level1: int
        - Level of the first node that is connected to the FSC. This node will provide the input tensor to the FSC. Must be
        smaller than level2 (i.e. the first node must be on a higher level in the U-Net 3+ since we are max-pooling), otherwise
        a ValueError exception is raised.
    level2: int
        - Level of the second node that is connected to the FSC. This node will receive the output of the FSC. Must be greater
        than level1 (i.e. the second node must be on a lower level in the U-Net 3+ since we are max-pooling), otherwise
        a ValueError exception is raised.
    pool_size: tuple or list
        - Pool size for the MaxPooling2D/MaxPooling3D layer.
    padding: str
        - Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        - If True, a bias vector will be used in the Conv2D/Conv3D layer.
    batch_normalization: bool
        - If True, a BatchNormalization layer will follow the Conv2D/Conv3D layer.
    activation: str
        - Activation function to use after the Conv2D/Conv3D layer (BatchNormalization layer, if batch_normalization is True).
        - Can be any of tf.keras.activations, 'prelu', 'leaky_relu', or 'smelu' (case-insensitive).
    kernel_initializer: str or tf.keras.initializers object
        - Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        - Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        - Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        - Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        - Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        - Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        - Constraint function applied to the bias vector in the Conv2D/Conv3D layers.
    name: str or None
        - Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        - Output tensor.
    """

    tensor_dims = len(tensor.shape)  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if level1 >= level2:
        raise ValueError("level2 must be greater than level1 in full-scale skip connections")

    # Arguments for the convolution module.
    module_kwargs = dict({})
    module_kwargs['filters'] = filters
    module_kwargs['kernel_size'] = kernel_size
    module_kwargs['num_modules'] = 1
    module_kwargs['padding'] = padding
    module_kwargs['use_bias'] = use_bias
    module_kwargs['batch_normalization'] = batch_normalization
    module_kwargs['activation'] = activation
    module_kwargs['kernel_initializer'] = kernel_initializer
    module_kwargs['bias_initializer'] = bias_initializer
    module_kwargs['kernel_regularizer'] = kernel_regularizer
    module_kwargs['bias_regularizer'] = bias_regularizer
    module_kwargs['activity_regularizer'] = activity_regularizer
    module_kwargs['kernel_constraint'] = kernel_constraint
    module_kwargs['bias_constraint'] = bias_constraint
    module_kwargs['name'] = name

    # Keyword arguments for the MaxPooling2D/MaxPooling3D layer
    pool_kwargs = dict({})
    pool_kwargs['name'] = f'{name}_MaxPool{tensor_dims - 2}D'
    pool_kwargs['pool_size'] = np.power(pool_size, abs(level1 - level2))

    if tensor_dims == 4:  # If the image is 2D
        pool_layer = MaxPooling2D
    elif tensor_dims == 5:  # If the image is 3D
        pool_layer = MaxPooling3D
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions")

    tensor = pool_layer(**pool_kwargs)(tensor)  # Pass the tensor through the MaxPooling2D/MaxPooling3D layer

    tensor = convolution_module(tensor, **module_kwargs)  # Pass the tensor through the convolution module

    return tensor


def conventional_skip_connection(tensor, filters, kernel_size, padding='same', use_bias=False, batch_normalization=True,
    activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
    activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name=None):
    """
    Connect two nodes in the U-Net 3+ with a conventional skip connection.

    Parameters
    ----------
    tensor: tf.Tensor
        - Input tensor for the convolution module.
    filters: int
        - Number of filters in the Conv2D/Conv3D layer.
    kernel_size: int or tuple
        - Size of the kernel in the Conv2D/Conv3D layer.
    padding: str
        - Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        - If True, a bias vector will be used in the Conv2D/Conv3D layer.
    batch_normalization: bool
        - If True, a BatchNormalization layer will follow the Conv2D/Conv3D layer.
    activation: str
        - Activation function to use after the Conv2D/Conv3D layer (BatchNormalization layer, if batch_normalization is True).
        - Can be any of tf.keras.activations, 'prelu', 'leaky_relu', or 'smelu' (case-insensitive).
    kernel_initializer: str or tf.keras.initializers object
        - Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        - Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        - Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        - Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        - Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        - Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        - Constraint function applied to the bias vector in the Conv2D/Conv3D layers.
    name: str or None
        - Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        - Output tensor.
    """

    # Arguments for the convolution module.
    module_kwargs = dict({})
    module_kwargs['filters'] = filters
    module_kwargs['kernel_size'] = kernel_size
    module_kwargs['num_modules'] = 1
    module_kwargs['padding'] = padding
    module_kwargs['use_bias'] = use_bias
    module_kwargs['batch_normalization'] = batch_normalization
    module_kwargs['activation'] = activation
    module_kwargs['kernel_initializer'] = kernel_initializer
    module_kwargs['bias_initializer'] = bias_initializer
    module_kwargs['kernel_regularizer'] = kernel_regularizer
    module_kwargs['bias_regularizer'] = bias_regularizer
    module_kwargs['activity_regularizer'] = activity_regularizer
    module_kwargs['kernel_constraint'] = kernel_constraint
    module_kwargs['bias_constraint'] = bias_constraint
    module_kwargs['name'] = name

    tensor = convolution_module(tensor, **module_kwargs)  # Pass the tensor through the convolution module

    return tensor


def max_pool(tensor, pool_size=None, name=None):
    """
    Connect two encoder nodes with a max-pooling operation.

    Parameters
    ----------
    tensor: tf.Tensor
        - Input tensor for the convolution module.
    pool_size: tuple or list
        - Pool size for the MaxPooling2D/MaxPooling3D layer.
    name: str or None
        - Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        - Output tensor.
    """

    if type(pool_size) != tuple and type(pool_size) != list:
        raise TypeError(f"pool_size can only be a tuple or list. Received type: {type(pool_size)}")

    tensor_dims = len(tensor.shape)  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    pool_kwargs = dict({})  # Keyword arguments in the MaxPooling layer
    pool_kwargs['name'] = f'{name}_MaxPool{tensor_dims - 2}D'
    pool_kwargs['pool_size'] = pool_size

    if tensor_dims == 4:  # If the image is 2D
        pool_layer = MaxPooling2D
        if len(pool_size) != 2:
            raise TypeError(f"For 2D max-pooling, the pool size must be a tuple or list with 2 integers. Received shape: {np.shape(pool_size)}")
    elif tensor_dims == 5:  # If the image is 3D
        pool_layer = MaxPooling3D
        if len(pool_size) != 3:
            raise TypeError(f"For 3D max-pooling, the pool size must be a tuple or list with 3 integers. Received shape: {np.shape(pool_size)}")
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions")

    pool_tensor = pool_layer(**pool_kwargs)(tensor)  # Pass the tensor through the MaxPooling2D/MaxPooling3D layer

    return pool_tensor


def upsample(tensor, filters, kernel_size, upsample_size, padding='same', use_bias=False, batch_normalization=True,
    activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
    activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name=None):
    """
    Connect decoder nodes with an up-sampling operation.

    Parameters
    ----------
    tensor: tf.Tensor
        - Input tensor for the convolution module.
    filters: int
        - Number of filters in the Conv2D/Conv3D layer.
    kernel_size: int or tuple
        - Size of the kernel in the Conv2D/Conv3D layer.
    upsample_size: tuple or list
        - Upsampling size in the UpSampling2D/UpSampling3D layer.
    padding: str
        - Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        - If True, a bias vector will be used in the Conv2D/Conv3D layer.
    batch_normalization: bool
        - If True, a BatchNormalization layer will follow the Conv2D/Conv3D layer.
    activation: str
        - Activation function to use after the Conv2D/Conv3D layer (BatchNormalization layer, if batch_normalization is True).
        - Can be any of tf.keras.activations, 'prelu', 'leaky_relu', or 'smelu' (case-insensitive).
    kernel_initializer: str or tf.keras.initializers object
        - Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        - Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        - Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        - Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        - Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        - Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        - Constraint function applied to the bias vector in the Conv2D/Conv3D layers.
    name: str or None
        - Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        - Output tensor.
    """

    tensor_dims = len(tensor.shape)  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if type(upsample_size) != tuple and type(upsample_size) != list:
        raise TypeError(f"upsample_size can only be a tuple or list. Received type: {type(upsample_size)}")

    # Arguments for the convolution module.
    module_kwargs = dict({})
    module_kwargs['filters'] = filters
    module_kwargs['kernel_size'] = kernel_size
    module_kwargs['num_modules'] = 1
    module_kwargs['padding'] = padding
    module_kwargs['use_bias'] = use_bias
    module_kwargs['batch_normalization'] = batch_normalization
    module_kwargs['activation'] = activation
    module_kwargs['kernel_initializer'] = kernel_initializer
    module_kwargs['bias_initializer'] = bias_initializer
    module_kwargs['kernel_regularizer'] = kernel_regularizer
    module_kwargs['bias_regularizer'] = bias_regularizer
    module_kwargs['activity_regularizer'] = activity_regularizer
    module_kwargs['kernel_constraint'] = kernel_constraint
    module_kwargs['bias_constraint'] = bias_constraint
    module_kwargs['name'] = name

    # Keyword arguments in the UpSampling layer
    upsample_kwargs = dict({})
    upsample_kwargs['name'] = f'{name}_UpSampling{tensor_dims - 2}D'
    upsample_kwargs['size'] = upsample_size

    if tensor_dims == 4:  # If the image is 2D
        upsample_layer = UpSampling2D
        if len(upsample_size) != 2:
            raise TypeError(f"For 2D up-sampling, the pool size must be a tuple or list with 2 integers. Received shape: {np.shape(upsample_size)}")
    elif tensor_dims == 5:  # If the image is 3D
        upsample_layer = UpSampling3D
        if len(upsample_size) != 3:
            raise TypeError(f"For 3D up-sampling, the pool size must be a tuple or list with 3 integers. Received shape: {np.shape(upsample_size)}")
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions")

    upsample_tensor = upsample_layer(**upsample_kwargs)(tensor)  # Pass the tensor through the UpSampling2D/UpSampling3D layer

    tensor = convolution_module(upsample_tensor, **module_kwargs)  # Pass the up-sampled tensor through a convolution module

    return tensor


def choose_activation_layer(activation: str):
    """
    Choose activation layer for the U-Net.

    Parameters
    ----------
    activation: str
        - Can be any of tf.keras.activations, 'prelu', 'leaky_relu', or 'smelu' (case-insensitive).

    Returns
    -------
    activation_layer: tf.keras.layers.Activation, tf.keras.layers.PReLU, or tf.keras.layers.LeakyReLU
        - Activation layer.
    """

    available_activations = ['elu', 'exponential', 'gelu', 'hard_sigmoid', 'leaky_relu', 'linear', 'prelu', 'relu', 'selu', 'sigmoid', 'softmax',
                            'softplus', 'softsign', 'swish', 'tanh', 'smelu']

    activation_kwargs = dict({})  # Keyword arguments that will be passed into the 'Activation' layer (if applicable)

    # Choose the activation layer
    if activation == 'leaky_relu':
        activation_layer = getattr(layers, 'LeakyReLU')
    elif activation == 'prelu':
        activation_layer = getattr(layers, 'PReLU')
    elif activation == 'smelu':
        activation_layer = custom_activations.SmeLU
    elif activation in available_activations:
        activation_layer = getattr(layers, 'Activation')
        activation_kwargs['activation'] = activation
    else:
        raise TypeError(f"'{activation}' is not a valid loss function and/or is not available, options are: {', '.join(list(available_activations))}")

    return activation_layer


def deep_supervision_side_output(tensor, num_classes, kernel_size, output_level, upsample_size, padding='same', use_bias=True,
    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None, name=None):
    """
    Deep supervision output. This is usually used on a decoder node in the U-Net 3+ or the final decoder node of a standard
    U-Net.

    Parameters
    ----------
    tensor: tf.Tensor
        - Input tensor for the convolution module.
    num_classes: int
        - Number of classes that the model is trying to predict.
    kernel_size: int or tuple
        - Size of the kernel in the Conv2D/Conv3D layer.
    output_level: int
        - Level of the decoder node from which the deep supervision output is based.
    upsample_size: tuple or list
        - Upsampling size for rows and columns in the UpSampling2D/UpSampling3D layer. Tuples are currently not supported
        but will be supported in a future update.
    padding: str
        - Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        - If True, a bias vector will be used in the Conv2D/Conv3D layer.
    kernel_initializer: str or tf.keras.initializers object
        - Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        - Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        - Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        - Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        - Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        - Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        - Constraint function applied to the bias vector in the Conv2D/Conv3D layers.
    name: str or None
        - Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        - Output tensor.
    """

    tensor_dims = len(tensor.shape)  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if tensor_dims == 4:  # If the image is 2D
        conv_layer = Conv2D
        upsample_layer = UpSampling2D
        if output_level > 1:
            upsample_size_1 = upsample_size
        else:
            upsample_size_1 = None

        if output_level > 2:
            upsample_size_2 = np.power(upsample_size, output_level - 2)
        else:
            upsample_size_2 = None

    elif tensor_dims == 5:  # If the image is 3D
        conv_layer = Conv3D
        upsample_layer = UpSampling3D
        if output_level > 1:
            upsample_size_1 = upsample_size
        else:
            upsample_size_1 = None

        if output_level > 2:
            upsample_size_2 = np.power(upsample_size, output_level - 2)
        else:
            upsample_size_2 = None

    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor can only have 4 or 5 dimensions")

    # Arguments for the Conv2D/Conv3D layer.
    conv_kwargs = dict({})
    conv_kwargs['kernel_size'] = kernel_size
    conv_kwargs['padding'] = padding
    conv_kwargs['use_bias'] = use_bias
    conv_kwargs['kernel_initializer'] = kernel_initializer
    conv_kwargs['bias_initializer'] = bias_initializer
    conv_kwargs['kernel_regularizer'] = kernel_regularizer
    conv_kwargs['bias_regularizer'] = bias_regularizer
    conv_kwargs['activity_regularizer'] = activity_regularizer
    conv_kwargs['kernel_constraint'] = kernel_constraint
    conv_kwargs['bias_constraint'] = bias_constraint
    conv_kwargs['name'] = f'{name}_Conv{tensor_dims - 2}D'

    if upsample_size_1 is not None:
        tensor = upsample_layer(size=upsample_size_1, name=f'{name}_UpSampling{tensor_dims - 2}D_1')(tensor)  # Pass the tensor through the UpSampling2D/UpSampling3D layer

    tensor = conv_layer(filters=num_classes, **conv_kwargs)(tensor)  # This convolution layer contains num_classes filters, one for each class

    if upsample_size_2 is not None:
        tensor = upsample_layer(size=upsample_size_2, name=f'{name}_UpSampling{tensor_dims - 2}D_2')(tensor)  # Pass the tensor through the UpSampling2D/UpSampling3D layer

    sup_output = Softmax(name=f'{name}_Softmax')(tensor)  # Final softmax output

    return sup_output
