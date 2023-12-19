"""
Functions for building components of U-Net models:
    - U-Net
    - U-Net ensemble
    - U-Net+
    - U-Net++
    - U-Net 3+
    - Attention U-Net

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.12.19.1
"""

import numpy as np
from tensorflow.keras.layers import Activation, Conv2D, Conv3D, BatchNormalization, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Softmax
from tensorflow.keras import layers
import tensorflow as tf
import custom_activations


def attention_gate(
    x: tf.Tensor,
    g: tf.Tensor,
    kernel_size: int | tuple[int],
    pool_size: tuple[int],
    name: str or None = None):
    """
    Attention gate for the Attention U-Net.

    Parameters
    ----------
    x: tf.Tensor
        Signal that originates from the encoder node on the same level as the attention gate.
    g: tf.Tensor
        Signal that originates from the level below the attention gate, which has higher resolution features.
    kernel_size: int or tuple
        Size of the kernel in the Conv2D/Conv3D layer(s). Only applies to layers that are not forced to a kernel size of 1.
    pool_size: tuple or list
        Pool size for the UpSampling layers, as well as the number of strides in the first
    name: str or None
        Prefix of the layer names. If left as None, the layer names are set automatically.

    References
    ----------
    https://towardsdatascience.com/a-detailed-explanation-of-the-attention-u-net-b371a5590831
    """

    conv_layer = getattr(tf.keras.layers, f'Conv{len(x.shape) - 2}D')  # Select the convolution layer for the x and g tensors
    upsample_layer = getattr(tf.keras.layers, f'UpSampling{len(x.shape) - 2}D')  # Select the upsampling layer

    shape_x = x.shape  # Shapes of the ORIGINAL inputs
    filters_x = shape_x[-1]

    """
    x: Get the x tensor to the same shape as the gating signal (g tensor)
    g: Perform a 1x1-style convolution on the gating signal so it has the same number of filters as the x signal  
    """
    x_conv = conv_layer(filters=filters_x,
                        kernel_size=kernel_size,
                        strides=pool_size,
                        padding='same',
                        name=f'{name}_Conv{len(x.shape) - 2}D_x')(x)
    g_conv = conv_layer(filters=filters_x, kernel_size=1, padding='same', name=f'{name}_Conv{len(x.shape) - 2}D_g')(g)

    xg = tf.add(x_conv, g_conv, name=f'{name}_sum')  # Sum the x and g signals element-wise
    xg = Activation(activation='relu', name=f'{name}_relu')(xg)  # Pass the summed signals through a ReLU activation layer

    xg_collapse = conv_layer(filters=1, kernel_size=1, padding='same', name=f'{name}_collapse')(xg)  # Collapse the number of filters to just 1
    xg_collapse = Activation(activation='sigmoid', name=f'{name}_sigmoid')(xg_collapse)  # Pass collapsed tensor through a sigmoid activation layer

    # Upsample the collapsed tensor so its dimensions match the original shape of the x signal, then expand the filters to match the g signal filters
    upsample_xg = upsample_layer(size=pool_size, name=f'{name}_UpSampling{len(x.shape) - 2}D')(xg_collapse)
    upsample_xg = tf.repeat(upsample_xg, filters_x, axis=-1, name=f'{name}_repeat')

    coeffs = tf.multiply(upsample_xg, x, name=f'{name}_multiply')  # Element-wise multiplication onto the original x signal

    attention_tensor = conv_layer(filters=filters_x, kernel_size=1, strides=1, padding='same', name=f'{name}_Conv{len(x.shape) - 2}D_coeffs')(coeffs)
    attention_tensor = BatchNormalization(name=f'{name}_BatchNorm')(attention_tensor)

    return attention_tensor


def convolution_module(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: int | tuple[int],
    num_modules: int = 1,
    padding: str = 'same',
    use_bias: bool = False,
    batch_normalization: bool = True,
    activation: str = 'relu',
    kernel_initializer = 'glorot_uniform',
    bias_initializer = 'zeros',
    kernel_regularizer = None,
    bias_regularizer = None,
    activity_regularizer = None,
    kernel_constraint = None,
    bias_constraint = None,
    shared_axes = None,
    name: str = None):
    """
    Insert modules into an encoder or decoder node.

    Parameters
    ----------
    tensor: tf.Tensor
        Input tensor for the convolution module(s).
    filters: int
        Number of filters in the Conv2D/Conv3D layer(s).
    kernel_size: int or tuple of ints
        Size of the kernel in the Conv2D/Conv3D layer(s).
    num_modules: int
        Number of convolution modules to insert. Must be greater than 0, otherwise a ValueError exception is raised.
    padding: str
        Padding in the Conv2D/Conv3D layer(s). 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        If True, a bias vector will be used in the Conv2D/Conv3D layers.
    batch_normalization: bool
        If True, a BatchNormalization layer will follow every Conv2D/Conv3D layer.
    activation: str
        Activation function to use after every Conv2D/Conv3D layer (BatchNormalization layer, if batch_normalization is True).
        See unet_utils.choose_activation_layer for all available activation functions.
    kernel_initializer: str or tf.keras.initializers object
        Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        Constraint function applied to the bias vector in the Conv2D/Conv3D layers.
    shared_axes: tuple or list of ints
        Axes along which to share the learnable parameters for the activation function.
    name: str or None
        Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        Output tensor.
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

    # Arguments for the Conv2D/Conv3D layer.
    conv_kwargs = dict({})
    for arg in ['filters', 'use_bias', 'kernel_size', 'padding', 'kernel_initializer', 'bias_initializer', 'kernel_regularizer',
                'bias_regularizer', 'activity_regularizer', 'kernel_constraint', 'bias_constraint']:
        conv_kwargs[arg] = locals()[arg]

    activation_layer = choose_activation_layer(activation)  # Choose activation layer for the convolution modules.

    activation_kwargs = dict({})
    if activation_layer == Activation:
        activation_kwargs['activation'] = activation
    elif activation in ['prelu', 'smelu', 'snake']:  # these activation functions have learnable parameters
        activation_kwargs['shared_axes'] = shared_axes

    # Insert convolution modules
    for module in range(num_modules):

        # Create names for the Conv2D/Conv3D layers and the activation layer.
        conv_kwargs['name'] = f'{name}_Conv{tensor_dims - 2}D_{module+1}'
        activation_kwargs['name'] = f'{name}_{activation}_{module+1}'

        conv_tensor = conv_layer(**conv_kwargs)(tensor)  # Perform convolution on the input tensor

        if batch_normalization:
            batch_norm_tensor = BatchNormalization(name=f'{name}_BatchNorm_{module+1}')(conv_tensor)  # Insert layer for batch normalization
            activation_tensor = activation_layer(**activation_kwargs)(batch_norm_tensor)  # Pass output tensor from BatchNormalization into the activation layer
        else:
            activation_tensor = activation_layer(**activation_kwargs)(conv_tensor)  # Pass output tensor from the convolution layer into the activation layer.

        tensor = activation_tensor

    return tensor


def aggregated_feature_map(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: int | tuple[int],
    level1: int,
    level2: int,
    upsample_size: tuple[int],
    padding: str = 'same',
    use_bias: bool = False,
    batch_normalization: bool = True,
    activation: str = 'relu',
    kernel_initializer = 'glorot_uniform',
    bias_initializer = 'zeros',
    kernel_regularizer = None,
    bias_regularizer = None,
    activity_regularizer = None,
    kernel_constraint = None,
    bias_constraint = None,
    shared_axes = None,
    name: str = None):
    """
    Connect two nodes in the U-Net 3+ with an aggregated feature map (AFM).

    Parameters
    ----------
    tensor: tf.Tensor
        Input tensor for the convolution module.
    filters: int
        Number of filters in the Conv2D/Conv3D layer.
    kernel_size: int or tuple
        Size of the kernel in the Conv2D/Conv3D layer.
    level1: int
        Level of the first node that is connected to the AFM. This node will provide the input tensor to the AFM. Must be
        greater than level2 (i.e. the first node must be on a lower level in the U-Net 3+ since we are up-sampling), otherwise
        a ValueError exception is raised.
    level2: int
        Level of the second node that is connected to the AFM. This node will receive the output of the AFM. Must be smaller
        than level1 (i.e. the second node must be on a higher level in the U-Net 3+ since we are up-sampling), otherwise
        a ValueError exception is raised.
    upsample_size: tuple or list of ints
        Upsampling size for rows and columns in the UpSampling2D/UpSampling3D layer.
    padding: str
        Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        If True, a bias vector will be used in the Conv2D/Conv3D layer.
    batch_normalization: bool
        If True, a BatchNormalization layer will follow the Conv2D/Conv3D layer.
    activation: str
        Activation function to use after the Conv2D/Conv3D layer (BatchNormalization layer, if batch_normalization is True).
        See unet_utils.choose_activation_layer for all available activation functions.
    kernel_initializer: str or tf.keras.initializers object
        Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        Constraint function applied to the bias vector in the Conv2D/Conv3D layers.
    shared_axes: tuple or list of ints
        Axes along which to share the learnable parameters for the activation function.
    name: str or None
        Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        Output tensor.
    """

    tensor_dims = len(tensor.shape)  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if level1 <= level2:
        raise ValueError("level2 must be smaller than level1 in aggregated feature maps")

    # Arguments for the convolution module.
    module_kwargs = dict({})
    module_kwargs['num_modules'] = 1
    for arg in ['filters', 'kernel_size', 'padding', 'use_bias', 'batch_normalization', 'activation', 'kernel_initializer',
                'bias_initializer', 'kernel_regularizer', 'bias_regularizer', 'activity_regularizer', 'kernel_constraint',
                'bias_constraint', 'shared_axes', 'name']:
        module_kwargs[arg] = locals()[arg]

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


def full_scale_skip_connection(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: int | tuple[int],
    level1: int,
    level2: int,
    pool_size: tuple[int],
    padding: str = 'same',
    use_bias: bool = False,
    batch_normalization: bool = True,
    activation: str = 'relu',
    kernel_initializer = 'glorot_uniform',
    bias_initializer = 'zeros',
    kernel_regularizer = None,
    bias_regularizer = None,
    activity_regularizer = None,
    kernel_constraint = None,
    bias_constraint = None,
    shared_axes = None,
    name: str = None):
    """
    Connect two nodes in the U-Net 3+ with a full-scale skip connection (FSC).

    Parameters
    ----------
    tensor: tf.Tensor
        Input tensor for the convolution module.
    filters: int
        Number of filters in the Conv2D/Conv3D layer.
    kernel_size: int or tuple
        Size of the kernel in the Conv2D/Conv3D layer.
    level1: int
        Level of the first node that is connected to the FSC. This node will provide the input tensor to the FSC. Must be
        smaller than level2 (i.e. the first node must be on a higher level in the U-Net 3+ since we are max-pooling), otherwise
        a ValueError exception is raised.
    level2: int
        Level of the second node that is connected to the FSC. This node will receive the output of the FSC. Must be greater
        than level1 (i.e. the second node must be on a lower level in the U-Net 3+ since we are max-pooling), otherwise
        a ValueError exception is raised.
    pool_size: tuple or list
        Pool size for the MaxPooling2D/MaxPooling3D layer.
    padding: str
        Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        If True, a bias vector will be used in the Conv2D/Conv3D layer.
    batch_normalization: bool
        If True, a BatchNormalization layer will follow the Conv2D/Conv3D layer.
    activation: str
        Activation function to use after the Conv2D/Conv3D layer (BatchNormalization layer, if batch_normalization is True).
        See unet_utils.choose_activation_layer for all available activation functions.
    kernel_initializer: str or tf.keras.initializers object
        Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        Constraint function applied to the bias vector in the Conv2D/Conv3D layers.
    shared_axes: tuple or list of ints
        Axes along which to share the learnable parameters for the activation function.
    name: str or None
        Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        Output tensor.
    """

    tensor_dims = len(tensor.shape)  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if level1 >= level2:
        raise ValueError("level2 must be greater than level1 in full-scale skip connections")

    # Arguments for the convolution module.
    module_kwargs = dict({})
    module_kwargs['num_modules'] = 1
    for arg in ['filters', 'kernel_size', 'padding', 'use_bias', 'batch_normalization', 'activation', 'kernel_initializer',
                'bias_initializer', 'kernel_regularizer', 'bias_regularizer', 'activity_regularizer', 'kernel_constraint',
                'bias_constraint', 'shared_axes', 'name']:
        module_kwargs[arg] = locals()[arg]

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


def conventional_skip_connection(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: int | tuple[int],
    padding: str = 'same',
    use_bias: bool = False,
    batch_normalization: bool = True,
    activation: str = 'relu',
    kernel_initializer = 'glorot_uniform',
    bias_initializer = 'zeros',
    kernel_regularizer = None,
    bias_regularizer = None,
    activity_regularizer = None,
    kernel_constraint = None,
    bias_constraint = None,
    shared_axes = None,
    name: str = None):
    """
    Connect two nodes in the U-Net 3+ with a conventional skip connection.

    Parameters
    ----------
    tensor: tf.Tensor
        Input tensor for the convolution module.
    filters: int
        Number of filters in the Conv2D/Conv3D layer.
    kernel_size: int or tuple
        Size of the kernel in the Conv2D/Conv3D layer.
    padding: str
        Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        If True, a bias vector will be used in the Conv2D/Conv3D layer.
    batch_normalization: bool
        If True, a BatchNormalization layer will follow the Conv2D/Conv3D layer.
    activation: str
        Activation function to use after the Conv2D/Conv3D layer (BatchNormalization layer, if batch_normalization is True).
        See unet_utils.choose_activation_layer for all available activation functions.
    kernel_initializer: str or tf.keras.initializers object
        Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        Constraint function applied to the bias vector in the Conv2D/Conv3D layers.
    shared_axes: tuple or list of ints
        Axes along which to share the learnable parameters for the activation function.
    name: str or None
        Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        Output tensor.
    """

    # Arguments for the convolution module.
    module_kwargs = dict({})
    module_kwargs['num_modules'] = 1
    for arg in ['filters', 'kernel_size', 'padding', 'use_bias', 'batch_normalization', 'activation', 'kernel_initializer',
                'bias_initializer', 'kernel_regularizer', 'bias_regularizer', 'activity_regularizer', 'kernel_constraint',
                'bias_constraint', 'shared_axes', 'name']:
        module_kwargs[arg] = locals()[arg]

    tensor = convolution_module(tensor, **module_kwargs)  # Pass the tensor through the convolution module

    return tensor


def max_pool(
    tensor: tf.Tensor,
    pool_size: tuple[int],
    name: str = None):
    """
    Connect two encoder nodes with a max-pooling operation.

    Parameters
    ----------
    tensor: tf.Tensor
        Input tensor for the convolution module.
    pool_size: tuple or list
        Pool size for the MaxPooling2D/MaxPooling3D layer.
    name: str or None
        Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        Output tensor.
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


def upsample(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: int | tuple[int],
    upsample_size: tuple[int],
    padding: str = 'same',
    use_bias: bool = False,
    batch_normalization: bool = True,
    activation: str = 'relu',
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer = None,
    bias_regularizer = None,
    activity_regularizer = None,
    kernel_constraint = None,
    bias_constraint = None,
    shared_axes = None,
    name: str = None):
    """
    Connect decoder nodes with an up-sampling operation.

    Parameters
    ----------
    tensor: tf.Tensor
        Input tensor for the convolution module.
    filters: int
        Number of filters in the Conv2D/Conv3D layer.
    kernel_size: int or tuple
        Size of the kernel in the Conv2D/Conv3D layer.
    upsample_size: tuple or list
        Upsampling size in the UpSampling2D/UpSampling3D layer.
    padding: str
        Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        If True, a bias vector will be used in the Conv2D/Conv3D layer.
    batch_normalization: bool
        If True, a BatchNormalization layer will follow the Conv2D/Conv3D layer.
    activation: str
        Activation function to use after the Conv2D/Conv3D layer (BatchNormalization layer, if batch_normalization is True).
        See unet_utils.choose_activation_layer for all available activation functions.
    kernel_initializer: str or tf.keras.initializers object
        Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        Constraint function applied to the bias vector in the Conv2D/Conv3D layers.
    shared_axes: tuple or list of ints
        Axes along which to share the learnable parameters for the activation function.
    name: str or None
        Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        Output tensor.
    """

    tensor_dims = len(tensor.shape)  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if type(upsample_size) != tuple and type(upsample_size) != list:
        raise TypeError(f"upsample_size can only be a tuple or list. Received type: {type(upsample_size)}")

    # Arguments for the convolution module.
    module_kwargs = dict({})
    module_kwargs['num_modules'] = 1
    for arg in ['filters', 'kernel_size', 'padding', 'use_bias', 'batch_normalization', 'activation', 'kernel_initializer',
                'bias_initializer', 'kernel_regularizer', 'bias_regularizer', 'activity_regularizer', 'kernel_constraint',
                'bias_constraint', 'shared_axes', 'name']:
        module_kwargs[arg] = locals()[arg]

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
        Can be any of tf.keras.activations, 'gaussian', 'gcu', 'leaky_relu', 'prelu', 'psigmoid', 'resech', 'smelu', 'snake' (case-insensitive).

    Returns
    -------
    activation_layer: tf.keras.layers.Activation, tf.keras.layers.PReLU, tf.keras.layers.LeakyReLU, or any layer from custom_activations
        Activation layer.
    """

    activation = activation.lower()

    available_activations = ['elliott', 'elu', 'exponential', 'gaussian', 'gcu', 'gelu', 'hard_sigmoid', 'hexpo', 'isigmoid', 'leaky_relu', 'linear',
        'lisht', 'prelu', 'psigmoid', 'ptanh', 'ptelu', 'relu', 'resech', 'selu', 'sigmoid', 'smelu', 'snake', 'softmax', 'softplus', 'softsign',
        'srs', 'stanh', 'swish', 'tanh', 'thresholded_relu']

    # Choose the activation layer
    if activation == 'elliott':
        activation_layer = custom_activations.Elliott
    elif activation == 'gaussian':
        activation_layer = custom_activations.Gaussian
    elif activation == 'gcu':
        activation_layer = custom_activations.GCU
    elif activation == 'hexpo':
        activation_layer = custom_activations.Hexpo
    elif activation == 'isigmoid':
        activation_layer = custom_activations.ISigmoid
    elif activation == 'leaky_relu':
        activation_layer = getattr(layers, 'LeakyReLU')
    elif activation == 'lisht':
        activation_layer = custom_activations.LiSHT
    elif activation == 'prelu':
        activation_layer = getattr(layers, 'PReLU')
    elif activation == 'psigmoid':
        activation_layer = custom_activations.PSigmoid
    elif activation == 'ptanh':
        activation_layer = custom_activations.PTanh
    elif activation == 'ptelu':
        activation_layer = custom_activations.PTELU
    elif activation == 'resech':
        activation_layer = custom_activations.ReSech
    elif activation == 'smelu':
        activation_layer = custom_activations.SmeLU
    elif activation == 'snake':
        activation_layer = custom_activations.Snake
    elif activation == 'srs':
        activation_layer = custom_activations.SRS
    elif activation == 'stanh':
        activation_layer = custom_activations.STanh
    elif activation == 'thresholded_relu':
        activation_layer = getattr(layers, 'ThresholdedReLU')
    elif activation in available_activations:
        activation_layer = getattr(layers, 'Activation')
    else:
        raise TypeError(f"'{activation}' is not a valid loss function and/or is not available, options are: {', '.join(sorted(list(available_activations)))}")

    return activation_layer


def deep_supervision_side_output(
    tensor: tf.Tensor,
    num_classes: int,
    kernel_size: int | tuple[int],
    output_level: int,
    upsample_size: tuple[int],
    padding: str = 'same',
    use_bias: bool = False,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer = None,
    bias_regularizer = None,
    activity_regularizer = None,
    kernel_constraint = None,
    bias_constraint = None,
    squeeze_axes: int | tuple[int] = None,
    name: str = None):
    """
    Deep supervision output. This is usually used on a decoder node in the U-Net 3+ or the final decoder node of a standard
    U-Net.

    Parameters
    ----------
    tensor: tf.Tensor
        Input tensor for the convolution module.
    num_classes: int
        Number of classes that the model is trying to predict.
    kernel_size: int or tuple
        Size of the kernel in the Conv2D/Conv3D layer.
    output_level: int
        Level of the decoder node from which the deep supervision output is based.
    upsample_size: tuple or list
        Upsampling size for rows and columns in the UpSampling2D/UpSampling3D layer. Tuples are currently not supported
        but will be supported in a future update.
    padding: str
        Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        If True, a bias vector will be used in the Conv2D/Conv3D layer.
    kernel_initializer: str or tf.keras.initializers object
        Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        Constraint function applied to the bias vector in the Conv2D/Conv3D layers.
    squeeze_axes: int, tuple, or None
        Axis or axes of the input tensor to squeeze.
    name: str or None
        Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        Output tensor.
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
    conv_kwargs['name'] = f'{name}_Conv{tensor_dims - 2}D'
    for arg in ['use_bias', 'kernel_size', 'padding', 'kernel_initializer', 'bias_initializer', 'kernel_regularizer', 'bias_regularizer',
                'activity_regularizer', 'kernel_constraint', 'bias_constraint']:
        conv_kwargs[arg] = locals()[arg]

    if upsample_size_1 is not None:
        tensor = upsample_layer(size=upsample_size_1, name=f'{name}_UpSampling{tensor_dims - 2}D_1')(tensor)  # Pass the tensor through the UpSampling2D/UpSampling3D layer

    tensor = conv_layer(filters=num_classes, **conv_kwargs)(tensor)  # This convolution layer contains num_classes filters, one for each class

    if upsample_size_2 is not None:
        tensor = upsample_layer(size=upsample_size_2, name=f'{name}_UpSampling{tensor_dims - 2}D_2')(tensor)  # Pass the tensor through the UpSampling2D/UpSampling3D layer

    ### Squeeze the given dimensions/axes ###
    if squeeze_axes is not None:

        conv_kwargs['kernel_size'] = [1 for _ in range(tensor_dims - 2)]

        if type(squeeze_axes) == int:
            squeeze_axes = [squeeze_axes, ]  # Turn integer into a list of length 1 to make indexing easier

        squeeze_axes_sizes = [tensor.shape[ax_to_squeeze] for ax_to_squeeze in squeeze_axes]

        for ax, size in enumerate(squeeze_axes_sizes):
            conv_kwargs['kernel_size'][squeeze_axes[ax] - 1] = size  # Kernel size of dimension to squeeze is equal to the size of the dimension because we want the final size to be 1 so it can be squeezed

        conv_kwargs['padding'] = 'valid'  # Padding cannot be 'same' since we want to modify the size of the dimension to be squeezed
        conv_kwargs['name'] = f'{name}_Conv{tensor_dims - 2}D_collapse'

        tensor = conv_layer(filters=num_classes, **conv_kwargs)(tensor)  # This convolution layer contains num_classes filters, one for each class
        tensor = tf.squeeze(tensor, axis=squeeze_axes)  # Squeeze the tensor and remove the dimension

    sup_output = Softmax(name=f'{name}_Softmax')(tensor)  # Final softmax output

    return sup_output
