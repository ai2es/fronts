"""
Functions for building U-Net models:
- U-Net
- U-Net 3+

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 4/13/2022 4:32 PM CDT

Known bugs:
- none

Please report any bugs to Andrew Justin: andrewjustinwx@gmail.com
"""

import numpy as np
from tensorflow.keras.layers import Conv2D, Conv3D, BatchNormalization, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Softmax
from tensorflow.keras import layers


def convolution_module(tensor, filters, kernel_size, num_modules=1, padding='same', use_bias=False, batch_normalization=True,
    activation='relu', name=None):
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
        - Can be any of tf.keras.activations, 'prelu', or 'leaky_relu' (case-insensitive).
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

    activation_layer = choose_activation_layer(activation)  # Choose activation layer for the convolution modules.

    """ 
    Arguments for the Activation layer(s), if applicable. 
    - If activation is 'prelu' or 'leaky_relu', a PReLU layer or a LeakyReLU layer will be used instead and this dictionary
      will have no effect.
    """
    activation_kwargs = dict({})
    if activation_layer == layers.Activation:
        activation_kwargs['activation'] = activation

    # Insert convolution modules
    for module in range(num_modules):

        # Create names for the Conv2D/Conv3D layers and the activation layer.
        conv_kwargs['name'] = f'{name}_Conv{tensor_dims - 2}D_{module+1}'
        activation_kwargs['name'] = f'{name}_{activation}_{module+1}'

        conv_tensor = conv_layer(**conv_kwargs)(tensor)  # Perform convolution on the input tensor

        if batch_normalization is True:
            batch_norm_tensor = BatchNormalization(name=f'{name}_BatchNorm_{module+1}')(conv_tensor)  # Insert layer for batch normalization
            activation_tensor = activation_layer(**activation_kwargs)(batch_norm_tensor)  # Pass output tensor from BatchNormalization into the activation layer
        else:
            activation_tensor = activation_layer(**activation_kwargs)(conv_tensor)  # Pass output tensor from the convolution layer into the activation layer.

        tensor = activation_tensor

    return tensor


def aggregated_feature_map(tensor, filters, kernel_size, level1, level2, upsample_size=2, padding='same', use_bias=False,
    batch_normalization=True, activation='relu', preserve_third_dimension=False, name=None):
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
    upsample_size: int
        - Upsampling factors for rows and columns in the UpSampling2D/UpSampling3D layer. Tuples are currently not supported
        but will be supported in a future update.
    padding: str
        - Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        - If True, a bias vector will be used in the Conv2D/Conv3D layer.
    batch_normalization: bool
        - If True, a BatchNormalization layer will follow the Conv2D/Conv3D layer.
    activation: str
        - Activation function to use after the Conv2D/Conv3D layer (BatchNormalization layer, if batch_normalization is True).
        - Can be any of tf.keras.activations, 'prelu', or 'leaky_relu' (case-insensitive).
    preserve_third_dimension: bool
        - If True, the third dimension of the image will not be affected by the up-sampling. This is particularly useful if
        the third dimension is small, as up-sampling can destroy features in images with a small vertical dimension.
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

    # Keyword arguments for the UpSampling2D/UpSampling3D models
    upsample_kwargs = dict({})
    upsample_kwargs['name'] = f'{name}_UpSampling{tensor_dims - 2}D'

    if tensor_dims == 4:  # If the image is 2D
        upsample_layer = UpSampling2D
        upsample_kwargs['size'] = (np.power(upsample_size, abs(level1 - level2)), np.power(upsample_size, abs(level1 - level2)))
        if preserve_third_dimension is True:
            raise ValueError(f"preserve_third_dimension is True but the tensor is not 5D. Provided tensor shape: {tensor.shape}")
    elif tensor_dims == 5:  # If the image is 3D
        upsample_layer = UpSampling3D
        if preserve_third_dimension is True:
            upsample_kwargs['size'] = (np.power(upsample_size, abs(level1 - level2)), np.power(upsample_size, abs(level1 - level2)), 1)
        else:
            upsample_kwargs['size'] = (np.power(upsample_size, abs(level1 - level2)), np.power(upsample_size, abs(level1 - level2)), np.power(upsample_size, abs(level1 - level2)))
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions")

    tensor = upsample_layer(**upsample_kwargs)(tensor)  # Pass the tensor through the UpSample2D/UpSample3D layer

    tensor = convolution_module(tensor, filters, kernel_size, num_modules=1, padding=padding, use_bias=use_bias, batch_normalization=batch_normalization,
        activation=activation, name=name)  # Pass input tensor through convolution module

    return tensor


def full_scale_skip_connection(tensor, filters, kernel_size, level1, level2, pool_size=2, padding='same', use_bias=False,
    batch_normalization=True, activation='relu', preserve_third_dimension=False, name=None):
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
    pool_size: int
        - Pool size for the MaxPooling2D/MaxPooling3D layer. Tuples are currently not supported but will be supported in a
        future update.
    padding: str
        - Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        - If True, a bias vector will be used in the Conv2D/Conv3D layer.
    batch_normalization: bool
        - If True, a BatchNormalization layer will follow the Conv2D/Conv3D layer.
    activation: str
        - Activation function to use after the Conv2D/Conv3D layer (BatchNormalization layer, if batch_normalization is True).
        - Can be any of tf.keras.activations, 'prelu', or 'leaky_relu' (case-insensitive).
    preserve_third_dimension: bool
        - If True, the third dimension of the image will not be affected by the max-pooling. This is particularly useful if
        the third dimension is small, as max-pooling can destroy features in images with a small vertical dimension.
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

    # Keyword arguments for the MaxPooling2D/MaxPooling3D layer
    pool_kwargs = dict({})
    pool_kwargs['name'] = f'{name}_MaxPool{tensor_dims - 2}D'

    if tensor_dims == 4:  # If the image is 2D
        pool_layer = MaxPooling2D
        pool_kwargs['pool_size'] = (np.power(pool_size, abs(level1 - level2)), np.power(pool_size, abs(level1 - level2)))
        if preserve_third_dimension is True:
            raise ValueError(f"preserve_third_dimension is True but the tensor is not 5D. Provided tensor shape: {tensor.shape}")
    elif tensor_dims == 5:  # If the image is 3D
        pool_layer = MaxPooling3D
        if preserve_third_dimension is True:
            pool_kwargs['pool_size'] = (np.power(pool_size, abs(level1 - level2)), np.power(pool_size, abs(level1 - level2)), 1)
        else:
            pool_kwargs['pool_size'] = (np.power(pool_size, abs(level1 - level2)), np.power(pool_size, abs(level1 - level2)), np.power(pool_size, abs(level1 - level2)))
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions")

    tensor = pool_layer(**pool_kwargs)(tensor)  # Pass the tensor through the MaxPooling2D/MaxPooling3D layer

    tensor = convolution_module(tensor, filters, kernel_size, num_modules=1, padding=padding, use_bias=use_bias, batch_normalization=batch_normalization,
        activation=activation, name=name)  # Pass the tensor through the convolution module

    return tensor


def conventional_skip_connection(tensor, filters, kernel_size, padding='same', use_bias=False, batch_normalization=True,
    activation='relu', name=None):
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
        - Can be any of tf.keras.activations, 'prelu', or 'leaky_relu' (case-insensitive).
    name: str or None
        - Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        - Output tensor.
    """

    tensor = convolution_module(tensor, filters, kernel_size, num_modules=1, padding=padding, use_bias=use_bias, batch_normalization=batch_normalization,
        activation=activation, name=name)  # Pass the tensor through the convolution module

    return tensor


def max_pool(tensor, pool_size=2, preserve_third_dimension=False, name=None):
    """
    Connect two encoder nodes with a max-pooling operation.

    Parameters
    ----------
    tensor: tf.Tensor
        - Input tensor for the convolution module.
    pool_size: int
        - Pool size for the MaxPooling2D/MaxPooling3D layer. Tuples are currently not supported but will be supported in a
        future update.
    preserve_third_dimension: bool
        - If True, the third dimension of the image will not be affected by the max-pooling. This is particularly useful if
        the third dimension is small, as max-pooling can destroy features in images with a small vertical dimension.
    name: str or None
        - Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        - Output tensor.
    """

    tensor_dims = len(tensor.shape)  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    pool_kwargs = dict({})  # Keyword arguments in the MaxPooling layer
    pool_kwargs['name'] = f'{name}_MaxPool{tensor_dims - 2}D'

    if tensor_dims == 4:  # If the image is 2D
        pool_layer = MaxPooling2D
        pool_kwargs['pool_size'] = (pool_size, pool_size)
        if preserve_third_dimension is True:
            raise ValueError(f"preserve_third_dimension is True but the tensor is not 5D. Provided tensor shape: {tensor.shape}")
    elif tensor_dims == 5:  # If the image is 3D
        pool_layer = MaxPooling3D
        if preserve_third_dimension is True:
            pool_kwargs['pool_size'] = (pool_size, pool_size, 1)
        else:
            pool_kwargs['pool_size'] = (pool_size, pool_size, pool_size)
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions")

    pool_tensor = pool_layer(**pool_kwargs)(tensor)  # Pass the tensor through the MaxPooling2D/MaxPooling3D layer

    return pool_tensor


def upsample(tensor, filters, kernel_size, upsample_size=2, padding='same', use_bias=False, preserve_third_dimension=False, batch_normalization=True,
    activation='relu', name=None):
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
    upsample_size: int
        - Upsampling factors for rows and columns in the UpSampling2D/UpSampling3D layer. Tuples are currently not supported
        but will be supported in a future update.
    padding: str
        - Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        - If True, a bias vector will be used in the Conv2D/Conv3D layer.
    batch_normalization: bool
        - If True, a BatchNormalization layer will follow the Conv2D/Conv3D layer.
    activation: str
        - Activation function to use after the Conv2D/Conv3D layer (BatchNormalization layer, if batch_normalization is True).
        - Can be any of tf.keras.activations, 'prelu', or 'leaky_relu' (case-insensitive).
    preserve_third_dimension: bool
        - If True, the third dimension of the image will not be affected by the up-sampling. This is particularly useful if
        the third dimension is small, as up-sampling can destroy features in images with a small vertical dimension.
    name: str or None
        - Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        - Output tensor.
    """

    tensor_dims = len(tensor.shape)  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    upsample_kwargs = dict({})  # Keyword arguments in the UpSampling layer
    upsample_kwargs['name'] = f'{name}_UpSampling{tensor_dims - 2}D'

    if tensor_dims == 4:  # If the image is 2D
        upsample_layer = UpSampling2D
        upsample_kwargs['size'] = (upsample_size, upsample_size)
        if preserve_third_dimension is True:
            raise ValueError(f"preserve_third_dimension is True but the tensor is not 5D. Provided tensor shape: {tensor.shape}")
    elif tensor_dims == 5:  # If the image is 3D
        upsample_layer = UpSampling3D
        if preserve_third_dimension is True:
            upsample_kwargs['size'] = (upsample_size, upsample_size, 1)
        else:
            upsample_kwargs['size'] = (upsample_size, upsample_size, upsample_size)
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor can only have 4 or 5 dimensions")

    upsample_tensor = upsample_layer(**upsample_kwargs)(tensor)  # Pass the tensor through the UpSampling2D/UpSampling3D layer

    tensor = convolution_module(upsample_tensor, filters, kernel_size, num_modules=1, padding=padding, use_bias=use_bias,
        batch_normalization=batch_normalization, activation=activation, name=name)  # Pass the up-sampled tensor through a convolution module

    return tensor


def choose_activation_layer(activation: str):
    """
    Choose activation layer for the U-Net.

    Parameters
    ----------
    activation: str
        - Can be any of tf.keras.activations, 'prelu', or 'leaky_relu' (case-insensitive).

    Returns
    -------
    activation_layer: tf.keras.layers.Activation, tf.keras.layers.PReLU, or tf.keras.layers.LeakyReLU
        - Activation layer.
    """

    available_activations = ['elu', 'exponential', 'gelu', 'hard_sigmoid', 'leaky_relu', 'linear', 'prelu', 'relu', 'selu', 'sigmoid', 'softmax',
                            'softplus', 'softsign', 'swish', 'tanh']

    activation_kwargs = dict({})  # Keyword arguments that will be passed into the 'Activation' layer (if applicable)

    # Choose the activation layer
    if activation == 'leaky_relu':
        activation_layer = getattr(layers, 'LeakyReLU')
    elif activation == 'prelu':
        activation_layer = getattr(layers, 'PReLU')
    elif activation in available_activations:
        activation_layer = getattr(layers, 'Activation')
        activation_kwargs['activation'] = activation
    else:
        raise TypeError(f"'{activation}' is not a valid loss function and/or is not available, options are: {', '.join(list(available_activations))}")

    return activation_layer


def deep_supervision_side_output(tensor, num_classes, kernel_size, output_level, upsample_size=2, padding='same', use_bias=True,
    preserve_third_dimension=False, name=None):
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
    upsample_size: int
        - Upsampling factors for rows and columns in the UpSampling2D/UpSampling3D layer. Tuples are currently not supported
        but will be supported in a future update.
    padding: str
        - Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        - If True, a bias vector will be used in the Conv2D/Conv3D layer.
    preserve_third_dimension: bool
        - If True, the third dimension of the image will not be affected by the up-sampling. This is particularly useful if
        the third dimension is small, as up-sampling can destroy features in images with a small vertical dimension.
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
        if preserve_third_dimension is True:
            raise ValueError(f"preserve_third_dimension is True but the tensor is not 5D. Provided tensor shape: {tensor.shape}")
        else:
            if output_level > 1:
                upsample_size_1 = (upsample_size, upsample_size)
            else:
                upsample_size_1 = None

            if output_level > 2:
                upsample_size_2 = (np.power(upsample_size, output_level - 2), np.power(upsample_size, output_level - 2))
            else:
                upsample_size_2 = None

    elif tensor_dims == 5:  # If the image is 3D
        conv_layer = Conv3D
        upsample_layer = UpSampling3D
        if preserve_third_dimension is True:
            if output_level > 1:
                upsample_size_1 = (upsample_size, upsample_size, 1)
            else:
                upsample_size_1 = None

            if output_level > 2:
                upsample_size_2 = (np.power(upsample_size, output_level - 2), np.power(upsample_size, output_level - 2), 1)
            else:
                upsample_size_2 = None
        else:
            if output_level > 1:
                upsample_size_1 = (upsample_size, upsample_size, upsample_size)
            else:
                upsample_size_1 = None

            if output_level > 2:
                upsample_size_2 = (np.power(upsample_size, output_level - 2), np.power(upsample_size, output_level - 2), np.power(upsample_size, output_level - 2))
            else:
                upsample_size_2 = None
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor can only have 4 or 5 dimensions")

    if upsample_size_1 is not None:
        tensor = upsample_layer(size=upsample_size_1, name=f'{name}_UpSampling{tensor_dims - 2}D_1')(tensor)  # Pass the tensor through the UpSampling2D/UpSampling3D layer

    tensor = conv_layer(filters=num_classes, kernel_size=kernel_size, padding=padding, use_bias=use_bias, name=f'{name}_Conv{tensor_dims - 2}D')(tensor)  # This convolution layer contains n_classes filters, one for each class

    if upsample_size_2 is not None:
        tensor = upsample_layer(size=upsample_size_2, name=f'{name}_UpSampling{tensor_dims - 2}D_2')(tensor)  # Pass the tensor through the UpSampling2D/UpSampling3D layer

    sup_output = Softmax(name=f'{name}_Softmax')(tensor)  # Final softmax output

    return sup_output
