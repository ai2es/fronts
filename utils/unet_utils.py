""" Functions for building the U-Net """
import numpy as np
from tensorflow.keras.layers import Conv2D, Conv3D, BatchNormalization, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Softmax
from tensorflow.keras import layers


def convolution_module(tensor, filters, kernel_size, num_modules=1, padding='same', use_bias=False, batch_normalization=True,
    activation='relu', name=None):
    """
    Insert modules into an encoder or decoder node.
    """

    tensor_dims = len(tensor.shape)  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if num_modules < 1:
        raise ValueError("num_modules must be greater than 0, at least one module must be added")

    if tensor_dims == 4:
        conv_layer = Conv2D
    elif tensor_dims == 5:
        conv_layer = Conv3D
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions")

    conv_kwargs = dict({})
    conv_kwargs['filters'] = filters
    conv_kwargs['kernel_size'] = kernel_size
    conv_kwargs['padding'] = padding
    conv_kwargs['use_bias'] = use_bias

    activation_layer = choose_activation_layer(activation)

    activation_kwargs = dict({})
    if activation_layer == layers.Activation:
        activation_kwargs['activation'] = activation

    for module in range(num_modules):

        conv_kwargs['name'] = f'{name}_Conv{tensor_dims - 2}D_{module+1}'
        activation_kwargs['name'] = f'{name}_{activation}_{module+1}'

        conv_tensor = conv_layer(**conv_kwargs)(tensor)
        if batch_normalization is True:
            batch_norm_tensor = BatchNormalization(name=f'{name}_BatchNorm_{module+1}')(conv_tensor)
            activation_tensor = activation_layer(**activation_kwargs)(batch_norm_tensor)
        else:
            activation_tensor = activation_layer(**activation_kwargs)(conv_tensor)
        tensor = activation_tensor

    return tensor


def aggregated_feature_map(tensor, filters, kernel_size, level1, level2, upsample_size=2, padding='same', use_bias=False,
    batch_normalization=True, activation='relu', preserve_third_dimension=False, name=None):
    """ Insert aggregated feature map into the model """

    tensor_dims = len(tensor.shape)  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if level1 <= level2:
        raise ValueError("level2 must be smaller than level1 in aggregated feature maps")

    upsample_kwargs = dict({})
    upsample_kwargs['size'] = np.power(upsample_size, abs(level1 - level2))
    upsample_kwargs['name'] = f'{name}_UpSampling{tensor_dims - 2}D'

    if tensor_dims == 4:
        upsample_layer = UpSampling2D
        upsample_kwargs['size'] = (np.power(upsample_size, abs(level1 - level2)), np.power(upsample_size, abs(level1 - level2)))
        if preserve_third_dimension is True:
            raise ValueError(f"preserve_third_dimension is True but the tensor is not 5D. Provided tensor shape: {tensor.shape}")
    elif tensor_dims == 5:
        upsample_layer = UpSampling3D
        upsample_kwargs['size'] = (np.power(upsample_size, abs(level1 - level2)), np.power(upsample_size, abs(level1 - level2)), np.power(upsample_size, abs(level1 - level2)))
        if preserve_third_dimension is True:
            upsample_kwargs['size'] = (np.power(upsample_size, abs(level1 - level2)), np.power(upsample_size, abs(level1 - level2)), 1)
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions")

    tensor = upsample_layer(**upsample_kwargs)(tensor)

    tensor = convolution_module(tensor, filters, kernel_size, num_modules=1, padding=padding, use_bias=use_bias, name=name,
        batch_normalization=batch_normalization, activation=activation)

    return tensor


def full_scale_skip_connection(tensor, filters, kernel_size, level1, level2, pool_size=2, padding='same', use_bias=False,
    batch_normalization=True, activation='relu', preserve_third_dimension=False, name=None):
    """ Insert full-scale skip connection into the model """

    tensor_dims = len(tensor.shape)  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if level1 >= level2:
        raise ValueError("level2 must be greater than level1 in full-scale skip connections")

    pool_kwargs = dict({})
    pool_kwargs['name'] = f'{name}_MaxPool{tensor_dims - 2}D'

    if tensor_dims == 4:
        pool_layer = MaxPooling2D
        pool_kwargs['pool_size'] = (np.power(pool_size, abs(level1 - level2)), np.power(pool_size, abs(level1 - level2)))
        if preserve_third_dimension is True:
            raise ValueError(f"preserve_third_dimension is True but the tensor is not 5D. Provided tensor shape: {tensor.shape}")
    elif tensor_dims == 5:
        pool_layer = MaxPooling3D
        if preserve_third_dimension is True:
            pool_kwargs['pool_size'] = (np.power(pool_size, abs(level1 - level2)), np.power(pool_size, abs(level1 - level2)), 1)
        else:
            pool_kwargs['pool_size'] = (np.power(pool_size, abs(level1 - level2)), np.power(pool_size, abs(level1 - level2)), np.power(pool_size, abs(level1 - level2)))
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions")

    tensor = pool_layer(**pool_kwargs)(tensor)

    tensor = convolution_module(tensor, filters, kernel_size, num_modules=1, padding=padding, use_bias=use_bias, name=name,
        batch_normalization=batch_normalization, activation=activation)

    return tensor


def conventional_skip_connection(tensor, filters, kernel_size, padding='same', use_bias=False, batch_normalization=True,
    activation='relu', name=None):
    """ Insert skip connection into the model """

    tensor = convolution_module(tensor, filters, kernel_size, num_modules=1, padding=padding, use_bias=use_bias, name=name,
        batch_normalization=batch_normalization, activation=activation)

    return tensor


def max_pool(tensor, pool_size=2, preserve_third_dimension=False, name=None):
    """ Connect two encoder nodes with a max-pooling operation """

    tensor_dims = len(tensor.shape)  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    pool_kwargs = dict({})  # Keyword arguments in the MaxPooling layer
    pool_kwargs['name'] = f'{name}_MaxPool{tensor_dims - 2}D'

    if tensor_dims == 4:
        pool_layer = MaxPooling2D
        pool_kwargs['pool_size'] = (pool_size, pool_size)
        if preserve_third_dimension is True:
            raise ValueError(f"preserve_third_dimension is True but the tensor is not 5D. Provided tensor shape: {tensor.shape}")
    elif tensor_dims == 5:
        pool_layer = MaxPooling3D
        if preserve_third_dimension is True:
            pool_kwargs['pool_size'] = (pool_size, pool_size, 1)
        else:
            pool_kwargs['pool_size'] = (pool_size, pool_size, pool_size)
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions")

    pool_tensor = pool_layer(**pool_kwargs)(tensor)

    return pool_tensor


def upsample(tensor, filters, kernel_size, upsample_size=2, padding='same', use_bias=False, preserve_third_dimension=False, batch_normalization=True,
    activation='relu', name=None):
    """ Connect decoder nodes with an up-sampling operation """

    tensor_dims = len(tensor.shape)  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    upsample_kwargs = dict({})  # Keyword arguments in the UpSampling layer
    upsample_kwargs['name'] = f'{name}_UpSampling{tensor_dims - 2}D'

    if tensor_dims == 4:
        upsample_layer = UpSampling2D
        upsample_kwargs['size'] = (upsample_size, upsample_size)
        if preserve_third_dimension is True:
            raise ValueError(f"preserve_third_dimension is True but the tensor is not 5D. Provided tensor shape: {tensor.shape}")
    elif tensor_dims == 5:
        upsample_layer = UpSampling3D
        if preserve_third_dimension is True:
            upsample_kwargs['size'] = (upsample_size, upsample_size, 1)
        else:
            upsample_kwargs['size'] = (upsample_size, upsample_size, upsample_size)
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor can only have 4 or 5 dimensions")

    upsample_tensor = upsample_layer(**upsample_kwargs)(tensor)

    tensor = convolution_module(upsample_tensor, filters, kernel_size, num_modules=1, padding=padding, use_bias=use_bias, name=name,
        batch_normalization=batch_normalization, activation=activation)

    return tensor


def choose_activation_layer(activation: str):
    """ Choose activation layer for the U-Net """

    available_activations = ['elu', 'exponential', 'gelu', 'hard_sigmoid', 'leaky_relu', 'linear', 'prelu', 'relu', 'selu', 'sigmoid', 'softmax',
                            'softplus', 'softsign', 'swish', 'tanh']

    activation_kwargs = dict({})  # Keyword arguments that will be passed into the 'Activation' layer (if applicable)

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
    """ Deep supervision output on the right side of the U-Net 3+ """

    tensor_dims = len(tensor.shape)  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if tensor_dims == 4:

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

    elif tensor_dims == 5:

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
        tensor = upsample_layer(size=upsample_size_1, name=f'{name}_UpSampling{tensor_dims - 2}D_1')(tensor)

    tensor = conv_layer(filters=num_classes, kernel_size=kernel_size, padding=padding, use_bias=use_bias, name=f'{name}_Conv{tensor_dims - 2}D')(tensor)

    if upsample_size_2 is not None:
        tensor = upsample_layer(size=upsample_size_2, name=f'{name}_UpSampling{tensor_dims - 2}D_2')(tensor)

    sup_output = Softmax(name=f'{name}_Softmax')(tensor)

    return sup_output
