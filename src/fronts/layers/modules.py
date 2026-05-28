from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv3D,
    MaxPooling2D,
    MaxPooling3D,
    Reshape,
    UpSampling2D,
    UpSampling3D,
)

from fronts.layers import keras_builders


def attention_gate(
    x: tf.Tensor,
    g: tf.Tensor,
    kernel_size: int | tuple[int, ...],
    pool_size: tuple[int, ...],
    name: str | None,
):
    """Attention gate for the Attention U-Net.

    Args:
        x: Signal from the encoder node on the same level as the attention gate.
        g: Signal from the level below the attention gate, which has higher-resolution features.
        kernel_size: Size of the kernel in the Conv2D/Conv3D layers. Only applies to layers not forced to kernel size 1.
        pool_size: Pool size for the UpSampling layers and the stride count in the first convolution.
        name: Prefix of the layer names. Auto-assigned when None.

    References:
        https://towardsdatascience.com/a-detailed-explanation-of-the-attention-u-net-b371a5590831
    """
    conv_layer = getattr(
        tf.keras.layers, f"Conv{len(x.shape) - 2}D"
    )  # Select the convolution layer for the x and g tensors
    upsample_layer = getattr(tf.keras.layers, f"UpSampling{len(x.shape) - 2}D")  # Select the upsampling layer

    shape_x = x.shape  # Shapes of the ORIGINAL inputs
    filters_x = shape_x[-1]

    x_conv = conv_layer(
        filters=filters_x,
        kernel_size=kernel_size,
        strides=pool_size,
        padding="same",
        name=f"{name}_Conv{len(x.shape) - 2}D_x",
    )(x)
    g_conv = conv_layer(
        filters=filters_x,
        kernel_size=1,
        padding="same",
        name=f"{name}_Conv{len(x.shape) - 2}D_g",
    )(g)

    xg = tf.add(x_conv, g_conv, name=f"{name}_sum")  # Sum the x and g signals element-wise
    xg = Activation(activation="relu", name=f"{name}_relu")(
        xg
    )  # Pass the summed signals through a ReLU activation layer

    xg_collapse = conv_layer(filters=1, kernel_size=1, padding="same", name=f"{name}_collapse")(
        xg
    )  # Collapse the number of filters to just 1
    xg_collapse = Activation(activation="sigmoid", name=f"{name}_sigmoid")(
        xg_collapse
    )  # Pass collapsed tensor through a sigmoid activation layer

    # Upsample the collapsed tensor to match x's spatial shape, then expand filters to match g
    upsample_xg = upsample_layer(size=pool_size, name=f"{name}_UpSampling{len(x.shape) - 2}D")(xg_collapse)
    upsample_xg = tf.repeat(upsample_xg, filters_x, axis=-1, name=f"{name}_repeat")

    coeffs = tf.multiply(
        upsample_xg, x, name=f"{name}_multiply"
    )  # Element-wise multiplication onto the original x signal

    attention_tensor = conv_layer(
        filters=filters_x,
        kernel_size=1,
        strides=1,
        padding="same",
        name=f"{name}_Conv{len(x.shape) - 2}D_coeffs",
    )(coeffs)
    attention_tensor = BatchNormalization(name=f"{name}_BatchNorm")(attention_tensor)

    return attention_tensor


def convolution(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: int | tuple[int, ...],
    num_modules: int = 1,
    padding: str = "same",
    use_bias: bool = False,
    batch_normalization: bool = True,
    activation: str = "relu",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    shared_axes=None,
    name: str | None = None,
):
    """Insert convolution modules into an encoder or decoder node.

    Args:
        tensor: Input tensor for the convolution module(s).
        filters: Number of filters in the Conv2D/Conv3D layer(s).
        kernel_size: Size of the kernel in the Conv2D/Conv3D layer(s).
        num_modules: Number of convolution modules to insert. Must be greater than 0.
        padding: Padding in the Conv2D/Conv3D layer(s). 'same' pads to preserve shape; 'valid' applies no padding.
        use_bias: If True, a bias vector is used in the Conv2D/Conv3D layers.
        batch_normalization: If True, a BatchNormalization layer follows every Conv2D/Conv3D layer.
        activation: Activation function after every Conv2D/Conv3D (or BatchNormalization) layer.
        kernel_initializer: Initializer for the kernel weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer applied to the kernel weights matrix.
        bias_regularizer: Regularizer applied to the bias vector.
        activity_regularizer: Regularizer applied to the layer output.
        kernel_constraint: Constraint applied to the kernel matrix.
        bias_constraint: Constraint applied to the bias vector.
        shared_axes: Axes along which to share learnable parameters for the activation function.
        name: Prefix of the layer names. Auto-assigned when None.

    Returns:
        Output tensor.

    Raises:
        ValueError: If num_modules < 1.
        TypeError: If the tensor does not have 4 or 5 dimensions.
    """
    tensor_dims = len(
        tensor.shape
    )  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if num_modules < 1:
        raise ValueError("num_modules must be greater than 0, at least one module must be added")

    if tensor_dims == 4:  # (batch, x, y, channels)
        conv_layer = Conv2D
    elif tensor_dims == 5:  # (batch, x, y, z, channels)
        conv_layer = Conv3D
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions")

    # Arguments for the Conv2D/Conv3D layer.
    conv_kwargs: dict[str, Any] = {}
    for arg in [
        "filters",
        "use_bias",
        "kernel_size",
        "padding",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
    ]:
        conv_kwargs[arg] = locals()[arg]

    activation_kwargs = {}
    if activation in [
        "prelu",
        "smelu",
        "snake",
    ]:  # these activation functions have learnable parameters
        activation_kwargs["shared_axes"] = shared_axes

    # Insert convolution modules
    for module in range(num_modules):
        # Create names for the Conv2D/Conv3D layers and the activation layer.
        conv_kwargs["name"] = f"{name}_Conv{tensor_dims - 2}D_{module + 1}"
        activation_kwargs["name"] = f"{name}_{activation}_{module + 1}"

        conv_tensor = conv_layer(**conv_kwargs)(tensor)  # Perform convolution on the input tensor

        activation_config = keras_builders.ActivationConfig(
            name=activation,  # pyrefly: ignore[bad-argument-type]
            config=activation_kwargs,
        )
        activation_layer = activation_config.build()

        if batch_normalization:
            batch_norm_tensor = BatchNormalization(name=f"{name}_BatchNorm_{module + 1}")(
                conv_tensor
            )  # Insert layer for batch normalization
            activation_tensor = activation_layer(
                batch_norm_tensor
            )  # Pass output tensor from BatchNormalization into the activation layer
        else:
            activation_tensor = activation_layer(
                conv_tensor
            )  # Pass output tensor from the convolution layer into the activation layer.

        tensor = activation_tensor

    return tensor


def aggregated_feature_map(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: int | tuple[int, ...],
    level1: int,
    level2: int,
    upsample_size: tuple[int, ...],
    padding: str = "same",
    use_bias: bool = False,
    batch_normalization: bool = True,
    activation: str = "relu",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    shared_axes=None,
    name: str | None = None,
):
    """Connect two U-Net 3+ nodes with an aggregated feature map (AFM).

    Args:
        tensor: Input tensor for the convolution module.
        filters: Number of filters in the Conv2D/Conv3D layer.
        kernel_size: Size of the kernel in the Conv2D/Conv3D layer.
        level1: Level of the first (lower, source) node. Must be greater than level2.
        level2: Level of the second (higher, destination) node. Must be smaller than level1.
        upsample_size: Upsampling size for rows and columns in the UpSampling2D/UpSampling3D layer.
        padding: Padding in the Conv2D/Conv3D layer. 'same' pads to preserve shape; 'valid' applies no padding.
        use_bias: If True, a bias vector is used in the Conv2D/Conv3D layer.
        batch_normalization: If True, a BatchNormalization layer follows the Conv2D/Conv3D layer.
        activation: Activation function after the Conv2D/Conv3D (or BatchNormalization) layer.
        kernel_initializer: Initializer for the kernel weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer applied to the kernel weights matrix.
        bias_regularizer: Regularizer applied to the bias vector.
        activity_regularizer: Regularizer applied to the layer output.
        kernel_constraint: Constraint applied to the kernel matrix.
        bias_constraint: Constraint applied to the bias vector.
        shared_axes: Axes along which to share learnable parameters for the activation function.
        name: Prefix of the layer names. Auto-assigned when None.

    Returns:
        Output tensor.

    Raises:
        ValueError: If level1 <= level2.
        TypeError: If the tensor does not have 4 or 5 dimensions.
    """
    tensor_dims = len(
        tensor.shape
    )  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if level1 <= level2:
        raise ValueError("level2 must be smaller than level1 in aggregated feature maps")

    # Arguments for the convolution module.
    module_kwargs: dict[str, Any] = {"num_modules": 1}
    for arg in [
        "filters",
        "kernel_size",
        "padding",
        "use_bias",
        "batch_normalization",
        "activation",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "shared_axes",
        "name",
    ]:
        module_kwargs[arg] = locals()[arg]

    # Keyword arguments for the UpSampling2D/UpSampling3D layers
    upsample_kwargs: dict[str, Any] = {}
    upsample_kwargs["name"] = f"{name}_UpSampling{tensor_dims - 2}D"
    upsample_kwargs["size"] = np.power(upsample_size, abs(level1 - level2))

    if tensor_dims == 4:  # If the image is 2D
        upsample_layer = UpSampling2D
        if len(upsample_size) != 2:
            raise TypeError(f"For 2D up-sampling, pool size must have 2 integers. Got shape: {np.shape(upsample_size)}")
    elif tensor_dims == 5:  # If the image is 3D
        upsample_layer = UpSampling3D
        if len(upsample_size) != 3:
            raise TypeError(f"For 3D up-sampling, pool size must have 3 integers. Got shape: {np.shape(upsample_size)}")
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions")

    tensor = upsample_layer(**upsample_kwargs)(tensor)  # Pass the tensor through the UpSample2D/UpSample3D layer

    tensor = convolution(tensor, **module_kwargs)  # Pass input tensor through convolution module

    return tensor


def full_scale_skip_connection(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: int | tuple[int, ...],
    level1: int,
    level2: int,
    pool_size: tuple[int, ...],
    padding: str = "same",
    use_bias: bool = False,
    batch_normalization: bool = True,
    activation: str = "relu",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    shared_axes=None,
    name: str | None = None,
):
    """Connect two U-Net 3+ nodes with a full-scale skip connection (FSC).

    Args:
        tensor: Input tensor for the convolution module.
        filters: Number of filters in the Conv2D/Conv3D layer.
        kernel_size: Size of the kernel in the Conv2D/Conv3D layer.
        level1: Level of the first (higher, source) node. Must be smaller than level2.
        level2: Level of the second (lower, destination) node. Must be greater than level1.
        pool_size: Pool size for the MaxPooling2D/MaxPooling3D layer.
        padding: Padding in the Conv2D/Conv3D layer. 'same' pads to preserve shape; 'valid' applies no padding.
        use_bias: If True, a bias vector is used in the Conv2D/Conv3D layer.
        batch_normalization: If True, a BatchNormalization layer follows the Conv2D/Conv3D layer.
        activation: Activation function after the Conv2D/Conv3D (or BatchNormalization) layer.
        kernel_initializer: Initializer for the kernel weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer applied to the kernel weights matrix.
        bias_regularizer: Regularizer applied to the bias vector.
        activity_regularizer: Regularizer applied to the layer output.
        kernel_constraint: Constraint applied to the kernel matrix.
        bias_constraint: Constraint applied to the bias vector.
        shared_axes: Axes along which to share learnable parameters for the activation function.
        name: Prefix of the layer names. Auto-assigned when None.

    Returns:
        Output tensor.

    Raises:
        ValueError: If level1 >= level2.
        TypeError: If the tensor does not have 4 or 5 dimensions.
    """
    tensor_dims = len(
        tensor.shape
    )  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if level1 >= level2:
        raise ValueError("level2 must be greater than level1 in full-scale skip connections")

    # Arguments for the convolution module.
    module_kwargs: dict[str, Any] = {"num_modules": 1}
    for arg in [
        "filters",
        "kernel_size",
        "padding",
        "use_bias",
        "batch_normalization",
        "activation",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "shared_axes",
        "name",
    ]:
        module_kwargs[arg] = locals()[arg]

    # Keyword arguments for the MaxPooling2D/MaxPooling3D layer
    pool_kwargs: dict[str, Any] = {}
    pool_kwargs["name"] = f"{name}_MaxPool{tensor_dims - 2}D"
    pool_kwargs["pool_size"] = np.power(pool_size, abs(level1 - level2))

    if tensor_dims == 4:  # If the image is 2D
        pool_layer = MaxPooling2D
    elif tensor_dims == 5:  # If the image is 3D
        pool_layer = MaxPooling3D
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions")

    tensor = pool_layer(**pool_kwargs)(tensor)  # Pass the tensor through the MaxPooling2D/MaxPooling3D layer

    tensor = convolution(tensor, **module_kwargs)  # Pass the tensor through the convolution module

    return tensor


def conventional_skip_connection(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: int | tuple[int, ...],
    padding: str = "same",
    use_bias: bool = False,
    batch_normalization: bool = True,
    activation: str = "relu",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    shared_axes=None,
    name: str | None = None,
):
    """Connect two U-Net 3+ nodes with a conventional skip connection.

    Args:
        tensor: Input tensor for the convolution module.
        filters: Number of filters in the Conv2D/Conv3D layer.
        kernel_size: Size of the kernel in the Conv2D/Conv3D layer.
        padding: Padding in the Conv2D/Conv3D layer. 'same' pads to preserve shape; 'valid' applies no padding.
        use_bias: If True, a bias vector is used in the Conv2D/Conv3D layer.
        batch_normalization: If True, a BatchNormalization layer follows the Conv2D/Conv3D layer.
        activation: Activation function after the Conv2D/Conv3D (or BatchNormalization) layer.
        kernel_initializer: Initializer for the kernel weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer applied to the kernel weights matrix.
        bias_regularizer: Regularizer applied to the bias vector.
        activity_regularizer: Regularizer applied to the layer output.
        kernel_constraint: Constraint applied to the kernel matrix.
        bias_constraint: Constraint applied to the bias vector.
        shared_axes: Axes along which to share learnable parameters for the activation function.
        name: Prefix of the layer names. Auto-assigned when None.

    Returns:
        Output tensor.
    """
    # Arguments for the convolution module.
    module_kwargs: dict[str, Any] = {"num_modules": 1}
    for arg in [
        "filters",
        "kernel_size",
        "padding",
        "use_bias",
        "batch_normalization",
        "activation",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "shared_axes",
        "name",
    ]:
        module_kwargs[arg] = locals()[arg]

    tensor = convolution(tensor, **module_kwargs)  # Pass the tensor through the convolution module

    return tensor


def max_pool(tensor: tf.Tensor, pool_size: tuple[int, ...], name: str | None = None):
    """Connect two encoder nodes with a max-pooling operation.

    Args:
        tensor: Input tensor for the max-pooling operation.
        pool_size: Pool size for the MaxPooling2D/MaxPooling3D layer.
        name: Prefix of the layer names. Auto-assigned when None.

    Returns:
        Output tensor after max-pooling.

    Raises:
        TypeError: If pool_size is not a tuple or list, or does not match the tensor's spatial dimensions.
        TypeError: If the tensor does not have 4 or 5 dimensions.
    """
    if not isinstance(pool_size, tuple) and not isinstance(pool_size, list):
        raise TypeError(f"pool_size can only be a tuple or list. Received type: {type(pool_size)}")

    tensor_dims = len(
        tensor.shape
    )  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    pool_kwargs: dict[str, Any] = {}  # Keyword arguments in the MaxPooling layer
    pool_kwargs["name"] = f"{name}_MaxPool{tensor_dims - 2}D"
    pool_kwargs["pool_size"] = pool_size

    if tensor_dims == 4:  # If the image is 2D
        pool_layer = MaxPooling2D
        if len(pool_size) != 2:
            raise TypeError(f"For 2D max-pooling, pool size must have 2 integers. Got shape: {np.shape(pool_size)}")
    elif tensor_dims == 5:  # If the image is 3D
        pool_layer = MaxPooling3D
        if len(pool_size) != 3:
            raise TypeError(f"For 3D max-pooling, pool size must have 3 integers. Got shape: {np.shape(pool_size)}")
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions")

    pool_tensor = pool_layer(**pool_kwargs)(tensor)  # Pass the tensor through the MaxPooling2D/MaxPooling3D layer

    return pool_tensor


def upsample(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: int | tuple[int, ...],
    upsample_size: tuple[int, ...],
    padding: str = "same",
    use_bias: bool = False,
    batch_normalization: bool = True,
    activation: str = "relu",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    shared_axes=None,
    name: str | None = None,
):
    """Connect decoder nodes with an up-sampling operation followed by a convolution.

    Args:
        tensor: Input tensor for the up-sampling operation.
        filters: Number of filters in the Conv2D/Conv3D layer.
        kernel_size: Size of the kernel in the Conv2D/Conv3D layer.
        upsample_size: Upsampling size in the UpSampling2D/UpSampling3D layer.
        padding: Padding in the Conv2D/Conv3D layer. 'same' pads to preserve shape; 'valid' applies no padding.
        use_bias: If True, a bias vector is used in the Conv2D/Conv3D layer.
        batch_normalization: If True, a BatchNormalization layer follows the Conv2D/Conv3D layer.
        activation: Activation function after the Conv2D/Conv3D (or BatchNormalization) layer.
        kernel_initializer: Initializer for the kernel weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer applied to the kernel weights matrix.
        bias_regularizer: Regularizer applied to the bias vector.
        activity_regularizer: Regularizer applied to the layer output.
        kernel_constraint: Constraint applied to the kernel matrix.
        bias_constraint: Constraint applied to the bias vector.
        shared_axes: Axes along which to share learnable parameters for the activation function.
        name: Prefix of the layer names. Auto-assigned when None.

    Returns:
        Output tensor after up-sampling and convolution.

    Raises:
        TypeError: If upsample_size is not a tuple or list, or does not match the tensor's spatial dimensions.
        TypeError: If the tensor does not have 4 or 5 dimensions.
    """
    tensor_dims = len(
        tensor.shape
    )  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if not isinstance(upsample_size, tuple) and not isinstance(upsample_size, list):
        raise TypeError(f"upsample_size can only be a tuple or list. Received type: {type(upsample_size)}")

    # Arguments for the convolution module.
    module_kwargs: dict[str, Any] = {"num_modules": 1}
    for arg in [
        "filters",
        "kernel_size",
        "padding",
        "use_bias",
        "batch_normalization",
        "activation",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "shared_axes",
        "name",
    ]:
        module_kwargs[arg] = locals()[arg]

    # Keyword arguments in the UpSampling layer
    upsample_kwargs: dict[str, Any] = {}
    upsample_kwargs["name"] = f"{name}_UpSampling{tensor_dims - 2}D"
    upsample_kwargs["size"] = upsample_size

    if tensor_dims == 4:  # If the image is 2D
        upsample_layer = UpSampling2D
        if len(upsample_size) != 2:
            raise TypeError(f"For 2D up-sampling, pool size must have 2 integers. Got shape: {np.shape(upsample_size)}")
    elif tensor_dims == 5:  # If the image is 3D
        upsample_layer = UpSampling3D
        if len(upsample_size) != 3:
            raise TypeError(f"For 3D up-sampling, pool size must have 3 integers. Got shape: {np.shape(upsample_size)}")
    else:
        raise TypeError(f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions")

    upsample_tensor = upsample_layer(**upsample_kwargs)(
        tensor
    )  # Pass the tensor through the UpSampling2D/UpSampling3D layer

    tensor = convolution(upsample_tensor, **module_kwargs)  # Pass the up-sampled tensor through a convolution module

    return tensor


def deep_supervision_side_output(
    tensor: tf.Tensor,
    num_classes: int,
    kernel_size: int | tuple[int, ...],
    output_level: int,
    upsample_size: tuple[int, ...],
    activation: str = "softmax",
    padding: str = "same",
    use_bias: bool = False,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    squeeze_axes: int | tuple[int, ...] | None = None,
    name: str | None = None,
):
    """Deep supervision side output for a U-Net 3+ decoder node or a standard U-Net's final node.

    Args:
        tensor: Input tensor for the convolution module.
        num_classes: Number of classes the model is predicting.
        kernel_size: Size of the kernel in the Conv2D/Conv3D layer.
        output_level: Level of the decoder node providing the deep supervision output.
        upsample_size: Upsampling size for rows and columns in the UpSampling2D/UpSampling3D layer.
        activation: Output activation function.
        padding: Padding in the Conv2D/Conv3D layer. 'same' pads to preserve shape; 'valid' applies no padding.
        use_bias: If True, a bias vector is used in the Conv2D/Conv3D layer.
        kernel_initializer: Initializer for the kernel weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer applied to the kernel weights matrix.
        bias_regularizer: Regularizer applied to the bias vector.
        activity_regularizer: Regularizer applied to the layer output.
        kernel_constraint: Constraint applied to the kernel matrix.
        bias_constraint: Constraint applied to the bias vector.
        squeeze_axes: Axis or axes of the input tensor to squeeze.
        name: Prefix of the layer names. Auto-assigned when None.

    Returns:
        Output tensor after convolution, optional up-sampling, and activation.

    Raises:
        TypeError: If the tensor does not have 4 or 5 dimensions.
    """
    tensor_dims = len(
        tensor.shape
    )  # Number of dims in the tensor (including the first 'None' dimension for batch size)

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
    conv_kwargs: dict[str, Any] = {}
    conv_kwargs["name"] = f"{name}_Conv{tensor_dims - 2}D"
    for arg in [
        "use_bias",
        "kernel_size",
        "padding",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
    ]:
        conv_kwargs[arg] = locals()[arg]

    if upsample_size_1 is not None:
        tensor = upsample_layer(size=upsample_size_1, name=f"{name}_UpSampling{tensor_dims - 2}D_1")(
            tensor
        )  # Pass the tensor through the UpSampling2D/UpSampling3D layer

    tensor = conv_layer(filters=num_classes, **conv_kwargs)(
        tensor
    )  # This convolution layer contains num_classes filters, one for each class

    if upsample_size_2 is not None:
        tensor = upsample_layer(size=upsample_size_2, name=f"{name}_UpSampling{tensor_dims - 2}D_2")(
            tensor
        )  # Pass the tensor through the UpSampling2D/UpSampling3D layer

    ### Squeeze the given dimensions/axes ###
    if squeeze_axes is not None:
        conv_kwargs["kernel_size"] = [1 for _ in range(tensor_dims - 2)]

        squeeze_axes_list: list[int] = [squeeze_axes] if isinstance(squeeze_axes, int) else list(squeeze_axes)

        squeeze_axes_sizes = [tensor.shape[ax_to_squeeze] for ax_to_squeeze in squeeze_axes_list]

        for ax, size in enumerate(squeeze_axes_sizes):
            # Set kernel to full dimension size so the output collapses to 1
            conv_kwargs["kernel_size"][squeeze_axes_list[ax] - 1] = size

        # "same" padding would preserve size; "valid" lets the kernel shrink the dim to 1
        conv_kwargs["padding"] = "valid"
        conv_kwargs["name"] = f"{name}_Conv{tensor_dims - 2}D_collapse"

        tensor = conv_layer(filters=num_classes, **conv_kwargs)(
            tensor
        )  # This convolution layer contains num_classes filters, one for each class
        # Remove singleton spatial dimensions (keep batch + channels)
        new_shape = []

        for _, dim in enumerate(tensor.shape[1:]):  # skip batch axis
            if dim != 1:
                new_shape.append(dim)

        tensor = Reshape(tuple(new_shape), name=f"{name}_Reshape")(tensor)

    activation_kwargs = {"name": f"{name}_{activation}"}
    activation_config = keras_builders.ActivationConfig(
        name=activation,  # pyrefly: ignore[bad-argument-type]
        config=activation_kwargs,
    )
    activation_layer = activation_config.build()
    sup_output = activation_layer(tensor)  # Final softmax output

    return sup_output
