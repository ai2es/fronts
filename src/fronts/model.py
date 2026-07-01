import dataclasses
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model

from fronts.layers import modules


@tf.keras.utils.register_keras_serializable(package="fronts")
class SharedTargetModel(Model):
    """A Model that trains multiple outputs against one shared target.

    Deep supervision gives this model one output per decoder level, all
    predicting the same front segmentation map. Keras 3 requires ``y`` to
    structurally match the (list-valued) ``y_pred`` for loss/metric
    computation, so this overrides the two hooks responsible for that
    matching to broadcast a single target across every output instead of
    requiring the caller to replicate it ahead of time.
    """

    def compute_loss(
        self,
        x: tf.Tensor | None = None,
        y: tf.Tensor | list[tf.Tensor] | None = None,
        y_pred: tf.Tensor | list[tf.Tensor] | None = None,
        sample_weight: tf.Tensor | None = None,
        training: bool = True,
    ) -> tf.Tensor:
        """Broadcasts a single target across all outputs before delegating to the default loss computation."""
        y_in = [y] * len(y_pred) if isinstance(y_pred, (list, tuple)) else y
        return super().compute_loss(x=x, y=y_in, y_pred=y_pred, sample_weight=sample_weight, training=training)

    def compute_metrics(
        self,
        x: tf.Tensor | None = None,
        y: tf.Tensor | list[tf.Tensor] | None = None,
        y_pred: tf.Tensor | list[tf.Tensor] | None = None,
        sample_weight: tf.Tensor | None = None,
    ) -> dict[str, tf.Tensor]:
        """Broadcasts a single target across all outputs before delegating to the default metrics computation."""
        y_in = [y] * len(y_pred) if isinstance(y_pred, (list, tuple)) else y
        return super().compute_metrics(x=x, y=y_in, y_pred=y_pred, sample_weight=sample_weight)


@dataclasses.dataclass
class ModelConfig:
    """Hyperparameter configuration for building a UNet3Plus model."""

    n_classes: int = 6
    n_channels: int = 30
    levels: int = 4
    filter_num: list[int] = dataclasses.field(default_factory=lambda: [32, 64, 128, 256])
    pool_size: tuple[int, int] = (2, 2)
    upsample_size: tuple[int, int] = (2, 2)
    squeeze_axes: int | None = None
    kernel_size: int = 3
    first_encoder_connections: bool = False
    deep_supervision: bool = True
    batch_normalization: bool = True
    activation: str = "gelu"
    output_activation: str = "softmax"
    modules_per_node: int = 2


@dataclasses.dataclass
class UNetBase:
    """Base class for a U-Net model.

    Attributes:
        input_shape: Shape of the inputs. The last number in the tuple represents the number of
            channels/predictors.
        num_classes: Number of classes/labels that the U-Net will try to predict.
        pool_size: Size of the mask in the MaxPooling layers.
        upsample_size: Size of the mask in the UpSampling layers.
        levels: Number of levels in the U-Net. Must be greater than 1.
        filter_num: Number of convolution filters on each level of the U-Net.
        kernel_size: Size of the kernel in the convolution layers.
        squeeze_axes: Axis or axes of the input tensor to squeeze.
        shared_axes: Axes along which to share the learnable parameters for the activation function.
            When left as None, parameters will be shared along all arbitrary dimensions.
        modules_per_node: Number of modules in each node of the U-Net.
        batch_normalization: If True, adds a batch normalization layer after every convolution in the modules.
        activation: Activation function to use in the modules. See utils.choose_activation_layer for
            all supported activation functions.
        output_activation: Output activation function.
        padding: Padding to use in the convolution layers.
        use_bias: If True, implements a bias vector in the convolution layers used in the modules.
        kernel_initializer: Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
        bias_initializer: Initializer for the bias vector in the Conv2D/Conv3D layers.
        kernel_regularizer: Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
        bias_regularizer: Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
        activity_regularizer: Regularizer function applied to the output of the Conv2D/Conv3D layers.
        kernel_constraint: Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
        bias_constraint: Constraint function applied to the bias vector in the Conv2D/Conv3D layers.

    Raises:
        ValueError: If levels < 2.
        ValueError: If input_shape does not have 3 or 4 dimensions.
        ValueError: If the length of filter_num does not match the number of levels.

    References:
        https://arxiv.org/pdf/1505.04597.pdf
    """

    input_shape: tuple[int | None, ...]
    num_classes: int
    pool_size: int | tuple[int, ...] | list[int]
    upsample_size: int | tuple[int, ...] | list[int]
    levels: int
    filter_num: tuple[int, ...] | list[int]
    kernel_size: int = 3
    squeeze_axes: int | tuple[int, ...] | list[int] | None = None
    shared_axes: int | tuple[int, ...] | list[int] | None = None
    modules_per_node: int = 2
    batch_normalization: bool = True
    activation: str = "relu"
    output_activation: str = "softmax"
    padding: str = "same"
    use_bias: bool = True
    kernel_initializer: str = "glorot_uniform"
    bias_initializer: str = "zeros"
    kernel_regularizer: str | None = None
    bias_regularizer: str | None = None
    activity_regularizer: str | None = None
    kernel_constraint: str | None = None
    bias_constraint: str | None = None

    def __post_init__(self) -> None:
        """Validate model configuration after dataclass initialization."""
        if self.levels not in [3, 4]:
            raise ValueError(
                "Input_shape can only have 3 or 4 dimensions (2D image + 1 dimension "
                "for channels OR a 3D image + 1 dimension for channels). Received "
                f"shape: {self.input_shape}"
            )
        if len(self.filter_num) != self.levels:
            raise ValueError(
                f"Length of filter_num ({len(self.filter_num)}) does not match the number of levels ({self.levels})"
            )
        self.ndims = (
            len(self.input_shape) - 1
        )  # Number of dimensions in the input image (excluding the last dimension reserved for channels)

        # Keyword arguments for the convolution modules
        self.module_kwargs: dict[str, Any] = {}
        self.module_kwargs["num_modules"] = self.modules_per_node
        for arg in [
            "activation",
            "batch_normalization",
            "padding",
            "kernel_size",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "activity_regularizer",
            "kernel_constraint",
            "bias_constraint",
            "shared_axes",
        ]:
            self.module_kwargs[arg] = getattr(self, arg)

        # MaxPooling keyword arguments
        self.pool_kwargs: dict[str, Any] = {"pool_size": self.pool_size}

        # Keyword arguments for upsampling
        self.upsample_kwargs: dict[str, Any] = {}
        for arg in [
            "activation",
            "batch_normalization",
            "padding",
            "kernel_size",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "activity_regularizer",
            "kernel_constraint",
            "bias_constraint",
            "upsample_size",
            "shared_axes",
        ]:
            self.upsample_kwargs[arg] = getattr(self, arg)

        # Keyword arguments for the deep supervision output in the final decoder node
        self.supervision_kwargs: dict[str, Any] = {}
        for arg in [
            "padding",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "activity_regularizer",
            "kernel_constraint",
            "bias_constraint",
            "upsample_size",
            "squeeze_axes",
            "num_classes",
        ]:
            self.supervision_kwargs[arg] = getattr(self, arg)


@dataclasses.dataclass
class UNet3Plus(UNetBase):
    """Builds a U-Net 3+ model with optional input normalization.

    Args:
        input_shape: Shape of the inputs. The last number in the tuple
            represents the number of channels/predictors.
        num_classes: Number of classes/labels that the U-Net 3+ will try
            to predict.
        pool_size: Size of the mask in the MaxPooling layers.
        upsample_size: Size of the mask in the UpSampling layers.
        levels: Number of levels in the U-Net 3+. Must be greater than 2.
        filter_num: Number of convolution filters in each encoder of the
            U-Net 3+. The length must be equal to ``levels``.
        filter_num_skip: Number of convolution filters in the conventional
            skip connections, full-scale skip connections, and aggregated
            feature maps. Defaults to the first value in ``filter_num``
            when left as None.
        filter_num_aggregate: Number of convolution filters in the decoder
            nodes after feature maps are concatenated. Defaults to
            ``filter_num_skip * levels`` when left as None.
        kernel_size: Size of the kernel in the convolution layers.
        first_encoder_connections: If True, creates full-scale skip
            connections from the first encoder node to all subsequent
            encoder levels, as described in the original UNET 3+ paper.
        squeeze_axes: Axis or axes of the input tensor to squeeze.
        shared_axes: Axes along which to share the learnable parameters
            for the activation function. When left as None, parameters
            will be shared along all arbitrary dimensions.
        modules_per_node: Number of convolution modules in each node of
            the U-Net 3+.
        batch_normalization: If True, adds a batch normalization layer
            after every convolution in the modules.
        deep_supervision: If True, adds deep supervision side outputs to
            each decoder node. The final decoder node always has deep
            supervision regardless of this setting.
        activation: Activation function to use in the modules. See
            ``utils.choose_activation_layer`` for all supported activation
            functions.
        output_activation: Output activation function.
        padding: Padding to use in the convolution layers.
        use_bias: If True, implements a bias vector in the convolution
            layers used in the modules.
        kernel_initializer: Initializer for the kernel weights matrix in
            the Conv2D/Conv3D layers.
        bias_initializer: Initializer for the bias vector in the
            Conv2D/Conv3D layers.
        kernel_regularizer: Regularizer function applied to the kernel
            weights matrix in the Conv2D/Conv3D layers.
        bias_regularizer: Regularizer function applied to the bias vector
            in the Conv2D/Conv3D layers.
        activity_regularizer: Regularizer function applied to the output
            of the Conv2D/Conv3D layers.
        kernel_constraint: Constraint function applied to the kernel
            matrix of the Conv2D/Conv3D layers.
        bias_constraint: Constraint function applied to the bias vector
            in the Conv2D/Conv3D layers.
        normalization_mean: Per-channel mean of shape (n_channels,). When provided
            alongside ``normalization_variance``, a ``tf.keras.layers.Normalization``
            layer is prepended with these statistics baked in as non-trainable weights.
            Raw unnormalized inputs can then be passed directly to the saved model.
        normalization_variance: Per-channel variance of shape (n_channels,). Must be
            provided together with ``normalization_mean``.

    Returns:
        A ``tf.keras.models.Model`` object representing the U-Net 3+ model.

    Raises:
        ValueError: If ``levels`` is less than 3.
        ValueError: If ``input_shape`` does not have 3 or 4 dimensions.
        ValueError: If the length of ``filter_num`` does not match
            ``levels``.

    References:
        Huang et al. (2020): https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf
    """

    filter_num_skip: int | None = None
    filter_num_aggregate: int | None = None
    first_encoder_connections: bool = False
    deep_supervision: bool = False
    normalization_mean: np.ndarray | None = None
    normalization_variance: np.ndarray | None = None

    def build(self) -> tf.keras.Model:
        """Builds and returns the U-Net 3+ Keras model."""
        ndims = len(self.input_shape) - 1

        if self.levels < 3:
            raise ValueError(f"levels must be greater than 2. Received value: {self.levels}")
        if len(self.input_shape) > 4 or len(self.input_shape) < 3:
            raise ValueError(f"input_shape can only have 3 or 4 dimensions. Received shape: {self.input_shape}")
        if len(self.filter_num) != self.levels:
            raise ValueError(
                f"length of filter_num ({len(self.filter_num)}) does not match the number of levels ({self.levels})"
            )

        filter_num_skip = self.filter_num[0] if self.filter_num_skip is None else self.filter_num_skip
        filter_num_aggregate = (
            self.levels * filter_num_skip if self.filter_num_aggregate is None else self.filter_num_aggregate
        )

        module_kwargs: dict[str, Any] = {}
        for arg in [
            "activation",
            "batch_normalization",
            "padding",
            "kernel_size",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "activity_regularizer",
            "kernel_constraint",
            "bias_constraint",
            "shared_axes",
        ]:
            module_kwargs[arg] = getattr(self, arg)
        module_kwargs["num_modules"] = self.modules_per_node

        pool_kwargs: dict[str, Any] = {"pool_size": self.pool_size}

        upsample_kwargs: dict[str, Any] = {}
        conventional_kwargs: dict[str, Any] = {}
        full_scale_kwargs: dict[str, Any] = {}
        aggregated_kwargs: dict[str, Any] = {}
        for arg in [
            "activation",
            "batch_normalization",
            "kernel_size",
            "padding",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "activity_regularizer",
            "kernel_constraint",
            "bias_constraint",
            "shared_axes",
        ]:
            upsample_kwargs[arg] = getattr(self, arg)
            conventional_kwargs[arg] = getattr(self, arg)
            full_scale_kwargs[arg] = getattr(self, arg)
            aggregated_kwargs[arg] = getattr(self, arg)

        conventional_kwargs["filters"] = filter_num_skip
        upsample_kwargs["filters"] = filter_num_skip
        upsample_kwargs["upsample_size"] = self.upsample_size
        full_scale_kwargs["filters"] = filter_num_skip
        full_scale_kwargs["pool_size"] = self.pool_size
        aggregated_kwargs["filters"] = filter_num_skip
        aggregated_kwargs["upsample_size"] = self.upsample_size

        supervision_kwargs: dict[str, Any] = {}
        for arg in [
            "kernel_size",
            "padding",
            "squeeze_axes",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "activity_regularizer",
            "kernel_constraint",
            "bias_constraint",
            "upsample_size",
        ]:
            supervision_kwargs[arg] = getattr(self, arg)
        supervision_kwargs["activation"] = self.output_activation
        supervision_kwargs["use_bias"] = True

        tensors: dict[str, Any] = {}
        tensors_with_supervision = []

        # Input + optional normalization layer
        # The normalization layer is adapted on X_train and its mean/std
        # are saved inside the model weights when you call model.save().
        # No separate normalization file is needed at inference time.
        tensors["input"] = Input(shape=self.input_shape, name="Input")

        if self.normalization_mean is not None and self.normalization_variance is not None:
            norm_layer = tf.keras.layers.Normalization(
                axis=-1,
                mean=self.normalization_mean,
                variance=self.normalization_variance,
                name="input_normalization",
            )
            first_tensor = norm_layer(tensors["input"])
        else:
            first_tensor = tensors["input"]

        # First encoder node — receives normalized input (or raw if no norm)
        tensors["En1"] = modules.convolution(
            first_tensor,  # <-- only change to the original build() logic
            filters=self.filter_num[0],
            name="En1",
            **module_kwargs,
        )

        if self.first_encoder_connections is True:
            for full_connection in range(2, self.levels):
                tensors[f"1---{full_connection}_full-scale"] = modules.full_scale_skip_connection(
                    tensors["En1"],
                    level1=1,
                    level2=full_connection,
                    name=f"1---{full_connection}_full-scale",
                    **full_scale_kwargs,
                )

        for encoder in np.arange(2, self.levels):
            pool_tensor = modules.max_pool(
                tensors[f"En{encoder - 1}"],
                name=f"En{encoder - 1}-En{encoder}",
                **pool_kwargs,
            )
            tensors[f"En{encoder}"] = modules.convolution(
                pool_tensor,
                filters=self.filter_num[encoder - 1],
                name=f"En{encoder}",
                **module_kwargs,
            )
            tensors[f"{encoder}---{encoder}_skip"] = modules.conventional_skip_connection(
                tensors[f"En{encoder}"],
                name=f"{encoder}---{encoder}_skip",
                **conventional_kwargs,
            )
            for full_connection in range(encoder + 1, self.levels):
                tensors[f"{encoder}---{full_connection}_full-scale"] = modules.full_scale_skip_connection(
                    tensors[f"En{encoder}"],
                    level1=encoder,
                    level2=full_connection,
                    name=f"{encoder}---{full_connection}_full-scale",
                    **full_scale_kwargs,
                )

        tensors[f"En{self.levels}"] = modules.max_pool(
            tensors[f"En{self.levels - 1}"],
            name=f"En{self.levels - 1}-En{self.levels}",
            **pool_kwargs,
        )
        tensors[f"En{self.levels}"] = modules.convolution(
            tensors[f"En{self.levels}"],
            filters=self.filter_num[self.levels - 1],
            name=f"En{self.levels}",
            **module_kwargs,
        )
        if self.deep_supervision:
            tensors[f"sup{self.levels}_output"] = modules.deep_supervision_side_output(
                tensors[f"En{self.levels}"],
                num_classes=self.num_classes,
                output_level=self.levels,
                name=f"sup{self.levels}",
                **supervision_kwargs,
            )
            tensors_with_supervision.append(tensors[f"sup{self.levels}_output"])

        for feature_map in range(1, self.levels - 1):
            tensors[f"{self.levels}---{feature_map}_feature"] = modules.aggregated_feature_map(
                tensors[f"En{self.levels}"],
                level1=self.levels,
                level2=feature_map,
                name=f"{self.levels}---{feature_map}_feature",
                **aggregated_kwargs,
            )

        for decoder in np.arange(1, self.levels)[::-1]:
            if decoder == self.levels - 1:
                tensors[f"De{decoder}"] = modules.upsample(
                    tensors[f"En{self.levels}"],
                    name=f"En{self.levels}-De{decoder}",
                    **upsample_kwargs,
                )
                tensors_to_concatenate = [
                    tensors[f"De{decoder}"],
                ]
                connections_to_add = sorted([tensor for tensor in tensors if f"---{decoder}" in tensor])[::-1]
                for connection in connections_to_add:
                    tensors_to_concatenate.append(tensors[connection])
            else:
                tensors[f"De{decoder}"] = modules.upsample(
                    tensors[f"De{decoder + 1}"],
                    name=f"De{decoder + 1}-De{decoder}",
                    **upsample_kwargs,
                )
                tensors_to_concatenate = sorted([tensor for tensor in tensors if f"---{decoder}" in tensor])[::-1]
                for index in range(len(tensors_to_concatenate)):
                    tensors_to_concatenate[index] = tensors[tensors_to_concatenate[index]]
                tensors_to_concatenate.insert(self.levels - 1 - decoder, tensors[f"De{decoder}"])

            tensors[f"De{decoder}"] = Concatenate(name=f"De{decoder}_Concatenate")(tensors_to_concatenate)
            tensors[f"De{decoder}"] = modules.convolution(
                tensors[f"De{decoder}"],
                filters=filter_num_aggregate,
                name=f"De{decoder}",
                **module_kwargs,
            )
            if self.deep_supervision or decoder == 1:
                tensors[f"sup{decoder}_output"] = modules.deep_supervision_side_output(
                    tensors[f"De{decoder}"],
                    num_classes=self.num_classes,
                    output_level=decoder,
                    name=f"sup{decoder}",
                    **supervision_kwargs,
                )
                tensors_with_supervision.append(tensors[f"sup{decoder}_output"])

            for feature_map in range(1, decoder - 1):
                tensors[f"{decoder}---{feature_map}_feature"] = modules.aggregated_feature_map(
                    tensors[f"De{decoder}"],
                    level1=decoder,
                    level2=feature_map,
                    name=f"{decoder}---{feature_map}_feature",
                    **aggregated_kwargs,
                )

        output_model = SharedTargetModel(
            inputs=tensors["input"],
            outputs=tensors_with_supervision[::-1],
            name=f"unet_3plus_{ndims}D",
        )

        return output_model
