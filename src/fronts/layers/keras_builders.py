import dataclasses
from typing import Any, ClassVar, Generic, Literal, TypeVar

import tensorflow as tf

from fronts.layers import activations, losses, metrics

T = TypeVar("T")


@dataclasses.dataclass
class BaseConfig(Generic[T]):
    """Base configuration class for building objects from a registry.

    Type parameter T is the type of object this config builds.
    """

    name: str
    config: dict[str, Any]

    # Subclasses must define this
    @property
    def registry(self) -> dict[str, Any]:
        """The registry mapping string names to classes or factory functions.

        Returns:
            A dictionary mapping string names to callables that can be built by this config.
        """
        raise NotImplementedError("Subclasses must define a registry property.")

    def build(self) -> T:
        """Builds the object from the configuration.

        Returns:
            An instance of the registered class.

        Raises:
            ValueError: If the name is not in the registry.
        """
        if self.name not in self.registry:
            raise ValueError(
                f"Unsupported {self.__class__.__name__}: {self.name}. Valid options are: {list(self.registry.keys())}"
            )

        method = self.registry[self.name]
        return method(**(self.config or {}))


@dataclasses.dataclass
class ConstraintConfig(BaseConfig[tf.keras.constraints.Constraint]):
    """Generic constraint configuration for training a model.

    Attributes:
        name: the string name of the constraint to use.
        config: a dictionary of keyword arguments to pass to the constraint constructor.
        registry: a dictionary mapping string names to constraint classes.
    """

    @property
    def registry(self) -> dict[str, Any]:
        """Return mapping from name strings to the corresponding Keras class."""
        return {
            "max_norm": tf.keras.constraints.MaxNorm,
            "min_max_norm": tf.keras.constraints.MinMaxNorm,
            "non_neg": tf.keras.constraints.NonNeg,
            "unit_norm": tf.keras.constraints.UnitNorm,
        }


@dataclasses.dataclass
class InitializerConfig(BaseConfig[tf.keras.initializers.Initializer]):
    """Initializer configuration for training a model.

    Attributes:
        name: the string name of the initializer to use.
        config: a dictionary of keyword arguments to pass to the initializer constructor.
    """

    @property
    def registry(self) -> dict[str, Any]:
        """Return mapping from name strings to the corresponding Keras class."""
        return {
            "glorot_normal": tf.keras.initializers.GlorotNormal,
            "glorot_uniform": tf.keras.initializers.GlorotUniform,
            "he_normal": tf.keras.initializers.HeNormal,
            "he_uniform": tf.keras.initializers.HeUniform,
            "identity": tf.keras.initializers.Identity,
            "lecun_normal": tf.keras.initializers.LecunNormal,
            "lecun_uniform": tf.keras.initializers.LecunUniform,
            "ones": tf.keras.initializers.Ones,
            "orthogonal": tf.keras.initializers.Orthogonal,
            "random_normal": tf.keras.initializers.RandomNormal,
            "random_uniform": tf.keras.initializers.RandomUniform,
            "truncated_normal": tf.keras.initializers.TruncatedNormal,
            "variance_scaling": tf.keras.initializers.VarianceScaling,
            "zeros": tf.keras.initializers.Zeros,
        }


@dataclasses.dataclass
class RegularizerConfig(BaseConfig[tf.keras.regularizers.Regularizer]):
    """Generic regularizer configuration for training a model.

    Attributes:
        name: the string name of the regularizer to use.
        config: a dictionary of keyword arguments to pass to the regularizer constructor.
    """

    @property
    def registry(self) -> dict[str, Any]:
        """Return mapping from name strings to the corresponding Keras class."""
        return {
            "l1": tf.keras.regularizers.L1,
            "l2": tf.keras.regularizers.L2,
            "l1_l2": tf.keras.regularizers.L1L2,
            "orthogonal_regularizer": tf.keras.regularizers.OrthogonalRegularizer,
        }


@dataclasses.dataclass
class OptimizerConfig(BaseConfig[tf.keras.optimizers.Optimizer]):
    """Optimizer configuration for training a model.

    Attributes:
        name: the string name of the optimizer to use.
        config: a dictionary of keyword arguments to pass to the optimizer constructor.
    """

    name: Literal["Adam"]  # pyrefly: ignore[bad-override-mutable-attribute]

    @property
    def registry(self) -> dict[str, Any]:
        """Return mapping from name strings to the corresponding Keras class."""
        return {
            "Adam": tf.keras.optimizers.Adam,
        }


@dataclasses.dataclass
class ActivationConfig(BaseConfig[tf.keras.layers.Activation | tf.keras.layers.Layer]):
    """Activation configuration for layers in a model.

    Attributes:
        name: the string name of the activation to use.
        config: a dictionary of keyword arguments to pass to the activation constructor.
    """

    name: Literal[  # pyrefly: ignore[bad-override-mutable-attribute]
        "elliott",
        "elu",
        "exponential",
        "gaussian",
        "gcu",
        "gelu",
        "hard_sigmoid",
        "hexpo",
        "isigmoid",
        "leaky_relu",
        "linear",
        "lisht",
        "prelu",
        "psigmoid",
        "ptanh",
        "ptelu",
        "relu",
        "resech",
        "selu",
        "sigmoid",
        "smelu",
        "snake",
        "softmax",
        "softplus",
        "softsign",
        "srs",
        "stanh",
        "swish",
        "tanh",
        "thresholded_relu",
    ]

    # Activations that are passed as string names to Keras layers directly.
    # These do not need instantiation — the UNet/convolution accepts
    # the string name via the Keras activation API.
    _BUILTIN_NAMES: ClassVar[frozenset] = frozenset(
        {
            "elu",
            "exponential",
            "gelu",
            "hard_sigmoid",
            "linear",
            "relu",
            "selu",
            "sigmoid",
            "softmax",
            "softplus",
            "softsign",
            "swish",
            "tanh",
        }
    )

    @property
    def registry(self) -> dict[str, Any]:
        """Return mapping from name strings to the corresponding Keras class."""
        return {
            "elliott": activations.Elliott,
            "gaussian": activations.Gaussian,
            "gcu": activations.GCU,
            "hexpo": activations.Hexpo,
            "isigmoid": activations.ISigmoid,
            "lisht": activations.LiSHT,
            "prelu": tf.keras.layers.PReLU,
            "psigmoid": activations.PSigmoid,
            "ptanh": activations.PTanh,
            "ptelu": activations.PTELU,
            "leaky_relu": tf.keras.layers.LeakyReLU,
            "resech": activations.ReSech,
            "smelu": activations.SmeLU,
            "snake": activations.Snake,
            "srs": activations.SRS,
            "stanh": activations.STanh,
            "thresholded_relu": tf.keras.layers.ThresholdedReLU,
        }

    def build(self):
        """Return a Activation layer for built-ins or an instantiated Layer for custom/parametric activations.

        The convolution function calls the result as a callable layer, so a Layer instance is always
        returned rather than a bare string.
        """
        if self.name in self._BUILTIN_NAMES:
            return tf.keras.layers.Activation(self.name)
        return super().build()


@dataclasses.dataclass
class LossConfig(BaseConfig[tf.keras.losses.Loss]):
    """Loss configuration for training a model.

    Attributes:
        name: the string name of the loss function to use.
        config: a dictionary of keyword arguments to pass to the loss constructor.
    """

    name: Literal[  # pyrefly: ignore[bad-override-mutable-attribute]
        "brier_skill_score",
        "critical_success_index",
        "fractions_skill_score",
        "probability_of_detection",
    ]

    @property
    def registry(self) -> dict[str, Any]:
        """Return mapping from name strings to the corresponding Keras class."""
        return {
            "brier_skill_score": losses.brier_skill_score,
            "critical_success_index": losses.critical_success_index,
            "fractions_skill_score": losses.fractions_skill_score,
            "probability_of_detection": losses.probability_of_detection,
        }


@dataclasses.dataclass
class MetricConfig(BaseConfig[tf.keras.metrics.Metric]):
    """Metric configuration for training a model.

    Attributes:
        name: the string name of the metric to use.
        config: a dictionary of keyword arguments to pass to the metric constructor.
    """

    name: Literal[  # pyrefly: ignore[bad-override-mutable-attribute]
        "brier_skill_score",
        "critical_success_index",
        "fractions_skill_score",
        "heidke_skill_score",
        "probability_of_detection",
    ]

    @property
    def registry(self) -> dict[str, Any]:
        """Return mapping from name strings to the corresponding Keras class."""
        return {
            "brier_skill_score": metrics.brier_skill_score,
            "critical_success_index": metrics.critical_success_index,
            "fractions_skill_score": metrics.fractions_skill_score,
            "heidke_skill_score": metrics.heidke_skill_score,
            "probability_of_detection": metrics.probability_of_detection,
        }


@dataclasses.dataclass
class ConvOutputConfig:
    """Convolution output config for training a model.

    Attributes:
        regularizer: a RegularizerConfig to apply to convolutional layer outputs.
    """

    regularizer: RegularizerConfig | None

    def build(self):
        """Builds the convolution output configuration.

        Returns:
            A dictionary of keyword arguments to pass to convolutional layer constructors.
        """
        regularizer_object = (
            self.regularizer.build() if self.regularizer is not None else None
        )
        return ConvOutput(activity_regularizer=regularizer_object)


@dataclasses.dataclass
class BiasVectorConfig:
    """Constraint configuration for bias vectors in a model.

    Attributes:
        constraint: a ConstraintConfig to apply to bias vectors.
        initializer: an InitializerConfig to use for bias vectors.
        regularizer: a RegularizerConfig to apply to bias vectors.
    """

    constraint: ConstraintConfig | None
    initializer: InitializerConfig
    regularizer: RegularizerConfig | None

    def build(self):
        """Builds the bias vector configuration.

        Returns:
            A dictionary of keyword arguments to pass to layer constructors for bias vectors.
        """
        constraint_object = (
            self.constraint.build() if self.constraint is not None else None
        )
        initializer_object = self.initializer.build()
        regularizer_object = (
            self.regularizer.build() if self.regularizer is not None else None
        )

        return BiasVector(
            bias_constraint=constraint_object,
            bias_initializer=initializer_object,
            bias_regularizer=regularizer_object,
        )


@dataclasses.dataclass
class KernelMatrixConfig:
    """Constraint configuration for kernel matrices in a model.

    Attributes:
        constraint: a ConstraintConfig to apply to kernel matrices.
        initializer: an InitializerConfig to use for kernel matrices.
        regularizer: a RegularizerConfig to apply to kernel matrices.
    """

    constraint: ConstraintConfig | None
    initializer: InitializerConfig
    regularizer: RegularizerConfig | None

    def build(self):
        """Builds the kernel matrix configuration.

        Returns:
            A dictionary of keyword arguments to pass to layer constructors for kernel matrices.
        """
        constraint_object = (
            self.constraint.build() if self.constraint is not None else None
        )
        initializer_object = self.initializer.build()
        regularizer_object = (
            self.regularizer.build() if self.regularizer is not None else None
        )

        return KernelMatrix(
            kernel_constraint=constraint_object,
            kernel_initializer=initializer_object,
            kernel_regularizer=regularizer_object,
        )


class ConvOutput:
    """Built Keras objects for the convolution layer output regularizer."""

    def __init__(
        self,
        activity_regularizer: tf.keras.regularizers.Regularizer,
    ):
        self.activity_regularizer = activity_regularizer


class KernelMatrix:
    """Built Keras objects for the convolution kernel (constraint, initializer, regularizer)."""

    def __init__(
        self,
        kernel_constraint: tf.keras.constraints.Constraint | None,
        kernel_initializer: tf.keras.initializers.Initializer,
        kernel_regularizer: tf.keras.regularizers.Regularizer | None,
    ):
        self.kernel_constraint = kernel_constraint
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer


class BiasVector:
    """Built Keras objects for the convolution bias vector (constraint, initializer, regularizer)."""

    def __init__(
        self,
        bias_constraint: tf.keras.constraints.Constraint | None,
        bias_initializer: tf.keras.initializers.Initializer,
        bias_regularizer: tf.keras.regularizers.Regularizer | None,
    ):
        self.bias_constraint = bias_constraint
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
