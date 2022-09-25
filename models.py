"""
Deep learning models:
    - U-Net
    - U-Net ensemble
    - U-Net+
    - U-Net++
    - U-Net 3+

TODO:
    * Allow models to have a unique number of encoder and decoder levels (e.g. 3 encoder levels and 5 decoder levels)
    * Add temporal U-Nets
    * Create help document with visualizations for in-code documentation

Code written by: Andrew Justin (andrewjustinwx@gmail.com)

Last updated: 9/25/2022 11:47 AM CT
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Input
from utils.unet_utils import *


def unet(input_shape, num_classes, pool_size, upsample_size, levels, filter_num, kernel_size=3, squeeze_dims=None, modules_per_node=5,
    batch_normalization=True, activation='relu', padding='same', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
    kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
    """
    Builds a U-Net model.

    Parameters
    ----------
    input_shape: tuple
        Shape of the inputs. The last number in the tuple represents the number of channels/predictors.
    num_classes: int
        Number of classes/labels that the U-Net will try to predict.
    pool_size: tuple or list
        Size of the mask in the MaxPooling layers.
    upsample_size: tuple or list
        Size of the mask in the UpSampling layers.
    levels: int
        Number of levels in the U-Net. Must be greater than 1.
    filter_num: iterable of ints
        Number of convolution filters on each level of the U-Net.
    kernel_size: int or tuple
        Size of the kernel in the convolution layers.
    squeeze_dims: int, tuple, or None
        Dimensions/axes of the input to squeeze such that the target (y_true) will be smaller than the input.
        - (e.g. to remove the third dimension, set this parameter to 2 [axis=2 for the third dimension])
    modules_per_node: int
        Number of modules in each node of the U-Net.
    batch_normalization: bool
        Setting this to True will add a batch normalization layer after every convolution in the modules.
    activation: str
        Activation function to use in the modules.
        Can be any of tf.keras.activations, 'gaussian', 'gcu', 'leaky_relu', 'prelu', 'smelu', 'snake' (case-insensitive).
    padding: str
        Padding to use in the convolution layers.
    use_bias: bool
        Setting this to True will implement a bias vector in the convolution layers used in the modules.
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

    Returns
    -------
    model: tf.keras.models.Model object
        U-Net model.

    Raises
    ------
    ValueError
        If levels < 2
        If input_shape does not have 3 nor 4 dimensions
        If the length of filter_num does not match the number of levels

    References
    ----------
    https://arxiv.org/pdf/1505.04597.pdf
    """

    ndims = len(input_shape) - 1  # Number of dimensions in the input image (excluding the last dimension reserved for channels)

    if levels < 2:
        raise ValueError(f"levels must be greater than 1. Received value: {levels}")

    if len(input_shape) > 4 or len(input_shape) < 3:
        raise ValueError(f"input_shape can only have 3 or 4 dimensions (2D image + 1 dimension for channels OR a 3D image + 1 dimension for channels). Received shape: {np.shape(input_shape)}")

    if len(filter_num) != levels:
        raise ValueError(f"length of filter_num ({len(filter_num)}) does not match the number of levels ({levels})")

    # Keyword arguments for the convolution modules
    module_kwargs = dict({})
    module_kwargs['activation'] = activation
    module_kwargs['batch_normalization'] = batch_normalization
    module_kwargs['num_modules'] = modules_per_node
    module_kwargs['padding'] = padding
    module_kwargs['use_bias'] = use_bias
    module_kwargs['kernel_initializer'] = kernel_initializer
    module_kwargs['bias_initializer'] = bias_initializer
    module_kwargs['kernel_regularizer'] = kernel_regularizer
    module_kwargs['bias_regularizer'] = bias_regularizer
    module_kwargs['activity_regularizer'] = activity_regularizer
    module_kwargs['kernel_constraint'] = kernel_constraint
    module_kwargs['bias_constraint'] = bias_constraint

    # MaxPooling keyword arguments
    pool_kwargs = dict({})
    pool_kwargs['pool_size'] = pool_size

    # Keyword arguments for upsampling
    upsample_kwargs = dict({})
    upsample_kwargs['activation'] = activation
    upsample_kwargs['batch_normalization'] = batch_normalization
    upsample_kwargs['padding'] = padding
    upsample_kwargs['kernel_initializer'] = kernel_initializer
    upsample_kwargs['bias_initializer'] = bias_initializer
    upsample_kwargs['kernel_regularizer'] = kernel_regularizer
    upsample_kwargs['bias_regularizer'] = bias_regularizer
    upsample_kwargs['activity_regularizer'] = activity_regularizer
    upsample_kwargs['kernel_constraint'] = kernel_constraint
    upsample_kwargs['bias_constraint'] = bias_constraint
    upsample_kwargs['upsample_size'] = upsample_size
    upsample_kwargs['use_bias'] = use_bias

    # Keyword arguments for the deep supervision output in the final decoder node
    supervision_kwargs = dict({})
    supervision_kwargs['upsample_size'] = upsample_size
    supervision_kwargs['use_bias'] = True
    supervision_kwargs['squeeze_dims'] = squeeze_dims
    supervision_kwargs['padding'] = padding
    supervision_kwargs['kernel_initializer'] = kernel_initializer
    supervision_kwargs['bias_initializer'] = bias_initializer
    supervision_kwargs['kernel_regularizer'] = kernel_regularizer
    supervision_kwargs['bias_regularizer'] = bias_regularizer
    supervision_kwargs['activity_regularizer'] = activity_regularizer
    supervision_kwargs['kernel_constraint'] = kernel_constraint
    supervision_kwargs['bias_constraint'] = bias_constraint

    tensors = dict({})  # Tensors associated with each node and skip connections

    """ Setup the first encoder node with an input layer and a convolution module """
    tensors['input'] = Input(shape=input_shape, name='Input')
    tensors['En1'] = convolution_module(tensors['input'], filters=filter_num[0], kernel_size=kernel_size, name='En1', **module_kwargs)

    """ The rest of the encoder nodes are handled here. Each encoder node is connected with a MaxPooling layer and contains convolution modules """
    for encoder in np.arange(2, levels+1):  # Iterate through the rest of the encoder nodes
        pool_tensor = max_pool(tensors[f'En{encoder - 1}'], name=f'En{encoder - 1}-En{encoder}', **pool_kwargs)  # Connect the next encoder node with a MaxPooling layer
        tensors[f'En{encoder}'] = convolution_module(pool_tensor, filters=filter_num[encoder - 1], kernel_size=kernel_size, name=f'En{encoder}', **module_kwargs)  # Convolution modules

    # Connect the bottom encoder node to a decoder node
    upsample_tensor = upsample(tensors[f'En{levels}'], filters=filter_num[levels - 2], kernel_size=kernel_size, name=f'En{levels}-De{levels}', **upsample_kwargs)

    """ Bottom decoder node """
    tensors[f'De{levels - 1}'] = Concatenate(name=f'De{levels - 1}_Concatenate')([tensors[f'En{levels - 1}'], upsample_tensor])  # Concatenate the upsampled tensor and skip connection
    tensors[f'De{levels - 1}'] = convolution_module(tensors[f'De{levels - 1}'], filters=filter_num[levels - 2], kernel_size=kernel_size, name=f'De{levels - 1}', **module_kwargs)  # Convolution module
    upsample_tensor = upsample(tensors[f'De{levels - 1}'], filters=filter_num[levels - 3], kernel_size=kernel_size, name=f'De{levels - 1}-De{levels - 2}', **upsample_kwargs)  # Connect the bottom decoder node to the next decoder node

    """ The rest of the decoder nodes (except the final node) are handled in this loop. Each node contains one concatenation of an upsampled tensor and a skip connection """
    for decoder in np.arange(2, levels-1)[::-1]:
        tensors[f'De{decoder}'] = Concatenate(name=f'De{decoder}_Concatenate')([tensors[f'En{decoder}'], upsample_tensor])  # Concatenate the upsampled tensor and skip connection
        tensors[f'De{decoder}'] = convolution_module(tensors[f'De{decoder}'], filters=filter_num[decoder - 1], kernel_size=kernel_size, name=f'De{decoder}', **module_kwargs)  # Convolution module
        upsample_tensor = upsample(tensors[f'De{decoder}'], filters=filter_num[decoder - 2], kernel_size=kernel_size, name=f'De{decoder}-De{decoder - 1}', **upsample_kwargs)  # Connect the bottom decoder node to the next decoder node

    """ Final decoder node begins with a concatenation and convolution module, followed by deep supervision """
    tensor_De1 = Concatenate(name='De1_Concatenate')([tensors['En1'], upsample_tensor])  # Concatenate the upsampled tensor and skip connection
    tensor_De1 = convolution_module(tensor_De1, filters=filter_num[0], kernel_size=kernel_size, name='De1', **module_kwargs)  # Convolution module
    tensors['output'] = deep_supervision_side_output(tensor_De1, num_classes=num_classes, kernel_size=1, output_level=1, name='final', **supervision_kwargs)  # Deep supervision - this layer will output the model's prediction

    model = Model(inputs=tensors['input'], outputs=tensors['output'], name=f'unet_{ndims}D')

    return model


def unet_ensemble(input_shape, num_classes, pool_size, upsample_size, levels, filter_num, kernel_size=3, squeeze_dims=None, modules_per_node=5,
    batch_normalization=True, activation='relu', padding='same', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
    kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
    """
    Builds a U-Net ensemble model.
    https://arxiv.org/pdf/1912.05074.pdf

    Parameters
    ----------
    input_shape: tuple
        Shape of the inputs. The last number in the tuple represents the number of channels/predictors.
    num_classes: int
        Number of classes/labels that the U-Net will try to predict.
    pool_size: tuple or list
        Size of the mask in the MaxPooling layers.
    upsample_size: tuple or list
        Size of the mask in the UpSampling layers.
    levels: int
        Number of levels in the U-Net. Must be greater than 1.
    filter_num: iterable of ints
        Number of convolution filters on each level of the U-Net.
    kernel_size: int or tuple
        Size of the kernel in the convolution layers.
    squeeze_dims: int, tuple, or None
        Dimensions/axes of the input to squeeze such that the target (y_true) will be smaller than the input.
        - (e.g. to remove the third dimension, set this parameter to 2 [axis=2 for the third dimension])
    modules_per_node: int
        Number of modules in each node of the U-Net.
    batch_normalization: bool
        Setting this to True will add a batch normalization layer after every convolution in the modules.
    activation: str
        Activation function to use in the modules.
        Can be any of tf.keras.activations, 'gaussian', 'gcu', 'leaky_relu', 'prelu', 'smelu', 'snake' (case-insensitive).
    padding: str
        Padding to use in the convolution layers.
    use_bias: bool
        Setting this to True will implement a bias vector in the convolution layers used in the modules.
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

    Returns
    -------
    model: tf.keras.models.Model object
        U-Net model.

    Raises
    ------
    ValueError
        If levels < 2
        If input_shape does not have 3 nor 4 dimensions
        If the length of filter_num does not match the number of levels
    """

    ndims = len(input_shape) - 1  # Number of dimensions in the input image (excluding the last dimension reserved for channels)

    if levels < 2:
        raise ValueError(f"levels must be greater than 1. Received value: {levels}")

    if len(input_shape) > 4 or len(input_shape) < 3:
        raise ValueError(f"input_shape can only have 3 or 4 dimensions (2D image + 1 dimension for channels OR a 3D image + 1 dimension for channels). Received shape: {np.shape(input_shape)}")

    if len(filter_num) != levels:
        raise ValueError(f"length of filter_num ({len(filter_num)}) does not match the number of levels ({levels})")

    # Keyword arguments for the convolution modules
    module_kwargs = dict({})
    module_kwargs['activation'] = activation
    module_kwargs['batch_normalization'] = batch_normalization
    module_kwargs['num_modules'] = modules_per_node
    module_kwargs['padding'] = padding
    module_kwargs['use_bias'] = use_bias
    module_kwargs['kernel_initializer'] = kernel_initializer
    module_kwargs['bias_initializer'] = bias_initializer
    module_kwargs['kernel_regularizer'] = kernel_regularizer
    module_kwargs['bias_regularizer'] = bias_regularizer
    module_kwargs['activity_regularizer'] = activity_regularizer
    module_kwargs['kernel_constraint'] = kernel_constraint
    module_kwargs['bias_constraint'] = bias_constraint

    # MaxPooling keyword arguments
    pool_kwargs = dict({})
    pool_kwargs['pool_size'] = pool_size

    # Keyword arguments for upsampling
    upsample_kwargs = dict({})
    upsample_kwargs['activation'] = activation
    upsample_kwargs['batch_normalization'] = batch_normalization
    upsample_kwargs['padding'] = padding
    upsample_kwargs['kernel_initializer'] = kernel_initializer
    upsample_kwargs['bias_initializer'] = bias_initializer
    upsample_kwargs['kernel_regularizer'] = kernel_regularizer
    upsample_kwargs['bias_regularizer'] = bias_regularizer
    upsample_kwargs['activity_regularizer'] = activity_regularizer
    upsample_kwargs['kernel_constraint'] = kernel_constraint
    upsample_kwargs['bias_constraint'] = bias_constraint
    upsample_kwargs['upsample_size'] = upsample_size
    upsample_kwargs['use_bias'] = use_bias

    # Keyword arguments for the deep supervision output in the final decoder node
    supervision_kwargs = dict({})
    supervision_kwargs['upsample_size'] = upsample_size
    supervision_kwargs['use_bias'] = True
    supervision_kwargs['squeeze_dims'] = squeeze_dims
    supervision_kwargs['padding'] = padding
    supervision_kwargs['kernel_initializer'] = kernel_initializer
    supervision_kwargs['bias_initializer'] = bias_initializer
    supervision_kwargs['kernel_regularizer'] = kernel_regularizer
    supervision_kwargs['bias_regularizer'] = bias_regularizer
    supervision_kwargs['activity_regularizer'] = activity_regularizer
    supervision_kwargs['kernel_constraint'] = kernel_constraint
    supervision_kwargs['bias_constraint'] = bias_constraint

    tensors = dict({})  # Tensors associated with each node and skip connections
    tensors_with_supervision = []  # list of output tensors. If deep supervision is used, more than one output will be produced

    """ Setup the first encoder node with an input layer and a convolution module """
    tensors['input'] = Input(shape=input_shape, name='Input')
    tensors['En1'] = convolution_module(tensors['input'], filters=filter_num[0], kernel_size=kernel_size, name='En1', **module_kwargs)

    """ The rest of the encoder nodes are handled here. Each encoder node is connected with a MaxPooling layer and contains convolution modules """
    for encoder in np.arange(2, levels+1):  # Iterate through the rest of the encoder nodes
        pool_tensor = max_pool(tensors[f'En{encoder - 1}'], name=f'En{encoder - 1}-En{encoder}', **pool_kwargs)  # Connect the next encoder node with a MaxPooling layer
        tensors[f'En{encoder}'] = convolution_module(pool_tensor, filters=filter_num[encoder - 1], kernel_size=kernel_size, name=f'En{encoder}', **module_kwargs)  # Convolution modules

    # Connect the bottom encoder node to a decoder node
    upsample_tensor = upsample(tensors[f'En{levels}'], filters=filter_num[levels - 2], kernel_size=kernel_size, name=f'En{levels}-De{levels}', **upsample_kwargs)

    """ Bottom decoder node """
    tensors[f'De{levels - 1}'] = Concatenate(name=f'De{levels - 1}_Concatenate')([upsample_tensor, tensors[f'En{levels - 1}']])  # Concatenate the upsampled tensor and skip connection
    tensors[f'De{levels - 1}'] = convolution_module(tensors[f'De{levels - 1}'], filters=filter_num[levels - 2], kernel_size=kernel_size, name=f'De{levels - 1}', **module_kwargs)  # Convolution module
    upsample_tensor = upsample(tensors[f'De{levels - 1}'], filters=filter_num[levels - 3], kernel_size=kernel_size, name=f'De{levels - 1}-De{levels - 2}', **upsample_kwargs)  # Connect the bottom decoder node to the next decoder node

    """ The rest of the decoder nodes (except the final node) are handled in this loop. Each node contains one concatenation of an upsampled tensor and a skip connection """
    for decoder in np.arange(1, levels-1)[::-1]:
        num_middle_nodes = levels - decoder - 1
        for node in range(1, num_middle_nodes + 1):
            if node == 1:  # if on the first middle node at the given level
                upsample_tensor_for_middle_node = upsample(tensors[f'En{decoder + 1}'], filters=filter_num[decoder - 2], kernel_size=kernel_size, name=f'En{decoder + 1}-Me{decoder}-1', **upsample_kwargs)
            else:
                upsample_tensor_for_middle_node = upsample(tensors[f'Me{decoder + 1}-{node - 1}'], filters=filter_num[decoder - 2], kernel_size=kernel_size, name=f'Me{decoder + 1}-{node - 1}-Me{decoder}-{node}', **upsample_kwargs)
            tensors[f'Me{decoder}-{node}'] = Concatenate(name=f'Me{decoder}-{node}_Concatenate')([tensors[f'En{decoder}'], upsample_tensor_for_middle_node])
            tensors[f'Me{decoder}-{node}'] = convolution_module(tensors[f'Me{decoder}-{node}'], filters=filter_num[decoder - 1], kernel_size=kernel_size, name=f'Me{decoder}-{node}', **module_kwargs)  # Convolution module
            if decoder == 1:
                tensors[f'sup{decoder}-{node}'] = deep_supervision_side_output(tensors[f'Me{decoder}-{node}'], num_classes=num_classes, output_level=1, kernel_size=1, name=f'sup{decoder}-{node}', **supervision_kwargs)  # deep supervision on middle node located on top level
                tensors_with_supervision.append(tensors[f'sup{decoder}-{node}'])
        tensors[f'De{decoder}'] = Concatenate(name=f'De{decoder}_Concatenate')([tensors[f'En{decoder}'], upsample_tensor])  # Concatenate the upsampled tensor and skip connection
        tensors[f'De{decoder}'] = convolution_module(tensors[f'De{decoder}'], filters=filter_num[decoder - 1], kernel_size=kernel_size, name=f'De{decoder}', **module_kwargs)  # Convolution module

        if decoder != 1:  # if not currently on the final decoder node (De1)
            upsample_tensor = upsample(tensors[f'De{decoder}'], filters=filter_num[decoder - 2], kernel_size=kernel_size, name=f'De{decoder}-De{decoder - 1}', **upsample_kwargs)  # Connect the bottom decoder node to the next decoder node
        else:
            tensors['output'] = deep_supervision_side_output(tensors['De1'], num_classes=num_classes, kernel_size=1, output_level=1, name='final', **supervision_kwargs)  # Deep supervision - this layer will output the model's prediction
            tensors_with_supervision.append(tensors['output'])

    model = Model(inputs=tensors['input'], outputs=tensors_with_supervision, name=f'unet_ensemble_{ndims}D')

    return model


def unet_plus(input_shape, num_classes, pool_size, upsample_size, levels, filter_num, kernel_size=3, squeeze_dims=None,
    modules_per_node=5, batch_normalization=True, deep_supervision=True, activation='relu', padding='same', use_bias=True,
    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None):
    """
    Builds a U-Net+ model.
    https://arxiv.org/pdf/1912.05074.pdf

    Parameters
    ----------
    input_shape: tuple
        Shape of the inputs. The last number in the tuple represents the number of channels/predictors.
    num_classes: int
        Number of classes/labels that the U-Net will try to predict.
    pool_size: tuple or list
        Size of the mask in the MaxPooling layers.
    upsample_size: tuple or list
        Size of the mask in the UpSampling layers.
    levels: int
        Number of levels in the U-Net. Must be greater than 1.
    filter_num: iterable of ints
        Number of convolution filters on each level of the U-Net.
    kernel_size: int or tuple
        Size of the kernel in the convolution layers.
    squeeze_dims: int, tuple, or None
        Dimensions/axes of the input to squeeze such that the target (y_true) will be smaller than the input.
        - (e.g. to remove the third dimension, set this parameter to 2 [axis=2 for the third dimension])
    modules_per_node: int
        Number of modules in each node of the U-Net.
    batch_normalization: bool
        Setting this to True will add a batch normalization layer after every convolution in the modules.
    deep_supervision: bool
        Add deep supervision side outputs to each top node.
        NOTE: The final decoder node requires deep supervision and is not affected if this parameter is False.
    activation: str
        Activation function to use in the modules.
        Can be any of tf.keras.activations, 'gaussian', 'gcu', 'leaky_relu', 'prelu', 'smelu', 'snake' (case-insensitive).
    padding: str
        Padding to use in the convolution layers.
    use_bias: bool
        Setting this to True will implement a bias vector in the convolution layers used in the modules.
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

    Returns
    -------
    model: tf.keras.models.Model object
        U-Net model.

    Raises
    ------
    ValueError
        If levels < 2
        If input_shape does not have 3 nor 4 dimensions
        If the length of filter_num does not match the number of levels
    """

    ndims = len(input_shape) - 1  # Number of dimensions in the input image (excluding the last dimension reserved for channels)

    if levels < 2:
        raise ValueError(f"levels must be greater than 1. Received value: {levels}")

    if len(input_shape) > 4 or len(input_shape) < 3:
        raise ValueError(f"input_shape can only have 3 or 4 dimensions (2D image + 1 dimension for channels OR a 3D image + 1 dimension for channels). Received shape: {np.shape(input_shape)}")

    if len(filter_num) != levels:
        raise ValueError(f"length of filter_num ({len(filter_num)}) does not match the number of levels ({levels})")

    # Keyword arguments for the convolution modules
    module_kwargs = dict({})
    module_kwargs['activation'] = activation
    module_kwargs['batch_normalization'] = batch_normalization
    module_kwargs['num_modules'] = modules_per_node
    module_kwargs['padding'] = padding
    module_kwargs['use_bias'] = use_bias
    module_kwargs['kernel_initializer'] = kernel_initializer
    module_kwargs['bias_initializer'] = bias_initializer
    module_kwargs['kernel_regularizer'] = kernel_regularizer
    module_kwargs['bias_regularizer'] = bias_regularizer
    module_kwargs['activity_regularizer'] = activity_regularizer
    module_kwargs['kernel_constraint'] = kernel_constraint
    module_kwargs['bias_constraint'] = bias_constraint

    # MaxPooling keyword arguments
    pool_kwargs = dict({})
    pool_kwargs['pool_size'] = pool_size

    # Keyword arguments for upsampling
    upsample_kwargs = dict({})
    upsample_kwargs['activation'] = activation
    upsample_kwargs['batch_normalization'] = batch_normalization
    upsample_kwargs['padding'] = padding
    upsample_kwargs['kernel_initializer'] = kernel_initializer
    upsample_kwargs['bias_initializer'] = bias_initializer
    upsample_kwargs['kernel_regularizer'] = kernel_regularizer
    upsample_kwargs['bias_regularizer'] = bias_regularizer
    upsample_kwargs['activity_regularizer'] = activity_regularizer
    upsample_kwargs['kernel_constraint'] = kernel_constraint
    upsample_kwargs['bias_constraint'] = bias_constraint
    upsample_kwargs['upsample_size'] = upsample_size
    upsample_kwargs['use_bias'] = use_bias

    # Keyword arguments for the deep supervision output in the final decoder node
    supervision_kwargs = dict({})
    supervision_kwargs['upsample_size'] = upsample_size
    supervision_kwargs['use_bias'] = True
    supervision_kwargs['squeeze_dims'] = squeeze_dims
    supervision_kwargs['padding'] = padding
    supervision_kwargs['kernel_initializer'] = kernel_initializer
    supervision_kwargs['bias_initializer'] = bias_initializer
    supervision_kwargs['kernel_regularizer'] = kernel_regularizer
    supervision_kwargs['bias_regularizer'] = bias_regularizer
    supervision_kwargs['activity_regularizer'] = activity_regularizer
    supervision_kwargs['kernel_constraint'] = kernel_constraint
    supervision_kwargs['bias_constraint'] = bias_constraint

    tensors = dict({})  # Tensors associated with each node and skip connections
    tensors_with_supervision = []  # list of output tensors. If deep supervision is used, more than one output will be produced

    """ Setup the first encoder node with an input layer and a convolution module """
    tensors['input'] = Input(shape=input_shape, name='Input')
    tensors['En1'] = convolution_module(tensors['input'], filters=filter_num[0], kernel_size=kernel_size, name='En1', **module_kwargs)

    """ The rest of the encoder nodes are handled here. Each encoder node is connected with a MaxPooling layer and contains convolution modules """
    for encoder in np.arange(2, levels+1):  # Iterate through the rest of the encoder nodes
        pool_tensor = max_pool(tensors[f'En{encoder - 1}'], name=f'En{encoder - 1}-En{encoder}', **pool_kwargs)  # Connect the next encoder node with a MaxPooling layer
        tensors[f'En{encoder}'] = convolution_module(pool_tensor, filters=filter_num[encoder - 1], kernel_size=kernel_size, name=f'En{encoder}', **module_kwargs)  # Convolution modules

    # Connect the bottom encoder node to a decoder node
    upsample_tensor = upsample(tensors[f'En{levels}'], filters=filter_num[levels - 2], kernel_size=kernel_size, name=f'En{levels}-De{levels}', **upsample_kwargs)

    """ Bottom decoder node """
    tensors[f'De{levels - 1}'] = Concatenate(name=f'De{levels - 1}_Concatenate')([upsample_tensor, tensors[f'En{levels - 1}']])  # Concatenate the upsampled tensor and skip connection
    tensors[f'De{levels - 1}'] = convolution_module(tensors[f'De{levels - 1}'], filters=filter_num[levels - 2], kernel_size=kernel_size, name=f'De{levels - 1}', **module_kwargs)  # Convolution module
    upsample_tensor = upsample(tensors[f'De{levels - 1}'], filters=filter_num[levels - 3], kernel_size=kernel_size, name=f'De{levels - 1}-De{levels - 2}', **upsample_kwargs)  # Connect the bottom decoder node to the next decoder node

    """ The rest of the decoder nodes (except the final node) are handled in this loop. Each node contains one concatenation of an upsampled tensor and a skip connection """
    for decoder in np.arange(1, levels-1)[::-1]:
        num_middle_nodes = levels - decoder - 1
        for node in range(1, num_middle_nodes + 1):
            if node == 1:  # if on the first middle node at the given level
                upsample_tensor_for_middle_node = upsample(tensors[f'En{decoder + 1}'], filters=filter_num[decoder - 2], kernel_size=kernel_size, name=f'En{decoder + 1}-Me{decoder}-1', **upsample_kwargs)
                tensors[f'Me{decoder}-1'] = Concatenate(name=f'Me{decoder}-1_Concatenate')([tensors[f'En{decoder}'], upsample_tensor_for_middle_node])
            else:
                upsample_tensor_for_middle_node = upsample(tensors[f'Me{decoder + 1}-{node - 1}'], filters=filter_num[decoder - 2], kernel_size=kernel_size, name=f'Me{decoder + 1}-{node - 1}-Me{decoder}-{node}', **upsample_kwargs)
                tensors[f'Me{decoder}-{node}'] = Concatenate(name=f'Me{decoder}-{node}_Concatenate')([tensors[f'Me{decoder}-{node - 1}'], upsample_tensor_for_middle_node])
            tensors[f'Me{decoder}-{node}'] = convolution_module(tensors[f'Me{decoder}-{node}'], filters=filter_num[decoder - 1], kernel_size=kernel_size, name=f'Me{decoder}-{node}', **module_kwargs)  # Convolution module
            if decoder == 1 and deep_supervision:
                tensors[f'sup{decoder}-{node}'] = deep_supervision_side_output(tensors[f'Me{decoder}-{node}'], num_classes=num_classes, output_level=1, kernel_size=1, name=f'sup{decoder}-{node}', **supervision_kwargs)  # deep supervision on middle node located on top level
                tensors_with_supervision.append(tensors[f'sup{decoder}-{node}'])
        tensors[f'De{decoder}'] = Concatenate(name=f'De{decoder}_Concatenate')([tensors[f'Me{decoder}-{num_middle_nodes}'], upsample_tensor])  # Concatenate the upsampled tensor and skip connection
        tensors[f'De{decoder}'] = convolution_module(tensors[f'De{decoder}'], filters=filter_num[decoder - 1], kernel_size=kernel_size, name=f'De{decoder}', **module_kwargs)  # Convolution module

        if decoder != 1:  # if not currently on the final decoder node (De1)
            upsample_tensor = upsample(tensors[f'De{decoder}'], filters=filter_num[decoder - 2], kernel_size=kernel_size, name=f'De{decoder}-De{decoder - 1}', **upsample_kwargs)  # Connect the bottom decoder node to the next decoder node
        else:
            tensors['output'] = deep_supervision_side_output(tensors['De1'], num_classes=num_classes, kernel_size=1, output_level=1, name='final', **supervision_kwargs)  # Deep supervision - this layer will output the model's prediction
            tensors_with_supervision.append(tensors['output'])

    model = Model(inputs=tensors['input'], outputs=tensors_with_supervision, name=f'unet_plus_{ndims}D')

    return model


def unet_2plus(input_shape, num_classes, pool_size, upsample_size, levels, filter_num, kernel_size=3, squeeze_dims=None,
    modules_per_node=5, batch_normalization=True, deep_supervision=True, activation='relu', padding='same', use_bias=True,
    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None):
    """
    Builds a U-Net++ model.
    https://arxiv.org/pdf/1912.05074.pdf

    Parameters
    ----------
    input_shape: tuple
        Shape of the inputs. The last number in the tuple represents the number of channels/predictors.
    num_classes: int
        Number of classes/labels that the U-Net will try to predict.
    pool_size: tuple or list
        Size of the mask in the MaxPooling layers.
    upsample_size: tuple or list
        Size of the mask in the UpSampling layers.
    levels: int
        Number of levels in the U-Net. Must be greater than 1.
    filter_num: iterable of ints
        Number of convolution filters on each level of the U-Net.
    kernel_size: int or tuple
        Size of the kernel in the convolution layers.
    squeeze_dims: int, tuple, or None
        Dimensions/axes of the input to squeeze such that the target (y_true) will be smaller than the input.
        - (e.g. to remove the third dimension, set this parameter to 2 [axis=2 for the third dimension])
    modules_per_node: int
        Number of modules in each node of the U-Net.
    batch_normalization: bool
        Setting this to True will add a batch normalization layer after every convolution in the modules.
    deep_supervision: bool
        Add deep supervision side outputs to each top node.
        NOTE: The final decoder node requires deep supervision and is not affected if this parameter is False.
    activation: str
        Activation function to use in the modules.
        Can be any of tf.keras.activations, 'gaussian', 'gcu', 'leaky_relu', 'prelu', 'smelu', 'snake' (case-insensitive).
    padding: str
        Padding to use in the convolution layers.
    use_bias: bool
        Setting this to True will implement a bias vector in the convolution layers used in the modules.
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

    Returns
    -------
    model: tf.keras.models.Model object
        U-Net model.

    Raises
    ------
    ValueError
        If levels < 2
        If input_shape does not have 3 nor 4 dimensions
        If the length of filter_num does not match the number of levels
    """

    ndims = len(input_shape) - 1  # Number of dimensions in the input image (excluding the last dimension reserved for channels)

    if levels < 2:
        raise ValueError(f"levels must be greater than 1. Received value: {levels}")

    if len(input_shape) > 4 or len(input_shape) < 3:
        raise ValueError(f"input_shape can only have 3 or 4 dimensions (2D image + 1 dimension for channels OR a 3D image + 1 dimension for channels). Received shape: {np.shape(input_shape)}")

    if len(filter_num) != levels:
        raise ValueError(f"length of filter_num ({len(filter_num)}) does not match the number of levels ({levels})")

    # Keyword arguments for the convolution modules
    module_kwargs = dict({})
    module_kwargs['activation'] = activation
    module_kwargs['batch_normalization'] = batch_normalization
    module_kwargs['num_modules'] = modules_per_node
    module_kwargs['padding'] = padding
    module_kwargs['use_bias'] = use_bias
    module_kwargs['kernel_initializer'] = kernel_initializer
    module_kwargs['bias_initializer'] = bias_initializer
    module_kwargs['kernel_regularizer'] = kernel_regularizer
    module_kwargs['bias_regularizer'] = bias_regularizer
    module_kwargs['activity_regularizer'] = activity_regularizer
    module_kwargs['kernel_constraint'] = kernel_constraint
    module_kwargs['bias_constraint'] = bias_constraint

    # MaxPooling keyword arguments
    pool_kwargs = dict({})
    pool_kwargs['pool_size'] = pool_size

    # Keyword arguments for upsampling
    upsample_kwargs = dict({})
    upsample_kwargs['activation'] = activation
    upsample_kwargs['batch_normalization'] = batch_normalization
    upsample_kwargs['padding'] = padding
    upsample_kwargs['kernel_initializer'] = kernel_initializer
    upsample_kwargs['bias_initializer'] = bias_initializer
    upsample_kwargs['kernel_regularizer'] = kernel_regularizer
    upsample_kwargs['bias_regularizer'] = bias_regularizer
    upsample_kwargs['activity_regularizer'] = activity_regularizer
    upsample_kwargs['kernel_constraint'] = kernel_constraint
    upsample_kwargs['bias_constraint'] = bias_constraint
    upsample_kwargs['upsample_size'] = upsample_size
    upsample_kwargs['use_bias'] = use_bias

    # Keyword arguments for the deep supervision output in the final decoder node
    supervision_kwargs = dict({})
    supervision_kwargs['upsample_size'] = upsample_size
    supervision_kwargs['use_bias'] = True
    supervision_kwargs['squeeze_dims'] = squeeze_dims
    supervision_kwargs['padding'] = padding
    supervision_kwargs['kernel_initializer'] = kernel_initializer
    supervision_kwargs['bias_initializer'] = bias_initializer
    supervision_kwargs['kernel_regularizer'] = kernel_regularizer
    supervision_kwargs['bias_regularizer'] = bias_regularizer
    supervision_kwargs['activity_regularizer'] = activity_regularizer
    supervision_kwargs['kernel_constraint'] = kernel_constraint
    supervision_kwargs['bias_constraint'] = bias_constraint

    tensors = dict({})  # Tensors associated with each node and skip connections
    tensors_with_supervision = []  # list of output tensors. If deep supervision is used, more than one output will be produced

    """ Setup the first encoder node with an input layer and a convolution module """
    tensors['input'] = Input(shape=input_shape, name='Input')
    tensors['En1'] = convolution_module(tensors['input'], filters=filter_num[0], kernel_size=kernel_size, name='En1', **module_kwargs)

    """ The rest of the encoder nodes are handled here. Each encoder node is connected with a MaxPooling layer and contains convolution modules """
    for encoder in np.arange(2, levels+1):  # Iterate through the rest of the encoder nodes
        pool_tensor = max_pool(tensors[f'En{encoder - 1}'], name=f'En{encoder - 1}-En{encoder}', **pool_kwargs)  # Connect the next encoder node with a MaxPooling layer
        tensors[f'En{encoder}'] = convolution_module(pool_tensor, filters=filter_num[encoder - 1], kernel_size=kernel_size, name=f'En{encoder}', **module_kwargs)  # Convolution modules

    # Connect the bottom encoder node to a decoder node
    upsample_tensor = upsample(tensors[f'En{levels}'], filters=filter_num[levels - 2], kernel_size=kernel_size, name=f'En{levels}-De{levels}', **upsample_kwargs)

    """ Bottom decoder node """
    tensors[f'De{levels - 1}'] = Concatenate(name=f'De{levels - 1}_Concatenate')([upsample_tensor, tensors[f'En{levels - 1}']])  # Concatenate the upsampled tensor and skip connection
    tensors[f'De{levels - 1}'] = convolution_module(tensors[f'De{levels - 1}'], filters=filter_num[levels - 2], kernel_size=kernel_size, name=f'De{levels - 1}', **module_kwargs)  # Convolution module
    upsample_tensor = upsample(tensors[f'De{levels - 1}'], filters=filter_num[levels - 3], kernel_size=kernel_size, name=f'De{levels - 1}-De{levels - 2}', **upsample_kwargs)  # Connect the bottom decoder node to the next decoder node

    """ The rest of the decoder nodes (except the final node) are handled in this loop. Each node contains one concatenation of an upsampled tensor and a skip connection """
    for decoder in np.arange(1, levels-1)[::-1]:
        num_middle_nodes = levels - decoder - 1
        for node in range(1, num_middle_nodes + 1):
            if node == 1:  # if on the first middle node at the given level
                upsample_tensor_for_middle_node = upsample(tensors[f'En{decoder + 1}'], filters=filter_num[decoder - 2], kernel_size=kernel_size, name=f'En{decoder + 1}-Me{decoder}-1', **upsample_kwargs)
                tensors[f'Me{decoder}-1'] = Concatenate(name=f'Me{decoder}-1_Concatenate')([tensors[f'En{decoder}'], upsample_tensor_for_middle_node])
            else:
                upsample_tensor_for_middle_node = upsample(tensors[f'Me{decoder + 1}-{node - 1}'], filters=filter_num[decoder - 2], kernel_size=kernel_size, name=f'Me{decoder + 1}-{node - 1}-Me{decoder}-{node}', **upsample_kwargs)
                tensors_to_concatenate = []  # Tensors to concatenate in the middle node
                connections_to_add = sorted([tensor for tensor in tensors if f'Me{decoder}' in tensor])[::-1]  # skip connections to add to the list of tensors to concatenate
                for connection in connections_to_add:
                    tensors_to_concatenate.append(tensors[connection])
                tensors_to_concatenate.append(tensors[f'En{decoder}'])
                tensors_to_concatenate.append(upsample_tensor_for_middle_node)
                tensors[f'Me{decoder}-{node}'] = Concatenate(name=f'Me{decoder}-{node}_Concatenate')(tensors_to_concatenate)
            tensors[f'Me{decoder}-{node}'] = convolution_module(tensors[f'Me{decoder}-{node}'], filters=filter_num[decoder - 1], kernel_size=kernel_size, name=f'Me{decoder}-{node}', **module_kwargs)  # Convolution module

            if decoder == 1 and deep_supervision:
                tensors[f'sup{decoder}-{node}'] = deep_supervision_side_output(tensors[f'Me{decoder}-{node}'], num_classes=num_classes, output_level=1, kernel_size=1, name=f'sup{decoder}-{node}', **supervision_kwargs)  # deep supervision on middle node located on top level
                tensors_with_supervision.append(tensors[f'sup{decoder}-{node}'])

        tensors_to_concatenate = []  # tensors to concatenate in the decoder node
        connections_to_add = sorted([tensor for tensor in tensors if f'Me{decoder}' in tensor])[::-1]  # skip connections to add to the list of tensors to concatenate
        for connection in connections_to_add:
            tensors_to_concatenate.append(tensors[connection])
        tensors_to_concatenate.append(tensors[f'En{decoder}'])
        tensors_to_concatenate.append(upsample_tensor)
        tensors[f'De{decoder}'] = Concatenate(name=f'De{decoder}_Concatenate')(tensors_to_concatenate)  # Concatenate the upsampled tensor and skip connection
        tensors[f'De{decoder}'] = convolution_module(tensors[f'De{decoder}'], filters=filter_num[decoder - 1], kernel_size=kernel_size, name=f'De{decoder}', **module_kwargs)  # Convolution module

        if decoder != 1:  # if not currently on the final decoder node (De1)
            upsample_tensor = upsample(tensors[f'De{decoder}'], filters=filter_num[decoder - 2], kernel_size=kernel_size, name=f'De{decoder}-De{decoder - 1}', **upsample_kwargs)  # Connect the bottom decoder node to the next decoder node
        else:
            tensors['output'] = deep_supervision_side_output(tensors['De1'], num_classes=num_classes, kernel_size=1, output_level=1, name='final', **supervision_kwargs)  # Deep supervision - this layer will output the model's prediction
            tensors_with_supervision.append(tensors['output'])

    model = Model(inputs=tensors['input'], outputs=tensors_with_supervision, name=f'unet_2plus_{ndims}D')

    return model


def unet_3plus(input_shape, num_classes, pool_size, upsample_size, levels, filter_num, filter_num_skip=None, filter_num_aggregate=None,
    kernel_size=3, squeeze_dims=None, first_encoder_connections=True, modules_per_node=5, batch_normalization=True, deep_supervision=True,
    activation='relu', padding='same', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
    """
    Creates a U-Net 3+.
    https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf

    Parameters
    ----------
    input_shape: tuple
        Shape of the inputs. The last number in the tuple represents the number of channels/predictors.
    num_classes: int
        Number of classes/labels that the U-Net 3+ will try to predict.
    pool_size: tuple or list
        Size of the mask in the MaxPooling layers.
    upsample_size: tuple or list
        Size of the mask in the UpSampling layers.
    levels: int
        Number of levels in the U-Net 3+. Must be greater than 2.
    filter_num: iterable of ints
        Number of convolution filters in each encoder of the U-Net 3+. The length must be equal to 'levels'.
    filter_num_skip: int or None
        Number of convolution filters in the conventional skip connections, full-scale skip connections, and aggregated feature maps.
        NOTE: When left as None, this will default to the first value in the 'filter_num' iterable.
    filter_num_aggregate: int or None
        Number of convolution filters in the decoder nodes after images are concatenated.
        When left as None, this will be equal to the product of filter_num_skip and the number of levels.
    kernel_size: int or tuple
        Size of the kernel in the convolution layers.
    first_encoder_connections: bool
        Setting this to True will create full-scale skip connections attached to the first encoder node.
    squeeze_dims: int, tuple, or None
        Dimensions/axes of the input to squeeze such that the target (y_true) will be smaller than the input.
        - (e.g. to remove the third dimension, set this parameter to 2 [axis=2 for the third dimension])
    modules_per_node: int
        Number of modules in each node of the U-Net 3+.
    batch_normalization: bool
        Setting this to True will add a batch normalization layer after every convolution in the modules.
    deep_supervision: bool
        Add deep supervision side outputs to each decoder node.
        NOTE: The final decoder node requires deep supervision and is not affected if this parameter is False.
    activation: str
        Activation function to use in the modules.
        Can be any of tf.keras.activations, 'gaussian', 'gcu', 'leaky_relu', 'prelu', 'smelu', 'snake' (case-insensitive).
    padding: str
        Padding to use in the convolution layers.
    use_bias: bool
        Setting this to True will implement a bias vector in the convolution layers used in the modules.
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
    pool_size: tuple or list
        Size of the mask in the MaxPooling layers.
    upsample_size: tuple or list
        Size of the mask in the UpSampling layers.

    Returns
    -------
    model: tf.keras.models.Model object
        U-Net 3+ model.
    """

    ndims = len(input_shape) - 1  # Number of dimensions in the input image (excluding the last dimension reserved for channels)

    if levels < 3:
        raise ValueError(f"levels must be greater than 2. Received value: {levels}")

    if len(input_shape) > 4 or len(input_shape) < 3:
        raise ValueError(f"input_shape can only have 3 or 4 dimensions (2D image + 1 dimension for channels OR a 3D image + 1 dimension for channels). Received shape: {np.shape(input_shape)}")

    if len(filter_num) != levels:
        raise ValueError(f"length of filter_num ({len(filter_num)}) does not match the number of levels ({levels})")

    if filter_num_skip is None:
        filter_num_skip = filter_num[0]

    if filter_num_aggregate is None:
        filter_num_aggregate = levels * filter_num_skip

    # print(f"\nCreating model: {ndims}D U-Net 3+")

    module_kwargs = dict({})
    module_kwargs['kernel_size'] = kernel_size
    module_kwargs['activation'] = activation
    module_kwargs['batch_normalization'] = batch_normalization
    module_kwargs['num_modules'] = modules_per_node
    module_kwargs['padding'] = padding
    module_kwargs['use_bias'] = use_bias
    module_kwargs['kernel_initializer'] = kernel_initializer
    module_kwargs['bias_initializer'] = bias_initializer
    module_kwargs['kernel_regularizer'] = kernel_regularizer
    module_kwargs['bias_regularizer'] = bias_regularizer
    module_kwargs['activity_regularizer'] = activity_regularizer
    module_kwargs['kernel_constraint'] = kernel_constraint
    module_kwargs['bias_constraint'] = bias_constraint

    pool_kwargs = dict({})
    pool_kwargs['pool_size'] = pool_size

    upsample_kwargs = dict({})
    upsample_kwargs['activation'] = activation
    upsample_kwargs['batch_normalization'] = batch_normalization
    upsample_kwargs['kernel_size'] = kernel_size
    upsample_kwargs['filters'] = filter_num_skip
    upsample_kwargs['padding'] = padding
    upsample_kwargs['kernel_initializer'] = kernel_initializer
    upsample_kwargs['bias_initializer'] = bias_initializer
    upsample_kwargs['kernel_regularizer'] = kernel_regularizer
    upsample_kwargs['bias_regularizer'] = bias_regularizer
    upsample_kwargs['activity_regularizer'] = activity_regularizer
    upsample_kwargs['kernel_constraint'] = kernel_constraint
    upsample_kwargs['bias_constraint'] = bias_constraint
    upsample_kwargs['upsample_size'] = upsample_size
    upsample_kwargs['use_bias'] = use_bias

    conventional_kwargs = dict({})
    conventional_kwargs['filters'] = filter_num_skip
    conventional_kwargs['kernel_size'] = kernel_size
    conventional_kwargs['activation'] = activation
    conventional_kwargs['batch_normalization'] = batch_normalization
    conventional_kwargs['padding'] = padding
    conventional_kwargs['use_bias'] = use_bias
    conventional_kwargs['kernel_initializer'] = kernel_initializer
    conventional_kwargs['bias_initializer'] = bias_initializer
    conventional_kwargs['kernel_regularizer'] = kernel_regularizer
    conventional_kwargs['bias_regularizer'] = bias_regularizer
    conventional_kwargs['activity_regularizer'] = activity_regularizer
    conventional_kwargs['kernel_constraint'] = kernel_constraint
    conventional_kwargs['bias_constraint'] = bias_constraint

    full_scale_kwargs = dict({})
    full_scale_kwargs['filters'] = filter_num_skip
    full_scale_kwargs['kernel_size'] = kernel_size
    full_scale_kwargs['activation'] = activation
    full_scale_kwargs['batch_normalization'] = batch_normalization
    full_scale_kwargs['use_bias'] = use_bias
    full_scale_kwargs['padding'] = padding
    full_scale_kwargs['pool_size'] = pool_size
    full_scale_kwargs['kernel_initializer'] = kernel_initializer
    full_scale_kwargs['bias_initializer'] = bias_initializer
    full_scale_kwargs['kernel_regularizer'] = kernel_regularizer
    full_scale_kwargs['bias_regularizer'] = bias_regularizer
    full_scale_kwargs['activity_regularizer'] = activity_regularizer
    full_scale_kwargs['kernel_constraint'] = kernel_constraint
    full_scale_kwargs['bias_constraint'] = bias_constraint

    aggregated_kwargs = dict({})
    aggregated_kwargs['filters'] = filter_num_skip
    aggregated_kwargs['kernel_size'] = kernel_size
    aggregated_kwargs['activation'] = activation
    aggregated_kwargs['batch_normalization'] = batch_normalization
    aggregated_kwargs['padding'] = padding
    aggregated_kwargs['upsample_size'] = upsample_size
    aggregated_kwargs['use_bias'] = use_bias
    aggregated_kwargs['kernel_initializer'] = kernel_initializer
    aggregated_kwargs['bias_initializer'] = bias_initializer
    aggregated_kwargs['kernel_regularizer'] = kernel_regularizer
    aggregated_kwargs['bias_regularizer'] = bias_regularizer
    aggregated_kwargs['activity_regularizer'] = activity_regularizer
    aggregated_kwargs['kernel_constraint'] = kernel_constraint
    aggregated_kwargs['bias_constraint'] = bias_constraint

    supervision_kwargs = dict({})
    supervision_kwargs['upsample_size'] = upsample_size
    supervision_kwargs['kernel_size'] = kernel_size
    supervision_kwargs['use_bias'] = True
    supervision_kwargs['padding'] = padding
    supervision_kwargs['squeeze_dims'] = squeeze_dims
    supervision_kwargs['kernel_initializer'] = kernel_initializer
    supervision_kwargs['bias_initializer'] = bias_initializer
    supervision_kwargs['kernel_regularizer'] = kernel_regularizer
    supervision_kwargs['bias_regularizer'] = bias_regularizer
    supervision_kwargs['activity_regularizer'] = activity_regularizer
    supervision_kwargs['kernel_constraint'] = kernel_constraint
    supervision_kwargs['bias_constraint'] = bias_constraint

    tensors = dict({})  # Tensors associated with each node and skip connections
    tensors_with_supervision = []  # Outputs of deep supervision

    """ Setup the first encoder node with an input layer and a convolution module (we are not using skip connections here) """
    tensors['input'] = Input(shape=input_shape, name='Input')
    tensors['En1'] = convolution_module(tensors['input'], filters=filter_num[0], name='En1', **module_kwargs)

    if first_encoder_connections is True:
        for full_connection in range(2, levels):
            tensors[f'1---{full_connection}_full-scale'] = full_scale_skip_connection(tensors[f'En1'], level1=1, level2=full_connection, name=f'1---{full_connection}_full-scale', **full_scale_kwargs)

    """ The rest of the encoder nodes are handled here. Each encoder node is connected with a MaxPooling layer and contains convolution modules """
    for encoder in np.arange(2, levels):  # Iterate through the rest of the encoder nodes
        pool_tensor = max_pool(tensors[f'En{encoder - 1}'], name=f'En{encoder - 1}-En{encoder}', **pool_kwargs)  # Connect the next encoder node with a MaxPooling layer
        tensors[f'En{encoder}'] = convolution_module(pool_tensor, filters=filter_num[encoder - 1], name=f'En{encoder}', **module_kwargs)  # Convolution modules
        tensors[f'{encoder}---{encoder}_skip'] = conventional_skip_connection(tensors[f'En{encoder}'], name=f'{encoder}---{encoder}_skip', **conventional_kwargs)

        # Create full-scale skip connections
        for full_connection in range(encoder + 1, levels):
            tensors[f'{encoder}---{full_connection}_full-scale'] = full_scale_skip_connection(tensors[f'En{encoder}'], level1=encoder, level2=full_connection, name=f'{encoder}---{full_connection}_full-scale', **full_scale_kwargs)

    # Bottom encoder node
    tensors[f'En{levels}'] = max_pool(tensors[f'En{levels - 1}'], name=f'En{levels - 1}-En{levels}', **pool_kwargs)
    tensors[f'En{levels}'] = convolution_module(tensors[f'En{levels}'], filters=filter_num[levels - 1], name=f'En{levels}', **module_kwargs)
    if deep_supervision:
        tensors[f'sup{levels}_output'] = deep_supervision_side_output(tensors[f'En{levels}'], num_classes=num_classes, output_level=levels, name=f'sup{levels}', **supervision_kwargs)
        tensors_with_supervision.append(tensors[f'sup{levels}_output'])

    # Add aggregated feature maps using the bottom encoder node
    for feature_map in range(1, levels - 1):
        tensors[f'{levels}---{feature_map}_feature'] = aggregated_feature_map(tensors[f'En{levels}'], level1=levels, level2=feature_map, name=f'{levels}---{feature_map}_feature', **aggregated_kwargs)

    """ Build the rest of the decoder nodes """
    for decoder in np.arange(1, levels)[::-1]:

        """ The lowest decoder node (levels - 1) is attached to the bottom encoder node via upsampling, so concatenation is slightly different """
        if decoder == levels - 1:
            tensors[f'De{decoder}'] = upsample(tensors[f'En{levels}'], name=f'En{levels}-De{decoder}', **upsample_kwargs)

            # Tensors to concatenate in the Concatenate layer
            tensors_to_concatenate = [tensors[f'De{decoder}'], ]
            connections_to_add = sorted([tensor for tensor in tensors if f'---{decoder}' in tensor])[::-1]
            for connection in connections_to_add:
                tensors_to_concatenate.append(tensors[connection])
        else:
            tensors[f'De{decoder}'] = upsample(tensors[f'De{decoder + 1}'], name=f'De{decoder + 1}-De{decoder}', **upsample_kwargs)

            # Tensors to concatenate in the Concatenate layer
            tensors_to_concatenate = sorted([tensor for tensor in tensors if f'---{decoder}' in tensor])[::-1]
            for index in range(len(tensors_to_concatenate)):
                tensors_to_concatenate[index] = tensors[tensors_to_concatenate[index]]
            tensors_to_concatenate.insert(levels - 1 - decoder, tensors[f'De{decoder}'])

        # Concatenate tensors, pass through convolution modules, then use deep supervision to create a side output
        tensors[f'De{decoder}'] = Concatenate(name=f'De{decoder}_Concatenate')(tensors_to_concatenate)
        tensors[f'De{decoder}'] = convolution_module(tensors[f'De{decoder}'], filters=filter_num_aggregate, name=f'De{decoder}', **module_kwargs)
        if deep_supervision or decoder == 1:  # Decoder node 1 must always have deep supervision
            tensors[f'sup{decoder}_output'] = deep_supervision_side_output(tensors[f'De{decoder}'], num_classes=num_classes, output_level=decoder, name=f'sup{decoder}', **supervision_kwargs)
            tensors_with_supervision.append(tensors[f'sup{decoder}_output'])

        """ Add aggregated feature maps """
        for feature_map in range(1, decoder - 1):
            tensors[f'{decoder}---{feature_map}_feature'] = aggregated_feature_map(tensors[f'De{decoder}'], level1=decoder, level2=feature_map, name=f'{decoder}---{feature_map}_feature', **aggregated_kwargs)

    model = Model(inputs=tensors['input'], outputs=tensors_with_supervision, name=f'unet_3plus_{ndims}D')

    return model
