"""
Models for front detection

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 4/16/2022 6:47 PM CDT

Known bugs:
- none

Please report any bugs to Andrew Justin: andrewjustinwx@gmail.com
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Input
from utils.unet_utils import *


def unet(input_shape, num_classes, kernel_size=3, levels=5, filter_num=(16, 32, 64, 128, 256, 512), modules_per_node=5, batch_normalization=False,
    activation='relu', padding='same', use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, preserve_third_dimension=False,
    pool_size=2, upsample_size=2):
    """
    Builds a U-Net model.

    Parameters
    ----------
    input_shape: tuple
        - Shape of the inputs.
    num_classes: int
        - Number of classes/labels that the U-Net will try to predict.
    kernel_size: int or tuple
        - Size of the kernel in the convolution layers.
    levels: int
        - Number of levels in the U-Net. Must be greater than 1.
    filter_num: iterable of ints
        - Number of convolution filters on each level of the U-Net.
    modules_per_node: int
        - Number of modules in each node of the U-Net.
    batch_normalization: bool
        - Setting this to True will add a batch normalization layer after every convolution in the modules.
    activation: str
        - Activation function to use in the modules.
        - Can be any of tf.keras.activations, 'prelu', 'leaky_relu', or 'smelu' (case-insensitive).
    padding: str
        - Padding to use in the convolution layers.
    use_bias: bool
        - Setting this to True will implement a bias vector in the convolution layers used in the modules.
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
    preserve_third_dimension: bool
        - Setting this to True will preserve the third dimension of the U-Net so that it is not modified by the MaxPooling and UpSampling layers.
    pool_size: int
        - Size of the mask in the MaxPooling layers.
    upsample_size: int
        - Size of the mask in the UpSampling layers.

    Returns
    -------
    model: tf.keras.models.Model object
        - U-Net model.
    """

    ndims = len(input_shape) - 1  # Number of dimensions in the input image (excluding the last dimension reserved for channels)

    if levels < 2:
        raise ValueError(f"levels must be greater than 1. Received value: {levels}")

    if len(input_shape) > 4 or len(input_shape) < 3:
        raise ValueError(f"input_shape can only have 3 or 4 dimensions (2D image + 1 dimension for channels OR a 3D image + 1 dimension for channels). Received shape: {np.shape(input_shape)}")

    if preserve_third_dimension is True and len(input_shape) != 4:
        raise ValueError(f"preserve_third_dimension is True but input_shape does not have 4 dimensions (3D image + 1 dimension for channels). Received shape: {np.shape(input_shape)}")

    if len(filter_num) != levels:
        raise ValueError(f"length of filter_num ({len(filter_num)}) does not match the number of levels ({levels})")

    print(f"\nCreating model: {ndims}D U-Net")

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
    if ndims == 3:
        pool_kwargs['preserve_third_dimension'] = preserve_third_dimension

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
    if ndims == 3:
        upsample_kwargs['preserve_third_dimension'] = preserve_third_dimension
    upsample_kwargs['upsample_size'] = upsample_size
    upsample_kwargs['use_bias'] = use_bias

    # Keyword arguments for the conventional skip connections
    conventional_kwargs = dict({})
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

    # Keyword arguments for the deep supervision output in the final decoder node
    supervision_kwargs = dict({})
    supervision_kwargs['upsample_size'] = upsample_size
    supervision_kwargs['use_bias'] = True
    supervision_kwargs['padding'] = padding
    supervision_kwargs['kernel_initializer'] = kernel_initializer
    supervision_kwargs['bias_initializer'] = bias_initializer
    supervision_kwargs['kernel_regularizer'] = kernel_regularizer
    supervision_kwargs['bias_regularizer'] = bias_regularizer
    supervision_kwargs['activity_regularizer'] = activity_regularizer
    supervision_kwargs['kernel_constraint'] = kernel_constraint
    supervision_kwargs['bias_constraint'] = bias_constraint
    if ndims == 3:
        supervision_kwargs['preserve_third_dimension'] = preserve_third_dimension

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
    tensors[f'De{levels - 1}'] = Concatenate(name=f'De{levels - 1}_Concatenate')([upsample_tensor, tensors[f'En{levels - 1}']])  # Concatenate the upsampled tensor and skip connection
    tensors[f'De{levels - 1}'] = convolution_module(tensors[f'De{levels - 1}'], filters=filter_num[levels - 2], kernel_size=kernel_size, name=f'De{levels - 1}', **module_kwargs)  # Convolution module
    upsample_tensor = upsample(tensors[f'De{levels - 1}'], filters=filter_num[levels - 3], kernel_size=kernel_size, name=f'De{levels - 1}-De{levels - 2}', **upsample_kwargs)  # Connect the bottom decoder node to the next decoder node

    """ The rest of the decoder nodes (except the final node) are handled in this loop. Each node contains one concatenation of an upsampled tensor and a skip connection """
    for decoder in np.arange(2, levels-1)[::-1]:
        tensors[f'De{decoder}'] = Concatenate(name=f'De{decoder}_Concatenate')([upsample_tensor, tensors[f'En{decoder}']])  # Concatenate the upsampled tensor and skip connection
        tensors[f'De{decoder}'] = convolution_module(tensors[f'De{decoder}'], filters=filter_num[decoder - 1], kernel_size=kernel_size, name=f'De{decoder}', **module_kwargs)  # Convolution module
        upsample_tensor = upsample(tensors[f'De{decoder}'], filters=filter_num[decoder - 2], kernel_size=kernel_size, name=f'De{decoder}-De{decoder - 1}', **upsample_kwargs)  # Connect the bottom decoder node to the next decoder node

    """ Final decoder node begins with a concatenation and convolution module, followed by deep supervision """
    tensor_De1 = Concatenate(name='De1_Concatenate')([upsample_tensor, tensors['En1']])  # Concatenate the upsampled tensor and skip connection
    tensor_De1 = convolution_module(tensor_De1, filters=filter_num[0], kernel_size=kernel_size, name='De1', **module_kwargs)  # Convolution module
    tensors['output'] = deep_supervision_side_output(tensor_De1, num_classes=num_classes, kernel_size=1, output_level=1, name='final', **supervision_kwargs)  # Deep supervision - this layer will output the model's prediction

    model = Model(inputs=tensors['input'], outputs=tensors['output'], name=f'U-Net ({ndims}D)')
    model.summary()  # Prints summary of the model, contains information about the model's layers and structure

    return model


def unet_3plus(input_shape, num_classes, kernel_size=3, filter_levels=(16, 32, 64, 128, 256, 512), filter_num_skip=None,
    filter_num_aggregate=None, modules_per_node=5, batch_normalization=True, activation='relu', padding='same', use_bias=False,
    preserve_third_dimension=False, pool_size=2, upsample_size=2):
    """
    Creates a U-Net 3+.

    Parameters
    ----------
    input_shape: tuple
        - Shape of the inputs.
    num_classes: int
        - Number of classes/labels that the U-Net will try to predict.
    kernel_size: int or tuple
        - Size of the kernel in the convolution layers.
    filter_levels: iterable of ints
        - Number of convolution filters in each encoder of the U-Net.
    filter_num_skip: int or None
        - Number of convolution filters in the conventional skip connections, full-scale skip connections, and aggregated feature maps.
        - NOTE: When left as None, this will default to the first value in the 'filter_levels' iterable.
    filter_num_aggregate: int or None
        - Number of convolution filters in the decoder nodes after images are concatenated.
        - When left as None, this will be equal to the product of filter_num_skip and the length of the filter_levels iterable.
    modules_per_node: int
        - Number of modules in each node of the U-Net 3+.
    batch_normalization: bool
        - Setting this to True will add a batch normalization layer after every convolution in the modules.
    activation: str
        - Activation function to use in the modules.
        - Can be any of tf.keras.activations, 'prelu', 'leaky_relu', or 'smelu' (case-insensitive).
    padding: str
        - Padding to use in the convolution layers.
    use_bias: bool
        - Setting this to True will implement a bias vector in the convolution layers used in the modules.
    preserve_third_dimension: bool
        - Setting this to True will preserve the third dimension of the U-Net so that it is not modified by the MaxPooling and UpSampling layers.
    pool_size: int
        - Size of the mask in the MaxPooling layers.
    upsample_size: int
        - Size of the mask in the UpSampling layers.

    Returns
    -------
    model: tf.keras.models.Model object
        - U-Net 3+ model.
    """

    ndims = len(input_shape) - 1  # Number of dimensions in the input image (excluding the last dimension reserved for channels)

    if preserve_third_dimension is True and len(input_shape) != 4:
        raise ValueError(f"preserve_third_dimension is True but the input size does not have 4 dimensions (3D image + 1 dimension for channels). Received shape: {np.shape(input_shape)}")

    if filter_num_skip is None:
        filter_num_skip = filter_levels[0]

    if filter_num_aggregate is None:
        filter_num_aggregate = len(filter_levels) * filter_num_skip

    print(f"\nCreating model: {ndims}D U-Net 3+")

    module_kwargs = dict({})
    module_kwargs['activation'] = activation
    module_kwargs['batch_normalization'] = batch_normalization
    module_kwargs['num_modules'] = modules_per_node
    module_kwargs['padding'] = padding
    module_kwargs['use_bias'] = use_bias

    pool_kwargs = dict({})
    pool_kwargs['pool_size'] = pool_size
    if ndims == 3:
        pool_kwargs['preserve_third_dimension'] = preserve_third_dimension

    upsample_kwargs = dict({})
    upsample_kwargs['activation'] = activation
    upsample_kwargs['batch_normalization'] = batch_normalization
    upsample_kwargs['padding'] = padding
    if ndims == 3:
        upsample_kwargs['preserve_third_dimension'] = preserve_third_dimension
    upsample_kwargs['upsample_size'] = upsample_size
    upsample_kwargs['use_bias'] = use_bias

    conventional_kwargs = dict({})
    conventional_kwargs['activation'] = activation
    conventional_kwargs['batch_normalization'] = batch_normalization
    conventional_kwargs['padding'] = padding
    conventional_kwargs['use_bias'] = use_bias

    full_scale_kwargs = dict({})
    full_scale_kwargs['activation'] = activation
    full_scale_kwargs['batch_normalization'] = batch_normalization
    full_scale_kwargs['padding'] = padding
    full_scale_kwargs['pool_size'] = pool_size
    if ndims == 3:
        full_scale_kwargs['preserve_third_dimension'] = preserve_third_dimension
    full_scale_kwargs['use_bias'] = use_bias

    aggregated_kwargs = dict({})
    aggregated_kwargs['activation'] = activation
    aggregated_kwargs['batch_normalization'] = batch_normalization
    aggregated_kwargs['padding'] = padding
    aggregated_kwargs['preserve_third_dimension'] = preserve_third_dimension
    aggregated_kwargs['upsample_size'] = upsample_size
    aggregated_kwargs['use_bias'] = use_bias

    supervision_kwargs = dict({})
    supervision_kwargs['upsample_size'] = upsample_size
    supervision_kwargs['use_bias'] = True
    supervision_kwargs['padding'] = padding
    if ndims == 3:
        supervision_kwargs['preserve_third_dimension'] = preserve_third_dimension

    input_tensor = Input(shape=input_shape, name='Input')

    """ Encoder 1 """
    tensor_En1 = convolution_module(input_tensor, filters=filter_levels[0], kernel_size=kernel_size, name='En1', **module_kwargs)

    """ Encoder 2 """
    tensor_En2 = max_pool(tensor_En1, name='En1-En2', **pool_kwargs)
    tensor_En2 = convolution_module(tensor_En2, filters=filter_levels[1], kernel_size=kernel_size, name='En2', **module_kwargs)

    # Skip connections
    tensor_En2De2 = conventional_skip_connection(tensor_En2, filters=filter_num_skip, kernel_size=kernel_size, name='En2---De2', **conventional_kwargs)
    tensor_En2De3 = full_scale_skip_connection(tensor_En2, filters=filter_num_skip, kernel_size=kernel_size, level1=2, level2=3, name='En2---De3', **full_scale_kwargs)
    tensor_En2De4 = full_scale_skip_connection(tensor_En2, filters=filter_num_skip, kernel_size=kernel_size, level1=2, level2=4, name='En2---De4', **full_scale_kwargs)
    tensor_En2De5 = full_scale_skip_connection(tensor_En2, filters=filter_num_skip, kernel_size=kernel_size, level1=2, level2=5, name='En2---De5', **full_scale_kwargs)

    """ Encoder 3 """
    tensor_En3 = max_pool(tensor_En2, name='En2-En3', **pool_kwargs)
    tensor_En3 = convolution_module(tensor_En3, filters=filter_levels[2], kernel_size=kernel_size, name='En3', **module_kwargs)

    # Skip connections #
    tensor_En3De3 = conventional_skip_connection(tensor_En3, filters=filter_num_skip, kernel_size=kernel_size, name='En3---De3', **conventional_kwargs)
    tensor_En3De4 = full_scale_skip_connection(tensor_En3, filters=filter_num_skip, kernel_size=kernel_size, level1=3, level2=4, name='En3---De4', **full_scale_kwargs)
    tensor_En3De5 = full_scale_skip_connection(tensor_En3, filters=filter_num_skip, kernel_size=kernel_size, level1=3, level2=5, name='En3---De5', **full_scale_kwargs)

    """ Encoder 4 """
    tensor_En4 = max_pool(tensor_En3, name='En3-En4', **pool_kwargs)
    tensor_En4 = convolution_module(tensor_En4, filters=filter_levels[3], kernel_size=kernel_size, name='En4', **module_kwargs)

    # Skip connections #
    tensor_En4De4 = conventional_skip_connection(tensor_En4, filters=filter_num_skip, kernel_size=kernel_size, name='En4---De4', **conventional_kwargs)
    tensor_En4De5 = full_scale_skip_connection(tensor_En4, filters=filter_num_skip, kernel_size=kernel_size, level1=4, level2=5, name='En4---De5', **full_scale_kwargs)

    """ Encoder 5 """
    tensor_En5 = max_pool(tensor_En4, name='En4-En5', **pool_kwargs)
    tensor_En5 = convolution_module(tensor_En5, filters=filter_levels[4], kernel_size=kernel_size, name='En5', **module_kwargs)

    # Skip connections #
    tensor_En5De5 = conventional_skip_connection(tensor_En5, filters=filter_num_skip, kernel_size=kernel_size, name='En5---De5', **conventional_kwargs)

    """ Encoder 6 (bottom layer) """
    tensor_En6 = max_pool(tensor_En5, name='En5-En6', **pool_kwargs)
    tensor_En6 = convolution_module(tensor_En6, filters=filter_levels[5], kernel_size=kernel_size, name='En6', **module_kwargs)
    sup6_output = deep_supervision_side_output(tensor_En6, num_classes=num_classes, kernel_size=kernel_size, output_level=6, name='sup6', **supervision_kwargs)

    # Skip connections #
    tensor_En6De1 = aggregated_feature_map(tensor_En6, filters=filter_num_skip, kernel_size=kernel_size, level1=6, level2=1, name='En6---De1', **aggregated_kwargs)
    tensor_En6De2 = aggregated_feature_map(tensor_En6, filters=filter_num_skip, kernel_size=kernel_size, level1=6, level2=2, name='En6---De2', **aggregated_kwargs)
    tensor_En6De3 = aggregated_feature_map(tensor_En6, filters=filter_num_skip, kernel_size=kernel_size, level1=6, level2=3, name='En6---De3', **aggregated_kwargs)
    tensor_En6De4 = aggregated_feature_map(tensor_En6, filters=filter_num_skip, kernel_size=kernel_size, level1=6, level2=4, name='En6---De4', **aggregated_kwargs)

    """ Decoder 5 """
    tensor_En6De5 = upsample(tensor_En6, filters=filter_num_skip, kernel_size=kernel_size, name='En6-De5', **upsample_kwargs)
    tensor_De5 = Concatenate(name='De5_Concatenate')([tensor_En6De5, tensor_En5De5, tensor_En4De5, tensor_En3De5, tensor_En2De5])
    tensor_De5 = convolution_module(tensor_De5, filters=filter_num_aggregate, kernel_size=kernel_size, name='De5', **module_kwargs)
    sup5_output = deep_supervision_side_output(tensor_De5, num_classes=num_classes, kernel_size=kernel_size, output_level=5, name='sup5', **supervision_kwargs)

    # Skip connections #
    tensor_De5De1 = aggregated_feature_map(tensor_De5, filters=filter_num_skip, kernel_size=kernel_size, level1=5, level2=1, name='De5---De1', **aggregated_kwargs)
    tensor_De5De2 = aggregated_feature_map(tensor_De5, filters=filter_num_skip, kernel_size=kernel_size, level1=5, level2=2, name='De5---De2', **aggregated_kwargs)
    tensor_De5De3 = aggregated_feature_map(tensor_De5, filters=filter_num_skip, kernel_size=kernel_size, level1=5, level2=3, name='De5---De3', **aggregated_kwargs)

    """ Decoder 4 """
    tensor_De5De4 = upsample(tensor_De5, filters=filter_num_skip, kernel_size=kernel_size, name='De5-De4', **upsample_kwargs)
    tensor_De4 = Concatenate(name='De4_Concatenate')([tensor_En6De4, tensor_De5De4, tensor_En4De4, tensor_En3De4, tensor_En2De4])
    tensor_De4 = convolution_module(tensor_De4, filters=filter_num_aggregate, kernel_size=kernel_size, name='De4', **module_kwargs)
    sup4_output = deep_supervision_side_output(tensor_De4, num_classes=num_classes, kernel_size=kernel_size, output_level=4, name='sup4', **supervision_kwargs)

    # Skip connections #
    tensor_De4De1 = aggregated_feature_map(tensor_De4, filters=filter_num_skip, kernel_size=kernel_size, level1=4, level2=1, name='De4---De1', **aggregated_kwargs)
    tensor_De4De2 = aggregated_feature_map(tensor_De4, filters=filter_num_skip, kernel_size=kernel_size, level1=4, level2=2, name='De4---De2', **aggregated_kwargs)

    """ Decoder 3 """
    tensor_De4De3 = upsample(tensor_De4, filters=filter_num_skip, kernel_size=kernel_size, name='De4-De3', **upsample_kwargs)
    tensor_De3 = Concatenate(name='De3_Concatenate')([tensor_En6De3, tensor_De5De3, tensor_De4De3, tensor_En3De3, tensor_En2De3])
    tensor_De3 = convolution_module(tensor_De3, filters=filter_num_aggregate, kernel_size=kernel_size, name='De3', **module_kwargs)
    sup3_output = deep_supervision_side_output(tensor_De3, num_classes=num_classes, kernel_size=kernel_size, output_level=3, name='sup3', **supervision_kwargs)

    # Skip connection #
    tensor_De3De1 = aggregated_feature_map(tensor_De3, filters=filter_num_skip, kernel_size=kernel_size, level1=3, level2=1, name='De3---De1', **aggregated_kwargs)

    """ Decoder 2 """
    tensor_De3De2 = upsample(tensor_De3, filters=filter_num_skip, kernel_size=kernel_size, name='De3-De2', **upsample_kwargs)
    tensor_De2 = Concatenate(name='De2_Concatenate')([tensor_En6De2, tensor_De5De2, tensor_De4De2, tensor_De3De2, tensor_En2De2])
    tensor_De2 = convolution_module(tensor_De2, filters=filter_num_aggregate, kernel_size=kernel_size, name='De2', **module_kwargs)
    sup2_output = deep_supervision_side_output(tensor_De2, num_classes=num_classes, kernel_size=kernel_size, output_level=2, name='sup2', **supervision_kwargs)

    """ Decoder 1 """
    tensor_De2De1 = upsample(tensor_De2, filters=filter_num_skip, kernel_size=kernel_size, name='De2-De1', **upsample_kwargs)
    tensor_De1 = Concatenate(name='De1_Concatenate')([tensor_En6De1, tensor_De5De1, tensor_De4De1, tensor_De3De1, tensor_De2De1])
    tensor_De1 = convolution_module(tensor_De1, filters=filter_num_aggregate, kernel_size=kernel_size, name='De1', **module_kwargs)
    final_output = deep_supervision_side_output(tensor_De1, num_classes=num_classes, kernel_size=kernel_size, output_level=1, name='final', **supervision_kwargs)

    model = Model(inputs=input_tensor, outputs=[final_output, sup2_output, sup3_output, sup4_output, sup5_output, sup6_output], name=f'3plus{ndims}D')
    model.summary()  # Prints summary of the model, contains information about the model's layers and structure

    return model
