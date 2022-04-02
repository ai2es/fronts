"""
Models for front detection

Author: Andrew Justin (andrewjustin@ou.edu)
Last updated: 4/1/2022 8:55 PM CDT
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Input
from utils.unet_utils import *


def unet(input_size, num_classes, kernel_size=3, filter_num=(16, 32, 64, 128, 256, 512), modules_per_node=5, batch_normalization=False,
    activation='relu', padding='same', use_bias=False, preserve_third_dimension=True, pool_size=2, upsample_size=2):
    """
    Creates a 3-dimensional U-Net.

    Parameters
    ----------
    activation: str
        Activation function to use in the modules.
    batch_normalization: bool
        Setting this to True will add a batch normalization layer after every convolution in the modules.
    filter_num: iterable of ints
        Number of convolution filters on each level of the U-Net.
    input_size: tuple of 4 ints
        Size of the inputs. The shape of the tuple should be: (image_size_x, image_size_y, image_size_z, number of predictors)
    kernel_size: int or tuple
        Size of the kernel in the convolution layers.
    modules_per_node: int
        Number of modules in each node of the U-Net.
    num_classes: int
        Number of classes/labels that the U-Net will try to predict.
    padding: str
        Padding to use in the convolution layers.
    pool_size: int
        Size of the mask in the MaxPooling layers.
    preserve_third_dimension: bool
        Setting this to True will preserve the third dimension of the U-Net so that it is not modified by the MaxPooling and UpSampling layers.
    upsample_size: int
        Size of the mask in the UpSampling layers.
    use_bias: bool
        Setting this to True will implement a bias vector in the convolution layers used in the modules.

    Returns
    -------
    model: U-Net
    """

    ndims = len(input_size) - 1  # Number of dimensions in the input image (excluding the last dimension reserved for channels)

    if preserve_third_dimension is True and len(input_size) != 4:
        raise ValueError(f"preserve_third_dimension is True but input_size does not have 4 dimensions (3D image + 1 dimension for channels). Received shape: {np.shape(input_size)}")

    if len(input_size) > 4 or len(input_size) < 3:
        raise ValueError(f"input_size can only have 3 or 4 dimensions (2D image + 1 dimension for channels OR a 3D image + 1 dimension for channels). Received shape: {np.shape(input_size)}")

    print(f"\nCreating model: {ndims}D U-Net")

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

    supervision_kwargs = dict({})
    supervision_kwargs['upsample_size'] = upsample_size
    supervision_kwargs['use_bias'] = True
    supervision_kwargs['padding'] = padding
    if ndims == 3:
        supervision_kwargs['preserve_third_dimension'] = preserve_third_dimension

    input_tensor = Input(shape=input_size, name='Input')

    """ Encoder 1 """
    tensor_En1 = convolution_module(input_tensor, filters=filter_num[0], kernel_size=kernel_size, name='En1', **module_kwargs)

    """ Encoder 2 """
    tensor_En2 = max_pool(tensor_En1, name='En1-En2', **pool_kwargs)
    tensor_En2 = convolution_module(tensor_En2, filters=filter_num[1], kernel_size=kernel_size, name='En2', **module_kwargs)

    """ Encoder 3 """
    tensor_En3 = max_pool(tensor_En2, name='En2-En3', **pool_kwargs)
    tensor_En3 = convolution_module(tensor_En3, filters=filter_num[2], kernel_size=kernel_size, name='En3', **module_kwargs)

    """ Encoder 4 """
    tensor_En4 = max_pool(tensor_En3, name='En3-En4', **pool_kwargs)
    tensor_En4 = convolution_module(tensor_En4, filters=filter_num[3], kernel_size=kernel_size, name='En4', **module_kwargs)

    """ Encoder 5 """
    tensor_En5 = max_pool(tensor_En4, name='En4-En5', **pool_kwargs)
    tensor_En5 = convolution_module(tensor_En5, filters=filter_num[4], kernel_size=kernel_size, name='En5', **module_kwargs)

    """ Encoder 6 """
    tensor_En6 = max_pool(tensor_En5, name='En5-En6', **pool_kwargs)
    tensor_En6 = convolution_module(tensor_En6, filters=filter_num[5], kernel_size=kernel_size, name='En6', **module_kwargs)
    tensor_En6De5 = upsample(tensor_En6, filters=filter_num[4], kernel_size=kernel_size, name='En6-De5', **upsample_kwargs)

    """ Decoder 5 """
    tensor_De5 = Concatenate(name='De5_Concatenate')([tensor_En6De5, tensor_En5])
    tensor_De5 = convolution_module(tensor_De5, filters=filter_num[4], kernel_size=kernel_size, name='De5', **module_kwargs)
    tensor_De5De4 = upsample(tensor_De5, filters=filter_num[3], kernel_size=kernel_size, name='De5-De4', **upsample_kwargs)

    """ Decoder 4 """
    tensor_De4 = Concatenate(name='De4_Concatenate')([tensor_De5De4, tensor_En4])
    tensor_De4 = convolution_module(tensor_De4, filters=filter_num[3], kernel_size=kernel_size, name='De4', **module_kwargs)
    tensor_De4De3 = upsample(tensor_De4, filters=filter_num[2], kernel_size=kernel_size, name='De4-De3', **upsample_kwargs)

    """ Decoder 3 """
    tensor_De3 = Concatenate(name='De3_Concatenate')([tensor_De4De3, tensor_En3])
    tensor_De3 = convolution_module(tensor_De3, filters=filter_num[2], kernel_size=kernel_size, name='De3', **module_kwargs)
    tensor_De3De2 = upsample(tensor_De3, filters=filter_num[1], kernel_size=kernel_size, name='De3-De2', **upsample_kwargs)

    """ Decoder 2 """
    tensor_De2 = Concatenate(name='De2_Concatenate')([tensor_De3De2, tensor_En2])
    tensor_De2 = convolution_module(tensor_De2, filters=filter_num[1], kernel_size=kernel_size, name='De2', **module_kwargs)
    tensor_De2De1 = upsample(tensor_De2, filters=filter_num[0], kernel_size=kernel_size, name='De2-De1', **upsample_kwargs)

    """ Decoder 1 """
    tensor_De1 = Concatenate(name='De1_Concatenate')([tensor_De2De1, tensor_En1])
    tensor_De1 = convolution_module(tensor_De1, filters=filter_num[0], kernel_size=kernel_size, name='De1', **module_kwargs)
    final_output = deep_supervision_side_output(tensor_De1, num_classes=num_classes, kernel_size=1, output_level=1, name='final', **supervision_kwargs)

    model = Model(inputs=input_tensor, outputs=final_output, name='U-Net (3D)')
    model.summary()  # Prints summary of the model, contains information about the model's layers and structure

    return model


def unet_3plus(input_size, num_classes, kernel_size=3, filter_num_down=(16, 32, 64, 128, 256, 512), filter_num_skip=None,
    filter_num_aggregate=None, modules_per_node=5, batch_normalization=True, activation='relu', padding='same', use_bias=False,
    preserve_third_dimension=False, pool_size=2, upsample_size=2):
    """
    Creates a U-Net 3+.

    Parameters
    ----------
    activation: str
        Activation function to use in the modules.
    batch_normalization: bool
        Setting this to True will add a batch normalization layer after every convolution in the modules.
    filter_num_aggregate: int or None
        Number of convolution filters in the decoder nodes after images are concatenated.
        ** NOTE: When left as None, this will be equal to the product of filter_num_skip and the length of the filter_num_down iterable.
    filter_num_down: iterable of ints
        Number of convolution filters in each encoder of the U-Net.
    filter_num_skip: int or None
        Number of convolution filters in the conventional skip connections, full-scale skip connections, and aggregated feature maps.
        ** NOTE: When left as None, this will default to the first value in the 'filter_num_down' iterable.
    input_size: tuple of 4 ints
        Size of the inputs. The shape of the tuple should be: (image_size_x, image_size_y, image_size_z, number of predictors)
    kernel_size: int or tuple
        Size of the kernel in the convolution layers.
    modules_per_node: int
        Number of modules in each node of the U-Net 3+.
    num_classes: int
        Number of classes/labels that the U-Net will try to predict.
    padding: str
        Padding to use in the convolution layers.
    pool_size: int
        Size of the mask in the MaxPooling layers.
    preserve_third_dimension: bool
        Setting this to True will preserve the third dimension of the U-Net so that it is not modified by the MaxPooling and UpSampling layers.
    upsample_size: int
        Size of the mask in the UpSampling layers.
    use_bias: bool
        Setting this to True will implement a bias vector in the convolution layers used in the modules.

    Returns
    -------
    model: U-Net 3+
    """

    ndims = len(input_size) - 1  # Number of dimensions in the input image (excluding the last dimension reserved for channels)

    if preserve_third_dimension is True and len(input_size) != 4:
        raise ValueError(f"preserve_third_dimension is True but the input size does not have 4 dimensions (3D image + 1 dimension for channels). Received shape: {np.shape(input_size)}")

    if filter_num_skip is None:
        filter_num_skip = filter_num_down[0]

    if filter_num_aggregate is None:
        filter_num_aggregate = len(filter_num_down) * filter_num_skip

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

    input_tensor = Input(shape=input_size, name='Input')

    """ Encoder 1 """
    tensor_En1 = convolution_module(input_tensor, filters=filter_num_down[0], kernel_size=kernel_size, name='En1', **module_kwargs)

    """ Encoder 2 """
    tensor_En2 = max_pool(tensor_En1, name='En1-En2', **pool_kwargs)
    tensor_En2 = convolution_module(tensor_En2, filters=filter_num_down[1], kernel_size=kernel_size, name='En2', **module_kwargs)

    # Skip connections
    tensor_En2De2 = conventional_skip_connection(tensor_En2, filters=filter_num_skip, kernel_size=kernel_size, name='En2---De2', **conventional_kwargs)
    tensor_En2De3 = full_scale_skip_connection(tensor_En2, filters=filter_num_skip, kernel_size=kernel_size, level1=2, level2=3, name='En2---De3', **full_scale_kwargs)
    tensor_En2De4 = full_scale_skip_connection(tensor_En2, filters=filter_num_skip, kernel_size=kernel_size, level1=2, level2=4, name='En2---De4', **full_scale_kwargs)
    tensor_En2De5 = full_scale_skip_connection(tensor_En2, filters=filter_num_skip, kernel_size=kernel_size, level1=2, level2=5, name='En2---De5', **full_scale_kwargs)

    """ Encoder 3 """
    tensor_En3 = max_pool(tensor_En2, name='En2-En3', **pool_kwargs)
    tensor_En3 = convolution_module(tensor_En3, filters=filter_num_down[2], kernel_size=kernel_size, name='En3', **module_kwargs)

    # Skip connections #
    tensor_En3De3 = conventional_skip_connection(tensor_En3, filters=filter_num_skip, kernel_size=kernel_size, name='En3---De3', **conventional_kwargs)
    tensor_En3De4 = full_scale_skip_connection(tensor_En3, filters=filter_num_skip, kernel_size=kernel_size, level1=3, level2=4, name='En3---De4', **full_scale_kwargs)
    tensor_En3De5 = full_scale_skip_connection(tensor_En3, filters=filter_num_skip, kernel_size=kernel_size, level1=3, level2=5, name='En3---De5', **full_scale_kwargs)

    """ Encoder 4 """
    tensor_En4 = max_pool(tensor_En3, name='En3-En4', **pool_kwargs)
    tensor_En4 = convolution_module(tensor_En4, filters=filter_num_down[3], kernel_size=kernel_size, name='En4', **module_kwargs)

    # Skip connections #
    tensor_En4De4 = conventional_skip_connection(tensor_En4, filters=filter_num_skip, kernel_size=kernel_size, name='En4---De4', **conventional_kwargs)
    tensor_En4De5 = full_scale_skip_connection(tensor_En4, filters=filter_num_skip, kernel_size=kernel_size, level1=4, level2=5, name='En4---De5', **full_scale_kwargs)

    """ Encoder 5 """
    tensor_En5 = max_pool(tensor_En4, name='En4-En5', **pool_kwargs)
    tensor_En5 = convolution_module(tensor_En5, filters=filter_num_down[4], kernel_size=kernel_size, name='En5', **module_kwargs)

    # Skip connections #
    tensor_En5De5 = conventional_skip_connection(tensor_En5, filters=filter_num_skip, kernel_size=kernel_size, name='En5---De5', **conventional_kwargs)

    """ Encoder 6 (bottom layer) """
    tensor_En6 = max_pool(tensor_En5, name='En5-En6', **pool_kwargs)
    tensor_En6 = convolution_module(tensor_En6, filters=filter_num_down[5], kernel_size=kernel_size, name='En6', **module_kwargs)
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