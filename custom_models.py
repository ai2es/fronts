"""
Custom U-Net Models

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 9/18/2021 4:42 PM CDT
"""

import keras
from keras.layers import Conv2D, Conv3D, BatchNormalization, MaxPooling2D, MaxPooling3D, PReLU, Concatenate, Input, UpSampling2D, UpSampling3D, Softmax


def UNet_3plus_3D(map_dim_x, map_dim_y, num_classes):
    """
    Creates a 3-dimensional U-Net 3+.

    Parameters
    ----------
    map_dim_x: int
        Integer that determines the X dimension of the image (map) to be fed into the Unet.
    map_dim_y: int
        Integer that determines the Y dimension of the image (map) to be fed into the Unet.
    num_classes: int
        Number of classes/labels that the U-Net will try to predict.

    Returns
    -------
    model: Keras model
        3D U-Net 3+.
    """
    # U-Net 3+ #
    filter_num_down = [16, 32, 64, 128, 256, 512]
    filter_num_skip = 16
    filter_num_aggregate = 96
    kernel_size = 3

    print("\n3D U-Net 3+")
    print("filter_num_down:", filter_num_down)
    print("filter_num_skip: %d" % filter_num_skip)
    print("filter_num_aggregate: %d" % filter_num_aggregate)
    print("kernel size:",kernel_size)

    inputs = Input(shape=(map_dim_x, map_dim_y, 5, 12))

    """ Encoder 1 """
    x = Conv3D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(inputs)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    """ Encoder 2 """
    x = MaxPooling3D(pool_size=(2,2,1))(x)
    x = Conv3D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    # Skip connections #
    En2De2 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    En2De2 = BatchNormalization()(En2De2)
    En2De2 = PReLU()(En2De2)

    En2De3 = MaxPooling3D(pool_size=(2,2,1))(x)
    En2De3 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En2De3)
    En2De3 = BatchNormalization()(En2De3)
    En2De3 = PReLU()(En2De3)

    En2De4 = MaxPooling3D(pool_size=(4,4,1))(x)
    En2De4 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En2De4)
    En2De4 = BatchNormalization()(En2De4)
    En2De4 = PReLU()(En2De4)

    En2De5 = MaxPooling3D(pool_size=(8,8,1))(x)
    En2De5 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En2De5)
    En2De5 = BatchNormalization()(En2De5)
    En2De5 = PReLU()(En2De5)

    """ Encoder 3 """
    x = MaxPooling3D(pool_size=(2,2,1))(x)
    x = Conv3D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    # Skip connections #
    En3De3 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    En3De3 = BatchNormalization()(En3De3)
    En3De3 = PReLU()(En3De3)

    En3De4 = MaxPooling3D(pool_size=(2,2,1))(x)
    En3De4 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En3De4)
    En3De4 = BatchNormalization()(En3De4)
    En3De4 = PReLU()(En3De4)

    En3De5 = MaxPooling3D(pool_size=(4,4,1))(x)
    En3De5 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En3De5)
    En3De5 = BatchNormalization()(En3De5)
    En3De5 = PReLU()(En3De5)

    # Encoder 4 #
    x = MaxPooling3D(pool_size=(2,2,1))(x)
    x = Conv3D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    # Skip connections #
    En4De4 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    En4De4 = BatchNormalization()(En4De4)
    En4De4 = PReLU()(En4De4)

    En4De5 = MaxPooling3D(pool_size=(2,2,1))(x)
    En4De5 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En4De5)
    En4De5 = BatchNormalization()(En4De5)
    En4De5 = PReLU()(En4De5)

    # Encoder 5 #
    x = MaxPooling3D(pool_size=(2,2,1))(x)
    x = Conv3D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    # Skip connections #
    En5De5 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    En5De5 = BatchNormalization()(En5De5)
    En5De5 = PReLU()(En5De5)

    # Encoder 6 (bottom layer) #
    x = MaxPooling3D(pool_size=(2,2,1))(x)
    x = Conv3D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = UpSampling3D(size=(2,2,1))(x)
    sup4_conv = Conv3D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, data_format='channels_last')(x)
    sup4_upsample = UpSampling3D(size=(16,16,1))(sup4_conv)
    sup4_output = Softmax()(sup4_upsample)
    # Skip connections #
    En6De1 = UpSampling3D(size=(16,16,1))(x)
    En6De1 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En6De1)
    En6De1 = BatchNormalization()(En6De1)
    En6De1 = PReLU()(En6De1)

    En6De2 = UpSampling3D(size=(8,8,1))(x)
    En6De2 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En6De2)
    En6De2 = BatchNormalization()(En6De2)
    En6De2 = PReLU()(En6De2)

    En6De3 = UpSampling3D(size=(4,4,1))(x)
    En6De3 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En6De3)
    En6De3 = BatchNormalization()(En6De3)
    En6De3 = PReLU()(En6De3)

    En6De4 = UpSampling3D(size=(2,2,1))(x)
    En6De4 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En6De4)
    En6De4 = BatchNormalization()(En6De4)
    En6De4 = PReLU()(En6De4)

    # Decoder 5 #
    x = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Concatenate()([x, En5De5, En4De5, En3De5, En2De5])
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = UpSampling3D(size=(2,2,1))(x)
    sup3_conv = Conv3D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, data_format='channels_last')(x)
    sup3_upsample = UpSampling3D(size=(8,8,1))(sup3_conv)
    sup3_output = Softmax()(sup3_upsample)
    # Skip connections #
    De5De1 = UpSampling3D(size=(8,8,1))(x)
    De5De1 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(De5De1)
    De5De1 = BatchNormalization()(De5De1)
    De5De1 = PReLU()(De5De1)

    De5De2 = UpSampling3D(size=(4,4,1))(x)
    De5De2 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(De5De2)
    De5De2 = BatchNormalization()(De5De2)
    De5De2 = PReLU()(De5De2)

    De5De3 = UpSampling3D(size=(2,2,1))(x)
    De5De3 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(De5De3)
    De5De3 = BatchNormalization()(De5De3)
    De5De3 = PReLU()(De5De3)

    # Decoder 4 #
    x = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Concatenate()([En6De4, x, En4De4, En3De4, En2De4])
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = UpSampling3D(size=(2,2,1))(x)
    sup2_conv = Conv3D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, data_format='channels_last')(x)
    sup2_upsample = UpSampling3D(size=(4,4,1))(sup2_conv)
    sup2_output = Softmax()(sup2_upsample)
    # Skip connection #
    De4De1 = UpSampling3D(size=(4,4,1))(x)
    De4De1 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(De4De1)
    De4De1 = BatchNormalization()(De4De1)
    De4De1 = PReLU()(De4De1)

    De4De2 = UpSampling3D(size=(2,2,1))(x)
    De4De2 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(De4De2)
    De4De2 = BatchNormalization()(De4De2)
    De4De2 = PReLU()(De4De2)

    # Decoder 3 #
    x = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Concatenate()([En6De3, De5De3, x, En3De3, En2De3])
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = UpSampling3D(size=(2,2,1))(x)
    sup1_conv = Conv3D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, data_format='channels_last')(x)
    sup1_upsample = UpSampling3D(size=(2,2,1))(sup1_conv)
    sup1_output = Softmax()(sup1_upsample)
    # Skip connection #
    De3De1 = UpSampling3D(size=(2,2,1))(x)
    De3De1 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(De3De1)
    De3De1 = BatchNormalization()(De3De1)
    De3De1 = PReLU()(De3De1)

    # Decoder 2 #
    x = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Concatenate()([En6De2, De5De2, De4De2, x, En2De2])
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = UpSampling3D(size=(2,2,1))(x)
    sup0_conv = Conv3D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, data_format='channels_last')(x)
    sup0_output = Softmax()(sup0_conv)

    # Decoder 1 #
    x = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Concatenate()([En6De1, De5De1, De4De1, De3De1, x])
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, data_format='channels_last')(x)
    final_output = Softmax()(x)
    model = keras.Model(inputs=inputs, outputs=[final_output,sup0_output,sup1_output,sup2_output,sup3_output,sup4_output], name='3plus3D')

    return model


def UNet_3plus_2D(map_dim_x, map_dim_y, num_classes):
    """
    Creates a 2-dimensional U-Net 3+.

    Parameters
    ----------
    map_dim_x: int
        Integer that determines the X dimension of the image (map) to be fed into the Unet.
    map_dim_y: int
        Integer that determines the Y dimension of the image (map) to be fed into the Unet.
    num_classes: int
        Number of classes/labels that the U-Net will try to predict.

    Returns
    -------
    model: Keras model
        2D U-Net 3+.
    """
    # U-Net 3+ #
    filter_num_down = [64, 128, 256, 512, 1024, 2048]
    filter_num_skip = 64
    filter_num_aggregate = 384
    kernel_size = (3,3)
    strides = (1,1)

    print("\n2D U-Net 3+")
    print("filter_num_down:", filter_num_down)
    print("filter_num_skip: %d" % filter_num_skip)
    print("filter_num_aggregate: %d" % filter_num_aggregate)
    print("kernel size: %dx%d" % (kernel_size[0], kernel_size[1]))
    print("strides: (%d,%d)" % (strides[0], strides[1]))

    inputs = Input(shape=(map_dim_x, map_dim_y, 60))

    """ Encoder 1 """
    x = Conv2D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(inputs)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    """ Encoder 2 """
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    # Skip connections #
    En2De2 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    En2De2 = BatchNormalization()(En2De2)
    En2De2 = PReLU()(En2De2)

    En2De3 = MaxPooling2D(pool_size=(2,2))(x)
    En2De3 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En2De3)
    En2De3 = BatchNormalization()(En2De3)
    En2De3 = PReLU()(En2De3)

    En2De4 = MaxPooling2D(pool_size=(4,4))(x)
    En2De4 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En2De4)
    En2De4 = BatchNormalization()(En2De4)
    En2De4 = PReLU()(En2De4)

    En2De5 = MaxPooling2D(pool_size=(8,8))(x)
    En2De5 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En2De5)
    En2De5 = BatchNormalization()(En2De5)
    En2De5 = PReLU()(En2De5)

    """ Encoder 3 """
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    # Skip connections #
    En3De3 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    En3De3 = BatchNormalization()(En3De3)
    En3De3 = PReLU()(En3De3)

    En3De4 = MaxPooling2D(pool_size=(2,2))(x)
    En3De4 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En3De4)
    En3De4 = BatchNormalization()(En3De4)
    En3De4 = PReLU()(En3De4)

    En3De5 = MaxPooling2D(pool_size=(4,4))(x)
    En3De5 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En3De5)
    En3De5 = BatchNormalization()(En3De5)
    En3De5 = PReLU()(En3De5)

    # Encoder 4 #
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    # Skip connections #
    En4De4 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    En4De4 = BatchNormalization()(En4De4)
    En4De4 = PReLU()(En4De4)

    En4De5 = MaxPooling2D(pool_size=(2,2))(x)
    En4De5 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En4De5)
    En4De5 = BatchNormalization()(En4De5)
    En4De5 = PReLU()(En4De5)

    # Encoder 5 #
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    # Skip connections #
    En5De5 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    En5De5 = BatchNormalization()(En5De5)
    En5De5 = PReLU()(En5De5)

    # Encoder 6 (bottom layer) #
    x = MaxPooling3D(pool_size=(2,2))(x)
    x = Conv2D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = UpSampling2D(size=(2,2))(x)
    sup4_conv = Conv2D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, data_format='channels_last')(x)
    sup4_upsample = UpSampling2D(size=(16,16))(sup4_conv)
    sup4_output = Softmax()(sup4_upsample)
    # Skip connections #
    En6De1 = UpSampling2D(size=(16,16))(x)
    En6De1 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En6De1)
    En6De1 = BatchNormalization()(En6De1)
    En6De1 = PReLU()(En6De1)

    En6De2 = UpSampling2D(size=(8,8))(x)
    En6De2 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En6De2)
    En6De2 = BatchNormalization()(En6De2)
    En6De2 = PReLU()(En6De2)

    En6De3 = UpSampling2D(size=(4,4))(x)
    En6De3 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En6De3)
    En6De3 = BatchNormalization()(En6De3)
    En6De3 = PReLU()(En6De3)

    En6De4 = UpSampling2D(size=(2,2))(x)
    En6De4 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En6De4)
    En6De4 = BatchNormalization()(En6De4)
    En6De4 = PReLU()(En6De4)

    # Decoder 5 #
    x = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Concatenate()([x, En5De5, En4De5, En3De5, En2De5])
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = UpSampling2D(size=(2,2))(x)
    sup3_conv = Conv2D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, data_format='channels_last')(x)
    sup3_upsample = UpSampling2D(size=(8,8))(sup3_conv)
    sup3_output = Softmax()(sup3_upsample)
    # Skip connections #
    De5De1 = UpSampling2D(size=(8,8))(x)
    De5De1 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(De5De1)
    De5De1 = BatchNormalization()(De5De1)
    De5De1 = PReLU()(De5De1)

    De5De2 = UpSampling2D(size=(4,4))(x)
    De5De2 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(De5De2)
    De5De2 = BatchNormalization()(De5De2)
    De5De2 = PReLU()(De5De2)

    De5De3 = UpSampling2D(size=(2,2))(x)
    De5De3 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(De5De3)
    De5De3 = BatchNormalization()(De5De3)
    De5De3 = PReLU()(De5De3)

    # Decoder 4 #
    x = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Concatenate()([En6De4, x, En4De4, En3De4, En2De4])
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = UpSampling2D(size=(2,2))(x)
    sup2_conv = Conv2D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, data_format='channels_last')(x)
    sup2_upsample = UpSampling2D(size=(4,4))(sup2_conv)
    sup2_output = Softmax()(sup2_upsample)
    # Skip connection #
    De4De1 = UpSampling2D(size=(4,4))(x)
    De4De1 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(De4De1)
    De4De1 = BatchNormalization()(De4De1)
    De4De1 = PReLU()(De4De1)

    De4De2 = UpSampling2D(size=(2,2))(x)
    De4De2 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(De4De2)
    De4De2 = BatchNormalization()(De4De2)
    De4De2 = PReLU()(De4De2)

    # Decoder 3 #
    x = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Concatenate()([En6De3, De5De3, x, En3De3, En2De3])
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = UpSampling2D(size=(2,2))(x)
    sup1_conv = Conv2D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, data_format='channels_last')(x)
    sup1_upsample = UpSampling2D(size=(2,2))(sup1_conv)
    sup1_output = Softmax()(sup1_upsample)
    # Skip connection #
    De3De1 = UpSampling2D(size=(2,2))(x)
    De3De1 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(De3De1)
    De3De1 = BatchNormalization()(De3De1)
    De3De1 = PReLU()(De3De1)

    # Decoder 2 #
    x = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Concatenate()([En6De2, De5De2, De4De2, x, En2De2])
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = UpSampling2D(size=(2,2))(x)
    sup0_conv = Conv2D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, data_format='channels_last')(x)
    sup0_output = Softmax()(sup0_conv)

    # Decoder 1 #
    x = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Concatenate()([En6De1, De5De1, De4De1, De3De1, x])
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, data_format='channels_last')(x)
    final_output = Softmax()(x)
    model = keras.Model(inputs=inputs, outputs=[final_output,sup0_output,sup1_output,sup2_output,sup3_output,sup4_output], name='3plus2D')

    return model
