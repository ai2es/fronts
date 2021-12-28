"""
Custom U-Net Models

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 12/27/2021 5:39 PM CST
"""

import keras
from keras.layers import Conv2D, Conv3D, BatchNormalization, MaxPooling2D, MaxPooling3D, Concatenate, Input, \
    UpSampling2D, UpSampling3D, Softmax, ReLU


def UNet_3plus_2D(map_dim_x, map_dim_y, num_classes):
    """
    Creates a 2-dimensional U-Net 3+.

    Parameters
    ----------
    map_dim_x: Integer that determines the X dimension of the image (map) to be fed into the Unet.
    map_dim_y: Integer that determines the Y dimension of the image (map) to be fed into the Unet.
    num_classes: Number of classes/labels that the U-Net will try to predict.

    Returns
    -------
    model: 2D U-Net 3+
    """
    # U-Net 3+ #
    filter_num_down = [64, 128, 256, 512, 1024, 2048]
    filter_num_skip = 64
    filter_num_aggregate = 384
    kernel_size = 3

    print("\n2D U-Net 3+")
    print("filter_num_down:", filter_num_down)
    print("filter_num_skip: %d" % filter_num_skip)
    print("filter_num_aggregate: %d" % filter_num_aggregate)
    print("kernel size: %d" % kernel_size)

    inputs = Input(shape=(map_dim_x, map_dim_y, 60))

    """ Encoder 1 """
    x = Conv2D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    """ Encoder 2 """
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # Skip connections #
    En2De2 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    En2De2 = BatchNormalization()(En2De2)
    En2De2 = ReLU()(En2De2)

    En2De3 = MaxPooling2D(pool_size=(2,2))(x)
    En2De3 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En2De3)
    En2De3 = BatchNormalization()(En2De3)
    En2De3 = ReLU()(En2De3)

    En2De4 = MaxPooling2D(pool_size=(4,4))(x)
    En2De4 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En2De4)
    En2De4 = BatchNormalization()(En2De4)
    En2De4 = ReLU()(En2De4)

    En2De5 = MaxPooling2D(pool_size=(8,8))(x)
    En2De5 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En2De5)
    En2De5 = BatchNormalization()(En2De5)
    En2De5 = ReLU()(En2De5)

    """ Encoder 3 """
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # Skip connections #
    En3De3 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    En3De3 = BatchNormalization()(En3De3)
    En3De3 = ReLU()(En3De3)

    En3De4 = MaxPooling2D(pool_size=(2,2))(x)
    En3De4 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En3De4)
    En3De4 = BatchNormalization()(En3De4)
    En3De4 = ReLU()(En3De4)

    En3De5 = MaxPooling2D(pool_size=(4,4))(x)
    En3De5 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En3De5)
    En3De5 = BatchNormalization()(En3De5)
    En3De5 = ReLU()(En3De5)

    # Encoder 4 #
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # Skip connections #
    En4De4 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    En4De4 = BatchNormalization()(En4De4)
    En4De4 = ReLU()(En4De4)

    En4De5 = MaxPooling2D(pool_size=(2,2))(x)
    En4De5 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En4De5)
    En4De5 = BatchNormalization()(En4De5)
    En4De5 = ReLU()(En4De5)

    # Encoder 5 #
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # Skip connections #
    En5De5 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    En5De5 = BatchNormalization()(En5De5)
    En5De5 = ReLU()(En5De5)

    # Encoder 6 (bottom layer) #
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling2D(size=(2,2))(x)
    sup4_conv = Conv2D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, data_format='channels_last')(x)
    sup4_upsample = UpSampling2D(size=(16,16))(sup4_conv)
    sup4_output = Softmax()(sup4_upsample)
    # Skip connections #
    En6De1 = UpSampling2D(size=(16,16))(x)
    En6De1 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En6De1)
    En6De1 = BatchNormalization()(En6De1)
    En6De1 = ReLU()(En6De1)

    En6De2 = UpSampling2D(size=(8,8))(x)
    En6De2 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En6De2)
    En6De2 = BatchNormalization()(En6De2)
    En6De2 = ReLU()(En6De2)

    En6De3 = UpSampling2D(size=(4,4))(x)
    En6De3 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En6De3)
    En6De3 = BatchNormalization()(En6De3)
    En6De3 = ReLU()(En6De3)

    En6De4 = UpSampling2D(size=(2,2))(x)
    En6De4 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(En6De4)
    En6De4 = BatchNormalization()(En6De4)
    En6De4 = ReLU()(En6De4)

    # Decoder 5 #
    x = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Concatenate()([x, En5De5, En4De5, En3De5, En2De5])
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling2D(size=(2,2))(x)
    sup3_conv = Conv2D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, data_format='channels_last')(x)
    sup3_upsample = UpSampling2D(size=(8,8))(sup3_conv)
    sup3_output = Softmax()(sup3_upsample)
    # Skip connections #
    De5De1 = UpSampling2D(size=(8,8))(x)
    De5De1 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(De5De1)
    De5De1 = BatchNormalization()(De5De1)
    De5De1 = ReLU()(De5De1)

    De5De2 = UpSampling2D(size=(4,4))(x)
    De5De2 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(De5De2)
    De5De2 = BatchNormalization()(De5De2)
    De5De2 = ReLU()(De5De2)

    De5De3 = UpSampling2D(size=(2,2))(x)
    De5De3 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(De5De3)
    De5De3 = BatchNormalization()(De5De3)
    De5De3 = ReLU()(De5De3)

    # Decoder 4 #
    x = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Concatenate()([En6De4, x, En4De4, En3De4, En2De4])
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling2D(size=(2,2))(x)
    sup2_conv = Conv2D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, data_format='channels_last')(x)
    sup2_upsample = UpSampling2D(size=(4,4))(sup2_conv)
    sup2_output = Softmax()(sup2_upsample)
    # Skip connection #
    De4De1 = UpSampling2D(size=(4,4))(x)
    De4De1 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(De4De1)
    De4De1 = BatchNormalization()(De4De1)
    De4De1 = ReLU()(De4De1)

    De4De2 = UpSampling2D(size=(2,2))(x)
    De4De2 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(De4De2)
    De4De2 = BatchNormalization()(De4De2)
    De4De2 = ReLU()(De4De2)

    # Decoder 3 #
    x = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Concatenate()([En6De3, De5De3, x, En3De3, En2De3])
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling2D(size=(2,2))(x)
    sup1_conv = Conv2D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, data_format='channels_last')(x)
    sup1_upsample = UpSampling2D(size=(2,2))(sup1_conv)
    sup1_output = Softmax()(sup1_upsample)
    # Skip connection #
    De3De1 = UpSampling2D(size=(2,2))(x)
    De3De1 = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(De3De1)
    De3De1 = BatchNormalization()(De3De1)
    De3De1 = ReLU()(De3De1)

    # Decoder 2 #
    x = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Concatenate()([En6De2, De5De2, De4De2, x, En2De2])
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling2D(size=(2,2))(x)
    sup0_conv = Conv2D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, data_format='channels_last')(x)
    sup0_output = Softmax()(sup0_conv)

    # Decoder 1 #
    x = Conv2D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Concatenate()([En6De1, De5De1, De4De1, De3De1, x])
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, data_format='channels_last')(x)
    final_output = Softmax()(x)
    model = keras.Model(inputs=inputs, outputs=[final_output,sup0_output,sup1_output,sup2_output,sup3_output,sup4_output], name='3plus2D')

    return model


def UNet_3plus_3D(map_dim_x, map_dim_y, num_classes):
    """
    Creates a 3-dimensional U-Net 3+.

    Parameters
    ----------
    map_dim_x: Integer that determines the X dimension of the image (map) to be fed into the Unet.
    map_dim_y: Integer that determines the Y dimension of the image (map) to be fed into the Unet.
    num_classes: Number of classes/labels that the U-Net will try to predict.

    Returns
    -------
    model: 3D U-Net 3+
    """

    filter_num_down = [16, 32, 64, 128, 256, 512]
    filter_num_skip = 16
    filter_num_aggregate = 96
    kernel_size = 3

    print("\n3D U-Net 3+")
    print("filter_num_down:", filter_num_down)
    print("filter_num_skip: %d" % filter_num_skip)
    print("filter_num_aggregate: %d" % filter_num_aggregate)
    print("kernel size:",kernel_size)

    inputs = Input(shape=(map_dim_x, map_dim_y, 5, 12), name='Input')

    """ Encoder 1 """
    x = Conv3D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, data_format='channels_last', name='En1_Conv3D_1')(inputs)
    x = BatchNormalization(name='En1_BatchNorm_1')(x)
    x = ReLU(name='En1_ReLU_1')(x)
    x = Conv3D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En1_Conv3D_2')(x)
    x = BatchNormalization(name='En1_BatchNorm_2')(x)
    x = ReLU(name='En1_ReLU_2')(x)
    x = Conv3D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En1_Conv3D_3')(x)
    x = BatchNormalization(name='En1_BatchNorm_3')(x)
    x = ReLU(name='En1_ReLU_3')(x)
    x = Conv3D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En1_Conv3D_4')(x)
    x = BatchNormalization(name='En1_BatchNorm_4')(x)
    x = ReLU(name='En1_ReLU_4')(x)
    x = Conv3D(filters=filter_num_down[0], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En1_Conv3D_5')(x)
    x = BatchNormalization(name='En1_BatchNorm_5')(x)
    x = ReLU(name='En1_ReLU_5')(x)

    """ Encoder 2 """
    x = MaxPooling3D(pool_size=(2,2,1),name='En1->En2_MaxPool3D')(x)
    x = Conv3D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En2_Conv3D_1')(x)
    x = BatchNormalization(name='En2_BatchNorm_1')(x)
    x = ReLU(name='En2_ReLU_1')(x)
    x = Conv3D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En2_Conv3D_2')(x)
    x = BatchNormalization(name='En2_BatchNorm_2')(x)
    x = ReLU(name='En2_ReLU_2')(x)
    x = Conv3D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En2_Conv3D_3')(x)
    x = BatchNormalization(name='En2_BatchNorm_3')(x)
    x = ReLU(name='En2_ReLU_3')(x)
    x = Conv3D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En2_Conv3D_4')(x)
    x = BatchNormalization(name='En2_BatchNorm_4')(x)
    x = ReLU(name='En2_ReLU_4')(x)
    x = Conv3D(filters=filter_num_down[1], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En2_Conv3D_5')(x)
    x = BatchNormalization(name='En2_BatchNorm_5')(x)
    x = ReLU(name='En2_ReLU_5')(x)
    # Skip connections #
    En2De2 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En2--->De2_Conv3D')(x)
    En2De2 = BatchNormalization(name='En2--->De2_BatchNorm')(En2De2)
    En2De2 = ReLU(name='En2--->De2_ReLU')(En2De2)

    En2De3 = MaxPooling3D(pool_size=(2,2,1), name='En2--->De3_MaxPool3D')(x)
    En2De3 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En2--->De3_Conv3D')(En2De3)
    En2De3 = BatchNormalization(name='En2--->De3_BatchNorm')(En2De3)
    En2De3 = ReLU(name='En2--->De3_ReLU')(En2De3)

    En2De4 = MaxPooling3D(pool_size=(4,4,1), name='En2--->De4_MaxPool3D')(x)
    En2De4 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En2--->De4_Conv3D')(En2De4)
    En2De4 = BatchNormalization(name='En2--->De4_BatchNorm')(En2De4)
    En2De4 = ReLU(name='En2--->De4_ReLU')(En2De4)

    En2De5 = MaxPooling3D(pool_size=(8,8,1), name='En2--->De5_MaxPool3D')(x)
    En2De5 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En2--->De5_Conv3D')(En2De5)
    En2De5 = BatchNormalization(name='En2--->De5_BatchNorm')(En2De5)
    En2De5 = ReLU(name='En2--->De5_ReLU')(En2De5)

    """ Encoder 3 """
    x = MaxPooling3D(pool_size=(2,2,1), name='En2->En3_MaxPool3D')(x)
    x = Conv3D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En3_Conv3D_1')(x)
    x = BatchNormalization(name='En3_BatchNorm_1')(x)
    x = ReLU(name='En3_ReLU_1')(x)
    x = Conv3D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En3_Conv3D_2')(x)
    x = BatchNormalization(name='En3_BatchNorm_2')(x)
    x = ReLU(name='En3_ReLU_2')(x)
    x = Conv3D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En3_Conv3D_3')(x)
    x = BatchNormalization(name='En3_BatchNorm_3')(x)
    x = ReLU(name='En3_ReLU_3')(x)
    x = Conv3D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En3_Conv3D_4')(x)
    x = BatchNormalization(name='En3_BatchNorm_4')(x)
    x = ReLU(name='En3_ReLU_4')(x)
    x = Conv3D(filters=filter_num_down[2], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En3_Conv3D_5')(x)
    x = BatchNormalization(name='En3_BatchNorm_5')(x)
    x = ReLU(name='En3_ReLU_5')(x)
    # Skip connections #
    En3De3 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En3--->De3_Conv3D')(x)
    En3De3 = BatchNormalization(name='En3--->De3_BatchNorm')(En3De3)
    En3De3 = ReLU(name='En3--->De3_ReLU')(En3De3)

    En3De4 = MaxPooling3D(pool_size=(2,2,1), name='En3--->De4_MaxPool3D')(x)
    En3De4 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En3--->De4_Conv3D')(En3De4)
    En3De4 = BatchNormalization(name='En3--->De4_BatchNorm')(En3De4)
    En3De4 = ReLU(name='En3--->De4_ReLU')(En3De4)

    En3De5 = MaxPooling3D(pool_size=(4,4,1), name='En3--->De5_MaxPool3D')(x)
    En3De5 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En3--->De5_Conv3D')(En3De5)
    En3De5 = BatchNormalization(name='En3--->De5_BatchNorm')(En3De5)
    En3De5 = ReLU(name='En3--->De5_ReLU')(En3De5)

    # Encoder 4 #
    x = MaxPooling3D(pool_size=(2,2,1), name='En3->En4_MaxPool3D')(x)
    x = Conv3D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En4_Conv3D_1')(x)
    x = BatchNormalization(name='En4_BatchNorm_1')(x)
    x = ReLU(name='En4_ReLU_1')(x)
    x = Conv3D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En4_Conv3D_2')(x)
    x = BatchNormalization(name='En4_BatchNorm_2')(x)
    x = ReLU(name='En4_ReLU_2')(x)
    x = Conv3D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En4_Conv3D_3')(x)
    x = BatchNormalization(name='En4_BatchNorm_3')(x)
    x = ReLU(name='En4_ReLU_3')(x)
    x = Conv3D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En4_Conv3D_4')(x)
    x = BatchNormalization(name='En4_BatchNorm_4')(x)
    x = ReLU(name='En4_ReLU_4')(x)
    x = Conv3D(filters=filter_num_down[3], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En4_Conv3D_5')(x)
    x = BatchNormalization(name='En4_BatchNorm_5')(x)
    x = ReLU(name='En4_ReLU_5')(x)
    # Skip connections #
    En4De4 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En4--->De4_Conv3D')(x)
    En4De4 = BatchNormalization(name='En4--->De4_BatchNorm')(En4De4)
    En4De4 = ReLU(name='En4--->De4_ReLU')(En4De4)

    En4De5 = MaxPooling3D(pool_size=(2,2,1), name='En4--->De5_MaxPool3D')(x)
    En4De5 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En4--->De5_Conv3D')(En4De5)
    En4De5 = BatchNormalization(name='En4--->De5_BatchNorm')(En4De5)
    En4De5 = ReLU(name='En4--->De5_ReLU')(En4De5)

    # Encoder 5 #
    x = MaxPooling3D(pool_size=(2,2,1), name='En4->En5_MaxPool3D')(x)
    x = Conv3D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En5_Conv3D_1')(x)
    x = BatchNormalization(name='En5_BatchNorm_1')(x)
    x = ReLU(name='En5_ReLU_1')(x)
    x = Conv3D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En5_Conv3D_2')(x)
    x = BatchNormalization(name='En5_BatchNorm_2')(x)
    x = ReLU(name='En5_ReLU_2')(x)
    x = Conv3D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En5_Conv3D_3')(x)
    x = BatchNormalization(name='En5_BatchNorm_3')(x)
    x = ReLU(name='En5_ReLU_3')(x)
    x = Conv3D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En5_Conv3D_4')(x)
    x = BatchNormalization(name='En5_BatchNorm_4')(x)
    x = ReLU(name='En5_ReLU_4')(x)
    x = Conv3D(filters=filter_num_down[4], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En5_Conv3D_5')(x)
    x = BatchNormalization(name='En5_BatchNorm_5')(x)
    x = ReLU(name='En5_ReLU_5')(x)
    # Skip connections #
    En5De5 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En5->De5_Conv3D')(x)
    En5De5 = BatchNormalization(name='En5->De5_BatchNorm')(En5De5)
    En5De5 = ReLU(name='En5->De5_ReLU')(En5De5)

    # Encoder 6 (bottom layer) #
    x = MaxPooling3D(pool_size=(2,2,1), name='En5->En6_MaxPool3D')(x)
    x = Conv3D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En6_Conv3D_1')(x)
    x = BatchNormalization(name='En6_BatchNorm_1')(x)
    x = ReLU(name='En6_ReLU_1')(x)
    x = Conv3D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En6_Conv3D_2')(x)
    x = BatchNormalization(name='En6_BatchNorm_2')(x)
    x = ReLU(name='En6_ReLU_2')(x)
    x = Conv3D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En6_Conv3D_3')(x)
    x = BatchNormalization(name='En6_BatchNorm_3')(x)
    x = ReLU(name='En6_ReLU_3')(x)
    x = Conv3D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En6_Conv3D_4')(x)
    x = BatchNormalization(name='En6_BatchNorm_4')(x)
    x = ReLU(name='En6_ReLU_4')(x)
    x = Conv3D(filters=filter_num_down[5], kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En6_Conv3D_5')(x)
    x = BatchNormalization(name='En6_BatchNorm_5')(x)
    x = ReLU(name='En6_ReLU_5')(x)
    x = UpSampling3D(size=(2,2,1), name='En6->De5_UpSample3D')(x)
    sup4_conv = Conv3D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, name='sup4_Conv3D')(x)
    sup4_upsample = UpSampling3D(size=(16,16,1), name='sup4_UpSample3D')(sup4_conv)
    sup4_output = Softmax(name='sup4_Softmax')(sup4_upsample)
    # Skip connections #
    En6De1 = UpSampling3D(size=(16,16,1), name='En6--->De1_UpSample3D')(x)
    En6De1 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En6--->De1_Conv3D')(En6De1)
    En6De1 = BatchNormalization(name='En6--->De1_BatchNorm')(En6De1)
    En6De1 = ReLU(name='En6--->De1_ReLU')(En6De1)

    En6De2 = UpSampling3D(size=(8,8,1), name='En6--->De2_UpSample3D')(x)
    En6De2 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En6--->De2_Conv3D')(En6De2)
    En6De2 = BatchNormalization(name='En6--->De2_BatchNorm')(En6De2)
    En6De2 = ReLU(name='En6--->De2_ReLU')(En6De2)

    En6De3 = UpSampling3D(size=(4,4,1), name='En6--->De3_UpSample3D')(x)
    En6De3 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En6--->De3_Conv3D')(En6De3)
    En6De3 = BatchNormalization(name='En6--->De3_BatchNorm')(En6De3)
    En6De3 = ReLU(name='En6--->De3_ReLU')(En6De3)

    En6De4 = UpSampling3D(size=(2,2,1), name='En6--->De4_UpSample3D')(x)
    En6De4 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En6--->De4_Conv3D')(En6De4)
    En6De4 = BatchNormalization(name='En6--->De4_BatchNorm')(En6De4)
    En6De4 = ReLU(name='En6--->De4_ReLU')(En6De4)

    # Decoder 5 #
    x = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='En6->De5_Conv3D')(x)
    x = BatchNormalization(name='En6->De5_BatchNorm')(x)
    x = ReLU(name='En6->De5_ReLU')(x)
    x = Concatenate(name='De5_Concatenate')([x, En5De5, En4De5, En3De5, En2De5])
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De5_Conv3D_1')(x)
    x = BatchNormalization(name='De5_BatchNorm_1')(x)
    x = ReLU(name='De5_ReLU_1')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De5_Conv3D_2')(x)
    x = BatchNormalization(name='De5_BatchNorm_2')(x)
    x = ReLU(name='De5_ReLU_2')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De5_Conv3D_3')(x)
    x = BatchNormalization(name='De5_BatchNorm_3')(x)
    x = ReLU(name='De5_ReLU_3')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De5_Conv3D_4')(x)
    x = BatchNormalization(name='De5_BatchNorm_4')(x)
    x = ReLU(name='De5_ReLU_4')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De5_Conv3D_5')(x)
    x = BatchNormalization(name='De5_BatchNorm_5')(x)
    x = ReLU(name='De5_ReLU_5')(x)

    x = UpSampling3D(size=(2,2,1), name='De5->De4_UpSample3D')(x)
    sup3_conv = Conv3D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, name='sup3_Conv3D')(x)
    sup3_upsample = UpSampling3D(size=(8,8,1), name='sup3_UpSample3D')(sup3_conv)
    sup3_output = Softmax(name='sup3_Softmax')(sup3_upsample)

    # Skip connections #
    De5De1 = UpSampling3D(size=(8,8,1), name='De5--->De1_UpSample3D')(x)
    De5De1 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De5--->De1_Conv3D')(De5De1)
    De5De1 = BatchNormalization(name='De5--->De1_BatchNorm')(De5De1)
    De5De1 = ReLU(name='De5--->De1_ReLU')(De5De1)

    De5De2 = UpSampling3D(size=(4,4,1), name='De5--->De2_UpSample3D')(x)
    De5De2 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De5--->De2_Conv3D')(De5De2)
    De5De2 = BatchNormalization(name='De5--->De2_BatchNorm')(De5De2)
    De5De2 = ReLU(name='De5--->De2_ReLU')(De5De2)

    De5De3 = UpSampling3D(size=(2,2,1), name='De5--->De3_UpSample3D')(x)
    De5De3 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De5--->De3_Conv3D')(De5De3)
    De5De3 = BatchNormalization(name='De5--->De3_BatchNorm')(De5De3)
    De5De3 = ReLU(name='De5--->De3_ReLU')(De5De3)

    # Decoder 4 #
    x = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De5->De4_Conv3D')(x)
    x = BatchNormalization(name='De5->De4_BatchNorm')(x)
    x = ReLU(name='De5->De4_ReLU')(x)
    x = Concatenate(name='De4_Concatenate')([En6De4, x, En4De4, En3De4, En2De4])
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De4_Conv3D_1')(x)
    x = BatchNormalization(name='De4_BatchNorm_1')(x)
    x = ReLU(name='De4_ReLU_1')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De4_Conv3D_2')(x)
    x = BatchNormalization(name='De4_BatchNorm_2')(x)
    x = ReLU(name='De4_ReLU_2')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De4_Conv3D_3')(x)
    x = BatchNormalization(name='De4_BatchNorm_3')(x)
    x = ReLU(name='De4_ReLU_3')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De4_Conv3D_4')(x)
    x = BatchNormalization(name='De4_BatchNorm_4')(x)
    x = ReLU(name='De4_ReLU_4')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De4_Conv3D_5')(x)
    x = BatchNormalization(name='De4_BatchNorm_5')(x)
    x = ReLU(name='De4_ReLU_5')(x)

    x = UpSampling3D(size=(2,2,1), name='De4->De3_UpSample3D')(x)
    sup2_conv = Conv3D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, name='sup2_Conv3D')(x)
    sup2_upsample = UpSampling3D(size=(4,4,1), name='sup2_UpSample3D')(sup2_conv)
    sup2_output = Softmax(name='sup2_Softmax')(sup2_upsample)

    # Skip connection #
    De4De1 = UpSampling3D(size=(4,4,1), name='De4--->De1_UpSample3D')(x)
    De4De1 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De4--->De1_Conv3D')(De4De1)
    De4De1 = BatchNormalization(name='De4--->De1_BatchNorm')(De4De1)
    De4De1 = ReLU(name='De4--->De1_ReLU')(De4De1)

    De4De2 = UpSampling3D(size=(2,2,1), name='De4--->De2_UpSample3D')(x)
    De4De2 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De4--->De2_Conv3D')(De4De2)
    De4De2 = BatchNormalization(name='De4--->De2_BatchNorm')(De4De2)
    De4De2 = ReLU(name='De4--->De2_ReLU')(De4De2)

    # Decoder 3 #
    x = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De4->De3_Conv3D')(x)
    x = BatchNormalization(name='De4->De3_BatchNorm')(x)
    x = ReLU(name='De4->De3_ReLU')(x)
    x = Concatenate(name='De3_Concatenate')([En6De3, De5De3, x, En3De3, En2De3])
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De3_Conv3D_1')(x)
    x = BatchNormalization(name='De3_BatchNorm_1')(x)
    x = ReLU(name='De3_ReLU_1')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De3_Conv3D_2')(x)
    x = BatchNormalization(name='De3_BatchNorm_2')(x)
    x = ReLU(name='De3_ReLU_2')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De3_Conv3D_3')(x)
    x = BatchNormalization(name='De3_BatchNorm_3')(x)
    x = ReLU(name='De3_ReLU_3')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De3_Conv3D_4')(x)
    x = BatchNormalization(name='De3_BatchNorm_4')(x)
    x = ReLU(name='De3_ReLU_4')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De3_Conv3D_5')(x)
    x = BatchNormalization(name='De3_BatchNorm_5')(x)
    x = ReLU(name='De3_ReLU_5')(x)

    x = UpSampling3D(size=(2,2,1), name='De3->De2_UpSample3D')(x)
    sup1_conv = Conv3D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, name='sup1_Conv3D')(x)
    sup1_upsample = UpSampling3D(size=(2,2,1), name='sup1_UpSample3D')(sup1_conv)
    sup1_output = Softmax(name='sup1_Softmax')(sup1_upsample)

    # Skip connection #
    De3De1 = UpSampling3D(size=(2,2,1), name='De3--->De1_UpSample3D')(x)
    De3De1 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De3--->De1_Conv3D')(De3De1)
    De3De1 = BatchNormalization(name='De3--->De1_BatchNorm')(De3De1)
    De3De1 = ReLU(name='De3--->De1_ReLU')(De3De1)

    # Decoder 2 #
    x = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De3->De2_Conv3D')(x)
    x = BatchNormalization(name='De3->De2_BatchNorm')(x)
    x = ReLU(name='De3->De2_ReLU')(x)
    x = Concatenate(name='De2_Concatenate')([En6De2, De5De2, De4De2, x, En2De2])
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De2_Conv3D_1')(x)
    x = BatchNormalization(name='De2_BatchNorm_1')(x)
    x = ReLU(name='De2_ReLU_1')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De2_Conv3D_2')(x)
    x = BatchNormalization(name='De2_BatchNorm_2')(x)
    x = ReLU(name='De2_ReLU_2')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De2_Conv3D_3')(x)
    x = BatchNormalization(name='De2_BatchNorm_3')(x)
    x = ReLU(name='De2_ReLU_3')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De2_Conv3D_4')(x)
    x = BatchNormalization(name='De2_BatchNorm_4')(x)
    x = ReLU(name='De2_ReLU_4')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De2_Conv3D_5')(x)
    x = BatchNormalization(name='De2_BatchNorm_5')(x)
    x = ReLU(name='De2_ReLU_5')(x)

    x = UpSampling3D(size=(2,2,1), name='De2->De1_UpSample3D')(x)
    sup0_conv = Conv3D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, name='sup0_Conv3D')(x)
    sup0_output = Softmax(name='sup0_Softmax')(sup0_conv)

    # Decoder 1 #
    x = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De2->De1_Conv3D')(x)
    x = BatchNormalization(name='De2->De1_BatchNorm')(x)
    x = ReLU(name='De2->De1_ReLU')(x)
    x = Concatenate(name='De1_Concatenate')([En6De1, De5De1, De4De1, De3De1, x])
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De1_Conv3D_1')(x)
    x = BatchNormalization(name='De1_BatchNorm_1')(x)
    x = ReLU(name='De1_ReLU_1')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De1_Conv3D_2')(x)
    x = BatchNormalization(name='De1_BatchNorm_2')(x)
    x = ReLU(name='De1_ReLU_2')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De1_Conv3D_3')(x)
    x = BatchNormalization(name='De1_BatchNorm_3')(x)
    x = ReLU(name='De1_ReLU_3')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De1_Conv3D_4')(x)
    x = BatchNormalization(name='De1_BatchNorm_4')(x)
    x = ReLU(name='De1_ReLU_4')(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=False, name='De1_Conv3D_5')(x)
    x = BatchNormalization(name='De1_BatchNorm_5')(x)
    x = ReLU(name='De1_ReLU_5')(x)

    x = Conv3D(filters=num_classes, kernel_size=kernel_size, dilation_rate=1, padding='same', use_bias=True, name='final_Conv3D')(x)
    final_output = Softmax(name='final_Softmax')(x)
    model = keras.Model(inputs=inputs, outputs=[final_output,sup0_output,sup1_output,sup2_output,sup3_output,sup4_output], name='3plus3D')

    return model
