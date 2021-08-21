import keras
from keras.layers import Conv3D, BatchNormalization, MaxPooling3D, PReLU, Concatenate, Input, UpSampling3D, Softmax

def UNet_3plus_3d(map_dim_x, map_dim_y, num_classes):
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
    filter_num_aggregate = 112
    kernel_size = (3,3,3)
    strides = (1,1,1)
    
    inputs = Input(shape=(map_dim_x, map_dim_y, 5, 12))

    """ Encoder 1 """
    x = Conv3D(filters=filter_num_down[0], kernel_size=kernel_size, padding='same', strides=strides, data_format='channels_last')(inputs)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[0], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[0], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[0], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[0], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    # Skip connections #
    En1De1 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(x)
    En1De1 = BatchNormalization()(En1De1)
    En1De1 = PReLU()(En1De1)

    En1De2 = MaxPooling3D(pool_size=(2,2,1), padding='same')(x)
    En1De2 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(En1De2)
    En1De2 = BatchNormalization()(En1De2)
    En1De2 = PReLU()(En1De2)

    En1De3 = MaxPooling3D(pool_size=(4,4,1), padding='same')(x)
    En1De3 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(En1De3)
    En1De3 = BatchNormalization()(En1De3)
    En1De3 = PReLU()(En1De3)

    En1De4 = MaxPooling3D(pool_size=(8,8,1), padding='same')(x)
    En1De4 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(En1De4)
    En1De4 = BatchNormalization()(En1De4)
    En1De4 = PReLU()(En1De4)

    En1De5 = MaxPooling3D(pool_size=(16,16,1), padding='same')(x)
    En1De5 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(En1De5)
    En1De5 = BatchNormalization()(En1De5)
    En1De5 = PReLU()(En1De5)

    """ Encoder 2 """
    x = MaxPooling3D(pool_size=(2,2,1), padding='same')(x)
    x = Conv3D(filters=filter_num_down[1], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[1], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[1], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[1], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[1], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    # Skip connections #
    En2De2 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(x)
    En2De2 = BatchNormalization()(En2De2)
    En2De2 = PReLU()(En2De2)

    En2De3 = MaxPooling3D(pool_size=(2,2,1), padding='same')(x)
    En2De3 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(En2De3)
    En2De3 = BatchNormalization()(En2De3)
    En2De3 = PReLU()(En2De3)

    En2De4 = MaxPooling3D(pool_size=(4,4,1), padding='same')(x)
    En2De4 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(En2De4)
    En2De4 = BatchNormalization()(En2De4)
    En2De4 = PReLU()(En2De4)

    En2De5 = MaxPooling3D(pool_size=(8,8,1), padding='same')(x)
    En2De5 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(En2De5)
    En2De5 = BatchNormalization()(En2De5)
    En2De5 = PReLU()(En2De5)

    """ Encoder 3 """
    x = MaxPooling3D(pool_size=(2,2,1), padding='same')(x)
    x = Conv3D(filters=filter_num_down[2], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[2], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[2], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[2], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[2], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    # Skip connections #
    print(x)
    En3De3 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(x)
    En3De3 = BatchNormalization()(En3De3)
    En3De3 = PReLU()(En3De3)
    print(En3De3)

    En3De4 = MaxPooling3D(pool_size=(2,2,1), padding='same')(x)
    En3De4 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(En3De4)
    En3De4 = BatchNormalization()(En3De4)
    En3De4 = PReLU()(En3De4)

    En3De5 = MaxPooling3D(pool_size=(4,4,1), padding='same')(x)
    En3De5 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(En3De5)
    En3De5 = BatchNormalization()(En3De5)
    En3De5 = PReLU()(En3De5)

    # Encoder 4 #
    x = MaxPooling3D(pool_size=(2,2,1), padding='same')(x)
    x = Conv3D(filters=filter_num_down[3], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[3], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[3], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[3], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[3], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    # Skip connections #
    En4De4 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(x)
    En4De4 = BatchNormalization()(En4De4)
    En4De4 = PReLU()(En4De4)

    En4De5 = MaxPooling3D(pool_size=(2,2,1), padding='same')(x)
    En4De5 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(En4De5)
    En4De5 = BatchNormalization()(En4De5)
    En4De5 = PReLU()(En4De5)

    # Encoder 5 #
    x = MaxPooling3D(pool_size=(2,2,1), padding='same')(x)
    x = Conv3D(filters=filter_num_down[4], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[4], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[4], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[4], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[4], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    # Skip connections #
    En5De5 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(x)
    En5De5 = BatchNormalization()(En5De5)
    En5De5 = PReLU()(En5De5)

    # Encoder 6 (bottom layer) #
    x = MaxPooling3D(pool_size=(2,2,1), padding='same')(x)
    x = Conv3D(filters=filter_num_down[5], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[5], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[5], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[5], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_down[5], kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = UpSampling3D(size=(2,2,1))(x)
    sup4_conv = Conv3D(filters=num_classes, kernel_size=kernel_size, padding='same', strides=strides)(x)
    sup4_upsample = UpSampling3D(size=(16,16,1))(sup4_conv)
    sup4_output = Softmax()(sup4_upsample)
    # Skip connections #
    En6De1 = UpSampling3D(size=(16,16,1))(x)
    En6De1 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(En6De1)
    En6De1 = BatchNormalization()(En6De1)
    En6De1 = PReLU()(En6De1)

    En6De2 = UpSampling3D(size=(8,8,1))(x)
    En6De2 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(En6De2)
    En6De2 = BatchNormalization()(En6De2)
    En6De2 = PReLU()(En6De2)

    En6De3 = UpSampling3D(size=(4,4,1))(x)
    En6De3 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(En6De3)
    En6De3 = BatchNormalization()(En6De3)
    En6De3 = PReLU()(En6De3)

    En6De4 = UpSampling3D(size=(2,2,1))(x)
    En6De4 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(En6De4)
    En6De4 = BatchNormalization()(En6De4)
    En6De4 = PReLU()(En6De4)

    # Decoder 5 #
    x = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Concatenate()([x, En5De5, En4De5, En3De5, En2De5, En1De5])
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = UpSampling3D(size=(2,2,1))(x)
    sup3_conv = Conv3D(filters=num_classes, kernel_size=kernel_size, padding='same', strides=strides)(x)
    sup3_upsample = UpSampling3D(size=(8,8,1))(sup3_conv)
    sup3_output = Softmax()(sup3_upsample)
    # Skip connections #
    De5De1 = UpSampling3D(size=(8,8,1))(x)
    De5De1 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(De5De1)
    De5De1 = BatchNormalization()(De5De1)
    De5De1 = PReLU()(De5De1)

    De5De2 = UpSampling3D(size=(4,4,1))(x)
    De5De2 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(De5De2)
    De5De2 = BatchNormalization()(De5De2)
    De5De2 = PReLU()(De5De2)

    De5De3 = UpSampling3D(size=(2,2,1))(x)
    De5De3 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(De5De3)
    De5De3 = BatchNormalization()(De5De3)
    De5De3 = PReLU()(De5De3)

    # Decoder 4 #
    x = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Concatenate()([En6De4, x, En4De4, En3De4, En2De4, En1De4])
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = UpSampling3D(size=(2,2,1))(x)
    sup2_conv = Conv3D(filters=num_classes, kernel_size=kernel_size, padding='same', strides=strides)(x)
    sup2_upsample = UpSampling3D(size=(4,4,1))(sup2_conv)
    sup2_output = Softmax()(sup2_upsample)
    # Skip connection #
    De4De1 = UpSampling3D(size=(4,4,1))(x)
    De4De1 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(De4De1)
    De4De1 = BatchNormalization()(De4De1)
    De4De1 = PReLU()(De4De1)

    De4De2 = UpSampling3D(size=(2,2,1))(x)
    De4De2 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(De4De2)
    De4De2 = BatchNormalization()(De4De2)
    De4De2 = PReLU()(De4De2)

    # Decoder 3 #
    x = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Concatenate()([En6De3, De5De3, x, En3De3, En2De3, En1De3])
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = UpSampling3D(size=(2,2,1))(x)
    sup1_conv = Conv3D(filters=num_classes, kernel_size=kernel_size, padding='same', strides=strides)(x)
    sup1_upsample = UpSampling3D(size=(2,2,1))(sup1_conv)
    sup1_output = Softmax()(sup1_upsample)
    # Skip connection #
    De3De1 = UpSampling3D(size=(2,2,1))(x)
    De3De1 = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(De3De1)
    De3De1 = BatchNormalization()(De3De1)
    De3De1 = PReLU()(De3De1)


    # Decoder 2 #
    x = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Concatenate()([En6De2, De5De2, De4De2, x, En2De2, En1De2])
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = UpSampling3D(size=(2,2,1))(x)
    sup0_conv = Conv3D(filters=num_classes, kernel_size=kernel_size, padding='same', strides=strides)(x)
    sup0_output = Softmax()(sup0_conv)

    # Decoder 1 #
    x = Conv3D(filters=filter_num_skip, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Concatenate()([En6De1, De5De1, De4De1, De3De1, x, En1De1])
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_num_aggregate, kernel_size=kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=num_classes, kernel_size=kernel_size, padding='same', strides=strides)(x)
    final_output = Softmax()(x)
    model = keras.Model(inputs=inputs, outputs=[final_output,sup0_output,sup1_output,sup2_output,sup3_output,sup4_output], name='3plus3d')

    return model

