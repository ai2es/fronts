import pandas as pd
import tensorflow as tf
import custom_losses
import keras
import file_manager as fm
import numpy as np
import random
from keras.layers import Conv2D, Conv3D, BatchNormalization, MaxPooling3D, PReLU, Concatenate, Input, UpSampling3D, Softmax

front_files, variable_files = fm.load_file_lists(60, 'CFWF', 'conus', file_dimensions=(289,129))

map_dim_x, map_dim_y, channels = 128, 128, 60
normalization_method = 1
file_dimensions = (289,129)

print(len(front_files), len(variable_files))

variable_ds = pd.read_pickle(variable_files[0])
variable_list = list(variable_ds.keys())

variable_list_sfc = variable_list[0:12]
variable_list_850 = variable_list[12:24]
variable_list_900 = variable_list[24:36]
variable_list_950 = variable_list[36:48]
variable_list_1000 = variable_list[48:60]

norm_params = pd.read_csv('normalization_parameters.csv', index_col='Variable')

lon_index = random.choices(range(file_dimensions[0] - map_dim_x))[0]
lat_index = random.choices(range(file_dimensions[1] - map_dim_y))[0]
lons = variable_ds.longitude.values[lon_index:lon_index + map_dim_x]
lats = variable_ds.latitude.values[lat_index:lat_index + map_dim_y]

for j in range(len(variable_list)):
    var = variable_list[j]
    if normalization_method == 1:
        # Min-max normalization
        variable_ds[var].values = np.nan_to_num((variable_ds[var].values - norm_params.loc[var,'Min']) /
                                                (norm_params.loc[var,'Max'] - norm_params.loc[var,'Min']))
    elif normalization_method == 2:
        # Mean normalization
        variable_ds[var].values = np.nan_to_num((variable_ds[var].values - norm_params.loc[var,'Mean']) /
                                                (norm_params.loc[var,'Max'] - norm_params.loc[var,'Min']))
variable_ds_new = np.nan_to_num(variable_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, map_dim_x,
    map_dim_y, len(variable_list)))

variable_ds_sfc = variable_ds_new[:,:,:,0:12]
variable_ds_1000 = variable_ds_new[:,:,:,48:60]
variable_ds_950 = variable_ds_new[:,:,:,36:48]
variable_ds_900 = variable_ds_new[:,:,:,24:36]
variable_ds_850 = variable_ds_new[:,:,:,12:24]

variable_ds_3d = np.array([variable_ds_sfc, variable_ds_1000, variable_ds_950, variable_ds_900, variable_ds_850]).reshape((1,128,128,5,12))

inputs = Input(shape=(128, 128, 5, 12))
filters = [64, 128, 256, 512, 1024, 2048]

""" Encoder 1 """
x = Conv3D(filters=filters[0], kernel_size=3, strides=1, data_format='channels_last')(inputs)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[0], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[0], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[0], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[0], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
# Skip Connections #
En1De1 = Conv3D(filters=64, kernel_size=3)(x)

En1De2 = MaxPooling3D(pool_size=(2,2,1))(x)
En1De2 = Conv3D(filters=64, kernel_size=3)(En1De2)

En1De3 = MaxPooling3D(pool_size=(4,4,1))(x)
En1De3 = Conv3D(filters=64, kernel_size=3)(En1De3)

En1De4 = MaxPooling3D(pool_size=(8,8,1))(x)
En1De4 = Conv3D(filters=64, kernel_size=3)(En1De4)

""" Encoder 2 """
x = MaxPooling3D(pool_size=(2,2,1))(x)
x = Conv3D(filters=filters[1], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[1], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[1], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[1], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[1], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
# Skip connections #
En2De2 = Conv3D(filters=64, kernel_size=3)(x)
En2De2 = BatchNormalization()(En2De2)
En2De2 = PReLU()(En2De2)

En2De3 = MaxPooling3D(pool_size=(2,2,1))(x)
En2De3 = Conv3D(filters=64, kernel_size=3)(En2De3)
En2De3 = BatchNormalization()(En2De3)
En2De3 = PReLU()(En2De3)

En2De4 = MaxPooling3D(pool_size=(4,4,1))(x)
En2De4 = Conv3D(filters=64, kernel_size=3)(En2De4)
En2De4 = BatchNormalization()(En2De4)
En2De4 = PReLU()(En2De4)

En2De5 = MaxPooling3D(pool_size=(8,8,1))(x)
En2De5 = Conv3D(filters=64, kernel_size=3)(En2De5)
En2De5 = BatchNormalization()(En2De5)
En2De5 = PReLU()(En2De5)

""" Encoder 3 """
x = MaxPooling3D(pool_size=(2,2,1))(x)
x = Conv3D(filters=filters[2], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[2], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[2], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[2], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[2], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
# Skip connections #
En3De3 = Conv3D(filters=64, kernel_size=3)(x)
En3De3 = BatchNormalization()(En3De3)
En3De3 = PReLU()(En3De3)

En3De4 = MaxPooling3D(pool_size=(2,2,1))(x)
En3De4 = Conv3D(filters=64, kernel_size=3)(En3De4)
En3De4 = BatchNormalization()(En3De4)
En3De4 = PReLU()(En3De4)

En3De5 = MaxPooling3D(pool_size=(4,4,1))(x)
En3De5 = Conv3D(filters=64, kernel_size=3)(En3De5)
En3De5 = BatchNormalization()(En3De5)
En3De5 = PReLU()(En3De5)

# Encoder 4 #
x = MaxPooling3D(pool_size=(2,2,1))(x)
x = Conv3D(filters=filters[3], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[3], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[3], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[3], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[3], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
# Skip connections #
En4De4 = Conv3D(filters=64, kernel_size=3)(x)
En4De4 = BatchNormalization()(En4De4)
En4De4 = PReLU()(En4De4)

En4De5 = MaxPooling3D(pool_size=(2,2,1))(x)
En4De5 = Conv3D(filters=64, kernel_size=3)(En4De5)
En4De5 = BatchNormalization()(En4De5)
En4De5 = PReLU()(En4De5)

# Encoder 5 #
x = MaxPooling3D(pool_size=(2,2,1))(x)
x = Conv3D(filters=filters[4], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[4], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[4], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[4], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[4], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
# Skip connections #
En5De5 = Conv3D(filters=64, kernel_size=3)(x)
En5De5 = BatchNormalization()(En5De5)
En5De5 = PReLU()(En5De5)

# Encoder 6 (bottom layer) #
x = MaxPooling3D(pool_size=(2,2,1))(x)
x = Conv3D(filters=filters[5], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[5], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[5], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[5], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=filters[5], kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = UpSampling3D(size=(2,2,1))(x)
sup4_conv = Conv3D(filters=3, kernel_size=3, strides=1)(x)
sup4_upsample = UpSampling3D(size=(16,16,1))(sup4_conv)
sup4_output = Softmax()(sup4_upsample)
# Skip connections #
En6De1 = UpSampling3D(size=(16,16,1))(x)
En6De1 = Conv3D(filters=64, kernel_size=3)(En6De1)
En6De1 = BatchNormalization()(En6De1)
En6De1 = PReLU()(En6De1)

En6De2 = UpSampling3D(size=(8,8,1))(x)
En6De2 = Conv3D(filters=64, kernel_size=3)(En6De2)
En6De2 = BatchNormalization()(En6De2)
En6De2 = PReLU()(En6De2)

En6De3 = UpSampling3D(size=(4,4,1))(x)
En6De3 = Conv3D(filters=64, kernel_size=3)(En6De3)
En6De3 = BatchNormalization()(En6De3)
En6De3 = PReLU()(En6De3)

En6De4 = UpSampling3D(size=(2,2,1))(x)
En6De4 = Conv3D(filters=64, kernel_size=3)(En6De4)
En6De4 = BatchNormalization()(En6De4)
En6De4 = PReLU()(En6De4)

# Decoder 5 #
x = Conv3D(filters=64, kernel_size=3)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Concatenate()([x,En5De5, En4De5, En3De5, En2De5])
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = UpSampling3D(size=(2,2,1))(x)
sup3_conv = Conv3D(filters=3, kernel_size=3, strides=1)(x)
sup3_upsample = UpSampling3D(size=(8,8,1))(sup3_conv)
sup3_output = Softmax()(sup3_upsample)
# Skip connections #
De5De1 = UpSampling3D(size=(8,8,1))(x)
De5De1 = Conv3D(filters=64, kernel_size=3)(De5De1)
De5De1 = BatchNormalization()(De5De1)
De5De1 = PReLU()(De5De1)

De5De2 = UpSampling3D(size=(4,4,1))(x)
De5De2 = Conv3D(filters=64, kernel_size=3)(De5De2)
De5De2 = BatchNormalization()(De5De2)
De5De2 = PReLU()(De5De2)

De5De3 = UpSampling3D(size=(2,2,1))(x)
De5De3 = Conv3D(filters=64, kernel_size=3)(De5De3)
De5De3 = BatchNormalization()(De5De3)
De5De3 = PReLU()(De5De3)

# Decoder 4 #
x = Conv3D(filters=64, kernel_size=3)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Concatenate()([En6De4,x,En4De4,En3De4,En2De4])
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = UpSampling3D(size=(2,2,1))(x)
sup2_conv = Conv3D(filters=3, kernel_size=3, strides=1)(x)
sup2_upsample = UpSampling3D(size=(4,4,1))(sup2_conv)
sup2_output = Softmax()(sup2_upsample)
# Skip connection #
De4De1 = UpSampling3D(size=(4,4,1))(x)
De4De1 = Conv3D(filters=64, kernel_size=3)(De4De1)
De4De1 = BatchNormalization()(De4De1)
De4De1 = PReLU()(De4De1)

De4De2 = UpSampling3D(size=(2,2,1))(x)
De4De2 = Conv3D(filters=64, kernel_size=3)(De4De2)
De4De2 = BatchNormalization()(De4De2)
De4De2 = PReLU()(De4De2)

# Decoder 3 #
x = Conv3D(filters=64, kernel_size=3)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Concatenate()([En6De3,De5De3,x,En3De3,En2De3])
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = UpSampling3D(size=(2,2,1))(x)
sup1_conv = Conv3D(filters=3, kernel_size=3, strides=1)(x)
sup1_upsample = UpSampling3D(size=(2,2,1))(sup1_conv)
sup1_output = Softmax()(sup1_upsample)
# Skip connection #
De3De1 = UpSampling3D(size=(2,2,1))(x)
De3De1 = Conv3D(filters=64, kernel_size=3)(De3De1)
De3De1 = BatchNormalization()(De3De1)
De3De1 = PReLU()(De3De1)


# Decoder 2 #
x = Conv3D(filters=64, kernel_size=3)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Concatenate()([En6De2,De5De2,De4De2,x,En2De2])
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = UpSampling3D(size=(2,2,1))(x)
sup0_conv = Conv3D(filters=3, kernel_size=3, strides=1)(x)
sup0_output = Softmax()(sup0_conv)

# Decoder 1 #
x = Conv3D(filters=64, kernel_size=3)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Concatenate()([En6De1,De5De1,De4De1,De3De1,x])
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=384, kernel_size=3, strides=1)(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Conv3D(filters=3, kernel_size=3, strides=1)(x)
final_output = Softmax()(x)

model = keras.Model(inputs=inputs, outputs=[final_output,sup0_output,sup1_output,sup2_output,sup3_output,sup4_output], name='3plus3d')

model.summary()
