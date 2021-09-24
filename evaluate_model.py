"""
Functions used for evaluating a U-Net model. The functions can be used to make predictions or plot learning curves.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 9/23/2021 8:20 PM CDT
"""

import random
import pandas as pd
import argparse
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import file_manager as fm
import Fronts_Aggregate_Plot as fplot
import xarray as xr
from errors import check_arguments
import pickle
import matplotlib as mpl
from expand_fronts import one_pixel_expansion as ope


def calculate_performance_stats(model_number, model_dir, num_variables, num_dimensions, front_types, domain, file_dimensions,
    test_years, normalization_method, loss, fss_mask_size, fss_c, pixel_expansion, metric, num_images, longitude_domain_length,
    image_trim):
    """
    Function that calculates the number of true positives, false positives, true negatives, and false negatives on a testing set.

    Parameters
    ----------
    model_number: int
        Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        Main directory for the models.
    normalization_method: int
        Normalization method for the data (described near the end of the script).
    loss: str
        Loss function for the Unet.
    fss_mask_size: int
        Size of the mask for the FSS loss function.
    fss_c: float
        C hyperparameter for the FSS loss' sigmoid function.
    front_types: str
        Fronts in the data.
    domain: str
        Domain which the front and variable files cover.
    file_dimensions: int (x2)
        Dimensions of the data files.
    test_years: list of ints
        Years for the test set used for calculating performance stats for cross-validation purposes.
    pixel_expansion: int
        Number of pixels to expand the fronts by in all directions.
    metric: str
        Metric used for evaluating the U-Net during training.
    num_dimensions: int
        Number of dimensions for the U-Net's convolutions, maxpooling, and upsampling.
    num_variables: int
        Number of variables in the datasets.
    num_images: int
        Number of images to stitch together for the final probability maps.
    longitude_domain_length: int
        Number of pixels in the longitude dimension of the final stitched map.
    image_trim: int
        Number of pixels to trim each image by in the longitude dimension before averaging the overlapping pixels.
    """

    if test_years is not None:
        front_files, variable_files = fm.load_test_files(num_variables, front_types, domain, file_dimensions, test_years)
    else:
        front_files, variable_files = fm.load_file_lists(num_variables, front_types, domain, file_dimensions)
    
    print("Front file count:", len(front_files))
    print("Variable file count:", len(variable_files))

    model = fm.load_model(model_number, model_dir, loss, fss_mask_size, fss_c, metric, num_dimensions)

    n = 0  # Counter for the number of down layers in the model
    map_dim_x = model.layers[0].input_shape[0][1]  # Longitudinal dimension of the U-Net images
    map_dim_y = model.layers[0].input_shape[0][2]  # Latitudinal dimension of the U-Net images
    if num_dimensions == 2:
        for layer in model.layers:
            if layer.__class__.__name__ == 'MaxPooling2D':
                n += 1
        n = int((n - 1)/2)
        channels = model.layers[0].input_shape[0][3]  # Number of variables used
    if num_dimensions == 3:
        for layer in model.layers:
            if layer.__class__.__name__ == 'MaxPooling3D':
                n += 1
        n = int((n - 1)/2)
        levels = model.layers[0].input_shape[0][3]  # Number of levels to the U-Net variables
        channels = model.layers[0].input_shape[0][4]

    norm_params = pd.read_csv('normalization_parameters.csv', index_col='Variable')

    tp_cold = np.zeros(shape=[4,100])
    fp_cold = np.zeros(shape=[4,100])
    tn_cold = np.zeros(shape=[4,100])
    fn_cold = np.zeros(shape=[4,100])

    tp_warm = np.zeros(shape=[4,100])
    fp_warm = np.zeros(shape=[4,100])
    tn_warm = np.zeros(shape=[4,100])
    fn_warm = np.zeros(shape=[4,100])

    tp_stationary = np.zeros(shape=[4,100])
    fp_stationary = np.zeros(shape=[4,100])
    tn_stationary = np.zeros(shape=[4,100])
    fn_stationary = np.zeros(shape=[4,100])

    tp_occluded = np.zeros(shape=[4,100])
    fp_occluded = np.zeros(shape=[4,100])
    tn_occluded = np.zeros(shape=[4,100])
    fn_occluded = np.zeros(shape=[4,100])

    tp_dryline = np.zeros(shape=[4,100])
    fp_dryline = np.zeros(shape=[4,100])
    tn_dryline = np.zeros(shape=[4,100])
    fn_dryline = np.zeros(shape=[4,100])

    model_longitude_length = map_dim_x
    raw_image_index_array = np.empty(shape=[num_images,2])
    longitude_domain_length_trim = longitude_domain_length - 2*image_trim
    image_spacing = int((longitude_domain_length - model_longitude_length)/(num_images-1))
    latitude_domain_length = 128

    for x in range(2):
        
        print("Prediction %d/%d....0/%d" % (x+1, len(front_files), num_images), end='\r')

        # Open random pair of files
        index = x
        fronts_filename = front_files[index]
        variables_filename = variable_files[index]
        fronts_ds = pd.read_pickle(fronts_filename)
        for i in range(pixel_expansion):
            fronts_ds = ope(fronts_ds)

        for i in range(num_images):
            if i == 0:
                raw_image_index_array[i][0] = image_trim
                raw_image_index_array[i][1] = model_longitude_length - image_trim
            elif i != num_images - 1:
                raw_image_index_array[i][0] = raw_image_index_array[i-1][0] + image_spacing
                raw_image_index_array[i][1] = raw_image_index_array[i-1][1] + image_spacing
            else:
                raw_image_index_array[i][0] = raw_image_index_array[i-1][0] + image_spacing
                raw_image_index_array[i][1] = raw_image_index_array[i-1][1] + image_spacing

        lon_pixels_per_image = int(raw_image_index_array[0][1] - raw_image_index_array[0][0])

        image_no_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_cold_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_warm_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_stationary_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_occluded_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_dryline_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_lats = fronts_ds.latitude.values[0:128]
        image_lons = fronts_ds.longitude.values[image_trim:longitude_domain_length-image_trim]

        for image in range(num_images):
            
            print("Prediction %d/%d....%d/%d" % (x+1, len(front_files), image+1, num_images), end='\r')
            lat_index = 0
            variable_ds = pd.read_pickle(variables_filename)
            if image == 0:
                lon_index = 0
            else:
                lon_index = int(image*image_spacing)

            lons = variable_ds.longitude.values[lon_index:lon_index + map_dim_x]
            lats = variable_ds.latitude.values[lat_index:lat_index + map_dim_y]

            variable_list = list(variable_ds.keys())
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

            if num_dimensions == 2:
                variable_ds_new = np.nan_to_num(variable_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, map_dim_x,
                    map_dim_y, channels))
            elif num_dimensions == 3:
                variables_sfc = variable_ds[['t2m','d2m','sp','u10','v10','theta_w','mix_ratio','rel_humid','virt_temp','wet_bulb','theta_e',
                                             'q']].sel(longitude=lons, latitude=lats).to_array().T.values
                variables_1000 = variable_ds[['t_1000','d_1000','z_1000','u_1000','v_1000','theta_w_1000','mix_ratio_1000','rel_humid_1000','virt_temp_1000',
                                              'wet_bulb_1000','theta_e_1000','q_1000']].sel(longitude=lons, latitude=lats).to_array().T.values
                variables_950 = variable_ds[['t_950','d_950','z_950','u_950','v_950','theta_w_950','mix_ratio_950','rel_humid_950','virt_temp_950',
                                             'wet_bulb_950','theta_e_950','q_950']].sel(longitude=lons, latitude=lats).to_array().T.values
                variables_900 = variable_ds[['t_900','d_900','z_900','u_900','v_900','theta_w_900','mix_ratio_900','rel_humid_900','virt_temp_900',
                                             'wet_bulb_900','theta_e_900','q_900']].sel(longitude=lons, latitude=lats).to_array().T.values
                variables_850 = variable_ds[['t_850','d_850','z_850','u_850','v_850','theta_w_850','mix_ratio_850','rel_humid_850','virt_temp_850',
                                             'wet_bulb_850','theta_e_850','q_850']].sel(longitude=lons, latitude=lats).to_array().T.values
                variable_ds_new = np.array([variables_sfc,variables_1000,variables_950,variables_900,variables_850]).reshape(1,map_dim_x,map_dim_y,levels,channels)

            prediction = model.predict(variable_ds_new)

            # time = str(fronts.time.values)[0:13].replace('T', '-') + 'z'
            # print(time)

            # Arrays of probabilities for all front types
            no_probs = np.zeros([map_dim_x, map_dim_y])
            cold_probs = np.zeros([map_dim_x, map_dim_y])
            warm_probs = np.zeros([map_dim_x, map_dim_y])
            stationary_probs = np.zeros([map_dim_x, map_dim_y])
            occluded_probs = np.zeros([map_dim_x, map_dim_y])
            dryline_probs = np.zeros([map_dim_x, map_dim_y])

            thresholds = np.linspace(0.01,1,100)
            boundaries = np.array([25,50,75,100])

            if front_types == 'CFWF':
                if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':
                    print(model.name, "n =", n)
                    for i in range(0, map_dim_x):
                        for j in range(0, map_dim_y):
                            no_probs[i][j] = prediction[n][0][i][j][0]
                            cold_probs[i][j] = prediction[n][0][i][j][1]
                            warm_probs[i][j] = prediction[n][0][i][j][2]
                if model.name == '3plus3D':
                    l = 0  # Level index (0 = surface, 1 = 1000mb, 2 = 950mb, 3 = 900mb, 4 = 850mb)
                    for i in range(0, map_dim_x):
                        for j in range(0, map_dim_y):
                            no_probs[i][j] = prediction[n][0][i][j][l][0]
                            cold_probs[i][j] = prediction[n][0][i][j][l][1]
                            warm_probs[i][j] = prediction[n][0][i][j][l][2]
                if image == 0:
                    image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                    image_cold_probs[0: model_longitude_length - image_trim][:] = cold_probs[image_trim: model_longitude_length][:]
                    image_warm_probs[0: model_longitude_length - image_trim][:] = warm_probs[image_trim: model_longitude_length][:]
                elif image != num_images - 1:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                else:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    probs_ds = xr.Dataset(
                        {"no_probs": (("longitude", "latitude"), image_no_probs), "cold_probs": (("longitude", "latitude"), image_cold_probs),
                         "warm_probs": (("longitude", "latitude"), image_warm_probs)}, coords={"latitude": image_lats, "longitude": image_lons}).transpose()

                    for boundary in range(4):
                        fronts = pd.read_pickle(fronts_filename)
                        for y in range(boundary+1):
                            fronts = ope(fronts)

                        new_fronts = fronts
                        t_cold_ds = xr.where(new_fronts == 1, 1, 0)
                        t_cold_probs = t_cold_ds.identifier * probs_ds.cold_probs
                        new_fronts = fronts
                        f_cold_ds = xr.where(new_fronts == 1, 0, 1)
                        f_cold_probs = f_cold_ds.identifier * probs_ds.cold_probs
                        new_fronts = fronts

                        t_warm_ds = xr.where(new_fronts == 2, 1, 0)
                        t_warm_probs = t_warm_ds.identifier * probs_ds.warm_probs
                        new_fronts = fronts
                        f_warm_ds = xr.where(new_fronts == 2, 0, 1)
                        f_warm_probs = f_warm_ds.identifier * probs_ds.warm_probs
                        new_fronts = fronts

                        for i in range(100):
                            tp_cold[boundary,i] += len(np.where(t_cold_probs > thresholds[i])[0])
                            tn_cold[boundary,i] += len(np.where((f_cold_probs < thresholds[i]) & (f_cold_probs != 0))[0])
                            fp_cold[boundary,i] += len(np.where(f_cold_probs > thresholds[i])[0])
                            fn_cold[boundary,i] += len(np.where((t_cold_probs < thresholds[i]) & (t_cold_probs != 0))[0])
                            tp_warm[boundary,i] += len(np.where(t_warm_probs > thresholds[i])[0])
                            tn_warm[boundary,i] += len(np.where((f_warm_probs < thresholds[i]) & (f_warm_probs != 0))[0])
                            fp_warm[boundary,i] += len(np.where(f_warm_probs > thresholds[i])[0])
                            fn_warm[boundary,i] += len(np.where((t_warm_probs < thresholds[i]) & (t_warm_probs != 0))[0])
                    
                    print("Prediction %d/%d....done" % (x+1, len(front_files)))

            elif front_types == 'SFOF':
                if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':
                    for i in range(0, map_dim_x):
                        for j in range(0, map_dim_y):
                            no_probs[i][j] = prediction[n][0][i][j][0]
                            stationary_probs[i][j] = prediction[n][0][i][j][1]
                            occluded_probs[i][j] = prediction[n][0][i][j][2]
                if model.name == '3plus3D':
                    l = 0  # Level index (0 = surface, 1 = 1000mb, 2 = 950mb, 3 = 900mb, 4 = 850mb)
                    for i in range(0, map_dim_x):
                        for j in range(0, map_dim_y):
                            no_probs[i][j] = prediction[n][0][i][j][l][0]
                            stationary_probs[i][j] = prediction[n][0][i][j][l][1]
                            occluded_probs[i][j] = prediction[n][0][i][j][l][2]
                if image == 0:
                    image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                    image_stationary_probs[0: model_longitude_length - image_trim][:] = stationary_probs[image_trim: model_longitude_length][:]
                    image_occluded_probs[0: model_longitude_length - image_trim][:] = occluded_probs[image_trim: model_longitude_length][:]
                elif image != num_images - 1:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                else:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    probs_ds = xr.Dataset(
                        {"no_probs": (("longitude", "latitude"), image_no_probs), "stationary_probs": (("longitude", "latitude"), image_stationary_probs),
                         "occluded_probs": (("longitude", "latitude"), image_occluded_probs)}, coords={"latitude": image_lats, "longitude": image_lons})

                    for boundary in range(4):
                        fronts = pd.read_pickle(fronts_filename)
                        for y in range(boundary+1):
                            fronts = ope(fronts)

                        t_stationary_ds = xr.where(new_fronts == 3, 1, 0)
                        t_stationary_probs = t_stationary_ds.identifier * probs_ds.stationary_probs
                        new_fronts = fronts
                        f_stationary_ds = xr.where(new_fronts == 3, 0, 1)
                        f_stationary_probs = f_stationary_ds.identifier * probs_ds.stationary_probs
                        new_fronts = fronts

                        t_occluded_ds = xr.where(new_fronts == 4, 1, 0)
                        t_occluded_probs = t_occluded_ds.identifier * probs_ds.occluded_probs
                        new_fronts = fronts
                        f_occluded_ds = xr.where(new_fronts == 4, 0, 1)
                        f_occluded_probs = f_occluded_ds.identifier * probs_ds.occluded_probs

                        for i in range(100):
                            tp_stationary[boundary,i] += len(np.where(t_stationary_probs > thresholds[i])[0])
                            tn_stationary[boundary,i] += len(np.where((f_stationary_probs < thresholds[i]) & (f_stationary_probs != 0))[0])
                            fp_stationary[boundary,i] += len(np.where(f_stationary_probs > thresholds[i])[0])
                            fn_stationary[boundary,i] += len(np.where((t_stationary_probs < thresholds[i]) & (t_stationary_probs != 0))[0])
                            tp_occluded[boundary,i] += len(np.where(t_occluded_probs > thresholds[i])[0])
                            tn_occluded[boundary,i] += len(np.where((f_occluded_probs < thresholds[i]) & (f_occluded_probs != 0))[0])
                            fp_occluded[boundary,i] += len(np.where(f_occluded_probs > thresholds[i])[0])
                            fn_occluded[boundary,i] += len(np.where((t_occluded_probs < thresholds[i]) & (t_occluded_probs != 0))[0])
                    
                    print("Prediction %d/%d....done" % (x+1, len(front_files)))

            elif front_types == 'DL':
                if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':
                    for i in range(0, map_dim_x):
                        for j in range(0, map_dim_y):
                            no_probs[i][j] = prediction[n][0][i][j][0]
                            dryline_probs[i][j] = prediction[n][0][i][j][1]
                if model.name == '3plus3D':
                    l = 0  # Level index (0 = surface, 1 = 1000mb, 2 = 950mb, 3 = 900mb, 4 = 850mb)
                    for i in range(0, map_dim_x):
                        for j in range(0, map_dim_y):
                            no_probs[i][j] = prediction[n][0][i][j][l][0]
                            dryline_probs[i][j] = prediction[n][0][i][j][l][1]
                if image == 0:
                    image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                    image_dryline_probs[0: model_longitude_length - image_trim][:] = dryline_probs[image_trim: model_longitude_length][:]
                elif image != num_images - 1:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                else:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    probs_ds = xr.Dataset(
                        {"no_probs": (("latitude", "longitude"), no_probs), "dryline_probs": (("latitude", "longitude"), image_dryline_probs)},
                        coords={"latitude": lats, "longitude": lons})

                    for boundary in range(4):
                        fronts = pd.read_pickle(fronts_filename)
                        for y in range(boundary+1):
                            fronts = ope(fronts)

                        t_dryline_ds = xr.where(new_fronts == 5, 1, 0)
                        t_dryline_probs = t_dryline_ds.identifier * probs_ds.dryline_probs
                        new_fronts = fronts
                        f_dryline_ds = xr.where(new_fronts == 5, 0, 1)
                        f_dryline_probs = f_dryline_ds.identifier * probs_ds.dryline_probs

                        for i in range(100):
                            tp_dryline[boundary,i] += len(np.where(t_dryline_probs > thresholds[i])[0])
                            tn_dryline[boundary,i] += len(np.where((f_dryline_probs < thresholds[i]) & (f_dryline_probs != 0))[0])
                            fp_dryline[boundary,i] += len(np.where(f_dryline_probs > thresholds[i])[0])
                            fn_dryline[boundary,i] += len(np.where((t_dryline_probs < thresholds[i]) & (t_dryline_probs != 0))[0])

                    print("Prediction %d/%d....done" % (x+1, len(front_files)))

            elif front_types == 'ALL':
                if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':
                    for i in range(0, map_dim_x):
                        for j in range(0, map_dim_y):
                            no_probs[i][j] = prediction[n][0][i][j][0]
                            cold_probs[i][j] = prediction[n][0][i][j][1]
                            warm_probs[i][j] = prediction[n][0][i][j][2]
                            stationary_probs[i][j] = prediction[n][0][i][j][3]
                            occluded_probs[i][j] = prediction[n][0][i][j][4]
                            dryline_probs[i][j] = prediction[n][0][i][j][5]
                if model.name == '3plus3D':
                    l = 0  # Level index (0 = surface, 1 = 1000mb, 2 = 950mb, 3 = 900mb, 4 = 850mb)
                    for i in range(0, map_dim_x):
                        for j in range(0, map_dim_y):
                            no_probs[i][j] = prediction[n][0][i][j][l][0]
                            cold_probs[i][j] = prediction[n][0][i][j][l][1]
                            warm_probs[i][j] = prediction[n][0][i][j][l][2]
                            stationary_probs[i][j] = prediction[n][0][i][j][l][3]
                            occluded_probs[i][j] = prediction[n][0][i][j][l][4]
                            dryline_probs[i][j] = prediction[n][0][i][j][l][5]
                if image == 0:
                    image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                    image_cold_probs[0: model_longitude_length - image_trim][:] = cold_probs[image_trim: model_longitude_length][:]
                    image_warm_probs[0: model_longitude_length - image_trim][:] = warm_probs[image_trim: model_longitude_length][:]
                    image_stationary_probs[0: model_longitude_length - image_trim][:] = stationary_probs[image_trim: model_longitude_length][:]
                    image_occluded_probs[0: model_longitude_length - image_trim][:] = occluded_probs[image_trim: model_longitude_length][:]
                elif image != num_images - 1:
                    image_dryline_probs[0: model_longitude_length - image_trim][:] = dryline_probs[image_trim: model_longitude_length][:]

                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                else:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    probs_ds = xr.Dataset(
                        {"no_probs": (("latitude", "longitude"), image_no_probs), "cold_probs": (("latitude", "longitude"), image_cold_probs),
                         "warm_probs": (("latitude", "longitude"), image_warm_probs), "stationary_probs": (("latitude", "longitude"), image_stationary_probs),
                         "occluded_probs": (("latitude", "longitude"), image_occluded_probs), "dryline_probs": (("latitude", "longitude"), image_dryline_probs)},
                        coords={"latitude": lats, "longitude": lons})

                    for boundary in range(4):
                        fronts = pd.read_pickle(fronts_filename)
                        for y in range(boundary+1):
                            fronts = ope(fronts)

                        new_fronts = fronts
                        t_cold_ds = xr.where(new_fronts == 1, 1, 0)
                        t_cold_probs = t_cold_ds.identifier * probs_ds.cold_probs
                        new_fronts = fronts
                        f_cold_ds = xr.where(new_fronts == 1, 0, 1)
                        f_cold_probs = f_cold_ds.identifier * probs_ds.cold_probs
                        new_fronts = fronts

                        t_warm_ds = xr.where(new_fronts == 2, 1, 0)
                        t_warm_probs = t_warm_ds.identifier * probs_ds.warm_probs
                        new_fronts = fronts
                        f_warm_ds = xr.where(new_fronts == 2, 0, 1)
                        f_warm_probs = f_warm_ds.identifier * probs_ds.warm_probs
                        new_fronts = fronts

                        t_stationary_ds = xr.where(new_fronts == 3, 1, 0)
                        t_stationary_probs = t_stationary_ds.identifier * probs_ds.stationary_probs
                        new_fronts = fronts
                        f_stationary_ds = xr.where(new_fronts == 3, 0, 1)
                        f_stationary_probs = f_stationary_ds.identifier * probs_ds.stationary_probs
                        new_fronts = fronts

                        t_occluded_ds = xr.where(new_fronts == 4, 1, 0)
                        t_occluded_probs = t_occluded_ds.identifier * probs_ds.occluded_probs
                        new_fronts = fronts
                        f_occluded_ds = xr.where(new_fronts == 4, 0, 1)
                        f_occluded_probs = f_occluded_ds.identifier * probs_ds.occluded_probs

                        t_dryline_ds = xr.where(new_fronts == 5, 1, 0)
                        t_dryline_probs = t_dryline_ds.identifier * probs_ds.dryline_probs
                        new_fronts = fronts
                        f_dryline_ds = xr.where(new_fronts == 5, 0, 1)
                        f_dryline_probs = f_dryline_ds.identifier * probs_ds.dryline_probs

                        for i in range(100):
                            tp_cold[boundary,i] += len(np.where(t_cold_probs > thresholds[i])[0])
                            tn_cold[boundary,i] += len(np.where((f_cold_probs < thresholds[i]) & (f_cold_probs != 0))[0])
                            fp_cold[boundary,i] += len(np.where(f_cold_probs > thresholds[i])[0])
                            fn_cold[boundary,i] += len(np.where((t_cold_probs < thresholds[i]) & (t_cold_probs != 0))[0])
                            tp_warm[boundary,i] += len(np.where(t_warm_probs > thresholds[i])[0])
                            tn_warm[boundary,i] += len(np.where((f_warm_probs < thresholds[i]) & (f_warm_probs != 0))[0])
                            fp_warm[boundary,i] += len(np.where(f_warm_probs > thresholds[i])[0])
                            fn_warm[boundary,i] += len(np.where((t_warm_probs < thresholds[i]) & (t_warm_probs != 0))[0])
                            tp_stationary[boundary,i] += len(np.where(t_stationary_probs > thresholds[i])[0])
                            tn_stationary[boundary,i] += len(np.where((f_stationary_probs < thresholds[i]) & (f_stationary_probs != 0))[0])
                            fp_stationary[boundary,i] += len(np.where(f_stationary_probs > thresholds[i])[0])
                            fn_stationary[boundary,i] += len(np.where((t_stationary_probs < thresholds[i]) & (t_stationary_probs != 0))[0])
                            tp_occluded[boundary,i] += len(np.where(t_occluded_probs > thresholds[i])[0])
                            tn_occluded[boundary,i] += len(np.where((f_occluded_probs < thresholds[i]) & (f_occluded_probs != 0))[0])
                            fp_occluded[boundary,i] += len(np.where(f_occluded_probs > thresholds[i])[0])
                            fn_occluded[boundary,i] += len(np.where((t_occluded_probs < thresholds[i]) & (t_occluded_probs != 0))[0])
                            tp_dryline[boundary,i] += len(np.where(t_dryline_probs > thresholds[i])[0])
                            tn_dryline[boundary,i] += len(np.where((f_dryline_probs < thresholds[i]) & (f_dryline_probs != 0))[0])
                            fp_dryline[boundary,i] += len(np.where(f_dryline_probs > thresholds[i])[0])
                            fn_dryline[boundary,i] += len(np.where((t_dryline_probs < thresholds[i]) & (t_dryline_probs != 0))[0])

                    print("Prediction %d/%d....done" % (x+1, len(front_files)))

    if front_types == 'CFWF':
        performance_ds = xr.Dataset({"tp_cold": (["boundary", "threshold"], tp_cold), "tp_warm": (["boundary", "threshold"], tp_warm),
                                     "fp_cold": (["boundary", "threshold"], fp_cold), "fp_warm": (["boundary", "threshold"], fp_warm),
                                     "tn_cold": (["boundary", "threshold"], tn_cold), "tn_warm": (["boundary", "threshold"], tn_warm),
                                     "fn_cold": (["boundary", "threshold"], fn_cold), "fn_warm": (["boundary", "threshold"], fn_warm)}, coords={"boundary": boundaries, "threshold": thresholds})
    elif front_types == 'SFOF':
        performance_ds = xr.Dataset({"tp_stationary": ("threshold", tp_stationary), "tp_occluded": ("threshold", tp_occluded),
                                     "fp_stationary": ("threshold", fp_stationary), "fp_occluded": ("threshold", fp_occluded),
                                     "tn_stationary": ("threshold", tn_stationary), "tn_occluded": ("threshold", tn_occluded),
                                     "fn_stationary": ("threshold", fn_stationary), "fn_occluded": ("threshold", fn_occluded)}, coords={"boundary": boundaries, "threshold": thresholds})
    elif front_types == 'DL':
        performance_ds = xr.Dataset({"tp_dryline": ("threshold", tp_dryline), "fp_dryline": ("threshold", tp_dryline),
                                     "fn_dryline": ("threshold", fp_dryline), "tn_dryline": ("threshold", tn_dryline)}, coords={"boundary": boundaries, "threshold": thresholds})
    elif front_types == 'ALL':
        performance_ds = xr.Dataset({"tp_cold": ("threshold", tp_cold), "tp_warm": ("threshold", tp_warm),
                                     "tp_stationary": ("threshold", tp_stationary), "tp_occluded": ("threshold", tp_occluded),
                                     "fp_cold": ("threshold", fp_cold), "fp_warm": ("threshold", fp_warm),
                                     "fp_stationary": ("threshold", fp_stationary), "fp_occluded": ("threshold", fp_occluded),
                                     "tn_cold": ("threshold", tn_cold), "tn_warm": ("threshold", tn_warm),
                                     "tn_stationary": ("threshold", tn_stationary), "tn_occluded": ("threshold", tn_occluded),
                                     "fn_cold": ("threshold", fn_cold), "fn_warm": ("threshold", fn_warm),
                                     "fn_stationary": ("threshold", fn_stationary), "fn_occluded": ("threshold", fn_occluded)}, coords={"boundary": boundaries, "threshold": thresholds})

    print(performance_ds)
    with open("%s/model_%d/model_%d_performance_stats_%dimage_%dtrim.pkl" % (model_dir, model_number, model_number, num_images, image_trim), "wb") as f:
        pickle.dump(performance_ds, f)


def find_matches_for_domain(longitude_domain_length, model_longitude_length):
    """
    Function that outputs the number of images that can be stitched together with the specified domain length and the length
    of the domain dimension output by the model.

    Parameters
    ----------
    longitude_domain_length: int
        Number of pixels in the longitude dimension of the final stitched map.
    model_longitude_length: int
        Number of pixels in the longitude dimension of the model's output.
    """

    # latitude_domain_length = 128
    # model_latitude_length = 128
    num_matches = 0
    for i in range(2,longitude_domain_length-model_longitude_length):  # Image counter
        image_spacing_match = (longitude_domain_length-model_longitude_length)/(i-1)
        if image_spacing_match - int(image_spacing_match) == 0 and image_spacing_match > 1 and model_longitude_length-image_spacing_match > 0:
            num_matches += 1
            print("MATCH: (Images, Spacing, Overlap, Max Trim)", i, int(image_spacing_match), int(model_longitude_length-image_spacing_match), int(np.floor(image_spacing_match/2)))
    print("\nMatches found: %d" % num_matches)


def make_prediction(model_number, model_dir, front_file_list, variable_file_list, normalization_method, loss, fss_mask_size,
    fss_c, front_types, pixel_expansion, metric, num_dimensions, num_images, longitude_domain_length, image_trim, 
    year, month, day, hour):
    """
    Function that makes random predictions using the provided model.

    Parameters
    ----------
    model_number: int
        Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        Main directory for the models.
    front_file_list: list
        List of filenames that contain front data.
    variable_file_list: list
        List of filenames that contain variable data.
    normalization_method: int
        Normalization method for the data (described near the end of the script).
    loss: str
        Loss function for the Unet.
    fss_mask_size: int
        Size of the mask for the FSS loss function.
    fss_c: float
        C hyperparameter for the FSS loss' sigmoid function.
    front_types: str
        Fronts in the data.
    pixel_expansion: int
        Number of pixels to expand the fronts by in all directions.
    metric: str
        Metric used for evaluating the U-Net during training.
    num_dimensions: int
        Number of dimensions for the U-Net's convolutions, maxpooling, and upsampling.
    num_images: int
        Number of images to stitch together for the final probability maps.
    longitude_domain_length: int
        Number of pixels in the longitude dimension of the final stitched map.
    image_trim: int
        Number of pixels to trim each image by in the longitude dimension before averaging the overlapping pixels.
    year: int
    month: int
    day: int
    hour: int
    """
    model = fm.load_model(model_number, model_dir, loss, fss_mask_size, fss_c, metric, num_dimensions)

    n = 0  # Counter for the number of down layers in the model
    map_dim_x = model.layers[0].input_shape[0][1]  # Longitudinal dimension of the U-Net images
    map_dim_y = model.layers[0].input_shape[0][2]  # Latitudinal dimension of the U-Net images
    if num_dimensions == 2:
        for layer in model.layers:
            if layer.__class__.__name__ == 'MaxPooling2D':
                n += 1
        n = int((n - 1)/2)
        channels = model.layers[0].input_shape[0][3]  # Number of variables used
    if num_dimensions == 3:
        for layer in model.layers:
            if layer.__class__.__name__ == 'MaxPooling3D':
                n += 1
        n = int((n - 1)/2)
        levels = model.layers[0].input_shape[0][3]  # Number of levels to the U-Net variables
        channels = model.layers[0].input_shape[0][4]

    model_longitude_length = map_dim_x
    raw_image_index_array = np.empty(shape=[num_images,2])
    longitude_domain_length_trim = longitude_domain_length - 2*image_trim
    image_spacing = int((longitude_domain_length - model_longitude_length)/(num_images-1))
    latitude_domain_length = 128

    norm_params = pd.read_csv('normalization_parameters.csv', index_col='Variable')

    front_filename_no_dir = 'FrontObjects_%s_%d%02d%02d%02d_%s_%dx%d.pkl' % (args.front_types, args.year, args.month,
        args.day, args.hour, args.domain, args.file_dimensions[0], args.file_dimensions[1])
    variable_filename_no_dir = 'Data_%dvar_%d%02d%02d%02d_%s_%dx%d.pkl' % (60, args.year, args.month, args.day, args.hour,
        args.domain, args.file_dimensions[0], args.file_dimensions[1])

    front_file = [front_filename for front_filename in front_file_list if front_filename_no_dir in front_filename][0]
    variable_file = [variable_filename for variable_filename in variable_file_list if variable_filename_no_dir in variable_filename][0]

    fronts_ds = pd.read_pickle(front_file)

    for i in range(num_images):
        if i == 0:
            raw_image_index_array[i][0] = image_trim
            raw_image_index_array[i][1] = model_longitude_length - image_trim
        elif i != num_images - 1:
            raw_image_index_array[i][0] = raw_image_index_array[i-1][0] + image_spacing
            raw_image_index_array[i][1] = raw_image_index_array[i-1][1] + image_spacing
        else:
            raw_image_index_array[i][0] = raw_image_index_array[i-1][0] + image_spacing
            raw_image_index_array[i][1] = raw_image_index_array[i-1][1] + image_spacing + image_trim - image_trim

    lon_pixels_per_image = int(raw_image_index_array[0][1] - raw_image_index_array[0][0])

    image_no_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
    image_cold_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
    image_warm_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
    image_stationary_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
    image_occluded_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
    image_dryline_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
    fronts = fronts_ds.sel(longitude=fronts_ds.longitude.values[image_trim:longitude_domain_length-image_trim],
                           latitude=fronts_ds.latitude.values[0:128])
    image_lats = fronts_ds.latitude.values[0:128]
    image_lons = fronts_ds.longitude.values[image_trim:longitude_domain_length-image_trim]

    for image in range(num_images):
        print("%d-%02d-%02d-%02dz...%d/%d" % (year, month, day, hour, image+1, num_images), end='\r')
        lat_index = 0
        variable_ds = pd.read_pickle(variable_file)
        if image == 0:
            lon_index = 0
        else:
            lon_index = int(image*image_spacing)

        lons = variable_ds.longitude.values[lon_index:lon_index + map_dim_x]
        lats = variable_ds.latitude.values[lat_index:lat_index + map_dim_y]

        variable_list = list(variable_ds.keys())
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

        if num_dimensions == 2:
            variable_ds_new = np.nan_to_num(variable_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, map_dim_x,
                map_dim_y, channels))
        elif num_dimensions == 3:
            variables_sfc = variable_ds[['t2m','d2m','sp','u10','v10','theta_w','mix_ratio','rel_humid','virt_temp','wet_bulb','theta_e',
                                         'q']].sel(longitude=lons, latitude=lats).to_array().T.values
            variables_1000 = variable_ds[['t_1000','d_1000','z_1000','u_1000','v_1000','theta_w_1000','mix_ratio_1000','rel_humid_1000','virt_temp_1000',
                                          'wet_bulb_1000','theta_e_1000','q_1000']].sel(longitude=lons, latitude=lats).to_array().T.values
            variables_950 = variable_ds[['t_950','d_950','z_950','u_950','v_950','theta_w_950','mix_ratio_950','rel_humid_950','virt_temp_950',
                                         'wet_bulb_950','theta_e_950','q_950']].sel(longitude=lons, latitude=lats).to_array().T.values
            variables_900 = variable_ds[['t_900','d_900','z_900','u_900','v_900','theta_w_900','mix_ratio_900','rel_humid_900','virt_temp_900',
                                         'wet_bulb_900','theta_e_900','q_900']].sel(longitude=lons, latitude=lats).to_array().T.values
            variables_850 = variable_ds[['t_850','d_850','z_850','u_850','v_850','theta_w_850','mix_ratio_850','rel_humid_850','virt_temp_850',
                                         'wet_bulb_850','theta_e_850','q_850']].sel(longitude=lons, latitude=lats).to_array().T.values
            variable_ds_new = np.array([variables_sfc,variables_1000,variables_950,variables_900,variables_850]).reshape(1,map_dim_x,map_dim_y,levels,channels)

        prediction = model.predict(variable_ds_new)

        time = str(fronts.time.values)[0:13].replace('T', '-') + 'z'
        # print(time)

        # Arrays of probabilities for all front types
        no_probs = np.zeros([map_dim_x, map_dim_y])
        cold_probs = np.zeros([map_dim_x, map_dim_y])
        warm_probs = np.zeros([map_dim_x, map_dim_y])
        stationary_probs = np.zeros([map_dim_x, map_dim_y])
        occluded_probs = np.zeros([map_dim_x, map_dim_y])
        dryline_probs = np.zeros([map_dim_x, map_dim_y])

        if front_types == 'CFWF':
            if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':
                for i in range(0, map_dim_x):
                    for j in range(0, map_dim_y):
                        no_probs[i][j] = prediction[n][0][i][j][0]
                        cold_probs[i][j] = prediction[n][0][i][j][1]
                        warm_probs[i][j] = prediction[n][0][i][j][2]
            if model.name == '3plus3D':
                l = 0  # Level index (0 = surface, 1 = 1000mb, 2 = 950mb, 3 = 900mb, 4 = 850mb)
                for i in range(0, map_dim_x):
                    for j in range(0, map_dim_y):
                        no_probs[i][j] = prediction[n][0][i][j][l][0]
                        cold_probs[i][j] = prediction[n][0][i][j][l][1]
                        warm_probs[i][j] = prediction[n][0][i][j][l][2]
            if image == 0:
                image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                image_cold_probs[0: model_longitude_length - image_trim][:] = cold_probs[image_trim: model_longitude_length][:]
                image_warm_probs[0: model_longitude_length - image_trim][:] = warm_probs[image_trim: model_longitude_length][:]
            elif image != num_images - 1:
                image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
            else:
                image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                probs_ds = xr.Dataset(
                    {"no_probs": (("longitude", "latitude"), image_no_probs), "cold_probs": (("longitude", "latitude"), image_cold_probs),
                     "warm_probs": (("longitude", "latitude"), image_warm_probs)}, coords={"latitude": image_lats, "longitude": image_lons}).transpose()
                prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, num_images, image_trim)

        elif front_types == 'SFOF':
            if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':
                for i in range(0, map_dim_x):
                    for j in range(0, map_dim_y):
                        no_probs[i][j] = prediction[n][0][i][j][0]
                        stationary_probs[i][j] = prediction[n][0][i][j][1]
                        occluded_probs[i][j] = prediction[n][0][i][j][2]
            if model.name == '3plus3D':
                l = 0  # Level index (0 = surface, 1 = 1000mb, 2 = 950mb, 3 = 900mb, 4 = 850mb)
                for i in range(0, map_dim_x):
                    for j in range(0, map_dim_y):
                        no_probs[i][j] = prediction[n][0][i][j][l][0]
                        stationary_probs[i][j] = prediction[n][0][i][j][l][1]
                        occluded_probs[i][j] = prediction[n][0][i][j][l][2]
            if image == 0:
                image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                image_stationary_probs[0: model_longitude_length - image_trim][:] = stationary_probs[image_trim: model_longitude_length][:]
                image_occluded_probs[0: model_longitude_length - image_trim][:] = occluded_probs[image_trim: model_longitude_length][:]
            elif image != num_images - 1:
                image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
            else:
                image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                probs_ds = xr.Dataset(
                    {"no_probs": (("longitude", "latitude"), image_no_probs), "stationary_probs": (("longitude", "latitude"), image_stationary_probs),
                     "occluded_probs": (("longitude", "latitude"), image_occluded_probs)}, coords={"latitude": image_lats, "longitude": image_lons}).transpose()
                prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, num_images, image_trim)

        elif front_types == 'DL':
            if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':
                for i in range(0, map_dim_x):
                    for j in range(0, map_dim_y):
                        no_probs[i][j] = prediction[n][0][i][j][0]
                        dryline_probs[i][j] = prediction[n][0][i][j][1]
            if model.name == '3plus3D':
                l = 0  # Level index (0 = surface, 1 = 1000mb, 2 = 950mb, 3 = 900mb, 4 = 850mb)
                for i in range(0, map_dim_x):
                    for j in range(0, map_dim_y):
                        no_probs[i][j] = prediction[n][0][i][j][l][0]
                        dryline_probs[i][j] = prediction[n][0][i][j][l][1]
            if image == 0:
                image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                image_dryline_probs[0: model_longitude_length - image_trim][:] = dryline_probs[image_trim: model_longitude_length][:]
            elif image != num_images - 1:
                image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
            else:
                image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                probs_ds = xr.Dataset(
                    {"no_probs": (("latitude", "longitude"), no_probs), "dryline_probs": (("latitude", "longitude"), image_dryline_probs)},
                    coords={"latitude": lats, "longitude": lons})
                prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, num_images, image_trim)

        elif front_types == 'ALL':
            if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':
                for i in range(0, map_dim_x):
                    for j in range(0, map_dim_y):
                        no_probs[i][j] = prediction[n][0][i][j][0]
                        cold_probs[i][j] = prediction[n][0][i][j][1]
                        warm_probs[i][j] = prediction[n][0][i][j][2]
                        stationary_probs[i][j] = prediction[n][0][i][j][3]
                        occluded_probs[i][j] = prediction[n][0][i][j][4]
                        dryline_probs[i][j] = prediction[n][0][i][j][5]
            if model.name == '3plus3D':
                l = 0  # Level index (0 = surface, 1 = 1000mb, 2 = 950mb, 3 = 900mb, 4 = 850mb)
                for i in range(0, map_dim_x):
                    for j in range(0, map_dim_y):
                        no_probs[i][j] = prediction[n][0][i][j][l][0]
                        cold_probs[i][j] = prediction[n][0][i][j][l][1]
                        warm_probs[i][j] = prediction[n][0][i][j][l][2]
                        stationary_probs[i][j] = prediction[n][0][i][j][l][3]
                        occluded_probs[i][j] = prediction[n][0][i][j][l][4]
                        dryline_probs[i][j] = prediction[n][0][i][j][l][5]
            if image == 0:
                image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                image_cold_probs[0: model_longitude_length - image_trim][:] = cold_probs[image_trim: model_longitude_length][:]
                image_warm_probs[0: model_longitude_length - image_trim][:] = warm_probs[image_trim: model_longitude_length][:]
                image_stationary_probs[0: model_longitude_length - image_trim][:] = stationary_probs[image_trim: model_longitude_length][:]
                image_occluded_probs[0: model_longitude_length - image_trim][:] = occluded_probs[image_trim: model_longitude_length][:]
                image_dryline_probs[0: model_longitude_length - image_trim][:] = dryline_probs[image_trim: model_longitude_length][:]
            elif image != num_images - 1:
                image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
            else:
                image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                probs_ds = xr.Dataset(
                    {"no_probs": (("latitude", "longitude"), image_no_probs), "cold_probs": (("latitude", "longitude"), image_cold_probs),
                     "warm_probs": (("latitude", "longitude"), image_warm_probs), "stationary_probs": (("latitude", "longitude"), image_stationary_probs),
                     "occluded_probs": (("latitude", "longitude"), image_occluded_probs), "dryline_probs": (("latitude", "longitude"), image_dryline_probs)},
                    coords={"latitude": lats, "longitude": lons})
                prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, num_images, image_trim)


def make_random_predictions(model_number, model_dir, front_files, variable_files, predictions, normalization_method,
    loss, fss_mask_size, fss_c, front_types, pixel_expansion, metric, num_dimensions, num_images, longitude_domain_length,
    image_trim):
    """
    Function that makes random predictions using the provided model and an optional test_years argument.

    Parameters
    ----------
    model_number: int
        Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        Main directory for the models.
    front_files: list
        List of filenames that contain front data.
    variable_files: list
        List of filenames that contain variable data.
    predictions: int
        Number of random predictions to make.
    normalization_method: int
        Normalization method for the data (described near the end of the script).
    loss: str
        Loss function for the Unet.
    fss_mask_size: int
        Size of the mask for the FSS loss function.
    fss_c: float
        C hyperparameter for the FSS loss' sigmoid function.
    front_types: str
        Fronts in the data.
    pixel_expansion: int
        Number of pixels to expand the fronts by in all directions.
    metric: str
        Metric used for evaluating the U-Net during training.
    num_dimensions: int
        Number of dimensions for the U-Net's convolutions, maxpooling, and upsampling.
    num_images: int
        Number of images to stitch together for the final probability maps.
    longitude_domain_length: int
        Number of pixels in the longitude dimension of the final stitched map.
    image_trim: int
        Number of pixels to trim each image by in the longitude dimension before averaging the overlapping pixels.
    """
    model = fm.load_model(model_number, model_dir, loss, fss_mask_size, fss_c, metric, num_dimensions)

    n = 0  # Counter for the number of down layers in the model
    map_dim_x = model.layers[0].input_shape[0][1]  # Longitudinal dimension of the U-Net images
    map_dim_y = model.layers[0].input_shape[0][2]  # Latitudinal dimension of the U-Net images
    if num_dimensions == 2:
        for layer in model.layers:
            if layer.__class__.__name__ == 'MaxPooling2D':
                n += 1
        n = int((n - 1)/2)
        channels = model.layers[0].input_shape[0][3]  # Number of variables used
    if num_dimensions == 3:
        for layer in model.layers:
            if layer.__class__.__name__ == 'MaxPooling3D':
                n += 1
        n = int((n - 1)/2)
        levels = model.layers[0].input_shape[0][3]  # Number of levels to the U-Net variables
        channels = model.layers[0].input_shape[0][4]

    model_longitude_length = map_dim_x
    raw_image_index_array = np.empty(shape=[num_images,2])
    longitude_domain_length_trim = longitude_domain_length - 2*image_trim
    image_spacing = int((longitude_domain_length - model_longitude_length)/(num_images-1))
    latitude_domain_length = 128

    norm_params = pd.read_csv('normalization_parameters.csv', index_col='Variable')

    for x in range(predictions):

        # Open random pair of files
        
        index = random.choices(range(len(front_files) - 1), k=1)[0]
        fronts_filename = front_files[index]
        variables_filename = variable_files[index]
        fronts_ds = pd.read_pickle(fronts_filename)

        for i in range(num_images):
            if i == 0:
                raw_image_index_array[i][0] = image_trim
                raw_image_index_array[i][1] = model_longitude_length - image_trim
            elif i != num_images - 1:
                raw_image_index_array[i][0] = raw_image_index_array[i-1][0] + image_spacing
                raw_image_index_array[i][1] = raw_image_index_array[i-1][1] + image_spacing
            else:
                raw_image_index_array[i][0] = raw_image_index_array[i-1][0] + image_spacing
                raw_image_index_array[i][1] = raw_image_index_array[i-1][1] + image_spacing + image_trim - image_trim

        lon_pixels_per_image = int(raw_image_index_array[0][1] - raw_image_index_array[0][0])

        image_no_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_cold_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_warm_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_stationary_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_occluded_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_dryline_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        fronts = fronts_ds.sel(longitude=fronts_ds.longitude.values[image_trim:longitude_domain_length-image_trim], latitude=fronts_ds.latitude.values[0:128])
        image_lats = fronts_ds.latitude.values[0:128]
        image_lons = fronts_ds.longitude.values[image_trim:longitude_domain_length-image_trim]

        for image in range(num_images):
            print("Prediction %d/%d....%d/%d" % (x+1, predictions, image+1, num_images), end='\r')
            lat_index = 0
            variable_ds = pd.read_pickle(variables_filename)
            if image == 0:
                lon_index = 0
            else:
                lon_index = int(image*image_spacing)

            lons = variable_ds.longitude.values[lon_index:lon_index + map_dim_x]
            lats = variable_ds.latitude.values[lat_index:lat_index + map_dim_y]

            variable_list = list(variable_ds.keys())
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

            if num_dimensions == 2:
                variable_ds_new = np.nan_to_num(variable_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, map_dim_x,
                    map_dim_y, channels))
            elif num_dimensions == 3:
                variables_sfc = variable_ds[['t2m','d2m','sp','u10','v10','theta_w','mix_ratio','rel_humid','virt_temp','wet_bulb','theta_e',
                                             'q']].sel(longitude=lons, latitude=lats).to_array().T.values
                variables_1000 = variable_ds[['t_1000','d_1000','z_1000','u_1000','v_1000','theta_w_1000','mix_ratio_1000','rel_humid_1000','virt_temp_1000',
                                              'wet_bulb_1000','theta_e_1000','q_1000']].sel(longitude=lons, latitude=lats).to_array().T.values
                variables_950 = variable_ds[['t_950','d_950','z_950','u_950','v_950','theta_w_950','mix_ratio_950','rel_humid_950','virt_temp_950',
                                             'wet_bulb_950','theta_e_950','q_950']].sel(longitude=lons, latitude=lats).to_array().T.values
                variables_900 = variable_ds[['t_900','d_900','z_900','u_900','v_900','theta_w_900','mix_ratio_900','rel_humid_900','virt_temp_900',
                                             'wet_bulb_900','theta_e_900','q_900']].sel(longitude=lons, latitude=lats).to_array().T.values
                variables_850 = variable_ds[['t_850','d_850','z_850','u_850','v_850','theta_w_850','mix_ratio_850','rel_humid_850','virt_temp_850',
                                             'wet_bulb_850','theta_e_850','q_850']].sel(longitude=lons, latitude=lats).to_array().T.values
                variable_ds_new = np.array([variables_sfc,variables_1000,variables_950,variables_900,variables_850]).reshape(1,map_dim_x,map_dim_y,levels,channels)

            prediction = model.predict(variable_ds_new)

            time = str(fronts.time.values)[0:13].replace('T', '-') + 'z'
            # print(time)

            # Arrays of probabilities for all front types
            no_probs = np.zeros([map_dim_x, map_dim_y])
            cold_probs = np.zeros([map_dim_x, map_dim_y])
            warm_probs = np.zeros([map_dim_x, map_dim_y])
            stationary_probs = np.zeros([map_dim_x, map_dim_y])
            occluded_probs = np.zeros([map_dim_x, map_dim_y])
            dryline_probs = np.zeros([map_dim_x, map_dim_y])

            if front_types == 'CFWF':
                if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':
                    for i in range(0, map_dim_x):
                        for j in range(0, map_dim_y):
                            no_probs[i][j] = prediction[n][0][i][j][0]
                            cold_probs[i][j] = prediction[n][0][i][j][1]
                            warm_probs[i][j] = prediction[n][0][i][j][2]
                if model.name == '3plus3D':
                    l = 0  # Level index (0 = surface, 1 = 1000mb, 2 = 950mb, 3 = 900mb, 4 = 850mb)
                    for i in range(0, map_dim_x):
                        for j in range(0, map_dim_y):
                            no_probs[i][j] = prediction[n][0][i][j][l][0]
                            cold_probs[i][j] = prediction[n][0][i][j][l][1]
                            warm_probs[i][j] = prediction[n][0][i][j][l][2]
                if image == 0:
                    image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                    image_cold_probs[0: model_longitude_length - image_trim][:] = cold_probs[image_trim: model_longitude_length][:]
                    image_warm_probs[0: model_longitude_length - image_trim][:] = warm_probs[image_trim: model_longitude_length][:]
                elif image != num_images - 1:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                else:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    probs_ds = xr.Dataset(
                        {"no_probs": (("longitude", "latitude"), image_no_probs), "cold_probs": (("longitude", "latitude"), image_cold_probs),
                         "warm_probs": (("longitude", "latitude"), image_warm_probs)}, coords={"latitude": image_lats, "longitude": image_lons}).transpose()
                    prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, num_images, image_trim)

            elif front_types == 'SFOF':
                if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':
                    for i in range(0, map_dim_x):
                        for j in range(0, map_dim_y):
                            no_probs[i][j] = prediction[n][0][i][j][0]
                            stationary_probs[i][j] = prediction[n][0][i][j][1]
                            occluded_probs[i][j] = prediction[n][0][i][j][2]
                if model.name == '3plus3D':
                    l = 0  # Level index (0 = surface, 1 = 1000mb, 2 = 950mb, 3 = 900mb, 4 = 850mb)
                    for i in range(0, map_dim_x):
                        for j in range(0, map_dim_y):
                            no_probs[i][j] = prediction[n][0][i][j][l][0]
                            stationary_probs[i][j] = prediction[n][0][i][j][l][1]
                            occluded_probs[i][j] = prediction[n][0][i][j][l][2]
                if image == 0:
                    image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                    image_stationary_probs[0: model_longitude_length - image_trim][:] = stationary_probs[image_trim: model_longitude_length][:]
                    image_occluded_probs[0: model_longitude_length - image_trim][:] = occluded_probs[image_trim: model_longitude_length][:]
                elif image != num_images - 1:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                else:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    probs_ds = xr.Dataset(
                        {"no_probs": (("longitude", "latitude"), image_no_probs), "stationary_probs": (("longitude", "latitude"), image_stationary_probs),
                         "occluded_probs": (("longitude", "latitude"), image_occluded_probs)}, coords={"latitude": image_lats, "longitude": image_lons}).transpose()
                    prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, num_images, image_trim)

            elif front_types == 'DL':
                if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':
                    for i in range(0, map_dim_x):
                        for j in range(0, map_dim_y):
                            no_probs[i][j] = prediction[n][0][i][j][0]
                            dryline_probs[i][j] = prediction[n][0][i][j][1]
                if model.name == '3plus3D':
                    l = 0  # Level index (0 = surface, 1 = 1000mb, 2 = 950mb, 3 = 900mb, 4 = 850mb)
                    for i in range(0, map_dim_x):
                        for j in range(0, map_dim_y):
                            no_probs[i][j] = prediction[n][0][i][j][l][0]
                            dryline_probs[i][j] = prediction[n][0][i][j][l][1]
                if image == 0:
                    image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                    image_dryline_probs[0: model_longitude_length - image_trim][:] = dryline_probs[image_trim: model_longitude_length][:]
                elif image != num_images - 1:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                else:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    probs_ds = xr.Dataset(
                        {"no_probs": (("latitude", "longitude"), no_probs), "dryline_probs": (("latitude", "longitude"), image_dryline_probs)},
                        coords={"latitude": lats, "longitude": lons})
                    prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, num_images, image_trim)

            elif front_types == 'ALL':
                if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':
                    for i in range(0, map_dim_x):
                        for j in range(0, map_dim_y):
                            no_probs[i][j] = prediction[n][0][i][j][0]
                            cold_probs[i][j] = prediction[n][0][i][j][1]
                            warm_probs[i][j] = prediction[n][0][i][j][2]
                            stationary_probs[i][j] = prediction[n][0][i][j][3]
                            occluded_probs[i][j] = prediction[n][0][i][j][4]
                            dryline_probs[i][j] = prediction[n][0][i][j][5]
                if model.name == '3plus3D':
                    l = 0  # Level index (0 = surface, 1 = 1000mb, 2 = 950mb, 3 = 900mb, 4 = 850mb)
                    for i in range(0, map_dim_x):
                        for j in range(0, map_dim_y):
                            no_probs[i][j] = prediction[n][0][i][j][l][0]
                            cold_probs[i][j] = prediction[n][0][i][j][l][1]
                            warm_probs[i][j] = prediction[n][0][i][j][l][2]
                            stationary_probs[i][j] = prediction[n][0][i][j][l][3]
                            occluded_probs[i][j] = prediction[n][0][i][j][l][4]
                            dryline_probs[i][j] = prediction[n][0][i][j][l][5]
                if image == 0:
                    image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                    image_cold_probs[0: model_longitude_length - image_trim][:] = cold_probs[image_trim: model_longitude_length][:]
                    image_warm_probs[0: model_longitude_length - image_trim][:] = warm_probs[image_trim: model_longitude_length][:]
                    image_stationary_probs[0: model_longitude_length - image_trim][:] = stationary_probs[image_trim: model_longitude_length][:]
                    image_occluded_probs[0: model_longitude_length - image_trim][:] = occluded_probs[image_trim: model_longitude_length][:]
                    image_dryline_probs[0: model_longitude_length - image_trim][:] = dryline_probs[image_trim: model_longitude_length][:]
                elif image != num_images - 1:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                else:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    probs_ds = xr.Dataset(
                        {"no_probs": (("latitude", "longitude"), image_no_probs), "cold_probs": (("latitude", "longitude"), image_cold_probs),
                         "warm_probs": (("latitude", "longitude"), image_warm_probs), "stationary_probs": (("latitude", "longitude"), image_stationary_probs),
                         "occluded_probs": (("latitude", "longitude"), image_occluded_probs), "dryline_probs": (("latitude", "longitude"), image_dryline_probs)},
                        coords={"latitude": lats, "longitude": lons})
                    prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, num_images, image_trim)


def plot_performance_diagrams(model_dir, model_number, front_types, num_images, image_trim):
    """
    Plots CSI performance diagram for different front types.

    Parameters
    ----------
    model_dir: str
        Main directory for the models.
    model_number: int
        Slurm job number for the model. This is the number in the model's filename.
    front_types: str
        Fronts in the data.
    num_images: int
        Number of images to stitch together for the final probability maps.
    image_trim: int
        Number of pixels to trim each image by in the longitude dimension before averaging the overlapping pixels.
    """
    stats = pd.read_pickle("%s/model_%d/model_%d_performance_stats_%dimage_%dtrim.pkl" % (model_dir, model_number, model_number, num_images, image_trim))
    stats_25km = stats.sel(boundary=25)
    stats_50km = stats.sel(boundary=50)
    stats_75km = stats.sel(boundary=75)
    stats_100km = stats.sel(boundary=100)

    # Code for performance diagram matrices sourced from Ryan Lagerquist's (lagerqui@ualberta.ca) thunderhoser repository:
    # https://github.com/thunderhoser/GewitterGefahr/blob/master/gewittergefahr/plotting/model_eval_plotting.py
    success_ratio_matrix, pod_matrix = np.linspace(0,1,100), np.linspace(0,1,100)
    x, y = np.meshgrid(success_ratio_matrix, pod_matrix)
    csi_matrix = (x ** -1 + y ** -1 - 1.) ** -1
    CSI_LEVELS = np.linspace(0,1,11)
    cmap = 'Blues'

    if front_types == 'CFWF' or front_types == 'ALL':
        POD_cold_25km = stats_25km['tp_cold']/(stats_25km['tp_cold'] + stats_25km['fn_cold'])
        SR_cold_25km = stats_25km['tp_cold']/(stats_25km['tp_cold'] + stats_25km['fp_cold'])
        CSI_cold_25km = stats_25km['tp_cold']/(stats_25km['tp_cold'] + stats_25km['fp_cold'] + stats_25km['fn_cold'])
        POD_cold_50km = stats_50km['tp_cold']/(stats_50km['tp_cold'] + stats_50km['fn_cold'])
        SR_cold_50km = stats_50km['tp_cold']/(stats_50km['tp_cold'] + stats_50km['fp_cold'])
        CSI_cold_50km = stats_50km['tp_cold']/(stats_50km['tp_cold'] + stats_50km['fp_cold'] + stats_50km['fn_cold'])
        POD_cold_75km = stats_75km['tp_cold']/(stats_75km['tp_cold'] + stats_75km['fn_cold'])
        SR_cold_75km = stats_75km['tp_cold']/(stats_75km['tp_cold'] + stats_75km['fp_cold'])
        CSI_cold_75km = stats_75km['tp_cold']/(stats_75km['tp_cold'] + stats_75km['fp_cold'] + stats_75km['fn_cold'])
        POD_cold_100km = stats_100km['tp_cold']/(stats_100km['tp_cold'] + stats_100km['fn_cold'])
        SR_cold_100km = stats_100km['tp_cold']/(stats_100km['tp_cold'] + stats_100km['fp_cold'])
        CSI_cold_100km = stats_100km['tp_cold']/(stats_100km['tp_cold'] + stats_100km['fp_cold'] + stats_100km['fn_cold'])
        POD_warm_25km = stats_25km['tp_warm']/(stats_25km['tp_warm'] + stats_25km['fn_warm'])
        SR_warm_25km = stats_25km['tp_warm']/(stats_25km['tp_warm'] + stats_25km['fp_warm'])
        CSI_warm_25km = stats_25km['tp_warm']/(stats_25km['tp_warm'] + stats_25km['fp_warm'] + stats_25km['fn_warm'])
        POD_warm_50km = stats_50km['tp_warm']/(stats_50km['tp_warm'] + stats_50km['fn_warm'])
        SR_warm_50km = stats_50km['tp_warm']/(stats_50km['tp_warm'] + stats_50km['fp_warm'])
        CSI_warm_50km = stats_50km['tp_warm']/(stats_50km['tp_warm'] + stats_50km['fp_warm'] + stats_50km['fn_warm'])
        POD_warm_75km = stats_75km['tp_warm']/(stats_75km['tp_warm'] + stats_75km['fn_warm'])
        SR_warm_75km = stats_75km['tp_warm']/(stats_75km['tp_warm'] + stats_75km['fp_warm'])
        CSI_warm_75km = stats_75km['tp_warm']/(stats_75km['tp_warm'] + stats_75km['fp_warm'] + stats_75km['fn_warm'])
        POD_warm_100km = stats_100km['tp_warm']/(stats_100km['tp_warm'] + stats_100km['fn_warm'])
        SR_warm_100km = stats_100km['tp_warm']/(stats_100km['tp_warm'] + stats_100km['fp_warm'])
        CSI_warm_100km = stats_100km['tp_warm']/(stats_100km['tp_warm'] + stats_100km['fp_warm'] + stats_100km['fn_warm'])

        plt.figure()
        plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
        plt.colorbar(label='Critical Success Index (CSI)')
        plt.plot(SR_cold_25km, POD_cold_25km, color='red', label='25km boundary')
        plt.plot(SR_cold_25km[np.where(CSI_cold_25km == np.max(CSI_cold_25km))], POD_cold_25km[np.where(CSI_cold_25km == np.max(CSI_cold_25km))], color='red', marker='*', markersize=9)
        plt.text(0.01, 0.01, s=str('25km: %.4f' % np.max(CSI_cold_25km)), color='red')
        plt.plot(SR_cold_50km, POD_cold_50km, color='purple', label='50km boundary')
        plt.plot(SR_cold_50km[np.where(CSI_cold_50km == np.max(CSI_cold_50km))], POD_cold_50km[np.where(CSI_cold_50km == np.max(CSI_cold_50km))], color='purple', marker='*', markersize=9)
        plt.text(0.01, 0.05, s=str('50km: %.4f' % np.max(CSI_cold_50km)), color='purple')
        plt.plot(SR_cold_75km, POD_cold_75km, color='brown', label='75km boundary')
        plt.plot(SR_cold_75km[np.where(CSI_cold_75km == np.max(CSI_cold_75km))], POD_cold_75km[np.where(CSI_cold_75km == np.max(CSI_cold_75km))], color='brown', marker='*', markersize=9)
        plt.text(0.01, 0.09, s=str('75km: %.4f' % np.max(CSI_cold_75km)), color='brown')
        plt.plot(SR_cold_100km, POD_cold_100km, color='green', label='100km boundary')
        plt.plot(SR_cold_100km[np.where(CSI_cold_100km == np.max(CSI_cold_100km))], POD_cold_100km[np.where(CSI_cold_100km == np.max(CSI_cold_100km))], color='green', marker='*', markersize=9)
        plt.text(0.01, 0.13, s=str('100km: %.4f' % np.max(CSI_cold_100km)), color='green')
        plt.text(0.01, 0.17, s='CSI values', style='oblique')
        plt.legend(loc='upper right')
        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.title("Model %d Performance for Cold Fronts" % model_number)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.show()
        plt.savefig("%s/model_%d/model_%d_performance_cold_%d_%d.png" % (model_dir, model_number, model_number, num_images, image_trim),bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
        plt.colorbar(label='Critical Success Index (CSI)')
        plt.plot(SR_warm_25km, POD_warm_25km, color='red', label='25km boundary')
        plt.plot(SR_warm_25km[np.where(CSI_warm_25km == np.max(CSI_warm_25km))], POD_warm_25km[np.where(CSI_warm_25km == np.max(CSI_warm_25km))], color='red', marker='*', markersize=9)
        plt.text(0.01, 0.01, s=str('25km: %.4f' % np.max(CSI_warm_25km)), color='red')
        plt.plot(SR_warm_50km, POD_warm_50km, color='purple', label='50km boundary')
        plt.plot(SR_warm_50km[np.where(CSI_warm_50km == np.max(CSI_warm_50km))], POD_warm_50km[np.where(CSI_warm_50km == np.max(CSI_warm_50km))], color='purple', marker='*', markersize=9)
        plt.text(0.01, 0.05, s=str('50km: %.4f' % np.max(CSI_warm_50km)), color='purple')
        plt.plot(SR_warm_75km, POD_warm_75km, color='brown', label='75km boundary')
        plt.plot(SR_warm_75km[np.where(CSI_warm_75km == np.max(CSI_warm_75km))], POD_warm_75km[np.where(CSI_warm_75km == np.max(CSI_warm_75km))], color='brown', marker='*', markersize=9)
        plt.text(0.01, 0.09, s=str('75km: %.4f' % np.max(CSI_warm_75km)), color='brown')
        plt.plot(SR_warm_100km, POD_warm_100km, color='green', label='100km boundary')
        plt.plot(SR_warm_100km[np.where(CSI_warm_100km == np.max(CSI_warm_100km))], POD_warm_100km[np.where(CSI_warm_100km == np.max(CSI_warm_100km))], color='green', marker='*', markersize=9)
        plt.text(0.01, 0.13, s=str('100km: %.4f' % np.max(CSI_warm_100km)), color='green')
        plt.text(0.01, 0.17, s='CSI values', style='oblique')
        plt.legend(loc='upper right')
        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.title("Model %d Performance for Warm Fronts" % model_number)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.show()
        plt.savefig("%s/model_%d/model_%d_performance_warm_%d_%d.png" % (model_dir, model_number, model_number, num_images, image_trim),bbox_inches='tight')
        plt.close()

        if front_types == 'ALL':
            POD_stationary_25km = stats_25km['tp_stationary']/(stats_25km['tp_stationary'] + stats_25km['fn_stationary'])
            SR_stationary_25km = stats_25km['tp_stationary']/(stats_25km['tp_stationary'] + stats_25km['fp_stationary'])
            CSI_stationary_25km = stats_25km['tp_stationary']/(stats_25km['tp_stationary'] + stats_25km['fp_stationary'] + stats_25km['fn_stationary'])
            POD_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fn_stationary'])
            SR_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fp_stationary'])
            CSI_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fp_stationary'] + stats_100km['fn_stationary'])
            POD_occluded_25km = stats_25km['tp_occluded']/(stats_25km['tp_occluded'] + stats_25km['fn_occluded'])
            SR_occluded_25km = stats_25km['tp_occluded']/(stats_25km['tp_occluded'] + stats_25km['fp_occluded'])
            CSI_occluded_25km = stats_25km['tp_occluded']/(stats_25km['tp_occluded'] + stats_25km['fp_occluded'] + stats_25km['fn_occluded'])
            POD_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fn_occluded'])
            SR_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fp_occluded'])
            CSI_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fp_occluded'] + stats_100km['fn_occluded'])
            POD_dryline_25km = stats_25km['tp_dryline']/(stats_25km['tp_dryline'] + stats_25km['fn_dryline'])
            SR_dryline_25km = stats_25km['tp_dryline']/(stats_25km['tp_dryline'] + stats_25km['fp_dryline'])
            CSI_dryline_25km = stats_25km['tp_dryline']/(stats_25km['tp_dryline'] + stats_25km['fp_dryline'] + stats_25km['fn_dryline'])
            POD_dryline_100km = stats_100km['tp_dryline']/(stats_100km['tp_dryline'] + stats_100km['fn_dryline'])
            SR_dryline_100km = stats_100km['tp_dryline']/(stats_100km['tp_dryline'] + stats_100km['fp_dryline'])
            CSI_dryline_100km = stats_100km['tp_dryline']/(stats_100km['tp_dryline'] + stats_100km['fp_dryline'] + stats_100km['fn_dryline'])

            plt.figure()
            plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
            plt.colorbar(label='Critical Success Index (CSI)')
            plt.plot(SR_stationary_25km, POD_stationary_25km, 'r', label='25km boundary')
            plt.plot(SR_stationary_25km[np.where(CSI_stationary_25km == np.max(CSI_stationary_25km))], POD_stationary_25km[np.where(CSI_stationary_25km == np.max(CSI_stationary_25km))], color='red', marker='*', markersize=9)
            plt.text(SR_stationary_25km[np.where(CSI_stationary_25km == np.max(CSI_stationary_25km))]+0.01, POD_stationary_25km[np.where(CSI_stationary_25km == np.max(CSI_stationary_25km))], s=str('%.4f' % np.max(CSI_stationary_25km)), color='red')
            plt.plot(SR_stationary_100km, POD_stationary_100km, color='green', label='100km boundary')
            plt.plot(SR_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))], POD_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))], color='green', marker='*', markersize=9)
            plt.text(SR_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))]+0.01, POD_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))], s=str('%.4f' % np.max(CSI_stationary_100km)), color='green')
            plt.legend()
            plt.xlabel("Success Ratio (1 - FAR)")
            plt.ylabel("Probability of Detection (POD)")
            plt.title("Model %d Performance for Stationary Fronts" % model_number)
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.show()
            plt.savefig("%s/model_%d/model_%d_performance_stationary_%d_%d.png" % (model_dir, model_number, model_number, num_images, image_trim),bbox_inches='tight')
            plt.close()

            plt.figure()
            plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
            plt.colorbar(label='Critical Success Index (CSI)')
            plt.plot(SR_occluded_25km, POD_occluded_25km, 'r', label='25km boundary')
            plt.plot(SR_occluded_25km[np.where(CSI_occluded_25km == np.max(CSI_occluded_25km))], POD_occluded_25km[np.where(CSI_occluded_25km == np.max(CSI_occluded_25km))], color='red', marker='*', markersize=9)
            plt.text(SR_occluded_25km[np.where(CSI_occluded_25km == np.max(CSI_occluded_25km))]+0.01, POD_occluded_25km[np.where(CSI_occluded_25km == np.max(CSI_occluded_25km))], s=str('%.4f' % np.max(CSI_occluded_25km)), color='red')
            plt.plot(SR_occluded_100km, POD_occluded_100km, color='green', label='100km boundary')
            plt.plot(SR_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))], POD_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))], color='green', marker='*', markersize=9)
            plt.text(SR_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))]+0.01, POD_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))], s=str('%.4f' % np.max(CSI_occluded_100km)), color='green')
            plt.legend()
            plt.xlabel("Success Ratio (1 - FAR)")
            plt.ylabel("Probability of Detection (POD)")
            plt.title("Model %d Performance for Occluded Fronts" % model_number)
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.show()
            plt.savefig("%s/model_%d/model_%d_performance_occluded_%d_%d.png" % (model_dir, model_number, model_number, num_images, image_trim),bbox_inches='tight')
            plt.close()

            plt.figure()
            plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
            plt.colorbar(label='Critical Success Index (CSI)')
            plt.plot(SR_dryline_25km, POD_dryline_25km, 'r', label='25km boundary')
            plt.plot(SR_dryline_25km[np.where(CSI_dryline_25km == np.max(CSI_dryline_25km))], POD_dryline_25km[np.where(CSI_dryline_25km == np.max(CSI_dryline_25km))], color='red', marker='*', markersize=9)
            plt.text(SR_dryline_25km[np.where(CSI_dryline_25km == np.max(CSI_dryline_25km))]+0.01, POD_dryline_25km[np.where(CSI_dryline_25km == np.max(CSI_dryline_25km))], s=str('%.4f' % np.max(CSI_dryline_25km)), color='red')
            plt.plot(SR_dryline_100km, POD_dryline_100km, color='green', label='100km boundary')
            plt.plot(SR_dryline_100km[np.where(CSI_dryline_100km == np.max(CSI_dryline_100km))], POD_dryline_100km[np.where(CSI_dryline_100km == np.max(CSI_dryline_100km))], color='green', marker='*', markersize=9)
            plt.text(SR_dryline_100km[np.where(CSI_dryline_100km == np.max(CSI_dryline_100km))]+0.01, POD_dryline_100km[np.where(CSI_dryline_100km == np.max(CSI_dryline_100km))], s=str('%.4f' % np.max(CSI_dryline_100km)), color='green')
            plt.legend()
            plt.xlabel("Success Ratio (1 - FAR)")
            plt.ylabel("Probability of Detection (POD)")
            plt.title("Model %d Performance for Dryline Fronts" % model_number)
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.show()
            plt.savefig("%s/model_%d/model_%d_performance_dryline_%d_%d.png" % (model_dir, model_number, model_number, num_images, image_trim),bbox_inches='tight')
            plt.close()

    elif front_types == 'SFOF':
        POD_stationary_25km = stats_25km['tp_stationary']/(stats_25km['tp_stationary'] + stats_25km['fn_stationary'])
        SR_stationary_25km = stats_25km['tp_stationary']/(stats_25km['tp_stationary'] + stats_25km['fp_stationary'])
        CSI_stationary_25km = stats_25km['tp_stationary']/(stats_25km['tp_stationary'] + stats_25km['fp_stationary'] + stats_25km['fn_stationary'])
        POD_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fn_stationary'])
        SR_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fp_stationary'])
        CSI_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fp_stationary'] + stats_100km['fn_stationary'])
        POD_occluded_25km = stats_25km['tp_occluded']/(stats_25km['tp_occluded'] + stats_25km['fn_occluded'])
        SR_occluded_25km = stats_25km['tp_occluded']/(stats_25km['tp_occluded'] + stats_25km['fp_occluded'])
        CSI_occluded_25km = stats_25km['tp_occluded']/(stats_25km['tp_occluded'] + stats_25km['fp_occluded'] + stats_25km['fn_occluded'])
        POD_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fn_occluded'])
        SR_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fp_occluded'])
        CSI_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fp_occluded'] + stats_100km['fn_occluded'])

        plt.figure()
        plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
        plt.colorbar(label='Critical Success Index (CSI)')
        plt.plot(SR_stationary_25km, POD_stationary_25km, 'r', label='25km boundary')
        plt.plot(SR_stationary_25km[np.where(CSI_stationary_25km == np.max(CSI_stationary_25km))], POD_stationary_25km[np.where(CSI_stationary_25km == np.max(CSI_stationary_25km))], color='red', marker='*', markersize=9)
        plt.text(SR_stationary_25km[np.where(CSI_stationary_25km == np.max(CSI_stationary_25km))]+0.01, POD_stationary_25km[np.where(CSI_stationary_25km == np.max(CSI_stationary_25km))], s=str('%.4f' % np.max(CSI_stationary_25km)), color='red')
        plt.plot(SR_stationary_100km, POD_stationary_100km, color='green', label='100km boundary')
        plt.plot(SR_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))], POD_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))], color='green', marker='*', markersize=9)
        plt.text(SR_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))]+0.01, POD_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))], s=str('%.4f' % np.max(CSI_stationary_100km)), color='green')
        plt.legend()
        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.title("Model %d Performance for Stationary Fronts" % model_number)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.show()
        plt.savefig("%s/model_%d/model_%d_performance_stationary_%d_%d.png" % (model_dir, model_number, model_number, num_images, image_trim),bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
        plt.colorbar(label='Critical Success Index (CSI)')
        plt.plot(SR_occluded_25km, POD_occluded_25km, 'r', label='25km boundary')
        plt.plot(SR_occluded_25km[np.where(CSI_occluded_25km == np.max(CSI_occluded_25km))], POD_occluded_25km[np.where(CSI_occluded_25km == np.max(CSI_occluded_25km))], color='red', marker='*', markersize=9)
        plt.text(SR_occluded_25km[np.where(CSI_occluded_25km == np.max(CSI_occluded_25km))]+0.01, POD_occluded_25km[np.where(CSI_occluded_25km == np.max(CSI_occluded_25km))], s=str('%.4f' % np.max(CSI_occluded_25km)), color='red')
        plt.plot(SR_occluded_100km, POD_occluded_100km, color='green', label='100km boundary')
        plt.plot(SR_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))], POD_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))], color='green', marker='*', markersize=9)
        plt.text(SR_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))]+0.01, POD_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))], s=str('%.4f' % np.max(CSI_occluded_100km)), color='green')
        plt.legend()
        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.title("Model %d Performance for Occluded Fronts" % model_number)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.show()
        plt.savefig("%s/model_%d/model_%d_performance_occluded_%d_%d.png" % (model_dir, model_number, model_number, num_images, image_trim),bbox_inches='tight')
        plt.close()


def prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, num_images, image_trim):
    """
    Function that uses generated predictions to make probability maps along with the 'true' fronts and saves out the
    subplots.

    Parameters
    ----------
    fronts: DataArray
        Xarray DataArray containing the 'true' front data.
    probs_ds: Dataset
        Xarray dataset containing prediction (probability) data for warm and cold fronts.
    time: str
        Timestring for the prediction plot title.
    model_number: int
        Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        Main directory for the models.
    front_types: str
        Fronts in the data.
    pixel_expansion: int
        Number of pixels to expand the fronts by in all directions.
    num_images: int
        Number of images to stitch together for the final probability maps.
    image_trim: int
        Number of pixels to trim each image by in the longitude dimension before averaging the overlapping pixels.
    """
    extent = np.array([220, 300, 29, 53])
    crs = ccrs.LambertConformal(central_longitude=250)

    # Create custom colorbar for the different front types of the 'truth' plot
    cmap = mpl.colors.ListedColormap(["white","blue","red",'green','purple','orange'], name='from_list', N=None)
    norm = mpl.colors.Normalize(vmin=0,vmax=6)

    cold_norm = mpl.colors.Normalize(vmin=0, vmax=0.6)
    warm_norm = mpl.colors.Normalize(vmin=0, vmax=0.6)
    stationary_norm = mpl.colors.Normalize(vmin=0, vmax=0.6)
    occluded_norm = mpl.colors.Normalize(vmin=0, vmax=0.6)
    dryline_norm = mpl.colors.Normalize(vmin=0.02, vmax=0.6)

    if pixel_expansion == 1:
        fronts = ope(fronts)  # 1-pixel expansion
    elif pixel_expansion == 2:
        fronts = ope(ope(fronts))  # 2-pixel expansion

    probs_ds = xr.where(probs_ds > 0.02, probs_ds, float("NaN"))

    if front_types == 'CFWF':
        fig, axarr = plt.subplots(1, 2, figsize=(20, 5), subplot_kw={'projection': crs}, gridspec_kw={'width_ratios': [1,1.3]})
        axlist = axarr.flatten()
        for ax in axlist:
            fplot.plot_background(ax, extent)
        fronts.identifier.plot(ax=axlist[0], cmap=cmap, norm=norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[0].title.set_text("%s Truth" % time)
        probs_ds.cold_probs.plot(ax=axlist[1], cmap="Blues", norm=cold_norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        probs_ds.warm_probs.plot(ax=axlist[1], cmap="Reds", norm=warm_norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[1].title.set_text("%s Front probabilities (images=%d, trim=%d)" % (time, num_images, image_trim))
        plt.savefig('%s/model_%d/predictions/model_%d_%s_plot_%dimages_%dtrim.png' % (model_dir, model_number, model_number,
            time, num_images, image_trim), bbox_inches='tight', dpi=300)
        plt.close()

    elif front_types == 'SFOF':
        fig, axarr = plt.subplots(2, 1, figsize=(12, 14), subplot_kw={'projection': crs})
        axlist = axarr.flatten()
        for ax in axlist:
            fplot.plot_background(ax, extent)
        fronts.identifier.plot(ax=axlist[0], cmap=cmap, norm=norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[0].title.set_text("%s Truth" % time)
        probs_ds.stationary_probs.plot(ax=axlist[1], cmap='Greens', norm=stationary_norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[1].title.set_text("%s SF probability" % time)
        probs_ds.occluded_probs.plot(ax=axlist[1], cmap='Purples', norm=occluded_norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[1].title.set_text("%s Front probabilities (images=%d, trim=%d)" % (time, num_images, image_trim))
        plt.savefig('%s/model_%d/predictions/model_%d_%s_plot_%dimages_%dtrim.png' % (model_dir, model_number, model_number,
            time, num_images, image_trim), bbox_inches='tight', dpi=300)
        plt.close()

    elif front_types == 'DL':
        fig, axarr = plt.subplots(2, 1, figsize=(12, 14), subplot_kw={'projection': crs})
        axlist = axarr.flatten()
        for ax in axlist:
            fplot.plot_background(ax, extent)
        fronts.identifier.plot(ax=axlist[0], cmap=cmap, norm=norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[0].title.set_text("%s Truth" % time)
        probs_ds.dryline_probs.plot(ax=axlist[1], cmap='Oranges', norm=dryline_norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[1].title.set_text("%s Front probabilities (images=%d, trim=%d)" % (time, num_images, image_trim))
        plt.savefig('%s/model_%d/predictions/model_%d_%s_plot_%dimages_%dtrim.png' % (model_dir, model_number, model_number,
            time, num_images, image_trim), bbox_inches='tight', dpi=300)
        plt.close()

    elif front_types == 'ALL':
        fig, axarr = plt.subplots(2, 1, figsize=(12, 14), subplot_kw={'projection': crs})
        axlist = axarr.flatten()
        for ax in axlist:
            fplot.plot_background(ax, extent)
        fronts.identifier.plot(ax=axlist[0], cmap=cmap, norm=norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[0].title.set_text("%s Truth" % time)
        probs_ds.cold_probs.plot(ax=axlist[1], cmap='Blues', norm=cold_norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        probs_ds.warm_probs.plot(ax=axlist[1], cmap='Reds', norm=warm_norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        probs_ds.stationary_probs.plot(ax=axlist[1], cmap='Greens', norm=stationary_norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        probs_ds.occluded_probs.plot(ax=axlist[1], cmap='Purples', norm=occluded_norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        probs_ds.dryline_probs.plot(ax=axlist[1], cmap='Oranges', norm=dryline_norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[1].title.set_text("%s Front probabilities (images=%d, trim=%d)" % (time, num_images, image_trim))
        plt.savefig('%s/model_%d/predictions/model_%d_%s_plot_%dimages_%dtrim.png' % (model_dir, model_number, model_number,
            time, num_images, image_trim), bbox_inches='tight', dpi=300)
        plt.close()


def learning_curve(include_validation_plots, model_number, model_dir, loss, fss_mask_size, fss_c, metric):
    """
    Function that plots learning curves for the specified model.

    Parameters
    ----------
    include_validation_plots: bool
        Include validation data in learning curve plots?
    model_number: int
        Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        Main directory for the models.
    loss: str
        Loss function for the U-Net.
    fss_mask_size: int
        Size of the mask for the FSS loss function.
    fss_c: float
        C hyperparameter for the FSS loss' sigmoid function.
    metric: str
        Metric used for evaluating the U-Net during training.
    """
    with open("%s/model_%d/model_%d_history.csv" % (model_dir, model_number, model_number), 'rb') as f:
        history = pd.read_csv(f)

    if loss == 'fss':
        loss_title = 'Fractions Skill Score (mask=%d, c=%.1f)' % (fss_mask_size, fss_c)
    elif loss == 'bss':
        loss_title = 'Brier Skill Score'
    elif loss == 'cce':
        loss_title = 'Categorical Cross-Entropy'
    elif loss == 'dice':
        loss_title = 'Dice coefficient'
    elif loss == 'tversky':
        loss_title = 'Tversky coefficient'

    if metric == 'fss':
        metric_title = 'Fractions Skill Score (mask=%d, c=%.1f)' % (fss_mask_size, fss_c)
        metric_string = 'FSS_loss'
    elif metric == 'bss':
        metric_title = 'Brier Skill Score'
        metric_string = 'brier_skill_score'
    elif metric == 'auc':
        metric_title = 'Area Under the Curve'
        metric_string = 'auc'
    elif metric == 'dice':
        metric_title = 'Dice coefficient'
        metric_string = 'dice'
    elif metric == 'tversky':
        metric_title = 'Tversky coefficient'
        metric_string = 'tversky'

    if include_validation_plots is True:
        nrows = 2
        figsize = (14,12)
    else:
        nrows = 1
        figsize = (14,7)

    plt.subplots(nrows, 2, figsize=figsize, dpi=500)

    plt.subplot(nrows, 2, 1)
    plt.title("Training loss: %s" % loss_title)
    plt.grid()

    if 'softmax_loss' in history:
        plt.plot(history['softmax_loss'], label='Encoder 6')
        plt.plot(history['softmax_1_loss'], label='Decoder 5')
        plt.plot(history['softmax_2_loss'], label='Decoder 4')
        plt.plot(history['softmax_3_loss'], label='Decoder 3')
        plt.plot(history['softmax_4_loss'], label='Decoder 2')
        plt.plot(history['softmax_5_loss'], label='Decoder 1 (final)', color='black')
        plt.plot(history['loss'], label='total', color='black')
    elif 'unet_output_final_activation_loss' in history:
        plt.plot(history['unet_output_sup0_activation_loss'], label='sup0')
        plt.plot(history['unet_output_sup1_activation_loss'], label='sup1')
        plt.plot(history['unet_output_sup2_activation_loss'], label='sup2')
        plt.plot(history['unet_output_sup3_activation_loss'], label='sup3')
        plt.plot(history['unet_output_sup4_activation_loss'], label='sup4')
        plt.plot(history['unet_output_final_activation_loss'], label='final')
        plt.plot(history['loss'], label='total', color='black')
    else:
        plt.plot(history['loss'], color='black')

    plt.legend(loc='best')
    plt.xlim(xmin=0)
    plt.xlabel('Epochs')
    plt.ylim(ymin=1e-6, ymax=1e-4)  # Limits of the loss function graph, adjust these as needed
    plt.yscale('log')  # Turns y-axis into a logarithmic scale. Useful if loss functions appear as very sharp curves.

    plt.subplot(nrows, 2, 2)
    plt.title("Training metric: %s" % metric_title)
    plt.grid()

    if 'softmax_loss' in history:
        plt.plot(history['softmax_%s' % metric_string], label='Encoder 6')
        plt.plot(history['softmax_1_%s' % metric_string], label='Decoder 5')
        plt.plot(history['softmax_2_%s' % metric_string], label='Decoder 4')
        plt.plot(history['softmax_3_%s' % metric_string], label='Decoder 3')
        plt.plot(history['softmax_4_%s' % metric_string], label='Decoder 2')
        plt.plot(history['softmax_5_%s' % metric_string], label='Decoder 1 (final)', color='black')
    elif 'unet_output_final_activation_loss' in history:
        plt.plot(history['unet_output_sup0_activation_%s' % metric_string], label='sup0')
        plt.plot(history['unet_output_sup1_activation_%s' % metric_string], label='sup1')
        plt.plot(history['unet_output_sup2_activation_%s' % metric_string], label='sup2')
        plt.plot(history['unet_output_sup3_activation_%s' % metric_string], label='sup3')
        plt.plot(history['unet_output_sup4_activation_%s' % metric_string], label='sup4')
        plt.plot(history['unet_output_final_activation_%s' % metric_string], label='final', color='black')
    else:
        plt.plot(history[metric_string], 'r')

    plt.legend(loc='best')
    plt.xlim(xmin=0)
    plt.xlabel('Epochs')
    plt.ylim(ymin=1e-3, ymax=1e-1)  # Limits of the metric graph, adjust as needed
    plt.yscale('log')

    if include_validation_plots is True:
        plt.subplot(nrows, 2, 3)
        plt.title("Validation loss: %s" % loss_title)
        plt.grid()

        if 'softmax_loss' in history:
            plt.plot(history['val_softmax_loss'], label='Encoder 6')
            plt.plot(history['val_softmax_1_loss'], label='Decoder 5')
            plt.plot(history['val_softmax_2_loss'], label='Decoder 4')
            plt.plot(history['val_softmax_3_loss'], label='Decoder 3')
            plt.plot(history['val_softmax_4_loss'], label='Decoder 2')
            plt.plot(history['val_softmax_5_loss'], label='Decoder 1 (final)', color='black')
            plt.plot(history['val_loss'], label='total', color='black')
        elif 'unet_output_final_activation_loss' in history:
            plt.plot(history['val_unet_output_sup0_activation_loss'], label='sup0')
            plt.plot(history['val_unet_output_sup1_activation_loss'], label='sup1')
            plt.plot(history['val_unet_output_sup2_activation_loss'], label='sup2')
            plt.plot(history['val_unet_output_sup3_activation_loss'], label='sup3')
            plt.plot(history['val_unet_output_sup4_activation_loss'], label='sup4')
            plt.plot(history['val_unet_output_final_activation_loss'], label='final')
            plt.plot(history['val_loss'], label='total', color='black')
        else:
            plt.plot(history['val_loss'], color='black')

        plt.legend(loc='best')
        plt.xlim(xmin=0)
        plt.xlabel('Epochs')
        plt.ylim(ymin=1e-6, ymax=1e-4)  # Limits of the loss function graph, adjust these as needed
        plt.yscale('log')  # Turns y-axis into a logarithmic scale. Useful if loss functions appear as very sharp curves.

        plt.subplot(nrows, 2, 4)
        plt.title("Validation metric: %s" % metric_title)
        plt.grid()

        if 'softmax_loss' in history:
            plt.plot(history['val_softmax_%s' % metric_string], label='Encoder 6')
            plt.plot(history['val_softmax_1_%s' % metric_string], label='Decoder 5')
            plt.plot(history['val_softmax_2_%s' % metric_string], label='Decoder 4')
            plt.plot(history['val_softmax_3_%s' % metric_string], label='Decoder 3')
            plt.plot(history['val_softmax_4_%s' % metric_string], label='Decoder 2')
            plt.plot(history['val_softmax_5_%s' % metric_string], label='Decoder 1 (final)', color='black')
        elif 'unet_output_final_activation_loss' in history:
            plt.plot(history['val_unet_output_sup0_activation_%s' % metric_string], label='sup0')
            plt.plot(history['val_unet_output_sup1_activation_%s' % metric_string], label='sup1')
            plt.plot(history['val_unet_output_sup2_activation_%s' % metric_string], label='sup2')
            plt.plot(history['val_unet_output_sup3_activation_%s' % metric_string], label='sup3')
            plt.plot(history['val_unet_output_sup4_activation_%s' % metric_string], label='sup4')
            plt.plot(history['val_unet_output_final_activation_%s' % metric_string], label='final', color='black')
        else:
            plt.plot(history['val_%s' % metric_string], 'r')

        plt.legend(loc='best')
        plt.xlim(xmin=0)
        plt.xlabel('Epochs')
        plt.ylim(ymin=1e-3, ymax=1e-1)  # Limits of the metric graph, adjust as needed
        plt.yscale('log')

    plt.savefig("%s/model_%d/model_%d_learning_curve.png" % (model_dir, model_number, model_number), bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    """ Main arguments """
    parser.add_argument('--calculate_performance_stats', type=bool, required=False,
                        help='Are you calculating performance stats for a model?')
    parser.add_argument('--day', type=int, required=False, help='Day for the prediction.')
    parser.add_argument('--domain', type=str, required=False, help='Domain of the data.')
    parser.add_argument('--file_dimensions', type=int, nargs=2, required=False,
                        help='Dimensions of the file size. Two integers need to be passed.')
    parser.add_argument('--find_matches', type=bool, required=False, help='Find matches for stitching predictions?')
    parser.add_argument('--front_types', type=str, required=False,
                        help='Front format of the file. If your files contain warm and cold fronts, pass this argument'
                             'as CFWF. If your files contain only drylines, pass this argument as DL. If your files '
                             'contain all fronts, pass this argument as ALL.')
    parser.add_argument('--fss_c', type=float, required=False, help="C hyperparameter for the FSS loss' sigmoid function.")
    parser.add_argument('--fss_mask_size', type=int, required=False, help='Mask size for the FSS loss function.')
    parser.add_argument('--hour', type=int, required=False, help='Hour for the prediction.')
    parser.add_argument('--image_trim', type=int, required=False,
                        help='Number of pixels to trim the images for stitching before averaging overlapping pixels.')
    parser.add_argument('--include_validation_plots', type=bool, required=False,
                        help='Include validation data in learning curve plots?')
    parser.add_argument('--learning_curve', type=bool, required=False, help='Plot learning curve?')
    parser.add_argument('--longitude_domain_length', type=int, required=False,
                        help='Number of pixels in the longitude dimension of the final stitched map.')
    parser.add_argument('--loss', type=str, required=False, help='Loss function used for training the U-Net.')
    parser.add_argument('--make_prediction', type=bool, required=False,
                        help='Make a prediction for a specific date and time?')
    parser.add_argument('--make_random_predictions', type=bool, required=False, help='Generate prediction plots?')
    parser.add_argument('--metric', type=str, required=False, help='Metric used for evaluating the U-Net during training.')
    parser.add_argument('--model_longitude_length', type=int, required=False,
                        help="Length of the longitude dimension of the model's prediction.")
    parser.add_argument('--model_dir', type=str, required=False, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, required=False, help='Model number.')
    parser.add_argument('--month', type=int, required=False, help='Month for the prediction.')
    parser.add_argument('--num_dimensions', type=int, required=False,
                        help='Number of dimensions of the U-Net convolutions, maxpooling, and upsampling. (2 or 3)')
    parser.add_argument('--num_images', type=int, required=False, help='Number of images to stitch together for each prediction.')
    parser.add_argument('--num_variables', type=int, required=False, help='Number of variables in the variable datasets.')
    parser.add_argument('--normalization_method', type=int, required=False,
                        help='Normalization method for the data. 0 - No normalization, 1 - Min-max normalization, '
                             '2 - Mean normalization')
    parser.add_argument('--pixel_expansion', type=int, required=False, help='Number of pixels to expand the fronts by.')
    parser.add_argument('--plot_performance_diagrams', type=bool, required=False, help='Plot performance diagrams for a model?')
    parser.add_argument('--predictions', type=int, required=False, help='Number of predictions to make.')
    parser.add_argument('--test_years', type=int, nargs="+", required=False,
                        help='Test years for cross-validating the current model or making predictions.')
    parser.add_argument('--year', type=int, required=False, help='Year for the prediction.')

    args = parser.parse_args()
    provided_arguments = vars(args)

    if args.loss == 'fss':
        required_arguments = ['fss_c', 'fss_mask_size']
        print("Checking arguments for FSS loss function....", end='')
        check_arguments(provided_arguments, required_arguments)

    if args.calculate_performance_stats is True:
        required_arguments = ['domain', 'file_dimensions', 'front_types', 'image_trim', 'longitude_domain_length',
                              'loss', 'metric', 'model_dir', 'model_number', 'normalization_method', 'num_dimensions',
                              'num_images', 'num_variables', 'pixel_expansion']
        print("Checking arguments for 'calculate_performance_stats'....", end='')
        check_arguments(provided_arguments, required_arguments)
        calculate_performance_stats(args.model_number, args.model_dir, args.num_variables, args.num_dimensions, args.front_types,
            args.domain, args.file_dimensions, args.test_years, args.normalization_method, args.loss, args.fss_mask_size,
            args.fss_c, args.pixel_expansion, args.metric, args.num_images, args.longitude_domain_length, args.image_trim)

    if args.find_matches is True:
        required_arguments = ['longitude_domain_length', 'model_longitude_length']
        print("Checking arguments for 'find_matches'....", end='')
        check_arguments(provided_arguments, required_arguments)
        find_matches_for_domain(args.longitude_domain_length, args.model_longitude_length)

    if args.learning_curve is True:
        required_arguments = ['include_validation_plots', 'loss', 'metric', 'model_dir', 'model_number']
        print("Checking arguments for 'learning_curve'....", end='')
        check_arguments(provided_arguments, required_arguments)
        learning_curve(args.include_validation_plots, args.model_number, args.model_dir, args.loss, args.fss_mask_size,
                       args.fss_c, args.metric)

    if args.make_prediction is True:
        required_arguments = ['model_number', 'model_dir', 'num_variables', 'num_dimensions', 'front_types', 'domain',
            'file_dimensions', 'normalization_method', 'loss', 'pixel_expansion', 'metric', 'num_images',
            'longitude_domain_length', 'image_trim', 'year', 'month', 'day', 'hour']
        print("Checking arguments for 'make_prediction'....", end='')
        check_arguments(provided_arguments, required_arguments)
        front_files, variable_files = fm.load_file_lists(args.num_variables, args.front_types, args.domain,
            args.file_dimensions)
        make_prediction(args.model_number, args.model_dir, front_files, variable_files, args.normalization_method,
            args.loss, args.fss_mask_size, args.fss_c, args.front_types, args.pixel_expansion, args.metric, args.num_dimensions,
            args.num_images, args.longitude_domain_length, args.image_trim, args.year, args.month, args.day, args.hour)

    if args.make_random_predictions is True:
        required_arguments = ['model_number', 'model_dir', 'num_variables', 'num_dimensions', 'front_types', 'domain',
            'file_dimensions', 'normalization_method', 'loss', 'pixel_expansion', 'metric', 'num_images',
            'longitude_domain_length', 'image_trim', 'predictions']
        print("Checking arguments for 'make_random_predictions'....", end='')
        check_arguments(provided_arguments, required_arguments)
        if args.test_years is not None:
            front_files, variable_files = fm.load_test_files(args.num_variables, args.front_types, args.domain, args.file_dimensions, args.test_years)
        else:
            front_files, variable_files = fm.load_file_lists(args.num_variables, args.front_types, args.domain,
                args.file_dimensions)
        make_random_predictions(args.model_number, args.model_dir, front_files, variable_files, args.predictions,
            args.normalization_method, args.loss, args.fss_mask_size, args.fss_c, args.front_types, args.pixel_expansion,
            args.metric, args.num_dimensions, args.num_images, args.longitude_domain_length, args.image_trim)

    if args.plot_performance_diagrams is True:
        required_arguments = ['model_number', 'model_dir', 'front_types', 'num_images', 'image_trim']
        print("Checking arguments for 'plot_performance_diagrams'....", end='')
        check_arguments(provided_arguments, required_arguments)
        plot_performance_diagrams(args.model_dir, args.model_number, args.front_types, args.num_images, args.image_trim)
