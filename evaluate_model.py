"""
Functions used for evaluating a U-Net model.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 12/11/2021 8:02 PM CST
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
from errors import check_arguments, ArgumentConflictError
import pickle
import matplotlib as mpl
from matplotlib import cm  # Here we explicitly import the cm module to suppress a PyCharm bug
from expand_fronts import one_pixel_expansion as ope
from variables import normalize


def add_image_to_map(stitched_map_probs, image_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
    model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing,
    lon_pixels_per_image, lat_pixels_per_image):
    """
    Add prediction to the stitched map

    Parameters
    ----------
    domain_images_lat: number of images along the latitude dimension of the domain
    domain_images_lon: number of images along the longitude dimension of the domain
    domain_trim_lat: number of pixels by which the images will be trimmed along the latitude dimension before taking the maximum of overlapping pixels
    domain_trim_lon: number of pixels by which the images will be trimmed along the longitude dimension before taking the maximum of overlapping pixels
    image_probs: array of front probabilities for the current prediction/image
    lat_image: integer that represents the current image number along the latitude dimension
    lat_image_spacing: number of pixels between each image along the latitude dimension
    lat_pixels_per_image: number of pixels along the latitude dimension of each image
    lon_image: integer that represents the current image number along the longitude dimension
    lon_image_spacing: number of pixels between each image along the longitude dimension
    lon_pixels_per_image: number of pixels along the longitude dimension of each image
    map_created: boolean that declares whether or not the final map has been completed
    model_length_lat: number of pixels along the latitude dimension of the model predictions
    model_length_lon: number of pixels along the longitude dimension of the model predictions
    stitched_map_probs: array of front probabilities for the final map
    
    
    Returns
    -------
    map_created: boolean that declares whether or not the final map has been completed
    stitched_map_probs: array of front probabilities for the final map
    """
    if lon_image == 0:  # If the image is on the western edge of the domain
        if lat_image == 0:  # If the image is on the northern edge of the domain
            # Add first image to map
            stitched_map_probs[0: model_length_lon - domain_trim_lon, 0: model_length_lat - domain_trim_lat] = \
                image_probs[domain_trim_lon: model_length_lon, domain_trim_lat: model_length_lat]

            if domain_images_lon == 1 and domain_images_lat == 1:
                map_created = True

        elif lat_image != domain_images_lat - 1:  # If the image is not on the northern nor the southern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant latitude
            stitched_map_probs[0: model_length_lon - domain_trim_lon, int(lat_image*lat_image_spacing):int((lat_image-1)*lat_image_spacing) + lat_pixels_per_image] = \
                np.maximum(stitched_map_probs[0: model_length_lon - domain_trim_lon, int(lat_image*lat_image_spacing):int((lat_image-1)*lat_image_spacing) + lat_pixels_per_image],
                           image_probs[domain_trim_lon: model_length_lon, domain_trim_lat:domain_trim_lat + lat_pixels_per_image - lat_image_spacing])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[0: model_length_lon - domain_trim_lon, int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image:lat_image_spacing * lat_image + lat_pixels_per_image] = \
                image_probs[domain_trim_lon: model_length_lon, domain_trim_lat + lat_pixels_per_image - lat_image_spacing:domain_trim_lat + lat_pixels_per_image]

            if domain_images_lon == 1 and domain_images_lat == 2:
                map_created = True

        else:  # If the image is on the southern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant latitude
            stitched_map_probs[0: model_length_lon - domain_trim_lon, int(lat_image*lat_image_spacing):int((lat_image-1)*lat_image_spacing) + lat_pixels_per_image] = \
                np.maximum(stitched_map_probs[domain_trim_lon: model_length_lon, int(lat_image*lat_image_spacing):int((lat_image-1)*lat_image_spacing) + lat_pixels_per_image],
                           image_probs[domain_trim_lon: model_length_lon, domain_trim_lat:domain_trim_lat + lat_pixels_per_image - lat_image_spacing])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[0: model_length_lon - domain_trim_lon, int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image:] = \
                image_probs[domain_trim_lon: model_length_lon, domain_trim_lat + lat_pixels_per_image - lat_image_spacing:model_length_lat-domain_trim_lat]

            if domain_images_lon == 1 and domain_images_lat > 2:
                map_created = True

    elif lon_image != domain_images_lon - 1:  # If the image is not on the western nor the eastern edge of the domain
        if lat_image == 0:  # If the image is on the northern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant longitude
            stitched_map_probs[int(lon_image*lon_image_spacing):int((lon_image-1)*lon_image_spacing) + lon_pixels_per_image, 0: model_length_lat - domain_trim_lat] = \
                np.maximum(stitched_map_probs[int(lon_image*lon_image_spacing):int((lon_image-1)*lon_image_spacing) + lon_pixels_per_image, 0: model_length_lat - domain_trim_lat],
                           image_probs[domain_trim_lon:domain_trim_lon + lon_pixels_per_image - lon_image_spacing, domain_trim_lat: model_length_lat])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image:lon_image_spacing * lon_image + lon_pixels_per_image, 0: model_length_lat - domain_trim_lat] = \
                image_probs[domain_trim_lon + lon_pixels_per_image - lon_image_spacing:domain_trim_lon + lon_pixels_per_image, domain_trim_lat: model_length_lat]

            if domain_images_lon == 2 and domain_images_lat == 1:
                map_created = True

        elif lat_image != domain_images_lat - 1:  # If the image is not on the northern nor the southern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant longitude
            stitched_map_probs[int(lon_image * lon_image_spacing):int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image, int(lat_image*lat_image_spacing)+domain_trim_lat:int(lat_image)*int(lat_image_spacing)+model_length_lat-domain_trim_lat] = \
                np.maximum(stitched_map_probs[int(lon_image * lon_image_spacing):int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image, int(lat_image*lat_image_spacing)+domain_trim_lat:int(lat_image)*int(lat_image_spacing)+model_length_lat-domain_trim_lat],
                           image_probs[domain_trim_lon:domain_trim_lon + lon_pixels_per_image - lon_image_spacing, domain_trim_lat:model_length_lat-domain_trim_lat])

            # Take the maximum of the overlapping pixels along sets of constant latitude
            stitched_map_probs[int(lon_image*lon_image_spacing):int(lon_image*lon_image_spacing)+model_length_lon-domain_trim_lon, int(lat_image * lat_image_spacing):int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image] = \
                np.maximum(stitched_map_probs[int(lon_image*lon_image_spacing):int(lon_image*lon_image_spacing)+model_length_lon-domain_trim_lon, int(lat_image * lat_image_spacing):int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image],
                           image_probs[domain_trim_lon:model_length_lon, domain_trim_lat:domain_trim_lat + lat_pixels_per_image - lat_image_spacing])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image:lon_image_spacing * lon_image + lon_pixels_per_image, int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image:lat_image_spacing * lat_image + lat_pixels_per_image] = \
                image_probs[domain_trim_lon + lon_pixels_per_image - lon_image_spacing:domain_trim_lon + lon_pixels_per_image, domain_trim_lat + lat_pixels_per_image - lat_image_spacing:domain_trim_lat + lat_pixels_per_image]

            if domain_images_lon == 2 and domain_images_lat == 2:
                map_created = True

        else:  # If the image is on the southern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant longitude
            stitched_map_probs[int(lon_image * lon_image_spacing):int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image, int(lat_image*lat_image_spacing):] = \
                np.maximum(stitched_map_probs[int(lon_image * lon_image_spacing):int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image, int(lat_image*lat_image_spacing):],
                           image_probs[domain_trim_lon:domain_trim_lon + lon_pixels_per_image - lon_image_spacing, domain_trim_lat:model_length_lat-domain_trim_lat])

            # Take the maximum of the overlapping pixels along sets of constant latitude
            stitched_map_probs[int(lon_image*lon_image_spacing):int(lon_image*lon_image_spacing)+model_length_lon-domain_trim_lon, int(lat_image * lat_image_spacing):int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image] = \
                np.maximum(stitched_map_probs[int(lon_image*lon_image_spacing):int(lon_image*lon_image_spacing)+model_length_lon-domain_trim_lon, int(lat_image * lat_image_spacing):int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image],
                           image_probs[domain_trim_lon:model_length_lon, domain_trim_lat:domain_trim_lat + lat_pixels_per_image - lat_image_spacing])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image:lon_image_spacing * lon_image + lon_pixels_per_image, int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image:] = \
                image_probs[domain_trim_lon + lon_pixels_per_image - lon_image_spacing:domain_trim_lon + lon_pixels_per_image, domain_trim_lat + lat_pixels_per_image - lat_image_spacing:domain_trim_lat + lat_pixels_per_image]

            if domain_images_lon == 2 and domain_images_lat > 2:
                map_created = True
    else:
        if lat_image == 0:  # If the image is on the northern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant longitude
            stitched_map_probs[int(lon_image * lon_image_spacing):int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image, 0: model_length_lat - domain_trim_lat] = \
                np.maximum(stitched_map_probs[int(lon_image * lon_image_spacing):int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image, 0: model_length_lat - domain_trim_lat],
                           image_probs[domain_trim_lon:domain_trim_lon + lon_pixels_per_image - lon_image_spacing, domain_trim_lat: model_length_lat])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image:, 0: model_length_lat - domain_trim_lat] = \
                image_probs[domain_trim_lon + lon_pixels_per_image - lon_image_spacing:model_length_lon-domain_trim_lon, domain_trim_lat: model_length_lat]

            if domain_images_lon > 2 and domain_images_lat == 1:
                map_created = True

        elif lat_image != domain_images_lat - 1:  # If the image is not on the northern nor the southern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant longitude
            stitched_map_probs[int(lon_image * lon_image_spacing):int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image, int(lat_image*lat_image_spacing)+domain_trim_lat:int(lat_image)*int(lat_image_spacing)+model_length_lat-domain_trim_lat] = \
                np.maximum(stitched_map_probs[int(lon_image * lon_image_spacing):int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image, int(lat_image*lat_image_spacing)+domain_trim_lat:int(lat_image)*int(lat_image_spacing)+model_length_lat-domain_trim_lat], image_probs[domain_trim_lon:domain_trim_lon + lon_pixels_per_image - lon_image_spacing, domain_trim_lat:model_length_lat-domain_trim_lat])

            # Take the maximum of the overlapping pixels along sets of constant latitude
            stitched_map_probs[int(lon_image*lon_image_spacing):, int(lat_image * lat_image_spacing):int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image] = \
                np.maximum(stitched_map_probs[int(lon_image*lon_image_spacing):, int(lat_image * lat_image_spacing):int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image], image_probs[domain_trim_lon:model_length_lon-domain_trim_lon, domain_trim_lat:domain_trim_lat + lat_pixels_per_image - lat_image_spacing])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image:, int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image:lat_image_spacing * lat_image + lat_pixels_per_image] = image_probs[domain_trim_lon + lon_pixels_per_image - lon_image_spacing:model_length_lon-domain_trim_lon, domain_trim_lat + lat_pixels_per_image - lat_image_spacing:domain_trim_lat + lat_pixels_per_image]

            if domain_images_lon > 2 and domain_images_lat == 2:
                map_created = True
        else:  # If the image is on the southern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant longitude
            stitched_map_probs[int(lon_image * lon_image_spacing):, int(lat_image*lat_image_spacing):] = \
                np.maximum(stitched_map_probs[int(lon_image * lon_image_spacing):, int(lat_image*lat_image_spacing):],
                           image_probs[domain_trim_lon:model_length_lon-domain_trim_lon, domain_trim_lat:model_length_lat-domain_trim_lat])

            # Take the maximum of the overlapping pixels along sets of constant latitude
            stitched_map_probs[int(lon_image*lon_image_spacing):, int(lat_image * lat_image_spacing):int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image] = \
                np.maximum(stitched_map_probs[int(lon_image*lon_image_spacing):, int(lat_image * lat_image_spacing):int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image],
                           image_probs[domain_trim_lon:model_length_lon-domain_trim_lon, domain_trim_lat:domain_trim_lat + lat_pixels_per_image - lat_image_spacing])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image:, int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image:lat_image_spacing * lat_image + lat_pixels_per_image] = \
                image_probs[domain_trim_lon + lon_pixels_per_image - lon_image_spacing:model_length_lon-domain_trim_lon, domain_trim_lat + lat_pixels_per_image - lat_image_spacing:domain_trim_lat + lat_pixels_per_image]

            map_created = True

    return stitched_map_probs, map_created


def calculate_performance_stats(model_number, model_dir, num_variables, num_dimensions, front_types, domain, test_years,
    normalization_method, loss, fss_mask_size, fss_c, pixel_expansion, metric, domain_images, domain_lengths, domain_trim,
    random_variable=None):
    """
    Function that calculates the number of true positives, false positives, true negatives, and false negatives on a testing set.

    Parameters
    ----------
    domain: Domain which the front and variable files cover.
    domain_images: Number of images along each dimension of the final stitched map (lon lat).
    domain_lengths: Number of pixels along each dimension of the final stitched map (lon lat).
    domain_trim: Number of pixels to trim each image by along each dimension before taking the maximum of the overlapping pixels.
    front_types: Fronts in the data.
    fss_c: C hyperparameter for the FSS loss' sigmoid function.
    fss_mask_size: Size of the mask for the FSS loss function.
    loss: Loss function for the Unet.
    metric: Metric used for evaluating the U-Net during training.
    model_number: Slurm job number for the model. This is the number in the model's filename.
    model_dir: Main directory for the models.
    normalization_method: Normalization method for the data (described near the end of the script).
    num_dimensions: Number of dimensions for the U-Net's convolutions, maxpooling, and upsampling.
    num_variables: Number of variables in the datasets.
    pixel_expansion: Number of pixels to expand the fronts by in all directions.
    random_variable: Variable to randomize.
    test_years: Years for the test set used for calculating performance stats for cross-validation purposes.
    """

    # If test_years is provided, load files corresponding to those years, otherwise just load all files
    if test_years is not None:
        if front_types == 'ALL_bin':
            front_files, variable_files = fm.load_test_files(num_variables, 'ALL', domain, test_years)
        else:
            front_files, variable_files = fm.load_test_files(num_variables, front_types, domain, test_years)
    else:
        if front_types == 'ALL_bin':
            front_files, variable_files = fm.load_file_lists(num_variables, 'ALL', domain)
        else:
            front_files, variable_files = fm.load_file_lists(num_variables, front_types, domain)
        front_files, variable_files = list(front_files), list(variable_files)
        print("Front file count:", len(front_files))
        print("Variable file count:", len(variable_files))

    model = fm.load_model(model_number, model_dir, loss, fss_mask_size, fss_c, metric, num_dimensions)  #

    n = 0  # Counter for the number of down layers in the model
    map_dim_x = model.layers[0].input_shape[0][1]  # Longitudinal dimension of the U-Net images
    map_dim_y = model.layers[0].input_shape[0][2]  # Latitudinal dimension of the U-Net images
    if num_dimensions == 2:
        for layer in model.layers:
            if layer.__class__.__name__ == 'MaxPooling2D':
                n += 1
        channels = model.layers[0].input_shape[0][3]  # Number of variables used
    elif num_dimensions == 3:
        for layer in model.layers:
            if layer.__class__.__name__ == 'MaxPooling3D':
                n += 1
        channels = model.layers[0].input_shape[0][4]  # Number of variables at each level
    else:
        raise ValueError("Invalid dimensions value: %d" % num_dimensions)
    n = int((n - 1)/2)

    """
    Performance stats
    tp_<front>: Array for the numbers of true positives of the given front and threshold
    tn_<front>: Array for the numbers of true negatives of the given front and threshold
    fp_<front>: Array for the numbers of false positives of the given front and threshold
    fn_<front>: Array for the numbers of false negatives of the given front and threshold
    """
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
    tp_front = np.zeros(shape=[4,100])
    fp_front = np.zeros(shape=[4,100])
    tn_front = np.zeros(shape=[4,100])
    fn_front = np.zeros(shape=[4,100])

    """ Properties of the final map made from stitched images """
    domain_images_lon, domain_images_lat = domain_images[0], domain_images[1]
    domain_length_lon, domain_length_lat = domain_lengths[0], domain_lengths[1]
    domain_trim_lon, domain_trim_lat = domain_trim[0], domain_trim[1]
    model_length_lon, model_length_lat = map_dim_x, map_dim_y  # Dimensions of the model's predictions
    domain_length_lon_trimmed = domain_length_lon - 2*domain_trim_lon  # Longitude dimension of the full stitched map after trimming
    domain_length_lat_trimmed = domain_length_lat - 2*domain_trim_lat  # Latitude dimension of the full stitched map after trimming
    lon_pixels_per_image = int(model_length_lon - 2*domain_trim_lon)  # Longitude dimension of each image after trimming
    lat_pixels_per_image = int(model_length_lat - 2*domain_trim_lat)  # Latitude dimension of each image after trimming

    if domain_images_lon > 1:
        lon_image_spacing = int((domain_length_lon - model_length_lon)/(domain_images_lon-1))
    else:
        lon_image_spacing = 0

    if domain_images_lat > 1:
        lat_image_spacing = int((domain_length_lat - model_length_lat)/(domain_images_lat-1))
    else:
        lat_image_spacing = 0

    thresholds = np.linspace(0.01,1,100)  # Probability thresholds for calculating performance statistics
    boundaries = np.array([50,100,150,200])  # Boundaries for checking whether or not a front is present (kilometers)

    for index in range(len(front_files)):

        # Open random pair of files
        fronts_filename = front_files[index]
        variables_filename = variable_files[index]
        fronts_ds = pd.read_pickle(fronts_filename)
        raw_variable_ds = normalize(pd.read_pickle(variables_filename), normalization_method)  # This dataset can be thought of as a "backup" that is used to reset the variable dataset.
        for i in range(pixel_expansion):  # Expand fronts by the given number of pixels
            fronts_ds = ope(fronts_ds)  # ope: one_pixel_expansion function in expand_fronts.py

        # Randomize variable (if applicable)
        if random_variable is not None:
            domain_dim_lon = len(raw_variable_ds['longitude'].values)  # Length of the full domain in the longitude direction (# of pixels)
            domain_dim_lat = len(raw_variable_ds['latitude'].values)  # Length of the full domain in the latitude direction (# of pixels)

            var_values = raw_variable_ds[random_variable].values.flatten()
            np.random.shuffle(var_values)
            raw_variable_ds[random_variable].values = var_values.reshape(domain_dim_lat,domain_dim_lon)

        # Create arrays containing probabilities for each front type in the final stitched map
        stitched_map_cold_probs = np.empty(shape=[domain_length_lon_trimmed,domain_length_lat_trimmed])
        stitched_map_warm_probs = np.empty(shape=[domain_length_lon_trimmed,domain_length_lat_trimmed])
        stitched_map_stationary_probs = np.empty(shape=[domain_length_lon_trimmed,domain_length_lat_trimmed])
        stitched_map_occluded_probs = np.empty(shape=[domain_length_lon_trimmed,domain_length_lat_trimmed])

        stitched_map_front_probs = np.empty(shape=[domain_length_lon_trimmed,domain_length_lat_trimmed])  # Array for binary fronts

        # Save longitude and latitude domain for making the prediction plot
        image_lats = fronts_ds.latitude.values[domain_trim_lat:domain_length_lat-domain_trim_lat]
        image_lons = fronts_ds.longitude.values[domain_trim_lon:domain_length_lon-domain_trim_lon]

        time = str(fronts_ds.time.values)[0:13].replace('T', '-') + 'z'

        map_created = False  # Boolean that dtermines whether or not the final stitched map has been created

        for lat_image in range(domain_images_lat):
            lat_index = int(lat_image*lat_image_spacing)  # Index of the latitude coordinates array
            for lon_image in range(domain_images_lon):
                print("%s....%d/%d" % (time, int(lat_image*domain_images_lon)+lon_image+1, int(domain_images_lon*domain_images_lat)),end='\r')
                lon_index = int(lon_image*lon_image_spacing)  # Index of the longitude coordinates array

                variable_ds = raw_variable_ds  # Reset variable dataset with "backup" copy
                lons = variable_ds.longitude.values[lon_index:lon_index + map_dim_x]  # Longitude values for the current image
                lats = variable_ds.latitude.values[lat_index:lat_index + map_dim_y]  # Latitude values for the current image

                if num_dimensions == 2:
                    variable_ds_new = np.nan_to_num(variable_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, map_dim_x,
                        map_dim_y, channels))  # Create new variable dataset for the current image
                else:
                    # Create new variable datasets for each level (surface, 1000mb, 950mb, 900mb, 850mb)
                    variables_sfc = variable_ds[['t2m','d2m','sp','u10','v10','theta_w','mix_ratio','rel_humid','virt_temp','wet_bulb','theta_e',
                                                 'q']].sel(longitude=lons, latitude=lats).to_array().values
                    variables_1000 = variable_ds[['t_1000','d_1000','z_1000','u_1000','v_1000','theta_w_1000','mix_ratio_1000','rel_humid_1000','virt_temp_1000',
                                                  'wet_bulb_1000','theta_e_1000','q_1000']].sel(longitude=lons, latitude=lats).to_array().values
                    variables_950 = variable_ds[['t_950','d_950','z_950','u_950','v_950','theta_w_950','mix_ratio_950','rel_humid_950','virt_temp_950',
                                                 'wet_bulb_950','theta_e_950','q_950']].sel(longitude=lons, latitude=lats).to_array().values
                    variables_900 = variable_ds[['t_900','d_900','z_900','u_900','v_900','theta_w_900','mix_ratio_900','rel_humid_900','virt_temp_900',
                                                 'wet_bulb_900','theta_e_900','q_900']].sel(longitude=lons, latitude=lats).to_array().values
                    variables_850 = variable_ds[['t_850','d_850','z_850','u_850','v_850','theta_w_850','mix_ratio_850','rel_humid_850','virt_temp_850',
                                                 'wet_bulb_850','theta_e_850','q_850']].sel(longitude=lons, latitude=lats).to_array().values
                    variable_ds_new = np.expand_dims(np.array([variables_sfc,variables_1000,variables_950,variables_900,variables_850]).transpose([3,2,0,1]), axis=0)

                prediction = model.predict(variable_ds_new)

                # Arrays of probabilities for all front types
                cold_probs = np.zeros([map_dim_x, map_dim_y])
                warm_probs = np.zeros([map_dim_x, map_dim_y])
                stationary_probs = np.zeros([map_dim_x, map_dim_y])
                occluded_probs = np.zeros([map_dim_x, map_dim_y])

                front_probs = np.zeros([map_dim_x, map_dim_y])  # Array for fronts if all are labeled as one type

                # Use predictions to build stitched map
                if front_types == 'CFWF':
                    if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':  # Names of the previous 2D models
                        for i in range(0, map_dim_x):
                            for j in range(0, map_dim_y):
                                cold_probs[i][j] = prediction[n][0][i][j][1]
                                warm_probs[i][j] = prediction[n][0][i][j][2]
                    elif model.name == '3plus3D':
                        for i in range(0, map_dim_x):
                            for j in range(0, map_dim_y):
                                cold_probs[i][j] = np.amax(prediction[0][0][i][j][:,1])
                                warm_probs[i][j] = np.amax(prediction[0][0][i][j][:,2])

                    # Add predictions to the map
                    stitched_map_cold_probs, map_created = add_image_to_map(stitched_map_cold_probs, cold_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)
                    stitched_map_warm_probs, map_created = add_image_to_map(stitched_map_warm_probs, warm_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)

                    if map_created is True:
                        print("%s....%d/%d" % (time, int(domain_images_lon*domain_images_lat), int(domain_images_lon*domain_images_lat)))
                        probs_ds = xr.Dataset(
                            {"cold_probs": (("longitude", "latitude"), stitched_map_cold_probs),
                             "warm_probs": (("longitude", "latitude"), stitched_map_warm_probs)}, coords={"latitude": image_lats, "longitude": image_lons}).transpose()
                        for boundary in range(4):
                            fronts = pd.read_pickle(fronts_filename)  # This is the "backup" dataset that can be used to reset the 'new_fronts' dataset
                            for y in range(int(2*boundary+1)):
                                fronts = ope(fronts)  # ope: one_pixel_expansion function in expand_fronts.py
                            """
                            t_<front>_ds: Pixels where the specific front type is present are set to 1, and 0 otherwise.
                            f_<front>_ds: Pixels where the specific front type is NOT present are set to 1, and 0 otherwise.
                            
                            'new_fronts' dataset is kept separate from the 'fronts' dataset to so it can be repeatedly modified and reset
                            new_fronts = fronts  <---- this line resets the front dataset after it is modified by xr.where()
                            """
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

                            """
                            Performance stats
                            tp_<front>: Number of true positives of the given front
                            tn_<front>: Number of true negatives of the given front
                            fp_<front>: Number of false positives of the given front
                            fn_<front>: Number of false negatives of the given front
                            """
                            for i in range(100):
                                tp_cold[boundary,i] += len(np.where(t_cold_probs > thresholds[i])[0])
                                tn_cold[boundary,i] += len(np.where((f_cold_probs < thresholds[i]) & (f_cold_probs != 0))[0])
                                fp_cold[boundary,i] += len(np.where(f_cold_probs > thresholds[i])[0])
                                fn_cold[boundary,i] += len(np.where((t_cold_probs < thresholds[i]) & (t_cold_probs != 0))[0])
                                tp_warm[boundary,i] += len(np.where(t_warm_probs > thresholds[i])[0])
                                tn_warm[boundary,i] += len(np.where((f_warm_probs < thresholds[i]) & (f_warm_probs != 0))[0])
                                fp_warm[boundary,i] += len(np.where(f_warm_probs > thresholds[i])[0])
                                fn_warm[boundary,i] += len(np.where((t_warm_probs < thresholds[i]) & (t_warm_probs != 0))[0])

                elif front_types == 'SFOF':
                    if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':  # Names of the previous 2D models
                        for i in range(0, map_dim_x):
                            for j in range(0, map_dim_y):
                                stationary_probs[i][j] = prediction[n][0][i][j][1]
                                occluded_probs[i][j] = prediction[n][0][i][j][2]
                    elif model.name == '3plus3D':
                        for i in range(0, map_dim_x):
                            for j in range(0, map_dim_y):
                                stationary_probs[i][j] = np.amax(prediction[0][0][i][j][:,1])
                                occluded_probs[i][j] = np.amax(prediction[0][0][i][j][:,2])

                    # Add predictions to the map
                    stitched_map_stationary_probs, map_created = add_image_to_map(stitched_map_stationary_probs, stationary_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)
                    stitched_map_occluded_probs, map_created = add_image_to_map(stitched_map_occluded_probs, occluded_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)

                    if map_created is True:
                        print("%s....%d/%d" % (time, int(domain_images_lon*domain_images_lat), int(domain_images_lon*domain_images_lat)))
                        probs_ds = xr.Dataset(
                            {"stationary_probs": (("longitude", "latitude"), stitched_map_stationary_probs),
                             "occluded_probs": (("longitude", "latitude"), stitched_map_occluded_probs)}, coords={"latitude": image_lats, "longitude": image_lons}).transpose()
                        for boundary in range(4):
                            fronts = pd.read_pickle(fronts_filename)  # This is the "backup" dataset that can be used to reset the 'new_fronts' dataset
                            for y in range(int(2*boundary+1)):
                                fronts = ope(fronts)  # ope: one_pixel_expansion function in expand_fronts.py
                            """
                            t_<front>_ds: Pixels where the specific front type is present are set to 1, and 0 otherwise.
                            f_<front>_ds: Pixels where the specific front type is NOT present are set to 1, and 0 otherwise.
                            
                            'new_fronts' dataset is kept separate from the 'fronts' dataset to so it can be repeatedly modified and reset
                            new_fronts = fronts  <---- this line resets the front dataset after it is modified by xr.where()
                            """
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

                            """
                            Performance stats
                            tp_<front>: Number of true positives of the given front
                            tn_<front>: Number of true negatives of the given front
                            fp_<front>: Number of false positives of the given front
                            fn_<front>: Number of false negatives of the given front
                            """
                            for i in range(100):
                                tp_stationary[boundary,i] += len(np.where(t_stationary_probs > thresholds[i])[0])
                                tn_stationary[boundary,i] += len(np.where((f_stationary_probs < thresholds[i]) & (f_stationary_probs != 0))[0])
                                fp_stationary[boundary,i] += len(np.where(f_stationary_probs > thresholds[i])[0])
                                fn_stationary[boundary,i] += len(np.where((t_stationary_probs < thresholds[i]) & (t_stationary_probs != 0))[0])
                                tp_occluded[boundary,i] += len(np.where(t_occluded_probs > thresholds[i])[0])
                                tn_occluded[boundary,i] += len(np.where((f_occluded_probs < thresholds[i]) & (f_occluded_probs != 0))[0])
                                fp_occluded[boundary,i] += len(np.where(f_occluded_probs > thresholds[i])[0])
                                fn_occluded[boundary,i] += len(np.where((t_occluded_probs < thresholds[i]) & (t_occluded_probs != 0))[0])

                elif front_types == 'ALL':
                    if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':  # Names of the previous 2D models
                        for i in range(0, map_dim_x):
                            for j in range(0, map_dim_y):
                                cold_probs[i][j] = prediction[n][0][i][j][1]
                                warm_probs[i][j] = prediction[n][0][i][j][2]
                                stationary_probs[i][j] = prediction[n][0][i][j][3]
                                occluded_probs[i][j] = prediction[n][0][i][j][4]
                    elif model.name == '3plus3D':
                        for i in range(0, map_dim_x):
                            for j in range(0, map_dim_y):
                                cold_probs[i][j] = np.amax(prediction[0][0][i][j][:,1])
                                warm_probs[i][j] = np.amax(prediction[0][0][i][j][:,2])
                                stationary_probs[i][j] = np.amax(prediction[0][0][i][j][:,3])
                                occluded_probs[i][j] = np.amax(prediction[0][0][i][j][:,4])

                    # Add predictions to the map
                    stitched_map_cold_probs, map_created = add_image_to_map(stitched_map_cold_probs, cold_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)
                    stitched_map_warm_probs, map_created = add_image_to_map(stitched_map_warm_probs, warm_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)
                    stitched_map_stationary_probs, map_created = add_image_to_map(stitched_map_stationary_probs, stationary_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)
                    stitched_map_occluded_probs, map_created = add_image_to_map(stitched_map_occluded_probs, occluded_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)

                    if map_created is True:
                        print("%s....%d/%d" % (time, int(domain_images_lon*domain_images_lat), int(domain_images_lon*domain_images_lat)))
                        probs_ds = xr.Dataset(
                            {"cold_probs": (("longitude", "latitude"), stitched_map_cold_probs),
                             "warm_probs": (("longitude", "latitude"), stitched_map_warm_probs), "stationary_probs": (("longitude", "latitude"), stitched_map_stationary_probs),
                             "occluded_probs": (("longitude", "latitude"), stitched_map_occluded_probs)},
                            coords={"latitude": image_lats, "longitude": image_lons})
                        for boundary in range(4):
                            fronts = pd.read_pickle(fronts_filename)  # This is the "backup" dataset that can be used to reset the 'new_fronts' dataset
                            for y in range(int(2*(boundary+1))):
                                fronts = ope(fronts)  # ope: one_pixel_expansion function in expand_fronts.py

                            """
                            t_<front>_ds: Pixels where the specific front type is present are set to 1, and 0 otherwise.
                            f_<front>_ds: Pixels where the specific front type is NOT present are set to 1, and 0 otherwise.
                            
                            'new_fronts' dataset is kept separate from the 'fronts' dataset to so it can be repeatedly modified and reset
                            new_fronts = fronts  <---- this line resets the front dataset after it is modified by xr.where()
                            """
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

                            """
                            Performance stats
                            tp_<front>: Number of true positives of the given front
                            tn_<front>: Number of true negatives of the given front
                            fp_<front>: Number of false positives of the given front
                            fn_<front>: Number of false negatives of the given front
                            """
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

                elif front_types == 'ALL_bin':
                    if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':  # Names of the previous 2D models
                        for i in range(0, map_dim_x):
                            for j in range(0, map_dim_y):
                                front_probs[i][j] = prediction[n][0][i][j][1]
                    elif model.name == '3plus3D':
                        for i in range(0, map_dim_x):
                            for j in range(0, map_dim_y):
                                front_probs[i][j] = np.amax(prediction[0][0][i][j][:,1])

                    # Add predictions to the map
                    stitched_map_front_probs, map_created = add_image_to_map(stitched_map_front_probs, front_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)

                    if map_created is True:
                        print("%s....%d/%d" % (time, int(domain_images_lon*domain_images_lat), int(domain_images_lon*domain_images_lat)))
                        probs_ds = xr.Dataset({"front_probs": (("longitude","latitude"), stitched_map_front_probs)},
                                              coords={"latitude": image_lats, "longitude": image_lons}).transpose()
                        for boundary in range(4):
                            fronts = pd.read_pickle(fronts_filename)  # This is the "backup" dataset that can be used to reset the 'new_fronts' dataset
                            for y in range(int(2*(boundary+1))):
                                fronts = ope(fronts)  # ope: one_pixel_expansion function in expand_fronts.py
                            """
                            t_<front>_ds: Pixels where the specific front type is present are set to 1, and 0 otherwise.
                            f_<front>_ds: Pixels where the specific front type is NOT present are set to 1, and 0 otherwise.
                            
                            'new_fronts' dataset is kept separate from the 'fronts' dataset to so it can be repeatedly modified and reset
                            new_fronts = fronts  <---- this line resets the front dataset after it is modified by xr.where()
                            """
                            new_fronts = fronts
                            t_front_ds = xr.where(new_fronts == 5, 0, new_fronts)
                            t_front_ds = xr.where(t_front_ds > 0, 1, 0)
                            t_front_probs = t_front_ds.identifier * probs_ds.front_probs
                            new_fronts = fronts
                            f_front_ds = xr.where(new_fronts == 5, 0, new_fronts)
                            f_front_ds = xr.where(f_front_ds > 0, 0, 1)
                            f_front_probs = f_front_ds.identifier * probs_ds.front_probs

                            """
                            Performance stats
                            tp_<front>: Number of true positives of the given front
                            tn_<front>: Number of true negatives of the given front
                            fp_<front>: Number of false positives of the given front
                            fn_<front>: Number of false negatives of the given front
                            """
                            for i in range(100):
                                tp_front[boundary,i] += len(np.where(t_front_probs > thresholds[i])[0])
                                tn_front[boundary,i] += len(np.where((f_front_probs < thresholds[i]) & (f_front_probs != 0))[0])
                                fp_front[boundary,i] += len(np.where(f_front_probs > thresholds[i])[0])
                                fn_front[boundary,i] += len(np.where((t_front_probs < thresholds[i]) & (t_front_probs != 0))[0])

    if front_types == 'CFWF':
        performance_ds = xr.Dataset({"tp_cold": (["boundary", "threshold"], tp_cold), "tp_warm": (["boundary", "threshold"], tp_warm),
                                     "fp_cold": (["boundary", "threshold"], fp_cold), "fp_warm": (["boundary", "threshold"], fp_warm),
                                     "tn_cold": (["boundary", "threshold"], tn_cold), "tn_warm": (["boundary", "threshold"], tn_warm),
                                     "fn_cold": (["boundary", "threshold"], fn_cold), "fn_warm": (["boundary", "threshold"], fn_warm)}, coords={"boundary": boundaries, "threshold": thresholds})
    elif front_types == 'SFOF':
        performance_ds = xr.Dataset({"tp_stationary": (["boundary", "threshold"], tp_stationary), "tp_occluded": (["boundary", "threshold"], tp_occluded),
                                     "fp_stationary": (["boundary", "threshold"], fp_stationary), "fp_occluded": (["boundary", "threshold"], fp_occluded),
                                     "tn_stationary": (["boundary", "threshold"], tn_stationary), "tn_occluded": (["boundary", "threshold"], tn_occluded),
                                     "fn_stationary": (["boundary", "threshold"], fn_stationary), "fn_occluded": (["boundary", "threshold"], fn_occluded)}, coords={"boundary": boundaries, "threshold": thresholds})
    elif front_types == 'ALL':
        performance_ds = xr.Dataset({"tp_cold": (["boundary", "threshold"], tp_cold), "tp_warm": (["boundary", "threshold"], tp_warm),
                                     "tp_stationary": (["boundary", "threshold"], tp_stationary), "tp_occluded": (["boundary", "threshold"], tp_occluded),
                                     "fp_cold": (["boundary", "threshold"], fp_cold), "fp_warm": (["boundary", "threshold"], fp_warm),
                                     "fp_stationary": (["boundary", "threshold"], fp_stationary), "fp_occluded": (["boundary", "threshold"], fp_occluded),
                                     "tn_cold": (["boundary", "threshold"], tn_cold), "tn_warm": (["boundary", "threshold"], tn_warm),
                                     "tn_stationary": (["boundary", "threshold"], tn_stationary), "tn_occluded": (["boundary", "threshold"], tn_occluded),
                                     "fn_cold": (["boundary", "threshold"], fn_cold), "fn_warm": (["boundary", "threshold"], fn_warm),
                                     "fn_stationary": (["boundary", "threshold"], fn_stationary), "fn_occluded": (["boundary", "threshold"], fn_occluded)}, coords={"boundary": boundaries, "threshold": thresholds})
    elif front_types == 'ALL_bin':
        performance_ds = xr.Dataset({"tp_front": (["boundary", "threshold"], tp_front), "fp_front": (["boundary", "threshold"], fp_front),
                                     "tn_front": (["boundary", "threshold"], tn_front), "fn_front": (["boundary", "threshold"], fn_front)}, coords={"boundary": boundaries, "threshold": thresholds})
    else:
        raise NameError("Unable to determine front types")

    print(performance_ds)

    if random_variable is None:
        filename = "%s/model_%d/model_%d_performance_stats_%dx%dimage_%dx%dtrim.pkl" % (model_dir, model_number, model_number, domain_images_lon, domain_images_lat, domain_trim_lon, domain_trim_lat)
    else:
        filename = "%s/model_%d/model_%d_performance_stats_%dx%dimage_%dx%dtrim_%s.pkl" % (model_dir, model_number, model_number, domain_images_lon, domain_images_lat, domain_trim_lon, domain_trim_lat, random_variable)

    with open(filename, "wb") as f:
        pickle.dump(performance_ds, f)


def find_matches_for_domain(domain_lengths, model_lengths, compatibility_mode=False, compat_images=None):
    """
    Function that outputs the number of images that can be stitched together with the specified domain length and the length
    of the domain dimension output by the model. This is also used to determine the compatibility of declared image and
    trim parameters for model predictions. However, if the number of images in either the longitude or latitude direction is 1,
    this function cannot be used and errors may occur when creating the final stitched maps.

    Parameters
    ----------
    compat_images: Number of images declared for the stitched map in each dimension (lon lat). (Compatibility mode only)
    compatibility_mode: Boolean that declares whether or not the function is being used to check compatibility of given parameters. (Compatibility mode only)
    domain_lengths: Number of pixels along each dimension of the final stitched map (lon lat).
    model_lengths: Number of pixels along each dimension of the model's output (lon lat).
    """
    if compatibility_mode is True:
        """ These parameters are used when checking the compatibility of image stitching arguments. """
        compat_images_lon = compat_images[0]  # Number of images in the longitude direction
        compat_images_lat = compat_images[1]  # Number of images in the latitude direction
    else:
        compat_images_lon, compat_images_lat = None, None

    # All of these boolean variables must be True after the compatibility check or else a ValueError is returned
    lon_images_are_compatible = False
    lat_images_are_compatible = False

    num_matches = [0,0]  # Total number of matching image arguments found for each dimension

    lon_image_matches = []
    lat_image_matches = []

    for lon_images in range(1,domain_lengths[0]-model_lengths[0]):  # Image counter for longitude dimension
        if lon_images > 1:
            lon_spacing = (domain_lengths[0]-model_lengths[0])/(lon_images-1)  # Spacing between images in the longitude dimension
        else:
            lon_spacing = 0
        if lon_spacing - int(lon_spacing) == 0 and lon_spacing > 1 and model_lengths[0]-lon_spacing > 0:  # Check compatibility of latitude image spacing
            lon_image_matches.append(lon_images)  # Add longitude image match to list
            num_matches[0] += 1
            if compatibility_mode is True:
                if compat_images_lon == lon_images:  # If the number of images for the compatibility check equals the match
                    lon_images_are_compatible = True
        elif lon_spacing == 0 and domain_lengths[0] - model_lengths[0] == 0:
            lon_image_matches.append(lon_images)  # Add lonitude image match to list
            num_matches[0] += 1
            if compatibility_mode is True:
                if compat_images_lon == lon_images:  # If the number of images for the compatibility check equals the match
                    lon_images_are_compatible = True

    if num_matches[0] == 0:
        raise ValueError(f"No compatible value for domain_images[0] was found with domain_lengths[0]={domain_lengths[0]} and model_lengths[0]={model_lengths[0]}.")
    if compatibility_mode is True:
        if lon_images_are_compatible is False:
            raise ValueError(f"domain_images[0]={compat_images_lon} is not compatible with domain_lengths[0]={domain_lengths[0]} "
                             f"and model_lengths[0]={model_lengths[0]}.\n"
                             f"====> Compatible values for domain_images[0] given domain_lengths[0]={domain_lengths[0]} "
                             f"and model_lengths[0]={model_lengths[0]}: {lon_image_matches}")
    else:
        print(f"Compatible longitude images: {lon_image_matches}")

    for lat_images in range(1,domain_lengths[1]-model_lengths[1]+2):  # Image counter for latitude dimension
        if lat_images > 1:
            lat_spacing = (domain_lengths[1]-model_lengths[1])/(lat_images-1)  # Spacing between images in the latitude dimension
        else:
            lat_spacing = 0
        if lat_spacing - int(lat_spacing) == 0 and lat_spacing > 1 and model_lengths[1]-lat_spacing > 0:  # Check compatibility of latitude image spacing
            lat_image_matches.append(lat_images)  # Add latitude image match to list
            num_matches[1] += 1
            if compatibility_mode is True:
                if compat_images_lat == lat_images:  # If the number of images for the compatibility check equals the match
                    lat_images_are_compatible = True
        elif lat_spacing == 0 and domain_lengths[1] - model_lengths[1] == 0:
            lat_image_matches.append(lat_images)  # Add latitude image match to list
            num_matches[1] += 1
            if compatibility_mode is True:
                if compat_images_lat == lat_images:  # If the number of images for the compatibility check equals the match
                    lat_images_are_compatible = True

    if num_matches[1] == 0:
        raise ValueError(f"No compatible value for domain_images[1] was found with domain_lengths[1]={domain_lengths[1]} and model_lengths[1]={model_lengths[1]}.")
    if compatibility_mode is True:
        if lat_images_are_compatible is False:
            raise ValueError(f"domain_images[1]={compat_images_lat} is not compatible with domain_lengths[1]={domain_lengths[1]} "
                             f"and model_lengths[1]={model_lengths[1]}.\n"
                             f"====> Compatible values for domain_images[1] given domain_lengths[1]={domain_lengths[1]} "
                             f"and model_lengths[1]={model_lengths[1]}: {lat_image_matches}")
    else:
        print(f"Compatible latitude images: {lat_image_matches}")


def generate_predictions(model_number, model_dir, front_files, variable_files, predictions, normalization_method,
    loss, fss_mask_size, fss_c, front_types, pixel_expansion, metric, num_dimensions, domain_images, domain_lengths,
    domain_trim, year, month, day, hour, random_variable=None, save_probabilities=False):
    """
    Function that makes random predictions using the provided model.

    Parameters
    ----------
    domain_images: Number of images along each dimension of the final stitched map (lon lat).
    domain_lengths: Number of pixels along each dimension of the final stitched map (lon lat).
    domain_trim: Number of pixels to trim each image by along each dimension before taking the maximum of the overlapping pixels.
    front_files: List of filenames that contain front data.
    front_types: Fronts in the data.
    fss_c: C hyperparameter for the FSS loss' sigmoid function.
    fss_mask_size: Size of the mask for the FSS loss function.
    loss: Loss function for the Unet.
    metric: Metric used for evaluating the U-Net during training.
    model_dir: Main directory for the models.
    model_number: Slurm job number for the model. This is the number in the model's filename.
    normalization_method: Normalization method for the data (described near the end of the script).
    num_dimensions: Number of dimensions for the U-Net's convolutions, maxpooling, and upsampling.
    pixel_expansion: Number of pixels to expand the fronts by in all directions.
    predictions: Number of random predictions to make.
    save_probabilities: Setting this to true will save the model's predictions for fronts into a pickle file.
    random_variable: Variable to randomize.
    variable_files: List of filenames that contain variable data.
    year: year
    month: month
    day: day
    hour: hour
    """
    model = fm.load_model(model_number, model_dir, loss, fss_mask_size, fss_c, metric, num_dimensions)

    n = 0  # Counter for the number of down layers in the model
    map_dim_x = model.layers[0].input_shape[0][1]  # Longitudinal dimension of the U-Net images
    map_dim_y = model.layers[0].input_shape[0][2]  # Latitudinal dimension of the U-Net images
    if num_dimensions == 2:
        for layer in model.layers:
            if layer.__class__.__name__ == 'MaxPooling2D':
                n += 1
        channels = model.layers[0].input_shape[0][3]  # Number of variables used
    elif num_dimensions == 3:
        for layer in model.layers:
            if layer.__class__.__name__ == 'MaxPooling3D':
                n += 1
        channels = model.layers[0].input_shape[0][4]  # Number of variables at each level
    else:
        raise ValueError("Invalid dimensions value: %d" % num_dimensions)
    n = int((n - 1)/2)

    # Properties of the final map made from stitched images
    domain_images_lon, domain_images_lat = domain_images[0], domain_images[1]
    domain_length_lon, domain_length_lat = domain_lengths[0], domain_lengths[1]
    domain_trim_lon, domain_trim_lat = domain_trim[0], domain_trim[1]
    model_length_lon, model_length_lat = map_dim_x, map_dim_y  # Dimensions of the model's predictions
    domain_length_lon_trimmed = domain_length_lon - 2*domain_trim_lon  # Longitude dimension of the full stitched map after trimming
    domain_length_lat_trimmed = domain_length_lat - 2*domain_trim_lat  # Latitude dimension of the full stitched map after trimming
    lon_pixels_per_image = int(model_length_lon - 2*domain_trim_lon)  # Longitude dimension of each image after trimming
    lat_pixels_per_image = int(model_length_lat - 2*domain_trim_lat)  # Latitude dimension of each image after trimming

    if domain_images_lon > 1:
        lon_image_spacing = int((domain_length_lon - model_length_lon)/(domain_images_lon-1))
    else:
        lon_image_spacing = 0

    if domain_images_lat > 1:
        lat_image_spacing = int((domain_length_lat - model_length_lat)/(domain_images_lat-1))
    else:
        lat_image_spacing = 0

    # Find files with provided date and time to make a prediction (if applicable)
    if year is not None and month is not None and day is not None and hour is not None and predictions is None:
        predictions = 1
        if front_types == 'ALL_bin':
            front_filename_no_dir = 'FrontObjects_%s_%d%02d%02d%02d_%s.pkl' % ('ALL', year, month, day, hour, args.domain)
        else:
            front_filename_no_dir = 'FrontObjects_%s_%d%02d%02d%02d_%s.pkl' % (front_types, year, month, day, hour, args.domain)
        variable_filename_no_dir = 'Data_%dvar_%d%02d%02d%02d_%s.pkl' % (60, year, month, day, hour, args.domain)
        front_files = [front_filename for front_filename in front_files if front_filename_no_dir in front_filename][0]
        variable_files = [variable_filename for variable_filename in variable_files if variable_filename_no_dir in variable_filename][0]
        """
        The 'indices' array is normally used only if the number of predictions is explicitly declared. However, here we declare
        the array to prevent some Python programs raising warnings about the array being referenced before its assignment.      
        """
        indices = []
    else:
        indices = random.choices(range(len(front_files) - 1), k=predictions)

    for x in range(predictions):
        if year is not None and month is not None and day is not None and hour is not None and predictions == 1:
            fronts_filename = front_files
            variables_filename = variable_files
        else:
            fronts_filename = front_files[indices[x]]
            variables_filename = variable_files[indices[x]]

        # Create arrays containing probabilities for each front type in the final stitched map
        stitched_map_cold_probs = np.empty(shape=[domain_length_lon_trimmed,domain_length_lat_trimmed])
        stitched_map_warm_probs = np.empty(shape=[domain_length_lon_trimmed,domain_length_lat_trimmed])
        stitched_map_stationary_probs = np.empty(shape=[domain_length_lon_trimmed,domain_length_lat_trimmed])
        stitched_map_occluded_probs = np.empty(shape=[domain_length_lon_trimmed,domain_length_lat_trimmed])

        stitched_map_front_probs = np.empty(shape=[domain_length_lon_trimmed,domain_length_lat_trimmed])  # If all fronts are labeled as the same type

        # Create a backup variable dataset that will be called to reset the variable dataset after it is modified.
        raw_variable_ds = normalize(pd.read_pickle(variables_filename), normalization_method)

        # Randomize variable
        if random_variable is not None:
            domain_dim_lon = len(raw_variable_ds['longitude'].values)  # Length of the full domain in the longitude direction (# of pixels)
            domain_dim_lat = len(raw_variable_ds['latitude'].values)  # Length of the full domain in the latitude direction (# of pixels)
            var_values = raw_variable_ds[random_variable].values.flatten()
            np.random.shuffle(var_values)
            raw_variable_ds[random_variable].values = var_values.reshape(domain_dim_lat,domain_dim_lon)

        fronts_ds = pd.read_pickle(fronts_filename)
        fronts = fronts_ds.sel(longitude=fronts_ds.longitude.values[domain_trim_lon:domain_length_lon-domain_trim_lon],
                               latitude=fronts_ds.latitude.values[domain_trim_lat:domain_length_lat-domain_trim_lat])

        # Latitude and longitude points in the domain
        image_lats = fronts_ds.latitude.values[domain_trim_lat:domain_length_lat-domain_trim_lat]
        image_lons = fronts_ds.longitude.values[domain_trim_lon:domain_length_lon-domain_trim_lon]

        time = str(fronts.time.values)[0:13].replace('T', '-') + 'z'

        map_created = False  # Boolean that determines whether or not the final stitched map has been created

        for lat_image in range(domain_images_lat):
            lat_index = int(lat_image*lat_image_spacing)
            for lon_image in range(domain_images_lon):
                print("%s....%d/%d" % (time, int(lat_image*domain_images_lon)+lon_image, int(domain_images_lon*domain_images_lat)),end='\r')
                lon_index = int(lon_image*lon_image_spacing)

                variable_ds = raw_variable_ds
                lons = variable_ds.longitude.values[lon_index:lon_index + map_dim_x]  # Longitude points for the current image
                lats = variable_ds.latitude.values[lat_index:lat_index + map_dim_y]  # Latitude points for the current image

                # Randomize variable
                if random_variable is not None:
                    domain_dim_lon = len(variable_ds['longitude'].values)  # Length of the full domain in the longitude direction (# of pixels)
                    domain_dim_lat = len(variable_ds['latitude'].values)  # Length of the full domain in the latitude direction (# of pixels)

                    var_values = variable_ds[random_variable].values.flatten()
                    np.random.shuffle(var_values)
                    variable_ds[random_variable].values = var_values.reshape(domain_dim_lat,domain_dim_lon)

                if num_dimensions == 2:
                    variable_ds_new = np.nan_to_num(variable_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, map_dim_x,
                        map_dim_y, channels))
                elif num_dimensions == 3:
                    variables_sfc = variable_ds[['t2m','d2m','sp','u10','v10','theta_w','mix_ratio','rel_humid','virt_temp','wet_bulb','theta_e',
                                                 'q']].sel(longitude=lons, latitude=lats).to_array().values
                    variables_1000 = variable_ds[['t_1000','d_1000','z_1000','u_1000','v_1000','theta_w_1000','mix_ratio_1000','rel_humid_1000','virt_temp_1000',
                                                  'wet_bulb_1000','theta_e_1000','q_1000']].sel(longitude=lons, latitude=lats).to_array().values
                    variables_950 = variable_ds[['t_950','d_950','z_950','u_950','v_950','theta_w_950','mix_ratio_950','rel_humid_950','virt_temp_950',
                                                 'wet_bulb_950','theta_e_950','q_950']].sel(longitude=lons, latitude=lats).to_array().values
                    variables_900 = variable_ds[['t_900','d_900','z_900','u_900','v_900','theta_w_900','mix_ratio_900','rel_humid_900','virt_temp_900',
                                                 'wet_bulb_900','theta_e_900','q_900']].sel(longitude=lons, latitude=lats).to_array().values
                    variables_850 = variable_ds[['t_850','d_850','z_850','u_850','v_850','theta_w_850','mix_ratio_850','rel_humid_850','virt_temp_850',
                                                 'wet_bulb_850','theta_e_850','q_850']].sel(longitude=lons, latitude=lats).to_array().values
                    variable_ds_new = np.expand_dims(np.array([variables_sfc,variables_1000,variables_950,variables_900,variables_850]).transpose([3,2,0,1]), axis=0)
                else:
                    raise ValueError("Invalid number of dimensions: %d" % num_dimensions)

                prediction = model.predict(variable_ds_new)

                # Arrays of probabilities for all front types
                cold_probs = np.zeros([map_dim_x, map_dim_y])
                warm_probs = np.zeros([map_dim_x, map_dim_y])
                stationary_probs = np.zeros([map_dim_x, map_dim_y])
                occluded_probs = np.zeros([map_dim_x, map_dim_y])

                front_probs = np.zeros([map_dim_x, map_dim_y])  # Array for fronts if all are labeled as one type

                if front_types == 'CFWF':
                    if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':  # Names of the previous 2D models
                        for i in range(0, map_dim_x):
                            for j in range(0, map_dim_y):
                                cold_probs[i][j] = prediction[n][0][i][j][1]
                                warm_probs[i][j] = prediction[n][0][i][j][2]
                    elif model.name == '3plus3D':
                        for i in range(0, map_dim_x):
                            for j in range(0, map_dim_y):
                                cold_probs[i][j] = np.amax(prediction[0][0][i][j][:,1])
                                warm_probs[i][j] = np.amax(prediction[0][0][i][j][:,2])

                    # Add predictions to the map
                    stitched_map_cold_probs, map_created = add_image_to_map(stitched_map_cold_probs, cold_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)
                    stitched_map_warm_probs, map_created = add_image_to_map(stitched_map_warm_probs, warm_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)

                    if map_created is True:
                        print("%s....%d/%d" % (time, int(domain_images_lon*domain_images_lat), int(domain_images_lon*domain_images_lat)))
                        probs_ds = xr.Dataset(
                            {"cold_probs": (("longitude", "latitude"), stitched_map_cold_probs),
                             "warm_probs": (("longitude", "latitude"), stitched_map_warm_probs)}, coords={"latitude": image_lats, "longitude": image_lons}).transpose()
                        prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, domain_images, domain_trim)
                        if save_probabilities is True:
                            outfile = '%s/model_%d/predictions/model_%d_%s_probabilities.pkl' % (model_dir, model_number, model_number, time)
                            with open(outfile,'wb') as f:
                                pickle.dump(probs_ds, f)

                elif front_types == 'SFOF':
                    if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':  # Names of the previous 2D models
                        for i in range(0, map_dim_x):
                            for j in range(0, map_dim_y):
                                stationary_probs[i][j] = prediction[n][0][i][j][1]
                                occluded_probs[i][j] = prediction[n][0][i][j][2]
                    elif model.name == '3plus3D':
                        for i in range(0, map_dim_x):
                            for j in range(0, map_dim_y):
                                stationary_probs[i][j] = np.amax(prediction[0][0][i][j][:,1])
                                occluded_probs[i][j] = np.amax(prediction[0][0][i][j][:,2])

                    # Add predictions to the map
                    stitched_map_stationary_probs, map_created = add_image_to_map(stitched_map_stationary_probs, stationary_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)
                    stitched_map_occluded_probs, map_created = add_image_to_map(stitched_map_occluded_probs, occluded_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)

                    if map_created is True:
                        print("%s....%d/%d" % (time, int(domain_images_lon*domain_images_lat), int(domain_images_lon*domain_images_lat)))
                        probs_ds = xr.Dataset(
                            {"stationary_probs": (("longitude", "latitude"), stitched_map_stationary_probs),
                             "occluded_probs": (("longitude", "latitude"), stitched_map_occluded_probs)}, coords={"latitude": image_lats, "longitude": image_lons}).transpose()
                        prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, domain_images, domain_trim)
                        if save_probabilities is True:
                            outfile = '%s/model_%d/predictions/model_%d_%s_probabilities.pkl' % (model_dir, model_number, model_number, time)
                            with open(outfile,'wb') as f:
                                pickle.dump(probs_ds, f)

                elif front_types == 'ALL':
                    if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':  # Names of the previous 2D models
                        for i in range(0, map_dim_x):
                            for j in range(0, map_dim_y):
                                cold_probs[i][j] = prediction[n][0][i][j][1]
                                warm_probs[i][j] = prediction[n][0][i][j][2]
                                stationary_probs[i][j] = prediction[n][0][i][j][3]
                                occluded_probs[i][j] = prediction[n][0][i][j][4]
                    elif model.name == '3plus3D':
                        for i in range(0, map_dim_x):
                            for j in range(0, map_dim_y):
                                cold_probs[i][j] = np.amax(prediction[0][0][i][j][:,1])
                                warm_probs[i][j] = np.amax(prediction[0][0][i][j][:,2])
                                stationary_probs[i][j] = np.amax(prediction[0][0][i][j][:,3])
                                occluded_probs[i][j] = np.amax(prediction[0][0][i][j][:,4])

                    # Add predictions to the map
                    stitched_map_cold_probs, map_created = add_image_to_map(stitched_map_cold_probs, cold_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)
                    stitched_map_warm_probs, map_created = add_image_to_map(stitched_map_warm_probs, warm_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)
                    stitched_map_stationary_probs, map_created = add_image_to_map(stitched_map_stationary_probs, stationary_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)
                    stitched_map_occluded_probs, map_created = add_image_to_map(stitched_map_occluded_probs, occluded_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)

                    if map_created is True:
                        print("%s....%d/%d" % (time, int(domain_images_lon*domain_images_lat), int(domain_images_lon*domain_images_lat)))
                        probs_ds = xr.Dataset(
                            {"cold_probs": (("longitude", "latitude"), stitched_map_cold_probs),
                             "warm_probs": (("longitude", "latitude"), stitched_map_warm_probs), "stationary_probs": (("longitude", "latitude"), stitched_map_stationary_probs),
                             "occluded_probs": (("longitude", "latitude"), stitched_map_occluded_probs)},
                            coords={"latitude": image_lats, "longitude": image_lons})
                        prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, domain_images, domain_trim)
                        if save_probabilities is True:
                            outfile = '%s/model_%d/predictions/model_%d_%s_probabilities.pkl' % (model_dir, model_number, model_number, time)
                            with open(outfile,'wb') as f:
                                pickle.dump(probs_ds, f)

                elif front_types == 'ALL_bin':
                    if model.name == 'U-Net' or model.name == 'unet' or model.name == 'model':  # Names of the previous 2D models
                        for i in range(0, map_dim_x):
                            for j in range(0, map_dim_y):
                                front_probs[i][j] = prediction[n][0][i][j][1]
                    elif model.name == '3plus3D':
                        for i in range(0, map_dim_x):
                            for j in range(0, map_dim_y):
                                front_probs[i][j] = np.amax(prediction[0][0][i][j][:,1])

                    # Add predictions to the map
                    stitched_map_front_probs, map_created = add_image_to_map(stitched_map_front_probs, front_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        model_length_lon, model_length_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)

                    if map_created is True:
                        print("%s....%d/%d" % (time, int(domain_images_lon*domain_images_lat), int(domain_images_lon*domain_images_lat)))
                        probs_ds = xr.Dataset({"front_probs": (("longitude","latitude"), stitched_map_front_probs)},
                                              coords={"latitude": image_lats, "longitude": image_lons}).transpose()
                        prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, domain_images, domain_trim)
                        if save_probabilities is True:
                            outfile = '%s/model_%d/predictions/model_%d_%s_probabilities.pkl' % (model_dir, model_number, model_number, time)
                            with open(outfile,'wb') as f:
                                pickle.dump(probs_ds, f)


def plot_performance_diagrams(model_dir, model_number, front_types, domain_images, domain_trim, random_variable=None):
    """
    Plots CSI performance diagram for different front types.

    Parameters
    ----------
    domain_images: Number of images along each dimension of the final stitched map (lon lat).
    domain_trim: Number of pixels to trim each image by along each dimension before taking the maximum of the overlapping pixels (lon lat).
    front_types: Fronts in the data.
    model_dir: Main directory for the models.
    model_number: Slurm job number for the model. This is the number in the model's filename.
    random_variable: Variable that was randomized when performance statistics were calculated.
    """

    if random_variable is None:
        filename = "%s/model_%d/model_%d_performance_stats_%dx%dimage_%dx%dtrim.pkl" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1])
    else:
        filename = "%s/model_%d/model_%d_performance_stats_%dx%dimage_%dx%dtrim_%s.pkl" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1], random_variable)

    stats = pd.read_pickle(filename)
    stats_50km = stats.sel(boundary=50)
    stats_100km = stats.sel(boundary=100)
    stats_150km = stats.sel(boundary=150)
    stats_200km = stats.sel(boundary=200)

    # Code for performance diagram matrices sourced from Ryan Lagerquist's (lagerqui@ualberta.ca) thunderhoser repository:
    # https://github.com/thunderhoser/GewitterGefahr/blob/master/gewittergefahr/plotting/model_eval_plotting.py
    success_ratio_matrix, pod_matrix = np.linspace(0,1,100), np.linspace(0,1,100)
    x, y = np.meshgrid(success_ratio_matrix, pod_matrix)
    csi_matrix = (x ** -1 + y ** -1 - 1.) ** -1
    CSI_LEVELS = np.linspace(0,1,11)
    cmap = 'Blues'

    if front_types == 'CFWF' or front_types == 'ALL':
        CSI_cold_50km = stats_50km['tp_cold']/(stats_50km['tp_cold'] + stats_50km['fp_cold'] + stats_50km['fn_cold'])
        CSI_cold_100km = stats_100km['tp_cold']/(stats_100km['tp_cold'] + stats_100km['fp_cold'] + stats_100km['fn_cold'])
        CSI_cold_150km = stats_150km['tp_cold']/(stats_150km['tp_cold'] + stats_150km['fp_cold'] + stats_150km['fn_cold'])
        CSI_cold_200km = stats_200km['tp_cold']/(stats_200km['tp_cold'] + stats_200km['fp_cold'] + stats_200km['fn_cold'])
        CSI_warm_50km = stats_50km['tp_warm']/(stats_50km['tp_warm'] + stats_50km['fp_warm'] + stats_50km['fn_warm'])
        CSI_warm_100km = stats_100km['tp_warm']/(stats_100km['tp_warm'] + stats_100km['fp_warm'] + stats_100km['fn_warm'])
        CSI_warm_150km = stats_150km['tp_warm']/(stats_150km['tp_warm'] + stats_150km['fp_warm'] + stats_150km['fn_warm'])
        CSI_warm_200km = stats_200km['tp_warm']/(stats_200km['tp_warm'] + stats_200km['fp_warm'] + stats_200km['fn_warm'])

        POD_cold_50km = stats_50km['tp_cold']/(stats_50km['tp_cold'] + stats_50km['fn_cold'])
        POD_cold_100km = stats_100km['tp_cold']/(stats_100km['tp_cold'] + stats_100km['fn_cold'])
        POD_cold_150km = stats_150km['tp_cold']/(stats_150km['tp_cold'] + stats_150km['fn_cold'])
        POD_cold_200km = stats_200km['tp_cold']/(stats_200km['tp_cold'] + stats_200km['fn_cold'])
        POD_warm_50km = stats_50km['tp_warm']/(stats_50km['tp_warm'] + stats_50km['fn_warm'])
        POD_warm_100km = stats_100km['tp_warm']/(stats_100km['tp_warm'] + stats_100km['fn_warm'])
        POD_warm_150km = stats_150km['tp_warm']/(stats_150km['tp_warm'] + stats_150km['fn_warm'])
        POD_warm_200km = stats_200km['tp_warm']/(stats_200km['tp_warm'] + stats_200km['fn_warm'])
        POD_cold_50km[0] = 1
        POD_cold_100km[0] = 1
        POD_cold_150km[0] = 1
        POD_cold_200km[0] = 1
        POD_warm_50km[0] = 1
        POD_warm_100km[0] = 1
        POD_warm_150km[0] = 1
        POD_warm_200km[0] = 1

        POFD_cold_50km = stats_50km['fp_cold']/(stats_50km['fp_cold'] + stats_50km['tn_cold'])
        POFD_cold_100km = stats_100km['fp_cold']/(stats_100km['fp_cold'] + stats_100km['tn_cold'])
        POFD_cold_150km = stats_150km['fp_cold']/(stats_150km['fp_cold'] + stats_150km['tn_cold'])
        POFD_cold_200km = stats_200km['fp_cold']/(stats_200km['fp_cold'] + stats_200km['tn_cold'])
        POFD_warm_50km = stats_50km['fp_warm']/(stats_50km['fp_warm'] + stats_50km['tn_warm'])
        POFD_warm_100km = stats_100km['fp_warm']/(stats_100km['fp_warm'] + stats_100km['tn_warm'])
        POFD_warm_150km = stats_150km['fp_warm']/(stats_150km['fp_warm'] + stats_150km['tn_warm'])
        POFD_warm_200km = stats_200km['fp_warm']/(stats_200km['fp_warm'] + stats_200km['tn_warm'])
        POFD_cold_50km[0] = 1
        POFD_cold_100km[0] = 1
        POFD_cold_150km[0] = 1
        POFD_cold_200km[0] = 1
        POFD_warm_50km[0] = 1
        POFD_warm_100km[0] = 1
        POFD_warm_150km[0] = 1
        POFD_warm_200km[0] = 1

        SR_cold_50km = stats_50km['tp_cold']/(stats_50km['tp_cold'] + stats_50km['fp_cold'])
        SR_cold_100km = stats_100km['tp_cold']/(stats_100km['tp_cold'] + stats_100km['fp_cold'])
        SR_cold_150km = stats_150km['tp_cold']/(stats_150km['tp_cold'] + stats_150km['fp_cold'])
        SR_cold_200km = stats_200km['tp_cold']/(stats_200km['tp_cold'] + stats_200km['fp_cold'])
        SR_warm_50km = stats_50km['tp_warm']/(stats_50km['tp_warm'] + stats_50km['fp_warm'])
        SR_warm_100km = stats_100km['tp_warm']/(stats_100km['tp_warm'] + stats_100km['fp_warm'])
        SR_warm_150km = stats_150km['tp_warm']/(stats_150km['tp_warm'] + stats_150km['fp_warm'])
        SR_warm_200km = stats_200km['tp_warm']/(stats_200km['tp_warm'] + stats_200km['fp_warm'])

        F1_cold_50km = 2/(1/POD_cold_50km + 1/SR_cold_50km)
        F1_cold_100km = 2/(1/POD_cold_100km + 1/SR_cold_100km)
        F1_cold_150km = 2/(1/POD_cold_150km + 1/SR_cold_150km)
        F1_cold_200km = 2/(1/POD_cold_200km + 1/SR_cold_200km)
        F1_warm_50km = 2/(1/POD_warm_50km + 1/SR_warm_50km)
        F1_warm_100km = 2/(1/POD_warm_100km + 1/SR_warm_100km)
        F1_warm_150km = 2/(1/POD_warm_150km + 1/SR_warm_150km)
        F1_warm_200km = 2/(1/POD_warm_200km + 1/SR_warm_200km)

        plt.figure()
        plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
        plt.colorbar(label='Critical Success Index (CSI)')
        plt.plot(SR_cold_50km, POD_cold_50km, color='red', label='50km boundary')
        plt.plot(SR_cold_50km[np.where(CSI_cold_50km == np.max(CSI_cold_50km))], POD_cold_50km[np.where(CSI_cold_50km == np.max(CSI_cold_50km))], color='red', marker='*', markersize=9)
        plt.text(0.01, 0.01, s=str('50km: %.4f' % np.max(CSI_cold_50km)), color='red')
        plt.plot(SR_cold_100km, POD_cold_100km, color='purple', label='100km boundary')
        plt.plot(SR_cold_100km[np.where(CSI_cold_100km == np.max(CSI_cold_100km))], POD_cold_100km[np.where(CSI_cold_100km == np.max(CSI_cold_100km))], color='purple', marker='*', markersize=9)
        plt.text(0.01, 0.05, s=str('100km: %.4f' % np.max(CSI_cold_100km)), color='purple')
        plt.plot(SR_cold_150km, POD_cold_150km, color='brown', label='150km boundary')
        plt.plot(SR_cold_150km[np.where(CSI_cold_150km == np.max(CSI_cold_150km))], POD_cold_150km[np.where(CSI_cold_150km == np.max(CSI_cold_150km))], color='brown', marker='*', markersize=9)
        plt.text(0.01, 0.09, s=str('150km: %.4f' % np.max(CSI_cold_150km)), color='brown')
        plt.plot(SR_cold_200km, POD_cold_200km, color='green', label='200km boundary')
        plt.plot(SR_cold_200km[np.where(CSI_cold_200km == np.max(CSI_cold_200km))], POD_cold_200km[np.where(CSI_cold_200km == np.max(CSI_cold_200km))], color='green', marker='*', markersize=9)
        plt.text(0.01, 0.13, s=str('200km: %.4f' % np.max(CSI_cold_200km)), color='green')
        plt.text(0.01, 0.17, s='CSI values', style='oblique')
        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.title("Model %d Performance for Cold Fronts" % model_number)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.savefig("%s/model_%d/model_%d_performance_cold_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.plot(F1_cold_50km, color='red', label='50km boundary')
        plt.plot(np.where(F1_cold_50km == np.max(F1_cold_50km)), np.max(F1_cold_50km), color='red', marker='*', markersize=9)
        plt.text(len(F1_cold_200km)*0.02, np.max(F1_cold_200km)*0.01, s=str('50km: %.4f' % np.max(F1_cold_50km)), color='red')
        plt.plot(F1_cold_100km, color='purple', label='100km boundary')
        plt.plot(np.where(F1_cold_100km == np.max(F1_cold_100km)), np.max(F1_cold_100km), color='purple', marker='*', markersize=9)
        plt.text(len(F1_cold_200km)*0.02, np.max(F1_cold_200km)*0.05, s=str('100km: %.4f' % np.max(F1_cold_100km)), color='purple')
        plt.plot(F1_cold_150km, color='brown', label='150km boundary')
        plt.plot(np.where(F1_cold_150km == np.max(F1_cold_150km)), np.max(F1_cold_150km), color='brown', marker='*', markersize=9)
        plt.text(len(F1_cold_200km)*0.02, np.max(F1_cold_200km)*0.09, s=str('150km: %.4f' % np.max(F1_cold_150km)), color='brown')
        plt.plot(F1_cold_200km, color='green', label='200km boundary')
        plt.plot(np.where(F1_cold_200km == np.max(F1_cold_200km)), np.max(F1_cold_200km), color='green', marker='*', markersize=9)
        plt.text(len(F1_cold_200km)*0.02, np.max(F1_cold_200km)*0.13, s=str('200km: %.4f' % np.max(F1_cold_200km)), color='green')
        plt.text(len(F1_cold_200km)*0.02, np.max(F1_cold_200km)*0.17, s='F1 scores', style='oblique')
        plt.xlim(0,100)
        plt.ylim(0)
        plt.xlabel("Probability Threshold (%)")
        plt.ylabel("F1 Score")
        plt.title("Model %d F1 Score for Cold Fronts" % model_number)
        plt.savefig("%s/model_%d/model_%d_F1_cold_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
        plt.colorbar(label='Critical Success Index (CSI)')
        plt.plot(SR_warm_50km, POD_warm_50km, color='red', label='50km boundary')
        plt.plot(SR_warm_50km[np.where(CSI_warm_50km == np.max(CSI_warm_50km))], POD_warm_50km[np.where(CSI_warm_50km == np.max(CSI_warm_50km))], color='red', marker='*', markersize=9)
        plt.text(0.01, 0.01, s=str('50km: %.4f' % np.max(CSI_warm_50km)), color='red')
        plt.plot(SR_warm_100km, POD_warm_100km, color='purple', label='100km boundary')
        plt.plot(SR_warm_100km[np.where(CSI_warm_100km == np.max(CSI_warm_100km))], POD_warm_100km[np.where(CSI_warm_100km == np.max(CSI_warm_100km))], color='purple', marker='*', markersize=9)
        plt.text(0.01, 0.05, s=str('100km: %.4f' % np.max(CSI_warm_100km)), color='purple')
        plt.plot(SR_warm_150km, POD_warm_150km, color='brown', label='150km boundary')
        plt.plot(SR_warm_150km[np.where(CSI_warm_150km == np.max(CSI_warm_150km))], POD_warm_150km[np.where(CSI_warm_150km == np.max(CSI_warm_150km))], color='brown', marker='*', markersize=9)
        plt.text(0.01, 0.09, s=str('150km: %.4f' % np.max(CSI_warm_150km)), color='brown')
        plt.plot(SR_warm_200km, POD_warm_200km, color='green', label='200km boundary')
        plt.plot(SR_warm_200km[np.where(CSI_warm_200km == np.max(CSI_warm_200km))], POD_warm_200km[np.where(CSI_warm_200km == np.max(CSI_warm_200km))], color='green', marker='*', markersize=9)
        plt.text(0.01, 0.13, s=str('200km: %.4f' % np.max(CSI_warm_200km)), color='green')
        plt.text(0.01, 0.17, s='CSI values', style='oblique')
        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.title("Model %d Performance for Warm Fronts" % model_number)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.savefig("%s/model_%d/model_%d_performance_warm_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
        plt.close()

        AUC_cold_50km, AUC_cold_100km, AUC_cold_150km, AUC_cold_200km = 0,0,0,0
        AUC_warm_50km, AUC_warm_100km, AUC_warm_150km, AUC_warm_200km = 0,0,0,0

        POD_cold_50km = np.append(np.array([1]),np.append(POD_cold_50km[0:99],0))
        POD_cold_100km = np.append(np.array([1]),np.append(POD_cold_100km[0:99],0))
        POD_cold_150km = np.append(np.array([1]),np.append(POD_cold_150km[0:99],0))
        POD_cold_200km = np.append(np.array([1]),np.append(POD_cold_200km[0:99],0))
        POFD_cold_50km = np.append(np.array([1]),np.append(POFD_cold_50km[0:99],0))
        POFD_cold_100km = np.append(np.array([1]),np.append(POFD_cold_100km[0:99],0))
        POFD_cold_150km = np.append(np.array([1]),np.append(POFD_cold_150km[0:99],0))
        POFD_cold_200km = np.append(np.array([1]),np.append(POFD_cold_200km[0:99],0))

        POD_warm_50km = np.append(np.array([1]),np.append(POD_warm_50km[0:99],0))
        POD_warm_100km = np.append(np.array([1]),np.append(POD_warm_100km[0:99],0))
        POD_warm_150km = np.append(np.array([1]),np.append(POD_warm_150km[0:99],0))
        POD_warm_200km = np.append(np.array([1]),np.append(POD_warm_200km[0:99],0))
        POFD_warm_50km = np.append(np.array([1]),np.append(POFD_warm_50km[0:99],0))
        POFD_warm_100km = np.append(np.array([1]),np.append(POFD_warm_100km[0:99],0))
        POFD_warm_150km = np.append(np.array([1]),np.append(POFD_warm_150km[0:99],0))
        POFD_warm_200km = np.append(np.array([1]),np.append(POFD_warm_200km[0:99],0))

        for threshold in range(99):
            AUC_cold_50km += POD_cold_50km[threshold]*(POFD_cold_50km[threshold]-POFD_cold_50km[threshold+1]) - \
                0.5*(POFD_cold_50km[threshold]-POFD_cold_50km[threshold+1])*(POD_cold_50km[threshold]-POD_cold_50km[threshold+1])
            AUC_cold_100km += POD_cold_100km[threshold]*(POFD_cold_100km[threshold]-POFD_cold_100km[threshold+1]) - \
                0.5*(POFD_cold_100km[threshold]-POFD_cold_100km[threshold+1])*(POD_cold_100km[threshold]-POD_cold_100km[threshold+1])
            AUC_cold_150km += POD_cold_150km[threshold]*(POFD_cold_150km[threshold]-POFD_cold_150km[threshold+1]) - \
                0.5*(POFD_cold_150km[threshold]-POFD_cold_150km[threshold+1])*(POD_cold_150km[threshold]-POD_cold_150km[threshold+1])
            AUC_cold_200km += POD_cold_200km[threshold]*(POFD_cold_200km[threshold]-POFD_cold_200km[threshold+1]) - \
                0.5*(POFD_cold_200km[threshold]-POFD_cold_200km[threshold+1])*(POD_cold_200km[threshold]-POD_cold_200km[threshold+1])
            AUC_warm_50km += POD_warm_50km[threshold]*(POFD_warm_50km[threshold]-POFD_warm_50km[threshold+1]) - \
                0.5*(POFD_warm_50km[threshold]-POFD_warm_50km[threshold+1])*(POD_warm_50km[threshold]-POD_warm_50km[threshold+1])
            AUC_warm_100km += POD_warm_100km[threshold]*(POFD_warm_100km[threshold]-POFD_warm_100km[threshold+1]) - \
                0.5*(POFD_warm_100km[threshold]-POFD_warm_100km[threshold+1])*(POD_warm_100km[threshold]-POD_warm_100km[threshold+1])
            AUC_warm_150km += POD_warm_150km[threshold]*(POFD_warm_150km[threshold]-POFD_warm_150km[threshold+1]) - \
                0.5*(POFD_warm_150km[threshold]-POFD_warm_150km[threshold+1])*(POD_warm_150km[threshold]-POD_warm_150km[threshold+1])
            AUC_warm_200km += POD_warm_200km[threshold]*(POFD_warm_200km[threshold]-POFD_warm_200km[threshold+1]) - \
                0.5*(POFD_warm_200km[threshold]-POFD_warm_200km[threshold+1])*(POD_warm_200km[threshold]-POD_warm_200km[threshold+1])

        plt.figure()
        plt.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),'k--')
        plt.plot(POFD_cold_50km, POD_cold_50km, color='red', label='50km boundary')
        plt.plot(POFD_cold_100km, POD_cold_100km, color='purple', label='100km boundary')
        plt.plot(POFD_cold_150km, POD_cold_150km, color='brown', label='150km boundary')
        plt.plot(POFD_cold_200km, POD_cold_200km, color='green', label='200km boundary')
        plt.text(0.8, 0.01, s=str('50km: %.4f' % AUC_cold_50km), color='red')
        plt.text(0.782, 0.05, s=str('100km: %.4f' % AUC_cold_100km), color='purple')
        plt.text(0.782, 0.09, s=str('150km: %.4f' % AUC_cold_150km), color='brown')
        plt.text(0.782, 0.13, s=str('200km: %.4f' % AUC_cold_200km), color='green')
        plt.text(0.605, 0.17, s='Area Under the Curve (AUC)', style='oblique')
        plt.xlabel("Probability of False Detection (POFD)")
        plt.ylabel("Probability of Detection (POD)")
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.title("Model %d ROC Curve for Cold Fronts" % model_number)
        plt.savefig("%s/model_%d/model_%d_AUC_cold_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.plot(F1_warm_50km, color='red', label='50km boundary')
        plt.plot(np.where(F1_warm_50km == np.max(F1_warm_50km)), np.max(F1_warm_50km), color='red', marker='*', markersize=9)
        plt.text(len(F1_warm_200km)*0.02, np.max(F1_warm_200km)*0.01, s=str('50km: %.4f' % np.max(F1_warm_50km)), color='red')
        plt.plot(F1_warm_100km, color='purple', label='100km boundary')
        plt.plot(np.where(F1_warm_100km == np.max(F1_warm_100km)), np.max(F1_warm_100km), color='purple', marker='*', markersize=9)
        plt.text(len(F1_warm_200km)*0.02, np.max(F1_warm_200km)*0.05, s=str('100km: %.4f' % np.max(F1_warm_100km)), color='purple')
        plt.plot(F1_warm_150km, color='brown', label='150km boundary')
        plt.plot(np.where(F1_warm_150km == np.max(F1_warm_150km)), np.max(F1_warm_150km), color='brown', marker='*', markersize=9)
        plt.text(len(F1_warm_200km)*0.02, np.max(F1_warm_200km)*0.09, s=str('150km: %.4f' % np.max(F1_warm_150km)), color='brown')
        plt.plot(F1_warm_200km, color='green', label='200km boundary')
        plt.plot(np.where(F1_warm_200km == np.max(F1_warm_200km)), np.max(F1_warm_200km), color='green', marker='*', markersize=9)
        plt.text(len(F1_warm_200km)*0.02, np.max(F1_warm_200km)*0.13, s=str('200km: %.4f' % np.max(F1_warm_200km)), color='green')
        plt.text(len(F1_warm_200km)*0.02, np.max(F1_warm_200km)*0.17, s='F1 scores', style='oblique')
        plt.xlim(0,100)
        plt.ylim(0)
        plt.xlabel("Probability Threshold (%)")
        plt.ylabel("F1 Score")
        plt.title("Model %d F1 Score for Warm Fronts" % model_number)
        plt.savefig("%s/model_%d/model_%d_F1_warm_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),'k--')
        plt.plot(POFD_warm_50km, POD_warm_50km, color='red', label='50km boundary')
        plt.plot(POFD_warm_100km, POD_warm_100km, color='purple', label='100km boundary')
        plt.plot(POFD_warm_150km, POD_warm_150km, color='brown', label='150km boundary')
        plt.plot(POFD_warm_200km, POD_warm_200km, color='green', label='200km boundary')
        plt.text(0.8, 0.01, s=str('50km: %.4f' % AUC_warm_50km), color='red')
        plt.text(0.782, 0.05, s=str('100km: %.4f' % AUC_warm_100km), color='purple')
        plt.text(0.782, 0.09, s=str('150km: %.4f' % AUC_warm_150km), color='brown')
        plt.text(0.782, 0.13, s=str('200km: %.4f' % AUC_warm_200km), color='green')
        plt.text(0.605, 0.17, s='Area Under the Curve (AUC)', style='oblique')
        plt.xlabel("Probability of False Detection (POFD)")
        plt.ylabel("Probability of Detection (POD)")
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.title("Model %d ROC Curve for Warm Fronts" % model_number)
        plt.savefig("%s/model_%d/model_%d_AUC_warm_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
        plt.close()

        if front_types == 'ALL':
            CSI_stationary_50km = stats_50km['tp_stationary']/(stats_50km['tp_stationary'] + stats_50km['fp_stationary'] + stats_50km['fn_stationary'])
            CSI_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fp_stationary'] + stats_100km['fn_stationary'])
            CSI_stationary_150km = stats_150km['tp_stationary']/(stats_150km['tp_stationary'] + stats_150km['fp_stationary'] + stats_150km['fn_stationary'])
            CSI_stationary_200km = stats_200km['tp_stationary']/(stats_200km['tp_stationary'] + stats_200km['fp_stationary'] + stats_200km['fn_stationary'])
            CSI_occluded_50km = stats_50km['tp_occluded']/(stats_50km['tp_occluded'] + stats_50km['fp_occluded'] + stats_50km['fn_occluded'])
            CSI_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fp_occluded'] + stats_100km['fn_occluded'])
            CSI_occluded_150km = stats_150km['tp_occluded']/(stats_150km['tp_occluded'] + stats_150km['fp_occluded'] + stats_150km['fn_occluded'])
            CSI_occluded_200km = stats_200km['tp_occluded']/(stats_200km['tp_occluded'] + stats_200km['fp_occluded'] + stats_200km['fn_occluded'])

            POD_stationary_50km = stats_50km['tp_stationary']/(stats_50km['tp_stationary'] + stats_50km['fn_stationary'])
            POD_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fn_stationary'])
            POD_stationary_150km = stats_150km['tp_stationary']/(stats_150km['tp_stationary'] + stats_150km['fn_stationary'])
            POD_stationary_200km = stats_200km['tp_stationary']/(stats_200km['tp_stationary'] + stats_200km['fn_stationary'])
            POD_occluded_50km = stats_50km['tp_occluded']/(stats_50km['tp_occluded'] + stats_50km['fn_occluded'])
            POD_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fn_occluded'])
            POD_occluded_150km = stats_150km['tp_occluded']/(stats_150km['tp_occluded'] + stats_150km['fn_occluded'])
            POD_occluded_200km = stats_200km['tp_occluded']/(stats_200km['tp_occluded'] + stats_200km['fn_occluded'])

            POFD_stationary_50km = stats_50km['fp_stationary']/(stats_50km['fp_stationary'] + stats_50km['tn_stationary'])
            POFD_stationary_100km = stats_100km['fp_stationary']/(stats_100km['fp_stationary'] + stats_100km['tn_stationary'])
            POFD_stationary_150km = stats_150km['fp_stationary']/(stats_150km['fp_stationary'] + stats_150km['tn_stationary'])
            POFD_stationary_200km = stats_200km['fp_stationary']/(stats_200km['fp_stationary'] + stats_200km['tn_stationary'])
            POFD_occluded_50km = stats_50km['fp_occluded']/(stats_50km['fp_occluded'] + stats_50km['tn_occluded'])
            POFD_occluded_100km = stats_100km['fp_occluded']/(stats_100km['fp_occluded'] + stats_100km['tn_occluded'])
            POFD_occluded_150km = stats_150km['fp_occluded']/(stats_150km['fp_occluded'] + stats_150km['tn_occluded'])
            POFD_occluded_200km = stats_200km['fp_occluded']/(stats_200km['fp_occluded'] + stats_200km['tn_occluded'])

            SR_stationary_50km = stats_50km['tp_stationary']/(stats_50km['tp_stationary'] + stats_50km['fp_stationary'])
            SR_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fp_stationary'])
            SR_stationary_150km = stats_150km['tp_stationary']/(stats_150km['tp_stationary'] + stats_150km['fp_stationary'])
            SR_stationary_200km = stats_200km['tp_stationary']/(stats_200km['tp_stationary'] + stats_200km['fp_stationary'])
            SR_occluded_50km = stats_50km['tp_occluded']/(stats_50km['tp_occluded'] + stats_50km['fp_occluded'])
            SR_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fp_occluded'])
            SR_occluded_150km = stats_150km['tp_occluded']/(stats_150km['tp_occluded'] + stats_150km['fp_occluded'])
            SR_occluded_200km = stats_200km['tp_occluded']/(stats_200km['tp_occluded'] + stats_200km['fp_occluded'])

            F1_stationary_50km = 2/(1/POD_stationary_50km + 1/SR_stationary_50km)
            F1_stationary_100km = 2/(1/POD_stationary_100km + 1/SR_stationary_100km)
            F1_stationary_150km = 2/(1/POD_stationary_150km + 1/SR_stationary_150km)
            F1_stationary_200km = 2/(1/POD_stationary_200km + 1/SR_stationary_200km)
            F1_occluded_50km = 2/(1/POD_occluded_50km + 1/SR_occluded_50km)
            F1_occluded_100km = 2/(1/POD_occluded_100km + 1/SR_occluded_100km)
            F1_occluded_150km = 2/(1/POD_occluded_150km + 1/SR_occluded_150km)
            F1_occluded_200km = 2/(1/POD_occluded_200km + 1/SR_occluded_200km)

            plt.figure()
            plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
            plt.colorbar(label='Critical Success Index (CSI)')
            plt.plot(SR_stationary_50km, POD_stationary_50km, color='red', label='50km boundary')
            plt.plot(SR_stationary_50km[np.where(CSI_stationary_50km == np.max(CSI_stationary_50km))], POD_stationary_50km[np.where(CSI_stationary_50km == np.max(CSI_stationary_50km))], color='red', marker='*', markersize=9)
            plt.text(0.01, 0.01, s=str('50km: %.4f' % np.max(CSI_stationary_50km)), color='red')
            plt.plot(SR_stationary_100km, POD_stationary_100km, color='purple', label='100km boundary')
            plt.plot(SR_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))], POD_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))], color='purple', marker='*', markersize=9)
            plt.text(0.01, 0.05, s=str('100km: %.4f' % np.max(CSI_stationary_100km)), color='purple')
            plt.plot(SR_stationary_150km, POD_stationary_150km, color='brown', label='150km boundary')
            plt.plot(SR_stationary_150km[np.where(CSI_stationary_150km == np.max(CSI_stationary_150km))], POD_stationary_150km[np.where(CSI_stationary_150km == np.max(CSI_stationary_150km))], color='brown', marker='*', markersize=9)
            plt.text(0.01, 0.09, s=str('150km: %.4f' % np.max(CSI_stationary_150km)), color='brown')
            plt.plot(SR_stationary_200km, POD_stationary_200km, color='green', label='200km boundary')
            plt.plot(SR_stationary_200km[np.where(CSI_stationary_200km == np.max(CSI_stationary_200km))], POD_stationary_200km[np.where(CSI_stationary_200km == np.max(CSI_stationary_200km))], color='green', marker='*', markersize=9)
            plt.text(0.01, 0.13, s=str('200km: %.4f' % np.max(CSI_stationary_200km)), color='green')
            plt.text(0.01, 0.17, s='CSI values', style='oblique')
            plt.xlabel("Success Ratio (1 - FAR)")
            plt.ylabel("Probability of Detection (POD)")
            plt.title("Model %d Performance for Stationary Fronts" % model_number)
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.savefig("%s/model_%d/model_%d_performance_stationary_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
            plt.close()

            plt.figure()
            plt.plot(F1_stationary_50km, color='red', label='50km boundary')
            plt.plot(np.where(F1_stationary_50km == np.max(F1_stationary_50km)), np.max(F1_stationary_50km), color='red', marker='*', markersize=9)
            plt.text(len(F1_stationary_200km)*0.02, np.max(F1_stationary_200km)*0.01, s=str('50km: %.4f' % np.max(F1_stationary_50km)), color='red')
            plt.plot(F1_stationary_100km, color='purple', label='100km boundary')
            plt.plot(np.where(F1_stationary_100km == np.max(F1_stationary_100km)), np.max(F1_stationary_100km), color='purple', marker='*', markersize=9)
            plt.text(len(F1_stationary_200km)*0.02, np.max(F1_stationary_200km)*0.05, s=str('100km: %.4f' % np.max(F1_stationary_100km)), color='purple')
            plt.plot(F1_stationary_150km, color='brown', label='150km boundary')
            plt.plot(np.where(F1_stationary_150km == np.max(F1_stationary_150km)), np.max(F1_stationary_150km), color='brown', marker='*', markersize=9)
            plt.text(len(F1_stationary_200km)*0.02, np.max(F1_stationary_200km)*0.09, s=str('150km: %.4f' % np.max(F1_stationary_150km)), color='brown')
            plt.plot(F1_stationary_200km, color='green', label='200km boundary')
            plt.plot(np.where(F1_stationary_200km == np.max(F1_stationary_200km)), np.max(F1_stationary_200km), color='green', marker='*', markersize=9)
            plt.text(len(F1_stationary_200km)*0.02, np.max(F1_stationary_200km)*0.13, s=str('200km: %.4f' % np.max(F1_stationary_200km)), color='green')
            plt.text(len(F1_stationary_200km)*0.02, np.max(F1_stationary_200km)*0.17, s='F1 scores', style='oblique')
            plt.xlim(0,100)
            plt.ylim(0)
            plt.xlabel("Probability Threshold (%)")
            plt.ylabel("F1 Score")
            plt.title("Model %d F1 Score for Stationary Fronts" % model_number)
            plt.savefig("%s/model_%d/model_%d_F1_stationary_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
            plt.close()

            plt.figure()
            plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
            plt.colorbar(label='Critical Success Index (CSI)')
            plt.plot(SR_occluded_50km, POD_occluded_50km, color='red', label='50km boundary')
            plt.plot(SR_occluded_50km[np.where(CSI_occluded_50km == np.max(CSI_occluded_50km))], POD_occluded_50km[np.where(CSI_occluded_50km == np.max(CSI_occluded_50km))], color='red', marker='*', markersize=9)
            plt.text(0.01, 0.01, s=str('50km: %.4f' % np.max(CSI_occluded_50km)), color='red')
            plt.plot(SR_occluded_100km, POD_occluded_100km, color='purple', label='100km boundary')
            plt.plot(SR_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))], POD_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))], color='purple', marker='*', markersize=9)
            plt.text(0.01, 0.05, s=str('100km: %.4f' % np.max(CSI_occluded_100km)), color='purple')
            plt.plot(SR_occluded_150km, POD_occluded_150km, color='brown', label='150km boundary')
            plt.plot(SR_occluded_150km[np.where(CSI_occluded_150km == np.max(CSI_occluded_150km))], POD_occluded_150km[np.where(CSI_occluded_150km == np.max(CSI_occluded_150km))], color='brown', marker='*', markersize=9)
            plt.text(0.01, 0.09, s=str('150km: %.4f' % np.max(CSI_occluded_150km)), color='brown')
            plt.plot(SR_occluded_200km, POD_occluded_200km, color='green', label='200km boundary')
            plt.plot(SR_occluded_200km[np.where(CSI_occluded_200km == np.max(CSI_occluded_200km))], POD_occluded_200km[np.where(CSI_occluded_200km == np.max(CSI_occluded_200km))], color='green', marker='*', markersize=9)
            plt.text(0.01, 0.13, s=str('200km: %.4f' % np.max(CSI_occluded_200km)), color='green')
            plt.text(0.01, 0.17, s='CSI values', style='oblique')
            plt.xlabel("Success Ratio (1 - FAR)")
            plt.ylabel("Probability of Detection (POD)")
            plt.title("Model %d Performance for Occluded Fronts" % model_number)
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.savefig("%s/model_%d/model_%d_performance_occluded_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
            plt.close()

            plt.figure()
            plt.plot(F1_occluded_50km, color='red', label='50km boundary')
            plt.plot(np.where(F1_occluded_50km == np.max(F1_occluded_50km)), np.max(F1_occluded_50km), color='red', marker='*', markersize=9)
            plt.text(len(F1_occluded_200km)*0.02, np.max(F1_occluded_200km)*0.01, s=str('50km: %.4f' % np.max(F1_occluded_50km)), color='red')
            plt.plot(F1_occluded_100km, color='purple', label='100km boundary')
            plt.plot(np.where(F1_occluded_100km == np.max(F1_occluded_100km)), np.max(F1_occluded_100km), color='purple', marker='*', markersize=9)
            plt.text(len(F1_occluded_200km)*0.02, np.max(F1_occluded_200km)*0.05, s=str('100km: %.4f' % np.max(F1_occluded_100km)), color='purple')
            plt.plot(F1_occluded_150km, color='brown', label='150km boundary')
            plt.plot(np.where(F1_occluded_150km == np.max(F1_occluded_150km)), np.max(F1_occluded_150km), color='brown', marker='*', markersize=9)
            plt.text(len(F1_occluded_200km)*0.02, np.max(F1_occluded_200km)*0.09, s=str('150km: %.4f' % np.max(F1_occluded_150km)), color='brown')
            plt.plot(F1_occluded_200km, color='green', label='200km boundary')
            plt.plot(np.where(F1_occluded_200km == np.max(F1_occluded_200km)), np.max(F1_occluded_200km), color='green', marker='*', markersize=9)
            plt.text(len(F1_occluded_200km)*0.02, np.max(F1_occluded_200km)*0.13, s=str('200km: %.4f' % np.max(F1_occluded_200km)), color='green')
            plt.text(len(F1_occluded_200km)*0.02, np.max(F1_occluded_200km)*0.17, s='F1 scores', style='oblique')
            plt.xlim(0,100)
            plt.ylim(0)
            plt.xlabel("Probability Threshold (%)")
            plt.ylabel("F1 Score")
            plt.title("Model %d F1 Score for Occluded Fronts" % model_number)
            plt.savefig("%s/model_%d/model_%d_F1_occluded_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
            plt.close()

            AUC_stationary_50km, AUC_stationary_100km, AUC_stationary_150km, AUC_stationary_200km = 0,0,0,0
            AUC_occluded_50km, AUC_occluded_100km, AUC_occluded_150km, AUC_occluded_200km = 0,0,0,0

            POD_stationary_50km = np.append(np.array([1]),np.append(POD_stationary_50km[0:99],0))
            POD_stationary_100km = np.append(np.array([1]),np.append(POD_stationary_100km[0:99],0))
            POD_stationary_150km = np.append(np.array([1]),np.append(POD_stationary_150km[0:99],0))
            POD_stationary_200km = np.append(np.array([1]),np.append(POD_stationary_200km[0:99],0))
            POFD_stationary_50km = np.append(np.array([1]),np.append(POFD_stationary_50km[0:99],0))
            POFD_stationary_100km = np.append(np.array([1]),np.append(POFD_stationary_100km[0:99],0))
            POFD_stationary_150km = np.append(np.array([1]),np.append(POFD_stationary_150km[0:99],0))
            POFD_stationary_200km = np.append(np.array([1]),np.append(POFD_stationary_200km[0:99],0))

            POD_occluded_50km = np.append(np.array([1]),np.append(POD_occluded_50km[0:99],0))
            POD_occluded_100km = np.append(np.array([1]),np.append(POD_occluded_100km[0:99],0))
            POD_occluded_150km = np.append(np.array([1]),np.append(POD_occluded_150km[0:99],0))
            POD_occluded_200km = np.append(np.array([1]),np.append(POD_occluded_200km[0:99],0))
            POFD_occluded_50km = np.append(np.array([1]),np.append(POFD_occluded_50km[0:99],0))
            POFD_occluded_100km = np.append(np.array([1]),np.append(POFD_occluded_100km[0:99],0))
            POFD_occluded_150km = np.append(np.array([1]),np.append(POFD_occluded_150km[0:99],0))
            POFD_occluded_200km = np.append(np.array([1]),np.append(POFD_occluded_200km[0:99],0))

            for threshold in range(99):
                AUC_stationary_50km += POD_stationary_50km[threshold]*(POFD_stationary_50km[threshold]-POFD_stationary_50km[threshold+1]) - \
                    0.5*(POFD_stationary_50km[threshold]-POFD_stationary_50km[threshold+1])*(POD_stationary_50km[threshold]-POD_stationary_50km[threshold+1])
                AUC_stationary_100km += POD_stationary_100km[threshold]*(POFD_stationary_100km[threshold]-POFD_stationary_100km[threshold+1]) - \
                    0.5*(POFD_stationary_100km[threshold]-POFD_stationary_100km[threshold+1])*(POD_stationary_100km[threshold]-POD_stationary_100km[threshold+1])
                AUC_stationary_150km += POD_stationary_150km[threshold]*(POFD_stationary_150km[threshold]-POFD_stationary_150km[threshold+1]) - \
                    0.5*(POFD_stationary_150km[threshold]-POFD_stationary_150km[threshold+1])*(POD_stationary_150km[threshold]-POD_stationary_150km[threshold+1])
                AUC_stationary_200km += POD_stationary_200km[threshold]*(POFD_stationary_200km[threshold]-POFD_stationary_200km[threshold+1]) - \
                    0.5*(POFD_stationary_200km[threshold]-POFD_stationary_200km[threshold+1])*(POD_stationary_200km[threshold]-POD_stationary_200km[threshold+1])
                AUC_occluded_50km += POD_occluded_50km[threshold]*(POFD_occluded_50km[threshold]-POFD_occluded_50km[threshold+1]) - \
                    0.5*(POFD_occluded_50km[threshold]-POFD_occluded_50km[threshold+1])*(POD_occluded_50km[threshold]-POD_occluded_50km[threshold+1])
                AUC_occluded_100km += POD_occluded_100km[threshold]*(POFD_occluded_100km[threshold]-POFD_occluded_100km[threshold+1]) - \
                    0.5*(POFD_occluded_100km[threshold]-POFD_occluded_100km[threshold+1])*(POD_occluded_100km[threshold]-POD_occluded_100km[threshold+1])
                AUC_occluded_150km += POD_occluded_150km[threshold]*(POFD_occluded_150km[threshold]-POFD_occluded_150km[threshold+1]) - \
                    0.5*(POFD_occluded_150km[threshold]-POFD_occluded_150km[threshold+1])*(POD_occluded_150km[threshold]-POD_occluded_150km[threshold+1])
                AUC_occluded_200km += POD_occluded_200km[threshold]*(POFD_occluded_200km[threshold]-POFD_occluded_200km[threshold+1]) - \
                    0.5*(POFD_occluded_200km[threshold]-POFD_occluded_200km[threshold+1])*(POD_occluded_200km[threshold]-POD_occluded_200km[threshold+1])

            plt.figure()
            plt.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),'k--')
            plt.plot(POFD_stationary_50km, POD_stationary_50km, color='red', label='50km boundary')
            plt.plot(POFD_stationary_100km, POD_stationary_100km, color='purple', label='100km boundary')
            plt.plot(POFD_stationary_150km, POD_stationary_150km, color='brown', label='150km boundary')
            plt.plot(POFD_stationary_200km, POD_stationary_200km, color='green', label='200km boundary')
            plt.text(0.8, 0.01, s=str('50km: %.4f' % AUC_stationary_50km), color='red')
            plt.text(0.782, 0.05, s=str('100km: %.4f' % AUC_stationary_100km), color='purple')
            plt.text(0.782, 0.09, s=str('150km: %.4f' % AUC_stationary_150km), color='brown')
            plt.text(0.782, 0.13, s=str('200km: %.4f' % AUC_stationary_200km), color='green')
            plt.text(0.605, 0.17, s='Area Under the Curve (AUC)', style='oblique')
            plt.xlabel("Probability of False Detection (POFD)")
            plt.ylabel("Probability of Detection (POD)")
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.title("Model %d ROC Curve for Stationary Fronts" % model_number)
            plt.savefig("%s/model_%d/model_%d_AUC_stationary_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
            plt.close()

            plt.figure()
            plt.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),'k--')
            plt.plot(POFD_occluded_50km, POD_occluded_50km, color='red', label='50km boundary')
            plt.plot(POFD_occluded_100km, POD_occluded_100km, color='purple', label='100km boundary')
            plt.plot(POFD_occluded_150km, POD_occluded_150km, color='brown', label='150km boundary')
            plt.plot(POFD_occluded_200km, POD_occluded_200km, color='green', label='200km boundary')
            plt.text(np.max(POFD_occluded_50km)*0.02, np.max(POD_occluded_50km)*0.80, s=str('50km: %.4f' % AUC_occluded_50km), color='red')
            plt.text(np.max(POFD_occluded_50km)*0.02, np.max(POD_occluded_50km)*0.84, s=str('100km: %.4f' % AUC_occluded_100km), color='purple')
            plt.text(np.max(POFD_occluded_50km)*0.02, np.max(POD_occluded_50km)*0.88, s=str('150km: %.4f' % AUC_occluded_150km), color='brown')
            plt.text(np.max(POFD_occluded_50km)*0.02, np.max(POD_occluded_50km)*0.92, s=str('200km: %.4f' % AUC_occluded_200km), color='green')
            plt.text(np.max(POFD_occluded_50km)*0.02, np.max(POD_occluded_50km)*0.96, s='Area Under the Curve (AUC)', style='oblique')
            plt.xlabel("Probability of False Detection (POFD)")
            plt.ylabel("Probability of Detection (POD)")
            plt.xlim(0,np.max(POFD_occluded_50km))
            plt.ylim(0,np.max(POD_occluded_50km))
            plt.title("Model %d ROC Curve for Occluded Fronts" % model_number)
            plt.savefig("%s/model_%d/model_%d_AUC_occluded_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
            plt.close()

    elif front_types == 'SFOF':
        CSI_stationary_50km = stats_50km['tp_stationary']/(stats_50km['tp_stationary'] + stats_50km['fp_stationary'] + stats_50km['fn_stationary'])
        CSI_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fp_stationary'] + stats_100km['fn_stationary'])
        CSI_stationary_150km = stats_150km['tp_stationary']/(stats_150km['tp_stationary'] + stats_150km['fp_stationary'] + stats_150km['fn_stationary'])
        CSI_stationary_200km = stats_200km['tp_stationary']/(stats_200km['tp_stationary'] + stats_200km['fp_stationary'] + stats_200km['fn_stationary'])
        CSI_occluded_50km = stats_50km['tp_occluded']/(stats_50km['tp_occluded'] + stats_50km['fp_occluded'] + stats_50km['fn_occluded'])
        CSI_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fp_occluded'] + stats_100km['fn_occluded'])
        CSI_occluded_150km = stats_150km['tp_occluded']/(stats_150km['tp_occluded'] + stats_150km['fp_occluded'] + stats_150km['fn_occluded'])
        CSI_occluded_200km = stats_200km['tp_occluded']/(stats_200km['tp_occluded'] + stats_200km['fp_occluded'] + stats_200km['fn_occluded'])

        POD_stationary_50km = stats_50km['tp_stationary']/(stats_50km['tp_stationary'] + stats_50km['fn_stationary'])
        POD_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fn_stationary'])
        POD_stationary_150km = stats_150km['tp_stationary']/(stats_150km['tp_stationary'] + stats_150km['fn_stationary'])
        POD_stationary_200km = stats_200km['tp_stationary']/(stats_200km['tp_stationary'] + stats_200km['fn_stationary'])
        POD_occluded_50km = stats_50km['tp_occluded']/(stats_50km['tp_occluded'] + stats_50km['fn_occluded'])
        POD_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fn_occluded'])
        POD_occluded_150km = stats_150km['tp_occluded']/(stats_150km['tp_occluded'] + stats_150km['fn_occluded'])
        POD_occluded_200km = stats_200km['tp_occluded']/(stats_200km['tp_occluded'] + stats_200km['fn_occluded'])

        POFD_stationary_50km = stats_50km['fp_stationary']/(stats_50km['fp_stationary'] + stats_50km['tn_stationary'])
        POFD_stationary_100km = stats_100km['fp_stationary']/(stats_100km['fp_stationary'] + stats_100km['tn_stationary'])
        POFD_stationary_150km = stats_150km['fp_stationary']/(stats_150km['fp_stationary'] + stats_150km['tn_stationary'])
        POFD_stationary_200km = stats_200km['fp_stationary']/(stats_200km['fp_stationary'] + stats_200km['tn_stationary'])
        POFD_occluded_50km = stats_50km['fp_occluded']/(stats_50km['fp_occluded'] + stats_50km['tn_occluded'])
        POFD_occluded_100km = stats_100km['fp_occluded']/(stats_100km['fp_occluded'] + stats_100km['tn_occluded'])
        POFD_occluded_150km = stats_150km['fp_occluded']/(stats_150km['fp_occluded'] + stats_150km['tn_occluded'])
        POFD_occluded_200km = stats_200km['fp_occluded']/(stats_200km['fp_occluded'] + stats_200km['tn_occluded'])

        SR_stationary_50km = stats_50km['tp_stationary']/(stats_50km['tp_stationary'] + stats_50km['fp_stationary'])
        SR_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fp_stationary'])
        SR_stationary_150km = stats_150km['tp_stationary']/(stats_150km['tp_stationary'] + stats_150km['fp_stationary'])
        SR_stationary_200km = stats_200km['tp_stationary']/(stats_200km['tp_stationary'] + stats_200km['fp_stationary'])
        SR_occluded_50km = stats_50km['tp_occluded']/(stats_50km['tp_occluded'] + stats_50km['fp_occluded'])
        SR_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fp_occluded'])
        SR_occluded_150km = stats_150km['tp_occluded']/(stats_150km['tp_occluded'] + stats_150km['fp_occluded'])
        SR_occluded_200km = stats_200km['tp_occluded']/(stats_200km['tp_occluded'] + stats_200km['fp_occluded'])

        F1_stationary_50km = 2/(1/POD_stationary_50km + 1/SR_stationary_50km)
        F1_stationary_100km = 2/(1/POD_stationary_100km + 1/SR_stationary_100km)
        F1_stationary_150km = 2/(1/POD_stationary_150km + 1/SR_stationary_150km)
        F1_stationary_200km = 2/(1/POD_stationary_200km + 1/SR_stationary_200km)
        F1_occluded_50km = 2/(1/POD_occluded_50km + 1/SR_occluded_50km)
        F1_occluded_100km = 2/(1/POD_occluded_100km + 1/SR_occluded_100km)
        F1_occluded_150km = 2/(1/POD_occluded_150km + 1/SR_occluded_150km)
        F1_occluded_200km = 2/(1/POD_occluded_200km + 1/SR_occluded_200km)

        plt.figure()
        plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
        plt.colorbar(label='Critical Success Index (CSI)')
        plt.plot(SR_stationary_50km, POD_stationary_50km, color='red', label='50km boundary')
        plt.plot(SR_stationary_50km[np.where(CSI_stationary_50km == np.max(CSI_stationary_50km))], POD_stationary_50km[np.where(CSI_stationary_50km == np.max(CSI_stationary_50km))], color='red', marker='*', markersize=9)
        plt.text(0.01, 0.01, s=str('50km: %.4f' % np.max(CSI_stationary_50km)), color='red')
        plt.plot(SR_stationary_100km, POD_stationary_100km, color='purple', label='100km boundary')
        plt.plot(SR_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))], POD_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))], color='purple', marker='*', markersize=9)
        plt.text(0.01, 0.05, s=str('100km: %.4f' % np.max(CSI_stationary_100km)), color='purple')
        plt.plot(SR_stationary_150km, POD_stationary_150km, color='brown', label='150km boundary')
        plt.plot(SR_stationary_150km[np.where(CSI_stationary_150km == np.max(CSI_stationary_150km))], POD_stationary_150km[np.where(CSI_stationary_150km == np.max(CSI_stationary_150km))], color='brown', marker='*', markersize=9)
        plt.text(0.01, 0.09, s=str('150km: %.4f' % np.max(CSI_stationary_150km)), color='brown')
        plt.plot(SR_stationary_200km, POD_stationary_200km, color='green', label='200km boundary')
        plt.plot(SR_stationary_200km[np.where(CSI_stationary_200km == np.max(CSI_stationary_200km))], POD_stationary_200km[np.where(CSI_stationary_200km == np.max(CSI_stationary_200km))], color='green', marker='*', markersize=9)
        plt.text(0.01, 0.13, s=str('200km: %.4f' % np.max(CSI_stationary_200km)), color='green')
        plt.text(0.01, 0.17, s='CSI values', style='oblique')
        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.title("Model %d Performance for Stationary Fronts" % model_number)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.savefig("%s/model_%d/model_%d_performance_stationary_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.plot(F1_stationary_50km, color='red', label='50km boundary')
        plt.plot(np.where(F1_stationary_50km == np.max(F1_stationary_50km)), np.max(F1_stationary_50km), color='red', marker='*', markersize=9)
        plt.text(len(F1_stationary_200km)*0.02, np.max(F1_stationary_200km)*0.01, s=str('50km: %.4f' % np.max(F1_stationary_50km)), color='red')
        plt.plot(F1_stationary_100km, color='purple', label='100km boundary')
        plt.plot(np.where(F1_stationary_100km == np.max(F1_stationary_100km)), np.max(F1_stationary_100km), color='purple', marker='*', markersize=9)
        plt.text(len(F1_stationary_200km)*0.02, np.max(F1_stationary_200km)*0.05, s=str('100km: %.4f' % np.max(F1_stationary_100km)), color='purple')
        plt.plot(F1_stationary_150km, color='brown', label='150km boundary')
        plt.plot(np.where(F1_stationary_150km == np.max(F1_stationary_150km)), np.max(F1_stationary_150km), color='brown', marker='*', markersize=9)
        plt.text(len(F1_stationary_200km)*0.02, np.max(F1_stationary_200km)*0.09, s=str('150km: %.4f' % np.max(F1_stationary_150km)), color='brown')
        plt.plot(F1_stationary_200km, color='green', label='200km boundary')
        plt.plot(np.where(F1_stationary_200km == np.max(F1_stationary_200km)), np.max(F1_stationary_200km), color='green', marker='*', markersize=9)
        plt.text(len(F1_stationary_200km)*0.02, np.max(F1_stationary_200km)*0.13, s=str('200km: %.4f' % np.max(F1_stationary_200km)), color='green')
        plt.text(len(F1_stationary_200km)*0.02, np.max(F1_stationary_200km)*0.17, s='F1 scores', style='oblique')
        plt.xlim(0,100)
        plt.ylim(0)
        plt.xlabel("Probability Threshold (%)")
        plt.ylabel("F1 Score")
        plt.title("Model %d F1 Score for Stationary Fronts" % model_number)
        plt.savefig("%s/model_%d/model_%d_F1_stationary_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
        plt.colorbar(label='Critical Success Index (CSI)')
        plt.plot(SR_occluded_50km, POD_occluded_50km, color='red', label='50km boundary')
        plt.plot(SR_occluded_50km[np.where(CSI_occluded_50km == np.max(CSI_occluded_50km))], POD_occluded_50km[np.where(CSI_occluded_50km == np.max(CSI_occluded_50km))], color='red', marker='*', markersize=9)
        plt.text(0.01, 0.01, s=str('50km: %.4f' % np.max(CSI_occluded_50km)), color='red')
        plt.plot(SR_occluded_100km, POD_occluded_100km, color='purple', label='100km boundary')
        plt.plot(SR_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))], POD_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))], color='purple', marker='*', markersize=9)
        plt.text(0.01, 0.05, s=str('100km: %.4f' % np.max(CSI_occluded_100km)), color='purple')
        plt.plot(SR_occluded_150km, POD_occluded_150km, color='brown', label='150km boundary')
        plt.plot(SR_occluded_150km[np.where(CSI_occluded_150km == np.max(CSI_occluded_150km))], POD_occluded_150km[np.where(CSI_occluded_150km == np.max(CSI_occluded_150km))], color='brown', marker='*', markersize=9)
        plt.text(0.01, 0.09, s=str('150km: %.4f' % np.max(CSI_occluded_150km)), color='brown')
        plt.plot(SR_occluded_200km, POD_occluded_200km, color='green', label='200km boundary')
        plt.plot(SR_occluded_200km[np.where(CSI_occluded_200km == np.max(CSI_occluded_200km))], POD_occluded_200km[np.where(CSI_occluded_200km == np.max(CSI_occluded_200km))], color='green', marker='*', markersize=9)
        plt.text(0.01, 0.13, s=str('200km: %.4f' % np.max(CSI_occluded_200km)), color='green')
        plt.text(0.01, 0.17, s='CSI values', style='oblique')
        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.title("Model %d Performance for Occluded Fronts" % model_number)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.savefig("%s/model_%d/model_%d_performance_occluded_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.plot(F1_occluded_50km, color='red', label='50km boundary')
        plt.plot(np.where(F1_occluded_50km == np.max(F1_occluded_50km)), np.max(F1_occluded_50km), color='red', marker='*', markersize=9)
        plt.text(len(F1_occluded_200km)*0.02, np.max(F1_occluded_200km)*0.01, s=str('50km: %.4f' % np.max(F1_occluded_50km)), color='red')
        plt.plot(F1_occluded_100km, color='purple', label='100km boundary')
        plt.plot(np.where(F1_occluded_100km == np.max(F1_occluded_100km)), np.max(F1_occluded_100km), color='purple', marker='*', markersize=9)
        plt.text(len(F1_occluded_200km)*0.02, np.max(F1_occluded_200km)*0.05, s=str('100km: %.4f' % np.max(F1_occluded_100km)), color='purple')
        plt.plot(F1_occluded_150km, color='brown', label='150km boundary')
        plt.plot(np.where(F1_occluded_150km == np.max(F1_occluded_150km)), np.max(F1_occluded_150km), color='brown', marker='*', markersize=9)
        plt.text(len(F1_occluded_200km)*0.02, np.max(F1_occluded_200km)*0.09, s=str('150km: %.4f' % np.max(F1_occluded_150km)), color='brown')
        plt.plot(F1_occluded_200km, color='green', label='200km boundary')
        plt.plot(np.where(F1_occluded_200km == np.max(F1_occluded_200km)), np.max(F1_occluded_200km), color='green', marker='*', markersize=9)
        plt.text(len(F1_occluded_200km)*0.02, np.max(F1_occluded_200km)*0.13, s=str('200km: %.4f' % np.max(F1_occluded_200km)), color='green')
        plt.text(len(F1_occluded_200km)*0.02, np.max(F1_occluded_200km)*0.17, s='F1 scores', style='oblique')
        plt.xlim(0,100)
        plt.ylim(0)
        plt.xlabel("Probability Threshold (%)")
        plt.ylabel("F1 Score")
        plt.title("Model %d F1 Score for Occluded Fronts" % model_number)
        plt.savefig("%s/model_%d/model_%d_F1_occluded_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
        plt.close()

        AUC_stationary_50km, AUC_stationary_100km, AUC_stationary_150km, AUC_stationary_200km = 0,0,0,0
        AUC_occluded_50km, AUC_occluded_100km, AUC_occluded_150km, AUC_occluded_200km = 0,0,0,0

        POD_stationary_50km = np.append(np.array([1]),np.append(POD_stationary_50km[0:99],0))
        POD_stationary_100km = np.append(np.array([1]),np.append(POD_stationary_100km[0:99],0))
        POD_stationary_150km = np.append(np.array([1]),np.append(POD_stationary_150km[0:99],0))
        POD_stationary_200km = np.append(np.array([1]),np.append(POD_stationary_200km[0:99],0))
        POFD_stationary_50km = np.append(np.array([1]),np.append(POFD_stationary_50km[0:99],0))
        POFD_stationary_100km = np.append(np.array([1]),np.append(POFD_stationary_100km[0:99],0))
        POFD_stationary_150km = np.append(np.array([1]),np.append(POFD_stationary_150km[0:99],0))
        POFD_stationary_200km = np.append(np.array([1]),np.append(POFD_stationary_200km[0:99],0))

        POD_occluded_50km = np.append(np.array([1]),np.append(POD_occluded_50km[0:99],0))
        POD_occluded_100km = np.append(np.array([1]),np.append(POD_occluded_100km[0:99],0))
        POD_occluded_150km = np.append(np.array([1]),np.append(POD_occluded_150km[0:99],0))
        POD_occluded_200km = np.append(np.array([1]),np.append(POD_occluded_200km[0:99],0))
        POFD_occluded_50km = np.append(np.array([1]),np.append(POFD_occluded_50km[0:99],0))
        POFD_occluded_100km = np.append(np.array([1]),np.append(POFD_occluded_100km[0:99],0))
        POFD_occluded_150km = np.append(np.array([1]),np.append(POFD_occluded_150km[0:99],0))
        POFD_occluded_200km = np.append(np.array([1]),np.append(POFD_occluded_200km[0:99],0))

        for threshold in range(99):
            AUC_stationary_50km += POD_stationary_50km[threshold]*(POFD_stationary_50km[threshold]-POFD_stationary_50km[threshold+1]) - \
                0.5*(POFD_stationary_50km[threshold]-POFD_stationary_50km[threshold+1])*(POD_stationary_50km[threshold]-POD_stationary_50km[threshold+1])
            AUC_stationary_100km += POD_stationary_100km[threshold]*(POFD_stationary_100km[threshold]-POFD_stationary_100km[threshold+1]) - \
                0.5*(POFD_stationary_100km[threshold]-POFD_stationary_100km[threshold+1])*(POD_stationary_100km[threshold]-POD_stationary_100km[threshold+1])
            AUC_stationary_150km += POD_stationary_150km[threshold]*(POFD_stationary_150km[threshold]-POFD_stationary_150km[threshold+1]) - \
                0.5*(POFD_stationary_150km[threshold]-POFD_stationary_150km[threshold+1])*(POD_stationary_150km[threshold]-POD_stationary_150km[threshold+1])
            AUC_stationary_200km += POD_stationary_200km[threshold]*(POFD_stationary_200km[threshold]-POFD_stationary_200km[threshold+1]) - \
                0.5*(POFD_stationary_200km[threshold]-POFD_stationary_200km[threshold+1])*(POD_stationary_200km[threshold]-POD_stationary_200km[threshold+1])
            AUC_occluded_50km += POD_occluded_50km[threshold]*(POFD_occluded_50km[threshold]-POFD_occluded_50km[threshold+1]) - \
                0.5*(POFD_occluded_50km[threshold]-POFD_occluded_50km[threshold+1])*(POD_occluded_50km[threshold]-POD_occluded_50km[threshold+1])
            AUC_occluded_100km += POD_occluded_100km[threshold]*(POFD_occluded_100km[threshold]-POFD_occluded_100km[threshold+1]) - \
                0.5*(POFD_occluded_100km[threshold]-POFD_occluded_100km[threshold+1])*(POD_occluded_100km[threshold]-POD_occluded_100km[threshold+1])
            AUC_occluded_150km += POD_occluded_150km[threshold]*(POFD_occluded_150km[threshold]-POFD_occluded_150km[threshold+1]) - \
                0.5*(POFD_occluded_150km[threshold]-POFD_occluded_150km[threshold+1])*(POD_occluded_150km[threshold]-POD_occluded_150km[threshold+1])
            AUC_occluded_200km += POD_occluded_200km[threshold]*(POFD_occluded_200km[threshold]-POFD_occluded_200km[threshold+1]) - \
                0.5*(POFD_occluded_200km[threshold]-POFD_occluded_200km[threshold+1])*(POD_occluded_200km[threshold]-POD_occluded_200km[threshold+1])

        plt.figure()
        plt.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),'k--')
        plt.plot(POFD_stationary_50km, POD_stationary_50km, color='red', label='50km boundary')
        plt.plot(POFD_stationary_100km, POD_stationary_100km, color='purple', label='100km boundary')
        plt.plot(POFD_stationary_150km, POD_stationary_150km, color='brown', label='150km boundary')
        plt.plot(POFD_stationary_200km, POD_stationary_200km, color='green', label='200km boundary')
        plt.text(0.8, 0.01, s=str('50km: %.4f' % AUC_stationary_50km), color='red')
        plt.text(0.782, 0.05, s=str('100km: %.4f' % AUC_stationary_100km), color='purple')
        plt.text(0.782, 0.09, s=str('150km: %.4f' % AUC_stationary_150km), color='brown')
        plt.text(0.782, 0.13, s=str('200km: %.4f' % AUC_stationary_200km), color='green')
        plt.text(0.605, 0.17, s='Area Under the Curve (AUC)', style='oblique')
        plt.xlabel("Probability of False Detection (POFD)")
        plt.ylabel("Probability of Detection (POD)")
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.title("Model %d ROC Curve for Stationary Fronts" % model_number)
        plt.savefig("%s/model_%d/model_%d_AUC_stationary_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),'k--')
        plt.plot(POFD_occluded_50km, POD_occluded_50km, color='red', label='50km boundary')
        plt.plot(POFD_occluded_100km, POD_occluded_100km, color='purple', label='100km boundary')
        plt.plot(POFD_occluded_150km, POD_occluded_150km, color='brown', label='150km boundary')
        plt.plot(POFD_occluded_200km, POD_occluded_200km, color='green', label='200km boundary')
        plt.text(np.max(POFD_occluded_50km)*0.02, np.max(POD_occluded_50km)*0.80, s=str('50km: %.4f' % AUC_occluded_50km), color='red')
        plt.text(np.max(POFD_occluded_50km)*0.02, np.max(POD_occluded_50km)*0.84, s=str('100km: %.4f' % AUC_occluded_100km), color='purple')
        plt.text(np.max(POFD_occluded_50km)*0.02, np.max(POD_occluded_50km)*0.88, s=str('150km: %.4f' % AUC_occluded_150km), color='brown')
        plt.text(np.max(POFD_occluded_50km)*0.02, np.max(POD_occluded_50km)*0.92, s=str('200km: %.4f' % AUC_occluded_200km), color='green')
        plt.text(np.max(POFD_occluded_50km)*0.02, np.max(POD_occluded_50km)*0.96, s='Area Under the Curve (AUC)', style='oblique')
        plt.xlabel("Probability of False Detection (POFD)")
        plt.ylabel("Probability of Detection (POD)")
        plt.xlim(0,np.max(POFD_occluded_50km))
        plt.ylim(0,np.max(POD_occluded_50km))
        plt.title("Model %d ROC Curve for Occluded Fronts" % model_number)
        plt.savefig("%s/model_%d/model_%d_AUC_occluded_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
        plt.close()

    elif front_types == 'ALL_bin':
        CSI_front_50km = stats_50km['tp_front']/(stats_50km['tp_front'] + stats_50km['fp_front'] + stats_50km['fn_front'])
        CSI_front_100km = stats_100km['tp_front']/(stats_100km['tp_front'] + stats_100km['fp_front'] + stats_100km['fn_front'])
        CSI_front_150km = stats_150km['tp_front']/(stats_150km['tp_front'] + stats_150km['fp_front'] + stats_150km['fn_front'])
        CSI_front_200km = stats_200km['tp_front']/(stats_200km['tp_front'] + stats_200km['fp_front'] + stats_200km['fn_front'])

        POD_front_50km = stats_50km['tp_front']/(stats_50km['tp_front'] + stats_50km['fn_front'])
        POD_front_100km = stats_100km['tp_front']/(stats_100km['tp_front'] + stats_100km['fn_front'])
        POD_front_150km = stats_150km['tp_front']/(stats_150km['tp_front'] + stats_150km['fn_front'])
        POD_front_200km = stats_200km['tp_front']/(stats_200km['tp_front'] + stats_200km['fn_front'])

        POFD_front_50km = stats_50km['fp_front']/(stats_50km['fp_front'] + stats_50km['tn_front'])
        POFD_front_100km = stats_100km['fp_front']/(stats_100km['fp_front'] + stats_100km['tn_front'])
        POFD_front_150km = stats_150km['fp_front']/(stats_150km['fp_front'] + stats_150km['tn_front'])
        POFD_front_200km = stats_200km['fp_front']/(stats_200km['fp_front'] + stats_200km['tn_front'])

        SR_front_50km = stats_50km['tp_front']/(stats_50km['tp_front'] + stats_50km['fp_front'])
        SR_front_100km = stats_100km['tp_front']/(stats_100km['tp_front'] + stats_100km['fp_front'])
        SR_front_150km = stats_150km['tp_front']/(stats_150km['tp_front'] + stats_150km['fp_front'])
        SR_front_200km = stats_200km['tp_front']/(stats_200km['tp_front'] + stats_200km['fp_front'])

        F1_front_50km = 2/(1/POD_front_50km + 1/SR_front_50km)
        F1_front_100km = 2/(1/POD_front_100km + 1/SR_front_100km)
        F1_front_150km = 2/(1/POD_front_150km + 1/SR_front_150km)
        F1_front_200km = 2/(1/POD_front_200km + 1/SR_front_200km)

        plt.figure()
        plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
        plt.colorbar(label='Critical Success Index (CSI)')
        plt.plot(SR_front_50km, POD_front_50km, color='red', label='50km boundary')
        plt.plot(SR_front_50km[np.where(CSI_front_50km == np.max(CSI_front_50km))], POD_front_50km[np.where(CSI_front_50km == np.max(CSI_front_50km))], color='red', marker='*', markersize=9)
        plt.text(0.01, 0.01, s=str('50km: %.4f' % np.max(CSI_front_50km)), color='red')
        plt.plot(SR_front_100km, POD_front_100km, color='purple', label='100km boundary')
        plt.plot(SR_front_100km[np.where(CSI_front_100km == np.max(CSI_front_100km))], POD_front_100km[np.where(CSI_front_100km == np.max(CSI_front_100km))], color='purple', marker='*', markersize=9)
        plt.text(0.01, 0.05, s=str('100km: %.4f' % np.max(CSI_front_100km)), color='purple')
        plt.plot(SR_front_150km, POD_front_150km, color='brown', label='150km boundary')
        plt.plot(SR_front_150km[np.where(CSI_front_150km == np.max(CSI_front_150km))], POD_front_150km[np.where(CSI_front_150km == np.max(CSI_front_150km))], color='brown', marker='*', markersize=9)
        plt.text(0.01, 0.09, s=str('150km: %.4f' % np.max(CSI_front_150km)), color='brown')
        plt.plot(SR_front_200km, POD_front_200km, color='green', label='200km boundary')
        plt.plot(SR_front_200km[np.where(CSI_front_200km == np.max(CSI_front_200km))], POD_front_200km[np.where(CSI_front_200km == np.max(CSI_front_200km))], color='green', marker='*', markersize=9)
        plt.text(0.01, 0.13, s=str('200km: %.4f' % np.max(CSI_front_200km)), color='green')
        plt.text(0.01, 0.17, s='CSI values', style='oblique')
        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.title("Model %d Performance for fronts" % model_number)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.savefig("%s/model_%d/model_%d_performance_front_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.plot(F1_front_50km, color='red', label='50km boundary')
        plt.plot(np.where(F1_front_50km == np.max(F1_front_50km)), np.max(F1_front_50km), color='red', marker='*', markersize=9)
        plt.text(len(F1_front_200km)*0.02, np.max(F1_front_200km)*0.01, s=str('50km: %.4f' % np.max(F1_front_50km)), color='red')
        plt.plot(F1_front_100km, color='purple', label='100km boundary')
        plt.plot(np.where(F1_front_100km == np.max(F1_front_100km)), np.max(F1_front_100km), color='purple', marker='*', markersize=9)
        plt.text(len(F1_front_200km)*0.02, np.max(F1_front_200km)*0.05, s=str('100km: %.4f' % np.max(F1_front_100km)), color='purple')
        plt.plot(F1_front_150km, color='brown', label='150km boundary')
        plt.plot(np.where(F1_front_150km == np.max(F1_front_150km)), np.max(F1_front_150km), color='brown', marker='*', markersize=9)
        plt.text(len(F1_front_200km)*0.02, np.max(F1_front_200km)*0.09, s=str('150km: %.4f' % np.max(F1_front_150km)), color='brown')
        plt.plot(F1_front_200km, color='green', label='200km boundary')
        plt.plot(np.where(F1_front_200km == np.max(F1_front_200km)), np.max(F1_front_200km), color='green', marker='*', markersize=9)
        plt.text(len(F1_front_200km)*0.02, np.max(F1_front_200km)*0.13, s=str('200km: %.4f' % np.max(F1_front_200km)), color='green')
        plt.text(len(F1_front_200km)*0.02, np.max(F1_front_200km)*0.17, s='F1 scores', style='oblique')
        plt.xlim(0,100)
        plt.ylim(0)
        plt.xlabel("Probability Threshold (%)")
        plt.ylabel("F1 Score")
        plt.title("Model %d F1 Score for fronts" % model_number)
        plt.savefig("%s/model_%d/model_%d_F1_front_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
        plt.close()

        POD_front_50km = np.append(np.array([1]),np.append(POD_front_50km[0:99],0))
        POD_front_100km = np.append(np.array([1]),np.append(POD_front_100km[0:99],0))
        POD_front_150km = np.append(np.array([1]),np.append(POD_front_150km[0:99],0))
        POD_front_200km = np.append(np.array([1]),np.append(POD_front_200km[0:99],0))
        POFD_front_50km = np.append(np.array([1]),np.append(POFD_front_50km[0:99],0))
        POFD_front_100km = np.append(np.array([1]),np.append(POFD_front_100km[0:99],0))
        POFD_front_150km = np.append(np.array([1]),np.append(POFD_front_150km[0:99],0))
        POFD_front_200km = np.append(np.array([1]),np.append(POFD_front_200km[0:99],0))

        AUC_front_50km, AUC_front_100km, AUC_front_150km, AUC_front_200km = 0,0,0,0

        for threshold in range(99):
            AUC_front_50km += POD_front_50km[threshold]*(POFD_front_50km[threshold]-POFD_front_50km[threshold+1]) - \
                0.5*(POFD_front_50km[threshold]-POFD_front_50km[threshold+1])*(POD_front_50km[threshold]-POD_front_50km[threshold+1])
            AUC_front_100km += POD_front_100km[threshold]*(POFD_front_100km[threshold]-POFD_front_100km[threshold+1]) - \
                0.5*(POFD_front_100km[threshold]-POFD_front_100km[threshold+1])*(POD_front_100km[threshold]-POD_front_100km[threshold+1])
            AUC_front_150km += POD_front_150km[threshold]*(POFD_front_150km[threshold]-POFD_front_150km[threshold+1]) - \
                0.5*(POFD_front_150km[threshold]-POFD_front_150km[threshold+1])*(POD_front_150km[threshold]-POD_front_150km[threshold+1])
            AUC_front_200km += POD_front_200km[threshold]*(POFD_front_200km[threshold]-POFD_front_200km[threshold+1]) - \
                0.5*(POFD_front_200km[threshold]-POFD_front_200km[threshold+1])*(POD_front_200km[threshold]-POD_front_200km[threshold+1])

        plt.figure()
        plt.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),'k--')
        plt.plot(POFD_front_50km, POD_front_50km, color='red', label='50km boundary')
        plt.plot(POFD_front_100km, POD_front_100km, color='purple', label='100km boundary')
        plt.plot(POFD_front_150km, POD_front_150km, color='brown', label='150km boundary')
        plt.plot(POFD_front_200km, POD_front_200km, color='green', label='200km boundary')
        plt.text(0.8, 0.01, s=str('50km: %.4f' % AUC_front_50km), color='red')
        plt.text(0.782, 0.05, s=str('100km: %.4f' % AUC_front_100km), color='purple')
        plt.text(0.782, 0.09, s=str('150km: %.4f' % AUC_front_150km), color='brown')
        plt.text(0.782, 0.13, s=str('200km: %.4f' % AUC_front_200km), color='green')
        plt.text(0.605, 0.17, s='Area Under the Curve (AUC)', style='oblique')
        plt.xlabel("Probability of False Detection (POFD)")
        plt.ylabel("Probability of Detection (POD)")
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.title("Model %d ROC Curve for fronts" % model_number)
        plt.savefig("%s/model_%d/model_%d_AUC_front_%dx%dimages_%dx%dtrim.png" % (model_dir, model_number, model_number, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]),bbox_inches='tight')
        plt.close()


def prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, domain_images, domain_trim,
                    same_map=True, probability_mask=0.10):
    """
    Function that uses generated predictions to make probability maps along with the 'true' fronts and saves out the
    subplots.

    Parameters
    ----------
    domain_images: Number of images along each dimension of the final stitched map (lon lat).
    domain_trim: Number of pixels to trim each image by along each dimension before taking the maximum of the overlapping pixels (lon lat).
    front_types: Fronts in the data.
    fronts: Xarray DataArray containing the 'true' front data.
    model_number: Slurm job number for the model. This is the number in the model's filename.
    model_dir: Main directory for the models.
    pixel_expansion: Number of pixels to expand the fronts by in all directions.
    probability_mask: Mask for front probabilities. Any probabilities smaller than this number will not be plotted.
        (Must be a multiple of 0.1, greater than 0, and no greater than 0.9)
    probs_ds: Xarray dataset containing prediction (probability) data for fronts.
    same_map: Setting this to true will plot the probabilities and the ground truth on the same plot.
    time: Timestring for the prediction plot title.
    """
    # extent = np.array([220, 300, 25, 52])  # Extent for CONUS
    extent = np.array([120, 380, 0, 80])  # Extent for full domain
    crs = ccrs.LambertConformal(central_longitude=250)

    cmap_cold, cmap_warm, cmap_stationary, cmap_occluded, cmap_front = cm.get_cmap('Blues', 10), cm.get_cmap('Reds', 10), \
        cm.get_cmap('Greens', 10), cm.get_cmap('Purples', 10), cm.get_cmap('Greys', 10)

    prob_norm = mpl.colors.Normalize(vmin=0, vmax=1)
    levels = np.arange(0,1.1,0.1)
    cbar_ticks = np.arange(probability_mask,1.1,0.1)
    cbar_tick_labels = [None, "10% ≤ P(front) < 20%", "20% ≤ P(front) < 30%", "30% ≤ P(front) < 40%", "40% ≤ P(front) < 50%",
                        "50% ≤ P(front) < 60%", "60% ≤ P(front) < 70%", "70% ≤ P(front) < 80%", "80% ≤ P(front) < 90%",
                        "P(front) ≥ 90%"]

    probability_plot_title = "%s front probabilities (images=[%d,%d], trim=[%d,%d], mask=%d%s)" % (time,
        domain_images[0], domain_images[1], domain_trim[0], domain_trim[1], int(probability_mask*100), "%")

    for expansion in range(pixel_expansion):
        fronts = ope(fronts)  # ope: one_pixel_expansion function in expand_fronts.py

    if front_types == 'ALL_bin':
        probs_ds = xr.where(probs_ds > probability_mask, probs_ds, float("NaN"))
    else:
        probs_ds = xr.where(probs_ds > probability_mask, probs_ds, 0)

    fronts = xr.where(fronts == 0, float("NaN"), fronts)
    fronts = xr.where(fronts == 5, float("NaN"), fronts)

    if front_types == 'CFWF':

        # Create custom colorbar for the different front types of the 'truth' plot
        cmap_identifier = mpl.colors.ListedColormap(['blue','red'], name='from_list', N=None)
        norm_identifier = mpl.colors.Normalize(vmin=1,vmax=3)

        """ Create datasets for each front type and modify them so they do not overlap in the prediction plot """
        probs_ds_copy_cold, probs_ds_copy_warm = probs_ds, probs_ds
        probs_ds_cold = probs_ds_copy_cold.drop(labels="warm_probs").rename({"cold_probs": "probs"})
        probs_ds_warm = probs_ds_copy_warm.drop(labels="cold_probs").rename({"warm_probs": "probs"})
        cold_ds = xr.where(probs_ds_cold > probs_ds_warm, probs_ds_cold, float("NaN")).rename({"probs": "cold_probs"})
        probs_ds_copy_cold, probs_ds_copy_warm = probs_ds, probs_ds
        probs_ds_cold = probs_ds_copy_cold.drop(labels="warm_probs").rename({"cold_probs": "probs"})
        probs_ds_warm = probs_ds_copy_warm.drop(labels="cold_probs").rename({"warm_probs": "probs"})
        warm_ds = xr.where(probs_ds_warm > probs_ds_cold, probs_ds_warm, float("NaN")).rename({"probs": "warm_probs"})

        # Create prediction plot
        if same_map is True:
            fig, ax = plt.subplots(1, 1, figsize=(20, 5), subplot_kw={'projection': crs})
            fplot.plot_background(ax, extent)  # Plot background on main subplot containing fronts and probabilities
            cold_ds.cold_probs.isel().plot.contourf(ax=ax, x='longitude', y='latitude', norm=prob_norm, levels=levels, cmap="Blues",
                transform=ccrs.PlateCarree(), alpha=0.60, add_colorbar=False)
            warm_ds.warm_probs.isel().plot.contourf(ax=ax, x='longitude', y='latitude', norm=prob_norm, levels=levels, cmap='Reds',
                transform=ccrs.PlateCarree(), alpha=0.60, add_colorbar=False)
            fronts.identifier.plot(ax=ax, cmap=cmap_identifier, norm=norm_identifier, x='longitude', y='latitude', transform=ccrs.PlateCarree(), add_colorbar=False)
            ax.title.set_text(probability_plot_title)

            cbar_cold = plt.colorbar(cm.ScalarMappable(norm=prob_norm, cmap=cmap_cold), ax=ax, pad=-0.165, boundaries=cbar_ticks, alpha=0.6)
            cbar_cold.set_ticks([])
            cbar_warm = plt.colorbar(cm.ScalarMappable(norm=prob_norm, cmap=cmap_warm), ax=ax, pad=-0.4, boundaries=cbar_ticks, alpha=0.6)
            cbar_warm.set_ticks(cbar_ticks + 0.05)
            cbar_warm.set_ticklabels(cbar_tick_labels[int(probability_mask*10):])

            cbar = plt.colorbar(cm.ScalarMappable(norm=norm_identifier, cmap=cmap_identifier), ax=ax, orientation='horizontal', fraction=0.046)
            cbar.set_ticks(np.arange(1,5,1)+0.5)
            cbar.set_ticklabels(['Cold','Warm'])
            cbar.set_label('Front type')

        else:
            fig, axarr = plt.subplots(2, 1, figsize=(20, 8), subplot_kw={'projection': crs})
            plt.subplots_adjust(left=None, bottom=None, right=0.6, top=None, wspace=None, hspace=0.04)
            axlist = axarr.flatten()
            for ax in axlist:
                fplot.plot_background(ax, extent)
            fronts.identifier.plot(ax=axlist[0], cmap=cmap_identifier, norm=norm_identifier, x='longitude', y='latitude', transform=ccrs.PlateCarree(), add_colorbar=False)
            axlist[0].title.set_text(probability_plot_title)
            cold_ds.cold_probs.isel().plot.contourf(ax=axlist[1], x='longitude', y='latitude', norm=prob_norm, levels=levels, cmap="Blues",
                transform=ccrs.PlateCarree(), alpha=0.60, add_colorbar=False)
            warm_ds.warm_probs.isel().plot.contourf(ax=axlist[1], x='longitude', y='latitude', norm=prob_norm, levels=levels, cmap='Reds',
                transform=ccrs.PlateCarree(), alpha=0.60, add_colorbar=False)

            cbar_cold = plt.colorbar(cm.ScalarMappable(norm=prob_norm, cmap=cmap_cold), ax=axlist[1], pad=-0.168, boundaries=cbar_ticks, alpha=0.6)
            cbar_cold.set_ticks([])
            cbar_warm = plt.colorbar(cm.ScalarMappable(norm=prob_norm, cmap=cmap_warm), ax=axlist[1], boundaries=cbar_ticks, alpha=0.6)
            cbar_warm.set_ticks(cbar_ticks + 0.05)
            cbar_warm.set_ticklabels(cbar_tick_labels[int(probability_mask*10):])

            cbar = plt.colorbar(cm.ScalarMappable(norm=norm_identifier, cmap=cmap_identifier), ax=axlist[0], pad=0.1385, fraction=0.046)
            cbar.set_ticks(np.arange(1,5,1)+0.5)
            cbar.set_ticklabels(['Cold','Warm'])
            cbar.set_label('Front type')

    elif front_types == 'SFOF':

        # Create custom colorbar for the different front types of the 'truth' plot
        cmap_identifier = mpl.colors.ListedColormap(['green','purple'], name='from_list', N=None)
        norm_identifier = mpl.colors.Normalize(vmin=3,vmax=5)

        """ Create datasets for each front type and modify them so they do not overlap in the prediction plot """
        probs_ds_copy_stationary, probs_ds_copy_occluded = probs_ds, probs_ds
        probs_ds_stationary = probs_ds_copy_stationary.drop(labels="occluded_probs").rename({"stationary_probs": "probs"})
        probs_ds_occluded = probs_ds_copy_occluded.drop(labels="stationary_probs").rename({"occluded_probs": "probs"})
        stationary_ds = xr.where(probs_ds_stationary > probs_ds_occluded, probs_ds_stationary, float("NaN")).rename({"probs": "stationary_probs"})
        probs_ds_copy_stationary, probs_ds_copy_occluded = probs_ds, probs_ds
        probs_ds_stationary = probs_ds_copy_stationary.drop(labels="occluded_probs").rename({"stationary_probs": "probs"})
        probs_ds_occluded = probs_ds_copy_occluded.drop(labels="stationary_probs").rename({"occluded_probs": "probs"})
        occluded_ds = xr.where(probs_ds_occluded > probs_ds_stationary, probs_ds_occluded, float("NaN")).rename({"probs": "occluded_probs"})

        # Create prediction plot
        if same_map is True:
            fig, ax = plt.subplots(1, 1, figsize=(20, 5), subplot_kw={'projection': crs})
            fplot.plot_background(ax, extent)  # Plot background on main subplot containing fronts and probabilities
            stationary_ds.stationary_probs.isel().plot.contourf(ax=ax, x='longitude', y='latitude', norm=prob_norm, levels=levels, cmap="Greens",
                transform=ccrs.PlateCarree(), alpha=0.60, add_colorbar=False)
            occluded_ds.occluded_probs.isel().plot.contourf(ax=ax, x='longitude', y='latitude', norm=prob_norm, levels=levels, cmap="Purples",
                transform=ccrs.PlateCarree(), alpha=0.60, add_colorbar=False)
            fronts.identifier.plot(ax=ax, cmap=cmap_identifier, norm=norm_identifier, x='longitude', y='latitude', transform=ccrs.PlateCarree(), add_colorbar=False)
            ax.title.set_text(probability_plot_title)

            cbar_stationary = plt.colorbar(cm.ScalarMappable(norm=prob_norm, cmap=cmap_stationary), ax=ax, pad=-0.165, boundaries=cbar_ticks, alpha=0.6)
            cbar_stationary.set_ticks([])
            cbar_occluded = plt.colorbar(cm.ScalarMappable(norm=prob_norm, cmap=cmap_occluded), ax=ax, pad=-0.4, boundaries=cbar_ticks, alpha=0.6)
            cbar_occluded.set_ticks(cbar_ticks + 0.05)
            cbar_occluded.set_ticklabels(cbar_tick_labels[int(probability_mask*10):])

            cbar = plt.colorbar(cm.ScalarMappable(norm=norm_identifier, cmap=cmap_identifier), ax=ax, orientation='horizontal', fraction=0.046)
            cbar.set_ticks(np.arange(1,5,1)+0.5)
            cbar.set_ticklabels(['Stationary','Occluded'])
            cbar.set_label('Front type')

        else:
            fig, axarr = plt.subplots(2, 1, figsize=(20, 8), subplot_kw={'projection': crs})
            plt.subplots_adjust(left=None, bottom=None, right=0.6, top=None, wspace=None, hspace=0.04)
            axlist = axarr.flatten()
            for ax in axlist:
                fplot.plot_background(ax, extent)
            fronts.identifier.plot(ax=axlist[0], cmap=cmap_identifier, norm=norm_identifier, x='longitude', y='latitude', transform=ccrs.PlateCarree(), add_colorbar=False)
            axlist[0].title.set_text(probability_plot_title)
            stationary_ds.stationary_probs.isel().plot.contourf(ax=axlist[1], x='longitude', y='latitude', norm=prob_norm, levels=levels, cmap="Greens",
                transform=ccrs.PlateCarree(), alpha=0.60, add_colorbar=False)
            occluded_ds.occluded_probs.isel().plot.contourf(ax=axlist[1], x='longitude', y='latitude', norm=prob_norm, levels=levels, cmap='Purples',
                transform=ccrs.PlateCarree(), alpha=0.60, add_colorbar=False)

            cbar_stationary = plt.colorbar(cm.ScalarMappable(norm=prob_norm, cmap=cmap_stationary), ax=axlist[1], pad=-0.168, boundaries=cbar_ticks, alpha=0.6)
            cbar_stationary.set_ticks([])
            cbar_occluded = plt.colorbar(cm.ScalarMappable(norm=prob_norm, cmap=cmap_occluded), ax=axlist[1], boundaries=cbar_ticks, alpha=0.6)
            cbar_occluded.set_ticks(cbar_ticks + 0.05)
            cbar_occluded.set_ticklabels(cbar_tick_labels[int(probability_mask*10):])

            cbar = plt.colorbar(cm.ScalarMappable(norm=norm_identifier, cmap=cmap_identifier), ax=axlist[0], pad=0.1385, fraction=0.046)
            cbar.set_ticks(np.arange(3,5,1)+0.5)
            cbar.set_ticklabels(['Stationary','Occluded'])
            cbar.set_label('Front type')

    elif front_types == 'ALL':

        # Create custom colorbar for the different front types of the 'truth' plot
        cmap_subplot1 = mpl.colors.ListedColormap(["blue","red"], name='from_list', N=None)
        norm_subplot1 = mpl.colors.Normalize(vmin=1,vmax=3)
        cmap_subplot2 = mpl.colors.ListedColormap(["green","purple"], name='from_list', N=None)
        norm_subplot2 = mpl.colors.Normalize(vmin=3,vmax=5)
        cmap_identifier = mpl.colors.ListedColormap(["blue","red",'green','purple'], name='from_list', N=None)
        norm_identifier = mpl.colors.Normalize(vmin=1,vmax=5)

        fronts_subplot1 = fronts
        fronts_subplot2 = fronts

        fronts_subplot1 = xr.where(fronts_subplot1 > 2, float("NaN"), fronts_subplot1)
        fronts_subplot2 = xr.where(fronts_subplot2 < 3, float("NaN"), fronts_subplot2)

        """ Create datasets for each front type and modify them so they do not overlap in the prediction plot """
        probs_ds_copy_cold, probs_ds_copy_warm = probs_ds, probs_ds
        probs_ds_cold = probs_ds_copy_cold.drop(labels="warm_probs").rename({"cold_probs": "probs"})
        probs_ds_warm = probs_ds_copy_warm.drop(labels="cold_probs").rename({"warm_probs": "probs"})
        cold_ds = xr.where(probs_ds_cold > probs_ds_warm, probs_ds_cold, float("NaN")).rename({"probs": "cold_probs"})
        probs_ds_copy_cold, probs_ds_copy_warm = probs_ds, probs_ds
        probs_ds_cold = probs_ds_copy_cold.drop(labels="warm_probs").rename({"cold_probs": "probs"})
        probs_ds_warm = probs_ds_copy_warm.drop(labels="cold_probs").rename({"warm_probs": "probs"})
        warm_ds = xr.where(probs_ds_warm > probs_ds_cold, probs_ds_warm, float("NaN")).rename({"probs": "warm_probs"})

        probs_ds_copy_stationary, probs_ds_copy_occluded = probs_ds, probs_ds
        probs_ds_stationary = probs_ds_copy_stationary.drop(labels="occluded_probs").rename({"stationary_probs": "probs"})
        probs_ds_occluded = probs_ds_copy_occluded.drop(labels="stationary_probs").rename({"occluded_probs": "probs"})
        stationary_ds = xr.where(probs_ds_stationary > probs_ds_occluded, probs_ds_stationary, float("NaN")).rename({"probs": "stationary_probs"})
        probs_ds_copy_stationary, probs_ds_copy_occluded = probs_ds, probs_ds
        probs_ds_stationary = probs_ds_copy_stationary.drop(labels="occluded_probs").rename({"stationary_probs": "probs"})
        probs_ds_occluded = probs_ds_copy_occluded.drop(labels="stationary_probs").rename({"occluded_probs": "probs"})
        occluded_ds = xr.where(probs_ds_occluded > probs_ds_stationary, probs_ds_occluded, float("NaN")).rename({"probs": "occluded_probs"})

        fig, axarr = plt.subplots(2, 1, figsize=(20, 8), subplot_kw={'projection': crs}, gridspec_kw={'height_ratios': [1,1.245]})
        plt.subplots_adjust(left=None, bottom=None, right=0.6, top=None, wspace=None, hspace=0.04)
        axlist = axarr.flatten()
        for ax in axlist:
            fplot.plot_background(ax, extent)
        cold_ds.cold_probs.isel().plot.contourf(ax=axlist[0], x='longitude', y='latitude', norm=prob_norm, levels=levels, cmap="Blues",
            transform=ccrs.PlateCarree(), alpha=0.60, add_colorbar=False)
        warm_ds.warm_probs.isel().plot.contourf(ax=axlist[0], x='longitude', y='latitude', norm=prob_norm, levels=levels, cmap='Reds',
            transform=ccrs.PlateCarree(), alpha=0.60, add_colorbar=False)
        stationary_ds.stationary_probs.isel().plot.contourf(ax=axlist[1], x='longitude', y='latitude', norm=prob_norm, levels=levels, cmap="Greens",
            transform=ccrs.PlateCarree(), alpha=0.60, add_colorbar=False)
        occluded_ds.occluded_probs.isel().plot.contourf(ax=axlist[1], x='longitude', y='latitude', norm=prob_norm, levels=levels, cmap="Purples",
            transform=ccrs.PlateCarree(), alpha=0.60, add_colorbar=False)
        fronts_subplot1.identifier.plot(ax=axlist[0], cmap=cmap_subplot1, norm=norm_subplot1, x='longitude', y='latitude', transform=ccrs.PlateCarree(), add_colorbar=False)
        fronts_subplot2.identifier.plot(ax=axlist[1], cmap=cmap_subplot2, norm=norm_subplot2, x='longitude', y='latitude', transform=ccrs.PlateCarree(), add_colorbar=False)
        axlist[0].title.set_text(probability_plot_title)
        axlist[1].title.set_text("")

        cbar_cold_ax = fig.add_axes([0.5065, 0.1913, 0.015, 0.688])
        cbar_warm_ax = fig.add_axes([0.521, 0.1913, 0.015, 0.688])
        cbar_stationary_ax = fig.add_axes([0.5355, 0.1913, 0.015, 0.688])
        cbar_occluded_ax = fig.add_axes([0.55, 0.1913, 0.015, 0.688])

        cbar_cold = plt.colorbar(cm.ScalarMappable(norm=prob_norm, cmap=cmap_cold), cax=cbar_cold_ax, pad=0, boundaries=cbar_ticks, alpha=0.6)
        cbar_cold.set_ticks([])
        cbar_warm = plt.colorbar(cm.ScalarMappable(norm=prob_norm, cmap=cmap_warm), cax=cbar_warm_ax, pad=0, boundaries=cbar_ticks, alpha=0.6)
        cbar_warm.set_ticks([])
        cbar_stationary = plt.colorbar(cm.ScalarMappable(norm=prob_norm, cmap=cmap_stationary), cax=cbar_stationary_ax, pad=0, boundaries=cbar_ticks, alpha=0.6)
        cbar_stationary.set_ticks([])
        cbar_occluded = plt.colorbar(cm.ScalarMappable(norm=prob_norm, cmap=cmap_occluded), cax=cbar_occluded_ax, pad=-0.2, boundaries=cbar_ticks, alpha=0.6)
        cbar_occluded.set_ticks(cbar_ticks + 0.05)
        cbar_occluded.set_ticklabels(cbar_tick_labels[int(probability_mask*10):])

        cbar = plt.colorbar(cm.ScalarMappable(norm=norm_identifier, cmap=cmap_identifier), ax=axlist[1], orientation='horizontal', fraction=0.046)
        cbar.set_ticks(np.arange(1,5,1)+0.5)
        cbar.set_ticklabels(['Cold','Warm','Stationary','Occluded'])
        cbar.set_label('Front type')

    elif front_types == 'ALL_bin':

        # Create custom colorbar for the different front types of the 'truth' plot
        cmap_identifier = mpl.colors.ListedColormap(["blue","red",'green','purple'], name='from_list', N=None)
        norm_identifier = mpl.colors.Normalize(vmin=1,vmax=5)

        fig, ax = plt.subplots(1, 1, figsize=(20, 5), subplot_kw={'projection': crs})
        fplot.plot_background(ax, extent)
        probs_ds.front_probs.isel().plot.contourf(ax=ax, x='longitude', y='latitude', norm=prob_norm, levels=levels, cmap=cmap_front,
            transform=ccrs.PlateCarree(), alpha=0.9, add_colorbar=False)
        fronts.identifier.plot(ax=ax, cmap=cmap_identifier, norm=norm_identifier, x='longitude', y='latitude', transform=ccrs.PlateCarree(), add_colorbar=False)
        ax.title.set_text(probability_plot_title)

        cbar_front = plt.colorbar(cm.ScalarMappable(norm=prob_norm, cmap=cmap_front), ax=ax, pad=-0.41, boundaries=cbar_ticks, alpha=0.9)
        cbar_front.set_ticks(cbar_ticks + 0.05)
        cbar_front.set_ticklabels(cbar_tick_labels[int(probability_mask*10):])

        cbar = plt.colorbar(cm.ScalarMappable(norm=norm_identifier, cmap=cmap_identifier), ax=ax, orientation='horizontal', fraction=0.046)
        cbar.set_ticks(np.arange(1,5,1)+0.5)
        cbar.set_ticklabels(['Cold','Warm','Stationary','Occluded'])
        cbar.set_label('Front type')

    plt.savefig('%s/model_%d/predictions/model_%d_%s_plot_%dx%dimages_%dx%dtrim.png' % (model_dir, model_number, model_number,
        time, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1]), bbox_inches='tight', dpi=1000)
    plt.close()


def learning_curve(model_number, model_dir, loss, fss_mask_size, fss_c, metric, include_validation_plots=True):
    """
    Function that plots learning curves for the specified model.

    Parameters
    ----------
    fss_mask_size: Size of the mask for the FSS loss function.
    fss_c: C hyperparameter for the FSS loss' sigmoid function.
    include_validation_plots: Setting this to True will plot validation data in addition to training data.
    loss: Loss function for the U-Net.
    metric: Metric used for evaluating the U-Net during training.
    model_number: Slurm job number for the model. This is the number in the model's filename.
    model_dir: Main directory for the models.
    """

    """
    loss_title: Title of the loss plots on the learning curves
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
    else:
        loss_title = None

    """
    metric_title: Title of the metric plots on the learning curves
    metric_string: Metric as it appears in the history files
    """
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
    else:
        metric_title = None
        metric_string = None

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

    if 'final_Softmax_loss' in history:
        plt.plot(history['sup4_Softmax_loss'], label='Encoder 6')
        plt.plot(history['sup3_Softmax_loss'], label='Decoder 5')
        plt.plot(history['sup2_Softmax_loss'], label='Decoder 4')
        plt.plot(history['sup1_Softmax_loss'], label='Decoder 3')
        plt.plot(history['sup0_Softmax_loss'], label='Decoder 2')
        plt.plot(history['final_Softmax_loss'], label='Decoder 1 (final)', color='black')
        plt.plot(history['loss'], label='total', color='black')
        if 1e-4 > np.min(history['final_Softmax_loss']) > 1e-5:
            loss_lower_limit, loss_upper_limit = 1e-5, 1e-3
        elif 1e-5 > np.min(history['final_Softmax_loss']) > 1e-6:
            loss_lower_limit, loss_upper_limit = 1e-6, 1e-4
        elif 1e-6 > np.min(history['final_Softmax_loss']) > 1e-7:
            loss_lower_limit, loss_upper_limit = 1e-7, 1e-5
        else:
            loss_lower_limit = np.min(history['unet_output_sup0_activation_loss'])
            loss_upper_limit = np.max(history['unet_output_sup0_activation_loss'])
    elif 'unet_output_final_activation_loss' in history:
        plt.plot(history['unet_output_sup0_activation_loss'], label='sup0')
        plt.plot(history['unet_output_sup1_activation_loss'], label='sup1')
        plt.plot(history['unet_output_sup2_activation_loss'], label='sup2')
        plt.plot(history['unet_output_sup3_activation_loss'], label='sup3')
        plt.plot(history['unet_output_sup4_activation_loss'], label='sup4')
        plt.plot(history['unet_output_final_activation_loss'], label='final')
        plt.plot(history['loss'], label='total', color='black')
        if 1e-4 > np.min(history['unet_output_sup0_activation_loss']) > 1e-5:
            loss_lower_limit, loss_upper_limit = 1e-5, 1e-3
        elif 1e-5 > np.min(history['unet_output_sup0_activation_loss']) > 1e-6:
            loss_lower_limit, loss_upper_limit = 1e-6, 1e-4
        elif 1e-6 > np.min(history['unet_output_sup0_activation_loss']) > 1e-7:
            loss_lower_limit, loss_upper_limit = 1e-7, 1e-5
        else:
            loss_lower_limit = np.min(history['unet_output_sup0_activation_loss'])
            loss_upper_limit = np.max(history['unet_output_sup0_activation_loss'])
    else:
        plt.plot(history['loss'], color='black')
        loss_lower_limit = np.min(history['loss'])
        loss_upper_limit = np.max(history['loss'])

    plt.legend(loc='best')
    plt.xlim(xmin=0)
    plt.xlabel('Epochs')

    plt.ylim(ymin=loss_lower_limit, ymax=loss_upper_limit)  # Limits of the loss function graph
    plt.yscale('log')  # Turns y-axis into a logarithmic scale. Useful if loss functions appear as very sharp curves.

    plt.subplot(nrows, 2, 2)
    plt.title("Training metric: %s" % metric_title)
    plt.grid()

    if 'final_Softmax_loss' in history:
        plt.plot(history['sup4_Softmax_%s' % metric_string], label='Encoder 6')
        plt.plot(history['sup3_Softmax_%s' % metric_string], label='Decoder 5')
        plt.plot(history['sup2_Softmax_%s' % metric_string], label='Decoder 4')
        plt.plot(history['sup1_Softmax_%s' % metric_string], label='Decoder 3')
        plt.plot(history['sup0_Softmax_%s' % metric_string], label='Decoder 2')
        plt.plot(history['final_Softmax_%s' % metric_string], label='Decoder 1 (final)', color='black')
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
    plt.yscale('log')  # Turns y-axis into a logarithmic scale. Useful if loss functions or metrics appear as very sharp curves.

    if include_validation_plots is True:
        plt.subplot(nrows, 2, 3)
        plt.title("Validation loss: %s" % loss_title)
        plt.grid()

        if 'final_Softmax_loss' in history:
            plt.plot(history['val_sup4_Softmax_loss'], label='Encoder 6')
            plt.plot(history['val_sup3_Softmax_loss'], label='Decoder 5')
            plt.plot(history['val_sup2_Softmax_loss'], label='Decoder 4')
            plt.plot(history['val_sup1_Softmax_loss'], label='Decoder 3')
            plt.plot(history['val_sup0_Softmax_loss'], label='Decoder 2')
            plt.plot(history['val_final_Softmax_loss'], label='Decoder 1 (final)', color='black')
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
        plt.ylim(ymin=loss_lower_limit, ymax=loss_upper_limit)  # Limits of the loss function graph
        plt.yscale('log')  # Turns y-axis into a logarithmic scale. Useful if loss functions or metrics appear as very sharp curves.

        plt.subplot(nrows, 2, 4)
        plt.title("Validation metric: %s" % metric_title)
        plt.grid()

        if 'final_Softmax_loss' in history:
            plt.plot(history['val_sup4_Softmax_%s' % metric_string], label='Encoder 6')
            plt.plot(history['val_sup3_Softmax_%s' % metric_string], label='Decoder 5')
            plt.plot(history['val_sup2_Softmax_%s' % metric_string], label='Decoder 4')
            plt.plot(history['val_sup1_Softmax_%s' % metric_string], label='Decoder 3')
            plt.plot(history['val_sup0_Softmax_%s' % metric_string], label='Decoder 2')
            plt.plot(history['val_final_Softmax_%s' % metric_string], label='Decoder 1 (final)', color='black')
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
        plt.yscale('log')  # Turns y-axis into a logarithmic scale. Useful if loss functions or metrics appear as very sharp curves.

    plt.savefig("%s/model_%d/model_%d_learning_curve.png" % (model_dir, model_number, model_number), bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    """ Main arguments """
    parser.add_argument('--calculate_performance_stats', type=bool, required=False,
                        help='Are you calculating performance stats for a model?')
    parser.add_argument('--day', type=int, required=False, help='Day for the prediction.')
    parser.add_argument('--domain', type=str, required=False, help='Domain of the data.')
    parser.add_argument('--domain_images', type=int, nargs=2, required=False,
                        help='Number of images for each dimension the final stitched map for predictions (lon lat).')
    parser.add_argument('--domain_lengths', type=int, nargs=2, required=False,
                        help='Lengths of the dimensions of the final stitched map for predictions (lon lat).')
    parser.add_argument('--domain_trim', type=int, nargs=2, required=False,
                        help='Number of pixels to trim the images by along each dimension for stitching before taking the '
                             'maximum across overlapping pixels.')
    parser.add_argument('--find_matches', type=bool, required=False, help='Find matches for stitching predictions?')
    parser.add_argument('--front_types', type=str, required=False,
                        help='Front format of the file. If your files contain warm and cold fronts, pass this argument'
                             'as CFWF. If your files contain only drylines, pass this argument as DL. If your files '
                             'contain all fronts, pass this argument as ALL.')
    parser.add_argument('--fss_c', type=float, required=False, help="C hyperparameter for the FSS loss' sigmoid function.")
    parser.add_argument('--fss_mask_size', type=int, required=False, help='Mask size for the FSS loss function.')
    parser.add_argument('--generate_predictions', type=bool, required=False, help='Generate prediction plots?')
    parser.add_argument('--hour', type=int, required=False, help='Hour for the prediction.')
    parser.add_argument('--learning_curve', type=bool, required=False, help='Plot learning curve?')
    parser.add_argument('--loss', type=str, required=False, help='Loss function used for training the U-Net.')
    parser.add_argument('--metric', type=str, required=False, help='Metric used for evaluating the U-Net during training.')
    parser.add_argument('--model_lengths', type=int, nargs=2, required=False,
                        help="Number of pixels along each dimension of the model's output (lon lat).")
    parser.add_argument('--model_dir', type=str, required=False, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, required=False, help='Model number.')
    parser.add_argument('--month', type=int, required=False, help='Month for the prediction.')
    parser.add_argument('--num_dimensions', type=int, required=False,
                        help='Number of dimensions of the U-Net convolutions, maxpooling, and upsampling. (2 or 3)')
    parser.add_argument('--num_variables', type=int, required=False, help='Number of variables in the variable datasets.')
    parser.add_argument('--normalization_method', type=int, required=False,
                        help='Normalization method for the data. 0 - No normalization, 1 - Min-max normalization, '
                             '2 - Mean normalization')
    parser.add_argument('--pixel_expansion', type=int, required=False, help='Number of pixels to expand the fronts by.')
    parser.add_argument('--plot_performance_diagrams', type=bool, required=False, help='Plot performance diagrams for a model?')
    parser.add_argument('--predictions', type=int, required=False, help='Number of predictions to make.')
    parser.add_argument('--random_variable', type=str, required=False,
                        help="Variable to randomize in 'calculate_performance_stats' or 'generate_predictions'.")
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
        required_arguments = ['domain', 'domain_images', 'domain_lengths', 'domain_trim', 'front_types', 'loss', 'metric',
                              'model_dir', 'model_number', 'normalization_method', 'num_dimensions', 'num_variables',
                              'pixel_expansion', 'model_lengths']
        print("Checking arguments for 'calculate_performance_stats'....", end='')
        check_arguments(provided_arguments, required_arguments)
        print("Checking compatibility of image stitching arguments....", end='')
        find_matches_for_domain(args.domain_lengths, args.model_lengths, compatibility_mode=True, compat_images=args.domain_images)
        print("done")
        calculate_performance_stats(args.model_number, args.model_dir, args.num_variables, args.num_dimensions, args.front_types,
            args.domain, args.test_years, args.normalization_method, args.loss, args.fss_mask_size, args.fss_c,
            args.pixel_expansion, args.metric, args.domain_images, args.domain_lengths, args.domain_trim,
            random_variable=args.random_variable)

    if args.find_matches is True:
        required_arguments = ['domain_lengths', 'model_lengths']
        print("Checking arguments for 'find_matches'....", end='')
        check_arguments(provided_arguments, required_arguments)
        find_matches_for_domain(args.domain_lengths, args.model_lengths)

    if args.learning_curve is True:
        required_arguments = ['loss', 'metric', 'model_dir', 'model_number']
        print("Checking arguments for 'learning_curve'....", end='')
        check_arguments(provided_arguments, required_arguments)
        learning_curve(args.model_number, args.model_dir, args.loss, args.fss_mask_size, args.fss_c, args.metric)

    if args.generate_predictions is True:
        required_arguments = ['model_number', 'model_dir', 'num_variables', 'num_dimensions', 'front_types', 'domain',
            'normalization_method', 'loss', 'pixel_expansion', 'metric', 'domain_images', 'domain_lengths',
            'domain_trim', 'model_lengths']
        print("Checking arguments for 'generate_predictions'....", end='')
        check_arguments(provided_arguments, required_arguments)

        print("Checking compatibility of image stitching arguments....", end='')
        find_matches_for_domain(args.domain_lengths, args.model_lengths, compatibility_mode=True, compat_images=args.domain_images)
        print("done")

        print("Checking compatibility of prediction arguments....",end='')
        if args.predictions is not None:
            if args.year is not None or args.month is not None or args.day is not None or args.hour is not None:
                print("error")
                raise ArgumentConflictError("If 'predictions' is not None (i.e. you want to make random predictions), "
                                            "then 'year', 'month', 'day', and 'hour' cannot be passed.")
            else:
                print("done")
        else:
            if args.year is not None or args.month is not None or args.day is not None or args.hour is not None:
                if args.test_years is not None:
                    print("error")
                    raise ArgumentConflictError("If 'predictions' is None (i.e. you want to make a prediction for a specific date and time), "
                                                "then 'test_years' cannot be passed.")
                else:
                    required_arguments = ['year', 'month', 'day', 'hour']
                    check_arguments(provided_arguments, required_arguments)

        if args.test_years is not None:
            if args.front_types == 'ALL_bin':
                front_files, variable_files = fm.load_test_files(args.num_variables, 'ALL', args.domain, args.test_years)
            else:
                front_files, variable_files = fm.load_test_files(args.num_variables, args.front_types, args.domain, args.test_years)
        else:
            if args.front_types == 'ALL_bin':
                front_files, variable_files = fm.load_file_lists(args.num_variables, 'ALL', args.domain)
            else:
                front_files, variable_files = fm.load_file_lists(args.num_variables, args.front_types, args.domain)
        generate_predictions(args.model_number, args.model_dir, front_files, variable_files, args.predictions, args.normalization_method,
            args.loss, args.fss_mask_size, args.fss_c, args.front_types, args.pixel_expansion, args.metric, args.num_dimensions,
            args.domain_images, args.domain_lengths, args.domain_trim, args.year, args.month, args.day, args.hour, random_variable=args.random_variable)

    if args.plot_performance_diagrams is True:
        required_arguments = ['model_number', 'model_dir', 'front_types', 'domain_images', 'domain_trim']
        print("Checking arguments for 'plot_performance_diagrams'....", end='')
        check_arguments(provided_arguments, required_arguments)
        plot_performance_diagrams(args.model_dir, args.model_number, args.front_types, args.domain_images, args.domain_trim,
                                  random_variable=args.random_variable)
