"""
Functions used for evaluating a U-Net model.

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 2/9/2023 10:49 PM CT

TODO:
    * Clean up code (much needed)
    * Remove the need for separate pickle files to be generated for spatial CSI maps
    * Add more documentation
"""

import itertools
import os
import pickle
import random
import argparse
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import errors
import file_manager as fm
import xarray as xr
from errors import check_arguments
import tensorflow as tf
from matplotlib import cm, colors  # Here we explicitly import the cm and color modules to suppress a PyCharm bug
from matplotlib.ticker import FixedLocator
from utils import data_utils, settings
from utils.plotting_utils import plot_background
from glob import glob
from matplotlib.font_manager import FontProperties
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def add_image_to_map(stitched_map_probs, image_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
    image_size_lon, image_size_lat, lon_image_spacing, lat_image_spacing):
    """
    Add model prediction to the stitched map.

    Parameters
    ----------
    stitched_map_probs: Numpy array
        - Array of front probabilities for the final map.
    image_probs: Numpy array
        - Array of front probabilities for the current prediction/image.
    map_created: bool
        - Boolean flag that declares whether or not the final map has been completed.
    domain_images_lon: int
        - Number of images along the longitude dimension of the domain.
    domain_images_lat: int
        - Number of images along the latitude dimension of the domain.
    lon_image: int
        - Current image number along the longitude dimension.
    lat_image: int
        - Current image number along the latitude dimension.
    image_size_lon: int
        - Number of pixels along the longitude dimension of the model predictions.
    image_size_lat: int
        - Number of pixels along the latitude dimension of the model predictions.
    lon_image_spacing: int
        - Number of pixels between each image along the longitude dimension.
    lat_image_spacing: int
        - Number of pixels between each image along the latitude dimension.

    Returns
    -------
    map_created: bool
        - Boolean flag that declares whether or not the final map has been completed.
    stitched_map_probs: array
        - Array of front probabilities for the final map.
    """

    if lon_image == 0:  # If the image is on the western edge of the domain
        if lat_image == 0:  # If the image is on the northern edge of the domain
            # Add first image to map
            stitched_map_probs[:, 0: image_size_lon, 0: image_size_lat] = \
                image_probs[:, :image_size_lon, :image_size_lat]

            if domain_images_lon == 1 and domain_images_lat == 1:
                map_created = True

        elif lat_image != domain_images_lat - 1:  # If the image is not on the northern nor the southern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant latitude
            stitched_map_probs[:, 0: image_size_lon, int(lat_image * lat_image_spacing):int((lat_image-1)*lat_image_spacing) + image_size_lat] = \
                np.maximum(stitched_map_probs[:, 0: image_size_lon, int(lat_image * lat_image_spacing):int((lat_image-1)*lat_image_spacing) + image_size_lat],
                           image_probs[:, :image_size_lon, :image_size_lat - lat_image_spacing])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[:, 0: image_size_lon, int(lat_image_spacing * (lat_image-1)) + image_size_lat:int(lat_image_spacing * lat_image) + image_size_lat] = \
                image_probs[:, :image_size_lon, image_size_lat - lat_image_spacing:image_size_lat]

            if domain_images_lon == 1 and domain_images_lat == 2:
                map_created = True

        else:  # If the image is on the southern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant latitude
            stitched_map_probs[:, 0: image_size_lon, int(lat_image * lat_image_spacing):int((lat_image-1)*lat_image_spacing) + image_size_lat] = \
                np.maximum(stitched_map_probs[:, :image_size_lon, int(lat_image * lat_image_spacing):int((lat_image-1)*lat_image_spacing) + image_size_lat],
                           image_probs[:, :image_size_lon, :image_size_lat - lat_image_spacing])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[:, 0: image_size_lon, int(lat_image_spacing * (lat_image-1)) + image_size_lat:] = \
                image_probs[:, :image_size_lon,  image_size_lat - lat_image_spacing:image_size_lat]

            if domain_images_lon == 1 and domain_images_lat > 2:
                map_created = True

    elif lon_image != domain_images_lon - 1:  # If the image is not on the western nor the eastern edge of the domain
        if lat_image == 0:  # If the image is on the northern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant longitude
            stitched_map_probs[:, int(lon_image * lon_image_spacing):int((lon_image-1)*lon_image_spacing) + image_size_lon, 0: image_size_lat] = \
                np.maximum(stitched_map_probs[:, int(lon_image * lon_image_spacing):int((lon_image-1)*lon_image_spacing) + image_size_lon, 0: image_size_lat],
                           image_probs[:, :image_size_lon - lon_image_spacing, :image_size_lat])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[:, int(lon_image_spacing * (lon_image-1)) + image_size_lon:lon_image_spacing * lon_image + image_size_lon, 0: image_size_lat] = \
                image_probs[:, image_size_lon - lon_image_spacing:image_size_lon, :image_size_lat]

            if domain_images_lon == 2 and domain_images_lat == 1:
                map_created = True

        elif lat_image != domain_images_lat - 1:  # If the image is not on the northern nor the southern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant longitude
            stitched_map_probs[:, int(lon_image * lon_image_spacing):int(lon_image_spacing * (lon_image-1)) + image_size_lon, int(lat_image * lat_image_spacing):int(lat_image * lat_image_spacing) + image_size_lat] = \
                np.maximum(stitched_map_probs[:, int(lon_image * lon_image_spacing):int(lon_image_spacing * (lon_image-1)) + image_size_lon, int(lat_image * lat_image_spacing):int(lat_image * lat_image_spacing) + image_size_lat],
                           image_probs[:, :image_size_lon - lon_image_spacing, :image_size_lat])

            # Take the maximum of the overlapping pixels along sets of constant latitude
            stitched_map_probs[:, int(lon_image * lon_image_spacing): int(lon_image * lon_image_spacing) + image_size_lon, int(lat_image * lat_image_spacing):int(lat_image_spacing * (lat_image-1)) + image_size_lat] = \
                np.maximum(stitched_map_probs[:, int(lon_image * lon_image_spacing): int(lon_image * lon_image_spacing) + image_size_lon, int(lat_image * lat_image_spacing):int(lat_image_spacing * (lat_image-1)) + image_size_lat],
                           image_probs[:, :image_size_lon, :image_size_lat - lat_image_spacing])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[:, int(lon_image_spacing * (lon_image-1)) + image_size_lon:lon_image_spacing * lon_image + image_size_lon, int(lat_image_spacing * (lat_image-1)) + image_size_lat:int(lat_image_spacing * lat_image) + image_size_lat] = \
                image_probs[:, image_size_lon - lon_image_spacing:image_size_lon, image_size_lat - lat_image_spacing:image_size_lat]

            if domain_images_lon == 2 and domain_images_lat == 2:
                map_created = True

        else:  # If the image is on the southern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant longitude
            stitched_map_probs[:, int(lon_image * lon_image_spacing):int(lon_image_spacing * (lon_image-1)) + image_size_lon, int(lat_image * lat_image_spacing):] = \
                np.maximum(stitched_map_probs[:, int(lon_image * lon_image_spacing):int(lon_image_spacing * (lon_image-1)) + image_size_lon, int(lat_image * lat_image_spacing):],
                           image_probs[:, :image_size_lon - lon_image_spacing, :image_size_lat])

            # Take the maximum of the overlapping pixels along sets of constant latitude
            stitched_map_probs[:, int(lon_image * lon_image_spacing):int(lon_image * lon_image_spacing) + image_size_lon, int(lat_image * lat_image_spacing):int(lat_image_spacing * (lat_image-1)) + image_size_lat] = \
                np.maximum(stitched_map_probs[:, int(lon_image * lon_image_spacing):int(lon_image * lon_image_spacing) + image_size_lon, int(lat_image * lat_image_spacing):int(lat_image_spacing * (lat_image-1)) + image_size_lat],
                           image_probs[:, :image_size_lon, :image_size_lat - lat_image_spacing])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[:, int(lon_image_spacing * (lon_image-1)) + image_size_lon:lon_image_spacing * lon_image + image_size_lon, int(lat_image_spacing * (lat_image-1)) + image_size_lat:] = \
                image_probs[:, image_size_lon - lon_image_spacing:image_size_lon, image_size_lat - lat_image_spacing:image_size_lat]

            if domain_images_lon == 2 and domain_images_lat > 2:
                map_created = True
    else:
        if lat_image == 0:  # If the image is on the northern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant longitude
            stitched_map_probs[:, int(lon_image * lon_image_spacing):int(lon_image_spacing * (lon_image-1)) + image_size_lon, 0: image_size_lat] = \
                np.maximum(stitched_map_probs[:, int(lon_image * lon_image_spacing):int(lon_image_spacing * (lon_image-1)) + image_size_lon, 0: image_size_lat],
                           image_probs[:, :image_size_lon - lon_image_spacing, :image_size_lat])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[:, int(lon_image_spacing * (lon_image-1)) + image_size_lon:, 0: image_size_lat] = \
                image_probs[:, image_size_lon - lon_image_spacing:image_size_lon, :image_size_lat]

            if domain_images_lon > 2 and domain_images_lat == 1:
                map_created = True

        elif lat_image != domain_images_lat - 1:  # If the image is not on the northern nor the southern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant longitude
            stitched_map_probs[:, int(lon_image * lon_image_spacing):int(lon_image_spacing * (lon_image-1)) + image_size_lon, int(lat_image * lat_image_spacing):int(lat_image * lat_image_spacing) + image_size_lat] = \
                np.maximum(stitched_map_probs[:, int(lon_image * lon_image_spacing):int(lon_image_spacing * (lon_image-1)) + image_size_lon, int(lat_image * lat_image_spacing):int(lat_image * lat_image_spacing) + image_size_lat], image_probs[:, :image_size_lon - lon_image_spacing, :image_size_lat])

            # Take the maximum of the overlapping pixels along sets of constant latitude
            stitched_map_probs[:, int(lon_image * lon_image_spacing):, int(lat_image * lat_image_spacing):int(lat_image_spacing * (lat_image-1)) + image_size_lat] = \
                np.maximum(stitched_map_probs[:, int(lon_image * lon_image_spacing):, int(lat_image * lat_image_spacing):int(lat_image_spacing * (lat_image-1)) + image_size_lat], image_probs[:, :image_size_lon, :image_size_lat - lat_image_spacing])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[:, int(lon_image_spacing * (lon_image-1)) + image_size_lon:, int(lat_image_spacing * (lat_image-1)) + image_size_lat:int(lat_image_spacing * lat_image) + image_size_lat] = image_probs[:, image_size_lon - lon_image_spacing:image_size_lon, image_size_lat - lat_image_spacing:image_size_lat]

            if domain_images_lon > 2 and domain_images_lat == 2:
                map_created = True
        else:  # If the image is on the southern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant longitude
            stitched_map_probs[:, int(lon_image * lon_image_spacing):, int(lat_image * lat_image_spacing):] = \
                np.maximum(stitched_map_probs[:, int(lon_image * lon_image_spacing):, int(lat_image * lat_image_spacing):],
                           image_probs[:, :image_size_lon, :image_size_lat])

            # Take the maximum of the overlapping pixels along sets of constant latitude
            stitched_map_probs[:, int(lon_image * lon_image_spacing):, int(lat_image * lat_image_spacing):int(lat_image_spacing * (lat_image-1)) + image_size_lat] = \
                np.maximum(stitched_map_probs[:, int(lon_image * lon_image_spacing):, int(lat_image * lat_image_spacing):int(lat_image_spacing * (lat_image-1)) + image_size_lat],
                           image_probs[:, :image_size_lon, :image_size_lat - lat_image_spacing])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[:, int(lon_image_spacing * (lon_image-1)) + image_size_lon:, int(lat_image_spacing * (lat_image-1)) + image_size_lat:int(lat_image_spacing * lat_image) + image_size_lat] = \
                image_probs[:, image_size_lon - lon_image_spacing:image_size_lon, image_size_lat - lat_image_spacing:image_size_lat]

            map_created = True

    return stitched_map_probs, map_created


def calculate_performance_stats(model_number, model_dir, fronts_netcdf_dir, timestep, domain, domain_images, forecast_hour=None, variable_data_source='era5'):
    """
    Calculate performance statistics (true positives, false positives, false negatives, true negatives) for a model over a specified domain.

    Parameters
    ----------
    model_number: int
        - Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        - Main directory for the models.
    fronts_netcdf_dir: str
        - Directory where the frontal object netCDF files are stored.
    timestep: iterable object with 4 ints
        - 4 integers in the following order: year, month, day, hour
    domain: str
        - Domain of the data.
    domain_images: iterable object with 2 ints
        - Number of images along each dimension of the final stitched map (lon lat).
    variable_data_source: str
        - Variable data to use for training the model. Options are: 'era5', 'gdas', or 'gfs' (case-insensitive)
    forecast_hour: int
        - Forecast hour for the data. This will only be used if the data source is not ERA5.
    """

    year, month, day, hour = timestep[0], timestep[1], timestep[2], timestep[3]

    # Grab the model predictions/probabilities for the current timestep
    probs_dir = f'{model_dir}/model_{model_number}/probabilities/{domain}_{domain_images[0]}x{domain_images[1]}'
    probs_file = f'{probs_dir}/model_{model_number}_{year}-%02d-%02d-%02dz_{domain}_{domain_images[0]}x{domain_images[1]}_probabilities.nc' % (month, day, hour)

    if forecast_hour is not None:
        forecast_timestep = data_utils.add_or_subtract_hours_to_timestep('%d%02d%02d%02d' % (year, month, day, hour), num_hours=forecast_hour)
        new_year, new_month, new_day, new_hour = forecast_timestep[:4], forecast_timestep[4:6], forecast_timestep[6:8], forecast_timestep[8:]
        fronts_file = '%s/%s/%s/%s/FrontObjects_%s%s%s%s_full.nc' % (fronts_netcdf_dir, new_year, new_month, new_day, new_year, new_month, new_day, new_hour)
        probs_file = probs_file.replace(domain, f'{domain}_{variable_data_source}_f%03d' % forecast_hour)
    else:
        fronts_file = '%s/%d/%02d/%02d/FrontObjects_%d%02d%02d%02d_full.nc' % (fronts_netcdf_dir, year, month, day, year, month, day, hour)

    model_properties = pd.read_pickle(f"{model_dir}/model_{model_number}/model_{model_number}_properties.pkl")
    front_types = model_properties['front_types']

    fronts_ds = xr.open_dataset(fronts_file).isel(longitude=slice(settings.DEFAULT_DOMAIN_INDICES[domain][0], settings.DEFAULT_DOMAIN_INDICES[domain][1]),
                                                  latitude=slice(settings.DEFAULT_DOMAIN_INDICES[domain][2], settings.DEFAULT_DOMAIN_INDICES[domain][3]))
    fronts_ds = data_utils.reformat_fronts(fronts_ds, front_types)

    num_front_types = fronts_ds.attrs['num_types']

    probs_ds = xr.open_dataset(probs_file)

    bool_tn_fn_dss = dict({front: xr.where(fronts_ds == front_no + 1, 1, 0)['identifier'] for front_no, front in enumerate(front_types)})
    bool_tp_fp_dss = dict({front: None for front in front_types})
    probs_dss = dict({front: probs_ds[front] for front in front_types})

    # Cumulative statistics arrays (no spatial dimension)
    tp_array = np.zeros(shape=[num_front_types, 5, 100]).astype('int32')
    fp_array = np.zeros(shape=[num_front_types, 5, 100]).astype('int32')
    tn_array = np.zeros(shape=[num_front_types, 5, 100]).astype('int32')
    fn_array = np.zeros(shape=[num_front_types, 5, 100]).astype('int32')

    thresholds = np.linspace(0.01, 1, 100)  # Probability thresholds for calculating performance statistics
    boundaries = np.array([50, 100, 150, 200, 250])  # Boundaries for checking whether or not a front is present (kilometers)

    performance_ds = xr.Dataset(coords={'boundary': boundaries, 'threshold': thresholds})

    for front_no, front_type in enumerate(front_types):

        ### Calculate true/false negatives ###
        for i in range(100):
            """
            True negative ==> model correctly predicts the lack of a front at a given point
            False negative ==> model does not predict a front, but a front exists
            
            The numbers of true negatives and false negatives are the same for all neighborhoods and are calculated WITHOUT expanding the fronts.
            If we were to calculate the negatives separately for each neighborhood, the number of misses would be artificially inflated, lowering the 
            final CSI scores and making the neighborhood method effectively useless.
            """
            tn_array[front_no, :, i] = len(np.where((probs_dss[front_type] < thresholds[i]) & (bool_tn_fn_dss[front_type] == 0))[0])
            fn_array[front_no, :, i] = len(np.where((probs_dss[front_type] < thresholds[i]) & (bool_tn_fn_dss[front_type] == 1))[0])

        ### Calculate true/false positives ###
        for boundary in range(5):
            new_fronts_ds = xr.open_dataset(fronts_file).isel(longitude=slice(settings.DEFAULT_DOMAIN_INDICES[domain][0], settings.DEFAULT_DOMAIN_INDICES[domain][1]),
                                                              latitude=slice(settings.DEFAULT_DOMAIN_INDICES[domain][2], settings.DEFAULT_DOMAIN_INDICES[domain][3]))
            new_fronts = data_utils.reformat_fronts(new_fronts_ds, front_types)
            front_identifier = data_utils.expand_fronts(new_fronts, iterations=int(2*(boundary+1)))  # Expand fronts
            bool_tp_fp_dss[front_type] = xr.where(front_identifier == front_no + 1, 1, 0)['identifier']  # 1 = cold front, 0 = not a cold front
            for i in range(100):
                """
                True positive ==> model correctly identifies a front
                False positive ==> model predicts a front, but no front is present within the given neighborhood
                """
                tp_array[front_no, boundary, i] = len(np.where((probs_dss[front_type] > thresholds[i]) & (bool_tp_fp_dss[front_type] == 1))[0])
                fp_array[front_no, boundary, i] = len(np.where((probs_dss[front_type] > thresholds[i]) & (bool_tp_fp_dss[front_type] == 0))[0])

        performance_ds["tp_%s" % front_type] = (('boundary', 'threshold'), tp_array[front_no])
        performance_ds["fp_%s" % front_type] = (('boundary', 'threshold'), fp_array[front_no])
        performance_ds["tn_%s" % front_type] = (('boundary', 'threshold'), tn_array[front_no])
        performance_ds["fn_%s" % front_type] = (('boundary', 'threshold'), fn_array[front_no])

    performance_ds = performance_ds.expand_dims({'time': np.atleast_1d(probs_ds['time'].values)})
    if forecast_hour is not None:
        performance_ds = performance_ds.expand_dims({'forecast_hour': np.atleast_1d(probs_ds['forecast_hour'].values)})

    performance_ds.to_netcdf(path=probs_file.replace('probabilities', 'statistics'), mode='w', engine='netcdf4')


def find_matches_for_domain(domain_size, image_size, compatibility_mode=False, compat_images=None):
    """
    Function that outputs the number of images that can be stitched together with the specified domain length and the length
    of the domain dimension output by the model. This is also used to determine the compatibility of declared image and
    parameters for model predictions.

    Parameters
    ----------
    domain_size: iterable object with 2 integers
        - Number of pixels along each dimension of the final stitched map (lon lat).
    image_size: iterable object with 2 integers
        - Number of pixels along each dimension of the model's output (lon lat).
    compatibility_mode: bool
        - Boolean flag that declares whether or not the function is being used to check compatibility of given parameters.
    compat_images: iterable object with 2 integers
        - Number of images declared for the stitched map in each dimension (lon lat). (Compatibility mode only)
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

    num_matches = [0, 0]  # Total number of matching image arguments found for each dimension

    lon_image_matches = []
    lat_image_matches = []

    for lon_images in range(1, domain_size[0]-image_size[0]):  # Image counter for longitude dimension
        if lon_images > 1:
            lon_spacing = (domain_size[0]-image_size[0])/(lon_images-1)  # Spacing between images in the longitude dimension
        else:
            lon_spacing = 0
        if lon_spacing - int(lon_spacing) == 0 and lon_spacing > 1 and image_size[0]-lon_spacing > 0:  # Check compatibility of latitude image spacing
            lon_image_matches.append(lon_images)  # Add longitude image match to list
            num_matches[0] += 1
            if compatibility_mode is True:
                if compat_images_lon == lon_images:  # If the number of images for the compatibility check equals the match
                    lon_images_are_compatible = True
        elif lon_spacing == 0 and domain_size[0] - image_size[0] == 0:
            lon_image_matches.append(lon_images)  # Add longitude image match to list
            num_matches[0] += 1
            if compatibility_mode is True:
                if compat_images_lon == lon_images:  # If the number of images for the compatibility check equals the match
                    lon_images_are_compatible = True

    if num_matches[0] == 0:
        raise ValueError(f"No compatible value for domain_images[0] was found with domain_size[0]={domain_size[0]} and image_size[0]={image_size[0]}.")
    if compatibility_mode is True:
        if lon_images_are_compatible is False:
            raise ValueError(f"domain_images[0]={compat_images_lon} is not compatible with domain_size[0]={domain_size[0]} "
                             f"and image_size[0]={image_size[0]}.\n"
                             f"====> Compatible values for domain_images[0] given domain_size[0]={domain_size[0]} "
                             f"and image_size[0]={image_size[0]}: {lon_image_matches}")
    else:
        print(f"Compatible longitude images: {lon_image_matches}")

    for lat_images in range(1, domain_size[1]-image_size[1]+2):  # Image counter for latitude dimension
        if lat_images > 1:
            lat_spacing = (domain_size[1]-image_size[1])/(lat_images-1)  # Spacing between images in the latitude dimension
        else:
            lat_spacing = 0
        if lat_spacing - int(lat_spacing) == 0 and lat_spacing > 1 and image_size[1]-lat_spacing > 0:  # Check compatibility of latitude image spacing
            lat_image_matches.append(lat_images)  # Add latitude image match to list
            num_matches[1] += 1
            if compatibility_mode is True:
                if compat_images_lat == lat_images:  # If the number of images for the compatibility check equals the match
                    lat_images_are_compatible = True
        elif lat_spacing == 0 and domain_size[1] - image_size[1] == 0:
            lat_image_matches.append(lat_images)  # Add latitude image match to list
            num_matches[1] += 1
            if compatibility_mode is True:
                if compat_images_lat == lat_images:  # If the number of images for the compatibility check equals the match
                    lat_images_are_compatible = True

    if num_matches[1] == 0:
        raise ValueError(f"No compatible value for domain_images[1] was found with domain_size[1]={domain_size[1]} and image_size[1]={image_size[1]}.")
    if compatibility_mode is True:
        if lat_images_are_compatible is False:
            raise ValueError(f"domain_images[1]={compat_images_lat} is not compatible with domain_size[1]={domain_size[1]} "
                             f"and image_size[1]={image_size[1]}.\n"
                             f"====> Compatible values for domain_images[1] given domain_size[1]={domain_size[1]} "
                             f"and image_size[1]={image_size[1]}: {lat_image_matches}")
    else:
        print(f"Compatible latitude images: {lat_image_matches}")


def generate_predictions(model_number, model_dir, variables_netcdf_indir, prediction_method, domain='full',
    domain_images='min', dataset=None, datetime=None, random_variables=None, variable_data_source='gdas', forecast_hours=None):
    """
    Generate predictions with a model.

    Parameters
    ----------
    model_number: int
        - Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        - Main directory for the models.
    variables_netcdf_indir: str
        - Input directory for the ERA5 netcdf files.
    domain: str
        - Domain of the datasets.
    domain_images: str or iterable object with 2 ints
        - If a string, valid values are 'min', 'balanced', 'max'. Default is 'min'.
            * 'min' uses the minimum possible number of images when making the predictions.
            * 'balanced' chooses the number of images that creates the closest possible balance of images in the longitude and latitude direction.
            * 'max' chooses the maximum number of images that can be stitched together (NOT RECOMMENDED)
        - If an iterable with 2 integers, values represent the number of images in the longitude and latitude dimensions.
    prediction_method: str
        - Prediction method. Options are: 'datetime', 'random', 'all'
    domain: str
        - Domain over which the predictions will be made. Options are: 'conus', 'full'. Default is 'full'.
    datetime: iterable object with 4 integers
        - 4 values for the date and time: year, month, day, hour
    dataset: str
        - Dataset for which to make predictions if prediction_method is 'random' or 'all'.
    random_variables: str, or iterable of strings
        - Variables to randomize.
    variable_data_source: str
        - Variable data to use for training the model. Options are: 'era5', 'gdas', or 'gfs' (case-insensitive)
    forecast_hours: int or tuple of ints
        - Forecast hours to make predictions with if making predictions with GDAS data.
    """

    variable_data_source = variable_data_source.lower()

    ### Model properties ###
    model_properties = pd.read_pickle(f"{model_dir}/model_{model_number}/model_{model_number}_properties.pkl")
    model_type = model_properties['model_type']
    image_size = model_properties['image_size']  # The image size does not include the last dimension of the input size as it only represents the number of channels
    front_types = model_properties['front_types']
    classes = model_properties['classes']
    variables = model_properties['variables']
    pressure_levels = model_properties['pressure_levels']
    normalization_parameters = model_properties['normalization_parameters']

    num_dimensions = len(image_size)
    if model_number not in [7805504, 7866106, 7961517]:  # The model numbers in this list have 2D kernels
        num_dimensions += 1

    test_years, valid_years = model_properties['test_years'], model_properties['validation_years']

    if domain_images is None:
        domain_images = settings.DEFAULT_DOMAIN_IMAGES[domain]
    domain_extent_indices = settings.DEFAULT_DOMAIN_INDICES[domain]

    ### Properties of the final map made from stitched images ###
    domain_images_lon, domain_images_lat = domain_images[0], domain_images[1]
    domain_size_lon, domain_size_lat = domain_extent_indices[1] - domain_extent_indices[0], domain_extent_indices[3] - domain_extent_indices[2]
    image_size_lon, image_size_lat = image_size[0], image_size[1]  # Dimensions of the model's predictions

    if domain_images_lon > 1:
        lon_image_spacing = int((domain_size_lon - image_size_lon)/(domain_images_lon-1))
    else:
        lon_image_spacing = 0

    if domain_images_lat > 1:
        lat_image_spacing = int((domain_size_lat - image_size_lat)/(domain_images_lat-1))
    else:
        lat_image_spacing = 0

    model = fm.load_model(model_number, model_dir)

    ############################################### Load variable files ################################################
    variable_files_obj = getattr(fm, '%sfiles' % variable_data_source.upper())  # Initialize the object for variable files
    variable_files_obj = variable_files_obj(variables_netcdf_indir, file_type='netcdf')
    variable_files_obj.validation_years = valid_years
    variable_files_obj.test_years = test_years

    if dataset is not None:
        variable_files = getattr(variable_files_obj, f'{variable_data_source}_files_' + dataset)
    else:
        variable_files = getattr(variable_files_obj, '%s_files' % variable_data_source)
    ####################################################################################################################

    dataset_kwargs = {'engine': 'netcdf4'}  # Keyword arguments for loading variable files with xarray
    coords_isel_kwargs = {'longitude': slice(domain_extent_indices[0], domain_extent_indices[1]), 'latitude': slice(domain_extent_indices[2], domain_extent_indices[3])}

    if prediction_method == 'datetime':
        timestep_str = '%d%02d%02d%02d' % (datetime[0], datetime[1], datetime[2], datetime[3])
        if variable_data_source == 'era5':
            datetime_index = [index for index, file in enumerate(variable_files) if timestep_str in file][0]
            variable_files = [variable_files[datetime_index], ]
        else:
            variable_files = [file for file in variable_files if timestep_str in file]

    subdir_base = '%s_%dx%d' % (domain, domain_images[0], domain_images[1])
    if random_variables is not None:
        subdir_base += '_' + '-'.join(sorted(random_variables))

    num_files = len(variable_files)

    num_chunks = int(np.ceil(num_files / settings.MAX_FILE_CHUNK_SIZE))  # Number of files/timesteps to process at once
    chunk_indices = np.linspace(0, num_files, num_chunks + 1, dtype=int)

    for chunk_no in range(num_chunks):

        files_in_chunk = variable_files[chunk_indices[chunk_no]:chunk_indices[chunk_no + 1]]
        print(f"Preparing chunk {chunk_no + 1}/{num_chunks}")
        variable_ds = xr.open_mfdataset(files_in_chunk, **dataset_kwargs).isel(**coords_isel_kwargs)[variables]

        if variable_data_source == 'era5':

            variable_ds = variable_ds.transpose('time', 'longitude', 'latitude', 'pressure_level')

        else:

            variable_ds = variable_ds.sel(pressure_level=pressure_levels).transpose('time', 'forecast_hour', 'longitude', 'latitude', 'pressure_level')
            forecast_hours = variable_ds['forecast_hour'].values

        # Older 2D models were trained with the pressure levels not in the proper order
        if model_number in [7805504, 7866106, 7961517]:
            variable_ds = variable_ds.isel(pressure_level=[0, 4, 3, 2, 1])

        image_lats = variable_ds.latitude.values[:domain_size_lat]
        image_lons = variable_ds.longitude.values[:domain_size_lon]

        # Randomize variable
        if random_variables is not None:
            variable_ds = data_utils.randomize_variables(variable_ds, random_variables)

        timestep_predict_size = settings.TIMESTEP_PREDICT_SIZE[domain]

        if variable_data_source != 'era5':
            num_forecast_hours = len(forecast_hours)
            timestep_predict_size /= num_forecast_hours
            timestep_predict_size = int(timestep_predict_size)

        num_timesteps = len(variable_ds['time'].values)
        num_batches = int(np.ceil(num_timesteps / timestep_predict_size))

        for batch_no in range(num_batches):

            print(f"======== Chunk {chunk_no + 1}/{num_chunks}: batch {batch_no + 1}/{num_batches} ========")

            variable_batch_ds = variable_ds.isel(time=slice(batch_no * timestep_predict_size, (batch_no + 1) * timestep_predict_size))  # Select timesteps for the current batch
            variable_batch_ds = data_utils.normalize_variables(variable_batch_ds, normalization_parameters).astype('float16')

            timesteps = variable_batch_ds['time'].values
            num_timesteps_in_batch = len(timesteps)
            map_created = False  # Boolean that determines whether or not the final stitched map has been created

            if variable_data_source == 'era5':
                stitched_map_probs = np.empty(shape=[num_timesteps_in_batch, classes-1, domain_size_lon, domain_size_lat])
            else:
                stitched_map_probs = np.empty(shape=[num_timesteps_in_batch, len(forecast_hours), classes-1, domain_size_lon, domain_size_lat])

            for lat_image in range(domain_images_lat):
                lat_index = int(lat_image * lat_image_spacing)
                for lon_image in range(domain_images_lon):
                    print(f"image %d/%d" % (int(lat_image*domain_images_lon) + lon_image + 1, int(domain_images_lon*domain_images_lat)))
                    lon_index = int(lon_image * lon_image_spacing)

                    # Select the current image
                    variable_batch_ds_new = variable_batch_ds[variables].isel(longitude=slice(lon_index, lon_index + image_size[0]),
                                                                              latitude=slice(lat_index, lat_index + image_size[1])).to_array().values

                    if variable_data_source == 'era5':
                        variable_batch_ds_new = variable_batch_ds_new.transpose([1, 2, 3, 4, 0])  # (time, longitude, latitude, pressure level, variable)
                    else:
                        variable_batch_ds_new = variable_batch_ds_new.transpose([1, 2, 3, 4, 5, 0])  # (time, forecast hour, longitude, latitude, pressure level, variable)

                    if num_dimensions == 2:

                        ### Combine pressure levels and variables into one dimension ###
                        variable_batch_ds_new_shape = np.shape(variable_batch_ds_new)
                        variable_batch_ds_new = variable_batch_ds_new.reshape(*[dim_size for dim_size in variable_batch_ds_new_shape[:-2]], variable_batch_ds_new_shape[-2] * variable_batch_ds_new_shape[-1])

                        ### Variables in older 2D models are in a weird order, so we will reshape the input array to account for this ###
                        if model_number in [7805504, 7866106, 7961517]:
                            variable_batch_ds_new_2D = np.empty(shape=np.shape(variable_batch_ds_new))
                            variable_index_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                                    23, 12, 15, 16, 14, 13, 18, 19, 22, 17, 20, 21,
                                                    35, 24, 27, 28, 26, 25, 30, 31, 34, 29, 32, 33,
                                                    47, 36, 39, 40, 38, 37, 42, 43, 46, 41, 44, 45,
                                                    59, 48, 51, 52, 50, 49, 54, 55, 58, 53, 56, 57]

                            for variable_index_new, variable_index_old in enumerate(variable_index_order):
                                variable_batch_ds_new_2D[:, :, :, variable_index_new] = np.array(variable_batch_ds_new[:, :, :, variable_index_old])
                            variable_batch_ds_new = variable_batch_ds_new_2D

                    transpose_indices = (0, 3, 1, 2)  # New order of indices for model predictions (time, front type, longitude, latitude)

                    ##################################### Generate the predictions #####################################
                    if variable_data_source == 'era5':

                        prediction = model.predict(variable_batch_ds_new, batch_size=settings.GPU_PREDICT_BATCH_SIZE, verbose=0)

                        if num_dimensions == 2:
                            if model_type == 'unet_3plus':
                                image_probs = np.transpose(prediction[0][:, :, :, 1:], transpose_indices)  # transpose the predictions
                        else:  # if num_dimensions == 3
                            if model_type == 'unet_3plus':
                                image_probs = np.transpose(np.amax(prediction[0][:, :, :, :, 1:], axis=3), transpose_indices)  # Take the maximum probability over the vertical dimension and transpose the predictions
                            elif model_type == 'unet':
                                image_probs = np.transpose(np.amax(prediction[:, :, :, :, 1:], axis=3), transpose_indices)  # Take the maximum probability over the vertical dimension and transpose the predictions
                    else:

                        ### Combine time and forecast hour into one dimension ###
                        gdas_variable_ds_new_shape = np.shape(variable_batch_ds_new)
                        variable_batch_ds_new = variable_batch_ds_new.reshape(gdas_variable_ds_new_shape[0] * gdas_variable_ds_new_shape[1], *[dim_size for dim_size in gdas_variable_ds_new_shape[2:]])

                        prediction = model.predict(variable_batch_ds_new, batch_size=settings.GPU_PREDICT_BATCH_SIZE)

                        if num_dimensions == 2:
                            image_probs = np.transpose(prediction[0][:, :, :, 1:], transpose_indices)
                        elif num_dimensions == 3:
                            image_probs = np.transpose(np.amax(prediction[0][:, :, :, :, 1:], axis=3), transpose_indices)

                    # Add predictions to the map
                    if variable_data_source != 'era5':
                        for timestep in range(num_timesteps_in_batch):
                            for fcst_hr_index in range(num_forecast_hours):
                                stitched_map_probs[timestep][fcst_hr_index], map_created = add_image_to_map(stitched_map_probs[timestep][fcst_hr_index], image_probs[timestep * num_forecast_hours + fcst_hr_index], map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                                    image_size_lon, image_size_lat, lon_image_spacing, lat_image_spacing)

                    else:  # if variable_data_source == 'era5'
                        for timestep in range(num_timesteps_in_batch):
                            stitched_map_probs[timestep], map_created = add_image_to_map(stitched_map_probs[timestep], image_probs[timestep], map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                                image_size_lon, image_size_lat, lon_image_spacing, lat_image_spacing)
                    ####################################################################################################

                    if map_created is True:

                        ### Create subdirectories for the data if they do not exist ###
                        if not os.path.isdir('%s/model_%d/maps/%s' % (model_dir, model_number, subdir_base)):
                            os.mkdir('%s/model_%d/maps/%s' % (model_dir, model_number, subdir_base))
                            print("New subdirectory made:", '%s/model_%d/maps/%s' % (model_dir, model_number, subdir_base))
                        if not os.path.isdir('%s/model_%d/probabilities/%s' % (model_dir, model_number, subdir_base)):
                            os.mkdir('%s/model_%d/probabilities/%s' % (model_dir, model_number, subdir_base))
                            print("New subdirectory made:", '%s/model_%d/probabilities/%s' % (model_dir, model_number, subdir_base))
                        if not os.path.isdir('%s/model_%d/statistics/%s' % (model_dir, model_number, subdir_base)):
                            os.mkdir('%s/model_%d/statistics/%s' % (model_dir, model_number, subdir_base))
                            print("New subdirectory made:", '%s/model_%d/statistics/%s' % (model_dir, model_number, subdir_base))

                        if variable_data_source != 'era5':

                            for timestep_no, timestep in enumerate(timesteps):
                                timestep = str(timestep)
                                for fcst_hr_index, forecast_hour in enumerate(forecast_hours):
                                    time = f'{timestep[:4]}-%s-%s-%sz' % (timestep[5:7], timestep[8:10], timestep[11:13])
                                    probs_ds = create_model_prediction_dataset(stitched_map_probs[timestep_no][fcst_hr_index], image_lats, image_lons, front_types)
                                    probs_ds = probs_ds.expand_dims({'time': np.atleast_1d(timestep), 'forecast_hour': np.atleast_1d(forecast_hours[fcst_hr_index])})
                                    filename_base = 'model_%d_%s_%s_%s_f%03d_%dx%d' % (model_number, time, domain, variable_data_source, forecast_hours[fcst_hr_index], domain_images_lon, domain_images_lat)
                                    if random_variables is not None:
                                        filename_base += '_' + '-'.join(sorted(random_variables))

                                    outfile = '%s/model_%d/probabilities/%s/%s_probabilities.nc' % (model_dir, model_number, subdir_base, filename_base)
                                    probs_ds.to_netcdf(path=outfile, engine='netcdf4', mode='w')

                        else:

                            for timestep_no, timestep in enumerate(timesteps):
                                time = f'{timestep[:4]}-%s-%s-%sz' % (timestep[5:7], timestep[8:10], timestep[11:13])
                                probs_ds = create_model_prediction_dataset(stitched_map_probs[timestep_no], image_lats, image_lons, front_types)
                                probs_ds = probs_ds.expand_dims({'time': np.atleast_1d(timestep)})
                                filename_base = 'model_%d_%s_%s_%dx%d' % (model_number, time, domain, domain_images_lon, domain_images_lat)
                                if random_variables is not None:
                                    filename_base += '_' + '-'.join(sorted(random_variables))

                                outfile = '%s/model_%d/probabilities/%s/%s_probabilities.nc' % (model_dir, model_number, subdir_base, filename_base)
                                probs_ds.to_netcdf(path=outfile, engine='netcdf4', mode='w')


def create_model_prediction_dataset(stitched_map_probs: np.array, lats, lons, front_types: str or list):
    """
    Create an Xarray dataset containing model predictions.

    Parameters
    ----------
    stitched_map_probs: np.array
        - Numpy array with probabilities for the given front type(s).
        - Shape/dimensions: [front types, longitude, latitude]
    lats: np.array
        - 1D array of latitude values.
    lons: np.array
        - 1D array of longitude values.
    front_types: str or list
        - Front types within the dataset. See documentation in utils.data_utils.reformat fronts for more information.

    Returns
    -------
    probs_ds: xr.Dataset
        - Xarray dataset containing front probabilities predicted by the model for each front type.
    """

    if front_types == 'F_BIN' or front_types == 'MERGED-F_BIN' or front_types == 'MERGED-T':
        probs_ds = xr.Dataset(
            {front_types: (('longitude', 'latitude'), stitched_map_probs[0])},
            coords={'latitude': lats, 'longitude': lons})
    elif front_types == 'MERGED-F':
        probs_ds = xr.Dataset(
            {'CF_merged': (('longitude', 'latitude'), stitched_map_probs[0]),
             'WF_merged': (('longitude', 'latitude'), stitched_map_probs[1]),
             'SF_merged': (('longitude', 'latitude'), stitched_map_probs[2]),
             'OF_merged': (('longitude', 'latitude'), stitched_map_probs[3])},
            coords={'latitude': lats, 'longitude': lons})
    elif front_types == 'MERGED-ALL':
        probs_ds = xr.Dataset(
            {'CF_merged': (('longitude', 'latitude'), stitched_map_probs[0]),
             'WF_merged': (('longitude', 'latitude'), stitched_map_probs[1]),
             'SF_merged': (('longitude', 'latitude'), stitched_map_probs[2]),
             'OF_merged': (('longitude', 'latitude'), stitched_map_probs[3]),
             'TROF_merged': (('longitude', 'latitude'), stitched_map_probs[4]),
             'INST': (('longitude', 'latitude'), stitched_map_probs[5]),
             'DL': (('longitude', 'latitude'), stitched_map_probs[6])},
            coords={'latitude': lats, 'longitude': lons})
    elif type(front_types) == list:
        probs_ds_dict = dict({})
        for probs_ds_index, front_type in enumerate(front_types):
            probs_ds_dict[front_type] = (('longitude', 'latitude'), stitched_map_probs[probs_ds_index])
        probs_ds = xr.Dataset(probs_ds_dict, coords={'latitude': lats, 'longitude': lons})
    else:
        raise ValueError(f"'{front_types}' is not a valid set of front types.")

    return probs_ds


def calibrate_model(model_number, model_dir, domain, domain_images, variable_data_source):

    model_properties = pd.read_pickle('%s/model_%d/model_%d_properties.pkl' % (model_dir, model_number, model_number))
    front_types = model_properties['front_types']

    if type(front_types) == str:
        front_types = [front_types, ]
    
    try:
        _ = model_properties['calibration_models']  # Check to see if the model has already been calibrated before
    except KeyError:
        model_properties['calibration_models'] = dict()

    model_properties['calibration_models']['%s_%dx%d' % (domain, domain_images[0], domain_images[1])] = dict()

    subdir_base = '%s_%dx%d' % (domain, domain_images[0], domain_images[1])

    if variable_data_source != 'era5':
        files = sorted(glob(f'%s\\model_%d\\statistics\\%s\\*_{variable_data_source}_f%03d_{domain}_{domain_images[0]}x{domain_images[1]}*statistics.nc' % (model_dir, model_number, subdir_base, forecast_hour)))
    else:
        files = sorted(glob(f'%s\\model_%d\\statistics\\%s\\*{domain}_{domain_images[0]}x{domain_images[1]}*statistics.nc' % (model_dir, model_number, subdir_base)))

    ### If evaluating over the full domain, remove non-synoptic hours (3z, 9z, 15z, 21z) ###
    if domain == 'full':
        if variable_data_source == 'era5':
            hours_to_remove = [3, 9, 15, 21]
            for hour in hours_to_remove:
                string = '%02dz_' % hour
                files = list(filter(lambda hour: string not in hour, files))
        else:
            forecast_hours_to_remove = [3, 9]
            for hour in forecast_hours_to_remove:
                string = '_f%03d_' % hour
                files = list(filter(lambda hour: string not in hour, files))

    print("opening datasets")
    if variable_data_source != 'era5':
        stats_ds = xr.open_mfdataset(files, combine='nested').isel(forecast_hour=0).transpose('time', 'boundary', 'threshold')
    else:
        stats_ds = xr.open_mfdataset(files, combine='nested', concat_dim='time')

    axis_ticks = np.arange(0.1, 1.1, 0.1)

    for front_label in front_types:

        model_properties['calibration_models']['%s_%dx%d' % (domain, domain_images[0], domain_images[1])][front_label] = dict()

        true_positives = stats_ds[f'tp_{front_label}'].values
        false_positives = stats_ds[f'fp_{front_label}'].values

        thresholds = stats_ds['threshold'].values

        ### Sum the true positives along the 'time' axis ###
        true_positives_sum = np.sum(true_positives, axis=0)
        false_positives_sum = np.sum(false_positives, axis=0)

        ### Find the number of true positives and false positives in each probability bin ###
        true_positives_diff = np.abs(np.diff(true_positives_sum))
        false_positives_diff = np.abs(np.diff(false_positives_sum))
        observed_relative_frequency = np.divide(true_positives_diff, true_positives_diff + false_positives_diff)

        boundary_colors = ['red', 'purple', 'brown', 'darkorange', 'darkgreen']

        calibrated_probabilities = []

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        axs[0].plot(thresholds, thresholds, color='black', linestyle='--', linewidth=0.5, label='Perfect Reliability')

        for boundary, color in enumerate(boundary_colors):

            ####################### Test different calibration methods to see which performs best ######################

            x = [threshold for threshold, frequency in zip(thresholds[1:], observed_relative_frequency[boundary]) if not np.isnan(frequency)]
            y = [frequency for threshold, frequency in zip(thresholds[1:], observed_relative_frequency[boundary]) if not np.isnan(frequency)]

            ### Isotonic Regression ###
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit_transform(x, y)
            calibrated_probabilities.append(ir.predict(x))
            r_squared = r2_score(y, calibrated_probabilities[boundary])

            axs[0].plot(x, y, color=color, linewidth=1, label='%d km' % ((boundary + 1) * 50))
            axs[1].plot(x, calibrated_probabilities[boundary], color=color, linestyle='--', linewidth=1, label=r'%d km ($R^2$ = %.3f)' % ((boundary + 1) * 50, r_squared))
            model_properties['calibration_models']['%s_%dx%d' % (domain, domain_images[0], domain_images[1])][front_label]['%d km' % ((boundary + 1) * 50)] = ir

        for ax in axs:

            axs[0].set_xlabel("Forecast Probability (uncalibrated)")
            ax.set_xticks(axis_ticks)
            ax.set_yticks(axis_ticks)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid()
            ax.legend()

        axs[0].set_title('Reliability Diagram')
        axs[1].set_title('Calibration (isotonic regression)')
        axs[0].set_ylabel("Observed Relative Frequency")
        axs[1].set_ylabel("Forecast Probability (calibrated)")

        with open('%s/model_%d/model_%d_properties.pkl' % (model_dir, model_number, model_number), 'wb') as f:
            pickle.dump(model_properties, f)

        plt.suptitle(f'Model {model_number} reliability/calibration: {settings.DEFAULT_FRONT_NAMES[front_label]}')
        plt.savefig(f'%s/model_%d/model_%d_calibration_%s_%dx%d_%s.png' % (model_dir, model_number, model_number, domain, domain_images[0], domain_images[1], front_label),
                    bbox_inches='tight', dpi=300)
        plt.close()


def plot_performance_diagrams(model_dir, model_number, fronts_netcdf_indir, domain, domain_images, bootstrap=True, random_variables=None,
    variable_data_source='era5', calibrated=False, num_iterations=10000, forecast_hour=None):
    """
    Plots CSI performance diagram for different front types.

    Parameters
    ----------
    model_dir: str
        - Main directory for the models.
    model_number: int
        - Slurm job number for the model. This is the number in the model's filename.
    domain: str
        - Domain of the data.
    domain_images: iterable object with 2 ints
        - Number of images along each dimension of the final stitched map (lon lat).
    bootstrap: bool
        - Setting this to true will plot confidence intervals onto the performance diagrams.
    random_variables: str or list of strs
        - Variable(s) that were randomized when performance statistics were calculated.
    variable_data_source: str
        - Variable data to use for training the model. Options are: 'era5', 'gdas', or 'gfs' (case-insensitive)
    calibrated: bool
        - Indicates whether or not the statistics to be plotted are from a calibrated model.
    num_iterations: int
        - Number of iterations when bootstrapping the data.
    forecast_hour: int
        - Forecast hour for the data. This will only be used if the data source is not ERA5.
    """

    model_properties = pd.read_pickle(f"{model_dir}\\model_{model_number}\\model_{model_number}_properties.pkl")
    front_types = model_properties['front_types']
    num_front_types = model_properties['classes'] - 1  # model_properties['classes'] - 1 ===> number of front types (we ignore the 'no front' type)

    if model_number in [7805504, 7866106, 7961517]:
        num_dimensions = 2
    else:
        num_dimensions = 3

    subdir_base = '%s_%dx%d' % (domain, domain_images[0], domain_images[1])
    stats_plot_base = 'model_%d_%s_%dx%d' % (model_number, domain, domain_images[0], domain_images[1])
    if random_variables is not None:
        subdir_base += '_' + '-'.join(sorted(random_variables))

    domain_extent_indices = settings.DEFAULT_DOMAIN_INDICES[domain]

    if variable_data_source != 'era5':
        files = sorted(glob(f'%s\\model_%d\\statistics\\%s\\*_{variable_data_source}_f%03d_{domain}_{domain_images[0]}x{domain_images[1]}*statistics.nc' % (model_dir, model_number, subdir_base, forecast_hour)))
    else:
        files = sorted(glob(f'%s\\model_%d\\statistics\\%s\\*{domain}_{domain_images[0]}x{domain_images[1]}*statistics.nc' % (model_dir, model_number, subdir_base)))

    ### If evaluating over the full domain, remove non-synoptic hours (3z, 9z, 15z, 21z) ###
    if domain == 'full':
        if variable_data_source == 'era5':
            hours_to_remove = [3, 9, 15, 21]
            for hour in hours_to_remove:
                string = '%02dz_' % hour
                files = list(filter(lambda hour: string not in hour, files))
        else:
            forecast_hours_to_remove = [3, 9]
            for hour in forecast_hours_to_remove:
                string = '_f%03d_' % hour
                files = list(filter(lambda hour: string not in hour, files))

        synoptic_only = True
    else:
        synoptic_only = False

    print("opening datasets")
    if variable_data_source != 'era5':
        stats_ds = xr.open_mfdataset(files, combine='nested').isel(forecast_hour=0).transpose('time', 'boundary', 'threshold')
    else:
        stats_ds = xr.open_mfdataset(files, combine='nested', concat_dim='time')

    # spatial_csi_obj = fm.SpatialCSIfiles(model_number, model_dir, domain, domain_images, variable_data_source, forecast_hour)
    # spatial_csi_obj.pair_with_fronts(front_indir=fronts_netcdf_indir, synoptic_only=synoptic_only, sort_fronts=True)
    # spatial_csi_obj.test_years = [2019, 2020]
    # probs_files = spatial_csi_obj.probs_files_test
    #
    # probs_ds = xr.open_mfdataset(probs_files)

    if front_types == ['CF', 'WF']:
        filename_str = 'CFWF'
    elif front_types == ['SF', 'OF']:
        filename_str = 'SFOF'
    else:
        filename_str = 'F_BIN'

    start_lon, end_lon = domain_extent_indices[0], domain_extent_indices[1]
    start_lat, end_lat = domain_extent_indices[2], domain_extent_indices[3]

    expanded_fronts_array = pd.read_pickle(f'expanded_fronts_{filename_str}.pkl')[:, start_lon:end_lon, start_lat:end_lat]
    fronts_array = pd.read_pickle(f'fronts_{filename_str}.pkl')[0, :, start_lon:end_lon, start_lat:end_lat]

    # If domain is not CONUS, select timesteps in the fronts arrays only pertaining to synoptic hours
    if domain != 'conus':
        expanded_fronts_array = expanded_fronts_array[::2, :, :]
        fronts_array = fronts_array[::2, :, :]
    print("done")

    num_timesteps = len(stats_ds['time'].values)

    POD_array = np.empty([num_front_types, num_iterations, 5, 100])  # Probability of detection
    SR_array = np.empty([num_front_types, num_iterations, 5, 100])  # Success ratio

    ### 95% confidence intervals for POD and SR ###
    CI_lower_POD = np.empty([num_front_types, 5, 100])
    CI_lower_SR = np.empty([num_front_types, 5, 100])
    CI_upper_POD = np.empty([num_front_types, 5, 100])
    CI_upper_SR = np.empty([num_front_types, 5, 100])

    selectable_indices = range(num_timesteps)

    if type(front_types) == str:
        front_types = [front_types, ]

    for front_no, front_label in enumerate(front_types):

        ################################ CSI and reliability diagrams (panels a and b) #################################
        true_positives = stats_ds[f'tp_{front_label}'].values
        false_positives = stats_ds[f'fp_{front_label}'].values
        false_negatives = stats_ds[f'fn_{front_label}'].values

        thresholds = stats_ds['threshold'].values

        ### Sum the true positives along the 'time' axis ###
        true_positives_sum = np.sum(true_positives, axis=0)
        false_positives_sum = np.sum(false_positives, axis=0)
        false_negatives_sum = np.sum(false_negatives, axis=0)

        ### Find the number of true positives and false positives in each probability bin ###
        true_positives_diff = np.abs(np.diff(true_positives_sum))
        false_positives_diff = np.abs(np.diff(false_positives_sum))
        observed_relative_frequency = np.divide(true_positives_diff, true_positives_diff + false_positives_diff)

        pod = np.divide(true_positives_sum, true_positives_sum + false_negatives_sum)  # Probability of detection
        sr = np.divide(true_positives_sum, true_positives_sum + false_positives_sum)  # Success ratio

        if bootstrap:

            for iteration in range(num_iterations):
                print(f"Iteration {iteration}/{num_iterations}", end='\r')
                indices = random.choices(selectable_indices, k=num_timesteps)  # Select a sample equal to the total number of timesteps

                POD_array[front_no, iteration, :, :] = np.nan_to_num(np.divide(np.sum(true_positives[indices, :, :], axis=0),
                                                                        np.add(np.sum(true_positives[indices, :, :], axis=0),
                                                                               np.sum(false_negatives[indices, :, :], axis=0))))
                SR_array[front_no, iteration, :, :] = np.nan_to_num(np.divide(np.sum(true_positives[indices, :, :], axis=0),
                                                                       np.add(np.sum(true_positives[indices, :, :], axis=0),
                                                                              np.sum(false_positives[indices, :, :], axis=0))))

            # Calculate confidence intervals at each probability bin
            for percent in np.arange(0, 100):
                CI_lower_POD[front_no, :, percent] = np.percentile(POD_array[front_no, :, :, percent], q=2.5, axis=0)
                CI_upper_POD[front_no, :, percent] = np.percentile(POD_array[front_no, :, :, percent], q=97.5, axis=0)
                CI_lower_SR[front_no, :, percent] = np.percentile(SR_array[front_no, :, :, percent], q=2.5, axis=0)
                CI_upper_SR[front_no, :, percent] = np.percentile(SR_array[front_no, :, :, percent], q=97.5, axis=0)

            """ 
            Some of the percentage thresholds have no data and will show up as zero in the CIs, so we not include them when interpolating the CIs.
            These arrays have four values: [50km boundary, 100km, 150km, 200km]
            These also represent the first index where data is missing, so only values before the index will be included 
            """
            CI_zero_index = np.min([np.min(np.where(CI_lower_SR[front_no, 0] == 0)[0]), np.min(np.where(CI_lower_SR[front_no, 1] == 0)[0]),
                            np.min(np.where(CI_lower_SR[front_no, 2] == 0)[0]), np.min(np.where(CI_lower_SR[front_no, 3] == 0)[0]),
                            np.min(np.where(CI_upper_SR[front_no, 0] == 0)[0]), np.min(np.where(CI_upper_SR[front_no, 1] == 0)[0]),
                            np.min(np.where(CI_upper_SR[front_no, 2] == 0)[0]), np.min(np.where(CI_upper_SR[front_no, 3] == 0)[0]),
                            np.min(np.where(CI_lower_POD[front_no, 0] == 0)[0]), np.min(np.where(CI_lower_POD[front_no, 1] == 0)[0]),
                            np.min(np.where(CI_lower_POD[front_no, 2] == 0)[0]), np.min(np.where(CI_lower_POD[front_no, 3] == 0)[0]),
                            np.min(np.where(CI_upper_POD[front_no, 0] == 0)[0]), np.min(np.where(CI_upper_POD[front_no, 1] == 0)[0]),
                            np.min(np.where(CI_upper_POD[front_no, 2] == 0)[0]), np.min(np.where(CI_upper_POD[front_no, 3] == 0)[0])])

        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        axarr = axs.flatten()

        sr_matrix, pod_matrix = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        csi_matrix = (sr_matrix ** -1 + pod_matrix ** -1 - 1.) ** -1  # CSI coordinates
        fb_matrix = pod_matrix * (sr_matrix ** -1)  # Frequency Bias coordinates
        CSI_LEVELS = np.linspace(0, 1, 11)  # CSI contour levels
        FB_LEVELS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3]  # Frequency Bias levels
        cmap = 'Blues'  # Colormap for the CSI contours
        axis_ticks = np.arange(0, 1.1, 0.1)

        cs = axarr[0].contour(sr_matrix, pod_matrix, fb_matrix, FB_LEVELS, colors='black', linewidths=0.5, linestyles='--')  # Plot FB levels
        axarr[0].clabel(cs, FB_LEVELS, fontsize=8)

        csi_contour = axarr[0].contourf(sr_matrix, pod_matrix, csi_matrix, CSI_LEVELS, cmap=cmap)  # Plot CSI contours in 0.1 increments
        cbar = fig.colorbar(csi_contour, ax=axarr[0], pad=0.02, label='Critical Success Index (CSI)')
        cbar.set_ticks(axis_ticks)

        axarr[1].plot(thresholds, thresholds, color='black', linestyle='--', linewidth=0.5, label='Perfect Reliability')

        cell_text = []  # List of strings that will be used in the table near the bottom of this function

        ### CSI and reliability lines for each boundary ###
        boundary_colors = ['red', 'purple', 'brown', 'darkorange', 'darkgreen']
        max_CSI_scores_by_boundary = np.empty(shape=(5,))
        for boundary, color in enumerate(boundary_colors):
            csi = np.power((1/sr[boundary]) + (1/pod[boundary]) - 1, -1)
            max_CSI_scores_by_boundary[boundary] = np.nanmax(csi)
            max_CSI_index = np.where(csi == max_CSI_scores_by_boundary[boundary])[0]
            max_CSI_threshold = thresholds[max_CSI_index][0]  # Probability threshold where CSI is maximized
            max_CSI_pod = pod[boundary][max_CSI_index][0]  # POD where CSI is maximized
            max_CSI_sr = sr[boundary][max_CSI_index][0]  # SR where CSI is maximized
            max_CSI_fb = max_CSI_pod / max_CSI_sr  # Frequency bias

            cell_text.append(['%.2f' % max_CSI_threshold, '%.2f' % max_CSI_scores_by_boundary[boundary], '%.2f' % max_CSI_pod, '%.2f' % max_CSI_sr, '%.2f' % (1 - max_CSI_sr), '%.2f' % max_CSI_fb])

            # Plot CSI lines
            axarr[0].plot(max_CSI_sr, max_CSI_pod, color=color, marker='*', markersize=10)
            axarr[0].plot(sr[boundary], pod[boundary], color=color, linewidth=1)

            # Plot reliability curve
            axarr[1].plot(thresholds[1:] + 0.005, observed_relative_frequency[boundary], color=color, linewidth=1)

            if bootstrap:
                xs = np.concatenate([CI_lower_SR[front_no, boundary, :CI_zero_index], CI_upper_SR[front_no, boundary, :CI_zero_index][::-1]])
                ys = np.concatenate([CI_lower_POD[front_no, boundary, :CI_zero_index], CI_upper_POD[front_no, boundary, :CI_zero_index][::-1]])
                axarr[0].fill(xs, ys, alpha=0.3, color=color)  # Shade the confidence interval

        axarr[0].set_xlabel("Success Ratio (SR = 1 - FAR)")
        axarr[0].set_ylabel("Probability of Detection (POD)")
        axarr[0].set_title('a) CSI diagram')

        if calibrated:
            axarr[1].set_xlabel("Forecast Probability (calibrated)")
        else:
            axarr[1].set_xlabel("Forecast Probability (uncalibrated)")
        axarr[1].set_ylabel("Observed Relative Frequency")
        axarr[1].set_title('b) Reliability diagram')

        for ax in axarr:
            ax.set_xticks(axis_ticks)
            ax.set_yticks(axis_ticks)
            ax.grid(color='black', alpha=0.1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ################################################################################################################

        ############################################# Data table (panel c) #############################################
        columns = ['Threshold*', 'CSI', 'POD', 'SR', 'FAR', 'FB']  # Column names
        rows = ['50 km', '100 km', '150 km', '200 km', '250 km']  # Row names

        table_axis = plt.axes([0.063, -0.06, 0.4, 0.2])
        table_axis.set_title('c) Data table', x=0.415, y=0.135, pad=-4)
        table_axis.axis('off')
        table_axis.text(0.16, -2.7, '* probability threshold where CSI is maximized')  # Add disclaimer for probability threshold column
        stats_table = table_axis.table(cellText=cell_text, rowLabels=rows, rowColours=boundary_colors, colLabels=columns, cellLoc='center')
        stats_table.scale(1, 3)  # Make the table larger

        ### Shade the cells and make the cell text larger ###
        for cell in stats_table._cells:
            stats_table._cells[cell].set_alpha(.7)
            stats_table._cells[cell].set_text_props(fontproperties=FontProperties(size='xx-large', stretch='expanded'))
        ################################################################################################################

        ########################################## Spatial CSI map (panel d) ###########################################
        # probs_array = probs_ds[front_label].values
        #
        # front_label_classes = {'CF': 1, 'WF': 2, 'SF': 1, 'OF': 2, 'F_BIN': 1}
        #
        # ### Calculate true positives (TP), false positives (FP), and false negatives (FN) for the CSI ###
        # spatial_tp = np.where((probs_array >= thresholds[max_CSI_index]) & (expanded_fronts_array == front_label_classes[front_label]), 1, 0).sum(0)
        # spatial_fp = np.where((probs_array >= thresholds[max_CSI_index]) & (expanded_fronts_array != front_label_classes[front_label]), 1, 0).sum(0)
        # spatial_fn = np.where((probs_array < thresholds[max_CSI_index]) & (fronts_array == front_label_classes[front_label]), 1, 0).sum(0)
        # spatial_csi = spatial_tp / (spatial_tp + spatial_fp + spatial_fn)
        #
        # # Create CSI dataset
        # spatial_stats_ds = xr.Dataset(data_vars={'CSI': (('longitude', 'latitude'), spatial_csi)},
        #                               coords={'latitude': probs_ds['latitude'].values, 'longitude': probs_ds['longitude'].values})

        # Colorbar keyword arguments
        # cbar_kwargs = {'label': 'CSI', 'pad': 0}
        #
        # # Adjust the spatial CSI plot based on the domain
        # if domain == 'conus':
        #     spatial_axis_extent = [0.52, -0.59, 0.48, 0.535]
        #     spatial_plot_xlabels = [-140, -105, -70]
        #     spatial_plot_ylabels = [30, 40, 50]
        #     bottom_labels = False  # Disable longitude labels on the bottom of the subplot
        # else:
        #     spatial_axis_extent = [0.538, -0.6, 0.48, 0.577]
        #     cbar_kwargs['shrink'] = 0.862
        #     spatial_plot_xlabels = [-150, -120, -90, -60, -30, 0, 120, 150, 180]
        #     spatial_plot_ylabels = [0, 20, 40, 60, 80]
        #     bottom_labels = True  # Longitude labels on the bottom of the subplot
        #
        # right_labels = False  # Disable latitude labels on the right side of the subplot
        # top_labels = True  # Longitude labels on top of the subplot
        # left_labels = True  # Latitude labels on the left side of the subplot
        #
        # spatial_stats_ds = xr.where(spatial_stats_ds > 0.1, spatial_stats_ds, float("NaN"))  # Mask out CSI scores less than 0.1

        ### Set up the spatial CSI plot ###
        # extent = settings.DEFAULT_DOMAIN_EXTENTS[domain]
        # spatial_axis = plt.axes(spatial_axis_extent, projection=ccrs.LambertConformal(central_longitude=250))
        # spatial_axis_title_text = '250 km CSI map'
        # plot_background(extent=extent, ax=spatial_axis)
        # norm_probs = colors.Normalize(vmin=0.1, vmax=1)
        # spatial_stats_ds['CSI'].plot(ax=spatial_axis, x='longitude', y='latitude', norm=norm_probs, cmap='gnuplot2', transform=ccrs.PlateCarree(), alpha=0.35, cbar_kwargs=cbar_kwargs)
        # spatial_axis.set_title(spatial_axis_title_text)
        # gl = spatial_axis.gridlines(draw_labels=True, zorder=0, dms=True, x_inline=False, y_inline=False)
        # gl.right_labels = right_labels
        # gl.top_labels = top_labels
        # gl.left_labels = left_labels
        # gl.bottom_labels = bottom_labels
        # gl.xlocator = FixedLocator(spatial_plot_xlabels)
        # gl.ylocator = FixedLocator(spatial_plot_ylabels)
        # gl.xlabel_style = {'size': 7}
        # gl.ylabel_style = {'size': 8}
        ################################################################################################################

        ###################################### Generate title for the whole plot #######################################
        ### Kernel size text ###
        kernel_text = '%s' % model_properties['kernel_size']
        for dim in range(num_dimensions - 1):
            kernel_text += 'x%s' % model_properties['kernel_size']

        ### Front name text ###
        front_text = settings.DEFAULT_FRONT_NAMES[front_label]
        if type(front_types) == list and front_types != ['F_BIN']:
            front_text += 's'
        elif front_types == ['F_BIN']:
            front_text = 'Binary fronts (front / no front)'

        ### Domain text ###
        if domain == 'conus':
            domain_text = 'CONUS'
        else:
            domain_text = domain
        domain_text += f' domain ({int(domain_images[0] * domain_images[1])} images per map)'

        plt.suptitle(f'{num_dimensions}D U-Net 3+ ({kernel_text} kernel): {front_text} over {domain_text}', fontsize=20)  # Create and plot the main title
        ################################################################################################################

        filename = f"%s/model_%d/%s_performance_%s_{variable_data_source}.png" % (model_dir, model_number, stats_plot_base, front_label)
        if variable_data_source != 'era5':
            filename = filename.replace('.png', '_f%03d.png' % forecast_hour)  # Add forecast hour to the end of the filename

        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()


def prediction_plot(model_number, model_dir, fronts_netcdf_dir, timestep, domain, domain_images, forecast_hour=None,
    variable_data_source='era5', probability_mask=(0.10, 0.10), calibration_km=None, same_map=True):
    """
    Function that uses generated predictions to make probability maps along with the 'true' fronts and saves out the
    subplots.

    Parameters
    ----------
    model_number: int
        - Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        - Main directory for the models.
    fronts_netcdf_dir: str
        - Input directory for the front object netcdf files.
    timestep: str
        - Timestring for the prediction plot title.
    domain: str
        - Domain of the data.
    domain_images: iterable object with 2 ints
        - Number of images along each dimension of the final stitched map (lon lat).
    forecast_hour: int or None
        - Forecast hour for the timestep to plot. Has no effect if plotting predictions from ERA5 data.
    variable_data_source: str
        - Variable data to use for training the model. Options are: 'era5', 'gdas', or 'gfs' (case-insensitive)
    probability_mask: tuple with 2 floats
        - Probability mask and the step/interval for the probability contours. Probabilities smaller than the mask will
            not be plotted.
    calibration_km: int or None
        - Neighborhood calibration distance in kilometers. Possible neighborhoods are 50, 100, 150, 200, and 250 km.
    same_map: bool
        - Plot the model predictions on the same map.
    """

    DEFAULT_COLORBAR_POSITION = {'conus': 0.76, 'full': 0.83}
    cbar_position = DEFAULT_COLORBAR_POSITION[domain]

    model_properties = pd.read_pickle(f"{model_dir}/model_{model_number}/model_{model_number}_properties.pkl")

    variable_data_source = variable_data_source.lower()

    extent = settings.DEFAULT_DOMAIN_EXTENTS[domain]

    year, month, day, hour = int(timestep[0]), int(timestep[1]), int(timestep[2]), int(timestep[3])

    subdir_base = '%s_%dx%d' % (domain, domain_images[0], domain_images[1])
    probs_dir = f'{model_dir}/model_{model_number}/probabilities/{subdir_base}'

    if forecast_hour is not None:
        forecast_timestep = data_utils.add_or_subtract_hours_to_timestep('%d%02d%02d%02d' % (year, month, day, hour), num_hours=forecast_hour)
        new_year, new_month, new_day, new_hour = forecast_timestep[:4], forecast_timestep[4:6], forecast_timestep[6:8], int(forecast_timestep[8:]) - (int(forecast_timestep[8:]) % 3)
        fronts_file = '%s/%s/%s/%s/FrontObjects_%s%s%s%02d_full.nc' % (fronts_netcdf_dir, new_year, new_month, new_day, new_year, new_month, new_day, new_hour)
        filename_base = f'model_%d_{year}-%02d-%02d-%02dz_%s_%s_f%03d_%dx%d' % (model_number, month, day, hour, domain, variable_data_source, forecast_hour, domain_images[0], domain_images[1])
    else:
        fronts_file = '%s/%d/%02d/%02d/FrontObjects_%d%02d%02d%02d_full.nc' % (fronts_netcdf_dir, year, month, day, year, month, day, hour)
        filename_base = f'model_%d_{year}-%02d-%02d-%02dz_%s_%dx%d' % (model_number, month, day, hour, domain, domain_images[0], domain_images[1])
        variable_data_source = 'era5'

    probs_file = f'{probs_dir}/{filename_base}_probabilities.nc'

    fronts = xr.open_dataset(fronts_file).sel(longitude=slice(extent[0], extent[1]), latitude=slice(extent[3], extent[2]))
    probs_ds = xr.open_mfdataset(probs_file)

    crs = ccrs.LambertConformal(central_longitude=250)

    image_size = model_properties['image_size']  # The image size does not include the last dimension of the input size as it only represents the number of channels
    front_types = model_properties['front_types']

    fronts = data_utils.reformat_fronts(fronts, front_types=front_types)
    fronts = xr.where(fronts == 0, float('NaN'), fronts)

    if type(front_types) == str:
        front_types = [front_types, ]

    mask, prob_int = probability_mask[0], probability_mask[1]  # Probability mask, contour interval for probabilities
    vmax, cbar_tick_adjust, cbar_label_adjust, n_colors = 1, prob_int, 10, 11
    levels = np.around(np.arange(0, 1 + prob_int, prob_int), 2)
    cbar_ticks = np.around(np.arange(mask, 1 + prob_int, prob_int), 2)

    contour_maps_by_type = [settings.DEFAULT_CONTOUR_CMAPS[label] for label in front_types]
    front_colors_by_type = [settings.DEFAULT_FRONT_COLORS[label] for label in front_types]
    front_names_by_type = [settings.DEFAULT_FRONT_NAMES[label] for label in front_types]

    cmap_front = colors.ListedColormap(front_colors_by_type, name='from_list', N=len(front_colors_by_type))
    norm_front = colors.Normalize(vmin=1, vmax=len(front_colors_by_type) + 1)

    if same_map:

        if type(front_types) == list and len(front_types) > 1:

            data_arrays = {}
            for key in list(probs_ds.keys()):
                if calibration_km is not None:
                    ir_model = model_properties['calibration_models']['%s_%dx%d' % (domain, domain_images[0], domain_images[1])][key]['%d km' % calibration_km]
                    original_shape = np.shape(probs_ds[key].values)
                    data_arrays[key] = ir_model.predict(probs_ds[key].values.flatten()).reshape(original_shape)
                    cbar_label = 'Probability (calibrated - %d km)' % calibration_km
                else:
                    data_arrays[key] = probs_ds[key].values
                    cbar_label = 'Probability (uncalibrated)'

            all_possible_front_combinations = itertools.permutations(front_types, r=2)
            for combination in all_possible_front_combinations:
                probs_ds[combination[0]].values = np.where(data_arrays[combination[0]] > data_arrays[combination[1]] - 0.02, data_arrays[combination[0]], 0)

        if variable_data_source != 'era5':
            probs_ds = xr.where(probs_ds > mask, probs_ds, float("NaN")).isel(time=0).sel(forecast_hour=forecast_hour)
            valid_time = data_utils.add_or_subtract_hours_to_timestep(f'%d%02d%02d%02d' % (year, month, day, hour), num_hours=forecast_hour)
            data_title = f'Run: {variable_data_source.upper()} {year}-%02d-%02d-%02dz F%03d \nPredictions valid: {valid_time[:4]}-{valid_time[4:6]}-{valid_time[6:8]}-{valid_time[8:]}z' % (month, day, hour, forecast_hour)
            fronts_valid_title = f'Fronts valid: {new_year}-{"%02d" % int(new_month)}-{"%02d" % int(new_day)}-{"%02d" % new_hour}z'
        else:
            probs_ds = xr.where(probs_ds > mask, probs_ds, float("NaN")).isel(time=0)
            data_title = 'Data: ERA5 reanalysis %d-%02d-%02d-%02dz' % (year, month, day, hour)
            fronts_valid_title = f'Fronts valid: %d-%02d-%02d-%02dz' % (year, month, day, hour)

        fig, ax = plt.subplots(1, 1, figsize=(20, 8), subplot_kw={'projection': crs})
        plot_background(extent, ax=ax, linewidth=0.5)
        # ax.gridlines(draw_labels=True, zorder=0)

        for front_no, front_key, front_name, front_label, cmap in zip(range(1, len(front_names_by_type) + 1), list(probs_ds.keys()), front_names_by_type, front_types, contour_maps_by_type):

            cmap_probs, norm_probs = cm.get_cmap(cmap, n_colors), colors.Normalize(vmin=0, vmax=vmax)
            probs_ds[front_key].plot.contourf(ax=ax, x='longitude', y='latitude', norm=norm_probs, levels=levels, cmap=cmap_probs, transform=ccrs.PlateCarree(), alpha=0.75, add_colorbar=False)
            fronts['identifier'].plot(ax=ax, x='longitude', y='latitude', cmap=cmap_front, norm=norm_front, transform=ccrs.PlateCarree(), add_colorbar=False)

            cbar_ax = fig.add_axes([cbar_position + (front_no * 0.015), 0.11, 0.015, 0.77])
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm_probs, cmap=cmap_probs), cax=cbar_ax, boundaries=levels[1:], alpha=0.75)
            cbar.set_ticklabels([])

        cbar.set_label(cbar_label, rotation=90)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticks)

        ax.set_title(f"{'/'.join(front_name.replace(' front', '') for front_name in front_names_by_type)} predictions")
        ax.set_title(data_title, loc='left')
        ax.set_title(fronts_valid_title, loc='right')

        plt.savefig('%s/model_%d/maps/%s/%s-same.png' % (model_dir, model_number, subdir_base, filename_base), bbox_inches='tight', dpi=300)
        plt.close()

    else:
        probs_ds = xr.where(probs_ds > probability_mask, probs_ds, float("NaN")).isel(time=0).sel(forecast_hour=forecast_hour)
        for front_no, front_key, front_name, front_label, cmap in zip(range(1, len(front_names_by_type) + 1), list(probs_ds.keys()), front_names_by_type, labels, contour_maps_by_type):
            if variable_data_source != 'era5':
                fig, ax = plt.subplots(1, 1, figsize=(20, 8), subplot_kw={'projection': crs})
                plot_background(extent, ax=ax, linewidth=0.5)
                current_fronts = fronts
                current_fronts = xr.where(current_fronts != front_no, float("NaN"), front_no)
                cmap_probs, norm_probs = cm.get_cmap(cmap, n_colors), colors.Normalize(vmin=0, vmax=vmax)
                probs_ds[front_key].sel(forecast_hour=forecast_hour).plot.contourf(ax=ax, x='longitude', y='latitude', norm=norm_probs, levels=levels, cmap=cmap_probs,
                    transform=ccrs.PlateCarree(), alpha=0.75, add_colorbar=False)
                # current_fronts['identifier'].plot(ax=ax, x='longitude', y='latitude', cmap=cmap_front, norm=norm_front, transform=ccrs.PlateCarree(), add_colorbar=False)
                valid_time = data_utils.add_or_subtract_hours_to_timestep(f'{year}%02d%02d%02d' % (month, day, hour), num_hours=forecast_hour)
                ax.set_title(f'{front_name} predictions')
                ax.set_title(f'Run: {variable_data_source.upper()} {year}-%02d-%02d-%02dz F%03d \nPredictions valid: {valid_time[:4]}-{valid_time[4:6]}-{valid_time[6:8]}-{valid_time[8:]}z' % (month, day, hour, forecast_hour), loc='left')
                # ax.set_title(f'Fronts valid: {new_year}-{"%02d" % int(new_month)}-{"%02d" % int(new_day)}-{"%02d" % new_hour}z', loc='right')
                cbar_ax = fig.add_axes([0.8365, 0.11, 0.015, 0.77])
                cbar = plt.colorbar(cm.ScalarMappable(norm=norm_probs, cmap=cmap_probs), cax=cbar_ax, boundaries=levels[1:], alpha=0.75)
                cbar.set_label('Probability (uncalibrated)', rotation=90)
                cbar.set_ticks(cbar_ticks)
                cbar.set_ticklabels(cbar_tick_labels[int(probability_mask*cbar_label_adjust):])
                plt.savefig('%s/model_%d/maps/%s/%s-%s.png' % (model_dir, model_number, subdir_base, filename_base, front_label), bbox_inches='tight', dpi=300)
                plt.close()
            else:
                # current_fronts = fronts
                # current_fronts = xr.where(current_fronts != front_no, float("NaN"), front_no)
                fig, ax = plt.subplots(1, 1, figsize=(20, 8), subplot_kw={'projection': crs})
                plot_background(extent, ax=ax, linewidth=0.5)
                cmap_probs, norm_probs = cm.get_cmap(cmap, n_colors), colors.Normalize(vmin=0, vmax=vmax)
                probs_ds[front_key].isel().plot.contourf(ax=ax, x='longitude', y='latitude', norm=norm_probs, levels=levels, cmap=cmap_probs,
                    transform=ccrs.PlateCarree(), alpha=0.75, add_colorbar=False)
                ax.set_title(f'Data: ERA5 reanalysis {year}-%02d-%02d-%02dz' % (month, day, hour), loc='left')
                # current_fronts['identifier'].plot(ax=ax, x='longitude', y='latitude', cmap=cmap_front, norm=norm_front, transform=ccrs.PlateCarree(), add_colorbar=False)
                ax.set_title(f'{front_name} predictions')
                cbar_ax = fig.add_axes([0.8365, 0.11, 0.015, 0.77])
                cbar = plt.colorbar(cm.ScalarMappable(norm=norm_probs, cmap=cmap_probs), cax=cbar_ax, boundaries=levels[1:], alpha=0.75)
                cbar.set_label(f'{front_name} probability', rotation=90)
                cbar.set_ticks(cbar_ticks)
                cbar.set_ticklabels(cbar_tick_labels[int(probability_mask*cbar_label_adjust):])
                plt.savefig('%s/model_%d/maps/%s/%s-%s.png' % (model_dir, model_number, subdir_base, filename_base, front_label), bbox_inches='tight', dpi=300)
                plt.close()


def learning_curve(model_number, model_dir):
    """
    Function that plots learning curves for the specified model.

    Parameters
    ----------
    model_number: int
        - Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        - Main directory for the models.
    include_validation_plots: bool
        - Setting this to True will plot validation data in addition to training data.
    """

    """
    loss_title: Title of the loss plots on the learning curves
    """
    with open("%s/model_%d/model_%d_history.csv" % (model_dir, model_number, model_number), 'rb') as f:
        history = pd.read_csv(f)

    model_properties = pd.read_pickle(f"{model_dir}/model_{model_number}/model_{model_number}_properties.pkl")

    # Model properties
    loss = model_properties['loss_function']

    if loss == 'fss':
        fss_mask_size, fss_c = model_properties['fss_mask_c'][0], model_properties['fss_mask_c'][1]
        loss_title = 'Fractions Skill Score (mask=%d, c=%.1f)' % (fss_mask_size, fss_c)
    elif loss == 'bss':
        loss_title = 'Brier Skill Score'
    elif loss == 'cce':
        loss_title = 'Categorical Cross-Entropy'
    else:
        loss_title = None

    min_loss_epoch = np.where(history['loss'] == np.min(history['loss']))[0][0] + 1
    min_val_loss_epoch = np.where(history['val_loss'] == np.min(history['val_loss']))[0][0] + 1

    num_epochs = len(history['val_loss'])

    plt.subplots(1, 1, dpi=300)

    plt.title(loss_title)
    plt.plot(np.arange(1, num_epochs + 1), history['loss'], color='blue', label='Training loss')
    plt.plot(np.arange(1, num_epochs + 1), history['val_loss'], color='red', label='Validation loss')
    plt.hlines(y=history['loss'][min_loss_epoch - 1], xmin=0, xmax=num_epochs + 2, linestyle='--', linewidths=0.6, color='blue')
    plt.hlines(y=history['val_loss'][min_val_loss_epoch - 1], xmin=0, xmax=num_epochs + 2, linestyle='--', linewidths=0.6, color='red')
    plt.text(x=num_epochs + 2, y=history['loss'][min_loss_epoch - 1], s='%.4e (%d)' % (history['loss'][min_loss_epoch - 1], min_loss_epoch), color='blue', va='center', fontdict=dict(fontsize=7))
    plt.text(x=num_epochs + 2, y=history['val_loss'][min_val_loss_epoch - 1], s='%.4e (%d)' % (history['val_loss'][min_val_loss_epoch - 1], min_val_loss_epoch), color='red', va='center', fontdict=dict(fontsize=7))
    plt.xlim(xmin=0, xmax=len(history['val_loss']) + 1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid()
    plt.yscale('log')  # Turns y-axis into a logarithmic scale. Useful if loss functions appear as very sharp curves.

    plt.savefig("%s/model_%d/model_%d_learning_curve.png" % (model_dir, model_number, model_number), bbox_inches='tight')


if __name__ == '__main__':
    """
    All arguments listed in the examples are listed via argparse in alphabetical order below this comment block.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', action='store_true', help='Bootstrap data?')
    parser.add_argument('--dataset', type=str, help="Dataset for which to make predictions if prediction_method is 'random' or 'all'. Options are:"
                                                    "'training', 'validation', 'test'")
    parser.add_argument('--datetime', type=int, nargs=4, help='Date and time of the data. Pass 4 ints in the following order: year, month, day, hour')
    parser.add_argument('--timestep', type=int, nargs=3, help='Date and time of the data. Pass 3 ints in the following order: year, month, day ')
    parser.add_argument('--domain', type=str, help='Domain of the data.')
    parser.add_argument('--domain_images', type=int, nargs=2, help='Number of images for each dimension the final stitched map for predictions: lon, lat')
    parser.add_argument('--domain_size', type=int, nargs=2, help='Lengths of the dimensions of the final stitched map for predictions: lon, lat')
    parser.add_argument('--forecast_hour', type=int, help='Forecast hour for the GDAS data')
    parser.add_argument('--find_matches', action='store_true', help='Find matches for stitching predictions?')
    parser.add_argument('--generate_predictions', action='store_true', help='Generate prediction plots?')
    parser.add_argument('--calculate_stats', action='store_true', help='generate stats')
    parser.add_argument('--calibrate_model', action='store_true', help='Calibrate model')
    parser.add_argument('--gpu_device', type=int, help='GPU device number.')
    parser.add_argument('--image_size', type=int, nargs=2, help="Number of pixels along each dimension of the model's output: lon, lat")
    parser.add_argument('--learning_curve', action='store_true', help='Plot learning curve')
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth on the GPU')
    parser.add_argument('--model_dir', type=str, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, help='Model number.')
    parser.add_argument('--num_iterations', type=int, default=10000, help='Number of iterations to perform when bootstrapping the data.')
    parser.add_argument('--num_rand_predictions', type=int, default=10, help='Number of random predictions to make.')
    parser.add_argument('--fronts_netcdf_indir', type=str, help='Main directory for the netcdf files containing frontal objects.')
    parser.add_argument('--variables_netcdf_indir', type=str, help='Main directory for the netcdf files containing variable data.')
    parser.add_argument('--plot_performance_diagrams', action='store_true', help='Plot performance diagrams for a model?')
    parser.add_argument('--prediction_method', type=str, help="Prediction method. Options are: 'datetime', 'random', 'all'")
    parser.add_argument('--prediction_plot', action='store_true', help='Create plot')
    parser.add_argument('--random_variables', type=str, nargs="+", default=None, help="Variables to randomize when generating predictions.")
    parser.add_argument('--save_map', action='store_true', help='Save maps of the model predictions?')
    parser.add_argument('--save_probabilities', action='store_true', help='Save model prediction data out to netcdf files?')
    parser.add_argument('--save_statistics', action='store_true', help='Save performance statistics data out to netcdf files?')
    parser.add_argument('--variable_data_source', type=str, default='era5', help='Data source for variables')

    args = parser.parse_args()
    provided_arguments = vars(args)

    ### Use a GPU ###
    if args.gpu_device is not None:
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        tf.config.experimental.set_visible_devices(devices=gpus[args.gpu_device], device_type='GPU')

        # Allow for memory growth on the GPU. This will only use the GPU memory that is required rather than allocating all of the GPU's memory.
        if args.memory_growth:
            tf.config.experimental.set_memory_growth(device=gpus[args.gpu_device], enable=True)

    if args.find_matches:
        required_arguments = ['domain_size', 'image_size']
        check_arguments(provided_arguments, required_arguments)
        find_matches_for_domain(args.domain_size, args.image_size)

    if args.learning_curve:
        required_arguments = ['model_dir', 'model_number']
        check_arguments(provided_arguments, required_arguments)
        learning_curve(args.model_number, args.model_dir)

    if args.generate_predictions:
        required_arguments = ['domain', 'model_number', 'model_dir', 'prediction_method', 'variables_netcdf_indir', 'fronts_netcdf_indir']
        check_arguments(provided_arguments, required_arguments)

        if args.prediction_method == 'datetime' and args.datetime is None:
            raise errors.MissingArgumentError("'datetime' argument must be passed: 'prediction_method' was set to 'datetime' ")

        model_properties = pd.read_pickle(f"{args.model_dir}/model_{args.model_number}/model_{args.model_number}_properties.pkl")
        image_size = model_properties['image_size']  # We are only concerned about longitude and latitude when checking compatibility

        domain_indices = settings.DEFAULT_DOMAIN_INDICES[args.domain]
        domain_images = settings.DEFAULT_DOMAIN_IMAGES[args.domain]

        # Verify the compatibility of image stitching arguments
        find_matches_for_domain((domain_indices[1] - domain_indices[0], domain_indices[3] - domain_indices[2]), image_size, compatibility_mode=True, compat_images=domain_images)

        generate_predictions(args.model_number, args.model_dir, args.variables_netcdf_indir, args.prediction_method,
            domain=args.domain, domain_images=args.domain_images, dataset=args.dataset, datetime=args.datetime, 
            random_variables=args.random_variables, variable_data_source=args.variable_data_source)

    if args.calculate_stats:
        required_arguments = ['model_number', 'model_dir', 'fronts_netcdf_indir', 'timestep', 'domain_images']

        if args.domain == 'full' or args.variable_data_source != 'era5':
            hours = range(0, 24, 6)
        else:
            hours = range(0, 24, 3)

        for hour in hours:
            if args.variable_data_source != 'era5':
                for forecast_hour in range(0, 12, 3):
                    timestep = (args.timestep[0], args.timestep[1], args.timestep[2], hour)
                    calculate_performance_stats(args.model_number, args.model_dir, args.fronts_netcdf_indir, timestep, args.domain,
                        args.domain_images, forecast_hour, args.variable_data_source)
            else:
                timestep = (args.timestep[0], args.timestep[1], args.timestep[2], hour)
                calculate_performance_stats(args.model_number, args.model_dir, args.fronts_netcdf_indir, timestep, args.domain,
                    args.domain_images, args.domain_trim)

    if args.calibrate_model:

        calibrate_model(args.model_number, args.model_dir, args.domain, args.domain_images, args.variable_data_source)

    if args.prediction_plot:
        required_arguments = ['model_number', 'model_dir', 'fronts_netcdf_indir', 'datetime', 'domain_images']
        prediction_plot(args.model_number, args.model_dir, args.fronts_netcdf_indir, args.datetime, args.domain,
            args.domain_images, forecast_hour=args.forecast_hour, variable_data_source=args.variable_data_source)

    if args.plot_performance_diagrams:
        required_arguments = ['model_number', 'model_dir', 'fronts_netcdf_indir', 'domain', 'domain_images']
        check_arguments(provided_arguments, required_arguments)
        plot_performance_diagrams(args.model_dir, args.model_number, args.fronts_netcdf_indir, args.domain, args.domain_images,
            random_variables=args.random_variables, variable_data_source=args.variable_data_source, bootstrap=args.bootstrap,
            num_iterations=args.num_iterations, forecast_hour=args.forecast_hour)
