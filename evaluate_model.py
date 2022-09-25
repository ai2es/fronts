"""
Functions used for evaluating a U-Net model.

Code written by: Andrew Justin (andrewjustinwx@gmail.com)

TODO:
    * Finish separating plots and statistics calculations from the main prediction method
    * Touch up prediction method for quicker predictions?

Last updated: 9/25/2022 11:51 AM CT
"""

import os
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
from matplotlib.animation import FuncAnimation
from utils import data_utils
from utils.plotting_utils import plot_background
from glob import glob

# default values for extents and images on different domains
DEFAULT_DOMAIN_EXTENTS = {'full': [130, 370, 0, 80], 'conus': [228, 299.75, 25, 56.75]}
DEFAULT_DOMAIN_IMAGES = {'full': [8, 3], 'conus': [3, 1]}


def add_image_to_map(stitched_map_probs, image_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
    image_size_lon, image_size_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing,
    lon_pixels_per_image, lat_pixels_per_image):
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
    domain_trim_lon: int
        - Number of pixels by which the images will be trimmed along the longitude dimension before taking the maximum of overlapping pixels.
    domain_trim_lat: int
        - Number of pixels by which the images will be trimmed along the latitude dimension before taking the maximum of overlapping pixels.
    lon_image_spacing: int
        - Number of pixels between each image along the longitude dimension.
    lat_image_spacing: int
        - Number of pixels between each image along the latitude dimension.
    lon_pixels_per_image: int
        - Number of pixels along the longitude dimension of each image.
    lat_pixels_per_image: int
        - Number of pixels along the latitude dimension of each image.

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
            stitched_map_probs[:, 0: image_size_lon - domain_trim_lon, 0: image_size_lat - domain_trim_lat] = \
                image_probs[:, domain_trim_lon: image_size_lon, domain_trim_lat: image_size_lat]

            if domain_images_lon == 1 and domain_images_lat == 1:
                map_created = True

        elif lat_image != domain_images_lat - 1:  # If the image is not on the northern nor the southern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant latitude
            stitched_map_probs[:, 0: image_size_lon - domain_trim_lon, int(lat_image*lat_image_spacing):int((lat_image-1)*lat_image_spacing) + lat_pixels_per_image] = \
                np.maximum(stitched_map_probs[:, 0: image_size_lon - domain_trim_lon, int(lat_image*lat_image_spacing):int((lat_image-1)*lat_image_spacing) + lat_pixels_per_image],
                           image_probs[:, domain_trim_lon: image_size_lon, domain_trim_lat:domain_trim_lat + lat_pixels_per_image - lat_image_spacing])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[:, 0: image_size_lon - domain_trim_lon, int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image:lat_image_spacing * lat_image + lat_pixels_per_image] = \
                image_probs[:, domain_trim_lon: image_size_lon, domain_trim_lat + lat_pixels_per_image - lat_image_spacing:domain_trim_lat + lat_pixels_per_image]

            if domain_images_lon == 1 and domain_images_lat == 2:
                map_created = True

        else:  # If the image is on the southern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant latitude
            stitched_map_probs[:, 0: image_size_lon - domain_trim_lon, int(lat_image*lat_image_spacing):int((lat_image-1)*lat_image_spacing) + lat_pixels_per_image] = \
                np.maximum(stitched_map_probs[:, domain_trim_lon: image_size_lon, int(lat_image*lat_image_spacing):int((lat_image-1)*lat_image_spacing) + lat_pixels_per_image],
                           image_probs[:, domain_trim_lon: image_size_lon, domain_trim_lat:domain_trim_lat + lat_pixels_per_image - lat_image_spacing])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[:, 0: image_size_lon - domain_trim_lon, int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image:] = \
                image_probs[:, domain_trim_lon: image_size_lon, domain_trim_lat + lat_pixels_per_image - lat_image_spacing:image_size_lat-domain_trim_lat]

            if domain_images_lon == 1 and domain_images_lat > 2:
                map_created = True

    elif lon_image != domain_images_lon - 1:  # If the image is not on the western nor the eastern edge of the domain
        if lat_image == 0:  # If the image is on the northern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant longitude
            stitched_map_probs[:, int(lon_image*lon_image_spacing):int((lon_image-1)*lon_image_spacing) + lon_pixels_per_image, 0: image_size_lat - domain_trim_lat] = \
                np.maximum(stitched_map_probs[:, int(lon_image*lon_image_spacing):int((lon_image-1)*lon_image_spacing) + lon_pixels_per_image, 0: image_size_lat - domain_trim_lat],
                           image_probs[:, domain_trim_lon:domain_trim_lon + lon_pixels_per_image - lon_image_spacing, domain_trim_lat: image_size_lat])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[:, int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image:lon_image_spacing * lon_image + lon_pixels_per_image, 0: image_size_lat - domain_trim_lat] = \
                image_probs[:, domain_trim_lon + lon_pixels_per_image - lon_image_spacing:domain_trim_lon + lon_pixels_per_image, domain_trim_lat: image_size_lat]

            if domain_images_lon == 2 and domain_images_lat == 1:
                map_created = True

        elif lat_image != domain_images_lat - 1:  # If the image is not on the northern nor the southern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant longitude
            stitched_map_probs[:, int(lon_image * lon_image_spacing):int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image, int(lat_image*lat_image_spacing)+domain_trim_lat:int(lat_image)*int(lat_image_spacing)+image_size_lat-domain_trim_lat] = \
                np.maximum(stitched_map_probs[:, int(lon_image * lon_image_spacing):int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image, int(lat_image*lat_image_spacing)+domain_trim_lat:int(lat_image)*int(lat_image_spacing)+image_size_lat-domain_trim_lat],
                           image_probs[:, domain_trim_lon:domain_trim_lon + lon_pixels_per_image - lon_image_spacing, domain_trim_lat:image_size_lat-domain_trim_lat])

            # Take the maximum of the overlapping pixels along sets of constant latitude
            stitched_map_probs[:, int(lon_image*lon_image_spacing):int(lon_image*lon_image_spacing)+image_size_lon-domain_trim_lon, int(lat_image * lat_image_spacing):int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image] = \
                np.maximum(stitched_map_probs[:, int(lon_image*lon_image_spacing):int(lon_image*lon_image_spacing)+image_size_lon-domain_trim_lon, int(lat_image * lat_image_spacing):int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image],
                           image_probs[:, domain_trim_lon:image_size_lon, domain_trim_lat:domain_trim_lat + lat_pixels_per_image - lat_image_spacing])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[:, int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image:lon_image_spacing * lon_image + lon_pixels_per_image, int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image:lat_image_spacing * lat_image + lat_pixels_per_image] = \
                image_probs[:, domain_trim_lon + lon_pixels_per_image - lon_image_spacing:domain_trim_lon + lon_pixels_per_image, domain_trim_lat + lat_pixels_per_image - lat_image_spacing:domain_trim_lat + lat_pixels_per_image]

            if domain_images_lon == 2 and domain_images_lat == 2:
                map_created = True

        else:  # If the image is on the southern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant longitude
            stitched_map_probs[:, int(lon_image * lon_image_spacing):int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image, int(lat_image*lat_image_spacing):] = \
                np.maximum(stitched_map_probs[:, int(lon_image * lon_image_spacing):int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image, int(lat_image*lat_image_spacing):],
                           image_probs[:, domain_trim_lon:domain_trim_lon + lon_pixels_per_image - lon_image_spacing, domain_trim_lat:image_size_lat-domain_trim_lat])

            # Take the maximum of the overlapping pixels along sets of constant latitude
            stitched_map_probs[:, int(lon_image*lon_image_spacing):int(lon_image*lon_image_spacing)+image_size_lon-domain_trim_lon, int(lat_image * lat_image_spacing):int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image] = \
                np.maximum(stitched_map_probs[:, int(lon_image*lon_image_spacing):int(lon_image*lon_image_spacing)+image_size_lon-domain_trim_lon, int(lat_image * lat_image_spacing):int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image],
                           image_probs[:, domain_trim_lon:image_size_lon, domain_trim_lat:domain_trim_lat + lat_pixels_per_image - lat_image_spacing])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[:, int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image:lon_image_spacing * lon_image + lon_pixels_per_image, int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image:] = \
                image_probs[:, domain_trim_lon + lon_pixels_per_image - lon_image_spacing:domain_trim_lon + lon_pixels_per_image, domain_trim_lat + lat_pixels_per_image - lat_image_spacing:domain_trim_lat + lat_pixels_per_image]

            if domain_images_lon == 2 and domain_images_lat > 2:
                map_created = True
    else:
        if lat_image == 0:  # If the image is on the northern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant longitude
            stitched_map_probs[:, int(lon_image * lon_image_spacing):int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image, 0: image_size_lat - domain_trim_lat] = \
                np.maximum(stitched_map_probs[:, int(lon_image * lon_image_spacing):int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image, 0: image_size_lat - domain_trim_lat],
                           image_probs[:, domain_trim_lon:domain_trim_lon + lon_pixels_per_image - lon_image_spacing, domain_trim_lat: image_size_lat])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[:, int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image:, 0: image_size_lat - domain_trim_lat] = \
                image_probs[:, domain_trim_lon + lon_pixels_per_image - lon_image_spacing:image_size_lon-domain_trim_lon, domain_trim_lat: image_size_lat]

            if domain_images_lon > 2 and domain_images_lat == 1:
                map_created = True

        elif lat_image != domain_images_lat - 1:  # If the image is not on the northern nor the southern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant longitude
            stitched_map_probs[:, int(lon_image * lon_image_spacing):int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image, int(lat_image*lat_image_spacing)+domain_trim_lat:int(lat_image)*int(lat_image_spacing)+image_size_lat-domain_trim_lat] = \
                np.maximum(stitched_map_probs[:, int(lon_image * lon_image_spacing):int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image, int(lat_image*lat_image_spacing)+domain_trim_lat:int(lat_image)*int(lat_image_spacing)+image_size_lat-domain_trim_lat], image_probs[:, domain_trim_lon:domain_trim_lon + lon_pixels_per_image - lon_image_spacing, domain_trim_lat:image_size_lat-domain_trim_lat])

            # Take the maximum of the overlapping pixels along sets of constant latitude
            stitched_map_probs[:, int(lon_image*lon_image_spacing):, int(lat_image * lat_image_spacing):int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image] = \
                np.maximum(stitched_map_probs[:, int(lon_image*lon_image_spacing):, int(lat_image * lat_image_spacing):int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image], image_probs[:, domain_trim_lon:image_size_lon-domain_trim_lon, domain_trim_lat:domain_trim_lat + lat_pixels_per_image - lat_image_spacing])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[:, int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image:, int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image:lat_image_spacing * lat_image + lat_pixels_per_image] = image_probs[:, domain_trim_lon + lon_pixels_per_image - lon_image_spacing:image_size_lon-domain_trim_lon, domain_trim_lat + lat_pixels_per_image - lat_image_spacing:domain_trim_lat + lat_pixels_per_image]

            if domain_images_lon > 2 and domain_images_lat == 2:
                map_created = True
        else:  # If the image is on the southern edge of the domain
            # Take the maximum of the overlapping pixels along sets of constant longitude
            stitched_map_probs[:, int(lon_image * lon_image_spacing):, int(lat_image*lat_image_spacing):] = \
                np.maximum(stitched_map_probs[:, int(lon_image * lon_image_spacing):, int(lat_image*lat_image_spacing):],
                           image_probs[:, domain_trim_lon:image_size_lon-domain_trim_lon, domain_trim_lat:image_size_lat-domain_trim_lat])

            # Take the maximum of the overlapping pixels along sets of constant latitude
            stitched_map_probs[:, int(lon_image*lon_image_spacing):, int(lat_image * lat_image_spacing):int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image] = \
                np.maximum(stitched_map_probs[:, int(lon_image*lon_image_spacing):, int(lat_image * lat_image_spacing):int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image],
                           image_probs[:, domain_trim_lon:image_size_lon-domain_trim_lon, domain_trim_lat:domain_trim_lat + lat_pixels_per_image - lat_image_spacing])

            # Add the remaining pixels of the current image to the map
            stitched_map_probs[:, int(lon_image_spacing*(lon_image-1)) + lon_pixels_per_image:, int(lat_image_spacing*(lat_image-1)) + lat_pixels_per_image:lat_image_spacing * lat_image + lat_pixels_per_image] = \
                image_probs[:, domain_trim_lon + lon_pixels_per_image - lon_image_spacing:image_size_lon-domain_trim_lon, domain_trim_lat + lat_pixels_per_image - lat_image_spacing:domain_trim_lat + lat_pixels_per_image]

            map_created = True

    return stitched_map_probs, map_created


def calculate_performance_stats(model_number, model_dir, fronts_netcdf_dir, timestep, domain, domain_images, domain_trim, forecast_hour=None):
    """
    """

    year, month, day, hour = timestep[0], timestep[1], timestep[2], timestep[3]

    probs_dir = f'{model_dir}/model_{model_number}/probabilities/{domain}_{domain_images[0]}x{domain_images[1]}images_{domain_trim[0]}x{domain_trim[1]}trim'

    if forecast_hour is not None:
        forecast_timestep = data_utils.add_or_subtract_hours_to_timestep('%d%02d%02d%02d' % (year, month, day, hour), num_hours=forecast_hour)
        new_year, new_month, new_day, new_hour = forecast_timestep[:4], forecast_timestep[4:6], forecast_timestep[6:8], forecast_timestep[8:]
        fronts_file = '%s/%s/%s/%s/FrontObjects_%s%s%s%s_full.nc' % (fronts_netcdf_dir, new_year, new_month, new_day, new_year, new_month, new_day, new_hour)
        probs_file = f'{probs_dir}/model_{model_number}_{year}-%02d-%02d-%02dz_{domain}_f%03d_{domain_images[0]}x{domain_images[1]}images_{domain_trim[0]}x{domain_trim[1]}trim_probabilities.nc' % (month, day, hour, forecast_hour)
    else:
        fronts_file = '%s/%d/%02d/%02d/FrontObjects_%d%02d%02d%02d_full.nc' % (fronts_netcdf_dir, year, month, day, year, month, day, hour)
        probs_file = f'{probs_dir}/model_{model_number}_{year}-%02d-%02d-%02dz_{domain}_{domain_images[0]}x{domain_images[1]}images_{domain_trim[0]}x{domain_trim[1]}trim_probabilities.nc' % (month, day, hour)

    model_properties = pd.read_pickle(f"{model_dir}/model_{model_number}/model_{model_number}_properties.pkl")
    front_types = model_properties['front_types']

    fronts_ds = xr.open_dataset(fronts_file).sel(longitude=slice(DEFAULT_DOMAIN_EXTENTS[domain][0], DEFAULT_DOMAIN_EXTENTS[domain][1]),
                                                 latitude=slice(DEFAULT_DOMAIN_EXTENTS[domain][3], DEFAULT_DOMAIN_EXTENTS[domain][2]))
    fronts_ds = data_utils.reformat_fronts(front_types, fronts_ds)
    probs_ds = xr.open_dataset(probs_file)

    if front_types == 'F_BIN' or front_types == 'MERGED-F_BIN' or front_types == 'MERGED-T':
        num_front_types = 1

    elif front_types == 'MERGED-F':
        num_front_types = 4

    elif front_types == 'MERGED-ALL':
        num_front_types = 7

    else:
        num_front_types = len(front_types)

    if type(front_types) == str:
        front_types = [front_types, ]

    if len(front_types) == 1:
        front_types_arg_reformat = front_types[0]
    else:
        front_types_arg_reformat = front_types

    bool_tn_fn_dss = dict({front: xr.where(fronts_ds == front_no + 1, 1, 0)['identifier'] for front_no, front in enumerate(front_types)})
    bool_tp_fp_dss = dict({front: None for front in front_types})
    probs_dss = dict({front: probs_ds[front] for front in front_types})

    tp_array = np.zeros(shape=[num_front_types, 5, 100])
    fp_array = np.zeros(shape=[num_front_types, 5, 100])
    tn_array = np.zeros(shape=[num_front_types, 5, 100])
    fn_array = np.zeros(shape=[num_front_types, 5, 100])

    thresholds = np.linspace(0.01, 1, 100)  # Probability thresholds for calculating performance statistics
    boundaries = np.array([50, 100, 150, 200, 250])  # Boundaries for checking whether or not a front is present (kilometers)

    for front_no, front_type in enumerate(front_types):
        for i in range(100):
            """
            True negative ==> model correctly predicts the lack of a front at a given point
            False negative ==> model does not predict a front, but a front exists
            """
            tn_array[front_no, :, i] = len(np.where((probs_dss[front_type] < thresholds[i]) & (bool_tn_fn_dss[front_type] == 0))[0])
            fn_array[front_no, :, i] = len(np.where((probs_dss[front_type] < thresholds[i]) & (bool_tn_fn_dss[front_type] == 1))[0])

        """ Calculate true positives and false positives """
        for boundary in range(5):
            new_fronts_ds = xr.open_dataset(fronts_file).sel(longitude=slice(DEFAULT_DOMAIN_EXTENTS[domain][0], DEFAULT_DOMAIN_EXTENTS[domain][1]),
                                                          latitude=slice(DEFAULT_DOMAIN_EXTENTS[domain][3], DEFAULT_DOMAIN_EXTENTS[domain][2]))
            new_fronts = data_utils.reformat_fronts(front_types_arg_reformat, fronts_ds=new_fronts_ds)
            front_identifier = data_utils.expand_fronts(new_fronts, iterations=int(2*(boundary+1)))  # Expand fronts
            bool_tp_fp_dss[front_type] = xr.where(front_identifier == front_no + 1, 1, 0)['identifier']  # 1 = cold front, 0 = not a cold front
            for i in range(100):
                """
                True positive ==> model correctly identifies a front
                False positive ==> model predicts a front, but it does not exist
                """
                tp_array[front_no, boundary, i] = len(np.where((probs_dss[front_type] > thresholds[i]) & (bool_tp_fp_dss[front_type] == 1))[0])
                fp_array[front_no, boundary, i] = len(np.where((probs_dss[front_type] > thresholds[i]) & (bool_tp_fp_dss[front_type] == 0))[0])

        if front_no == 0:
            performance_ds = xr.Dataset({"tp_%s" % front_type: (["boundary", "threshold"], tp_array[front_no]),
                                         "fp_%s" % front_type: (["boundary", "threshold"], fp_array[front_no]),
                                         "tn_%s" % front_type: (["boundary", "threshold"], tn_array[front_no]),
                                         "fn_%s" % front_type: (["boundary", "threshold"], fn_array[front_no])},
                                        coords={"boundary": boundaries, "threshold": thresholds})
        else:
            performance_ds["tp_%s" % front_type] = (('boundary', 'threshold'), tp_array[front_no])
            performance_ds["fp_%s" % front_type] = (('boundary', 'threshold'), fp_array[front_no])
            performance_ds["tn_%s" % front_type] = (('boundary', 'threshold'), tn_array[front_no])
            performance_ds["fn_%s" % front_type] = (('boundary', 'threshold'), fn_array[front_no])

    performance_ds = performance_ds.expand_dims({'time': np.atleast_1d(probs_ds['time'].values)})
    if forecast_hour is not None:
        performance_ds = performance_ds.expand_dims({'forecast_hour': np.atleast_1d(probs_ds['forecast_hour'].values)})

    performance_ds.to_netcdf(path=probs_file.replace('probabilities', 'statistics'), mode='w', engine='scipy')


def find_matches_for_domain(domain_size, image_size, compatibility_mode=False, compat_images=None):
    """
    Function that outputs the number of images that can be stitched together with the specified domain length and the length
    of the domain dimension output by the model. This is also used to determine the compatibility of declared image and
    trim parameters for model predictions.

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


def gdas_prediction_animated(model_number, model_dir, fronts_netcdf_indir, domain, domain_images, domain_trim, timestep, forecast_hours='all',
    probability_mask_2D=0.05, probability_mask_3D=0.10):
    """

    """

    year, month, day, hour = timestep[0], timestep[1], timestep[2], timestep[3]

    if domain == 'conus':
        extent = np.array([220, 300, 25, 52])  # Extent for CONUS
    else:
        extent = np.array([120, 380, 0, 80])  # Extent for full domain
    crs = ccrs.LambertConformal(central_longitude=250)

    model_properties = pd.read_pickle(f"{model_dir}/model_{model_number}/model_{model_number}_properties.pkl")
    path_to_probs_folder = f'{model_dir}/model_{model_number}/probabilities/{domain}_{domain_images[0]}x{domain_images[1]}images_{domain_trim[0]}x{domain_trim[1]}trim/'
    probs_ds_filenames = f'model_{model_number}_{year}-%02d-%02d-%02dz_{domain}_f*_{domain_images[0]}x{domain_images[1]}images_{domain_trim[0]}x{domain_trim[1]}trim_probabilities.nc' % (month, day, hour)
    probs_ds = xr.open_mfdataset(path_to_probs_folder + probs_ds_filenames).isel(time=0)
    fronts_ds = xr.open_dataset(f'{fronts_netcdf_indir}\\{year}\\%02d\\%02d\\FrontObjects_{year}%02d%02d%02d_{domain}.nc' % (month, day, month, day, hour))

    # Model properties
    image_size = model_properties['input_size'][:-1]  # The image size does not include the last dimension of the input size as it only represents the number of channels
    front_types = model_properties['front_types']
    num_dimensions = len(image_size)

    if num_dimensions == 2:
        probability_mask = probability_mask_2D
        vmax, cbar_tick_adjust, cbar_label_adjust, n_colors = 0.55, 0.025, 20, 11
        levels = np.arange(0, 0.6, 0.05)
        cbar_ticks = np.arange(probability_mask, 0.6, 0.05)
        cbar_tick_labels = [None, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    else:
        probability_mask = probability_mask_3D
        vmax, cbar_tick_adjust, cbar_label_adjust, n_colors = 1, 0.05, 10, 11
        levels = np.arange(0, 1.1, 0.1)
        cbar_ticks = np.arange(probability_mask, 1.1, 0.1)
        cbar_tick_labels = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    fronts, names, labels, colors_types, colors_probs = data_utils.reformat_fronts(fronts_ds, front_types, return_names=True, return_colors=True)
    fronts = data_utils.expand_fronts(fronts)
    fronts = xr.where(fronts == 0, float('NaN'), fronts)

    probs_ds = xr.where(probs_ds > probability_mask, probs_ds, float("NaN"))

    cmap_front = colors.ListedColormap(colors_types, name='from_list', N=len(names))
    norm_front = colors.Normalize(vmin=1, vmax=len(names) + 1)

    fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=300, subplot_kw={'projection': crs})

    for front_no, front_key, front_name, front_label, cmap in zip(range(1, len(names) + 1), list(probs_ds.keys()), names, labels, colors_probs):

        def gdas_forecast_animation(i):
            ax.cla()
            plot_background(extent, ax=ax, linewidth=0.5)
            cmap_probs, norm_probs = cm.get_cmap(cmap, n_colors), colors.Normalize(vmin=0, vmax=vmax)
            probs_ds[front_key].sel(forecast_hour=i).plot.contourf(ax=ax, x='longitude', y='latitude', norm=norm_probs, levels=levels, cmap=cmap_probs,
                transform=ccrs.PlateCarree(), alpha=0.75, add_colorbar=False)
            valid_time = data_utils.add_or_subtract_hours_to_timestep(f'{year}%02d%02d%02d' % (month, day, hour), num_hours=i)
            ax.set_title(f'Run: GDAS {year}-%02d-%02d-%02dz F%03d \nPredictions valid: {valid_time[:4]}-{valid_time[4:6]}-{valid_time[6:8]}-{valid_time[8:]}z' % (month, day, hour, i), loc='left')
            ax.set_title(f'{front_name} predictions')
            cbar_ax = fig.add_axes([0.8365, 0.11, 0.015, 0.77])
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm_probs, cmap=cmap_probs), cax=cbar_ax, boundaries=levels[1:], alpha=0.75)
            cbar.set_label('Probability', rotation=90)
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_tick_labels[int(probability_mask*cbar_label_adjust):])

        ani = FuncAnimation(fig, gdas_forecast_animation, frames=10, interval=500, repeat=False)
        ani.save(f'{path_to_probs_folder}test_animation_{front_key}.mp4', writer='ffmpeg', fps=5)


def generate_predictions(model_number, model_dir, variables_netcdf_indir, fronts_netcdf_indir, prediction_method, domain='full',
    domain_images='min', domain_trim=(0, 0), dataset=None, datetime=None, num_rand_predictions=None, random_variables=None,
    variable_data_source='gdas', forecast_hours=None):
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
    fronts_netcdf_indir: str
        - Input directory for the front object netcdf files.
    domain: str
        - Domain of the datasets.
    domain_images: str or iterable object with 2 ints
        - If a string, valid values are 'min', 'balanced', 'max'. Default is 'min'.
            * 'min' uses the minimum possible number of images when making the predictions.
            * 'balanced' chooses the number of images that creates the closest possible balance of images in the longitude and latitude direction.
            * 'max' chooses the maximum number of images that can be stitched together (NOT RECOMMENDED)
        - If an iterable with 2 integers, values represent the number of images in the longitude and latitude dimensions.
    domain_trim: None or iterable object with 2 ints
        - Number of pixels to trim each image by along each dimension before taking the maximum of the overlapping pixels (lon lat).
    prediction_method: str
        - Prediction method. Options are: 'datetime', 'random', 'all'
    domain: str
        - Domain over which the predictions will be made. Options are: 'conus', 'full'. Default is 'full'.
    datetime: iterable object with 4 integers
        - 4 values for the date and time: year, month, day, hour
    dataset: str
        - Dataset for which to make predictions if prediction_method is 'random' or 'all'.
    num_rand_predictions: int
        - Number of random predictions to make.
    random_variables: str, or iterable of strings
        - Variables to randomize.
    variable_data_source: str
        - Variable data to use for training the model. Options are: 'era5', 'gdas', or 'gfs' (case-insensitive)
    forecast_hours: int or tuple of ints
        - Forecast hours to make predictions with if making predictions with GDAS data.
    """

    ### Model properties ###
    model_properties = pd.read_pickle(f"{model_dir}/model_{model_number}/model_{model_number}_properties.pkl")
    image_size = model_properties['input_size'][:-1]  # The image size does not include the last dimension of the input size as it only represents the number of channels
    front_types = model_properties['front_types']
    classes = model_properties['classes']
    num_dimensions = len(image_size)
    test_years, valid_years = model_properties['test_years'], model_properties['validation_years']

    domain_images = DEFAULT_DOMAIN_IMAGES[domain]
    domain_extent = DEFAULT_DOMAIN_EXTENTS[domain]

    ### Properties of the final map made from stitched images ###
    domain_images_lon, domain_images_lat = domain_images[0], domain_images[1]
    domain_size_lon, domain_size_lat = int((domain_extent[1] - domain_extent[0])*4) + 1, int((domain_extent[3] - domain_extent[2])*4) + 1
    domain_trim_lon, domain_trim_lat = domain_trim[0], domain_trim[1]
    image_size_lon, image_size_lat = image_size[0], image_size[1]  # Dimensions of the model's predictions
    domain_size_lon_trimmed = domain_size_lon - 2*domain_trim_lon  # Longitude dimension of the full stitched map after trimming
    domain_size_lat_trimmed = domain_size_lat - 2*domain_trim_lat  # Latitude dimension of the full stitched map after trimming
    lon_pixels_per_image = int(image_size_lon - 2*domain_trim_lon)  # Longitude dimension of each image after trimming
    lat_pixels_per_image = int(image_size_lat - 2*domain_trim_lat)  # Latitude dimension of each image after trimming

    if domain_images_lon > 1:
        lon_image_spacing = int((domain_size_lon - image_size_lon)/(domain_images_lon-1))
    else:
        lon_image_spacing = 0

    if domain_images_lat > 1:
        lat_image_spacing = int((domain_size_lat - image_size_lat)/(domain_images_lat-1))
    else:
        lat_image_spacing = 0

    model = fm.load_model(model_number, model_dir)

    if variable_data_source.lower() == 'era5':

        era5_files_obj = fm.ERA5files(variables_netcdf_indir)
        era5_files_obj.variables = ['T', 'Td', 'sp_z', 'u', 'v', 'theta_w', 'r', 'RH', 'Tv', 'Tw', 'theta_e', 'q']
        era5_files_obj.validation_years = valid_years
        era5_files_obj.test_years = test_years
        era5_files_obj.sort_by_timestep()

        if dataset is not None:
            variable_files = getattr(era5_files_obj, 'era5_files_' + dataset)
        else:
            variable_files = era5_files_obj.era5_files

    elif variable_data_source.lower() == 'gdas':

        gdas_files_obj = fm.GDASfiles(variables_netcdf_indir)
        gdas_files_obj.variables = ['T', 'Td', 'sp_z', 'u', 'v', 'theta_w', 'r', 'RH', 'Tv', 'Tw', 'theta_e', 'q']
        gdas_files_obj.validation_years = valid_years
        gdas_files_obj.test_years = test_years
        gdas_files_obj.forecast_hours = (0, 3, 6, 9)
        gdas_files_obj.pair_with_fronts(fronts_netcdf_indir)

        if dataset is not None:
            variable_files = getattr(gdas_files_obj, 'gdas_files_' + dataset)
        else:
            variable_files = gdas_files_obj.gdas_files

    if prediction_method == 'datetime':
        timestep_str = '%d%02d%02d%02d' % (datetime[0], datetime[1], datetime[2], datetime[3])
        datetime_index = [index for index, files_for_timestep in enumerate(variable_files) if all(timestep_str in file for file in files_for_timestep)][0]
        variable_files = [variable_files[datetime_index], ]

    subdir_base = '%s_%dx%dimages_%dx%dtrim' % (domain, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1])
    if random_variables is not None:
        subdir_base += '_' + '-'.join(sorted(random_variables))

    for file_index in range(len(variable_files)):
        """
        Create array that contains all probabilities across the whole domain map for every front type that the model is 
        predicting. The first dimension is 1 unit less than the number of classes because we are not concerned about the 
        'no front' front type in the model.
        """

        if variable_data_source == 'era5':
            variable_ds = xr.open_mfdataset(variable_files[file_index]).transpose('time', 'longitude', 'latitude', 'pressure_level').sel(longitude=slice(domain_extent[0], domain_extent[1]), latitude=slice(domain_extent[3], domain_extent[2]))
            stitched_map_probs = np.empty(shape=[classes-1, domain_size_lon_trimmed, domain_size_lat_trimmed])
        elif variable_data_source == 'gdas':
            variable_ds = xr.open_mfdataset(variable_files[file_index]).transpose('time', 'forecast_hour', 'longitude', 'latitude', 'pressure_level').sel(pressure_level=['surface', '1000', '950', '900', '850'], longitude=slice(domain_extent[0], domain_extent[1]), latitude=slice(domain_extent[3], domain_extent[2]))
            forecast_hours = variable_ds['forecast_hour'].values
            stitched_map_probs = np.empty(shape=[len(forecast_hours), classes-1, domain_size_lon_trimmed, domain_size_lat_trimmed])

        # Latitude and longitude points in the domain
        image_lats = variable_ds.latitude.values[domain_trim_lat:domain_size_lat-domain_trim_lat]
        image_lons = variable_ds.longitude.values[domain_trim_lon:domain_size_lon-domain_trim_lon]

        timestep = str(variable_ds['time'].values[0])
        variable_ds = data_utils.normalize_variables(variable_ds)

        # Randomize variable
        if random_variables is not None:
            variable_ds = data_utils.randomize_variables(variable_ds, random_variables)

        time = f'{timestep[:4]}-%s-%s-%sz' % (timestep[5:7], timestep[8:10], timestep[11:13])

        map_created = False  # Boolean that determines whether or not the final stitched map has been created

        for lat_image in range(domain_images_lat):
            lat_index = int(lat_image*lat_image_spacing)
            for lon_image in range(domain_images_lon):
                print("%s....%d/%d" % (time, int(lat_image*domain_images_lon)+lon_image, int(domain_images_lon*domain_images_lat)))
                lon_index = int(lon_image*lon_image_spacing)

                lons = variable_ds.longitude.values[lon_index:lon_index + image_size[0]]  # Longitude points for the current image
                lats = variable_ds.latitude.values[lat_index:lat_index + image_size[1]]  # Latitude points for the current image

                variable_ds_new = variable_ds[['T', 'Td', 'sp_z', 'u', 'v', 'theta_w', 'r', 'RH', 'Tv', 'Tw', 'theta_e', 'q']].sel(longitude=lons, latitude=lats).to_array().values

                if num_dimensions == 3:
                    if variable_data_source == 'era5':
                        variable_ds_new = variable_ds_new.transpose([1, 2, 3, 4, 0])
                    elif variable_data_source == 'gdas':
                        variable_ds_new = variable_ds_new.transpose([1, 2, 3, 4, 5, 0])[0]
                        forecast_hours = variable_ds['forecast_hour'].values
                else:
                    raise ValueError("Invalid number of dimensions: %d" % num_dimensions)

                prediction = model.predict(variable_ds_new)

                if num_dimensions == 2:
                    if variable_data_source == 'era5':
                        image_probs = np.transpose(prediction[0][0][:, :, 1:], (2, 0, 1))  # Final index is front+1 because we skip over the 'no front' type
                else:
                    if variable_data_source == 'era5':
                        image_probs = np.transpose(np.amax(prediction[0][0][:, :, :, 1:], axis=2), (2, 0, 1))  # Final index is front+1 because we skip over the 'no front' type
                    elif variable_data_source == 'gdas':
                        image_probs = np.transpose(np.amax(prediction[0][:, :, :, :, 1:], axis=3), (0, 3, 1, 2))  # Final index is front+1 because we skip over the 'no front' type

                # Add predictions to the map
                if variable_data_source == 'gdas':
                    for fcst_hr_index in range(len(forecast_hours)):
                        stitched_map_probs[fcst_hr_index], map_created = add_image_to_map(stitched_map_probs[fcst_hr_index], image_probs[fcst_hr_index], map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                            image_size_lon, image_size_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)
                else:
                    stitched_map_probs, map_created = add_image_to_map(stitched_map_probs, image_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        image_size_lon, image_size_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)

                if map_created is True:
                    print("%s....%d/%d" % (time, int(domain_images_lon*domain_images_lat), int(domain_images_lon*domain_images_lat)))

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

                    if variable_data_source == 'gdas':

                        for fcst_hr_index, forecast_hour in enumerate(forecast_hours):
                            probs_ds = create_model_prediction_dataset(stitched_map_probs[fcst_hr_index], image_lats, image_lons, front_types)
                            probs_ds = probs_ds.expand_dims({'time': np.atleast_1d(timestep), 'forecast_hour': np.atleast_1d(forecast_hours[fcst_hr_index])})
                            filename_base = 'model_%d_%s_%s_f%03d_%dx%dimages_%dx%dtrim' % (model_number, time, domain, forecast_hours[fcst_hr_index], domain_images_lon, domain_images_lat, domain_trim_lon, domain_trim_lat)
                            if random_variables is not None:
                                filename_base += '_' + '-'.join(sorted(random_variables))

                            outfile = '%s/model_%d/probabilities/%s/%s_probabilities.nc' % (model_dir, model_number, subdir_base, filename_base)
                            probs_ds.to_netcdf(path=outfile, engine='scipy', mode='w')

                    elif variable_data_source == 'era5':
                        probs_ds = create_model_prediction_dataset(stitched_map_probs, image_lats, image_lons, front_types)
                        probs_ds = probs_ds.expand_dims({'time': np.atleast_1d(timestep)})
                        filename_base = 'model_%d_%s_%s_%dx%dimages_%dx%dtrim' % (model_number, time, domain, domain_images_lon, domain_images_lat, domain_trim_lon, domain_trim_lat)
                        if random_variables is not None:
                            filename_base += '_' + '-'.join(sorted(random_variables))

                        outfile = '%s/model_%d/probabilities/%s/%s_probabilities.nc' % (model_dir, model_number, subdir_base, filename_base)
                        probs_ds.to_netcdf(path=outfile, engine='scipy', mode='w')


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


def plot_performance_diagrams(model_dir, model_number, domain, domain_images, domain_trim, bootstrap=True, random_variables=None,
    calibrated=False, num_iterations=10000):
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
    domain_trim: iterable object with 2 ints
        - Number of pixels to trim each image by along each dimension before taking the maximum of the overlapping pixels (lon lat).
    bootstrap: bool
        - Setting this to true will plot confidence intervals onto the performance diagrams.
    random_variables: str or list of strs
        - Variable(s) that were randomized when performance statistics were calculated.
    calibrated: bool
        - Indicates whether or not the statistics to be plotted are from a calibrated model.
    num_iterations: int
        - Number of iterations when bootstrapping the data.
    """

    model_properties = pd.read_pickle(f"{model_dir}\\model_{model_number}\\model_{model_number}_properties.pkl")
    front_types = model_properties['front_types']
    num_front_types = model_properties['classes'] - 1  # model_properties['classes'] - 1 ===> number of front types (we ignore the 'no front' type)

    if num_front_types > 1:
        _, labels = data_utils.reformat_fronts(front_types, return_names=True)
    else:
        labels = [front_types, ]

    subdir_base = '%s_%dx%dimages_%dx%dtrim' % (domain, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1])
    stats_plot_base = 'model_%d_%s_%dx%dimages_%dx%dtrim' % (model_number, domain, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1])
    if random_variables is not None:
        subdir_base += '_' + '-'.join(sorted(random_variables))

    files = sorted(glob('%s\\model_%d\\statistics\\%s\\*statistics.nc' % (model_dir, model_number, subdir_base)))
    num_files = len(files)

    # If evaluating over the full domain, remove non-synoptic hours (3z, 9z, 15z, 21z)
    if domain == 'full':
        hours_to_remove = [3, 9, 15, 21]
        for hour in hours_to_remove:
            string = '%02dz_' % hour
            files = list(filter(lambda hour: string not in hour, files))

    stats_ds = xr.open_mfdataset(files, combine='nested', concat_dim='time')  # Open the first dataset so information on variables can be retrieved

    POD_array = np.empty([num_front_types, num_iterations, 5, 100])  # Probability of detection
    SR_array = np.empty([num_front_types, num_iterations, 5, 100])  # Success ratio

    """
    Confidence Interval (CI) for POD and SR

    Shape of arrays: (number of front types, number of boundaries, number of thresholds)
    """
    CI_lower_POD = np.empty([num_front_types, 5, 100])
    CI_lower_SR = np.empty([num_front_types, 5, 100])
    CI_upper_POD = np.empty([num_front_types, 5, 100])
    CI_upper_SR = np.empty([num_front_types, 5, 100])

    selectable_indices = range(num_files)

    front_names, _ = data_utils.reformat_fronts(front_types, return_names=True)

    for front_no, front_label in enumerate(labels):

        true_positives = stats_ds[f'tp_{front_label}'].values
        false_positives = stats_ds[f'fp_{front_label}'].values
        false_negatives = stats_ds[f'fn_{front_label}'].values
        thresholds = stats_ds['threshold'].values

        true_positives_sum = np.sum(true_positives, axis=0)
        false_positives_sum = np.sum(false_positives, axis=0)
        false_negatives_sum = np.sum(false_negatives, axis=0)

        true_positives_diff = np.abs(np.diff(true_positives_sum))
        false_positives_diff = np.abs(np.diff(false_positives_sum))
        observed_relative_frequency = np.divide(true_positives_diff, true_positives_diff + false_positives_diff)

        pod = np.divide(true_positives_sum, true_positives_sum + false_negatives_sum)
        sr = np.divide(true_positives_sum, true_positives_sum + false_positives_sum)

        if bootstrap is True:

            for iteration in range(num_iterations):
                print(f"Iteration {iteration}/{num_iterations}", end='\r')
                indices = random.choices(selectable_indices, k=num_files)  # Select a sample equal to the total number of files

                POD_array[front_no, iteration, :, :] = np.nan_to_num(np.divide(np.sum(true_positives[indices, :, :], axis=0),
                                                                        np.add(np.sum(true_positives[indices, :, :], axis=0),
                                                                               np.sum(false_negatives[indices, :, :], axis=0))))
                SR_array[front_no, iteration, :, :] = np.nan_to_num(np.divide(np.sum(true_positives[indices, :, :], axis=0),
                                                                       np.add(np.sum(true_positives[indices, :, :], axis=0),
                                                                              np.sum(false_positives[indices, :, :], axis=0))))

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

        # Code for performance diagram matrices sourced from Ryan Lagerquist's (lagerqui@ualberta.ca) thunderhoser repository:
        # https://github.com/thunderhoser/GewitterGefahr/blob/master/gewittergefahr/plotting/model_eval_plotting.py
        success_ratio_matrix, pod_matrix = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
        x, y = np.meshgrid(success_ratio_matrix, pod_matrix)
        csi_matrix = (x ** -1 + y ** -1 - 1.) ** -1
        fb_matrix = y * (x ** -1)
        CSI_LEVELS = np.linspace(0, 1, 11)
        FB_LEVELS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3]
        cmap = 'Blues'
        axis_ticks = np.arange(0, 1.1, 0.1)

        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        axarr = axs.flatten()
        cs = axarr[0].contour(x, y, fb_matrix, FB_LEVELS, colors='black', linewidths=0.5, linestyles='--')
        axarr[0].clabel(cs, FB_LEVELS, fontsize=8)  # Plot FB contours
        csi_contour = axarr[0].contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)  # Plot CSI contours in 0.1 increments
        cbar = fig.colorbar(csi_contour, ax=axarr[0], pad=0.02, label='Critical Success Index (CSI)')
        cbar.set_ticks(axis_ticks)

        cell_text = []  # List of strings that will be used in the table near the bottom of this function

        # CSI lines for each boundary
        boundary_colors = ['red', 'purple', 'brown', 'darkorange', 'darkgreen']
        max_CSI_scores_by_boundary = np.empty(shape=(5,))
        for boundary, color in enumerate(boundary_colors):
            csi = np.power((1/sr[boundary]) + (1/pod[boundary]) - 1, -1)
            max_CSI_scores_by_boundary[boundary] = np.nanmax(csi)
            max_CSI_index = np.where(csi == max_CSI_scores_by_boundary[boundary])[0]
            max_CSI_threshold = thresholds[max_CSI_index][0]
            max_CSI_pod = pod[boundary][max_CSI_index][0]  # POD where CSI is maximized
            max_CSI_sr = sr[boundary][max_CSI_index][0]  # SR where CSI is maximized
            max_CSI_fb = max_CSI_pod / max_CSI_sr

            cell_text.append(['%.2f' % max_CSI_threshold, '%.2f' % max_CSI_scores_by_boundary[boundary], '%.2f' % max_CSI_pod, '%.2f' % max_CSI_sr, '%.2f' % (1 - max_CSI_sr), '%.2f' % max_CSI_fb])

            axarr[0].plot(max_CSI_sr, max_CSI_pod, color=color, marker='*', markersize=10)
            axarr[0].plot(sr[boundary], pod[boundary], color=color, linewidth=1)
            axarr[1].plot(thresholds[1:] + 0.005, observed_relative_frequency[boundary], color=color, linewidth=1)
            axarr[1].plot(thresholds, thresholds, color='black', linestyle='--', linewidth=0.5, label='Perfect Reliability')

            if bootstrap:
                xs = np.concatenate([CI_lower_SR[front_no, boundary, :CI_zero_index], CI_upper_SR[front_no, boundary, :CI_zero_index][::-1]])
                ys = np.concatenate([CI_lower_POD[front_no, boundary, :CI_zero_index], CI_upper_POD[front_no, boundary, :CI_zero_index][::-1]])
                axarr[0].fill(xs, ys, alpha=0.3, color=color)

        axarr[0].set_xlabel("Success Ratio (SR = 1 - FAR)")
        axarr[0].set_ylabel("Probability of Detection (POD)")
        if calibrated:
            axarr[1].set_xlabel("Forecast Probability (calibrated)")
        else:
            axarr[1].set_xlabel("Forecast Probability (uncalibrated)")
        axarr[1].set_ylabel("Observed Relative Frequency")

        for ax in axarr:
            ax.set_xticks(axis_ticks)
            ax.set_yticks(axis_ticks)
            ax.grid(color='black', alpha=0.1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        ### Create table of statistics ###

        columns = ['Threshold*', 'CSI', 'POD', 'SR', 'FAR', 'FB']
        rows = ['50 km', '100 km', '150 km', '200 km', '250 km']

        table_axis = plt.axes([0.063, 0, 0.4, 0.2])
        table_axis.axis('off')
        stats_table = table_axis.table(cellText=cell_text, rowLabels=rows, rowColours=boundary_colors, colLabels=columns, cellLoc='center')
        stats_table.set_fontsize(14)
        stats_table.scale(1, 2)

        plt.text(0.16, -1.85, '* probability threshold where CSI is maximized')

        ### Text for the plot ###

        fontdict = {'fontsize': 14}

        model_text = f"Model type: {len(model_properties['input_size']) - 1}D "
        if model_properties['model_type'] == 'unet_3plus':
            model_text += 'U-Net 3+'
        plt.text(1.2, -0.25, model_text, fontdict=fontdict)

        kernel_text = 'Kernel size: %s' % model_properties['kernel_size']
        for dim in range(len(model_properties['input_size']) - 2):
            kernel_text += 'x%s' % model_properties['kernel_size']

        front_text = 'Front type: '
        if front_types == 'F_BIN':
            front_text += 'Binary (front / no front)**'
            plt.text(1.2, -1.55, '** binary: cold, warm, stationary, and occluded fronts all treated as one type')
        elif type(front_types) == list:
            front_text += front_names[front_no]

        domain_text = 'Domain: '
        if domain == 'conus':
            domain_text += domain.upper() + f' ({int(domain_images[0] * domain_images[1])} images per stitched map)'

        plt.text(1.2, -0.5, kernel_text, fontdict=fontdict)
        plt.text(1.2, -0.75, front_text, fontdict=fontdict)
        plt.text(1.2, -1, domain_text, fontdict=fontdict)

        for cell in stats_table._cells:
            stats_table._cells[cell].set_alpha(.7)

        plt.tight_layout()
        plt.savefig("%s/model_%d/%s_performance_%s.png" % (model_dir, model_number, stats_plot_base, front_label), bbox_inches='tight')
        plt.close()


def prediction_plot(model_number, model_dir, fronts_netcdf_dir, timestep, domain, domain_images, domain_trim, forecast_hour=None, probability_mask_2D=0.05, probability_mask_3D=0.10):
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
    probability_mask_2D: float
        - Mask for front probabilities with 2D models. Any probabilities smaller than this number will not be plotted.
        - Must be a multiple of 0.05, greater than 0, and no greater than 0.45.
    probability_mask_3D: float
        - Mask for front probabilities with 3D models. Any probabilities smaller than this number will not be plotted.
        - Must be a multiple of 0.1, greater than 0, and no greater than 0.9.
    """

    extent = DEFAULT_DOMAIN_EXTENTS[domain]

    year, month, day, hour = int(timestep[0]), int(timestep[1]), int(timestep[2]), int(timestep[3])

    subdir_base = '%s_%dx%dimages_%dx%dtrim' % (domain, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1])
    probs_dir = f'{model_dir}/model_{model_number}/probabilities/{subdir_base}'

    if forecast_hour is not None:
        forecast_timestep = data_utils.add_or_subtract_hours_to_timestep('%d%02d%02d%02d' % (year, month, day, hour), num_hours=forecast_hour)
        new_year, new_month, new_day, new_hour = forecast_timestep[:4], forecast_timestep[4:6], forecast_timestep[6:8], int(forecast_timestep[8:]) - (int(forecast_timestep[8:]) % 3)
        fronts_file = '%s/%s/%s/%s/FrontObjects_%s%s%s%02d_full.nc' % (fronts_netcdf_dir, new_year, new_month, new_day, new_year, new_month, new_day, new_hour)
        filename_base = f'model_%d_{year}-%02d-%02d-%02dz_%s_f%03d_%dx%dimages_%dx%dtrim' % (model_number, month, day, hour, domain, forecast_hour, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1])
        variable_data_source = 'gdas'
    else:
        fronts_file = '%s/%d/%02d/%02d/FrontObjects_%d%02d%02d%02d_full.nc' % (fronts_netcdf_dir, year, month, day, year, month, day, hour)
        filename_base = f'model_%d_{year}-%02d-%02d-%02dz_%s_%dx%dimages_%dx%dtrim' % (model_number, month, day, hour, domain, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1])
        variable_data_source = 'era5'

    probs_file = f'{probs_dir}/{filename_base}_probabilities.nc'

    print(fronts_file)
    fronts = xr.open_mfdataset(fronts_file).sel(longitude=slice(extent[0], extent[1]), latitude=slice(extent[3], extent[2]))
    probs_ds = xr.open_mfdataset(probs_file)

    crs = ccrs.LambertConformal(central_longitude=250)

    model_properties = pd.read_pickle(f"{model_dir}/model_{model_number}/model_{model_number}_properties.pkl")

    # Model properties
    image_size = model_properties['input_size'][:-1]  # The image size does not include the last dimension of the input size as it only represents the number of channels
    front_types = model_properties['front_types']
    num_dimensions = len(image_size)

    if num_dimensions == 2:
        probability_mask = probability_mask_2D
        vmax, cbar_tick_adjust, cbar_label_adjust, n_colors = 0.55, 0.025, 20, 11
        levels = np.arange(0, 0.6, 0.05)
        cbar_ticks = np.arange(probability_mask, 0.6, 0.05)
        cbar_tick_labels = [None, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    else:
        probability_mask = probability_mask_3D
        vmax, cbar_tick_adjust, cbar_label_adjust, n_colors = 1, 0.05, 10, 11
        levels = np.arange(0, 1.1, 0.1)
        cbar_ticks = np.arange(probability_mask, 1.1, 0.1)
        cbar_tick_labels = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    probs_ds = xr.where(probs_ds > probability_mask, probs_ds, float("NaN")).isel(time=0)

    fronts, names, labels, colors_types, colors_probs = data_utils.reformat_fronts(fronts, front_types, return_names=True, return_colors=True)
    fronts = data_utils.expand_fronts(fronts)
    fronts = xr.where(fronts == 0, float('NaN'), fronts)

    cmap_front = colors.ListedColormap(colors_types, name='from_list', N=len(names))
    norm_front = colors.Normalize(vmin=1, vmax=len(names) + 1)

    for front_no, front_key, front_name, front_label, cmap in zip(range(1, len(names) + 1), list(probs_ds.keys()), names, labels, colors_probs):
        if variable_data_source == 'gdas':
            fig, ax = plt.subplots(1, 1, figsize=(20, 8), subplot_kw={'projection': crs})
            plot_background(extent, ax=ax, linewidth=0.5)
            current_fronts = fronts
            current_fronts = xr.where(current_fronts != front_no, float("NaN"), front_no)
            cmap_probs, norm_probs = cm.get_cmap(cmap, n_colors), colors.Normalize(vmin=0, vmax=vmax)
            probs_ds[front_key].sel(forecast_hour=forecast_hour).plot.contourf(ax=ax, x='longitude', y='latitude', norm=norm_probs, levels=levels, cmap=cmap_probs,
                transform=ccrs.PlateCarree(), alpha=0.75, add_colorbar=False)
            current_fronts['identifier'].plot(ax=ax, x='longitude', y='latitude', cmap=cmap_front, norm=norm_front, transform=ccrs.PlateCarree(), add_colorbar=False)
            valid_time = data_utils.add_or_subtract_hours_to_timestep(f'{year}%02d%02d%02d' % (month, day, hour), num_hours=forecast_hour)
            ax.set_title(f'{front_name} predictions and ground truth')
            ax.set_title(f'Run: GDAS {year}-%02d-%02d-%02dz F%03d \nPredictions valid: {valid_time[:4]}-{valid_time[4:6]}-{valid_time[6:8]}-{valid_time[8:]}z' % (month, day, hour, forecast_hour), loc='left')
            ax.set_title(f'Fronts valid: {new_year}-{"%02d" % int(new_month)}-{"%02d" % int(new_day)}-{"%02d" % new_hour}z', loc='right')
            cbar_ax = fig.add_axes([0.8365, 0.11, 0.015, 0.77])
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm_probs, cmap=cmap_probs), cax=cbar_ax, boundaries=levels[1:], alpha=0.75)
            cbar.set_label('Probability', rotation=90)
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_tick_labels[int(probability_mask*cbar_label_adjust):])
            plt.savefig('%s/model_%d/maps/%s/%s-%s.png' % (model_dir, model_number, subdir_base, filename_base, front_label), bbox_inches='tight', dpi=300)
            plt.close()
        else:
            current_fronts = fronts
            current_fronts = xr.where(current_fronts != front_no, float("NaN"), front_no)
            fig, ax = plt.subplots(1, 1, figsize=(20, 8), subplot_kw={'projection': crs})
            plot_background(extent, ax=ax, linewidth=0.5)
            cmap_probs, norm_probs = cm.get_cmap(cmap, n_colors), colors.Normalize(vmin=0, vmax=vmax)
            probs_ds[front_key].isel().plot.contourf(ax=ax, x='longitude', y='latitude', norm=norm_probs, levels=levels, cmap=cmap_probs,
                transform=ccrs.PlateCarree(), alpha=0.75, add_colorbar=False)
            ax.set_title(f'Data: ERA5 reanalysis {year}-%02d-%02d-%02dz' % (month, day, hour), loc='left')
            current_fronts['identifier'].plot(ax=ax, x='longitude', y='latitude', cmap=cmap_front, norm=norm_front, transform=ccrs.PlateCarree(), add_colorbar=False)
            ax.set_title(f'{front_name} predictions')
            cbar_ax = fig.add_axes([0.8365, 0.11, 0.015, 0.77])
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm_probs, cmap=cmap_probs), cax=cbar_ax, boundaries=levels[1:], alpha=0.75)
            cbar.set_label(f'{front_name} probability', rotation=90)
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_tick_labels[int(probability_mask*cbar_label_adjust):])
            plt.savefig('%s/model_%d/maps/%s/%s-%s.png' % (model_dir, model_number, subdir_base, filename_base, front_label), bbox_inches='tight', dpi=300)
            plt.close()


def learning_curve(model_number, model_dir, include_validation_plots=True):
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
    metric = model_properties['metric']

    if loss == 'fss':
        fss_mask_size, fss_c = model_properties['fss_mask_c'][0], model_properties['fss_mask_c'][1]
        loss_title = 'Fractions Skill Score (mask=%d, c=%.1f)' % (fss_mask_size, fss_c)
    elif loss == 'bss':
        loss_title = 'Brier Skill Score'
    elif loss == 'cce':
        loss_title = 'Categorical Cross-Entropy'
    else:
        loss_title = None

    """
    metric_title: Title of the metric plots on the learning curves
    metric_string: Metric as it appears in the history files
    """
    if metric == 'fss':
        fss_mask_size, fss_c = model_properties['fss_mask_c'][0], model_properties['fss_mask_c'][1]
        metric_title = 'Fractions Skill Score (mask=%d, c=%.1f)' % (fss_mask_size, fss_c)
        metric_string = 'FSS_loss'
    elif metric == 'bss':
        metric_title = 'Brier Skill Score'
        metric_string = 'brier_skill_score'
    elif metric == 'auc':
        metric_title = 'Area Under the Curve'
        metric_string = 'auc'
    else:
        metric_title = None
        metric_string = None

    if include_validation_plots is True:
        nrows = 2
        figsize = (14, 12)
    else:
        nrows = 1
        figsize = (14, 7)

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

    if 'softmax_loss' in history:
        plt.plot(history['softmax_%s' % metric_string], label='Encoder 6')
        plt.plot(history['softmax_%s' % metric_string], label='Decoder 5')
        plt.plot(history['softmax_%s' % metric_string], label='Decoder 4')
        plt.plot(history['softmax_%s' % metric_string], label='Decoder 3')
        plt.plot(history['softmax_%s' % metric_string], label='Decoder 2')
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

    # plt.savefig("%s/model_%d/model_%d_learning_curve.png" % (model_dir, model_number, model_number), bbox_inches='tight')


if __name__ == '__main__':
    """
    All arguments listed in the examples are listed via argparse in alphabetical order below this comment block.
    
    Examples
    --------
    Finding compatible image dimensions:
        ==================================================================================
        python evaluate_model.py --find_matches --domain_size 288 128 --image_size 128 128
        ==================================================================================
        Required arguments: --find_matches, --domain_size, --image_size
        
    Generating model predictions:
        ==========================================================================================================================
        python evaluate_model.py --generate_predictions --save_probabilities --save_statistics --save_map --prediction_method all 
        --domain conus --model_dir /home/my_model_folder --netcdf_dir /home/netcdf_files --model_number 6846496 --domain_images 3 1
        --domain_size 288 128 --dataset test
        =========================================================================================================================
        Required arguments: --generate_predictions, --model_number, --model_dir, --domain, --domain_images, --domain_size, 
                            --prediction_method, --netcdf_dir
        Optional arguments: --domain_trim, --save_probabilities, --save_statistics, --save_map, --random_variables, --dataset
        Conditional arguments: --datetime - must be passed if --prediction_method == 'datetime'.
                               --num_rand_predictions - can be passed if --prediction_method == 'random'. Defaults to 10.
        *** NOTE: GPUs can be used to generate the predictions by passing --gpu_devices
    
    Generating learning curves:
        ==================================================================================================
        python evaluate_model.py --learning_curve --model_dir /home/my_model_folder --model_number 7805504
        ==================================================================================================
        Required arguments: --learning_curve, --model_number, --model_dir
    
    Generating performance diagrams:
        ============================================================================================================================================================
        python evaluate_model.py --plot_performance_diagrams --bootstrap --domain conus --model_number 7961517 --model_dir /home/my_model_folder --domain_images 3 1
        ============================================================================================================================================================
        Required arguments: --plot_performance_diagrams, --model_number, --model_dir, --domain, --domain_images
        Optional arguments: --bootstrap, --random_variables, --num_iterations, --domain_trim
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
    parser.add_argument('--domain_trim', type=int, nargs=2, default=[0, 0],
                        help='Number of pixels to trim the images by along each dimension for stitching before taking the '
                             'maximum across overlapping pixels.')
    parser.add_argument('--find_matches', action='store_true', help='Find matches for stitching predictions?')
    parser.add_argument('--generate_predictions', action='store_true', help='Generate prediction plots?')
    parser.add_argument('--calculate_stats', action='store_true', help='generate stats')
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
    parser.add_argument('--random_variables', type=str, nargs="+", default=None, help="Variables to randomize when generating predictions.")
    parser.add_argument('--save_map', action='store_true', help='Save maps of the model predictions?')
    parser.add_argument('--save_probabilities', action='store_true', help='Save model prediction data out to netcdf files?')
    parser.add_argument('--save_statistics', action='store_true', help='Save performance statistics data out to netcdf files?')
    parser.add_argument('--variable_data_source', type=str, default='era5', help='Data source for variables')

    args = parser.parse_args()
    provided_arguments = vars(args)

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
        image_size = model_properties['input_size'][0:2]  # We are only concerned about longitude and latitude when checking compatibility

        # Verify the compatibility of image stitching arguments
        # find_matches_for_domain(args.domain_size, image_size, compatibility_mode=True, compat_images=args.domain_images)

        generate_predictions(args.model_number, args.model_dir, args.variables_netcdf_indir, args.fronts_netcdf_indir, args.prediction_method,
            domain=args.domain, domain_images=args.domain_images, domain_trim=args.domain_trim, dataset=args.dataset,
            datetime=args.datetime, num_rand_predictions=args.num_rand_predictions, random_variables=args.random_variables,
            variable_data_source=args.variable_data_source)

    if args.calculate_stats:
        required_arguments = ['model_number', 'model_dir', 'fronts_netcdf_indir', 'timestep', 'domain_images', 'domain_trim']
        for hour in range(0, 24, 3):
            timestep = (args.timestep[0], args.timestep[1], args.timestep[2], hour)
            calculate_performance_stats(args.model_number, args.model_dir, args.fronts_netcdf_indir, timestep, args.domain,
                args.domain_images, args.domain_trim)

    if args.plot_performance_diagrams:
        required_arguments = ['model_number', 'model_dir', 'domain', 'domain_images']
        check_arguments(provided_arguments, required_arguments)
        plot_performance_diagrams(args.model_dir, args.model_number, args.domain, args.domain_images, args.domain_trim,
            random_variables=args.random_variables, bootstrap=args.bootstrap, num_iterations=args.num_iterations)
