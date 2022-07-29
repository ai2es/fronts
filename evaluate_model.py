"""
Functions used for evaluating a U-Net model.

Code written by: Andrew Justin (andrewjustinwx@gmail.com)

Last updated: 7/28/2022 11:53 PM CDT
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
import pickle
import tensorflow as tf
from matplotlib import cm, colors  # Here we explicitly import the cm and color modules to suppress a PyCharm bug
from utils.data_utils import expand_fronts, reformat_fronts
from utils.plotting_utils import plot_background
from variables import normalize_era5, normalize_gdas
from glob import glob


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


def calculate_performance_stats(probs_ds, front_types, fronts_filename):
    """
    Function that calculates the number of true positives, false positives, true negatives, and false negatives for a model
    prediction.

    Parameters
    ----------
    probs_ds: xarray Dataset
        - Dataset containing the model's predictions.
    front_types: str
        - Special code representing the front types.
    fronts_filename: str
        - Filename for the file containing the front objects.

    Returns
    -------
    performance_ds: xarray Dataset
        - Performance statistics for the model prediction.
    """

    """
    Performance stats arrays
    tp_<front>: Array for the numbers of true positives of the given front and threshold
    tn_<front>: Array for the numbers of true negatives of the given front and threshold
    fp_<front>: Array for the numbers of false positives of the given front and threshold
    fn_<front>: Array for the numbers of false negatives of the given front and threshold
    """
    tp_cold = np.zeros(shape=[5, 100])
    fp_cold = np.zeros(shape=[5, 100])
    tn_cold = np.zeros(shape=[5, 100])
    fn_cold = np.zeros(shape=[5, 100])
    tp_warm = np.zeros(shape=[5, 100])
    fp_warm = np.zeros(shape=[5, 100])
    tn_warm = np.zeros(shape=[5, 100])
    fn_warm = np.zeros(shape=[5, 100])
    tp_stationary = np.zeros(shape=[5, 100])
    fp_stationary = np.zeros(shape=[5, 100])
    tn_stationary = np.zeros(shape=[5, 100])
    fn_stationary = np.zeros(shape=[5, 100])
    tp_occluded = np.zeros(shape=[5, 100])
    fp_occluded = np.zeros(shape=[5, 100])
    tn_occluded = np.zeros(shape=[5, 100])
    fn_occluded = np.zeros(shape=[5, 100])
    tp_front = np.zeros(shape=[5, 100])
    fp_front = np.zeros(shape=[5, 100])
    tn_front = np.zeros(shape=[5, 100])
    fn_front = np.zeros(shape=[5, 100])

    thresholds = np.linspace(0.01, 1, 100)  # Probability thresholds for calculating performance statistics
    boundaries = np.array([50, 100, 150, 200, 250])  # Boundaries for checking whether or not a front is present (kilometers)

    front_identifier = pd.read_pickle(fronts_filename)

    if front_types == ['CF', 'WF']:

        """ Calculate initial true negatives and false negatives """
        cf_bool = xr.where(front_identifier == 1, 1, 0)['identifier']  # 1 = cold front, 0 = not a cold front
        wf_bool = xr.where(front_identifier == 2, 1, 0)['identifier']  # 1 = warm front, 0 = no warm front
        probs_cf = probs_ds['CF']  # Model predictions for cold fronts
        probs_wf = probs_ds['WF']  # Model predictions for warm fronts
        for i in range(100):
            """ 
            True negative ==> model correctly predicts the lack of a front at a given point 
            False negative ==> model does not predict a front, but a front exists
            """
            tn_cold[:, i] += len(np.where((probs_cf < thresholds[i]) & (cf_bool == 0))[0])
            fn_cold[:, i] += len(np.where((probs_cf < thresholds[i]) & (cf_bool == 1))[0])
            tn_warm[:, i] += len(np.where((probs_wf < thresholds[i]) & (wf_bool == 0))[0])
            fn_warm[:, i] += len(np.where((probs_wf < thresholds[i]) & (wf_bool == 1))[0])

        """ Calculate true positives and false positives """
        for boundary in range(5):
            front_identifier = pd.read_pickle(fronts_filename)
            for y in range(int(2*(boundary+1))):
                front_identifier = expand_fronts(front_identifier)  # Expand fronts
            cf_bool = xr.where(front_identifier == 1, 1, 0)['identifier']  # 1 = cold front, 0 = not a cold front
            wf_bool = xr.where(front_identifier == 2, 1, 0)['identifier']  # 1 = warm front, 0 = no warm front
            for i in range(100):
                """
                True positive ==> model correctly identifies a front
                False positive ==> model predicts a front, but it does not exist
                """
                tp_cold[boundary, i] += len(np.where((probs_cf > thresholds[i]) & (cf_bool == 1))[0])
                fp_cold[boundary, i] += len(np.where((probs_cf > thresholds[i]) & (cf_bool == 0))[0])
                tp_warm[boundary, i] += len(np.where((probs_wf > thresholds[i]) & (wf_bool == 1))[0])
                fp_warm[boundary, i] += len(np.where((probs_wf > thresholds[i]) & (wf_bool == 0))[0])

        performance_ds = xr.Dataset({"tp_cold": (["boundary", "threshold"], tp_cold), "tp_warm": (["boundary", "threshold"], tp_warm),
                                     "fp_cold": (["boundary", "threshold"], fp_cold), "fp_warm": (["boundary", "threshold"], fp_warm),
                                     "tn_cold": (["boundary", "threshold"], tn_cold), "tn_warm": (["boundary", "threshold"], tn_warm),
                                     "fn_cold": (["boundary", "threshold"], fn_cold), "fn_warm": (["boundary", "threshold"], fn_warm)}, coords={"boundary": boundaries, "threshold": thresholds})

    elif front_types == ['SF', 'OF']:

        """ Calculate initial true negatives and false negatives """
        sf_bool = xr.where(front_identifier == 3, 1, 0)['identifier']  # 1 = stationary front, 0 = not a stationary front
        of_bool = xr.where(front_identifier == 4, 1, 0)['identifier']  # 1 = occluded front, 0 = no occluded front
        probs_sf = probs_ds['SF']  # Model predictions for stationary fronts
        probs_of = probs_ds['OF']  # Model predictions for occluded fronts
        for i in range(100):
            """ 
            True negative ==> model correctly predicts the lack of a front at a given point 
            False negative ==> model does not predict a front, but a front exists
            """
            tn_stationary[:, i] += len(np.where((probs_sf < thresholds[i]) & (sf_bool == 0))[0])
            fn_stationary[:, i] += len(np.where((probs_sf < thresholds[i]) & (sf_bool == 1))[0])
            tn_occluded[:, i] += len(np.where((probs_of < thresholds[i]) & (of_bool == 0))[0])
            fn_occluded[:, i] += len(np.where((probs_of < thresholds[i]) & (of_bool == 1))[0])

        """ Calculate true positives and false positives """
        for boundary in range(5):
            front_identifier = pd.read_pickle(fronts_filename)
            for y in range(int(2*(boundary+1))):
                front_identifier = expand_fronts(front_identifier)  # Expand fronts
            sf_bool = xr.where(front_identifier == 3, 1, 0)['identifier']  # 1 = stationary front, 0 = not a stationary front
            of_bool = xr.where(front_identifier == 4, 1, 0)['identifier']  # 1 = occluded front, 0 = no occluded front
            for i in range(100):
                """
                True positive ==> model correctly identifies a front
                False positive ==> model predicts a front, but it does not exist
                """
                tp_stationary[boundary, i] += len(np.where((probs_sf > thresholds[i]) & (sf_bool == 1))[0])
                fp_stationary[boundary, i] += len(np.where((probs_sf > thresholds[i]) & (sf_bool == 0))[0])
                tp_occluded[boundary, i] += len(np.where((probs_of > thresholds[i]) & (of_bool == 1))[0])
                fp_occluded[boundary, i] += len(np.where((probs_of > thresholds[i]) & (of_bool == 0))[0])

        performance_ds = xr.Dataset({"tp_stationary": (["boundary", "threshold"], tp_stationary), "tp_occluded": (["boundary", "threshold"], tp_occluded),
                                     "fp_stationary": (["boundary", "threshold"], fp_stationary), "fp_occluded": (["boundary", "threshold"], fp_occluded),
                                     "tn_stationary": (["boundary", "threshold"], tn_stationary), "tn_occluded": (["boundary", "threshold"], tn_occluded),
                                     "fn_stationary": (["boundary", "threshold"], fn_stationary), "fn_occluded": (["boundary", "threshold"], fn_occluded)}, coords={"boundary": boundaries, "threshold": thresholds})

    elif front_types == ['CF', 'WF', 'SF', 'OF']:
        for boundary in range(5):
            fronts = pd.read_pickle(fronts_filename)  # This is the "backup" dataset that can be used to reset the 'new_fronts' dataset
            for y in range(int(2*(boundary+1))):
                fronts = expand_fronts(fronts)

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
                tp_cold[boundary, i] += len(np.where(t_cold_probs > thresholds[i])[0])
                tn_cold[boundary, i] += len(np.where((f_cold_probs < thresholds[i]) & (f_cold_probs != 0))[0])
                fp_cold[boundary, i] += len(np.where(f_cold_probs > thresholds[i])[0])
                fn_cold[boundary, i] += len(np.where((t_cold_probs < thresholds[i]) & (t_cold_probs != 0))[0])
                tp_warm[boundary, i] += len(np.where(t_warm_probs > thresholds[i])[0])
                tn_warm[boundary, i] += len(np.where((f_warm_probs < thresholds[i]) & (f_warm_probs != 0))[0])
                fp_warm[boundary, i] += len(np.where(f_warm_probs > thresholds[i])[0])
                fn_warm[boundary, i] += len(np.where((t_warm_probs < thresholds[i]) & (t_warm_probs != 0))[0])
                tp_stationary[boundary, i] += len(np.where(t_stationary_probs > thresholds[i])[0])
                tn_stationary[boundary, i] += len(np.where((f_stationary_probs < thresholds[i]) & (f_stationary_probs != 0))[0])
                fp_stationary[boundary, i] += len(np.where(f_stationary_probs > thresholds[i])[0])
                fn_stationary[boundary, i] += len(np.where((t_stationary_probs < thresholds[i]) & (t_stationary_probs != 0))[0])
                tp_occluded[boundary, i] += len(np.where(t_occluded_probs > thresholds[i])[0])
                tn_occluded[boundary, i] += len(np.where((f_occluded_probs < thresholds[i]) & (f_occluded_probs != 0))[0])
                fp_occluded[boundary, i] += len(np.where(f_occluded_probs > thresholds[i])[0])
                fn_occluded[boundary, i] += len(np.where((t_occluded_probs < thresholds[i]) & (t_occluded_probs != 0))[0])

        performance_ds = xr.Dataset({"tp_cold": (["boundary", "threshold"], tp_cold), "tp_warm": (["boundary", "threshold"], tp_warm),
                 "tp_stationary": (["boundary", "threshold"], tp_stationary), "tp_occluded": (["boundary", "threshold"], tp_occluded),
                 "fp_cold": (["boundary", "threshold"], fp_cold), "fp_warm": (["boundary", "threshold"], fp_warm),
                 "fp_stationary": (["boundary", "threshold"], fp_stationary), "fp_occluded": (["boundary", "threshold"], fp_occluded),
                 "tn_cold": (["boundary", "threshold"], tn_cold), "tn_warm": (["boundary", "threshold"], tn_warm),
                 "tn_stationary": (["boundary", "threshold"], tn_stationary), "tn_occluded": (["boundary", "threshold"], tn_occluded),
                 "fn_cold": (["boundary", "threshold"], fn_cold), "fn_warm": (["boundary", "threshold"], fn_warm),
                 "fn_stationary": (["boundary", "threshold"], fn_stationary), "fn_occluded": (["boundary", "threshold"], fn_occluded)}, coords={"boundary": boundaries, "threshold": thresholds})

    elif front_types == 'F_BIN':

        """ Calculate initial true negatives and false negatives """
        front_bin_bool = reformat_fronts(pd.read_pickle(fronts_filename), front_types)['identifier']  # 1 = front exists, 0 = front does not exist
        probs_front_bin = probs_ds['F_BIN']  # Model predictions for binary fronts
        for i in range(100):
            """ 
            True negative ==> model correctly predicts the lack of a front at a given point 
            False negative ==> model does not predict a front, but a front exists
            """
            tn_front[:, i] += len(np.where((probs_front_bin < thresholds[i]) & (front_bin_bool == 0))[0])
            fn_front[:, i] += len(np.where((probs_front_bin < thresholds[i]) & (front_bin_bool == 1))[0])

        """ Calculate true positives and false positives """
        for boundary in range(5):
            fronts_bin_bool = reformat_fronts(pd.read_pickle(fronts_filename), front_types)  # 1 = front exists, 0 = front does not exist
            for y in range(int(2*(boundary+1))):
                fronts_bin_bool = expand_fronts(fronts_bin_bool)

            fronts_bin_bool = fronts_bin_bool['identifier']
            for i in range(100):
                """
                True positive ==> model correctly identifies a front
                False positive ==> model predicts a front, but it does not exist
                """
                tp_front[boundary, i] += len(np.where((probs_front_bin > thresholds[i]) & (fronts_bin_bool == 1))[0])
                fp_front[boundary, i] += len(np.where((probs_front_bin > thresholds[i]) & (fronts_bin_bool == 0))[0])

        performance_ds = xr.Dataset({"tp_front": (["boundary", "threshold"], tp_front), "fp_front": (["boundary", "threshold"], fp_front),
                                     "tn_front": (["boundary", "threshold"], tn_front), "fn_front": (["boundary", "threshold"], fn_front)},
                                    coords={"boundary": boundaries, "threshold": thresholds})

    else:
        raise NameError("Unable to determine front types")

    return performance_ds


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


def generate_predictions(model_number, model_dir, variables_pickle_indir, fronts_pickle_indir, domain, domain_images, domain_size, domain_trim,
    prediction_method, datetime, dataset=None, num_rand_predictions=None, random_variables=None, save_map=True,
    save_probabilities=False, save_statistics=False, variable_data_source='gdas'):
    """
    Generate predictions with a model.

    Parameters
    ----------
    model_number: int
        - Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        - Main directory for the models.
    variables_pickle_indir: str
        - Input directory for the ERA5 pickle files.
    fronts_pickle_indir: str
        - Input directory for the front object pickle files.
    domain: str
        - Domain of the datasets.
    domain_images: iterable object with 2 ints
        - Number of images along each dimension of the final stitched map (lon lat).
    domain_size: iterable object with 2 ints
        - Number of pixels along each dimension of the final stitched map (lon lat).
    domain_trim: iterable object with 2 ints
        - Number of pixels to trim each image by along each dimension before taking the maximum of the overlapping pixels (lon lat).
    prediction_method: str
        - Prediction method. Options are: 'datetime', 'random', 'all'
    datetime: iterable object with 4 integers
        - 4 values for the date and time: year, month, day, hour
    dataset: str
        - Dataset for which to make predictions if prediction_method is 'random' or 'all'.
    num_rand_predictions: int
        - Number of random predictions to make.
    random_variables: str, or iterable of strings
        - Variables to randomize.
    save_map: bool
        - Setting this to true will save the prediction maps.
    save_probabilities: bool
        - Setting this to true will save front probability data to a pickle file.
    save_statistics: bool
        - Setting this to true will save performance statistics data to a pickle file.
    variable_data_source: str
        - Variable data to use for training the model. Options are: 'era5', 'gdas', or 'gfs' (case-insensitive)
    """

    # Model properties
    model_properties = pd.read_pickle(f"{model_dir}/model_{model_number}/model_{model_number}_properties.pkl")
    image_size = model_properties['input_size'][:-1]  # The image size does not include the last dimension of the input size as it only represents the number of channels
    channels = model_properties['input_size'][-1]
    front_types = model_properties['front_types']
    classes = model_properties['classes']
    num_dimensions = len(image_size)
    num_variables = model_properties['num_variables']
    test_years, valid_years = model_properties['test_years'], model_properties['validation_years']

    # Properties of the final map made from stitched images
    domain_images_lon, domain_images_lat = domain_images[0], domain_images[1]
    domain_size_lon, domain_size_lat = domain_size[0], domain_size[1]
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

    if prediction_method == 'datetime':
        dataset = 'timestep'
        timestep_tuple = tuple(datetime)
    else:
        timestep_tuple = None

    if variable_data_source.lower() == 'era5':
        variable_files, front_files = fm.load_era5_and_fronts_pickle_files(variables_pickle_indir, fronts_pickle_indir, num_variables, domain, dataset=dataset, test_years=test_years, validation_years=valid_years, timestep_tuple=timestep_tuple)
    elif variable_data_source.lower() == 'gdas' or variable_data_source.lower() == 'gfs':
        variable_files, front_files = fm.load_gdas_or_gfs_and_fronts_pickle_files(variables_pickle_indir, fronts_pickle_indir, domain, dataset=dataset, test_years=test_years, validation_years=valid_years, gdas_or_gfs=variable_data_source.lower(), timestep_tuple=timestep_tuple)
    else:
        raise TypeError("Invalid source for variable data, available options are: 'era5', 'gdas', or 'gfs' (case-insensitive)")

    # Find files with provided date and time to make a prediction (if applicable)
    if prediction_method == 'random':
        if prediction_method == 'random':  # Select a random set of files for which to generate predictions
            indices = random.choices(range(len(front_files) - 1), k=num_rand_predictions)
            if num_rand_predictions > 1:
                front_files, variable_files = list(front_files[index] for index in indices), list(variable_files[index] for index in indices)
            else:
                front_files, variable_files = [front_files,], [variable_files,]
    elif prediction_method != 'all' and prediction_method != 'datetime':
        raise ValueError(f"'{prediction_method}' is not a valid prediction method. Options are: 'datetime', 'random', 'all'")

    subdir_base = '%s_%dx%dimages_%dx%dtrim' % (domain, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1])  #
    if random_variables is not None:
        subdir_base += '_' + '-'.join(sorted(random_variables))

    model = fm.load_model(model_number, model_dir)
    n = 0  # Counter for the number of down layers in the model
    if num_dimensions == 2:
        for layer in model.layers:
            if layer.__class__.__name__ == 'MaxPooling2D':
                n += 1
    elif num_dimensions == 3:
        for layer in model.layers:
            if layer.__class__.__name__ == 'MaxPooling3D':
                n += 1
    else:
        raise ValueError("Invalid dimensions value: %d" % num_dimensions)
    n = int((n - 1)/2)

    for file_index in range(len(front_files)):

        """ 
        Create array that contains all probabilities across the whole domain map for every front type that the model is 
        predicting. The first dimension is 1 unit less than the number of classes because we are not concerned about the 
        'no front' front type in the model.
        """
        stitched_map_probs = np.empty(shape=[classes-1, domain_size_lon_trimmed, domain_size_lat_trimmed])

        if variable_data_source == 'era5':
            fronts_filename = front_files[file_index]
            variables_filename = variable_files[file_index]

            variable_ds = normalize_era5(pd.read_pickle(variables_filename))

            # Randomize variable
            if random_variables is not None:
                for random_variable in random_variables:
                    domain_dim_lon = len(variable_ds['longitude'].values)  # Length of the full domain in the longitude direction (# of pixels)
                    domain_dim_lat = len(variable_ds['latitude'].values)  # Length of the full domain in the latitude direction (# of pixels)
                    var_values = variable_ds[random_variable].values.flatten()
                    np.random.shuffle(var_values)
                    variable_ds[random_variable].values = var_values.reshape(domain_dim_lat, domain_dim_lon)

            fronts_ds = pd.read_pickle(fronts_filename)
            fronts = fronts_ds.sel(longitude=fronts_ds.longitude.values[domain_trim_lon:domain_size_lon-domain_trim_lon],
                                   latitude=fronts_ds.latitude.values[domain_trim_lat:domain_size_lat-domain_trim_lat])

            # Latitude and longitude points in the domain
            image_lats = fronts_ds.latitude.values[domain_trim_lat:domain_size_lat-domain_trim_lat]
            image_lons = fronts_ds.longitude.values[domain_trim_lon:domain_size_lon-domain_trim_lon]

            time = str(fronts.time.values)[0:13].replace('T', '-') + 'z'

            map_created = False  # Boolean that determines whether or not the final stitched map has been created

            for lat_image in range(domain_images_lat):
                lat_index = int(lat_image*lat_image_spacing)
                for lon_image in range(domain_images_lon):
                    print("%s....%d/%d" % (time, int(lat_image*domain_images_lon)+lon_image, int(domain_images_lon*domain_images_lat)), end='\r')
                    lon_index = int(lon_image*lon_image_spacing)

                    # variable_ds = raw_variable_ds
                    lons = variable_ds.longitude.values[lon_index:lon_index + image_size[0]]  # Longitude points for the current image
                    lats = variable_ds.latitude.values[lat_index:lat_index + image_size[1]]  # Latitude points for the current image

                    if num_dimensions == 2:
                        variable_ds_new = np.nan_to_num(variable_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, image_size[0], image_size[1], channels))
                    elif num_dimensions == 3:
                        variables_sfc = variable_ds[['t2m', 'd2m', 'sp', 'u10', 'v10', 'theta_w', 'mix_ratio', 'rel_humid', 'virt_temp', 'wet_bulb', 'theta_e',
                                                     'q']].sel(longitude=lons, latitude=lats).to_array().values
                        variables_1000 = variable_ds[['t_1000', 'd_1000', 'z_1000', 'u_1000', 'v_1000', 'theta_w_1000', 'mix_ratio_1000', 'rel_humid_1000', 'virt_temp_1000',
                                                      'wet_bulb_1000', 'theta_e_1000', 'q_1000']].sel(longitude=lons, latitude=lats).to_array().values
                        variables_950 = variable_ds[['t_950', 'd_950', 'z_950', 'u_950', 'v_950', 'theta_w_950', 'mix_ratio_950', 'rel_humid_950', 'virt_temp_950',
                                                     'wet_bulb_950', 'theta_e_950', 'q_950']].sel(longitude=lons, latitude=lats).to_array().values
                        variables_900 = variable_ds[['t_900', 'd_900', 'z_900', 'u_900', 'v_900', 'theta_w_900', 'mix_ratio_900', 'rel_humid_900', 'virt_temp_900',
                                                     'wet_bulb_900', 'theta_e_900', 'q_900']].sel(longitude=lons, latitude=lats).to_array().values
                        variables_850 = variable_ds[['t_850', 'd_850', 'z_850', 'u_850', 'v_850', 'theta_w_850', 'mix_ratio_850', 'rel_humid_850', 'virt_temp_850',
                                                     'wet_bulb_850', 'theta_e_850', 'q_850']].sel(longitude=lons, latitude=lats).to_array().values
                        variable_ds_new = np.expand_dims(np.array([variables_sfc, variables_1000, variables_950, variables_900, variables_850]).transpose([3, 2, 0, 1]), axis=0)
                    else:
                        raise ValueError("Invalid number of dimensions: %d" % num_dimensions)

                    prediction = model.predict(variable_ds_new)

                    if num_dimensions == 2:
                        image_probs = np.transpose(prediction[0][0][:, :, 1:], (2, 0, 1))  # Final index is front+1 because we skip over the 'no front' type
                    else:
                        image_probs = np.transpose(np.amax(prediction[0][0][:, :, :, 1:], axis=2), (2, 0, 1))  # Final index is front+1 because we skip over the 'no front' type

                    # Add predictions to the map
                    stitched_map_probs, map_created = add_image_to_map(stitched_map_probs, image_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        image_size_lon, image_size_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)

                    if map_created is True:
                        print("%s....%d/%d" % (time, int(domain_images_lon*domain_images_lat), int(domain_images_lon*domain_images_lat)))
                        if front_types == 'F_BIN' or front_types == 'MERGED-F_BIN' or front_types == 'MERGED-T':
                            probs_ds = xr.Dataset(
                                {front_types: (('longitude', 'latitude'), stitched_map_probs[0])},
                                coords={'latitude': image_lats, 'longitude': image_lons})
                        elif front_types == 'MERGED-F':
                            probs_ds = xr.Dataset(
                                {'CF_merged': (('longitude', 'latitude'), stitched_map_probs[0]),
                                 'WF_merged': (('longitude', 'latitude'), stitched_map_probs[1]),
                                 'SF_merged': (('longitude', 'latitude'), stitched_map_probs[2]),
                                 'OF_merged': (('longitude', 'latitude'), stitched_map_probs[3])},
                                coords={'latitude': image_lats, 'longitude': image_lons})
                        elif front_types == 'MERGED-ALL':
                            probs_ds = xr.Dataset(
                                {'CF_merged': (('longitude', 'latitude'), stitched_map_probs[0]),
                                 'WF_merged': (('longitude', 'latitude'), stitched_map_probs[1]),
                                 'SF_merged': (('longitude', 'latitude'), stitched_map_probs[2]),
                                 'OF_merged': (('longitude', 'latitude'), stitched_map_probs[3]),
                                 'TROF_merged': (('longitude', 'latitude'), stitched_map_probs[4]),
                                 'INST': (('longitude', 'latitude'), stitched_map_probs[5]),
                                 'DL': (('longitude', 'latitude'), stitched_map_probs[6])},
                                coords={'latitude': image_lats, 'longitude': image_lons})
                        elif type(front_types) == list:
                            probs_ds_dict = dict({})
                            probs_ds_index = 0
                            for front_type in front_types:
                                probs_ds_dict[front_type] = (('longitude', 'latitude'), stitched_map_probs[probs_ds_index])
                                probs_ds_index += 1
                            probs_ds = xr.Dataset(probs_ds_dict, coords={'latitude': image_lats, 'longitude': image_lons})
                        else:
                            raise ValueError(f"'{front_types}' is not a valid set of front types.")

                        filename_base = 'model_%d_%s_%s_%dx%dimages_%dx%dtrim' % (model_number, time, domain, domain_images_lon,
                            domain_images_lat, domain_trim_lon, domain_trim_lat)
                        if random_variables is not None:
                            filename_base += '_' + '-'.join(sorted(random_variables))

                        if save_map:
                            # Check that the necessary subdirectory exists and make it if it doesn't exist
                            if not os.path.isdir('%s/model_%d/maps/%s' % (model_dir, model_number, subdir_base)):
                                os.mkdir('%s/model_%d/maps/%s' % (model_dir, model_number, subdir_base))
                                print("New subdirectory made:", '%s/model_%d/maps/%s' % (model_dir, model_number, subdir_base))
                            prediction_plot(fronts, probs_ds, time, model_number, model_dir, domain, subdir_base, filename_base)  # Generate prediction plot

                        if save_probabilities:
                            # Check that the necessary subdirectory exists and make it if it doesn't exist
                            if not os.path.isdir('%s/model_%d/probabilities/%s' % (model_dir, model_number, subdir_base)):
                                os.mkdir('%s/model_%d/probabilities/%s' % (model_dir, model_number, subdir_base))
                                print("New subdirectory made:", '%s/model_%d/probabilities/%s' % (model_dir, model_number, subdir_base))
                            outfile = '%s/model_%d/probabilities/%s/%s_probabilities.pkl' % (model_dir, model_number, subdir_base, filename_base)

                            with open(outfile, 'wb') as f:
                                pickle.dump(probs_ds, f)  # Save probabilities dataset

                        if save_statistics:

                            # Check that the necessary subdirectory exists and make it if it doesn't exist
                            if not os.path.isdir('%s/model_%d/statistics/%s' % (model_dir, model_number, subdir_base)):
                                os.mkdir('%s/model_%d/statistics/%s' % (model_dir, model_number, subdir_base))
                                print("New subdirectory made:", '%s/model_%d/statistics/%s' % (model_dir, model_number, subdir_base))

                            performance_ds = calculate_performance_stats(probs_ds, front_types, fronts_filename)
                            outfile = '%s/model_%d/statistics/%s/%s_statistics.pkl' % (model_dir, model_number, subdir_base, filename_base)
                            with open(outfile, 'wb') as f:
                                pickle.dump(performance_ds, f)

        elif variable_data_source == 'gdas':

            fronts_filename = front_files[file_index]
            gdas_filenames = variable_files[file_index]

            print(gdas_filenames)

            # Create a backup variable dataset that will be called to reset the variable dataset after it is modified.
            gdas_surface = normalize_gdas(pd.read_pickle(gdas_filenames[0]), 'surface')
            gdas_1000 = normalize_gdas(pd.read_pickle(gdas_filenames[1]), 1000)
            gdas_950 = normalize_gdas(pd.read_pickle(gdas_filenames[3]), 950)
            gdas_900 = normalize_gdas(pd.read_pickle(gdas_filenames[5]), 900)
            gdas_850 = normalize_gdas(pd.read_pickle(gdas_filenames[6]), 850)

            # TODO: Develop randomization code for GDAS data
            # # Randomize variable
            # if random_variables is not None:
            #     for random_variable in random_variables:
            #         domain_dim_lon = len(variable_ds['longitude'].values)  # Length of the full domain in the longitude direction (# of pixels)
            #         domain_dim_lat = len(variable_ds['latitude'].values)  # Length of the full domain in the latitude direction (# of pixels)
            #         var_values = variable_ds[random_variable].values.flatten()
            #         np.random.shuffle(var_values)
            #         raw_variable_ds[random_variable].values = var_values.reshape(domain_dim_lat, domain_dim_lon)

            fronts_ds = pd.read_pickle(fronts_filename)
            fronts = fronts_ds.sel(longitude=fronts_ds.longitude.values[domain_trim_lon:domain_size_lon-domain_trim_lon],
                                   latitude=fronts_ds.latitude.values[domain_trim_lat:domain_size_lat-domain_trim_lat])

            # Latitude and longitude points in the domain
            image_lats = fronts_ds.latitude.values[domain_trim_lat:domain_size_lat-domain_trim_lat]
            image_lons = fronts_ds.longitude.values[domain_trim_lon:domain_size_lon-domain_trim_lon]

            time = str(fronts.time.values)[0:13].replace('T', '-') + 'z'

            map_created = False  # Boolean that determines whether or not the final stitched map has been created

            for lat_image in range(domain_images_lat):
                lat_index = int(lat_image*lat_image_spacing)
                for lon_image in range(domain_images_lon):
                    print("%s....%d/%d" % (time, int(lat_image*domain_images_lon)+lon_image, int(domain_images_lon*domain_images_lat)), end='\r')
                    lon_index = int(lon_image*lon_image_spacing)

                    lons = fronts_ds.longitude.values[lon_index:lon_index + image_size[0]]  # Longitude points for the current image
                    lats = fronts_ds.latitude.values[lat_index:lat_index + image_size[1]]  # Latitude points for the current image

                    # TODO: Program GDAS datasets for 2D models
                    # if num_dimensions == 2:
                    #     variable_ds_new = np.nan_to_num(variable_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, image_size[0], image_size[1], channels))
                    if num_dimensions == 3:
                        variables_sfc = gdas_surface[['T', 'Td', 'mslp', 'u', 'v', 'theta_w', 'r', 'RH', 'Tv', 'Tw', 'theta_e', 'q']].sel(longitude=lons, latitude=lats).to_array().values
                        variables_1000 = gdas_1000[['T', 'Td', 'z', 'u', 'v', 'theta_w', 'r', 'RH', 'Tv', 'Tw', 'theta_e', 'q']].sel(longitude=lons, latitude=lats).to_array().values
                        variables_950 = gdas_950[['T', 'Td', 'z', 'u', 'v', 'theta_w', 'r', 'RH', 'Tv', 'Tw', 'theta_e', 'q']].sel(longitude=lons, latitude=lats).to_array().values
                        variables_900 = gdas_900[['T', 'Td', 'z', 'u', 'v', 'theta_w', 'r', 'RH', 'Tv', 'Tw', 'theta_e', 'q']].sel(longitude=lons, latitude=lats).to_array().values
                        variables_850 = gdas_850[['T', 'Td', 'z', 'u', 'v', 'theta_w', 'r', 'RH', 'Tv', 'Tw', 'theta_e', 'q']].sel(longitude=lons, latitude=lats).to_array().values

                        variable_ds_new = np.expand_dims(np.array([variables_sfc, variables_1000, variables_950, variables_900, variables_850]).transpose([3, 2, 0, 1]), axis=0)
                    else:
                        raise ValueError("Invalid number of dimensions: %d" % num_dimensions)

                    prediction = model.predict(variable_ds_new)

                    if num_dimensions == 2:
                        image_probs = np.transpose(prediction[0][0][:, :, 1:], (2, 0, 1))  # Final index is front+1 because we skip over the 'no front' type
                    else:
                        image_probs = np.transpose(np.amax(prediction[0][0][:, :, :, 1:], axis=2), (2, 0, 1))  # Final index is front+1 because we skip over the 'no front' type

                    # Add predictions to the map
                    stitched_map_probs, map_created = add_image_to_map(stitched_map_probs, image_probs, map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                        image_size_lon, image_size_lat, domain_trim_lon, domain_trim_lat, lon_image_spacing, lat_image_spacing, lon_pixels_per_image, lat_pixels_per_image)

                    if map_created is True:
                        print("%s....%d/%d" % (time, int(domain_images_lon*domain_images_lat), int(domain_images_lon*domain_images_lat)))
                        if front_types == 'F_BIN' or front_types == 'MERGED-F_BIN' or front_types == 'MERGED-T':
                            probs_ds = xr.Dataset(
                                {front_types: (('longitude', 'latitude'), stitched_map_probs[0])},
                                coords={'latitude': image_lats, 'longitude': image_lons})
                        elif front_types == 'MERGED-F':
                            probs_ds = xr.Dataset(
                                {'CF_merged': (('longitude', 'latitude'), stitched_map_probs[0]),
                                 'WF_merged': (('longitude', 'latitude'), stitched_map_probs[1]),
                                 'SF_merged': (('longitude', 'latitude'), stitched_map_probs[2]),
                                 'OF_merged': (('longitude', 'latitude'), stitched_map_probs[3])},
                                coords={'latitude': image_lats, 'longitude': image_lons})
                        elif front_types == 'MERGED-ALL':
                            probs_ds = xr.Dataset(
                                {'CF_merged': (('longitude', 'latitude'), stitched_map_probs[0]),
                                 'WF_merged': (('longitude', 'latitude'), stitched_map_probs[1]),
                                 'SF_merged': (('longitude', 'latitude'), stitched_map_probs[2]),
                                 'OF_merged': (('longitude', 'latitude'), stitched_map_probs[3]),
                                 'TROF_merged': (('longitude', 'latitude'), stitched_map_probs[4]),
                                 'INST': (('longitude', 'latitude'), stitched_map_probs[5]),
                                 'DL': (('longitude', 'latitude'), stitched_map_probs[6])},
                                coords={'latitude': image_lats, 'longitude': image_lons})
                        elif type(front_types) == list:
                            probs_ds_dict = dict({})
                            probs_ds_index = 0
                            for front_type in front_types:
                                probs_ds_dict[front_type] = (('longitude', 'latitude'), stitched_map_probs[probs_ds_index])
                                probs_ds_index += 1
                            probs_ds = xr.Dataset(probs_ds_dict, coords={'latitude': image_lats, 'longitude': image_lons})
                        else:
                            raise ValueError(f"'{front_types}' is not a valid set of front types.")

                        filename_base = 'model_%d_%s_%s_%dx%dimages_%dx%dtrim' % (model_number, time, domain, domain_images_lon,
                            domain_images_lat, domain_trim_lon, domain_trim_lat)
                        if random_variables is not None:
                            filename_base += '_' + '-'.join(sorted(random_variables))

                        if save_map:
                            # Check that the necessary subdirectory exists and make it if it doesn't exist
                            if not os.path.isdir('%s/model_%d/maps/%s' % (model_dir, model_number, subdir_base)):
                                os.mkdir('%s/model_%d/maps/%s' % (model_dir, model_number, subdir_base))
                                print("New subdirectory made:", '%s/model_%d/maps/%s' % (model_dir, model_number, subdir_base))
                            prediction_plot(fronts, probs_ds, time, model_number, model_dir, domain, subdir_base, filename_base)  # Generate prediction plot

                        if save_probabilities:
                            # Check that the necessary subdirectory exists and make it if it doesn't exist
                            if not os.path.isdir('%s/model_%d/probabilities/%s' % (model_dir, model_number, subdir_base)):
                                os.mkdir('%s/model_%d/probabilities/%s' % (model_dir, model_number, subdir_base))
                                print("New subdirectory made:", '%s/model_%d/probabilities/%s' % (model_dir, model_number, subdir_base))
                            outfile = '%s/model_%d/probabilities/%s/%s_probabilities.pkl' % (model_dir, model_number, subdir_base, filename_base)

                            with open(outfile, 'wb') as f:
                                pickle.dump(probs_ds, f)  # Save probabilities dataset

                        if save_statistics:

                            # Check that the necessary subdirectory exists and make it if it doesn't exist
                            if not os.path.isdir('%s/model_%d/statistics/%s' % (model_dir, model_number, subdir_base)):
                                os.mkdir('%s/model_%d/statistics/%s' % (model_dir, model_number, subdir_base))
                                print("New subdirectory made:", '%s/model_%d/statistics/%s' % (model_dir, model_number, subdir_base))

                            performance_ds = calculate_performance_stats(probs_ds, front_types, fronts_filename)
                            outfile = '%s/model_%d/statistics/%s/%s_statistics.pkl' % (model_dir, model_number, subdir_base, filename_base)
                            with open(outfile, 'wb') as f:
                                pickle.dump(performance_ds, f)


def plot_performance_diagrams(model_dir, model_number, domain, domain_images, domain_trim, bootstrap=True, random_variables=None,
    num_iterations=10000):
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
    num_iterations: int
        - Number of iterations when bootstrapping the data.
    """

    model_properties = pd.read_pickle(f"{model_dir}\\model_{model_number}\\model_{model_number}_properties.pkl")
    front_types = model_properties['front_types']

    subdir_base = '%s_%dx%dimages_%dx%dtrim' % (domain, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1])
    stats_plot_base = 'model_%d_%s_%dx%dimages_%dx%dtrim' % (model_number, domain, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1])
    if random_variables is not None:
        subdir_base += '_' + '-'.join(sorted(random_variables))

    files = sorted(glob('%s\\model_%d\\statistics\\%s\\*statistics.pkl' % (model_dir, model_number, subdir_base)))

    # If evaluating over the full domain, remove non-synoptic hours (3z, 9z, 15z, 21z)
    if domain == 'full':
        hours_to_remove = [3, 9, 15, 21]
        for hour in hours_to_remove:
            string = '%02dz_' % hour
            files = list(filter(lambda hour: string not in hour, files))

    stats = pd.read_pickle(files[0])  # Open the first dataset so information on variables can be retrieved
    ds_keys = list(stats.keys())  # List of the names of the variables in the stats datasets.
    num_keys = len(ds_keys)  # Number of data variables
    num_files = len(files)  # Number of files

    """
    These arrays are used for calculating confidence intervals. We use these instead of the original datasets because numpy
    cannot efficiently perform calculations across xarray datasets.
    """
    statistics_array_50km = np.empty(shape=(num_files, num_keys, 100))
    statistics_array_100km = np.empty(shape=(num_files, num_keys, 100))
    statistics_array_150km = np.empty(shape=(num_files, num_keys, 100))
    statistics_array_200km = np.empty(shape=(num_files, num_keys, 100))

    for file_no in range(num_files):
        print(f"File {file_no}/{num_files}", end='\r')
        if 'total_stats' not in locals():  # If this is the first file
            stats = pd.read_pickle(files[0])  # Open the first dataset
            total_stats = stats
        else:
            stats = pd.read_pickle(files[file_no])
            total_stats += stats  # Add the rest of the stats to the first file to create one final dataset
        stats_50km = stats.sel(boundary=50)
        stats_100km = stats.sel(boundary=100)
        stats_150km = stats.sel(boundary=150)
        stats_200km = stats.sel(boundary=200)
        for key_no in range(num_keys):  # Iterate through the data variables (statistics)
            statistics_array_50km[file_no, key_no, :] = stats_50km[ds_keys[key_no]].values
            statistics_array_100km[file_no, key_no, :] = stats_100km[ds_keys[key_no]].values
            statistics_array_150km[file_no, key_no, :] = stats_150km[ds_keys[key_no]].values
            statistics_array_200km[file_no, key_no, :] = stats_200km[ds_keys[key_no]].values

    # Split datasets by boundary
    stats_50km = total_stats.sel(boundary=50)
    stats_100km = total_stats.sel(boundary=100)
    stats_150km = total_stats.sel(boundary=150)
    stats_200km = total_stats.sel(boundary=200)
    stats_250km = total_stats.sel(boundary=250)

    num_front_types = model_properties['classes'] - 1  # model_properties['classes'] - 1 ===> number of front types (we ignore the 'no front' type)

    """
    Probability of Detection (POD) and Success Ratio (SR) 

    Shape of arrays: (number of front types, num_iterations, number of thresholds)
    """
    POD_array_50km = np.empty([num_front_types, num_iterations, 100])
    POD_array_100km = np.empty([num_front_types, num_iterations, 100])
    POD_array_150km = np.empty([num_front_types, num_iterations, 100])
    POD_array_200km = np.empty([num_front_types, num_iterations, 100])
    SR_array_50km = np.empty([num_front_types, num_iterations, 100])
    SR_array_100km = np.empty([num_front_types, num_iterations, 100])
    SR_array_150km = np.empty([num_front_types, num_iterations, 100])
    SR_array_200km = np.empty([num_front_types, num_iterations, 100])

    if bootstrap is True:
        """
        Confidence Interval (CI) for POD and SR
        
        Shape of arrays: (number of front types, number of boundaries, number of thresholds)
        """
        CI_lower_POD = np.empty([model_properties['classes'] - 1, 4, 100])
        CI_lower_SR = np.empty([model_properties['classes'] - 1, 4, 100])
        CI_upper_POD = np.empty([model_properties['classes'] - 1, 4, 100])
        CI_upper_SR = np.empty([model_properties['classes'] - 1, 4, 100])

        for iteration in range(num_iterations):
            print(f"Iteration {iteration}/{num_iterations}", end='\r')
            indices = random.choices(range(num_files), k=num_files)  # Select a sample equal to the total number of files
            for front_type in range(num_front_types):
                true_positive_index = front_type
                false_positive_index = num_front_types + front_type
                true_negative_index = num_front_types*2 + front_type
                false_negative_index = num_front_types*3 + front_type

                POD_array_50km[front_type, iteration, :] = np.nan_to_num(np.divide(np.sum(statistics_array_50km[indices, true_positive_index, :], axis=0),
                                                                            np.add(np.sum(statistics_array_50km[indices, true_positive_index, :], axis=0),
                                                                                   np.sum(statistics_array_50km[indices, false_negative_index, :], axis=0))))
                SR_array_50km[front_type, iteration, :] = np.nan_to_num(np.divide(np.sum(statistics_array_50km[indices, true_positive_index, :], axis=0),
                                                                           np.add(np.sum(statistics_array_50km[indices, true_positive_index, :], axis=0),
                                                                                  np.sum(statistics_array_50km[indices, false_positive_index, :], axis=0))))
                POD_array_100km[front_type, iteration, :] = np.nan_to_num(np.divide(np.sum(statistics_array_100km[indices, true_positive_index, :], axis=0),
                                                                             np.add(np.sum(statistics_array_100km[indices, true_positive_index, :], axis=0),
                                                                                    np.sum(statistics_array_100km[indices, false_negative_index, :], axis=0))))
                SR_array_100km[front_type, iteration, :] = np.nan_to_num(np.divide(np.sum(statistics_array_100km[indices, true_positive_index, :], axis=0),
                                                                            np.add(np.sum(statistics_array_100km[indices, true_positive_index, :], axis=0),
                                                                                   np.sum(statistics_array_100km[indices, false_positive_index, :], axis=0))))
                POD_array_150km[front_type, iteration, :] = np.nan_to_num(np.divide(np.sum(statistics_array_150km[indices, true_positive_index, :], axis=0),
                                                                             np.add(np.sum(statistics_array_150km[indices, true_positive_index, :], axis=0),
                                                                                    np.sum(statistics_array_150km[indices, false_negative_index, :], axis=0))))
                SR_array_150km[front_type, iteration, :] = np.nan_to_num(np.divide(np.sum(statistics_array_150km[indices, true_positive_index, :], axis=0),
                                                                            np.add(np.sum(statistics_array_150km[indices, true_positive_index, :], axis=0),
                                                                                   np.sum(statistics_array_150km[indices, false_positive_index, :], axis=0))))
                POD_array_200km[front_type, iteration, :] = np.nan_to_num(np.divide(np.sum(statistics_array_200km[indices, true_positive_index, :], axis=0),
                                                                             np.add(np.sum(statistics_array_200km[indices, true_positive_index, :], axis=0),
                                                                                    np.sum(statistics_array_200km[indices, false_negative_index, :], axis=0))))
                SR_array_200km[front_type, iteration, :] = np.nan_to_num(np.divide(np.sum(statistics_array_200km[indices, true_positive_index, :], axis=0),
                                                                            np.add(np.sum(statistics_array_200km[indices, true_positive_index, :], axis=0),
                                                                                   np.sum(statistics_array_200km[indices, false_positive_index, :], axis=0))))

        # Calculate 95% confidence intervals
        for front_type in range(num_front_types):
            for percent in np.arange(0, 100):
                # 50km
                CI_lower_POD[front_type, 0, percent] = np.percentile(POD_array_50km[front_type, :, percent], q=2.5)
                CI_upper_POD[front_type, 0, percent] = np.percentile(POD_array_50km[front_type, :, percent], q=97.5)
                CI_lower_SR[front_type, 0, percent] = np.percentile(SR_array_50km[front_type, :, percent], q=2.5)
                CI_upper_SR[front_type, 0, percent] = np.percentile(SR_array_50km[front_type, :, percent], q=97.5)

                # 100km
                CI_lower_POD[front_type, 1, percent] = np.percentile(POD_array_100km[front_type, :, percent], q=2.5)
                CI_upper_POD[front_type, 1, percent] = np.percentile(POD_array_100km[front_type, :, percent], q=97.5)
                CI_lower_SR[front_type, 1, percent] = np.percentile(SR_array_100km[front_type, :, percent], q=2.5)
                CI_upper_SR[front_type, 1, percent] = np.percentile(SR_array_100km[front_type, :, percent], q=97.5)

                # 150km
                CI_lower_POD[front_type, 2, percent] = np.percentile(POD_array_150km[front_type, :, percent], q=2.5)
                CI_upper_POD[front_type, 2, percent] = np.percentile(POD_array_150km[front_type, :, percent], q=97.5)
                CI_lower_SR[front_type, 2, percent] = np.percentile(SR_array_150km[front_type, :, percent], q=2.5)
                CI_upper_SR[front_type, 2, percent] = np.percentile(SR_array_150km[front_type, :, percent], q=97.5)

                # 200km
                CI_lower_POD[front_type, 3, percent] = np.percentile(POD_array_200km[front_type, :, percent], q=2.5)
                CI_upper_POD[front_type, 3, percent] = np.percentile(POD_array_200km[front_type, :, percent], q=97.5)
                CI_lower_SR[front_type, 3, percent] = np.percentile(SR_array_200km[front_type, :, percent], q=2.5)
                CI_upper_SR[front_type, 3, percent] = np.percentile(SR_array_200km[front_type, :, percent], q=97.5)

    # Code for performance diagram matrices sourced from Ryan Lagerquist's (lagerqui@ualberta.ca) thunderhoser repository:
    # https://github.com/thunderhoser/GewitterGefahr/blob/master/gewittergefahr/plotting/model_eval_plotting.py
    success_ratio_matrix, pod_matrix = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    x, y = np.meshgrid(success_ratio_matrix, pod_matrix)
    csi_matrix = (x ** -1 + y ** -1 - 1.) ** -1
    CSI_LEVELS = np.linspace(0, 1, 11)
    cmap = 'Blues'
    axis_ticks = np.arange(0, 1.1, 0.1)

    if front_types == ['CF', 'WF'] or front_types == 'CFWFSFOF':
        CSI_cold_50km = np.nan_to_num(stats_50km['tp_cold']/(stats_50km['tp_cold'] + stats_50km['fp_cold'] + stats_50km['fn_cold']))
        CSI_cold_100km = np.nan_to_num(stats_100km['tp_cold']/(stats_100km['tp_cold'] + stats_100km['fp_cold'] + stats_100km['fn_cold']))
        CSI_cold_150km = np.nan_to_num(stats_150km['tp_cold']/(stats_150km['tp_cold'] + stats_150km['fp_cold'] + stats_150km['fn_cold']))
        CSI_cold_200km = np.nan_to_num(stats_200km['tp_cold']/(stats_200km['tp_cold'] + stats_200km['fp_cold'] + stats_200km['fn_cold']))
        CSI_cold_250km = np.nan_to_num(stats_250km['tp_cold']/(stats_250km['tp_cold'] + stats_250km['fp_cold'] + stats_250km['fn_cold']))
        CSI_warm_50km = np.nan_to_num(stats_50km['tp_warm']/(stats_50km['tp_warm'] + stats_50km['fp_warm'] + stats_50km['fn_warm']))
        CSI_warm_100km = np.nan_to_num(stats_100km['tp_warm']/(stats_100km['tp_warm'] + stats_100km['fp_warm'] + stats_100km['fn_warm']))
        CSI_warm_150km = np.nan_to_num(stats_150km['tp_warm']/(stats_150km['tp_warm'] + stats_150km['fp_warm'] + stats_150km['fn_warm']))
        CSI_warm_200km = np.nan_to_num(stats_200km['tp_warm']/(stats_200km['tp_warm'] + stats_200km['fp_warm'] + stats_200km['fn_warm']))
        CSI_warm_250km = np.nan_to_num(stats_250km['tp_warm']/(stats_250km['tp_warm'] + stats_250km['fp_warm'] + stats_250km['fn_warm']))

        # Probability of detection for each front type and boundary
        POD_cold_50km = np.nan_to_num(stats_50km['tp_cold']/(stats_50km['tp_cold'] + stats_50km['fn_cold']))
        POD_cold_100km = np.nan_to_num(stats_100km['tp_cold']/(stats_100km['tp_cold'] + stats_100km['fn_cold']))
        POD_cold_150km = np.nan_to_num(stats_150km['tp_cold']/(stats_150km['tp_cold'] + stats_150km['fn_cold']))
        POD_cold_200km = np.nan_to_num(stats_200km['tp_cold']/(stats_200km['tp_cold'] + stats_200km['fn_cold']))
        POD_warm_50km = np.nan_to_num(stats_50km['tp_warm']/(stats_50km['tp_warm'] + stats_50km['fn_warm']))
        POD_warm_100km = np.nan_to_num(stats_100km['tp_warm']/(stats_100km['tp_warm'] + stats_100km['fn_warm']))
        POD_warm_150km = np.nan_to_num(stats_150km['tp_warm']/(stats_150km['tp_warm'] + stats_150km['fn_warm']))
        POD_warm_200km = np.nan_to_num(stats_200km['tp_warm']/(stats_200km['tp_warm'] + stats_200km['fn_warm']))

        # Success ratio for each front type and boundary
        SR_cold_50km = np.nan_to_num(stats_50km['tp_cold']/(stats_50km['tp_cold'] + stats_50km['fp_cold']))
        SR_cold_100km = np.nan_to_num(stats_100km['tp_cold']/(stats_100km['tp_cold'] + stats_100km['fp_cold']))
        SR_cold_150km = np.nan_to_num(stats_150km['tp_cold']/(stats_150km['tp_cold'] + stats_150km['fp_cold']))
        SR_cold_200km = np.nan_to_num(stats_200km['tp_cold']/(stats_200km['tp_cold'] + stats_200km['fp_cold']))
        SR_warm_50km = np.nan_to_num(stats_50km['tp_warm']/(stats_50km['tp_warm'] + stats_50km['fp_warm']))
        SR_warm_100km = np.nan_to_num(stats_100km['tp_warm']/(stats_100km['tp_warm'] + stats_100km['fp_warm']))
        SR_warm_150km = np.nan_to_num(stats_150km['tp_warm']/(stats_150km['tp_warm'] + stats_150km['fp_warm']))
        SR_warm_200km = np.nan_to_num(stats_200km['tp_warm']/(stats_200km['tp_warm'] + stats_200km['fp_warm']))

        # SR and POD values where CSI is maximized for each front type and boundary
        SR_cold_50km_CSI = SR_cold_50km[np.where(CSI_cold_50km == np.max(CSI_cold_50km))]
        POD_cold_50km_CSI = POD_cold_50km[np.where(CSI_cold_50km == np.max(CSI_cold_50km))]
        SR_cold_100km_CSI = SR_cold_100km[np.where(CSI_cold_100km == np.max(CSI_cold_100km))]
        POD_cold_100km_CSI = POD_cold_100km[np.where(CSI_cold_100km == np.max(CSI_cold_100km))]
        SR_cold_150km_CSI = SR_cold_150km[np.where(CSI_cold_150km == np.max(CSI_cold_150km))]
        POD_cold_150km_CSI = POD_cold_150km[np.where(CSI_cold_150km == np.max(CSI_cold_150km))]
        SR_cold_200km_CSI = SR_cold_200km[np.where(CSI_cold_200km == np.max(CSI_cold_200km))]
        POD_cold_200km_CSI = POD_cold_200km[np.where(CSI_cold_200km == np.max(CSI_cold_200km))]
        SR_warm_50km_CSI = SR_warm_50km[np.where(CSI_warm_50km == np.max(CSI_warm_50km))]
        POD_warm_50km_CSI = POD_warm_50km[np.where(CSI_warm_50km == np.max(CSI_warm_50km))]
        SR_warm_100km_CSI = SR_warm_100km[np.where(CSI_warm_100km == np.max(CSI_warm_100km))]
        POD_warm_100km_CSI = POD_warm_100km[np.where(CSI_warm_100km == np.max(CSI_warm_100km))]
        SR_warm_150km_CSI = SR_warm_150km[np.where(CSI_warm_150km == np.max(CSI_warm_150km))]
        POD_warm_150km_CSI = POD_warm_150km[np.where(CSI_warm_150km == np.max(CSI_warm_150km))]
        SR_warm_200km_CSI = SR_warm_200km[np.where(CSI_warm_200km == np.max(CSI_warm_200km))]
        POD_warm_200km_CSI = POD_warm_200km[np.where(CSI_warm_200km == np.max(CSI_warm_200km))]

        # First index where the SR or POD go to zero. We will stop plotting the CSI curves at this index
        cold_50km_stop = np.min([np.where(SR_cold_50km == 0), np.where(POD_cold_50km == 0)])
        cold_100km_stop = np.min([np.where(SR_cold_100km == 0), np.where(POD_cold_100km == 0)])
        cold_150km_stop = np.min([np.where(SR_cold_150km == 0), np.where(POD_cold_150km == 0)])
        cold_200km_stop = np.min([np.where(SR_cold_200km == 0), np.where(POD_cold_200km == 0)])
        warm_50km_stop = np.min([np.where(SR_warm_50km == 0), np.where(POD_warm_50km == 0)])
        warm_100km_stop = np.min([np.where(SR_warm_100km == 0), np.where(POD_warm_100km == 0)])
        warm_150km_stop = np.min([np.where(SR_warm_150km == 0), np.where(POD_warm_150km == 0)])
        warm_200km_stop = np.min([np.where(SR_warm_200km == 0), np.where(POD_warm_200km == 0)])

        ###############################################################################################################
        ########################################### Cold front CSI diagram ############################################
        ###############################################################################################################

        plt.figure()
        plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)  # Plot CSI contours in 0.1 increments
        plt.colorbar(label='Critical Success Index (CSI)')

        # CSI lines for each boundary
        plt.plot(SR_cold_50km, POD_cold_50km, color='red', linewidth=1)
        plt.plot(SR_cold_100km, POD_cold_100km, color='purple', linewidth=1)
        plt.plot(SR_cold_150km, POD_cold_150km, color='brown', linewidth=1)
        plt.plot(SR_cold_200km, POD_cold_200km, color='green', linewidth=1)

        # Mark points of maximum CSI for each boundary
        plt.plot(SR_cold_50km_CSI, POD_cold_50km_CSI, color='red', marker='*', markersize=9)
        plt.plot(SR_cold_100km_CSI, POD_cold_100km_CSI, color='purple', marker='*', markersize=9)
        plt.plot(SR_cold_150km_CSI, POD_cold_150km_CSI, color='brown', marker='*', markersize=9)
        plt.plot(SR_cold_200km_CSI, POD_cold_200km_CSI, color='green', marker='*', markersize=9)

        if bootstrap is True:
            cold_index = 0
            """ 
            Some of the percentage thresholds have no data and will show up as zero in the CIs, so we not include them when interpolating the CIs.
            These arrays have four values: [50km boundary, 100km, 150km, 200km]
            These also represent the first index where data is missing, so only values before the index will be included 
            """
            CI_zero_index = np.min([np.min(np.where(CI_lower_SR[cold_index, 0] == 0)[0]), np.min(np.where(CI_lower_SR[cold_index, 1] == 0)[0]),
                            np.min(np.where(CI_lower_SR[cold_index, 2] == 0)[0]), np.min(np.where(CI_lower_SR[cold_index, 3] == 0)[0]),
                            np.min(np.where(CI_upper_SR[cold_index, 0] == 0)[0]), np.min(np.where(CI_upper_SR[cold_index, 1] == 0)[0]),
                            np.min(np.where(CI_upper_SR[cold_index, 2] == 0)[0]), np.min(np.where(CI_upper_SR[cold_index, 3] == 0)[0]),
                            np.min(np.where(CI_lower_POD[cold_index, 0] == 0)[0]), np.min(np.where(CI_lower_POD[cold_index, 1] == 0)[0]),
                            np.min(np.where(CI_lower_POD[cold_index, 2] == 0)[0]), np.min(np.where(CI_lower_POD[cold_index, 3] == 0)[0]),
                            np.min(np.where(CI_upper_POD[cold_index, 0] == 0)[0]), np.min(np.where(CI_upper_POD[cold_index, 1] == 0)[0]),
                            np.min(np.where(CI_upper_POD[cold_index, 2] == 0)[0]), np.min(np.where(CI_upper_POD[cold_index, 3] == 0)[0])])

            # 50km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[cold_index, 0, :CI_zero_index], CI_upper_SR[cold_index, 0, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[cold_index, 0, :CI_zero_index], CI_upper_POD[cold_index, 0, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='red')

            # 100km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[cold_index, 1, :CI_zero_index], CI_upper_SR[cold_index, 1, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[cold_index, 1, :CI_zero_index], CI_upper_POD[cold_index, 1, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='purple')

            # 150km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[cold_index, 2, :CI_zero_index], CI_upper_SR[cold_index, 2, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[cold_index, 2, :CI_zero_index], CI_upper_POD[cold_index, 2, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='brown')

            # 200km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[cold_index, 3, :CI_zero_index], CI_upper_SR[cold_index, 3, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[cold_index, 3, :CI_zero_index], CI_upper_POD[cold_index, 3, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='green')

        # Plot CSI scores on a white rectangle in the upper-left portion of the CSI plot
        rectangle = plt.Rectangle((0, 0.795), 0.25, 0.205, facecolor='white', edgecolor='black', zorder=3)
        plt.gca().add_patch(rectangle)
        plt.text(0.005, 0.96, s='CSI scores (*)', style='oblique')
        plt.text(0.005, 0.92, s=str('50km: %.3f' % np.max(CSI_cold_50km)), color='red')
        plt.text(0.005, 0.88, s=str('100km: %.3f' % np.max(CSI_cold_100km)), color='purple')
        plt.text(0.005, 0.84, s=str('150km: %.3f' % np.max(CSI_cold_150km)), color='brown')
        plt.text(0.005, 0.80, s=str('200km: %.3f' % np.max(CSI_cold_200km)), color='green')
        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.yticks(axis_ticks)
        plt.xticks(axis_ticks)
        plt.grid(color='black', alpha=0.1)
        plt.title("3D CF/WF model performance (3x3x3 kernel): Cold fronts")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # plt.savefig("%s/model_%d/%s_performance_cold.png" % (model_dir, model_number, stats_plot_base), bbox_inches='tight')
        plt.close()

        ###############################################################################################################
        ########################################### Warm front CSI diagram ############################################
        ###############################################################################################################

        plt.figure()
        plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
        plt.colorbar(label='Critical Success Index (CSI)')

        # CSI lines for each boundary
        plt.plot(SR_warm_50km, POD_warm_50km, color='red', linewidth=1)
        plt.plot(SR_warm_100km, POD_warm_100km, color='purple', linewidth=1)
        plt.plot(SR_warm_150km, POD_warm_150km, color='brown', linewidth=1)
        plt.plot(SR_warm_200km, POD_warm_200km, color='green', linewidth=1)

        if bootstrap is True:
            warm_index = 1
            """ 
            Some of the percentage thresholds have no data and will show up as zero in the CIs, so we not include them when interpolating the CIs.
            These arrays have four values: [50km boundary, 100km, 150km, 200km]
            These also represent the first index where data is missing, so only values before the index will be included 
            """
            CI_zero_index = np.min([np.min(np.where(CI_lower_SR[warm_index, 0] == 0)[0]), np.min(np.where(CI_lower_SR[warm_index, 1] == 0)[0]),
                            np.min(np.where(CI_lower_SR[warm_index, 2] == 0)[0]), np.min(np.where(CI_lower_SR[warm_index, 3] == 0)[0]),
                            np.min(np.where(CI_upper_SR[warm_index, 0] == 0)[0]), np.min(np.where(CI_upper_SR[warm_index, 1] == 0)[0]),
                            np.min(np.where(CI_upper_SR[warm_index, 2] == 0)[0]), np.min(np.where(CI_upper_SR[warm_index, 3] == 0)[0]),
                            np.min(np.where(CI_lower_POD[warm_index, 0] == 0)[0]), np.min(np.where(CI_lower_POD[warm_index, 1] == 0)[0]),
                            np.min(np.where(CI_lower_POD[warm_index, 2] == 0)[0]), np.min(np.where(CI_lower_POD[warm_index, 3] == 0)[0]),
                            np.min(np.where(CI_upper_POD[warm_index, 0] == 0)[0]), np.min(np.where(CI_upper_POD[warm_index, 1] == 0)[0]),
                            np.min(np.where(CI_upper_POD[warm_index, 2] == 0)[0]), np.min(np.where(CI_upper_POD[warm_index, 3] == 0)[0])])

            # 50km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[warm_index, 0, :CI_zero_index], CI_upper_SR[warm_index, 0, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[warm_index, 0, :CI_zero_index], CI_upper_POD[warm_index, 0, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='red')

            # 100km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[warm_index, 1, :CI_zero_index], CI_upper_SR[warm_index, 1, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[warm_index, 1, :CI_zero_index], CI_upper_POD[warm_index, 1, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='purple')

            # 150km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[warm_index, 2, :CI_zero_index], CI_upper_SR[warm_index, 2, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[warm_index, 2, :CI_zero_index], CI_upper_POD[warm_index, 2, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='brown')

            # 200km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[warm_index, 3, :CI_zero_index], CI_upper_SR[warm_index, 3, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[warm_index, 3, :CI_zero_index], CI_upper_POD[warm_index, 3, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='green')

        # Mark points of maximum CSI for each boundary
        plt.plot(SR_warm_50km_CSI, POD_warm_50km_CSI, color='red', marker='*', markersize=9)
        plt.plot(SR_warm_100km_CSI, POD_warm_100km_CSI, color='purple', marker='*', markersize=9)
        plt.plot(SR_warm_150km_CSI, POD_warm_150km_CSI, color='brown', marker='*', markersize=9)
        plt.plot(SR_warm_200km_CSI, POD_warm_200km_CSI, color='green', marker='*', markersize=9)

        # Plot CSI scores on a white rectangle in the upper-left portion of the CSI plot
        rectangle = plt.Rectangle((0, 0.795), 0.25, 0.205, facecolor='white', edgecolor='black', zorder=3)
        plt.gca().add_patch(rectangle)
        plt.text(0.005, 0.96, s='CSI scores (*)', style='oblique')
        plt.text(0.005, 0.92, s=str('50km: %.3f' % np.max(CSI_warm_50km)), color='red')
        plt.text(0.005, 0.88, s=str('100km: %.3f' % np.max(CSI_warm_100km)), color='purple')
        plt.text(0.005, 0.84, s=str('150km: %.3f' % np.max(CSI_warm_150km)), color='brown')
        plt.text(0.005, 0.80, s=str('200km: %.3f' % np.max(CSI_warm_200km)), color='green')

        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.yticks(axis_ticks)
        plt.xticks(axis_ticks)
        plt.grid(color='black', alpha=0.1)
        plt.title("3D CF/WF model performance (3x3x3 kernel): Warm fronts")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig("%s/model_%d/%s_performance_warm.png" % (model_dir, model_number, stats_plot_base), bbox_inches='tight')
        plt.close()

    if front_types == ['SF', 'OF'] or front_types == 'CFWFSFOF':
        CSI_stationary_50km = stats_50km['tp_stationary']/(stats_50km['tp_stationary'] + stats_50km['fp_stationary'] + stats_50km['fn_stationary']).values
        CSI_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fp_stationary'] + stats_50km['fn_stationary']).values
        CSI_stationary_150km = stats_150km['tp_stationary']/(stats_150km['tp_stationary'] + stats_150km['fp_stationary'] + stats_50km['fn_stationary']).values
        CSI_stationary_200km = stats_200km['tp_stationary']/(stats_200km['tp_stationary'] + stats_200km['fp_stationary'] + stats_50km['fn_stationary']).values
        CSI_stationary_250km = stats_250km['tp_stationary']/(stats_250km['tp_stationary'] + stats_250km['fp_stationary'] + stats_50km['fn_stationary']).values
        CSI_occluded_50km = stats_50km['tp_occluded']/(stats_50km['tp_occluded'] + stats_50km['fp_occluded'] + stats_50km['fn_occluded']).values
        CSI_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fp_occluded'] + stats_50km['fn_occluded']).values
        CSI_occluded_150km = stats_150km['tp_occluded']/(stats_150km['tp_occluded'] + stats_150km['fp_occluded'] + stats_50km['fn_occluded']).values
        CSI_occluded_200km = stats_200km['tp_occluded']/(stats_200km['tp_occluded'] + stats_200km['fp_occluded'] + stats_50km['fn_occluded']).values
        CSI_occluded_250km = stats_250km['tp_occluded']/(stats_250km['tp_occluded'] + stats_250km['fp_occluded'] + stats_50km['fn_occluded']).values

        # Probability of detection for each front type and boundary
        POD_stationary_50km = stats_50km['tp_stationary']/(stats_50km['tp_stationary'] + stats_50km['fn_stationary'])
        POD_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_50km['fn_stationary'])
        POD_stationary_150km = stats_150km['tp_stationary']/(stats_150km['tp_stationary'] + stats_50km['fn_stationary'])
        POD_stationary_200km = stats_200km['tp_stationary']/(stats_200km['tp_stationary'] + stats_50km['fn_stationary'])
        POD_occluded_50km = stats_50km['tp_occluded']/(stats_50km['tp_occluded'] + stats_50km['fn_occluded'])
        POD_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_50km['fn_occluded'])
        POD_occluded_150km = stats_150km['tp_occluded']/(stats_150km['tp_occluded'] + stats_50km['fn_occluded'])
        POD_occluded_200km = stats_200km['tp_occluded']/(stats_200km['tp_occluded'] + stats_50km['fn_occluded'])

        # Success ratio for each front type and boundary
        SR_stationary_50km = stats_50km['tp_stationary']/(stats_50km['tp_stationary'] + stats_50km['fp_stationary'])
        SR_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fp_stationary'])
        SR_stationary_150km = stats_150km['tp_stationary']/(stats_150km['tp_stationary'] + stats_150km['fp_stationary'])
        SR_stationary_200km = stats_200km['tp_stationary']/(stats_200km['tp_stationary'] + stats_200km['fp_stationary'])
        SR_occluded_50km = stats_50km['tp_occluded']/(stats_50km['tp_occluded'] + stats_50km['fp_occluded'])
        SR_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fp_occluded'])
        SR_occluded_150km = stats_150km['tp_occluded']/(stats_150km['tp_occluded'] + stats_150km['fp_occluded'])
        SR_occluded_200km = stats_200km['tp_occluded']/(stats_200km['tp_occluded'] + stats_200km['fp_occluded'])

        # SR and POD values where CSI is maximized for each front type and boundary
        SR_stationary_50km_CSI = SR_stationary_50km[np.where(CSI_stationary_50km == np.max(CSI_stationary_50km))].values[0]
        POD_stationary_50km_CSI = POD_stationary_50km[np.where(CSI_stationary_50km == np.max(CSI_stationary_50km))].values[0]
        SR_stationary_100km_CSI = SR_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))].values[0]
        POD_stationary_100km_CSI = POD_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))].values[0]
        SR_stationary_150km_CSI = SR_stationary_150km[np.where(CSI_stationary_150km == np.max(CSI_stationary_150km))].values[0]
        POD_stationary_150km_CSI = POD_stationary_150km[np.where(CSI_stationary_150km == np.max(CSI_stationary_150km))].values[0]
        SR_stationary_200km_CSI = SR_stationary_200km[np.where(CSI_stationary_200km == np.max(CSI_stationary_200km))].values[0]
        POD_stationary_200km_CSI = POD_stationary_200km[np.where(CSI_stationary_200km == np.max(CSI_stationary_200km))].values[0]
        SR_occluded_50km_CSI = SR_occluded_50km[np.where(CSI_occluded_50km == np.max(CSI_occluded_50km))].values[0]
        POD_occluded_50km_CSI = POD_occluded_50km[np.where(CSI_occluded_50km == np.max(CSI_occluded_50km))].values[0]
        SR_occluded_100km_CSI = SR_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))].values[0]
        POD_occluded_100km_CSI = POD_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))].values[0]
        SR_occluded_150km_CSI = SR_occluded_150km[np.where(CSI_occluded_150km == np.max(CSI_occluded_150km))].values[0]
        POD_occluded_150km_CSI = POD_occluded_150km[np.where(CSI_occluded_150km == np.max(CSI_occluded_150km))].values[0]
        SR_occluded_200km_CSI = SR_occluded_200km[np.where(CSI_occluded_200km == np.max(CSI_occluded_200km))].values[0]
        POD_occluded_200km_CSI = POD_occluded_200km[np.where(CSI_occluded_200km == np.max(CSI_occluded_200km))].values[0]

        ###############################################################################################################
        ######################################## Stationary front CSI diagram #########################################
        ###############################################################################################################

        plt.figure()
        plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)  # Plot CSI contours in 0.1 increments
        plt.colorbar(label='Critical Success Index (CSI)')

        # CSI lines for each boundary
        plt.plot(SR_stationary_50km, POD_stationary_50km, color='red', linewidth=1)
        plt.plot(SR_stationary_100km, POD_stationary_100km, color='purple', linewidth=1)
        plt.plot(SR_stationary_150km, POD_stationary_150km, color='brown', linewidth=1)
        plt.plot(SR_stationary_200km, POD_stationary_200km, color='green', linewidth=1)

        # Mark points of maximum CSI for each boundary
        plt.plot(SR_stationary_50km_CSI, POD_stationary_50km_CSI, color='red', marker='*', markersize=9)
        plt.plot(SR_stationary_100km_CSI, POD_stationary_100km_CSI, color='purple', marker='*', markersize=9)
        plt.plot(SR_stationary_150km_CSI, POD_stationary_150km_CSI, color='brown', marker='*', markersize=9)
        plt.plot(SR_stationary_200km_CSI, POD_stationary_200km_CSI, color='green', marker='*', markersize=9)

        if bootstrap is True:
            if front_types == 'CFWFSFOF':
                stationary_index = 2
            else:
                stationary_index = 0

            """ 
            Some of the percentage thresholds have no data and will show up as zero in the CIs, so we not include them when interpolating the CIs.
            These arrays have four values: [50km boundary, 100km, 150km, 200km]
            These also represent the first index where data is missing, so only values before the index will be included 
            """
            CI_zero_index = np.min([np.min(np.where(CI_lower_SR[stationary_index, 0] == 0)[0]), np.min(np.where(CI_lower_SR[stationary_index, 1] == 0)[0]),
                            np.min(np.where(CI_lower_SR[stationary_index, 2] == 0)[0]), np.min(np.where(CI_lower_SR[stationary_index, 3] == 0)[0]),
                            np.min(np.where(CI_upper_SR[stationary_index, 0] == 0)[0]), np.min(np.where(CI_upper_SR[stationary_index, 1] == 0)[0]),
                            np.min(np.where(CI_upper_SR[stationary_index, 2] == 0)[0]), np.min(np.where(CI_upper_SR[stationary_index, 3] == 0)[0]),
                            np.min(np.where(CI_lower_POD[stationary_index, 0] == 0)[0]), np.min(np.where(CI_lower_POD[stationary_index, 1] == 0)[0]),
                            np.min(np.where(CI_lower_POD[stationary_index, 2] == 0)[0]), np.min(np.where(CI_lower_POD[stationary_index, 3] == 0)[0]),
                            np.min(np.where(CI_upper_POD[stationary_index, 0] == 0)[0]), np.min(np.where(CI_upper_POD[stationary_index, 1] == 0)[0]),
                            np.min(np.where(CI_upper_POD[stationary_index, 2] == 0)[0]), np.min(np.where(CI_upper_POD[stationary_index, 3] == 0)[0])])

            # 50km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[stationary_index, 0, :CI_zero_index], CI_upper_SR[stationary_index, 0, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[stationary_index, 0, :CI_zero_index], CI_upper_POD[stationary_index, 0, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='red')

            # 100km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[stationary_index, 1, :CI_zero_index], CI_upper_SR[stationary_index, 1, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[stationary_index, 1, :CI_zero_index], CI_upper_POD[stationary_index, 1, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='purple')

            # 150km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[stationary_index, 2, :CI_zero_index], CI_upper_SR[stationary_index, 2, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[stationary_index, 2, :CI_zero_index], CI_upper_POD[stationary_index, 2, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='brown')

            # 200km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[stationary_index, 3, :CI_zero_index], CI_upper_SR[stationary_index, 3, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[stationary_index, 3, :CI_zero_index], CI_upper_POD[stationary_index, 3, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='green')

        # Plot CSI scores on a white rectangle in the upper-left portion of the CSI plot
        rectangle = plt.Rectangle((0, 0.795), 0.25, 0.205, facecolor='white', edgecolor='black', zorder=3)
        plt.gca().add_patch(rectangle)
        plt.text(0.005, 0.96, s='CSI scores (*)', style='oblique')
        plt.text(0.005, 0.92, s=str('50km: %.3f' % np.max(CSI_stationary_50km)), color='red')
        plt.text(0.005, 0.88, s=str('100km: %.3f' % np.max(CSI_stationary_100km)), color='purple')
        plt.text(0.005, 0.84, s=str('150km: %.3f' % np.max(CSI_stationary_150km)), color='brown')
        plt.text(0.005, 0.80, s=str('200km: %.3f' % np.max(CSI_stationary_200km)), color='green')

        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.yticks(axis_ticks)
        plt.xticks(axis_ticks)
        plt.grid(color='black', alpha=0.1)
        plt.title("3D SF/OF model performance (3x3x3 kernel): Stationary fronts")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # plt.savefig("%s/model_%d/%s_performance_stationary.png" % (model_dir, model_number, stats_plot_base), bbox_inches='tight')
        plt.close()

        ###############################################################################################################
        ######################################### Occluded front CSI diagram ##########################################
        ###############################################################################################################

        plt.figure()
        plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
        plt.colorbar(label='Critical Success Index (CSI)')

        # CSI lines for each boundary
        plt.plot(SR_occluded_50km, POD_occluded_50km, color='red', linewidth=1)
        plt.plot(SR_occluded_100km, POD_occluded_100km, color='purple', linewidth=1)
        plt.plot(SR_occluded_150km, POD_occluded_150km, color='brown', linewidth=1)
        plt.plot(SR_occluded_200km, POD_occluded_200km, color='green', linewidth=1)

        if bootstrap is True:
            if front_types == 'CFWFSFOF':
                occluded_index = 3
            else:
                occluded_index = 1

            """ 
            Some of the percentage thresholds have no data and will show up as zero in the CIs, so we not include them when interpolating the CIs.
            These arrays have four values: [50km boundary, 100km, 150km, 200km]
            These also represent the first index where data is missing, so only values before the index will be included 
            """
            CI_zero_index = np.min([np.min(np.where(CI_lower_SR[occluded_index, 0] == 0)[0]), np.min(np.where(CI_lower_SR[occluded_index, 1] == 0)[0]),
                            np.min(np.where(CI_lower_SR[occluded_index, 2] == 0)[0]), np.min(np.where(CI_lower_SR[occluded_index, 3] == 0)[0]),
                            np.min(np.where(CI_upper_SR[occluded_index, 0] == 0)[0]), np.min(np.where(CI_upper_SR[occluded_index, 1] == 0)[0]),
                            np.min(np.where(CI_upper_SR[occluded_index, 2] == 0)[0]), np.min(np.where(CI_upper_SR[occluded_index, 3] == 0)[0]),
                            np.min(np.where(CI_lower_POD[occluded_index, 0] == 0)[0]), np.min(np.where(CI_lower_POD[occluded_index, 1] == 0)[0]),
                            np.min(np.where(CI_lower_POD[occluded_index, 2] == 0)[0]), np.min(np.where(CI_lower_POD[occluded_index, 3] == 0)[0]),
                            np.min(np.where(CI_upper_POD[occluded_index, 0] == 0)[0]), np.min(np.where(CI_upper_POD[occluded_index, 1] == 0)[0]),
                            np.min(np.where(CI_upper_POD[occluded_index, 2] == 0)[0]), np.min(np.where(CI_upper_POD[occluded_index, 3] == 0)[0])])

            # 50km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[occluded_index, 0, :CI_zero_index], CI_upper_SR[occluded_index, 0, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[occluded_index, 0, :CI_zero_index], CI_upper_POD[occluded_index, 0, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='red')

            # 100km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[occluded_index, 1, :CI_zero_index], CI_upper_SR[occluded_index, 1, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[occluded_index, 1, :CI_zero_index], CI_upper_POD[occluded_index, 1, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='purple')

            # 150km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[occluded_index, 2, :CI_zero_index], CI_upper_SR[occluded_index, 2, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[occluded_index, 2, :CI_zero_index], CI_upper_POD[occluded_index, 2, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='brown')

            # 200km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[occluded_index, 3, :CI_zero_index], CI_upper_SR[occluded_index, 3, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[occluded_index, 3, :CI_zero_index], CI_upper_POD[occluded_index, 3, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='green')

        # Mark points of maximum CSI for each boundary
        plt.plot(SR_occluded_50km_CSI, POD_occluded_50km_CSI, color='red', marker='*', markersize=9)
        plt.plot(SR_occluded_100km_CSI, POD_occluded_100km_CSI, color='purple', marker='*', markersize=9)
        plt.plot(SR_occluded_150km_CSI, POD_occluded_150km_CSI, color='brown', marker='*', markersize=9)
        plt.plot(SR_occluded_200km_CSI, POD_occluded_200km_CSI, color='green', marker='*', markersize=9)

        # Plot CSI scores on a white rectangle in the upper-left portion of the CSI plot
        rectangle = plt.Rectangle((0, 0.795), 0.25, 0.205, facecolor='white', edgecolor='black', zorder=3)
        plt.gca().add_patch(rectangle)
        plt.text(0.005, 0.96, s='CSI scores (*)', style='oblique')
        plt.text(0.005, 0.92, s=str('50km: %.3f' % np.max(CSI_occluded_50km)), color='red')
        plt.text(0.005, 0.88, s=str('100km: %.3f' % np.max(CSI_occluded_100km)), color='purple')
        plt.text(0.005, 0.84, s=str('150km: %.3f' % np.max(CSI_occluded_150km)), color='brown')
        plt.text(0.005, 0.80, s=str('200km: %.3f' % np.max(CSI_occluded_200km)), color='green')

        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.yticks(axis_ticks)
        plt.xticks(axis_ticks)
        plt.grid(color='black', alpha=0.1)
        plt.title("3D SF/OF model performance (3x3x3 kernel): Occluded fronts")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # plt.savefig("%s/model_%d/%s_performance_occluded.png" % (model_dir, model_number, stats_plot_base), bbox_inches='tight')
        plt.close()

    if front_types == 'F_BIN':
        CSI_front_50km = stats_50km['tp_front']/(stats_50km['tp_front'] + stats_50km['fp_front'] + stats_50km['fn_front']).values
        CSI_front_100km = stats_100km['tp_front']/(stats_100km['tp_front'] + stats_100km['fp_front'] + stats_100km['fn_front']).values
        CSI_front_150km = stats_150km['tp_front']/(stats_150km['tp_front'] + stats_150km['fp_front'] + stats_150km['fn_front']).values
        CSI_front_200km = stats_200km['tp_front']/(stats_200km['tp_front'] + stats_200km['fp_front'] + stats_200km['fn_front']).values
        CSI_front_250km = stats_250km['tp_front']/(stats_250km['tp_front'] + stats_250km['fp_front'] + stats_250km['fn_front']).values

        # Probability of detection
        POD_front_50km = stats_50km['tp_front']/(stats_50km['tp_front'] + stats_50km['fn_front'])
        POD_front_100km = stats_100km['tp_front']/(stats_100km['tp_front'] + stats_100km['fn_front'])
        POD_front_150km = stats_150km['tp_front']/(stats_150km['tp_front'] + stats_150km['fn_front'])
        POD_front_200km = stats_200km['tp_front']/(stats_200km['tp_front'] + stats_200km['fn_front'])

        # Success ratio
        SR_front_50km = stats_50km['tp_front']/(stats_50km['tp_front'] + stats_50km['fp_front'])
        SR_front_100km = stats_100km['tp_front']/(stats_100km['tp_front'] + stats_100km['fp_front'])
        SR_front_150km = stats_150km['tp_front']/(stats_150km['tp_front'] + stats_150km['fp_front'])
        SR_front_200km = stats_200km['tp_front']/(stats_200km['tp_front'] + stats_200km['fp_front'])

        # SR and POD values where CSI is maximized for each boundary
        SR_front_50km_CSI = SR_front_50km[np.where(CSI_front_50km == np.max(CSI_front_50km))].values[0]
        POD_front_50km_CSI = POD_front_50km[np.where(CSI_front_50km == np.max(CSI_front_50km))].values[0]
        SR_front_100km_CSI = SR_front_100km[np.where(CSI_front_100km == np.max(CSI_front_100km))].values[0]
        POD_front_100km_CSI = POD_front_100km[np.where(CSI_front_100km == np.max(CSI_front_100km))].values[0]
        SR_front_150km_CSI = SR_front_150km[np.where(CSI_front_150km == np.max(CSI_front_150km))].values[0]
        POD_front_150km_CSI = POD_front_150km[np.where(CSI_front_150km == np.max(CSI_front_150km))].values[0]
        SR_front_200km_CSI = SR_front_200km[np.where(CSI_front_200km == np.max(CSI_front_200km))].values[0]
        POD_front_200km_CSI = POD_front_200km[np.where(CSI_front_200km == np.max(CSI_front_200km))].values[0]

        ##########################################################################################################
        ######################################## F/NF CSI diagram ################################################
        ##########################################################################################################

        plt.figure()
        plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)  # Plot CSI contours in 0.1 increments
        plt.colorbar(label='Critical Success Index (CSI)')

        # CSI lines for each boundary
        plt.plot(SR_front_50km, POD_front_50km, color='red', linewidth=1)
        plt.plot(SR_front_100km, POD_front_100km, color='purple', linewidth=1)
        plt.plot(SR_front_150km, POD_front_150km, color='brown', linewidth=1)
        plt.plot(SR_front_200km, POD_front_200km, color='green', linewidth=1)

        # Mark points of maximum CSI for each boundary
        plt.plot(SR_front_50km_CSI, POD_front_50km_CSI, color='red', marker='*', markersize=9)
        plt.plot(SR_front_100km_CSI, POD_front_100km_CSI, color='purple', marker='*', markersize=9)
        plt.plot(SR_front_150km_CSI, POD_front_150km_CSI, color='brown', marker='*', markersize=9)
        plt.plot(SR_front_200km_CSI, POD_front_200km_CSI, color='green', marker='*', markersize=9)

        if bootstrap is True:
            front_index = 0
            """ 
            Some of the percentage thresholds have no data and will show up as zero in the CIs, so we not include them when interpolating the CIs.
            These arrays have four values: [50km boundary, 100km, 150km, 200km]
            These also represent the first index where data is missing, so only values before the index will be included 
            """
            CI_zero_index = np.min([np.min(np.where(CI_lower_SR[front_index, 0] == 0)[0]), np.min(np.where(CI_lower_SR[front_index, 1] == 0)[0]),
                            np.min(np.where(CI_lower_SR[front_index, 2] == 0)[0]), np.min(np.where(CI_lower_SR[front_index, 3] == 0)[0]),
                            np.min(np.where(CI_upper_SR[front_index, 0] == 0)[0]), np.min(np.where(CI_upper_SR[front_index, 1] == 0)[0]),
                            np.min(np.where(CI_upper_SR[front_index, 2] == 0)[0]), np.min(np.where(CI_upper_SR[front_index, 3] == 0)[0]),
                            np.min(np.where(CI_lower_POD[front_index, 0] == 0)[0]), np.min(np.where(CI_lower_POD[front_index, 1] == 0)[0]),
                            np.min(np.where(CI_lower_POD[front_index, 2] == 0)[0]), np.min(np.where(CI_lower_POD[front_index, 3] == 0)[0]),
                            np.min(np.where(CI_upper_POD[front_index, 0] == 0)[0]), np.min(np.where(CI_upper_POD[front_index, 1] == 0)[0]),
                            np.min(np.where(CI_upper_POD[front_index, 2] == 0)[0]), np.min(np.where(CI_upper_POD[front_index, 3] == 0)[0])])

            # 50km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[front_index, 0, :CI_zero_index], CI_upper_SR[front_index, 0, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[front_index, 0, :CI_zero_index], CI_upper_POD[front_index, 0, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='red')

            # 100km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[front_index, 1, :CI_zero_index], CI_upper_SR[front_index, 1, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[front_index, 1, :CI_zero_index], CI_upper_POD[front_index, 1, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='purple')

            # 150km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[front_index, 2, :CI_zero_index], CI_upper_SR[front_index, 2, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[front_index, 2, :CI_zero_index], CI_upper_POD[front_index, 2, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='brown')

            # 200km confidence interval polygon
            xs = np.concatenate([CI_lower_SR[front_index, 3, :CI_zero_index], CI_upper_SR[front_index, 3, :CI_zero_index][::-1]])
            ys = np.concatenate([CI_lower_POD[front_index, 3, :CI_zero_index], CI_upper_POD[front_index, 3, :CI_zero_index][::-1]])
            plt.fill(xs, ys, alpha=0.3, color='green')

        # Plot CSI scores on a white rectangle in the upper-left portion of the CSI plot
        rectangle = plt.Rectangle((0, 0.795), 0.25, 0.205, facecolor='white', edgecolor='black', zorder=3)
        plt.gca().add_patch(rectangle)
        plt.text(0.005, 0.96, s='CSI scores (*)', style='oblique')
        plt.text(0.005, 0.92, s=str('50km: %.3f' % np.max(CSI_front_50km)), color='red')
        plt.text(0.005, 0.88, s=str('100km: %.3f' % np.max(CSI_front_100km)), color='purple')
        plt.text(0.005, 0.84, s=str('150km: %.3f' % np.max(CSI_front_150km)), color='brown')
        plt.text(0.005, 0.80, s=str('200km: %.3f' % np.max(CSI_front_200km)), color='green')

        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.yticks(axis_ticks)
        plt.xticks(axis_ticks)
        plt.grid(color='black', alpha=0.1)
        plt.title("3D F/NF model performance (3x3x3 kernel)")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig("%s/model_%d/%s_performance_fronts.png" % (model_dir, model_number, stats_plot_base), bbox_inches='tight')
        plt.close()


def prediction_plot(fronts, probs_ds, time, model_number, model_dir, domain, subdir_base, filename_base, probability_mask_2D=0.05, probability_mask_3D=0.10):
    """
    Function that uses generated predictions to make probability maps along with the 'true' fronts and saves out the
    subplots.

    Parameters
    ----------
    fronts: Xarray DataArray
        - DataArray containing the 'true' front data.
    probs_ds: Xarray Dataset
        - Dataset containing prediction (probability) data for fronts.
    time: str
        - Timestring for the prediction plot title.
    model_number: int
        - Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        - Main directory for the models.
    domain: str
        - Domain of the data.
    subdir_base: str
        - Name for the map subdirectory.
    filename_base: str
        - Base of the map filename.
    probability_mask_2D: float
        - Mask for front probabilities with 2D models. Any probabilities smaller than this number will not be plotted.
        - Must be a multiple of 0.05, greater than 0, and no greater than 0.45.
    probability_mask_3D: float
        - Mask for front probabilities with 3D models. Any probabilities smaller than this number will not be plotted.
        - Must be a multiple of 0.1, greater than 0, and no greater than 0.9.
    """
    if domain == 'conus':
        extent = np.array([220, 300, 25, 52])  # Extent for CONUS
    else:
        extent = np.array([120, 380, 0, 80])  # Extent for full domain
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

    fronts, names, labels, colors_types, colors_probs = reformat_fronts(fronts, front_types, return_names=True, return_colors=True)
    fronts = expand_fronts(fronts)
    fronts = xr.where(fronts == 0, float('NaN'), fronts)

    probs_ds = xr.where(probs_ds > probability_mask, probs_ds, float("NaN"))

    cmap_front = colors.ListedColormap(colors_types, name='from_list', N=len(names))
    norm_front = colors.Normalize(vmin=1, vmax=len(names) + 1)

    for front_no, front_key, front_name, front_label, cmap in zip(range(1, len(names) + 1), list(probs_ds.keys()), names, labels, colors_probs):
        current_fronts = fronts
        current_fronts = xr.where(current_fronts != front_no, float("NaN"), front_no)
        fig, ax = plt.subplots(1, 1, figsize=(20, 8), subplot_kw={'projection': crs})
        plot_background(extent, ax=ax, linewidth=0.5)
        cmap_probs, norm_probs = cm.get_cmap(cmap, n_colors), colors.Normalize(vmin=0, vmax=vmax)
        probs_ds[front_key].isel().plot.contourf(ax=ax, x='longitude', y='latitude', norm=norm_probs, levels=levels, cmap=cmap_probs,
            transform=ccrs.PlateCarree(), alpha=0.75, add_colorbar=False)
        current_fronts['identifier'].plot(ax=ax, x='longitude', y='latitude', cmap=cmap_front, norm=norm_front, transform=ccrs.PlateCarree(), add_colorbar=False)
        ax.set_title(f'{front_name} predictions and ground truth: {time}')
        cbar_ax = fig.add_axes([0.8365, 0.11, 0.015, 0.77])
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm_probs, cmap=cmap_probs), cax=cbar_ax, boundaries=levels[1:], alpha=0.75)
        cbar.set_label('Probability', rotation=90)
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
        --domain conus --model_dir /home/my_model_folder --pickle_dir /home/pickle_files --model_number 6846496 --domain_images 3 1
        --domain_size 288 128 --dataset test
        =========================================================================================================================
        Required arguments: --generate_predictions, --model_number, --model_dir, --domain, --domain_images, --domain_size, 
                            --prediction_method, --pickle_dir
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
    parser.add_argument('--domain', type=str, help='Domain of the data.')
    parser.add_argument('--domain_images', type=int, nargs=2, help='Number of images for each dimension the final stitched map for predictions: lon, lat')
    parser.add_argument('--domain_size', type=int, nargs=2, help='Lengths of the dimensions of the final stitched map for predictions: lon, lat')
    parser.add_argument('--domain_trim', type=int, nargs=2, default=[0, 0],
                        help='Number of pixels to trim the images by along each dimension for stitching before taking the '
                             'maximum across overlapping pixels.')
    parser.add_argument('--find_matches', action='store_true', help='Find matches for stitching predictions?')
    parser.add_argument('--generate_predictions', action='store_true', help='Generate prediction plots?')
    parser.add_argument('--gpu_device', type=int, help='GPU device number.')
    parser.add_argument('--image_size', type=int, nargs=2, help="Number of pixels along each dimension of the model's output: lon, lat")
    parser.add_argument('--learning_curve', action='store_true', help='Plot learning curve')
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth on the GPU')
    parser.add_argument('--model_dir', type=str, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, help='Model number.')
    parser.add_argument('--num_iterations', type=int, default=10000, help='Number of iterations to perform when bootstrapping the data.')
    parser.add_argument('--num_rand_predictions', type=int, default=10, help='Number of random predictions to make.')
    parser.add_argument('--fronts_pickle_indir', type=str, help='Main directory for the pickle files containing frontal objects.')
    parser.add_argument('--variables_pickle_indir', type=str, help='Main directory for the pickle files containing variable data.')
    parser.add_argument('--plot_performance_diagrams', action='store_true', help='Plot performance diagrams for a model?')
    parser.add_argument('--prediction_method', type=str, help="Prediction method. Options are: 'datetime', 'random', 'all'")
    parser.add_argument('--random_variables', type=str, nargs="+", default=None, help="Variables to randomize when generating predictions.")
    parser.add_argument('--save_map', action='store_true', help='Save maps of the model predictions?')
    parser.add_argument('--save_probabilities', action='store_true', help='Save model prediction data out to pickle files?')
    parser.add_argument('--save_statistics', action='store_true', help='Save performance statistics data out to pickle files?')

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
        required_arguments = ['domain', 'model_number', 'model_dir', 'domain_images', 'domain_size', 'prediction_method', 'variables_pickle_indir', 'fronts_pickle_indir']
        check_arguments(provided_arguments, required_arguments)

        if args.prediction_method == 'datetime' and args.datetime is None:
            raise errors.MissingArgumentError("'datetime' argument must be passed: 'prediction_method' was set to 'datetime' ")

        model_properties = pd.read_pickle(f"{args.model_dir}/model_{args.model_number}/model_{args.model_number}_properties.pkl")
        image_size = model_properties['input_size'][0:2]  # We are only concerned about longitude and latitude when checking compatibility

        # Verify the compatibility of image stitching arguments
        find_matches_for_domain(args.domain_size, image_size, compatibility_mode=True, compat_images=args.domain_images)

        generate_predictions(args.model_number, args.model_dir, args.variables_pickle_indir, args.fronts_pickle_indir,
            args.domain, args.domain_images, args.domain_size, args.domain_trim, args.prediction_method, args.datetime,
            dataset=args.dataset, num_rand_predictions=args.num_rand_predictions, random_variables=args.random_variables,
            save_map=args.save_map, save_probabilities=args.save_probabilities, save_statistics=args.save_statistics)

    if args.plot_performance_diagrams:
        required_arguments = ['model_number', 'model_dir', 'domain', 'domain_images', 'num_iterations']
        check_arguments(provided_arguments, required_arguments)
        plot_performance_diagrams(args.model_dir, args.model_number, args.domain, args.domain_images, args.domain_trim,
            random_variables=args.random_variables, bootstrap=args.bootstrap, num_iterations=args.num_iterations)
