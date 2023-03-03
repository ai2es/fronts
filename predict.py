"""
Generate predictions with a U-Net model

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 3/17/2023 4:28 PM CT
"""

import argparse
import os.path
import pandas as pd
import numpy as np
import xarray as xr
import tensorflow as tf
from utils import data_utils, settings
from glob import glob
import custom_losses
import custom_metrics


def load_model(model_number, model_dir):
    """
    Load a saved model.

    Parameters
    ----------
    model_number: int
        - Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        - Main directory for the models.
    """

    model_path = f"{model_dir}/model_{model_number}/model_{model_number}.h5"
    model_properties = pd.read_pickle(f"{model_dir}/model_{model_number}/model_{model_number}_properties.pkl")

    try:
        loss_string = model_properties['loss_string']
    except KeyError:
        loss_string = model_properties['loss']  # Error in the training sometimes resulted in the incorrect key ('loss' instead of 'loss_string')
    loss_args = model_properties['loss_args']

    try:
        metric_string = model_properties['metric_string']
    except KeyError:
        metric_string = model_properties['metric']  # Error in the training sometimes resulted in the incorrect key ('metric' instead of 'metric_string')
    metric_args = model_properties['metric_args']

    custom_objects = {}

    if 'fss' in loss_string.lower():
        custom_objects[loss_string] = custom_losses.fractions_skill_score(**loss_args)

    if 'brier' in metric_string.lower():
        custom_objects[metric_string] = custom_metrics.brier_skill_score(**metric_args)

    if 'csi' in metric_string.lower():
        custom_objects[metric_string] = custom_metrics.critical_success_index(**metric_args)

    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


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

    for lon_images in range(1, domain_size[0]-image_size[0] + 2):  # Image counter for longitude dimension
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


def generate_predictions(model_number, model_dir, netcdf_indir, init_time, domain='full', domain_images='min',
    variable_data_source='gdas'):
    """
    Generate predictions with a model.

    Parameters
    ----------
    model_number: int
        - Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        - Main directory for the models.
    netcdf_indir: str
        - Input directory for the netcdf files.
    init_time: iterable object with 4 integers
        - 4 values for the date and time: year, month, day, hour
    domain: str
        - Domain over which the predictions will be made. Options are: 'conus', 'full'. Default is 'full'.
    domain_images: str or iterable object with 2 ints
        - If a string, valid values are 'min', 'balanced', 'max'. Default is 'min'.
            * 'min' uses the minimum possible number of images when making the predictions.
            * 'balanced' chooses the number of images that creates the closest possible balance of images in the longitude and latitude direction.
            * 'max' chooses the maximum number of images that can be stitched together (NOT RECOMMENDED)
        - If an iterable with 2 integers, values represent the number of images in the longitude and latitude dimensions.
    variable_data_source: str
        - Variable data to use for training the model. Options are: 'gdas', or 'gfs' (case-insensitive)
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

    num_dimensions = len(image_size)
    if model_number not in [7805504, 7866106, 7961517]:  # The model numbers in this list have 2D kernels
        num_dimensions += 1

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

    model = load_model(model_number, model_dir)

    ### Load variable files ###
    variable_files = glob(f'%s/{variable_data_source}_%d%02d%02d%02d_f*.nc' % (netcdf_indir, init_time[0], init_time[1], init_time[2], init_time[3]))

    dataset_kwargs = {'engine': 'netcdf4'}  # Keyword arguments for loading variable files with xarray
    coords_isel_kwargs = {'longitude': slice(domain_extent_indices[0], domain_extent_indices[1]), 'latitude': slice(domain_extent_indices[2], domain_extent_indices[3])}

    num_files = len(variable_files)

    num_chunks = int(np.ceil(num_files / settings.MAX_FILE_CHUNK_SIZE))  # Number of files/timesteps to process at once
    chunk_indices = np.linspace(0, num_files, num_chunks + 1, dtype=int)

    for chunk_no in range(num_chunks):

        files_in_chunk = variable_files[chunk_indices[chunk_no]:chunk_indices[chunk_no + 1]]
        print(f"Preparing chunk {chunk_no + 1}/{num_chunks}")
        variable_ds = xr.open_mfdataset(files_in_chunk, **dataset_kwargs).isel(**coords_isel_kwargs)[variables]

        variable_ds = variable_ds.sel(pressure_level=pressure_levels).transpose('time', 'forecast_hour', 'longitude', 'latitude', 'pressure_level')
        forecast_hours = variable_ds['forecast_hour'].values

        # Older 2D models were trained with the pressure levels not in the proper order
        if model_number in [7805504, 7866106, 7961517]:
            variable_ds = variable_ds.isel(pressure_level=[0, 4, 3, 2, 1])

        image_lats = variable_ds.latitude.values[:domain_size_lat]
        image_lons = variable_ds.longitude.values[:domain_size_lon]

        timestep_predict_size = settings.TIMESTEP_PREDICT_SIZE[domain]

        num_forecast_hours = len(forecast_hours)
        timestep_predict_size /= num_forecast_hours
        timestep_predict_size = int(timestep_predict_size)

        num_timesteps = len(variable_ds['time'].values)
        num_batches = int(np.ceil(num_timesteps / timestep_predict_size))

        for batch_no in range(num_batches):

            print(f"======== Chunk {chunk_no + 1}/{num_chunks}: batch {batch_no + 1}/{num_batches} ========")

            variable_batch_ds = variable_ds.isel(time=slice(batch_no * timestep_predict_size, (batch_no + 1) * timestep_predict_size))  # Select timesteps for the current batch
            variable_batch_ds = data_utils.normalize_variables(variable_batch_ds).astype('float16')

            timesteps = variable_batch_ds['time'].values
            num_timesteps_in_batch = len(timesteps)
            map_created = False  # Boolean that determines whether or not the final stitched map has been created

            stitched_map_probs = np.empty(shape=[num_timesteps_in_batch, len(forecast_hours), classes-1, domain_size_lon, domain_size_lat])

            for lat_image in range(domain_images_lat):
                lat_index = int(lat_image * lat_image_spacing)
                for lon_image in range(domain_images_lon):
                    print(f"image %d/%d" % (int(lat_image*domain_images_lon) + lon_image + 1, int(domain_images_lon*domain_images_lat)))
                    lon_index = int(lon_image * lon_image_spacing)

                    # Select the current image
                    variable_batch_ds_new = variable_batch_ds[variables].isel(longitude=slice(lon_index, lon_index + image_size[0]),
                                                                              latitude=slice(lat_index, lat_index + image_size[1])).to_array().values

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

                    ### Combine time and forecast hour into one dimension ###
                    gdas_variable_ds_new_shape = np.shape(variable_batch_ds_new)
                    variable_batch_ds_new = variable_batch_ds_new.reshape(gdas_variable_ds_new_shape[0] * gdas_variable_ds_new_shape[1], *[dim_size for dim_size in gdas_variable_ds_new_shape[2:]])

                    prediction = model.predict(variable_batch_ds_new, batch_size=settings.GPU_PREDICT_BATCH_SIZE, verbose=0)

                    num_dims_in_pred = len(np.shape(prediction))

                    if model_type == 'unet':
                        if num_dims_in_pred == 4:  # 2D labels, prediction shape: (time, lat, lon, front type)
                            image_probs = np.transpose(prediction[:, :, :, 1:], transpose_indices)  # transpose the predictions
                        else:  # if num_dims_in_pred == 5; 3D labels, prediction shape: (time, lat, lon, pressure level, front type)
                            image_probs = np.transpose(np.amax(prediction[:, :, :, :, 1:], axis=3), transpose_indices)  # Take the maximum probability over the vertical dimension and transpose the predictions

                    elif model_type == 'unet_3plus':
                        if num_dims_in_pred == 5:  # 2D labels, prediction shape: (output level, time, lon, lat, front type)
                            image_probs = np.transpose(prediction[0][:, :, :, 1:], transpose_indices)  # transpose the predictions
                        else:  # if num_dims_in_pred == 6; 3D labels, prediction shape: (output level, time, lat, lon, pressure level, front type)
                            image_probs = np.transpose(np.amax(prediction[0][:, :, :, :, 1:], axis=3), transpose_indices)  # Take the maximum probability over the vertical dimension and transpose the predictions

                    # Add predictions to the map
                    for timestep in range(num_timesteps_in_batch):
                        for fcst_hr_index in range(num_forecast_hours):
                            stitched_map_probs[timestep][fcst_hr_index], map_created = add_image_to_map(stitched_map_probs[timestep][fcst_hr_index], image_probs[timestep * num_forecast_hours + fcst_hr_index], map_created, domain_images_lon, domain_images_lat, lon_image, lat_image,
                                image_size_lon, image_size_lat, lon_image_spacing, lat_image_spacing)

                    ####################################################################################################

                    if map_created is True:

                        for timestep_no, timestep in enumerate(timesteps):
                            for fcst_hr_index, forecast_hour in enumerate(forecast_hours):
                                time = f'{init_time[0]}-%02d-%02d-%02dz' % (init_time[1], init_time[2], init_time[3])
                                probs_ds = create_model_prediction_dataset(stitched_map_probs[timestep_no][fcst_hr_index], image_lats, image_lons, front_types)
                                probs_ds = probs_ds.expand_dims({'time': np.atleast_1d(timestep)})
                                filename_base = f'model_%d_%s_{variable_data_source}_f%03d_%s_%dx%d' % (model_number, time, forecast_hour, domain, domain_images_lon, domain_images_lat)

                                if not os.path.isdir('%s/model_%d/predictions' % (model_dir, model_number)):
                                    os.mkdir('%s/model_%d/predictions' % (model_dir, model_number))
                                outfile = '%s/model_%d/predictions/%s_probabilities.nc' % (model_dir, model_number, filename_base)
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


if __name__ == '__main__':
    """
    All arguments listed in the examples are listed via argparse in alphabetical order below this comment block.
    """
    parser = argparse.ArgumentParser()

    ### Initialization time ###
    parser.add_argument('--init_time', type=int, nargs=4, required=True, help='Date and time of the data. Pass 4 ints in the following order: year, month, day, hour')

    ### Domain arguments ###
    parser.add_argument('--domain', type=str, required=True, help='Domain of the data.')
    parser.add_argument('--domain_images', type=int, nargs=2, help='Number of images for each dimension the final stitched map for predictions: lon, lat')

    ### Model / data arguments ###
    parser.add_argument('--netcdf_indir', type=str, required=True, help='Main directory for the netcdf files containing variable data.')
    parser.add_argument('--variable_data_source', type=str, required=True, help='Data source for variables (GDAS or GFS)')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, required=True, help='Model number.')
    parser.add_argument('--image_size', type=int, nargs=2, default=(128, 128), help='Image size used for predictions (lon, lat)')

    ### GPU arguments ###
    parser.add_argument('--gpu_device', type=int, help='GPU device number.')
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth on the GPU')

    args = parser.parse_args()
    provided_arguments = vars(args)

    ### Use a GPU ###
    if args.gpu_device is not None:
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        tf.config.experimental.set_visible_devices(devices=gpus[args.gpu_device], device_type='GPU')

        # Allow for memory growth on the GPU. This will only use the GPU memory that is required rather than allocating all of the GPU's memory.
        if args.memory_growth:
            tf.config.experimental.set_memory_growth(device=gpus[args.gpu_device], enable=True)

    model_properties = pd.read_pickle(f"{args.model_dir}/model_{args.model_number}/model_{args.model_number}_properties.pkl")
    image_size = args.image_size

    domain_indices = settings.DEFAULT_DOMAIN_INDICES[args.domain]
    if args.domain_images is None:
        domain_images = settings.DEFAULT_DOMAIN_IMAGES[args.domain]
    else:
        domain_images = args.domain_images

    # Verify the compatibility of image stitching arguments
    find_matches_for_domain((domain_indices[1] - domain_indices[0], domain_indices[3] - domain_indices[2]), image_size, compatibility_mode=True, compat_images=domain_images)

    generate_predictions(args.model_number, args.model_dir, args.netcdf_indir, args.init_time,
        domain=args.domain, domain_images=domain_images, variable_data_source=args.variable_data_source)
