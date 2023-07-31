"""
Generate predictions with a U-Net model

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 7/30/2023 6:18 PM CT
"""

import argparse
import os.path
import pandas as pd
import numpy as np
import xarray as xr
import tensorflow as tf
from utils import data_utils, settings
from glob import glob


def load_model(model_number: int, model_dir: str):
    """
    Load a saved model.

    Parameters
    ----------
    model_number: int
        Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        Main directory for the models.
    """

    ######################################### Check the parameters for errors ##########################################
    if not isinstance(model_number, int):
        raise TypeError(f"model_number must be an integer, received {type(model_number)}")
    if not isinstance(model_dir, str):
        raise TypeError(f"model_dir must be a string, received {type(model_dir)}")
    ####################################################################################################################

    from tensorflow.keras.models import load_model as lm
    import custom_losses
    import custom_metrics

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
        if model_number in [6846496, 7236500, 7507525]:
            loss_string = 'fss_loss'
        custom_objects[loss_string] = custom_losses.fractions_skill_score(**loss_args)

    if 'brier' in metric_string.lower() or 'bss' in metric_string.lower():
        if model_number in [6846496, 7236500, 7507525]:
            metric_string = 'bss'
        custom_objects[metric_string] = custom_metrics.brier_skill_score(**metric_args)

    if 'csi' in metric_string.lower():
        custom_objects[metric_string] = custom_metrics.critical_success_index(**metric_args)

    return lm(model_path, custom_objects=custom_objects)


def generate_predictions(model_number, model_dir, netcdf_indir, init_time, domain='full', variable_data_source='gdas'):
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
    variable_data_source: str
        - Variable data to use for training the model. Options are: 'gdas', or 'gfs' (case-insensitive)
    """

    variable_data_source = variable_data_source.lower()

    ### Model properties ###
    model_properties = pd.read_pickle(f"{model_dir}/model_{model_number}/model_{model_number}_properties.pkl")
    model_type = model_properties['model_type']
    front_types = model_properties['dataset_properties']['front_types']
    variables = model_properties['dataset_properties']['variables']
    pressure_levels = model_properties['dataset_properties']['pressure_levels']
    num_dims = model_properties['dataset_properties']['num_dims']

    lons = np.arange(settings.DEFAULT_DOMAIN_EXTENTS[domain][0], settings.DEFAULT_DOMAIN_EXTENTS[domain][1] + 0.25, 0.25)
    lats = np.arange(settings.DEFAULT_DOMAIN_EXTENTS[domain][2], settings.DEFAULT_DOMAIN_EXTENTS[domain][3] + 0.25, 0.25)[::-1]

    model = load_model(model_number, model_dir)

    ### Load variable files ###
    variable_files = glob(f'%s/{variable_data_source}_%d%02d%02d%02d_f*.nc' % (netcdf_indir, init_time[0], init_time[1], init_time[2], init_time[3]))

    dataset_kwargs = {'engine': 'netcdf4'}  # Keyword arguments for loading variable files with xarray

    num_files = len(variable_files)

    num_chunks = int(np.ceil(num_files / settings.MAX_FILE_CHUNK_SIZE))  # Number of files/timesteps to process at once
    chunk_indices = np.linspace(0, num_files, num_chunks + 1, dtype=int)

    for chunk_no in range(num_chunks):

        files_in_chunk = variable_files[chunk_indices[chunk_no]:chunk_indices[chunk_no + 1]]
        print(f"Preparing chunk {chunk_no + 1}/{num_chunks}")
        variable_ds = xr.open_mfdataset(files_in_chunk, **dataset_kwargs)[variables]

        variable_ds = variable_ds.sel(pressure_level=pressure_levels).transpose('time', 'forecast_hour', 'longitude', 'latitude', 'pressure_level')
        forecast_hours = variable_ds['forecast_hour'].values

        variable_ds = data_utils.normalize_variables(variable_ds).astype('float16')  # normalize variables, shrink to float16 to save memory

        variable_ds_new = variable_ds[variables].to_array().values  # convert the xarray dataset to a large numpy array
        variable_ds_new = variable_ds_new.transpose([1, 2, 3, 4, 5, 0])  # (time, forecast hour, longitude, latitude, pressure level, variable)

        ### If the model inputs are 2D, combine pressure levels and variables into one dimension ###
        if num_dims[0] == 2:
            variable_ds_new_shape = np.shape(variable_ds_new)
            variable_ds_new = variable_ds_new.reshape(*[dim_size for dim_size in variable_ds_new_shape[:-2]], variable_ds_new_shape[-2] * variable_ds_new_shape[-1])

        transpose_indices = (0, 3, 1, 2)  # New order of indices for model predictions (time, front type, longitude, latitude)

        ########################################### Generate the predictions ###########################################

        ### Combine time and forecast hour into one dimension ###
        gdas_variable_ds_new_shape = np.shape(variable_ds_new)
        variable_ds_new = variable_ds_new.reshape(gdas_variable_ds_new_shape[0] * gdas_variable_ds_new_shape[1], *[dim_size for dim_size in gdas_variable_ds_new_shape[2:]])

        prediction = model.predict(variable_ds_new, batch_size=settings.GPU_PREDICT_BATCH_SIZE, verbose=1)

        if model_type == 'unet':
            if num_dims[1] == 2:  # 2D labels, prediction shape: (time, lat, lon, front type)
                prediction = np.transpose(prediction[:, :, :, 1:], transpose_indices)  # transpose the predictions
            else:  # if num_dims[1] == 3; 3D labels, prediction shape: (time, lat, lon, pressure level, front type)
                prediction = np.transpose(np.amax(prediction[:, :, :, :, 1:], axis=3), transpose_indices)  # Take the maximum probability over the vertical dimension and transpose the predictions

        elif model_type == 'unet_3plus':
            if num_dims[1] == 2:  # 2D labels, prediction shape: (output level, time, lon, lat, front type)
                prediction = np.transpose(prediction[0][:, :, :, 1:], transpose_indices)  # transpose the predictions
            else:  # if num_dims[1] == 3; 3D labels, prediction shape: (output level, time, lat, lon, pressure level, front type)
                prediction = np.transpose(np.amax(prediction[0][:, :, :, :, 1:], axis=3), transpose_indices)  # Take the maximum probability over the vertical dimension and transpose the predictions

        ################################################################################################################

        for fcst_hr_index, forecast_hour in enumerate(forecast_hours):
            time = f'{init_time[0]}-%02d-%02d-%02dz' % (init_time[1], init_time[2], init_time[3])
            probs_ds = create_model_prediction_dataset(prediction[fcst_hr_index], lats, lons, front_types)
            probs_ds = probs_ds.expand_dims({'time': np.atleast_1d(variable_ds['time'].values),
                                             'forecast_hour': np.atleast_1d(forecast_hour)})
            filename_base = f'model_%d_%s_{variable_data_source}_f%03d_%s' % (model_number, time, forecast_hour, domain)

            if not os.path.isdir('%s/model_%d/predictions' % (model_dir, model_number)):
                os.mkdir('%s/model_%d/predictions' % (model_dir, model_number))
            outfile = '%s/model_%d/predictions/%s_probabilities.nc' % (model_dir, model_number, filename_base)
            probs_ds.to_netcdf(path=outfile, engine='netcdf4', mode='w')

            if args['calibration'] is not None:
                for front_type in front_types:
                    try:
                        ir_model = model_properties['calibration_models'][args['domain']][front_type]['%d km' % args['calibration']]
                    except KeyError:
                        ir_model = model_properties['calibration_models']['conus'][front_type]['%d km' % args['calibration']]
                    original_shape = np.shape(probs_ds[front_type].values)
                    probs_ds[front_type].values = ir_model.predict(probs_ds[front_type].values.flatten()).reshape(original_shape)

                outfile = '%s/model_%d/predictions/%s_probabilities_calibrated.nc' % (model_dir, model_number, filename_base)
                probs_ds.astype('float32').to_netcdf(path=outfile, engine='netcdf4', mode='w')


def create_model_prediction_dataset(prediction: np.array, lats, lons, front_types: str or list):
    """
    Create an Xarray dataset containing model predictions.

    Parameters
    ----------
    prediction: np.array
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
            {front_types: (('longitude', 'latitude'), prediction[0])},
            coords={'latitude': lats, 'longitude': lons})
    elif front_types == 'MERGED-F':
        probs_ds = xr.Dataset(
            {'CF_merged': (('longitude', 'latitude'), prediction[0]),
             'WF_merged': (('longitude', 'latitude'), prediction[1]),
             'SF_merged': (('longitude', 'latitude'), prediction[2]),
             'OF_merged': (('longitude', 'latitude'), prediction[3])},
            coords={'latitude': lats, 'longitude': lons})
    elif front_types == 'MERGED-ALL':
        probs_ds = xr.Dataset(
            {'CF_merged': (('longitude', 'latitude'), prediction[0]),
             'WF_merged': (('longitude', 'latitude'), prediction[1]),
             'SF_merged': (('longitude', 'latitude'), prediction[2]),
             'OF_merged': (('longitude', 'latitude'), prediction[3]),
             'TROF_merged': (('longitude', 'latitude'), prediction[4]),
             'INST': (('longitude', 'latitude'), prediction[5]),
             'DL': (('longitude', 'latitude'), prediction[6])},
            coords={'latitude': lats, 'longitude': lons})
    elif type(front_types) == list:

        probs_ds_dict = dict({})

        for probs_ds_index, front_type in enumerate(front_types):
            probs_ds_dict[front_type] = (('longitude', 'latitude'), prediction[probs_ds_index])
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

    parser.add_argument('--domain', type=str, required=True, help='Domain of the data.')

    ### Model / data arguments ###
    parser.add_argument('--netcdf_indir', type=str, required=True, help='Main directory for the netcdf files containing variable data.')
    parser.add_argument('--variable_data_source', type=str, required=True, help='Data source for variables (GDAS or GFS)')
    parser.add_argument('--calibration', type=int, help='Neighborhood to use for calibrating model probabilities')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, required=True, help='Model number.')

    ### GPU arguments ###
    parser.add_argument('--gpu_device', type=int, help='GPU device number.')
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth on the GPU')

    args = vars(parser.parse_args())

    ### Use a GPU ###
    if args['gpu_device'] is not None:
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        tf.config.experimental.set_visible_devices(devices=gpus[args['gpu_device']], device_type='GPU')

        # Allow for memory growth on the GPU. This will only use the GPU memory that is required rather than allocating all of the GPU's memory.
        if args['memory_growth']:
            tf.config.experimental.set_memory_growth(device=gpus[args['gpu_device']], enable=True)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    model_properties = pd.read_pickle(f"{args['model_dir']}/model_{args['model_number']}/model_{args['model_number']}_properties.pkl")

    generate_predictions(args['model_number'], args['model_dir'], args['netcdf_indir'], args['init_time'],
        domain=args['domain'], variable_data_source=args['variable_data_source'])
