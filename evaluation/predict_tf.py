"""
**** EXPERIMENTAL SCRIPT TO REPLACE 'predict.py' IN THE NEAR FUTURE ****

Generate predictions using a model with tensorflow datasets.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 6/9/2023 11:23 AM CT
"""
import argparse
import file_manager as fm
import numpy as np
import pandas as pd
from utils.data_utils import combine_datasets
from utils.settings import *
import xarray as xr
import tensorflow as tf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
        help="Dataset for which to make predictions. Options are: 'training', 'validation', 'test'")
    parser.add_argument('--model_dir', type=str, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, help='Model number.')
    parser.add_argument('--tf_indir', type=str, help='Directory for the tensorflow dataset that will be used when generating predictions.')
    parser.add_argument('--data_source', type=str, default='era5', help='Data source for variables')
    parser.add_argument('--gpu_device', type=int, nargs='+', help='GPU device numbers.')
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth on the GPU')

    args = vars(parser.parse_args())

    if args['gpu_device'] is not None:
        gpus = tf.config.list_physical_devices(device_type='GPU')
        tf.config.set_visible_devices(devices=[gpus[gpu] for gpu in args['gpu_device']], device_type='GPU')

        # Allow for memory growth on the GPU. This will only use the GPU memory that is required rather than allocating all of the GPU's memory.
        if args['memory_growth']:
            tf.config.experimental.set_memory_growth(device=[gpus[gpu] for gpu in args['gpu_device']][0], enable=True)

    model_properties = pd.read_pickle('%s/model_%d/model_%d_properties.pkl' % (args['model_dir'], args['model_number'], args['model_number']))
    dataset_properties = pd.read_pickle('%s/dataset_properties.pkl' % args['tf_indir'])

    domain = dataset_properties['domain']
    front_types = dataset_properties['front_types']
    years = model_properties['%s_years' % args['dataset']]
    num_dims = (model_properties['num_dimensions'], model_properties['loss_args']['num_dims'])

    # The axis that the predicts will be concatenated on depends on the shape of the output, which is determined by deep supervision
    if model_properties['deep_supervision'] is True:
        concat_axis = 1
    else:
        concat_axis = 0

    tf_ds_obj = fm.DataFileLoader(args['tf_indir'], data_file_type='%s-tensorflow' % args['data_source'])

    lons = np.arange(DEFAULT_DOMAIN_EXTENTS[domain][0], DEFAULT_DOMAIN_EXTENTS[domain][1] + 0.25, 0.25)
    lats = np.arange(DEFAULT_DOMAIN_EXTENTS[domain][2], DEFAULT_DOMAIN_EXTENTS[domain][3] + 0.25, 0.25)[::-1]

    for year in years:

        tf_ds_obj.test_years = [year, ]
        test_input_files = tf_ds_obj.data_files_test

        time_array = np.array([], dtype=np.datetime64)
        for file in test_input_files:
            time_array = np.append(time_array,
                                   np.arange(np.datetime64(f"{file[-9:-5]}-{file[-5:-3]}"),
                                             np.datetime64(f"{file[-9:-5]}-{file[-5:-3]}") + np.timedelta64(1, "M"),
                                             np.timedelta64(3, "h")))

        test_dataset = combine_datasets(test_input_files)
        assert len(test_dataset) == len(time_array)  # make sure the tensorflow dataset has one element (image) per timestep
        del test_dataset

        model = fm.load_model(args['model_number'], args['model_dir'])

        for file in test_input_files:
            tf_ds = tf.data.Dataset.load(file).batch(GPU_PREDICT_BATCH_SIZE)
            if 'prediction' not in locals():
                prediction = np.array(model.predict(tf_ds)).astype(np.float16)  # shape: (time, longitude, latitude, pressure level, front type)
            else:
                prediction = np.concatenate((prediction, np.array(model.predict(tf_ds)).astype(np.float16)), axis=concat_axis)

        if model_properties['deep_supervision']:
            prediction = prediction[0, ...]  # select the top output of the model, since it is the only one we care about

        if num_dims[1] == 3:
            # Take the maxmimum probability for each front type over the vertical dimension (pressure levels)
            prediction = np.amax(prediction, axis=3)  # shape: (time, longitude, latitude, front type)

        prediction = prediction[:, :, :, 1:]  # remove the 'no front' type from the array
        prediction = np.transpose(prediction, (0, 2, 1, 3))  # shape: (time, latitude, longitude, front type)

        xr.Dataset(data_vars={front_type: (('time', 'latitude', 'longitude'), prediction[:, :, :, front_type_no])
                              for front_type_no, front_type in enumerate(front_types)},
                   coords={'time': time_array, 'longitude': lons, 'latitude': lats}).astype('float32').\
            to_netcdf(path='%s/model_%d/probabilities/model_%d_pred_%s_%d.nc' % (args['model_dir'], args['model_number'], args['model_number'], domain, year),
                      mode='w', engine='netcdf4')