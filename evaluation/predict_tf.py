"""
**** EXPERIMENTAL SCRIPT TO REPLACE 'predict.py' IN THE NEAR FUTURE ****

Generate predictions using a model with tensorflow datasets.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.8.21
"""
import argparse
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # this line allows us to import scripts outside the current directory
import file_manager as fm
from utils.data_utils import *
from utils.misc import initialize_gpus
import xarray as xr
import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset for which to make predictions. Options are: 'training', 'validation', 'test'")
    parser.add_argument('--year_and_month', type=int, nargs=2, help="Year and month for which to make predictions.")
    parser.add_argument('--model_dir', type=str, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, help='Model number.')
    parser.add_argument('--tf_indir', type=str, help='Directory for the tensorflow dataset that will be used when generating predictions.')
    parser.add_argument('--data_source', type=str, default='era5', help='Data source for variables')
    parser.add_argument('--gpu_device', type=int, nargs='+', help='GPU device numbers.')
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for the model predictions.")
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth on the GPU')
    parser.add_argument('--overwrite', action='store_true', help="Overwrite any existing prediction files.")
    args = vars(parser.parse_args())
    
    model_properties = pd.read_pickle('%s/model_%d/model_%d_properties.pkl' % (args['model_dir'], args['model_number'], args['model_number']))
    dataset_properties = pd.read_pickle('%s/dataset_properties.pkl' % args['tf_indir'])

    domain = dataset_properties['domain']

    hour_interval = 3 if domain == 'conus' else 6

    # Some older models do not have the 'dataset_properties' dictionary
    try:
        front_types = model_properties['dataset_properties']['front_types']
        num_dims = model_properties['dataset_properties']['num_dims']
    except KeyError:
        front_types = model_properties['front_types']
        if args['model_number'] in [6846496, 7236500, 7507525]:
            num_dims = (3, 3)

    if args['dataset'] is not None and args['year_and_month'] is not None:
        raise ValueError("--dataset and --year_and_month cannot be passed together.")
    elif args['dataset'] is None and args['year_and_month'] is None:
        raise ValueError("At least one of [--dataset, --year_and_month] must be passed.")
    elif args['year_and_month'] is not None:
        years, months = [args['year_and_month'][0]], [args['year_and_month'][1]]
    else:
        years, months = model_properties['%s_years' % args['dataset']], range(1, 13)

    ### Make sure that the dataset has the same attributes as the model ###
    if model_properties['normalization_parameters'] != dataset_properties['normalization_parameters']:
        raise ValueError("Cannot evaluate model with the selected dataset. Reason: normalization parameters do not match")
    if model_properties['dataset_properties']['front_types'] != dataset_properties['front_types']:
        raise ValueError("Cannot evaluate model with the selected dataset. Reason: front types do not match "
                         f"(model: {model_properties['dataset_properties']['front_types']}, dataset: {dataset_properties['front_types']})")
    if model_properties['dataset_properties']['variables'] != dataset_properties['variables']:
        raise ValueError("Cannot evaluate model with the selected dataset. Reason: variables do not match "
                         f"(model: {model_properties['dataset_properties']['variables']}, dataset: {dataset_properties['variables']})")
    if model_properties['dataset_properties']['pressure_levels'] != dataset_properties['pressure_levels']:
        raise ValueError("Cannot evaluate model with the selected dataset. Reason: pressure levels do not match "
                         f"(model: {model_properties['dataset_properties']['pressure_levels']}, dataset: {dataset_properties['pressure_levels']})")

    if args['gpu_device'] is not None:
        initialize_gpus(args['gpu_device'], args['memory_growth'])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # The axis that the predicts will be concatenated on depends on the shape of the output, which is determined by deep supervision
    concat_axis = 1 if model_properties['deep_supervision'] else 0
    
    if dataset_properties['override_extent'] is not None:
        extent = dataset_properties['override_extent']
        lons = np.arange(extent[0], extent[1] + 0.25, 0.25)
        lats = np.arange(extent[2], extent[3] + 0.25, 0.25)[::-1]
    else:
        lons = np.arange(DOMAIN_EXTENTS[domain][0], DOMAIN_EXTENTS[domain][1] + 0.25, 0.25)
        lats = np.arange(DOMAIN_EXTENTS[domain][2], DOMAIN_EXTENTS[domain][3] + 0.25, 0.25)[::-1]

    model = fm.load_model(args['model_number'], args['model_dir'])

    for year in years:
        
        tf_ds_obj = fm.DataFileLoader(args['tf_indir'], data_type='inputs', file_format='tensorflow', years=years)
        files_for_year = tf_ds_obj.files[0]
        
        for month in months:

            prediction_dataset_path = '%s/model_%d/probabilities/model_%d_pred_%s_%d%02d.nc' % (args['model_dir'], args['model_number'], args['model_number'], domain, year, month)
            if os.path.isfile(prediction_dataset_path) and not args['overwrite']:
                print("WARNING: %s exists, pass the --overwrite argument to overwrite existing data." % prediction_dataset_path)
                continue

            input_file = [file for file in files_for_year if '_%d%02d' % (year, month) in file][0]
            tf_ds = tf.data.Dataset.load(input_file)
            time_array = np.arange(np.datetime64(f"{input_file[-9:-5]}-{input_file[-5:-3]}"),
                                   np.datetime64(f"{input_file[-9:-5]}-{input_file[-5:-3]}") + np.timedelta64(1, "M"),
                                   np.timedelta64(hour_interval, "h"))

            ### remove timesteps that do not have data ###
            missing_indices = np.array([])
            missing_fronts = "%d-%02d" % (year, month) in missing_fronts_ind
            if missing_fronts:
                missing_indices = np.append(missing_indices, missing_fronts_ind["%d-%02d" % (year, month)])

            model_uses_satellite = any(['band_' in var for var in model_properties['dataset_properties']['variables']])
            if model_uses_satellite and "%d-%02d" % (year, month) in missing_satellite_ind:
                missing_indices = np.append(missing_indices, missing_satellite_ind["%d-%02d" % (year, month)])
                missing_indices = np.unique(missing_indices.flatten())

            if missing_fronts or model_uses_satellite:
                if hour_interval == 6:
                    missing_indices = np.array([int(ind / 2) for ind in missing_indices if ind % 2 == 0])
                time_array = np.delete(time_array, missing_indices.astype('int32'))
            ##############################################

            assert len(tf_ds) == len(time_array)  # make sure tensorflow dataset has all timesteps

            tf_ds = tf_ds.batch(args['batch_size'])
            prediction = np.array(model.predict(tf_ds)).astype(np.float16)

            if model_properties['deep_supervision']:
                prediction = prediction[0, ...]  # select the top output of the model, since it is the only one we care about

            if num_dims[1] == 3:
                # Take the maxmimum probability for each front type over the vertical dimension (pressure levels)
                prediction = np.amax(prediction, axis=3)  # shape: (time, longitude, latitude, front type)

            prediction = prediction[..., 1:]  # remove the 'no front' type from the array
            prediction = np.transpose(prediction, (0, 2, 1, 3))  # shape: (time, latitude, longitude, front type)

            xr.Dataset(data_vars={front_type: (('time', 'latitude', 'longitude'), prediction[:, :, :, front_type_no])
                                  for front_type_no, front_type in enumerate(front_types)},
                       coords={'time': time_array, 'longitude': lons, 'latitude': lats}).astype('float32').\
                to_netcdf(path=prediction_dataset_path, mode='w', engine='netcdf4')

            del prediction  # Delete the prediction variable so it can be recreated for the next month
