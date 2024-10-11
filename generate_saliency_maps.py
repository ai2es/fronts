"""
Generate saliency maps for a model.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.10.10
"""
import argparse
import os
from utils import data_utils
import numpy as np
import pandas as pd
import file_manager as fm
import xarray as xr
import tensorflow as tf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='test', help="Dataset for which to make saliency maps. Options are: 'training', 'validation', 'test'")
    parser.add_argument('--year_and_month', type=int, nargs=2, help="Year and month for which to make saleincy maps.")
    parser.add_argument('--tf_indir', type=str, required=True, help="Input directory for the tensorflow dataset(s).")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory where the models are stored.")
    parser.add_argument('--model_number', type=int, required=True, help="Model number.")
    parser.add_argument('--batch_size', type=int, default=8,
        help="Batch size for the model predictions. Since the gradients will also be retrieved, this should be lower than "
             "the batch sizes used during training.")
    parser.add_argument('--freq', type=str, required=True, help="Timestep frequency of the input data.")
    parser.add_argument('--verbose', action='store_true', help="Print out the progress of saliency map generation by batch.")
    args = vars(parser.parse_args())

    gpus = tf.config.list_physical_devices(device_type='GPU')  # Find available GPUs
    if len(gpus) > 0:
        tf.config.set_visible_devices(devices=gpus[0], device_type='GPU')
        gpus = tf.config.get_visible_devices(device_type='GPU')  # List of selected GPUs
        tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)  # allow memory growth on GPU

    model_properties = pd.read_pickle('%s/model_%d/model_%d_properties.pkl' % (args['model_dir'], args['model_number'], args['model_number']))
    dataset_properties = pd.read_pickle('%s/dataset_properties.pkl' % args['tf_indir'])  # properties of the dataset being used for saliency maps

    variables = model_properties['dataset_properties']['variables']
    pressure_levels = model_properties['dataset_properties']['pressure_levels']
    num_classes = model_properties['classes']
    front_types = model_properties['dataset_properties']['front_types']
    test_years = model_properties['test_years']
    domain = dataset_properties['domain']

    model = fm.load_model(args['model_number'], args['model_dir'])

    file_loader = fm.DataFileLoader(args['tf_indir'], data_file_type='era5-tensorflow')

    if args['year_and_month'] is not None:
        years, months = [args['year_and_month'][0]], [args['year_and_month'][1]]
    else:
        years, months = model_properties['%s_years' % args['dataset']], range(1, 13)

    for year in years:

        file_loader.test_years = [year, ]
        files_for_year = file_loader.data_files_test

        for month in months:

            gradients = None

            try:
                tf_ds = tf.data.Dataset.load([file for file in files_for_year if '_%d%02d' % (year, month) in file][0])
            except IndexError:
                print("ERA5 tensorflow dataset not found for %d-%02d in %s" % (year, month, args['tf_indir']))
                continue

            next_year = year + 1 if month == 12 else year
            next_month = 1 if month == 12 else month + 1

            init_times = pd.date_range('%s-%02d' % (year, month), '%s-%02d' % (next_year, next_month), freq=args['freq'])[:-1]

            os.makedirs('%s/model_%d/saliencymaps' % (args['model_dir'], args['model_number']), exist_ok=True)  # make directory for the saliency maps
            salmap_dataset_path = '%s/model_%d/saliencymaps/model_%d_salmap_%s_%d%02d.nc' % (args['model_dir'], args['model_number'], args['model_number'], domain, year, month)

            assert len(tf_ds) == len(init_times), \
                "Length of provided tensorflow dataset (%d) does not match the number of timesteps (%d) in %d-%02d with the provided frequency (%s)" % (len(tf_ds), len(init_times), year, month, args['freq'])

            tf_ds = tf_ds.batch(args['batch_size'])  # split dataset into batches (necessary for saliency maps because of memory issues)
            num_batches = len(tf_ds)

            print("Generating saliency maps for %d-%02d" % (year, month))

            for batch_num, batch in enumerate(tf_ds, start=1):
                if args['verbose']:
                    print("Current batch: %d/%d" % (batch_num, num_batches), end='\r')
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(batch)
                    predictions = model(batch)[0]

                    # batch_gradient: model gradients for the current batch, values must be converted to float32 for netcdf4 support
                    batch_gradient = np.stack([np.max(tape.gradient(predictions[..., class_idx+1:class_idx+2], batch).numpy(), axis=-1) for class_idx in range(num_classes-1)], axis=-1).astype('float32')
                    gradients = batch_gradient if gradients is None else np.concatenate([gradients, batch_gradient], axis=0)

            domain_ext = data_utils.DOMAIN_EXTENTS[domain]

            domain_size = (int((domain_ext[1] - domain_ext[0]) // 0.25) + 1,
                           int((domain_ext[3] - domain_ext[2]) // 0.25) + 1)

            lons = np.linspace(domain_ext[0], domain_ext[1], domain_size[0])
            lats = np.linspace(domain_ext[2], domain_ext[3], domain_size[1])[::-1]  # lats in descending order (north-south)

            salmaps = xr.Dataset(data_vars=dict({'%s' % front_type: (('time', 'longitude', 'latitude'), np.max(gradients[..., idx], axis=-1)) for idx, front_type in enumerate(front_types)} |
                                                {'%s_pl' % front_type: (('time', 'longitude', 'latitude', 'pressure_level'), gradients[..., idx]) for idx, front_type in enumerate(front_types)}),
                                 coords={'time': init_times, 'longitude': lons, 'latitude': lats, 'pressure_level': pressure_levels},
                                 attrs={'model_number': args['model_number']})
            salmaps.to_netcdf(salmap_dataset_path, mode='w', engine='netcdf4')