"""
Generate performance statistics for a model.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.4.29
"""
import argparse
import glob
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import xarray as xr
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # this line allows us to import scripts outside the current directory
import file_manager as fm
from utils import data_utils
from utils.settings import DOMAIN_EXTENTS


def combine_statistics_for_dataset():
    """
    Combine monthly statistics into one aggregate statistics file for a large dataset.
    """
    statistics_files = []

    for year in years:
        statistics_files += list(sorted(glob.glob('%s/model_%d/statistics/model_%d_statistics_%s_%d*.nc' %
                                                  (args['model_dir'], args['model_number'], args['model_number'], args['domain'], year))))

    datasets_by_front_type = []

    for front_no, front_type in enumerate(front_types):

        ### Temporal and spatial datasets need to be loaded separately because of differing dimensions (xarray bugs) ###
        dataset_performance_ds_temporal = xr.open_dataset(statistics_files[0], chunks={'time': 16})[['%s_temporal_%s' % (stat, front_type) for stat in ['tp', 'fp', 'tn', 'fn']]]
        dataset_performance_ds_spatial = xr.open_dataset(statistics_files[0], chunks={'time': 16})[['%s_spatial_%s' % (stat, front_type) for stat in ['tp', 'fp', 'tn', 'fn']]]
        for stats_file in statistics_files[1:]:
            dataset_performance_ds_spatial += xr.open_dataset(stats_file, chunks={'time': 16})[['%s_spatial_%s' % (stat, front_type) for stat in ['tp', 'fp', 'tn', 'fn']]]
            dataset_performance_ds_temporal = xr.merge([dataset_performance_ds_temporal, xr.open_dataset(stats_file, chunks={'time': 16})[['%s_temporal_%s' % (stat, front_type) for stat in ['tp', 'fp', 'tn', 'fn']]]])
        dataset_performance_ds = xr.merge([dataset_performance_ds_spatial, dataset_performance_ds_temporal])  # Combine spatial and temporal data into one dataset

        tp_array_temporal = dataset_performance_ds['tp_temporal_%s' % front_type].values
        fp_array_temporal = dataset_performance_ds['fp_temporal_%s' % front_type].values
        fn_array_temporal = dataset_performance_ds['fn_temporal_%s' % front_type].values

        time_array = dataset_performance_ds['time'].values

        ### Bootstrap the temporal statistics to find confidence intervals ###
        POD_array = np.zeros([num_front_types, args['num_iterations'], 5, 100])  # probability of detection = TP / (TP + FN)
        SR_array = np.zeros([num_front_types, args['num_iterations'], 5, 100])  # success ratio = 1 - False Alarm Ratio = TP / (TP + FP)

        # 3 confidence intervals: 90, 95, and 99%
        CI_lower_POD = np.zeros([num_front_types, 3, 5, 100])
        CI_lower_SR = np.zeros([num_front_types, 3, 5, 100])
        CI_upper_POD = np.zeros([num_front_types, 3, 5, 100])
        CI_upper_SR = np.zeros([num_front_types, 3, 5, 100])

        num_timesteps = len(time_array)
        selectable_indices = range(num_timesteps)

        for iteration in range(args['num_iterations']):
            print(f"Iteration {iteration}/{args['num_iterations']}", end='\r')
            indices = random.choices(selectable_indices, k=num_timesteps)  # Select a sample equal to the total number of timesteps

            POD_array[front_no, iteration, :, :] = np.divide(np.sum(tp_array_temporal[indices, :, :], axis=0),
                                                             np.add(np.sum(tp_array_temporal[indices, :, :], axis=0),
                                                                    np.sum(fn_array_temporal[indices, :, :], axis=0)))
            SR_array[front_no, iteration, :, :] = np.divide(np.sum(tp_array_temporal[indices, :, :], axis=0),
                                                            np.add(np.sum(tp_array_temporal[indices, :, :], axis=0),
                                                                   np.sum(fp_array_temporal[indices, :, :], axis=0)))
        print(f"Iteration {args['num_iterations']}/{args['num_iterations']}")

        ## Turn NaNs to zeros
        POD_array = np.nan_to_num(POD_array)
        SR_array = np.nan_to_num(SR_array)

        # Calculate confidence intervals at each probability bin
        for percent in np.arange(0, 100):
            CI_lower_POD[front_no, 0, :, percent] = np.percentile(POD_array[front_no, :, :, percent], q=5, axis=0)  # lower bound for 90% confidence interval
            CI_lower_POD[front_no, 1, :, percent] = np.percentile(POD_array[front_no, :, :, percent], q=2.5, axis=0)  # lower bound for 95% confidence interval
            CI_lower_POD[front_no, 2, :, percent] = np.percentile(POD_array[front_no, :, :, percent], q=0.5, axis=0)  # lower bound for 99% confidence interval
            CI_upper_POD[front_no, 0, :, percent] = np.percentile(POD_array[front_no, :, :, percent], q=95, axis=0)  # upper bound for 90% confidence interval
            CI_upper_POD[front_no, 1, :, percent] = np.percentile(POD_array[front_no, :, :, percent], q=97.5, axis=0)  # upper bound for 95% confidence interval
            CI_upper_POD[front_no, 2, :, percent] = np.percentile(POD_array[front_no, :, :, percent], q=99.5, axis=0)  # upper bound for 99% confidence interval

            CI_lower_SR[front_no, 0, :, percent] = np.percentile(SR_array[front_no, :, :, percent], q=5, axis=0)  # lower bound for 90% confidence interval
            CI_lower_SR[front_no, 1, :, percent] = np.percentile(SR_array[front_no, :, :, percent], q=2.5, axis=0)  # lower bound for 95% confidence interval
            CI_lower_SR[front_no, 2, :, percent] = np.percentile(SR_array[front_no, :, :, percent], q=0.5, axis=0)  # lower bound for 99% confidence interval
            CI_upper_SR[front_no, 0, :, percent] = np.percentile(SR_array[front_no, :, :, percent], q=95, axis=0)  # upper bound for 90% confidence interval
            CI_upper_SR[front_no, 1, :, percent] = np.percentile(SR_array[front_no, :, :, percent], q=97.5, axis=0)  # upper bound for 95% confidence interval
            CI_upper_SR[front_no, 2, :, percent] = np.percentile(SR_array[front_no, :, :, percent], q=99.5, axis=0)  # upper bound for 99% confidence interval

        dataset_performance_ds["POD_0.5_%s" % front_type] = (('boundary', 'threshold'), CI_lower_POD[front_no, 2, :, :])
        dataset_performance_ds["POD_2.5_%s" % front_type] = (('boundary', 'threshold'), CI_lower_POD[front_no, 1, :, :])
        dataset_performance_ds["POD_5_%s" % front_type] = (('boundary', 'threshold'), CI_lower_POD[front_no, 0, :, :])
        dataset_performance_ds["POD_99.5_%s" % front_type] = (('boundary', 'threshold'), CI_upper_POD[front_no, 2, :, :])
        dataset_performance_ds["POD_97.5_%s" % front_type] = (('boundary', 'threshold'), CI_upper_POD[front_no, 1, :, :])
        dataset_performance_ds["POD_95_%s" % front_type] = (('boundary', 'threshold'), CI_upper_POD[front_no, 0, :, :])
        dataset_performance_ds["SR_0.5_%s" % front_type] = (('boundary', 'threshold'), CI_lower_SR[front_no, 2, :, :])
        dataset_performance_ds["SR_2.5_%s" % front_type] = (('boundary', 'threshold'), CI_lower_SR[front_no, 1, :, :])
        dataset_performance_ds["SR_5_%s" % front_type] = (('boundary', 'threshold'), CI_lower_SR[front_no, 0, :, :])
        dataset_performance_ds["SR_99.5_%s" % front_type] = (('boundary', 'threshold'), CI_upper_SR[front_no, 2, :, :])
        dataset_performance_ds["SR_97.5_%s" % front_type] = (('boundary', 'threshold'), CI_upper_SR[front_no, 1, :, :])
        dataset_performance_ds["SR_95_%s" % front_type] = (('boundary', 'threshold'), CI_upper_SR[front_no, 0, :, :])

        datasets_by_front_type.append(dataset_performance_ds)

    final_performance_ds = xr.merge(datasets_by_front_type)
    final_performance_ds.to_netcdf(path='%s/model_%d/statistics/model_%d_statistics_%s_%s.nc' % (args['model_dir'], args['model_number'], args['model_number'], args['domain'], args['dataset']), mode='w', engine='netcdf4')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset for which to make predictions. Options are: 'training', 'validation', 'test'")
    parser.add_argument('--year_and_month', type=int, nargs=2, help="Year and month for which to make predictions.")
    parser.add_argument('--combine', action='store_true', help="Combine calculated statistics for a dataset.")
    parser.add_argument('--domain', type=str, help='Domain of the data.')
    parser.add_argument('--forecast_hour', type=int, help='Forecast hour for the GDAS data')
    parser.add_argument('--gpu_device', type=int, nargs='+', help='GPU device number.')
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth on the GPU')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, required=True, help='Model number.')
    parser.add_argument('--num_iterations', type=int, default=10000, help='Number of iterations to perform when bootstrapping the data.')
    parser.add_argument('--fronts_netcdf_indir', type=str, help='Main directory for the netcdf files containing frontal objects.')
    parser.add_argument('--data_source', type=str, default='era5', help='Data source for variables')
    parser.add_argument('--overwrite', action='store_true', help="Overwrite any existing statistics files.")

    args = vars(parser.parse_args())

    model_properties = pd.read_pickle('%s/model_%d/model_%d_properties.pkl' % (args['model_dir'], args['model_number'], args['model_number']))
    domain = args['domain']

    # Some older models do not have the 'dataset_properties' dictionary
    try:
        front_types = model_properties['dataset_properties']['front_types']
        num_dims = model_properties['dataset_properties']['num_dims']
    except KeyError:
        front_types = model_properties['front_types']
        if args['model_number'] in [6846496, 7236500, 7507525]:
            num_dims = (3, 3)

    num_front_types = model_properties['classes'] - 1  # remove the "no front" class type

    if args['dataset'] is not None and args['year_and_month'] is not None:
        raise ValueError("--dataset and --year_and_month cannot be passed together.")
    elif args['dataset'] is None and args['year_and_month'] is None:
        raise ValueError("At least one of [--dataset, --year_and_month] must be passed.")
    elif args['year_and_month'] is not None:
        years, months = [args['year_and_month'][0]], [args['year_and_month'][1]]
    else:
        years, months = model_properties['%s_years' % args['dataset']], range(1, 13)

    if args['dataset'] is not None and args['combine']:
        combine_statistics_for_dataset()
        exit()

    if args['gpu_device'] is not None:
        gpus = tf.config.list_physical_devices(device_type='GPU')
        tf.config.set_visible_devices(devices=[gpus[gpu] for gpu in args['gpu_device']], device_type='GPU')

        # Allow for memory growth on the GPU. This will only use the GPU memory that is required rather than allocating all the GPU's memory.
        if args['memory_growth']:
            tf.config.experimental.set_memory_growth(device=[gpus[gpu] for gpu in args['gpu_device']][0], enable=True)

    for year in years:

        era5_files_obj = fm.DataFileLoader(args['fronts_netcdf_indir'], data_file_type='fronts-netcdf')
        era5_files_obj.test_years = [year, ]  # does not matter which year attribute we set the years to
        front_files = era5_files_obj.front_files_test

        for month in months:

            front_files_month = [file for file in front_files if '_%d%02d' % (year, month) in file]

            if args['domain'] == 'full':
                print("full")
                for front_file in front_files_month:
                    if any(['%02d_full.nc' % hour in front_file for hour in np.arange(3, 27, 6)]):
                        front_files_month.pop(front_files_month.index(front_file))

            prediction_file = f'%s/model_%d/probabilities/model_%d_pred_%s_%d%02d.nc' % \
                              (args['model_dir'], args['model_number'], args['model_number'], args['domain'], year, month)

            stats_dataset_path = '%s/model_%d/statistics/model_%d_statistics_%s_%d%02d.nc' % (args['model_dir'], args['model_number'], args['model_number'], args['domain'], year, month)
            if os.path.isfile(stats_dataset_path) and not args['overwrite']:
                print("WARNING: %s exists, pass the --overwrite argument to overwrite existing data." % stats_dataset_path)
                continue

            probs_ds = xr.open_dataset(prediction_file)
            lons = probs_ds['longitude'].values
            lats = probs_ds['latitude'].values

            fronts_ds = xr.open_mfdataset(front_files_month, combine='nested', concat_dim='time')\
                .sel(longitude=slice(DOMAIN_EXTENTS[args['domain']][0], DOMAIN_EXTENTS[args['domain']][1]),
                     latitude=slice(DOMAIN_EXTENTS[args['domain']][3], DOMAIN_EXTENTS[args['domain']][2]))

            fronts_ds_month = data_utils.reformat_fronts(fronts_ds.sel(time='%d-%02d' % (year, month)), front_types)

            time_array = probs_ds['time'].values
            num_timesteps = len(time_array)

            tp_array_spatial = np.zeros(shape=[num_front_types, len(lats), len(lons), 5, 100]).astype('int64')
            fp_array_spatial = np.zeros(shape=[num_front_types, len(lats), len(lons), 5, 100]).astype('int64')
            tn_array_spatial = np.zeros(shape=[num_front_types, len(lats), len(lons), 5, 100]).astype('int64')
            fn_array_spatial = np.zeros(shape=[num_front_types, len(lats), len(lons), 5, 100]).astype('int64')

            tp_array_temporal = np.zeros(shape=[num_front_types, num_timesteps, 5, 100]).astype('int64')
            fp_array_temporal = np.zeros(shape=[num_front_types, num_timesteps, 5, 100]).astype('int64')
            tn_array_temporal = np.zeros(shape=[num_front_types, num_timesteps, 5, 100]).astype('int64')
            fn_array_temporal = np.zeros(shape=[num_front_types, num_timesteps, 5, 100]).astype('int64')

            thresholds = np.linspace(0.01, 1, 100)  # Probability thresholds for calculating performance statistics
            boundaries = np.array([50, 100, 150, 200, 250])  # Boundaries for checking whether a front is present (kilometers)

            bool_tn_fn_dss = dict({front: tf.convert_to_tensor(xr.where(fronts_ds_month == front_no + 1, 1, 0)['identifier'].values) for front_no, front in enumerate(front_types)})
            bool_tp_fp_dss = dict({front: None for front in front_types})
            probs_dss = dict({front: tf.convert_to_tensor(probs_ds[front].values) for front in front_types})

            performance_ds = xr.Dataset(coords={'time': time_array, 'longitude': lons, 'latitude': lats, 'boundary': boundaries, 'threshold': thresholds})

            for front_no, front_type in enumerate(front_types):
                fronts_ds_month = data_utils.reformat_fronts(fronts_ds.sel(time='%d-%02d' % (year, month)), front_types)
                print("%d-%02d: %s (TN/FN)" % (year, month, front_type))
                ### Calculate true/false negatives ###
                for i in range(100):
                    """
                    True negative ==> model correctly predicts the lack of a front at a given point
                    False negative ==> model does not predict a front, but a front exists
                    
                    The numbers of true negatives and false negatives are the same for all neighborhoods and are calculated WITHOUT expanding the fronts.
                    If we were to calculate the negatives separately for each neighborhood, the number of misses would be artificially inflated, lowering the 
                    final CSI scores and making the neighborhood method effectively useless.
                    """
                    tn = tf.where((probs_dss[front_type] < thresholds[i]) & (bool_tn_fn_dss[front_type] == 0), 1, 0)
                    fn = tf.where((probs_dss[front_type] < thresholds[i]) & (bool_tn_fn_dss[front_type] == 1), 1, 0)

                    tn_array_spatial[front_no, :, :, :, i] = tf.tile(tf.expand_dims(tf.reduce_sum(tn, axis=0), axis=-1), (1, 1, 5))
                    fn_array_spatial[front_no, :, :, :, i] = tf.tile(tf.expand_dims(tf.reduce_sum(fn, axis=0), axis=-1), (1, 1, 5))
                    tn_array_temporal[front_no, :, :, i] = tf.tile(tf.expand_dims(tf.reduce_sum(tn, axis=(1, 2)), axis=-1), (1, 5))
                    fn_array_temporal[front_no, :, :, i] = tf.tile(tf.expand_dims(tf.reduce_sum(fn, axis=(1, 2)), axis=-1), (1, 5))

                ### Calculate true/false positives ###
                for boundary in range(5):
                    fronts_ds_month = data_utils.expand_fronts(fronts_ds_month, iterations=2)  # Expand fronts by 50km
                    bool_tp_fp_dss[front_type] = tf.convert_to_tensor(xr.where(fronts_ds_month == front_no + 1, 1, 0)['identifier'].values)  # 1 = cold front, 0 = not a cold front
                    print("%d-%02d: %s (%d km)" % (year, month, front_type, (boundary + 1) * 50))
                    for i in range(100):
                        """
                        True positive ==> model correctly identifies a front
                        False positive ==> model predicts a front, but no front is present within the given neighborhood
                        """
                        tp = tf.where((probs_dss[front_type] > thresholds[i]) & (bool_tp_fp_dss[front_type] == 1), 1, 0)
                        fp = tf.where((probs_dss[front_type] > thresholds[i]) & (bool_tp_fp_dss[front_type] == 0), 1, 0)

                        tp_array_spatial[front_no, :, :, boundary, i] = tf.reduce_sum(tp, axis=0)
                        fp_array_spatial[front_no, :, :, boundary, i] = tf.reduce_sum(fp, axis=0)
                        tp_array_temporal[front_no, :, boundary, i] = tf.reduce_sum(tp, axis=(1, 2))
                        fp_array_temporal[front_no, :, boundary, i] = tf.reduce_sum(fp, axis=(1, 2))

                performance_ds["tp_spatial_%s" % front_type] = (('latitude', 'longitude', 'boundary', 'threshold'), tp_array_spatial[front_no])
                performance_ds["fp_spatial_%s" % front_type] = (('latitude', 'longitude', 'boundary', 'threshold'), fp_array_spatial[front_no])
                performance_ds["tn_spatial_%s" % front_type] = (('latitude', 'longitude', 'boundary', 'threshold'), tn_array_spatial[front_no])
                performance_ds["fn_spatial_%s" % front_type] = (('latitude', 'longitude', 'boundary', 'threshold'), fn_array_spatial[front_no])
                performance_ds["tp_temporal_%s" % front_type] = (('time', 'boundary', 'threshold'), tp_array_temporal[front_no])
                performance_ds["fp_temporal_%s" % front_type] = (('time', 'boundary', 'threshold'), fp_array_temporal[front_no])
                performance_ds["tn_temporal_%s" % front_type] = (('time', 'boundary', 'threshold'), tn_array_temporal[front_no])
                performance_ds["fn_temporal_%s" % front_type] = (('time', 'boundary', 'threshold'), fn_array_temporal[front_no])

            performance_ds.to_netcdf(path=stats_dataset_path, mode='w', engine='netcdf4')
