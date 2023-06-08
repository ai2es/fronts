"""
Generate performance statistics for a model.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 6/8/2023 6:45 PM CT
"""
import argparse
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import xarray as xr
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # this line allows us to import scripts outside of the current directoryimport file_manager as fm
from utils import data_utils


if __name__ == '__main__':
    """
    All arguments listed in the examples are listed via argparse in alphabetical order below this comment block.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
        help="Dataset for which the statistics will be calculated. 'training, 'validation', 'test'")
    parser.add_argument('--datetime', type=int, nargs=4, help='Date and time of the data. Pass 4 ints in the following order: year, month, day, hour')
    parser.add_argument('--domain', type=str, help='Domain of the data.')
    parser.add_argument('--domain_size', type=int, nargs=2, help='Lengths of the dimensions of the final stitched map for predictions: lon, lat')
    parser.add_argument('--forecast_hour', type=int, help='Forecast hour for the GDAS data')
    parser.add_argument('--gpu_device', type=int, help='GPU device number.')
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth on the GPU')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, required=True, help='Model number.')
    parser.add_argument('--num_iterations', type=int, default=10000, help='Number of iterations to perform when bootstrapping the data.')
    parser.add_argument('--fronts_netcdf_indir', type=str, required=True, help='Main directory for the netcdf files containing frontal objects.')
    parser.add_argument('--data_source', type=str, default='era5', help='Data source for variables')

    args = vars(parser.parse_args())

    model_properties = pd.read_pickle('%s/model_%d/model_%d_properties.pkl' % (args['model_dir'], args['model_number'], args['model_number']))
    front_types = model_properties['front_types']
    years = model_properties['%s_years' % args['dataset']]


    era5_files_obj = fm.DataFileLoader(args['fronts_netcdf_indir'], data_file_type='fronts-netcdf')
    era5_files_obj.test_years = years  # does not matter which year attribute we set the years to
    front_files = era5_files_obj.front_files_test

    probs_ds = xr.open_dataset('%s/model_%d/probabilities/model_%d_pred_%s.nc' % (args['model_dir'], args['model_number'], args['model_number'], args['dataset']))

    print("Opening fronts dataset")
    fronts_ds = xr.open_mfdataset(front_files, combine='nested', concat_dim='time').sel(longitude=slice(228, 299.75), latitude=slice(56.75, 25))
    print("Reformatting fronts")
    fronts_ds = data_utils.reformat_fronts(fronts_ds, front_types)

    time_array = probs_ds['time'].values
    lons = probs_ds['longitude'].values
    lats = probs_ds['latitude'].values
    num_timesteps = len(time_array)

    tp_array_spatial = np.zeros(shape=[1, len(lats), len(lons), 5, 100]).astype('int32')
    fp_array_spatial = np.zeros(shape=[1, len(lats), len(lons), 5, 100]).astype('int32')
    tn_array_spatial = np.zeros(shape=[1, len(lats), len(lons), 5, 100]).astype('int32')
    fn_array_spatial = np.zeros(shape=[1, len(lats), len(lons), 5, 100]).astype('int32')

    tp_array_temporal = np.zeros(shape=[1, num_timesteps, 5, 100]).astype('int32')
    fp_array_temporal = np.zeros(shape=[1, num_timesteps, 5, 100]).astype('int32')
    tn_array_temporal = np.zeros(shape=[1, num_timesteps, 5, 100]).astype('int32')
    fn_array_temporal = np.zeros(shape=[1, num_timesteps, 5, 100]).astype('int32')

    thresholds = np.linspace(0.01, 1, 100)  # Probability thresholds for calculating performance statistics
    boundaries = np.array([50, 100, 150, 200, 250])  # Boundaries for checking whether or not a front is present (kilometers)

    bool_tn_fn_dss = dict({front: tf.convert_to_tensor(xr.where(fronts_ds == front_no + 1, 1, 0)['identifier'].values) for front_no, front in enumerate(front_types)})
    bool_tp_fp_dss = dict({front: None for front in front_types})
    probs_dss = dict({front: tf.convert_to_tensor(probs_ds[front].values) for front in front_types})

    performance_ds = xr.Dataset(coords={'time': time_array, 'longitude': lons, 'latitude': lats, 'boundary': boundaries, 'threshold': thresholds})

    for front_no, front_type in enumerate(front_types):
        ### Calculate true/false negatives ###
        for i in range(100):
            print("TN/FN", i + 1, end='\r')
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
            fronts_ds = data_utils.expand_fronts(fronts_ds, iterations=2)  # Expand fronts
            bool_tp_fp_dss[front_type] = tf.convert_to_tensor(xr.where(fronts_ds == front_no + 1, 1, 0)['identifier'].values)  # 1 = cold front, 0 = not a cold front
            for i in range(100):
                print("TP/FP", boundary, i + 1, end='\r')
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

        ### Bootstrap the temporal statistics to find confidence intervals ###
        POD_array = np.zeros([1, 10000, 5, 100])  # probability of detection = TP / (TP + FN)
        SR_array = np.zeros([1, 10000, 5, 100])  # success ratio = 1 - False Alarm Ratio = TP / (TP + FP)

        # 3 confidence intervals: 90, 95, and 99%
        CI_lower_POD = np.zeros([1, 3, 5, 100])
        CI_lower_SR = np.zeros([1, 3, 5, 100])
        CI_upper_POD = np.zeros([1, 3, 5, 100])
        CI_upper_SR = np.zeros([1, 3, 5, 100])

        selectable_indices = range(len(time_array))

        for iteration in range(args['num_iterations']):
            print(f"Iteration {iteration}/{args['num_iterations']}", end='\r')
            indices = random.choices(selectable_indices, k=num_timesteps)  # Select a sample equal to the total number of timesteps

            POD_array[front_no, iteration, :, :] = np.divide(np.sum(tp_array_temporal[front_no, indices, :, :], axis=0),
                                                             np.add(np.sum(tp_array_temporal[front_no, indices, :, :], axis=0),
                                                                    np.sum(fn_array_temporal[front_no, indices, :, :], axis=0)))
            SR_array[front_no, iteration, :, :] = np.divide(np.sum(tp_array_temporal[front_no, indices, :, :], axis=0),
                                                            np.add(np.sum(tp_array_temporal[front_no, indices, :, :], axis=0),
                                                                   np.sum(fp_array_temporal[front_no, indices, :, :], axis=0)))

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

        performance_ds["tp_spatial_%s" % front_type] = (('latitude', 'longitude', 'boundary', 'threshold'), tp_array_spatial[front_no])
        performance_ds["fp_spatial_%s" % front_type] = (('latitude', 'longitude', 'boundary', 'threshold'), fp_array_spatial[front_no])
        performance_ds["tn_spatial_%s" % front_type] = (('latitude', 'longitude', 'boundary', 'threshold'), tn_array_spatial[front_no])
        performance_ds["fn_spatial_%s" % front_type] = (('latitude', 'longitude', 'boundary', 'threshold'), fn_array_spatial[front_no])
        performance_ds["tp_temporal_%s" % front_type] = (('time', 'boundary', 'threshold'), tp_array_temporal[front_no])
        performance_ds["fp_temporal_%s" % front_type] = (('time', 'boundary', 'threshold'), fp_array_temporal[front_no])
        performance_ds["tn_temporal_%s" % front_type] = (('time', 'boundary', 'threshold'), tn_array_temporal[front_no])
        performance_ds["fn_temporal_%s" % front_type] = (('time', 'boundary', 'threshold'), fn_array_temporal[front_no])
        performance_ds["POD_0.5_%s" % front_type] = (('boundary', 'threshold'), CI_lower_POD[front_no, 2, :, :])
        performance_ds["POD_2.5_%s" % front_type] = (('boundary', 'threshold'), CI_lower_POD[front_no, 1, :, :])
        performance_ds["POD_5_%s" % front_type] = (('boundary', 'threshold'), CI_lower_POD[front_no, 0, :, :])
        performance_ds["POD_99.5_%s" % front_type] = (('boundary', 'threshold'), CI_upper_POD[front_no, 2, :, :])
        performance_ds["POD_97.5_%s" % front_type] = (('boundary', 'threshold'), CI_upper_POD[front_no, 1, :, :])
        performance_ds["POD_95_%s" % front_type] = (('boundary', 'threshold'), CI_upper_POD[front_no, 0, :, :])
        performance_ds["SR_0.5_%s" % front_type] = (('boundary', 'threshold'), CI_lower_SR[front_no, 2, :, :])
        performance_ds["SR_2.5_%s" % front_type] = (('boundary', 'threshold'), CI_lower_SR[front_no, 1, :, :])
        performance_ds["SR_5_%s" % front_type] = (('boundary', 'threshold'), CI_lower_SR[front_no, 0, :, :])
        performance_ds["SR_99.5_%s" % front_type] = (('boundary', 'threshold'), CI_upper_SR[front_no, 2, :, :])
        performance_ds["SR_97.5_%s" % front_type] = (('boundary', 'threshold'), CI_upper_SR[front_no, 1, :, :])
        performance_ds["SR_95_%s" % front_type] = (('boundary', 'threshold'), CI_upper_SR[front_no, 0, :, :])

    performance_ds.to_netcdf(path='%s/model_%d/statistics/model_%d_statistics_test.nc' % (args['model_dir'], args['model_number'], args['model_number']), mode='w', engine='netcdf4')
