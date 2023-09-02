"""
Calibrate a trained model.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.6.24
"""
import argparse
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # this line allows us to import scripts outside of the current directory
from utils.settings import DEFAULT_FRONT_NAMES
import matplotlib.pyplot as plt
import pickle
import xarray as xr
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score


if __name__ == '__main__':
    """
    All arguments listed in the examples are listed via argparse in alphabetical order below this comment block.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset for which to make predictions if prediction_method is 'random' or 'all'. Options are:"
                                                    "'training', 'validation', 'test'")
    parser.add_argument('--domain', type=str, help='Domain of the data.')
    parser.add_argument('--model_dir', type=str, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, help='Model number.')
    parser.add_argument('--data_source', type=str, default='era5', help='Data source for variables')

    args = vars(parser.parse_args())

    model_properties = pd.read_pickle('%s/model_%d/model_%d_properties.pkl' % (args['model_dir'], args['model_number'], args['model_number']))

    ### front_types argument is being moved into the dataset_properties dictionary within model_properties ###
    try:
        front_types = model_properties['front_types']
    except KeyError:
        front_types = model_properties['dataset_properties']['front_types']

    if type(front_types) == str:
        front_types = [front_types, ]

    try:
        _ = model_properties['calibration_models']  # Check to see if the model has already been calibrated before
    except KeyError:
        model_properties['calibration_models'] = dict()

    model_properties['calibration_models'][args['domain']] = dict()

    stats_ds = xr.open_dataset('%s/model_%d/statistics/model_%d_statistics_%s_%s.nc' % (args['model_dir'], args['model_number'], args['model_number'], args['domain'], args['dataset']))

    axis_ticks = np.arange(0.1, 1.1, 0.1)

    for front_label in front_types:

        model_properties['calibration_models'][args['domain']][front_label] = dict()

        true_positives = stats_ds[f'tp_temporal_{front_label}'].values
        false_positives = stats_ds[f'fp_temporal_{front_label}'].values

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
            model_properties['calibration_models'][args['domain']][front_label]['%d km' % ((boundary + 1) * 50)] = ir

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

        with open('%s/model_%d/model_%d_properties.pkl' % (args['model_dir'], args['model_number'], args['model_number']), 'wb') as f:
            pickle.dump(model_properties, f)

        plt.suptitle(f"Model {args['model_number']} reliability/calibration: {DEFAULT_FRONT_NAMES[front_label]}")
        plt.savefig(f'%s/model_%d/model_%d_calibration_%s_%s.png' % (args['model_dir'], args['model_number'], args['model_number'], args['domain'], front_label),
                    bbox_inches='tight', dpi=300)
        plt.close()
