"""
Functions used for evaluating a U-Net model.

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 7/30/2023 6:24 PM CT

TODO:
    * Clean up code (much needed)
    * Remove the need for separate pickle files to be generated for spatial CSI maps
    * Add more documentation
"""

import itertools
import argparse
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm, colors  # Here we explicitly import the cm and color modules to suppress a PyCharm bug
from utils import data_utils, settings
from utils.plotting_utils import plot_background
from skimage.morphology import skeletonize


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--init_time', type=int, nargs=4, required=True, help='Date and time of the data. Pass 4 ints in the following order: year, month, day, hour')
    parser.add_argument('--forecast_hours', type=int, nargs='+', required=True, help='Forecast hour for the GDAS data')

    parser.add_argument('--domain', type=str, required=True, help='Domain of the data.')
    parser.add_argument('--domain_images', type=int, nargs=2, help='Number of images for each dimension the final stitched map for predictions: lon, lat')
    parser.add_argument('--calibration', type=int, help='Neighborhood to use for calibrating model probabilities')
    parser.add_argument('--splines', action='store_true', help='Plot fronts as deterministic splines (best first-guess)')
    parser.add_argument('--contours', action='store_true', help='Plot fronts as probability contours')

    parser.add_argument('--model_dir', type=str, required=True, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, required=True, help='Model number.')
    parser.add_argument('--plot_dir', type=str, help='plot directory')

    parser.add_argument('--variable_data_source', type=str, default='gdas', help='Data source for variables')

    args = vars(parser.parse_args())

    DEFAULT_COLORBAR_POSITION = {'conus': 0.77, 'full': 0.87}

    extent = settings.DEFAULT_DOMAIN_EXTENTS[args['domain']]

    year, month, day, hour = args['init_time']
    probs_dir = f"{args['model_dir']}/model_{args['model_number']}/predictions"

    cbar_position = DEFAULT_COLORBAR_POSITION[args['domain']]

    for forecast_hour in args['forecast_hours']:
        filename_base = f'model_%d_{year}-%02d-%02d-%02dz_%s_f%03d_%s' % (args['model_number'], month, day, hour, args['variable_data_source'].lower(), forecast_hour, args['domain'])

        probs_file = f'{probs_dir}/{filename_base}_probabilities.nc'
        probs_ds = xr.open_mfdataset(probs_file)

        crs = ccrs.PlateCarree(central_longitude=np.mean(extent[:2]))

        model_properties = pd.read_pickle(f"{args['model_dir']}/model_{args['model_number']}/model_{args['model_number']}_properties.pkl")

        front_types = model_properties['dataset_properties']['front_types']

        if type(front_types) == str:
            front_types = [front_types, ]

        mask, prob_int = 0.10, 0.10  # Probability mask, contour interval for probabilities
        vmax, cbar_tick_adjust, cbar_label_adjust, n_colors = 1, prob_int, 10, 11
        levels = np.around(np.arange(0, 1 + prob_int, prob_int), 2)
        cbar_ticks = np.around(np.arange(mask, 1 + prob_int, prob_int), 2)

        contour_maps_by_type = [settings.DEFAULT_CONTOUR_CMAPS[label] for label in front_types]
        front_names_by_type = [settings.DEFAULT_FRONT_NAMES[label] for label in front_types]
        front_colors_by_type = [settings.DEFAULT_FRONT_COLORS[label] for label in front_types]

        probs_ds = xr.where(probs_ds > mask, probs_ds, 0).isel(time=0, forecast_hour=0)

        for key in list(probs_ds.keys()):

            if args['splines']:
                try:
                    spline_threshold = model_properties['front_obj_thresholds'][args['domain']][key]['250']
                except KeyError:
                    spline_calibration_domain = list(model_properties['front_obj_thresholds'].keys())[0]
                    spline_threshold = model_properties['front_obj_thresholds'][spline_calibration_domain][key]['250']
                    print("No calibration data available for the %s domain, using %s domain instead." % (args['domain'], spline_calibration_domain))
                probs_ds[f'{key}_obj'] = (('longitude', 'latitude'), skeletonize(xr.where(probs_ds[key] > spline_threshold, 1, 0).values))

            if args['calibration'] is not None:
                try:
                    ir_model = model_properties['calibration_models'][args['domain']][key]['%d km' % args['calibration']]
                except KeyError:
                    ir_model = model_properties['calibration_models']['conus'][key]['%d km' % args['calibration']]
                original_shape = np.shape(probs_ds[key].values)
                probs_ds[key].values = ir_model.predict(probs_ds[key].values.flatten()).reshape(original_shape)
                cbar_label = 'Probability (calibrated - %d km)' % args['calibration']
            else:
                cbar_label = 'Probability (uncalibrated)'

        if len(front_types) > 1:
            all_possible_front_combinations = itertools.permutations(front_types, r=2)
            for combination in all_possible_front_combinations:
                probs_ds[combination[0]].values = np.where(probs_ds[combination[0]].values > probs_ds[combination[1]].values - 0.02, probs_ds[combination[0]].values, 0)

        probs_ds = xr.where(probs_ds == 0, float("NaN"), probs_ds)

        valid_time = data_utils.add_or_subtract_hours_to_timestep(f'%d%02d%02d%02d' % (year, month, day, hour), num_hours=forecast_hour)
        data_title = f"Run: {args['variable_data_source'].upper()} {year}-%02d-%02d-%02dz F%03d \nPredictions valid: {valid_time[:4]}-{valid_time[4:6]}-{valid_time[6:8]}-{valid_time[8:]}z" % (month, day, hour, forecast_hour)

        fig, ax = plt.subplots(1, 1, figsize=(20, 8), subplot_kw={'projection': crs})
        plot_background(extent, ax=ax, linewidth=0.5)

        cbar_front_labels = []
        cbar_front_ticks = []

        for front_no, front_key, front_name, front_label, cmap in zip(range(1, len(front_names_by_type) + 1), front_types, front_names_by_type, front_types, contour_maps_by_type):

            cmap_probs, norm_probs = cm.get_cmap(cmap, n_colors), colors.Normalize(vmin=0, vmax=vmax)

            if args['contours']:
                probs_ds[front_key].plot.contourf(ax=ax, x='longitude', y='latitude', norm=norm_probs, levels=levels, cmap=cmap_probs, transform=ccrs.PlateCarree(), alpha=0.75, add_colorbar=False)
                cbar_ax = fig.add_axes([cbar_position + (front_no * 0.015), 0.11, 0.015, 0.77])
                cbar = plt.colorbar(cm.ScalarMappable(norm=norm_probs, cmap=cmap_probs), cax=cbar_ax, boundaries=levels[1:], alpha=0.75)
                cbar.set_ticklabels([])

            if args['splines']:
                cmap_front = colors.ListedColormap(['None', front_colors_by_type[front_no - 1]], name='from_list', N=2)
                norm_front = colors.Normalize(vmin=0, vmax=1)
                probs_ds[f'{front_key}_obj'].plot(ax=ax, x='longitude', y='latitude', cmap=cmap_front, norm=norm_front, transform=ccrs.PlateCarree(), add_colorbar=False)

            cbar_front_labels.append(front_name)
            cbar_front_ticks.append(front_no + 0.5)

        if args['contours']:
            cbar.set_label(cbar_label, rotation=90)
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_ticks)

        cmap_front = colors.ListedColormap(front_colors_by_type, name='from_list', N=len(front_colors_by_type))
        norm_front = colors.Normalize(vmin=1, vmax=len(front_colors_by_type) + 1)

        cbar_front = plt.colorbar(cm.ScalarMappable(norm=norm_front, cmap=cmap_front), ax=ax, alpha=0.75, orientation='horizontal', shrink=0.5, pad=0.06)
        cbar_front.set_ticks(cbar_front_ticks)
        cbar_front.set_ticklabels(cbar_front_labels)
        cbar_front.set_label('Front type')

        gl = ax.gridlines(draw_labels=True, alpha=0.3)
        gl.right_labels = False
        gl.top_labels = False

        ax.set_title(f"U-Net 3+ predictions", loc='right')
        ax.set_title('')
        ax.set_title(data_title, loc='left')

        plt.savefig('%s/%s.png' % (args['plot_dir'], filename_base), bbox_inches='tight', dpi=300)
        plt.close()
