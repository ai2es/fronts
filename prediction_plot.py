"""
Functions used for evaluating a U-Net model.

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 2/9/2023 10:49 PM CT

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


def prediction_plot(model_number, model_dir, init_time, forecast_hour, domain, domain_images, variable_data_source='gdas',
    probability_mask_2D=0.05, probability_mask_3D=0.10, same_map=True):
    """
    Function that uses generated predictions to make probability maps along with the 'true' fronts and saves out the
    subplots.

    Parameters
    ----------
    model_number: int
        - Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        - Main directory for the models.
    init_time: str
        - Timestring for the prediction plot title.
    probability_mask_2D: float
        - Mask for front probabilities with 2D models. Any probabilities smaller than this number will not be plotted.
        - Must be a multiple of 0.05, greater than 0, and no greater than 0.45.
    probability_mask_3D: float
        - Mask for front probabilities with 3D models. Any probabilities smaller than this number will not be plotted.
        - Must be a multiple of 0.1, greater than 0, and no greater than 0.9.
    """

    variable_data_source = variable_data_source.lower()

    extent = settings.DEFAULT_DOMAIN_EXTENTS[domain]

    year, month, day, hour = int(init_time[0]), int(init_time[1]), int(init_time[2]), int(init_time[3])

    subdir_base = '%s_%dx%d' % (domain, domain_images[0], domain_images[1])
    probs_dir = f'{model_dir}/model_{model_number}/probabilities/{subdir_base}'

    forecast_timestep = data_utils.add_or_subtract_hours_to_timestep('%d%02d%02d%02d' % (year, month, day, hour), num_hours=forecast_hour)
    new_year, new_month, new_day, new_hour = forecast_timestep[:4], forecast_timestep[4:6], forecast_timestep[6:8], int(forecast_timestep[8:]) - (int(forecast_timestep[8:]) % 3)
    filename_base = f'model_%d_{year}-%02d-%02d-%02dz_%s_%s_f%03d_%dx%d' % (model_number, month, day, hour, domain, variable_data_source, forecast_hour, domain_images[0], domain_images[1])

    probs_file = f'{probs_dir}/{filename_base}_probabilities.nc'

    probs_ds = xr.open_mfdataset(probs_file)

    crs = ccrs.LambertConformal(central_longitude=250)

    model_properties = pd.read_pickle(f"{model_dir}/model_{model_number}/model_{model_number}_properties.pkl")

    image_size = model_properties['image_size']  # The image size does not include the last dimension of the input size as it only represents the number of channels
    front_types = model_properties['front_types']

    if type(front_types) == str:
        front_types = [front_types, ]

    num_dimensions = len(image_size)
    if model_number not in [7805504, 7866106, 7961517]:
        num_dimensions += 1

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

    contour_maps_by_type = [settings.DEFAULT_CONTOUR_CMAPS[label] for label in front_types]
    front_names_by_type = [settings.DEFAULT_FRONT_NAMES[label] for label in front_types]

    if same_map:

        if type(front_types) == list and len(front_types) > 1:

            data_arrays = {}
            for key in list(probs_ds.keys()):
                data_arrays[key] = probs_ds[key].values

            all_possible_front_combinations = itertools.permutations(front_types, r=2)
            for combination in all_possible_front_combinations:
                probs_ds[combination[0]].values = np.where(data_arrays[combination[0]] > data_arrays[combination[1]] - 0.02, data_arrays[combination[0]], 0)

        probs_ds = xr.where(probs_ds > probability_mask, probs_ds, float("NaN")).isel(time=0).sel(forecast_hour=forecast_hour)
        valid_time = data_utils.add_or_subtract_hours_to_timestep(f'%d%02d%02d%02d' % (year, month, day, hour), num_hours=forecast_hour)
        data_title = f'Run: {variable_data_source.upper()} {year}-%02d-%02d-%02dz F%03d \nPredictions valid: {valid_time[:4]}-{valid_time[4:6]}-{valid_time[6:8]}-{valid_time[8:]}z' % (month, day, hour, forecast_hour)
        fronts_valid_title = f'Fronts valid: {new_year}-{"%02d" % int(new_month)}-{"%02d" % int(new_day)}-{"%02d" % new_hour}z'

        fig, ax = plt.subplots(1, 1, figsize=(20, 8), subplot_kw={'projection': crs})
        plot_background(extent, ax=ax, linewidth=0.5)
        # ax.gridlines(draw_labels=True, zorder=0)

        for front_no, front_key, front_name, front_label, cmap in zip(range(1, len(front_names_by_type) + 1), list(probs_ds.keys()), front_names_by_type, front_types, contour_maps_by_type):

            cmap_probs, norm_probs = cm.get_cmap(cmap, n_colors), colors.Normalize(vmin=0, vmax=vmax)
            probs_ds[front_key].plot.contourf(ax=ax, x='longitude', y='latitude', norm=norm_probs, levels=levels, cmap=cmap_probs, transform=ccrs.PlateCarree(), alpha=0.75, add_colorbar=False)

            cbar_ax = fig.add_axes([0.7365 + (front_no * 0.015), 0.11, 0.015, 0.77])
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm_probs, cmap=cmap_probs), cax=cbar_ax, boundaries=levels[1:], alpha=0.75)
            cbar.set_ticklabels([])

        cbar.set_label('Probability (uncalibrated)', rotation=90)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_tick_labels[int(probability_mask*cbar_label_adjust):])

        ax.set_title(f"{'/'.join(front_name.replace(' front', '') for front_name in front_names_by_type)} predictions")
        ax.set_title(data_title, loc='left')
        ax.set_title(fronts_valid_title, loc='right')

        plt.savefig('%s/model_%d/maps/%s/%s-same.png' % (model_dir, model_number, subdir_base, filename_base), bbox_inches='tight', dpi=300)
        plt.close()

    else:

        probs_ds = xr.where(probs_ds > probability_mask, probs_ds, float("NaN")).isel(time=0).sel(forecast_hour=forecast_hour)
        for front_no, front_key, front_name, front_label, cmap in zip(range(1, len(front_names_by_type) + 1), list(probs_ds.keys()), front_names_by_type, front_types, contour_maps_by_type):
            fig, ax = plt.subplots(1, 1, figsize=(20, 8), subplot_kw={'projection': crs})
            plot_background(extent, ax=ax, linewidth=0.5)
            cmap_probs, norm_probs = cm.get_cmap(cmap, n_colors), colors.Normalize(vmin=0, vmax=vmax)
            probs_ds[front_key].sel(forecast_hour=forecast_hour).plot.contourf(ax=ax, x='longitude', y='latitude', norm=norm_probs, levels=levels, cmap=cmap_probs,
                transform=ccrs.PlateCarree(), alpha=0.75, add_colorbar=False)
            valid_time = data_utils.add_or_subtract_hours_to_timestep(f'{year}%02d%02d%02d' % (month, day, hour), num_hours=forecast_hour)
            ax.set_title(f'{front_name} predictions')
            ax.set_title(f'Run: {variable_data_source.upper()} {year}-%02d-%02d-%02dz F%03d \nPredictions valid: {valid_time[:4]}-{valid_time[4:6]}-{valid_time[6:8]}-{valid_time[8:]}z' % (month, day, hour, forecast_hour), loc='left')
            cbar_ax = fig.add_axes([0.8365, 0.11, 0.015, 0.77])
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm_probs, cmap=cmap_probs), cax=cbar_ax, boundaries=levels[1:], alpha=0.75)
            cbar.set_label('Probability (uncalibrated)', rotation=90)
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_tick_labels[int(probability_mask*cbar_label_adjust):])
            plt.savefig('%s/model_%d/maps/%s/%s-%s.png' % (model_dir, model_number, subdir_base, filename_base, front_label), bbox_inches='tight', dpi=300)
            plt.close()


if __name__ == '__main__':
    """
    All arguments listed in the examples are listed via argparse in alphabetical order below this comment block.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', action='store_true', help='Bootstrap data?')
    parser.add_argument('--dataset', type=str, help="Dataset for which to make predictions if prediction_method is 'random' or 'all'. Options are:"
                                                    "'training', 'validation', 'test'")
    parser.add_argument('--datetime', type=int, nargs=4, help='Date and time of the data. Pass 4 ints in the following order: year, month, day, hour')
    parser.add_argument('--init_time', type=int, nargs=3, help='Date and time of the data. Pass 3 ints in the following order: year, month, day ')
    parser.add_argument('--domain', type=str, help='Domain of the data.')
    parser.add_argument('--domain_images', type=int, nargs=2, help='Number of images for each dimension the final stitched map for predictions: lon, lat')
    parser.add_argument('--domain_size', type=int, nargs=2, help='Lengths of the dimensions of the final stitched map for predictions: lon, lat')
    parser.add_argument('--forecast_hour', type=int, help='Forecast hour for the GDAS data')
    parser.add_argument('--find_matches', action='store_true', help='Find matches for stitching predictions?')
    parser.add_argument('--generate_predictions', action='store_true', help='Generate prediction plots?')
    parser.add_argument('--calculate_stats', action='store_true', help='generate stats')
    parser.add_argument('--gpu_device', type=int, help='GPU device number.')
    parser.add_argument('--image_size', type=int, nargs=2, help="Number of pixels along each dimension of the model's output: lon, lat")
    parser.add_argument('--learning_curve', action='store_true', help='Plot learning curve')
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth on the GPU')
    parser.add_argument('--model_dir', type=str, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, help='Model number.')
    parser.add_argument('--num_iterations', type=int, default=10000, help='Number of iterations to perform when bootstrapping the data.')
    parser.add_argument('--num_rand_predictions', type=int, default=10, help='Number of random predictions to make.')
    parser.add_argument('--fronts_netcdf_indir', type=str, help='Main directory for the netcdf files containing frontal objects.')
    parser.add_argument('--variables_netcdf_indir', type=str, help='Main directory for the netcdf files containing variable data.')
    parser.add_argument('--plot_performance_diagrams', action='store_true', help='Plot performance diagrams for a model?')
    parser.add_argument('--prediction_method', type=str, help="Prediction method. Options are: 'datetime', 'random', 'all'")
    parser.add_argument('--prediction_plot', action='store_true', help='Create plot')
    parser.add_argument('--random_variables', type=str, nargs="+", default=None, help="Variables to randomize when generating predictions.")
    parser.add_argument('--save_map', action='store_true', help='Save maps of the model predictions?')
    parser.add_argument('--save_probabilities', action='store_true', help='Save model prediction data out to netcdf files?')
    parser.add_argument('--save_statistics', action='store_true', help='Save performance statistics data out to netcdf files?')
    parser.add_argument('--variable_data_source', type=str, default='era5', help='Data source for variables')

    args = parser.parse_args()
    provided_arguments = vars(args)

    if args.prediction_plot:
        required_arguments = ['model_number', 'model_dir', 'fronts_netcdf_indir', 'datetime', 'domain_images']
        prediction_plot(args.model_number, args.model_dir, args.init_time, args.forecast_hour, args.domain,
            args.domain_images, forecast_hour=args.forecast_hour, variable_data_source=args.variable_data_source)
