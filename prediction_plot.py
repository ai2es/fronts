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


def prediction_plot(model_number, model_dir, plot_dir, init_time, forecast_hours, domain, domain_images, variable_data_source='gdas',
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
    probs_dir = f'{model_dir}/model_{model_number}/predictions'

    if domain == 'conus':
        cbar_loc = 0.8365
    elif domain == 'full':
        cbar_loc = 0.8965

    for forecast_hour in forecast_hours:
        filename_base = f'model_%d_{year}-%02d-%02d-%02dz_%s_f%03d_%s_%dx%d' % (model_number, month, day, hour, variable_data_source, forecast_hour, domain, domain_images[0], domain_images[1])

        probs_file = f'{probs_dir}/{filename_base}_probabilities.nc'

        probs_ds = xr.open_mfdataset(probs_file)

        crs = ccrs.PlateCarree(central_longitude=250)

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

            probs_ds['CF_obj'] = xr.where(probs_ds['CF'] > 0.38, 1, 0)
            probs_ds['WF_obj'] = xr.where(probs_ds['WF'] > 0.33, 1, 0)
            probs_ds['SF_obj'] = xr.where(probs_ds['SF'] > 0.20, 1, 0)
            probs_ds['OF_obj'] = xr.where(probs_ds['OF'] > 0.21, 1, 0)
            probs_ds['F_BIN_obj'] = xr.where(probs_ds['F_BIN'] > 0.32, 1, 0)

            probs_ds = xr.where(probs_ds > probability_mask, probs_ds, 0).isel(time=0)
            probs_ds = xr.where(probs_ds == 0, float("NaN"), probs_ds)
            valid_time = data_utils.add_or_subtract_hours_to_timestep(f'%d%02d%02d%02d' % (year, month, day, hour), num_hours=forecast_hour)
            data_title = f'Run: {variable_data_source.upper()} {year}-%02d-%02d-%02dz F%03d \nPredictions valid: {valid_time[:4]}-{valid_time[4:6]}-{valid_time[6:8]}-{valid_time[8:]}z' % (month, day, hour, forecast_hour)

            fig, ax = plt.subplots(1, 1, figsize=(20, 8), subplot_kw={'projection': crs})
            plot_background(extent, ax=ax, linewidth=0.5)
            ax.gridlines(draw_labels=True, zorder=0, alpha=0.4)

            for front_no, front_key, front_name, front_label, cmap in zip(range(1, len(front_names_by_type) + 1), list(probs_ds.keys()), front_names_by_type, front_types, contour_maps_by_type):

                cmap_probs, norm_probs = cm.get_cmap(cmap, n_colors), colors.Normalize(vmin=0, vmax=vmax)
                probs_ds[front_key].plot.contourf(ax=ax, x='longitude', y='latitude', norm=norm_probs, levels=levels, cmap=cmap_probs, transform=ccrs.PlateCarree(), alpha=0.75, add_colorbar=False)
                probs_ds[f'{front_key}_obj'].plot(ax=ax, x='longitude', y='latitude', cmap=settings.DEFAULT_CONTOUR_CMAPS[front_key], transform=ccrs.PlateCarree(), add_colorbar=False)

                cbar_ax = fig.add_axes([cbar_loc + (front_no * 0.015), 0.11, 0.015, 0.77])
                cbar = plt.colorbar(cm.ScalarMappable(norm=norm_probs, cmap=cmap_probs), cax=cbar_ax, boundaries=levels[1:], alpha=0.75)
                cbar.set_ticklabels([])

            cbar.set_label('Probability (uncalibrated)', rotation=90)
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_tick_labels[int(probability_mask*cbar_label_adjust):])

            ax.set_title(f"{'/'.join(front_name.replace(' front', '') for front_name in front_names_by_type)} predictions")
            ax.set_title(data_title, loc='left')

            plt.savefig('%s/%s.png' % (plot_dir, filename_base), bbox_inches='tight', dpi=300)
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
                plt.savefig('%s/%s-%s.png' % (plot_dir, filename_base, front_label), bbox_inches='tight', dpi=300)
                plt.close()


if __name__ == '__main__':
    """
    All arguments listed in the examples are listed via argparse in alphabetical order below this comment block.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--init_time', type=int, nargs=4, required=True, help='Date and time of the data. Pass 4 ints in the following order: year, month, day, hour')
    parser.add_argument('--forecast_hours', type=int, nargs='+', required=True, help='Forecast hour for the GDAS data')

    parser.add_argument('--domain', type=str, required=True, help='Domain of the data.')
    parser.add_argument('--domain_images', type=int, nargs=2, help='Number of images for each dimension the final stitched map for predictions: lon, lat')

    parser.add_argument('--model_dir', type=str, required=True, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, required=True, help='Model number.')
    parser.add_argument('--plot_dir', type=str, help='plot directory')

    parser.add_argument('--variable_data_source', type=str, default='gdas', help='Data source for variables')

    args = parser.parse_args()

    if args.domain_images is None:
        domain_images = settings.DEFAULT_DOMAIN_IMAGES[args.domain]
    else:
        domain_images = args.domain_images

    prediction_plot(args.model_number, args.model_dir, args.plot_dir, args.init_time, args.forecast_hours, args.domain,
        domain_images, variable_data_source=args.variable_data_source)
