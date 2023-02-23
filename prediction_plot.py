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
    probability_mask=(0.10, 0.10), calibration_km=None, objects=False):
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
    """

    DEFAULT_COLORBAR_POSITION = {'conus': 0.76, 'full': 0.83}
    FRONT_OBJ_THRESHOLDS = {'CF': {'conus': {'50': 0.35, '100': 0.31, '150': 0.27, '200': 0.25, '250': 0.24},
                                   'full': {'50': 0.48, '100': 0.44, '150': 0.42, '200': 0.39, '250': 0.37}},
                            'WF': {'conus': {'50': 0.31, '100': 0.29, '150': 0.28, '200': 0.27, '250': 0.26},
                                   'full': {'50': 0.37, '100': 0.35, '150': 0.34, '200': 0.33, '250': 0.33}},
                            'SF': {'conus': {'50': 0.29, '100': 0.26, '150': 0.24, '200': 0.22, '250': 0.20},
                                   'full': {'50': 0.38, '100': 0.36, '150': 0.34, '200': 0.33, '250': 0.32}},
                            'OF': {'conus': {'50': 0.26, '100': 0.24, '150': 0.23, '200': 0.21, '250': 0.21},
                                   'full': {'50': 0.24, '100': 0.23, '150': 0.22, '200': 0.21, '250': 0.20}},
                            'F_BIN': {'conus': {'50': 0.42, '100': 0.37, '150': 0.32, '200': 0.29, '250': 0.26},
                                      'full': {'50': 0.50, '100': 0.46, '150': 0.42, '200': 0.39, '250': 0.38}}}

    variable_data_source = variable_data_source.lower()
    extent = settings.DEFAULT_DOMAIN_EXTENTS[domain]

    year, month, day, hour = int(init_time[0]), int(init_time[1]), int(init_time[2]), int(init_time[3])
    probs_dir = f'{model_dir}/model_{model_number}/predictions'

    cbar_position = DEFAULT_COLORBAR_POSITION[domain]

    for forecast_hour in forecast_hours:
        filename_base = f'model_%d_{year}-%02d-%02d-%02dz_%s_f%03d_%s_%dx%d' % (model_number, month, day, hour, variable_data_source, forecast_hour, domain, domain_images[0], domain_images[1])

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

        mask, prob_int = probability_mask[0], probability_mask[1]  # Probability mask, contour interval for probabilities
        vmax, cbar_tick_adjust, cbar_label_adjust, n_colors = 1, prob_int, 10, 11
        levels = np.around(np.arange(0, 1 + prob_int, prob_int), 2)
        cbar_ticks = np.around(np.arange(mask, 1 + prob_int, prob_int), 2)

        contour_maps_by_type = [settings.DEFAULT_CONTOUR_CMAPS[label] for label in front_types]
        front_names_by_type = [settings.DEFAULT_FRONT_NAMES[label] for label in front_types]
        front_colors_by_type = [settings.DEFAULT_FRONT_COLORS[label] for label in front_types]

        probs_ds = xr.where(probs_ds > mask, probs_ds, 0).isel(time=0)

        data_arrays = {}
        for key in list(probs_ds.keys()):
            if calibration_km is not None:
                ir_model = model_properties['calibration_models']['%s_%dx%d' % (domain, domain_images[0], domain_images[1])][key]['%d km' % calibration_km]
                original_shape = np.shape(probs_ds[key].values)
                data_arrays[key] = ir_model.predict(probs_ds[key].values.flatten()).reshape(original_shape)
                cbar_label = 'Probability (calibrated - %d km)' % calibration_km
                probs_ds[f'{key}_obj'] = xr.where(probs_ds[key] > FRONT_OBJ_THRESHOLDS[key][domain][str(calibration_km)], 1, 0)
            else:
                data_arrays[key] = probs_ds[key].values
                cbar_label = 'Probability (uncalibrated)'
                probs_ds[f'{key}_obj'] = xr.where(probs_ds[key] > FRONT_OBJ_THRESHOLDS[key][domain]['50'], 1, 0)

        if type(front_types) == list and len(front_types) > 1:

            all_possible_front_combinations = itertools.permutations(front_types, r=2)
            for combination in all_possible_front_combinations:
                probs_ds[combination[0]].values = np.where(data_arrays[combination[0]] > data_arrays[combination[1]] - 0.02, data_arrays[combination[0]], 0)

        probs_ds = xr.where(probs_ds == 0, float("NaN"), probs_ds)

        valid_time = data_utils.add_or_subtract_hours_to_timestep(f'%d%02d%02d%02d' % (year, month, day, hour), num_hours=forecast_hour)
        data_title = f'Run: {variable_data_source.upper()} {year}-%02d-%02d-%02dz F%03d \nPredictions valid: {valid_time[:4]}-{valid_time[4:6]}-{valid_time[6:8]}-{valid_time[8:]}z' % (month, day, hour, forecast_hour)

        fig, ax = plt.subplots(1, 1, figsize=(20, 8), subplot_kw={'projection': crs})
        plot_background(extent, ax=ax, linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, zorder=0, alpha=0.4)
        gl.right_labels = False

        for front_no, front_key, front_name, front_label, cmap in zip(range(1, len(front_names_by_type) + 1), front_types, front_names_by_type, front_types, contour_maps_by_type):
            cmap_probs, norm_probs = cm.get_cmap(cmap, n_colors), colors.Normalize(vmin=0, vmax=vmax)
            if not objects:
                probs_ds[front_key].plot.contourf(ax=ax, x='longitude', y='latitude', norm=norm_probs, levels=levels, cmap=cmap_probs, transform=ccrs.PlateCarree(), alpha=0.75, add_colorbar=False)
                cbar_ax = fig.add_axes([cbar_position + (front_no * 0.015), 0.11, 0.015, 0.77])
                cbar = plt.colorbar(cm.ScalarMappable(norm=norm_probs, cmap=cmap_probs), cax=cbar_ax, boundaries=levels[1:], alpha=0.75)
                cbar.set_ticklabels([])
            else:
                cmap_front = colors.ListedColormap(['None', front_colors_by_type[front_no - 1]], name='from_list', N=2)
                norm_front = colors.Normalize(vmin=0, vmax=1)
                probs_ds[f'{front_key}_obj'].plot(ax=ax, x='longitude', y='latitude', cmap=cmap_front, norm=norm_front, transform=ccrs.PlateCarree(), alpha=0.9, add_colorbar=False)

        if not objects:
            cbar.set_label(cbar_label, rotation=90)
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_ticks)

        ax.set_title(f"{'/'.join(front_name.replace(' front', '') for front_name in front_names_by_type)} predictions")
        ax.set_title(data_title, loc='left')

        plt.savefig('%s/%s.png' % (plot_dir, filename_base), bbox_inches='tight', dpi=300)
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
    parser.add_argument('--calibration_km', type=int, help='Neighborhood to use for calibrating model probabilities')
    parser.add_argument('--objects', action='store_true', help='Plot fronts as objects instead of probability contours')

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
        domain_images, variable_data_source=args.variable_data_source, calibration_km=args.calibration_km, objects=args.objects)
