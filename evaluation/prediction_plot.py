"""
Plot model predictions.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.1.5

TODO: Fix colorbar position issues
"""
import itertools
import argparse
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm, colors  # Here we explicitly import the cm and color modules to suppress a PyCharm bug
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # this line allows us to import scripts outside the current directory
from utils import data_utils, settings
from utils.plotting_utils import plot_background
from skimage.morphology import skeletonize


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_time', type=int, nargs=4, help='Date and time of the data. Pass 4 ints in the following order: year, month, day, hour')
    parser.add_argument('--domain', type=str, required=True, help='Domain of the data.')
    parser.add_argument('--domain_images', type=int, nargs=2, help='Number of images for each dimension the final stitched map for predictions: lon, lat')
    parser.add_argument('--forecast_hour', type=int, help='Forecast hour for the GDAS data')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, required=True, help='Model number.')
    parser.add_argument('--fronts_netcdf_indir', type=str, help='Main directory for the netcdf files containing frontal objects.')
    parser.add_argument('--data_source', type=str, default='era5', help="Source of the variable data (ERA5, GDAS, etc.)")
    parser.add_argument('--prob_mask', type=float, nargs=2, default=[0.1, 0.1],
        help="Probability mask and the step/interval for the probability contours. Probabilities smaller than the mask will not be plotted.")
    parser.add_argument('--calibration', type=int,
        help="Neighborhood calibration distance in kilometers. Possible neighborhoods are 50, 100, 150, 200, and 250 km.")
    parser.add_argument('--deterministic', action='store_true', help="Plot deterministic splines.")
    parser.add_argument('--targets', action='store_true', help="Plot ground truth targets/labels.")
    parser.add_argument('--open_contours', action='store_true', help="Plot probability contours.")
    parser.add_argument('--filled_contours', action='store_true', help="Plot probability contours.")

    args = vars(parser.parse_args())

    if args['deterministic'] and args['targets']:
        raise TypeError("Cannot plot deterministic splines and ground truth targets at the same time. Only one of --deterministic, --targets may be passed")

    if args['domain_images'] is None:
        args['domain_images'] = [1, 1]

    DEFAULT_COLORBAR_POSITION = {'conus': 0.78, 'full': 0.84, 'global': 0.74}
    cbar_position = DEFAULT_COLORBAR_POSITION['conus']

    model_properties = pd.read_pickle(f"{args['model_dir']}/model_{args['model_number']}/model_{args['model_number']}_properties.pkl")

    args['data_source'] = args['data_source'].lower()

    extent = settings.DOMAIN_EXTENTS[args['domain']]

    year, month, day, hour = args['init_time'][0], args['init_time'][1], args['init_time'][2], args['init_time'][3]

    ### Attempt to pull predictions from a monthly netcdf file generated with tensorflow datasets, otherwise try to pull a single netcdf file ###
    try:
        probs_file = f"{args['model_dir']}/model_{args['model_number']}/probabilities/model_{args['model_number']}_pred_{args['domain']}_{year}%02d.nc" % month
        fronts_file = '%s/%d%02d/FrontObjects_%d%02d%02d%02d_full.nc' % (args['fronts_netcdf_indir'], year, month, year, month, day, hour)
        plot_filename = '%s/model_%d/maps/model_%d_%d%02d%02d%02d_%s.png' % (args['model_dir'], args['model_number'], args['model_number'], year, month, day, hour, args['domain'])
        probs_ds = xr.open_mfdataset(probs_file).sel(time=['%d-%02d-%02dT%02d' % (year, month, day, hour), ])
    except OSError:
        subdir_base = '%s_%dx%d' % (args['domain'], args['domain_images'][0], args['domain_images'][1])
        probs_dir = f"{args['model_dir']}/model_{args['model_number']}/probabilities/{subdir_base}"

        if args['forecast_hour'] is not None:
            timestep = np.datetime64('%d-%02d-%02dT%02d' % (year, month, day, hour)).astype(object)
            forecast_timestep = timestep if args['forecast_hour'] == 0 else timestep + np.timedelta64(args['forecast_hour'], 'h').astype(object)
            new_year, new_month, new_day, new_hour = forecast_timestep.year, forecast_timestep.month, forecast_timestep.day, forecast_timestep.hour - (forecast_timestep.hour % 3)
            fronts_file = '%s/%s%s/FrontObjects_%s%s%s%02d_full.nc' % (args['fronts_netcdf_indir'], new_year, new_month, new_year, new_month, new_day, new_hour)
            filename_base = f'model_%d_{year}%02d%02d%02d_%s_%s_f%03d_%dx%d' % (args['model_number'], month, day, hour, args['domain'], args['data_source'], args['forecast_hour'], args['domain_images'][0], args['domain_images'][1])
        else:
            fronts_file = '%s/%d%02d/FrontObjects_%d%02d%02d%02d_full.nc' % (args['fronts_netcdf_indir'], year, month, year, month, day, hour)
            filename_base = f'model_%d_{year}%02d%02d%02d_%s_%dx%d' % (args['model_number'], month, day, hour, args['domain'], args['domain_images'][0], args['domain_images'][1])
            args['data_source'] = 'era5'

        plot_filename = '%s/model_%d/maps/%s/%s-same.png' % (args['model_dir'], args['model_number'], subdir_base, filename_base)
        probs_file = f'{probs_dir}/{filename_base}_probabilities.nc'
        probs_ds = xr.open_mfdataset(probs_file)

    try:
        front_types = model_properties['dataset_properties']['front_types']
    except KeyError:
        front_types = model_properties['front_types']

    labels = front_types
    fronts_found = False

    if args['targets']:
        right_title = 'Splines: NOAA fronts'
        try:
            fronts = xr.open_dataset(fronts_file).sel(longitude=slice(extent[0], extent[1]), latitude=slice(extent[3], extent[2]))
            fronts = data_utils.reformat_fronts(fronts, front_types=front_types)
            labels = fronts.attrs['labels']
            fronts = xr.where(fronts == 0, float('NaN'), fronts)
            fronts_found = True
        except FileNotFoundError:
            print("No ground truth fronts found")

    if type(front_types) == str:
        front_types = [front_types, ]

    mask, prob_int = args['prob_mask'][0], args['prob_mask'][1]  # Probability mask, contour interval for probabilities
    vmax, cbar_tick_adjust, cbar_label_adjust, n_colors = 1, prob_int, 10, 11
    levels = np.around(np.arange(0, 1 + prob_int, prob_int), 2)
    cbar_ticks = np.around(np.arange(mask, 1 + prob_int, prob_int), 2)

    contour_maps_by_type = [settings.CONTOUR_CMAPS[label] for label in labels]
    front_colors_by_type = [settings.FRONT_COLORS[label] for label in labels]
    front_names_by_type = [settings.FRONT_NAMES[label] for label in labels]

    cmap_front = colors.ListedColormap(front_colors_by_type, name='from_list', N=len(front_colors_by_type))
    norm_front = colors.Normalize(vmin=1, vmax=len(front_colors_by_type) + 1)

    probs_ds = probs_ds.isel(time=0) if args['data_source'] == 'era5' else probs_ds.isel(time=0, forecast_hour=0)
    probs_ds = probs_ds.transpose('latitude', 'longitude')

    for key in list(probs_ds.keys()):

        if args['deterministic']:
            spline_threshold = model_properties['front_obj_thresholds'][args['domain']][key]['100']
            probs_ds[f'{key}_obj'] = (('latitude', 'longitude'), skeletonize(xr.where(probs_ds[key] > spline_threshold, 1, 0).values.copy(order='C')))

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

    if not args['open_contours']:
        if len(front_types) > 1:
            all_possible_front_combinations = itertools.permutations(front_types, r=2)
            for combination in all_possible_front_combinations:
                probs_ds[combination[0]].values = np.where(probs_ds[combination[0]].values > probs_ds[combination[1]].values - 0.02, probs_ds[combination[0]].values, 0)

    probs_ds = xr.where(probs_ds > mask, probs_ds, float("NaN"))
    if args['data_source'] != 'era5':
        valid_time = timestep + np.timedelta64(args['forecast_hour'], 'h').astype(object)
        data_title = f"Run: {args['data_source'].upper()} {year}-%02d-%02d-%02dz F%03d \nPredictions valid: %d-%02d-%02d-%02dz" % (month, day, hour, args['forecast_hour'], valid_time.year, valid_time.month, valid_time.day, valid_time.hour)
    else:
        data_title = 'Data: ERA5 reanalysis %d-%02d-%02d-%02dz\n' \
                     'Predictions valid: %d-%02d-%02d-%02dz' % (year, month, day, hour, year, month, day, hour)

    fig, ax = plt.subplots(1, 1, figsize=(22, 8), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
    plot_background(extent, ax=ax, linewidth=0.5)
    # ax.gridlines(draw_labels=True, zorder=0)

    cbar_front_labels = []
    cbar_front_ticks = []

    for front_no, front_key, front_name, front_label, cmap in zip(range(1, len(front_names_by_type) + 1), list(probs_ds.keys()), front_names_by_type, front_types, contour_maps_by_type):

        if args['filled_contours']:
            cmap_probs, norm_probs = cm.get_cmap(cmap, n_colors), colors.Normalize(vmin=0, vmax=vmax)
            probs_ds[front_key].plot.contourf(ax=ax, x='longitude', y='latitude', norm=norm_probs, levels=levels, cmap=cmap_probs,
                                              transform=ccrs.PlateCarree(), alpha=0.75, add_colorbar=False)
            cbar_ax = fig.add_axes([cbar_position + (front_no * 0.015), 0.24, 0.015, 0.64])
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm_probs, cmap=cmap_probs), cax=cbar_ax, boundaries=levels[1:], alpha=0.75)
            cbar.set_ticklabels([])

        if args['open_contours']:
            probs_ds[front_key].plot.contour(ax=ax, x='longitude', y='latitude', levels=levels,
                colors=front_colors_by_type[front_no - 1], transform=ccrs.PlateCarree(), alpha=0.75, add_colorbar=False)

        if args['deterministic']:
            right_title = 'Splines: Deterministic first-guess fronts'
            cmap_deterministic = colors.ListedColormap(['None', front_colors_by_type[front_no - 1]], name='from_list', N=2)
            norm_deterministic = colors.Normalize(vmin=0, vmax=1)
            probs_ds[f'{front_key}_obj'].plot(ax=ax, x='longitude', y='latitude', cmap=cmap_deterministic, norm=norm_deterministic,
                                              transform=ccrs.PlateCarree(), alpha=0.9, add_colorbar=False)

        if fronts_found:
            fronts['identifier'].plot(ax=ax, x='longitude', y='latitude', cmap=cmap_front, norm=norm_front, transform=ccrs.PlateCarree(), add_colorbar=False)

        cbar_front_labels.append(front_name)
        cbar_front_ticks.append(front_no + 0.5)

    if args['filled_contours']:
        cbar.set_label(cbar_label, rotation=90)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticks)

    cbar_front = plt.colorbar(cm.ScalarMappable(norm=norm_front, cmap=cmap_front), ax=ax, alpha=0.75, orientation='horizontal', shrink=0.5, pad=0.02)
    cbar_front.set_ticks(cbar_front_ticks)
    cbar_front.set_ticklabels(cbar_front_labels)
    cbar_front.set_label(r'$\bf{Front}$ $\bf{type}$')

    if fronts_found or args['deterministic']:
        ax.set_title(right_title, loc='right')

    ax.set_title('')
    ax.set_title(data_title, loc='left')

    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    plt.close()
