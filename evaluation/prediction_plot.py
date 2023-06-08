"""
Plot model predictions.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 6/8/2023 1:38 PM CT
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
sys.path.append(os.getcwd())
from utils import data_utils, settings
from utils.plotting_utils import plot_background


if __name__ == '__main__':
    """
    All arguments listed in the examples are listed via argparse in alphabetical order below this comment block.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--datetime', type=int, nargs=4, help='Date and time of the data. Pass 4 ints in the following order: year, month, day, hour')
    parser.add_argument('--domain', type=str, required=True, help='Domain of the data.')
    parser.add_argument('--domain_images', type=int, nargs=2, help='Number of images for each dimension the final stitched map for predictions: lon, lat')
    parser.add_argument('--forecast_hour', type=int, help='Forecast hour for the GDAS data')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, required=True, help='Model number.')
    parser.add_argument('--fronts_netcdf_indir', type=str, help='Main directory for the netcdf files containing frontal objects.')
    parser.add_argument('--data_source', type=str, default='era5', help="Source of the variable data (ERA5, GDAS, etc.)")
    parser.add_argument('--prob_mask', type=float, nargs=2, default=[0.1, 0.1],
        help="Probability mask and the step/interval for the probability contours. Probabilities smaller than the mask will not be plotted.")
    parser.add_argument('--calibration_km', type=int,
        help="Neighborhood calibration distance in kilometers. Possible neighborhoods are 50, 100, 150, 200, and 250 km.")

    args = vars(parser.parse_args())
    
    DEFAULT_COLORBAR_POSITION = {'conus': 0.71, 'full': 0.80, 'global': 0.74}
    cbar_position = DEFAULT_COLORBAR_POSITION[args['domain']]

    model_properties = pd.read_pickle(f"{args['model_dir']}/model_{args['model_number']}/model_{args['model_number']}_properties.pkl")

    args['data_source'] = args['data_source'].lower()

    extent = settings.DEFAULT_DOMAIN_EXTENTS[args['domain']]

    year, month, day, hour = args['datetime'][0], args['datetime'][1], args['datetime'][2], args['datetime'][3]

    subdir_base = '%s_%dx%d' % (args['domain'], args['domain_images'][0], args['domain_images'][1])
    probs_dir = f"{args['model_dir']}/model_{args['model_number']}/probabilities/{subdir_base}"

    if args['forecast_hour'] is not None:
        forecast_timestep = data_utils.add_or_subtract_hours_to_timestep('%d%02d%02d%02d' % (year, month, day, hour), num_hours=args['forecast_hour'])
        new_year, new_month, new_day, new_hour = forecast_timestep[:4], forecast_timestep[4:6], forecast_timestep[6:8], int(forecast_timestep[8:]) - (int(forecast_timestep[8:]) % 3)
        fronts_file = '%s/%s%s/FrontObjects_%s%s%s%02d_full.nc' % (args['fronts_netcdf_indir'], new_year, new_month, new_year, new_month, new_day, new_hour)
        filename_base = f'model_%d_{year}-%02d-%02d-%02dz_%s_%s_f%03d_%dx%d' % (args['model_number'], month, day, hour, args['domain'], args['data_source'], args['forecast_hour'], args['domain_images'][0], args['domain_images'][1])
    else:
        fronts_file = '%s/%d%02d/FrontObjects_%d%02d%02d%02d_full.nc' % (args['fronts_netcdf_indir'], year, month, year, month, day, hour)
        filename_base = f'model_%d_{year}-%02d-%02d-%02dz_%s_%dx%d' % (args['model_number'], month, day, hour, args['domain'], args['domain_images'][0], args['domain_images'][1])
        args['data_source'] = 'era5'

    probs_file = f'{probs_dir}/{filename_base}_probabilities.nc'

    probs_ds = xr.open_mfdataset(probs_file)

    front_types = model_properties['front_types']

    try:
        fronts = xr.open_dataset(fronts_file).sel(longitude=slice(extent[0], extent[1]), latitude=slice(extent[3], extent[2]))
        fronts = data_utils.reformat_fronts(fronts, front_types=front_types)
        labels = fronts.attrs['labels']
        fronts = xr.where(fronts == 0, float('NaN'), fronts)
        fronts_found = True
    except FileNotFoundError:
        labels = front_types
        fronts_found = False

    if type(front_types) == str:
        front_types = [front_types, ]

    mask, prob_int = args['prob_mask'][0], args['prob_mask'][1]  # Probability mask, contour interval for probabilities
    vmax, cbar_tick_adjust, cbar_label_adjust, n_colors = 1, prob_int, 10, 11
    levels = np.around(np.arange(0, 1 + prob_int, prob_int), 2)
    cbar_ticks = np.around(np.arange(mask, 1 + prob_int, prob_int), 2)

    contour_maps_by_type = [settings.DEFAULT_CONTOUR_CMAPS[label] for label in labels]
    front_colors_by_type = [settings.DEFAULT_FRONT_COLORS[label] for label in labels]
    front_names_by_type = [settings.DEFAULT_FRONT_NAMES[label] for label in labels]

    cmap_front = colors.ListedColormap(front_colors_by_type, name='from_list', N=len(front_colors_by_type))
    norm_front = colors.Normalize(vmin=1, vmax=len(front_colors_by_type) + 1)

    data_arrays = {}
    for key in list(probs_ds.keys()):
        if args['calibration_km'] is not None:
            ir_model = model_properties['calibration_models']['%s_%dx%d' % (args['domain'], args['domain_images'][0], args['domain_images'][1])][key]['%d km' % args['calibration_km']]
            original_shape = np.shape(probs_ds[key].values)
            data_arrays[key] = ir_model.predict(probs_ds[key].values.flatten()).reshape(original_shape)
            cbar_label = 'Probability (calibrated - %d km)' % args['calibration_km']
        else:
            data_arrays[key] = probs_ds[key].values
            cbar_label = 'Probability (uncalibrated)'

    all_possible_front_combinations = itertools.permutations(front_types, r=2)
    for combination in all_possible_front_combinations:
        probs_ds[combination[0]].values = np.where(probs_ds[combination[0]].values > probs_ds[combination[1]].values - 0.02, probs_ds[combination[0]].values, 0)

    if args['data_source'] != 'era5':
        probs_ds = xr.where(probs_ds > mask, probs_ds, float("NaN")).isel(time=0).sel(forecast_hour=args['forecast_hour'])
        valid_time = data_utils.add_or_subtract_hours_to_timestep(f'%d%02d%02d%02d' % (year, month, day, hour), num_hours=args['forecast_hour'])
        data_title = f"Run: {args['data_source'].upper()} {year}-%02d-%02d-%02dz F%03d \nPredictions valid: {valid_time[:4]}-{valid_time[4:6]}-{valid_time[6:8]}-{valid_time[8:]}z" % (month, day, hour, args['forecast_hour'])
        fronts_valid_title = f'Fronts valid: {new_year}-{"%02d" % int(new_month)}-{"%02d" % int(new_day)}-{"%02d" % new_hour}z'
    else:
        probs_ds = xr.where(probs_ds > mask, probs_ds, float("NaN")).isel(time=0)
        data_title = 'Data: ERA5 reanalysis %d-%02d-%02d-%02dz\n' \
                     'Predictions valid: %d-%02d-%02d-%02dz' % (year, month, day, hour, year, month, day, hour)
        fronts_valid_title = f'Fronts valid: %d-%02d-%02d-%02dz' % (year, month, day, hour)

    fig, ax = plt.subplots(1, 1, figsize=(20, 8), subplot_kw={'projection': ccrs.Miller(central_longitude=250)})
    plot_background(extent, ax=ax, linewidth=0.5)
    # ax.gridlines(draw_labels=True, zorder=0)

    cbar_front_labels = []
    cbar_front_ticks = []

    for front_no, front_key, front_name, front_label, cmap in zip(range(1, len(front_names_by_type) + 1), list(probs_ds.keys()), front_names_by_type, front_types, contour_maps_by_type):

        cmap_probs, norm_probs = cm.get_cmap(cmap, n_colors), colors.Normalize(vmin=0, vmax=vmax)
        probs_ds[front_key].plot.contourf(ax=ax, x='longitude', y='latitude', norm=norm_probs, levels=levels, cmap=cmap_probs, transform=ccrs.PlateCarree(), alpha=0.75, add_colorbar=False)

        if fronts_found:
            fronts['identifier'].plot(ax=ax, x='longitude', y='latitude', cmap=cmap_front, norm=norm_front, transform=ccrs.PlateCarree(), add_colorbar=False)

        cbar_ax = fig.add_axes([cbar_position + (front_no * 0.015), 0.24, 0.015, 0.64])
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm_probs, cmap=cmap_probs), cax=cbar_ax, boundaries=levels[1:], alpha=0.75)
        cbar.set_ticklabels([])

        cbar_front_labels.append(front_name)
        cbar_front_ticks.append(front_no + 0.5)

    cbar.set_label(cbar_label, rotation=90)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticks)

    if fronts_found:
        cbar_front = plt.colorbar(cm.ScalarMappable(norm=norm_front, cmap=cmap_front), ax=ax, alpha=0.75, orientation='horizontal', shrink=0.5, pad=0.02)
        cbar_front.set_ticks(cbar_front_ticks)
        cbar_front.set_ticklabels(cbar_front_labels)
        ax.set_title(fronts_valid_title, loc='right')

    ax.set_title('')
    # ax.set_title(f"{'/'.join(front_name.replace(' front', '') for front_name in front_names_by_type)} predictions")
    ax.set_title(data_title, loc='left')

    plt.savefig('%s/model_%d/maps/%s/%s-same.png' % (args['model_dir'], args['model_number'], subdir_base, filename_base), bbox_inches='tight', dpi=500)
    plt.close()
