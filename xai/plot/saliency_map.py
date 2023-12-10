"""
Plot saliency maps for model predictions.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.12.9
"""

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
from utils import settings
from utils.plotting_utils import plot_background


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, required=True, help='Model number.')
    parser.add_argument('--plot_outdir', type=str, help='Directory for the saliency maps.')
    parser.add_argument('--init_time', type=str, help="Initialization time of the data. Format: YYYY-MM-DD-HH.")
    parser.add_argument('--domain', type=str, required=True, help='Domain of the data.')
    parser.add_argument('--calibration', type=int,
        help="Neighborhood calibration distance in kilometers. Possible neighborhoods are 50, 100, 150, 200, and 250 km.")
    parser.add_argument('--data_source', type=str, default='era5', help="Source of the variable data (ERA5, GDAS, etc.)")
    parser.add_argument('--cmap', type=str, default='viridis', help="Colormap to use for the saliency maps.")
    parser.add_argument('--extent', type=int, nargs=4,
        help="Extent of the saliency maps to plot. If this argument is not provided, use the default domain extent.")

    args = vars(parser.parse_args())

    init_time = pd.date_range(args['init_time'], args['init_time'])[0]
    extent = settings.DEFAULT_DOMAIN_EXTENTS[args['domain']] if args['extent'] is None else args['extent']

    model_properties = pd.read_pickle(f"{args['model_dir']}/model_{args['model_number']}/model_{args['model_number']}_properties.pkl")
    front_types = model_properties['dataset_properties']['front_types']

    salmap_folder = "%s/model_%d/saliencymaps" % (args['model_dir'], args['model_number'])
    salmap_ds = xr.open_dataset('%s/model_%d_salmap_%s_%s_%d%02d.nc' % (salmap_folder, args['model_number'], args['domain'], args['data_source'], init_time.year, init_time.month))

    probs_folder = '%s/model_%d/probabilities' % (args['model_dir'], args['model_number'])
    probs_ds = xr.open_dataset('%s/model_%d_pred_%s_%d%02d.nc' % (probs_folder, args['model_number'], args['domain'], init_time.year, init_time.month)).sel(time=init_time)
    levels = np.around(np.arange(0, 1.1, 0.1), 2)

    for front_type in front_types:

        if args['calibration'] is not None:
            try:
                ir_model = model_properties['calibration_models'][args['domain']][front_type]['%d km' % args['calibration']]
            except KeyError:
                ir_model = model_properties['calibration_models']['conus'][front_type]['%d km' % args['calibration']]
            original_shape = np.shape(probs_ds[front_type].values)
            probs_ds[front_type].values = ir_model.predict(probs_ds[front_type].values.flatten()).reshape(original_shape)
            cbar_label = 'Probability (calibrated - %d km)' % args['calibration']
        else:
            cbar_label = 'Probability (uncalibrated)'

        # mask out low probabilities
        probs_ds[front_type].values = np.where(probs_ds[front_type].values < 0.1, np.nan, probs_ds[front_type].values)

        cmap_probs, norm = cm.get_cmap(settings.DEFAULT_CONTOUR_CMAPS[front_type], 11), colors.Normalize(vmin=0, vmax=1)

        salmap_for_type = salmap_ds[front_type].sel(time=init_time)
        salmap_for_type_pl = salmap_ds[front_type + '_pl'].sel(time=init_time)
        max_gradient, min_gradient = salmap_for_type.max(), salmap_for_type.min()
        salmap_for_type = (salmap_for_type - min_gradient) / (max_gradient - min_gradient)
        salmap_for_type_pl = (salmap_for_type_pl - min_gradient) / (max_gradient - min_gradient)

        fig, axs = plt.subplots(3, 2, subplot_kw={'projection': ccrs.PlateCarree(central_longitude=(extent[0] + extent[1]) / 2)})
        axarr = axs.flatten()
        for ax_ind, ax in enumerate(axarr):
            plot_background(extent, ax=ax, linewidth=0.5)
            if ax_ind == 0:
                probs_ds[front_type].plot.contourf(ax=ax, x='longitude', y='latitude', cmap=cmap_probs, norm=norm, levels=levels, transform=ccrs.PlateCarree(), alpha=0.8, add_colorbar=False)
            else:
                probs_ds[front_type].plot.contour(ax=ax, x='longitude', y='latitude', colors='black', linewidths=0.25, norm=norm, levels=levels, transform=ccrs.PlateCarree(), alpha=0.8)
                salmap_for_type_pl.isel(pressure_level=ax_ind-1).plot(ax=ax, x='longitude', y='latitude', cmap=args['cmap'], norm=norm, transform=ccrs.PlateCarree(), alpha=0.6, add_colorbar=False)

        axarr[0].set_title("a) Model predictions")
        axarr[1].set_title("b) Saliency map - surface")
        axarr[2].set_title("c) Saliency map - 1000 hPa")
        axarr[3].set_title("d) Saliency map - 950 hPa")
        axarr[4].set_title("e) Saliency map - 900 hPa")
        axarr[5].set_title("f) Saliency map - 850 hPa")

        # if a directory for the plots is not provided, save the plots to the folder containing saliency maps
        plot_outdir = args['plot_outdir'] if args['plot_outdir'] is not None else salmap_folder

        plt.tight_layout()
        plt.suptitle("%s predictions: %d-%02d-%02d-%02dz" % (settings.DEFAULT_FRONT_NAMES[front_type], init_time.year, init_time.month, init_time.day, init_time.hour), y=1.05)
        plt.savefig('%s/model_%d_salmap_%d%02d%02d%02d_%s_%s.png' % (plot_outdir, args['model_number'], init_time.year, init_time.month, init_time.day, init_time.hour, args['domain'], front_type), bbox_inches='tight', dpi=500)
        plt.close()