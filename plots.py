"""
Code for generating various plots.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.7.24

TODO: Add functions for plotting ERA5, GDAS, and GFS data.
"""

import argparse
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import settings
from utils.data_utils import reformat_fronts, expand_fronts
from utils.plotting_utils import plot_background
import numpy as np
import cartopy.crs as ccrs


def plot_fronts(netcdf_indir, plot_outdir, timestep, front_types, domain, extent=(-180, 180, -90, 90)):

    year, month, day, hour = timestep

    fronts_ds = xr.open_dataset('%s/%d%02d/FrontObjects_%d%02d%02d%02d_%s.nc' % (netcdf_indir, year, month, year, month, day, hour, domain))

    if front_types is not None:
        fronts_ds = reformat_fronts(fronts_ds, front_types)
    labels = fronts_ds.attrs['labels']

    fronts_ds = expand_fronts(fronts_ds, iterations=1)
    fronts_ds = xr.where(fronts_ds == 0, np.nan, fronts_ds)

    front_colors_by_type = [settings.DEFAULT_FRONT_COLORS[label] for label in labels]
    front_names_by_type = [settings.DEFAULT_FRONT_NAMES[label] for label in labels]
    cmap_front = mpl.colors.ListedColormap(front_colors_by_type, name='from_list', N=len(front_colors_by_type))
    norm_front = mpl.colors.Normalize(vmin=1, vmax=len(front_colors_by_type) + 1)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=np.mean(extent[:2]))})
    plot_background(extent, ax=ax, linewidth=0.25)

    cbar_front = plt.colorbar(mpl.cm.ScalarMappable(norm=norm_front, cmap=cmap_front), ax=ax, alpha=0.75, shrink=0.8, pad=0.02)
    cbar_front.set_ticks(np.arange(1, len(front_colors_by_type) + 1) + 0.5)
    cbar_front.set_ticklabels(front_names_by_type)
    cbar_front.set_label('Front Type')

    fronts_ds['identifier'].plot(ax=ax, x='longitude', y='latitude', cmap=cmap_front, norm=norm_front, transform=ccrs.PlateCarree(),
                                 add_colorbar=False)
    ax.gridlines(alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"%s/fronts_%d%02d%02d%02d_{domain}.png" % (plot_outdir, year, month, day, hour), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestep', type=int, nargs=4, help='Year, month, day, and hour of the data.')
    parser.add_argument('--netcdf_indir', type=str, help='Directory for the netcdf files.')
    parser.add_argument('--plot_outdir', type=str, help='Directory for the plots.')
    parser.add_argument('--front_types', type=str, nargs='+', help='Directory for the netcdf files.')
    parser.add_argument('--domain', type=str, default='full', help="Domain for which the fronts will be plotted.")
    parser.add_argument('--extent', type=float, nargs=4, default=[-180., 180., -90., 90.], help="Extent of the plot [min lon, max lon, min lat, max lat]")
    args = vars(parser.parse_args())

    plot_fronts(args['netcdf_indir'], args['plot_outdir'], args['timestep'], args['front_types'], args['domain'], args['extent'])
