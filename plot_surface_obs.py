"""
Function that plots interpolated surface observations.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 11/15/2021 7:20 PM CST by Andrew Justin
"""

import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import argparse
from errors import check_arguments
import matplotlib.pyplot as plt
import Fronts_Aggregate_Plot as fplot
import xarray as xr
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=False)
    parser.add_argument('--month', type=int, required=False)
    parser.add_argument('--day', type=int, required=False)
    parser.add_argument('--hour', type=int, required=False)
    parser.add_argument('--pickle_indir', type=str, required=False, help='Path where the pickle files are saved.')
    parser.add_argument('--image_outdir', type=str, required=False, help='Path where the images will be saved.')
    parser.add_argument('--variable_asos', type=str, required=False,
                        help='Variable to plot. Provide the ASOS code for the variable with this argument.')

    args = parser.parse_args()
    provided_arguments = vars(args)
    required_arguments = ['year','month','day','hour','image_outdir','pickle_indir']

    print("Checking arguments....",end='')
    check_arguments(provided_arguments, required_arguments)

    surface_obs = pd.read_pickle('%s/%d/%02d/%02d/SurfaceObs_%d%02d%02d%02d_%s.pkl' % (args.pickle_indir, args.year,
                    args.month, args.day, args.year, args.month, args.day, args.hour, args.variable_asos))
    fronts = pd.read_pickle('%s/%d/%02d/%02d/FrontObjects_ALL_%d%02d%02d%02d_full.pkl' % (args.pickle_indir, args.year,
                    args.month, args.day, args.year, args.month, args.day, args.hour))

    fronts = xr.where(fronts == 0, float("NaN"), fronts)

    extent = np.array([237, 285, 25, 55])
    crs = ccrs.LambertConformal(central_longitude=250)

    fronts_norm = Normalize(vmin=1,vmax=5)
    fronts_cmap = ListedColormap(["blue","red",'green','purple'], name='from_list', N=None)

    variable_norm = Normalize(vmin=20, vmax=80)
    variable_cmap = 'terrain_r'

    fig, ax = plt.subplots(1, 1, figsize=(10,5), subplot_kw={'projection': crs})
    fplot.plot_background(ax, extent)
    surface_obs[args.variable_asos].plot(ax=ax, x='longitude', y='latitude', norm=variable_norm, cmap=variable_cmap,
                                         alpha=0.75, transform=ccrs.PlateCarree())
    fronts['identifier'].plot(ax=ax, x='longitude', y='latitude', norm=fronts_norm, cmap=fronts_cmap, transform=ccrs.PlateCarree(),
                              add_colorbar=False)

    cbar = plt.colorbar(ScalarMappable(norm=fronts_norm, cmap=fronts_cmap), ax=ax)
    cbar.set_ticks(np.arange(1,5,1)+0.5)
    cbar.set_ticklabels(['Cold','Warm','Stationary','Occluded'])
    cbar.set_label('Front type')

    plt.savefig('%s/SurfaceObs_%d%02d%02d%02d_%s_plot.png' % (args.image_outdir, args.year, args.month, args.day,
        args.hour, args.variable_asos), dpi=300)
    plt.close()
