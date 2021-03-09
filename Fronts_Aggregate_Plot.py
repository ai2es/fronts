"""
Function used to create the map subplots.

Code written by: John Allen (allen4jt@cmich.edu)
Last updated: 3/9/2021 12:07 CST by Andrew Justin (andrewjustin@ou.edu)
"""

import matplotlib.pyplot as plt
import xarray as xr

plt.switch_backend('agg')
from glob import glob
import argparse
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_background(ax, extent):
    """
    Returns new background for the plot.

    Parameters
    ----------
    ax: GeoAxesSubplot
        Initial subplot axes.
    extent: ndarray
        Numpy array containing the extent/boundaries of the plot in the format of [min lon, max lon, min lat, max lat].

    Returns
    -------
    ax: GeoAxesSubplot
        New plot background.

    """
    ax.gridlines()
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS,linewidth=0.5)
    ax.add_feature(cfeature.STATES,linewidth=0.5)
    ax.set_extent(extent)
    return ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create labeled data for the specified day")
    parser.add_argument('--year', type=int, required=True, help="year for the data to be read in")
    parser.add_argument('--month', type=int, required=False, help="month for the data to be read in")
    parser.add_argument('--day', type=int, required=False, help="day for the data to be read in")
    parser.add_argument('--image_outdir', type=str, required=False, help="output directory for image files")
    parser.add_argument('--netcdf_indir', type=str, required=False, help="input directory for netcdf files")
    args = parser.parse_args()

    base_dir = args.netcdf_indir

    if args.year == None:
        print('Failed to identify year')
    elif args.month == None:
        netcdf_intime = '%04d' % args.year
        in_file_name = 'FrontalCounts_%s*.nc' % netcdf_intime
        all_files = sorted(glob(os.path.join(base_dir, in_file_name)))
        plot_name = '%04d' % args.year
    elif args.day == None:
        netcdf_intime = str('%04d%02d' % (args.year, args.month))
        in_file_name = 'FrontalCounts_%s*.nc' % netcdf_intime
        all_files = sorted(glob(os.path.join(base_dir, in_file_name)))
        plot_name = '%04d-%02d' % (args.year, args.month)
    else:
        netcdf_intime = str('%04d%02d%02d' % (args.year, args.month, args.day))
        in_file_name = 'FrontalCounts_%s.nc' % netcdf_intime
        all_files = sorted(glob(os.path.join(base_dir, in_file_name)))
        plot_name = '%04d-%02d-%02d' % (args.year, args.month, args.day)

    ds = xr.open_mfdataset("%s/%s" % (args.netcdf_indir, in_file_name), combine='nested', concat_dim='Date')

    total_frequency = ds.sum(dim='Date')

    crs = ccrs.LambertConformal(central_longitude=250)
    fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(20, 14),
                          subplot_kw={'projection': crs})
    extent = [130, 370, 0, 80]
    axlist = axarr.flatten()
    for ax in axlist:
        plot_background(ax, extent)

    total_frequency.Frequency.sel(Type='COLD_FRONT').plot(ax=axlist[0] ,x='Longitude', y='Latitude', transform=ccrs.PlateCarree())
    total_frequency.Frequency.sel(Type='WARM_FRONT').plot(ax=axlist[1], x='Longitude', y='Latitude', transform=ccrs.PlateCarree())
    total_frequency.Frequency.sel(Type='OCCLUDED_FRONT').plot(ax=axlist[2], x='Longitude', y='Latitude', transform=ccrs.PlateCarree())
    total_frequency.Frequency.sel(Type='STATIONARY_FRONT').plot(ax=axlist[3], x='Longitude', y='Latitude', transform=ccrs.PlateCarree())

    plt.savefig(os.path.join(args.image_outdir,'%s_frequency_plot.png' % plot_name), bbox_inches='tight', dpi=300)
    print("Plot saved. File path: %s/%s_frequency_plot.png" % (args.image_outdir, plot_name))
