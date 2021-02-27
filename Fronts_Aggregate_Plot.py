# Function used to create the map subplots

import matplotlib.pyplot as plt
import xarray as xr

plt.switch_backend('agg')
from glob import glob
import argparse
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
from datetime import timedelta

if __name__ == "__main__":
    start_time = time.time()
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
        print("(1/11) Collecting netCDF files for %04d...." % args.year, end=' ')
        netcdf_intime = '%04d' % args.year
        in_file_name = 'FrontalCounts_%s*.nc' % netcdf_intime
        all_files = sorted(glob(os.path.join(base_dir, in_file_name)))
        plot_name = '%04d' % args.year
    elif args.day == None:
        print("(1/11) Collecting netCDF files for %04d-%02d...." % (args.year, args.month), end=' ')
        netcdf_intime = str('%04d%02d' % (args.year, args.month))
        in_file_name = 'FrontalCounts_%s*.nc' % netcdf_intime
        all_files = sorted(glob(os.path.join(base_dir, in_file_name)))
        plot_name = '%04d-%02d' % (args.year, args.month)
    else:
        print("(1/11) Collecting netCDF files for %04d-%02d-%02d...." % (args.year, args.month, args.day), end=' ')
        netcdf_intime = str('%04d%02d%02d' % (args.year, args.month, args.day))
        in_file_name = 'FrontalCounts_%s.nc' % netcdf_intime
        all_files = sorted(glob(os.path.join(base_dir, in_file_name)))
        plot_name = '%04d-%02d-%02d' % (args.year, args.month, args.day)


    # you have to make sure it's only the data you want to concatenate
    # if you have an "intruder" (like a file that is not the same as others)
    # it will fail
    print("(2/11) Concatenating data files....", end=' ')
    ds = xr.open_mfdataset("%s/%s" % (args.netcdf_indir, in_file_name), combine='nested', concat_dim='Date')
    print("done")

    print("(3/11) Calculating maximum frequency....", end=' ')
    total_frequency = ds.sum(dim='Date')
    print("done")

    def plot_background(ax):
        ax.gridlines()
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
        ax.add_feature(cfeature.BORDERS,linewidth=0.5)
        ax.add_feature(cfeature.STATES,linewidth=0.5)
        #ax.set_extent([130, 370, 0, 80])
        return ax

    print("(4/11) Creating plot projections....", end=' ')
    crs = ccrs.LambertConformal(central_longitude=250)
    print("done")

    print("(5/11) Formatting plot file....", end=' ')
    fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(20, 14),
                          subplot_kw={'projection': crs})
    print("done")

    axlist = axarr.flatten()
    print("(6/11) Adding features and setting plot extents....", end=' ')
    for ax in axlist:
        plot_background(ax)
    print("done")

    print("(7/11) Plotting cold fronts....", end=' ')
    total_frequency.Frequency.sel(Type='COLD_FRONT').plot(ax=axlist[0] ,x='Longitude', y='Latitude', transform=ccrs.PlateCarree())
    print("done")

    print("(8/11) Plotting warm fronts....", end=' ')
    total_frequency.Frequency.sel(Type='WARM_FRONT').plot(ax=axlist[1], x='Longitude', y='Latitude', transform=ccrs.PlateCarree())
    print("done")

    print("(9/11) Plotting occluded fronts....", end=' ')
    total_frequency.Frequency.sel(Type='OCCLUDED_FRONT_DISS').plot(ax=axlist[2], x='Longitude', y='Latitude', transform=ccrs.PlateCarree())
    print("done")

    print("(10/11) Plotting stationary fronts....", end=' ')
    total_frequency.Frequency.sel(Type='STATIONARY_FRONT').plot(ax=axlist[3], x='Longitude', y='Latitude', transform=ccrs.PlateCarree())
    print("done")

    print("(11/11) Saving plot....", end=' ')
    plt.savefig(os.path.join(args.image_outdir,'%s_frequency_plot.png' % plot_name), bbox_inches='tight', dpi=300)
    print("done")

    print("Plot saved. File path: %s/%s_frequency_plot.png" % (args.image_outdir, plot_name))

    print("Time elapsed: %s" % timedelta(seconds=time.time()-start_time))
