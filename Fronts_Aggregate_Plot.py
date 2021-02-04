# Daily cartopy maps
# Function used to create the map subplots

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

plt.switch_backend('agg')
import glob
import argparse
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create labeled data for the specified day")
    parser.add_argument('--year', type=int, required=True, help="year for the data to be read in")
    parser.add_argument('--month', type=int, required=False, help="month for the data to be read in")
    parser.add_argument('--day', type=int, required=False, help="day for the data to be read in")
    #parser.add_argument('--image_outdir', type=str, required=False, help="output directory for image files")
    parser.add_argument('--netcdf_indir', type=str, required=False, help="input directory for netcdf files")
    args = parser.parse_args()
    
    base_dir = args.netcdf_indir
    
    if args.year == None:
        print('Failed to identify year')
    elif args.month == None:
        netcdf_intime = str('%04d' % (args.year))
        in_file_name = 'FrontalCounts_%s.nc' % netcdf_intime
        all_files = sorted(glob(os.path.join(base_dir, in_file_name)))
    elif args.day == None:
        netcdf_intime = str('%04d%02d' % (args.year, args.month))
        in_file_name = 'FrontalCounts_%s.nc' % netcdf_intime
        all_files = sorted(glob(os.path.join(base_dir, in_file_name)))
    else: 
        netcdf_intime = str('%04d%02d%02d' % (args.year, args.month, args.day))
        in_file_name = 'FrontalCounts_%s.nc' % netcdf_intime
        all_files = sorted(glob(os.path.join(base_dir, in_file_name)))
    
    # you have to make sure it's only the data you want to concatenate
    # if you have an "intruder" (like a file that is not hte same as others)
    # it will fail
    ds = xr.concat([xr.open_dataset(fname) for fname in all_files],
                   dim='Date')

    total_frequency = ds.sum(dim='Date')

    def plot_background(ax):
        ax.gridlines()
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
        ax.add_feature(cfeature.BORDERS,linewidth=0.5)
        ax.add_feature(cfeature.STATESm,linewidth=0.5)
        ax.set_extent([130, 370, 0, 80])
        return ax

    crs = ccrs.LambertConformal(ccrs.LambertConformal(central_longitude=250))

    fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(20, 14), constrained_layout=True,
                          subplot_kw={'projection': crs})
    axlist = axarr.flatten()
    for ax in axlist:
        plot_background(ax)

    total_frequency.Frequency.sel(Type='COLD_FRONT').isel(Date=0).plot(ax=axlist[0] ,x='Longitude', y='Latitude', transform=ccrs.PlateCarree())
    total_frequency.Frequency.sel(Type='WARM_FRONT').isel(Date=0).plot(ax=axlist[1], x='Longitude', y='Latitude', transform=ccrs.PlateCarree())
    total_frequency.Frequency.sel(Type='OCCLUDED_FRONT_DISS').isel(Date=0).plot(ax=axlist[2], x='Longitude', y='Latitude', transform=ccrs.PlateCarree())
    total_frequency.Frequency.sel(Type='STATIONARY_FRONT').isel(Date=0).plot(ax=axlist[4], x='Longitude', y='Latitude', transform=ccrs.PlateCarree())
    plt.savefig(os.path.join(args.image_outdir,'%d_frequencyplot.png' % (args.year)), bbox_inches='tight',dpi=300)
    print("Plot Created")
