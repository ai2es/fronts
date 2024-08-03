"""
Script that processes raw GOES-16/17 data into smaller netCDF files.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.8.3
"""
import argparse
import datetime as dt
import numpy as np
import os
import pandas as pd
import scipy
from utils.satellite import calculate_lat_lon_from_dataset
import xarray as xr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--satellite_indir', type=str, required=True, help='Parent input directory for the raw satellite data.')
    parser.add_argument('--init_time', type=str, required=True, help='Initialization time of the model. Format: YYYY-MM-DDTHH.')
    parser.add_argument('--netcdf_outdir', type=str, required=True, help='Output directory for the processed satellite netCDF files.')
    parser.add_argument('--band_nums', type=int, nargs='+',
        help='Band numbers to include in the final datasets. If this argument is not passed, all band numbers (1-16) will be included.')
    args = vars(parser.parse_args())
    
    init_time = pd.date_range(args['init_time'], args['init_time'])[0]
    year, month, day, hour = init_time.year, init_time.month, init_time.day, init_time.hour
    
    print(f"[{dt.datetime.utcnow()}]", "Opening datasets")
    ds_goes16 = xr.open_dataset('%s/%d%02d/goes16_%d%02d%02d%02d_full-disk.nc' % (args['satellite_indir'], year, month, year, month, day, hour), engine='netcdf4')
    ds_goes17 = xr.open_dataset('%s/%d%02d/goes17_%d%02d%02d%02d_full-disk.nc' % (args['satellite_indir'], year, month, year, month, day, hour), engine='netcdf4')
    
    lon_array = np.arange(-196, -13, 0.25)
    lon_array_g16 = np.arange(-141, -13, 0.25)
    lon_array_g17 = np.arange(-196, -76, 0.25)
    lat_array = np.arange(20, 68, 0.25)
    
    print(f"[{dt.datetime.utcnow()}]", "Converting goes16 coordinates")
    lat_g16, lon_g16 = calculate_lat_lon_from_dataset(ds_goes16)
    extent_g16 = [np.nanmin(lon_g16), np.nanmax(lon_g16), np.nanmin(lat_g16), np.nanmax(lat_g16)]
    lat_g16, lon_g16 = np.nan_to_num(lat_g16), np.nan_to_num(lon_g16)
    
    print(f"[{dt.datetime.utcnow()}]", "Converting goes17 coordinates")
    lat_g17, lon_g17 = calculate_lat_lon_from_dataset(ds_goes17)
    extent_g17 = [np.nanmin(lon_g17), np.nanmax(lon_g17), np.nanmin(lat_g17), np.nanmax(lat_g17)]
    lat_g17, lon_g17 = np.nan_to_num(lat_g17), np.nan_to_num(lon_g17)
    
    new_lat_g16, new_lon_g16 = np.meshgrid(lat_array, lon_array_g16)
    new_lat_g17, new_lon_g17 = np.meshgrid(lat_array, lon_array_g17)
    
    band_nums = args['band_nums'] if args['band_nums'] is not None else np.arange(1, 17)

    # xarray dataset that will contain the merged satellite data
    ds_merged = xr.Dataset(coords={'longitude': lon_array, 'latitude': lat_array})

    for band_num in band_nums:
        
        band_str = 'CMI_C%02d' % band_num
        
        variable_g16 = ds_goes16[band_str].values
        variable_g17 = ds_goes17[band_str].values
    
        print(f"[{dt.datetime.utcnow()}]", "Interpolating %s" % band_str)
        variable_g16 = scipy.interpolate.griddata((lat_g16.ravel(), lon_g16.ravel()), variable_g16.ravel(), (new_lat_g16, new_lon_g16), method='nearest')
        variable_g17 = scipy.interpolate.griddata((lat_g17.ravel(), lon_g17.ravel()), variable_g17.ravel(), (new_lat_g17, new_lon_g17), method='nearest')
    
        # blend overlapping portions of images together
        overlap_g16 = variable_g16[:260]
        overlap_g17 = variable_g17[-260:]
        overlap_mask = np.linspace(0, 1, 260)[:, np.newaxis]
        overlap_blend = (overlap_g16 * overlap_mask) + (overlap_g17 * (1 - overlap_mask))
    
        merged_data = np.vstack([variable_g16[:-260, :], overlap_blend, variable_g17[260:]])
        
        ds_merged[band_str] = (('longitude', 'latitude'), merged_data.astype(np.float32))
        ds_merged[band_str].attrs = ds_goes16[band_str].attrs
    
    converted_ds_filepath = '%s/%d%02d/satellite_%d%02d%02d%02d_full.nc' % (args['netcdf_outdir'], year, month, year, month, day, hour)
    converted_ds_exists = os.path.isfile(converted_ds_filepath)
    
    # directory check
    os.makedirs('%s/%d%02d' % (args['netcdf_outdir'], year, month), exist_ok=True)
    
    # save dataset
    ds_merged.to_netcdf(converted_ds_filepath, engine='netcdf4', mode='w')