"""
Script that processes raw GOES data into smaller netCDF files.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.8.25
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
    parser.add_argument('--multiprocessing', action='store_true', help='Interpolate the satellite data with one band per CPU thread.')
    args = vars(parser.parse_args())
    
    init_time = pd.date_range(args['init_time'], args['init_time'])[0]
    year, month, day, hour = init_time.year, init_time.month, init_time.day, init_time.hour
    
    lon_array = np.arange(-196, -13, 0.25)
    lon_array_sat1 = np.arange(-141, -13, 0.25)
    lon_array_sat2 = np.arange(-196, -76, 0.25)
    lat_array = np.arange(20, 68, 0.25)
    
    if year <= 2017:
        sat1, sat2 = 'goes13', 'goes15'
    else:
        sat1, sat2 = 'goes16', 'goes17'
    
    print(f"[{dt.datetime.utcnow()}]", "Opening datasets")
    ds_sat1 = xr.open_dataset('%s/%d%02d/%s_%d%02d%02d%02d_full-disk.nc' % (args['satellite_indir'], year, month, sat1, year, month, day, hour), engine='netcdf4')
    ds_sat2 = xr.open_dataset('%s/%d%02d/%s_%d%02d%02d%02d_full-disk.nc' % (args['satellite_indir'], year, month, sat2, year, month, day, hour), engine='netcdf4')
    
    if sat1 == 'goes16':
        
        print(f"[{dt.datetime.utcnow()}]", f"Converting {sat1} coordinates")
        lat_sat1, lon_sat1 = calculate_lat_lon_from_dataset(ds_sat1)
        lat_sat1, lon_sat1 = np.nan_to_num(lat_sat1), np.nan_to_num(lon_sat1)
        
        print(f"[{dt.datetime.utcnow()}]", f"Converting {sat2} coordinates")
        lat_sat2, lon_sat2 = calculate_lat_lon_from_dataset(ds_sat2)
        lat_sat2, lon_sat2 = np.nan_to_num(lat_sat2), np.nan_to_num(lon_sat2)
    
    else:
        
        ds_sat1 = ds_sat1.isel(time=0)
        ds_sat2 = ds_sat2.isel(time=0)
        
        lat_sat1, lon_sat1 = ds_sat1['lat'].values, ds_sat1['lon'].values
        lat_sat2, lon_sat2 = ds_sat2['lat'].values, ds_sat2['lon'].values
    
    new_lat_sat1, new_lon_sat1 = np.meshgrid(lat_array, lon_array_sat1)
    new_lat_sat2, new_lon_sat2 = np.meshgrid(lat_array, lon_array_sat2)
    
    band_nums = args['band_nums'] if args['band_nums'] is not None else np.arange(1, 17)
    
    # xarray dataset that will contain the merged satellite data
    ds_merged = xr.Dataset(coords={'longitude': lon_array + 360, 'latitude': lat_array})

    for band_num in band_nums:
        
        if sat1 == 'goes13':
            band_str = 'ch%d' % int(band_num)
        else:
            band_str = 'CMI_C%02d' % int(band_num)
        
        variable_sat1 = np.nan_to_num(ds_sat1[band_str].values)
        variable_sat2 = np.nan_to_num(ds_sat2[band_str].values)

        print(f"[{dt.datetime.utcnow()}]", "Interpolating %s" % band_str)
        
        if sat1 == 'goes13':
            variable_sat1 = scipy.interpolate.RegularGridInterpolator((lat_sat1, lon_sat1), variable_sat1, method='nearest')((new_lat_sat1, new_lon_sat1))
            variable_sat2 = scipy.interpolate.RegularGridInterpolator((lat_sat2, lon_sat2), variable_sat2, method='nearest')((new_lat_sat2, new_lon_sat2))
        else:
            variable_sat1 = scipy.interpolate.griddata((lat_sat1.ravel(), lon_sat1.ravel()), variable_sat1.ravel(), (new_lat_sat1, new_lon_sat1), method='nearest')
            variable_sat2 = scipy.interpolate.griddata((lat_sat2.ravel(), lon_sat2.ravel()), variable_sat2.ravel(), (new_lat_sat2, new_lon_sat2), method='nearest')
        
        # blend overlapping portions of images together
        overlap_sat1 = variable_sat1[:260]
        overlap_sat2 = variable_sat2[-260:]
        overlap_mask = np.linspace(0, 1, 260)[:, np.newaxis]
        overlap_blend = (overlap_sat1 * overlap_mask) + (overlap_sat2 * (1 - overlap_mask))
        
        merged_data = np.vstack([variable_sat2[:-260, :], overlap_blend, variable_sat1[260:]])

        ds_merged['band_%d' % band_num] = (('longitude', 'latitude'), merged_data.astype(np.float32))
        ds_merged['band_%d' % band_num].attrs = ds_sat1[band_str].attrs
    
    converted_ds_filepath = '%s/%d%02d/satellite_%d%02d%02d%02d_full.nc' % (args['netcdf_outdir'], year, month, year, month, day, hour)
    converted_ds_exists = os.path.isfile(converted_ds_filepath)
    
    os.makedirs('%s/%d%02d' % (args['netcdf_outdir'], year, month), exist_ok=True)  # directory check
    
    ds_merged = ds_merged.expand_dims({'time': np.atleast_1d(init_time).astype('datetime64[ns]')})
    ds_merged = ds_merged.reindex(latitude=ds_merged['latitude'].values[::-1])  # reverse latitude values so they are ordered north-south
    ds_merged.to_netcdf(converted_ds_filepath, engine='netcdf4', mode='w')  # save dataset