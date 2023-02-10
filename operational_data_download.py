"""
Functions in this script create netcdf files containing ERA5, GDAS, or frontal object data.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 12/24/2022 5:58 PM CT

TODO: modify code so GDAS and GFS grib files have the correct units (units are different prior to 2022)
"""

import argparse
import time
import urllib.error
import requests
from bs4 import BeautifulSoup
import xarray as xr
import glob
import numpy as np
import wget
import os
import variables
import sys


def bar_progress(current, total):
  progress_message = "Downloading: %d%% [%.1f / %.1f] MB" % (current / total * 100, current / 1e6, total / 1e6)
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--netcdf_outdir', type=str, required=True, help="output directory for netcdf files")
    parser.add_argument('--init_time', type=int, required=True, nargs=4, help="Initialization time: year, month, day, hour")
    parser.add_argument('--forecast_hours', type=int, required=True, nargs='+', help='forecast hours to download')
    parser.add_argument('--model', type=str, required=True, help="'GDAS' or 'GFS' (case-insensitive)")

    args = parser.parse_args()
    provided_arguments = vars(args)
    
    model = args.model.lower()
    year, month, day, hour = args.init_time[0], args.init_time[1], args.init_time[2], args.init_time[3]
    forecast_hours = args.forecast_hours
    netcdf_outdir = args.netcdf_outdir
    
    ################################################### Download data ##################################################
    files = [f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/{args.model}.{year}%02d%02d/%02d/atmos/{args.model}.t%02dz.pgrb2.0p25.f%03d' % (month, day, hour, hour, forecast_hour)
             for forecast_hour in forecast_hours]

    local_grib_filenames = [f'{args.model}_{year}%02d%02d%02d_f%03d.grib' % (month, day, hour, forecast_hour)
                            for forecast_hour in forecast_hours]

    for file, local_filename in zip(files, local_grib_filenames):

        ### If the directory does not exist, check to see if the file link is valid. If the file link is NOT valid, then the directory will not be created since it will be empty. ###
        if not os.path.isdir(netcdf_outdir):
            if requests.head(file).status_code == requests.codes.ok or requests.head(file.replace('/atmos', '')).status_code == requests.codes.ok:
                os.mkdir(netcdf_outdir)

        full_file_path = f'{netcdf_outdir}/{local_filename}'

        if not os.path.isfile(full_file_path) and os.path.isdir(netcdf_outdir):
            try:
                wget.download(file, out=full_file_path)
            except urllib.error.HTTPError:
                try:
                    wget.download(file.replace('/atmos', ''), out=full_file_path, bar=bar_progress)
                except urllib.error.HTTPError:
                    print(f"Error downloading {file}")
                    pass

        elif os.path.isfile(full_file_path):
            print(f"{full_file_path} already exists, skipping file....")
    ####################################################################################################################

    print(stop)

    ################################################ Convert to netCDF #################################################
    keys_to_extract = ['gh', 'mslet', 'r', 'sp', 't', 'u', 'v']

    pressure_level_file_indices = [0, 2, 4, 5, 6]
    surface_data_file_indices = [2, 4, 5, 6]
    raw_pressure_data_file_index = 3
    mslp_data_file_index = 1

    # all lon/lat values in degrees
    start_lon, end_lon = 130, 370  # western boundary, eastern boundary
    start_lat, end_lat = 80, 0  # northern boundary, southern boundary
    unified_longitude_indices = np.append(np.arange(start_lon / 0.25, (end_lon - 10) / 0.25), np.arange(0, (end_lon - 360) / 0.25 + 1)).astype(int)
    unified_latitude_indices = np.arange((90 - start_lat) / 0.25, (90 - end_lat) / 0.25 + 1).astype(int)
    lon_coords_360 = np.arange(start_lon, end_lon + 0.25, 0.25)

    domain_indices_isel = {'longitude': unified_longitude_indices,
                           'latitude': unified_latitude_indices}

    isobaricInhPa_isel = [0, 1, 2, 3, 4, 5, 6]  # Dictionary to unpack for selecting indices in the datasets

    chunk_sizes = {'latitude': 321, 'longitude': 961}
    dataset_dimensions = ('forecast_hour', 'pressure_level', 'latitude', 'longitude')

    grib_files = [f'%s/{model}_%d%02d%02d%02d_f%03d.grib' % (netcdf_outdir, year, month, day, hour, forecast_hour)
                  for forecast_hour in forecast_hours]

    individual_variable_filename_format = f'%s/{model}_*_%d%02d%02d%02d_*.grib' % (netcdf_outdir, year, month, day, hour)

    ### Split grib files into one file per variable ###
    for key in keys_to_extract:
        output_file = f'%s{model}.%s.t%02dz.pgrb2.0p25' % (grib_indir, year, month, day, key, hour)
        if (os.path.isfile(output_file) and overwrite_grib) or not os.path.isfile(output_file):
            os.system(f'grib_copy -w shortName={key} {" ".join(grib_files)} {output_file}')

    if delete_original_grib:
        [os.remove(file) for file in grib_files]

    time.sleep(5)  # Pause the code for 5 seconds to ensure that all contents of the individual files are preserved

    # grib files by variable
    grib_files = sorted(glob.glob(individual_variable_filename_format))

    pressure_level_files = [grib_files[index] for index in pressure_level_file_indices]
    surface_data_files = [grib_files[index] for index in surface_data_file_indices]

    raw_pressure_data_file = grib_files[raw_pressure_data_file_index]
    if 'mslp_data_file_index' in locals():
        mslp_data_file = grib_files[mslp_data_file_index]
        mslp_data = xr.open_dataset(mslp_data_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}}, chunks={'latitude': 721, 'longitude': 1440}).isel(**domain_indices_isel).drop_vars(['step'])

    pressure_levels = [1000, 975, 950, 925, 900, 850, 700]

    # Open the datasets
    pressure_level_data = xr.open_mfdataset(pressure_level_files, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}}, chunks=chunk_sizes, combine='nested').isel(isobaricInhPa=isobaricInhPa_isel, **domain_indices_isel).drop_vars(['step'])
    surface_data = xr.open_mfdataset(surface_data_files, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'sigma'}}, chunks=chunk_sizes).isel(**domain_indices_isel).drop_vars(['step'])
    raw_pressure_data = xr.open_dataset(raw_pressure_data_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'stepType': 'instant'}}, chunks=chunk_sizes).isel(**domain_indices_isel).drop_vars(['step'])

    # Calculate the forecast hours using the surface_data dataset
    try:
        run_time = surface_data['time'].values.astype('int64')
    except KeyError:
        run_time = surface_data['run_time'].values.astype('int64')

    valid_time = surface_data['valid_time'].values.astype('int64')
    forecast_hours = np.array((valid_time - int(run_time)) / 3.6e12, dtype='int32')

    try:
        num_forecast_hours = len(forecast_hours)
    except TypeError:
        num_forecast_hours = 1
        forecast_hours = [forecast_hours, ]

    # Reformat the longitude coordinates to the 360 degree system
    if model in ['gdas', 'gfs']:
        pressure_level_data['longitude'] = lon_coords_360
        surface_data['longitude'] = lon_coords_360
        raw_pressure_data['longitude'] = lon_coords_360
        mslp_data['longitude'] = lon_coords_360

        mslp = mslp_data['mslet'].values  # mean sea level pressure (eta model reduction)
        mslp_z = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
        mslp_z[:, 0, :, :] = mslp / 100  # convert to hectopascals

    P = np.empty(shape=(num_forecast_hours, len(pressure_levels), chunk_sizes['latitude'], chunk_sizes['longitude']))  # create 3D array of pressure levels to match the shape of variable arrays
    for pressure_level_index, pressure_level in enumerate(pressure_levels):
        P[:, pressure_level_index, :, :] = pressure_level * 100

    print("Generating pressure level variables")

    T_pl = pressure_level_data['t'].values
    RH_pl = pressure_level_data['r'].values / 100
    vap_pres_pl = RH_pl * variables.vapor_pressure(T_pl)
    Td_pl = variables.dewpoint_from_vapor_pressure(vap_pres_pl)
    Tv_pl = variables.virtual_temperature_from_dewpoint(T_pl, Td_pl, P)
    Tw_pl = variables.wet_bulb_temperature(T_pl, Td_pl)
    r_pl = variables.mixing_ratio_from_dewpoint(Td_pl, P) * 1000  # Convert to g/kg
    q_pl = variables.specific_humidity_from_dewpoint(Td_pl, P) * 1000  # Convert to g/kg
    theta_pl = variables.potential_temperature(T_pl, P)
    theta_e_pl = variables.equivalent_potential_temperature(T_pl, Td_pl, P)
    theta_v_pl = variables.virtual_potential_temperature(T_pl, Td_pl, P)
    theta_w_pl = variables.wet_bulb_potential_temperature(T_pl, Td_pl, P)
    z = pressure_level_data['gh'].values / 10  # Convert to dam
    u_pl = pressure_level_data['u'].values
    v_pl = pressure_level_data['v'].values

    if 'mslp_data_file_index' in locals():
        mslp_z[:, 1:, :, :] = z

    # Create arrays of coordinates for the surface data
    surface_data_latitudes = pressure_level_data['latitude'].values

    print("Generating surface variables")

    sp = raw_pressure_data['sp'].values
    T_sigma = surface_data['t'].values
    RH_sigma = surface_data['r'].values / 100
    vap_pres_sigma = RH_sigma * variables.vapor_pressure(T_sigma)
    Td_sigma = variables.dewpoint_from_vapor_pressure(vap_pres_sigma)
    Tv_sigma = variables.virtual_temperature_from_dewpoint(T_sigma, Td_sigma, sp)
    Tw_sigma = variables.wet_bulb_temperature(T_sigma, Td_sigma)
    r_sigma = variables.mixing_ratio_from_dewpoint(Td_sigma, sp) * 1000  # Convert to g/kg
    q_sigma = variables.specific_humidity_from_dewpoint(Td_sigma, sp) * 1000  # Convert to g/kg
    theta_sigma = variables.potential_temperature(T_sigma, sp)
    theta_e_sigma = variables.equivalent_potential_temperature(T_sigma, Td_sigma, sp)
    theta_v_sigma = variables.virtual_potential_temperature(T_sigma, Td_sigma, sp)
    theta_w_sigma = variables.wet_bulb_potential_temperature(T_sigma, Td_sigma, sp)
    u_sigma = surface_data['u'].values
    v_sigma = surface_data['v'].values

    T = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    Td = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    Tv = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    Tw = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    theta = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    theta_e = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    theta_v = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    theta_w = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    RH = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    r = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    q = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    u = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    v = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    sp_z = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))

    T[:, 0, :, :] = T_sigma
    T[:, 1:, :, :] = T_pl
    Td[:, 0, :, :] = Td_sigma
    Td[:, 1:, :, :] = Td_pl
    Tv[:, 0, :, :] = Tv_sigma
    Tv[:, 1:, :, :] = Tv_pl
    Tw[:, 0, :, :] = Tw_sigma
    Tw[:, 1:, :, :] = Tw_pl
    theta[:, 0, :, :] = theta_sigma
    theta[:, 1:, :, :] = theta_pl
    theta_e[:, 0, :, :] = theta_e_sigma
    theta_e[:, 1:, :, :] = theta_e_pl
    theta_v[:, 0, :, :] = theta_v_sigma
    theta_v[:, 1:, :, :] = theta_v_pl
    theta_w[:, 0, :, :] = theta_w_sigma
    theta_w[:, 1:, :, :] = theta_w_pl
    RH[:, 0, :, :] = RH_sigma
    RH[:, 1:, :, :] = RH_pl
    r[:, 0, :, :] = r_sigma
    r[:, 1:, :, :] = r_pl
    q[:, 0, :, :] = q_sigma
    q[:, 1:, :, :] = q_pl
    u[:, 0, :, :] = u_sigma
    u[:, 1:, :, :] = u_pl
    v[:, 0, :, :] = v_sigma
    v[:, 1:, :, :] = v_pl
    sp_z[:, 0, :, :] = sp / 100
    sp_z[:, 1:, :, :] = z

    pressure_levels = ['surface', '1000', '975', '950', '925', '900', '850', '700']

    print("Building final dataset")

    full_dataset_coordinates = dict(forecast_hour=forecast_hours, pressure_level=pressure_levels)

    full_dataset_variables = dict(T=(dataset_dimensions, T),
                                  Td=(dataset_dimensions, Td),
                                  Tv=(dataset_dimensions, Tv),
                                  Tw=(dataset_dimensions, Tw),
                                  theta=(dataset_dimensions, theta),
                                  theta_e=(dataset_dimensions, theta_e),
                                  theta_v=(dataset_dimensions, theta_v),
                                  theta_w=(dataset_dimensions, theta_w),
                                  RH=(dataset_dimensions, RH),
                                  r=(dataset_dimensions, r),
                                  q=(dataset_dimensions, q),
                                  u=(dataset_dimensions, u),
                                  v=(dataset_dimensions, v),
                                  sp_z=(dataset_dimensions, sp_z))

    if 'mslp_data_file_index' in locals():
        full_dataset_variables['mslp_z'] = (('forecast_hour', 'pressure_level', 'latitude', 'longitude'), mslp_z)

    full_dataset_coordinates['latitude'] = surface_data_latitudes
    full_dataset_coordinates['longitude'] = lon_coords_360

    full_grib_dataset = xr.Dataset(data_vars=full_dataset_variables,
                                   coords=full_dataset_coordinates).astype('float32')

    full_grib_dataset = full_grib_dataset.expand_dims({'time': np.atleast_1d(pressure_level_data['time'].values)})

    for fcst_hr_index, forecast_hour in enumerate(forecast_hours):
        full_grib_dataset.isel(forecast_hour=np.atleast_1d(fcst_hr_index)).to_netcdf(path=f'%s/{model}_%d%02d%02d%02d_f%03d.nc' % (netcdf_outdir, year, month, day, year, month, day, hour, forecast_hour), mode='w', engine='netcdf4')
