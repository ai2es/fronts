"""
Convert GDAS and/or GFS grib files to netCDF files.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.3.27
"""

import argparse
import time
import xarray as xr
from utils import variables
import glob
import numpy as np
import os
import tensorflow as tf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grib_indir', type=str, required=True, help="Input directory for grib files.")
    parser.add_argument('--netcdf_outdir', type=str, required=True, help="Output directory for the netcdf files.")
    parser.add_argument('--model', required=True, type=str, help="NWP model for the grib files.")
    parser.add_argument('--init_time', type=str, required=True, help="Initialization time of the model. Format: YYYY-MM-DDTHH.")
    parser.add_argument('--pressure_levels', type=str, nargs="+", default=["surface", "1000", "950", "900", "850"], help="Pressure levels to extract from the grib files.")
    parser.add_argument('--gpu', action='store_true',
        help="Use a GPU to perform calculations of additional variables. This can provide speedups when generating very "
             "large amounts of data.")
    parser.add_argument('--ignore_warnings', action='store_true', help="Disable runtime warnings in variable calculations.")
    args = vars(parser.parse_args())

    if args["ignore_warnings"]:  # suppress divide by zero RuntimeWarnings
        import warnings
        warnings.filterwarnings('ignore')

    gpus = tf.config.list_physical_devices(device_type='GPU')
    if len(gpus) > 0 and args['gpu']:
        print("Using GPU for variable derivations")
        tf.config.set_visible_devices(devices=gpus[0], device_type='GPU')
        gpus = tf.config.get_visible_devices(device_type='GPU')
        tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
    else:
        print("Using CPUs for variable derivations")
        tf.config.set_visible_devices([], 'GPU')

    args['model'] = args['model'].lower()

    date = np.datetime64(args['init_time']).astype(object)
    year, month, day, hour = date.year, date.month, date.day, date.hour

    # filename format of the downloaded grib files
    grib_filename_format = f"%s/%d%02d/%s_%d%02d%02d%02d_f*.grib" % (args['grib_indir'], year, month, args['model'], year, month, day, hour)
    grib_files = list(glob.glob(grib_filename_format))

    # list of pressure levels (NOT including surface level)
    pressure_levels = [lvl for lvl in args["pressure_levels"] if lvl != "surface"]

    # boolean flags
    include_pressure_level_data = len(pressure_levels) > 0
    include_surface_data = True if "surface" in args["pressure_levels"] else False

    # keyword arguments used in xr.open_mfdataset
    open_ds_args = dict(engine="cfgrib", errors="ignore", combine="nested", concat_dim="valid_time")

    if include_pressure_level_data:

        print("Reading pressure level data", end='')
        start_time = time.time()
        # open pressure level data from grib files
        if args["model"] not in ["nam-12km", "namnest-conus"]:
            pressure_level_data = xr.open_mfdataset(grib_files, backend_kwargs={"filter_by_keys": {'typeOfLevel': 'isobaricInhPa'}}, **open_ds_args)
            pressure_level_data = pressure_level_data.sel(isobaricInhPa=pressure_levels)  # select pressure levels
            pressure_level_data = pressure_level_data[["t", "u", "v", "r", "gh"]]  # select variables

            # lat/lon coordinates, time variable
            latitude = pressure_level_data["latitude"].values
            longitude = pressure_level_data["longitude"].values
            timestep = pressure_level_data["time"].values

            T = pressure_level_data["t"].values  # temperature
            u = pressure_level_data["u"].values  # u-wind
            v = pressure_level_data["v"].values  # v-wind
            RH = pressure_level_data["r"].values / 100  # relative humidity
            sp_z = pressure_level_data["gh"].values / 10  # geopotential height (dam)

        else:  # need to open NAM grib files one variable at a time (typical cfgrib nonsense)
            T = xr.open_mfdataset(grib_files, backend_kwargs={"filter_by_keys": {'typeOfLevel': 'isobaricInhPa', 'cfVarName': 't'}}, **open_ds_args).sel(isobaricInhPa=pressure_levels)['t'].values  # temperature
            u = xr.open_mfdataset(grib_files, backend_kwargs={"filter_by_keys": {'typeOfLevel': 'isobaricInhPa', 'cfVarName': 'u'}}, **open_ds_args).sel(isobaricInhPa=pressure_levels)['u'].values  # u-wind
            v = xr.open_mfdataset(grib_files, backend_kwargs={"filter_by_keys": {'typeOfLevel': 'isobaricInhPa', 'cfVarName': 'v'}}, **open_ds_args).sel(isobaricInhPa=pressure_levels)['v'].values  # v-wind
            RH = xr.open_mfdataset(grib_files, backend_kwargs={"filter_by_keys": {'typeOfLevel': 'isobaricInhPa', 'cfVarName': 'r'}}, **open_ds_args).sel(isobaricInhPa=pressure_levels)['r'].values / 100  # relative humidity
            sp_z = xr.open_mfdataset(grib_files, backend_kwargs={"filter_by_keys": {'typeOfLevel': 'isobaricInhPa', 'cfVarName': 'gh'}}, **open_ds_args).sel(isobaricInhPa=pressure_levels)['gh'].values / 10  # geopotential height (dam)

            # lat/lon coordinates
            latitude = xr.open_mfdataset(grib_files, backend_kwargs={"filter_by_keys": {'typeOfLevel': 'isobaricInhPa'}}, **open_ds_args)["latitude"].values
            longitude = xr.open_mfdataset(grib_files, backend_kwargs={"filter_by_keys": {'typeOfLevel': 'isobaricInhPa'}}, **open_ds_args)["longitude"].values
            timestep = xr.open_mfdataset(grib_files, backend_kwargs={"filter_by_keys": {'typeOfLevel': 'isobaricInhPa'}}, **open_ds_args)["time"].values

        # pressure array for calculating additional variables. shape: (forecast hour, pressure level, latitude/y, longitude/x)
        P = np.zeros_like(T)
        for i, lvl in enumerate(pressure_levels):
            P[:, i, ...] = int(lvl) * 100  # convert pressure to Pa
        print(" (%.1f seconds)" % (time.time() - start_time))

    if include_surface_data:

        print("Reading surface data", end='')
        start_time = time.time()
        if args["model"] in ["gfs", "gdas"]:
            sp_data = xr.open_mfdataset(grib_files, backend_kwargs={"filter_by_keys": {'typeOfLevel': 'surface', 'stepType': 'instant'}}, **open_ds_args)["sp"].values[:, np.newaxis, ...]
            surface_data = xr.open_mfdataset(grib_files, backend_kwargs={"filter_by_keys": {'typeOfLevel': 'sigma', 'stepType': 'instant'}}, **open_ds_args)
        elif args["model"] == "hrrr":
            sp_data = xr.open_mfdataset(grib_files, backend_kwargs={"filter_by_keys": {'typeOfLevel': 'surface', 'stepType': 'instant'}}, **open_ds_args)["sp"].values[:, np.newaxis, ...]
            surface_data = xr.open_mfdataset(grib_files, backend_kwargs={"filter_by_keys": {'typeOfLevel': 'surface', 'stepType': 'instant'}}, **open_ds_args)
        else:  # NAM
            raise NotImplementedError("NAM surface data currently not supported.")
            # TODO: Figure out how to get NAM surface data implemented
            # T_da = xr.open_mfdataset(grib_files, backend_kwargs={"filter_by_keys": {'typeOfLevel': 'sigma', 'cfVarName': 't'}}, **open_ds_args)['t']  # temperature
            # u_da = xr.open_mfdataset(grib_files, backend_kwargs={"filter_by_keys": {'typeOfLevel': 'sigma', 'cfVarName': 'u'}}, **open_ds_args)['u']  # u-wind
            # v_da = xr.open_mfdataset(grib_files, backend_kwargs={"filter_by_keys": {'typeOfLevel': 'sigma', 'cfVarName': 'v'}}, **open_ds_args)['v']  # v-wind
            # RH_da = xr.open_mfdataset(grib_files, backend_kwargs={"filter_by_keys": {'typeOfLevel': 'sigma', 'cfVarName': 'r'}}, **open_ds_args)['r'] / 100  # relative humidity
            # sp_z_da = xr.open_mfdataset(grib_files, backend_kwargs={"filter_by_keys": {'typeOfLevel': 'sigma', 'cfVarName': 'sp'}}, **open_ds_args)['sp'] / 10  # geopotential height (dam)

        # lat/lon coordinates, time variable
        latitude = surface_data["latitude"].values
        longitude = surface_data["longitude"].values
        timestep = surface_data["time"].values

        if include_pressure_level_data:
            # combine surface and pressure level data
            T = np.concatenate([surface_data["t"].values[:, np.newaxis, ...], T], axis=1)
            u = np.concatenate([surface_data["u"].values[:, np.newaxis, ...], u], axis=1)
            v = np.concatenate([surface_data["v"].values[:, np.newaxis, ...], v], axis=1)
            RH = np.concatenate([surface_data["r"].values[:, np.newaxis, ...] / 100, RH], axis=1)
            P = np.concatenate([sp_data, P], axis=1)
            sp_z = np.concatenate([sp_data / 100, sp_z], axis=1)
        else:
            T = surface_data["t"].values[:, np.newaxis, ...]
            u = surface_data["u"].values[:, np.newaxis, ...]
            v = surface_data["v"].values[:, np.newaxis, ...]
            RH = surface_data["r"].values[:, np.newaxis, ...] / 100
            P = sp_data
            sp_z = sp_data / 100
        print(" (%.1f seconds)" % (time.time() - start_time))

    # if using a GPU, convert the numpy arrays to tensors
    print("Calculating additional variables", end='')
    start_time = time.time()
    if len(gpus) > 0 and args['gpu']:
        T = tf.convert_to_tensor(T)
        u = tf.convert_to_tensor(u)
        v = tf.convert_to_tensor(v)
        RH = tf.convert_to_tensor(RH)
        P = tf.convert_to_tensor(P)

    # calculate additional variables
    q = variables.specific_humidity_from_relative_humidity(RH, T, P)
    Td = variables.dewpoint_from_specific_humidity(P, T, q)
    Tv = variables.virtual_temperature_from_dewpoint(T, Td, P)
    r = variables.mixing_ratio_from_dewpoint(Td, P) * 1000  # convert back to g/kg
    theta = variables.potential_temperature(T, P)
    theta_e = variables.equivalent_potential_temperature(T, Td, P)
    theta_v = variables.virtual_potential_temperature(T, Td, P)
    print(" (%.1f seconds)" % (time.time() - start_time))

    # if a GPU was used to calculate additional variables, turn the tensors back to numpy arrays
    if len(gpus) > 0 and args['gpu']:
        T = T.numpy()
        Td = Td.numpy()
        Tv = Tv.numpy()
        q = q.numpy()
        r = r.numpy()
        u = u.numpy()
        v = v.numpy()
        RH = RH.numpy()
        theta = theta.numpy()
        theta_e = theta_e.numpy()
        theta_v = theta_v.numpy()

    forecast_hours = [int(filename.split('_f')[1][:3]) for filename in grib_files]  # can pull forecast hours straight from the grib filenames

    if args["model"] in ["gfs", "gdas"]:
        dataset_dimensions = ('forecast_hour', 'pressure_level', 'latitude', 'longitude')
    else:  # HRRR/NAM - both have non-uniform lat/lon grids
        dataset_dimensions = ('forecast_hour', 'pressure_level', 'y', 'x')
        latitude = (('y', 'x'), latitude)
        longitude = (('y', 'x'), longitude)

    print("Building final datasets", end='')
    start_time = time.time()
    full_dataset_coordinates = dict(forecast_hour=forecast_hours, pressure_level=args["pressure_levels"], latitude=latitude, longitude=longitude)

    full_dataset_variables = dict(T=(dataset_dimensions, T),
                                  Td=(dataset_dimensions, Td),
                                  Tv=(dataset_dimensions, Tv),
                                  theta=(dataset_dimensions, theta),
                                  theta_e=(dataset_dimensions, theta_e),
                                  theta_v=(dataset_dimensions, theta_v),
                                  RH=(dataset_dimensions, RH),
                                  r=(dataset_dimensions, r),
                                  q=(dataset_dimensions, q),
                                  u=(dataset_dimensions, u),
                                  v=(dataset_dimensions, v),
                                  sp_z=(dataset_dimensions, sp_z))

    full_dataset = xr.Dataset(data_vars=full_dataset_variables, coords=full_dataset_coordinates).astype('float32')

    # final dataset attributes
    full_dataset.attrs = dict(grib_to_netcdf_script_version="2024.3.27", model=args["model"], Nx=np.shape(T)[-1], Ny=np.shape(T)[-2])

    # turn the time coordinate into a dimension, allows for concatenation when opening multiple datasets
    full_dataset = full_dataset.expand_dims({"time": np.atleast_1d(timestep)})
    print(" (%.1f seconds)" % (time.time() - start_time))

    # folder containing netcdf data for the month of the dataset
    monthly_dir = '%s/%d%02d' % (args['netcdf_outdir'], year, month)
    if not os.path.isdir(monthly_dir):
        os.mkdir(monthly_dir)

    # save out netcdf files, one for each forecast hour
    for idx, forecast_hour in enumerate(forecast_hours):
        filepath = f"%s/%s_%d%02d%02d%02d_f%03d.nc" % (monthly_dir, args["model"], year, month, day, hour, forecast_hour)
        print("Saving: %s" % filepath)
        full_dataset.isel(forecast_hour=[idx, ]).to_netcdf(path=filepath, mode='w', engine='netcdf4')