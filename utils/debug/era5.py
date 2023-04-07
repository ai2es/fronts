"""
Debugging tools for ERA5 data

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 3/27/2023 9:11 PM CT
"""


import file_manager as fm
import numpy as np
import xarray as xr
from utils.data_utils import normalize_variables
import os


def check_era5_variables(era5_netcdf_indir: str, timestep: tuple):
    """
    Function that lists the minimum, mean, and maximum values of all variables in an ERA5 dataset before and after normalization.

    Parameters
    ----------
    era5_netcdf_indir: str
        Directory where the ERA5 netcdf files are stored.
    timestep: tuple with 4 ints
        Timestep of the ERA5 dataset to analyze. The tuple must be passed with 4 integers in the following format: (year, month, day, hour)
    """

    ######################################### Check the parameters for errors ##########################################
    if era5_netcdf_indir is not None and not isinstance(era5_netcdf_indir, str):
        raise TypeError(f"era5_netcdf_indir must be a string, received {type(era5_netcdf_indir)}")

    if not isinstance(timestep, tuple):
        raise TypeError(f"Expected a tuple for timestep, received {type(timestep)}")
    elif len(timestep) != 2:
        raise TypeError(f"Tuple for timestep must be length 4, received length {len(timestep)}")
    ####################################################################################################################

    era5_files_obj = fm.ERA5files(era5_netcdf_indir)
    era5_files_obj.variables = ['T', 'Td', 'u', 'v', 'sp_z', 'theta_e', 'theta_w', 'Tw', 'Tv', 'RH', 'r', 'q']
    era5_files_obj.sort_by_timestep()
    era5_files = era5_files_obj.era5_files

    timestep_str = '%d%02d%02d%02d' % (timestep[0], timestep[1], timestep[2], timestep[3])
    timestep_index = [index for index, files_for_timestep in enumerate(era5_files) if all(timestep_str in file for file in files_for_timestep)][0]
    era5_files = era5_files[timestep_index]
    era5_dataset = xr.open_mfdataset(era5_files, engine='scipy').transpose('time', 'longitude', 'latitude', 'pressure_level')
    
    print("Before normalization")
    for key in list(era5_dataset.keys()):
        for pressure_level in era5_dataset['pressure_level'].values:
            current_var = era5_dataset.sel(pressure_level=pressure_level)[key]
            print(key, pressure_level, np.nanmin(current_var), np.nanmean(current_var), np.nanmax(current_var))

    era5_dataset_norm = normalize_variables(era5_dataset)

    print("After normalization")
    for key in list(era5_dataset_norm.keys()):
        for pressure_level in era5_dataset_norm['pressure_level'].values:
            current_var = era5_dataset_norm.sel(pressure_level=pressure_level)[key]
            print(key, pressure_level, np.nanmin(current_var), np.nanmean(current_var), np.nanmax(current_var))


def find_missing_era5_data(era5_netcdf_indir: str):
    """
    Function that scans a directory and searches for missing ERA5 netcdf files.

    Parameters
    ----------
    era5_netcdf_indir: str
        Directory where the ERA5 netcdf files are stored.
    """

    if era5_netcdf_indir is not None and not isinstance(era5_netcdf_indir, str):
        raise TypeError(f"era5_netcdf_indir must be a string, received {type(era5_netcdf_indir)}")

    years = np.arange(2008, 2021)
    missing_indices = dict({str(year): [] for year in years})

    for year in years:

        if year % 4 == 0:  # If it is a leap year
            month_2_days = 29
        else:
            month_2_days = 28

        days_per_month = [31, month_2_days, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        for month in range(1, 13):
            for day in range(1, days_per_month[month - 1] + 1):
                for hour in range(0, 24, 3):
                    if not os.path.isfile(f'{era5_netcdf_indir}/%d/%02d/%02d/era5_%d%02d%02d%02d_full.nc' % (year, month, day, year, month, day, hour)):

                        index = int(np.sum(days_per_month[:month-1]) + day) - 1
                        missing_indices[str(year)].append(index)

    for year in years:
        year = str(year)
        missing_indices[year] = np.unique(missing_indices[year])
        print(f"\nMissing indices [{year}]: {','.join(missing_indices[year].astype(str))}")
