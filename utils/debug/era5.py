"""
Debugging tools for ERA5 data

Code written by: Andrew Justin (andrewjustinwx@gmail.com)

Last updated: 10/13/2022 1:51 PM CT
"""


import file_manager as fm
import numpy as np
import xarray as xr
from utils.data_utils import normalize_variables
import os


def check_era5_variables(era5_netcdf_indir, timestep):
    """
    Function that lists the minimum, mean, and maximum values of all variables in an ERA5 dataset before and after normalization.

    Parameters
    ----------
    era5_netcdf_indir: str
        - Directory where the ERA5 netcdf files are stored.
    timestep: tuple with 4 ints
        - Timestep of the ERA5 dataset to analyze. The tuple must be passed with 4 integers in the following format: (year, month, day, hour)
    """
    
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


def check_for_corrupt_era5_files(era5_netcdf_indir, lower_bound_MB=84.3, upper_bound_MB=84.4):
    """
    Function that scans the size of all ERA5 files to search for corrupt files. Files with sizes less than 'lower_bound_MB'
    or larger than 'upper_bound_MB' are deemed to be corrupt.

    Parameters
    ----------
    era5_netcdf_indir: str
        - Directory where the ERA5 netcdf files are stored.
    lower_bound_MB: float or int
        - The minimum file size in megabytes that an ERA5 file must be for it to be deemed not corrupt. In other words, ERA5 files with
          sizes smaller than this number will be marked as corrupt.
    upper_bound_MB: float or int
        - The maximum file size in megabytes that an ERA5 file can be before it is marked as corrupt. In other words, ERA5 files with
          sizes larger than this number will be marked as corrupt.
    """
    era5_files = fm.ERA5files(era5_netcdf_indir).era5_files
    files_found = len(era5_files)

    corrupt_files = []
    corrupt_sizes = []
    num_corrupt_files = 0

    print(f"Scanning {files_found} ERA5 files")
    for index, file in enumerate(era5_files):
        print(f"{index + 1}/{files_found}", end='\r')
        size_of_file_mb = os.path.getsize(file)/1e6
        if size_of_file_mb > upper_bound_MB or size_of_file_mb < lower_bound_MB:

            num_corrupt_files += 1
            corrupt_files.append(file)
            corrupt_sizes.append(size_of_file_mb)

    print("Corrupt files found:", num_corrupt_files)
    if num_corrupt_files > 0:
        [print(file) for file in corrupt_files]


def find_missing_era5_data(era5_netcdf_indir):
    """
    Function that scans a directory and searches for missing ERA5 netcdf files.

    Parameters
    ----------
    era5_netcdf_indir: str
        - Directory where the ERA5 netcdf files are stored.
    """

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
