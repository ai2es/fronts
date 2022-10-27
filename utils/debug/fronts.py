"""
Debugging tools for front objects

Code written by: Andrew Justin (andrewjustinwx@gmail.com)

Last updated: 10/14/2022 9:02 PM CDT
"""


import os
import numpy as np


def find_missing_fronts_data(fronts_netcdf_indir):
    """
    Function that scans a directory and searches for missing front object netcdf files.

    Parameters
    ----------
    fronts_netcdf_indir: str
        - Directory where the front object netcdf files are stored.
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
                    if not os.path.isfile(f'{fronts_netcdf_indir}/%d/%02d/%02d/FrontObjects_%d%02d%02d%02d_full.nc' % (year, month, day, year, month, day, hour)):

                        index = int(np.sum(days_per_month[:month-1]) + day) - 1
                        missing_indices[str(year)].append(index)

    for year in years:
        year = str(year)
        missing_indices[year] = np.unique(missing_indices[year])
        print(f"\nMissing indices [{year}]: {','.join(missing_indices[year].astype(str))}")
