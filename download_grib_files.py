"""
Download grib files for GDAS and/or GFS data.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 6/25/2023 12:07 AM CT
"""

import argparse
from bs4 import BeautifulSoup
import numpy as np
import os
import requests
import urllib.error
import warnings
import wget


def from_day_range(grib_outdir: str, year: int, day_range: tuple | list, model: str, forecast_hours: tuple | list):
    """
    Download grib files containing data for a NWP model from the AWS server within a given day range.

    Parameters
    ----------
    grib_outdir: str
        Output directory for the downloaded GRIB files.
    year: year
    day_range: tuple or list
        Tuple of two integers marking the start and end indices of the days of the year to download. The end index IS included.
        Example: to download all days (1-365), enter (0, 364).
    model: str
        Model/source of the data. Current options are: 'gdas', 'gfs'.
    forecast_hours: tuple or list
        List of forecast hours to download for the given days.
    """

    ######################################### Check the parameters for errors ##########################################
    if not isinstance(grib_outdir, str):
        raise TypeError(f"grib_outdir must be a string, received {type(grib_outdir)}")
    if not isinstance(year, int):
        raise TypeError(f"year must be an integer, received {type(year)}")
    if not isinstance(model, str):
        raise TypeError(f"model must be a string, received {type(model)}")
    if not isinstance(day_range, (tuple, list)):
        raise TypeError(f"day_range must be a tuple or list, received {type(day_range)}")
    elif len(day_range) != 2:
        raise TypeError(f"Tuple for day_range must be length 2, received length {len(day_range)}")
    ####################################################################################################################

    model_lowercase = model.lower()

    if year % 4 == 0:
        month_2_days = 29  # Leap year
    else:
        month_2_days = 28  # Not a leap year

    days_per_month = [31, month_2_days, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    ### Generate a list of dates [[year, 1, 1], [year, ..., ...], [year, 12, 31]] ###
    date_list = []
    for month in range(12):
        for day in range(days_per_month[month]):
            date_list.append([year, month + 1, day + 1])

    for day_index in range(day_range[0], day_range[1] + 1):

        month, day = date_list[day_index][1], date_list[day_index][2]
        hours = np.arange(0, 24, 6)[::-1]  # Hours that the GDAS model runs forecasts for

        monthly_directory = grib_outdir + '/%d%02d' % (year, month)  # Directory for the grib files for the given days

        # GDAS files have different naming patterns based on which year the data is for
        if year < 2014:

            files = [f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.{year}%02d%02d/%02d/gdas1.t%02dz.pgrbf%02d.grib2' % (month, day, hour, hour, forecast_hour)
                     for hour in hours
                     for forecast_hour in forecast_hours]
            local_filenames = ['gdas1.t%02dz.pgrbf%02d.grib2' % (hour, forecast_hour)
                               for hour in hours
                               for forecast_hour in forecast_hours]

        elif year < 2021:

            files = [f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.{year}%02d%02d/%02d/gdas1.t%02dz.pgrb2.0p25.f%03d' % (month, day, hour, hour, forecast_hour)
                     for hour in hours
                     for forecast_hour in forecast_hours]
            local_filenames = ['gdas.t%02dz.pgrb2.0p25.f%03d' % (hour, forecast_hour)
                               for hour in hours
                               for forecast_hour in forecast_hours]

        else:

            files = [f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/{model_lowercase}.{year}%02d%02d/%02d/atmos/{model_lowercase}.t%02dz.pgrb2.0p25.f%03d' % (month, day, hour, hour, forecast_hour)
                     for hour in hours
                     for forecast_hour in forecast_hours]

            local_filenames = [f'{model}.t%02dz.pgrb2.0p25.f%03d' % (hour, forecast_hour)
                               for hour in hours
                               for forecast_hour in forecast_hours]

        for file, local_filename in zip(files, local_filenames):

            ### If the directory does not exist, check to see if the file link is valid. If the file link is NOT valid, then the directory will not be created since it will be empty. ###
            if not os.path.isdir(monthly_directory):
                if requests.head(file).status_code == requests.codes.ok or requests.head(file.replace('/atmos', '')).status_code == requests.codes.ok:
                    os.mkdir(monthly_directory)

            full_file_path = f'{monthly_directory}/{local_filename}'

            if not os.path.isfile(full_file_path) and os.path.isdir(monthly_directory):
                try:
                    wget.download(file, out=full_file_path)
                except urllib.error.HTTPError:
                    try:
                        wget.download(file.replace('/atmos', ''), out=full_file_path)
                    except urllib.error.HTTPError:
                        print(f"Error downloading {file}")
                        pass

            elif not os.path.isdir(monthly_directory):
                warnings.warn(f"Unknown problem encountered when creating the following directory: {monthly_directory}, "
                              f"Consider checking the AWS server to make sure that data exists for the given day ({year}-%02d-%02d): "
                              f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/index.html#{model_lowercase}.{year}%02d%02d" % (date_list[day_index][1], date_list[day_index][2], date_list[day_index][1], date_list[day_index][2]))
                break

            elif os.path.isfile(full_file_path):
                print(f"{full_file_path} already exists, skipping file....")


def latest_data(grib_outdir: str, model: str = 'gdas'):
    """
    Download the latest model grib files from NCEP.

    Parameters
    ----------
    grib_outdir: str
        Output directory for the downloaded GRIB files.
    model: str
        Model/source of the data. Current options are: 'gdas', 'gfs'.
    """

    ######################################### Check the parameters for errors ##########################################
    if not isinstance(grib_outdir, str):
        raise TypeError(f"grib_outdir must be a string, received {type(grib_outdir)}")
    if not isinstance(model, str):
        raise TypeError(f"model must be a string, received {type(model)}")
    ####################################################################################################################

    model_uppercase = model.upper()
    model_lowercase = model.lower()

    print(f"Updating {model_uppercase} files in directory: {grib_outdir}")

    print(f"Connecting to main {model_uppercase} source")
    url = f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/'
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    gdas_folders_by_day = [url + node.get('href') for node in soup.find_all('a') if model_lowercase in node.get('href')]
    timesteps_available = [folder[-9:-1] for folder in gdas_folders_by_day if f'prod/{model_lowercase}' in folder][::-1]

    files_added = 0

    for timestep in timesteps_available:

        year, month, day = timestep[:4], timestep[4:6], timestep[6:]
        monthly_directory = '%s/%s' % (grib_outdir, timestep[:6])

        url = f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/{model_lowercase}.{timestep}/'
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'html.parser')

        gdas_folders_by_hour = [url + node.get('href') for node in soup.find_all('a')]
        hours_available = [folder[-3:-1] for folder in gdas_folders_by_hour if any(hour in folder[-3:-1] for hour in ['00', '06', '12', '18'])][::-1]
        forecast_hours = np.arange(0, 54, 6)

        for hour in hours_available:

            print(f"\nSearching for available data: {year}-{month}-{day}-%02dz" % float(hour))

            url = f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/{model_lowercase}.{timestep}/%02d/atmos/' % float(hour)
            page = requests.get(url).text
            soup = BeautifulSoup(page, 'html.parser')
            files = [url + node.get('href') for node in soup.find_all('a') if any('.pgrb2.0p25.f%03d' % forecast_hour in node.get('href') for forecast_hour in forecast_hours)]

            if not os.path.isdir(monthly_directory):
                os.mkdir(monthly_directory)

            for file in files:
                local_filename = file.replace(url, '')
                full_file_path = f'{monthly_directory}/{local_filename}'
                if not os.path.isfile(full_file_path):
                    try:
                        wget.download(file, out=full_file_path)
                        files_added += 1
                    except urllib.error.HTTPError:
                        print(f"Error downloading {url}{file}")
                        pass
                else:
                    print(f"{full_file_path} already exists")

    print(f"{model_uppercase} directory update complete, %d files added" % files_added)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grib_outdir', type=str, required=True, help='Output directory for GDAS grib files downloaded from NCEP.')
    parser.add_argument('--model', type=str, required=True, help="model of the data for which grib files will be split: 'GDAS' or 'GFS'")
    parser.add_argument('--forecast_hours', type=int, nargs="+", required=True, help="List of forecast hours to download for the given day.")
    parser.add_argument('--latest', action='store_true', help="Download the latest model data.")

    ### Both of these arguments must be passed together ###
    parser.add_argument('--year', type=int, help="Year for the data to be downloaded.")
    parser.add_argument('--day_range', type=int, nargs=2, help="Start and end days of the year for grabbing grib files")

    args = vars(parser.parse_args())

    if args['latest']:
        latest_data(args['grib_outdir'], args['model'])

    if args['year'] is not None and args['day_range'] is not None:
        from_day_range(args['grib_outdir'], args['year'], args['day_range'], args['model'], args['forecast_hours'])
