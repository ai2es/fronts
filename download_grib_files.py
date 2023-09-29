"""
Download grib files for GDAS and/or GFS data.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.9.28
"""

import argparse
import os
import pandas as pd
import requests
import urllib.error
import wget
import sys
import datetime


def bar_progress(current, total, width=None):
    progress_message = "Downloading %s: %d%% [%d/%d] MB       " % (local_filename, current / total * 100, current / 1e6, total / 1e6)
    sys.stdout.write("\r" + progress_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grib_outdir', type=str, required=True, help='Output directory for GDAS grib files downloaded from NCEP.')
    parser.add_argument('--model', type=str, required=True, help="NWP model to use as the data source.")
    parser.add_argument('--init_time', type=str, help="Initialization time of the model. Format: YYYY-MM-DD-HH.")
    parser.add_argument('--range', type=str, nargs=3,
        help="Download model data between a range of dates. Three arguments must be passed, with the first two arguments "
             "marking the bounds of the date range in the format YYYY-MM-DD-HH. The third argument is the frequency (e.g. 6H), "
             "which has the same formatting as the 'freq' keyword argument in pandas.date_range().")
    parser.add_argument('--forecast_hours', type=int, nargs="+", required=True, help="List of forecast hours to download for the given day.")
    parser.add_argument('--verbose', action='store_true', help="Include a progress bar for download progress.")

    args = vars(parser.parse_args())

    args['model'] = args['model'].lower()

    # If --verbose is passed, include a progress bar to show the download progress
    bar = bar_progress if args['verbose'] else None

    if args['init_time'] is not None and args['range'] is not None:
        raise ValueError("Only one of the following arguments can be passed: --init_time, --range")
    elif args['init_time'] is None and args['range'] is None:
        raise ValueError("One of the following arguments must be passed: --init_time, --range")

    init_times = pd.date_range(args['init_time'], args['init_time']) if args['init_time'] is not None else pd.date_range(*args['range'][:2], freq=args['range'][-1])

    files = []  # complete urls for the files to pull from AWS
    local_filenames = []  # filenames for the local files after downloading
    
    for init_time in init_times:
        if args['model'] == 'gdas':
            if datetime.datetime(init_time.year, init_time.month, init_time.day, init_time.hour) < datetime.datetime(2015, 6, 23, 0):
                raise ConnectionAbortedError("Cannot download GDAS data prior to June 23, 2015.")
            elif datetime.datetime(init_time.year, init_time.month, init_time.day, init_time.hour) < datetime.datetime(2017, 7, 20, 0):
                [files.append(f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.%d%02d%02d/%02d/gdas1.t%02dz.pgrb2.0p25.f%03d' % (init_time.year, init_time.month, init_time.day, init_time.hour, init_time.hour, forecast_hour))
                 for forecast_hour in args['forecast_hours']]
            elif init_time.year < 2021:
                [files.append(f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.%d%02d%02d/%02d/gdas.t%02dz.pgrb2.0p25.f%03d' % (init_time.year, init_time.month, init_time.day, init_time.hour, init_time.hour, forecast_hour))
                 for forecast_hour in args['forecast_hours']]
            else:
                [files.append(f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.%d%02d%02d/%02d/atmos/gdas.t%02dz.pgrb2.0p25.f%03d" % (init_time.year, init_time.month, init_time.day, init_time.hour, init_time.hour, forecast_hour))
                 for forecast_hour in args['forecast_hours']]
        elif args['model'] == 'gfs':
            if datetime.datetime(init_time.year, init_time.month, init_time.day, init_time.hour) < datetime.datetime(2021, 2, 26, 0):
                raise ConnectionAbortedError("Cannot download GFS data prior to February 26, 2021.")
            elif datetime.datetime(init_time.year, init_time.month, init_time.day, init_time.hour) < datetime.datetime(2021, 3, 22, 0):
                [files.append(f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.%d%02d%02d/%02d/gfs.t%02dz.pgrb2.0p25.f%03d" % (init_time.year, init_time.month, init_time.day, init_time.hour, init_time.hour, forecast_hour))
                 for forecast_hour in args['forecast_hours']]
            else:
                [files.append(f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.%d%02d%02d/%02d/atmos/gfs.t%02dz.pgrb2.0p25.f%03d" % (init_time.year, init_time.month, init_time.day, init_time.hour, init_time.hour, forecast_hour))
                 for forecast_hour in args['forecast_hours']]
        elif args['model'] == 'hrrr':
            [files.append(f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.%d%02d%02d/conus/hrrr.t%02dz.wrfprsf%02d.grib2" % (init_time.year, init_time.month, init_time.day, init_time.hour, forecast_hour))
             for forecast_hour in args['forecast_hours']]
        elif args['model'] == 'rap':
            [files.append(f"https://noaa-rap-pds.s3.amazonaws.com/rap.%d%02d%02d/rap.t%02dz.wrfprsf%02d.grib2" % (init_time.year, init_time.month, init_time.day, init_time.hour, forecast_hour))
             for forecast_hour in args['forecast_hours']]
        elif 'namnest' in args['model']:
            nest = args['model'].split('_')[-1]
            [files.append(f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/nam/prod/nam.%d%02d%02d/nam.t%02dz.%snest.hiresf%02d.tm00.grib2" % (init_time.year, init_time.month, init_time.day, init_time.hour, nest, forecast_hour))
             for forecast_hour in args['forecast_hours']]
        elif args['model'] == 'nam_12km':
            for forecast_hour in args['forecast_hours']:
                if forecast_hour in [0, 1, 2, 3, 6]:
                    folder = 'analysis'  # use the analysis folder as it contains more accurate data
                else:
                    folder = 'forecast'  # forecast hours other than 0, 1, 2, 3, 6 do not have analysis data
                files.append(f"https://www.ncei.noaa.gov/data/north-american-mesoscale-model/access/%s/%d%02d/%d%02d%02d/nam_218_%d%02d%02d_%02d00_%03d.grb2" %
                             (folder, init_time.year, init_time.month, init_time.year, init_time.month, init_time.day, init_time.year, init_time.month, init_time.day, init_time.hour, forecast_hour))
        [local_filenames.append("%s_%d%02d%02d%02d_f%03d.grib" % (args['model'].replace('_', ''), init_time.year, init_time.month, init_time.day, init_time.hour, forecast_hour)) for forecast_hour in args['forecast_hours']]

    for file, local_filename in zip(files, local_filenames):

        init_time = local_filename.split('_')[1] if 'nam' not in args['model'] else local_filename.split('_')[2]
        init_time = pd.to_datetime(f'{init_time[:4]}-{init_time[4:6]}-{init_time[6:8]}-{init_time[8:10]}')

        monthly_directory = '%s/%d%02d' % (args['grib_outdir'], init_time.year, init_time.month)  # Directory for the grib files for the given days

        ### If the directory does not exist, check to see if the file link is valid. If the file link is NOT valid, then the directory will not be created since it will be empty. ###
        if not os.path.isdir(monthly_directory):
            if requests.head(file).status_code == requests.codes.ok or requests.head(file.replace('/atmos', '')).status_code == requests.codes.ok:
                os.mkdir(monthly_directory)

        full_file_path = f'{monthly_directory}/{local_filename}'

        if not os.path.isfile(full_file_path):
            try:
                wget.download(file, out=full_file_path, bar=bar)
            except urllib.error.HTTPError:
                print(f"Error downloading {file}")
        else:
            print(f"{full_file_path} already exists, skipping file....")
