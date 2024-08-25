"""
Download grib files containing NWP model data.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.8.14
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
    progress_message = 'Downloading %s: %d%% [%d/%d] MB       ' % (local_filename, current / total * 100, current / 1e6, total / 1e6)
    sys.stdout.write('\r' + progress_message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grib_outdir', type=str, required=True, help="Output directory for GDAS grib files downloaded from NCEP.")
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
        raise ValueError('Only one of the following arguments can be passed: --init_time, --range')
    elif args['init_time'] is None and args['range'] is None:
        raise ValueError('One of the following arguments must be passed: --init_time, --range')

    init_times = pd.date_range(args['init_time'], args['init_time']) if args['init_time'] is not None else pd.date_range(*args['range'][:2], freq=args['range'][-1])

    files = []  # complete urls for the files to pull from AWS
    local_filenames = []  # filenames for the local files after downloading
    
    for init_time in init_times:
        yr, mo, dy, hr = init_time.year, init_time.month, init_time.day, init_time.hour
        if args['model'] == 'gdas':
            if datetime.datetime(yr, mo, dy, hr) < datetime.datetime(2015, 6, 23, 0):
                raise ConnectionAbortedError('Cannot download GDAS data prior to June 23, 2015.')
            elif datetime.datetime(yr, mo, dy, hr) < datetime.datetime(2017, 7, 20, 0):
                [files.append('https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.%d%02d%02d/%02d/gdas1.t%02dz.pgrb2.0p25.f%03d' % (yr, mo, dy, hr, hr, forecast_hour))
                 for forecast_hour in args['forecast_hours']]
            elif yr < 2021:
                [files.append('https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.%d%02d%02d/%02d/gdas.t%02dz.pgrb2.0p25.f%03d' % (yr, mo, dy, hr, hr, forecast_hour))
                 for forecast_hour in args['forecast_hours']]
            else:
                [files.append('https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.%d%02d%02d/%02d/atmos/gdas.t%02dz.pgrb2.0p25.f%03d' % (yr, mo, dy, hr, hr, forecast_hour))
                 for forecast_hour in args['forecast_hours']]
        elif 'gefs' in args['model']:
            member = args['model'].split('-')[-1]
            if datetime.datetime(yr, mo, dy, hr) < datetime.datetime(2017, 1, 1, 0):
                raise ConnectionAbortedError('Cannot download GEFS data prior to January 1, 2017.')
            elif datetime.datetime(yr, mo, dy, hr) < datetime.datetime(2018, 7, 27, 0):
                [files.append('https://noaa-gefs-pds.s3.amazonaws.com/gefs.%d%02d%02d/%02d/ge%s.t%02dz.pgrb2af%03d' % (yr, mo, dy, hr, member, hr, forecast_hour))
                 for forecast_hour in args['forecast_hours']]
            elif datetime.datetime(yr, mo, dy, hr) < datetime.datetime(2020, 9, 24, 0):
                [files.append('https://noaa-gefs-pds.s3.amazonaws.com/gefs.%d%02d%02d/%02d/pgrb2a/ge%s.t%02dz.pgrb2af%02d' % (yr, mo, dy, hr, member, hr, forecast_hour))
                 for forecast_hour in args['forecast_hours']]
            else:
                [files.append('https://noaa-gefs-pds.s3.amazonaws.com/gefs.%d%02d%02d/%02d/atmos/pgrb2ap5/ge%s.t%02dz.pgrb2a.0p50.f%03d' % (yr, mo, dy, hr, member, hr, forecast_hour))
                 for forecast_hour in args['forecast_hours']]
        elif args['model'] == 'gfs':
            if datetime.datetime(yr, mo, dy, hr) < datetime.datetime(2021, 2, 26, 0):
                raise ConnectionAbortedError('Cannot download GFS data prior to February 26, 2021.')
            elif datetime.datetime(yr, mo, dy, hr) < datetime.datetime(2021, 3, 22, 0):
                [files.append('https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.%d%02d%02d/%02d/gfs.t%02dz.pgrb2.0p25.f%03d' % (yr, mo, dy, hr, hr, forecast_hour))
                 for forecast_hour in args['forecast_hours']]
            else:
                [files.append('https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.%d%02d%02d/%02d/atmos/gfs.t%02dz.pgrb2.0p25.f%03d' % (yr, mo, dy, hr, hr, forecast_hour))
                 for forecast_hour in args['forecast_hours']]
        elif args['model'] == 'hrrr':
            if datetime.datetime(yr, mo, dy, hr) < datetime.datetime(2014, 7, 30, 18):
                raise ConnectionAbortedError('Cannot download HRRR data prior to 18z July 30, 2014.')
            [files.append('https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.%d%02d%02d/conus/hrrr.t%02dz.wrfprsf%02d.grib2' % (yr, mo, dy, hr, forecast_hour))
             for forecast_hour in args['forecast_hours']]
        elif args['model'] == 'rap':
            if datetime.datetime(yr, mo, dy, hr) < datetime.datetime(2021, 2, 22, 0):
                raise ConnectionAbortedError('Cannot download RAP data prior to 0z February 22, 2021.')
            [files.append('https://noaa-rap-pds.s3.amazonaws.com/rap.%d%02d%02d/rap.t%02dz.wrfprsf%02d.grib2' % (yr, mo, dy, hr, forecast_hour))
             for forecast_hour in args['forecast_hours']]
        elif 'ecmwf' in args['model']:
            ecmwf_model = args['model'].split('-')[1]
            [files.append('https://data.ecmwf.int/forecasts/%d%02d%02d/%02dz/%s/0p25/oper/%d%02d%02d%02d0000-%dh-oper-fc.grib2' % (yr, mo, dy, hr, ecmwf_model, yr, mo, dy, hr, forecast_hour))
             for forecast_hour in args['forecast_hours']]
        elif 'namnest' in args['model']:
            nest = args['model'].split('-')[-1]
            [files.append('https://nomads.ncep.noaa.gov/pub/data/nccf/com/nam/prod/nam.%d%02d%02d/nam.t%02dz.%snest.hiresf%02d.tm00.grib2' % (yr, mo, dy, hr, nest, forecast_hour))
             for forecast_hour in args['forecast_hours']]
        elif args['model'] == 'nam-12km':
            for forecast_hour in args['forecast_hours']:
                if forecast_hour in [0, 1, 2, 3, 6]:
                    folder = 'analysis'  # use the analysis folder as it contains more accurate data
                else:
                    folder = 'forecast'  # forecast hours other than 0, 1, 2, 3, 6 do not have analysis data

                if datetime.datetime(yr, mo, dy, hr) > datetime.datetime.utcnow() - datetime.timedelta(days=7):
                    files.append(f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/nam/prod/nam.%d%02d%02d/nam.t%02dz.awphys%02d.tm00.grib2' %
                                 (yr, mo, dy, hr, forecast_hour))
                else:
                    files.append(f'https://www.ncei.noaa.gov/data/north-american-mesoscale-model/access/%s/%d%02d/%d%02d%02d/nam_218_%d%02d%02d_%02d00_%03d.grb2' %
                                 (folder, yr, mo, yr, mo, dy, yr, mo, dy, hr, forecast_hour))
        [local_filenames.append('%s_%d%02d%02d%02d_f%03d.grib' % (args['model'], yr, mo, dy, hr, forecast_hour)) for forecast_hour in args['forecast_hours']]

    for file, local_filename in zip(files, local_filenames):

        timestring = local_filename.split('_')[1]
        year, month = timestring[:4], timestring[4:6]
        monthly_directory = '%s/%s%s' % (args['grib_outdir'], year, month)  # Directory for the grib files for the given days

        ### If the directory does not exist, check to see if the file link is valid. If the file link is NOT valid, then the directory will not be created since it will be empty. ###
        if not os.path.isdir(monthly_directory):
            if requests.head(file).status_code == requests.codes.ok or requests.head(file.replace('/atmos', '')).status_code == requests.codes.ok:
                os.mkdir(monthly_directory)

        full_file_path = f'{monthly_directory}/{local_filename}'

        if not os.path.isfile(full_file_path):
            try:
                wget.download(file, out=full_file_path, bar=bar)
            except urllib.error.HTTPError:
                print('Error downloading %s' % file)
        else:
            print('%s already exists, skipping file....' % full_file_path)
