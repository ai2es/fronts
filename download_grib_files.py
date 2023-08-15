"""
Download grib files for GDAS and/or GFS data.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 8/10/2023 6:02 PM CT
"""

import argparse
import os
import requests
import urllib.error
import warnings
import wget
import sys
import datetime


def bar_progress(current, total, width=None):
    progress_message = "Downloading: %d%% [%d/%d] MB       " % (current / total * 100, current / 1e6, total / 1e6)
    sys.stdout.write("\r" + progress_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grib_outdir', type=str, required=True, help='Output directory for GDAS grib files downloaded from NCEP.')
    parser.add_argument('--model', type=str, required=True, help="NWP model to use as the data source.")
    parser.add_argument('--init_time', type=int, nargs=4, required=True, help="Initialization time of the model.")
    parser.add_argument('--forecast_hours', type=int, nargs="+", required=True, help="List of forecast hours to download for the given day.")

    args = vars(parser.parse_args())

    year, month, day, hour = args['init_time']
    args['model'] = args['model'].lower()

    # GDAS files have different naming patterns based on which year the data is for
    if args['model'] == 'gdas':
        if datetime.datetime(year, month, day, hour) < datetime.datetime(2015, 6, 23, 0):
            raise ConnectionAbortedError("Cannot download GDAS data prior to June 23, 2015.")
        elif datetime.datetime(year, month, day, hour) < datetime.datetime(2017, 7, 20, 0):
            files = [f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.%d%02d%02d/%02d/gdas1.t%02dz.pgrb2.0p25.f%03d' % (year, month, day, hour, hour, forecast_hour)
                     for forecast_hour in args['forecast_hours']]
        elif year < 2021:
            files = [f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.%d%02d%02d/%02d/gdas.t%02dz.pgrb2.0p25.f%03d' % (year, month, day, hour, hour, forecast_hour)
                     for forecast_hour in args['forecast_hours']]
        else:
            files = [f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.%d%02d%02d/%02d/atmos/gdas.t%02dz.pgrb2.0p25.f%03d" % (year, month, day, hour, hour, forecast_hour)
                     for forecast_hour in args['forecast_hours']]
    elif args['model'] == 'gfs':
        if datetime.datetime(year, month, day, hour) < datetime.datetime(2021, 2, 26, 0):
            raise ConnectionAbortedError("Cannot download GFS data prior to February 26, 2021.")
        files = [f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.%d%02d%02d/%02d/atmos/gfs.t%02dz.pgrb2.0p25.f%03d" % (year, month, day, hour, hour, forecast_hour)
                 for forecast_hour in args['forecast_hours']]
    elif args['model'] == 'hrrr':
        files = [f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.%d%02d%02d/conus/hrrr.t%02dz.wrfprsf%02d.grib2" % (year, month, day, hour, forecast_hour)
                 for forecast_hour in args['forecast_hours']]
    elif args['model'] == 'rap':
        files = [f"https://noaa-rap-pds.s3.amazonaws.com/rap.%d%02d%02d/rap.t%02dz.wrfprsf%02d.grib2" % (year, month, day, hour, forecast_hour)
                 for forecast_hour in args['forecast_hours']]
    elif 'namnest' in args['model']:
        nest = args['model'].split('_')[-1]
        files = [f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/nam/prod/nam.%d%02d%02d/nam.t%02dz.%snest.hiresf%02d.tm00.grib2" % (year, month, day, hour, nest, forecast_hour)
                 for forecast_hour in args['forecast_hours']]
    elif args['model'] == 'nam_12km':
        files = []
        for forecast_hour in args['forecast_hours']:
            if forecast_hour in [0, 1, 2, 3, 6]:
                folder = 'analysis'  # use the analysis folder as it contains more accurate data
            else:
                folder = 'forecast'  # forecast hours other than 0, 1, 2, 3, 6 do not have analysis data
            files.append(f"https://www.ncei.noaa.gov/data/north-american-mesoscale-model/access/%s/%d%02d/%d%02d%02d/nam_218_%d%02d%02d_%02d00_%03d.grb2" %
                         (folder, year, month, year, month, day, year, month, day, hour, forecast_hour))

    local_filenames = ["%s_%d%02d%02d%02d_f%03d.grib" % (args['model'], year, month, day, hour, forecast_hour) for forecast_hour in args['forecast_hours']]

    for file, local_filename in zip(files, local_filenames):

        monthly_directory = '%s/%d%02d' % (args['grib_outdir'], year, month)  # Directory for the grib files for the given days

        ### If the directory does not exist, check to see if the file link is valid. If the file link is NOT valid, then the directory will not be created since it will be empty. ###
        if not os.path.isdir(monthly_directory):
            if requests.head(file).status_code == requests.codes.ok or requests.head(file.replace('/atmos', '')).status_code == requests.codes.ok:
                os.mkdir(monthly_directory)

        full_file_path = f'{monthly_directory}/{local_filename}'

        if not os.path.isfile(full_file_path) and os.path.isdir(monthly_directory):
            try:
                wget.download(file, out=full_file_path, bar=bar_progress)
            except urllib.error.HTTPError:
                try:
                    wget.download(file.replace('/atmos', ''), out=full_file_path, bar=bar_progress)
                except urllib.error.HTTPError:
                    print(f"Error downloading {file}")
                    pass

        elif not os.path.isdir(monthly_directory):
            warnings.warn(f"Unknown problem encountered when creating the following directory: %s, "
                          f"Consider checking the AWS server to make sure that data exists for the given day (%d-%02d-%02d): "
                          f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/index.html#%s.%d%02d%02d" %
                          (monthly_directory, year, month, day, args['model'], year, month, day))
            break

        elif os.path.isfile(full_file_path):
            print(f"{full_file_path} already exists, skipping file....")
