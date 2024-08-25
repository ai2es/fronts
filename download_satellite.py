"""
Download netCDF files containing GOES satellite data.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.8.14
"""

import argparse
import os
import pandas as pd
import requests
import s3fs
import sys
import urllib.error
import wget


def bar_progress(current, total, width=None):
    progress_message = "Downloading %s: %d%% [%d/%d] MB       " % (local_filename, current / total * 100, current / 1e6, total / 1e6)
    sys.stdout.write("\r" + progress_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--netcdf_outdir', type=str, required=True, help="Output directory for the satellite netCDF files.")
    parser.add_argument('--satellite', type=str, default='goes16', help="GOES satellite source. Options are 'goes16', 'goes17', 'goes18'.")
    parser.add_argument('--domain', type=str, default='full-disk', help="Domain of the satellite data. Options are 'full_disk', 'conus', 'meso'.")
    parser.add_argument('--product', type=str, default='ABI-L2-MCMIP', help="Satellite product to download.")
    parser.add_argument('--init_time', type=str, help="Initialization time for which to search for satellite data. Format: YYYY-MM-DD-HH.")
    parser.add_argument('--range', type=str, nargs=3,
        help="Download satellite data between a range of dates. Three arguments must be passed, with the first two arguments "
             "marking the bounds of the date range in the format YYYY-MM-DD-HH. The third argument is the frequency (e.g. 6H), "
             "which has the same formatting as the 'freq' keyword argument in pandas.date_range().")
    parser.add_argument('--verbose', action='store_true', help="Include a progress bar for download progress.")
    args = vars(parser.parse_args())

    init_times = pd.date_range(args['init_time'], args['init_time']) if args['init_time'] is not None else pd.date_range(*args['range'][:2], freq=args['range'][-1])

    files = []  # complete urls for the files to pull from AWS
    local_filenames = []  # filenames for the local files after downloading

    if args['satellite'] not in ['goes13', 'goes15', 'goes16', 'goes17', 'goes18']:
        raise ValueError("'%s' is not a valid satellite source. Options are 'goes16', 'goes17', 'goes18'." % args['satellite'])

    if args['domain'] == 'full-disk':
        domain_str = 'F'
    elif args['domain'] == 'conus':
        domain_str = 'C'
    elif args['domain'] == 'meso':
        domain_str = 'M'
    else:
        raise ValueError("'%s' is not a valid domain. Options are 'full-disk', 'conus', 'meso'." % args['domain'])

    files = []  # complete urls for the files to pull from AWS
    local_filenames = []  # filenames for the local files after downloading
    
    if args['satellite'] in ['goes16', 'goes17', 'goes18']:
        
        fs = s3fs.S3FileSystem(anon=True)
        
        for i, init_time in enumerate(init_times):
            
            day_of_year = init_time.timetuple().tm_yday
    
            s3_folder = 's3://noaa-%s/%s%s/%d/%03d/%02d' % (args['satellite'], args['product'], domain_str, init_time.year, day_of_year, init_time.hour)
            s3_timestring = '%d%03d%02d' % (init_time.year, day_of_year, init_time.hour)
            s3_glob_str = s3_folder + "/*s" + s3_timestring + "*.nc"
            
            try:
                files.append(list(sorted(fs.glob(s3_glob_str)))[0].replace("noaa-%s" % args['satellite'], "https://noaa-%s.s3.amazonaws.com" % args['satellite']))
            except IndexError:
                print("NO DATA FOUND (ind %d): satellite=[%s] domain=[%s] product=[%s] init_time=[%s] " % (i, args['satellite'], args['domain'], args['product'], init_time))
            else:
                print("Gathering data: satellite=[%s] domain=[%s] product=[%s] init_time=[%s]" % (args['satellite'], args['domain'], args['product'], init_time), end='\r')
                local_filenames.append("%s_%d%02d%02d%02d_%s.nc" % (args['satellite'], init_time.year, init_time.month, init_time.day, init_time.hour, args['domain']))
    
    else:
    
        for i, init_time in enumerate(init_times):
            
            files.append('https://www.ncei.noaa.gov/data/gridsat-goes/access/goes/%d/%02d/GridSat-GOES.%s.%d.%02d.%02d.%02d00.v01.nc' %
                         (init_time.year, init_time.month, args['satellite'], init_time.year, init_time.month, init_time.day, init_time.hour))
            local_filenames.append("%s_%d%02d%02d%02d_%s.nc" % (args['satellite'], init_time.year, init_time.month, init_time.day, init_time.hour, args['domain']))
            
    # If --verbose is passed, include a progress bar to show the download progress
    bar = bar_progress if args['verbose'] else None

    if args['init_time'] is not None and args['range'] is not None:
        raise ValueError("Only one of the following arguments can be passed: --init_time, --range")
    elif args['init_time'] is None and args['range'] is None:
        raise ValueError("One of the following arguments must be passed: --init_time, --range")

    for file, local_filename in zip(files, local_filenames):

        timestring = local_filename.split('_')[1]
        year, month = timestring[:4], timestring[4:6]
        monthly_directory = '%s/%s%s' % (args['netcdf_outdir'], year, month)  # Directory for the netCDF files for the given days

        ### If the directory does not exist, check to see if the file link is valid. If the file link is NOT valid, then the directory will not be created since it will be empty. ###
        if not os.path.isdir(monthly_directory):
            if requests.head(file).status_code == requests.codes.ok:
                os.mkdir(monthly_directory)

        full_file_path = f'{monthly_directory}/{local_filename}'

        if not os.path.isfile(full_file_path):
            try:
                wget.download(file, out=full_file_path, bar=bar)
            except urllib.error.HTTPError:
                print(f"Error downloading {file}")
        else:
            print(f"{full_file_path} already exists, skipping file....")
