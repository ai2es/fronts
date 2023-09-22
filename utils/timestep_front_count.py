"""
Tool for counting the number of fronts in each timestep so that tensorflow datasets can be quickly generated. This script
effectively prevents empty timesteps from being analyzed by 'convert_netcdf_to_tf.py', saving potentially large amounts
of time when generating tensorflow datasets.

A dictionary containing front counts for timesteps across a given domain will be saved to a pickle file in a directory for
tensorflow datasets.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.6.13
"""
import os
import sys
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # this line allows us to import scripts outside the current directory
import file_manager as fm
from utils.settings import DEFAULT_DOMAIN_INDICES
import argparse
import numpy as np
import xarray as xr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fronts_netcdf_indir', type=str, required=True,
        help="Input directory for the netCDF files containing frontal boundary data.")
    parser.add_argument('--tf_outdir', type=str, required=True,
        help="Output directory for future tensorflow datasets. This is where the pickle file containing frontal counts will "
             "also be saved.")
    parser.add_argument('--domain', type=str, default='conus', help='Domain from which to pull the images.')

    args = vars(parser.parse_args())

    front_files_obj = fm.DataFileLoader(args['fronts_netcdf_indir'], data_file_type='fronts-netcdf')
    front_files = front_files_obj.front_files

    isel_kwargs = {'longitude': slice(DEFAULT_DOMAIN_INDICES[args['domain']][0], DEFAULT_DOMAIN_INDICES[args['domain']][1]),
                   'latitude': slice(DEFAULT_DOMAIN_INDICES[args['domain']][2], DEFAULT_DOMAIN_INDICES[args['domain']][3])}

    fieldnames = ['File', 'CF', 'CF-F', 'CF-D', 'WF', 'WF-F', 'WF-D', 'SF', 'SF-F', 'SF-D', 'OF', 'OF-F', 'OF-D', 'INST',
                  'TROF', 'TT', 'DL']

    front_count_csv_file = '%s/timestep_front_counts_%s.csv' % (args['tf_outdir'], args['domain'])

    with open('%s/timestep_front_counts.csv' % args['tf_outdir'], 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(fieldnames)

        for file_no, front_file in enumerate(front_files):
            print(front_file, end='\r')
            front_dataset = xr.open_dataset(front_file, engine='netcdf4').isel(**isel_kwargs).expand_dims('time', axis=0).astype('float16')
            front_bins = np.bincount(front_dataset['identifier'].values.astype('int64').flatten(), minlength=17)[1:]  # counts for each front type ('no front' type removed)

            row = [os.path.basename(front_file), *front_bins]

            csvwriter.writerow(row)
