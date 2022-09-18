"""
Script for running debug commands

Code written by: Andrew Justin (andrewjustinwx@gmail.com)

Last updated: 9/18/2022 4:34 PM CT
"""

import argparse
from errors import check_arguments
from utils.debug import model_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset for which to make predictions if prediction_method is 'random' or 'all'. Options are:"
                                                    "'training', 'validation', 'test'")
    parser.add_argument('--domain', type=str, help='Domain of the data.')
    parser.add_argument('--domain_images', type=int, nargs=2, help='Number of images for each dimension the final stitched map for predictions: lon, lat')
    parser.add_argument('--domain_trim', type=int, nargs=2, default=[0, 0],
                        help='Number of pixels to trim the images by along each dimension for stitching before taking the '
                             'maximum across overlapping pixels.')
    parser.add_argument('--find_missing_statistics', action='store_true', help='Analyze a directory of statistics to find missing data')
    parser.add_argument('--model_dir', type=str, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, help='Model number.')
    parser.add_argument('--fronts_netcdf_indir', type=str, help='Main directory for the netcdf files containing frontal objects.')
    parser.add_argument('--variables_netcdf_indir', type=str, help='Main directory for the netcdf files containing variable data.')
    parser.add_argument('--random_variables', type=str, nargs="+", default=None, help="Variables to randomize when generating predictions.")
    parser.add_argument('--variables_data_source', type=str, default='era5', help='Data source for variables')

    args = parser.parse_args()
    provided_arguments = vars(args)

    if args.find_missing_statistics:
        required_arguments = ['model_dir', 'model_number', 'domain', 'domain_images', 'domain_trim', 'variables_data_source',
                              'variables_netcdf_indir', 'fronts_netcdf_indir', 'dataset']
        check_arguments(provided_arguments, required_arguments)
        model_data.find_missing_statistics(args.model_dir, args.model_number, args.domain, args.domain_images, args.domain_trim,
                                           args.variables_data_source, args.variables_netcdf_indir, args.fronts_netcdf_indir, dataset=args.dataset)
