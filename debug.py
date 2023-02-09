"""
Script for running debug commands

Code written by: Andrew Justin (andrewjustinwx@gmail.com)

Last updated: 10/1/2022 7:30 PM CT
"""

import argparse
import os.path

from errors import check_arguments
import utils.debug.era5, utils.debug.fronts, utils.debug.tf, utils.debug.model_data

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
    parser.add_argument('--find_missing_era5_files', action='store_true', help='Analyze a directory to find missing ERA5 files')
    parser.add_argument('--find_missing_front_files', action='store_true', help='Analyze a directory to find missing front object files')
    parser.add_argument('--find_missing_tf_datasets', action='store_true', help='Analyze a directory to find missing tensorflow datasets')
    parser.add_argument('--check_era5_variables', action='store_true', help='Check variables in era5 dataset')
    parser.add_argument('--check_for_corrupt_era5_files', action='store_true', help='Check variables in era5 dataset')
    parser.add_argument('--model_dir', type=str, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, help='Model number.')
    parser.add_argument('--netcdf_indir', type=str, help='Main directory for netcdf files')
    parser.add_argument('--tf_indir', type=str, help='Main directory for tensorflow datasets')
    parser.add_argument('--timestep', type=int, nargs=4, help='Timestep for ERA5 data (year, month, day, hour)')
    parser.add_argument('--fronts_xml_indir', type=str, help='Main directory for the xml files containing frontal objects.')
    parser.add_argument('--variables_netcdf_indir', type=str, help='Main directory for the netcdf files containing variable data.')
    parser.add_argument('--random_variables', type=str, nargs="+", default=None, help="Variables to randomize when generating predictions.")
    parser.add_argument('--variables_data_source', type=str, default='era5', help='Data source for variables')

    args = parser.parse_args()
    provided_arguments = vars(args)

    if args.find_missing_statistics:
        required_arguments = ['model_dir', 'model_number', 'domain', 'domain_images', 'domain_trim', 'variables_data_source',
                              'variables_netcdf_indir', 'netcdf_indir', 'fronts_xml_indir', 'dataset']
        check_arguments(provided_arguments, required_arguments)
        utils.debug.model_data.find_missing_statistics(args.model_dir, args.model_number, args.domain, args.domain_images, args.domain_trim,
            args.variables_data_source, args.variables_netcdf_indir, args.netcdf_indir, args.fronts_xml_indir, dataset=args.dataset)

    if args.check_era5_variables:
        required_arguments = ['netcdf_indir', 'timestep']
        check_arguments(provided_arguments, required_arguments)
        utils.debug.era5.check_era5_variables(args.netcdf_indir, args.timestep)

    if args.check_for_corrupt_era5_files:
        required_arguments = ['netcdf_indir']
        check_arguments(provided_arguments, required_arguments)
        utils.debug.era5.check_for_corrupt_era5_files(args.netcdf_indir)

    if args.find_missing_era5_files:
        required_arguments = ['netcdf_indir']
        check_arguments(provided_arguments, required_arguments)
        utils.debug.era5.find_missing_era5_data(args.netcdf_indir)

    if args.find_missing_front_files:
        required_arguments = ['netcdf_indir']
        check_arguments(provided_arguments, required_arguments)
        utils.debug.fronts.find_missing_fronts_data(args.netcdf_indir)

    if args.find_missing_tf_datasets:
        required_arguments = ['tf_indir']
        assert os.path.isdir(args.tf_indir)
        check_arguments(provided_arguments, required_arguments)
        utils.debug.tf.find_missing_tf_datasets(args.tf_indir)