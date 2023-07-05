"""
Convert netCDF files containing ERA5 and frontal boundary data into tensorflow datasets for model training.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 7/1/2023 8:06 PM CT
"""
import argparse
from glob import glob
import itertools
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
from utils import data_utils, settings
import xarray as xr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year_and_month', type=int, nargs=2, required=True,
        help="Year and month for the netcdf data to be converted to tensorflow datasets.")
    parser.add_argument('--era5_netcdf_indir', type=str, required=True,
        help="Input directory for the netCDF files containing ERA5 data.")
    parser.add_argument('--fronts_netcdf_indir', type=str, required=True,
        help="Input directory for the netCDF files containing frontal boundary data.")
    parser.add_argument('--tf_outdir', type=str, required=True,
        help="Output directory for the generated tensorflow datasets.")
    parser.add_argument('--front_types', type=str, nargs='+', required=True,
        help="Code(s) for the front types that will be generated in the tensorflow datasets. Refer to documentation in 'utils.data_utils.reformat_fronts' "
             "for more information on these codes.")
    parser.add_argument('--variables', type=str, nargs='+', help='ERA5 variables to select')
    parser.add_argument('--pressure_levels', type=str, nargs='+', help='ERA5 pressure levels to select')
    parser.add_argument('--num_dims', type=int, nargs=2, default=[3, 3], help='Number of dimensions in the ERA5 and front object images, repsectively.')
    parser.add_argument('--domain', type=str, default='conus', help='Domain from which to pull the images.')
    parser.add_argument('--evaluation_dataset', action='store_true',
        help=''' 
            Boolean flag that determines if the dataset being generated will be used for evaluating a model.
            If this flag is True, all of the following keyword arguments will be set and any values provided to 'netcdf_to_tf'
                by the user will be overriden:
                * num_dims = (_, 2)  <=== NOTE: The first value of this tuple will NOT be overriden.
                * images = (1, 1)
                * image_size will be set to the size of the domain.
                * keep_fraction will have no effect
                * shuffle_timesteps = False
                * shuffle_images = False
                * noise_fraction = 0.0
                * rotate_chance = 0.0
                * flip_chance_lon = 0.0
                * flip_chance_lat = 0.0
            ''')
    parser.add_argument('--images', type=int, nargs=2, default=[9, 1],
        help='Number of ERA5/front images along the longitude and latitude dimensions to generate for each timestep. The product of the 2 integers '
             'will be the total number of images generated per timestep.')
    parser.add_argument('--image_size', type=int, nargs=2, default=[128, 128], help='Size of the longitude and latitude dimensions of the images.')
    parser.add_argument('--shuffle_timesteps', action='store_true',
        help='Shuffle the timesteps when generating the dataset. This is particularly useful when generating very large '
             'datasets that cannot be shuffled on the fly during training.')
    parser.add_argument('--shuffle_images', action='store_true',
        help='Shuffle the order of the images in each timestep. This does NOT shuffle the entire dataset for the provided '
             'month, but rather only the images in each respective timestep. This is particularly useful when generating '
             'very large datasets that cannot be shuffled on the fly during training.')
    parser.add_argument('--add_previous_fronts', type=str, nargs='+',
        help='Optional front types from previous timesteps to include as predictors. If the dataset is over conus, the fronts '
             'will be pulled from the last 3-hour timestep. If the dataset is over the full domain, the fronts will be pulled '
             'from the last 6-hour timestep.')
    parser.add_argument('--front_dilation', type=int, default=0, help='Number of pixels to expand the fronts by in all directions.')
    parser.add_argument('--keep_fraction', type=float, default=0.0,
        help='The fraction of timesteps WITHOUT all necessary front types that will be retained in the dataset. Can be any float 0 <= x <= 1.')
    parser.add_argument('--noise_fraction', type=float, default=0.0,
        help='The fraction of pixels in each image that will contain noise. Can be any float 0 <= x < 1.')
    parser.add_argument('--rotate_chance', type=float, default=0.0,
        help='The probability that the current image will be rotated (in any direction, up to 270 degrees). Can be any float 0 <= x <= 1.')
    parser.add_argument('--flip_chance_lon', type=float, default=0.0,
        help='The probability that the current image will have its longitude dimension reversed. Can be any float 0 <= x <= 1.')
    parser.add_argument('--flip_chance_lat', type=float, default=0.0,
        help='The probability that the current image will have its latitude dimension reversed. Can be any float 0 <= x <= 1.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the contents of any existing ERA5 and fronts data.')
    parser.add_argument('--status_printout', action='store_true', help='Print out the progress of the dataset generation.')
    parser.add_argument('--gpu_device', type=int, nargs='+', help='GPU device numbers.')
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth on the GPU')

    args = vars(parser.parse_args())

    if args['gpu_device'] is not None:
        gpus = tf.config.list_physical_devices(device_type='GPU')
        tf.config.set_visible_devices(devices=[gpus[gpu] for gpu in args['gpu_device']], device_type='GPU')

        # Allow for memory growth on the GPU. This will only use the GPU memory that is required rather than allocating all of the GPU's memory.
        if args['memory_growth']:
            tf.config.experimental.set_memory_growth(device=[gpus[gpu] for gpu in args['gpu_device']][0], enable=True)

    year, month = args['year_and_month'][0], args['year_and_month'][1]

    tf_dataset_folder_era5 = f'%s/era5_%d%02d_tf' % (args['tf_outdir'], year, month)
    tf_dataset_folder_fronts = f"%s/fronts_{'_'.join(front_type for front_type in args['front_types'])}_%d%02d_tf" % (args['tf_outdir'], year, month)

    if os.path.isdir(tf_dataset_folder_era5) or os.path.isdir(tf_dataset_folder_fronts):
        if args['overwrite']:
            print("WARNING: Tensorflow dataset(s) already exist for the provided year and month and will be overwritten.")
        else:
            raise FileExistsError("Tensorflow dataset(s) already exist for the provided year and month. If you would like to "
                                  "overwrite the existing datasets, pass the --overwrite flag into the command line.")

    if not os.path.isdir(args['tf_outdir']):
        try:
            os.mkdir(args['tf_outdir'])
        except FileExistsError:  # When running in parallel, sometimes multiple instances will try to create this directory at once, resulting in a FileExistsError
            pass

    dataset_props_file = '%s/dataset_properties.pkl' % args['tf_outdir']

    if not os.path.isfile(dataset_props_file):
        """
        Save critical dataset information to a pickle file so it can be referenced later when generating data for other months.
        """

        if args['evaluation_dataset']:
            """
            Override all keyword arguments so the dataset will be prepared for model evaluation.
            """
            print("WARNING: This dataset will be used for model evaluation, so the following arguments will be set and "
                  "any provided values for these arguments will be overriden:")
            args['num_dims'][1] = 2
            args['num_dims'] = tuple(args['num_dims'])
            args['images'] = (1, 1)
            args['image_size'] = (settings.DEFAULT_DOMAIN_INDICES[args['domain']][1] - settings.DEFAULT_DOMAIN_INDICES[args['domain']][0],
                                  settings.DEFAULT_DOMAIN_INDICES[args['domain']][3] - settings.DEFAULT_DOMAIN_INDICES[args['domain']][2])
            args['shuffle_timesteps'] = False
            args['shuffle_images'] = False
            args['noise_fraction'] = 0.0
            args['rotate_chance'] = 0.0
            args['flip_chance_lon'] = 0.0
            args['flip_chance_lat'] = 0.0

            print("num_dims[1] = 2\n"
                  f"images = {args['images']}\n"
                  f"image_size = {args['image_size']}\n"
                  f"shuffle_timesteps = False\n"
                  f"shuffle_images = False\n"
                  f"noise_fraction = 0.0\n"
                  f"rotate_chance = 0.0\n"
                  f"flip_chance_lon = 0.0\n"
                  f"flip_chance_lat = 0.0\n")

        dataset_props = dict({})
        dataset_props['normalization_parameters'] = data_utils.normalization_parameters
        for key in sorted(['front_types', 'variables', 'pressure_levels', 'num_dims', 'images', 'image_size', 'front_dilation',
                    'noise_fraction', 'rotate_chance', 'flip_chance_lon', 'flip_chance_lat', 'shuffle_images', 'shuffle_timesteps',
                    'domain', 'evaluation_dataset', 'add_previous_fronts', 'keep_fraction']):
            dataset_props[key] = args[key]

        with open(dataset_props_file, 'wb') as f:
            pickle.dump(dataset_props, f)

        with open('%s/dataset_properties.txt' % args['tf_outdir'], 'w') as f:
            for key in sorted(dataset_props.keys()):
                f.write(f"{key}: {dataset_props[key]}\n")

    else:

        print("WARNING: Dataset properties file was found in %s. The following settings will be used from the file." % args['tf_outdir'])
        dataset_props = pd.read_pickle(dataset_props_file)

        for key in sorted(['front_types', 'variables', 'pressure_levels', 'num_dims', 'images', 'image_size', 'front_dilation',
                           'noise_fraction', 'rotate_chance', 'flip_chance_lon', 'flip_chance_lat', 'shuffle_images', 'shuffle_timesteps',
                           'domain', 'evaluation_dataset', 'add_previous_fronts', 'keep_fraction']):
            args[key] = dataset_props[key]
            print(f"%s: {args[key]}" % key)

    all_variables = ['T', 'Td', 'sp_z', 'u', 'v', 'theta_w', 'r', 'RH', 'Tv', 'Tw', 'theta_e', 'q', 'theta', 'theta_v']
    all_pressure_levels = ['surface', '1000', '950', '900', '850']

    era5_monthly_directory = '%s/%d%02d' % (args['era5_netcdf_indir'], year, month)
    fronts_monthly_directory = '%s/%d%02d' % (args['fronts_netcdf_indir'], year, month)

    era5_netcdf_files = sorted(glob('%s/era5_%d%02d*_full.nc' % (era5_monthly_directory, year, month)))
    fronts_netcdf_files = sorted(glob('%s/FrontObjects_%d%02d*_full.nc' % (fronts_monthly_directory, year, month)))

    ### Grab front files from previous timesteps so previous fronts can be used as predictors ###
    if args['add_previous_fronts'] is not None:
        files_to_remove = []  # era5 and front files that will be removed from the dataset
        previous_fronts_netcdf_files = []
        for file in fronts_netcdf_files:
            current_timestep = np.datetime64(f'{file[-18:-14]}-{file[-14:-12]}-{file[-12:-10]}T{file[-10:-8]}')
            previous_timestep = (current_timestep - np.timedelta64(3, "h")).astype(object)
            prev_year, prev_month, prev_day, prev_hour = previous_timestep.year, previous_timestep.month, previous_timestep.day, previous_timestep.hour
            previous_fronts_file = '%s/%d%02d/FrontObjects_%d%02d%02d%02d_full.nc' % (args['fronts_netcdf_indir'], prev_year, prev_month, prev_year, prev_month, prev_day, prev_hour)
            if os.path.isfile(previous_fronts_file):
                previous_fronts_netcdf_files.append(previous_fronts_file)  # Add the previous fronts to the dataset
            else:
                files_to_remove.append(file)

        ### Remove files from the dataset if previous fronts are not available ###
        if len(files_to_remove) > 0:
            for file in files_to_remove:
                index_to_pop = fronts_netcdf_files.index(file)
                era5_netcdf_files.pop(index_to_pop), fronts_netcdf_files.pop(index_to_pop)

    if args['shuffle_timesteps']:
        zipped_list = list(zip(era5_netcdf_files, fronts_netcdf_files))
        np.random.shuffle(zipped_list)
        era5_netcdf_files, fronts_netcdf_files = zip(*zipped_list)

    files_match_flag = all(era5_file[-18:] == fronts_file[-18:] for era5_file, fronts_file in zip(era5_netcdf_files, fronts_netcdf_files))

    isel_kwargs = {'longitude': slice(settings.DEFAULT_DOMAIN_INDICES[args['domain']][0], settings.DEFAULT_DOMAIN_INDICES[args['domain']][1]),
                   'latitude': slice(settings.DEFAULT_DOMAIN_INDICES[args['domain']][2], settings.DEFAULT_DOMAIN_INDICES[args['domain']][3])}

    if not files_match_flag:
        raise OSError("ERA5/fronts files do not match")

    # Normalization parameters are not available for theta_v and theta, so we will only select the following variables
    if args['variables'] is None:
        variables_to_use = all_variables
    else:
        variables_to_use = args['variables']

    if args['pressure_levels'] is None:
        args['pressure_levels'] = all_pressure_levels
    else:
        args['pressure_levels'] = [lvl for lvl in all_pressure_levels if lvl in args['pressure_levels']]

    num_timesteps = len(era5_netcdf_files)
    timesteps_kept = 0
    timesteps_discarded = 0

    for timestep_no in range(num_timesteps):

        front_dataset = xr.open_dataset(fronts_netcdf_files[timestep_no], engine='netcdf4').isel(**isel_kwargs).expand_dims('time', axis=0).astype('float16')

        ### Reformat the fronts in the current timestep ###
        if args['front_types'] is not None:
            front_dataset = data_utils.reformat_fronts(front_dataset, args['front_types'])
            num_front_types = front_dataset.attrs['num_types'] + 1
        else:
            num_front_types = 17

        if args['front_dilation'] > 0:
            front_dataset = data_utils.expand_fronts(front_dataset, iterations=args['front_dilation'])  # expand the front labels

        keep_timestep = np.random.random() <= args['keep_fraction']  # boolean flag for keeping timesteps without all front types

        front_dataset = front_dataset.isel(time=0).to_array().transpose('longitude', 'latitude', 'variable')
        front_bins = np.bincount(front_dataset.values.astype('int64').flatten(), minlength=num_front_types)  # counts for each front type
        all_fronts_present = all([front_count > 0 for front_count in front_bins]) > 0  # boolean flag that says if all front types are present in the current timestep

        if all_fronts_present or keep_timestep or args['evaluation_dataset']:

            era5_dataset = xr.open_dataset(era5_netcdf_files[timestep_no], engine='netcdf4')[variables_to_use].isel(**isel_kwargs).sel(pressure_level=args['pressure_levels']).transpose('time', 'longitude', 'latitude', 'pressure_level').astype('float16')
            era5_dataset = data_utils.normalize_variables(era5_dataset).isel(time=0).transpose('longitude', 'latitude', 'pressure_level').astype('float16')

            ### Reformat the fronts from the previous timestep ###
            if args['add_previous_fronts'] is not None:
                previous_front_dataset = xr.open_dataset(previous_fronts_netcdf_files[timestep_no], engine='netcdf4').isel(**isel_kwargs).astype('float16')
                previous_front_dataset = data_utils.reformat_fronts(previous_front_dataset, args['add_previous_fronts'])

                if args['front_dilation'] > 0:
                    previous_front_dataset = data_utils.expand_fronts(previous_front_dataset, iterations=args['front_dilation'])

                previous_front_dataset = previous_front_dataset.transpose('longitude', 'latitude')

                previous_fronts = np.zeros([settings.DEFAULT_DOMAIN_INDICES[args['domain']][1] - settings.DEFAULT_DOMAIN_INDICES[args['domain']][0],
                                            settings.DEFAULT_DOMAIN_INDICES[args['domain']][3] - settings.DEFAULT_DOMAIN_INDICES[args['domain']][2],
                                            len(args['pressure_levels'])], dtype=np.float16)

                for front_type_no, previous_front_type in enumerate(args['add_previous_fronts']):
                    previous_fronts[..., 0] = np.where(previous_front_dataset['identifier'].values == front_type_no + 1, 1, 0)  # Place previous front labels at the surface level
                    era5_dataset[previous_front_type] = (('longitude', 'latitude', 'pressure_level'), previous_fronts)  # Add previous fronts to the predictor dataset

            era5_dataset = era5_dataset.to_array().transpose('longitude', 'latitude', 'pressure_level', 'variable')

            try:
                start_indices_lon = np.arange(0, settings.DEFAULT_DOMAIN_INDICES[args['domain']][1] - settings.DEFAULT_DOMAIN_INDICES[args['domain']][0] - args['image_size'][0] + 1,
                                              int((settings.DEFAULT_DOMAIN_INDICES[args['domain']][1] - settings.DEFAULT_DOMAIN_INDICES[args['domain']][0] - args['image_size'][0]) / (args['images'][0] - 1)))
            except ZeroDivisionError:
                start_indices_lon = np.zeros([args['images'][0]], dtype=int)

            try:
                start_indices_lat = np.arange(0, settings.DEFAULT_DOMAIN_INDICES[args['domain']][3] - settings.DEFAULT_DOMAIN_INDICES[args['domain']][2] - args['image_size'][1] + 1,
                                              int((settings.DEFAULT_DOMAIN_INDICES[args['domain']][3] - settings.DEFAULT_DOMAIN_INDICES[args['domain']][2] - args['image_size'][1]) / (args['images'][1] - 1)))
            except ZeroDivisionError:
                start_indices_lat = np.zeros([args['images'][1]], dtype=int)

            image_order = list(itertools.product(start_indices_lon, start_indices_lat))  # Every possible combination of longitude and latitude starting points
            if args['shuffle_images']:
                np.random.shuffle(image_order)

            for image_start_indices in image_order:

                start_index_lon = image_start_indices[0]
                end_index_lon = start_index_lon + args['image_size'][0]
                start_index_lat = image_start_indices[1]
                end_index_lat = start_index_lat + args['image_size'][1]

                # boolean flags for rotating and flipping images
                rotate_image = np.random.random() <= args['rotate_chance']
                flip_lon = np.random.random() <= args['flip_chance_lon']
                flip_lat = np.random.random() <= args['flip_chance_lat']

                if rotate_image:
                    rotation_direction = np.random.randint(0, 2)  # 0 = clockwise, 1 = counter-clockwise
                    num_rotations = np.random.randint(1, 4)  # n * 90 degrees

                era5_tensor = tf.convert_to_tensor(era5_dataset[start_index_lon:end_index_lon, start_index_lat:end_index_lat, :, :], dtype=tf.float16)
                if flip_lon:
                    era5_tensor = tf.reverse(era5_tensor, axis=[0])  # Reverse values along the longitude dimension
                if flip_lat:
                    era5_tensor = tf.reverse(era5_tensor, axis=[1])  # Reverse values along the latitude dimension
                if rotate_image:
                    for rotation in range(num_rotations):
                        era5_tensor = tf.reverse(tf.transpose(era5_tensor, perm=[1, 0, 2, 3]), axis=[rotation_direction])  # Rotate image 90 degrees

                    if args['noise_fraction'] > 0:
                        ### Add noise to image ###
                        random_values = tf.random.uniform(shape=era5_tensor.shape)
                        era5_tensor = tf.where(random_values < args['noise_fraction'] / 2, 0.0, era5_tensor)  # add 0s to image
                        era5_tensor = tf.where(random_values > 1.0 - (args['noise_fraction'] / 2), 1.0, era5_tensor)  # add 1s to image

                    if args['num_dims'][0] == 2:
                        era5_tensor_shape_3d = era5_tensor.shape
                        # Combine pressure level and variables dimensions, making the images 2D (excluding the final dimension)
                        era5_tensor = tf.reshape(era5_tensor, [era5_tensor_shape_3d[0], era5_tensor_shape_3d[1], era5_tensor_shape_3d[2] * era5_tensor_shape_3d[3]])

                era5_tensor_for_timestep = tf.data.Dataset.from_tensors(era5_tensor)
                if 'era5_tensors_for_month' not in locals():
                    era5_tensors_for_month = era5_tensor_for_timestep
                else:
                    era5_tensors_for_month = era5_tensors_for_month.concatenate(era5_tensor_for_timestep)

                front_tensor = tf.convert_to_tensor(front_dataset[start_index_lon:end_index_lon, start_index_lat:end_index_lat, :], dtype=tf.int32)

                if flip_lon:
                    front_tensor = tf.reverse(front_tensor, axis=[0])  # Reverse values along the longitude dimension
                if flip_lat:
                    front_tensor = tf.reverse(front_tensor, axis=[1])  # Reverse values along the latitude dimension
                if rotate_image:
                    for rotation in range(num_rotations):
                        front_tensor = tf.reverse(tf.transpose(front_tensor, perm=[1, 0, 2]), axis=[rotation_direction])  # Rotate image 90 degrees

                if args['num_dims'][1] == 3:
                    # Make the front object images 3D, with the size of the 3rd dimension equal to the number of pressure levels
                    front_tensor = tf.tile(front_tensor, (1, 1, len(args['pressure_levels'])))
                else:
                    front_tensor = front_tensor[:, :, 0]

                front_tensor = tf.cast(tf.one_hot(front_tensor, num_front_types), tf.float16)  # One-hot encode the labels
                front_tensor_for_timestep = tf.data.Dataset.from_tensors(front_tensor)
                if 'front_tensors_for_month' not in locals():
                    front_tensors_for_month = front_tensor_for_timestep
                else:
                    front_tensors_for_month = front_tensors_for_month.concatenate(front_tensor_for_timestep)

            timesteps_kept += 1
        else:
            timesteps_discarded += 1

        if args['status_printout']:
            print("Timesteps complete: %d/%d  (Retained/discarded: %d/%d)" % (timesteps_kept + timesteps_discarded, num_timesteps, timesteps_kept, timesteps_discarded), end='\r')

    print("Timesteps complete: %d/%d  (Retained/discarded: %d/%d)" % (timesteps_kept + timesteps_discarded, num_timesteps, timesteps_kept, timesteps_discarded))

    if args['overwrite']:
        if os.path.isdir(tf_dataset_folder_era5):
            os.rmdir(tf_dataset_folder_era5)
        if os.path.isdir(tf_dataset_folder_fronts):
            os.rmdir(tf_dataset_folder_fronts)

    try:
        tf.data.Dataset.save(era5_tensors_for_month, path=tf_dataset_folder_era5)
        tf.data.Dataset.save(front_tensors_for_month, path=tf_dataset_folder_fronts)
        print("Tensorflow datasets for %d-%02d saved to %s." % (year, month, args['tf_outdir']))
    except NameError:
        print("No images could be retained with the provided arguments.")
