"""
Convert netCDF files containing variable, satellite, and frontal boundary data into tensorflow datasets for model training.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.8.31
"""
import argparse
import itertools
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
import file_manager as fm
from utils import data_utils, misc
from datetime import datetime
import xarray as xr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variables_netcdf_indir', type=str, required=True,
        help="Input directory for the netCDF files containing variable data.")
    parser.add_argument('--fronts_netcdf_indir', type=str, required=True,
        help="Input directory for the netCDF files containing frontal boundary data.")
    parser.add_argument('--satellite_netcdf_indir', type=str, required=False,
        help="Input directory for the netCDF files containing GOES satellite data.")
    parser.add_argument('--tf_outdir', type=str, required=True,
        help="Output directory for the generated tensorflow datasets.")
    parser.add_argument('--year_and_month', type=int, nargs=2, required=True,
        help="Year and month for the netcdf data to be converted to tensorflow datasets.")
    parser.add_argument('--data_source', type=str, default='era5', help="Data source or model containing the variable data.")
    parser.add_argument('--front_types', type=str, nargs='+', required=True,
        help="Code(s) for the front types that will be generated in the tensorflow datasets. Refer to documentation in 'utils.data_utils.reformat_fronts' "
             "for more information on these codes.")
    parser.add_argument('--variables', type=str, nargs='+', required=True, help='Variables to select')
    parser.add_argument('--pressure_levels', type=str, nargs='+', help='Variables pressure levels to select')
    parser.add_argument('--num_dims', type=int, nargs=2, default=[3, 3], help='Number of dimensions in the variables and front object images, repsectively.')
    parser.add_argument('--domain', type=str, default='conus', help='Domain from which to pull the images.')
    parser.add_argument('--override_extent', type=float, nargs=4,
        help='Override the default domain extent by selecting a custom extent. [min lon, max lon, min lat, max lat]')
    parser.add_argument('--images', type=int, nargs=2, default=[9, 1],
        help='Number of variables/front images along the longitude and latitude dimensions to generate for each timestep. The product of the 2 integers '
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
    parser.add_argument('--timestep_fraction', type=float, default=0.0,
        help='The fraction of timesteps WITHOUT all necessary front types that will be retained in the dataset. Can be any float 0 <= x <= 1.')
    parser.add_argument('--image_fraction', type=float, default=1.0,
        help='The fraction of images WITHOUT all necessary front types in the selected timesteps that will be retained in the dataset. Can be any float 0 <= x <= 1. '
             'By default, all images are retained.')
    parser.add_argument('--noise_fraction', type=float, default=0.0,
        help='The fraction of pixels in each image that will contain noise. Can be any float 0 <= x < 1.')
    parser.add_argument('--flip_chance_lon', type=float, default=0.0,
        help='The probability that the current image will have its longitude dimension reversed. Can be any float 0 <= x <= 1.')
    parser.add_argument('--flip_chance_lat', type=float, default=0.0,
        help='The probability that the current image will have its latitude dimension reversed. Can be any float 0 <= x <= 1.')
    parser.add_argument('--verbose', action='store_true', help='Print out the progress of the dataset generation.')
    parser.add_argument('--gpu_device', type=int, nargs='+', help='GPU device numbers.')
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth on the GPU(s).')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 2**31 - 1),
        help="Seed for the random number generators. The same seed will be used for all months within a particular dataset.")

    args = vars(parser.parse_args())
    
    # configure GPU devices
    if args['gpu_device'] is not None:
        misc.initialize_gpus(args['gpu_device'], args['memory_growth'])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    year, month = args['year_and_month']
    
    """
    all_data_vars: Variables found in ERA5 reanalysis and NWP models
    all_sat_vars: Satellite variables
    """
    all_data_vars = ['T', 'Td', 'sp_z', 'u', 'v', 'theta_w', 'r', 'RH', 'Tv', 'Tw', 'theta_e', 'q', 'theta', 'theta_v']
    all_sat_vars = ['Tb',]
    
    all_pressure_levels = ['surface', '1000', '950', '900', '850'] if args['data_source'] == 'era5' else ['surface', '1013', '1000', '950', '900', '850', '700', '500']
    
    # check for invalid variables
    invalid_vars = [var for var in args['variables'] if var not in all_data_vars and var not in all_sat_vars]
    assert len(invalid_vars) == 0, 'Invalid variables (%d): %s' % (len(invalid_vars), ', '.join(invalid_vars))
    
    data_vars = [var for var in args['variables'] if var in all_data_vars]
    sat_vars = [var for var in args['variables'] if var in all_sat_vars]
    
    load_satellite = args['satellite_netcdf_indir'] is not None and len(sat_vars) > 0  # boolean flag that says if satellite data will be loaded

    os.makedirs(args['tf_outdir'], exist_ok=True)  # ensure that a folder exists for the dataset
    
    tf_dataset_folder_inputs = f'%s/inputs_%d%02d_tf' % (args['tf_outdir'], year, month)
    tf_dataset_folder_fronts = tf_dataset_folder_inputs.replace('inputs', 'fronts')
    
    if os.path.isdir(tf_dataset_folder_inputs) or os.path.isdir(tf_dataset_folder_fronts):
        raise FileExistsError("Tensorflow dataset(s) already exist for the provided year and month.")

    dataset_props_file = '%s/dataset_properties.pkl' % args['tf_outdir']

    if not os.path.isfile(dataset_props_file):
        """
        Save critical dataset information to a pickle file so it can be referenced later when generating data for other months.
        """
        print("Setting seed: %d" % args["seed"])

        dataset_props = dict({})
        dataset_props['normalization_parameters'] = data_utils.normalization_parameters
        for key in sorted(['front_types', 'variables', 'pressure_levels', 'num_dims', 'images', 'image_size', 'front_dilation',
                    'noise_fraction', 'flip_chance_lon', 'flip_chance_lat', 'shuffle_images', 'shuffle_timesteps',
                    'domain', 'add_previous_fronts', 'timestep_fraction', 'image_fraction', 'override_extent', 'seed']):
            dataset_props[key] = args[key]

        with open(dataset_props_file, 'wb') as f:
            pickle.dump(dataset_props, f)

        with open('%s/dataset_properties.txt' % args['tf_outdir'], 'w') as f:
            for key in sorted(dataset_props.keys()):
                f.write(f"{key}: {dataset_props[key]}\n")
            f.write(f"\n\n\nFile generated at {datetime.utcnow()} UTC\n")
            f.write(f"convert_netcdf_to_tf.py script version: 2024.8.31")

    else:

        print("WARNING: Dataset properties file was found in %s. The following settings will be used from the file." % args['tf_outdir'])
        dataset_props = pd.read_pickle(dataset_props_file)

        for key in sorted(['front_types', 'variables', 'pressure_levels', 'num_dims', 'images', 'image_size', 'front_dilation',
                           'noise_fraction', 'flip_chance_lon', 'flip_chance_lat', 'shuffle_images', 'shuffle_timesteps',
                           'domain', 'add_previous_fronts', 'timestep_fraction', 'image_fraction', 'override_extent']):
            args[key] = dataset_props[key]
            print(f"%s: {args[key]}" % key)

        if "seed" in list(dataset_props.keys()):
            args["seed"] = dataset_props["seed"]
            print(f"%s: {args['seed']}" % "seed")

    # set the seeds
    tf.random.set_seed(args["seed"])
    np.random.seed(args["seed"])
    
    file_obj = fm.DataFileLoader(args['variables_netcdf_indir'], 'era5', 'netcdf', years=year, months=month, domains='full')
    file_obj.add_file_list(args['fronts_netcdf_indir'], 'fronts')
    
    ### add satellite data files ###
    if load_satellite:
        file_obj.add_file_list(args['satellite_netcdf_indir'], 'MERGIR')
        variables_netcdf_files, fronts_netcdf_files, satellite_netcdf_files = file_obj.files
    else:
        variables_netcdf_files, fronts_netcdf_files = file_obj.files
    
    ### Grab front files from previous timesteps so previous fronts can be used as predictors ###
    if args['add_previous_fronts'] is not None:
        files_to_remove = []  # variables and front files that will be removed from the dataset
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
                variables_netcdf_files.pop(index_to_pop), fronts_netcdf_files.pop(index_to_pop)
                if load_satellite:
                    satellite_netcdf_files.pop(index_to_pop)
    
    synoptic_only = True if args['domain'] not in ['conus', 'hrrr'] else False
    if synoptic_only:
        synoptic_ind = [variables_netcdf_files.index(file) for file in variables_netcdf_files if any(['%02d_' % hr in file for hr in [0, 6, 12, 18]])]
        variables_netcdf_files = list([variables_netcdf_files[i] for i in synoptic_ind])
        fronts_netcdf_files = list([fronts_netcdf_files[i] for i in synoptic_ind])
        if load_satellite:
            satellite_netcdf_files = list([satellite_netcdf_files[i] for i in synoptic_ind])
        
    if args['shuffle_timesteps']:
        zipped_list = list(zip(variables_netcdf_files, fronts_netcdf_files)) if not load_satellite else list(zip(variables_netcdf_files, fronts_netcdf_files, satellite_netcdf_files))
        np.random.shuffle(zipped_list)
        if load_satellite:
            variables_netcdf_files, fronts_netcdf_files, satellite_netcdf_files = zip(*zipped_list)
        else:
            variables_netcdf_files, fronts_netcdf_files = zip(*zipped_list)

    if args["domain"] in ["conus", "full"]:
        if args['override_extent'] is None:
            sel_kwargs = {'longitude': slice(data_utils.DOMAIN_EXTENTS[args['domain']][0], data_utils.DOMAIN_EXTENTS[args['domain']][1]),
                          'latitude': slice(data_utils.DOMAIN_EXTENTS[args['domain']][3], data_utils.DOMAIN_EXTENTS[args['domain']][2])}
        else:
            sel_kwargs = {'longitude': slice(args['override_extent'][0], args['override_extent'][1]),
                          'latitude': slice(args['override_extent'][3], args['override_extent'][2])}
    else:
        sel_kwargs = {}

    args['pressure_levels'] = all_pressure_levels if args['pressure_levels'] is None else [lvl for lvl in all_pressure_levels if lvl in args['pressure_levels']]

    num_timesteps = len(variables_netcdf_files)
    images_kept = 0
    images_discarded = 0
    timesteps_kept = 0
    timesteps_discarded = 0

    isel_kwargs = dict(forecast_hour=0) if args['data_source'] != 'era5' else dict()

    for timestep_no in range(num_timesteps):

        keep_timestep = np.random.random() <= args['timestep_fraction']  # boolean flag for keeping timesteps without all front types

        ### open front dataset ###
        front_dataset = xr.open_dataset(fronts_netcdf_files[timestep_no], engine='netcdf4').isel(**isel_kwargs)
        if args["data_source"] not in ["hrrr", "namnest-conus", "nam-12km"]:
            front_dataset = front_dataset.sel(**sel_kwargs).astype('float16')
            transpose_dims = ("longitude", "latitude")  # spatial dimensions that need to be transposed
        else:
            transpose_dims = ("x", "y")  # spatial dimensions that need to be transposed
        domain_size = (len(front_dataset[transpose_dims[0]]), len(front_dataset[transpose_dims[1]]))
        
        ### Reformat the fronts in the current timestep ###
        if args['front_types'] is not None:
            front_dataset = data_utils.reformat_fronts(front_dataset, args['front_types'])
            num_front_types = front_dataset.attrs['num_front_types'] + 1
        else:
            num_front_types = 16

        ### Expand the front labels ###
        if args['front_dilation'] > 0:
            front_dataset = data_utils.expand_fronts(front_dataset, iterations=args['front_dilation'])

        ### Check for all front types in the dataset ###
        front_dataset = front_dataset.isel(time=0) if 'time' in front_dataset.dims else front_dataset
        front_dataset = front_dataset.to_array().transpose(*transpose_dims, 'variable')
        timestep_front_bins = np.bincount(front_dataset.values.astype('int64').flatten(), minlength=num_front_types)  # counts for each front type
        all_fronts_in_timestep = all([front_count > 0 for front_count in timestep_front_bins]) > 0  # boolean flag that says if all front types are present in the current timestep

        if args['verbose']:
            print("%d-%02d Dataset progress (kept/discarded):  (%d/%d timesteps, %d/%d images)" % (year, month, timesteps_kept, timesteps_discarded, images_kept, images_discarded), end='\r')

        if all_fronts_in_timestep or keep_timestep:

            ### Open variables dataset ###
            variables_dataset = xr.open_dataset(variables_netcdf_files[timestep_no], engine='netcdf4')[data_vars].sel(pressure_level=args['pressure_levels'], **sel_kwargs).isel(**isel_kwargs).transpose('time', *transpose_dims, 'pressure_level').astype('float16')
            variables_dataset = variables_dataset.isel(time=0).transpose(*transpose_dims, 'pressure_level').astype('float16')
            
            ### open satellite data ###
            if load_satellite:
                satellite_dataset = xr.open_dataset(satellite_netcdf_files[timestep_no], engine='netcdf4')
                satellite_dataset = satellite_dataset.sel(**sel_kwargs).isel(time=0, **isel_kwargs).transpose(*transpose_dims)
            
            ### Reformat the fronts from the previous timestep ###
            if args['add_previous_fronts'] is not None:
                previous_front_dataset = xr.open_dataset(previous_fronts_netcdf_files[timestep_no], engine='netcdf4').sel(**sel_kwargs).isel(**isel_kwargs).astype('float16')
                previous_front_dataset = data_utils.reformat_fronts(previous_front_dataset, args['add_previous_fronts'])

                # expand fronts in the previous timestep
                if args['front_dilation'] > 0:
                    previous_front_dataset = data_utils.expand_fronts(previous_front_dataset, iterations=args['front_dilation'])
                previous_front_dataset = previous_front_dataset.transpose(*transpose_dims)  # transpose dimensions
                previous_fronts = np.zeros([len(previous_front_dataset[transpose_dims[0]].values),
                                            len(previous_front_dataset[transpose_dims[1]].values),
                                            len(args['pressure_levels'])], dtype=np.float16)

                for front_type_no, previous_front_type in enumerate(args['add_previous_fronts']):
                    previous_fronts[..., 0] = np.where(previous_front_dataset['identifier'].values == front_type_no + 1, 1, 0)  # Place previous front labels at the surface level
                    variables_dataset[previous_front_type] = ((*transpose_dims, 'pressure_level'), previous_fronts)  # Add previous fronts to the predictor dataset

            if args['images'][0] > 1 and domain_size[0] > args['image_size'][0] + args['images'][0]:
                start_indices_lon = np.linspace(0, domain_size[0] - args['image_size'][0], args['images'][0]).astype(int)
            else:
                start_indices_lon = np.zeros((args['images'][0], ), dtype=int)

            if args['images'][1] > 1 and domain_size[1] > args['image_size'][1] + args['images'][1]:
                start_indices_lat = np.linspace(0, domain_size[1] - args['image_size'][1], args['images'][1]).astype(int)
            else:
                start_indices_lat = np.zeros((args['images'][1], ), dtype=int)

            image_order = list(itertools.product(start_indices_lon, start_indices_lat))  # Every possible combination of longitude and latitude starting points

            if args['shuffle_images']:
                np.random.shuffle(image_order)

            images_to_keep = np.random.random(size=len(image_order)) <= args["image_fraction"]
            
            for i, image_start_indices in enumerate(image_order):

                if args['verbose']:
                    print("%d-%02d Dataset progress (kept/discarded):  (%d/%d timesteps, %d/%d images)" % (year, month, timesteps_kept, timesteps_discarded, images_kept, images_discarded), end='\r')

                start_index_lon = image_start_indices[0]
                end_index_lon = start_index_lon + args['image_size'][0]
                start_index_lat = image_start_indices[1]
                end_index_lat = start_index_lat + args['image_size'][1]

                front_image = front_dataset[start_index_lon:end_index_lon, start_index_lat:end_index_lat, :]
                image_front_bins = np.bincount(front_image.values.astype('int64').flatten(), minlength=num_front_types)  # counts for each front type
                all_fronts_in_image = all([front_count > 0 for front_count in image_front_bins]) > 0  # boolean flag that says if all front types are present in the current timestep

                if not all_fronts_in_image and not images_to_keep[i]:
                    images_discarded += 1
                    continue

                images_kept += 1

                new_variables_dataset = variables_dataset.copy()  # copy variables dataset to isolate it in memory

                # boolean flags for rotating and flipping images
                flip_lon = np.random.random() <= args['flip_chance_lon']
                flip_lat = np.random.random() <= args['flip_chance_lat']

                ### before flipping images, we will apply the necessary changes to the wind components to account for reflections ###
                if flip_lon and "u" in args["variables"]:
                    new_variables_dataset["u"] = -new_variables_dataset["u"]  # need to reverse u-wind component if flipping the longitude axis
                if flip_lat and "v" in args["variables"]:
                    new_variables_dataset["v"] = -new_variables_dataset["v"]  # need to reverse v-wind component if flipping the latitude axis

                new_variables_dataset = data_utils.normalize_variables(new_variables_dataset).to_array().transpose(*transpose_dims, 'pressure_level', 'variable')
                input_tensor = tf.convert_to_tensor(new_variables_dataset[start_index_lon:end_index_lon, start_index_lat:end_index_lat, :, :], dtype=tf.float16)
                
                random_values = tf.random.uniform(shape=input_tensor.shape)  # random noise values, will not necessarily be used
                
                ### rotate input variables image ###
                if flip_lon:
                    input_tensor = tf.reverse(input_tensor, axis=[0])  # Reverse values along the longitude dimension
                if flip_lat:
                    input_tensor = tf.reverse(input_tensor, axis=[1])  # Reverse values along the latitude dimension

                if args['noise_fraction'] > 0:
                    ### Add salt and pepper noise to images ###
                    input_tensor = tf.where(random_values < args['noise_fraction'] / 2, 0.0, input_tensor)  # add 0s to image
                    input_tensor = tf.where(random_values > 1.0 - (args['noise_fraction'] / 2), 1.0, input_tensor)  # add 1s to image
                
                if args['num_dims'][0] == 2:
                    ### Combine pressure level and variables dimensions, making the images 2D (excluding the final dimension) ###
                    input_tensor_shape_3d = input_tensor.shape
                    input_tensor = tf.reshape(input_tensor, [input_tensor_shape_3d[0], input_tensor_shape_3d[1], input_tensor_shape_3d[2] * input_tensor_shape_3d[3]])

                ### process satellite image ###
                if load_satellite:
                    new_satellite_dataset = satellite_dataset.copy()  # copy satellite dataset to isolate it in memory
                    new_satellite_dataset = data_utils.normalize_satellite(new_satellite_dataset).to_array().transpose(*transpose_dims, 'variable')
                    satellite_tensor = tf.convert_to_tensor(new_satellite_dataset[start_index_lon:end_index_lon, start_index_lat:end_index_lat], dtype=tf.float16)
                    
                    ### rotate satellite image ###
                    if flip_lon:
                        satellite_tensor = tf.reverse(satellite_tensor, axis=[0])  # Reverse values along the longitude dimension
                    if flip_lat:
                        satellite_tensor = tf.reverse(satellite_tensor, axis=[1])  # Reverse values along the latitude dimension
                    
                    if args['noise_fraction'] > 0:
                        satellite_tensor = tf.where(random_values < args['noise_fraction'] / 2, 0.0, satellite_tensor)  # add 0s to image
                        satellite_tensor = tf.where(random_values > 1.0 - (args['noise_fraction'] / 2), 1.0, satellite_tensor)  # add 1s to image
                    
                    if args['num_dims'][0] == 3:
                        satellite_tensor = tf.expand_dims(satellite_tensor, axis=2)  # create a vertical dimension
                        satellite_tensor = tf.tile(satellite_tensor, (1, 1, len(args['pressure_levels']), 1))
                    
                    input_tensor = tf.concat([input_tensor, satellite_tensor], axis=-1)
                    
                ### add input images to tensorflow dataset ###
                input_tensor_for_timestep = tf.data.Dataset.from_tensors(input_tensor)
                if 'input_tensors_for_month' not in locals():
                    input_tensors_for_month = input_tensor_for_timestep
                else:
                    input_tensors_for_month = input_tensors_for_month.concatenate(input_tensor_for_timestep)

                front_tensor = tf.convert_to_tensor(front_image, dtype=tf.int32)
                
                if flip_lon:
                    front_tensor = tf.reverse(front_tensor, axis=[0])  # Reverse values along the longitude dimension
                if flip_lat:
                    front_tensor = tf.reverse(front_tensor, axis=[1])  # Reverse values along the latitude dimension

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

    print("%d-%02d Dataset progress (kept/discarded):  (%d/%d timesteps, %d/%d images)" % (year, month, timesteps_kept, timesteps_discarded, images_kept, images_discarded))
    
    ### save the tensorflow datasets ###
    try:
        tf.data.Dataset.save(input_tensors_for_month, path=tf_dataset_folder_inputs)
        tf.data.Dataset.save(front_tensors_for_month, path=tf_dataset_folder_fronts)
        print("Tensorflow datasets for %d-%02d saved to %s." % (year, month, args['tf_outdir']))
    except NameError:
        print("No images could be retained with the provided arguments.")