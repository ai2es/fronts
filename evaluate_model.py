"""
Functions used for evaluating a U-Net model. The functions can be used to make predictions or plot learning curves.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 9/13/2021 2:33 PM CDT
"""

import random
import pandas as pd
import argparse
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import file_manager as fm
import Fronts_Aggregate_Plot as fplot
import xarray as xr
import errors
import pickle
import matplotlib as mpl
from expand_fronts import one_pixel_expansion as ope


def calculate_performance_stats(model_number, model_dir, num_variables, num_dimensions, front_types, domain, file_dimensions,
    test_year, normalization_method, loss, fss_mask_size, fss_c, pixel_expansion, metric, num_images, longitude_domain_length,
    image_trim):
    """
    Function that calculates the number of true positives, false positives, true negatives, and false negatives on a testing set.

    Parameters
    ----------
    model_number: int
        Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        Main directory for the models.
    normalization_method: int
        Normalization method for the data (described near the end of the script).
    loss: str
        Loss function for the Unet.
    fss_mask_size: int
        Size of the mask for the FSS loss function.
    fss_c: float
        C hyperparameter for the FSS loss' sigmoid function.
    front_types: str
        Fronts in the data.
    domain: str
        Domain which the front and variable files cover.
    file_dimensions: int (x2)
        Dimensions of the data files.
    test_year: int
        Year for the test set used for calculating performance stats for cross-validation purposes.
    pixel_expansion: int
        Number of pixels to expand the fronts by in all directions.
    metric: str
        Metric used for evaluating the U-Net during training.
    num_dimensions: int
        Number of dimensions for the U-Net's convolutions, maxpooling, and upsampling.
    num_variables: int
        Number of variables in the datasets.
    num_images: int
        Number of images to stitch together for the final probability maps.
    longitude_domain_length: int
        Number of pixels in the longitude dimension of the final stitched map.
    image_trim: int
        Number of pixels to trim each image by in the longitude dimension before averaging the overlapping pixels.
    """

    if test_year is not None:
        front_files, variable_files = fm.load_test_files(num_variables, front_types, domain, file_dimensions, test_year)
    else:
        front_files, variable_files = fm.load_file_lists(num_variables, front_types, domain, file_dimensions)
    print("Front file count:", len(front_files))
    print("Variable file count:", len(variable_files))

    model = fm.load_model(model_number, model_dir, loss, fss_mask_size, fss_c, metric, num_dimensions)

    map_dim_x = model.layers[0].input_shape[0][1]  # Longitudinal dimension of the U-Net images
    map_dim_y = model.layers[0].input_shape[0][2]  # Latitudinal dimension of the U-Net images
    if num_dimensions == 2:
        channels = model.layers[0].input_shape[0][3]  # Number of variables used
    if num_dimensions == 3:
        levels = model.layers[0].input_shape[0][3]
        channels = model.layers[0].input_shape[0][4]

    """
    IMPORTANT!!!! Parameters for normalization were changed on August 5, 2021.
    
    If your model was trained prior to August 5, 2021, you MUST import the old parameters as follows:
        norm_params = pd.read_csv('normalization_parameters_old.csv', index_col='Variable')
    
    For all models created AFTER August 5, 2021, import the parameters as follows:
        norm_params = pd.read_csv('normalization_parameters.csv', index_col='Variable')
        
    If the prediction plots come out as a solid color across the domain with all probabilities near 0, you may be importing
    the wrong normalization parameters.
    """
    norm_params = pd.read_csv('normalization_parameters.csv', index_col='Variable')

    tp_cold = np.zeros(shape=[100])
    fp_cold = np.zeros(shape=[100])
    tn_cold = np.zeros(shape=[100])
    fn_cold = np.zeros(shape=[100])

    tp_warm = np.zeros(shape=[100])
    fp_warm = np.zeros(shape=[100])
    tn_warm = np.zeros(shape=[100])
    fn_warm = np.zeros(shape=[100])

    tp_stationary = np.zeros(shape=[100])
    fp_stationary = np.zeros(shape=[100])
    tn_stationary = np.zeros(shape=[100])
    fn_stationary = np.zeros(shape=[100])

    tp_occluded = np.zeros(shape=[100])
    fp_occluded = np.zeros(shape=[100])
    tn_occluded = np.zeros(shape=[100])
    fn_occluded = np.zeros(shape=[100])

    tp_dryline = np.zeros(shape=[100])
    fp_dryline = np.zeros(shape=[100])
    tn_dryline = np.zeros(shape=[100])
    fn_dryline = np.zeros(shape=[100])

    model_longitude_length = map_dim_x
    raw_image_index_array = np.empty(shape=[num_images,2])
    longitude_domain_length_trim = longitude_domain_length - 2*image_trim
    image_spacing = int((longitude_domain_length - model_longitude_length)/(num_images-1))
    latitude_domain_length = 128

    for x in range(len(front_files)):
        print("Prediction %d/%d....0/%d" % (x+1, len(front_files), num_images), end='\r')

        # Open random pair of files
        index = x
        fronts_filename = front_files[index]
        variables_filename = variable_files[index]
        fronts_ds = pd.read_pickle(fronts_filename)
        for i in range(pixel_expansion):
            fronts_ds = ope(fronts_ds)

        for i in range(num_images):
            if i == 0:
                raw_image_index_array[i][0] = image_trim
                raw_image_index_array[i][1] = model_longitude_length - image_trim
            elif i != num_images - 1:
                raw_image_index_array[i][0] = raw_image_index_array[i-1][0] + image_spacing
                raw_image_index_array[i][1] = raw_image_index_array[i-1][1] + image_spacing
            else:
                raw_image_index_array[i][0] = raw_image_index_array[i-1][0] + image_spacing
                raw_image_index_array[i][1] = raw_image_index_array[i-1][1] + image_spacing + image_trim - image_trim

        lon_pixels_per_image = int(raw_image_index_array[0][1] - raw_image_index_array[0][0])

        image_no_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_cold_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_warm_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_stationary_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_occluded_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_dryline_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        fronts = fronts_ds.sel(longitude=fronts_ds.longitude.values[image_trim:longitude_domain_length-image_trim], latitude=fronts_ds.latitude.values[0:128])
        image_lats = fronts_ds.latitude.values[0:128]
        image_lons = fronts_ds.longitude.values[image_trim:longitude_domain_length-image_trim]

        for image in range(num_images):
            # print("Prediction %d/%d....%d/%d" % (x+1, len(front_files_test), image+1, num_images), end='\r')
            print("Prediction %d/%d....%d/%d" % (x+1, 4, image+1, num_images), end='\r')
            lat_index = 0
            variable_ds = pd.read_pickle(variables_filename)
            if image == 0:
                lon_index = 0
            else:
                lon_index = int(image*image_spacing)

            lons = variable_ds.longitude.values[lon_index:lon_index + map_dim_x]
            lats = variable_ds.latitude.values[lat_index:lat_index + map_dim_y]

            variable_list = list(variable_ds.keys())
            for j in range(len(variable_list)):
                var = variable_list[j]
                if normalization_method == 1:
                    # Min-max normalization
                    variable_ds[var].values = np.nan_to_num((variable_ds[var].values - norm_params.loc[var,'Min']) /
                                                            (norm_params.loc[var,'Max'] - norm_params.loc[var,'Min']))
                elif normalization_method == 2:
                    # Mean normalization
                    variable_ds[var].values = np.nan_to_num((variable_ds[var].values - norm_params.loc[var,'Mean']) /
                                                            (norm_params.loc[var,'Max'] - norm_params.loc[var,'Min']))

            if num_dimensions == 2:
                variable_ds_new = np.nan_to_num(variable_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, map_dim_x,
                    map_dim_y, channels))
            elif num_dimensions == 3:
                variables_sfc = variable_ds[['t2m','d2m','sp','u10','v10','theta_w','mix_ratio','rel_humid','virt_temp','wet_bulb','theta_e',
                                             'q']].sel(longitude=lons, latitude=lats).to_array().T.values
                variables_1000 = variable_ds[['t_1000','d_1000','z_1000','u_1000','v_1000','theta_w_1000','mix_ratio_1000','rel_humid_1000','virt_temp_1000',
                                              'wet_bulb_1000','theta_e_1000','q_1000']].sel(longitude=lons, latitude=lats).to_array().T.values
                variables_950 = variable_ds[['t_950','d_950','z_950','u_950','v_950','theta_w_950','mix_ratio_950','rel_humid_950','virt_temp_950',
                                             'wet_bulb_950','theta_e_950','q_950']].sel(longitude=lons, latitude=lats).to_array().T.values
                variables_900 = variable_ds[['t_900','d_900','z_900','u_900','v_900','theta_w_900','mix_ratio_900','rel_humid_900','virt_temp_900',
                                             'wet_bulb_900','theta_e_900','q_900']].sel(longitude=lons, latitude=lats).to_array().T.values
                variables_850 = variable_ds[['t_850','d_850','z_850','u_850','v_850','theta_w_850','mix_ratio_850','rel_humid_850','virt_temp_850',
                                             'wet_bulb_850','theta_e_850','q_850']].sel(longitude=lons, latitude=lats).to_array().T.values
                variable_ds_new = np.array([variables_sfc,variables_1000,variables_950,variables_900,variables_850]).reshape(1,map_dim_x,map_dim_y,levels,channels)

            prediction = model.predict(variable_ds_new)

            # time = str(fronts.time.values)[0:13].replace('T', '-') + 'z'
            # print(time)

            # Arrays of probabilities for all front types
            no_probs = np.zeros([map_dim_x, map_dim_y])
            cold_probs = np.zeros([map_dim_x, map_dim_y])
            warm_probs = np.zeros([map_dim_x, map_dim_y])
            stationary_probs = np.zeros([map_dim_x, map_dim_y])
            occluded_probs = np.zeros([map_dim_x, map_dim_y])
            dryline_probs = np.zeros([map_dim_x, map_dim_y])

            thresholds = np.linspace(0.01,1,100)

            """
            Reformatting predictions: change the lines inside the loops according to your U-Net type.
            
            - <front> is the front type
            - i and j are loop indices, do NOT change these
            - n is the number of down layers in the model
            - l is the level index (0 = surface, 1 = 1000mb, 2 = 950mb, 3 = 900mb, 4 = 850mb)
            - x is the index of the front type in the softmax output. Refer to your data and the model structure to set this
              to the correct value for each front type.
            
            ### U-Net ###
            <front>_probs[i][j] = prediction[0][j][i][x]
            
            ### U-Net 3+ (2D) ###
            <front>_probs[i][j] = prediction[n][0][j][i][x]
            
            ### U-Net 3+ (3D) ###
            <front>_probs[i][j] = prediction[0][0][j][i][l][x]
            """
            if front_types == 'CFWF':
                for i in range(0, map_dim_x):
                    for j in range(0, map_dim_y):
                        no_probs[i][j] = prediction[5][0][i][j][0]
                        cold_probs[i][j] = prediction[5][0][i][j][1]
                        warm_probs[i][j] = prediction[5][0][i][j][2]
                if image == 0:
                    image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                    image_cold_probs[0: model_longitude_length - image_trim][:] = cold_probs[image_trim: model_longitude_length][:]
                    image_warm_probs[0: model_longitude_length - image_trim][:] = warm_probs[image_trim: model_longitude_length][:]
                elif image != num_images - 1:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                else:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    probs_ds = xr.Dataset(
                        {"no_probs": (("longitude", "latitude"), image_no_probs), "cold_probs": (("longitude", "latitude"), image_cold_probs),
                         "warm_probs": (("longitude", "latitude"), image_warm_probs)}, coords={"latitude": image_lats, "longitude": image_lons}).transpose()

                    new_fronts = fronts
                    t_cold_ds = xr.where(new_fronts == 1, 1, 0)
                    t_cold_probs = t_cold_ds.identifier * probs_ds.cold_probs
                    new_fronts = fronts
                    f_cold_ds = xr.where(new_fronts == 1, 0, 1)
                    f_cold_probs = f_cold_ds.identifier * probs_ds.cold_probs
                    new_fronts = fronts

                    t_warm_ds = xr.where(new_fronts == 2, 1, 0)
                    t_warm_probs = t_warm_ds.identifier * probs_ds.warm_probs
                    new_fronts = fronts
                    f_warm_ds = xr.where(new_fronts == 2, 0, 1)
                    f_warm_probs = f_warm_ds.identifier * probs_ds.warm_probs
                    new_fronts = fronts

                    for i in range(100):
                        tp_cold[i] += len(np.where(t_cold_probs > thresholds[i])[0])
                        tn_cold[i] += len(np.where((f_cold_probs < thresholds[i]) & (f_cold_probs != 0))[0])
                        fp_cold[i] += len(np.where(f_cold_probs > thresholds[i])[0])
                        fn_cold[i] += len(np.where((t_cold_probs < thresholds[i]) & (t_cold_probs != 0))[0])
                        tp_warm[i] += len(np.where(t_warm_probs > thresholds[i])[0])
                        tn_warm[i] += len(np.where((f_warm_probs < thresholds[i]) & (f_warm_probs != 0))[0])
                        fp_warm[i] += len(np.where(f_warm_probs > thresholds[i])[0])
                        fn_warm[i] += len(np.where((t_warm_probs < thresholds[i]) & (t_warm_probs != 0))[0])

                    # print("Prediction %d/%d....done" % (x+1, len(front_files_test)))
                    print("Prediction %d/%d....done" % (x+1, 4))

            elif front_types == 'SFOF':
                for i in range(0, map_dim_x):
                    for j in range(0, map_dim_y):
                        no_probs[i][j] = prediction[5][0][i][j][0]
                        stationary_probs[i][j] = prediction[5][0][i][j][1]
                        occluded_probs[i][j] = prediction[5][0][i][j][2]
                if image == 0:
                    image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                    image_stationary_probs[0: model_longitude_length - image_trim][:] = stationary_probs[image_trim: model_longitude_length][:]
                    image_occluded_probs[0: model_longitude_length - image_trim][:] = occluded_probs[image_trim: model_longitude_length][:]
                elif image != num_images - 1:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                else:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    probs_ds = xr.Dataset(
                        {"no_probs": (("longitude", "latitude"), image_no_probs), "stationary_probs": (("longitude", "latitude"), image_stationary_probs),
                         "occluded_probs": (("longitude", "latitude"), image_occluded_probs)}, coords={"latitude": image_lats, "longitude": image_lons})

                    t_stationary_ds = xr.where(new_fronts == 3, 1, 0)
                    t_stationary_probs = t_stationary_ds.identifier * probs_ds.stationary_probs
                    new_fronts = fronts
                    f_stationary_ds = xr.where(new_fronts == 3, 0, 1)
                    f_stationary_probs = f_stationary_ds.identifier * probs_ds.stationary_probs
                    new_fronts = fronts

                    t_occluded_ds = xr.where(new_fronts == 4, 1, 0)
                    t_occluded_probs = t_occluded_ds.identifier * probs_ds.occluded_probs
                    new_fronts = fronts
                    f_occluded_ds = xr.where(new_fronts == 4, 0, 1)
                    f_occluded_probs = f_occluded_ds.identifier * probs_ds.occluded_probs

                    for i in range(100):
                        tp_stationary[i] += len(np.where(t_stationary_probs > thresholds[i])[0])
                        tn_stationary[i] += len(np.where((f_stationary_probs < thresholds[i]) & (f_stationary_probs != 0))[0])
                        fp_stationary[i] += len(np.where(f_stationary_probs > thresholds[i])[0])
                        fn_stationary[i] += len(np.where((t_stationary_probs < thresholds[i]) & (t_stationary_probs != 0))[0])
                        tp_occluded[i] += len(np.where(t_occluded_probs > thresholds[i])[0])
                        tn_occluded[i] += len(np.where((f_occluded_probs < thresholds[i]) & (f_occluded_probs != 0))[0])
                        fp_occluded[i] += len(np.where(f_occluded_probs > thresholds[i])[0])
                        fn_occluded[i] += len(np.where((t_occluded_probs < thresholds[i]) & (t_occluded_probs != 0))[0])
                    print("Prediction %d/%d....done" % (x+1, len(front_files)))

            elif front_types == 'DL':
                for i in range(0, map_dim_x):
                    for j in range(0, map_dim_y):
                        no_probs[i][j] = prediction[5][0][i][j][0]
                        dryline_probs[i][j] = prediction[5][0][i][j][1]
                if image == 0:
                    image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                    image_dryline_probs[0: model_longitude_length - image_trim][:] = dryline_probs[image_trim: model_longitude_length][:]
                elif image != num_images - 1:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                else:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    probs_ds = xr.Dataset(
                        {"no_probs": (("latitude", "longitude"), no_probs), "dryline_probs": (("latitude", "longitude"), image_dryline_probs)},
                        coords={"latitude": lats, "longitude": lons})

                    t_dryline_ds = xr.where(new_fronts == 5, 1, 0)
                    t_dryline_probs = t_dryline_ds.identifier * probs_ds.dryline_probs
                    new_fronts = fronts
                    f_dryline_ds = xr.where(new_fronts == 5, 0, 1)
                    f_dryline_probs = f_dryline_ds.identifier * probs_ds.dryline_probs

                    for i in range(100):
                        tp_dryline[i] += len(np.where(t_dryline_probs > thresholds[i])[0])
                        tn_dryline[i] += len(np.where((f_dryline_probs < thresholds[i]) & (f_dryline_probs != 0))[0])
                        fp_dryline[i] += len(np.where(f_dryline_probs > thresholds[i])[0])
                        fn_dryline[i] += len(np.where((t_dryline_probs < thresholds[i]) & (t_dryline_probs != 0))[0])

                    print("Prediction %d/%d....done" % (x+1, len(front_files)))

            elif front_types == 'ALL':
                for i in range(0, map_dim_x):
                    for j in range(0, map_dim_y):
                        no_probs[i][j] = prediction[5][0][i][j][0]
                        cold_probs[i][j] = prediction[5][0][i][j][1]
                        warm_probs[i][j] = prediction[5][0][i][j][2]
                        stationary_probs[i][j] = prediction[5][0][i][j][3]
                        occluded_probs[i][j] = prediction[5][0][i][j][4]
                        dryline_probs[i][j] = prediction[5][0][i][j][5]
                if image == 0:
                    image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                    image_cold_probs[0: model_longitude_length - image_trim][:] = cold_probs[image_trim: model_longitude_length][:]
                    image_warm_probs[0: model_longitude_length - image_trim][:] = warm_probs[image_trim: model_longitude_length][:]
                    image_stationary_probs[0: model_longitude_length - image_trim][:] = stationary_probs[image_trim: model_longitude_length][:]
                    image_occluded_probs[0: model_longitude_length - image_trim][:] = occluded_probs[image_trim: model_longitude_length][:]
                elif image != num_images - 1:
                    image_dryline_probs[0: model_longitude_length - image_trim][:] = dryline_probs[image_trim: model_longitude_length][:]

                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                else:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]

                    new_fronts = fronts
                    t_cold_ds = xr.where(new_fronts == 1, 1, 0)
                    t_cold_probs = t_cold_ds.identifier * probs_ds.cold_probs
                    new_fronts = fronts
                    f_cold_ds = xr.where(new_fronts == 1, 0, 1)
                    f_cold_probs = f_cold_ds.identifier * probs_ds.cold_probs
                    new_fronts = fronts

                    t_warm_ds = xr.where(new_fronts == 2, 1, 0)
                    t_warm_probs = t_warm_ds.identifier * probs_ds.warm_probs
                    new_fronts = fronts
                    f_warm_ds = xr.where(new_fronts == 2, 0, 1)
                    f_warm_probs = f_warm_ds.identifier * probs_ds.warm_probs
                    new_fronts = fronts

                    t_stationary_ds = xr.where(new_fronts == 3, 1, 0)
                    t_stationary_probs = t_stationary_ds.identifier * probs_ds.stationary_probs
                    new_fronts = fronts
                    f_stationary_ds = xr.where(new_fronts == 3, 0, 1)
                    f_stationary_probs = f_stationary_ds.identifier * probs_ds.stationary_probs
                    new_fronts = fronts

                    t_occluded_ds = xr.where(new_fronts == 4, 1, 0)
                    t_occluded_probs = t_occluded_ds.identifier * probs_ds.occluded_probs
                    new_fronts = fronts
                    f_occluded_ds = xr.where(new_fronts == 4, 0, 1)
                    f_occluded_probs = f_occluded_ds.identifier * probs_ds.occluded_probs

                    t_dryline_ds = xr.where(new_fronts == 5, 1, 0)
                    t_dryline_probs = t_dryline_ds.identifier * probs_ds.dryline_probs
                    new_fronts = fronts
                    f_dryline_ds = xr.where(new_fronts == 5, 0, 1)
                    f_dryline_probs = f_dryline_ds.identifier * probs_ds.dryline_probs

                    for i in range(100):
                        tp_cold[i] += len(np.where(t_cold_probs > thresholds[i])[0])
                        tn_cold[i] += len(np.where((f_cold_probs < thresholds[i]) & (f_cold_probs != 0))[0])
                        fp_cold[i] += len(np.where(f_cold_probs > thresholds[i])[0])
                        fn_cold[i] += len(np.where((t_cold_probs < thresholds[i]) & (t_cold_probs != 0))[0])
                        tp_warm[i] += len(np.where(t_warm_probs > thresholds[i])[0])
                        tn_warm[i] += len(np.where((f_warm_probs < thresholds[i]) & (f_warm_probs != 0))[0])
                        fp_warm[i] += len(np.where(f_warm_probs > thresholds[i])[0])
                        fn_warm[i] += len(np.where((t_warm_probs < thresholds[i]) & (t_warm_probs != 0))[0])
                        tp_stationary[i] += len(np.where(t_stationary_probs > thresholds[i])[0])
                        tn_stationary[i] += len(np.where((f_stationary_probs < thresholds[i]) & (f_stationary_probs != 0))[0])
                        fp_stationary[i] += len(np.where(f_stationary_probs > thresholds[i])[0])
                        fn_stationary[i] += len(np.where((t_stationary_probs < thresholds[i]) & (t_stationary_probs != 0))[0])
                        tp_occluded[i] += len(np.where(t_occluded_probs > thresholds[i])[0])
                        tn_occluded[i] += len(np.where((f_occluded_probs < thresholds[i]) & (f_occluded_probs != 0))[0])
                        fp_occluded[i] += len(np.where(f_occluded_probs > thresholds[i])[0])
                        fn_occluded[i] += len(np.where((t_occluded_probs < thresholds[i]) & (t_occluded_probs != 0))[0])
                        tp_dryline[i] += len(np.where(t_dryline_probs > thresholds[i])[0])
                        tn_dryline[i] += len(np.where((f_dryline_probs < thresholds[i]) & (f_dryline_probs != 0))[0])
                        fp_dryline[i] += len(np.where(f_dryline_probs > thresholds[i])[0])
                        fn_dryline[i] += len(np.where((t_dryline_probs < thresholds[i]) & (t_dryline_probs != 0))[0])

                    print("Prediction %d/%d....done" % (x+1, len(front_files)))

    if front_types == 'CFWF':
        performance_ds = xr.Dataset({"tp_cold": ("threshold", tp_cold), "tp_warm": ("threshold", tp_warm),
                                     "fp_cold": ("threshold", fp_cold), "fp_warm": ("threshold", fp_warm),
                                     "tn_cold": ("threshold", tn_cold), "tn_warm": ("threshold", tn_warm),
                                     "fn_cold": ("threshold", fn_cold), "fn_warm": ("threshold", fn_warm)}, coords={"threshold": thresholds})
    elif front_types == 'SFOF':
        performance_ds = xr.Dataset({"tp_stationary": ("threshold", tp_stationary), "tp_occluded": ("threshold", tp_occluded),
                                     "fp_stationary": ("threshold", fp_stationary), "fp_occluded": ("threshold", fp_occluded),
                                     "tn_stationary": ("threshold", tn_stationary), "tn_occluded": ("threshold", tn_occluded),
                                     "fn_stationary": ("threshold", fn_stationary), "fn_occluded": ("threshold", fn_occluded)}, coords={"threshold": thresholds})
    elif front_types == 'DL':
        performance_ds = xr.Dataset({"tp_dryline": ("threshold", tp_dryline), "tp_dryline": ("threshold", tp_dryline),
                                     "fp_dryline": ("threshold", fp_dryline), "tn_dryline": ("threshold", tn_dryline)}, coords={"threshold": thresholds})
    elif front_types == 'ALL':
        performance_ds = xr.Dataset({"tp_cold": ("threshold", tp_cold), "tp_warm": ("threshold", tp_warm),
                                     "tp_stationary": ("threshold", tp_stationary), "tp_occluded": ("threshold", tp_occluded),
                                     "fp_cold": ("threshold", fp_cold), "fp_warm": ("threshold", fp_warm),
                                     "fp_stationary": ("threshold", fp_stationary), "fp_occluded": ("threshold", fp_occluded),
                                     "tn_cold": ("threshold", tn_cold), "tn_warm": ("threshold", tn_warm),
                                     "tn_stationary": ("threshold", tn_stationary), "tn_occluded": ("threshold", tn_occluded),
                                     "fn_cold": ("threshold", fn_cold), "fn_warm": ("threshold", fn_warm),
                                     "fn_stationary": ("threshold", fn_stationary), "fn_occluded": ("threshold", fn_occluded)}, coords={"threshold": thresholds})

    print(performance_ds)
    with open('%s/model_%d/model_%d_performance_stats_%dkm.pkl' % (model_dir, model_number, model_number, int(pixel_expansion*25)), 'wb') as f:
        pickle.dump(performance_ds, f)


def find_matches_for_domain(longitude_domain_length, model_longitude_length):
    """
    Function that outputs the number of images that can be stitched together with the specified domain length and the length
    of the domain dimension output by the model.

    Parameters
    ----------
    longitude_domain_length: int
        Number of pixels in the longitude dimension of the final stitched map.
    model_longitude_length: int
        Number of pixels in the longitude dimension of the model's output.
    """

    # latitude_domain_length = 128
    # model_latitude_length = 128
    num_matches = 0
    for i in range(2,longitude_domain_length-model_longitude_length):  # Image counter
        image_spacing_match = (longitude_domain_length-model_longitude_length)/(i-1)
        if image_spacing_match - int(image_spacing_match) == 0 and image_spacing_match > 1 and model_longitude_length-image_spacing_match > 0:
            num_matches += 1
            print("MATCH: (Images, Spacing, Overlap, Max Trim)", i, int(image_spacing_match), int(model_longitude_length-image_spacing_match), int(np.floor(image_spacing_match/2)))
    print("\nMatches found: %d" % num_matches)


def make_prediction(model_number, model_dir, front_file_list, variable_file_list, normalization_method, loss, fss_mask_size,
    fss_c, front_types, pixel_expansion, metric, num_dimensions, num_images, longitude_domain_length, image_trim, 
    year, month, day, hour):
    """
    Function that makes random predictions using the provided model.

    Parameters
    ----------
    model_number: int
        Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        Main directory for the models.
    front_file_list: list
        List of filenames that contain front data.
    variable_file_list: list
        List of filenames that contain variable data.
    normalization_method: int
        Normalization method for the data (described near the end of the script).
    loss: str
        Loss function for the Unet.
    fss_mask_size: int
        Size of the mask for the FSS loss function.
    fss_c: float
        C hyperparameter for the FSS loss' sigmoid function.
    front_types: str
        Fronts in the data.
    pixel_expansion: int
        Number of pixels to expand the fronts by in all directions.
    metric: str
        Metric used for evaluating the U-Net during training.
    num_dimensions: int
        Number of dimensions for the U-Net's convolutions, maxpooling, and upsampling.
    num_images: int
        Number of images to stitch together for the final probability maps.
    longitude_domain_length: int
        Number of pixels in the longitude dimension of the final stitched map.
    image_trim: int
        Number of pixels to trim each image by in the longitude dimension before averaging the overlapping pixels.
    year: int
    month: int
    day: int
    hour: int
    """
    model = fm.load_model(model_number, model_dir, loss, fss_mask_size, fss_c, metric, num_dimensions)

    map_dim_x = model.layers[0].input_shape[0][1]  # Longitudinal dimension of the U-Net images
    map_dim_y = model.layers[0].input_shape[0][2]  # Latitudinal dimension of the U-Net images
    if num_dimensions == 2:
        channels = model.layers[0].input_shape[0][3]  # Number of variables used
    if num_dimensions == 3:
        levels = model.layers[0].input_shape[0][3]
        channels = model.layers[0].input_shape[0][4]

    model_longitude_length = map_dim_x
    raw_image_index_array = np.empty(shape=[num_images,2])
    longitude_domain_length_trim = longitude_domain_length - 2*image_trim
    image_spacing = int((longitude_domain_length - model_longitude_length)/(num_images-1))
    latitude_domain_length = 128

    """
    IMPORTANT!!!! Parameters for normalization were changed on August 5, 2021.
    
    If your model was trained prior to August 5, 2021, you MUST import the old parameters as follows:
        norm_params = pd.read_csv('normalization_parameters_old.csv', index_col='Variable')
    
    For all models created AFTER August 5, 2021, import the parameters as follows:
        norm_params = pd.read_csv('normalization_parameters.csv', index_col='Variable')
        
    If the prediction plots come out as a solid color across the domain with all probabilities near 0, you may be importing
    the wrong normalization parameters.
    """
    norm_params = pd.read_csv('normalization_parameters.csv', index_col='Variable')

    front_filename_no_dir = 'FrontObjects_%s_%d%02d%02d%02d_%s_%dx%d.pkl' % (args.front_types, args.year, args.month,
        args.day, args.hour, args.domain, args.file_dimensions[0], args.file_dimensions[1])
    variable_filename_no_dir = 'Data_%dvar_%d%02d%02d%02d_%s_%dx%d.pkl' % (60, args.year, args.month, args.day, args.hour,
        args.domain, args.file_dimensions[0], args.file_dimensions[1])

    front_file = [front_filename for front_filename in front_file_list if front_filename_no_dir in front_filename][0]
    variable_file = [variable_filename for variable_filename in variable_file_list if variable_filename_no_dir in variable_filename][0]

    fronts_ds = pd.read_pickle(front_file)

    for i in range(num_images):
        if i == 0:
            raw_image_index_array[i][0] = image_trim
            raw_image_index_array[i][1] = model_longitude_length - image_trim
        elif i != num_images - 1:
            raw_image_index_array[i][0] = raw_image_index_array[i-1][0] + image_spacing
            raw_image_index_array[i][1] = raw_image_index_array[i-1][1] + image_spacing
        else:
            raw_image_index_array[i][0] = raw_image_index_array[i-1][0] + image_spacing
            raw_image_index_array[i][1] = raw_image_index_array[i-1][1] + image_spacing + image_trim - image_trim

    lon_pixels_per_image = int(raw_image_index_array[0][1] - raw_image_index_array[0][0])

    image_no_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
    image_cold_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
    image_warm_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
    image_stationary_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
    image_occluded_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
    image_dryline_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
    fronts = fronts_ds.sel(longitude=fronts_ds.longitude.values[image_trim:longitude_domain_length-image_trim],
                           latitude=fronts_ds.latitude.values[0:128])
    image_lats = fronts_ds.latitude.values[0:128]
    image_lons = fronts_ds.longitude.values[image_trim:longitude_domain_length-image_trim]

    for image in range(num_images):
        print("%d-%02d-%02d-%02dz...%d/%d" % (year, month, day, hour, image+1, num_images), end='\r')
        lat_index = 0
        variable_ds = pd.read_pickle(variable_file)
        if image == 0:
            lon_index = 0
        else:
            lon_index = int(image*image_spacing)

        lons = variable_ds.longitude.values[lon_index:lon_index + map_dim_x]
        lats = variable_ds.latitude.values[lat_index:lat_index + map_dim_y]

        variable_list = list(variable_ds.keys())
        for j in range(len(variable_list)):
            var = variable_list[j]
            if normalization_method == 1:
                # Min-max normalization
                variable_ds[var].values = np.nan_to_num((variable_ds[var].values - norm_params.loc[var,'Min']) /
                                                        (norm_params.loc[var,'Max'] - norm_params.loc[var,'Min']))
            elif normalization_method == 2:
                # Mean normalization
                variable_ds[var].values = np.nan_to_num((variable_ds[var].values - norm_params.loc[var,'Mean']) /
                                                        (norm_params.loc[var,'Max'] - norm_params.loc[var,'Min']))

        if num_dimensions == 2:
            variable_ds_new = np.nan_to_num(variable_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, map_dim_x,
                map_dim_y, channels))
        elif num_dimensions == 3:
            variables_sfc = variable_ds[['t2m','d2m','sp','u10','v10','theta_w','mix_ratio','rel_humid','virt_temp','wet_bulb','theta_e',
                                         'q']].sel(longitude=lons, latitude=lats).to_array().T.values
            variables_1000 = variable_ds[['t_1000','d_1000','z_1000','u_1000','v_1000','theta_w_1000','mix_ratio_1000','rel_humid_1000','virt_temp_1000',
                                          'wet_bulb_1000','theta_e_1000','q_1000']].sel(longitude=lons, latitude=lats).to_array().T.values
            variables_950 = variable_ds[['t_950','d_950','z_950','u_950','v_950','theta_w_950','mix_ratio_950','rel_humid_950','virt_temp_950',
                                         'wet_bulb_950','theta_e_950','q_950']].sel(longitude=lons, latitude=lats).to_array().T.values
            variables_900 = variable_ds[['t_900','d_900','z_900','u_900','v_900','theta_w_900','mix_ratio_900','rel_humid_900','virt_temp_900',
                                         'wet_bulb_900','theta_e_900','q_900']].sel(longitude=lons, latitude=lats).to_array().T.values
            variables_850 = variable_ds[['t_850','d_850','z_850','u_850','v_850','theta_w_850','mix_ratio_850','rel_humid_850','virt_temp_850',
                                         'wet_bulb_850','theta_e_850','q_850']].sel(longitude=lons, latitude=lats).to_array().T.values
            variable_ds_new = np.array([variables_sfc,variables_1000,variables_950,variables_900,variables_850]).reshape(1,map_dim_x,map_dim_y,levels,channels)

        prediction = model.predict(variable_ds_new)

        time = str(fronts.time.values)[0:13].replace('T', '-') + 'z'
        # print(time)

        # Arrays of probabilities for all front types
        no_probs = np.zeros([map_dim_x, map_dim_y])
        cold_probs = np.zeros([map_dim_x, map_dim_y])
        warm_probs = np.zeros([map_dim_x, map_dim_y])
        stationary_probs = np.zeros([map_dim_x, map_dim_y])
        occluded_probs = np.zeros([map_dim_x, map_dim_y])
        dryline_probs = np.zeros([map_dim_x, map_dim_y])

        """
        Reformatting predictions: change the lines inside the loops according to your U-Net type.
        
        - <front> is the front type
        - i and j are loop indices, do NOT change these
        - n is the number of down layers in the model
        - l is the level index (0 = surface, 1 = 1000mb, 2 = 950mb, 3 = 900mb, 4 = 850mb)
        - x is the index of the front type in the softmax output. Refer to your data and the model structure to set this
          to the correct value for each front type.
        
        ### U-Net ###
        <front>_probs[i][j] = prediction[0][j][i][x]
        
        ### U-Net 3+ (2D) ###
        <front>_probs[i][j] = prediction[n][0][j][i][x]
        
        ### U-Net 3+ (3D) ###
        <front>_probs[i][j] = prediction[0][0][j][i][l][x]
        """
        if front_types == 'CFWF':
            for i in range(0, map_dim_x):
                for j in range(0, map_dim_y):
                    no_probs[i][j] = prediction[5][0][i][j][0]
                    cold_probs[i][j] = prediction[5][0][i][j][1]
                    warm_probs[i][j] = prediction[5][0][i][j][2]
            if image == 0:
                image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                image_cold_probs[0: model_longitude_length - image_trim][:] = cold_probs[image_trim: model_longitude_length][:]
                image_warm_probs[0: model_longitude_length - image_trim][:] = warm_probs[image_trim: model_longitude_length][:]
            elif image != num_images - 1:
                image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
            else:
                image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                probs_ds = xr.Dataset(
                    {"no_probs": (("longitude", "latitude"), image_no_probs), "cold_probs": (("longitude", "latitude"), image_cold_probs),
                     "warm_probs": (("longitude", "latitude"), image_warm_probs)}, coords={"latitude": image_lats, "longitude": image_lons}).transpose()
                prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, num_images, image_trim)

        elif front_types == 'SFOF':
            for i in range(0, map_dim_y):
                for j in range(0, map_dim_x):
                    no_probs[i][j] = prediction[0][j][i][0]
                    stationary_probs[i][j] = prediction[0][j][i][1]
                    occluded_probs[i][j] = prediction[0][j][i][2]
            if image == 0:
                image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                image_stationary_probs[0: model_longitude_length - image_trim][:] = stationary_probs[image_trim: model_longitude_length][:]
                image_occluded_probs[0: model_longitude_length - image_trim][:] = occluded_probs[image_trim: model_longitude_length][:]
            elif image != num_images - 1:
                image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
            else:
                image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                probs_ds = xr.Dataset(
                    {"no_probs": (("longitude", "latitude"), image_no_probs), "stationary_probs": (("longitude", "latitude"), image_stationary_probs),
                     "occluded_probs": (("longitude", "latitude"), image_occluded_probs)}, coords={"latitude": image_lats, "longitude": image_lons}).transpose()
                prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, num_images, image_trim)

        elif front_types == 'DL':
            for i in range(0, map_dim_y):
                for j in range(0, map_dim_x):
                    no_probs[i][j] = prediction[0][j][i][0]
                    dryline_probs[i][j] = prediction[0][j][i][1]
            if image == 0:
                image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                image_dryline_probs[0: model_longitude_length - image_trim][:] = dryline_probs[image_trim: model_longitude_length][:]
            elif image != num_images - 1:
                image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
            else:
                image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                probs_ds = xr.Dataset(
                    {"no_probs": (("latitude", "longitude"), no_probs), "dryline_probs": (("latitude", "longitude"), image_dryline_probs)},
                    coords={"latitude": lats, "longitude": lons})
                prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, num_images, image_trim)

        elif front_types == 'ALL':
            for i in range(0, map_dim_y):
                for j in range(0, map_dim_x):
                    no_probs[i][j] = prediction[0][j][i][0]
                    cold_probs[i][j] = prediction[0][j][i][1]
                    warm_probs[i][j] = prediction[0][j][i][2]
                    stationary_probs[i][j] = prediction[0][j][i][3]
                    occluded_probs[i][j] = prediction[0][j][i][4]
                    dryline_probs[i][j] = prediction[0][j][i][5]
            if image == 0:
                image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                image_cold_probs[0: model_longitude_length - image_trim][:] = cold_probs[image_trim: model_longitude_length][:]
                image_warm_probs[0: model_longitude_length - image_trim][:] = warm_probs[image_trim: model_longitude_length][:]
                image_stationary_probs[0: model_longitude_length - image_trim][:] = stationary_probs[image_trim: model_longitude_length][:]
                image_occluded_probs[0: model_longitude_length - image_trim][:] = occluded_probs[image_trim: model_longitude_length][:]
                image_dryline_probs[0: model_longitude_length - image_trim][:] = dryline_probs[image_trim: model_longitude_length][:]
            elif image != num_images - 1:
                image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
            else:
                image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                probs_ds = xr.Dataset(
                    {"no_probs": (("latitude", "longitude"), image_no_probs), "cold_probs": (("latitude", "longitude"), image_cold_probs),
                     "warm_probs": (("latitude", "longitude"), image_warm_probs), "stationary_probs": (("latitude", "longitude"), image_stationary_probs),
                     "occluded_probs": (("latitude", "longitude"), image_occluded_probs), "dryline_probs": (("latitude", "longitude"), image_dryline_probs)},
                    coords={"latitude": lats, "longitude": lons})
                prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, num_images, image_trim)


def make_random_predictions(model_number, model_dir, fronts_files_list, variables_files_list, predictions, normalization_method,
    loss, fss_mask_size, fss_c, front_types, pixel_expansion, metric, num_dimensions, num_images, longitude_domain_length,
    image_trim):
    """
    Function that makes random predictions using the provided model.

    Parameters
    ----------
    model_number: int
        Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        Main directory for the models.
    fronts_files_list: list
        List of filenames that contain front data.
    variables_files_list: list
        List of filenames that contain variable data.
    predictions: int
        Number of random predictions to make.
    normalization_method: int
        Normalization method for the data (described near the end of the script).
    loss: str
        Loss function for the Unet.
    fss_mask_size: int
        Size of the mask for the FSS loss function.
    fss_c: float
        C hyperparameter for the FSS loss' sigmoid function.
    front_types: str
        Fronts in the data.
    pixel_expansion: int
        Number of pixels to expand the fronts by in all directions.
    metric: str
        Metric used for evaluating the U-Net during training.
    num_dimensions: int
        Number of dimensions for the U-Net's convolutions, maxpooling, and upsampling.
    num_images: int
        Number of images to stitch together for the final probability maps.
    longitude_domain_length: int
        Number of pixels in the longitude dimension of the final stitched map.
    image_trim: int
        Number of pixels to trim each image by in the longitude dimension before averaging the overlapping pixels.
    """
    model = fm.load_model(model_number, model_dir, loss, fss_mask_size, fss_c, metric, num_dimensions)

    map_dim_x = model.layers[0].input_shape[0][1]  # Longitudinal dimension of the U-Net images
    map_dim_y = model.layers[0].input_shape[0][2]  # Latitudinal dimension of the U-Net images
    if num_dimensions == 2:
        channels = model.layers[0].input_shape[0][3]  # Number of variables used
    if num_dimensions == 3:
        levels = model.layers[0].input_shape[0][3]
        channels = model.layers[0].input_shape[0][4]

    model_longitude_length = map_dim_x
    raw_image_index_array = np.empty(shape=[num_images,2])
    longitude_domain_length_trim = longitude_domain_length - 2*image_trim
    image_spacing = int((longitude_domain_length - model_longitude_length)/(num_images-1))
    latitude_domain_length = 128

    """
    IMPORTANT!!!! Parameters for normalization were changed on August 5, 2021.
    
    If your model was trained prior to August 5, 2021, you MUST import the old parameters as follows:
        norm_params = pd.read_csv('normalization_parameters_old.csv', index_col='Variable')
    
    For all models created AFTER August 5, 2021, import the parameters as follows:
        norm_params = pd.read_csv('normalization_parameters.csv', index_col='Variable')
        
    If the prediction plots come out as a solid color across the domain with all probabilities near 0, you may be importing
    the wrong normalization parameters.
    """
    norm_params = pd.read_csv('normalization_parameters.csv', index_col='Variable')

    for x in range(predictions):

        # Open random pair of files
        index = random.choices(range(len(fronts_files_list) - 1), k=1)[0]
        fronts_filename = fronts_files_list[index]
        variables_filename = variables_files_list[index]
        fronts_ds = pd.read_pickle(fronts_filename)

        for i in range(num_images):
            if i == 0:
                raw_image_index_array[i][0] = image_trim
                raw_image_index_array[i][1] = model_longitude_length - image_trim
            elif i != num_images - 1:
                raw_image_index_array[i][0] = raw_image_index_array[i-1][0] + image_spacing
                raw_image_index_array[i][1] = raw_image_index_array[i-1][1] + image_spacing
            else:
                raw_image_index_array[i][0] = raw_image_index_array[i-1][0] + image_spacing
                raw_image_index_array[i][1] = raw_image_index_array[i-1][1] + image_spacing + image_trim - image_trim

        lon_pixels_per_image = int(raw_image_index_array[0][1] - raw_image_index_array[0][0])

        image_no_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_cold_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_warm_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_stationary_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_occluded_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        image_dryline_probs = np.empty(shape=[longitude_domain_length_trim,latitude_domain_length])
        fronts = fronts_ds.sel(longitude=fronts_ds.longitude.values[image_trim:longitude_domain_length-image_trim], latitude=fronts_ds.latitude.values[0:128])
        image_lats = fronts_ds.latitude.values[0:128]
        image_lons = fronts_ds.longitude.values[image_trim:longitude_domain_length-image_trim]

        for image in range(num_images):
            print("Prediction %d/%d....%d/%d" % (x+1, predictions, image+1, num_images), end='\r')
            lat_index = 0
            variable_ds = pd.read_pickle(variables_filename)
            if image == 0:
                lon_index = 0
            else:
                lon_index = int(image*image_spacing)

            lons = variable_ds.longitude.values[lon_index:lon_index + map_dim_x]
            lats = variable_ds.latitude.values[lat_index:lat_index + map_dim_y]

            variable_list = list(variable_ds.keys())
            for j in range(len(variable_list)):
                var = variable_list[j]
                if normalization_method == 1:
                    # Min-max normalization
                    variable_ds[var].values = np.nan_to_num((variable_ds[var].values - norm_params.loc[var,'Min']) /
                                                            (norm_params.loc[var,'Max'] - norm_params.loc[var,'Min']))
                elif normalization_method == 2:
                    # Mean normalization
                    variable_ds[var].values = np.nan_to_num((variable_ds[var].values - norm_params.loc[var,'Mean']) /
                                                            (norm_params.loc[var,'Max'] - norm_params.loc[var,'Min']))

            if num_dimensions == 2:
                variable_ds_new = np.nan_to_num(variable_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, map_dim_x,
                    map_dim_y, channels))
            elif num_dimensions == 3:
                variables_sfc = variable_ds[['t2m','d2m','sp','u10','v10','theta_w','mix_ratio','rel_humid','virt_temp','wet_bulb','theta_e',
                                             'q']].sel(longitude=lons, latitude=lats).to_array().T.values
                variables_1000 = variable_ds[['t_1000','d_1000','z_1000','u_1000','v_1000','theta_w_1000','mix_ratio_1000','rel_humid_1000','virt_temp_1000',
                                              'wet_bulb_1000','theta_e_1000','q_1000']].sel(longitude=lons, latitude=lats).to_array().T.values
                variables_950 = variable_ds[['t_950','d_950','z_950','u_950','v_950','theta_w_950','mix_ratio_950','rel_humid_950','virt_temp_950',
                                             'wet_bulb_950','theta_e_950','q_950']].sel(longitude=lons, latitude=lats).to_array().T.values
                variables_900 = variable_ds[['t_900','d_900','z_900','u_900','v_900','theta_w_900','mix_ratio_900','rel_humid_900','virt_temp_900',
                                             'wet_bulb_900','theta_e_900','q_900']].sel(longitude=lons, latitude=lats).to_array().T.values
                variables_850 = variable_ds[['t_850','d_850','z_850','u_850','v_850','theta_w_850','mix_ratio_850','rel_humid_850','virt_temp_850',
                                             'wet_bulb_850','theta_e_850','q_850']].sel(longitude=lons, latitude=lats).to_array().T.values
                variable_ds_new = np.array([variables_sfc,variables_1000,variables_950,variables_900,variables_850]).reshape(1,map_dim_x,map_dim_y,levels,channels)

            prediction = model.predict(variable_ds_new)

            time = str(fronts.time.values)[0:13].replace('T', '-') + 'z'
            # print(time)

            # Arrays of probabilities for all front types
            no_probs = np.zeros([map_dim_x, map_dim_y])
            cold_probs = np.zeros([map_dim_x, map_dim_y])
            warm_probs = np.zeros([map_dim_x, map_dim_y])
            stationary_probs = np.zeros([map_dim_x, map_dim_y])
            occluded_probs = np.zeros([map_dim_x, map_dim_y])
            dryline_probs = np.zeros([map_dim_x, map_dim_y])

            """
            Reformatting predictions: change the lines inside the loops according to your U-Net type.
            
            - <front> is the front type
            - i and j are loop indices, do NOT change these
            - n is the number of down layers in the model
            - l is the level index (0 = surface, 1 = 1000mb, 2 = 950mb, 3 = 900mb, 4 = 850mb)
            - x is the index of the front type in the softmax output. Refer to your data and the model structure to set this
              to the correct value for each front type.
            
            ### U-Net ###
            <front>_probs[i][j] = prediction[0][j][i][x]
            
            ### U-Net 3+ (2D) ###
            <front>_probs[i][j] = prediction[n][0][j][i][x]
            
            ### U-Net 3+ (3D) ###
            <front>_probs[i][j] = prediction[0][0][j][i][l][x]
            """
            if front_types == 'CFWF':
                for i in range(0, map_dim_x):
                    for j in range(0, map_dim_y):
                        no_probs[i][j] = prediction[5][0][i][j][0]
                        cold_probs[i][j] = prediction[5][0][i][j][1]
                        warm_probs[i][j] = prediction[5][0][i][j][2]
                if image == 0:
                    image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                    image_cold_probs[0: model_longitude_length - image_trim][:] = cold_probs[image_trim: model_longitude_length][:]
                    image_warm_probs[0: model_longitude_length - image_trim][:] = warm_probs[image_trim: model_longitude_length][:]
                elif image != num_images - 1:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                else:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    probs_ds = xr.Dataset(
                        {"no_probs": (("longitude", "latitude"), image_no_probs), "cold_probs": (("longitude", "latitude"), image_cold_probs),
                         "warm_probs": (("longitude", "latitude"), image_warm_probs)}, coords={"latitude": image_lats, "longitude": image_lons}).transpose()
                    prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, num_images, image_trim)

            elif front_types == 'SFOF':
                for i in range(0, map_dim_y):
                    for j in range(0, map_dim_x):
                        no_probs[i][j] = prediction[0][j][i][0]
                        stationary_probs[i][j] = prediction[0][j][i][1]
                        occluded_probs[i][j] = prediction[0][j][i][2]
                if image == 0:
                    image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                    image_stationary_probs[0: model_longitude_length - image_trim][:] = stationary_probs[image_trim: model_longitude_length][:]
                    image_occluded_probs[0: model_longitude_length - image_trim][:] = occluded_probs[image_trim: model_longitude_length][:]
                elif image != num_images - 1:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                else:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    probs_ds = xr.Dataset(
                        {"no_probs": (("longitude", "latitude"), image_no_probs), "stationary_probs": (("longitude", "latitude"), image_stationary_probs),
                         "occluded_probs": (("longitude", "latitude"), image_occluded_probs)}, coords={"latitude": image_lats, "longitude": image_lons}).transpose()
                    prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, num_images, image_trim)

            elif front_types == 'DL':
                for i in range(0, map_dim_y):
                    for j in range(0, map_dim_x):
                        no_probs[i][j] = prediction[0][j][i][0]
                        dryline_probs[i][j] = prediction[0][j][i][1]
                if image == 0:
                    image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                    image_dryline_probs[0: model_longitude_length - image_trim][:] = dryline_probs[image_trim: model_longitude_length][:]
                elif image != num_images - 1:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                else:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    probs_ds = xr.Dataset(
                        {"no_probs": (("latitude", "longitude"), no_probs), "dryline_probs": (("latitude", "longitude"), image_dryline_probs)},
                        coords={"latitude": lats, "longitude": lons})
                    prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, num_images, image_trim)

            elif front_types == 'ALL':
                for i in range(0, map_dim_y):
                    for j in range(0, map_dim_x):
                        no_probs[i][j] = prediction[0][j][i][0]
                        cold_probs[i][j] = prediction[0][j][i][1]
                        warm_probs[i][j] = prediction[0][j][i][2]
                        stationary_probs[i][j] = prediction[0][j][i][3]
                        occluded_probs[i][j] = prediction[0][j][i][4]
                        dryline_probs[i][j] = prediction[0][j][i][5]
                if image == 0:
                    image_no_probs[0: model_longitude_length - image_trim][:] = no_probs[image_trim: model_longitude_length][:]
                    image_cold_probs[0: model_longitude_length - image_trim][:] = cold_probs[image_trim: model_longitude_length][:]
                    image_warm_probs[0: model_longitude_length - image_trim][:] = warm_probs[image_trim: model_longitude_length][:]
                    image_stationary_probs[0: model_longitude_length - image_trim][:] = stationary_probs[image_trim: model_longitude_length][:]
                    image_occluded_probs[0: model_longitude_length - image_trim][:] = occluded_probs[image_trim: model_longitude_length][:]
                    image_dryline_probs[0: model_longitude_length - image_trim][:] = dryline_probs[image_trim: model_longitude_length][:]
                elif image != num_images - 1:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                    image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:image_spacing * image + lon_pixels_per_image][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:image_trim + lon_pixels_per_image][:]
                else:
                    image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_no_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], no_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_cold_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], cold_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_warm_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], warm_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_stationary_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], stationary_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_occluded_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], occluded_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])
                    image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:] = np.maximum(image_dryline_probs[int(image * image_spacing):int(image_spacing*(image-1)) + lon_pixels_per_image][:], dryline_probs[image_trim:image_trim + lon_pixels_per_image - image_spacing][:])

                    image_no_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = no_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_cold_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = cold_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_warm_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = warm_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_stationary_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = stationary_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_occluded_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = occluded_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    image_dryline_probs[int(image_spacing*(image-1)) + lon_pixels_per_image:][:] = dryline_probs[image_trim + lon_pixels_per_image - image_spacing:model_longitude_length-image_trim][:]
                    probs_ds = xr.Dataset(
                        {"no_probs": (("latitude", "longitude"), image_no_probs), "cold_probs": (("latitude", "longitude"), image_cold_probs),
                         "warm_probs": (("latitude", "longitude"), image_warm_probs), "stationary_probs": (("latitude", "longitude"), image_stationary_probs),
                         "occluded_probs": (("latitude", "longitude"), image_occluded_probs), "dryline_probs": (("latitude", "longitude"), image_dryline_probs)},
                        coords={"latitude": lats, "longitude": lons})
                    prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, num_images, image_trim)


def plot_performance_diagrams(model_dir, model_number, front_types):
    """
    Plots CSI performance diagram for different front types.

    Parameters
    ----------
    model_dir: str
        Main directory for the models.
    model_number: int
        Slurm job number for the model. This is the number in the model's filename.
    front_types: str
        Fronts in the data.
    """
    stats_25km = pd.read_pickle("%s/model_%d/model_%d_performance_stats_25km.pkl" % (model_dir, model_number, model_number))
    stats_100km = pd.read_pickle("%s/model_%d/model_%d_performance_stats_100km.pkl" % (model_dir, model_number, model_number))

    success_ratio_matrix, pod_matrix = np.linspace(0,1,100), np.linspace(0,1,100)
    x, y = np.meshgrid(success_ratio_matrix, pod_matrix)
    csi_matrix = (x ** -1 + y ** -1 - 1.) ** -1

    CSI_LEVELS = np.linspace(0,1,11)
    cmap = 'Blues'

    if front_types == 'CFWF' or front_types == 'ALL':
        POD_cold_25km = stats_25km['tp_cold']/(stats_25km['tp_cold'] + stats_25km['fn_cold'])
        SR_cold_25km = stats_25km['tp_cold']/(stats_25km['tp_cold'] + stats_25km['fp_cold'])
        CSI_cold_25km = stats_25km['tp_cold']/(stats_25km['tp_cold'] + stats_25km['fp_cold'] + stats_25km['fn_cold'])
        POD_cold_100km = stats_100km['tp_cold']/(stats_100km['tp_cold'] + stats_100km['fn_cold'])
        SR_cold_100km = stats_100km['tp_cold']/(stats_100km['tp_cold'] + stats_100km['fp_cold'])
        CSI_cold_100km = stats_100km['tp_cold']/(stats_100km['tp_cold'] + stats_100km['fp_cold'] + stats_100km['fn_cold'])
        POD_warm_25km = stats_25km['tp_warm']/(stats_25km['tp_warm'] + stats_25km['fn_warm'])
        SR_warm_25km = stats_25km['tp_warm']/(stats_25km['tp_warm'] + stats_25km['fp_warm'])
        CSI_warm_25km = stats_25km['tp_warm']/(stats_25km['tp_warm'] + stats_25km['fp_warm'] + stats_25km['fn_warm'])
        POD_warm_100km = stats_100km['tp_warm']/(stats_100km['tp_warm'] + stats_100km['fn_warm'])
        SR_warm_100km = stats_100km['tp_warm']/(stats_100km['tp_warm'] + stats_100km['fp_warm'])
        CSI_warm_100km = stats_100km['tp_warm']/(stats_100km['tp_warm'] + stats_100km['fp_warm'] + stats_100km['fn_warm'])

        fig = plt.figure()
        plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
        plt.colorbar(label='Critical Success Index (CSI)')
        plt.plot(SR_cold_25km, POD_cold_25km, 'r', label='25km boundary')
        plt.plot(SR_cold_25km[np.where(CSI_cold_25km == np.max(CSI_cold_25km))], POD_cold_25km[np.where(CSI_cold_25km == np.max(CSI_cold_25km))], color='red', marker='*', markersize=9)
        plt.text(SR_cold_25km[np.where(CSI_cold_25km == np.max(CSI_cold_25km))]+0.01, POD_cold_25km[np.where(CSI_cold_25km == np.max(CSI_cold_25km))], s=str('%.4f' % np.max(CSI_cold_25km)), color='red')
        plt.plot(SR_cold_100km, POD_cold_100km, color='green', label='100km boundary')
        plt.plot(SR_cold_100km[np.where(CSI_cold_100km == np.max(CSI_cold_100km))], POD_cold_100km[np.where(CSI_cold_100km == np.max(CSI_cold_100km))], color='green', marker='*', markersize=9)
        plt.text(SR_cold_100km[np.where(CSI_cold_100km == np.max(CSI_cold_100km))]+0.01, POD_cold_100km[np.where(CSI_cold_100km == np.max(CSI_cold_100km))], s=str('%.4f' % np.max(CSI_cold_100km)), color='green')
        plt.legend()
        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.title("Model %d Performance for Cold Fronts" % model_number)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.show()
        plt.savefig("%s/model_%d/model_%d_performance_cold.png" % (model_dir, model_number, model_number), bbox_inches='tight')
        plt.close()

        fig = plt.figure()
        plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
        plt.colorbar(label='Critical Success Index (CSI)')
        plt.plot(SR_warm_25km, POD_warm_25km, 'r', label='25km boundary')
        plt.plot(SR_warm_25km[np.where(CSI_warm_25km == np.max(CSI_warm_25km))], POD_warm_25km[np.where(CSI_warm_25km == np.max(CSI_warm_25km))], color='red', marker='*', markersize=9)
        plt.text(SR_warm_25km[np.where(CSI_warm_25km == np.max(CSI_warm_25km))]+0.01, POD_warm_25km[np.where(CSI_warm_25km == np.max(CSI_warm_25km))], s=str('%.4f' % np.max(CSI_warm_25km)), color='red')
        plt.plot(SR_warm_100km, POD_warm_100km, color='green', label='100km boundary')
        plt.plot(SR_warm_100km[np.where(CSI_warm_100km == np.max(CSI_warm_100km))], POD_warm_100km[np.where(CSI_warm_100km == np.max(CSI_warm_100km))], color='green', marker='*', markersize=9)
        plt.text(SR_warm_100km[np.where(CSI_warm_100km == np.max(CSI_warm_100km))]+0.01, POD_warm_100km[np.where(CSI_warm_100km == np.max(CSI_warm_100km))], s=str('%.4f' % np.max(CSI_warm_100km)), color='green')
        plt.legend()
        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.title("Model %d Performance for Warm Fronts" % model_number)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.show()
        plt.savefig("%s/model_%d/model_%d_performance_warm.png" % (model_dir, model_number, model_number), bbox_inches='tight')
        plt.close()

        if front_types == 'ALL':
            POD_stationary_25km = stats_25km['tp_stationary']/(stats_25km['tp_stationary'] + stats_25km['fn_stationary'])
            SR_stationary_25km = stats_25km['tp_stationary']/(stats_25km['tp_stationary'] + stats_25km['fp_stationary'])
            CSI_stationary_25km = stats_25km['tp_stationary']/(stats_25km['tp_stationary'] + stats_25km['fp_stationary'] + stats_25km['fn_stationary'])
            POD_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fn_stationary'])
            SR_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fp_stationary'])
            CSI_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fp_stationary'] + stats_100km['fn_stationary'])
            POD_occluded_25km = stats_25km['tp_occluded']/(stats_25km['tp_occluded'] + stats_25km['fn_occluded'])
            SR_occluded_25km = stats_25km['tp_occluded']/(stats_25km['tp_occluded'] + stats_25km['fp_occluded'])
            CSI_occluded_25km = stats_25km['tp_occluded']/(stats_25km['tp_occluded'] + stats_25km['fp_occluded'] + stats_25km['fn_occluded'])
            POD_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fn_occluded'])
            SR_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fp_occluded'])
            CSI_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fp_occluded'] + stats_100km['fn_occluded'])
            POD_dryline_25km = stats_25km['tp_dryline']/(stats_25km['tp_dryline'] + stats_25km['fn_dryline'])
            SR_dryline_25km = stats_25km['tp_dryline']/(stats_25km['tp_dryline'] + stats_25km['fp_dryline'])
            CSI_dryline_25km = stats_25km['tp_dryline']/(stats_25km['tp_dryline'] + stats_25km['fp_dryline'] + stats_25km['fn_dryline'])
            POD_dryline_100km = stats_100km['tp_dryline']/(stats_100km['tp_dryline'] + stats_100km['fn_dryline'])
            SR_dryline_100km = stats_100km['tp_dryline']/(stats_100km['tp_dryline'] + stats_100km['fp_dryline'])
            CSI_dryline_100km = stats_100km['tp_dryline']/(stats_100km['tp_dryline'] + stats_100km['fp_dryline'] + stats_100km['fn_dryline'])

            fig = plt.figure()
            plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
            plt.colorbar(label='Critical Success Index (CSI)')
            plt.plot(SR_stationary_25km, POD_stationary_25km, 'r', label='25km boundary')
            plt.plot(SR_stationary_25km[np.where(CSI_stationary_25km == np.max(CSI_stationary_25km))], POD_stationary_25km[np.where(CSI_stationary_25km == np.max(CSI_stationary_25km))], color='red', marker='*', markersize=9)
            plt.text(SR_stationary_25km[np.where(CSI_stationary_25km == np.max(CSI_stationary_25km))]+0.01, POD_stationary_25km[np.where(CSI_stationary_25km == np.max(CSI_stationary_25km))], s=str('%.4f' % np.max(CSI_stationary_25km)), color='red')
            plt.plot(SR_stationary_100km, POD_stationary_100km, color='green', label='100km boundary')
            plt.plot(SR_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))], POD_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))], color='green', marker='*', markersize=9)
            plt.text(SR_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))]+0.01, POD_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))], s=str('%.4f' % np.max(CSI_stationary_100km)), color='green')
            plt.legend()
            plt.xlabel("Success Ratio (1 - FAR)")
            plt.ylabel("Probability of Detection (POD)")
            plt.title("Model %d Performance for Stationary Fronts" % model_number)
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.show()
            plt.savefig("%s/model_%d/model_%d_performance_stationary.png" % (model_dir, model_number, model_number), bbox_inches='tight')
            plt.close()

            fig = plt.figure()
            plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
            plt.colorbar(label='Critical Success Index (CSI)')
            plt.plot(SR_occluded_25km, POD_occluded_25km, 'r', label='25km boundary')
            plt.plot(SR_occluded_25km[np.where(CSI_occluded_25km == np.max(CSI_occluded_25km))], POD_occluded_25km[np.where(CSI_occluded_25km == np.max(CSI_occluded_25km))], color='red', marker='*', markersize=9)
            plt.text(SR_occluded_25km[np.where(CSI_occluded_25km == np.max(CSI_occluded_25km))]+0.01, POD_occluded_25km[np.where(CSI_occluded_25km == np.max(CSI_occluded_25km))], s=str('%.4f' % np.max(CSI_occluded_25km)), color='red')
            plt.plot(SR_occluded_100km, POD_occluded_100km, color='green', label='100km boundary')
            plt.plot(SR_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))], POD_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))], color='green', marker='*', markersize=9)
            plt.text(SR_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))]+0.01, POD_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))], s=str('%.4f' % np.max(CSI_occluded_100km)), color='green')
            plt.legend()
            plt.xlabel("Success Ratio (1 - FAR)")
            plt.ylabel("Probability of Detection (POD)")
            plt.title("Model %d Performance for Occluded Fronts" % model_number)
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.show()
            plt.savefig("%s/model_%d/model_%d_performance_occluded.png" % (model_dir, model_number, model_number),bbox_inches='tight')
            plt.close()

            fig = plt.figure()
            plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
            plt.colorbar(label='Critical Success Index (CSI)')
            plt.plot(SR_dryline_25km, POD_dryline_25km, 'r', label='25km boundary')
            plt.plot(SR_dryline_25km[np.where(CSI_dryline_25km == np.max(CSI_dryline_25km))], POD_dryline_25km[np.where(CSI_dryline_25km == np.max(CSI_dryline_25km))], color='red', marker='*', markersize=9)
            plt.text(SR_dryline_25km[np.where(CSI_dryline_25km == np.max(CSI_dryline_25km))]+0.01, POD_dryline_25km[np.where(CSI_dryline_25km == np.max(CSI_dryline_25km))], s=str('%.4f' % np.max(CSI_dryline_25km)), color='red')
            plt.plot(SR_dryline_100km, POD_dryline_100km, color='green', label='100km boundary')
            plt.plot(SR_dryline_100km[np.where(CSI_dryline_100km == np.max(CSI_dryline_100km))], POD_dryline_100km[np.where(CSI_dryline_100km == np.max(CSI_dryline_100km))], color='green', marker='*', markersize=9)
            plt.text(SR_dryline_100km[np.where(CSI_dryline_100km == np.max(CSI_dryline_100km))]+0.01, POD_dryline_100km[np.where(CSI_dryline_100km == np.max(CSI_dryline_100km))], s=str('%.4f' % np.max(CSI_dryline_100km)), color='green')
            plt.legend()
            plt.xlabel("Success Ratio (1 - FAR)")
            plt.ylabel("Probability of Detection (POD)")
            plt.title("Model %d Performance for Dryline Fronts" % model_number)
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.show()
            plt.savefig("%s/model_%d/model_%d_performance_dryline.png" % (model_dir, model_number, model_number),bbox_inches='tight')
            plt.close()

    elif front_types == 'SFOF':
        POD_stationary_25km = stats_25km['tp_stationary']/(stats_25km['tp_stationary'] + stats_25km['fn_stationary'])
        SR_stationary_25km = stats_25km['tp_stationary']/(stats_25km['tp_stationary'] + stats_25km['fp_stationary'])
        CSI_stationary_25km = stats_25km['tp_stationary']/(stats_25km['tp_stationary'] + stats_25km['fp_stationary'] + stats_25km['fn_stationary'])
        POD_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fn_stationary'])
        SR_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fp_stationary'])
        CSI_stationary_100km = stats_100km['tp_stationary']/(stats_100km['tp_stationary'] + stats_100km['fp_stationary'] + stats_100km['fn_stationary'])
        POD_occluded_25km = stats_25km['tp_occluded']/(stats_25km['tp_occluded'] + stats_25km['fn_occluded'])
        SR_occluded_25km = stats_25km['tp_occluded']/(stats_25km['tp_occluded'] + stats_25km['fp_occluded'])
        CSI_occluded_25km = stats_25km['tp_occluded']/(stats_25km['tp_occluded'] + stats_25km['fp_occluded'] + stats_25km['fn_occluded'])
        POD_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fn_occluded'])
        SR_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fp_occluded'])
        CSI_occluded_100km = stats_100km['tp_occluded']/(stats_100km['tp_occluded'] + stats_100km['fp_occluded'] + stats_100km['fn_occluded'])

        fig = plt.figure()
        plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
        plt.colorbar(label='Critical Success Index (CSI)')
        plt.plot(SR_stationary_25km, POD_stationary_25km, 'r', label='25km boundary')
        plt.plot(SR_stationary_25km[np.where(CSI_stationary_25km == np.max(CSI_stationary_25km))], POD_stationary_25km[np.where(CSI_stationary_25km == np.max(CSI_stationary_25km))], color='red', marker='*', markersize=9)
        plt.text(SR_stationary_25km[np.where(CSI_stationary_25km == np.max(CSI_stationary_25km))]+0.01, POD_stationary_25km[np.where(CSI_stationary_25km == np.max(CSI_stationary_25km))], s=str('%.4f' % np.max(CSI_stationary_25km)), color='red')
        plt.plot(SR_stationary_100km, POD_stationary_100km, color='green', label='100km boundary')
        plt.plot(SR_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))], POD_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))], color='green', marker='*', markersize=9)
        plt.text(SR_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))]+0.01, POD_stationary_100km[np.where(CSI_stationary_100km == np.max(CSI_stationary_100km))], s=str('%.4f' % np.max(CSI_stationary_100km)), color='green')
        plt.legend()
        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.title("Model %d Performance for Stationary Fronts" % model_number)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.show()
        plt.savefig("%s/model_%d/model_%d_performance_stationary.png" % (model_dir, model_number, model_number),bbox_inches='tight')
        plt.close()

        fig = plt.figure()
        plt.contourf(x, y, csi_matrix, CSI_LEVELS, cmap=cmap)
        plt.colorbar(label='Critical Success Index (CSI)')
        plt.plot(SR_occluded_25km, POD_occluded_25km, 'r', label='25km boundary')
        plt.plot(SR_occluded_25km[np.where(CSI_occluded_25km == np.max(CSI_occluded_25km))], POD_occluded_25km[np.where(CSI_occluded_25km == np.max(CSI_occluded_25km))], color='red', marker='*', markersize=9)
        plt.text(SR_occluded_25km[np.where(CSI_occluded_25km == np.max(CSI_occluded_25km))]+0.01, POD_occluded_25km[np.where(CSI_occluded_25km == np.max(CSI_occluded_25km))], s=str('%.4f' % np.max(CSI_occluded_25km)), color='red')
        plt.plot(SR_occluded_100km, POD_occluded_100km, color='green', label='100km boundary')
        plt.plot(SR_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))], POD_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))], color='green', marker='*', markersize=9)
        plt.text(SR_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))]+0.01, POD_occluded_100km[np.where(CSI_occluded_100km == np.max(CSI_occluded_100km))], s=str('%.4f' % np.max(CSI_occluded_100km)), color='green')
        plt.legend()
        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.title("Model %d Performance for Occluded Fronts" % model_number)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.show()
        plt.savefig("%s/model_%d/model_%d_performance_occluded.png" % (model_dir, model_number, model_number),bbox_inches='tight')
        plt.close()


def prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion, num_images, image_trim):
    """
    Function that uses generated predictions to make probability maps along with the 'true' fronts and saves out the
    subplots.

    Parameters
    ----------
    fronts: DataArray
        Xarray DataArray containing the 'true' front data.
    probs_ds: Dataset
        Xarray dataset containing prediction (probability) data for warm and cold fronts.
    time: str
        Timestring for the prediction plot title.
    model_number: int
        Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        Main directory for the models.
    front_types: str
        Fronts in the data.
    pixel_expansion: int
        Number of pixels to expand the fronts by in all directions.
    num_images: int
        Number of images to stitch together for the final probability maps.
    image_trim: int
        Number of pixels to trim each image by in the longitude dimension before averaging the overlapping pixels.
    """
    extent = [220, 300, 29, 53]
    crs = ccrs.LambertConformal(central_longitude=250)

    # Create custom colorbar for the different front types of the 'truth' plot
    cmap = mpl.colors.ListedColormap(["white","blue","red",'green','purple','orange'], name='from_list', N=None)
    norm = mpl.colors.Normalize(vmin=0,vmax=6)

    cold_norm = mpl.colors.Normalize(vmin=0.2, vmax=0.6)
    warm_norm = mpl.colors.Normalize(vmin=0.2, vmax=0.6)

    if pixel_expansion == 1:
        fronts = ope(fronts)  # 1-pixel expansion
    elif pixel_expansion == 2:
        fronts = ope(ope(fronts))  # 2-pixel expansion

    probs_ds = xr.where(probs_ds > 0.2, probs_ds, float("NaN"))

    if front_types == 'CFWF':
        fig, axarr = plt.subplots(1, 2, figsize=(20, 5), subplot_kw={'projection': crs}, gridspec_kw={'width_ratios': [1,1.3]})
        axlist = axarr.flatten()
        for ax in axlist:
            fplot.plot_background(ax, extent)
        fronts.identifier.plot(ax=axlist[0], cmap=cmap, norm=norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[0].title.set_text("%s Truth" % time)
        probs_ds.cold_probs.plot(ax=axlist[1], cmap="Blues", norm=cold_norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        probs_ds.warm_probs.plot(ax=axlist[1], cmap="Reds", norm=warm_norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[1].title.set_text("%s Front probabilities (images=%d, trim=%d)" % (time, num_images, image_trim))
        plt.savefig('%s/model_%d/predictions/model_%d_%s_plot_%dimages_%dtrim.png' % (model_dir, model_number, model_number,
            time, num_images, image_trim), bbox_inches='tight', dpi=300)
        plt.close()

    elif front_types == 'SFOF':
        fig, axarr = plt.subplots(2, 1, figsize=(12, 14), subplot_kw={'projection': crs})
        axlist = axarr.flatten()
        for ax in axlist:
            fplot.plot_background(ax, extent)
        fronts.identifier.plot(ax=axlist[0], cmap=cmap, norm=norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[0].title.set_text("%s Truth" % time)
        probs_ds.stationary_probs.plot(ax=axlist[1], cmap='Greens', x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[1].title.set_text("%s SF probability" % time)
        probs_ds.occluded_probs.plot(ax=axlist[1], cmap='Purples', x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[1].title.set_text("%s Front probabilities (images=%d, trim=%d)" % (time, num_images, image_trim))
        plt.savefig('%s/model_%d/predictions/model_%d_%s_plot_%dimages_%dtrim.png' % (model_dir, model_number, model_number,
            time, num_images, image_trim), bbox_inches='tight', dpi=300)
        plt.close()

    elif front_types == 'DL':
        fig, axarr = plt.subplots(2, 1, figsize=(12, 14), subplot_kw={'projection': crs})
        axlist = axarr.flatten()
        for ax in axlist:
            fplot.plot_background(ax, extent)
        fronts.identifier.plot(ax=axlist[0], cmap=cmap, norm=norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[0].title.set_text("%s Truth" % time)
        probs_ds.dryline_probs.plot(ax=axlist[1], cmap='Oranges', x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[1].title.set_text("%s Front probabilities (images=%d, trim=%d)" % (time, num_images, image_trim))
        plt.savefig('%s/model_%d/predictions/model_%d_%s_plot_%dimages_%dtrim.png' % (model_dir, model_number, model_number,
            time, num_images, image_trim), bbox_inches='tight', dpi=300)
        plt.close()

    elif front_types == 'ALL':
        fig, axarr = plt.subplots(2, 1, figsize=(12, 14), subplot_kw={'projection': crs})
        axlist = axarr.flatten()
        for ax in axlist:
            fplot.plot_background(ax, extent)
        fronts.identifier.plot(ax=axlist[0], cmap=cmap, norm=norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[0].title.set_text("%s Truth" % time)
        probs_ds.cold_probs.plot(ax=axlist[1], cmap='Blues', x='longitude', y='latitude', transform=ccrs.PlateCarree())
        probs_ds.warm_probs.plot(ax=axlist[1], cmap='Reds', x='longitude', y='latitude', transform=ccrs.PlateCarree())
        probs_ds.stationary_probs.plot(ax=axlist[1], cmap='Greens', x='longitude', y='latitude', transform=ccrs.PlateCarree())
        probs_ds.occluded_probs.plot(ax=axlist[1], cmap='Purples', x='longitude', y='latitude', transform=ccrs.PlateCarree())
        probs_ds.dryline_probs.plot(ax=axlist[1], cmap='Oranges', x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[1].title.set_text("%s Front probabilities (images=%d, trim=%d)" % (time, num_images, image_trim))
        plt.savefig('%s/model_%d/predictions/model_%d_%s_plot_%dimages_%dtrim.png' % (model_dir, model_number, model_number,
            time, num_images, image_trim), bbox_inches='tight', dpi=300)
        plt.close()


def learning_curve(model_number, model_dir, loss, fss_mask_size, fss_c, metric):
    """
    Function that plots learning curves for the specified model.

    Parameters
    ----------
    model_number: int
        Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        Main directory for the models.
    loss: str
        Loss function for the U-Net.
    fss_mask_size: int
        Size of the mask for the FSS loss function.
    fss_c: float
        C hyperparameter for the FSS loss' sigmoid function.
    metric: str
        Metric used for evaluating the U-Net during training.
    """
    with open("%s/model_%d/model_%d_history.csv" % (model_dir, model_number, model_number), 'rb') as f:
        history = pd.read_csv(f)

    if loss == 'fss':
        loss_title = 'Fractions Skill Score (mask=%d, c=%.1f)' % (fss_mask_size, fss_c)
    elif loss == 'bss':
        loss_title = 'Brier Skill Score'
    elif loss == 'cce':
        loss_title = 'Categorical Cross-Entropy'
    elif loss == 'dice':
        loss_title = 'Dice coefficient'
    elif loss == 'tversky':
        loss_title = 'Tversky coefficient'

    if metric == 'fss':
        metric_title = 'Fractions Skill Score (mask=%d, c=%.1f)' % (fss_mask_size, fss_c)
        metric_string = 'FSS_loss'
    elif metric == 'bss':
        metric_title = 'Brier Skill Score'
        metric_string = 'brier_skill_score'
    elif metric == 'auc':
        metric_title = 'Area Under the Curve'
        metric_string = 'auc'
    elif metric == 'dice':
        metric_title = 'Dice coefficient'
        metric_string = 'dice'
    elif metric == 'tversky':
        metric_title = 'Tversky coefficient'
        metric_string = 'tversky'

    plt.subplots(1, 2, figsize=(14, 7), dpi=500)

    plt.subplot(1, 2, 1)
    plt.title("Loss: %s" % loss_title)
    plt.grid()

    """
    Loss curves: replace the line(s) below this block according to your U-Net type.
    
    ### U-Net ###
    plt.plot(history['loss'], color='black')
    
    ### U-Net 3+ ### (Make sure you know if you need to remove or add lines - refer on your U-Net's architecture)
    plt.plot(history['unet_output_sup0_activation_loss'], label='sup0')
    plt.plot(history['unet_output_sup1_activation_loss'], label='sup1')
    plt.plot(history['unet_output_sup2_activation_loss'], label='sup2')
    plt.plot(history['unet_output_sup3_activation_loss'], label='sup3')
    plt.plot(history['unet_output_sup4_activation_loss'], label='sup4')
    plt.plot(history['unet_output_final_activation_loss'], label='final')
    plt.plot(history['loss'], label='total', color='black')

    ### U-Net 3+ (3D) ### (Make sure you know if you need to remove or add lines - refer on your U-Net's architecture)
    plt.plot(history['softmax_loss'], label='Encoder 6')
    plt.plot(history['softmax_1_loss'], label='Decoder 5')
    plt.plot(history['softmax_2_loss'], label='Decoder 4')
    plt.plot(history['softmax_3_loss'], label='Decoder 3')
    plt.plot(history['softmax_4_loss'], label='Decoder 2')
    plt.plot(history['softmax_5_loss'], label='Decoder 1 (final)', color='black')
    plt.plot(history['loss'], label='total', color='black')
    """
    plt.plot(history['unet_output_sup0_activation_loss'], label='sup0')
    plt.plot(history['unet_output_sup1_activation_loss'], label='sup1')
    plt.plot(history['unet_output_sup2_activation_loss'], label='sup2')
    plt.plot(history['unet_output_sup3_activation_loss'], label='sup3')
    plt.plot(history['unet_output_sup4_activation_loss'], label='sup4')
    plt.plot(history['unet_output_final_activation_loss'], label='final')
    plt.plot(history['loss'], label='total', color='black')

    plt.legend()
    plt.xlim(xmin=0)
    plt.xlabel('Epochs')
    plt.ylim(ymin=1e-6, ymax=1e-4)  # Limits of the loss function graph, adjust these as needed
    plt.yscale('log')  # Turns y-axis into a logarithmic scale. Useful if loss functions appear as very sharp curves.

    plt.subplot(1, 2, 2)
    plt.title("Metric: %s" % metric_title)
    plt.grid()

    """
    Other metric curves: replace the line(s) below this block according to your U-Net type.
    
    ### U-Net ###
    plt.plot(history[metric_string], 'r')
    
    ### U-Net 3+ (2D) ### (Make sure you know if you need to remove or add lines - refer on your U-Net's architecture)
    plt.plot(history['unet_output_sup0_activation_%s' % metric_string], label='sup0')
    plt.plot(history['unet_output_sup1_activation_%s' % metric_string], label='sup1')
    plt.plot(history['unet_output_sup2_activation_%s' % metric_string], label='sup2')
    plt.plot(history['unet_output_sup3_activation_%s' % metric_string], label='sup3')
    plt.plot(history['unet_output_sup4_activation_%s' % metric_string], label='sup4')
    plt.plot(history['unet_output_final_activation_%s' % metric_string], label='final', color='black')

    ### U-Net 3+ (3D) ### (Make sure you know if you need to remove or add lines - refer on your U-Net's architecture)
    plt.plot(history['softmax_%s' % metric_string], label='Encoder 6')
    plt.plot(history['softmax_1_%s' % metric_string], label='Decoder 5')
    plt.plot(history['softmax_2_%s' % metric_string], label='Decoder 4')
    plt.plot(history['softmax_3_%s' % metric_string], label='Decoder 3')
    plt.plot(history['softmax_4_%s' % metric_string], label='Decoder 2')
    plt.plot(history['softmax_5_%s' % metric_string], label='Decoder 1 (final)', color='black')
    """
    plt.plot(history['unet_output_sup0_activation_%s' % metric_string], label='sup0')
    plt.plot(history['unet_output_sup1_activation_%s' % metric_string], label='sup1')
    plt.plot(history['unet_output_sup2_activation_%s' % metric_string], label='sup2')
    plt.plot(history['unet_output_sup3_activation_%s' % metric_string], label='sup3')
    plt.plot(history['unet_output_sup4_activation_%s' % metric_string], label='sup4')
    plt.plot(history['unet_output_final_activation_%s' % metric_string], label='final', color='black')

    plt.legend()
    plt.xlim(xmin=0)
    plt.xlabel('Epochs')
    plt.ylim(ymin=0.99, ymax=1)  # Limits of the metric graph, adjust as needed
    plt.savefig("%s/model_%d/model_%d_learning_curve.png" % (model_dir, model_number, model_number), bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    """ Main arguments """
    parser.add_argument('--calculate_performance_stats', type=bool, required=False,
                        help='Are you calculating performance stats for a model?')
    parser.add_argument('--day', type=int, required=False, help='Day for the prediction.')
    parser.add_argument('--domain', type=str, required=False, help='Domain of the data.')
    parser.add_argument('--file_dimensions', type=int, nargs=2, required=False,
                        help='Dimensions of the file size. Two integers need to be passed.')
    parser.add_argument('--find_matches', type=bool, required=False, help='Find matches for stitching predictions?')
    parser.add_argument('--front_types', type=str, required=False,
                        help='Front format of the file. If your files contain warm and cold fronts, pass this argument'
                             'as CFWF. If your files contain only drylines, pass this argument as DL. If your files '
                             'contain all fronts, pass this argument as ALL.')
    parser.add_argument('--fss_c', type=float, required=False, help="C hyperparameter for the FSS loss' sigmoid function.")
    parser.add_argument('--fss_mask_size', type=int, required=False, help='Mask size for the FSS loss function.')
    parser.add_argument('--hour', type=int, required=False, help='Hour for the prediction.')
    parser.add_argument('--image_trim', type=int, required=False,
                        help='Number of pixels to trim the images for stitching before averaging overlapping pixels.')
    parser.add_argument('--learning_curve', type=bool, required=False, help='Plot learning curve?')
    parser.add_argument('--longitude_domain_length', type=int, required=False,
                        help='Number of pixels in the longitude dimension of the final stitched map.')
    parser.add_argument('--loss', type=str, required=False, help='Loss function used for training the U-Net.')
    parser.add_argument('--make_prediction', type=bool, required=False,
                        help='Make a prediction for a specific date and time?')
    parser.add_argument('--make_random_predictions', type=bool, required=False, help='Generate prediction plots?')
    parser.add_argument('--metric', type=str, required=False, help='Metric used for evaluating the U-Net during training.')
    parser.add_argument('--model_longitude_length', type=int, required=False,
                        help="Length of the longitude dimension of the model's prediction.")
    parser.add_argument('--model_dir', type=str, required=False, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, required=False, help='Model number.')
    parser.add_argument('--month', type=int, required=False, help='Month for the prediction.')
    parser.add_argument('--num_dimensions', type=int, required=False,
                        help='Number of dimensions of the U-Net convolutions, maxpooling, and upsampling. (2 or 3)')
    parser.add_argument('--num_images', type=int, required=False, help='Number of images to stitch together for each prediction.')
    parser.add_argument('--num_variables', type=int, required=False, help='Number of variables in the variable datasets.')
    parser.add_argument('--normalization_method', type=int, required=False,
                        help='Normalization method for the data. 0 - No normalization, 1 - Min-max normalization, '
                             '2 - Mean normalization')
    parser.add_argument('--pixel_expansion', type=int, required=False, help='Number of pixels to expand the fronts by.')
    parser.add_argument('--plot_performance_diagrams', type=bool, required=False, help='Plot performance diagrams for a model?')
    parser.add_argument('--predictions', type=int, required=False, help='Number of predictions to make.')
    parser.add_argument('--test_year', type=int, required=False, help='Test year for cross-validating the current model.')
    parser.add_argument('--year', type=int, required=False, help='Year for the prediction.')

    args = parser.parse_args()

    print(args)

    if args.loss == 'fss':
        if args.fss_c is None or args.fss_mask_size is None:
            raise errors.MissingArgumentError("If loss is fss, the following arguments must be passed: fss_c, fss_mask_size")
    else:
        if args.fss_c is not None or args.fss_mask_size is not None:
            raise errors.ExtraArgumentError("loss is not fss but one or more of the following arguments was provided: "
                                            "fss_c, fss_mask_size")

    if args.calculate_performance_stats is True:
        if args.model_number is None or args.model_dir is None or args.num_variables is None or args.num_dimensions is None or \
            args.front_types is None or args.domain is None or args.file_dimensions is None or args.normalization_method is None or \
            args.loss is None or args.pixel_expansion is None or args.metric is None or args.num_images is None or \
            args.longitude_domain_length is None or args.image_trim is None:
            raise errors.MissingArgumentError("If calculate_performance_stats is True, the following arguments must be passed: "
                "domain, file_dimensions, front_types, image_trim, loss, metric, model_dir, model_number, normalization_method, "
                "num_dimensions, num_images, num_variables, pixel_expansion")
        else:
            calculate_performance_stats(args.model_number, args.model_dir, args.num_variables, args.num_dimensions, args.front_types,
                args.domain, args.file_dimensions, args.test_year, args.normalization_method, args.loss, args.fss_mask_size,
                args.fss_c, args.pixel_expansion, args.metric, args.num_images, args.longitude_domain_length, args.image_trim)
    else:
        if args.test_year is not None:
            raise errors.ExtraArgumentError("cross_validate is False but the following argument was provided: test_year")

    if args.find_matches is True:
        if args.longitude_domain_length is None or args.model_longitude_length is None:
            raise errors.MissingArgumentError("If find_matches is True, the following arguments must be provided: "
                                              "longitude_domain_length, model_longitude_length")
        else:
            find_matches_for_domain(args.longitude_domain_length, args.model_longitude_length)

    if args.learning_curve is True:
        if args.model_number is None or args.model_dir is None or args.loss is None or args.metric is None:
            raise errors.MissingArgumentError("If learning_curve is True, the following arguments must be passed: "
                                              "loss, metric, model_dir, model_number")
        else:
            learning_curve(args.model_number, args.model_dir, args.loss, args.fss_mask_size, args.fss_c, args.metric)

    if args.make_prediction is True:
        if args.model_number is None or args.model_dir is None or args.num_variables is None or args.num_dimensions is None or \
            args.front_types is None or args.domain is None or args.file_dimensions is None or args.normalization_method is None or \
            args.loss is None or args.pixel_expansion is None or args.metric is None or args.num_images is None or \
            args.longitude_domain_length is None or args.image_trim is None or args.year is None or args.month is None or \
            args.day is None or args.hour is None:
            raise errors.MissingArgumentError("If predict is True, the following arguments must be passed: "
                "day, domain, file_dimensions, front_types, hour, longitude_domain_length, loss, metric, model_dir, model_number, "
                "month, normalization_method, num_dimensions, num_images, num_variables, pixel_expansion, image_trim, "
                "year")
        else:
            fronts_files_list, variables_files_list = fm.load_file_lists(args.num_variables, args.front_types, args.domain,
                args.file_dimensions)
            make_prediction(args.model_number, args.model_dir, fronts_files_list, variables_files_list, args.normalization_method,
                args.loss, args.fss_mask_size, args.fss_c, args.front_types, args.pixel_expansion, args.metric, args.num_dimensions,
                args.num_images, args.longitude_domain_length, args.image_trim, args.year, args.month, args.day, args.hour)

    if args.make_random_predictions is True:
        if args.model_number is None or args.model_dir is None or args.num_variables is None or args.num_dimensions is None or \
            args.front_types is None or args.domain is None or args.file_dimensions is None or args.predictions is None or \
            args.normalization_method is None or args.loss is None or args.pixel_expansion is None or args.metric is None or \
            args.num_images is None or args.longitude_domain_length is None or args.image_trim is None:
            raise errors.MissingArgumentError("If make_random_predictions is True, the following arguments must be passed: "
                "domain, file_dimensions, front_types, image_trim, longitude_domain_length, loss, metric, model_dir, model_number, "
                "normalization_method, num_dimensions, num_images, num_variables, pixel_expansion, predictions")
        else:
            fronts_files_list, variables_files_list = fm.load_file_lists(args.num_variables, args.front_types, args.domain,
                args.file_dimensions)
            make_random_predictions(args.model_number, args.model_dir, fronts_files_list, variables_files_list, args.predictions,
                args.normalization_method, args.loss, args.fss_mask_size, args.fss_c, args.front_types, args.pixel_expansion,
                args.metric, args.num_dimensions, args.num_images, args.longitude_domain_length, args.image_trim)
    else:
        if args.predictions is not None:
            raise errors.ExtraArgumentError("make_random_predictions is False but the following argument was provided: predictions")

    if args.plot_performance_diagrams is True:
        if args.model_dir is None or args.model_number is None or args.front_types is None:
            raise errors.MissingArgumentError("If plot_performance_diagrams is True, the following arguments must be passed: "
                "front_types, model_dir, model_number")
        else:
            plot_performance_diagrams(args.model_dir, args.model_number, args.front_types)
