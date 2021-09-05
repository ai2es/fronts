"""
Functions used for evaluating a U-Net model. The functions can be used to make predictions or plot learning curves.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 9/5/2021 3:30 PM CDT
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


def cross_validate(model_number, model_dir, num_variables, num_dimensions, front_types, domain, file_dimensions, test_year,
                   normalization_method, loss, fss_mask_size, fss_c, pixel_expansion, metric):
    """
    Function that calculates, CSI, FB, POD, and POFD for thresholds from 1 to 100 percent for each front type.

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
    """

    front_files_test, variable_files_test = fm.load_test_files(num_variables, front_types, domain, file_dimensions, test_year)
    print("Front file count:", len(front_files_test))
    print("Variable file count:", len(variable_files_test))

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

    CSI_cold = np.zeros(shape=[100])
    FB_cold = np.zeros(shape=[100])
    POFD_cold = np.zeros(shape=[100])
    POD_cold = np.zeros(shape=[100])
    SR_cold = np.zeros(shape=[100])

    CSI_warm = np.zeros(shape=[100])
    FB_warm = np.zeros(shape=[100])
    POFD_warm = np.zeros(shape=[100])
    POD_warm = np.zeros(shape=[100])
    SR_warm = np.zeros(shape=[100])
    SR_warm_errors = 0

    CSI_stationary = np.zeros(shape=[100])
    FB_stationary = np.zeros(shape=[100])
    POFD_stationary = np.zeros(shape=[100])
    POD_stationary = np.zeros(shape=[100])
    SR_stationary = np.zeros(shape=[100])

    CSI_occluded = np.zeros(shape=[100])
    FB_occluded = np.zeros(shape=[100])
    POFD_occluded = np.zeros(shape=[100])
    POD_occluded = np.zeros(shape=[100])
    SR_occluded = np.zeros(shape=[100])

    CSI_dryline = np.zeros(shape=[100])
    FB_dryline = np.zeros(shape=[100])
    POFD_dryline = np.zeros(shape=[100])
    POD_dryline = np.zeros(shape=[100])
    SR_dryline = np.zeros(shape=[100])

    for x in range(len(front_files_test)):

        print("Prediction %d/%d....0/3" % (x+1, len(front_files_test)), end='\r')

        # Open random pair of files
        index = x
        fronts_filename = front_files_test[index]
        variables_filename = variable_files_test[index]
        fronts_ds = pd.read_pickle(fronts_filename)
        for i in range(pixel_expansion):
            fronts_ds = ope(fronts_ds)

        lon_indices = [21,80,139]
        image_no_probs = np.empty([238,128])
        image_cold_probs = np.empty([238,128])
        image_warm_probs = np.empty([238,128])
        image_stationary_probs = np.empty([238,128])
        image_occluded_probs = np.empty([238,128])
        image_dryline_probs = np.empty([238,128])
        fronts = fronts_ds.sel(longitude=fronts_ds.longitude.values[25:263], latitude=fronts_ds.latitude.values[0:128])
        image_lats = fronts_ds.latitude.values[0:128]
        image_lons = fronts_ds.longitude.values[25:263]

        for image in range(3):
            print("Prediction %d/%d....%d/3" % (x+1, len(front_files_test), image+1), end='\r')
            lon_index = lon_indices[image]
            lat_index = 0
            variable_ds = pd.read_pickle(variables_filename)
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
            #print(time)

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
                    image_no_probs[4:128][:] = no_probs[4:128][:]
                    image_cold_probs[4:128][:] = cold_probs[4:128][:]
                    image_warm_probs[4:128][:] = warm_probs[4:128][:]
                elif image == 1:
                    image_no_probs[66:122][:] = np.maximum(image_no_probs[66:122][:], no_probs[7:63][:])
                    image_cold_probs[66:122][:] = np.maximum(image_cold_probs[66:122][:], cold_probs[7:63][:])
                    image_warm_probs[66:122][:] = np.maximum(image_warm_probs[66:122][:], warm_probs[7:63][:])

                    image_no_probs[122:181][:] = no_probs[63:122][:]
                    image_cold_probs[122:181][:] = cold_probs[63:122][:]
                    image_warm_probs[122:181][:] = warm_probs[63:122][:]
                elif image == 2:
                    image_no_probs[128:193][:] = np.maximum(image_no_probs[128:193][:], no_probs[10:75][:])
                    image_cold_probs[128:193][:] = np.maximum(image_cold_probs[128:193][:], cold_probs[10:75][:])
                    image_warm_probs[128:193][:] = np.maximum(image_warm_probs[128:193][:], warm_probs[10:75][:])

                    image_no_probs[193:][:] = no_probs[79:124][:]
                    image_cold_probs[193:][:] = cold_probs[79:124][:]
                    image_warm_probs[193:][:] = warm_probs[79:124][:]
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
                        tp_cold = len(np.where(t_cold_probs > thresholds[i])[0])
                        tn_cold = len(np.where((f_cold_probs < thresholds[i]) & (f_cold_probs != 0))[0])
                        fp_cold = len(np.where(f_cold_probs > thresholds[i])[0])
                        fn_cold = len(np.where((t_cold_probs < thresholds[i]) & (t_cold_probs != 0))[0])
                        tp_warm = len(np.where(t_warm_probs > thresholds[i])[0])
                        tn_warm = len(np.where((f_warm_probs < thresholds[i]) & (f_warm_probs != 0))[0])
                        fp_warm = len(np.where(f_warm_probs > thresholds[i])[0])
                        fn_warm = len(np.where((t_warm_probs < thresholds[i]) & (t_warm_probs != 0))[0])

                        try:
                            CSI_cold[i] += tp_cold/(tp_cold + fp_cold + fn_cold)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            FB_cold[i] += (tp_cold + fp_cold)/(tp_cold + fn_cold)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            POFD_cold[i] += fp_cold/(fp_cold + fn_cold)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            POD_cold[i] += tp_cold/(tp_cold + fn_cold)/len(front_files_test)
                        except ZeroDivisionError:
                            pass

                        try:
                            CSI_warm[i] += tp_warm/(tp_warm + fp_warm + fn_warm)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            FB_warm[i] += (tp_warm + fp_warm)/(tp_warm + fn_warm)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            POFD_warm[i] += fp_warm/(fp_warm + fn_warm)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            POD_warm[i] += tp_warm/(tp_warm + fn_warm)/len(front_files_test)
                        except ZeroDivisionError:
                            pass

                    print("Prediction %d/%d....done" % (x+1, len(front_files_test)))


            elif front_types == 'SFOF':
                for i in range(0, map_dim_x):
                    for j in range(0, map_dim_y):
                        no_probs[i][j] = prediction[5][0][i][j][0]
                        stationary_probs[i][j] = prediction[5][0][i][j][1]
                        occluded_probs[i][j] = prediction[5][0][i][j][2]
                if image == 0:
                    image_no_probs[4:128][:] = no_probs[4:128][:]
                    image_stationary_probs[4:128][:] = stationary_probs[4:128][:]
                    image_occluded_probs[4:128][:] = occluded_probs[4:128][:]
                elif image == 1:
                    image_no_probs[66:122][:] = np.maximum(image_no_probs[66:122][:], no_probs[7:63][:])
                    image_stationary_probs[66:122][:] = np.maximum(image_stationary_probs[66:122][:], stationary_probs[7:63][:])
                    image_occluded_probs[66:122][:] = np.maximum(image_occluded_probs[66:122][:], occluded_probs[7:63][:])

                    image_no_probs[122:181][:] = no_probs[63:122][:]
                    image_stationary_probs[122:181][:] = stationary_probs[63:122][:]
                    image_occluded_probs[122:181][:] = occluded_probs[63:122][:]
                elif image == 2:
                    image_no_probs[128:193][:] = np.maximum(image_no_probs[128:193][:], no_probs[10:75][:])
                    image_stationary_probs[128:193][:] = np.maximum(image_stationary_probs[128:193][:], stationary_probs[10:75][:])
                    image_occluded_probs[128:193][:] = np.maximum(image_occluded_probs[128:193][:], occluded_probs[10:75][:])

                    image_no_probs[193:][:] = no_probs[79:124][:]
                    image_stationary_probs[193:][:] = stationary_probs[79:124][:]
                    image_occluded_probs[193:][:] = occluded_probs[79:124][:]
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
                        tp_stationary = len(np.where(t_stationary_probs > thresholds[i])[0])
                        tn_stationary = len(np.where((f_stationary_probs < thresholds[i]) & (f_stationary_probs != 0))[0])
                        fp_stationary = len(np.where(f_stationary_probs > thresholds[i])[0])
                        fn_stationary = len(np.where((t_stationary_probs < thresholds[i]) & (t_stationary_probs != 0))[0])
                        tp_occluded = len(np.where(t_occluded_probs > thresholds[i])[0])
                        tn_occluded = len(np.where((f_occluded_probs < thresholds[i]) & (f_occluded_probs != 0))[0])
                        fp_occluded = len(np.where(f_occluded_probs > thresholds[i])[0])
                        fn_occluded = len(np.where((t_occluded_probs < thresholds[i]) & (t_occluded_probs != 0))[0])

                        try:
                            CSI_stationary[i] += tp_stationary/(tp_stationary + fp_stationary + fn_stationary)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            FB_stationary[i] += (tp_stationary + fp_stationary)/(tp_stationary + fn_stationary)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            POFD_stationary[i] += fp_stationary/(fp_stationary + fn_stationary)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            POD_stationary[i] += tp_stationary/(tp_stationary + fn_stationary)/len(front_files_test)
                        except ZeroDivisionError:
                            pass

                        try:
                            CSI_occluded[i] += tp_occluded/(tp_occluded + fp_occluded + fn_occluded)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            FB_occluded[i] += (tp_occluded + fp_occluded)/(tp_occluded + fn_occluded)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            POFD_occluded[i] += fp_occluded/(fp_occluded + fn_occluded)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            POD_occluded[i] += tp_occluded/(tp_occluded + fn_occluded)/len(front_files_test)
                        except ZeroDivisionError:
                            pass

                    print("Prediction %d/%d....done" % (x+1, len(front_files_test)))


            elif front_types == 'DL':
                for i in range(0, map_dim_x):
                    for j in range(0, map_dim_y):
                        no_probs[i][j] = prediction[5][0][i][j][0]
                        dryline_probs[i][j] = prediction[5][0][i][j][1]
                if image == 0:
                    image_no_probs[4:128][:] = no_probs[4:128][:]
                    image_dryline_probs[4:128][:] = dryline_probs[4:128][:]
                elif image == 1:
                    image_no_probs[66:122][:] = np.maximum(image_no_probs[66:122][:], no_probs[7:63][:])
                    image_dryline_probs[66:122][:] = np.maximum(image_dryline_probs[66:122][:], dryline_probs[7:63][:])

                    image_no_probs[122:181][:] = no_probs[63:122][:]
                    image_dryline_probs[122:181][:] = dryline_probs[63:122][:]
                elif image == 2:
                    image_no_probs[128:193][:] = np.maximum(image_no_probs[128:193][:], no_probs[10:75][:])
                    image_dryline_probs[128:193][:] = np.maximum(image_dryline_probs[128:193][:], dryline_probs[10:75][:])

                    image_no_probs[193:][:] = no_probs[79:124][:]
                    image_dryline_probs[193:][:] = dryline_probs[79:124][:]
                    probs_ds = xr.Dataset(
                        {"no_probs": (("longitude", "latitude"), image_no_probs), "dryline_probs": (("longitude", "latitude"), image_dryline_probs)},
                        coords={"latitude": image_lats, "longitude": image_lons})

                    t_dryline_ds = xr.where(new_fronts == 5, 1, 0)
                    t_dryline_probs = t_dryline_ds.identifier * probs_ds.dryline_probs
                    new_fronts = fronts
                    f_dryline_ds = xr.where(new_fronts == 5, 0, 1)
                    f_dryline_probs = f_dryline_ds.identifier * probs_ds.dryline_probs

                    for i in range(100):
                        tp_dryline = len(np.where(t_dryline_probs > thresholds[i])[0])
                        tn_dryline = len(np.where((f_dryline_probs < thresholds[i]) & (f_dryline_probs != 0))[0])
                        fp_dryline = len(np.where(f_dryline_probs > thresholds[i])[0])
                        fn_dryline = len(np.where((t_dryline_probs < thresholds[i]) & (t_dryline_probs != 0))[0])

                        CSI_dryline[i] += tp_dryline/(tp_dryline + fp_dryline + fn_dryline)/len(front_files_test)
                        FB_dryline[i] += (tp_dryline + fp_dryline)/(tp_dryline + fn_dryline)/len(front_files_test)
                        POFD_dryline[i] += fp_dryline/(fp_dryline + fn_dryline)/len(front_files_test)
                        POD_dryline[i] += tp_dryline/(tp_dryline + fn_dryline)/len(front_files_test)

                    print("Prediction %d/%d....done" % (x+1, len(front_files_test)))


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
                    image_no_probs[4:128][:] = no_probs[4:128][:]
                    image_cold_probs[4:128][:] = cold_probs[4:128][:]
                    image_warm_probs[4:128][:] = warm_probs[4:128][:]
                    image_stationary_probs[4:128][:] = stationary_probs[4:128][:]
                    image_occluded_probs[4:128][:] = occluded_probs[4:128][:]
                    image_dryline_probs[4:128][:] = dryline_probs[4:128][:]
                elif image == 1:
                    image_no_probs[66:122][:] = np.maximum(image_no_probs[66:122][:], no_probs[7:63][:])
                    image_cold_probs[66:122][:] = np.maximum(image_cold_probs[66:122][:], cold_probs[7:63][:])
                    image_warm_probs[66:122][:] = np.maximum(image_warm_probs[66:122][:], warm_probs[7:63][:])
                    image_stationary_probs[66:122][:] = np.maximum(image_stationary_probs[66:122][:], stationary_probs[7:63][:])
                    image_occluded_probs[66:122][:] = np.maximum(image_occluded_probs[66:122][:], occluded_probs[7:63][:])
                    image_dryline_probs[66:122][:] = np.maximum(image_dryline_probs[66:122][:], dryline_probs[7:63][:])

                    image_no_probs[122:181][:] = no_probs[63:122][:]
                    image_cold_probs[122:181][:] = cold_probs[63:122][:]
                    image_warm_probs[122:181][:] = warm_probs[63:122][:]
                    image_stationary_probs[122:181][:] = stationary_probs[63:122][:]
                    image_occluded_probs[122:181][:] = occluded_probs[63:122][:]
                    image_dryline_probs[122:181][:] = dryline_probs[63:122][:]
                elif image == 2:
                    image_no_probs[128:193][:] = np.maximum(image_no_probs[128:193][:], no_probs[10:75][:])
                    image_cold_probs[128:193][:] = np.maximum(image_cold_probs[128:193][:], cold_probs[10:75][:])
                    image_warm_probs[128:193][:] = np.maximum(image_warm_probs[128:193][:], warm_probs[10:75][:])
                    image_stationary_probs[128:193][:] = np.maximum(image_stationary_probs[128:193][:], stationary_probs[10:75][:])
                    image_occluded_probs[128:193][:] = np.maximum(image_occluded_probs[128:193][:], occluded_probs[10:75][:])
                    image_dryline_probs[128:193][:] = np.maximum(image_dryline_probs[128:193][:], dryline_probs[10:75][:])

                    image_no_probs[193:][:] = no_probs[79:124][:]
                    image_cold_probs[193:][:] = cold_probs[79:124][:]
                    image_warm_probs[193:][:] = warm_probs[79:124][:]
                    image_stationary_probs[193:][:] = stationary_probs[79:124][:]
                    image_occluded_probs[193:][:] = occluded_probs[79:124][:]
                    image_dryline_probs[193:][:] = dryline_probs[79:124][:]
                    probs_ds = xr.Dataset(
                        {"no_probs": (("longitude", "latitude"), no_probs), "cold_probs": (("longitude", "latitude"), cold_probs),
                         "warm_probs": (("longitude", "latitude"), warm_probs), "stationary_probs": (("longitude", "latitude"), stationary_probs),
                         "occluded_probs": (("longitude", "latitude"), occluded_probs), "dryline_probs": (("longitude", "latitude"), dryline_probs)},
                        coords={"latitude": lats, "longitude": lons})

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
                        tp_cold = len(np.where(t_cold_probs > thresholds[i])[0])
                        tn_cold = len(np.where((f_cold_probs < thresholds[i]) & (f_cold_probs != 0))[0])
                        fp_cold = len(np.where(f_cold_probs > thresholds[i])[0])
                        fn_cold = len(np.where((t_cold_probs < thresholds[i]) & (t_cold_probs != 0))[0])
                        tp_warm = len(np.where(t_warm_probs > thresholds[i])[0])
                        tn_warm = len(np.where((f_warm_probs < thresholds[i]) & (f_warm_probs != 0))[0])
                        fp_warm = len(np.where(f_warm_probs > thresholds[i])[0])
                        fn_warm = len(np.where((t_warm_probs < thresholds[i]) & (t_warm_probs != 0))[0])
                        tp_stationary = len(np.where(t_stationary_probs > thresholds[i])[0])
                        tn_stationary = len(np.where((f_stationary_probs < thresholds[i]) & (f_stationary_probs != 0))[0])
                        fp_stationary = len(np.where(f_stationary_probs > thresholds[i])[0])
                        fn_stationary = len(np.where((t_stationary_probs < thresholds[i]) & (t_stationary_probs != 0))[0])
                        tp_occluded = len(np.where(t_occluded_probs > thresholds[i])[0])
                        tn_occluded = len(np.where((f_occluded_probs < thresholds[i]) & (f_occluded_probs != 0))[0])
                        fp_occluded = len(np.where(f_occluded_probs > thresholds[i])[0])
                        fn_occluded = len(np.where((t_occluded_probs < thresholds[i]) & (t_occluded_probs != 0))[0])
                        tp_dryline = len(np.where(t_dryline_probs > thresholds[i])[0])
                        tn_dryline = len(np.where((f_dryline_probs < thresholds[i]) & (f_dryline_probs != 0))[0])
                        fp_dryline = len(np.where(f_dryline_probs > thresholds[i])[0])
                        fn_dryline = len(np.where((t_dryline_probs < thresholds[i]) & (t_dryline_probs != 0))[0])

                        try:
                            CSI_cold[i] += tp_cold/(tp_cold + fp_cold + fn_cold)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            FB_cold[i] += (tp_cold + fp_cold)/(tp_cold + fn_cold)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            POFD_cold[i] += fp_cold/(fp_cold + fn_cold)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            POD_cold[i] += tp_cold/(tp_cold + fn_cold)/len(front_files_test)
                        except ZeroDivisionError:
                            pass

                        try:
                            CSI_warm[i] += tp_warm/(tp_warm + fp_warm + fn_warm)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            FB_warm[i] += (tp_warm + fp_warm)/(tp_warm + fn_warm)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            POFD_warm[i] += fp_warm/(fp_warm + fn_warm)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            POD_warm[i] += tp_warm/(tp_warm + fn_warm)/len(front_files_test)
                        except ZeroDivisionError:
                            pass

                        try:
                            CSI_stationary[i] += tp_stationary/(tp_stationary + fp_stationary + fn_stationary)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            FB_stationary[i] += (tp_stationary + fp_stationary)/(tp_stationary + fn_stationary)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            POFD_stationary[i] += fp_stationary/(fp_stationary + fn_stationary)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            POD_stationary[i] += tp_stationary/(tp_stationary + fn_stationary)/len(front_files_test)
                        except ZeroDivisionError:
                            pass

                        try:
                            CSI_occluded[i] += tp_occluded/(tp_occluded + fp_occluded + fn_occluded)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            FB_occluded[i] += (tp_occluded + fp_occluded)/(tp_occluded + fn_occluded)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            POFD_occluded[i] += fp_occluded/(fp_occluded + fn_occluded)/len(front_files_test)
                        except ZeroDivisionError:
                            pass
                        try:
                            POD_occluded[i] += tp_occluded/(tp_occluded + fn_occluded)/len(front_files_test)
                        except ZeroDivisionError:
                            pass

                    print("Prediction %d/%d....done" % (x+1, len(front_files_test)))

    if front_types == 'CFWF':
        performance_ds = xr.Dataset({"CSI_cold": ("threshold", CSI_cold), "CSI_warm": ("threshold", CSI_warm),
                                     "FB_cold": ("threshold", FB_cold), "FB_warm": ("threshold", FB_warm),
                                     "POFD_cold": ("threshold", POFD_cold), "POFD_warm": ("threshold", POFD_warm),
                                     "POD_cold": ("threshold", POD_cold), "POD_warm": ("threshold", POD_warm)}, coords={"threshold": thresholds})
    elif front_types == 'SFOF':
        performance_ds = xr.Dataset({"CSI_stationary": ("threshold", CSI_stationary), "CSI_occluded": ("threshold", CSI_occluded),
                                     "FB_stationary": ("threshold", FB_stationary), "FB_occluded": ("threshold", FB_occluded),
                                     "POFD_stationary": ("threshold", POFD_stationary), "POFD_occluded": ("threshold", POFD_occluded),
                                     "POD_stationary": ("threshold", POD_stationary), "POD_occluded": ("threshold", POD_occluded)}, coords={"threshold": thresholds})
    elif front_types == 'DL':
        performance_ds = xr.Dataset({"CSI_dryline": ("threshold", CSI_dryline), "CSI_dryline": ("threshold", CSI_dryline),
                                     "FB_dryline": ("threshold", FB_dryline), "POFD_dryline": ("threshold", POFD_dryline)}, coords={"threshold": thresholds})
    elif front_types == 'ALL':
        performance_ds = xr.Dataset({"CSI_cold": ("threshold", CSI_cold), "CSI_warm": ("threshold", CSI_warm),
                                     "CSI_stationary": ("threshold", CSI_stationary), "CSI_occluded": ("threshold", CSI_occluded),
                                     "FB_cold": ("threshold", FB_cold), "FB_warm": ("threshold", FB_warm),
                                     "FB_stationary": ("threshold", FB_stationary), "FB_occluded": ("threshold", FB_occluded),
                                     "POFD_cold": ("threshold", POFD_cold), "POFD_warm": ("threshold", POFD_warm),
                                     "POFD_stationary": ("threshold", POFD_stationary), "POFD_occluded": ("threshold", POFD_occluded),
                                     "POD_cold": ("threshold", POD_cold), "POD_warm": ("threshold", POD_warm),
                                     "POD_stationary": ("threshold", POD_stationary), "POD_occluded": ("threshold", POD_occluded)}, coords={"threshold": thresholds})

    print(performance_ds)
    with open('%s/model_%d/model_%d_performance_stats_25km.pkl' % (model_dir, model_number, model_number), 'wb') as f:
        pickle.dump(performance_ds, f)


def predict(model_number, model_dir, fronts_files_list, variables_files_list, predictions, normalization_method, loss,
    fss_mask_size, fss_c, front_types, file_dimensions, pixel_expansion, metric, num_dimensions):
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
    file_dimensions: int (x2)
        Dimensions of the data files.
    pixel_expansion: int
        Number of pixels to expand the fronts by in all directions.
    metric: str
        Metric used for evaluating the U-Net during training.
    num_dimensions: int
        Number of dimensions for the U-Net's convolutions, maxpooling, and upsampling.
    """
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

    for x in range(predictions):

        # Open random pair of files
        index = random.choices(range(len(fronts_files_list) - 1), k=1)[0]
        fronts_filename = fronts_files_list[index]
        variables_filename = variables_files_list[index]
        fronts_ds = pd.read_pickle(fronts_filename)
        variable_ds = pd.read_pickle(variables_filename)

        # Select a random portion of the map
        lon_index = random.choices(range(file_dimensions[0] - map_dim_x))[0]
        lat_index = random.choices(range(file_dimensions[1] - map_dim_y))[0]
        lons = fronts_ds.longitude.values[lon_index:lon_index + map_dim_x]
        lats = fronts_ds.latitude.values[lat_index:lat_index + map_dim_y]
        fronts = fronts_ds.sel(longitude=lons, latitude=lats)

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
        print(time)

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
            for i in range(0, map_dim_y):
                for j in range(0, map_dim_x):
                    no_probs[i][j] = prediction[0][0][j][i][0][0]
                    cold_probs[i][j] = prediction[0][0][j][i][0][1]
                    warm_probs[i][j] = prediction[0][0][j][i][0][2]
                    print(prediction[0][0][j][i][:][:])
            probs_ds = xr.Dataset(
                {"no_probs": (("latitude", "longitude"), no_probs), "cold_probs": (("latitude", "longitude"), cold_probs),
                 "warm_probs": (("latitude", "longitude"), warm_probs)}, coords={"latitude": lats, "longitude": lons})

        elif front_types == 'SFOF':
            for i in range(0, map_dim_y):
                for j in range(0, map_dim_x):
                    no_probs[i][j] = prediction[0][j][i][0]
                    stationary_probs[i][j] = prediction[0][j][i][1]
                    occluded_probs[i][j] = prediction[0][j][i][2]
            probs_ds = xr.Dataset(
                {"no_probs": (("latitude", "longitude"), no_probs), "stationary_probs": (("latitude", "longitude"), stationary_probs),
                 "occluded_probs": (("latitude", "longitude"), occluded_probs)}, coords={"latitude": lats, "longitude": lons})

        elif front_types == 'DL':
            for i in range(0, map_dim_y):
                for j in range(0, map_dim_x):
                    no_probs[i][j] = prediction[0][j][i][0]
                    dryline_probs[i][j] = prediction[0][j][i][1]
            probs_ds = xr.Dataset(
                {"no_probs": (("latitude", "longitude"), no_probs), "dryline_probs": (("latitude", "longitude"), dryline_probs)},
                coords={"latitude": lats, "longitude": lons})

        elif front_types == 'ALL':
            for i in range(0, map_dim_y):
                for j in range(0, map_dim_x):
                    no_probs[i][j] = prediction[0][j][i][0]
                    cold_probs[i][j] = prediction[0][j][i][1]
                    warm_probs[i][j] = prediction[0][j][i][2]
                    stationary_probs[i][j] = prediction[0][j][i][3]
                    occluded_probs[i][j] = prediction[0][j][i][4]
                    dryline_probs[i][j] = prediction[0][j][i][5]
            probs_ds = xr.Dataset(
                {"no_probs": (("latitude", "longitude"), no_probs), "cold_probs": (("latitude", "longitude"), cold_probs),
                 "warm_probs": (("latitude", "longitude"), warm_probs), "stationary_probs": (("latitude", "longitude"), stationary_probs),
                 "occluded_probs": (("latitude", "longitude"), occluded_probs), "dryline_probs": (("latitude", "longitude"), dryline_probs)},
                coords={"latitude": lats, "longitude": lons})

        print("Generating plots....", end='')
        prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion)
        print("done")


def prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types, pixel_expansion):
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
    """
    extent = [220, 300, 29, 53]
    crs = ccrs.LambertConformal(central_longitude=250)

    # Create custom colorbar for the different front types of the 'truth' plot
    cmap = mpl.colors.ListedColormap(['0.9',"blue","red",'green','purple','orange'], name='from_list', N=None)
    norm = mpl.colors.Normalize(vmin=0,vmax=6)

    if pixel_expansion == 1:
        fronts = ope(fronts)  # 1-pixel expansion
    elif pixel_expansion == 2:
        fronts = ope(ope(fronts))  # 2-pixel expansion

    if front_types == 'CFWF':
        fig, axarr = plt.subplots(3, 1, figsize=(12, 14), subplot_kw={'projection': crs})
        axlist = axarr.flatten()
        for ax in axlist:
            fplot.plot_background(ax, extent)
        fronts.identifier.plot(ax=axlist[0], cmap=cmap, norm=norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[0].title.set_text("%s Truth" % time)
        probs_ds.cold_probs.plot(ax=axlist[1], cmap='Blues', x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[1].title.set_text("%s CF probability" % time)
        probs_ds.warm_probs.plot(ax=axlist[2], cmap='Reds', x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[2].title.set_text("%s WF probability" % time)
        plt.savefig('%s/model_%d/predictions/model_%d_%s_plot.png' % (model_dir, model_number, model_number, time),
                    bbox_inches='tight', dpi=300)
        plt.close()

    elif front_types == 'SFOF':
        fig, axarr = plt.subplots(3, 1, figsize=(12, 14), subplot_kw={'projection': crs})
        axlist = axarr.flatten()
        for ax in axlist:
            fplot.plot_background(ax, extent)
        fronts.identifier.plot(ax=axlist[0], cmap=cmap, norm=norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[0].title.set_text("%s Truth" % time)
        probs_ds.stationary_probs.plot(ax=axlist[1], cmap='Greens', x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[1].title.set_text("%s SF probability" % time)
        probs_ds.occluded_probs.plot(ax=axlist[2], cmap='Purples', x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[2].title.set_text("%s OF probability" % time)
        plt.savefig('%s/model_%d/predictions/model_%d_%s_plot.png' % (model_dir, model_number, model_number, time),
                    bbox_inches='tight', dpi=300)
        plt.close()

    elif front_types == 'DL':
        fig, axarr = plt.subplots(2, 1, figsize=(12, 14), subplot_kw={'projection': crs})
        axlist = axarr.flatten()
        for ax in axlist:
            fplot.plot_background(ax, extent)
        fronts.identifier.plot(ax=axlist[0], cmap=cmap, norm=norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[0].title.set_text("%s Truth" % time)
        probs_ds.dryline_probs.plot(ax=axlist[1], cmap='Oranges', x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[1].title.set_text("%s DL probability" % time)
        plt.savefig('%s/model_%d/predictions/model_%d_%s_plot.png' % (model_dir, model_number, model_number, time),
                    bbox_inches='tight', dpi=300)
        plt.close()

    elif front_types == 'ALL':
        fig, axarr = plt.subplots(3, 2, figsize=(12, 14), subplot_kw={'projection': crs})
        axlist = axarr.flatten()
        for ax in axlist:
            fplot.plot_background(ax, extent)
        fronts.identifier.plot(ax=axlist[0], cmap=cmap, norm=norm, x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[0].title.set_text("%s Truth" % time)
        probs_ds.cold_probs.plot(ax=axlist[1], cmap='Blues', x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[1].title.set_text("%s CF probability" % time)
        probs_ds.warm_probs.plot(ax=axlist[2], cmap='Reds', x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[2].title.set_text("%s WF probability" % time)
        probs_ds.stationary_probs.plot(ax=axlist[3], cmap='Greens', x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[3].title.set_text("%s SF probability" % time)
        probs_ds.occluded_probs.plot(ax=axlist[4], cmap='Purples', x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[4].title.set_text("%s OF probability" % time)
        probs_ds.dryline_probs.plot(ax=axlist[5], cmap='Oranges', x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[5].title.set_text("%s DL probability" % time)
        plt.savefig('%s/model_%d/predictions/model_%d_%s_plot.png' % (model_dir, model_number, model_number, time),
                    bbox_inches='tight', dpi=300)
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
    plt.plot(history['unet_output_final_activation_loss'], label='final', color='black')

    ### U-Net 3+ (3D) ### (Make sure you know if you need to remove or add lines - refer on your U-Net's architecture)
    plt.plot(history['softmax_loss'], label='Encoder 6')
    plt.plot(history['softmax_1_loss'], label='Decoder 5')
    plt.plot(history['softmax_2_loss'], label='Decoder 4')
    plt.plot(history['softmax_3_loss'], label='Decoder 3')
    plt.plot(history['softmax_4_loss'], label='Decoder 2')
    plt.plot(history['softmax_5_loss'], label='Decoder 1 (final)', color='black')
    """
    plt.plot(history['softmax_loss'], label='Encoder 6')
    plt.plot(history['softmax_1_loss'], label='Decoder 5')
    plt.plot(history['softmax_2_loss'], label='Decoder 4')
    plt.plot(history['softmax_3_loss'], label='Decoder 3')
    plt.plot(history['softmax_4_loss'], label='Decoder 2')
    plt.plot(history['softmax_5_loss'], label='Decoder 1 (final)', color='black')

    plt.legend()
    plt.xlim(xmin=0)
    plt.xlabel('Epochs')
    plt.ylim(ymin=1e-6, ymax=1e-5)  # Limits of the loss function graph, adjust these as needed
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
    plt.plot(history['softmax_%s' % metric_string], label='Encoder 6')
    plt.plot(history['softmax_1_%s' % metric_string], label='Decoder 5')
    plt.plot(history['softmax_2_%s' % metric_string], label='Decoder 4')
    plt.plot(history['softmax_3_%s' % metric_string], label='Decoder 3')
    plt.plot(history['softmax_4_%s' % metric_string], label='Decoder 2')
    plt.plot(history['softmax_5_%s' % metric_string], label='Decoder 1 (final)', color='black')

    plt.legend()
    plt.xlim(xmin=0)
    plt.xlabel('Epochs')
    plt.ylim(ymin=1e-3, ymax=1e-1)  # Limits of the metric graph, adjust as needed
    plt.yscale('log')
    plt.savefig("%s/model_%d/model_%d_learning_curve.png" % (model_dir, model_number, model_number), bbox_inches='tight')


def average_max_probabilities(model_number, model_dir, variables_files_list, loss, normalization_method, front_types,
    fss_mask_size, fss_c, metric, num_dimensions):
    """
    Function that makes calculates maximum front probabilities for the provided model and saves the probabilities to
    a pickle file.

    Parameters
    ----------
    model_number: int
        Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        Main directory for the models.
    variables_files_list: list
        List of filenames that contain variable data.
    loss: str
        Loss function for the Unet.
    normalization_method: int
        Normalization method for the data (described near the end of the script).
    front_types: str
        Front format of the file.
    fss_mask_size: int
        Size of the mask for the FSS loss function.
    fss_c: float
        C hyperparameter for the FSS loss' sigmoid function.
    metric: str
        Metric used for evaluating the U-Net during training.
    num_dimensions: int
        Number of dimensions for the U-Net's convolutions, maxpooling, and upsampling.
    """

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

    max_cold_probs = []  # List of maximum cold front probabilities achieved in each file.
    max_warm_probs = []  # List of maximum warm front probabilities achieved in each file.
    max_stationary_probs = []  # List of maximum stationary front probabilities achieved in each file.
    max_occluded_probs = []  # List of maximum occluded front probabilities achieved in each file.
    max_dryline_probs = []  # List of maximum dryline front probabilities achieved in each file.
    dates = []  # List of dates for the files.
    num_files = len(variables_files_list)

    print("Calculating maximum probability statistics for model %d" % model_number)
    for x in range(num_files):

        variable_ds = pd.read_pickle(variables_files_list[x])
        dates.append(variable_ds.time.values)

        # Hold indices constant for a better evaluation
        lon_index = 97
        lat_index = 0
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
        - i and j are loop indices, do not change these
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
            for i in range(0, map_dim_y):
                for j in range(0, map_dim_x):
                    no_probs[i][j] = prediction[0][0][j][i][0][0]
                    cold_probs[i][j] = prediction[0][0][j][i][0][1]
                    warm_probs[i][j] = prediction[0][0][j][i][0][2]
            max_cold_probs.append(np.max(cold_probs*100))
            max_warm_probs.append(np.max(warm_probs*100))
            print("\r(%d/%d)  Avg CF/WF: %.1f%s / %.1f%s,  Max CF/WF: %.1f%s / %.1f%s, Stddev CF/WF: %.1f%s / %.1f%s "
                % (x+1, num_files, np.sum(max_cold_probs)/(x+1), '%', np.sum(max_warm_probs)/(x+1),  '%', np.max(max_cold_probs),
                '%', np.max(max_warm_probs),  '%', np.std(max_cold_probs),  '%', np.std(max_warm_probs),  '%',), end='')

        elif front_types == 'SFOF':
            for i in range(0, map_dim_y):
                for j in range(0, map_dim_x):
                    no_probs[i][j] = prediction[0][j][i][0]
                    stationary_probs[i][j] = prediction[0][j][i][1]
                    occluded_probs[i][j] = prediction[0][j][i][2]
            max_stationary_probs.append(np.max(stationary_probs*100))
            max_occluded_probs.append(np.max(occluded_probs*100))
            print("\r(%d/%d)  Avg SF/OF: %.1f%s / %.1f%s,  Max SF/OF: %.1f%s / %.1f%s, Stddev SF/OF: %.1f%s / %.1f%s "
                % (x+1, num_files, np.sum(max_stationary_probs)/(x+1), '%', np.sum(max_occluded_probs)/(x+1),  '%', np.max(max_stationary_probs),
                '%', np.max(max_occluded_probs),  '%', np.std(max_stationary_probs),  '%', np.std(max_occluded_probs),  '%',), end='')

        elif front_types == 'DL':
            for i in range(0, map_dim_y):
                for j in range(0, map_dim_x):
                    no_probs[i][j] = prediction[0][j][i][0]
                    dryline_probs[i][j] = prediction[0][j][i][1]
            max_dryline_probs.append(np.max(dryline_probs*100))
            print("\r(%d/%d)  Avg DL: %.1f%s,  Max DL: %.1f%s, Stddev DL: %.1f%s" % (x+1, num_files,
                np.sum(max_dryline_probs)/(x+1), '%', np.max(max_dryline_probs), '%', np.std(max_dryline_probs), '%'), end='')

        elif front_types == 'ALL':
            for i in range(0, map_dim_y):
                for j in range(0, map_dim_x):
                    no_probs[i][j] = prediction[0][j][i][0]
                    cold_probs[i][j] = prediction[0][j][i][1]
                    warm_probs[i][j] = prediction[0][j][i][2]
                    stationary_probs[i][j] = prediction[0][j][i][3]
                    occluded_probs[i][j] = prediction[0][j][i][4]
                    dryline_probs[i][j] = prediction[0][j][i][5]
            max_cold_probs.append(np.max(cold_probs*100))
            max_warm_probs.append(np.max(warm_probs*100))
            max_stationary_probs.append(np.max(stationary_probs*100))
            max_occluded_probs.append(np.max(occluded_probs*100))
            max_dryline_probs.append(np.max(dryline_probs*100))
            print("\r(%d/%d)  Avg CF/WF/SF/OF/DL: %.1f%s / %.1f%s / %.1f%s / %.1f%s / %.1f%s, "
                  " Max CF/WF/SF/OF/DL: %.1f%s / %.1f%s / %.1f%s / %.1f%s / %.1f%s, "
                  " Stddev CF/WF/SF/OF/DL: %.1f%s / %.1f%s / %.1f%s / %.1f%s / %.1f%s "
                % (x+1, num_files, np.sum(max_cold_probs)/(x+1), '%', np.sum(max_warm_probs)/(x+1),  '%', np.sum(max_stationary_probs)/(x+1), '%',
                   np.sum(max_occluded_probs)/(x+1),  '%', np.sum(max_dryline_probs)/(x+1),  '%', np.max(max_cold_probs), '%',
                   np.max(max_warm_probs),  '%', np.max(max_stationary_probs),  '%', np.max(max_occluded_probs),  '%',
                   np.max(max_dryline_probs),  '%', np.std(max_cold_probs),  '%', np.std(max_warm_probs),  '%',
                   np.std(max_stationary_probs),  '%', np.std(max_occluded_probs),  '%', np.std(max_dryline_probs),  '%'),
                  end='')

    if front_types == 'CFWF':
        max_probs_ds = xr.Dataset({"max_cold_probs": ("time", max_cold_probs), "max_warm_probs":
            ("time", max_warm_probs)}, coords={"time": dates})

    elif front_types == 'SFOF':
        max_probs_ds = xr.Dataset({"max_stationary_probs": ("time", max_stationary_probs), "max_occluded_probs":
            ("time", max_occluded_probs)}, coords={"time": dates})

    elif front_types == 'DL':
        max_probs_ds = xr.Dataset({"max_dryline_probs": ("time", max_dryline_probs)}, coords={"time": dates})

    elif front_types == 'ALL':
        max_probs_ds = xr.Dataset({"max_cold_probs": ("time", max_cold_probs), "max_warm_probs":
            ("time", max_warm_probs), "max_stationary_probs": ("time", max_stationary_probs), "max_occluded_probs":
            ("time", max_occluded_probs), "max_dryline_probs": ("time", max_dryline_probs)}, coords={"time": dates})

    print(max_probs_ds)

    with open('%s/model_%d/model_%d_maximum_probabilities.pkl' % (model_dir, model_number, model_number), 'wb') as f:
        pickle.dump(max_probs_ds, f)


def probability_distribution_plot(model_number, model_dir, front_types):
    """
    Function that takes an Xarray dataset containing maximum front probabilities for a provided model and creates a
    probability distribution plot.

    Parameters
    ----------
    model_number: int
        Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        Main directory for the models.
    front_types: str
        Front format of the file.
    """
    with open('%s/model_%d/model_%d_maximum_probabilities.pkl' % (model_dir, model_number, model_number), 'rb') as f:
        max_probs_ds = pd.read_pickle(f)

    labels = []  # Sets of probability labels for the plot

    """
    <front>_occurrences: Number of occurrences in each file, sorted in lists
    total_<front>_occurrences: Total number of occurrences over all files
    """
    cold_occurrences = []
    total_cold_occurrences = 0
    warm_occurrences = []
    total_warm_occurrences = 0
    stationary_occurrences = []
    total_stationary_occurrences = 0
    occluded_occurrences = []
    total_occluded_occurrences = 0
    dryline_occurrences = []
    total_dryline_occurrences = 0
    total_days = len(max_probs_ds.time.values)

    if front_types == 'CFWF':
        for i in range(20):
            cold_occurrence = len(np.where(max_probs_ds.max_cold_probs.values <= (5*(i+1)))[0]) - total_cold_occurrences
            total_cold_occurrences += cold_occurrence
            cold_occurrences.append(cold_occurrence)
            warm_occurrence = len(np.where(max_probs_ds.max_warm_probs.values <= (5*(i+1)))[0]) - total_warm_occurrences
            total_warm_occurrences += warm_occurrence
            warm_occurrences.append(warm_occurrence)
            labels.append("%d-%d" % (int(5*i), int(5*(i+1))))
        plt.figure(figsize=(10,5),dpi=300)
        plt.grid(axis='y', alpha=0.2)
        plt.xticks(ticks=np.linspace(0,19,20), labels=labels)
        plt.title("Model %d Probability Distribution" % model_number)
        plt.plot(np.array(cold_occurrences)*100/total_days, 'b',label='cold front')
        plt.plot(np.array(warm_occurrences)*100/total_days, 'r',label='warm front')
        plt.xticks(rotation=45)
        plt.xlabel("Maximum probability (%)")
        plt.ylabel("Occurrence in dataset (%)")
        plt.ylim(0,100)
        plt.legend()
        plt.savefig("%s/model_%d/model_%d_percentages.png" % (model_dir, model_number, model_number), bbox_inches='tight')
        plt.close()

    elif front_types == 'SFOF':
        for i in range(20):
            stationary_occurrence = len(np.where(max_probs_ds.max_stationary_probs.values <= (5*(i+1)))[0]) - total_stationary_occurrences
            total_stationary_occurrences += stationary_occurrence
            stationary_occurrences.append(stationary_occurrence)
            occluded_occurrence = len(np.where(max_probs_ds.max_occluded_probs.values <= (5*(i+1)))[0]) - total_occluded_occurrences
            total_occluded_occurrences += occluded_occurrence
            occluded_occurrences.append(occluded_occurrence)
            labels.append("%d-%d" % (int(5*i), int(5*(i+1))))
        plt.figure(figsize=(10,5),dpi=300)
        plt.grid(axis='y', alpha=0.2)
        plt.xticks(ticks=np.linspace(0,19,20), labels=labels)
        plt.title("Model %d Probability Distribution" % model_number)
        plt.plot(np.array(stationary_occurrences)*100/total_days, 'b',label='stationary front')
        plt.plot(np.array(occluded_occurrences)*100/total_days, 'r',label='occluded front')
        plt.xticks(rotation=45)
        plt.xlabel("Maximum probability (%)")
        plt.ylabel("Occurrence in dataset (%)")
        plt.ylim(0,100)
        plt.legend()
        plt.savefig("%s/model_%d/model_%d_percentages.png" % (model_dir, model_number, model_number), bbox_inches='tight')
        plt.close()

    elif front_types == 'DL':
        for i in range(20):
            dryline_occurrence = len(np.where(max_probs_ds.max_dryline_probs.values <= (5*(i+1)))[0]) - total_dryline_occurrences
            total_dryline_occurrences += dryline_occurrence
            dryline_occurrences.append(dryline_occurrence)
            labels.append("%d-%d" % (int(5*i), int(5*(i+1))))
        plt.figure(figsize=(10,5),dpi=300)
        plt.grid(axis='y', alpha=0.2)
        plt.xticks(ticks=np.linspace(0,19,20), labels=labels)
        plt.title("Model %d Probability Distribution" % model_number)
        plt.plot(np.array(dryline_occurrences)*100/total_days, 'r',label='dryline front')
        plt.xticks(rotation=45)
        plt.xlabel("Maximum probability (%)")
        plt.ylabel("Occurrence in dataset (%)")
        plt.ylim(0,100)
        plt.legend()
        plt.savefig("%s/model_%d/model_%d_percentages.png" % (model_dir, model_number, model_number), bbox_inches='tight')
        plt.close()

    elif front_types == 'ALL':
        for i in range(20):
            cold_occurrence = len(np.where(max_probs_ds.max_cold_probs.values <= (5*(i+1)))[0]) - total_cold_occurrences
            total_cold_occurrences += cold_occurrence
            cold_occurrences.append(cold_occurrence)
            warm_occurrence = len(np.where(max_probs_ds.max_warm_probs.values <= (5*(i+1)))[0]) - total_warm_occurrences
            total_warm_occurrences += warm_occurrence
            warm_occurrences.append(warm_occurrence)
            stationary_occurrence = len(np.where(max_probs_ds.max_stationary_probs.values <= (5*(i+1)))[0]) - total_stationary_occurrences
            total_stationary_occurrences += stationary_occurrence
            stationary_occurrences.append(stationary_occurrence)
            occluded_occurrence = len(np.where(max_probs_ds.max_occluded_probs.values <= (5*(i+1)))[0]) - total_occluded_occurrences
            total_occluded_occurrences += occluded_occurrence
            occluded_occurrences.append(occluded_occurrence)
            labels.append("%d-%d" % (int(5*i), int(5*(i+1))))
        plt.figure(figsize=(10,5), dpi=300)
        plt.grid(axis='y', alpha=0.2)
        plt.xticks(ticks=np.linspace(0,19,20), labels=labels)
        plt.title("Model %d Probability Distribution" % model_number)
        plt.plot(np.array(cold_occurrences)*100/total_days, 'b', label='cold front')
        plt.plot(np.array(warm_occurrences)*100/total_days, 'r', label='warm front')
        plt.plot(np.array(stationary_occurrences)*100/total_days, 'g', label='stationary front')
        plt.plot(np.array(occluded_occurrences)*100/total_days, color='purple', label='occluded front')
        plt.plot(np.array(dryline_occurrences)*100/total_days, color='orange', label='dryline front')
        plt.xticks(rotation=45)
        plt.xlabel("Maximum probability (%)")
        plt.ylabel("Occurrence in dataset (%)")
        plt.ylim(0,100)
        plt.legend()
        plt.savefig("%s/model_%d/model_%d_percentages.png" % (model_dir, model_number, model_number), bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    """ Main arguments """
    parser.add_argument('--cross_validate', type=bool, required=False,
                        help='Are you calculating performance stats for a model for cross-validation purposes?')
    parser.add_argument('--domain', type=str, required=False, help='Domain of the data.')
    parser.add_argument('--file_dimensions', type=int, nargs=2, required=False,
                        help='Dimensions of the file size. Two integers need to be passed.')
    parser.add_argument('--front_types', type=str, required=False,
                        help='Front format of the file. If your files contain warm and cold fronts, pass this argument'
                             'as CFWF. If your files contain only drylines, pass this argument as DL. If your files '
                             'contain all fronts, pass this argument as ALL.')
    parser.add_argument('--fss_c', type=float, required=False, help="C hyperparameter for the FSS loss' sigmoid function.")
    parser.add_argument('--fss_mask_size', type=int, required=False, help='Mask size for the FSS loss function.')
    parser.add_argument('--learning_curve', type=bool, required=False, help='Plot learning curve?')
    parser.add_argument('--loss', type=str, required=False, help='Loss function used for training the U-Net.')
    parser.add_argument('--metric', type=str, required=False, help='Metric used for evaluating the U-Net during training.')
    parser.add_argument('--model_dir', type=str, required=False, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, required=False, help='Model number.')
    parser.add_argument('--num_dimensions', type=int, required=False,
                        help='Number of dimensions of the U-Net convolutions, maxpooling, and upsampling. (2 or 3)')
    parser.add_argument('--num_variables', type=int, required=False, help='Number of variables in the variable datasets.')
    parser.add_argument('--normalization_method', type=int, required=False,
                        help='Normalization method for the data. 0 - No normalization, 1 - Min-max normalization, '
                             '2 - Mean normalization')
    parser.add_argument('--performance_statistics', type=bool, required=False,
                        help='Calculate performance statistics for a model?')
    parser.add_argument('--pixel_expansion', type=int, required=False, help='Number of pixels to expand the fronts by.')
    parser.add_argument('--predict', type=bool, required=False, help='Generate prediction plots?')
    parser.add_argument('--predictions', type=int, required=False, help='Number of predictions to make.')
    parser.add_argument('--probability_plot', type=bool, required=False, help='Create probability distribution plot?')
    parser.add_argument('--probability_statistics', type=bool, required=False,
                        help='Calculate maximum probability statistics?')
    parser.add_argument('--test_year', type=int, required=False, help='Test year for cross-validating the current model.')

    args = parser.parse_args()

    print(args)

    fronts_files_list, variables_files_list = fm.load_file_lists(args.num_variables, args.front_types, args.domain,
                                                                 args.file_dimensions)

    if args.loss == 'fss':
        if args.fss_c is None or args.fss_mask_size is None:
            raise errors.MissingArgumentError("If loss is fss, the following arguments must be passed: fss_c, fss_mask_size")
    else:
        if args.fss_c is not None or args.fss_mask_size is not None:
            raise errors.ExtraArgumentError("loss is not fss but one or more of the following arguments was provided: "
                                            "fss_c, fss_mask_size")

    if args.cross_validate is True:
        if args.model_number is None or args.model_dir is None or args.num_variables is None or args.num_dimensions is None or \
            args.front_types is None or args.domain is None or args.file_dimensions is None or args.test_year is None or \
            args.normalization_method is None or args.loss is None or args.pixel_expansion is None or args.metric:
            raise errors.MissingArgumentError("If cross_validate is True, the following arguments must be passed: "
                "domain, file_dimensions, front_types, loss, metric, model_dir, model_number, normalization_method, "
                "num_dimensions, num_variables, pixel_expansion, test_year")
        else:
            cross_validate(args.model_number, args.model_dir, args.num_variables, args.num_dimensions, args.front_types,
                args.domain, args.file_dimensions, args.test_year, args.normalization_method, args.loss, args.fss_mask_size,
                args.fss_c, args.pixel_expansion, args.metric)
    else:
        if args.predictions is not None:
            raise errors.ExtraArgumentError("cross_validate is False but the following argument was provided: test_year")

    if args.learning_curve is True:
        if args.model_number is None or args.model_dir is None or args.loss is None or args.metric is None:
            raise errors.MissingArgumentError("If learning_curve is True, the following arguments must be passed: "
                                              "loss, metric, model_dir, model_number")
        else:
            learning_curve(args.model_number, args.model_dir, args.loss, args.fss_mask_size, args.fss_c, args.metric)

    if args.predict is True:
        if args.model_number is None or args.model_dir is None or args.num_variables is None or args.num_dimensions is None or \
            args.front_types is None or args.domain is None or args.file_dimensions is None or args.predictions is None or \
            args.normalization_method is None or args.loss is None or args.pixel_expansion is None or args.metric:
            raise errors.MissingArgumentError("If predict is True, the following arguments must be passed: "
                "domain, file_dimensions, front_types, loss, metric, model_dir, model_number, normalization_method, "
                "num_dimensions, num_variables, pixel_expansion, predictions")
        else:
            predict(args.model_number, args.model_dir, fronts_files_list, variables_files_list, args.predictions,
                args.normalization_method, args.loss, args.fss_mask_size, args.fss_c, args.front_types, args.file_dimensions,
                args.pixel_expansion, args.metric, args.num_dimensions)
    else:
        if args.predictions is not None:
            raise errors.ExtraArgumentError("predict is False but the following argument was provided: predictions")

    if args.probability_statistics is True:
        if args.model_number is None or args.model_dir is None or args.num_dimensions is None or args.front_types is None or \
            args.normalization_method is None or args.loss is None or args.metric is None:
            raise errors.MissingArgumentError("If predict is True, the following arguments must be passed: "
                "file_dimensions, front_types, loss, metric, model_dir, model_number, normalization_method, num_dimensions")
        else:
            average_max_probabilities(args.model_number, args.model_dir, variables_files_list, args.loss, args.normalization_method,
                args.front_types, args.fss_mask_size, args.fss_c, args.metric, args.num_dimensions)

    if args.probability_plot is True:
        if args.model_number is None or args.model_dir is None or args.front_types is None:
            raise errors.MissingArgumentError("If learning_curve is True, the following arguments must be passed: "
                                              "front_types, model_dir, model_number")
        else:
            probability_distribution_plot(args.model_number, args.model_dir, args.front_types)
