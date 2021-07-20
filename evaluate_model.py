"""
Functions used for evaluating a U-Net model. The functions can be used to make predictions or plot learning curves.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 7/20/2021 2:21 PM CDT
"""

import random
import pandas as pd
import argparse
import tensorflow as tf
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import file_manager as fm
import Fronts_Aggregate_Plot as fplot
import xarray as xr
import errors
import custom_losses
import pickle


def predict(model_number, model_dir, fronts_files_list, variables_files_list, predictions, normalization_method, loss,
    fss_mask_size, front_types):
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
    front_types: str
        Fronts in the data.
    """
    print("\n=== MODEL EVALUATION ===")

    print("Loading model....", end='')
    if loss == 'cce':
        model = tf.keras.models.load_model('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number))
        print("done")
        print("Loss function: cce")
    elif loss == 'dice':
        model = tf.keras.models.load_model('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number), 
            custom_objects={'dice': custom_losses.dice})
        print("done")
        print("Loss function: dice")
    elif loss == 'tversky':
        model = tf.keras.models.load_model('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number), 
            custom_objects={'tversky': custom_losses.tversky})
        print("done")
        print("Loss function: tversky")
    elif loss == 'fss':
        model = tf.keras.models.load_model('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number), 
            custom_objects={'FSS_loss': custom_losses.make_FSS_loss(fss_mask_size)})
        print("done")
        print("Loss function: FSS(%d)" % fss_mask_size)

    map_dim_x = model.layers[0].input_shape[0][1]
    map_dim_y = model.layers[0].input_shape[0][2]
    channels = model.layers[0].input_shape[0][3]

    maxs = [326.3396301, 305.8016968, 107078.2344, 44.84565735, 135.2455444, 327, 70, 1, 333, 310, 415, 0.024951244,
        309.2406, 89.81117, 90.9507, 17612.004, 0.034231212, 309.2406, 89.853115, 91.11507, 13249.213, 0.046489507,
        309.2406, 62.46032, 91.073425, 9000.762, 0.048163727, 309.2406, 62.22315, 76.649796, 17522.139]
    mins = [192.2073669, 189.1588898, 47399.61719, -196.7885437, -96.90724182, 188, 0.0005, 0, 192, 193, 188,
        0.00000000466, 205.75833, -165.10022, -64.62073, -6912.213, 0.00000000466, 205.75833, -165.20557, -64.64681,
        903.0327, 0.00000000466, 205.75833, -148.51501, -66.1152, -3231.293, 0.00000000466, 205.75833, -165.27695,
        -58.405083, -6920.75]
    means = [278.8510794, 274.2647937, 96650.46322, -0.06747816, 0.1984011, 278.39639128, 4.291633, 0.7226335,
        279.5752426, 276.296217417, 293.69090226, 0.00462498, 274.6106082, 1.385064762, 0.148459298, 13762.46737, 
        0.005586943, 276.6008764, 0.839714324, 0.201385933, 9211.468268, 0.00656686, 278.2460963, 0.375778613, 
        0.207254872, 4877.725497, 0.007057154, 280.1310979, -0.050884628, 0.197406197, 736.070931]
    std_devs = [21.161467, 20.603729, 9590.54, 5.587448, 4.795126, 24.325, 12.2499125, 0.175, 24.675, 20.475, 39.725,
        0.004141041, 15.55585542, 8.250520488, 6.286386854, 1481.972616, 0.00473022, 15.8944975, 8.122294976, 
        6.424827792, 1313.379508, 0.005520186, 16.7592906, 7.689928269, 6.445098408, 1178.610181, 0.005908417, 
        18.16819064, 6.193227753, 5.342330733, 1083.730224]

    for x in range(predictions):
        index = random.choices(range(len(fronts_files_list) - 1), k=1)[0]
        fronts_filename = fronts_files_list[index]
        variables_filename = variables_files_list[index]
        fronts_ds = pd.read_pickle(fronts_filename)
        variable_ds = pd.read_pickle(variables_filename)

        lon_index = random.choices(range(289 - map_dim_x))[0]
        lat_index = random.choices(range(129 - map_dim_y))[0]
        lons = fronts_ds.longitude.values[lon_index:lon_index + map_dim_x]
        lats = fronts_ds.latitude.values[lat_index:lat_index + map_dim_y]

        fronts = fronts_ds.sel(longitude=lons, latitude=lats)
        variable_list = list(variable_ds.keys())
        for j in range(31):
            var = variable_list[j]
            if normalization_method == 1:
                # Z-score normalization
                variable_ds[var].values = np.nan_to_num((variable_ds[var].values - means[j]) / std_devs[j])
            elif normalization_method == 2:
                # Min-max normalization
                variable_ds[var].values = np.nan_to_num((variable_ds[var].values - mins[j]) / (maxs[j] - mins[j]))
            elif normalization_method == 3:
                # Mean normalization
                variable_ds[var].values = np.nan_to_num((variable_ds[var].values - means[j]) / (maxs[j] - mins[j]))
        variable = np.nan_to_num(variable_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, map_dim_x, 
            map_dim_y, channels))

        prediction = model.predict(variable)

        time = str(fronts.time.values)[0:13].replace('T', '-') + 'z'
        print(time)

        no_probs = np.zeros([map_dim_x, map_dim_y])
        cold_probs = np.zeros([map_dim_x, map_dim_y])
        warm_probs = np.zeros([map_dim_x, map_dim_y])
        stationary_probs = np.zeros([map_dim_x, map_dim_y])
        occluded_probs = np.zeros([map_dim_x, map_dim_y])
        dryline_probs = np.zeros([map_dim_x, map_dim_y])

        if front_types == 'CFWF':
            for i in range(0, map_dim_y):
                for j in range(0, map_dim_x):
                    no_probs[i][j] = prediction[0][j][i][0]
                    cold_probs[i][j] = prediction[0][j][i][1]
                    warm_probs[i][j] = prediction[0][j][i][2]
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
        prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types)
        print("done\n\n\n")


def prediction_plot(fronts, probs_ds, time, model_number, model_dir, front_types):
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
    """
    extent = [220, 300, 29, 53]
    crs = ccrs.LambertConformal(central_longitude=250)

    if front_types == 'CFWF':
        fig, axarr = plt.subplots(3, 1, figsize=(12, 14), subplot_kw={'projection': crs})
        axlist = axarr.flatten()
        for ax in axlist:
            fplot.plot_background(ax, extent)
        fronts.identifier.plot(ax=axlist[0], x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[0].title.set_text("%s Truth" % time)
        probs_ds.cold_probs.plot(ax=axlist[1], x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[1].title.set_text("%s CF probability" % time)
        probs_ds.warm_probs.plot(ax=axlist[2], x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[2].title.set_text("%s WF probability" % time)
        plt.savefig('%s/model_%d/predictions/model_%d_%s_plot.png' % (model_dir, model_number, model_number, time),
                    bbox_inches='tight', dpi=300)
        plt.close()

    elif front_types == 'SFOF':
        fig, axarr = plt.subplots(3, 1, figsize=(12, 14), subplot_kw={'projection': crs})
        axlist = axarr.flatten()
        for ax in axlist:
            fplot.plot_background(ax, extent)
        fronts.identifier.plot(ax=axlist[0], x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[0].title.set_text("%s Truth" % time)
        probs_ds.stationary_probs.plot(ax=axlist[1], x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[1].title.set_text("%s SF probability" % time)
        probs_ds.occluded_probs.plot(ax=axlist[2], x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[2].title.set_text("%s OF probability" % time)
        plt.savefig('%s/model_%d/predictions/model_%d_%s_plot.png' % (model_dir, model_number, model_number, time),
                    bbox_inches='tight', dpi=300)
        plt.close()

    elif front_types == 'DL':
        fig, axarr = plt.subplots(2, 1, figsize=(12, 14), subplot_kw={'projection': crs})
        axlist = axarr.flatten()
        for ax in axlist:
            fplot.plot_background(ax, extent)
        fronts.identifier.plot(ax=axlist[0], x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[0].title.set_text("%s Truth" % time)
        probs_ds.dryline_probs.plot(ax=axlist[1], x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[1].title.set_text("%s DL probability" % time)
        plt.savefig('%s/model_%d/predictions/model_%d_%s_plot.png' % (model_dir, model_number, model_number, time),
                    bbox_inches='tight', dpi=300)
        plt.close()

    elif front_types == 'ALL':
        fig, axarr = plt.subplots(3, 2, figsize=(12, 14), subplot_kw={'projection': crs})
        axlist = axarr.flatten()
        for ax in axlist:
            fplot.plot_background(ax, extent)
        fronts.identifier.plot(ax=axlist[0], x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[0].title.set_text("%s Truth" % time)
        probs_ds.cold_probs.plot(ax=axlist[1], x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[1].title.set_text("%s CF probability" % time)
        probs_ds.warm_probs.plot(ax=axlist[2], x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[2].title.set_text("%s WF probability" % time)
        probs_ds.stationary_probs.plot(ax=axlist[3], x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[3].title.set_text("%s SF probability" % time)
        probs_ds.occluded_probs.plot(ax=axlist[4], x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[4].title.set_text("%s OF probability" % time)
        probs_ds.dryline_probs.plot(ax=axlist[5], x='longitude', y='latitude', transform=ccrs.PlateCarree())
        axlist[5].title.set_text("%s DL probability" % time)
        plt.savefig('%s/model_%d/predictions/model_%d_%s_plot.png' % (model_dir, model_number, model_number, time),
                    bbox_inches='tight', dpi=300)
        plt.close()


def learning_curve(model_number, model_dir):
    """
    Function that plots learning curves for the specified model.
    
    Parameters
    ----------
    model_number: int
        Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        Main directory for the models.
    """
    with open("%s/model_%d/model_%d_history.csv" % (model_dir, model_number, model_number), 'rb') as f:
        history = pd.read_csv(f)

    plt.subplots(1, 2, figsize=(10, 5), dpi=500)
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.grid()
    plt.plot(history['loss'], 'b')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)

    plt.subplot(1, 2, 2)
    plt.title("AUC")
    plt.grid()
    plt.plot(history['auc'], 'r')
    plt.axhline(y=0.9871, color='black', linestyle='-')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0.98, ymax=1)
    plt.savefig("%s/model_%d/model_%d_learning_curve.png" % (model_dir, model_number, model_number),
                bbox_inches='tight')


def average_max_probabilities(model_number, model_dir, variables_files_list, loss, normalization_method,
    file_dimensions, front_types, fss_mask_size):
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
    file_dimensions: int (x2)
        Dimensions of the datasets.
    front_types: str
        Front format of the file.
    fss_mask_size: int
        Size of the mask for the FSS loss function.
    """
    print("\n=== MODEL EVALUATION ===")
    print("Loading model....", end='')

    if loss == 'cce':
        model = tf.keras.models.load_model('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number))
    else:
        if loss == 'dice':
            loss_function = custom_losses.dice
            print("done")
            print("Loss: dice")
        elif loss == 'tversky':
            loss_function = custom_losses.tversky
            print("done")
            print("Loss: tversky")
        elif loss == 'fss':
            loss = 'FSS_loss'  # Override the loss argument so keras can recognize the loss function in the model.
            loss_function = custom_losses.make_FSS_loss(fss_mask_size)
            print("done")
            print("Loss: FSS(%d)" % fss_mask_size)
        model = tf.keras.models.load_model('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number),
                                           custom_objects={loss: loss_function})

    map_dim_x = model.layers[0].input_shape[0][1]
    map_dim_y = model.layers[0].input_shape[0][2]
    channels = model.layers[0].input_shape[0][3]

    maxs = [326.3396301, 305.8016968, 107078.2344, 44.84565735, 135.2455444, 327, 70, 1, 333, 310, 415, 0.024951244,
            309.2406, 89.81117, 90.9507, 17612.004, 0.034231212, 309.2406, 89.853115, 91.11507, 13249.213,
            0.046489507, 309.2406, 62.46032, 91.073425, 9000.762, 0.048163727, 309.2406, 62.22315, 76.649796,
            17522.139]
    mins = [192.2073669, 189.1588898, 47399.61719, -196.7885437, -96.90724182, 188, 0.0005, 0, 192, 193, 188,
            0.00000000466, 205.75833, -165.10022, -64.62073, -6912.213, 0.00000000466, 205.75833, -165.20557,
            -64.64681, 903.0327, 0.00000000466, 205.75833, -148.51501, -66.1152, -3231.293, 0.00000000466,
            205.75833, -165.27695, -58.405083, -6920.75]
    means = [278.8510794, 274.2647937, 96650.46322, -0.06747816, 0.1984011, 278.39639128, 4.291633, 0.7226335,
             279.5752426, 276.296217417, 293.69090226, 0.00462498, 274.6106082, 1.385064762, 0.148459298,
             13762.46737, 0.005586943, 276.6008764, 0.839714324, 0.201385933, 9211.468268, 0.00656686, 278.2460963,
             0.375778613, 0.207254872, 4877.725497, 0.007057154, 280.1310979, -0.050884628, 0.197406197, 736.070931]
    std_devs = [21.161467, 20.603729, 9590.54, 5.587448, 4.795126, 24.325, 12.2499125, 0.175, 24.675, 20.475,
                39.725, 0.004141041, 15.55585542, 8.250520488, 6.286386854, 1481.972616, 0.00473022, 15.8944975,
                8.122294976, 6.424827792, 1313.379508, 0.005520186, 16.7592906, 7.689928269, 6.445098408,
                1178.610181, 0.005908417, 18.16819064, 6.193227753, 5.342330733, 1083.730224]

    max_cold_probs = []
    max_warm_probs = []
    max_stationary_probs = []
    max_occluded_probs = []
    max_dryline_probs = []
    dates = []
    num_files = len(variables_files_list)
    print("Calculating maximum probability statistics for model %d" % model_number)
    for x in range(num_files):
        variable_ds = pd.read_pickle(variables_files_list[x])
        time = variable_ds.time.values
        lon_index = random.choices(range(file_dimensions[0] - map_dim_x))[0]
        lat_index = random.choices(range(file_dimensions[1] - map_dim_y))[0]
        lons = variable_ds.longitude.values[lon_index:lon_index + map_dim_x]
        lats = variable_ds.latitude.values[lat_index:lat_index + map_dim_y]
        variable_list = list(variable_ds.keys())
        for j in range(31):
            var = variable_list[j]
            if normalization_method == 1:
                # Z-score normalization
                variable_ds[var].values = np.nan_to_num((variable_ds[var].values - means[j]) / std_devs[j])
            elif normalization_method == 2:
                # Min-max normalization
                variable_ds[var].values = np.nan_to_num((variable_ds[var].values - mins[j]) / (maxs[j] - mins[j]))
            elif normalization_method == 3:
                # Mean normalization
                variable_ds[var].values = np.nan_to_num((variable_ds[var].values - means[j]) / (maxs[j] - mins[j]))
        variable = np.nan_to_num(variable_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, map_dim_x,
            map_dim_y, channels))

        prediction = model.predict(variable)

        if front_types == 'CFWF':
            no_probs = np.zeros([map_dim_x, map_dim_y])
            cold_probs = np.zeros([map_dim_x, map_dim_y])
            warm_probs = np.zeros([map_dim_x, map_dim_y])
            for i in range(0, map_dim_y):
                for j in range(0, map_dim_x):
                    no_probs[i][j] = prediction[0][j][i][0]
                    cold_probs[i][j] = prediction[0][j][i][1]
                    warm_probs[i][j] = prediction[0][j][i][2]
            max_cold_probs.append(np.max(cold_probs*100))
            max_warm_probs.append(np.max(warm_probs*100))
            dates.append(time)
            print("\r(%d/%d)  Avg CF/WF: %.1f%s / %.1f%s,  Max CF/WF: %.1f%s / %.1f%s, Stddev CF/WF: %.1f%s / %.1f%s "
                % (x+1, num_files, np.sum(max_cold_probs)/x, '%', np.sum(max_warm_probs)/x,  '%', np.max(max_cold_probs),
                '%', np.max(max_warm_probs),  '%', np.std(max_cold_probs),  '%', np.std(max_warm_probs),  '%',), end='')

        elif front_types == 'SFOF':
            no_probs = np.zeros([map_dim_x, map_dim_y])
            stationary_probs = np.zeros([map_dim_x, map_dim_y])
            occluded_probs = np.zeros([map_dim_x, map_dim_y])
            for i in range(0, map_dim_y):
                for j in range(0, map_dim_x):
                    no_probs[i][j] = prediction[0][j][i][0]
                    stationary_probs[i][j] = prediction[0][j][i][1]
                    occluded_probs[i][j] = prediction[0][j][i][2]
            max_stationary_probs.append(np.max(stationary_probs*100))
            max_occluded_probs.append(np.max(occluded_probs*100))
            dates.append(time)
            print("\r(%d/%d)  Avg SF/OF: %.1f%s / %.1f%s,  Max SF/OF: %.1f%s / %.1f%s, Stddev SF/OF: %.1f%s / %.1f%s "
                % (x+1, num_files, np.sum(max_stationary_probs)/x, '%', np.sum(max_occluded_probs)/x,  '%', np.max(max_stationary_probs),
                '%', np.max(max_occluded_probs),  '%', np.std(max_stationary_probs),  '%', np.std(max_occluded_probs),  '%',), end='')

        elif front_types == 'DL':
            no_probs = np.zeros([map_dim_x, map_dim_y])
            dryline_probs = np.zeros([map_dim_x, map_dim_y])
            for i in range(0, map_dim_y):
                for j in range(0, map_dim_x):
                    no_probs[i][j] = prediction[0][j][i][0]
                    dryline_probs[i][j] = prediction[0][j][i][1]
            print("\r(%d/%d)  Avg DL: %.1f%s,  Max DL: %.1f%s, Stddev DL: %.1f%s" % (x+1, num_files,
                np.sum(max_dryline_probs)/x, '%', np.max(max_dryline_probs), '%', np.std(max_dryline_probs), '%'), end='')

        elif front_types == 'ALL':
            no_probs = np.zeros([map_dim_x, map_dim_y])
            cold_probs = np.zeros([map_dim_x, map_dim_y])
            warm_probs = np.zeros([map_dim_x, map_dim_y])
            stationary_probs = np.zeros([map_dim_x, map_dim_y])
            occluded_probs = np.zeros([map_dim_x, map_dim_y])
            dryline_probs = np.zeros([map_dim_x, map_dim_y])
            for i in range(0, map_dim_y):
                for j in range(0, map_dim_x):
                    no_probs[i][j] = prediction[0][j][i][0]
                    cold_probs[i][j] = prediction[0][j][i][1]
                    warm_probs[i][j] = prediction[0][j][i][2]
                    stationary_probs[i][j] = prediction[0][j][i][3]
                    occluded_probs[i][j] = prediction[0][j][i][4]
                    dryline_probs[i][j] = prediction[0][j][i][5]
            print("\r(%d/%d)  Avg CF/WF/SF/OF/DL: %.1f%s / %.1f%s / %.1f%s / %.1f%s / %.1f%s, "
                  " Max CF/WF/SF/OF/DL: %.1f%s / %.1f%s / %.1f%s / %.1f%s / %.1f%s, "
                  " Stddev CF/WF/SF/OF/DL: %.1f%s / %.1f%s / %.1f%s / %.1f%s / %.1f%s "
                % (x+1, num_files, np.sum(max_cold_probs)/x, '%', np.sum(max_warm_probs)/x,  '%', np.sum(max_stationary_probs)/x, '%',
                   np.sum(max_occluded_probs)/x,  '%', np.sum(max_dryline_probs)/x,  '%', np.max(max_cold_probs), '%',
                   np.max(max_warm_probs),  '%', np.max(max_stationary_probs),  '%', np.max(max_occluded_probs),  '%',
                   np.max(max_dryline_probs),  '%', np.std(max_cold_probs),  '%', np.std(max_warm_probs),  '%',
                   np.std(max_stationary_probs),  '%', np.std(max_occluded_probs),  '%', np.std(max_dryline_probs),  '%'),
                  end='')

    max_probs_ds = xr.Dataset({"max_cold_probs": ("time", max_cold_probs), "max_warm_probs":
        ("time", max_warm_probs)}, coords={"time": dates})

    print(max_probs_ds)

    with open('model_%d_maximum_probabilities.pkl' % model_number, 'wb') as f:
        pickle.dump(max_probs_ds, f)


def probability_distribution_plot(model_number, front_types):
    """
    Function that takes an Xarray dataset containing maximum front probabilities for a provided model and creates a
    probability distribution plot.

    Parameters
    ----------
    model_number: int
        Slurm job number for the model. This is the number in the model's filename.
    front_types: str
        Front format of the file.
    """
    with open('model_%d_maximum_probabilities.pkl' % model_number, 'rb') as f:
        max_probs_ds = pd.read_pickle(f)

    labels = []
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
        plt.savefig("model_%d_percentages.png" % model_number,bbox_inches='tight')
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
        plt.savefig("model_%d_percentages.png" % model_number,bbox_inches='tight')
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
        plt.savefig("model_%d_percentages.png" % model_number,bbox_inches='tight')
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
        plt.figure(figsize=(10,5),dpi=300)
        plt.grid(axis='y', alpha=0.2)
        plt.xticks(ticks=np.linspace(0,19,20), labels=labels)
        plt.title("Model %d Probability Distribution" % model_number)
        plt.plot(np.array(cold_occurrences)*100/total_days, 'b',label='cold front')
        plt.plot(np.array(warm_occurrences)*100/total_days, 'r',label='warm front')
        plt.plot(np.array(stationary_occurrences)*100/total_days, 'g', label='stationary front')
        plt.plot(np.array(occluded_occurrences)*100/total_days, color='purple', label='occluded front')
        plt.plot(np.array(dryline_occurrences)*100/total_days, color='orange', label='dryline front')
        plt.xticks(rotation=45)
        plt.xlabel("Maximum probability (%)")
        plt.ylabel("Occurrence in dataset (%)")
        plt.ylim(0,100)
        plt.legend()
        plt.savefig("model_%d_percentages.png" % model_number, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_number', type=int, required=True, help='Path of pickle files containing front object'
        ' and surface data.')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory for the models.')
    parser.add_argument('--predict', type=str, required=False, help='Generate prediction plots? (True/False)')
    parser.add_argument('--predictions', type=int, required=False, help='Number of predictions to make. Default is 25.')
    parser.add_argument('--learning_curve', type=str, required=False, help='Plot learning curves? (True/False)')
    parser.add_argument('--loss', type=str, required=True, help='Loss function used for training the Unet.')
    parser.add_argument('--fss_mask_size', type=int, required=True, help='Mask size for the FSS loss function'
        ' (if applicable).')
    parser.add_argument('--num_variables', type=int, required=True, help='Number of variables in the variable datasets.')
    parser.add_argument('--front_types', type=str, required=True, help='Front format of the file. If your files contain warm'
        ' and cold fronts, pass this argument as CFWF. If your files contain only drylines, pass this argument as DL. '
        'If your files contain all fronts, pass this argument as ALL.')
    parser.add_argument('--domain', type=str, required=True, help='Domain of the data. Possible values are: conus')
    parser.add_argument('--file_dimensions', type=int, nargs=2, required=True, help='Dimensions of the file size. Two integers'
        ' need to be passed.')
    # Normalization methods: 0 - No normalization, 1 - Z-score, 2 - Min-max normalization, 3 - Mean normalization
    parser.add_argument('--normalization_method', type=int, required=True, help='Normalization method for the data.')
    parser.add_argument('--probability_statistics', type=str, required=False, help='Calculate maximum probability statistics?'
        ' (True/False)')
    parser.add_argument('--probability_plot', type=str, required=False, help='Create probability distribution plot? (True/False)')
    args = parser.parse_args()

    fronts_files_list, variables_files_list = fm.load_file_lists(args.num_variables, args.front_types, args.domain, 
                                                                 args.file_dimensions)

    if args.loss == 'fss' and args.fss_mask_size is None:
        raise errors.MissingArgumentError("Argument '--fss_mask_size' must be passed if you are using the FSS loss function.")
    if args.loss != 'fss' and args.fss_mask_size is not None:
        raise errors.ExtraArgumentError("Argument '--fss_mask_size' can only be passed if you are using the FSS loss function.")

    if args.predict == 'True':
        if args.predictions is None:
            print("WARNING: '--predictions' argument not provided, defaulting to 25 predictions.")
            predictions = 25
        else:
            predictions = args.predictions
        predict(args.model_number, args.model_dir, fronts_files_list, variables_files_list, predictions,
                args.normalization_method, args.loss, args.fss_mask_size, args.front_types)
    else:
        if args.predictions is not None:
            raise errors.ExtraArgumentError("Argument '--predictions' cannot be passed if '--predict' is False or was not provided.")

    if args.learning_curve == 'True':
        learning_curve(args.model_number, args.model_dir)

    if args.probability_statistics == 'True':
        average_max_probabilities(args.model_number, args.model_dir, variables_files_list, args.loss, args.normalization_method,
            args.file_dimensions, args.front_types, args.fss_mask_size)

    if args.probability_plot == 'True':
        probability_distribution_plot(args.model_number, args.front_types)
