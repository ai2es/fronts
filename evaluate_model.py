"""
Functions used for evaluating a U-Net model. The functions can be used to make predictions or plot learning curves.
Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 7/17/2021 11:39 PM CDT
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

def predict(model_number, model_dir, fronts_files_list, variables_files_list, predictions, normalization_method):
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
    """
    print("\n=== MODEL EVALUATION ===")
    print("Loading model....", end='')
    model = tf.keras.models.load_model('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number), custom_objects={'FSS_loss': custom_losses.make_FSS_loss(2)})
    print("done")

    map_dim_x = model.layers[0].input_shape[0][1]
    map_dim_y = model.layers[0].input_shape[0][2]
    channels = model.layers[0].input_shape[0][3]
    maxs = [326.3396301, 305.8016968, 107078.2344, 44.84565735, 135.2455444, 327, 70, 1, 333, 310, 415, 0.024951244,
            309.2406, 89.81117, 90.9507, 17612.004, 0.034231212, 309.2406, 89.853115, 91.11507, 13249.213,
            0.046489507,
            309.2406, 62.46032, 91.073425, 9000.762, 0.048163727, 309.2406, 62.22315, 76.649796, 17522.139]
    mins = [192.2073669, 189.1588898, 47399.61719, -196.7885437, -96.90724182, 188, 0.0005, 0, 192, 193, 188,
            0.00000000466, 205.75833, -165.10022, -64.62073, -6912.213, 0.00000000466, 205.75833, -165.20557,
            -64.64681,
            903.0327, 0.00000000466, 205.75833, -148.51501, -66.1152, -3231.293, 0.00000000466, 205.75833,
            -165.27695,
            -58.405083, -6920.75]
    means = [278.8510794, 274.2647937, 96650.46322, -0.06747816, 0.1984011, 278.39639128, 4.291633, 0.7226335,
             279.5752426, 276.296217417, 293.69090226, 0.00462498, 274.6106082, 1.385064762, 0.148459298,
             13762.46737, 0.005586943, 276.6008764, 0.839714324, 0.201385933, 9211.468268, 0.00656686, 278.2460963,
             0.375778613, 0.207254872, 4877.725497, 0.007057154, 280.1310979, -0.050884628, 0.197406197, 736.070931]
    std_devs = [21.161467, 20.603729, 9590.54, 5.587448, 4.795126, 24.325, 12.2499125, 0.175, 24.675, 20.475,
                39.725,
                0.004141041, 15.55585542, 8.250520488, 6.286386854, 1481.972616, 0.00473022, 15.8944975,
                8.122294976, 6.424827792, 1313.379508, 0.005520186, 16.7592906, 7.689928269, 6.445098408,
                1178.610181,
                0.005908417, 18.16819064, 6.193227753, 5.342330733, 1083.730224]
    for x in range(predictions):
        index = random.choices(range(len(fronts_files_list) - 1), k=1)[0]
        fronts_filename = fronts_files_list[index]
        variables_filename = variables_files_list[index]
        fronts_ds = pd.read_pickle(fronts_filename)
        sfcdata_ds = pd.read_pickle(variables_filename)

        lon_index = random.choices(range(289 - map_dim_x))[0]
        lat_index = random.choices(range(129 - map_dim_y))[0]
        lons = fronts_ds.longitude.values[lon_index:lon_index + map_dim_x]
        lats = fronts_ds.latitude.values[lat_index:lat_index + map_dim_y]

        fronts = fronts_ds.sel(longitude=lons, latitude=lats)
        variable_list = list(sfcdata_ds.keys())
        for j in range(31):
            var = variable_list[j]
            if normalization_method == 1:
                # Z-score normalization
                sfcdata_ds[var].values = np.nan_to_num((sfcdata_ds[var].values - means[j]) / std_devs[j])
            elif normalization_method == 2:
                # Min-max normalization
                sfcdata_ds[var].values = np.nan_to_num((sfcdata_ds[var].values - mins[j]) / (maxs[j] - mins[j]))
            elif normalization_method == 3:
                # Mean normalization
                sfcdata_ds[var].values = np.nan_to_num((sfcdata_ds[var].values - means[j]) / (maxs[j] - mins[j]))
        sfcdata = np.nan_to_num(
            sfcdata_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, map_dim_x, map_dim_y,
                                                                                      channels))

        print("Predicting values....", end='')
        prediction = model.predict(sfcdata)
        print("done")

        print(prediction.shape)
        print(prediction)
        time = str(fronts.time.values)[0:13].replace('T', '-') + 'z'
        print(time)

        no_probs = np.zeros([map_dim_x, map_dim_y])
        cold_probs = np.zeros([map_dim_x, map_dim_y])
        warm_probs = np.zeros([map_dim_x, map_dim_y])

        print("Reformatting predictions....", end='')
        for i in range(0, map_dim_y):
            for j in range(0, map_dim_x):
                no_probs[i][j] = prediction[0][j][i][0]
                cold_probs[i][j] = prediction[0][j][i][1]
                warm_probs[i][j] = prediction[0][j][i][2]
        print("done")

        probs_ds = xr.Dataset(
            {"no_probs": (("latitude", "longitude"), no_probs), "cold_probs": (("latitude", "longitude"), cold_probs),
             "warm_probs": (("latitude", "longitude"), warm_probs)}, coords={"latitude": lats, "longitude": lons})

        print("Generating plots....", end='')
        prediction_plot(fronts, probs_ds, time, model_number, model_dir)
        print("done\n\n\n")


def prediction_plot(fronts, probs_ds, time, model_number, model_dir):
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
    """
    extent = [220, 300, 29, 53]
    crs = ccrs.LambertConformal(central_longitude=250)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_number', type=int, required=True, help='Path of pickle files containing front object'
                                                                        ' and surface data.')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory for the models.')
    parser.add_argument('--predict', type=str, required=False, help='Generate prediction plots? (True/False)')
    parser.add_argument('--predictions', type=int, required=False, help='Number of predictions to make. Default is 25.')
    parser.add_argument('--learning_curve', type=str, required=False, help='Plot learning curves? (True/False)')
    parser.add_argument('--num_variables', type=int, required=True, help='Number of variables in the variable datasets.')
    parser.add_argument('--front_types', type=str, required=True, help='Front format of the file. If your files contain warm'
                                                                       ' and cold fronts, pass this argument as CFWF.'
                                                                       ' If your files contain only drylines, pass this argument'
                                                                       ' as DL. If your files contain all fronts, pass this argument'
                                                                       ' as ALL.')
    parser.add_argument('--domain', type=str, required=True, help='Domain of the data. Possible values are: conus')
    parser.add_argument('--map_dimensions', type=int, nargs=2, required=True, help='Dimensions of the map size. Two integers'
                                                                                   ' need to be passed.')
    # Normalization methods: 0 - No normalization, 1 - Z-score, 2 - Min-max normalization, 3 - Mean normalization
    parser.add_argument('--normalize_method', type=int, required=True, help='Normalization method for the data.')
    args = parser.parse_args()

    fronts_files_list, variables_files_list = fm.load_file_lists(args.num_variables, args.front_types, args.domain, args.map_dimensions)

    if args.predict == 'True':
        if args.predictions is None:
            print("WARNING: '--predictions' argument not provided, defaulting to 25 predictions.")
            predictions = 25
        else:
            predictions = args.predictions
        predict(args.model_number, args.model_dir, fronts_files_list, variables_files_list, predictions,
                args.normalization_method)
    else:
        if args.predictions is not None:
            raise errors.MissingArgumentError(
                "'--predictions' cannot be passed if '--predict' is False or was not provided.")

    if args.learning_curve is not None:
        if args.learning_curve == 'True':
            learning_curve(args.model_number, args.model_dir)
