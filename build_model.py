"""UNet model"""

import xarray as xr
import pandas as pd
from glob import glob
import argparse
from keras_unet_collection import models
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import Plot_ERA5
import cartopy.crs as ccrs
import os
import matplotlib.pyplot as plt
import numpy as np
global list_IDs

def training_generator(frontobject_window_files, surfacedata_window_files, input_name, output_name):

    num_indices = len(frontobject_window_files)

    for i in range(0,num_indices,126):
        time = pd.read_pickle(frontobject_window_files[i]).time.values
        for j in range(0,7):
            lats = pd.read_pickle(frontobject_window_files[i+j]).latitude.values
            for k in range(0,6):
                fronts_dss = xr.concat(objs=(pd.read_pickle(frontobject_window_files[i+j+(21*k)]),
                                             pd.read_pickle(frontobject_window_files[i+j+(21*k)+7]),
                                             pd.read_pickle(frontobject_window_files[i+j+(21*k)+14])),
                                       dim='longitude').drop('time')
                sfcdata_dss = xr.concat(objs=(pd.read_pickle(surfacedata_window_files[i+j+(21*k)]),
                                              pd.read_pickle(surfacedata_window_files[i+j+(21*k)+7]),
                                              pd.read_pickle(surfacedata_window_files[i+j+(21*k)+14])),
                                        dim='longitude').drop('time')
                lons = fronts_dss.longitude.values
                for l in range(0,2):
                    for m in range(0,6):
                        fronts = fronts_dss.sel(latitude=slice(lats[0+l],lats[15+l]),
                                                longitude=slice(lons[7*m],lons[7*m+15])
                                                ).to_array().T.values.reshape(1,16,16,1)
                        sfcdata = sfcdata_dss.sel(latitude=slice(lats[0+l],lats[15+l]),
                                                  longitude=slice(lons[7*m],lons[7*m+15])
                                                  ).to_array().T.values.reshape(1,16,16,6)
                        yield({input_name: sfcdata,
                               output_name: fronts})

def load_files(pickle_indir):
    print("\n=== LOADING FILES ===")
    print("Collecting front object files....", end='')
    frontobject_window_files = sorted(glob("%s/*/*/*/FrontObjects*lon*lat*.pkl" % pickle_indir))
    print("done, %d files found" % len(frontobject_window_files))

    print("Collecting surface data files....", end='')
    surfacedata_window_files = sorted(glob("%s/*/*/*/SurfaceData*lon*lat*.pkl" % pickle_indir))
    print("done, %d files found\n" % len(surfacedata_window_files))

    return frontobject_window_files, surfacedata_window_files

def file_removal(frontobject_window_files, surfacedata_window_files, pickle_indir):

    print("=== FILE REMOVAL ===")

    extra_files = len(surfacedata_window_files) - len(frontobject_window_files)
    total_sfc_files = len(surfacedata_window_files)

    frontobject_window_files_no_prefix = []
    surfacedata_window_files_no_prefix = []

    for i in range(0,len(frontobject_window_files)):
        frontobject_window_files_no_prefix.append(frontobject_window_files[i].replace('FrontObjects_',''))
    for j in range(0,len(surfacedata_window_files)):
        surfacedata_window_files_no_prefix.append(surfacedata_window_files[j].replace('SurfaceData_',''))

    frontobject_window_files_no_prefix = np.array(frontobject_window_files_no_prefix)
    surfacedata_window_files_no_prefix = np.array(surfacedata_window_files_no_prefix)

    k = 0
    x = 0
    print("Removing %d extra files....0/%d" % (extra_files, extra_files), end='\r')
    for filename in surfacedata_window_files_no_prefix:
        k = k + 1
        if filename not in frontobject_window_files_no_prefix:
            os.rename(surfacedata_window_files[k-1],
                      surfacedata_window_files[k-1].replace("%s" % pickle_indir,
                                                            "%s/unusable_pickle_files" % pickle_indir))
            x += 1
        print("Removing %d extra files....%d%s (%d/%d)" % (extra_files, 100*(k/total_sfc_files), '%', x, extra_files),
              end='\r')
    print("Removing %d extra files....done")

# UNet model
def train_unet(frontobject_window_files, surfacedata_window_files):

    print("Creating unet....", end='')
    model = models.unet_2d((16, 16, 6), [16, 32, 64, 128, 256], n_labels=1, stack_num_down=3,
                            stack_num_up=3, activation='ReLU', output_activation='Softmax',
                            batch_norm=True, pool=True, unpool=True, name='unet')
    print('done')

    print('Compiling unet....', end='')
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=tf.keras.metrics.AUC())
    print('done')

    layer_names=[layer.name for layer in model.layers]
    input_name = layer_names[0]
    output_name = layer_names[-1]

    generator = training_generator(frontobject_window_files, surfacedata_window_files, input_name, output_name)

    history = model.fit_generator(generator=generator, steps_per_epoch=1, verbose=1, epochs=5)

def evaluate_model(X_test, filename, frontobject_conus_df, train_length):

    print("==========================\n"
          "==== MODEL EVALUATION ====\n"
          "==========================\n")
    print("Loading model....", end='')
    model = tf.keras.models.load_model(filename)
    print("done")
    print("Testing model....", end='')
    prediction = model.predict(X_test)
    print("done")

    print(prediction)
    print(prediction.shape)

    prediction_dss = frontobject_conus_df.to_xarray()
    prediction_df = prediction_dss.sel(time=prediction_dss.time.values[train_length+1::])
    print(prediction_df)

    identifier = np.empty((prediction.shape[0],prediction.shape[1],prediction.shape[2]),dtype=int)
    print(prediction.shape)
    print(identifier.shape)

    print("Binarizing predictions....",end='')
    for i in range(0, 1129):
        for j in range(0, 96):
            for k in range(0, 96):
                no_front_prob = prediction[i][j][k][0]
                cold_front_prob = prediction[i][j][k][1]
                warm_front_prob = prediction[i][j][k][2]
                if cold_front_prob > 0.17:
                    identifier[i][j][k] = 1
                elif warm_front_prob > 0.17:
                    identifier[i][j][k] = 2
                else:
                    identifier[i][j][k] = 0

    prediction_df.identifier.values = identifier.reshape(96,96,1129)

    extent = [238, 283, 25, 50]
    ax = Plot_ERA5.plot_background(extent)

    prediction_df.identifier.sel(time='2009-08-10T21:00:00').plot(ax=ax, x='longitude', y='latitude',
                                                                  transform=ccrs.PlateCarree())
    plt.savefig('E:/FrontsProjectData/models/model_prediction_plot.png', bbox_inches='tight', dpi=1000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_indir', type=str, required=True, help='Path of pickle files containing front object'
                                                                        ' and surface data.')
    parser.add_argument('--file_removal', type=str, required=False, help='Remove extra files that cannot be used? '
                                                                        '(True/False)')
    parser.add_argument('--model_outdir', type=str, required=False, help='Path where models will saved to')
    parser.add_argument('--model_filepath', type=str, required=False, help='Path where model for evaluation is saved')
    args = parser.parse_args()

    if args.pickle_indir is not None:
        frontobject_window_files, surfacedata_window_files = load_files(args.pickle_indir)
        if len(frontobject_window_files) != len(surfacedata_window_files):
            if args.file_removal == 'True':
                file_removal(frontobject_window_files, surfacedata_window_files, args.pickle_indir)
                print("====> NOTE: Extra files have been removed, a new list of files will now be loaded. <====")
                frontobject_window_files, surfacedata_window_files = load_files(args.pickle_indir)
            else:
                print("ERROR: The number of front object and surface data files are not the same. You must remove "
                      "the extra files before data\ncan be processed by setting the '--file_removal' argument to "
                      "'True'.")
        train_unet(frontobject_window_files, surfacedata_window_files)

    else:
        print("ERROR: No directory for pickle files included, please declare where pickle files are located using the "
              "'--pickle_indir'\nargument.")

