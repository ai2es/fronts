"""UNet model"""

import random
import pandas as pd
from glob import glob
import argparse
from keras_unet_collection import models, losses
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import gpu_config
import Plot_ERA5
import cartopy.crs as ccrs
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical

def generate_nonempty_file_lists(frontobject_window_files, surfacedata_window_files):
    print("\n=== GENERATING NON-EMPTY FILE LISTS ===")
    nonempty_front_files = []
    nonempty_sfcdata_files = []
    file_count = len(frontobject_window_files)
    for i in range(0,file_count):
        print("\rProcessing files....%d/%d" % (i,file_count), end='')
        fronts = pd.read_pickle(frontobject_window_files[i]).identifier.values.flatten()
        identifiers = np.count_nonzero(fronts)
        if identifiers > 0:
            nonempty_front_files.append(frontobject_window_files[i])
            nonempty_sfcdata_files.append(surfacedata_window_files[i])
    print("Processing files....done          ")
    print("Saving lists....", end='')
    with open('nonempty_front_files.pkl', 'wb') as f:
        pickle.dump(nonempty_front_files, f)
    with open('nonempty_sfcdata_files.pkl', 'wb') as f:
        pickle.dump(nonempty_sfcdata_files, f)
    print("done")

def training_generator(nonempty_front_files, nonempty_sfcdata_files, input_name, output_name, batch_size):

    while True:
        indices = random.choices(range(len(nonempty_front_files)), k=batch_size)
        for i in range(batch_size):
            lons = pd.read_pickle(nonempty_front_files[indices[i]]).longitude.values[0:16]
            lats = pd.read_pickle(nonempty_front_files[indices[i]]).latitude.values[0:16]
            fronts = pd.read_pickle(nonempty_front_files[indices[i]]).sel(longitude=lons, latitude=lats
                                                                          ).to_array().T.values.reshape(1,16,16)
            binarized_fronts = to_categorical(fronts, num_classes=3)
            sfcdata = pd.read_pickle(nonempty_sfcdata_files[indices[i]]).sel(longitude=lons, latitude=lats
                                                                             ).to_array().T.values.reshape(1,16,16,6)
            yield({input_name: sfcdata},
                  {output_name: binarized_fronts})

def load_files(pickle_indir):
    print("\n=== LOADING FILES ===")
    print("Collecting front object files....", end='')
    frontobject_window_files = sorted(glob("%s/*/*/*/FrontObjects*lon*lat*.pkl" % pickle_indir))
    print("done, %d files found" % len(frontobject_window_files))

    print("Collecting surface data files....", end='')
    surfacedata_window_files = sorted(glob("%s/*/*/*/SurfaceData*lon*lat*.pkl" % pickle_indir))
    print("done, %d files found" % len(surfacedata_window_files))

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
def train_unet():
    print("\n==== MODEL TRAINING ====")
    print("Creating unet....", end='')
    model = models.unet_2d((16, 16, 6), [16, 32, 64, 128, 256], n_labels=3, stack_num_down=3,
                            stack_num_up=3, activation='ReLU', output_activation='Softmax',
                            batch_norm=True, pool=True, unpool=True, name='unet')
    print('done')

    print('Compiling unet....', end='')
    dice = losses.dice
    tversky = losses.tversky
    adam = Adam(learning_rate=1e-4)

    model.compile(loss=dice, optimizer=adam, metrics=tf.keras.metrics.AUC())
    print('done')

    layer_names=[layer.name for layer in model.layers]
    input_name = layer_names[0]
    output_name = layer_names[-1]

    nonempty_front_files = pd.read_pickle('nonempty_front_files.pkl')
    nonempty_sfcdata_files = pd.read_pickle('nonempty_sfcdata_files.pkl')

    batch_size = 100

    generator = training_generator(nonempty_front_files, nonempty_sfcdata_files, input_name, output_name, batch_size)

    history = model.fit(generator, epochs=2, steps_per_epoch=100)

    model.save('E:/FrontsProjectData/models/front_model_dice.h5')

def evaluate_model(frontobject_window_files, surfacedata_window_files):

    print("\n==== MODEL EVALUATION ====")
    dice = losses.dice
    print("Loading model....", end='')
    model = tf.keras.models.load_model('E:/FrontsProjectData/models/front_model_dice.h5', custom_objects={'dice': dice})
    print("done")

    fronts_filename = 'E:/FrontsProjectData/pickle_files/2008/05/25/FrontObjects_2008052521_lon(260_264)_lat(41_45).pkl'
    sfcdata_filename = 'E:/FrontsProjectData/pickle_files/2008/05/25/SurfaceData_2008052521_lon(260_264)_lat(41_45).pkl'
    lons = pd.read_pickle(fronts_filename).longitude.values[1:]
    lats = pd.read_pickle(fronts_filename).latitude.values[1:]

    fronts = pd.read_pickle(fronts_filename).sel(longitude=lons, latitude=lats)
    sfcdata = pd.read_pickle(sfcdata_filename).sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1,16,16,6)

    prediction = model.predict(sfcdata).reshape(16,16,3)
    #print(prediction.shape)

    print(prediction)

    identifier = np.zeros([16,16])
    print("Reformatting predictions....", end='')
    for i in range(0,16):
        for j in range(0,16):
            if prediction[i][j][1] > prediction[i][j][0] and prediction[i][j][1] > prediction[i][j][2]:
                identifier[i][j] = 1
            elif prediction[i][j][2] > prediction[i][j][0] and prediction[i][j][2] > prediction[i][j][1]:
                identifier[i][j] = 2

    fronts.identifier.values = identifier

    extent = [238, 283, 25, 50]
    ax = Plot_ERA5.plot_background(extent)

    fronts.identifier.plot(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree())
    plt.savefig('E:/FrontsProjectData/models/model_prediction_plot.png', bbox_inches='tight', dpi=1000)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_indir', type=str, required=True, help='Path of pickle files containing front object'
                                                                        ' and surface data.')
    parser.add_argument('--file_removal', type=str, required=False, help='Remove extra files that cannot be used? '
                                                                         '(True/False)')
    parser.add_argument('--model_outdir', type=str, required=False, help='Path where models will saved to')
    parser.add_argument('--model_filepath', type=str, required=False, help='Path where model for evaluation is saved')
    args = parser.parse_args()

    gpu_config.memory_growth()

    if args.pickle_indir is not None:
        frontobject_window_files, surfacedata_window_files = load_files(args.pickle_indir)
        #generate_nonempty_file_lists(frontobject_window_files, surfacedata_window_files)
        if len(frontobject_window_files) != len(surfacedata_window_files):
            if args.file_removal == 'True':
                file_removal(frontobject_window_files, surfacedata_window_files, args.pickle_indir)
                print("====> NOTE: Extra files have been removed, a new list of files will now be loaded. <====")
                frontobject_window_files, surfacedata_window_files = load_files(args.pickle_indir)
            else:
                print("ERROR: The number of front object and surface data files are not the same. You must remove "
                      "the extra files before data\ncan be processed by setting the '--file_removal' argument to "
                      "'True'.")
        #train_unet(frontobject_window_files, surfacedata_window_files)
        train_unet()
        evaluate_model(frontobject_window_files, surfacedata_window_files)
    else:
        print("ERROR: No directory for pickle files included, please declare where pickle files are located using the "
              "'--pickle_indir'\nargument.")

