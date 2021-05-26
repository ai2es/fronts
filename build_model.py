"""UNet model"""

import random
import pandas as pd
import argparse
from keras_unet_collection import models, losses
import tensorflow as tf
import Plot_ERA5
import cartopy.crs as ccrs
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
import file_manager as fm

def training_generator(frontobject_conus_files, surfacedata_conus_files, map_size, empty_split, input_name, output_name,
                       batch_size):

    while True:
        front_dss = np.empty(shape=(batch_size,map_size,map_size,3))
        sfcdata_dss = np.empty(shape=(batch_size,map_size,map_size,6))
        indices = random.choices(range(len(frontobject_conus_files)-1), k=int(batch_size*(1-empty_split)))
        for i in range(len(indices)):
            with open(frontobject_conus_files[indices[i]], 'rb') as front_file:
                front_ds = pickle.load(front_file)
            with open(surfacedata_conus_files[indices[i]], 'rb') as sfcdata_file:
                sfcdata_ds = pickle.load(sfcdata_file)
            lon_index = random.choices(range(289-map_size))[0]
            lat_index = random.choices(range(113-map_size))[0]
            lons = front_ds.longitude.values[lon_index:lon_index+map_size]
            lats = front_ds.latitude.values[lat_index:lat_index+map_size]
            fronts = front_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, map_size, map_size)
            binarized_fronts = to_categorical(fronts, num_classes=3)
            sfcdata = sfcdata_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, map_size, map_size, 6)
            front_dss[i] = binarized_fronts[0]
            sfcdata_dss[i] = sfcdata[0]
            front_file.close()
            front_ds.close()
            sfcdata_file.close()
            sfcdata_ds.close()
        yield ({input_name: sfcdata_dss},
               {output_name: front_dss})

def validation_generator(frontobject_conus_files, surfacedata_conus_files, map_size, empty_split, input_name, output_name,
                         batch_size):

    while True:
        front_dss = np.empty(shape=(batch_size,map_size,map_size,3))
        sfcdata_dss = np.empty(shape=(batch_size,map_size,map_size,6))
        indices = random.choices(range(len(frontobject_conus_files)-1), k=int(batch_size*(1-empty_split)))
        for i in range(batch_size):
            with open(frontobject_conus_files[indices[i]], 'rb') as front_file:
                front_ds = pickle.load(front_file)
            with open(surfacedata_conus_files[indices[i]], 'rb') as sfcdata_file:
                sfcdata_ds = pickle.load(sfcdata_file)
            lon_index = random.choices(range(289-map_size))[0]
            lat_index = random.choices(range(113-map_size))[0]
            lons = front_ds.longitude.values[lon_index:lon_index+map_size]
            lats = front_ds.latitude.values[lat_index:lat_index+map_size]
            fronts = front_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, map_size, map_size)
            binarized_fronts = to_categorical(fronts, num_classes=3)
            sfcdata = sfcdata_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, map_size, map_size, 6)
            front_dss[i] = binarized_fronts[0]
            sfcdata_dss[i] = sfcdata[0]
            front_file.close()
            front_ds.close()
            sfcdata_file.close()
            sfcdata_ds.close()
        yield ({input_name: sfcdata_dss},
               {output_name: front_dss})


# UNet model
def train_unet(frontobject_conus_files, surfacedata_conus_files, map_size, empty_split,
               learning_rate, train_epochs, train_steps, train_batch_size, valid_steps, valid_batch_size, valid_freq,
               workers, job_number):
    """
    Function that trains the unet model and saves the model along with its weights.
    """
    print("\n=== MODEL TRAINING ===")
    print("Creating unet....", end='')

    model = models.unet_2d((map_size, map_size, 6), filter_num=[16, 32, 64, 128, 256], n_labels=3, stack_num_down=5,
                           stack_num_up=5, activation='LeakyReLU', output_activation='Softmax',
                           batch_norm=True, pool=True, unpool=True, name='unet')
    print('done')
    print('Compiling unet....', end='')
    dice = losses.dice
    tversky = losses.tversky
    adam = Adam(learning_rate=learning_rate)

    model.compile(loss=dice, optimizer=adam, metrics=tf.keras.metrics.AUC())
    print('done')

    model.summary()

    layer_names = [layer.name for layer in model.layers]
    input_name = layer_names[0]
    output_name = layer_names[-1]

    train_gen = training_generator(frontobject_conus_files, surfacedata_conus_files, map_size, empty_split, input_name,
                                   output_name, train_batch_size)
    valid_gen = validation_generator(frontobject_conus_files, surfacedata_conus_files, map_size, empty_split,
                                     input_name, output_name, valid_batch_size)

    history = model.fit(train_gen, validation_data=valid_gen, validation_freq=valid_freq, epochs=train_epochs,
                        steps_per_epoch=train_steps, validation_steps=valid_steps, verbose=2, workers=workers)

    model.save('/work/earnestb13/ajustin/fronts/models/model_%d.h5' % job_number)
    model.save_weights("/work/earnestb13/ajustin/fronts/models/model_%d_weights.h5" % job_number)


def evaluate_model(job_number, map_size):
    print("\n=== MODEL EVALUATION ===")
    print("Loading model....", end='')
    dice = losses.dice
    model = tf.keras.models.load_model('/work/earnestb13/ajustin/fronts/models/model_%d.h5' % job_number,
                                       custom_objects={'dice': dice})
    model.load_weights("/work/earnestb13/ajustin/fronts/models/model_%d_weights.h5" % job_number)
    print("done")

    fronts_filename = '/work/earnestb13/ajustin/fronts/pickle_files/2008/05/25/FrontObjects_2008052521_conus.pkl'
    sfcdata_filename = '/work/earnestb13/ajustin/fronts/pickle_files/2008/05/25/SurfaceData_2008052521_conus.pkl'
    fronts_ds = pd.read_pickle(fronts_filename)
    sfcdata_ds = pd.read_pickle(sfcdata_filename)

    lon_index = random.choices(range(289-map_size))[0]
    lat_index = random.choices(range(113-map_size))[0]
    lons = fronts_ds.longitude.values[lon_index:lon_index+map_size]
    lats = fronts_ds.latitude.values[lat_index:lat_index+map_size]

    fronts = fronts_ds.sel(longitude=lons, latitude=lats)
    fronts_new = fronts_ds.sel(longitude=lons, latitude=lats)
    sfcdata = sfcdata_ds.sel(longitude=lons, latitude=lats).to_array().T.values.reshape(1, map_size, map_size, 6)

    print("Predicting values....", end='')
    prediction = model.predict(sfcdata)
    print("done")

    print(prediction)

    identifier = np.zeros([map_size, map_size])
    print("Reformatting predictions....", end='')
    threshold = 0.5
    for i in range(0, map_size):
        for j in range(0, map_size):
            if prediction[0][j][i][1] > threshold:
                # if prediction[0][j][i][1] > prediction[0][j][i][0] and prediction[0][j][i][1] > prediction[0][j][i][2]:
                identifier[i][j] = 1
            elif prediction[0][j][i][2] > threshold:
                # elif prediction[0][j][i][2] > prediction[0][j][i][0] and prediction[0][j][i][2] > prediction[0][j][i][1]:
                identifier[i][j] = 2
    print("done")

    # The fronts dataset has dimensions latitude x longitude, while the identifier array was created with the format of
    # longitude x longitude, so we need to swap the axes.
    fronts_new.identifier.values = identifier
    time = fronts.time.values

    print("Generating plots....", end='')
    prediction_plot(fronts_new, time, job_number)
    truth_plot(fronts, time, job_number)
    print("done")


def prediction_plot(fronts_new, time, job_number):
    extent = [220, 300, 29, 53]
    ax = Plot_ERA5.plot_background(extent)
    fronts_new.identifier.plot(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree())
    plt.title("%s Prediction Plot" % time)
    plt.savefig('/work/earnestb13/ajustin/fronts/models/model_%d_prediction_plot.png' % job_number, bbox_inches='tight', dpi=300)
    plt.close()


def truth_plot(fronts, time, job_number):
    extent = [220, 300, 29, 53]
    ax = Plot_ERA5.plot_background(extent)
    fronts.identifier.plot(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree())
    plt.title("%s Truth Plot" % time)
    plt.savefig('/work/earnestb13/ajustin/fronts/models/model_%d_truth_plot.png' % job_number, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_indir', type=str, required=True, help='Path of pickle files containing front object'
                                                                        ' and surface data.')
    parser.add_argument('--file_removal', type=str, required=False, help='Remove extra files that cannot be used? '
                                                                         '(True/False)')
    parser.add_argument('--model_outdir', type=str, required=False, help='Path where models will saved to')
    parser.add_argument('--model_filepath', type=str, required=False, help='Path where model for evaluation is saved')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for Unet optimizer')
    parser.add_argument('--train_epochs', type=int, required=True, help='Number of epochs for the Unet training')
    parser.add_argument('--train_steps', type=int, required=True, help='Number of steps for each epoch in training')
    parser.add_argument('--train_batch_size', type=int, required=True, help='Batch size for the Unet training')
    parser.add_argument('--valid_steps', type=int, required=True, help='Number of steps for each epoch in validation')
    parser.add_argument('--valid_batch_size', type=int, required=True, help='Batch size for the Unet validation')
    parser.add_argument('--valid_freq', type=int, required=True, help='How many epochs to pass before each validation')
    parser.add_argument('--workers', type=int, required=True, help='Number of workers for the Unet')
    parser.add_argument('--empty_split', type=float, required=True, help='How much empty front object data to use (10% = 0.1)')
    parser.add_argument('--map_size', type=int, required=True, help='Size of the map for the Unet. This will be a square'
                                                                    ' with each side being the size of map_size. (i.e.'
                                                                    ' setting this to 32 yields a 32x32 Unet and map.')
    parser.add_argument('--job_number', type=int, required=True, help='Slurm job number')
    args = parser.parse_args()
	
    if args.pickle_indir is not None:
        frontobject_conus_files, surfacedata_conus_files = fm.load_conus_files(args.pickle_indir)
        if len(frontobject_conus_files) != len(surfacedata_conus_files):
            if args.file_removal == 'True':
                fm.file_removal(frontobject_conus_files, surfacedata_conus_files, args.pickle_indir)
                print("====> NOTE: Extra files have been removed, a new list of files will now be loaded. <====")
                frontobject_conus_files, surfacedata_conus_files = fm.load_conus_files(args.pickle_indir)
                #fm.generate_file_lists(frontobject_conus_files, surfacedata_conus_files)
            else:
                print("ERROR: The number of front object and surface data files are not the same. You must remove "
                      "the extra files before data\ncan be processed by setting the '--file_removal' argument to "
                      "'True'.")
        #fm.generate_file_lists(frontobject_conus_files, surfacedata_conus_files)
        train_unet(frontobject_conus_files, surfacedata_conus_files, args.map_size, args.empty_split, args.learning_rate,
                   args.train_epochs, args.train_steps, args.train_batch_size, args.valid_steps, args.valid_batch_size,
                   args.valid_freq, args.workers, args.job_number)
        evaluate_model(args.job_number, args.map_size)
    else:
        print("ERROR: No directory for pickle files included, please declare where pickle files are located using the "
              "'--pickle_indir'\nargument.")
