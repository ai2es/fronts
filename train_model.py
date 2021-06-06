"""
Function that creates, trains, and validates a Unet model.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 6/6/2021 11:45 AM CDT
"""

import random
import argparse

import keras.utils
from keras_unet_collection import models, losses
import tensorflow as tf
from keras.optimizers import Adam
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
import file_manager as fm
import os
import evaluate_model

class DataGenerator(keras.utils.Sequence):
    def __init__(self, front_files_list, sfcdata_files_list, map_size, batch_size):
        self.front_files_list = front_files_list
        self.sfcdata_files_list = sfcdata_files_list
        self.map_size = map_size
        self.batch_size = batch_size

    def __len__(self):
        return int(np.floor(len(self.front_files_list) / self.batch_size))

    def __getitem__(self, index):
        input, output = self.__data_generation()
        return input, output

    def __data_generation(self):
        front_dss = np.empty(shape=(self.batch_size,self.map_size,self.map_size,3))
        sfcdata_dss = np.empty(shape=(self.batch_size,self.map_size,self.map_size,6))
        for i in range(self.batch_size):
            # Open random files with random coordinate domains until a sample contains at least one front.
            identifiers = 0
            while identifiers == 0:
                index = random.choices(range(len(self.front_files_list)-1), k=1)[0]
                with open(self.front_files_list[index], 'rb') as front_file:
                    front_ds = pickle.load(front_file)
                lon_index = random.choices(range(289-self.map_size))[0]
                lat_index = random.choices(range(113-self.map_size))[0]
                lons = front_ds.longitude.values[lon_index:lon_index+self.map_size]
                lats = front_ds.latitude.values[lat_index:lat_index+self.map_size]
                identifiers = np.count_nonzero(front_ds.sel(longitude=lons, latitude=lats).identifier.values.flatten())
            with open(self.sfcdata_files_list[index], 'rb') as sfcdata_file:
                sfcdata_ds = pickle.load(sfcdata_file)
            fronts = front_ds.sel(longitude=lons, latitude=lats).to_array().T.values
            binarized_fronts = to_categorical(fronts, num_classes=3)
            sfcdata = sfcdata_ds.sel(longitude=lons, latitude=lats).to_array().T.values
            front_dss[i] = binarized_fronts
            sfcdata_dss[i] = sfcdata
        return sfcdata_dss, front_dss

# def training_generator(front_files_list, sfcdata_files_list, map_size, input_name, output_name, batch_size):
#     """
#     Training generator for the Unet that generates random batches of data.
#
#     Parameters
#     ----------
#     front_files_list: list
#         List of all files containing fronts. Includes files with no fronts present in their respective domains.
#     sfcdata_files_list: list
#         List of all files containing surface data.
#     input_name: str
#         Name of the input layer of the Unet.
#     output_name: str
#         Name of the output layer of the Unet.
#     map_size: int
#         Integer that determines the size of the window to be fed into the Unet. (e.g. Setting this to 32 would create a
#         Unet that takes input shape (batch_size, 32, 32, num_predictors)). This MUST be a multiple of 2^n.
#     batch_size: int
#         Number of images/datasets to process for each step of each epoch in the training generator. This is determined
#         by the 'train_batch_size' argument.
#     """
#     while True:
#         front_dss = np.empty(shape=(batch_size,map_size,map_size,3))
#         sfcdata_dss = np.empty(shape=(batch_size,map_size,map_size,6))
#         for i in range(batch_size):
#             # Open random files with random coordinate domains until a sample contains at least one front.
#             identifiers = 0
#             while identifiers == 0:
#                 index = random.choices(range(len(front_files_list)-1), k=1)[0]
#                 with open(front_files_list[index], 'rb') as front_file:
#                     front_ds = pickle.load(front_file)
#                 lon_index = random.choices(range(289-map_size))[0]
#                 lat_index = random.choices(range(113-map_size))[0]
#                 lons = front_ds.longitude.values[lon_index:lon_index+map_size]
#                 lats = front_ds.latitude.values[lat_index:lat_index+map_size]
#                 identifiers = np.count_nonzero(front_ds.sel(longitude=lons, latitude=lats).identifier.values.flatten())
#             with open(sfcdata_files_list[index], 'rb') as sfcdata_file:
#                 sfcdata_ds = pickle.load(sfcdata_file)
#             fronts = front_ds.sel(longitude=lons, latitude=lats).to_array().T.values
#             binarized_fronts = to_categorical(fronts, num_classes=3)
#             sfcdata = sfcdata_ds.sel(longitude=lons, latitude=lats).to_array().T.values
#             front_dss[i] = binarized_fronts
#             sfcdata_dss[i] = sfcdata
#         yield ({input_name: sfcdata_dss},
#                {output_name: front_dss})

# def validation_generator(front_files_list, sfcdata_files_list, map_size, input_name, output_name,
#                          batch_size, valid_fronts):
#     """
#     Validation generator for the Unet that generates random batches of data.
#
#     Parameters
#     ----------
#     front_files_list: list
#         List of all files containing fronts. Includes files with no fronts present in their respective domains.
#     sfcdata_files_list: list
#         List of all files containing surface data.
#     input_name: str
#         Name of the input layer of the Unet.
#     output_name: str
#         Name of the output layer of the Unet.
#     map_size: int
#         Integer that determines the size of the window to be fed into the Unet. (e.g. Setting this to 32 would create a
#         Unet that takes input shape (batch_size, 32, 32, num_predictors)). This MUST be a multiple of 2^n.
#     batch_size: int
#         Number of images/datasets to process for each step of each epoch in the training generator. This is determined
#         by the 'valid_batch_size' argument.
#     """
#     while True:
#         front_dss = np.empty(shape=(batch_size,map_size,map_size,3))
#         sfcdata_dss = np.empty(shape=(batch_size,map_size,map_size,6))
#         for i in range(batch_size):
#             identifiers = 0
#             # Open random files with random coordinate domains until a sample contains at least valid_fronts fronts. This is to
#             # prevent the model from predicting too few fronts.
#             while identifiers < valid_fronts:
#                 index = random.choices(range(len(front_files_list)-1), k=1)[0]
#                 with open(front_files_list[index], 'rb') as front_file:
#                     front_ds = pickle.load(front_file)
#                 lon_index = random.choices(range(289-map_size))[0]
#                 lat_index = random.choices(range(113-map_size))[0]
#                 lons = front_ds.longitude.values[lon_index:lon_index+map_size]
#                 lats = front_ds.latitude.values[lat_index:lat_index+map_size]
#                 identifiers = np.count_nonzero(front_ds.sel(longitude=lons, latitude=lats).identifier.values.flatten())
#             with open(sfcdata_files_list[index], 'rb') as sfcdata_file:
#                 sfcdata_ds = pickle.load(sfcdata_file)
#             fronts = front_ds.sel(longitude=lons, latitude=lats).to_array().T.values
#             binarized_fronts = to_categorical(fronts, num_classes=3)
#             sfcdata = sfcdata_ds.sel(longitude=lons, latitude=lats).to_array().T.values
#             front_dss[i] = binarized_fronts
#             sfcdata_dss[i] = sfcdata
#         yield ({input_name: sfcdata_dss},
#                {output_name: front_dss})



# UNet model
def train_unet(front_files_list, sfcdata_files_list, map_size, learning_rate, train_epochs, train_steps,
               train_batch_size, valid_steps, valid_batch_size, valid_freq, valid_fronts, loss, workers, job_number):
    """
    Function that trains the unet model and saves the model along with its weights.

    Parameters
    ----------
    front_files_list: list
        List of all files containing fronts. Includes files with no fronts present in their respective domains.
    sfcdata_files_list: list
        List of all files containing surface data.
    map_size: int
        Integer that determines the size of the window to be fed into the Unet. (e.g. Setting this to 32 would create a
        Unet that takes input shape (batch_size, 32, 32, num_predictors)). This MUST be a multiple of 2^n.
    learning_rate: float
        Value that determines how fast the optimization algorithm overrides old information (how fast the Unet learns).
    train_epochs: int
        Number of times that the Unet will cycle over the data, or in this case, the number of times that the model will
        run the training generator.
    train_steps: int
        Number of steps per epoch for the training generator. This is the number of times that a batch will be generated
        before the generator moves onto the next epoch.
    train_batch_size: int
        Number of images/datasets to process for each step of each epoch in the training generator.
    valid_steps: int
        Number of steps for each epoch in the validation generator.
    valid_batch_size: int
        Number of images/datasets to process for each step of each epoch in the validation generator.
    valid_freq: int
        This integer represents how often the model will be validated, or having its hyperparameters automatically
        tuned. For example, setting this to 4 means that the model will be validated every 4 epochs.
    loss: str
        Loss function for the Unet.
    workers: int
        Number of threads that will be generating batches in parallel.
    job_number: int
        Slurm job number. This is set automatically and should not be changed by the user.
    """
    print("\n=== MODEL TRAINING ===")
    print("Creating unet....", end='')

    model = models.unet_2d((map_size, map_size, 6), filter_num=[4, 8, 16, 32, 64], n_labels=3, stack_num_down=5,
                           stack_num_up=5, activation='ReLU', output_activation='Softmax',
                           batch_norm=True, pool=True, unpool=True, name='unet')
    print('done')
    print('Compiling unet....', end='')

    if loss == 'dice':
        loss_function = losses.dice
    elif loss == 'tversky':
        loss_function = losses.tversky

    adam = Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=adam, metrics=tf.keras.metrics.AUC())
    print('done')

    print("Loss function = %s" % loss)

    model.summary()

    layer_names = [layer.name for layer in model.layers]
    input_name = layer_names[0]
    output_name = layer_names[-1]

    train_dataset = tf.data.Dataset.from_generator(DataGenerator,
                                                   args = [front_files_list, sfcdata_files_list, map_size, train_batch_size],
                                                   output_types=(tf.float32, tf.float32))

    validation_dataset = tf.data.Dataset.from_generator(DataGenerator,
                                                        args = [front_files_list, sfcdata_files_list, map_size, valid_batch_size],
                                                        output_types=(tf.float32, tf.float32))

    #train_gen = DataGenerator(front_files_list, sfcdata_files_list, map_size, input_name, output_name,
    #                          train_batch_size)
    #valid_gen = DataGenerator(front_files_list, sfcdata_files_list, map_size, input_name, output_name,
    #                          valid_batch_size)

    history = model.fit(train_dataset, validation_data=validation_dataset, validation_freq=valid_freq, epochs=train_epochs,
                        steps_per_epoch=train_steps, validation_steps=valid_steps, verbose=2, workers=workers, use_multiprocessing=True)

    os.mkdir('/work/earnestb13/ajustin/fronts/models/model_%d' % job_number)
    os.mkdir('/work/earnestb13/ajustin/fronts/models/model_%d/predictions' % job_number)
    os.rename('/work/earnestb13/ajustin/fronts/TrainModel_%d_stdout.txt' % job_number,
              '/work/earnestb13/ajustin/fronts/models/model_%d/model_%d.txt' % (job_number, job_number))
    model.save('/work/earnestb13/ajustin/fronts/models/model_%d/model_%d.h5' % (job_number, job_number))
    model.save_weights("/work/earnestb13/ajustin/fronts/models/model_%d/model_%d_weights.h5" % (job_number, job_number))
    evaluate_model.evaluate_model(job_number, map_size, front_files_list, sfcdata_files_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_indir', type=str, required=True, help='Path of pickle files containing front object'
                                                                        ' and surface data.')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for Unet optimizer')
    parser.add_argument('--train_epochs', type=int, required=True, help='Number of epochs for the Unet training')
    parser.add_argument('--train_steps', type=int, required=True, help='Number of steps for each epoch in training')
    parser.add_argument('--train_batch_size', type=int, required=True, help='Batch size for the Unet training')
    parser.add_argument('--valid_steps', type=int, required=True, help='Number of steps for each epoch in validation')
    parser.add_argument('--valid_batch_size', type=int, required=True, help='Batch size for the Unet validation')
    parser.add_argument('--valid_freq', type=int, required=True, help='How many epochs to pass before each validation')
    parser.add_argument('--valid_fronts', type=int, required=True, help='How many fronts must be in a dataset for it'
                                                                        'to be passed through the validation generator')
    parser.add_argument('--loss', type=str, required=True, help='Loss function for the Unet')
    parser.add_argument('--workers', type=int, required=True, help='Number of workers for the Unet')
    parser.add_argument('--map_size', type=int, required=True, help='Size of the map for the Unet. This will be a square'
                                                                    ' with each side being the size of map_size. (i.e.'
                                                                    ' setting this to 32 yields a 32x32 Unet and map.')
    parser.add_argument('--job_number', type=int, required=True, help='Slurm job number')
    args = parser.parse_args()

    front_files_list, sfcdata_files_list = fm.load_file_lists()
    train_unet(front_files_list, sfcdata_files_list, args.map_size, args.learning_rate,
               args.train_epochs, args.train_steps, args.train_batch_size, args.valid_steps, args.valid_batch_size,
               args.valid_freq, args.valid_fronts, args.loss, args.workers, args.job_number)
