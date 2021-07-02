"""
Function that creates, trains, and validates a Unet model.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 7/2/2021 4:09 PM CDT
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
from tensorflow.keras.callbacks import EarlyStopping
import errors
from expand_fronts import one_pixel_expansion as ope


class DataGenerator(keras.utils.Sequence):
    """
    Data generator for the U-Net model.
    """

    def __init__(self, front_files, variable_files, map_dim_x, map_dim_y, batch_size, front_threshold):
        self.front_files = front_files
        self.variable_files = variable_files
        self.map_dim_x = map_dim_x
        self.map_dim_y = map_dim_y
        self.batch_size = batch_size
        self.front_threshold = front_threshold

    def __len__(self):
        return int(np.floor(len(self.front_files) / self.batch_size))

    def __getitem__(self, index):
        inputs, outputs = self.__data_generation()
        return inputs, outputs

    def __data_generation(self):
        front_dss = np.empty(shape=(self.batch_size, self.map_dim_x, self.map_dim_y, 3))
        sfcdata_dss = np.empty(shape=(self.batch_size, self.map_dim_x, self.map_dim_y, 31))
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
        for i in range(self.batch_size):
            # Open random files with random coordinate domains until a sample contains at least one front.
            identifiers = 0
            while identifiers < self.front_threshold:
                index = random.choices(range(len(self.front_files) - 1), k=1)[0]
                with open(self.front_files[index], 'rb') as front_file:
                    # front_ds = pickle.load(front_file)
                    front_ds = ope(pickle.load(front_file))
                lon_index = random.choices(range(289 - self.map_dim_x))[0]
                lat_index = random.choices(range(129 - self.map_dim_y))[0]
                lons = front_ds.longitude.values[lon_index:lon_index + self.map_dim_x]
                lats = front_ds.latitude.values[lat_index:lat_index + self.map_dim_y]
                identifiers = np.count_nonzero(front_ds.sel(longitude=lons, latitude=lats).identifier.values.flatten())
            with open(self.variable_files[index], 'rb') as sfcdata_file:
                sfcdata_ds = pickle.load(sfcdata_file)
            # print(sfcdata_ds)
            variable_list = list(sfcdata_ds.keys())
            # print(variable_list)
            for j in range(31):
                var = variable_list[j]
                # sfcdata_ds[var].values = np.nan_to_num((sfcdata_ds[var].values - means[j]) / std_devs[j])
                sfcdata_ds[var].values = np.nan_to_num((sfcdata_ds[var].values - mins[j]) / (maxs[j] - mins[j]))
                # sfcdata_ds[var].values = np.nan_to_num((sfcdata_ds[var].values - means[j]) / (maxs[j] - mins[j]))
            fronts = front_ds.sel(longitude=lons, latitude=lats).to_array().T.values
            binarized_fronts = to_categorical(fronts, num_classes=3)
            sfcdata = sfcdata_ds.sel(longitude=lons, latitude=lats).to_array().T.values
            front_dss[i] = binarized_fronts
            sfcdata_dss[i] = sfcdata
        return sfcdata_dss, front_dss


# UNet model
def train_new_unet(front_files, variable_files, map_dim_x, map_dim_y, learning_rate, train_epochs, train_steps,
                   train_batch_size, train_fronts, valid_steps, valid_batch_size, valid_freq, valid_fronts, loss,
                   workers, job_number, model_dir):
    """
    Function that trains the unet model and saves the model along with its weights.

    Parameters
    ----------
    front_files: list
        List of all files containing fronts. Includes files with no fronts present in their respective domains.
    variable_files: list
        List of all files containing variable data.
    map_dim_x: int
        Integer that determines the X dimension of the image (map) to be fed into the Unet.
    map_dim_y: int
        Integer that determines the Y dimension of the image (map) to be fed into the Unet.
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
    train_fronts: int
        Minimum number of pixels containing fronts that must be present in an image for it to be used for training the
        Unet.
    valid_steps: int
        Number of steps for each epoch in the validation generator.
    valid_batch_size: int
        Number of images/datasets to process for each step of each epoch in the validation generator.
    valid_freq: int
        This integer represents how often the model will be validated, or having its hyperparameters automatically
        tuned. For example, setting this to 4 means that the model will be validated every 4 epochs.
    valid_fronts: int
        Minimum number of pixels containing fronts that must be present in an image for it to be used in Unet
        validation.
    loss: str
        Loss function for the Unet.
    workers: int
        Number of threads that will be generating batches in parallel.
    job_number: int
        Slurm job number. This is set automatically and should not be changed by the user.
    model_dir: str
        Directory that the models are saved to.
    """
    print("\n=== MODEL TRAINING ===")
    print("Creating unet....", end='')

    model = models.unet_2d((map_dim_x, map_dim_y, 31), filter_num=[32, 64, 128, 256, 512, 1024], n_labels=3,
                           stack_num_down=5, stack_num_up=5, activation='PReLU', output_activation='Softmax',
                           batch_norm=True, pool=True, unpool=True, name='unet')

    print('done')
    print('Compiling unet....', end='')

    if loss == 'dice':
        loss_function = losses.dice
    elif loss == 'tversky':
        loss_function = losses.tversky
    elif loss == 'cce':
        loss_function = 'categorical_crossentropy'

    adam = Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=adam, metrics=tf.keras.metrics.AUC())
    print('done')

    print("Loss function = %s" % loss)

    model.summary()

    train_dataset = tf.data.Dataset.from_generator(DataGenerator,
                                                   args=[front_files, variable_files, map_dim_x, map_dim_y,
                                                         train_batch_size, train_fronts],
                                                   output_types=(tf.float32, tf.float32))

    validation_dataset = tf.data.Dataset.from_generator(DataGenerator,
                                                        args=[front_files, variable_files, map_dim_x,
                                                              map_dim_y, valid_batch_size, valid_fronts],
                                                        output_types=(tf.float32, tf.float32))

    os.mkdir('%s/model_%d' % (model_dir, job_number))
    os.mkdir('%s/model_%d/predictions' % (model_dir, job_number))
    model_filepath = '%s/model_%d/model_%d.h5' % (model_dir, job_number, job_number)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_filepath, monitor='loss', verbose=1, save_best_only=True,
                                                    save_weights_only=False, save_freq='epoch')
    early_stopping = EarlyStopping('loss', patience=500, verbose=1)

    history = model.fit(train_dataset.repeat(), validation_data=validation_dataset.repeat(), validation_freq=valid_freq,
                        epochs=train_epochs,
                        steps_per_epoch=train_steps, validation_steps=valid_steps,
                        callbacks=[early_stopping, checkpoint], verbose=2, workers=workers,
                        use_multiprocessing=True, max_queue_size=100000)

    os.rename('TrainModel_%d_stdout.txt' % job_number,
              '%s/model_%d/model_%d.txt' % (model_dir, job_number, job_number))
    with open('%s/model_%d/model_%d_history.pkl' % (model_dir, job_number, job_number), 'wb') as f:
        pickle.dump(history.history, f)
    model.save('%s/model_%d/model_%d.h5' % (model_dir, job_number, job_number))


# UNet model
def train_imported_unet(front_files, variable_files, learning_rate, train_epochs, train_steps,
                        train_batch_size, train_fronts, valid_steps, valid_batch_size, valid_freq, valid_fronts, loss,
                        workers, model_number, model_dir):
    """
    Function that trains the unet model and saves the model along with its weights.

    Parameters
    ----------
    front_files: list
        List of all files containing fronts. Includes files with no fronts present in their respective domains.
    variable_files: list
        List of all files containing variable data.
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
    train_fronts: int
        Minimum number of pixels containing fronts that must be present in an image for it to be used for training the
        Unet.
    valid_steps: int
        Number of steps for each epoch in the validation generator.
    valid_batch_size: int
        Number of images/datasets to process for each step of each epoch in the validation generator.
    valid_freq: int
        This integer represents how often the model will be validated, or having its hyperparameters automatically
        tuned. For example, setting this to 4 means that the model will be validated every 4 epochs.
    valid_fronts: int
        Minimum number of pixels containing fronts that must be present in an image for it to be used in Unet
        validation.
    loss: str
        Loss function for the Unet.
    workers: int
        Number of threads that will be generating batches in parallel.
    model_number: int
        Number of the imported model.
    model_dir: str
        Directory that the models are saved to.
    """
    print("\n=== MODEL TRAINING ===")
    print("Importing unet....", end='')

    model = tf.keras.models.load_model('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number))
    print('done')
    print('Compiling unet....', end='')

    if loss == 'dice':
        loss_function = losses.dice
    elif loss == 'tversky':
        loss_function = losses.tversky
    elif loss == 'cce':
        loss_function = 'categorical_crossentropy'

    adam = Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=adam, metrics=tf.keras.metrics.AUC())
    print('done')

    print("Loss function = %s" % loss)

    model.summary()

    map_dim_x = model.layers[0].input_shape[0][1]
    map_dim_y = model.layers[0].input_shape[0][2]

    train_dataset = tf.data.Dataset.from_generator(DataGenerator,
                                                   args=[front_files, variable_files, map_dim_x, map_dim_y,
                                                         train_batch_size, train_fronts],
                                                   output_types=(tf.float32, tf.float32))

    validation_dataset = tf.data.Dataset.from_generator(DataGenerator,
                                                        args=[front_files, variable_files, map_dim_x,
                                                              map_dim_y, valid_batch_size, valid_fronts],
                                                        output_types=(tf.float32, tf.float32))

    model_filepath = '%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_filepath, monitor='loss', verbose=1, save_best_only=True,
                                                    save_weights_only=False, save_freq='epoch')
    early_stopping = EarlyStopping('loss', patience=500, verbose=1)

    history = model.fit(train_dataset.repeat(), validation_data=validation_dataset.repeat(), validation_freq=valid_freq,
                        epochs=train_epochs, steps_per_epoch=train_steps, validation_steps=valid_steps,
                        callbacks=[early_stopping, checkpoint], verbose=2, workers=workers,
                        use_multiprocessing=True, max_queue_size=100000)

    with open('%s/model_%d/model_%d_history.pkl' % (model_dir, model_number, model_number), 'wb') as f:
        pickle.dump(history.history, f)
    model.save('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for Unet optimizer')
    parser.add_argument('--train_epochs', type=int, required=True, help='Number of epochs for the Unet training')
    parser.add_argument('--train_steps', type=int, required=True, help='Number of steps for each epoch in training')
    parser.add_argument('--train_batch_size', type=int, required=True, help='Batch size for the Unet training')
    parser.add_argument('--train_fronts', type=int, required=False, help='How many fronts must be in a dataset for it'
                                                                         'to be passed through the generator')
    parser.add_argument('--valid_steps', type=int, required=True, help='Number of steps for each epoch in validation')
    parser.add_argument('--valid_batch_size', type=int, required=True, help='Batch size for the Unet validation')
    parser.add_argument('--valid_freq', type=int, required=True, help='How many epochs to pass before each validation')
    parser.add_argument('--valid_fronts', type=int, required=False, help='How many fronts must be in a dataset for it'
                                                                         'to be passed through the generator')
    parser.add_argument('--loss', type=str, required=True, help='Loss function for the Unet')
    parser.add_argument('--workers', type=int, required=True, help='Number of workers for the Unet')
    parser.add_argument('--map_dim_x', type=int, required=False, help='X dimension of the Unet map')
    parser.add_argument('--map_dim_y', type=int, required=False, help='Y dimension of the Unet map')
    parser.add_argument('--job_number', type=int, required=False, help='Slurm job number')
    parser.add_argument('--import_model_number', type=int, required=False,
                        help='Number of the model that you would like to import.')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory where the models are or will be saved to.')
    parser.add_argument('--num_variables', type=int, required=True,
                        help='Number of variables in the variable datasets.')
    parser.add_argument('--front_types', type=str, required=True,
                        help='Front format of the file. If your files contain warm'
                             ' and cold fronts, pass this argument as CFWF.'
                             ' If your files contain only drylines, pass this argument'
                             ' as DL. If your files contain all fronts, pass this argument'
                             ' as ALL.')
    parser.add_argument('--domain', type=str, required=True, help='Domain of the data. Possible values are: conus')
    parser.add_argument('--map_dimensions', type=int, nargs=2, required=True,
                        help='Dimensions of the map size. Two integers'
                             ' need to be passed.')
    parser.add_argument('--generate_lists', type=str, required=False, help='Generate lists of new files? (True/False)')
    args = parser.parse_args()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    if args.train_fronts is None:
        print("WARNING: Argument '--train_fronts' was not passed, so it will be defaulted to 5.")
        train_fronts = 5
    else:
        train_fronts = args.train_fronts
    if args.valid_fronts is None:
        print("WARNING: Argument '--valid_fronts' was not passed, so it will be defaulted to 5.")
        valid_fronts = 5
    else:
        valid_fronts = args.valid_fronts

    if args.import_model_number is not None:
        if args.map_dim_x is not None or args.map_dim_y is not None:
            raise ValueError("Arguments '--map_dim_x' and '--map_dim_y' cannot be passed if you are importing a model.")
        else:
            print("WARNING: You have imported model %d for training." % args.import_model_number)
            front_files, variable_files = fm.load_file_lists(args.num_variables, args.front_types, args.domain,
                                                             args.map_dimensions)
            train_imported_unet(front_files, variable_files, args.learning_rate, args.train_epochs, args.train_steps,
                                args.train_batch_size, train_fronts, args.valid_steps, args.valid_batch_size,
                                args.valid_freq, valid_fronts, args.loss, args.workers, args.import_model_number,
                                args.model_dir)
    else:
        if args.job_number is None:
            raise errors.MissingArgumentError("Argument '--job_number' must be passed if you are creating a new model.")
        if args.map_dim_x is None or args.map_dim_y is None:
            raise ValueError(
                "Arguments '--map_dim_x' and '--map_dim_y' must be passed if you are creating a new model.")
        else:
            front_files, variable_files = fm.load_file_lists(args.num_variables, args.front_types, args.domain,
                                                             args.map_dimensions)
            train_new_unet(front_files, variable_files, args.map_dim_x, args.map_dim_y, args.learning_rate,
                           args.train_epochs, args.train_steps, args.train_batch_size, train_fronts,
                           args.valid_steps, args.valid_batch_size, args.valid_freq, valid_fronts, args.loss,
                           args.workers, args.job_number, args.model_dir)
