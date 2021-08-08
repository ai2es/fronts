"""
Function that trains a new or imported U-Net model.
Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 8/8/2021 2:05 PM CDT
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
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import errors
import custom_losses
from expand_fronts import one_pixel_expansion as ope
import pandas as pd


class DataGenerator(keras.utils.Sequence):
    """
    Data generator for the U-Net model.
    """
    def __init__(self, front_files, variable_files, map_dim_x, map_dim_y, batch_size, front_threshold, front_types,
                 normalization_method, num_classes, pixel_expansion, num_variables, file_dimensions):
        self.front_files = front_files
        self.variable_files = variable_files
        self.map_dim_x = map_dim_x
        self.map_dim_y = map_dim_y
        self.batch_size = batch_size
        self.front_threshold = front_threshold
        self.front_types = front_types  # NOTE: This is passed into the generator as a bytes object, NOT a string
        self.normalization_method = normalization_method
        self.num_classes = num_classes
        self.pixel_expansion = pixel_expansion
        self.num_variables = num_variables
        self.file_dimensions = file_dimensions

    def __len__(self):
        return int(np.floor(len(self.front_files) / self.batch_size))

    def __getitem__(self, index):
        inputs, outputs = self.__data_generation()
        return inputs, outputs

    def __data_generation(self):
        front_dss = np.empty(shape=(self.batch_size, self.map_dim_x, self.map_dim_y, self.num_classes))
        variable_dss = np.empty(shape=(self.batch_size, self.map_dim_x, self.map_dim_y, self.num_variables))

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

        for i in range(self.batch_size):
            # Open random files with random coordinate domains until a sample contains at least one front.
            identifiers = 0
            while identifiers < self.front_threshold:
                index = random.choices(range(len(self.front_files) - 1), k=1)[0]
                with open(self.front_files[index], 'rb') as front_file:
                    if self.pixel_expansion == 0:
                        front_ds = pickle.load(front_file)  # No expansion
                    elif self.pixel_expansion == 1:
                        front_ds = ope(pickle.load(front_file))  # 1-pixel expansion
                    elif self.pixel_expansion == 2:
                        front_ds = ope(ope(pickle.load(front_file)))  # 2-pixel expansion

                # Select a random portion of the map
                lon_index = random.choices(range(self.file_dimensions[0] - self.map_dim_x))[0]
                lat_index = random.choices(range(self.file_dimensions[1] - self.map_dim_y))[0]
                lons = front_ds.longitude.values[lon_index:lon_index + self.map_dim_x]
                lats = front_ds.latitude.values[lat_index:lat_index + self.map_dim_y]

                identifiers = np.count_nonzero(front_ds.sel(longitude=lons, latitude=lats).identifier.values.flatten())
            with open(self.variable_files[index], 'rb') as variable_file:
                variable_ds = pickle.load(variable_file)
            variable_list = list(variable_ds.keys())
            for j in range(self.num_variables):
                var = variable_list[j]
                if self.normalization_method == 1:
                    # Min-max normalization
                    variable_ds[var].values = np.nan_to_num((variable_ds[var].values - norm_params.loc[var,'Min']) /
                                                            (norm_params.loc[var,'Max'] - norm_params.loc[var,'Min']))
                elif self.normalization_method == 2:
                    # Mean normalization
                    variable_ds[var].values = np.nan_to_num((variable_ds[var].values - norm_params.loc[var,'Mean']) /
                                                            (norm_params.loc[var,'Max'] - norm_params.loc[var,'Min']))
            fronts = front_ds.sel(longitude=lons, latitude=lats).to_array().T.values
            if self.front_types == b'SFOF':
                # Prevents generator from making 5 classes
                fronts = np.where(np.where(fronts == 3, 1, fronts) == 4, 2, np.where(fronts == 3, 1, fronts))
            elif self.front_types == b'DL':
                # Prevents generator from making 6 classes
                fronts = np.where(np.where(fronts == 5, 1, fronts))
            """
            tensorflow.keras.utils.to_categorical: binarizes identifier (front label) values
            Example: [0, 2, 3, 4] ---> [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]] 
            """
            binarized_fronts = to_categorical(fronts, num_classes=self.num_classes)

            variable = variable_ds.sel(longitude=lons, latitude=lats).to_array().T.values
            front_dss[i] = binarized_fronts
            variable_dss[i] = variable
        return variable_dss, front_dss


# Creating and training a new U-Net model
def train_new_unet(front_files, variable_files, map_dim_x, map_dim_y, learning_rate, train_epochs, train_steps,
                   train_batch_size, train_fronts, valid_steps, valid_batch_size, valid_freq, valid_fronts, loss,
                   workers, job_number, model_dir, front_types, normalization_method, fss_mask_size, pixel_expansion,
                   num_variables, file_dimensions):
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
    front_types: str
        Fronts in the data.
    normalization_method: int
        Normalization method for the data (described near the end of the script).
    fss_mask_size: int
        Size of the mask for the FSS loss function.
    pixel_expansion: int
        Number of pixels to expand the fronts by in all directions.
    num_variables: int
        Number of variables in the datasets.
    file_dimensions: int (x2)
        Dimensions of the data files.
    """

    print("Creating unet....", end='')

    if front_types == 'CFWF' or front_types == 'SFOF':
        num_classes = 3
    elif front_types == 'DL':
        num_classes = 2
    else:
        num_classes = 6

    """
    Splitting a new model across multiple GPUs using tf.distribute.MultiWorkerMirroredStrategy()
    
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    
    with strategy.scope():
    
        model = models.unet_2d((map_dim_x, map_dim_y, num_variables), filter_num=[64, 128, 256, 512, 1024, 2048], n_labels=num_classes,
            stack_num_down=5, stack_num_up=5, activation='PReLU', output_activation='Softmax', batch_norm=True, pool=True,
            unpool=True, name='unet')
            
        if loss == 'dice':
            loss_function = losses.dice
        elif loss == 'tversky':
            loss_function = losses.tversky
        elif loss == 'cce':
            loss_function = 'categorical_crossentropy'
        elif loss == 'fss':
            loss_function = custom_losses.make_FSS_loss(fss_mask_size)
        
        adam = Adam(learning_rate=learning_rate)
        model.compile(loss=loss_function, optimizer=adam, metrics=tf.keras.metrics.AUC())
    
    model.summary()
    ....
    ....
    model.fit(...)
    """

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    with strategy.scope():
        """
        U-Net examples
        Visit the keras_unet_collection user guide for help with making models: https://pypi.org/project/keras-unet-collection/
        
        === U-Net ===
        model = models.unet_2d((map_dim_x, map_dim_y, num_variables), filter_num=[32, 64, 128, 256, 512, 1024], n_labels=num_classes,
            stack_num_down=5, stack_num_up=5, activation='PReLU', output_activation='Softmax', batch_norm=True, pool=True,
            unpool=True, name='unet')
        
        === U-Net++ ===
        model = models.unet_plus_2d((map_dim_x, map_dim_y, num_variables), filter_num=[32, 64, 128, 256, 512, 1024], n_labels=num_classes,
            stack_num_down=5, stack_num_up=5, activation='PReLU', output_activation='Softmax', batch_norm=True, pool=True,
            unpool=True, name='unet', deep_supervision=True)
        
        === U-Net 3+ ===
        model = models.unet_3plus_2d((map_dim_x, map_dim_y, num_variables), filter_num_down=[32, 64, 128, 256, 512, 1024],
            filter_num_skip='auto', filter_num_aggregate='auto', n_labels=num_classes, stack_num_down=5, stack_num_up=5,
            activation='PReLU', output_activation='Softmax', batch_norm=True, pool=True, unpool=True, name='unet',
            deep_supervision=True)
        """
        model = models.unet_3plus_2d((map_dim_x, map_dim_y, num_variables), filter_num_down=[64, 128, 256, 512, 1024, 2048],
            filter_num_skip='auto', filter_num_aggregate='auto', n_labels=num_classes, stack_num_down=5, stack_num_up=5,
            activation='PReLU', output_activation='Softmax', batch_norm=True, pool=True, unpool=True, name='unet',
            deep_supervision=True)
        print('done')

        if loss == 'dice':
            loss_function = losses.dice
        elif loss == 'tversky':
            loss_function = losses.tversky
        elif loss == 'cce':
            loss_function = 'categorical_crossentropy'
        elif loss == 'fss':
            loss_function = custom_losses.make_FSS_loss(fss_mask_size)

        print('Compiling unet....', end='')
        adam = Adam(learning_rate=learning_rate)
        model.compile(loss=loss_function, optimizer=adam, metrics=tf.keras.metrics.AUC())
        print('done')

    model.summary()

    train_dataset = tf.data.Dataset.from_generator(DataGenerator, args=[front_files, variable_files, map_dim_x,
        map_dim_y, train_batch_size, train_fronts, front_types, normalization_method, num_classes, pixel_expansion,
        num_variables, file_dimensions],
        output_types=(tf.float16, tf.float16))

    validation_dataset = tf.data.Dataset.from_generator(DataGenerator, args=[front_files, variable_files, map_dim_x,
        map_dim_y, valid_batch_size, valid_fronts, front_types, normalization_method, num_classes, pixel_expansion,
        num_variables, file_dimensions],
        output_types=(tf.float16, tf.float16))

    os.mkdir('%s/model_%d' % (model_dir, job_number))  # Make folder for model
    os.mkdir('%s/model_%d/predictions' % (model_dir, job_number))  # Make folder for model predictions
    model_filepath = '%s/model_%d/model_%d.h5' % (model_dir, job_number, job_number)
    history_filepath = '%s/model_%d/model_%d_history.csv' % (model_dir, job_number, job_number)

    # ModelCheckpoint: saves model at a specified interval
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_filepath, monitor='loss', verbose=1, save_best_only=True,
        save_weights_only=False, save_freq='epoch')
    # EarlyStopping: stops training early if a metric does not improve after a specified number of epochs (patience)
    early_stopping = EarlyStopping('loss', patience=500, verbose=1)
    history_logger = CSVLogger(history_filepath, separator=",", append=True)  # Saves loss/AUC data every epoch

    model.fit(train_dataset.repeat(), validation_data=validation_dataset.repeat(), validation_freq=valid_freq,
        epochs=train_epochs, steps_per_epoch=train_steps, validation_steps=valid_steps, callbacks=[early_stopping,
        checkpoint, history_logger], verbose=2, workers=workers, use_multiprocessing=True, max_queue_size=100000)


# Retraining a U-Net model
def train_imported_unet(front_files, variable_files, learning_rate, train_epochs, train_steps, train_batch_size,
    train_fronts, valid_steps, valid_batch_size, valid_freq, valid_fronts, loss, workers, model_number, model_dir,
    front_types, normalization_method, fss_mask_size, pixel_expansion, num_variables, file_dimensions):
    """
    Function that trains the U-Net model and saves the model along with its weights.
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
    front_types: str
        Fronts in the data.
    normalization_method: int
        Normalization method for the data (described near the end of the script).
    fss_mask_size: int
        Size of the mask for the FSS loss function.
    pixel_expansion: int
        Number of pixels to expand the fronts by in all directions.
    num_variables: int
        Number of variables in the datasets.
    file_dimensions: int (x2)
        Dimensions of the data files.
    """

    if front_types == 'CFWF' or front_types == 'SFOF':
        num_classes = 3
    elif front_types == 'DL':
        num_classes = 2
    else:
        num_classes = 6

    """
    Splitting an imported model across multiple GPUs using tf.distribute.MultiWorkerMirroredStrategy()
    
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    
    with strategy.scope():
        print("Importing unet....", end='')
        if loss == 'dice':
            loss_function = losses.dice
            model = tf.keras.models.load_model('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number),
                custom_objects={'dice': loss_function})
        elif loss == 'tversky':
            loss_function = losses.tversky
            model = tf.keras.models.load_model('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number),
                custom_objects={'tversky': loss_function})
        elif loss == 'cce':
            loss_function = 'categorical_crossentropy'
            model = tf.keras.models.load_model('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number))
        elif loss == 'fss':
            loss_function = custom_losses.make_FSS_loss(fss_mask_size)
            model = tf.keras.models.load_model('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number),
                custom_objects={'FSS_loss': loss_function})
        print("done")
    
        print('Compiling unet....', end='')
        adam = Adam(learning_rate=learning_rate)
        model.compile(loss=loss_function, optimizer=adam, metrics=tf.keras.metrics.AUC())
        print('done')
    
    model.summary()
    ....
    ....
    model.fit(...)
    """

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    with strategy.scope():

        print("Importing unet....", end='')
        if loss == 'dice':
            loss_function = losses.dice
            model = tf.keras.models.load_model('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number),
                custom_objects={'dice': loss_function})
        elif loss == 'tversky':
            loss_function = losses.tversky
            model = tf.keras.models.load_model('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number),
                custom_objects={'tversky': loss_function})
        elif loss == 'cce':
            loss_function = 'categorical_crossentropy'
            model = tf.keras.models.load_model('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number))
        elif loss == 'fss':
            loss_function = custom_losses.make_FSS_loss(fss_mask_size)
            model = tf.keras.models.load_model('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number),
                custom_objects={'FSS_loss': loss_function})
        print("done")

        print('Compiling unet....', end='')
        adam = Adam(learning_rate=learning_rate)
        model.compile(loss=loss_function, optimizer=adam, metrics=tf.keras.metrics.AUC())
        print('done')

    model.summary()

    map_dim_x = model.layers[0].input_shape[0][1]  # Longitudinal dimension of the U-Net
    map_dim_y = model.layers[0].input_shape[0][2]  # Latitudinal dimension of the U-Net

    train_dataset = tf.data.Dataset.from_generator(DataGenerator, args=[front_files, variable_files, map_dim_x,
        map_dim_y, train_batch_size, train_fronts, front_types, normalization_method, num_classes, pixel_expansion,
        num_variables, file_dimensions],
        output_types=(tf.float16, tf.float16))

    validation_dataset = tf.data.Dataset.from_generator(DataGenerator, args=[front_files, variable_files, map_dim_x,
        map_dim_y, valid_batch_size, valid_fronts, front_types, normalization_method, num_classes, pixel_expansion,
        num_variables, file_dimensions],
        output_types=(tf.float16, tf.float16))

    model_filepath = '%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number)
    history_filepath = '%s/model_%d/model_%d_history.csv' % (model_dir, model_number, model_number)

    # ModelCheckpoint: saves model at a specified interval
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_filepath, monitor='loss', verbose=1, save_best_only=True,
        save_weights_only=False, save_freq='epoch')
    # EarlyStopping: stops training early if a metric does not improve after a specified number of epochs (patience)
    early_stopping = EarlyStopping('loss', patience=500, verbose=1)
    history_logger = CSVLogger(history_filepath, separator=",", append=True)  # Saves loss/AUC data every epoch

    model.fit(train_dataset.repeat(), validation_data=validation_dataset.repeat(), validation_freq=valid_freq,
        epochs=train_epochs, steps_per_epoch=train_steps, validation_steps=valid_steps, callbacks=[early_stopping,
        checkpoint, history_logger], verbose=2, workers=workers, use_multiprocessing=True, max_queue_size=100000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    """
    Required arguments
    """
    parser.add_argument('--loss', type=str, required=True, help='Loss function for the U-Net')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory where the models are or will be saved to.')
    parser.add_argument('--workers', type=int, required=True, help='Number of workers for training the U-Net')
    parser.add_argument('--num_variables', type=int, required=True, help='Number of variables in the variable datasets.')
    parser.add_argument('--front_types', type=str, required=True, help='Front format of the file. If your files contain '
        'warm and cold fronts, pass this argument as CFWF. If your files contain only drylines, pass this argument as '
        'DL. If your files contain all fronts, pass this argument as ALL.')
    parser.add_argument('--domain', type=str, required=True, help='Domain of the data. Possible values are: conus')
    parser.add_argument('--file_dimensions', type=int, nargs=2, required=True, help='Dimensions of the file size. Two '
        'integers need to be passed.')
    parser.add_argument('--pixel_expansion', type=int, required=True, help='Number of pixels to expand the fronts by')
    # Normalization methods: 0 - No normalization, 1 - Min-max normalization, 2 - Mean normalization
    parser.add_argument('--normalization_method', type=int, required=True, help='Normalization method for the data.')

    """
    Optional arguments: these will default to a specified value if not explicitly passed.
    """
    parser.add_argument('--learning_rate', type=float, required=False, help='Learning rate for U-Net optimizer '
        '(Default: 1e-4)')
    parser.add_argument('--epochs', type=int, required=False, help='Number of epochs for the U-Net training '
        '(Default: 10000)')
    parser.add_argument('--train_valid_steps', type=int, required=False, nargs=2, help='Number of steps for each epoch. '
        '(Default: 20 20)')
    parser.add_argument('--train_valid_batch_size', type=int, required=False, nargs=2, help='Batch sizes for the U-Net.'
        '(Default: 64 64)')
    parser.add_argument('--train_valid_fronts', type=int, required=False, nargs=2, help='How many pixels with fronts an image must have for it'
        'to be passed through the generator. (Default: 5 5)')
    parser.add_argument('--valid_freq', type=int, required=False, help='How many epochs to pass before each validation (Default: 3)')


    """
    Conditional arguments
    """
    ### Must be passed if you are using the fractions skill score (FSS) loss function ###
    parser.add_argument('--fss_mask_size', type=int, required=False, help='Mask size for the FSS loss function'
        ' (if applicable).')
    ### Can only be passed when training a NEW U-Net ###
    parser.add_argument('--map_dim_x_y', type=int, required=False, nargs=2, help='Dimensions of the Unet map')
    parser.add_argument('--job_number', type=int, required=False, help='Slurm job number')
    ### Can only be passed when IMPORTING a U-Net ###
    parser.add_argument('--import_model_number', type=int, required=False, help='Number of the model that you would '
        'like to import.')

    args = parser.parse_args()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    print("=== ARGUMENTS/HYPERPARAMETERS ===")

    if args.loss == 'fss' and args.fss_mask_size is None:
        raise errors.MissingArgumentError("Argument '--fss_mask_size' must be passed if you are using the FSS loss function.")
    if args.loss != 'fss' and args.fss_mask_size is not None:
        raise errors.MissingArgumentError("Argument '--fss_mask_size' can only be passed if you are using the FSS loss function.")

    print("Model directory: %s" % args.model_dir)

    if args.domain is None:
        domain = 'conus'
        print("Domain: %s (default)" % domain)
    else:
        domain = args.domain
        print("Domain: %s" % domain)

    print("File dimensions: %d x %d" % (args.file_dimensions[0], args.file_dimensions[1]))
    print("Front types: %s" % args.front_types)

    if args.learning_rate is None:
        learning_rate = 0.0001
        print("Learning rate: %f (default)" % learning_rate)
    else:
        learning_rate = args.learning_rate
        print("Learning rate: %f" % learning_rate)

    if args.loss is None:
        loss = 'cce'
        print("Loss function: %s (default)" % loss)
    else:
        loss = args.loss
        if loss == 'fss':
            print("Loss function: FSS(%d)" % args.fss_mask_size)
        else:
            print("Loss function: %s" % loss)

    if args.map_dim_x_y is not None:
        print('Map dimensions (U-Net): %d x %d' % (args.map_dim_x_y[0], args.map_dim_x_y[1]))

    if args.normalization_method is None:
        normalization_method = 0
        print('Normalization method: 0 - No normalization (default)')
    else:
        normalization_method = args.normalization_method
        if normalization_method == 1:
            print('Normalization method: 1 - Min-max normalization')
        elif normalization_method == 2:
            print('Normalization method: 2 - Mean normalization')

    print("Number of variables: %d" % args.num_variables)

    if args.pixel_expansion is None:
        pixel_expansion = 0
        print("Pixel expansion: %d (default)" % pixel_expansion)
    else:
        pixel_expansion = args.pixel_expansion
        print("Pixel expansion: %d" % pixel_expansion)

    if args.epochs is None:
        epochs = 10000
        print("Epochs: %d (default)" % epochs)
    else:
        epochs = args.epochs
        print("Epochs: %d" % epochs)

    if args.train_valid_batch_size is None:
        train_valid_batch_size = (64, 64)
        print("Training/validation batch size: %d/%d (default)" % (train_valid_batch_size[0], train_valid_batch_size[1]))
    else:
        train_valid_batch_size = args.train_valid_batch_size
        print("Training/validation batch size: %d/%d" % (train_valid_batch_size[0], train_valid_batch_size[1]))

    if args.train_valid_fronts is None:
        train_valid_fronts = (5, 5)
        print("Training/validation front threshold: %d/%d (default)" % (train_valid_fronts[0], train_valid_fronts[1]))
    else:
        train_valid_fronts = args.train_valid_fronts
        print("Training/validation front threshold: %d/%d" % (train_valid_fronts[0], train_valid_fronts[1]))

    if args.train_valid_steps is None:
        train_valid_steps = (20, 20)
        print("Training/validation steps per epoch: %d/%d (default)" % (train_valid_steps[0], train_valid_steps[1]))
    else:
        train_valid_steps = args.train_valid_steps
        print("Training/validation steps per epoch: %d/%d" % (train_valid_steps[0], train_valid_steps[1]))

    if args.valid_freq is None:
        valid_freq = 3
        print("Validation frequency: %d epochs (default)" % valid_freq)
    else:
        valid_freq = args.valid_freq
        print("Validation frequency: %d epochs" % valid_freq)

    print("\n")

    if args.import_model_number is not None:
        if args.map_dim_x_y is not None:
            raise ValueError("Argument '--map_dim_x_y' cannot be passed if you are importing a model.")
        else:
            print("WARNING: You have imported model %d for training." % args.import_model_number)
            front_files, variable_files = fm.load_file_lists(args.num_variables, args.front_types, args.domain,
                args.file_dimensions)
            train_imported_unet(front_files, variable_files, learning_rate, epochs, train_valid_steps[0], train_valid_batch_size[0],
                train_valid_fronts[0], train_valid_steps[1], train_valid_batch_size[1], valid_freq, train_valid_fronts[1],
                loss, args.workers, args.import_model_number, args.model_dir, args.front_types, normalization_method, args.fss_mask_size,
                pixel_expansion, args.num_variables, args.file_dimensions)
    else:
        if args.job_number is None:
            raise errors.MissingArgumentError("Argument '--job_number' must be passed if you are creating a new model.")
        if args.map_dim_x_y is None:
            raise ValueError(
                "Argument '--map_dim_x_y' must be passed if you are creating a new model.")
        else:
            front_files, variable_files = fm.load_file_lists(args.num_variables, args.front_types, args.domain,
                args.file_dimensions)
            train_new_unet(front_files, variable_files, args.map_dim_x_y[0], args.map_dim_x_y[1], learning_rate, epochs,
                train_valid_steps[0], train_valid_batch_size[0], train_valid_fronts[0], train_valid_steps[1], train_valid_batch_size[1],
                valid_freq, train_valid_fronts[1], loss, args.workers, args.job_number, args.model_dir, args.front_types,
                normalization_method, args.fss_mask_size, pixel_expansion, args.num_variables, args.file_dimensions)
