"""
Function that trains a new or imported U-Net model.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 9/23/2021 8:48 PM CDT
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
from errors import check_arguments
import errors
import custom_losses
import custom_models
from expand_fronts import one_pixel_expansion as ope
import pandas as pd


class DataGenerator_2D(tf.keras.utils.Sequence):
    """
    Data generator for 2D U-Net models that grabs random files for training and validation.
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
        return int(np.floor(len(self.front_files) / self.batch_size))  # Tells model how many batches to expect

    def __getitem__(self, index):
        inputs, outputs = self.__data_generation()
        return inputs, outputs

    def __data_generation(self):
        front_dss = np.empty(shape=(self.batch_size, self.map_dim_x, self.map_dim_y, self.num_classes))
        variable_dss = np.empty(shape=(self.batch_size, self.map_dim_x, self.map_dim_y, self.num_variables))

        norm_params = pd.read_csv('normalization_parameters.csv', index_col='Variable')

        for i in range(self.batch_size):
            # Open random files with random coordinate domains until a sample contains at least one pixel with a front.
            num_identifiers = 0
            while num_identifiers < self.front_threshold:
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

                num_identifiers = np.count_nonzero(front_ds.sel(longitude=lons, latitude=lats).identifier.values.flatten())
            with open(self.variable_files[index], 'rb') as variable_file:
                variable_ds = pickle.load(variable_file)
            variable_list = list(variable_ds.keys())
            for j in range(self.num_variables):
                var = variable_list[j]
                if self.normalization_method == 1:
                    # Min-max normalization
                    """
                    numpy.nan_to_num: Turns NaN values into 0 and inf numbers into very large numbers
                    """
                    variable_ds[var].values = np.nan_to_num((variable_ds[var].values - norm_params.loc[var,'Min']) /
                                                            (norm_params.loc[var,'Max'] - norm_params.loc[var,'Min']))
                elif self.normalization_method == 2:
                    # Mean normalization
                    """
                    numpy.nan_to_num: Turns NaN values into zeros and inf numbers into very large numbers
                    """
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


class DataGenerator_3D(tf.keras.utils.Sequence):
    """
    Data generator for 3D U-Net models that grabs random files for training and validation.
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
        return int(np.floor(len(self.front_files) / self.batch_size))  # Tells model how many batches to expect

    def __getitem__(self, index):
        inputs, outputs = self.__data_generation()
        return inputs, outputs

    def __data_generation(self):
        front_dss = np.empty(shape=(self.batch_size, self.map_dim_x, self.map_dim_y, 5, self.num_classes))
        variable_dss = np.empty(shape=(self.batch_size, self.map_dim_x, self.map_dim_y, 5, 12))

        norm_params = pd.read_csv('normalization_parameters.csv', index_col='Variable')

        for i in range(self.batch_size):
            # Open random files with random coordinate domains until a sample contains at least one image with a front.
            num_identifiers = 0
            while num_identifiers < self.front_threshold:
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

                num_identifiers = np.count_nonzero(front_ds.sel(longitude=lons, latitude=lats).identifier.values.flatten())
            with open(self.variable_files[index], 'rb') as variable_file:
                variable_ds = pickle.load(variable_file)
            variable_list = list(variable_ds.keys())
            for j in range(self.num_variables):
                var = variable_list[j]
                if self.normalization_method == 1:
                    # Min-max normalization
                    """
                    numpy.nan_to_num: Turns NaN values into 0 and inf numbers into very large numbers
                    """
                    variable_ds[var].values = np.nan_to_num((variable_ds[var].values - norm_params.loc[var,'Min']) /
                                                            (norm_params.loc[var,'Max'] - norm_params.loc[var,'Min']))
                elif self.normalization_method == 2:
                    # Mean normalization
                    """
                    numpy.nan_to_num: Turns NaN values into zeros and inf numbers into very large numbers
                    """
                    variable_ds[var].values = np.nan_to_num((variable_ds[var].values - norm_params.loc[var,'Mean']) /
                                                            (norm_params.loc[var,'Max'] - norm_params.loc[var,'Min']))
            # Split variable dataset into 5 different datasets for the different levels
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
            # Concatenate datasets
            variables_all_levels = np.array([variables_sfc,variables_1000,variables_950,variables_900,variables_850]).reshape(1,self.map_dim_x,self.map_dim_y,5,12)

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
            # Duplicate front array for the 5 different levels in the data
            fronts_all_levels = np.array([binarized_fronts,binarized_fronts,binarized_fronts,binarized_fronts,binarized_fronts]).reshape(1,self.map_dim_x,self.map_dim_y,5,self.num_classes)
            front_dss[i] = fronts_all_levels
            variable_dss[i] = variables_all_levels
        return variable_dss, front_dss


def train_new_unet(front_files, variable_files, map_dim_x, map_dim_y, learning_rate, train_epochs, train_steps,
                   train_batch_size, train_fronts, valid_steps, valid_batch_size, valid_freq, valid_fronts, loss,
                   workers, job_number, model_dir, front_types, normalization_method, fss_mask_size, fss_c,
                   pixel_expansion, num_variables, file_dimensions, validation_years, test_years, num_dimensions, metric):
    """
    Function that train a new U-Net model and saves the model along with its weights.

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
    fss_c: float
        C hyperparameter for the FSS loss' sigmoid function.
    pixel_expansion: int
        Number of pixels to expand the fronts by in all directions.
    num_variables: int
        Number of variables in the datasets.
    file_dimensions: int (x2)
        Dimensions of the data files.
    validation_years: list of ints
        Years for the validation dataset.
    test_years: list of ints
        Years for the test dataset.
    num_dimensions: int
        Number of dimensions for the U-Net's convolutions, maxpooling, and upsampling.
    metric: str
        Metric used for evaluating the U-Net during training.
    """

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
            if num_dimensions == 2:
                loss_function = custom_losses.make_FSS_loss_2D(fss_mask_size, fss_c)
            elif num_dimensions == 3:
                loss_function = custom_losses.make_FSS_loss_3D(fss_mask_size, fss_c)
        
        if metric == 'bss':
            metric_function = custom_losses.brier_skill_score
        elif metric == 'auc':
            metric_function = 'auc'
    
        print('Compiling unet....', end='')
        adam = Adam(learning_rate=learning_rate)
        model.compile(loss=loss_function, optimizer=adam, metrics=metric_function)
        print('done')
    
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
        
        === U-Net (2D) ===
        model = models.unet_2d((map_dim_x, map_dim_y, num_variables), filter_num=[32, 64, 128, 256, 512, 1024], n_labels=num_classes,
            stack_num_down=5, stack_num_up=5, activation='PReLU', output_activation='Softmax', batch_norm=True, pool=True,
            unpool=True, name='unet')
        
        === U-Net++ (2D) ===
        model = models.unet_plus_2d((map_dim_x, map_dim_y, num_variables), filter_num=[32, 64, 128, 256, 512, 1024], n_labels=num_classes,
            stack_num_down=5, stack_num_up=5, activation='PReLU', output_activation='Softmax', batch_norm=True, pool=True,
            unpool=True, name='unet', deep_supervision=True)
        
        === U-Net 3+ (2D) ===
        model = models.unet_3plus_2d((map_dim_x, map_dim_y, num_variables), filter_num_down=[32, 64, 128, 256, 512, 1024],
            filter_num_skip='auto', filter_num_aggregate='auto', n_labels=num_classes, stack_num_down=5, stack_num_up=5,
            activation='PReLU', output_activation='Softmax', batch_norm=True, pool=True, unpool=True, name='unet',
            deep_supervision=True)
        
        === U-Net 3+ (3D) ===
        model = custom_models.UNet_3plus_3d(map_dim_x, map_dim_y, num_classes)
        """
        if num_dimensions == 2:
            print("Creating 2D U-Net....",end='')
            model = models.unet_3plus_2d((map_dim_x, map_dim_y, num_variables), filter_num_down=[64, 128, 256, 512, 1024, 2048],
                filter_num_skip='auto', filter_num_aggregate='auto', n_labels=num_classes, stack_num_down=5, stack_num_up=5,
                activation='PReLU', output_activation='Softmax', batch_norm=True, pool=True, unpool=True, name='U-Net',
                deep_supervision=True)
        elif num_dimensions == 3:
            print("Creating 3D U-Net....",end='')
            model = custom_models.UNet_3plus_3D(map_dim_x, map_dim_y, num_classes)
        print('done')

        if loss == 'dice':
            loss_function = losses.dice
        elif loss == 'tversky':
            loss_function = losses.tversky
        elif loss == 'cce':
            loss_function = 'categorical_crossentropy'
        elif loss == 'fss':
            if num_dimensions == 2:
                loss_function = custom_losses.make_FSS_loss_2D(fss_mask_size, fss_c)
            elif num_dimensions == 3:
                loss_function = custom_losses.make_FSS_loss_3D(fss_mask_size, fss_c)

        if metric == 'bss':
            metric_function = custom_losses.brier_skill_score
        elif metric == 'auc':
            metric_function = 'auc'

        print('Compiling unet....', end='')
        adam = Adam(learning_rate=learning_rate)
        model.compile(loss=loss_function, optimizer=adam, metrics=metric_function)
        print('done')

    model.summary()

    # If validation year and test year are provided, split data into training and validation sets
    if validation_years is not None and test_years is not None:
        front_files_training, front_files_validation, variable_files_training, variable_files_validation = \
            fm.split_file_lists(front_files, variable_files, validation_years, test_years)

        if num_dimensions == 2:
            train_dataset = tf.data.Dataset.from_generator(DataGenerator_2D, args=[front_files_training, variable_files_training,
                map_dim_x, map_dim_y, train_batch_size, train_fronts, front_types, normalization_method, num_classes, pixel_expansion,
                num_variables, file_dimensions],
                output_types=(tf.float16, tf.float16))
    
            validation_dataset = tf.data.Dataset.from_generator(DataGenerator_2D, args=[front_files_validation, variable_files_validation,
                map_dim_x, map_dim_y, valid_batch_size, valid_fronts, front_types, normalization_method, num_classes, pixel_expansion,
                num_variables, file_dimensions],
                output_types=(tf.float16, tf.float16))

        elif num_dimensions == 3:
            train_dataset = tf.data.Dataset.from_generator(DataGenerator_3D, args=[front_files_training, variable_files_training,
                map_dim_x, map_dim_y, train_batch_size, train_fronts, front_types, normalization_method, num_classes, pixel_expansion,
                num_variables, file_dimensions],
                output_types=(tf.float16, tf.float16))

            validation_dataset = tf.data.Dataset.from_generator(DataGenerator_3D, args=[front_files_validation, variable_files_validation,
                map_dim_x, map_dim_y, valid_batch_size, valid_fronts, front_types, normalization_method, num_classes, pixel_expansion,
                num_variables, file_dimensions],
                output_types=(tf.float16, tf.float16))
    else:
        if num_dimensions == 2:
            train_dataset = tf.data.Dataset.from_generator(DataGenerator_2D, args=[front_files, variable_files, map_dim_x,
                map_dim_y, train_batch_size, train_fronts, front_types, normalization_method, num_classes, pixel_expansion,
                num_variables, file_dimensions], output_types=(tf.float16, tf.float16))

            validation_dataset = tf.data.Dataset.from_generator(DataGenerator_2D, args=[front_files, variable_files, map_dim_x,
                map_dim_y, valid_batch_size, valid_fronts, front_types, normalization_method, num_classes, pixel_expansion,
                num_variables, file_dimensions], output_types=(tf.float16, tf.float16))

        elif num_dimensions == 3:
            train_dataset = tf.data.Dataset.from_generator(DataGenerator_3D, args=[front_files, variable_files, map_dim_x,
                map_dim_y, train_batch_size, train_fronts, front_types, normalization_method, num_classes, pixel_expansion,
                num_variables, file_dimensions], output_types=(tf.float16, tf.float16))

            validation_dataset = tf.data.Dataset.from_generator(DataGenerator_3D, args=[front_files, variable_files, map_dim_x,
                map_dim_y, valid_batch_size, valid_fronts, front_types, normalization_method, num_classes, pixel_expansion,
                num_variables, file_dimensions], output_types=(tf.float16, tf.float16))

    os.mkdir('%s/model_%d' % (model_dir, job_number))  # Make folder for model
    os.mkdir('%s/model_%d/predictions' % (model_dir, job_number))  # Make folder for model predictions
    model_filepath = '%s/model_%d/model_%d.h5' % (model_dir, job_number, job_number)
    history_filepath = '%s/model_%d/model_%d_history.csv' % (model_dir, job_number, job_number)

    # ModelCheckpoint: saves model at a specified interval
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, save_freq='epoch')
    # EarlyStopping: stops training early if a metric does not improve after a specified number of epochs (patience)
    early_stopping = EarlyStopping('val_loss', patience=150, verbose=1)
    history_logger = CSVLogger(history_filepath, separator=",", append=True)  # Saves loss/AUC data every epoch

    model.fit(train_dataset.repeat(), validation_data=validation_dataset.repeat(), validation_freq=valid_freq,
        epochs=train_epochs, steps_per_epoch=train_steps, validation_steps=valid_steps, callbacks=[early_stopping,
        checkpoint, history_logger], verbose=2, workers=workers, use_multiprocessing=True, max_queue_size=10000)


def train_imported_unet(front_files, variable_files, learning_rate, train_epochs, train_steps, train_batch_size,
    train_fronts, valid_steps, valid_batch_size, valid_freq, valid_fronts, loss, workers, model_number, model_dir,
    front_types, normalization_method, fss_mask_size, fss_c, pixel_expansion, num_variables, file_dimensions, validation_years,
    test_years, num_dimensions, metric):
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
    fss_c: float
        C hyperparameter for the FSS loss' sigmoid function.
    pixel_expansion: int
        Number of pixels to expand the fronts by in all directions.
    num_variables: int
        Number of variables in the datasets.
    file_dimensions: int (x2)
        Dimensions of the data files.
    validation_years: int
        Years for the validation dataset.
    test_years: int
        Years for the test dataset.
    num_dimensions: int
        Number of dimensions for the U-Net's convolutions, maxpooling, and upsampling.
    metric: str
        Metric used for evaluating the U-Net during training.
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
    
        print("Importing U-Net....",end='')
        model = fm.load_model(model_number, model_dir, loss, fss_mask_size, fss_c, metric, num_dimensions)
        print("done")

        if loss == 'dice':
            loss_function = losses.dice
        elif loss == 'tversky':
            loss_function = losses.tversky
        elif loss == 'cce':
            loss_function = 'categorical_crossentropy'
        elif loss == 'fss':
            if num_dimensions == 2:
                loss_function = custom_losses.make_FSS_loss_2D(fss_mask_size, fss_c)
            elif num_dimensions == 3:
                loss_function = custom_losses.make_FSS_loss_3D(fss_mask_size, fss_c)
        
        if metric == 'bss':
            metric_function = custom_losses.brier_skill_score
        elif metric == 'auc':
            metric_function = 'auc'
    
        print('Compiling unet....', end='')
        adam = Adam(learning_rate=learning_rate)
        model.compile(loss=loss_function, optimizer=adam, metrics=metric_function)
        print('done')
    
    model.summary()
    ....
    ....
    model.fit(...)
    """

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    with strategy.scope():

        model = fm.load_model(model_number, model_dir, loss, fss_mask_size, fss_c, metric, num_dimensions)

        if loss == 'dice':
            loss_function = losses.dice
        elif loss == 'tversky':
            loss_function = losses.tversky
        elif loss == 'cce':
            loss_function = 'categorical_crossentropy'
        elif loss == 'fss':
            if num_dimensions == 2:
                loss_function = custom_losses.make_FSS_loss_2D(fss_mask_size, fss_c)
            elif num_dimensions == 3:
                loss_function = custom_losses.make_FSS_loss_3D(fss_mask_size, fss_c)

        if metric == 'bss':
            metric_function = custom_losses.brier_skill_score
        elif metric == 'auc':
            metric_function = 'auc'

        print('Compiling unet....', end='')
        adam = Adam(learning_rate=learning_rate)
        model.compile(loss=loss_function, optimizer=adam, metrics=metric_function)
        print('done')

    model.summary()

    map_dim_x = model.layers[0].input_shape[0][1]  # Longitudinal dimension of the U-Net
    map_dim_y = model.layers[0].input_shape[0][2]  # Latitudinal dimension of the U-Net

    # If validation years and test years are provided, split data into training and validation sets
    if validation_years is not None and test_years is not None:
        front_files_training, front_files_validation, variable_files_training, variable_files_validation = \
            fm.split_file_lists(front_files, variable_files, validation_years, test_years)

        if num_dimensions == 2:
            train_dataset = tf.data.Dataset.from_generator(DataGenerator_2D, args=[front_files_training, variable_files_training,
                map_dim_x, map_dim_y, train_batch_size, train_fronts, front_types, normalization_method, num_classes, pixel_expansion,
                num_variables, file_dimensions],
                output_types=(tf.float16, tf.float16))

            validation_dataset = tf.data.Dataset.from_generator(DataGenerator_2D, args=[front_files_validation, variable_files_validation,
                map_dim_x, map_dim_y, valid_batch_size, valid_fronts, front_types, normalization_method, num_classes, pixel_expansion,
                num_variables, file_dimensions],
                output_types=(tf.float16, tf.float16))

        elif num_dimensions == 3:
            train_dataset = tf.data.Dataset.from_generator(DataGenerator_3D, args=[front_files_training, variable_files_training,
                map_dim_x, map_dim_y, train_batch_size, train_fronts, front_types, normalization_method, num_classes, pixel_expansion,
                num_variables, file_dimensions],
                output_types=(tf.float16, tf.float16))

            validation_dataset = tf.data.Dataset.from_generator(DataGenerator_3D, args=[front_files_validation, variable_files_validation,
                map_dim_x, map_dim_y, valid_batch_size, valid_fronts, front_types, normalization_method, num_classes, pixel_expansion,
                num_variables, file_dimensions],
                output_types=(tf.float16, tf.float16))
    else:
        if num_dimensions == 2:
            train_dataset = tf.data.Dataset.from_generator(DataGenerator_2D, args=[front_files, variable_files, map_dim_x,
                map_dim_y, train_batch_size, train_fronts, front_types, normalization_method, num_classes, pixel_expansion,
                num_variables, file_dimensions], output_types=(tf.float16, tf.float16))

            validation_dataset = tf.data.Dataset.from_generator(DataGenerator_2D, args=[front_files, variable_files, map_dim_x,
                map_dim_y, valid_batch_size, valid_fronts, front_types, normalization_method, num_classes, pixel_expansion,
                num_variables, file_dimensions], output_types=(tf.float16, tf.float16))

        elif num_dimensions == 3:
            train_dataset = tf.data.Dataset.from_generator(DataGenerator_3D, args=[front_files, variable_files, map_dim_x,
                map_dim_y, train_batch_size, train_fronts, front_types, normalization_method, num_classes, pixel_expansion,
                num_variables, file_dimensions], output_types=(tf.float16, tf.float16))

            validation_dataset = tf.data.Dataset.from_generator(DataGenerator_3D, args=[front_files, variable_files, map_dim_x,
                map_dim_y, valid_batch_size, valid_fronts, front_types, normalization_method, num_classes, pixel_expansion,
                num_variables, file_dimensions], output_types=(tf.float16, tf.float16))

    model_filepath = '%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number)
    history_filepath = '%s/model_%d/model_%d_history.csv' % (model_dir, model_number, model_number)

    # ModelCheckpoint: saves model at a specified interval
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, save_freq='epoch')
    # EarlyStopping: stops training early if a metric does not improve after a specified number of epochs (patience)
    early_stopping = EarlyStopping('val_loss', patience=150, verbose=1)
    history_logger = CSVLogger(history_filepath, separator=",", append=True)  # Saves loss/AUC data every epoch

    model.fit(train_dataset.repeat(), validation_data=validation_dataset.repeat(), validation_freq=valid_freq,
        epochs=train_epochs, steps_per_epoch=train_steps, validation_steps=valid_steps, callbacks=[early_stopping,
        checkpoint, history_logger], verbose=2, workers=workers, use_multiprocessing=True, max_queue_size=10000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--domain', type=str, required=False, help='Domain of the data.')
    parser.add_argument('--epochs', type=int, required=False, help='Number of epochs for the U-Net training.')
    parser.add_argument('--file_dimensions', type=int, nargs=2, required=False,
                        help='Dimensions of the file size. Two integers need to be passed.')
    parser.add_argument('--front_types', type=str, required=False, 
                        help='Front format of the file. If your files contain warm and cold fronts, pass this argument '
                             'as CFWF. If your files contain only drylines, pass this argument as DL. If your files '
                             'contain all fronts, pass this argument as ALL.')
    parser.add_argument('--fss_c', type=float, required=False, help='C hyperparameter for the FSS loss sigmoid function.')
    parser.add_argument('--fss_mask_size', type=int, required=False, help='Mask size for the FSS loss function.')
    parser.add_argument('--job_number', type=int, required=False, help='Slurm job number.')
    parser.add_argument('--import_model_number', type=int, required=False,
                        help='Number of the model that you would like to import.')
    parser.add_argument('--learning_rate', type=float, required=False, help='Learning rate for U-Net optimizer.')
    parser.add_argument('--loss', type=str, required=False, help='Loss function for the U-Net')
    parser.add_argument('--map_dim_x_y', type=int, required=False, nargs=2, help='Dimensions of the U-Net map.')
    parser.add_argument('--metric', type=str, required=False, help='Metric for evaluating the U-Net during training.')
    parser.add_argument('--model_dir', type=str, required=False, help='Directory where the models are or will be saved to.')
    parser.add_argument('--normalization_method', type=int, required=False,
                        help='Normalization method for the data. 0 - No normalization, 1 - Min-max normalization, '
                             '2 - Mean normalization')
    parser.add_argument('--num_dimensions', type=int, required=False,
                        help='Number of dimensions of the U-Net convolutions, maxpooling, and upsampling. (2 or 3)')
    parser.add_argument('--num_variables', type=int, required=False, help='Number of variables in the variable datasets.')
    parser.add_argument('--pixel_expansion', type=int, required=False, help='Number of pixels to expand the fronts by.')
    parser.add_argument('--test_years', type=int, nargs="+", required=False, help='Years for the test set.')
    parser.add_argument('--train_valid_batch_size', type=int, required=False, nargs=2, help='Batch sizes for the U-Net.')
    parser.add_argument('--train_valid_fronts', type=int, required=False, nargs=2,
                        help='How many pixels with fronts an image must have for it to be passed through the generator.')
    parser.add_argument('--train_valid_steps', type=int, required=False, nargs=2, help='Number of steps for each epoch.')
    parser.add_argument('--valid_freq', type=int, required=False, help='How many epochs to pass before each validation.')
    parser.add_argument('--validation_years', type=int, nargs="+", required=False, help='Years for the validation set.')
    parser.add_argument('--workers', type=int, required=False, help='Number of workers for training the U-Net.')

    args = parser.parse_args()
    provided_arguments = vars(args)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    if args.import_model_number is not None and args.job_number is not None:
        raise errors.ArgumentConflictError("import_model_number and job_number cannot be passed at the same time.")
    if args.import_model_number is None and args.job_number is None:
        raise errors.MissingArgumentError("one of the following arguments must be provided: import_model_number, job_number")

    if args.loss == 'fss':
        required_arguments = ['fss_c', 'fss_mask_size']
        print("Checking arguments for FSS loss function....", end='')
        check_arguments(provided_arguments, required_arguments)

    if args.import_model_number is not None:
        if args.map_dim_x_y is not None:
            raise errors.ExtraArgumentError('import_model_number is not None but the following argument was passed: map_dim_x_y')
        required_arguments = ['domain', 'epochs', 'file_dimensions', 'front_types', 'learning_rate', 'loss', 'metric',
                              'model_dir', 'normalization_method', 'num_dimensions', 'num_variables', 'pixel_expansion',
                              'train_valid_batch_size', 'train_valid_fronts', 'train_valid_steps', 'valid_freq', 'workers']
        print("Checking arguments for 'train_imported_unet'....", end='')
        check_arguments(provided_arguments, required_arguments)
        print("WARNING: You have imported model %d for training." % args.import_model_number)
        front_files, variable_files = fm.load_file_lists(args.num_variables, args.front_types, args.domain, args.file_dimensions)
        train_imported_unet(front_files, variable_files, args.learning_rate, args.epochs, args.train_valid_steps[0], args.train_valid_batch_size[0],
            args.train_valid_fronts[0], args.train_valid_steps[1], args.train_valid_batch_size[1], args.valid_freq, args.train_valid_fronts[1],
            args.loss, args.workers, args.import_model_number, args.model_dir, args.front_types, args.normalization_method, args.fss_mask_size,
            args.fss_c, args.pixel_expansion, args.num_variables, args.file_dimensions, args.validation_years, args.test_years, args.num_dimensions,
            args.metric)

    if args.job_number is not None:
        required_arguments = ['domain', 'epochs', 'file_dimensions', 'front_types', 'learning_rate', 'loss', 'map_dim_x_y',
                              'metric', 'model_dir', 'normalization_method', 'num_dimensions', 'num_variables', 'pixel_expansion',
                              'train_valid_batch_size', 'train_valid_fronts', 'train_valid_steps', 'valid_freq', 'workers']
        print("Checking arguments for 'train_new_unet'....", end='')
        check_arguments(provided_arguments, required_arguments)
        front_files, variable_files = fm.load_file_lists(args.num_variables, args.front_types, args.domain, args.file_dimensions)
        train_new_unet(front_files, variable_files, args.map_dim_x_y[0], args.map_dim_x_y[1], args.learning_rate, args.epochs,
            args.train_valid_steps[0], args.train_valid_batch_size[0], args.train_valid_fronts[0], args.train_valid_steps[1], args.train_valid_batch_size[1],
            args.valid_freq, args.train_valid_fronts[1], args.loss, args.workers, args.job_number, args.model_dir, args.front_types,
            args.normalization_method, args.fss_mask_size, args.fss_c, args.pixel_expansion, args.num_variables, args.file_dimensions,
            args.validation_years, args.test_years, args.num_dimensions, args.metric)
