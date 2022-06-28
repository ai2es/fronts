"""
Function that trains a new or imported U-Net model.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 5/10/2022 6:39 PM CDT
"""

import random
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import pickle
import numpy as np
import file_manager as fm
import os
from errors import check_arguments
import errors
import custom_losses
import models
from utils.data_utils import expand_fronts, reformat_fronts
from variables import normalize


class DataGenerator_2D(tf.keras.utils.Sequence):
    """
    Data generator for 2D U-Net models that grabs random files for training and validation.
    """
    def __init__(self, front_files, variable_files, image_size_x, image_size_y, batch_size, front_threshold, front_types,
                 num_classes, pixel_expansion, num_variables):
        self.front_files = front_files
        self.variable_files = variable_files
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        self.batch_size = batch_size
        self.front_threshold = front_threshold
        self.front_types = front_types  # NOTE: This is passed into the generator as a bytes object, NOT a string
        self.num_classes = num_classes
        self.pixel_expansion = pixel_expansion
        self.num_variables = num_variables

    def __len__(self):
        return int(np.floor(len(self.front_files) / self.batch_size))  # Tells model how many batches to expect

    def __getitem__(self, index):
        inputs, outputs = self.__data_generation()
        return inputs, outputs

    def __data_generation(self):

        # Create empty arrays that will contain the data for the final batch that will be sent to the GPU
        front_dss = np.empty(shape=(self.batch_size, self.image_size_x, self.image_size_y, self.num_classes))
        variable_dss = np.empty(shape=(self.batch_size, self.image_size_x, self.image_size_y, self.num_variables))

        for i in range(self.batch_size):
            # Open random files with random coordinate domains until a domain contains a specified number of pixels with fronts.
            num_identifiers = 0  # Initialize variable for the number of pixels in the images
            while num_identifiers < self.front_threshold:  # Continue opening files until the image contains at least front_threshold pixels
                index = random.choices(range(len(self.front_files) - 1), k=1)[0]  # Select random file
                with open(self.front_files[index], 'rb') as front_file:
                    front_ds = pickle.load(front_file)
                    for pixel in range(self.pixel_expansion):  # Expand fronts by a specified number of pixels in each direction
                        front_ds = expand_fronts(front_ds)
                domain_dim_lon = len(front_ds.longitude.values)
                domain_dim_lat = len(front_ds.latitude.values)
                lon_index = random.choices(range(domain_dim_lon - self.image_size_x))[0]  # Select a random part of the longitude domain
                lat_index = random.choices(range(domain_dim_lat - self.image_size_y))[0]  # Select a random part of the latitude domain
                lons = front_ds.longitude.values[lon_index:lon_index + self.image_size_x]  # Select longitude points in front dataset
                lats = front_ds.latitude.values[lat_index:lat_index + self.image_size_y]  # Select latitude points in front dataset
                num_identifiers = np.count_nonzero(front_ds.sel(longitude=lons, latitude=lats).identifier.values.flatten())  # Number of pixels in the new front dataset with fronts

            # Open corresponding variable file
            with open(self.variable_files[index], 'rb') as variable_file:
                variable_ds = pickle.load(variable_file)

            variable_ds = normalize(variable_ds)  # Normalize variables

            fronts = reformat_fronts(front_ds.sel(longitude=lons, latitude=lats), self.front_types.encode('utf-8')).to_array().T.values

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
    def __init__(self, front_files, variable_files, image_size_x, image_size_y, batch_size, front_threshold, front_types,
                 num_classes, pixel_expansion, num_variables):
        self.front_files = front_files
        self.variable_files = variable_files
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        self.batch_size = batch_size
        self.front_threshold = front_threshold
        self.front_types = front_types  # NOTE: This is passed into the generator as a bytes object, NOT a string
        self.num_classes = num_classes
        self.pixel_expansion = pixel_expansion
        self.num_variables = num_variables

    def __len__(self):
        return int(np.floor(len(self.front_files) / self.batch_size))  # Tells model how many batches to expect

    def __getitem__(self, index):
        inputs, outputs = self.__data_generation()
        return inputs, outputs

    def __data_generation(self):

        # Create empty arrays that will contain the data for the final batch that will be sent to the GPU
        front_dss = np.empty(shape=(self.batch_size, self.image_size_x, self.image_size_y, 5, self.num_classes))
        variable_dss = np.empty(shape=(self.batch_size, self.image_size_x, self.image_size_y, 5, 12))

        for i in range(self.batch_size):
            index = random.choices(range(len(self.front_files) - 1), k=1)[0]  # Select random file
            with open(self.front_files[index], 'rb') as front_file:
                front_ds = pickle.load(front_file)
                for pixel in range(self.pixel_expansion):  # Expand fronts by a specified number of pixels in each direction
                    front_ds = expand_fronts(front_ds)
                    front_ds_backup = front_ds
            # Open random files with random coordinate domains until a domain contains a specified number of pixels with fronts.
            front_ds = front_ds_backup
            num_identifiers = 0  # Initialize variable for the number of pixels in the images
            while num_identifiers < self.front_threshold:  # Continue opening files until the image contains at least front_threshold pixels
                domain_dim_lon = len(front_ds.longitude.values)
                domain_dim_lat = len(front_ds.latitude.values)
                lon_index = random.choices(range(domain_dim_lon - self.image_size_x))[0]  # Select a random part of the longitude domain
                if domain_dim_lat == self.image_size_y:
                    lat_index = 0
                else:
                    lat_index = random.choices(range(domain_dim_lat - self.image_size_y))[0]  # Select a random part of the latitude domain
                lons = front_ds.longitude.values[lon_index:lon_index + self.image_size_x]  # Select longitude points in front dataset
                lats = front_ds.latitude.values[lat_index:lat_index + self.image_size_y]  # Select latitude points in front dataset
                num_identifiers = np.count_nonzero(front_ds.sel(longitude=lons, latitude=lats).identifier.values.flatten())  # Number of pixels in the new front dataset with fronts

            # Open corresponding variable file
            with open(self.variable_files[index], 'rb') as variable_file:
                variable_ds = pickle.load(variable_file)

            variable_ds = normalize(variable_ds)  # Normalize variables
            
            # Split variable dataset into 5 different datasets for the different levels
            variables_sfc = variable_ds[['t2m', 'd2m', 'sp', 'u10', 'v10', 'theta_w', 'mix_ratio', 'rel_humid', 'virt_temp', 
                'wet_bulb', 'theta_e', 'q']].sel(longitude=lons, latitude=lats).to_array().values
            variables_1000 = variable_ds[['t_1000', 'd_1000', 'z_1000', 'u_1000', 'v_1000', 'theta_w_1000', 'mix_ratio_1000',
                'rel_humid_1000', 'virt_temp_1000', 'wet_bulb_1000', 'theta_e_1000', 'q_1000']].sel(longitude=lons, latitude=lats).to_array().values
            variables_950 = variable_ds[['t_950', 'd_950', 'z_950', 'u_950', 'v_950', 'theta_w_950', 'mix_ratio_950', 'rel_humid_950', 
                'virt_temp_950', 'wet_bulb_950', 'theta_e_950', 'q_950']].sel(longitude=lons, latitude=lats).to_array().values
            variables_900 = variable_ds[['t_900', 'd_900', 'z_900', 'u_900', 'v_900', 'theta_w_900', 'mix_ratio_900', 'rel_humid_900', 
                'virt_temp_900', 'wet_bulb_900', 'theta_e_900', 'q_900']].sel(longitude=lons, latitude=lats).to_array().values
            variables_850 = variable_ds[['t_850', 'd_850', 'z_850', 'u_850', 'v_850', 'theta_w_850', 'mix_ratio_850', 'rel_humid_850', 
                'virt_temp_850', 'wet_bulb_850', 'theta_e_850', 'q_850']].sel(longitude=lons, latitude=lats).to_array().values
            # Concatenate datasets
            variables_all_levels = np.expand_dims(np.array([variables_sfc, variables_1000, variables_950, variables_900, variables_850]).transpose([3, 2, 0, 1]), axis=0)

            fronts = reformat_fronts(front_ds.identifier.sel(longitude=lons, latitude=lats), self.front_types.decode('utf-8'))  # Turn dataset into array

            # Duplicate front array for the 5 different levels in the data
            fronts_all_levels = np.expand_dims(np.array([fronts, fronts, fronts, fronts, fronts]).transpose([2, 1, 0]), axis=0)

            """
            tensorflow.keras.utils.to_categorical: binarizes identifier (front label) values
            Example: [0, 2, 3, 4] ---> [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]] 
            """
            binarized_fronts = to_categorical(fronts_all_levels, num_classes=self.num_classes)

            front_dss[i] = binarized_fronts
            variable_dss[i] = variables_all_levels
        return variable_dss, front_dss


def train_new_unet(input_size, kernel_size, pool_size, upsample_size, levels, filter_num, epochs, loss, metric, fss_mask_size,
    fss_c, learning_rate, train_valid_domain, train_valid_batch_size, train_valid_steps, train_valid_fronts, valid_freq,
    front_types, pixel_expansion, test_years, validation_years, model_number, model_dir, model_type, pickle_dir):
    """
    Function that train a new U-Net model and saves the model along with its weights.

    Parameters
    ----------
    input_size: tuple
        - Input size for the U-Net. The tuple must contain a dimension for channels.
    kernel_size: int or tuple
        - Kernel size for the convolutions in the U-Net.
    pool_size: int or tuple
        - Pool size in the MaxPooling layers.
    upsample_size: int or tuple
        - Upsampling size in the UpSampling layers.
    levels: int
        - Number of levels in the U-Net.
    filter_num: iterable
        - Number of filters in each level of the U-Net. The length of this iterable must be equal to the 'levels'.
    epochs: int
        - Number of epochs for model training.
    loss: str
        - Loss function for the U-Net.
    metric: str
        - Metric to use when training the U-Net.
    fss_mask_size: int or tuple
        - Pool size in the AveragePooling layers of the FSS loss function.
    fss_c: float
        - C parameter in the sigmoid function inside of the FSS loss function.
    learning_rate: float
        - Learning rate for the U-Net.
    train_valid_domain: iterable of 2 strs
        - Domains over which the U-Net will be trained in evaluated. Two strings need to be passed.
    train_valid_batch_size: iterable of 2 ints
        - Batch sizes for model training and validation. Two integers need to be passed.
    train_valid_steps: iterable of 2 ints
        - Number of steps per epoch for model training and validation. Two integers need to be passed.
    train_valid_fronts: iterable of 2 ints
        - Minimum number of pixels that need to be present in an image for it to be used in training or validation. Two integers need to be passed.
    valid_freq: int
        - Number of epochs that the model will train over before performing validation.
    front_types: list
        - List of codes (or a single code) that tells the generator what front types that the model will be trained on.
    pixel_expansion: int
        - Number of pixels to expand the fronts by in all directions. 1 pixel ~ 25 km
    test_years: iterable
        - Years of datasets to leave out of training and validation.
    validation_years: iterable
        - Years of datasets to use in validation.
    model_number: int
        - Number that the model will be assigned when it is saved.
    model_dir: str
        - Directory where the model will be saved to.
    model_type: str
        - Type of model to train.
    pickle_dir: str
        - Directory of the pickle files that will be used to train and validate the model.
    """

    if front_types == 'MERGED-F-BIN' or front_types == 'MERGED-T':
        num_classes = 2
    elif front_types == 'MERGED-F':
        num_classes = 5
    elif front_types == 'MERGED-ALL':
        num_classes = 8
    else:
        num_classes = len(front_types) + 1

    if len(input_size) == 3:  # If the image size is 2D
        num_dimensions = 2
        num_variables = input_size[-1]
    elif len(input_size) == 4:  # If the image size is 3D
        num_dimensions = 3
        num_variables = input_size[2] * input_size[3]  # Number of levels * predictors per level
    else:
        raise ValueError(f"Invalid input size: {np.shape(input_size)}. The input tensor can only have 3 or 4 dimensions")

    if test_years is not None and validation_years is not None:
        front_files_training, variable_files_training = fm.load_files(pickle_dir, num_variables, train_valid_domain[0], dataset='training', test_years=test_years, validation_years=validation_years)
        front_files_validation, variable_files_validation = fm.load_files(pickle_dir, num_variables, train_valid_domain[1], dataset='validation', validation_years=validation_years)
    else:
        front_files_training, variable_files_training = fm.load_files(pickle_dir, num_variables, train_valid_domain[0])
        front_files_validation, variable_files_validation = fm.load_files(pickle_dir, num_variables, train_valid_domain[1])

    if model_type == 'unet':
        model = models.unet(input_size, num_classes, pool_size, upsample_size, levels, filter_num)
    elif model_type == 'unet_3plus':
        model = models.unet_3plus(input_size, num_classes, pool_size, upsample_size, levels, filter_num)
    else:
        raise ValueError(f"Invalid model type: {model_type}. Available options are: 'unet', 'unet_3plus'")
    model.summary()

    # Choose loss function
    if loss == 'cce':
        loss_function = 'categorical_crossentropy'
    elif loss == 'fss':
        loss_function = custom_losses.make_fractions_skill_score(fss_mask_size, c=fss_c)
    else:
        loss_function = loss

    # Choose metric
    if metric == 'bss':
        metric_function = custom_losses.brier_skill_score
    else:
        metric_function = metric

    adam = Adam(learning_rate=learning_rate)  # Adam optimizer for the U-Net
    model.compile(loss=loss_function, optimizer=adam, metrics=metric_function)  # Compile the model

    if num_dimensions == 2:
        train_dataset = tf.data.Dataset.from_generator(DataGenerator_2D, args=[front_files_training, variable_files_training,
            input_size[0], input_size[1], train_valid_batch_size[0], train_valid_fronts[0], front_types, num_classes, pixel_expansion, num_variables],
            output_types=(tf.float16, tf.float16))
        validation_dataset = tf.data.Dataset.from_generator(DataGenerator_2D, args=[front_files_validation, variable_files_validation,
            input_size[0], input_size[1], train_valid_batch_size[1], train_valid_fronts[1], front_types, num_classes, pixel_expansion, num_variables],
            output_types=(tf.float16, tf.float16))
    else:
        train_dataset = tf.data.Dataset.from_generator(DataGenerator_3D, args=[front_files_training, variable_files_training,
            input_size[0], input_size[1], train_valid_batch_size[0], train_valid_fronts[0], front_types, num_classes, pixel_expansion, num_variables],
            output_types=(tf.float16, tf.float16))
        validation_dataset = tf.data.Dataset.from_generator(DataGenerator_3D, args=[front_files_validation, variable_files_validation,
            input_size[0], input_size[1], train_valid_batch_size[1], train_valid_fronts[1], front_types, num_classes, pixel_expansion, num_variables],
            output_types=(tf.float16, tf.float16))

    training_years = np.arange(2008, 2021, 1)  # All years of data

    # Remove the validation and test years from the training years array in order to save the training years to the dictionary below
    if validation_years is not None:
        for validation_year in validation_years:
            training_years = training_years[np.where(training_years != validation_year)]
    if test_years is not None:
        for test_year in test_years:
            training_years = training_years[np.where(training_years != test_year)]

    # Create dictionary containing information about the model. This simplifies the process of loading the model
    model_properties = dict({})
    model_properties['model_number'] = model_number
    model_properties['training_years'] = training_years
    model_properties['validation_years'] = validation_years
    model_properties['test_years'] = test_years
    model_properties['domains'] = train_valid_domain
    model_properties['batch_sizes'] = train_valid_batch_size
    model_properties['steps_per_epoch'] = train_valid_steps
    model_properties['valid_freq'] = 1
    model_properties['optimizer'] = 'Adam'
    model_properties['learning_rate'] = learning_rate
    model_properties['input_size'] = input_size
    model_properties['kernel_size'] = kernel_size
    model_properties['num_variables'] = num_variables
    model_properties['front_threshold'] = train_valid_fronts
    model_properties['pixel_expansion'] = pixel_expansion
    model_properties['normalization'] = 'min-max'
    model_properties['loss_function'] = loss
    model_properties['fss_mask_c'] = (fss_mask_size, fss_c)
    model_properties['metric'] = metric
    model_properties['front_types'] = front_types
    model_properties['classes'] = num_classes

    os.mkdir('%s/model_%d' % (model_dir, model_number))  # Make folder for model
    os.mkdir('%s/model_%d/maps' % (model_dir, model_number))  # Make folder for model predicton maps
    os.mkdir('%s/model_%d/probabilities' % (model_dir, model_number))  # Make folder for prediction data files
    os.mkdir('%s/model_%d/statistics' % (model_dir, model_number))  # Make folder for statistics data files
    model_filepath = '%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number)
    history_filepath = '%s/model_%d/model_%d_history.csv' % (model_dir, model_number, model_number)
    with open('%s/model_%d/model_%d_properties.pkl' % (model_dir, model_number, model_number), 'wb') as f:
        pickle.dump(model_properties, f)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, save_freq='epoch')  # ModelCheckpoint: saves model at a specified interval
    early_stopping = EarlyStopping('val_loss', patience=250, verbose=1)  # EarlyStopping: stops training early if a metric does not improve after a specified number of epochs (patience)
    history_logger = CSVLogger(history_filepath, separator=",", append=True)  # Saves loss/AUC data every epoch

    model.fit(train_dataset.repeat(), validation_data=validation_dataset.repeat(), validation_freq=valid_freq,
        epochs=epochs, steps_per_epoch=train_valid_steps[0], validation_steps=train_valid_steps[1], callbacks=[early_stopping,
        checkpoint, history_logger], verbose=1, workers=10, use_multiprocessing=False, max_queue_size=1000)


def train_imported_unet(train_domain, valid_domain, learning_rate, train_epochs, train_steps, train_batch_size,
    train_fronts, valid_steps, valid_batch_size, valid_freq, valid_fronts, loss, model_number, model_dir, front_types, 
    fss_mask_size, fss_c, pixel_expansion, num_variables, validation_years, test_years, num_dimensions, metric):
    """
    Function that train a new U-Net model and saves the model along with its weights.

    Parameters
    ----------
    front_types: str
        Code that identifies the types of fronts that are being predicted by the U-Net.
    fss_c: float
        C hyperparameter for the FSS loss' sigmoid function.
    fss_mask_size: int
        Size of the mask for the FSS loss function.
    learning_rate: float
        Value that determines how fast the optimization algorithm overrides old information (how fast the U-Net learns).
    loss: str
        Loss function for the U-Net.
    metric: str
        Metric used for evaluating the U-Net during training.
    model_dir: str
        Directory that the models are saved to.
    model_number: int
        Model number.
    num_dimensions: int
        Number of dimensions for the U-Net's convolutions, maxpooling, and upsampling.
    num_variables: int
        Number of variables in the datasets.
    pixel_expansion: int
        Number of pixels to expand the fronts by in all directions.
    test_years: iterable of ints
        Years for the test dataset.
    train_batch_size: int
        Number of images to process for each step of each epoch in the training generator.
    train_domain: str
        Domain of the training data.
    train_epochs: int
        Number of times that the U-Net will cycle over the data, or in this case, the number of times that the model will
        run the training generator.
    train_fronts: int
        Minimum number of pixels containing fronts that must be present in an image for it to be used for training the
        U-net.
    train_steps: int
        Number of steps per epoch for the training generator. This is the number of times that a batch will be generated
        before the generator moves onto the next epoch.
    valid_batch_size: int
        Number of images/datasets to process for each step of each epoch in the validation generator.
    valid_domain: str
        Domain of the validation data.
    valid_freq: int
        This integer represents how often the model will be validated, or having its hyperparameters automatically
        tuned. For example, setting this to 4 means that the model will be validated every 4 epochs.
    valid_fronts: int
        Minimum number of pixels containing fronts that must be present in an image for it to be used in U-Net
        validation.
    valid_steps: int
        Number of steps for each epoch in the validation generator.
    validation_years: iterable of ints
        Years for the validation dataset.
    """

    if test_years is not None and validation_years is not None:
        front_files_training, variable_files_training = fm.load_files(train_domain, num_variables, dataset='training', test_years=test_years, validation_years=validation_years)
        front_files_validation, variable_files_validation = fm.load_files(valid_domain, num_variables, dataset='validation', validation_years=validation_years)
    else:
        front_files_training, variable_files_training = fm.load_files(train_domain, num_variables)
        front_files_validation, variable_files_validation = fm.load_files(valid_domain, num_variables)

    if front_types == 'CFWF' or front_types == 'SFOF':
        num_classes = 3
    elif front_types == 'ALL_bin':
        num_classes = 2
    else:
        num_classes = 5

    model = fm.load_model(model_number, model_dir)

    # Choose loss function
    if loss == 'fss':
        loss_function = custom_losses.make_fractions_skill_score(fss_mask_size, c=fss_c)
    else:
        loss_function = loss

    # Choose metric
    if metric == 'bss':
        metric_function = custom_losses.brier_skill_score
    else:
        metric_function = metric

    adam = Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=adam, metrics=metric_function)

    image_size_x = model.layers[0].input_shape[0][1]  # Longitudinal dimension of the U-Net
    image_size_y = model.layers[0].input_shape[0][2]  # Latitudinal dimension of the U-Net

    if num_dimensions == 2:
        train_dataset = tf.data.Dataset.from_generator(DataGenerator_2D, args=[front_files_training, variable_files_training,
            image_size_x, image_size_y, train_batch_size, train_fronts, front_types, num_classes, pixel_expansion, num_variables],
            output_types=(tf.float16, tf.float16))
        validation_dataset = tf.data.Dataset.from_generator(DataGenerator_2D, args=[front_files_validation, variable_files_validation,
            image_size_x, image_size_y, valid_batch_size, valid_fronts, front_types, num_classes, pixel_expansion, num_variables],
            output_types=(tf.float16, tf.float16))
    else:
        train_dataset = tf.data.Dataset.from_generator(DataGenerator_3D, args=[front_files_training, variable_files_training,
            image_size_x, image_size_y, train_batch_size, train_fronts, front_types, num_classes, pixel_expansion, num_variables],
            output_types=(tf.float16, tf.float16))
        validation_dataset = tf.data.Dataset.from_generator(DataGenerator_3D, args=[front_files_validation, variable_files_validation,
            image_size_x, image_size_y, valid_batch_size, valid_fronts, front_types, num_classes, pixel_expansion, num_variables],
            output_types=(tf.float16, tf.float16))

    model_filepath = '%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number)
    history_filepath = '%s/model_%d/model_%d_history.csv' % (model_dir, model_number, model_number)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, save_freq='epoch')  # ModelCheckpoint: saves model at a specified interval
    early_stopping = EarlyStopping('val_loss', patience=100, verbose=1, restore_best_weights=True)  # EarlyStopping: stops training early if a metric does not improve after a specified number of epochs (patience)
    history_logger = CSVLogger(history_filepath, separator=",", append=True)  # Saves loss/AUC data every epoch

    model.fit(train_dataset.repeat(), validation_data=validation_dataset.repeat(), validation_freq=valid_freq,
        epochs=train_epochs, steps_per_epoch=train_steps, validation_steps=valid_steps, callbacks=[early_stopping,
        checkpoint, history_logger], verbose=2, workers=8, use_multiprocessing=True, max_queue_size=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    """ Training parameters """
    parser.add_argument('--epochs', type=int, default=6000, help='Number of epochs for the U-Net training.')
    parser.add_argument('--filter_num', type=int, nargs='+', help='Number of filters on each level of the U-Net.')
    parser.add_argument('--front_types', type=str, nargs='+', required=False, help='Front types that the model will be trained on.')
    """
    Available options for individual front types (cannot be passed with any special codes):

    Code (class #): Front Type
    --------------------------
    CF (1): Cold front
    WF (2): Warm front
    SF (3): Stationary front
    OF (4): Occluded front
    CF-F (5): Cold front (forming)
    WF-F (6): Warm front (forming)
    SF-F (7): Stationary front (forming)
    OF-F (8): Occluded front (forming)
    CF-D (9): Cold front (dissipating)
    WF-D (10): Warm front (dissipating)
    SF-D (11): Stationary front (dissipating)
    OF-D (12): Occluded front (dissipating)
    INST (13): Instability axis
    TROF (14): Trough
    TT (15): Tropical Trough
    DL (16): Dryline
    
    
    Special codes (cannot be passed with any individual front codes):  
    -----------------------------------------------------------------
    MERGED-F (4 classes): Train on 1-12, but treat forming and dissipating fronts as a standard front. In other words, the following changes are made to the classes:
        CF-F (5), CF-D (9) are re-classified as: CF (1)
        WF-F (6), WF-D (10) are re-classified as: WF (2)
        SF-F (7), SF-D (11) are re-classified as: SF (3)
        OF-F (8), OF-D (12) are re-classified as: OF (4)
    
    MERGED-F-BIN (1 class): Train on 1-12, but treat all front types and stages as one type. This means that classes 1-12 will all be one class (1).
    
    MERGED-T (1 class): Train on 14-15, but treat troughs and tropical troughs as the same. In other words, TT (15) becomes TROF (14).

    MERGED-ALL (7 classes): Train on all classes (1-16), but make the changes in the MERGED-F and MERGED-T codes.
    
    **** NOTE: The number of classes does NOT include the 'no front' class.
    """
    parser.add_argument('--fss_c', type=float, required=False, help='C hyperparameter for the FSS loss sigmoid function.')
    parser.add_argument('--fss_mask_size', type=int, required=False, help='Mask size for the FSS loss function.')
    parser.add_argument('--gpu_device', type=int, help='GPU device numbers.')
    parser.add_argument('--import_model_number', type=int, required=False,
                        help='Number of the model that you would like to import.')
    parser.add_argument('--input_size', type=int, nargs='+', help='Size of the U-Net input.')
    parser.add_argument('--kernel_size', type=int, help='Size of the convolution kernels.')
    parser.add_argument('--learning_rate', type=float, required=False, help='Learning rate for U-Net optimizer.')
    parser.add_argument('--levels', type=int, help='Number of levels in the U-Net.')
    parser.add_argument('--loss', type=str, required=False, help='Loss function for the U-Net')
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth for GPUs')
    parser.add_argument('--metric', type=str, required=False, help='Metric for evaluating the U-Net during training.')
    parser.add_argument('--model_dir', type=str, required=False, help='Directory where the models are or will be saved to.')
    parser.add_argument('--model_number', type=int, required=False, help='Number that the model will be assigned.')
    parser.add_argument('--model_type', type=str, help='Model type.')
    parser.add_argument('--num_variables', type=int, required=False, help='Number of variables in the variable datasets.')
    parser.add_argument('--pickle_dir', type=str, help='Directory where the pickle files are stored.')
    parser.add_argument('--pixel_expansion', type=int, default=1, help='Number of pixels to expand the fronts by.')
    parser.add_argument('--pool_size', type=int, nargs='+', help='Pool size for the MaxPooling layers in the U-Net.')
    parser.add_argument('--test_years', type=int, nargs="+", required=False, help='Years for the test set.')
    parser.add_argument('--train_valid_batch_size', type=int, required=False, nargs=2, help='Batch sizes for the U-Net.')
    parser.add_argument('--train_valid_domain', type=str, required=False, nargs=2, help='Domains for training and validation')
    parser.add_argument('--train_valid_fronts', type=int, required=False, nargs=2,
                        help='How many pixels with fronts an image must have for it to be passed through the generator.')
    parser.add_argument('--train_valid_steps', type=int, nargs=2, default=[20, 20], help='Number of steps for each epoch.')
    parser.add_argument('--upsample_size', type=int, nargs='+', help='Upsample size for the UpSampling layers in the U-Net.')
    parser.add_argument('--valid_freq', type=int, default=1, help='How many epochs to pass before each validation.')
    parser.add_argument('--validation_years', type=int, nargs="+", required=False, help='Years for the validation set.')

    args = parser.parse_args()
    provided_arguments = vars(args)

    # Convert pool size and upsample size to tuples
    pool_size, upsample_size = tuple(args.pool_size), tuple(args.upsample_size)

    if len(args.front_types) == 1:
        front_types = args.front_types[0]
    else:
        front_types = args.front_types

    if args.gpu_device is not None:
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        tf.config.experimental.set_visible_devices(devices=gpus[args.gpu_device], device_type='GPU')

        # Allow for memory growth on the GPU. This will only use the GPU memory that is required rather than allocating all of the GPU's memory.
        if args.memory_growth:
            tf.config.experimental.set_memory_growth(device=gpus[args.gpu_device], enable=True)

    if args.import_model_number is not None and args.model_number is not None:
        raise errors.ArgumentConflictError("import_model_number and job_number cannot be passed at the same time.")
    if args.import_model_number is None and args.model_number is None:
        raise errors.MissingArgumentError("one of the following arguments must be provided: import_model_number, job_number")

    if args.loss == 'fss':
        required_arguments = ['fss_c', 'fss_mask_size']
        check_arguments(provided_arguments, required_arguments)

    if args.import_model_number is not None:
        if args.image_size is not None:
            raise errors.ArgumentConflictError('import_model_number is not None but the following argument was passed: image_size')
        required_arguments = ['epochs', 'front_types', 'learning_rate', 'loss', 'metric', 'model_dir', 'num_dimensions',
            'num_variables', 'pixel_expansion', 'train_valid_batch_size', 'train_valid_domains', 'train_valid_fronts',
            'train_valid_steps', 'valid_freq', 'workers']
        check_arguments(provided_arguments, required_arguments)
        print("WARNING: You are about to import model %d for training." % args.import_model_number)
        train_imported_unet(args.train_valid_domains[0], args.train_valid_domains[1], args.learning_rate, args.epochs, args.train_valid_steps[0],
            args.train_valid_batch_size[0], args.train_valid_fronts[0], args.train_valid_steps[1], args.train_valid_batch_size[1],
            args.valid_freq, args.train_valid_fronts[1], args.loss, args.import_model_number, args.model_dir,
            args.front_types, args.fss_mask_size, args.fss_c, args.pixel_expansion, args.num_variables, args.validation_years,
            args.test_years, args.num_dimensions, args.metric)

    if args.model_number is not None:
        required_arguments = ['input_size', 'kernel_size', 'pool_size', 'upsample_size', 'levels', 'filter_num', 'epochs',
            'loss', 'metric', 'fss_mask_size', 'fss_c', 'learning_rate', 'train_valid_domain', 'train_valid_batch_size',
            'train_valid_steps', 'train_valid_fronts', 'valid_freq', 'front_types', 'pixel_expansion', 'model_number',
            'model_dir', 'model_type', 'pickle_dir']

        check_arguments(provided_arguments, required_arguments)

        train_new_unet(args.input_size, args.kernel_size, pool_size, upsample_size, args.levels, args.filter_num, args.epochs,
            args.loss, args.metric, args.fss_mask_size, args.fss_c, args.learning_rate, args.train_valid_domain, args.train_valid_batch_size,
            args.train_valid_steps, args.train_valid_fronts, args.valid_freq, front_types, args.pixel_expansion, args.test_years,
            args.validation_years, args.model_number, args.model_dir, args.model_type, args.pickle_dir)
