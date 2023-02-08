"""
Function that trains a new or imported U-Net model.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 9/28/2022 8:25 PM CDT
"""

import random
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import pickle
import numpy as np
import file_manager as fm
import os
import custom_losses
import models
from utils import data_utils, settings
import xarray as xr
import datetime
import time
import sys

# tf.config.experimental.enable_tensor_float_32_execution(False)

def combine_datasets(dataset_files):

    complete_dataset = tf.data.Dataset.load(dataset_files.pop(0))
    for dataset_file in dataset_files:
        dataset = tf.data.Dataset.load(dataset_file)
        complete_dataset = complete_dataset.concatenate(dataset)

    return complete_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, required=True, help='Directory where the models are or will be saved to.')
    parser.add_argument('--model_number', type=int, help='Number that the model will be assigned.')
    parser.add_argument('--model_type', type=str, help='Model type.')
    parser.add_argument('--era5_tf_indir', type=str, required=True, help='Directory where the tensorflow datasets are stored.')

    ### GPU arguments ###
    parser.add_argument('--gpu_device', type=int, nargs='+', help='GPU device numbers.')
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth for GPUs')

    ### Hyperparameters ###
    parser.add_argument('--epochs', type=int, default=6000, help='Number of epochs for the U-Net training.')
    parser.add_argument('--fss_c', type=float, help='C hyperparameter for the FSS loss sigmoid function.')
    parser.add_argument('--fss_mask_size', type=int, help='Mask size for the FSS loss function.')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for U-Net optimizer.')
    parser.add_argument('--train_valid_batch_size', type=int, nargs=2, help='Batch sizes for the U-Net.')
    parser.add_argument('--train_valid_domain', type=str, nargs=2, help='Domains for training and validation')
    parser.add_argument('--train_valid_steps', type=int, nargs=2, default=[20, 20], help='Number of steps for each epoch.')
    parser.add_argument('--valid_freq', type=int, default=1, help='How many epochs to pass before each validation.')

    ### U-Net arguments ###
    parser.add_argument('--activation', type=str, required=True, help='Activation function to use in the U-Net')
    parser.add_argument('--batch_normalization', action='store_true', help='Use batch normalization in the model')
    parser.add_argument('--filter_num', type=int, nargs='+', required=True, help='Number of filters in each level of the U-Net.')
    parser.add_argument('--filter_num_aggregate', type=int, help='Number of filters in aggregated feature maps')
    parser.add_argument('--filter_num_skip', type=int, help='Number of filters in full-scale skip connections')
    parser.add_argument('--first_encoder_connections', action='store_true', help='Enable first encoder connections in the U-Net 3+')
    parser.add_argument('--image_size', type=int, nargs='+', required=False, default=(128, 128), help='Size of the U-Net input images.')
    parser.add_argument('--kernel_size', type=int, required=True, help='Size of the convolution kernels.')
    parser.add_argument('--levels', type=int, required=True, help='Number of levels in the U-Net.')
    parser.add_argument('--loss', type=str, required=True, help='Loss function for the U-Net')
    parser.add_argument('--metric', type=str, help='Metric for evaluating the U-Net during training.')
    parser.add_argument('--modules_per_node', type=int, default=5, help='Number of convolution modules in each node')
    parser.add_argument('--padding', type=str, default='same', help='Padding to use in the model')
    parser.add_argument('--pool_size', type=int, nargs='+', required=True, help='Pool size for the MaxPooling layers in the U-Net.')
    parser.add_argument('--squeeze_dims', type=int, help='Axis of the U-Net to squeeze')
    parser.add_argument('--upsample_size', type=int, nargs='+', required=True, help='Upsample size for the UpSampling layers in the U-Net.')
    parser.add_argument('--use_bias', action='store_true', help='Use bias parameters in the U-Net')

    ### Data arguments ###
    parser.add_argument('--front_types', type=str, nargs='+', required=True, help='Front types that the model will be trained on.')
    parser.add_argument('--pixel_expansion', type=int, default=1, help='Number of pixels to expand the fronts by.')
    parser.add_argument('--pressure_levels', type=str, nargs='+', required=True, help='Pressure levels to use')
    parser.add_argument('--variables', type=str, nargs='+', required=True, help='Variables to use')
    parser.add_argument('--num_test_years', type=int, help='Number of years for the test set.')
    parser.add_argument('--test_years', type=int, nargs="+", help='Years for the test set.')
    parser.add_argument('--num_validation_years', type=int, help='Number of years for the validation set.')
    parser.add_argument('--validation_years', type=int, nargs="+", help='Years for the validation set.')

    args = parser.parse_args()

    args.variables = sorted(args.variables)
    num_pressure_levels = len(args.pressure_levels)

    available_years = np.arange(2008, 2021)
    np.random.shuffle(available_years)

    if args.num_validation_years is not None:
        num_training_years = len(available_years) - args.num_validation_years - args.num_test_years
        training_years = sorted(available_years[:num_training_years])
        validation_years = sorted(available_years[num_training_years:num_training_years + args.num_validation_years])
        test_years = sorted(available_years[-args.num_test_years:])
    else:
        validation_indices = [np.where(available_years == val_year)[0][0] for val_year in args.validation_years if val_year in available_years]
        validation_years = sorted([available_years[index] for index in validation_indices])
        available_years = np.delete(available_years, validation_indices)

        test_indices = [np.where(available_years == test_year)[0][0] for test_year in args.test_years if test_year in available_years]
        test_years = sorted([available_years[index] for index in test_indices])
        training_years = sorted(np.delete(available_years, test_indices))

    if args.model_number is None:  # If no model number is provided, select a number based on the current date and time
        model_number = int(datetime.datetime.now().timestamp())
    else:
        model_number = args.model_number

    # Convert pool size and upsample size to tuples
    pool_size, upsample_size = tuple(args.pool_size), tuple(args.upsample_size)

    if len(args.front_types) == 1:
        front_types = args.front_types[0]
    else:
        front_types = args.front_types

    if args.gpu_device is not None:
        gpus = tf.config.list_physical_devices(device_type='GPU')
        tf.config.set_visible_devices(devices=[gpus[gpu] for gpu in args.gpu_device], device_type='GPU')

        # Allow for memory growth on the GPU. This will only use the GPU memory that is required rather than allocating all of the GPU's memory.
        if args.memory_growth:
            tf.config.experimental.set_memory_growth(device=[gpus[gpu] for gpu in args.gpu_device][0], enable=True)

    if front_types == 'MERGED-F_BIN' or front_types == 'MERGED-T' or front_types == 'F_BIN':
        num_classes = 2
    elif front_types == 'MERGED-F':
        num_classes = 5
    elif front_types == 'MERGED-ALL':
        num_classes = 8
    else:
        num_classes = len(front_types) + 1

    # Create dictionary containing information about the model. This simplifies the process of loading the model
    model_properties = dict({})
    model_properties['model_number'] = model_number
    model_properties['training_years'] = sorted(training_years)
    model_properties['validation_years'] = sorted(args.validation_years)
    model_properties['test_years'] = sorted(args.test_years)
    model_properties['domains'] = args.train_valid_domain
    model_properties['batch_sizes'] = args.train_valid_batch_size
    model_properties['steps_per_epoch'] = args.train_valid_steps
    model_properties['valid_freq'] = 1
    model_properties['optimizer'] = 'Adam'
    model_properties['learning_rate'] = args.learning_rate
    model_properties['image_size'] = args.image_size
    model_properties['kernel_size'] = args.kernel_size
    model_properties['pixel_expansion'] = args.pixel_expansion
    model_properties['normalization'] = 'min-max'
    model_properties['loss_function'] = args.loss
    model_properties['fss_mask_c'] = (args.fss_mask_size, args.fss_c)
    model_properties['variables'] = sorted(args.variables)
    model_properties['pressure_levels'] = args.pressure_levels
    model_properties['metric'] = args.metric
    model_properties['front_types'] = front_types
    model_properties['classes'] = num_classes
    model_properties['model_type'] = args.model_type
    model_properties['squeeze_dims'] = args.squeeze_dims

    os.mkdir('%s/model_%d' % (args.model_dir, model_number))  # Make folder for model
    os.mkdir('%s/model_%d/maps' % (args.model_dir, model_number))  # Make folder for model predicton maps
    os.mkdir('%s/model_%d/probabilities' % (args.model_dir, model_number))  # Make folder for prediction data files
    os.mkdir('%s/model_%d/statistics' % (args.model_dir, model_number))  # Make folder for statistics data files
    model_filepath = '%s/model_%d/model_%d.h5' % (args.model_dir, model_number, model_number)
    history_filepath = '%s/model_%d/model_%d_history.csv' % (args.model_dir, model_number, model_number)
    with open('%s/model_%d/model_%d_properties.pkl' % (args.model_dir, model_number, model_number), 'wb') as f:
        pickle.dump(model_properties, f)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, save_freq='epoch')  # ModelCheckpoint: saves model at a specified interval
    early_stopping = EarlyStopping('val_loss', patience=150, verbose=1)  # EarlyStopping: stops training early if a metric does not improve after a specified number of epochs (patience)
    history_logger = CSVLogger(history_filepath, separator=",", append=True)  # Saves loss/AUC data every epoch

    full_image_size = (None, None, None, 12)

    print("Building model")
    ### Select the model ###
    unet_kwargs = dict({'kernel_size': args.kernel_size,
                        'squeeze_dims': args.squeeze_dims,
                        'modules_per_node': args.modules_per_node,
                        'activation': args.activation,
                        'batch_normalization': args.batch_normalization,
                        'padding': args.padding,
                        'use_bias': args.use_bias})
    if args.model_type == 'unet_3plus':
        unet_kwargs['first_encoder_connections'] = args.first_encoder_connections

    unet_model = getattr(models, args.model_type)

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():

        model = unet_model(full_image_size, num_classes, args.pool_size, args.upsample_size, args.levels, args.filter_num, **unet_kwargs)

        loss_function = custom_losses.make_fractions_skill_score(args.fss_mask_size, c=args.fss_c)

        adam = Adam(learning_rate=args.learning_rate)
        model.compile(loss=loss_function, optimizer=adam)

    model.summary()

    print("Building datasets")
    era5_files_obj = fm.ERA5files(args.era5_tf_indir, file_type='tensorflow')
    era5_files_obj.training_years = training_years
    era5_files_obj.validation_years = validation_years

    ### Training dataset ###
    print(" ---> training")
    start_time = time.time()
    training_dataset_files = era5_files_obj.era5_files_training
    training_dataset = combine_datasets(training_dataset_files)
    training_dataset = training_dataset.shuffle(buffer_size=len(training_dataset))
    training_dataset = training_dataset.batch(args.train_valid_batch_size[0], drop_remainder=True, num_parallel_calls=20)
    training_dataset = training_dataset.prefetch(32)
    print(f"time elapsed: {time.time() - start_time}")

    ### Validation dataset ###
    print(" ---> validation")
    start_time = time.time()
    validation_dataset_files = era5_files_obj.era5_files_validation
    validation_dataset = combine_datasets(validation_dataset_files)
    validation_dataset = validation_dataset.batch(args.train_valid_batch_size[1], drop_remainder=True, num_parallel_calls=20)
    validation_dataset = validation_dataset.prefetch(32)
    print(f"time elapsed: {time.time() - start_time}")

    print("Fitting model")
    model.fit(training_dataset.repeat(), validation_data=validation_dataset.repeat(), validation_freq=args.valid_freq,
        epochs=args.epochs, steps_per_epoch=args.train_valid_steps[0], validation_steps=args.train_valid_steps[1], callbacks=[early_stopping, checkpoint, history_logger],
        verbose=2)
