"""
Function that trains a new U-Net model.

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 6/7/2023 11:46 PM CT
"""

import argparse
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import pickle
import numpy as np
import file_manager as fm
import os
import custom_losses
import custom_metrics
import models
import datetime
from utils import settings, misc, data_utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, required=True, help='Directory where the models are or will be saved to.')
    parser.add_argument('--model_number', type=int, help='Number that the model will be assigned.')
    parser.add_argument('--era5_tf_indir', type=str, required=True, help='Directory where the tensorflow datasets are stored.')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs for the U-Net training.')
    parser.add_argument('--patience', type=int, required=True, help='Patience for EarlyStopping callback')
    parser.add_argument('--verbose', type=int, default=2, help='Model.fit verbose')

    ### GPU arguments ###
    parser.add_argument('--gpu_device', type=int, nargs='+', help='GPU device numbers.')
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth for GPUs')
    parser.add_argument('--disable_tensorfloat32', action='store_true', help='Disable TensorFloat32 execution.')

    ### Hyperparameters ###
    parser.add_argument('--learning_rate', type=float, help='Learning rate for U-Net optimizer.')
    parser.add_argument('--train_valid_batch_size', type=int, nargs=2, help='Batch sizes for the U-Net.')
    parser.add_argument('--train_valid_domain', type=str, nargs=2, help='Domains for training and validation')
    parser.add_argument('--train_valid_steps', type=int, nargs=2, default=[20, 20], help='Number of steps for each epoch.')
    parser.add_argument('--valid_freq', type=int, default=1, help='How many epochs to pass before each validation.')

    ### U-Net arguments ###
    parser.add_argument('--model_type', type=str, help='Model type.')
    parser.add_argument('--activation', type=str, help='Activation function to use in the U-Net')
    parser.add_argument('--batch_normalization', action='store_true', help='Use batch normalization in the model')
    parser.add_argument('--deep_supervision', action='store_true', help='Use deep supervision in the U-Net.')
    parser.add_argument('--filter_num', type=int, nargs='+', help='Number of filters in each level of the U-Net.')
    parser.add_argument('--filter_num_aggregate', type=int, help='Number of filters in aggregated feature maps')
    parser.add_argument('--filter_num_skip', type=int, help='Number of filters in full-scale skip connections')
    parser.add_argument('--first_encoder_connections', action='store_true', help='Enable first encoder connections in the U-Net 3+')
    parser.add_argument('--kernel_size', type=int, nargs='+', help='Size of the convolution kernels.')
    parser.add_argument('--levels', type=int, help='Number of levels in the U-Net.')
    parser.add_argument('--loss', type=str, help='Loss function for the U-Net')
    parser.add_argument('--loss_args', type=str, help="String containing arguments for the loss function. See 'utils.misc.string_arg_to_dict' "
                                                      "for more details.")
    parser.add_argument('--metric', type=str, help='Metric for evaluating the U-Net during training.')
    parser.add_argument('--metric_args', type=str, help="String containing arguments for the metric. See 'utils.misc.string_arg_to_dict' "
                                                        "for more details.")
    parser.add_argument('--modules_per_node', type=int, default=5, help='Number of convolution modules in each node')
    parser.add_argument('--padding', type=str, default='same', help='Padding to use in the model')
    parser.add_argument('--pool_size', type=int, nargs='+', help='Pool size for the MaxPooling layers in the U-Net.')
    parser.add_argument('--squeeze_dims', type=int, help='Axis of the U-Net to squeeze')
    parser.add_argument('--upsample_size', type=int, nargs='+', help='Upsample size for the UpSampling layers in the U-Net.')
    parser.add_argument('--use_bias', action='store_true', help='Use bias parameters in the U-Net')

    ### Data arguments ###
    parser.add_argument('--num_training_years', type=int, help='Number of years for the training dataset.')
    parser.add_argument('--training_years', type=int, nargs="+", help='Years for the training dataset.')
    parser.add_argument('--num_validation_years', type=int, help='Number of years for the validation set.')
    parser.add_argument('--validation_years', type=int, nargs="+", help='Years for the validation set.')

    ### Retraining model ###
    parser.add_argument('--retrain', action='store_true', help='Retrain a model')

    args = parser.parse_args()

    if args.loss_args is not None:
        loss_args = misc.string_arg_to_dict(args.loss_args)
    else:
        loss_args = dict()

    if args.metric_args is not None:
        metric_args = misc.string_arg_to_dict(args.metric_args)
    else:
        metric_args = dict()

    if args.gpu_device is not None:
        gpus = tf.config.list_physical_devices(device_type='GPU')
        tf.config.set_visible_devices(devices=[gpus[gpu] for gpu in args.gpu_device], device_type='GPU')

        # Allow for memory growth on the GPU. This will only use the GPU memory that is required rather than allocating all of the GPU's memory.
        if args.memory_growth:
            tf.config.experimental.set_memory_growth(device=[gpus[gpu] for gpu in args.gpu_device][0], enable=True)

        if args.disable_tensorfloat32:
            tf.config.experimental.enable_tensor_float_32_execution(False)  # Disable TensorFloat32 for matrix multiplication

    dataset_properties = pd.read_pickle('%s/dataset_properties.pkl' % args.era5_tf_indir)
    front_types = dataset_properties['front_types']
    variables = dataset_properties['variables']
    pressure_levels = dataset_properties['pressure_levels']
    image_size = dataset_properties['image_size']
    num_dims = dataset_properties['num_dims']

    if not args.retrain:

        if args.loss == 'fractions_skill_score':
            loss_string = 'fss_loss'
        elif args.loss == 'critical_success_index':
            loss_string = 'csi_loss'
        elif args.loss == 'brier_skill_score':
            loss_string = 'bss_loss'
        else:
            loss_string = None

        if args.metric == 'fractions_skill_score':
            metric_string = 'fss'
        elif args.metric == 'critical_success_index':
            metric_string = 'csi'
        elif args.metric == 'brier_skill_score':
            metric_string = 'bss'
        else:
            metric_string = None
    
        all_years = np.arange(2008, 2021)

        if args.num_training_years is not None:
            if args.training_years is not None:
                raise TypeError("Cannot explicitly declare the training years if --num_training_years is passed")
            training_years = list(sorted(np.random.choice(all_years, args.num_training_years, replace=False)))
        else:
            if args.training_years is None:
                raise TypeError("Must pass one of the following arguments: --training_years, --num_training_years")
            training_years = list(sorted(args.training_years))

        if args.num_validation_years is not None:
            if args.validation_years is not None:
                raise TypeError("Cannot explicitly declare the validation years if --num_validation_years is passed")
            validation_years = list(sorted(np.random.choice([year for year in all_years if year not in training_years], args.num_validation_years, replace=False)))
        else:
            if args.validation_years is None:
                raise TypeError("Must pass one of the following arguments: --validation_years, --num_validation_years")
            validation_years = list(sorted(args.validation_years))

        if len(training_years) + len(validation_years) > 12:
            raise ValueError("No testing years are available: the total number of training and validation years cannot be greater than 12")

        test_years = [year for year in all_years if year not in training_years + validation_years]

        if args.model_number is None:  # If no model number is provided, select a number based on the current date and time
            model_number = int(datetime.datetime.now().timestamp() - 1677265215)
        else:
            model_number = args.model_number

        # Convert pool size and upsample size to tuples
        pool_size, upsample_size = tuple(args.pool_size), tuple(args.upsample_size)

        if any(front_type == front_types for front_type in [['MERGED-F_BIN'], ['MERGED-T'], ['F_BIN']]):
            num_classes = 2
        elif front_types == ['MERGED-F']:
            num_classes = 5
        elif front_types == ['MERGED-ALL']:
            num_classes = 8
        else:
            num_classes = len(front_types) + 1

        # Create dictionary containing information about the model. This simplifies the process of loading the model
        model_properties = dict({})
        model_properties['model_number'] = model_number
        model_properties['training_years'] = sorted(training_years)
        model_properties['validation_years'] = sorted(validation_years)
        model_properties['test_years'] = sorted(test_years)
        model_properties['num_dimensions'] = 3
        model_properties['domains'] = args.train_valid_domain
        model_properties['batch_sizes'] = args.train_valid_batch_size
        model_properties['steps_per_epoch'] = args.train_valid_steps
        model_properties['valid_freq'] = 1
        model_properties['optimizer'] = 'Adam'
        model_properties['learning_rate'] = args.learning_rate
        model_properties['image_size'] = image_size
        model_properties['kernel_size'] = args.kernel_size
        model_properties['normalization_parameters'] = data_utils.normalization_parameters
        model_properties['loss_string'] = loss_string
        model_properties['loss_args'] = loss_args
        model_properties['metric_string'] = metric_string
        model_properties['metric_args'] = metric_args
        model_properties['variables'] = variables
        model_properties['pressure_levels'] = pressure_levels
        model_properties['front_types'] = front_types
        model_properties['classes'] = num_classes
        model_properties['model_type'] = args.model_type
        model_properties['deep_supervision'] = args.deep_supervision

        unet_kwargs = dict({'kernel_size': args.kernel_size,
                            'squeeze_dims': args.squeeze_dims,
                            'modules_per_node': args.modules_per_node,
                            'activation': args.activation,
                            'batch_normalization': args.batch_normalization,
                            'padding': args.padding,
                            'use_bias': args.use_bias})

        if args.model_type == 'unet_3plus':
            unet_kwargs['first_encoder_connections'] = args.first_encoder_connections
            unet_kwargs['deep_supervision'] = args.deep_supervision

        train_valid_batch_size = args.train_valid_batch_size
        train_valid_steps = args.train_valid_steps
        valid_freq = args.valid_freq

        unet_model = getattr(models, args.model_type)

        print("Training years:", training_years)
        print("Validation years:", validation_years)
        print("Test years:", test_years)

    else:

        model_number = args.model_number

        model_properties = pd.read_pickle('%s/model_%d/model_%d_properties.pkl' % (args.model_dir, model_number, model_number))

        front_types = model_properties['front_types']

        training_years = model_properties['training_years']
        validation_years = model_properties['validation_years']
        test_years = model_properties['test_years']

        train_valid_batch_size = model_properties['batch_sizes']
        train_valid_steps = model_properties['steps_per_epoch']
        valid_freq = model_properties['valid_freq']

    model_filepath = '%s/model_%d/model_%d.h5' % (args.model_dir, model_number, model_number)
    history_filepath = '%s/model_%d/model_%d_history.csv' % (args.model_dir, model_number, model_number)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, save_freq='epoch')  # ModelCheckpoint: saves model at a specified interval
    early_stopping = EarlyStopping('val_loss', patience=args.patience, verbose=1)  # EarlyStopping: stops training early if a metric does not improve after a specified number of epochs (patience)
    history_logger = CSVLogger(history_filepath, separator=",", append=True)  # Saves loss/AUC data every epoch

    print("Building datasets")
    era5_files_obj = fm.DataFileLoader(args.era5_tf_indir, data_file_type='era5-tensorflow')
    era5_files_obj.training_years = training_years
    era5_files_obj.validation_years = validation_years
    era5_files_obj.pair_with_fronts(args.era5_tf_indir, front_types=front_types)

    ### Training dataset ###
    training_inputs = era5_files_obj.data_files_training
    training_labels = era5_files_obj.front_files_training

    # Shuffle monthly data
    training_files = list(zip(training_inputs, training_labels))
    np.random.shuffle(training_files)
    training_inputs, training_labels = zip(*training_files)

    training_dataset = data_utils.combine_datasets(training_inputs, training_labels)
    print(f"Images in training dataset: {len(training_dataset):,}")
    training_buffer_size = np.min([len(training_dataset), settings.MAX_TRAIN_BUFFER_SIZE])
    training_dataset = training_dataset.shuffle(buffer_size=training_buffer_size)
    training_dataset = training_dataset.batch(train_valid_batch_size[0], drop_remainder=True, num_parallel_calls=4)
    training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE)

    ### Validation dataset ###
    validation_inputs = era5_files_obj.data_files_validation
    validation_labels = era5_files_obj.front_files_validation
    validation_dataset = data_utils.combine_datasets(validation_inputs, validation_labels)
    print(f"Images in validation dataset: {len(validation_dataset):,}")
    validation_dataset = validation_dataset.batch(train_valid_batch_size[1], drop_remainder=True, num_parallel_calls=4)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    input_shape = list(training_dataset.take(0).element_spec[0].shape[1:])
    if len(input_shape) == 4:
        for i in range(len(input_shape) - 2):
            input_shape[i] = None
    else:
        for i in range(len(input_shape) - 1):
            input_shape[i] = None

    input_shape = tuple(input_shape)

    with tf.distribute.MirroredStrategy().scope():

        if not args.retrain:
            print("Building model")
            model = unet_model(input_shape, num_classes, args.pool_size, args.upsample_size, args.levels, args.filter_num, **unet_kwargs)
            loss_function = getattr(custom_losses, args.loss)(**loss_args)
            metric_function = getattr(custom_metrics, args.metric)(**metric_args)
            adam = Adam(learning_rate=args.learning_rate)
            model.compile(loss=loss_function, optimizer=adam, metrics=metric_function)
        else:
            model = fm.load_model(args.model_number, args.model_dir)

    model.summary()

    if not args.retrain:
        os.mkdir('%s/model_%d' % (args.model_dir, model_number))  # Make folder for model
        os.mkdir('%s/model_%d/maps' % (args.model_dir, model_number))  # Make folder for model predicton maps
        os.mkdir('%s/model_%d/probabilities' % (args.model_dir, model_number))  # Make folder for prediction data files
        os.mkdir('%s/model_%d/statistics' % (args.model_dir, model_number))  # Make folder for statistics data files
        with open('%s/model_%d/model_%d_properties.pkl' % (args.model_dir, model_number, model_number), 'wb') as f:
            pickle.dump(model_properties, f)

    print("Fitting model")
    model.fit(training_dataset.repeat(), validation_data=validation_dataset.repeat(), validation_freq=valid_freq,
        epochs=args.epochs, steps_per_epoch=train_valid_steps[0], validation_steps=train_valid_steps[1], callbacks=[early_stopping, checkpoint, history_logger],
        verbose=args.verbose)
