"""
Function that trains a new U-Net model.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.8.9
"""

import argparse
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import tensorflow as tf
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
    parser.add_argument('--era5_tf_indirs', type=str, required=True, nargs='+',
        help='Directories for the TensorFlow datasets. One or two paths can be passed. If only one path is passed, then the '
             'training and validation datasets will be pulled from this path. If two paths are passed, the training dataset '
             'will be pulled from the first path and the validation dataset from the second.')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs for the U-Net training.')
    parser.add_argument('--patience', type=int,
        help='Patience for EarlyStopping callback. If this argument is not provided, it will be set according to the size '
             'of the training dataset (images in training set divided by the product of the batch size and steps).')
    parser.add_argument('--verbose', type=int, default=2,
        help='Model.fit verbose. Unless you want a text file that is several hundred megabytes in size and takes 10 years '
             'to scroll through, I suggest you leave this at 2.')

    ### GPU and hardware arguments ###
    parser.add_argument('--gpu_device', type=int, nargs='+', help='GPU device numbers.')
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth for GPUs')
    parser.add_argument('--num_parallel_calls', type=int, default=4,
        help='Number of parallel calls for retrieving batches for the training and validation datasets.')
    parser.add_argument('--disable_tensorfloat32', action='store_true', help='Disable TensorFloat32 execution.')

    ### Hyperparameters ###
    parser.add_argument('--learning_rate', type=float,
        help='Learning rate for U-Net optimizer. If left as None, then the default optimizer learning rate will be used.')
    parser.add_argument('--batch_size', type=int, required=True, nargs='+',
        help='Batch sizes for the U-Net. Up to 2 arguments can be passed. If 1 argument is passed, the value will be both '
             'the training and validation batch sizes. If 2 arguments are passed, the first and second arguments will be '
             'the training and validation batch sizes, respectively.')
    parser.add_argument('--steps', type=int, required=True, nargs='+',
        help='Number of steps for each epoch. Up to 2 arguments can be passed. If 1 argument is passed, the value will only '
             'be applied to the number of steps per epoch, and the number of validation steps will be calculated by tensorflow '
             'such that the entire validation dataset is passed into the model during validation. If 2 arguments are passed, '
             'then the arguments are the number of steps in training and validation. If no arguments are passed, then the '
             'number of steps in both training and validation will be calculated by tensorflow.')
    parser.add_argument('--valid_freq', type=int, default=1, help='How many epochs to complete before each validation.')

    ### U-Net arguments ###
    parser.add_argument('--model_type', type=str,
        help='Model type. Options are: unet, unet_ensemble, unet_plus, unet_2plus, unet_3plus.')
    parser.add_argument('--activation', type=str,
        help='Activation function to use in the U-Net. Refer to utils.unet_utils.choose_activation_layer to see all available '
             'activation functions.')
    parser.add_argument('--batch_normalization', action='store_true',
        help='Use batch normalization in the model. This will place batch normalization layers after each convolution layer.')
    parser.add_argument('--deep_supervision', action='store_true', help='Use deep supervision in the U-Net.')
    parser.add_argument('--filter_num', type=int, nargs='+', help='Number of filters in each level of the U-Net.')
    parser.add_argument('--filter_num_aggregate', type=int,
        help='Number of filters in aggregated feature maps. This argument is only used in the U-Net 3+ model.')
    parser.add_argument('--filter_num_skip', type=int, help='Number of filters in full-scale skip connections in the U-Net 3+.')
    parser.add_argument('--first_encoder_connections', action='store_true', help='Enable first encoder connections in the U-Net 3+.')
    parser.add_argument('--kernel_size', type=int, nargs='+', help='Size of the convolution kernels.')
    parser.add_argument('--levels', type=int, help='Number of levels in the U-Net.')
    parser.add_argument('--loss', type=str, nargs='+',
        help="Loss function for the U-Net (arg 1), with keyword arguments (arg 2). Keyword arguments must be passed as a "
             "string in the second argument. See 'utils.misc.string_arg_to_dict' for more details. Raises a ValueError if "
             "more than 2 arguments are passed.")
    parser.add_argument('--metric', type=str, nargs='+',
        help="Metric for evaluating the U-Net during training (arg 1), with keyword arguments (arg 2). Keyword arguments "
             "must be passed as a string in the second argument. See 'utils.misc.string_arg_to_dict' for more details. Raises "
             "a ValueError if more than 2 arguments are passed.")
    parser.add_argument('--modules_per_node', type=int, default=5, help='Number of convolution modules in each node')
    parser.add_argument('--optimizer', type=str, nargs='+', default=['Adam', ],
        help="Optimizer to use during the training process (arg 1), with keyword arguments (arg 2). Keyword arguments "
             "must be passed as a string in the second argument. See 'utils.misc.string_arg_to_dict' for more details. Raises "
             "a ValueError if more than 2 arguments are passed.")
    parser.add_argument('--padding', type=str, default='same', help='Padding to use in the model')
    parser.add_argument('--pool_size', type=int, nargs='+', help='Pool size for the MaxPooling layers in the U-Net.')
    parser.add_argument('--upsample_size', type=int, nargs='+', help='Upsample size for the UpSampling layers in the U-Net.')
    parser.add_argument('--use_bias', action='store_true', help='Use bias parameters in the U-Net')

    ### Constraints, initializers, and regularizers ###
    parser.add_argument('--activity_regularizer', type=str, nargs='+', default=[None, ],
        help='Regularizer function applied to the output of the Conv2D/Conv3D layers. A second string argument can be passed '
             'containing keyword arguments for the regularizer.')
    parser.add_argument('--bias_constraint', type=str, nargs='+', default=[None, ],
        help='Constraint function applied to the bias vector of the Conv2D/Conv3D layers. A second string argument can be '
             'passed containing keyword arguments for the constraint.')
    parser.add_argument('--bias_initializer', type=str, default='zeros', help='Initializer for the bias vector in the Conv2D/Conv3D layers.')
    parser.add_argument('--bias_regularizer', type=str, nargs='+', default=[None, ],
        help='Regularizer function applied to the bias vector in the Conv2D/Conv3D layers. A second string argument can '
             'be passed containing keyword arguments for the regularizer.')
    parser.add_argument('--kernel_constraint', type=str, nargs='+', default=[None, ],
        help='Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers. A second string argument can '
             'be passed containing keyword arguments for the constraint.')
    parser.add_argument('--kernel_initializer', type=str, default='glorot_uniform', help='Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.')
    parser.add_argument('--kernel_regularizer', type=str, nargs='+', default=[None, ],
        help='Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers. A second string argument '
             'can be passed containing keyword arguments for the regularizer.')

    ### Data arguments ###
    parser.add_argument('--num_training_years', type=int, help='Number of years for the training dataset.')
    parser.add_argument('--training_years', type=int, nargs="+", help='Years for the training dataset.')
    parser.add_argument('--num_validation_years', type=int, help='Number of years for the validation set.')
    parser.add_argument('--validation_years', type=int, nargs="+", help='Years for the validation set.')
    parser.add_argument('--shuffle', type=str, default='full',
        help="Shuffling method for the training set. Valid options are 'lazy' or 'full' (default is 'full'). "
             "A 'lazy' shuffle will only shuffle the order of the monthly datasets but not the contents within. A 'full' "
             "shuffle will shuffle every image inside the dataset.")

    ### Retraining model ###
    parser.add_argument('--retrain', action='store_true', help='Retrain a model')

    parser.add_argument('--override_directory_check', action='store_true',
        help="Override the OSError caused by creating a new model directory that already exists. Normally, if the script "
             "crashes before or during the training of a new model, an OSError will be returned if the script is immediately "
             "ran again with the same model number as the model directory already exists. This is an intentional fail-safe "
             "designed to prevent models that already exist from being overwritten. Passing this boolean flag disables the "
             "fail-safe and can be useful if the script is being ran on a workload manager (e.g. SLURM) where jobs can fail "
             "and then be immediately requeued and ran again.")

    args = vars(parser.parse_args())

    if len(args['era5_tf_indirs']) > 2:
        raise ValueError("Only 1 or 2 paths can be passed into --era5_tf_indirs, received %d paths" % len(args['era5_tf_indirs']))
    elif len(args['era5_tf_indirs']) == 1:
        args['era5_tf_indirs'].append(args['era5_tf_indirs'][0])

    if args['shuffle'] != 'lazy' and args['shuffle'] != 'full':
        raise ValueError("Unrecognized shuffling method: %s. Valid methods are 'lazy' or 'full'" % args['shuffle'])

    # Check arguments that can only have a maximum length of 2
    for arg in ['loss', 'metric', 'optimizer', 'activity_regularizer', 'bias_constraint', 'bias_regularizer', 'kernel_constraint',
                'kernel_regularizer', 'batch_size', 'steps']:
        if len(args[arg]) > 2:
            raise ValueError("--%s can only take up to 2 arguments" % arg)

    ### Dictionary containing arguments that cannot be used for specific model types ###
    incompatible_args = {'unet': dict(deep_supervision=False, first_encoder_connections=False),
                         'unet_ensemble': dict(deep_supervision=False, first_encoder_connections=False),
                         'unet_plus': dict(first_encoder_connections=False),
                         'unet_2plus': dict(first_encoder_connections=False),
                         'unet_3plus': {},
                         'attention_unet': dict(upsample_size=None, deep_supervision=False, first_encoder_connections=False)}

    ### Make sure that incompatible arguments were not passed, and raise errors if they were passed ###
    incompatible_args_for_model = incompatible_args[args['model_type']]
    for arg in incompatible_args_for_model:
        if incompatible_args_for_model[arg] != args[arg]:
            raise ValueError(f"--{arg} must be '{incompatible_args_for_model[arg]}' when the model type is {args['model_type']}")

    ### Convert keyword argument strings to dictionaries ###
    loss_args = misc.string_arg_to_dict(args['loss'][1]) if len(args['loss']) > 1 else dict()
    metric_args = misc.string_arg_to_dict(args['metric'][1]) if len(args['metric']) > 1 else dict()
    optimizer_args = misc.string_arg_to_dict(args['optimizer'][1]) if len(args['optimizer']) > 1 else dict()
    activity_regularizer_args = misc.string_arg_to_dict(args['activity_regularizer'][1]) if len(args['activity_regularizer']) > 1 else dict()
    bias_constraint_args = misc.string_arg_to_dict(args['bias_constraint'][1]) if len(args['bias_constraint']) > 1 else dict()
    bias_regularizer_args = misc.string_arg_to_dict(args['bias_regularizer'][1]) if len(args['bias_regularizer']) > 1 else dict()
    kernel_constraint_args = misc.string_arg_to_dict(args['kernel_constraint'][1]) if len(args['kernel_constraint']) > 1 else dict()
    kernel_regularizer_args = misc.string_arg_to_dict(args['kernel_regularizer'][1]) if len(args['kernel_regularizer']) > 1 else dict()

    # learning rate is part of the optimizer
    if args['learning_rate'] is not None:
        optimizer_args['learning_rate'] = args['learning_rate']

    gpus = tf.config.list_physical_devices(device_type='GPU')  # Find available GPUs
    if len(gpus) > 0:

        print("Number of GPUs available: %d" % len(gpus))

        # Only make the selected GPU(s) visible to TensorFlow
        if args['gpu_device'] is not None:
            tf.config.set_visible_devices(devices=[gpus[gpu] for gpu in args['gpu_device']], device_type='GPU')
            gpus = tf.config.get_visible_devices(device_type='GPU')  # List of selected GPUs
            print("Using %d GPU(s):" % len(gpus), gpus)

        # Disable TensorFloat32 for matrix multiplication
        if args['disable_tensorfloat32']:
            tf.config.experimental.enable_tensor_float_32_execution(False)

        # Allow for memory growth on the GPU. This will only use the GPU memory that is required rather than allocating all of the GPU's memory.
        if args['memory_growth']:
            tf.config.experimental.set_memory_growth(device=[gpu for gpu in gpus][0], enable=True)

    else:
        print('WARNING: No GPUs found, all computations will be performed on CPUs.')
        tf.config.set_visible_devices([], 'GPU')

    train_dataset_properties = pd.read_pickle('%s/dataset_properties.pkl' % args['era5_tf_indirs'][0])
    valid_dataset_properties = pd.read_pickle('%s/dataset_properties.pkl' % args['era5_tf_indirs'][1])

    """ 
    Verify that the training and validation datasets have the same front types, variables, pressure levels, number of 
        dimensions, and normalization parameters.
    """
    try:
        assert train_dataset_properties['front_types'] == valid_dataset_properties['front_types']
        assert train_dataset_properties['variables'] == valid_dataset_properties['variables']
        assert train_dataset_properties['pressure_levels'] == valid_dataset_properties['pressure_levels']
        assert all(train_dataset_properties['num_dims'][num] == valid_dataset_properties['num_dims'][num] for num in range(2))
        assert train_dataset_properties['normalization_parameters'] == valid_dataset_properties['normalization_parameters']
    except AssertionError:
        raise TypeError("Training and validation dataset properties do not match. Select a different dataset(s) or choose "
                        "one dataset to use for both training and validation.")

    front_types = train_dataset_properties['front_types']
    variables = train_dataset_properties['variables']
    pressure_levels = train_dataset_properties['pressure_levels']
    image_size = train_dataset_properties['image_size']
    num_dims = train_dataset_properties['num_dims']

    if not args['retrain']:

        if args['loss'][0] == 'fractions_skill_score':
            loss_string = 'fss_loss'
        elif args['loss'][0] == 'critical_success_index':
            loss_string = 'csi_loss'
        elif args['loss'][0] == 'brier_skill_score':
            loss_string = 'bss_loss'
        else:
            loss_string = None

        if args['metric'][0] == 'fractions_skill_score':
            metric_string = 'fss'
        elif args['metric'][0] == 'critical_success_index':
            metric_string = 'csi'
        elif args['metric'][0] == 'brier_skill_score':
            metric_string = 'bss'
        else:
            metric_string = None
    
        all_years = np.arange(2008, 2021)

        if args['num_training_years'] is not None:
            if args['training_years'] is not None:
                raise TypeError("Cannot explicitly declare the training years if --num_training_years is passed")
            training_years = list(sorted(np.random.choice(all_years, args['num_training_years'], replace=False)))
        else:
            if args['training_years'] is None:
                raise TypeError("Must pass one of the following arguments: --training_years, --num_training_years")
            training_years = list(sorted(args['training_years']))

        if args['num_validation_years'] is not None:
            if args['validation_years'] is not None:
                raise TypeError("Cannot explicitly declare the validation years if --num_validation_years is passed")
            validation_years = list(sorted(np.random.choice([year for year in all_years if year not in training_years], args['num_validation_years'], replace=False)))
        else:
            if args['validation_years'] is None:
                raise TypeError("Must pass one of the following arguments: --validation_years, --num_validation_years")
            validation_years = list(sorted(args['validation_years']))

        if len(training_years) + len(validation_years) > 12:
            raise ValueError("No testing years are available: the total number of training and validation years cannot be greater than 12")

        test_years = [year for year in all_years if year not in training_years + validation_years]

        # If no model number was provided, select a number based on the current date and time.
        model_number = int(datetime.datetime.utcnow().timestamp() % 1e8) if args['model_number'] is None else args['model_number']

        # Convert pool size and upsample size to tuples
        pool_size = tuple(args['pool_size']) if args['pool_size'] is not None else None
        upsample_size = tuple(args['upsample_size']) if args['upsample_size'] is not None else None

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
        model_properties['domains'] = [train_dataset_properties['domain'], valid_dataset_properties['domain']]
        model_properties['normalization_parameters'] = data_utils.normalization_parameters
        model_properties['dataset_properties'] = train_dataset_properties
        model_properties['classes'] = num_classes

        # Place provided arguments into the model properties dictionary
        for arg in ['model_type', 'learning_rate', 'deep_supervision', 'model_number', 'kernel_size', 'modules_per_node',
                    'activation', 'batch_normalization', 'padding', 'use_bias', 'activity_regularizer', 'bias_constraint',
                    'bias_initializer', 'bias_regularizer', 'kernel_constraint', 'kernel_initializer', 'kernel_regularizer',
                    'first_encoder_connections', 'valid_freq', 'optimizer']:
            model_properties[arg] = args[arg]

        # Place local variables into the model properties dictionary
        for arg in ['loss_string', 'loss_args', 'metric_string', 'metric_args', 'image_size', 'training_years',
                    'validation_years', 'test_years']:
            model_properties[arg] = locals()[arg]

        # If using 3D inputs and 2D targets, squeeze out the vertical dimension of the model (index 2)
        squeeze_dims = 2 if num_dims == [3, 2] else None

        train_batch_size = args['batch_size'][0]
        valid_batch_size = args['batch_size'][0] if len(args['batch_size']) == 1 else args['batch_size'][1]
        train_steps = args['steps'][0]
        valid_steps = None if len(args['steps']) < 2 else args['steps'][1]
        valid_freq = args['valid_freq']

        model_properties['batch_sizes'] = [train_batch_size, valid_batch_size]
        model_properties['steps_per_epoch'] = [train_steps, valid_steps]

        unet_model = getattr(models, args['model_type'])
        unet_model_args = unet_model.__code__.co_varnames[:unet_model.__code__.co_argcount]  # pull argument names from unet function

        ### Arguments for the function used to build the U-Net ###
        unet_kwargs = {arg: args[arg] for arg in ['pool_size', 'upsample_size', 'levels', 'filter_num', 'kernel_size', 'modules_per_node',
            'activation', 'batch_normalization', 'padding', 'use_bias', 'bias_initializer', 'kernel_initializer', 'first_encoder_connections',
            'deep_supervision'] if arg in unet_model_args}
        unet_kwargs['squeeze_dims'] = squeeze_dims
        unet_kwargs['activity_regularizer'] = getattr(tf.keras.regularizers, args['activity_regularizer'][0])(**activity_regularizer_args) if args['activity_regularizer'][0] is not None else None
        unet_kwargs['bias_constraint'] = getattr(tf.keras.constraints, args['bias_constraint'][0])(**bias_constraint_args) if args['bias_constraint'][0] is not None else None
        unet_kwargs['kernel_constraint'] = getattr(tf.keras.constraints, args['kernel_constraint'][0])(**kernel_constraint_args) if args['kernel_constraint'][0] is not None else None
        unet_kwargs['bias_regularizer'] = getattr(tf.keras.regularizers, args['bias_regularizer'][0])(**bias_regularizer_args) if args['bias_regularizer'][0] is not None else None
        unet_kwargs['kernel_regularizer'] = getattr(tf.keras.regularizers, args['kernel_regularizer'][0])(**kernel_regularizer_args) if args['kernel_regularizer'][0] is not None else None

        print("Training years:", training_years)
        print("Validation years:", validation_years)
        print("Test years:", test_years)

    else:

        model_number = args['model_number']

        model_properties = pd.read_pickle('%s/model_%d/model_%d_properties.pkl' % (args['model_dir'], model_number, model_number))

        front_types = model_properties['front_types']

        training_years = model_properties['training_years']
        validation_years = model_properties['validation_years']
        test_years = model_properties['test_years']

        train_batch_size, valid_batch_size = model_properties['batch_sizes']
        train_steps, valid_steps = model_properties['steps_per_epoch']
        valid_freq = model_properties['valid_freq']

    model_filepath = '%s/model_%d/model_%d.h5' % (args['model_dir'], model_number, model_number)
    history_filepath = '%s/model_%d/model_%d_history.csv' % (args['model_dir'], model_number, model_number)

    ### Training dataset ###
    train_files_obj = fm.DataFileLoader(args['era5_tf_indirs'][0], data_file_type='era5-tensorflow')
    train_files_obj.training_years = training_years
    train_files_obj.pair_with_fronts(args['era5_tf_indirs'][0], front_types=front_types)
    training_inputs = train_files_obj.data_files_training
    training_labels = train_files_obj.front_files_training

    # Shuffle monthly data lazily
    if args['shuffle'] == 'lazy':
        training_files = list(zip(training_inputs, training_labels))
        np.random.shuffle(training_files)
        training_inputs, training_labels = zip(*training_files)

    training_dataset = data_utils.combine_datasets(training_inputs, training_labels)
    print(f"Images in training dataset: {len(training_dataset):,}")

    """
    If the patience argument is not explicitly provided, derive it from the size of the training dataset along with the
    batch size and number of steps per epoch.
    """
    if args['patience'] is None:
        patience = int(len(training_dataset) / (train_batch_size * train_steps)) + 1
        print("Using patience value of %d epochs for early stopping" % patience)
    else:
        patience = args['patience']

    # Shuffle the entire training dataset
    if args['shuffle'] == 'full':
        training_buffer_size = np.min([len(training_dataset), settings.MAX_TRAIN_BUFFER_SIZE])
        training_dataset = training_dataset.shuffle(buffer_size=training_buffer_size)

    training_dataset = training_dataset.batch(train_batch_size, drop_remainder=True, num_parallel_calls=args['num_parallel_calls'])
    training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE)

    ### Validation dataset ###
    valid_files_obj = fm.DataFileLoader(args['era5_tf_indirs'][1], data_file_type='era5-tensorflow')
    valid_files_obj.validation_years = validation_years
    valid_files_obj.pair_with_fronts(args['era5_tf_indirs'][1], front_types=front_types)
    validation_inputs = valid_files_obj.data_files_validation
    validation_labels = valid_files_obj.front_files_validation
    validation_dataset = data_utils.combine_datasets(validation_inputs, validation_labels)
    print(f"Images in validation dataset: {len(validation_dataset):,}")
    validation_dataset = validation_dataset.batch(valid_batch_size, drop_remainder=True, num_parallel_calls=args['num_parallel_calls'])
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    # Set the lat/lon dimensions to have a None shape so images of any sized can be passed into the U-Net
    input_shape = list(training_dataset.take(0).element_spec[0].shape[1:])
    for i in range(2):
        input_shape[i] = None
    input_shape = tuple(input_shape)

    with tf.distribute.MirroredStrategy().scope():

        if not args['retrain']:
            model = unet_model(input_shape, num_classes, **unet_kwargs)
            loss_function = getattr(custom_losses, args['loss'][0])(**loss_args)
            metric_function = getattr(custom_metrics, args['metric'][0])(**metric_args)
            optimizer = getattr(tf.keras.optimizers, args['optimizer'][0])(**optimizer_args)
            model.compile(loss=loss_function, optimizer=optimizer, metrics=metric_function)
        else:
            model = fm.load_model(args['model_number'], args['model_dir'])

    model.summary()

    if not args['retrain']:

        model_properties = {key: model_properties[key] for key in sorted(model_properties.keys())}  # Sort model properties dictionary alphabetically

        if not os.path.isdir('%s/model_%d' % (args['model_dir'], model_number)):
            os.mkdir('%s/model_%d' % (args['model_dir'], model_number))  # Make folder for model
            os.mkdir('%s/model_%d/maps' % (args['model_dir'], model_number))  # Make folder for model predicton maps
            os.mkdir('%s/model_%d/probabilities' % (args['model_dir'], model_number))  # Make folder for prediction data files
            os.mkdir('%s/model_%d/statistics' % (args['model_dir'], model_number))  # Make folder for statistics data files
        elif not args['override_directory_check']:
            raise OSError('%s/model_%d already exists. If model %d still needs to be created and trained, run this script '
                          'again with the --override_directory_check flag.' % (args['model_dir'], model_number, model_number))
        elif os.path.isfile(model_filepath):
            raise OSError('model %d already exists at %s. Choose a different model number and try again.' % (model_number, model_filepath))

        with open('%s/model_%d/model_%d_properties.pkl' % (args['model_dir'], model_number, model_number), 'wb') as f:
            pickle.dump(model_properties, f)

        with open('%s/model_%d/model_%d_properties.txt' % (args['model_dir'], model_number, model_number), 'w') as f:
            for key in model_properties.keys():
                f.write(f"{key}: {model_properties[key]}\n")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, save_freq='epoch')  # ModelCheckpoint: saves model at a specified interval
    early_stopping = EarlyStopping('val_loss', patience=patience, verbose=1)  # EarlyStopping: stops training early if the validation loss does not improve after a specified number of epochs (patience)
    history_logger = CSVLogger(history_filepath, separator=",", append=True)  # Saves loss and metric data every epoch

    model.fit(training_dataset.repeat(), validation_data=validation_dataset, validation_freq=valid_freq, epochs=args['epochs'],
        steps_per_epoch=train_steps, validation_steps=valid_steps, callbacks=[early_stopping, checkpoint, history_logger],
        verbose=args['verbose'])
