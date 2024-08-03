"""
Function that trains a new U-Net model.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.8.3
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
from utils import misc, data_utils
import wandb


class ArgumentParser(argparse.ArgumentParser):
    """ Custom argument parser class """
    def convert_arg_line_to_args(self, arg_line):
        """ Allow multiple arguments to be passed on one line if calling a text file with arguments (e.g., --arg 2 5 6) """
        return arg_line.split()


if __name__ == "__main__":
    parser = ArgumentParser(fromfile_prefix_chars='@')

    ### WandB ###
    parser.add_argument('--project', type=str, help="WandB project that will be used to store model training data.")
    parser.add_argument('--log_freq', type=int, default=1, help="WandB loss/metric logging frequency in epochs.")
    parser.add_argument('--upload_model', action='store_true', help="Upload model checkpoints to WandB.")
    parser.add_argument('--key', type=str, help="WandB API key.")
    parser.add_argument('--name', type=str,
        help="WandB name for the current model run. If no name is specified, it will default to the model number (e.g. model_129482).")

    ### General arguments ###
    parser.add_argument('--model_dir', type=str, required=True, help='Directory where the models are or will be saved to.')
    parser.add_argument('--model_number', type=int,
        help='Number that the model will be assigned. If no argument is passed, a number will be automatically assigned based '
             'on the current date and time.')
    parser.add_argument('--tf_indirs', type=str, required=True, nargs='+',
        help='Directories for the TensorFlow datasets. One or two paths can be passed. If only one path is passed, then the '
             'training and validation datasets will be pulled from this path. If two paths are passed, the training dataset '
             'will be pulled from the first path and the validation dataset from the second.')
    parser.add_argument('--epochs', type=int, required=True, help='Maximum number of epochs for model training.')
    parser.add_argument('--patience', type=int,
        help='Patience for EarlyStopping callback. If this argument is not provided, it will be set according to the size '
             'of the training dataset (images in training set divided by the product of the batch size and steps).')
    parser.add_argument('--verbose', type=int, default=2,
        help='Model.fit verbose. Unless you want a text file that is several hundred megabytes in size and takes 10 years '
             'to scroll through, I suggest you leave this at 2.')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 2**31 - 1),
        help="Seed for the random number generators. If a model is being retrained with the --retrain flag, this argument "
             "will be overriden by the previous seed used to train that model.")

    ### GPU and hardware arguments ###
    parser.add_argument('--gpu_device', type=int, nargs='+', help='GPU device numbers.')
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth for GPUs')
    parser.add_argument('--num_parallel_calls', type=int, default=4,
        help='Number of parallel calls for retrieving batches for the training and validation datasets.')
    parser.add_argument('--buffer_size', type=int, 
        help="Maximum buffer size used when shuffling the training dataset. By default, the entire training dataset will "
             "be shuffled, and the buffer size is equal to the number of images in the training dataset.")
    parser.add_argument('--cache', type=str,
        help="Directory where the datasets will be cached for training. Passing 'RAM' or an empty string will cache the "
             "datasets directly to RAM.")
    parser.add_argument('--disable_tensorfloat32', action='store_true', help='Disable TensorFloat32 execution.')

    ### Hyperparameters ###
    parser.add_argument('--learning_rate', type=float,
        help='Learning rate for U-Net optimizer. If left as None, then the default optimizer learning rate will be used.')
    parser.add_argument('--batch_size', type=int, required=True, nargs='+',
        help='Batch sizes for the U-Net. Up to 2 arguments can be passed. If 1 argument is passed, the value will be both '
             'the training and validation batch sizes. If 2 arguments are passed, the first and second arguments will be '
             'the training and validation batch sizes, respectively.')
    parser.add_argument('--steps', type=int, nargs='+',
        help='Number of steps for each epoch. Up to 2 arguments can be passed. If 1 argument is passed, the value will only '
             'be applied to the number of steps per epoch, and the number of validation steps will be calculated by tensorflow '
             'such that the entire validation dataset is passed into the model during validation. If 2 arguments are passed, '
             'then the arguments are the number of steps in training and validation. If no arguments are passed, then the '
             'number of steps in both training and validation will be calculated by tensorflow.')
    parser.add_argument('--valid_freq', type=int, default=1, help='How many epochs to complete before validation.')

    ### U-Net arguments ###
    parser.add_argument('--model_type', type=str,
        help='Model type. Options are: unet, unet_ensemble, unet_plus, unet_2plus, unet_3plus, attention_unet.')
    parser.add_argument('--activation', type=str,
        help='Activation function to use in the model. Refer to utils.unet_utils.choose_activation_layer to see all available '
             'activation functions.')
    parser.add_argument('--batch_normalization', action='store_true',
        help='Use batch normalization in the model. This will place batch normalization layers after each convolution layer.')
    parser.add_argument('--deep_supervision', action='store_true',
        help="Use deep supervision in the model. Deep supervision creates side outputs from the bottom encoder node and each decoder node.")
    parser.add_argument('--filter_num', type=int, nargs='+',
        help='Number of filters in each level of the U-Net. The number of arguments passed to --filter_num must be equal to the '
             'value passed to --levels.')
    parser.add_argument('--filter_num_aggregate', type=int,
        help='Number of filters in aggregated feature maps. This argument is only used in the U-Net 3+.')
    parser.add_argument('--filter_num_skip', type=int, help='Number of filters in full-scale skip connections in the U-Net 3+.')
    parser.add_argument('--first_encoder_connections', action='store_true', help='Enable first encoder connections in the U-Net 3+.')
    parser.add_argument('--kernel_size', type=int, nargs='+',
        help="Size of the convolution kernels. One integer can be passed to make the kernel dimensions have equal length (e.g. "
             "passing 3 has the same effect as passing 3 3 3 for 3-dimensional kernels.)")
    parser.add_argument('--levels', type=int, help="Number of levels in the model, also known as the 'depth' of the model.")
    parser.add_argument('--loss', type=str, nargs='+',
        help="Loss function for the U-Net (arg 1), with keyword arguments (arg 2). Keyword arguments must be passed as a "
             "string in the second argument. See 'utils.misc.string_arg_to_dict' for more details. Raises a ValueError if "
             "more than 2 arguments are passed.")
    parser.add_argument('--metric', type=str, nargs='+',
        help="Metric for evaluating the U-Net during training (arg 1), with keyword arguments (arg 2). Keyword arguments "
             "must be passed as a string in the second argument. See 'utils.misc.string_arg_to_dict' for more details. Raises "
             "a ValueError if more than 2 arguments are passed.")
    parser.add_argument('--modules_per_node', type=int, default=5,
        help="Number of convolution modules in each node. A convolution module consists of a convolution layer followed by "
             "an optional batch normalization layer and an activation layer. (e.g. Conv3D -> BatchNormalization -> PReLU; Conv3D -> PReLU)")
    parser.add_argument('--optimizer', type=str, nargs='+', default=['Adam', ],
        help="Optimizer to use during the training process (arg 1), with keyword arguments (arg 2). Keyword arguments "
             "must be passed as a string in the second argument. See 'utils.misc.string_arg_to_dict' for more details. Raises "
             "a ValueError if more than 2 arguments are passed.")
    parser.add_argument('--padding', type=str, default='same',
        help="Padding to use in the convolution layers. If 'same', then zero-padding will be added to the inputs such that the outputs "
             "of the layers will be the same shape as the inputs. If 'valid', no padding will be applied to the layers' inputs.")
    parser.add_argument('--pool_size', type=int, nargs='+',
        help="Pool size for the max pooling layers. One integer can be passed to make the pooling dimensions have equal length "
             "(e.g. passing 2 has the same effect as passing 2 2 2 for 3-dimensional max pooling.)")
    parser.add_argument('--upsample_size', type=int, nargs='+',
        help="Upsampling factors for the up-sampling layers. One integer can be passed to make the factors have equal size "
             "(e.g. passing 2 has the same effect as passing 2 2 2 for 3-dimensional up-sampling.)")
    parser.add_argument('--use_bias', action='store_true',
        help="Use bias parameters in the convolution layers.")

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

    ### Debug ###
    parser.add_argument('--no_train', action='store_true',
        help="Do not train the model. This argument will allow everything in the script to run as normal but will not start "
             "the training process. In addition, no directory for the model will be created and WandB will not be initialized. "
             "This argument is mainly meant for debugging purposes as well as being able to see the number of images in "
             "the training and validation datasets without starting the training process.")

    parser.add_argument('--override_directory_check', action='store_true',
        help="Override the OSError caused by creating a new model directory that already exists. Normally, if the script "
             "crashes before or during the training of a new model, an OSError will be returned if the script is immediately "
             "ran again with the same model number as the model directory already exists. This is an intentional fail-safe "
             "designed to prevent models that already exist from being overwritten. Passing this boolean flag disables the "
             "fail-safe and can be useful if the script is being ran on a workload manager (e.g. SLURM) where jobs can fail "
             "and then be immediately requeued and ran again.")

    args = vars(parser.parse_args())

    # Set the random seed. After a model is trained, the same seed will be used in subsequent retraining sessions for the same model.
    seed = pd.read_pickle('%s/model_%d/model_%d_properties.pkl' % (args['model_dir'], args["model_number"], args["model_number"]))["seed"] if args["retrain"] else args["seed"]
    tf.keras.utils.set_random_seed(seed)

    assert len(args['tf_indirs']) < 3, "Only 1 or 2 paths can be passed into --tf_indirs, received %d paths" % len(args['tf_indirs'])

    if len(args['tf_indirs']) == 1:
        args['tf_indirs'].append(args['tf_indirs'][0])

    train_dataset_properties = pd.read_pickle('%s/dataset_properties.pkl' % args['tf_indirs'][0])
    valid_dataset_properties = pd.read_pickle('%s/dataset_properties.pkl' % args['tf_indirs'][1])

    if args['shuffle'] != 'lazy' and args['shuffle'] != 'full':
        raise ValueError("Unrecognized shuffling method: %s. Valid methods are 'lazy' or 'full'" % args['shuffle'])

    # Check arguments that can only have a maximum length of 2
    for arg in ['loss', 'metric', 'optimizer', 'activity_regularizer', 'bias_constraint', 'bias_regularizer', 'kernel_constraint',
                'kernel_regularizer', 'batch_size', 'steps']:
        if args[arg] is None:  # need this line in here because 'steps' can be None
            continue
        elif len(args[arg]) > 2:
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

        # Allow for memory growth on the GPU. This will only use the GPU memory that is required rather than allocating all the GPU's memory.
        if args['memory_growth']:
            tf.config.experimental.set_memory_growth(device=[gpu for gpu in gpus][0], enable=True)

    else:
        print('WARNING: No GPUs found, all computations will be performed on CPUs.')
        tf.config.set_visible_devices([], 'GPU')

    """ 
    Verify that the training and validation datasets have the same front types, variables, pressure levels, number of 
        dimensions, and normalization parameters.
    """
    if args["tf_indirs"][0] != args["tf_indirs"][1]:
        assert train_dataset_properties['front_types'] == valid_dataset_properties['front_types'], \
            (f"The front types in the training and validation datasets must be the same! Received {train_dataset_properties['front_types']} "
             f"for training, {valid_dataset_properties['front_types']} for validation.")
        assert train_dataset_properties['variables'] == valid_dataset_properties['variables'], \
            (f"The variables in the training and validation datasets must be the same! Received {train_dataset_properties['variables']} "
             f"for training, {valid_dataset_properties['variables']} for validation.")
        assert train_dataset_properties['pressure_levels'] == valid_dataset_properties['pressure_levels'], \
            (f"The pressure levels in the training and validation datasets must be the same! Received {train_dataset_properties['pressure_levels']} "
             f"for training, {valid_dataset_properties['pressure_levels']} for validation.")
        assert all(train_dataset_properties['num_dims'][num] == valid_dataset_properties['num_dims'][num] for num in range(2)), \
            (f"The number of dimensions for the inputs and targets in the training and validation datasets must be the same! Received {train_dataset_properties['num_dims']} "
             f"for training, {valid_dataset_properties['num_dims']} for validation")
        assert train_dataset_properties['normalization_parameters'] == valid_dataset_properties['normalization_parameters'], \
            "Normalization parameters for the training and validation datasets must be the same!"

    front_types = train_dataset_properties['front_types']
    variables = train_dataset_properties['variables']
    pressure_levels = train_dataset_properties['pressure_levels']
    image_size = train_dataset_properties['image_size']
    num_dims = train_dataset_properties['num_dims']

    if not args['retrain']:

        all_years = np.arange(2007, 2022.1, 1)

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

        if len(training_years) + len(validation_years) > 15:
            raise ValueError("No testing years are available: the total number of training and validation years cannot be greater than 15")

        test_years = [year for year in all_years if year not in training_years + validation_years]

        # If no model number was provided, select a number based on the current date and time. This number changes once per minute.
        args["model_number"] = int(datetime.datetime.utcnow().timestamp() % 1e8 / 60) if args['model_number'] is None else args['model_number']

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
                    'first_encoder_connections', 'valid_freq', 'optimizer', 'seed']:
            model_properties[arg] = args[arg]
        model_properties['activation'] = model_properties['activation'].lower()

        # Place local variables into the model properties dictionary
        for arg in ['loss_args', 'metric_args', 'image_size', 'training_years', 'validation_years', 'test_years']:
            model_properties[arg] = locals()[arg]

        # If using 3D inputs and 2D targets, squeeze out the vertical dimension of the model (index 3)
        squeeze_axes = 3 if num_dims[0] == 3 and num_dims[1] == 2 else None

        unet_model = getattr(models, args['model_type'])
        unet_model_args = unet_model.__code__.co_varnames[:unet_model.__code__.co_argcount]  # pull argument names from unet function

        ### Arguments for the function used to build the U-Net ###
        unet_kwargs = {arg: args[arg] for arg in ['pool_size', 'upsample_size', 'levels', 'filter_num', 'kernel_size', 'modules_per_node',
            'activation', 'batch_normalization', 'padding', 'use_bias', 'bias_initializer', 'kernel_initializer', 'first_encoder_connections',
            'deep_supervision'] if arg in unet_model_args}
        unet_kwargs['squeeze_axes'] = squeeze_axes
        unet_kwargs['activity_regularizer'] = getattr(tf.keras.regularizers, args['activity_regularizer'][0])(**activity_regularizer_args) if args['activity_regularizer'][0] is not None else None
        unet_kwargs['bias_constraint'] = getattr(tf.keras.constraints, args['bias_constraint'][0])(**bias_constraint_args) if args['bias_constraint'][0] is not None else None
        unet_kwargs['kernel_constraint'] = getattr(tf.keras.constraints, args['kernel_constraint'][0])(**kernel_constraint_args) if args['kernel_constraint'][0] is not None else None
        unet_kwargs['bias_regularizer'] = getattr(tf.keras.regularizers, args['bias_regularizer'][0])(**bias_regularizer_args) if args['bias_regularizer'][0] is not None else None
        unet_kwargs['kernel_regularizer'] = getattr(tf.keras.regularizers, args['kernel_regularizer'][0])(**kernel_regularizer_args) if args['kernel_regularizer'][0] is not None else None

        print("Training years:", training_years)
        print("Validation years:", validation_years)
        print("Test years:", test_years)

    else:

        model_properties = pd.read_pickle('%s/model_%d/model_%d_properties.pkl' % (args['model_dir'], args['model_number'], args['model_number']))
        front_types = model_properties['front_types']

        training_years = model_properties['training_years']
        validation_years = model_properties['validation_years']
        test_years = model_properties['test_years']

        train_batch_size, valid_batch_size = model_properties['batch_sizes']
        train_steps, valid_steps = model_properties['steps_per_epoch']
        valid_freq = model_properties['valid_freq']

    model_filepath = '%s/model_%d/model_%d.h5' % (args['model_dir'], args['model_number'], args['model_number'])  # filepath for the actual model (.h5 file)
    history_filepath = '%s/model_%d/model_%d_history.csv' % (args['model_dir'], args['model_number'], args['model_number'])  # path of the CSV file containing loss and metric statistics

    if train_dataset_properties["domain"] in ["conus", "full"]:
        train_data_source = "era5"
    elif train_dataset_properties["domain"] == "global":
        train_data_source = "gfs"
    else:
        train_data_source = train_dataset_properties["domain"]

    train_batch_size = args['batch_size'][0]
    valid_batch_size = args['batch_size'][0] if len(args['batch_size']) == 1 else args['batch_size'][1]
    model_properties['batch_sizes'] = [train_batch_size, valid_batch_size]

    ### Training dataset ###
    try:
        train_files_obj = fm.DataFileLoader(args['tf_indirs'][0], data_file_type='%s-tensorflow' % train_data_source)
        train_files_obj.training_years = training_years
        train_files_obj.pair_with_fronts(args['tf_indirs'][0])
    except IndexError:
        train_files_obj = fm.DataFileLoader(args['tf_indirs'][0], data_file_type='%s-tensorflow' % train_data_source)
        train_files_obj.training_years = training_years
        train_files_obj.pair_with_fronts(args['tf_indirs'][0], underscore_skips=len(front_types))
    training_inputs = train_files_obj.data_files_training
    training_labels = train_files_obj.front_files_training

    # Shuffle monthly data lazily
    if args['shuffle'] == 'lazy':
        training_files = list(zip(training_inputs, training_labels))
        np.random.shuffle(training_files)
        training_inputs, training_labels = zip(*training_files)

    training_dataset = data_utils.combine_datasets(training_inputs, training_labels)
    images_in_training_dataset = len(training_dataset)
    print(f"Images in training dataset: {images_in_training_dataset:,}")

    # Shuffle the entire training dataset
    if args['shuffle'] == 'full':
        training_buffer_size = args["buffer_size"] if args["buffer_size"] is not None else images_in_training_dataset
        training_dataset = training_dataset.shuffle(buffer_size=training_buffer_size)

    ### Cache the training dataset ###
    if args["cache"] is not None:
        if args["cache"] == "" or args["cache"] == "RAM":
            training_dataset = training_dataset.cache()  # cache to RAM
        else:
            training_dataset = training_dataset.cache('%s/%d_training' % (args["cache"], args["model_number"]))  # cache to specified directory

    training_dataset = training_dataset.batch(train_batch_size, drop_remainder=True, num_parallel_calls=args['num_parallel_calls'])
    training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE)

    if valid_dataset_properties["domain"] in ["conus", "full"]:
        valid_data_source = "era5"
    elif valid_dataset_properties["domain"] == "global":
        valid_data_source = "gfs"
    else:
        valid_data_source = valid_dataset_properties["domain"]

    ### Validation dataset ###
    try:
        valid_files_obj = fm.DataFileLoader(args['tf_indirs'][1], data_file_type='%s-tensorflow' % valid_data_source)
        valid_files_obj.validation_years = validation_years
        valid_files_obj.pair_with_fronts(args['tf_indirs'][1])
    except IndexError:
        valid_files_obj = fm.DataFileLoader(args['tf_indirs'][1], data_file_type='%s-tensorflow' % valid_data_source)
        valid_files_obj.validation_years = validation_years
        valid_files_obj.pair_with_fronts(args['tf_indirs'][1], underscore_skips=len(front_types))
    validation_inputs = valid_files_obj.data_files_validation
    validation_labels = valid_files_obj.front_files_validation
    validation_dataset = data_utils.combine_datasets(validation_inputs, validation_labels)
    images_in_validation_dataset = len(validation_dataset)
    print(f"Images in validation dataset: {images_in_validation_dataset:,}")

    ### Cache the validation dataset ###
    if args["cache"] is not None:
        if args["cache"] == "" or args["cache"] == "RAM":
            validation_dataset = validation_dataset.cache()  # cache to RAM
        else:
            validation_dataset = validation_dataset.cache('%s/%d_validation' % (args["cache"], args["model_number"]))  # cache to specified directory

    validation_dataset = validation_dataset.batch(valid_batch_size, drop_remainder=True, num_parallel_calls=args['num_parallel_calls'])
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    """
    If the number of training steps is not passed, the number of training steps will be determined by the dataset size
    and the provided batch size. The calculated number of training steps will allow for one complete pass over the training
    dataset during each epoch.
    """
    if args['steps'] is None:
        train_steps = int(images_in_training_dataset/train_batch_size)
        print("Using %d training steps per epoch" % train_steps)
        valid_steps = None
    else:
        train_steps = args['steps'][0]
        valid_steps = None if len(args['steps']) < 2 else args['steps'][1]

    valid_freq = args['valid_freq']
    model_properties['steps_per_epoch'] = [train_steps, valid_steps]

    """
    If the patience argument is not explicitly provided, derive it from the size of the training dataset along with the
    batch size and number of steps per epoch.
    """
    if args['patience'] is None:
        patience = int(images_in_training_dataset / (train_batch_size * train_steps)) + 1
        print("Using patience value of %d epochs for early stopping" % patience)
    else:
        patience = args['patience']

    # Set the lat/lon dimensions to have a None shape so images of any size can be passed into the model
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

            model_properties["loss_parent_string"] = args['loss'][0]
            model_properties["loss_child_string"] = loss_function.function_spec._name
            model_properties["metric_parent_string"] = args['metric'][0]
            model_properties["metric_child_string"] = metric_function.function_spec._name
        else:
            model = fm.load_model(args['model_number'], args['model_dir'])

    model.summary()

    if not args['retrain'] and not args["no_train"]:

        model_properties = {key: model_properties[key] for key in sorted(model_properties.keys())}  # Sort model properties dictionary alphabetically

        if not os.path.isdir('%s/model_%d' % (args['model_dir'], args['model_number'])):
            os.makedirs('%s/model_%d/maps' % (args['model_dir'], args['model_number']))  # Make folder for model predicton maps
            os.mkdir('%s/model_%d/probabilities' % (args['model_dir'], args['model_number']))  # Make folder for prediction data files
            os.mkdir('%s/model_%d/statistics' % (args['model_dir'], args['model_number']))  # Make folder for statistics data files
        elif not args['override_directory_check']:
            raise OSError('%s/model_%d already exists. If model %d still needs to be created and trained, run this script '
                          'again with the --override_directory_check flag.' % (args['model_dir'], args['model_number'], args['model_number']))
        elif os.path.isfile(model_filepath):
            raise OSError('model %d already exists at %s. Choose a different model number and try again.' % (args['model_number'], model_filepath))

        with open('%s/model_%d/model_%d_properties.pkl' % (args['model_dir'], args['model_number'], args['model_number']), 'wb') as f:
            pickle.dump(model_properties, f)

        with open('%s/model_%d/model_%d_properties.txt' % (args['model_dir'], args['model_number'], args['model_number']), 'w') as f:
            for key in model_properties.keys():
                f.write(f"{key}: {model_properties[key]}\n")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, save_freq='epoch')  # ModelCheckpoint: saves model at a specified interval
    early_stopping = EarlyStopping('val_loss', patience=patience, verbose=1)  # EarlyStopping: stops training early if the validation loss does not improve after a specified number of epochs (patience)
    history_logger = CSVLogger(history_filepath, separator=",", append=True)  # Saves loss and metric data every epoch

    callbacks = [early_stopping, checkpoint, history_logger]

    if not args["no_train"]:

        ### Initialize WandB ###
        if args["project"] is not None:

            wandb_init_config = dict(
                {key: model_properties[key] for key in ["activation", "activity_regularizer", "batch_normalization",
                                                        "batch_sizes", "bias_constraint", "bias_initializer",
                                                        "bias_regularizer", "domains", "image_size",
                                                        "kernel_constraint",
                                                        "kernel_initializer", "kernel_regularizer", "kernel_size",
                                                        "learning_rate", "loss_parent_string", "metric_parent_string", "model_number",
                                                        "model_type", "modules_per_node", "optimizer", "padding",
                                                        "steps_per_epoch", "test_years", "training_years", "use_bias",
                                                        "validation_years"]})

            # add keys from dataset_properties dictionary
            for key in ["variables", "pressure_levels", "domain", "timestep_fraction", "image_fraction",
                        "flip_chance_lon", "flip_chance_lat", "front_dilation"]:
                wandb_init_config[key] = model_properties["dataset_properties"][key]

            wandb_init_name = "model_%d" % args['model_number'] if args["name"] is None else args["name"]

            if args["key"] is not None:
                wandb.login(key=args["key"])

            wandb.init(project=args["project"], config=wandb_init_config, name=wandb_init_name)
            callbacks.append(wandb.keras.WandbMetricsLogger(log_freq=args["log_freq"]))

            if args["upload_model"]:
                callbacks.append(wandb.keras.WandbModelCheckpoint("models"))  # upload model checkpoints to wandb

        model.fit(training_dataset.repeat(), validation_data=validation_dataset, validation_freq=valid_freq, epochs=args['epochs'],
            steps_per_epoch=train_steps, validation_steps=valid_steps, callbacks=callbacks, verbose=args['verbose'])

    else:

        print("NOTE: Remove the --no_train argument from the command line to start the training process.")