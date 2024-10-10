"""
Script for testing model prediction speed.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.10.10
"""
import argparse
import file_manager as fm
import numpy as np
import pandas as pd
import tensorflow as tf
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='Parent model directory.')
    parser.add_argument('--model_number', type=int, required=True, help='Model number.')
    parser.add_argument('--gpu_device', type=int, nargs='+', help='GPU device numbers.')
    parser.add_argument('--memory_growth', action='store_true', help='Use memory growth for GPUs')
    args = vars(parser.parse_args())

    gpus = tf.config.list_physical_devices(device_type='GPU')  # Find available GPUs
    if len(gpus) > 0:

        print("Number of GPUs available: %d" % len(gpus))

        # Only make the selected GPU(s) visible to TensorFlow
        if args['gpu_device'] is not None:
            tf.config.set_visible_devices(devices=[gpus[gpu] for gpu in args['gpu_device']], device_type='GPU')
            gpus = tf.config.get_visible_devices(device_type='GPU')  # List of selected GPUs
            print("Using %d GPU(s):" % len(gpus), gpus)

        # Allow for memory growth on the GPU. This will only use the GPU memory that is required rather than allocating all the GPU's memory.
        if args['memory_growth']:
            tf.config.experimental.set_memory_growth(device=[gpu for gpu in gpus][0], enable=True)

    else:
        print('WARNING: No GPUs found, all computations will be performed on CPUs.')
        tf.config.set_visible_devices([], 'GPU')
    
    model = fm.load_model(args['model_number'], args['model_dir'])
    model_properties = pd.read_pickle('%s/model_%d/model_%d_properties.pkl' % (args['model_dir'], args['model_number'], args['model_number']))
    
    N_runs = 30
    batch_size = 2
    N_pressure_levels = len(model_properties['dataset_properties']['pressure_levels'])
    N_variables = len(model_properties['dataset_properties']['variables'])
    image_size = [960, 320]

    time_elapsed_arr = []
    
    print("Initializing model")
    init_input_tensor = tf.convert_to_tensor(np.random.rand(batch_size, *image_size, N_pressure_levels, N_variables))
    model.predict(init_input_tensor, verbose=0)
    
    print("Starting benchmark...")
    for run in range(1, N_runs + 1):
        
        input_tensor = tf.convert_to_tensor(np.random.rand(batch_size, *image_size, N_pressure_levels, N_variables))
        start_time = time.time()
        model.predict(input_tensor, verbose=0)
        time_elapsed = time.time() - start_time
        
        print("Run %d/%d: %.3fs" % (run, N_runs, time_elapsed))
        time_elapsed_arr.append(time_elapsed)
    
    avg_time_elapsed = np.mean(np.array(time_elapsed_arr))
    print("Final benchmark: %.3fs, %.3fs/image" % (avg_time_elapsed, avg_time_elapsed / batch_size))