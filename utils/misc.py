"""
Miscellaneous tools.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.8.10
"""


def initialize_gpus(devices: int | list[int], memory_growth: bool = False):
    """
    Initialize GPU devices.
    
    devices: int or list of ints
        GPU device indices.
    memory_growth: bool
        Use memory growth on the GPU(s).
    """
    
    if isinstance(devices, int):
        devices = [devices, ]
    
    # placing tensorflow import in local scope to prevent loading the library for all functions
    import tensorflow as tf
    
    # configure GPU devices
    gpus = tf.config.list_physical_devices(device_type='GPU')
    tf.config.set_visible_devices(devices=[gpus[gpu] for gpu in devices], device_type='GPU')
    
    # Allow for memory growth on the GPU. This will only use the GPU memory that is required rather than allocating all the GPU's memory.
    if memory_growth:
        tf.config.experimental.set_memory_growth(device=[gpus[gpu] for gpu in devices][0], enable=True)


def string_arg_to_dict(arg_str: str):
    """
    Function that converts a string argument into a dictionary. Dictionaries cannot be passed through a command line, so
    this function takes a special string argument and converts it to a dictionary so arguments within a function can be
    explicitly called.
    
    Parameters
    ----------
    arg_str: str
    
    Returns
    -------
    arg_dict: dict
    """

    arg_str = arg_str.replace(' ', '')  # Remove all spaces from the string.
    args = arg_str.split(',')
    arg_dict = {}  # dictionary that will contain the final arguments and values

    for arg in args:
        arg_name, arg_val_str = arg.split('=')

        arg_is_tuple = '(' in arg_val_str and ')' in arg_val_str
        arg_is_list = '[' in arg_val_str and ']' in arg_val_str

        # if argument value is a list or tuple
        if arg_is_tuple or arg_is_list:
            list_vals = arg_val_str.replace('[', '').replace(']', '').replace('(', '').replace(')', '').split('*')
            new_list_vals = []
            for val in list_vals:
                if '.' in val:
                    new_list_vals.append(float(val))
                elif val == 'True':
                    new_list_vals.append(True)
                elif val == 'False':
                    new_list_vals.append(False)
                else:
                    try:
                        new_list_vals.append(int(val))
                    except ValueError:
                        new_list_vals.append(val)

            if arg_is_tuple:
                new_list_vals = tuple(new_list_vals)

            arg_dict[arg_name] = new_list_vals

        else:
            if '.' in arg_val_str:
                arg_val = float(arg_val_str)
            elif arg_val_str == 'True':
                arg_val = True
            elif arg_val_str == 'False':
                arg_val = False
            else:
                try:
                    arg_val = int(arg_val_str)
                except ValueError:
                    arg_val = arg_val_str

            arg_dict[arg_name] = arg_val

    return arg_dict
