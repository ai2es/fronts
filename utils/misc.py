"""
Miscellaneous tools.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.3.8
"""


def string_arg_to_dict(arg_str: str):
    """
    Function that converts a string argument into a dictionary. Dictionaries cannot be passed through a command line, so
    this function takes a special string argument and converts it to a dictionary so arguments within a function can be
    explicitly called.

    arg_str: str
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
