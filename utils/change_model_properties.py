"""
Changes values of dictionary keys in a model_properties.pkl file.
This script is mainly used to address bugs in train_model.py, where the dictionaries are created.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 8/12/2023 11:07 PM
"""
import argparse
import pandas as pd
import pickle
from utils.misc import string_arg_to_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='Directory for the model.')
    parser.add_argument('--model_number', type=int, help='Model number.')
    parser.add_argument('--changes', type=str,
        help="Changes to make to the model properties dictionary. See utils.misc.string_arg_to_dict for more information.")
    parser.add_argument('--permission_override', action='store_true',
        help="WARNING: Read the description for this argument CAREFULLY. This is a boolean flag that overrides permission "
             "errors when attempting to modify critical model information. Changing properties that raise a PermissionError "
             "can render a model unusable with this module. ALWAYS create a backup of the model_*_properties.pkl file if "
             "you plan to modify critical model information.")

    args = vars(parser.parse_args())

    model_properties_file = '%s/model_%d/model_%d_properties.pkl' % (args['model_dir'], args['model_number'], args['model_number'])
    model_properties = pd.read_pickle(model_properties_file)

    changes = string_arg_to_dict(args['changes'])

    critical_args = ['dataset_properties', 'normalization_parameters', 'training_years', 'validation_years', 'test_years', 'model_number']
    critical_args_passed = list([arg for arg in critical_args if arg in changes])

    if len(critical_args_passed) > 0:
        if not args['permission_override']:
            raise PermissionError(
                f"The following critical model properties were attempted to be modified: --{', --'.join(critical_args_passed)}. "
                "Changing these properties can render the model properties file to be incompatible with other scripts. "
                "If you would like to modify these properties, pass the --permission_override flag. ALWAYS CREATE A BACKUP "
                "model_*_properties.pkl file before proceeding.")

    for arg in changes:
        model_properties[arg] = changes[arg]

    # Rewrite the human-readable model properties text file
    with open(model_properties_file.replace('.pkl', '.txt'), 'w') as f:
        for key in model_properties.keys():
            f.write(f"{key}: {model_properties[key]}\n")

    # Save the model properties dictionary with the new changes.
    with open(model_properties_file, 'wb') as f:
        pickle.dump(model_properties, f)
