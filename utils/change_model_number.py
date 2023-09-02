"""
Script that changes the number of a model and its data files.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.6.26
"""
import os
from glob import glob
import argparse
import pandas as pd
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='Directory for the models.')
    parser.add_argument('--model_numbers', type=int, nargs=2, help='The original and new model numbers.')
    args = vars(parser.parse_args())

    assert not os.path.isdir('%s/model_%d' % (args['model_dir'], args['model_numbers'][1]))  # make sure the new model number is not already assigned to a model

    ### Change the model number in the model properties dictionary ###
    model_properties_file = '%s/model_%d/model_%d_properties.pkl' % (args['model_dir'], args['model_numbers'][0], args['model_numbers'][0])
    model_properties = pd.read_pickle(model_properties_file)
    model_properties['model_number'] = args['model_numbers'][1]

    with open(model_properties_file, 'wb') as f:
        pickle.dump(model_properties, f)

    os.rename('%s/model_%d' % (args['model_dir'], args['model_numbers'][0]), '%s/model_%d' % (args['model_dir'], args['model_numbers'][1]))  # rename the model number directory
    files_to_rename = list(sorted(glob('%s/model_%d/**/*' % (args['model_dir'], args['model_numbers'][1]), recursive=True)))  # files within the subdirectories to rename

    print("Renaming %d files" % len(files_to_rename))
    for file in files_to_rename:
        os.rename(file, file.replace(str(args['model_numbers'][0]), str(args['model_numbers'][1])))

    print("Successfully changed model number: %d -------> %d" % (args['model_numbers'][0], args['model_numbers'][1]))
