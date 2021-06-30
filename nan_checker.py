"""
Debugging tool used to check for corrupt and/or missing data in variable pickle files.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 6/30/2021 1:05 PM CDT
"""

import pickle
import numpy as np
import file_manager as fm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_indir', type=str, required=True, help='Path of pickle files containing front object'
                                                                        ' and variable data.')
    args = parser.parse_args()

    front_files, variable_files = fm.load_31var_files(args.pickle_indir)
    print(len(variable_files))
    file_count = 0
    for i in range(len(variable_files)):
        with open(variable_files[i], 'rb') as f:
            data = pickle.load(f)
        corrupt = 0
        variable_list = list(data.keys())
        for var in variable_list:
            nanvalues = np.where(np.isnan(data[var].values) == True)
            if len(nanvalues[0]) > 0 or len(nanvalues[1]) > 0:
                print(var)
                corrupt = 1
        if corrupt == 1:
            file_count += 1
            print("Corrupt file found: File #%d, %d corrupt files found" % (i, file_count))
            print(variable_files[i])
            print("\n")
    print(file_count)
