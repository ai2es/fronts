"""
Functions in this code prepare and organize data files before they are used in model training and validation.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 6/30/2021 12:59 PM CDT
"""

from glob import glob
import numpy as np
import os
import pickle
import argparse


def load_10var_files(pickle_indir):
    """
    Function that loads and returns lists of data files.

    Parameters
    ----------
    pickle_indir: str
        Directory where the created pickle files containing the domain data will be stored.

    Returns
    -------
    front_files: list
        List of all files containing fronts. Includes files with no fronts present in their respective domains.
    variable_files: list
        List of all files containing variable data.
    """
    print("\n=== LOADING FILES ===")
    print("Collecting front object files....", end='')
    front_files = sorted(glob("%s/*/*/*/FrontObjects_*_conus.pkl" % pickle_indir))
    print("done, %d files found" % len(front_files))
    print("Collecting variable data files....", end='')
    variable_files = sorted(glob("%s/*/*/*/SurfaceData_*_conus.pkl" % pickle_indir))
    print("done, %d files found" % len(variable_files))

    return front_files, variable_files


def load_31var_files(pickle_indir):
    """
    Function that loads and returns lists of data files.

    Parameters
    ----------
    pickle_indir: str
        Directory where the created pickle files containing the domain data will be stored.

    Returns
    -------
    front_files: list
        List of all files containing fronts. Includes files with no fronts present in their respective domains.
    variable_files: list
        List of all files containing variable data.
    """
    print("\n=== LOADING FILES ===")
    print("Collecting front object files....", end='')
    front_files = sorted(glob("%s/*/*/*/FrontObjects_*_conus_128.pkl" % pickle_indir))
    print("done, %d files found" % len(front_files))
    print("Collecting 31var data files....", end='')
    variable_files = sorted(glob("%s/*/*/*/Data_31var_*_conus_128.pkl" % pickle_indir))
    print("done, %d files found" % len(variable_files))

    return front_files, variable_files


def load_file_lists():
    """
    Opens files containing lists of filenames for fronts and variable data.

    Returns
    -------
    front_files_list: list
        List of filenames that contain front data.
    variable_files_list: list
        List of filenames that contain variable data.
    """
    with open('all_front_files_list.pkl', 'rb') as f:
        front_files_list = pickle.load(f)
    with open('all_variable_files_list.pkl', 'rb') as f:
        variable_files_list = pickle.load(f)

    return front_files_list, variable_files_list


def load_31var_file_lists():
    """
    Opens files containing lists of filenames for front and variable data.

    Returns
    -------
    front_files_list: list
        List of filenames that contain front data.
    variable_files_list: list
        List of filenames that contain variable data.
    """
    with open('all_31var_front_files_list.pkl', 'rb') as f:
        front_files_list = pickle.load(f)
    with open('all_31var_variable_files_list.pkl', 'rb') as f:
        variable_files_list = pickle.load(f)

    return front_files_list, variable_files_list


def file_removal(front_files, variable_files, pickle_indir):
    """
    Function that moves pickle files which cannot be used to new folders. Pickle files that cannot be used are defined
    as front files without a respective variable file or vice versa. In other words, if there are variable
    files for a specific day but there are no corresponding front files for the same day, the variable files
    will be moved to a new folder.

    Parameters
    ----------
    front_files: list
        List of all files containing fronts. Includes files with no fronts present in their respective domains.
    variable_files: list
        List of all files containing variable data.
    pickle_indir: str
        Directory where the created pickle files containing the domain data will be stored.
    """
    print("\n=== FILE REMOVAL ===")

    extra_files = len(variable_files) - len(front_files)
    total_variable_files = len(variable_files)

    front_files_no_prefix = []
    variable_files_no_prefix = []

    for i in range(0, len(front_files)):
        front_files_no_prefix.append(front_files[i].replace('FrontObjects_', ''))
    # for j in range(0, len(variable_files)):
    #     variable_files_no_prefix.append(variable_files[j].replace('SurfaceData_', ''))
    for j in range(0, len(variable_files)):
        variable_files_no_prefix.append(variable_files[j].replace('Data_31var_', ''))

    front_files_no_prefix = np.array(front_files_no_prefix)
    variable_files_no_prefix = np.array(variable_files_no_prefix)

    k = 0  # Counter for total number of files
    x = 0  # Counter for number of extra files
    print("Removing %d extra files....0/%d" % (extra_files, extra_files), end='\r')
    for filename in variable_files_no_prefix:
        k = k + 1
        if filename not in front_files_no_prefix:
            os.rename(variable_files[k - 1],
                      variable_files[k - 1].replace("%s" % pickle_indir,
                                                    "%s/unusable_pickle_files" % pickle_indir))
            x += 1
        print(
            "Removing %d extra files....%d%s (%d/%d)" % (extra_files, 100 * (k / total_variable_files), '%', x, extra_files),
            end='\r')
    print("Removing %d extra files....done               " % extra_files)


def generate_file_lists(front_files, variable_files):
    """
    Generates lists of files containing front and variable data. Creating separate lists and loading them saves time as
    opposed to just loading all of the files individually each time the code is ran. This function should ONLY be used
    AFTER file_removal has been ran so that the lists are the same length (i.e. each front data file has a corresponding
    variable data file).

    Parameters
    ----------
    front_files: list
        List of files containing front data.
    variable_files: list
        List of files containing variable data.
    """
    print("\n=== GENERATING FILE LISTS ===")
    all_front_files_list = []
    all_variable_files_list = []
    file_count = len(front_files)
    for i in range(0, file_count):
        print("Processing files....%d/%d" % (i, file_count), end='\r')
        all_front_files_list.append(front_files[i])
        all_variable_files_list.append(variable_files[i])
    print("Processing files....done          ")
    print("Saving lists....", end='')
    with open('all_front_files_list.pkl', 'wb') as f:
        pickle.dump(all_front_files_list, f)
    with open('all_variable_files_list.pkl', 'wb') as f:
        pickle.dump(all_variable_files_list, f)
    print("done")


def generate_31var_file_lists(front_files, variable_files):
    """
    Generates lists of files containing variable data. Creating separate lists and loading them saves time as
    opposed to just loading all of the files individually each time the code is ran.

    Parameters
    ----------
    front_files: list
        List of files containing front data.
    variable_files: list
        List of files containing variable data.
    """
    print("\n=== GENERATING 31var FILE LISTS ===")
    all_31var_front_files_list = []
    all_31var_variable_files_list = []
    file_count = len(variable_files)
    for i in range(0, file_count):
        print("Processing files....%d/%d" % (i, file_count), end='\r')
        all_31var_front_files_list.append(front_files[i])
        all_31var_variable_files_list.append(variable_files[i])
    print("Processing files....done          ")
    print("Saving lists....", end='')
    with open('all_31var_front_files_list.pkl', 'wb') as f:
        pickle.dump(all_31var_front_files_list, f)
    with open('all_31var_variable_files_list.pkl', 'wb') as f:
        pickle.dump(all_31var_variable_files_list, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_indir', type=str, required=True, help='Path of pickle files containing front object'
                                                                        ' and variable data.')
    parser.add_argument('--generate_lists', type=str, required=False, help='Generate lists of new files? (True/False)')
    args = parser.parse_args()

    # front_files, variable_files = load_10var_files(args.pickle_indir)
    front_files, variable_files = load_31var_files(args.pickle_indir)
    if len(front_files) != len(variable_files):
        file_removal(front_files, variable_files, args.pickle_indir)
        print("====> NOTE: Extra files have been removed, so a new list of files will now be loaded. <====")
        front_files, variable_files = load_31var_files(args.pickle_indir)
    else:
        print("No files need to be removed.")
    if args.generate_lists == 'True':
        generate_file_lists(front_files, variable_files)
    #    generate_31var_file_lists(front_files, variable_files)
    else:
        print("WARNING: Generation of file lists was not enabled. To generate file lists, pass '--generate_lists' as"
              " 'True'.")
