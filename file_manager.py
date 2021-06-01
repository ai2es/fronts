"""
Functions in this code prepare and organize data files before they are used in model training.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 6/1/2021 3:28 PM CDT
"""

from glob import glob
import numpy as np
import os
import pickle
import argparse

def load_conus_files(pickle_indir):
    """
    Function that loads and returns lists of data files.

    Parameters
    ----------
    pickle_indir: str
        Directory where the created pickle files containing the domain data will be stored.

    Returns
    -------
    frontobject_conus_files: list
        List of all files containing fronts. Includes files with no fronts present in their respective domains.
    surfacedata_conus_files: list
        List of all files containing surface data.
    """
    print("\n=== LOADING FILES ===")
    print("Collecting front object files....", end='')
    frontobject_conus_files = sorted(glob("%s/*/*/*/FrontObjects_*_conus.pkl" % pickle_indir))
    print("done, %d files found" % len(frontobject_conus_files))
    print("Collecting surface data files....", end='')
    surfacedata_conus_files = sorted(glob("%s/*/*/*/SurfaceData_*_conus.pkl" % pickle_indir))
    print("done, %d files found" % len(surfacedata_conus_files))

    return frontobject_conus_files, surfacedata_conus_files

def load_file_lists():
    """
    Opens files containing lists of filenames for fronts and surface data.

    Returns
    -------
    front_files_list: list
        List of filenames that contain front data.
    sfcdata_files_list: list
        List of filenames that contain surface data.
    """
    with open('all_front_files_list.pkl', 'rb') as f:
        front_files_list = pickle.load(f)
    with open('all_sfcdata_files_list.pkl', 'rb') as f:
        sfcdata_files_list = pickle.load(f)

    return front_files_list, sfcdata_files_list

def file_removal(frontobject_conus_files, surfacedata_conus_files, pickle_indir):
    """
    Function that moves pickle files which cannot be used to new folders. Pickle files that cannot be used are defined
    as frontobject files without a respective surfacedata file or vice versa. In other words, if there are surfacedata
    files for a specific day but there are no corresponding frontobject files for the same day, the surfacedata files
    will be moved to a new folder.

    Parameters
    ----------
    frontobject_conus_files: list
        List of all files containing fronts. Includes files with no fronts present in their respective domains.
    surfacedata_conus_files: list
        List of all files containing surface data.
    pickle_indir: str
        Directory where the created pickle files containing the domain data will be stored.
    """
    print("\n=== FILE REMOVAL ===")

    extra_files = len(surfacedata_conus_files) - len(frontobject_conus_files)
    total_sfc_files = len(surfacedata_conus_files)

    frontobject_conus_files_no_prefix = []
    surfacedata_conus_files_no_prefix = []

    for i in range(0, len(frontobject_conus_files)):
        frontobject_conus_files_no_prefix.append(frontobject_conus_files[i].replace('FrontObjects_', ''))
    for j in range(0, len(surfacedata_conus_files)):
        surfacedata_conus_files_no_prefix.append(surfacedata_conus_files[j].replace('SurfaceData_', ''))

    frontobject_conus_files_no_prefix = np.array(frontobject_conus_files_no_prefix)
    surfacedata_conus_files_no_prefix = np.array(surfacedata_conus_files_no_prefix)

    k = 0 # Counter for total number of files
    x = 0 # Counter for number of extra files
    print("Removing %d extra files....0/%d" % (extra_files, extra_files), end='\r')
    for filename in surfacedata_conus_files_no_prefix:
        k = k + 1
        if filename not in frontobject_conus_files_no_prefix:
            os.rename(surfacedata_conus_files[k - 1],
                      surfacedata_conus_files[k - 1].replace("%s" % pickle_indir,
                                                             "%s/unusable_pickle_files" % pickle_indir))
            x += 1
        print(
            "Removing %d extra files....%d%s (%d/%d)" % (extra_files, 100 * (k / total_sfc_files), '%', x, extra_files),
            end='\r')
    print("Removing %d extra files....done               " % extra_files)


def generate_file_lists(frontobject_conus_files, surfacedata_conus_files):
    """
    Generates lists of files containing front and surface data. Creating separate lists and loading them saves time as
    opposed to just loading all of the files individually each time the code is ran. This function should ONLY be used
    AFTER file_removal has been ran so that the lists are the same length (i.e. each front data file has a corresponding
    surface data file).

    Parameters
    ----------
    frontobject_conus_files: list
        List of files containing front data.
    surfacedata_conus_files: list
        List of files containing surface data.
    """
    print("\n=== GENERATING FILE LISTS ===")
    all_front_files_list = []
    all_sfcdata_files_list = []
    file_count = len(frontobject_conus_files)
    for i in range(0, file_count):
        print("Processing files....%d/%d" % (i, file_count), end='\r')
        all_front_files_list.append(frontobject_conus_files[i])
        all_sfcdata_files_list.append(surfacedata_conus_files[i])
    print("Processing files....done          ")
    print("Saving lists....", end='')
    with open('all_front_files_list.pkl', 'wb') as f:
        pickle.dump(all_front_files_list, f)
    with open('all_sfcdata_files_list.pkl', 'wb') as f:
        pickle.dump(all_sfcdata_files_list, f)
    print("done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_indir', type=str, required=True, help='Path of pickle files containing front object'
                                                                        ' and surface data.')
    parser.add_argument('--generate_lists', type=str, required=True, help='Generate lists of new files? (True/False)')
    args = parser.parse_args()

    frontobject_conus_files, surfacedata_conus_files = load_conus_files(args.pickle_indir)
    if len(frontobject_conus_files) != len(surfacedata_conus_files):
        file_removal(frontobject_conus_files, surfacedata_conus_files, args.pickle_indir)
        print("====> NOTE: Extra files have been removed, so a new list of files will now be loaded. <====")
        frontobject_conus_files, surfacedata_conus_files = load_conus_files(args.pickle_indir)
    else:
        print("No files need to be removed.")
    if args.generate_lists == 'True':
        generate_file_lists(frontobject_conus_files, surfacedata_conus_files)
    else:
        print("Generation of file lists was not enabled. To generate file lists, set the 'generate_lists' argument to"
              "'True'.")
