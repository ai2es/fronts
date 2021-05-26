from glob import glob
import numpy as np
import pandas as pd
import os
import pickle

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

    k = 0
    x = 0
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
    print("\n=== GENERATING FILE LISTS ===")
    empty_front_files = []
    empty_sfcdata_files = []
    nonempty_front_files = []
    nonempty_sfcdata_files = []
    file_count = len(frontobject_conus_files)
    for i in range(0, file_count):
        print("Processing files....%d/%d" % (i, file_count), end='\r')
        fronts = pd.read_pickle(frontobject_conus_files[i]).identifier.values.flatten()
        identifiers = np.count_nonzero(fronts)
        if identifiers > 0:
            nonempty_front_files.append(frontobject_conus_files[i])
            nonempty_sfcdata_files.append(surfacedata_conus_files[i])
        else:
            empty_front_files.append(frontobject_conus_files[i])
            empty_sfcdata_files.append(surfacedata_conus_files[i])
    print("Processing files....done          ")
    print("Saving lists....", end='')
    with open('empty_2016_front_files.pkl', 'wb') as f:
        pickle.dump(empty_front_files, f)
    with open('empty_2016_sfcdata_files.pkl', 'wb') as f:
        pickle.dump(empty_sfcdata_files, f)
    with open('nonempty_2016_front_files.pkl', 'wb') as f:
        pickle.dump(nonempty_front_files, f)
    with open('nonempty_2016_sfcdata_files.pkl', 'wb') as f:
        pickle.dump(nonempty_sfcdata_files, f)
    print("done")
