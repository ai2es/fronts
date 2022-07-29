"""
Functions in this code manage data files and directories.
    1. Create subdirectories organized by year, month, day, and/or hour to organize data files.
    2. Compress a large set of files into a tarfile.
    3. Extract the contents of an existing tarfile.
    4. Create new ERA5, GDAS, GFS, and frontal object pickle files across a smaller domain.
    5. Delete a large group of files using a common string in the filenames.
    6. Load ERA5, ERA5/fronts, GDAS, GDAS/fronts, GFS, GFS/fronts, or frontal object pickle files.
    7. Load a saved tensorflow model.

TODO:
    * Add examples to the following functions:
        - compress_files
        - create_subdirectories
        - delete_grouped_files
        - extract_tarfile
        - load_model

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 7/28/2022 11:46 PM CDT
"""

from glob import glob
import tarfile
import pickle
import argparse
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model as lm
import custom_losses
import itertools
from errors import check_arguments
from utils import data_utils


def compress_files(main_dir, glob_file_string, tar_filename, remove_files=False, status_printout=True):
    """
    Compress files into a tar file.

    Parameters
    ----------
    main_dir: str
        - Main directory where the files are located and where the tar file will be saved.
    glob_file_string: str
        - String of the names of the files to compress.
    tar_filename:
        - Name of the compressed tar file that will be made.
    remove_files: bool
        - Setting this to true will remove the files after they have been compressed to a tar file.
    status_printout: bool
        - Setting this to true will provide printouts of the status of the compression.
    """
    uncompressed_size = 0  # MB

    files = list(sorted(glob(f"{main_dir}/{glob_file_string}")))
    if len(files) == 0:
        raise OSError("No files found")
    else:
        print(f"{len(files)} files found")

    num_files = len(files)

    with tarfile.open(f"{main_dir}/{tar_filename}.tar.gz", "w:gz") as tarF:
        for file in range(num_files):
            tarF.add(files[file], arcname=files[file].replace(main_dir, ''))
            tarF_size = os.path.getsize(f"{main_dir}/{tar_filename}.tar.gz")/1e6  # MB
            uncompressed_size += os.path.getsize(files[file])/1e6
            if status_printout:
                print(f'({file+1}/{num_files})      {uncompressed_size:,.2f} MB ---> {tarF_size:,.2f} MB ({100*(1-(tarF_size/uncompressed_size)):.1f}% compression ratio)', end='\r')

    print(f"Successfully compressed {len(files)} files: ",
          f'{uncompressed_size:,.2f} MB ---> {tarF_size:,.2f} MB ({100*(1-(tarF_size/uncompressed_size)):.1f}% compression ratio)')

    if remove_files:
        for file in files:
            os.remove(file)
        print(f"Successfully deleted {len(files)} files")


def create_new_domain_files(current_domain_files, domain, new_domain, new_extent):
    """
    Function that takes files encompassing the specific domain and creates new files with a smaller domain.

    Parameters
    ----------
    current_domain_files: list
        - List of pickle files over the original domain.
    domain: str
        - Domain of the files from which the new files will be made.
    new_domain: str
        - Name of the new domain.
    new_extent: Iterable object with 4 ints
        - 4 values for the new extent: min_lon, max_lon, min_lat, max_lat
    """

    print("Create new files for the following domain:", new_domain)

    for file in current_domain_files:
        full_dataset = pd.read_pickle(file)
        new_domain_dataset = full_dataset.sel(longitude=slice(new_extent[0], new_extent[1]), latitude=slice(new_extent[3], new_extent[2]))
        new_domain_file = file.replace(domain, new_domain)
        with open(new_domain_file, "wb") as f:
            pickle.dump(new_domain_dataset, f)


def create_subdirectories(main_dir, hour_dirs=False):
    """
    Creates subdirectories for files sorted by year, month, day, and hour (hour is optional).

    Parameters
    ----------
    main_dir: str
        - Main directory for the subdirectories.
    hour_dirs: bool
        - Boolean flag that determines whether hourly subdirectories will be generated.
    """

    years = list(np.arange(2006, 2023, 1))

    month_1_days = np.linspace(1, 31, 31)
    month_3_days = np.linspace(1, 31, 31)
    month_4_days = np.linspace(1, 30, 30)
    month_5_days = np.linspace(1, 31, 31)
    month_6_days = np.linspace(1, 30, 30)
    month_7_days = np.linspace(1, 31, 31)
    month_8_days = np.linspace(1, 31, 31)
    month_9_days = np.linspace(1, 30, 30)
    month_10_days = np.linspace(1, 31, 31)
    month_11_days = np.linspace(1, 30, 30)
    month_12_days = np.linspace(1, 31, 31)

    for year in years:
        print(f"Making directories for year {year}")
        if year % 4 == 0:  # Check if the current year is a leap year
            month_2_days = np.linspace(1, 29, 29)
        else:
            month_2_days = np.linspace(1, 28, 28)
        month_days = [month_1_days, month_2_days, month_3_days, month_4_days, month_5_days, month_6_days, month_7_days, month_8_days,
                      month_9_days, month_10_days, month_11_days, month_12_days]
        if not os.path.isdir('%s/%d' % (main_dir,  year)):
            os.mkdir('%s/%d' % (main_dir,  year))
        for month in range(1, 13):
            if not os.path.isdir('%s/%d/%02d' % (main_dir, year, month)):
                os.mkdir('%s/%d/%02d' % (main_dir, year, month))
            for day in month_days[month-1]:
                if not os.path.isdir('%s/%d/%02d/%02d' % (main_dir, year, month, day)):
                    os.mkdir('%s/%d/%02d/%02d' % (main_dir, year, month, day))
                if hour_dirs:
                    for hour in range(0, 24, 3):
                        if not os.path.isdir('%s/%d/%02d/%02d/%02d' % (main_dir, year, month, day, hour)):
                            os.mkdir('%s/%d/%02d/%02d/%02d' % (main_dir, year, month, day, hour))


def delete_grouped_files(main_dir, glob_file_string, num_subdir):
    """
    Deletes grouped files with names matching given strings.

    Parameters
    ----------
    main_dir: str
        - Main directory or directories where the grouped files are located.
    glob_file_string: str
        - String of the names of the files to delete.
    num_subdir: int
        - Number of subdirectory layers in the main directory.
    """

    subdir_string = ''

    for i in range(num_subdir):
        subdir_string += '/*'
    glob_file_string = subdir_string + glob_file_string

    files_to_delete = list(sorted(glob("%s/%s" % (main_dir, glob_file_string))))

    print("Deleting %d files...." % len(files_to_delete), end='')
    for file in files_to_delete:
        os.remove(file)
    print("done")


def extract_tarfile(main_dir, tar_filename):
    """
    Extract all the contents of a tar file.

    main_dir: str
        - Main directory where the tar file is located. This is also where the extracted files will be placed.
    tar_filename: str
        - Name of the compressed tar file.
    """

    with tarfile.open(f"{main_dir}/{tar_filename}", 'r') as tarF:
        tarF.extractall(main_dir)
    print(f"Successfully extracted {main_dir}/{tar_filename}")


def load_era5_pickle_files(era5_pickle_indir, num_variables, domain, dataset=None, test_years=None, validation_years=None, timestep_tuple=None):
    """
    Function that loads and returns a list of ERA5 pickle files.

    Parameters
    ----------
    era5_pickle_indir: str
        - Input directory for the ERA5 pickle files.
    num_variables: int
        - Number of variables in the ERA5 datasets.
    domain: str
        - Domain covered by the ERA5 datasets.
    dataset: str or None
        - Dataset to load. Available options are 'training', 'validation', or 'test'. Leaving this as None will load the file lists without any filtering.
    test_years: list of ints or None
        - Years for the test set (will not be used in training or validation).
    validation_years: list of ints or None
        - Years for the validation set.
    timestep_tuple: tuple with 4 ints
        Timestep of the GDAS/GFS pickle files to load. This can only be passed if dataset='timestep'.

    Returns
    -------
    era5_files_<dataset>: list of strs
        - List of all files containing ERA5 data.

    Examples
    --------
    era5_files = fm.load_era5_pickle_files(era5_pickle_indir, num_variables=60, domain='full')  # Load all ERA5 pickle files containing 60 variables across the full domain
        * add the following parameters to load the TRAINING dataset: dataset='training', test_years=[list of ints], validation_years=[list of ints]
        * add the following parameters to load the VALIDATION dataset: dataset='validation', validation_years=[list of ints]
        * add the following parameters to load the TEST dataset: dataset='test', test_years=[list of ints]
        * add the following parameters to load an ERA5 file for a given timestep: dataset='timestep', timestep_tuple=(year, month, day, hour)
    """
    if dataset is not None:
        print(f"Loading ERA5 files: {dataset} dataset")
    else:
        print("Loading ERA5 files")

    if dataset == 'timestep':
        era5_files = sorted(glob("%s/%d/%02d/%02d/Data_%dvar_%d%02d%02d%02d_%s.pkl" % (era5_pickle_indir, timestep_tuple[0], timestep_tuple[1], timestep_tuple[2],
                                                                                       num_variables, timestep_tuple[0], timestep_tuple[1], timestep_tuple[2], timestep_tuple[3], domain)))

        if len(era5_files) == 0:
            raise OSError("No ERA5 pickle file found")

        print("Loaded ERA5 pickle file for %d-%02d-%02d-%02dz" % (timestep_tuple[0], timestep_tuple[1], timestep_tuple[2], timestep_tuple[3]))

        return era5_files

    else:
        era5_files = sorted(glob("%s/*/*/*/Data_%dvar_*_%s.pkl" % (era5_pickle_indir, num_variables, domain)))

        if domain == 'full':  # Filter out non-synoptic hours
            hours_to_remove = [3, 9, 15, 21]
            for hour in hours_to_remove:
                string = '%02d_' % hour
                era5_files = list(filter(lambda hour: string not in hour, era5_files))  # Remove files that do not contain the given hourly string

        separators = ['/', '\\']  # The subdirectory separator is different depending on your operating system, so we will iterate through both of these separators to make sure we grab the files
        loop_counter = 0  # Counter that increases by 1 if no files are found as we iterate through each of the file separators

        if dataset == 'training':

            if validation_years is None:
                raise ValueError("Validation years must be declared to pull the training set")
            if test_years is None:
                raise ValueError("Test years must be declared to pull the training set")

            era5_files_training = era5_files
            while len(era5_files_training) == len(era5_files) and loop_counter < 2:  # While no ERA5 files in the training dataset are found
                for validation_year in validation_years:
                    validation_year_string = separators[loop_counter] + str(validation_year) + separators[loop_counter]
                    era5_files_training = list(filter(lambda validation_year: validation_year_string not in validation_year, era5_files_training))
                for test_year in test_years:
                    test_year_string = separators[loop_counter] + str(test_year) + separators[loop_counter]
                    era5_files_training = list(filter(lambda test_year: test_year_string not in test_year, era5_files_training))
                loop_counter += 1

            print("ERA5 files in training dataset:", len(era5_files_training))
            return era5_files_training

        elif dataset == 'validation':

            if validation_years is None:
                raise ValueError("Validation years must be declared to pull the validation set")

            era5_files_validation = []
            while len(era5_files_validation) == 0 and loop_counter < 2:  # While no ERA5 files in the validation dataset are found
                for validation_year in validation_years:
                    validation_year_string = separators[loop_counter] + str(validation_year) + separators[loop_counter]
                    era5_files_validation.append(list(filter(lambda validation_year: validation_year_string in validation_year, era5_files)))
                era5_files_validation = list(sorted(itertools.chain(*era5_files_validation)))
                loop_counter += 1

            print(f"ERA5 files in validation dataset {validation_years}:", len(era5_files_validation))
            return era5_files_validation

        elif dataset == 'test':

            if test_years is None:
                raise ValueError("Test years must be declared to pull the test set")

            era5_files_test = []
            while len(era5_files_test) == 0 and loop_counter < 2:  # While no ERA5 files in the testing dataset are found
                for test_year in test_years:
                    test_year_string = separators[loop_counter] + str(test_year) + separators[loop_counter]
                    era5_files_test.append(list(filter(lambda test_year: test_year_string in test_year, era5_files)))
                era5_files_test = list(sorted(itertools.chain(*era5_files_test)))
                loop_counter += 1

            print(f"ERA5 files in testing dataset {test_years}:", len(era5_files_test))
            return era5_files_test

        elif dataset is None:
            print("ERA5 files:", len(era5_files))
            return era5_files

        else:
            raise ValueError(f"{dataset} is not a valid dataset, options are: None, training, validation, test")


def load_fronts_pickle_files(fronts_pickle_indir, domain, dataset=None, test_years=None, validation_years=None, timestep_tuple=None):
    """
    Function that loads and returns front object files with matching dates.

    Parameters
    ----------
    fronts_pickle_indir: str
        - Input directory for the front object pickle files.
    domain: str
        - Domain which the front and variable files cover.
    dataset: str or None
        - Dataset to load. Available options are 'training', 'validation', or 'test'. Leaving this as None will load the file lists without any filtering.
    test_years: list of ints or None
        - Years for the test set (will not be used in training or validation).
    validation_years: list of ints or None
        - Years for the validation set.
    timestep_tuple: tuple with 4 ints
        Timestep of the GDAS pickle files to load. This can only be passed if dataset='timestep'.

    Returns
    -------
    front_files_<dataset>: list of strs
        - List of all files containing fronts. Includes files with no fronts present in their respective domains.

    Examples
    --------
    front_files = fm.load_fronts_pickle_files(front_pickle_indir, domain='full')  # Load all frontal object pickle files
        * add the following parameters to load the TRAINING dataset: dataset='training', test_years=[list of ints], validation_years=[list of ints]
        * add the following parameters to load the VALIDATION dataset: dataset='validation', validation_years=[list of ints]
        * add the following parameters to load the TEST dataset: dataset='test', test_years=[list of ints]
        * add the following parameters to load a frontal object file for a given timestep: dataset='timestep', timestep_tuple=(year, month, day, hour)
    """
    if dataset is not None:
        print(f"Loading fronts files: {dataset} dataset")
    else:
        print("Loading fronts files")

    if dataset == 'timestep':
        front_files = sorted(glob("%s/%d/%02d/%02d/FrontObjects_%d%02d%02d%02d_%s.pkl" % (fronts_pickle_indir, timestep_tuple[0], timestep_tuple[1], timestep_tuple[2],
                                                                                          timestep_tuple[0], timestep_tuple[1], timestep_tuple[2], timestep_tuple[3], domain)))

        if len(front_files) == 0:
            raise OSError("No fronts file found")

        print("Loaded frontal object pickle file for %d-%02d-%02d-%02dz" % (timestep_tuple[0], timestep_tuple[1], timestep_tuple[2], timestep_tuple[3]))

        return front_files

    else:
        front_files = sorted(glob("%s/*/*/*/FrontObjects_*_%s.pkl" % (fronts_pickle_indir, domain)))

        if domain == 'full':  # Filter out non-synoptic hours
            hours_to_remove = [3, 9, 15, 21]
            for hour in hours_to_remove:
                string = '%02d_' % hour
                front_files = list(filter(lambda hour: string not in hour, front_files))

        separators = ['/', '\\']  # The subdirectory separator is different depending on your operating system, so we will iterate through both of these separators to make sure we grab the files
        loop_counter = 0  # Counter that increases by 1 if no files are found as we iterate through each of the file separators

        if dataset == 'training':

            if validation_years is None:
                raise ValueError("Validation years must be declared to pull the training set")
            if test_years is None:
                raise ValueError("Test years must be declared to pull the training set")

            front_files_training = front_files
            while len(front_files_training) == len(front_files) and loop_counter < 2:
                for validation_year in validation_years:
                    validation_year_string = separators[loop_counter] + str(validation_year) + separators[loop_counter]
                    front_files_training = list(filter(lambda validation_year: validation_year_string not in validation_year, front_files_training))
                for test_year in test_years:
                    test_year_string = separators[loop_counter] + str(test_year) + separators[loop_counter]
                    front_files_training = list(filter(lambda test_year: test_year_string not in test_year, front_files_training))
                loop_counter += 1

            print("Front files in training dataset:", len(front_files_training))
            return front_files_training

        elif dataset == 'validation':

            if validation_years is None:
                raise ValueError("Validation years must be declared to pull the validation set")

            front_files_validation = []
            while len(front_files_validation) == 0 and loop_counter < 2:
                for validation_year in validation_years:
                    validation_year_string = separators[loop_counter] + str(validation_year) + separators[loop_counter]
                    front_files_validation.append(list(filter(lambda validation_year: validation_year_string in validation_year, front_files)))
                front_files_validation = list(sorted(itertools.chain(*front_files_validation)))
                loop_counter += 1

            print(f"Front files in validation dataset {validation_years}:", len(front_files_validation))
            return front_files_validation

        elif dataset == 'test':

            if test_years is None:
                raise ValueError("Test years must be declared to pull the test set")

            front_files_test = []
            while len(front_files_test) == 0 and loop_counter < 2:
                for test_year in test_years:
                    test_year_string = separators[loop_counter] + str(test_year) + separators[loop_counter]
                    front_files_test.append(list(filter(lambda test_year: test_year_string in test_year, front_files)))
                front_files_test = list(sorted(itertools.chain(*front_files_test)))
                loop_counter += 1

            print(f"Front files in test dataset {test_years}:", len(front_files_test))
            return front_files_test

        elif dataset is None:
            print(f"Front files:", len(front_files))
            return front_files

        else:
            raise ValueError(f"{dataset} is not a valid dataset, options are: None, training, validation, test")


def load_era5_and_fronts_pickle_files(era5_pickle_indir, fronts_pickle_indir, num_variables, domain, dataset=None, test_years=None, validation_years=None, timestep_tuple=None):
    """
    Function that loads and returns lists of ERA5 pickle files and front object files with matching dates.

    Parameters
    ----------
    era5_pickle_indir: str
        - Input directory for the ERA5 pickle files.
    fronts_pickle_indir: str
        - Input directory for the front object pickle files.
    num_variables: int
        - Number of variables in the variable datasets.
    domain: str
        - Domain which the front and variable files cover.
    dataset: str or None
        - Dataset to load. Available options are 'training', 'validation', or 'test'. Leaving this as None will load the file lists without any filtering.
    test_years: list of ints
        - Years for the test set (will not be used in training or validation).
    validation_years: list of ints
        - Years for the validation set.
    timestep_tuple: tuple with 4 ints
        Timestep of the ERA5 pickle files to load. This can only be passed if dataset='timestep'.

    Returns
    -------
    era5_files_<dataset>: list of strs
        - List of all files containing ERA5 data.
    front_files_<dataset>: list of strs
        - List of all files containing fronts. Includes files with no fronts present in their respective domains.

    Examples
    --------
    era5_files, front_files = fm.load_era5_and_fronts_pickle_files(era5_pickle_indir, front_pickle_indir, num_variables=60, domain='full')  # Load all pairs of ERA5/front files
        * add the following parameters to load the TRAINING dataset: dataset='training', test_years=[list of ints], validation_years=[list of ints]
        * add the following parameters to load the VALIDATION dataset: dataset='validation', validation_years=[list of ints]
        * add the following parameters to load the TEST dataset: dataset='test', test_years=[list of ints]
        * add the following parameters to load a set of ERA5 files for a given timestep: dataset='timestep', timestep_tuple=(year, month, day, hour)
    """
    if dataset is not None:
        print(f"Loading ERA5/fronts {dataset} dataset")
    else:
        print("Loading ERA5/fronts files")

    if timestep_tuple is not None:
        if dataset != 'timestep':
            raise ValueError("dataset must be set to 'timestep' when loading ERA5/fronts files for a specific timestep")
        else:
            front_files = sorted(glob("%s/%d/%02d/%02d/FrontObjects_%d%02d%02d%02d_%s.pkl" % (fronts_pickle_indir, timestep_tuple[0], timestep_tuple[1], timestep_tuple[2],
                                                                                              timestep_tuple[0], timestep_tuple[1], timestep_tuple[2], timestep_tuple[3], domain)))
            era5_files = sorted(glob("%s/%d/%02d/%02d/Data_%dvar_%d%02d%02d%02d_%s.pkl" % (era5_pickle_indir, timestep_tuple[0], timestep_tuple[1], timestep_tuple[2],
                                                                                           num_variables, timestep_tuple[0], timestep_tuple[1], timestep_tuple[2], timestep_tuple[3], domain)))

            assert len(front_files) == 1
            assert len(era5_files) == 1

            print("Successfully loaded ERA5/front file pair for %d-%02d-%02d-%02dz" % (timestep_tuple[0], timestep_tuple[1], timestep_tuple[2], timestep_tuple[3]))

            return era5_files, front_files

    else:

        front_files = sorted(glob("%s/*/*/*/FrontObjects_*_%s.pkl" % (fronts_pickle_indir, domain)))
        era5_files = sorted(glob("%s/*/*/*/Data_%dvar_*_%s.pkl" % (era5_pickle_indir, num_variables, domain)))

        front_files_timestep = []
        era5_files_timestep = []
        front_files_list = []
        era5_files_list = []

        for i in range(len(front_files)):
            front_files_timestep.append(front_files[i].replace('FrontObjects_', ''))
        for j in range(len(era5_files)):
            era5_files_timestep.append(era5_files[j].replace('Data_%dvar_' % num_variables, ''))

        if len(era5_files) > len(front_files):
            total_era5_files = len(era5_files)
            total_front_files = len(front_files)
            for era5_file_no in range(total_era5_files):
                if era5_files_timestep[era5_file_no] in front_files_timestep:
                    era5_files_list.append(era5_files[era5_file_no])
            for front_file_no in range(total_front_files):
                if front_files_timestep[front_file_no] in era5_files_timestep:
                    front_files_list.append(front_files[front_file_no])
        else:
            total_front_files = len(front_files)
            total_era5_files = len(era5_files)
            for front_file_no in range(total_front_files):
                if front_files_timestep[front_file_no] in era5_files_timestep:
                    front_files_list.append(front_files[front_file_no])
            for era5_file_no in range(total_era5_files):
                if era5_files_timestep[era5_file_no] in front_files_timestep:
                    era5_files_list.append(era5_files[era5_file_no])

        front_files, era5_files = front_files_list, era5_files_list

        matches = 0
        mismatches = 0

        # Check the end of the filenames to match up files together
        for i in range(len(front_files_list)):
            if front_files_list[i][-15:] == era5_files_list[i][-15:]:
                matches += 1
            else:
                mismatches += 1

        if mismatches > 0:
            raise ValueError("The dates in the file lists do not match up. Please report this bug to Andrew Justin (andrewjustinwx@gmail.com)")

        if domain == 'full':  # Filter out non-synoptic hours
            hours_to_remove = [3, 9, 15, 21]
            for hour in hours_to_remove:
                string = '%02d_' % hour
                front_files = list(filter(lambda hour: string not in hour, front_files))
                era5_files = list(filter(lambda hour: string not in hour, era5_files))

        separators = ['/', '\\']  # The subdirectory separator is different depending on your operating system, so we will iterate through both of these separators to make sure we grab the files
        loop_counter = 0  # Counter that increases by 1 if no files are found as we iterate through each of the file separators

        if dataset == 'training':

            if validation_years is None:
                raise ValueError("Validation years must be declared to pull the training set")
            if test_years is None:
                raise ValueError("Test years must be declared to pull the training set")

            front_files_training = front_files
            era5_files_training = era5_files
            while len(front_files_training) == len(front_files) and len(era5_files_training) == len(era5_files) and loop_counter < 2:  # While no ERA5/front files in the training dataset are found
                for validation_year in validation_years:
                    validation_year_string = separators[loop_counter] + str(validation_year) + separators[loop_counter]
                    front_files_training = list(filter(lambda validation_year: validation_year_string not in validation_year, front_files_training))
                    era5_files_training = list(filter(lambda validation_year: validation_year_string not in validation_year, era5_files_training))
                for test_year in test_years:
                    test_year_string = separators[loop_counter] + str(test_year) + separators[loop_counter]
                    front_files_training = list(filter(lambda test_year: test_year_string not in test_year, front_files_training))
                    era5_files_training = list(filter(lambda test_year: test_year_string not in test_year, era5_files_training))
                loop_counter += 1

            print("ERA5/fronts file pairs in training dataset:", len(front_files_training))
            return era5_files_training, front_files_training

        elif dataset == 'validation':

            if validation_years is None:
                raise ValueError("Validation years must be declared to pull the validation set")

            front_files_validation = []
            era5_files_validation = []
            while len(front_files_validation) == 0 and len(era5_files_validation) == 0 and loop_counter < 2:  # While no ERA5/front files in the validation dataset are found
                for validation_year in validation_years:
                    validation_year_string = separators[loop_counter] + str(validation_year) + separators[loop_counter]
                    front_files_validation.append(list(filter(lambda validation_year: validation_year_string in validation_year, front_files)))
                    era5_files_validation.append(list(filter(lambda validation_year: validation_year_string in validation_year, era5_files)))
                front_files_validation = list(sorted(itertools.chain(*front_files_validation)))
                era5_files_validation = list(sorted(itertools.chain(*era5_files_validation)))
                loop_counter += 1

            print(f"ERA5/fronts file pairs in validation dataset {validation_years}:", len(front_files_validation))
            return era5_files_validation, front_files_validation

        elif dataset == 'test':

            if test_years is None:
                raise ValueError("Test years must be declared to pull the test set")

            front_files_test = []
            era5_files_test = []
            while len(front_files_test) == 0 and len(era5_files_test) == 0 and loop_counter < 2:  # While no ERA5/front files in the testing dataset are found
                for test_year in test_years:
                    test_year_string = separators[loop_counter] + str(test_year) + separators[loop_counter]
                    front_files_test.append(list(filter(lambda test_year: test_year_string in test_year, front_files)))
                    era5_files_test.append(list(filter(lambda test_year: test_year_string in test_year, era5_files)))
                front_files_test = list(sorted(itertools.chain(*front_files_test)))
                era5_files_test = list(sorted(itertools.chain(*era5_files_test)))
                loop_counter += 1

            print(f"ERA5/fronts file pairs in testing dataset {test_years}:", len(front_files_test))
            return era5_files_test, front_files_test

        elif dataset is None:
            print(f"ERA5/fronts file pairs:", len(front_files))
            return era5_files, front_files

        else:
            raise ValueError(f"{dataset} is not a valid dataset, options are: None, training, validation, test")


def load_gdas_or_gfs_pickle_files(gdas_or_gfs_pickle_indir, domain, gdas_or_gfs='gdas', dataset=None, test_years=None, validation_years=None, timestep_tuple=None, forecast_hour=6):
    """
    Function that loads and returns lists of GDAS or GFS pickle files.

    Parameters
    ----------
    gdas_or_gfs_pickle_indir: str
        Input directory for the GDAS/GFS pickle files.
    domain: str
        Domain covered by the GDAS/GFS pickle files.
    gdas_or_gfs: str
        Can be 'gdas' or 'gfs' (case-insensitive).
    dataset: str or None
        Dataset to load. Available options are 'training', 'validation', 'test', or 'timestep'. Leaving this as None will load the file lists without any filtering.
    test_years: list of ints or None
        Years for the test set (will not be used in training or validation).
    validation_years: list of ints or None
        Years for the validation set.
    timestep_tuple: tuple with 4 ints
        Timestep of the GDAS/GFS pickle files to load. This can only be passed if dataset='timestep'.
    forecast_hour: int
        Forecast hour to load.

    Returns
    -------
    gdas_or_gfs_files_<dataset>: list of strs
        List of all files containing GDAS/GFS data.

    Examples
    --------
    gdas_or_gfs_files = fm.load_gdas_or_gfs_pickle_files(gdas_or_gfs_pickle_indir, domain='full', gdas_or_gfs='gdas')  # Load all GDAS files
        * add the following parameters to load the TRAINING dataset: dataset='training', test_years=[list of ints], validation_years=[list of ints]
        * add the following parameters to load the VALIDATION dataset: dataset='validation', validation_years=[list of ints]
        * add the following parameters to load the TEST dataset: dataset='test', test_years=[list of ints]
        * add the following parameters to load a set of GDAS/GFS files for a given timestep: dataset='timestep', timestep_tuple=(year, month, day, hour)
    """
    if dataset is not None:
        print(f"Loading {gdas_or_gfs.upper()} {dataset} dataset")
    else:
        print(f"Loading {gdas_or_gfs.upper()} files")

    gdas_or_gfs_files_timestep = []
    gdas_or_gfs_timesteps = []  # Timesteps of GDAS/GFS files
    gdas_or_gfs_files_by_timestep = []  # List of GDAS/GFS files with each index representing a group of files with a common timestep
    gdas_or_gfs_files_list = []

    separators = ['/', '\\']  # The subdirectory separator is different depending on your operating system, so we will iterate through both of these separators to make sure we grab the files
    loop_counter = 0  # Counter that increases by 1 if no files are found as we iterate through each of the file separators

    required_pressure_levels = ['_surface_', '_1000_', '_975_', '_950_', '_925_', '_900_', '_850_', '_800_', '_750_', '_700_',
                                '_650_', '_600_', '_550_', '_500_', '_450_', '_400_', '_350_', '_300_', '_250_', '_200_',
                                '_150_', '_100_']

    if timestep_tuple is not None:
        if dataset != 'timestep':
            raise ValueError("dataset must be set to 'timestep' when loading GDAS/GFS and fronts files for a specific timestep")

        gdas_or_gfs_files = sorted(glob("%s/%d/%02d/%02d/%s_*_%d%02d%02d%02d_f%03d_%s.pkl" % (gdas_or_gfs_pickle_indir, timestep_tuple[0], timestep_tuple[1], timestep_tuple[2], gdas_or_gfs.lower(),
                                                                                              timestep_tuple[0], timestep_tuple[1], timestep_tuple[2], timestep_tuple[3], forecast_hour, domain)))

    else:

        gdas_or_gfs_files = sorted(glob("%s/*/*/*/%s_*_f%03d_%s.pkl" % (gdas_or_gfs_pickle_indir, gdas_or_gfs.lower(), forecast_hour, domain)))

    # Total number of files before any filtering is performed
    initial_gdas_or_gfs_files_count = len(gdas_or_gfs_files)

    if dataset == 'training':

        gdas_or_gfs_files_list = gdas_or_gfs_files

        if validation_years is None:
            raise ValueError("Validation years must be declared to pull the training set")
        if test_years is None:
            raise ValueError("Test years must be declared to pull the training set")

        while len(gdas_or_gfs_files) == initial_gdas_or_gfs_files_count and loop_counter < 2:  # While no GDAS/GFS and fronts files in the training dataset are found
            for validation_year in validation_years:
                validation_year_string = separators[loop_counter] + str(validation_year) + separators[loop_counter]
                gdas_or_gfs_files_list = list(filter(lambda validation_year: validation_year_string not in validation_year, gdas_or_gfs_files_list))
            for test_year in test_years:
                test_year_string = separators[loop_counter] + str(test_year) + separators[loop_counter]
                gdas_or_gfs_files_list = list(filter(lambda test_year: test_year_string not in test_year, gdas_or_gfs_files_list))
            loop_counter += 1

    elif dataset == 'validation':

        if validation_years is None:
            raise ValueError("Validation years must be declared to pull the validation set")

        while len(gdas_or_gfs_files) == initial_gdas_or_gfs_files_count and loop_counter < 2:  # While no GDAS/GFS and fronts files in the training dataset are found
            for validation_year in validation_years:
                validation_year_string = separators[loop_counter] + str(validation_year) + separators[loop_counter]
                gdas_or_gfs_files_list.append(list(filter(lambda validation_year: validation_year_string in validation_year, gdas_or_gfs_files)))
            loop_counter += 1

    elif dataset == 'test':

        if test_years is None:
            raise ValueError("Test years must be declared to pull the test set")

        while len(gdas_or_gfs_files) == initial_gdas_or_gfs_files_count and loop_counter < 2:  # While no GDAS/GFS and fronts files in the training dataset are found
            for test_year in test_years:
                test_year_string = separators[loop_counter] + str(test_year) + separators[loop_counter]
                gdas_or_gfs_files_list.append(list(filter(lambda test_year: test_year_string in test_year, gdas_or_gfs_files)))

            loop_counter += 1

    # Flatten new file lists
    gdas_or_gfs_files = list(sorted(itertools.chain(gdas_or_gfs_files_list)))

    # Find all timesteps in the GDAS/GFS files
    for j in range(len(gdas_or_gfs_files)):
        gdas_or_gfs_filename_start_index = gdas_or_gfs_files[j].find('%s_' % gdas_or_gfs.lower())
        gdas_or_gfs_filename_no_pressure_level_index = gdas_or_gfs_files[j][gdas_or_gfs_filename_start_index + 5:].find('_2') + 1
        gdas_or_gfs_timesteps.append(gdas_or_gfs_files[j][gdas_or_gfs_filename_start_index + gdas_or_gfs_filename_no_pressure_level_index + 5:gdas_or_gfs_filename_start_index + gdas_or_gfs_filename_no_pressure_level_index + 15])  # Timestep is the first 10 characters after the pressure level in the filename
    unique_gdas_or_gfs_fronts_timesteps = np.unique(sorted(gdas_or_gfs_timesteps))

    # Find all GDAS/GFS files for each timestep and order them by pressure level
    for timestep in unique_gdas_or_gfs_fronts_timesteps:
        files_for_timestep = sorted(filter(lambda filename: timestep in filename, gdas_or_gfs_files))

        # Move 1000 mb file to the end of the list, then remove its duplicate at the beginning
        files_for_timestep.append(files_for_timestep[0])
        files_for_timestep.pop(0)

        # Move surface file to the end of the list and remove its duplicate
        files_for_timestep.append(files_for_timestep[-2])
        files_for_timestep.pop(-3)

        # Reverse list so that pressure levels are in descending order / increasing with altitude
        files_for_timestep = files_for_timestep[::-1]

        # Check all files for all pressure levels
        pressure_levels_found = 0
        for pressure_level, file in zip(required_pressure_levels, files_for_timestep):
            if pressure_level in file:
                pressure_levels_found += 1
        if pressure_levels_found == 22:
            gdas_or_gfs_files_timestep_temp = []  # Temporary holding folder for GDAS/GFS files before they are added to the list of filenames
            for full_filename in range(22):
                gdas_or_gfs_filename_start_index = files_for_timestep[full_filename].find('%s_' % gdas_or_gfs.lower())
                gdas_or_gfs_filename_no_pressure_level_index = files_for_timestep[full_filename][gdas_or_gfs_filename_start_index + 5:].find('_2') + 1
                gdas_or_gfs_files_timestep_temp.append(files_for_timestep[full_filename][gdas_or_gfs_filename_start_index + gdas_or_gfs_filename_no_pressure_level_index + 5:gdas_or_gfs_filename_start_index + gdas_or_gfs_filename_no_pressure_level_index + 15])
            gdas_or_gfs_files_timestep.append(gdas_or_gfs_files_timestep_temp)
            gdas_or_gfs_files_by_timestep.append(files_for_timestep)

    # Filter out non-synoptic hours if loading files over the full domain
    if domain == 'full':
        hours_to_remove = [3, 9, 15, 21]
        for hour in hours_to_remove:
            string = '%02d_' % hour
            for gdas_or_gfs_file_set in range(len(gdas_or_gfs_files_by_timestep)):
                gdas_or_gfs_files_by_timestep[gdas_or_gfs_file_set] = list(filter(lambda hour: string not in hour, gdas_or_gfs_files_by_timestep[gdas_or_gfs_file_set]))

    if dataset == 'training':

        print(f"{gdas_or_gfs.upper()} files in training dataset:", len(gdas_or_gfs_files_by_timestep))

    elif dataset == 'validation':

        print(f"{gdas_or_gfs.upper()} files in validation dataset {validation_years}:", len(gdas_or_gfs_files_by_timestep))

    elif dataset == 'test':

        print(f"{gdas_or_gfs.upper()} files in testing dataset {test_years}:", len(gdas_or_gfs_files_by_timestep))

    elif dataset == 'timestep':

        if timestep_tuple is None:
            raise ValueError("Missing timestep parameter")

        gdas_or_gfs_files_by_timestep = gdas_or_gfs_files_by_timestep[0]

        print(f"{gdas_or_gfs.upper()} files for %d-%02d-%02d-%02dz: %d" % (timestep_tuple[0], timestep_tuple[1], timestep_tuple[2], timestep_tuple[3], len(gdas_or_gfs_files_by_timestep)))

    elif dataset is None:

        print(f"{gdas_or_gfs.upper()} files:", len(gdas_or_gfs_files_by_timestep))

    else:

        raise ValueError(f"{dataset} is not a valid dataset, options are: None, training, validation, test, timestep")

    return gdas_or_gfs_files_by_timestep


def load_gdas_or_gfs_and_fronts_pickle_files(gdas_or_gfs_pickle_indir, fronts_pickle_indir, domain, gdas_or_gfs='gdas', dataset=None, test_years=None, validation_years=None, timestep_tuple=None, forecast_hour=6):
    """
    Function that loads and returns lists of GDAS/GFS pickle files and front object files with matching dates.

    Parameters
    ----------
    gdas_or_gfs_pickle_indir: str
        - Input directory for the GDAS/GFS pickle files.
    fronts_pickle_indir: str
        - Input directory for the front object pickle files.
    domain: str
        - Domain which the front and GDAS/GFS files cover.
    gdas_or_gfs: str
        Can be 'gdas' or 'gfs' (case-insensitive).
    dataset: str or None
        - Dataset to load. Available options are 'training', 'validation', or 'test'. Leaving this as None will load the file lists without any filtering.
    test_years: list of ints or None
        - Years for the test set (will not be used in training or validation).
    validation_years: list of ints or None
        - Years for the validation set.
    timestep_tuple: tuple with 4 ints
        Timestep of the GDAS/GFS pickle files to load. This can only be passed if dataset='timestep'.
    forecast_hour: int
        Forecast hour to load.

    Returns
    -------
    gdas_or_gfs_files_<dataset>: list of strs
        - List of all files containing gdas data.
    front_files_<dataset>: list of strs
        - List of all files containing fronts. Includes files with no fronts present in their respective domains.

    Examples
    --------
    gdas_or_gfs_files, front_files = fm.load_gdas_or_gfs_and_fronts_pickle_files(gdas_or_gfs_pickle_indir, front_pickle_indir, domain='full', gdas_or_gfs='gdas')  # Load all paired sets of GDAS/GFS and front files
        * add the following parameters to load the TRAINING dataset: dataset='training', test_years=[list of ints], validation_years=[list of ints]
        * add the following parameters to load the VALIDATION dataset: dataset='validation', validation_years=[list of ints]
        * add the following parameters to load the TEST dataset: dataset='test', test_years=[list of ints]
        * add the following parameters to load a set of GDAS/GFS and fronts files for a given timestep: dataset='timestep', timestep_tuple=(year, month, day, hour)
    """
    if dataset is not None:
        print(f"Loading {gdas_or_gfs.upper()}/fronts {dataset} dataset")
    else:
        print(f"Loading {gdas_or_gfs.upper()}/fronts files")

    front_files_timestep = []
    gdas_or_gfs_files_timestep = []
    gdas_or_gfs_timesteps = []  # Timesteps of GDAS/GFS files
    gdas_or_gfs_files_by_timestep = []  # List of GDAS/GFS files with each index representing a group of files with a common timestep
    front_files_list = []
    gdas_or_gfs_files_list = []

    separators = ['/', '\\']  # The subdirectory separator is different depending on your operating system, so we will iterate through both of these separators to make sure we grab the files
    loop_counter = 0  # Counter that increases by 1 if no files are found as we iterate through each of the file separators

    required_pressure_levels = ['_surface_', '_1000_', '_975_', '_950_', '_925_', '_900_', '_850_', '_800_', '_750_', '_700_',
                                '_650_', '_600_', '_550_', '_500_', '_450_', '_400_', '_350_', '_300_', '_250_', '_200_',
                                '_150_', '_100_']

    if timestep_tuple is not None:
        if dataset != 'timestep':
            raise ValueError("dataset must be set to 'timestep' when loading GDAS/GFS and fronts files for a specific timestep")

        new_timestep_tuple = data_utils.add_or_subtract_hours_to_timestep(timestep_tuple, forecast_hour)

        front_files = sorted(glob("%s/%d/%02d/%02d/FrontObjects_%d%02d%02d%02d_%s.pkl" % (fronts_pickle_indir, new_timestep_tuple[0], new_timestep_tuple[1], new_timestep_tuple[2],
                                                                                          new_timestep_tuple[0], new_timestep_tuple[1], new_timestep_tuple[2], new_timestep_tuple[3], domain)))
        gdas_or_gfs_files = sorted(glob("%s/%d/%02d/%02d/%s_*_%d%02d%02d%02d_f%03d_%s.pkl" % (gdas_or_gfs_pickle_indir, timestep_tuple[0], timestep_tuple[1], timestep_tuple[2], gdas_or_gfs.lower(),
                                                                                              timestep_tuple[0], timestep_tuple[1], timestep_tuple[2], timestep_tuple[3], forecast_hour, domain)))

    else:

        front_files = sorted(glob("%s/*/*/*/FrontObjects_*_%s.pkl" % (fronts_pickle_indir, domain)))
        gdas_or_gfs_files = sorted(glob("%s/*/*/*/%s_*_f%03d_%s.pkl" % (gdas_or_gfs_pickle_indir, gdas_or_gfs.lower(), forecast_hour, domain)))

    # Total number of files before any filtering is performed
    initial_front_files_count = len(front_files)
    initial_gdas_or_gfs_files_count = len(gdas_or_gfs_files)

    if dataset == 'training':

        front_files_list = front_files
        gdas_or_gfs_files_list = gdas_or_gfs_files

        if validation_years is None:
            raise ValueError("Validation years must be declared to pull the training set")
        if test_years is None:
            raise ValueError("Test years must be declared to pull the training set")

        while len(front_files) == initial_front_files_count and len(gdas_or_gfs_files) == initial_gdas_or_gfs_files_count and loop_counter < 2:  # While no GDAS/GFS and fronts files in the training dataset are found
            for validation_year in validation_years:
                validation_year_string = separators[loop_counter] + str(validation_year) + separators[loop_counter]
                front_files_list = list(filter(lambda validation_year: validation_year_string not in validation_year, front_files_list))
                gdas_or_gfs_files_list = list(filter(lambda validation_year: validation_year_string not in validation_year, gdas_or_gfs_files_list))
            for test_year in test_years:
                test_year_string = separators[loop_counter] + str(test_year) + separators[loop_counter]
                front_files_list = list(filter(lambda test_year: test_year_string not in test_year, front_files_list))
                gdas_or_gfs_files_list = list(filter(lambda test_year: test_year_string not in test_year, gdas_or_gfs_files_list))
            loop_counter += 1

    elif dataset == 'validation':

        if validation_years is None:
            raise ValueError("Validation years must be declared to pull the validation set")

        while len(front_files) == initial_front_files_count and len(gdas_or_gfs_files) == initial_gdas_or_gfs_files_count and loop_counter < 2:  # While no GDAS/GFS and fronts files in the training dataset are found
            for validation_year in validation_years:
                validation_year_string = separators[loop_counter] + str(validation_year) + separators[loop_counter]
                front_files_list.append(list(filter(lambda validation_year: validation_year_string in validation_year, front_files)))
                gdas_or_gfs_files_list.append(list(filter(lambda validation_year: validation_year_string in validation_year, gdas_or_gfs_files)))
            loop_counter += 1

    elif dataset == 'test':

        if test_years is None:
            raise ValueError("Test years must be declared to pull the test set")

        while len(front_files) == initial_front_files_count and len(gdas_or_gfs_files) == initial_gdas_or_gfs_files_count and loop_counter < 2:  # While no GDAS/GFS and fronts files in the training dataset are found
            for test_year in test_years:
                test_year_string = separators[loop_counter] + str(test_year) + separators[loop_counter]
                front_files_list.append(list(filter(lambda test_year: test_year_string in test_year, front_files)))
                gdas_or_gfs_files_list.append(list(filter(lambda test_year: test_year_string in test_year, gdas_or_gfs_files)))
            loop_counter += 1

    # Flatten new file lists
    if len(front_files) > 1:
        front_files = list(sorted(itertools.chain(*front_files_list)))
        gdas_or_gfs_files = list(sorted(itertools.chain(*gdas_or_gfs_files_list)))

    # Reset the 'list' variables so they can be reused later during file matching
    front_files_list = []
    gdas_or_gfs_files_list = []

    # Collect all timesteps in the fronts files
    for i in range(len(front_files)):
        front_filename_start_index = front_files[i].find('FrontObjects_')
        front_files_timestep.append(front_files[i][front_filename_start_index + 13:front_filename_start_index + 23])  # Timestep is first 10 characters after 'FrontObjects_'

    # Find all timesteps in the GDAS/GFS files
    for j in range(len(gdas_or_gfs_files)):
        gdas_or_gfs_filename_start_index = gdas_or_gfs_files[j].find('%s_' % gdas_or_gfs.lower())
        gdas_or_gfs_filename_no_pressure_level_index = gdas_or_gfs_files[j][gdas_or_gfs_filename_start_index + 5:].find('_2') + 1
        gdas_or_gfs_timesteps.append(gdas_or_gfs_files[j][gdas_or_gfs_filename_start_index + gdas_or_gfs_filename_no_pressure_level_index + 5:gdas_or_gfs_filename_start_index + gdas_or_gfs_filename_no_pressure_level_index + 15])  # Timestep is the first 10 characters after the pressure level in the filename

    unique_gdas_or_gfs_fronts_timesteps = np.unique(sorted(gdas_or_gfs_timesteps))

    # Find all GDAS/GFS files for each timestep and order them by pressure level
    for timestep in unique_gdas_or_gfs_fronts_timesteps:
        files_for_timestep = sorted(filter(lambda filename: timestep in filename, gdas_or_gfs_files))

        # Move 1000 mb file to the end of the list, then remove its duplicate at the beginning
        files_for_timestep.append(files_for_timestep[0])
        files_for_timestep.pop(0)

        # Move surface file to the end of the list and remove its duplicate
        files_for_timestep.append(files_for_timestep[-2])
        files_for_timestep.pop(-3)

        # Reverse list so that pressure levels are in descending order / increasing with altitude
        files_for_timestep = files_for_timestep[::-1]

        # Check all files for all pressure levels
        pressure_levels_found = 0
        for pressure_level, file in zip(required_pressure_levels, files_for_timestep):
            if pressure_level in file:
                pressure_levels_found += 1
        if pressure_levels_found == 22:
            gdas_or_gfs_files_timestep_temp = []  # Temporary holding folder for GDAS/GFS files before they are added to the list of filenames
            for full_filename in range(22):
                gdas_or_gfs_filename_start_index = files_for_timestep[full_filename].find('%s_' % gdas_or_gfs.lower())
                gdas_or_gfs_filename_no_pressure_level_index = files_for_timestep[full_filename][gdas_or_gfs_filename_start_index + 5:].find('_2') + 1
                gdas_or_gfs_files_timestep_temp.append(files_for_timestep[full_filename][gdas_or_gfs_filename_start_index + gdas_or_gfs_filename_no_pressure_level_index + 5:gdas_or_gfs_filename_start_index + gdas_or_gfs_filename_no_pressure_level_index + 15])
            gdas_or_gfs_files_timestep.append(gdas_or_gfs_files_timestep_temp)
            gdas_or_gfs_files_by_timestep.append(files_for_timestep)

    new_unique_gdas_files_timesteps = np.unique(list(itertools.chain(*gdas_or_gfs_files_timestep)))
    new_unique_forecast_timesteps = []  # List of future timesteps that will be used to grab front files. A 12z GDAS timestep at forecast hour 6 will return an 18z front file

    for unique_timestep in unique_gdas_or_gfs_fronts_timesteps:
        new_unique_forecast_timesteps.append(data_utils.add_or_subtract_hours_to_timestep(str(unique_timestep), forecast_hour))

    """ File matching: only load files with common timesteps or the proper timesteps for the given forecast hour """
    if len(gdas_or_gfs_files) > len(front_files):
        total_gdas_or_gfs_files = len(gdas_or_gfs_files_timestep)
        total_front_files = len(front_files)
        for gdas_or_gfs_file_no in range(total_gdas_or_gfs_files):
            if new_unique_gdas_files_timesteps[gdas_or_gfs_file_no] in front_files_timestep:
                gdas_or_gfs_files_list.append(gdas_or_gfs_files_by_timestep[gdas_or_gfs_file_no])
        for front_file_no in range(total_front_files):
            if front_files_timestep[front_file_no] in new_unique_forecast_timesteps:
                front_files_list.append(front_files[front_file_no])
    else:
        total_front_files = len(front_files)
        total_gdas_or_gfs_files = len(gdas_or_gfs_files_timestep)

        for front_file_no in range(total_front_files):
            if any(front_files_timestep[front_file_no] in gdas_timestep for gdas_timestep in gdas_or_gfs_files_timestep):
                front_files_list.append(front_files[front_file_no])

        for gdas_or_gfs_file_no in range(total_gdas_or_gfs_files):
            if gdas_or_gfs_files_timestep[gdas_or_gfs_file_no][0] in front_files_timestep:
                gdas_or_gfs_files_list.append(gdas_or_gfs_files_by_timestep[gdas_or_gfs_file_no])

    # Filter out non-synoptic hours if loading files over the full domain
    if domain == 'full':
        hours_to_remove = [3, 9, 15, 21]
        for hour in hours_to_remove:
            string = '%02d_' % hour
            front_files_list = list(filter(lambda hour: string not in hour, front_files_list))
            for gdas_or_gfs_file_set in range(len(gdas_or_gfs_files_list)):
                gdas_or_gfs_files_list[gdas_or_gfs_file_set] = list(filter(lambda hour: string not in hour, gdas_or_gfs_files_list[gdas_or_gfs_file_set]))

    if dataset == 'training':

        print(f"{gdas_or_gfs.upper()}/fronts files in training dataset:", len(gdas_or_gfs_files_list), len(front_files_list))

    elif dataset == 'validation':

        print(f"{gdas_or_gfs.upper()}/fronts files in validation dataset {validation_years}:", len(gdas_or_gfs_files_list), len(front_files_list))

    elif dataset == 'test':

        print(f"{gdas_or_gfs.upper()}/fronts files in testing dataset {test_years}:", len(gdas_or_gfs_files_list), len(front_files_list))

    elif dataset == 'timestep':

        if timestep_tuple is None:
            raise ValueError("Missing timestep parameter")

        gdas_or_gfs_files_list = gdas_or_gfs_files_by_timestep
        front_files_list = [front_files[0], ]

        print(f"{gdas_or_gfs.upper()}/fronts files for %d-%02d-%02d-%02dz F%03d: %d/%d" % (timestep_tuple[0], timestep_tuple[1], timestep_tuple[2], timestep_tuple[3], len(gdas_or_gfs_files_list[0]), len(front_files_list), forecast_hour))

    elif dataset is None:

        print(f"{gdas_or_gfs.upper()}/fronts files:", len(gdas_or_gfs_files_list), len(front_files_list))

    else:

        raise ValueError(f"{dataset} is not a valid dataset, options are: None, training, validation, test, timestep")

    return gdas_or_gfs_files_list, front_files_list


def load_model(model_number, model_dir):
    """
    Load a saved model.

    Parameters
    ----------
    model_number: int
        - Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        - Main directory for the models.
    """

    model_properties = pd.read_pickle(f"{model_dir}/model_{model_number}/model_{model_number}_properties.pkl")

    loss = model_properties['loss_function']
    metric = model_properties['metric']

    if loss == 'cce' and metric == 'auc':
        model = lm('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number))
    else:
        if loss == 'fss':
            fss_mask_size, fss_c = model_properties['fss_mask_c'][0], model_properties['fss_mask_c'][1]  # First element is the mask size, second is the c parameter
            loss_function = custom_losses.make_fractions_skill_score(fss_mask_size, fss_c)
            loss_string = 'FSS_loss_3D'
        elif loss == 'bss':
            loss_string = 'brier_skill_score'
            loss_function = custom_losses.brier_skill_score
        else:
            loss_string = None
            loss_function = None

        if metric == 'fss':
            fss_mask_size, fss_c = model_properties['fss_mask_c'][0], model_properties['fss_mask_c'][1]  # First element is the mask size, second is the c parameter
            loss_function = custom_losses.make_fractions_skill_score(fss_mask_size, fss_c)
            metric_string = 'FSS_loss_3D'
        elif metric == 'dice':
            metric_string = 'dice'
            metric_function = custom_losses.dice
        elif metric == 'tversky':
            metric_string = 'tversky'
            metric_function = custom_losses.tversky
        elif metric == 'bss':
            metric_string = 'brier_skill_score'
            metric_function = custom_losses.brier_skill_score
        else:
            metric_string = None
            metric_function = None

        print(loss_string, metric_string)

        print("Loading model....", end='')
        if loss_string is not None:
            if metric_string is not None:
                model = lm('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number), custom_objects={loss_string: loss_function, metric_string: metric_function})
            else:
                model = lm('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number), custom_objects={loss_string: loss_function})
        else:
            model = lm('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number), custom_objects={metric_string: metric_function})
        print("done")

    return model


if __name__ == '__main__':
    """
    Warnings
        - Do not use leading zeros when declaring the month, day, and hour in 'date'. (ex: if the day is 2, do not type 02)
        - Longitude values in the 'new_extent' argument must in the 360-degree coordinate system.
    
    Examples
        Example 1a. Create subdirectories organized by year, month, and day to organize data files.
            > python file_manager.py --create_subdirectories --main_dir ./pickle_files
        
        Example 1b. Create subdirectories organized by year, month, day, and hour to organize data files.
            > python file_manager.py --create_subdirectories --main_dir ./pickle_files --hour_dirs
        
        Example 2. Compress a large set of files into a tarfile.
            > python file_manager.py --compress_files --main_dir ./file_directory --glob_file_string test_files_*_.pkl --tar_filename compressed_files.tar
        
        Example 3. Extract the contents of an existing tarfile.
            > python file_manager.py --extract_tarfile --main_dir ./file_directory --tar_filename compressed_files.tar
        
        Example 4. Create new ERA5, GDAS, GFS, and frontal object pickle files across a smaller domain.
            NOTE: This is currently not operational.
        
        Example 5. Delete a large group of files using a common string in the filenames.
            > python file_manager.py --delete_grouped_files --main_dir ./file_directory --num_subdir 3 --glob_file_string test_files_*_.pkl
            Using 3 for 'num_subdir' means that files will be returned from glob that match the following string:
                ./file_directory/*/*/*/test_files_*_.pkl
            - Using 5 for 'num_subdir':
                ./file_directory/*/*/*/*/*/test_files_*_.pkl

        Example 6a. Load ERA5 pickle files.
            Refer to Examples in function 'load_era5_pickle_files'.
        
        Example 6b. Load ERA5/front pickle files.
            Refer to Examples in function 'load_era5_and_fronts_pickle_files'.
        
        Example 6c. Load GDAS or GFS pickle files.
            Refer to Examples in function 'load_gdas_or_gfs_pickle_files'.
        
        Example 6d. Load GDAS/GFS and front pickle_files.
            Refer to Examples in function 'load_gdas_or_gfs_and_fronts_pickle_files.
        
        Example 6e. Load pickle files containing frontal objects.
            Refer to Examples in function 'load_fronts_pickle_files'.
        
        Example 7. Load a saved tensorflow model.
            Refer to Examples in function 'load_model'.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--compress_files', action='store_true', help='Compress files')
    parser.add_argument('--create_new_era5_files', action='store_true', help='Create ERA5 files for a new domain')
    parser.add_argument('--create_new_front_files', action='store_true', help='Create front files for a new domain')
    parser.add_argument('--create_new_gdas_or_gfs_files', action='store_true', help='Create GDAS or GFS files for a new domain')
    parser.add_argument('--create_subdirectories', action='store_true', help='Create new directories')
    parser.add_argument('--delete_grouped_files', action='store_true', help='Delete a set of files')
    parser.add_argument('--domain', type=str, help='Domain of the data')
    parser.add_argument('--extract_tarfile', action='store_true', help='Extract a tarfile')
    parser.add_argument('--glob_file_string', type=str, help='String of the names of the files to compress or delete.')
    parser.add_argument('--hour_dirs', action='store_true', help='Create hourly subdirectories?')
    parser.add_argument('--main_dir', type=str, help='Main directory for subdirectory creation or where the files in question are located.')
    parser.add_argument('--new_domain', type=str, help='Name of the new domain.')
    parser.add_argument('--new_extent', type=float, nargs=4, help='Extent of the new domain. Pass 4 values in the following order:'
                                                                  'min_lon, max_lon, min_lat, max_lat')
    parser.add_argument('--num_subdir', type=int, help='Number of subdirectory layers in the main directory.')
    parser.add_argument('--num_variables', type=int, help='Number of variables in the variable datasets.')
    parser.add_argument('--gdas_or_gfs_pickle_indir', type=str, help='Input directory for the GDAS or GFS pickle files.')
    parser.add_argument('--era5_pickle_indir', type=str, help='Input directory for the ERA5 pickle files.')
    parser.add_argument('--fronts_pickle_indir', type=str, help='Input directory for the front object files.')
    parser.add_argument('--tar_filename', type=str, help='Name of the tar file.')
    args = parser.parse_args()
    provided_arguments = vars(args)
    
    if args.compress_files:
        required_arguments = ['main_dir', 'glob_file_string', 'tar_filename']
        check_arguments(provided_arguments, required_arguments)
        compress_files(args.main_dir, args.glob_file_string, args.tar_filename)

    if args.create_new_era5_files:
        required_arguments = ['era5_pickle_indir', 'domain', 'num_variables', 'new_domain', 'new_extent']
        check_arguments(provided_arguments, required_arguments)
        era5_domain_files = load_era5_pickle_files(args.era5_pickle_indir, args.num_variables, args.domain)
        create_new_domain_files(era5_domain_files, args.domain, args.new_domain, args.new_extent)

    if args.create_new_front_files:
        required_arguments = ['fronts_pickle_indir', 'domain', 'new_domain', 'new_extent']
        check_arguments(provided_arguments, required_arguments)
        front_domain_files = load_fronts_pickle_files(args.fronts_pickle_indir, args.domain)
        create_new_domain_files(front_domain_files, args.domain, args.new_domain, args.new_extent)

    if args.create_new_gdas_or_gfs_files:
        required_arguments = ['gdas_or_gfs_pickle_indir', 'domain', 'new_domain', 'new_extent']
        check_arguments(provided_arguments, required_arguments)
        gdas_or_gfs_domain_files = load_gdas_or_gfs_pickle_files(args.gdas_or_gfs_pickle_indir, args.domain)
        create_new_domain_files(gdas_or_gfs_domain_files, args.domain, args.new_domain, args.new_extent)

    if args.create_subdirectories:
        required_arguments = ['main_dir']
        check_arguments(provided_arguments, required_arguments)
        create_subdirectories(args.main_dir, args.hour_dirs)

    if args.delete_grouped_files:
        required_arguments = ['glob_file_string', 'main_dir', 'num_subdir']
        check_arguments(provided_arguments, required_arguments)
        delete_grouped_files(args.main_dir, args.glob_file_string, args.num_subdir)

    if args.extract_tarfile:
        required_arguments = ['main_dir', 'tar_filename']
        check_arguments(provided_arguments, required_arguments)
        extract_tarfile(args.main_dir, args.tar_filename)
