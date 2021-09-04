"""
Functions in this code manage data files and directories.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 9/4/2021 2:15 PM CDT
"""

from glob import glob
import pickle
import argparse
import numpy as np
import os
import errors


def add_hourly_directories(main_dir_subdir, year, month, day):
    """
    Adds hourly subdirectories in daily subdirectories.

    Parameters
    ----------
    main_dir_subdir: str
        Main directory for the subdirectories.
    year: int
    month: int
    day: int
    """
    for hour in range(0,24,3):
        os.mkdir('%s/%d/%02d/%02d/%02d' % (main_dir_subdir, year, month, day, hour))


def create_subdirectories(main_dir_subdir, hour_dirs):
    """
    Creates subdirectories for files sorted by year, month, day, and hour (hour is optional).

    Parameters
    ----------
    main_dir_subdir: str
        Main directory for the subdirectories.
    hour_dirs: Boolean
        Boolean flag that determines whether or not hourly subdirectories will be generated.
    """

    years = list(np.linspace(2006,2020,15))

    month_1_days = np.linspace(1,31,31)
    month_3_days = np.linspace(1,31,31)
    month_4_days = np.linspace(1,30,30)
    month_5_days = np.linspace(1,31,31)
    month_6_days = np.linspace(1,30,30)
    month_7_days = np.linspace(1,31,31)
    month_8_days = np.linspace(1,31,31)
    month_9_days = np.linspace(1,30,30)
    month_10_days = np.linspace(1,31,31)
    month_11_days = np.linspace(1,30,30)
    month_12_days = np.linspace(1,31,31)

    print("Making directories....",end='')
    for year in years:
        if year % 4 == 0:  # Check if the current year is a leap year
            month_2_days = np.linspace(1,29,29)
        else:
            month_2_days = np.linspace(1,28,28)
        month_days = [month_1_days,month_2_days,month_3_days,month_4_days,month_5_days,month_6_days,month_7_days,month_8_days,
                      month_9_days,month_10_days,month_11_days,month_12_days]
        os.mkdir('%s/%d' % (main_dir_subdir, year))
        for month in range(1,13):
            os.mkdir('%s/%d/%02d' % (main_dir_subdir, year, month))
            for day in month_days[month-1]:
                os.mkdir('%s/%d/%02d/%02d' % (main_dir_subdir, year, month, day))
                if hour_dirs is True:
                    add_hourly_directories(main_dir_subdir, year, month, day)
    print("done")


def delete_grouped_files(main_dir_group, glob_file_string, num_subdir):
    """
    Deletes grouped files with names matching given strings.

    Parameters
    ----------
    main_dir_group: str
        Main directory or directories where the grouped files are located.
    glob_file_string: str
        String of the names of the files to delete.
    num_subdir: int
        Number of subdirectory layers in the main directory.
    """

    subdir_string = ''

    for i in range(num_subdir):
        subdir_string += '/*'
    glob_file_string = subdir_string + glob_file_string

    files_to_delete = list(sorted(glob("%s/%s" % (main_dir_group, glob_file_string))))

    print("Deleting %d files...." % len(files_to_delete),end='')
    for file in files_to_delete:
        os.remove(file)
    print("done")


def generate_file_lists(front_files, variable_files, num_variables, front_types, domain, file_dimensions):
    """
    Generates lists of files containing front and variable data. Creating separate lists and loading them saves time as
    opposed to just loading all of the files individually each time the code is ran.

    Parameters
    ----------
    front_files: list
        List of files containing front data.
    variable_files: list
        List of files containing variable data.
    num_variables: int
        Number of variables in the variable datasets.
    front_types: str
        Fronts in the frontobject datasets.
    domain: str
        Domain which the front and variable files cover.
    file_dimensions: int (x2)
        Dimensions of the domain/files.
    """
    front_files_no_prefix = []
    variable_files_no_prefix = []
    front_files_list = []
    variable_files_list = []

    print("Generating file lists....", end='')
    for i in range(len(front_files)):
        front_files_no_prefix.append(front_files[i].replace('FrontObjects_%s_' % front_types, ''))
    for j in range(len(variable_files)):
        variable_files_no_prefix.append(variable_files[j].replace('Data_%dvar_' % num_variables, ''))

    if len(variable_files) > len(front_files):
        total_variable_files = len(variable_files)
        total_front_files = len(front_files)
        for k in range(total_variable_files):
            if variable_files_no_prefix[k] in front_files_no_prefix:
                variable_files_list.append(variable_files[k])
        for l in range(total_front_files):
            if front_files_no_prefix[l] in variable_files_no_prefix:
                front_files_list.append(front_files[l])
    else:
        total_front_files = len(front_files)
        total_variable_files = len(variable_files)
        for k in range(total_front_files):
            if front_files_no_prefix[k] in variable_files_no_prefix:
                front_files_list.append(front_files[k])
        for l in range(total_variable_files):
            if variable_files_no_prefix[l] in front_files_no_prefix:
                variable_files_list.append(variable_files[l])
    print("done")

    print("Front file list length: %d" % len(front_files_list))
    print("Variable file list length: %d\n" % len(variable_files_list))

    matches = 0
    mismatches = 0

    # Check the end of the filenames to match up files together
    for i in range(len(front_files_list)):
        if front_files_list[0][-22-len(str(file_dimensions[0]))-len(str(file_dimensions[1])):] == \
            variable_files_list[0][-22-len(str(file_dimensions[0]))-len(str(file_dimensions[1])):]:
            matches += 1
        else:
            mismatches += 1

    print("Matched files: %d/%d" % (matches, len(front_files_list)))
    print("Mismatched files: %d/%d" % (mismatches, len(front_files_list)))

    if mismatches == 0:
        print("No errors found!\n")
    else:
        print("ERROR: File lists do not have matching dates, check your data and remake the lists.\n")

    print("Saving lists....", end='')
    with open('%dvar_%s_%s_%dx%d_front_files_list.pkl' % (num_variables, front_types, domain, file_dimensions[0],
                                                          file_dimensions[1]), 'wb') as f:
        pickle.dump(front_files_list, f)
    with open('%dvar_%s_%s_%dx%d_variable_files_list.pkl' % (num_variables, front_types, domain, file_dimensions[0],
                                                             file_dimensions[1]), 'wb') as f:
        pickle.dump(variable_files_list, f)
    print("done")


def load_files(pickle_indir, num_variables, front_types, domain, file_dimensions):
    """
    Function that loads and returns lists of data files.

    Parameters
    ----------
    pickle_indir: str
        Directory where the created pickle files containing the domain data will be stored.
    num_variables: int
        Number of variables in the variable datasets.
    front_types: str
        Fronts in the frontobject datasets.
    domain: str
        Domain which the front and variable files cover.
    file_dimensions: int (x2)
        Dimensions of the domain/files.

    Returns
    -------
    front_files: list
        List of all files containing fronts. Includes files with no fronts present in their respective domains.
    variable_files: list
        List of all files containing variable data.
    """
    print("Collecting front object files....", end='')
    front_files = sorted(glob("%s/*/*/*/FrontObjects_%s_*_%s_%dx%d.pkl" % (pickle_indir, front_types, domain, file_dimensions[0],
                                                                           file_dimensions[1])))
    print("done, %d files found" % len(front_files))
    print("Collecting variable data files....", end='')
    variable_files = sorted(glob("%s/*/*/*/Data_%dvar_*_%s_%dx%d.pkl" % (pickle_indir, num_variables, domain, file_dimensions[0],
                                                                         file_dimensions[1])))
    print("done, %d files found" % len(variable_files))

    return front_files, variable_files


def load_file_lists(num_variables, front_types, domain, file_dimensions):
    """
    Opens files containing lists of filenames for fronts and variable data.

    Parameters
    ----------
    num_variables: int
        Number of variables in the variable datasets.
    front_types: str
        Fronts in the frontobject datasets.
    domain: str
        Domain which the front and variable files cover.
    file_dimensions: int (x2)
        Dimensions of the domain/files.

    Returns
    -------
    front_files_list: list
        List of filenames that contain front data.
    variable_files_list: list
        List of filenames that contain variable data.
    """
    with open('%dvar_%s_%s_%dx%d_front_files_list.pkl' % (num_variables, front_types, domain, file_dimensions[0],
                                                          file_dimensions[1]), 'rb') as f:
        front_files_list = pickle.load(f)
    with open('%dvar_%s_%s_%dx%d_variable_files_list.pkl' % (num_variables, front_types, domain, file_dimensions[0],
                                                             file_dimensions[1]), 'rb') as f:
        variable_files_list = pickle.load(f)

    return front_files_list, variable_files_list


def load_test_files(num_variables, front_types, domain, file_dimensions, test_year):
    """
    Splits front and variable data files into training and validation sets.

    Parameters
    ----------
    num_variables: int
        Number of variables in the variable datasets.
    front_types: str
        Fronts in the frontobject datasets.
    domain: str
        Domain which the front and variable files cover.
    file_dimensions: int (x2)
        Dimensions of the domain/files.
    test_year: int
        Year for the test set (will not be used in training or validation).

    Returns
    -------
    front_files_test: list
        List of files containing front data for the test dataset.
    variable_files_test: list
        List of files containing variable data for the test dataset.
    """
    front_files, variable_files = load_file_lists(num_variables, front_types, domain, file_dimensions)

    print("Test year: %d" % test_year)
    test_year_string = "/" + str(test_year) + "/"

    print("Splitting files....", end='')
    front_files_test = list(filter(lambda year: test_year_string in year, front_files))
    variable_files_test = list(filter(lambda year: test_year_string in year, variable_files))
    print("done")

    return front_files_test, variable_files_test


def split_file_lists(front_files, variable_files, validation_year, test_year):
    """
    Splits front and variable data files into training and validation sets.

    Parameters
    ----------
    front_files: list
        List of files containing front data.
    variable_files: list
        List of files containing variable data.
    validation_year: int
        Year for the validation set.
    test_year: int
        Year for the test set (will not be used in training or validation).

    Returns
    -------
    front_files_training: list
        List of files containing front data for the training dataset.
    front_files_validation: list
        List of files containing front data for the validation dataset.
    variable_files_training: list
        List of files containing variable data for the training dataset.
    variable_files_validation: list
        List of files containing variable data for the validation dataset.
    """

    print("Validation year: %d" % validation_year)
    print("Test year: %d" % test_year)
    validation_year_string = "/" + str(validation_year) + "/"
    test_year_string = "/" + str(test_year) + "/"

    print("Splitting files....", end='')
    front_files_validation = list(filter(lambda year: validation_year_string in year, front_files))
    variable_files_validation = list(filter(lambda year: validation_year_string in year, variable_files))
    front_files_test = list(filter(lambda year: test_year_string in year, front_files))
    variable_files_test = list(filter(lambda year: test_year_string in year, variable_files))

    front_files_training_temp = list(filter(lambda year: validation_year_string not in year, front_files))
    variable_files_training_temp = list(filter(lambda year: validation_year_string not in year, variable_files))
    front_files_training = list(filter(lambda year: test_year_string not in year, front_files_training_temp))
    variable_files_training = list(filter(lambda year: test_year_string not in year, variable_files_training_temp))
    print("done")

    print("All files BEFORE splitting into sets: %d front files, %d variable files" % (len(front_files), len(variable_files)))
    print("Training set: %d front files, %d variable files" % (len(front_files_training), len(variable_files_training)))
    print("Validation set (%d): %d front files, %d variable files" % (validation_year, len(front_files_validation), len(variable_files_validation)))
    print("***Testing set (%d): %d front files, %d variable files   ***THESE WILL NOT BE USED IN TRAINING OR VALIDATION" % (test_year, len(front_files_test), len(variable_files_test)))
    return front_files_training, front_files_validation, variable_files_training, variable_files_validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    """ Optional Arguments """
    parser.add_argument('--create_subdirectories', type=bool, required=False, help='Create new directories?')
    parser.add_argument('--delete_grouped_files', type=bool, required=False, help='Delete a set of files?')
    parser.add_argument('--generate_lists', type=bool, required=False, help='Generate lists of new files?')

    """ Conditional Arguments """
    # Must be passed if create_subdirectories is True #
    parser.add_argument('--hour_dirs', type=bool, required=False, help='Create hourly subdirectories?')
    parser.add_argument('--main_dir_subdir', type=str, required=False,
                        help='Main directory where the subdirectories will be made.')

    # Must be passed if delete_grouped_files is True #
    parser.add_argument('--main_dir_group', type=str, required=False,
                        help='Directory or directories where the grouped files are located.')
    parser.add_argument('--glob_file_string', type=str, required=False, help='String of the names of the files to delete.')
    parser.add_argument('--num_subdir', type=int, required=False,
                        help='Number of subdirectory layers in the main directory.')

    # Must be passed if generate_lists is True #
    parser.add_argument('--domain', type=str, required=False, help='Domain of the data.')
    parser.add_argument('--file_dimensions', type=int, nargs=2, required=False,
                        help='Dimensions of the map size. Two integers need to be passed.')
    parser.add_argument('--front_types', type=str, required=False,
                        help='Front format of the file. If your files contain warm and cold fronts, pass this argument '
                             'as CFWF. If your files contain only drylines, pass this argument as DL. If your files '
                             'contain all fronts, pass this argument as ALL.')
    parser.add_argument('--num_variables', type=int, required=False, help='Number of variables in the variable datasets.')
    parser.add_argument('--pickle_indir', type=str, required=False,
                        help='Path of pickle files containing front object and variable data.')

    args = parser.parse_args()

    if args.create_subdirectories is True:
        if args.hour_dirs is None or args.main_dir_subdir is None:
            raise errors.MissingArgumentError("If create_subdirectories is True, the following arguments must be passed: "
                                              "hour_dirs, main_dir_subdir")
        else:
            create_subdirectories(args.main_dir_subdir, args.hour_dirs)

    if args.delete_grouped_files is True:
        if args.glob_file_string is None or args.main_dir_group is None or args.num_subdir is None:
            raise errors.MissingArgumentError("If delete_grouped_files is True, the following arguments must be passed: "
                                              "glob_file_string, main_dir_group, num_subdir")
        else:
            delete_grouped_files(args.main_dir_group, args.glob_file_string, args.num_subdir)

    if args.generate_lists is True:
        if args.domain is None or args.file_dimensions is None or args.front_types is None or args.num_variables is None or \
            args.pickle_indir is None:
            raise errors.MissingArgumentError("If generate_lists is True, the following arguments must be passed: "
                                              "domain, file_dimensions, front_types, num_variables, pickle_indir")
        else:
            front_files, variable_files = load_files(args.pickle_indir, args.num_variables, args.front_types, args.domain,
                args.file_dimensions)
            generate_file_lists(front_files, variable_files, args.num_variables, args.front_types, args.domain,
                args.file_dimensions)
