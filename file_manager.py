"""
Functions in this code prepare and organize data files before they are used in model training and validation.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 8/10/2021 5:12 PM CDT
"""

from glob import glob
import pickle
import argparse


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


def split_file_lists(front_files, variable_files, validation_year):
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
    validation_year_string = "/" + str(validation_year) + "/"

    print("Splitting files....", end='')
    front_files_validation = list(filter(lambda year: validation_year_string in year, front_files))
    variable_files_validation = list(filter(lambda year: validation_year_string in year, variable_files))
    front_files_training = list(filter(lambda year: validation_year_string not in year, front_files))
    variable_files_training = list(filter(lambda year: validation_year_string not in year, variable_files))
    print("done")

    print("Training set: %d front files, %d variable files" % (len(front_files_training), len(variable_files_training)))
    print("Validation set (%d): %d front files, %d variable files" % (validation_year, len(front_files_validation), len(variable_files_validation)))

    return front_files_training, front_files_validation, variable_files_training, variable_files_validation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_indir', type=str, required=True, help='Path of pickle files containing front object'
        ' and variable data.')
    parser.add_argument('--num_variables', type=int, required=True, help='Number of variables in the variable datasets.')
    parser.add_argument('--front_types', type=str, required=True, help='Front format of the file. If your files contain '
        'warm and cold fronts, pass this argument as CFWF. If your files contain only drylines, pass this argument as '
        'DL. If your files contain all fronts, pass this argument as ALL.')
    parser.add_argument('--domain', type=str, required=True, help='Domain of the data. Possible values are: conus')
    parser.add_argument('--file_dimensions', type=int, nargs=2, required=True, help='Dimensions of the map size. Two '
        'integers need to be passed.')
    parser.add_argument('--generate_lists', type=str, required=False, help='Generate lists of new files? (True/False)')
    args = parser.parse_args()

    front_files, variable_files = load_files(args.pickle_indir, args.num_variables, args.front_types, args.domain,
                                             args.file_dimensions)
    if len(front_files) == len(variable_files) and (args.generate_lists != 'True'):
        print("WARNING: File lists have equal length. If you would still like to create new lists, pass the "
              "'--generate_lists' argument as 'True'.")
    else:
        generate_file_lists(front_files, variable_files, args.num_variables, args.front_types, args.domain,
                            args.file_dimensions)
