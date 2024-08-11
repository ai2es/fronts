"""
Functions in this code manage data files and models.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.8.10
"""

import argparse
from glob import glob
import os
import numpy as np
import pandas as pd
import shutil
import tarfile


def compress_files(
    main_dir: str,
    glob_file_string: str,
    tar_filename: str,
    algorithm: str = "gz",
    remove_files: bool = False,
    status_printout: bool = True):
    """
    Compress files into a TAR file.

    Parameters
    ----------
    main_dir: str
        Main directory where the files are located and where the TAR file will be saved.
    glob_file_string: str
        String of the names of the files to compress.
    tar_filename: str
        Name of the compressed TAR file that will be made. Do not include the .tar.gz extension in the name, this is added automatically.
    algorithm: str
        Compression algorithm to use when generating the TAR file.
    remove_files: bool
        Setting this to true will remove the files after they have been compressed to a TAR file.
    status_printout: bool
        Setting this to true will provide printouts of the status of the compression.

    Examples
    --------
    <<<<< start example >>>>

        import fronts.file_manager as fm

        main_dir = 'C:/Users/username/data_files'
        glob_file_string = '*matching_string.pkl'
        tar_filename = 'matching_files'  # Do not add the .tar.gz extension, this is done automatically

        compress_files(main_dir, glob_file_string, tar_filename, remove_files=True, status_printout=False)  # Compress files and remove them after compression into a TAR file

    <<<<< end example >>>>>
    """

    ########################################### Check the parameters for errors ########################################
    if not isinstance(main_dir, str):
        raise TypeError(f"main_dir must be a string, received {type(main_dir)}")
    if not isinstance(glob_file_string, str):
        raise TypeError(f"glob_file_string must be a string, received {type(glob_file_string)}")
    if not isinstance(tar_filename, str):
        raise TypeError(f"tar_filename must be a string, received {type(tar_filename)}")
    if not isinstance(remove_files, bool):
        raise TypeError(f"remove_files must be a boolean, received {type(remove_files)}")
    if not isinstance(status_printout, bool):
        raise TypeError(f"status_printout must be a boolean, received {type(status_printout)}")
    ####################################################################################################################

    uncompressed_size = 0  # MB

    ### Gather a list of files containing the specified string ###
    files = list(sorted(glob(f"{main_dir}/{glob_file_string}")))
    if len(files) == 0:
        raise OSError("No files found")
    else:
        print(f"{len(files)} files found")

    num_files = len(files)  # Total number of files

    ### Create the TAR file ###
    with tarfile.open(f"{main_dir}/{tar_filename}.tar.{algorithm}", f"w:{algorithm}") as tarF:

        ### Iterate through all the available files ###
        for file in range(num_files):
            tarF.add(files[file], arcname=files[file].replace(main_dir, ''))  # Add the file to the TAR file
            tarF_size = os.path.getsize(f"{main_dir}/{tar_filename}.tar.{algorithm}")/1e6  # Compressed size of the files within the TAR file (megabytes)
            uncompressed_size += os.path.getsize(files[file])/1e6  # Uncompressed size of the files within the TAR file (megabytes)

            ### Print out the current status of the compression (if enabled) ###
            if status_printout:
                print(f'({file+1}/{num_files})      {uncompressed_size:,.2f} MB ---> {tarF_size:,.2f} MB ({100*(1-(tarF_size/uncompressed_size)):.1f}% compression ratio)', end='\r')

    # Completion message
    print(f"Successfully compressed {len(files)} files: ",
          f'{uncompressed_size:,.2f} MB ---> {tarF_size:,.2f} MB ({100*(1-(tarF_size/uncompressed_size)):.1f}% compression ratio)')

    ### Remove the files that were added to the TAR archive (if enabled; does NOT affect the contents of the TAR file just created) ###
    if remove_files:
        for file in files:
            os.remove(file)
        print(f"Successfully deleted {len(files)} files")


def delete_grouped_files(
    main_dir: str, 
    glob_file_string: str, 
    num_subdir: int):
    """
    Deletes grouped files with names matching given strings.

    Parameters
    ----------
    main_dir: str
        Main directory or directories where the grouped files are located.
    glob_file_string: str
        String of the names of the files to delete.
    num_subdir: int
        Number of subdirectory layers in the main directory.

    Examples
    --------
    <<<<< start example >>>>>

        import fronts.file_manager as fm

        main_dir = 'C:/Users/username/data_files'
        glob_file_string = '*matching_string.pkl'
        num_subdir = 3  # Check in the 3rd level of the directories within the main directory

        fm.delete_grouped_files(main_dir, glob_file_string, num_subdir)

    <<<<< end example >>>>>
    """

    ########################################### Check the parameters for errors ########################################
    if not isinstance(main_dir, str):
        raise TypeError(f"main_dir must be a string, received {type(main_dir)}")
    if not isinstance(glob_file_string, str):
        raise TypeError(f"glob_file_string must be a string, received {type(glob_file_string)}")
    if not isinstance(num_subdir, int):
        raise TypeError(f"num_subdir must be an integer, received {type(num_subdir)}")
    ####################################################################################################################

    subdir_string = ''  # This string will be modified depending on the provided value of num_subdir
    for _ in range(num_subdir):
        subdir_string += '/*'
    subdir_string += '/'
    glob_file_string = subdir_string + glob_file_string  # String that will be used to match with patterns in filenames

    files_to_delete = list(sorted(glob("%s%s" % (main_dir, glob_file_string))))  # Search for files in the given directory that have patterns matching the file string

    # Delete all the files
    print("Deleting %d files...." % len(files_to_delete), end='')
    for file in files_to_delete:
        try:
            os.remove(file)
        except PermissionError:
            shutil.rmtree(file)
    print("done")


def extract_tarfile(main_dir: str, 
                    tar_filename: str):
    """
    Extract all the contents of a TAR file.

    Parameters
    ----------
    main_dir: str
        Main directory where the TAR file is located. This is also where the extracted files will be placed.
    tar_filename: str
        Name of the compressed TAR file. Do NOT include the .tar.gz extension.

    Examples
    --------
    <<<<< start example >>>>

        import fronts.file_manager as fm

        main_dir = 'C:/Users/username/data_files'
        tar_filename = 'foo_tarfile'  # Do not add the .tar.gz extension

        fm.extract_tarfile(main_dir, glob_file_string, tar_filename, remove_files=True, status_printout=False)  # Compress files and remove them after compression into a TAR file

    <<<<< end example >>>>>
    """

    ########################################### Check the parameters for errors ########################################
    if not isinstance(main_dir, str):
        raise TypeError(f"main_dir must be a string, received {type(main_dir)}")
    if not isinstance(tar_filename, str):
        raise TypeError(f"tar_filename must be a string, received {type(tar_filename)}")
    ####################################################################################################################

    with tarfile.open(f"{main_dir}/{tar_filename}.tar.gz", 'r') as tarF:
        tarF.extractall(main_dir)
    print(f"Successfully extracted {main_dir}/{tar_filename}")


class DataFileLoader:
    """
    Objects that loads and manages various types of files containing weather data.
    """
    def __init__(
        self,
        file_dir: str,
        data_type: str,
        file_format: str,
        years = None,
        months = None,
        days = None,
        hours = None,
        domains = None):
        """
        file_dir: str
            Parent directory for the first set of various data files to load.
        data_type: str
            The source/type of the data files to load. Options for the data sources are: 'era5', 'fronts', 'gdas', 'gfs', 'nam-12km', 'satellite'.
        file_format: str
            Formatting of the data files to load. Options for the file/dataset type string are: 'grib', 'netcdf', 'tensorflow'.
                Note that this CANNOT be changed after it has been set, you must create an entirely separate DataFileLoader
                object to load a separate file type.
        years: int or iterable of ints, optional
            Year(s) to select from the available files.
        months: int or iterable of ints, optional
            Month(s) to select from the avilable files.
        days: int or iterable of ints, optional
            Day(s) to select from the available files.
        hours: int or iterable of ints, optional
            Hour(s) to select from the available files.
        domains: str or iterable of strs, optional
            Domain(s) to select from the available files.
        """
        
        self._file_format = file_format
        self._years = years
        self._months = months
        self._days = days
        self._hours = hours
        self._domains = domains
        
        self._format_args()
        
        if data_type == 'fronts' and self._file_format != 'tensorflow':
            data_type = 'FrontObjects'
        
        if self._file_format == 'grib':
            self._file_extension = '.grib'
        elif self._file_format == 'netcdf':
            self._file_extension = '.nc'
        elif self._file_format == 'tensorflow':
            self._file_extension = '_tf'
        else:
            raise ValueError('Unknown file type: %s' % self._file_format)
        
        if self._file_format in ['grib', 'netcdf']:
            glob_strs = [f'{file_dir}/{yr}{mo}/{data_type}_{yr}{mo}{dy}{hr}_{domain}{self._file_extension}'
                         for yr in self._years
                         for mo in self._months
                         for dy in self._days
                         for hr in self._hours
                         for domain in self._domains]
        else:
            glob_strs = [f'{file_dir}/{data_type}_{yr}{mo}{self._file_extension}'
                         for yr in self._years
                         for mo in self._months]
        
        # remove consecutive asterisks to prevent recursive searches
        glob_strs = [glob_str.replace('****', '*').replace('***', '*').replace('**', '*') for glob_str in glob_strs]
        
        self.files = []
        for glob_str in glob_strs:
            self.files.extend(glob(glob_str))
        self.files = [sorted(self.files)]
        
    def add_file_list(self, file_dir, data_type):
        """
        Add another file list.
        """
        num_lists = len(self.files)  # the current number of file lists
        
        if data_type == 'fronts' and self._file_format != 'tensorflow':
            data_type = 'FrontObjects'
        
        if self._file_format in ['grib', 'netcdf']:
            glob_strs = [f'{file_dir}/{yr}{mo}/{data_type}_{yr}{mo}{dy}{hr}_{domain}{self._file_extension}'
                         for yr in self._years
                         for mo in self._months
                         for dy in self._days
                         for hr in self._hours
                         for domain in self._domains]
        else:  # tensorflow
            glob_strs = [f'{file_dir}/{data_type}_{yr}{mo}{self._file_extension}'
                         for yr in self._years
                         for mo in self._months]
        
        # remove consecutive asterisks to prevent recursive searches
        glob_strs = [glob_str.replace('****', '*').replace('***', '*').replace('**', '*') for glob_str in glob_strs]
        
        current_file_list = []
        for glob_str in glob_strs:
            current_file_list.extend(glob(glob_str))
        current_file_list = sorted(current_file_list)
        
        # file basenames
        control_basename_list = [os.path.basename(file) for file in self.files[0]]
        current_basename_list = [os.path.basename(file) for file in current_file_list]
        
        # get details contained within the file names
        control_basename_info = [file.replace(self._file_extension, '').split('_')[1:] for file in control_basename_list]
        current_basename_info = [file.replace(self._file_extension, '').split('_')[1:] for file in current_basename_list]
        
        # if the file info has length 2, there is no forecast hour tag, so we need to add it
        if len(control_basename_info[0]) == 2:
            [file_info.insert(1, 'f000') for file_info in control_basename_info]
        if len(current_basename_info[0]) == 2:
            [file_info.insert(1, 'f000') for file_info in current_basename_info]
        
        # find indices where all file details match
        index_pairs = np.array([[data_idx, current_basename_info.index(file_info)] for data_idx, file_info in enumerate(control_basename_info) if file_info in current_basename_info])
        
        # filter files and add new list
        self.files = [[self.files[list_num][i] for i in index_pairs[:, 0]] for list_num in range(num_lists)]
        current_file_list = [current_file_list[i] for i in index_pairs[:, 1]]
        self.files.append(current_file_list)
        
    def _format_args(self):
        
        if isinstance(self._years, int):
            self._years = [f'{self._years:04d}', ]
        elif self._years is not None:
            self._years = ['%04d' % yr for yr in self._years]
        else:
            self._years = ['*', ]
        
        if isinstance(self._months, int):
            self._months = [f'{self._months:02d}', ]
        elif self._months is not None:
            self._months = ['%02d' % mo for mo in self._months]
        else:
            self._months = ['*', ]

        if isinstance(self._days, int):
            self._days = [f'{self._days:02d}', ]
        elif self._days is not None:
            self._days = ['%02d' % dy for dy in self._days]
        else:
            self._days = ['*', ]

        if isinstance(self._hours, int):
            self._hours = [f'{self._hours:02d}', ]
        elif self._hours is not None:
            self._hours = ['%02d' % hr for hr in self._hours]
        else:
            self._hours = ['*', ]

        if isinstance(self._domains, str):
            self._domains = [self._domains, ]
        elif self._domains is None:
            self._domains = ['*', ]



def load_model(model_number: int,
               model_dir: str):
    """
    Load a saved model.

    Parameters
    ----------
    model_number: int
        Slurm job number for the model. This is the number in the model's filename.
    model_dir: str
        Main directory for the models.
    """

    ######################################### Check the parameters for errors ##########################################
    if not isinstance(model_number, int):
        raise TypeError(f"model_number must be an integer, received {type(model_number)}")
    if not isinstance(model_dir, str):
        raise TypeError(f"model_dir must be a string, received {type(model_dir)}")
    ####################################################################################################################

    from tensorflow.keras.models import load_model as lm
    import custom_activations
    import custom_losses
    import custom_metrics

    model_path = f"{model_dir}/model_{model_number}/model_{model_number}.h5"
    model_properties = pd.read_pickle(f"{model_dir}/model_{model_number}/model_{model_number}_properties.pkl")

    custom_objects = {}

    loss_args = model_properties["loss_args"]
    loss_parent_string = model_properties["loss_parent_string"]
    loss_child_string = model_properties["loss_child_string"]

    metric_args = model_properties["metric_args"]
    metric_parent_string = model_properties["metric_parent_string"]
    metric_child_string = model_properties["metric_child_string"]

    # add the loss and metric functions to the custom_objects dictionary
    custom_objects[loss_child_string] = getattr(custom_losses, loss_parent_string)(**loss_args)
    custom_objects[metric_child_string] = getattr(custom_metrics, metric_parent_string)(**metric_args)

    # add the activation function to the custom_objects dictionary
    activation_string = model_properties['activation']
    if activation_string in ["elliott", "gaussian", "gcu", "hexpo", "isigmoid", "lisht", "psigmoid", "ptanh", "ptelu", "resech",
                             "smelu", "snake", "srs", "stanh"]:
        if activation_string == "elliott":
            activation = custom_activations.Elliott()
        elif activation_string == "gaussian":
            activation = custom_activations.Gaussian()
        elif activation_string == "gcu":
            activation = custom_activations.GCU()
        elif activation_string == "hexpo":
            activation = custom_activations.Hexpo()
        elif activation_string == "isigmoid":
            activation = custom_activations.ISigmoid()
        elif activation_string == "lisht":
            activation = custom_activations.LiSHT()
        elif activation_string == "psigmoid":
            activation = custom_activations.PSigmoid()
        elif activation_string == "ptanh":
            activation = custom_activations.PTanh()
        elif activation_string == "ptelu":
            activation = custom_activations.PTELU()
        elif activation_string == "resech":
            activation = custom_activations.ReSech()
        elif activation_string == "smelu":
            activation = custom_activations.SmeLU()
        elif activation_string == "snake":
            activation = custom_activations.Snake()
        elif activation_string == "srs":
            activation = custom_activations.SRS()
        else:  # activation_string == "stanh"
            activation = custom_activations.STanh()
        custom_objects[activation.__class__.__name__] = activation

    return lm(model_path, custom_objects=custom_objects)


if __name__ == '__main__':
    """
    Warnings
        Do not use leading zeros when declaring the month, day, and hour in 'date'. (ex: if the day is 2, do not type 02)
        Longitude values in the 'new_extent' argument must in the 360-degree coordinate system.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--compress_files', action='store_true', help='Compress files')
    parser.add_argument('--delete_grouped_files', action='store_true', help='Delete a set of files')
    parser.add_argument('--extract_tarfile', action='store_true', help='Extract a TAR file')
    parser.add_argument('--glob_file_string', type=str, help='String of the names of the files to compress or delete.')
    parser.add_argument('--main_dir', type=str, help='Main directory for subdirectory creation or where the files in question are located.')
    parser.add_argument('--num_subdir', type=int, help='Number of subdirectory layers in the main directory.')
    parser.add_argument('--tar_filename', type=str, help='Name of the TAR file.')
    args = parser.parse_args()
    provided_arguments = vars(args)
    
    if args.compress_files:
        compress_files(args.main_dir, args.glob_file_string, args.tar_filename)

    if args.delete_grouped_files:
        delete_grouped_files(args.main_dir, args.glob_file_string, args.num_subdir)

    if args.extract_tarfile:
        extract_tarfile(args.main_dir, args.tar_filename)
