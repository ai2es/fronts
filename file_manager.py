"""
Functions in this code manage data files and models.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.11.8
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
    with tarfile.open(f"{main_dir}/{tar_filename}.tar.gz", "w:gz") as tarF:

        ### Iterate through all the available files ###
        for file in range(num_files):
            tarF.add(files[file], arcname=files[file].replace(main_dir, ''))  # Add the file to the TAR file
            tarF_size = os.path.getsize(f"{main_dir}/{tar_filename}.tar.gz")/1e6  # Compressed size of the files within the TAR file (megabytes)
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
    Object that loads and manages ERA5, GDAS, GFS, and front object files.
    """
    def __init__(
        self, 
        file_dir: str, 
        data_file_type: str, 
        synoptic_only: bool = False):
        """
        When the DataFileLoader object is created, find all netCDF or tensorflow datasets.

        Parameters
        ----------
        file_dir: str
            Input directory for the netCDF or tensorflow datasets.
        data_file_type: str
            This string will contain two parts, separated by a hyphen: the source for the variable data, and the type of file/dataset.
            Options for the variable data sources are: 'era5', 'fronts', 'gdas', and 'gfs'.
            Options for the file/dataset type string are: 'netcdf', 'tensorflow'.
        synoptic_only: bool
            Setting this to True will remove any files with timesteps at non-synoptic hours (3, 9, 15, 21z).

        Examples
        --------
        <<<<< start example >>>>

            import file_manager as fm

            file_dir = '/home/user/data'  # Directory where the data is stored
            data_file_type = 'era5-netcdf'  # We want to load netCDF files containing ERA5 data

            era5_file_obj = fm.DataFileLoader(file_dir, data_file_type)  # Load the data files

        <<<<< end example >>>>
        """

        ######################################### Check the parameters for errors ######################################
        if not isinstance(file_dir, str):
            raise TypeError(f"file_dir must be a string, received {type(file_dir)}")
        if not isinstance(data_file_type, str):
            raise TypeError(f"data_file_type must be a string, received {type(data_file_type)}")
        if not isinstance(synoptic_only, bool):
            raise TypeError(f"synoptic_only must be a boolean, received {type(synoptic_only)}")
        ################################################################################################################

        valid_data_sources = ['era5', 'ecmwf', 'fronts', 'gdas', 'gfs']
        valid_file_types = ['netcdf', 'tensorflow']

        data_file_type = data_file_type.lower().split('-')
        self._file_type = data_file_type[1]

        if self._file_type == 'netcdf':
            self._file_extension = '.nc'
            self._subdir_glob = '/*/'
        elif self._file_type == 'tensorflow':
            self._file_extension = '_tf'
            self._subdir_glob = '/'
        else:
            raise TypeError(f"'%s' is not a valid file type, valid types are: {', '.join(valid_file_types)}" % self._file_type)

        if data_file_type[0] not in valid_data_sources:
            raise TypeError(f"'%s' is not a valid data source, valid sources are: {', '.join(valid_data_sources)}" % data_file_type[0])

        if data_file_type[0] == 'fronts':
            self._file_prefix = 'FrontObjects'
        else:
            self._file_prefix = data_file_type[0]

        self._all_data_files = sorted(glob("%s%s%s*%s" % (file_dir, self._subdir_glob, self._file_prefix, self._file_extension)))  # All data files without filtering
        self._data_file_information = [os.path.basename(file).split('_')[1:] for file in self._all_data_files]  # timesteps in the unfiltered data files
        [data_file_info.insert(1, 'f000') for data_file_info in self._data_file_information if len(data_file_info) == 2]

        if synoptic_only:
            ### Filter out non-synoptic hours (3, 9, 15, 21z) ###
            for file_info in self._data_file_information:
                if any('%02d' % hour in file_info[0][-2:] for hour in [3, 9, 15, 21]):
                    self._all_data_files.pop(self._data_file_information.index(file_info))
                    self._data_file_information.pop(self._data_file_information.index(file_info))

        ### All available options for specific filters ###
        self._all_forecast_hours = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        self._all_years = (2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023)

        self.reset_all_filters()  # Resetting the filters simply creates the list of data files

        ### Current values for the filters used in the files ###
        self._forecast_hours = self._all_forecast_hours
        self._training_years = self._all_years
        self._validation_years = self._all_years
        self._test_years = self._all_years

    def reset_all_filters(self):
        """
        Reset the lists of files back to their original states with no filters
        """

        if self._file_prefix == 'FrontObjects':
            self.front_files = self._all_data_files
            self.front_files_training = self.front_files
            self.front_files_validation = self.front_files
            self.front_files_test = self.front_files
        else:
            self.data_files = self._all_data_files
            self.data_files_training = self.data_files
            self.data_files_validation = self.data_files
            self.data_files_test = self.data_files

        ### If front object files have been loaded to be paired with variable files, reset the front file lists ###
        if hasattr(self, '_all_front_files'):
            self.front_files = self._all_front_files
            self.front_files_training = self.front_files
            self.front_files_validation = self.front_files
            self.front_files_test = self.front_files

    ####################################################################################################################

    def __get_training_years(self):
        """
        Return the list of training years used
        """

        return self._training_years

    def __set_training_years(self, training_years: tuple | list):
        """
        Select the training years to load
        """

        self.__reset_training_years()  # Return file list to last state before training years were modified (no effect is this is first training year selection)

        self._training_years = training_years

        ### Check that all selected training years are valid ###
        invalid_training_years = [year for year in training_years if year not in self._all_years]
        if len(invalid_training_years) > 0:
            raise TypeError(f"The following training years are not valid: {','.join(sorted(invalid_training_years))}")

        self._training_years_not_in_data = [year for year in self._all_years if year not in training_years]

        ### Remove unwanted years from the list of files ###
        for year in self._training_years_not_in_data:
            if self._file_prefix == 'FrontObjects':
                self.front_files_training = [file for file in self.front_files_training if '_%s' % str(year) not in file]
            else:
                self.data_files_training = [file for file in self.data_files_training if '_%s' % str(year) not in file]

    def __reset_training_years(self):
        """
        Reset training years
        """

        if not hasattr(self, '_filtered_data_files_before_training_year_selection'):  # If the training years have not been selected yet
            if self._file_prefix == 'FrontObjects':
                self._filtered_data_files_before_training_year_selection = self.front_files_training
            else:
                self._filtered_data_files_before_training_year_selection = self.data_files_training
        else:  # Return file list to last state before training years were modified
            if self._file_prefix == 'FrontObjects':
                self.front_files_training = self._filtered_data_files_before_training_year_selection
            else:
                self.data_files_training = self._filtered_data_files_before_training_year_selection

    training_years = property(__get_training_years, __set_training_years)  # Property method for setting training years

    ####################################################################################################################

    def __get_validation_years(self):
        """
        Return the list of validation years used
        """

        return self._validation_years

    def __set_validation_years(self, validation_years: tuple | list):
        """
        Select the validation years to load
        """

        self.__reset_validation_years()  # Return file list to last state before validation years were modified (no effect is this is first validation year selection)

        self._validation_years = validation_years

        ### Check that all selected validation years are valid ###
        invalid_validation_years = [year for year in validation_years if year not in self._all_years]
        if len(invalid_validation_years) > 0:
            raise TypeError(f"The following validation years are not valid: {','.join(sorted(invalid_validation_years))}")

        self._validation_years_not_in_data = [year for year in self._all_years if year not in validation_years]

        ### Remove unwanted years from the list of files ###
        for year in self._validation_years_not_in_data:
            if self._file_prefix == 'FrontObjects':
                self.front_files_validation = [file for file in self.front_files_validation if '_%s' % str(year) not in file]
            else:
                self.data_files_validation = [file for file in self.data_files_validation if '_%s' % str(year) not in file]

    def __reset_validation_years(self):
        """
        Reset validation years
        """

        if not hasattr(self, '_filtered_data_files_before_validation_year_selection'):  # If the validation years have not been selected yet
            if self._file_prefix == 'FrontObjects':
                self._filtered_data_files_before_validation_year_selection = self.front_files_validation
            else:
                self._filtered_data_files_before_validation_year_selection = self.data_files_validation
        else:  # Return file list to last state before validation years were modified
            if self._file_prefix == 'FrontObjects':
                self.front_files_validation = self._filtered_data_files_before_validation_year_selection
            else:
                self.data_files_validation = self._filtered_data_files_before_validation_year_selection

    validation_years = property(__get_validation_years, __set_validation_years)  # Property method for setting validation years

    ####################################################################################################################

    def __get_test_years(self):
        """
        Return the list of test years used
        """

        return self._test_years

    def __set_test_years(self, test_years: tuple | list):
        """
        Select the test years to load
        """

        self.__reset_test_years()  # Return file list to last state before test years were modified (no effect is this is first test year selection)

        self._test_years = test_years

        ### Check that all selected test years are valid ###
        invalid_test_years = [year for year in test_years if year not in self._all_years]
        if len(invalid_test_years) > 0:
            raise TypeError(f"The following test years are not valid: {','.join(sorted(invalid_test_years))}")

        self._test_years_not_in_data = [year for year in self._all_years if year not in test_years]

        ### Remove unwanted years from the list of files ###
        for year in self._test_years_not_in_data:
            if self._file_prefix == 'FrontObjects':
                self.front_files_test = [file for file in self.front_files_test if '_%s' % str(year) not in file]
            else:
                self.data_files_test = [file for file in self.data_files_test if '_%s' % str(year) not in file]

    def __reset_test_years(self):
        """
        Reset test years
        """

        if not hasattr(self, '_filtered_data_files_before_test_year_selection'):  # If the test years have not been selected yet
            if self._file_prefix == 'FrontObjects':
                self._filtered_data_files_before_test_year_selection = self.front_files_test
            else:
                self._filtered_data_files_before_test_year_selection = self.data_files_test
        else:  # Return file list to last state before test years were modified
            if self._file_prefix == 'FrontObjects':
                self.front_files_test = self._filtered_data_files_before_test_year_selection
            else:
                self.data_files_test = self._filtered_data_files_before_test_year_selection

    test_years = property(__get_test_years, __set_test_years)  # Property method for setting test years

    ####################################################################################################################

    def __get_forecast_hours(self):
        """
        Return the list of forecast hours used
        """

        if self._file_type == 'era5':
            return None
        else:
            return self._forecast_hours

    def __set_forecast_hours(self, forecast_hours: tuple | list):
        """
        Select the forecast hours to load
        """

        self.__reset_forecast_hours()  # Return file list to last state before forecast hours were modified (no effect is this is first forecast hour selection)

        self._forecast_hours = forecast_hours

        ### Check that all selected forecast hours are valid ###
        invalid_forecast_hours = [hour for hour in forecast_hours if hour not in self._all_forecast_hours]
        if len(invalid_forecast_hours) > 0:
            raise TypeError(f"The following forecast hours are not valid: {','.join(sorted(str(hour) for hour in invalid_forecast_hours))}")

        self._forecast_hours_not_in_data = [hour for hour in self._all_forecast_hours if hour not in forecast_hours]

        ### Remove unwanted forecast hours from the list of files ###
        for hour in self._forecast_hours_not_in_data:
            self.data_files = [file for file in self.data_files if '_f%03d_' % hour not in file]
            self.data_files_training = [file for file in self.data_files_training if '_f%03d_' % hour not in file]
            self.data_files_validation = [file for file in self.data_files_validation if '_f%03d_' % hour not in file]
            self.data_files_test = [file for file in self.data_files_test if '_f%03d_' % hour not in file]

    def __reset_forecast_hours(self):
        """
        Reset forecast hours in the data files
        """

        if not hasattr(self, '_filtered_data_files_before_forecast_hour_selection'):  # If the forecast hours have not been selected yet
            self._filtered_data_files_before_forecast_hour_selection = self.data_files
            self._filtered_data_training_files_before_forecast_hour_selection = self.data_files_training
            self._filtered_data_validation_files_before_forecast_hour_selection = self.data_files_validation
            self._filtered_data_test_files_before_forecast_hour_selection = self.data_files_test
        else:
            ### Return file lists to last state before forecast hours were modified ###
            self.data_files = self._filtered_data_files_before_forecast_hour_selection
            self.data_files_training = self._filtered_data_training_files_before_forecast_hour_selection
            self.data_files_validation = self._filtered_data_validation_files_before_forecast_hour_selection
            self.data_files_test = self._filtered_data_test_files_before_forecast_hour_selection

    forecast_hours = property(__get_forecast_hours, __set_forecast_hours)  # Property method for setting forecast hours

    ####################################################################################################################

    def __sort_files_by_dataset(self, data_timesteps_used, sort_fronts=False):
        """
        Filter files for the training, validation, and test datasets. This is done by finding indices for each timestep in the 'data_timesteps_used' list.

        data_timesteps_used: list
            List of timesteps used when sorting variable and/or frontal object files.
        sort_fronts: bool
            Setting this to True will sort the lists of frontal object files. This will only be True when sorting both variable and frontal object files at the
                same time.
        """

        ### Find all indices where timesteps for training, validation, and test datasets are present in the selected variable files ###
        training_indices = [index for index, timestep in enumerate(data_timesteps_used) if any('%d' % training_year in timestep[:4] for training_year in self._training_years)]
        validation_indices = [index for index, timestep in enumerate(data_timesteps_used) if any('%d' % validation_year in timestep[:4] for validation_year in self._validation_years)]
        test_indices = [index for index, timestep in enumerate(data_timesteps_used) if any('%d' % test_year in timestep[:4] for test_year in self._test_years)]

        ### Create new variable file lists for training, validation, and test datasets using the indices pulled from above ###
        self.data_files_training = [self.data_files[index] for index in training_indices]
        self.data_files_validation = [self.data_files[index] for index in validation_indices]
        self.data_files_test = [self.data_files[index] for index in test_indices]

        if sort_fronts:
            ### Create new frontal object file lists for training, validation, and test datasets using the indices pulled from above ###
            self.front_files_training = [self.front_files[index] for index in training_indices]
            self.front_files_validation = [self.front_files[index] for index in validation_indices]
            self.front_files_test = [self.front_files[index] for index in test_indices]

    def pair_with_fronts(self, front_indir, match='same', underscore_skips=0):
        """
        Parameters
        ----------
        front_indir: str
            Directory where the frontal object files are stored.
        match: 'same', 'data-forecast', 'fronts-forecast', 'forecast', or 'any'
            Method to use when creating the final paired file lists.
            * 'same' will only match variable and front files that have the same file information (timestep, forecast hour, domain)
            * 'data-forecast' will attempt to match variable files with forecast hours greater than 0 to front files with forecast hours of 0.
                For example, gfs_2023033100_f006_global.nc will be paired with FrontObjects_2023033106_f000_global.nc.
            * 'fronts-forecast' will attempt to match front files with forecast hours greater than 0 to variable files with forecast hours of 0.
                For example, FrontObjects_2023033100_f006_global.nc will be paired with gfs_2023033106_f000_global.nc.
            * 'forecast' will perform actions in 'data-forecast' and 'fronts-forecast'
            * 'any' will perform the actions in both 'same' and 'forecast'.
        underscore_skips: int
            This argument is meant to make older tensorflow datasets for front objects compatible with the file loader object.
            If using older tensorflow datasets with the front types in the filenames, this argument should be equal to the number of front types.
        """

        if self._file_prefix == 'FrontObjects':
            print("WARNING: 'DataFileLoader.pair_with_fronts' can only be used with ERA5, GDAS, or GFS files.")
            return

        ######################################### Check the parameters for errors ######################################
        if not isinstance(front_indir, str):
            raise TypeError(f"front_indir must be a string, received {type(front_indir)}")
        if match not in ['same', 'data-forecast', 'fronts-forecast', 'forecast', 'any']:
            raise ValueError("Invalid match type: '%s'" % match)
        ################################################################################################################

        if self._file_type == 'netcdf':
            file_prefix = 'FrontObjects'
        elif self._file_type == 'tensorflow':
            file_prefix = 'fronts'
            if match != 'same':
                raise TypeError(f"Matching method must be 'same' with tensorflow datasets. Received: {match}")
        else:
            raise ValueError(f"The available options for 'file_type' are: 'netcdf', 'tensorflow'. Received: {self._file_type}")

        self._all_front_files = sorted(glob("%s%s%s*%s" % (front_indir, self._subdir_glob, file_prefix, self._file_extension)))  # All front files without filtering
        front_file_information = [os.path.basename(file).split('_')[1 + underscore_skips:] for file in self._all_front_files]

        # Add forecast hour 0 to file information if a forecast hour is not in the filename(s)
        [front_file_info.insert(1, 'f000') for front_file_info in front_file_information if len(front_file_info) == 2]

        if self._file_type == 'tensorflow':
            # Dates in tensorflow dataset filenames are only the year and month, so we will add a day and hour to the file information
            self._data_file_information = [[data_file_info[0] + '0100', *data_file_info[1:]] for data_file_info in self._data_file_information]
            front_file_information = [[front_file_info[0] + '0100', *front_file_info[1:]] for front_file_info in front_file_information]

        new_data_file_list = []
        new_front_file_list = []
        data_timesteps_used = []  # timesteps used in the data; used to sort into training/validation/test datasets

        if match == 'same':
            # Indices in the data and front file lists where all file information is the same
            index_pairs = np.array([[data_idx, front_file_information.index(file_info)] for data_idx, file_info in enumerate(self._data_file_information)
                                    if file_info in front_file_information])

            # Files corresponding to the above indices
            _data_files = [self._all_data_files[idx] for idx in index_pairs[:, 0]]
            _front_files = [self._all_front_files[idx] for idx in index_pairs[:, 1]]

            # Update file information to only include information for the selected files
            self._data_file_information = [self._data_file_information[idx] for idx in index_pairs[:, 0]]
            front_file_information = [front_file_information[idx] for idx in index_pairs[:, 1]]

        if 'forecast' in match:

            data_file_info_to_retain = []
            front_file_info_to_retain = []

            if match == 'data-forecast':
                # Remove all data files with forecast hours equal to 0 and all front files with forecast hours NOT equal to 0
                data_file_info_to_retain.append([file_info for file_info in self._data_file_information if file_info[1] != 'f000'])
                front_file_info_to_retain.append([file_info for file_info in front_file_information if file_info[1] == 'f000'])

            elif match == 'fronts-forecast':
                # Remove all data files with forecast hours NOT equal to 0 and all front files with forecast hours equal to 0
                data_file_info_to_retain.append([file_info for file_info in self._data_file_information if file_info[1] == 'f000'])
                front_file_info_to_retain.append([file_info for file_info in front_file_information if file_info[1] != 'f000'])

            _data_files = [self._all_data_files[self._data_file_information.index(file_info)] for file_info in data_file_info_to_retain]
            _front_files = [self._all_front_files[front_file_information.index(file_info)] for file_info in front_file_info_to_retain]

            self._data_file_information = [self._data_file_information[self._data_file_information.index(file_info)] for file_info in data_file_info_to_retain]
            front_file_information = [front_file_information[front_file_information.index(file_info)] for file_info in front_file_info_to_retain]

        # Valid time = initialization time + forecast hours
        data_file_valid_times = [np.datetime64(
            '%s-%s-%sT%s' % (file_info[0][:4], file_info[0][4:6], file_info[0][6:8], file_info[0][8:]))
                                 + np.timedelta64(int(file_info[1][1:])) for file_info in
                                 self._data_file_information]
        front_file_valid_times = [np.datetime64(
            '%s-%s-%sT%s' % (file_info[0][:4], file_info[0][4:6], file_info[0][6:8], file_info[0][8:]))
                                  + np.timedelta64(int(file_info[1][1:])) for file_info in
                                  front_file_information]

        if match == 'any':
            _data_files = [self._all_data_files[self._data_file_information.index(file_info)] for file_info in self._data_file_information]
            _front_files = [self._all_front_files[front_file_information.index(file_info)] for file_info in front_file_information]

            index_pairs = [[idx, np.where(front_file_valid_times == data_file_valid_time)[0]] for idx, data_file_valid_time in enumerate(data_file_valid_times)]

            [new_data_file_list.append(_data_files[idx_pair[0]]) for idx_pair in index_pairs if len(idx_pair[1]) > 0 for _ in range(len(idx_pair[1]))]
            [new_front_file_list.append(_front_files[idx]) for idx_pair in index_pairs if len(idx_pair[1]) > 0 for idx in idx_pair[1]]
            [data_timesteps_used.append(pd.to_datetime(front_file_valid_times[idx]).strftime('%Y%m%d%H')) for idx_pair in index_pairs if len(idx_pair[1]) > 0 for idx in idx_pair[1]]

        else:
            _, data_indices, front_indices = np.intersect1d(data_file_valid_times, front_file_valid_times, return_indices=True)

            [new_data_file_list.append(_data_files[idx]) for idx in data_indices]
            [new_front_file_list.append(_front_files[idx]) for idx in front_indices]
            [data_timesteps_used.append(pd.to_datetime(data_file_valid_times[idx]).strftime('%Y%m%d%H')) for idx in data_indices]

        self.data_files = new_data_file_list
        self.front_files = new_front_file_list

        self.__sort_files_by_dataset(data_timesteps_used, sort_fronts=True)


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

    try:
        loss_string = model_properties['loss_string']
    except KeyError:
        loss_string = model_properties['loss']  # Error in the training sometimes resulted in the incorrect key ('loss' instead of 'loss_string')
    loss_args = model_properties['loss_args']

    try:
        metric_string = model_properties['metric_string']
    except KeyError:
        metric_string = model_properties['metric']  # Error in the training sometimes resulted in the incorrect key ('metric' instead of 'metric_string')
    metric_args = model_properties['metric_args']

    custom_objects = {}

    if 'fss' in loss_string.lower():
        if model_number in [6846496, 7236500, 7507525]:
            loss_string = 'fss_loss'
        custom_objects[loss_string] = custom_losses.fractions_skill_score(**loss_args)

    if 'brier' in metric_string.lower() or 'bss' in metric_string.lower():
        if model_number in [6846496, 7236500, 7507525]:
            metric_string = 'bss'
        custom_objects[metric_string] = custom_metrics.brier_skill_score(**metric_args)

    if 'csi' in metric_string.lower():
        custom_objects[metric_string] = custom_metrics.critical_success_index(**metric_args)

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
