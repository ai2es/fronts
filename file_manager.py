"""
Functions in this code manage data files.
    1. Compress a large set of files into a TAR file.
    2. Extract the contents of an existing TAR file.
    3. Extract members from a GDAS TAR file.
    4. Delete a large group of files using a common string in the filenames.
    5. Load ERA5, GDAS, and frontal object netCDF files.
    6. Load a saved tensorflow model.

TODO:
    * Add functions for managing GFS data once the data is obtained

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 8/24/2022 10:52 AM CT
"""

from glob import glob
import tarfile
import argparse
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model as lm
import custom_losses
from errors import check_arguments
from utils import data_utils


def compress_files(main_dir, glob_file_string, tar_filename, remove_files=False, status_printout=True):
    """
    Compress files into a TAR file.

    Parameters
    ----------
    main_dir: str
        - Main directory where the files are located and where the TAR file will be saved.
    glob_file_string: str
        - String of the names of the files to compress.
    tar_filename:
        - Name of the compressed TAR file that will be made. Do not include the .tar.gz extension in the name, this is added automatically.
    remove_files: bool
        - Setting this to true will remove the files after they have been compressed to a TAR file.
    status_printout: bool
        - Setting this to true will provide printouts of the status of the compression.

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

        ### Iterate through all of the available files ###
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

    subdir_string = ''  # This string will be modified depending on the provided value of num_subdir
    for i in range(num_subdir):
        subdir_string += '/*'
    subdir_string += '/'
    glob_file_string = subdir_string + glob_file_string  # String that will be used to match with patterns in filenames

    files_to_delete = list(sorted(glob("%s/%s" % (main_dir, glob_file_string))))  # Search for files in the given directory that have patterns matching the file string

    # Delete all the files
    print("Deleting %d files...." % len(files_to_delete), end='')
    for file in files_to_delete:
        os.remove(file)
    print("done")


def extract_tarfile(main_dir, tar_filename):
    """
    Extract all the contents of a TAR file.

    Parameters
    ----------
    main_dir: str
        - Main directory where the TAR file is located. This is also where the extracted files will be placed.
    tar_filename: str
        - Name of the compressed TAR file. Do NOT include the .tar.gz extension.

    Examples
    --------
    <<<<< start example >>>>

        import fronts.file_manager as fm

        main_dir = 'C:/Users/username/data_files'
        tar_filename = 'foo_tarfile'  # Do not add the .tar.gz extension

        fm.extract_tarfile(main_dir, glob_file_string, tar_filename, remove_files=True, status_printout=False)  # Compress files and remove them after compression into a TAR file

    <<<<< end example >>>>>
    """

    with tarfile.open(f"{main_dir}/{tar_filename}.tar.gz", 'r') as tarF:
        tarF.extractall(main_dir)
    print(f"Successfully extracted {main_dir}/{tar_filename}")


class ERA5files:
    """
    Object that loads and manages ERA5 netCDF files and datasets
    """
    def __init__(self, era5_netcdf_indir: str):
        """
        When the ERA5files object is created, open all ERA5 netCDF files

        era5_netcdf_indir: str
            - Input directory for the ERA5 netCDF files.
        """

        self._all_era5_netcdf_files = sorted(glob("%s/*/*/*/era5*.nc" % era5_netcdf_indir))  # All ERA5 files without filtering

        ### All available options for specific filters ###
        self._all_domains = ('full', 'conus')
        self._all_variables = ('q', 'r', 'RH', 'sp_z', 'T', 'Td', 'theta', 'theta_e', 'theta_v', 'theta_w', 'Tv', 'Tw', 'u', 'v')
        self._all_years = (2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022)

        self.reset_all_filters()  # Resetting the filters simply creates the ERA5 file lists

        ### Current values for the filters used in the files ###
        self._domains = self._all_domains
        self._variables = self._all_variables
        self._training_years = self._all_years[:11]  # default training years: 2006-2016
        self._validation_years = self._all_years[11:14]  # default validation years: 2017-2019
        self._test_years = self._all_years[14:]  # default test years: 2020-2022

    def reset_all_filters(self):
        """
        Reset the lists of ERA5 netCDF files and front object files
        """

        self.era5_files = self._all_era5_netcdf_files
        self.era5_files_training = [era5_file for era5_file in self._all_era5_netcdf_files if any('_%s' % str(year) in era5_file for year in self._all_years[:11])]
        self.era5_files_validation = [era5_file for era5_file in self._all_era5_netcdf_files if any('_%s' % str(year) in era5_file for year in self._all_years[11:14])]
        self.era5_files_test = [era5_file for era5_file in self._all_era5_netcdf_files if any('_%s' % str(year) in era5_file for year in self._all_years[14:])]

        ### If front files have been loaded to be paired with ERA5 files, reset the front file lists ###
        if hasattr(self, '_all_front_files'):
            self.front_files = self._all_front_files
            self.front_files_training = [front_file for front_file in self._all_front_files if any('_%s' % str(year) in front_file for year in self._all_years[:11])]
            self.front_files_validation = [front_file for front_file in self._all_front_files if any('_%s' % str(year) in front_file for year in self._all_years[11:14])]
            self.front_files_test = [front_file for front_file in self._all_front_files if any('_%s' % str(year) in front_file for year in self._all_years[14:])]

    ####################################################################################################################

    def __get_training_years(self):
        """
        Return the list of training years used
        """

        return self._training_years

    def __set_training_years(self, training_years: tuple or list):
        """
        Select the training years to load
        """

        self.__reset_training_years()  # Return file list to last state before training years were modified (no effect is this is first training year selection)

        self._training_years = training_years

        ### Check that all selected ERA5 training years are valid ###
        invalid_training_years = [year for year in training_years if year not in self._all_years]
        if len(invalid_training_years) > 0:
            raise TypeError(f"The following training years are not valid: {','.join(sorted(invalid_training_years))}")

        self._training_years_not_in_data = [year for year in self._all_years if year not in training_years]

        ### Remove unwanted years from the list of files ###
        for year in self._training_years_not_in_data:
            self.era5_files_training = [file for file in self.era5_files_training if '_%s' % str(year) not in file]

    def __reset_training_years(self):
        """
        Reset training years in the ERA5 files
        """

        if not hasattr(self, '_filtered_era5_files_before_training_year_selection'):  # If the training_years have not been selected yet
            self._filtered_era5_files_before_training_year_selection = self.era5_files_training
        else:
            self.era5_files_training = self._filtered_era5_files_before_training_year_selection  # Return file list to last state before training_years were modified

    training_years = property(__get_training_years, __set_training_years)  # Property method for setting training years

    ####################################################################################################################

    def __get_validation_years(self):
        """
        Return the list of validation years used
        """

        return self._validation_years

    def __set_validation_years(self, validation_years: tuple or list):
        """
        Select the validation years to load
        """

        self.__reset_validation_years()  # Return file list to last state before validation years were modified (no effect is this is first validation year selection)

        self._validation_years = validation_years

        ### Check that all selected ERA5 validation years are valid ###
        invalid_validation_years = [year for year in validation_years if year not in self._all_years]
        if len(invalid_validation_years) > 0:
            raise TypeError(f"The following validation years are not valid: {','.join(sorted(invalid_validation_years))}")

        self._validation_years_not_in_data = [year for year in self._all_years if year not in validation_years]

        ### Remove unwanted years from the list of files ###
        for year in self._validation_years_not_in_data:
            self.era5_files_validation = [file for file in self.era5_files_validation if '_%s' % str(year) not in file]

    def __reset_validation_years(self):
        """
        Reset validation years in the ERA5 files
        """

        if not hasattr(self, '_filtered_era5_files_before_validation_year_selection'):  # If the validation_years have not been selected yet
            self._filtered_era5_files_before_validation_year_selection = self.era5_files_validation
        else:
            self.era5_files_validation = self._filtered_era5_files_before_validation_year_selection  # Return file list to last state before validation_years were modified

    validation_years = property(__get_validation_years, __set_validation_years)  # Property method for setting validation years

    ####################################################################################################################

    def __get_test_years(self):
        """
        Return the list of test years used
        """

        return self._test_years

    def __set_test_years(self, test_years: tuple or list):
        """
        Select the test years to load
        """

        self.__reset_test_years()  # Return file list to last state before test years were modified (no effect is this is first test year selection)

        self._test_years = test_years

        ### Check that all selected ERA5 test years are valid ###
        invalid_test_years = [year for year in test_years if year not in self._all_years]
        if len(invalid_test_years) > 0:
            raise TypeError(f"The following test years are not valid: {','.join(sorted(invalid_test_years))}")

        self._test_years_not_in_data = [year for year in self._all_years if year not in test_years]

        ### Remove unwanted years from the list of files ###
        for year in self._test_years_not_in_data:
            self.era5_files_test = [file for file in self.era5_files_test if '_%s' % str(year) not in file]

    def __reset_test_years(self):
        """
        Reset test years in the ERA5 files
        """

        if not hasattr(self, '_filtered_era5_files_before_test_year_selection'):  # If the test_years have not been selected yet
            self._filtered_era5_files_before_test_year_selection = self.era5_files_test
        else:
            self.era5_files_test = self._filtered_era5_files_before_test_year_selection  # Return file list to last state before test_years were modified

    test_years = property(__get_test_years, __set_test_years)  # Property method for setting test years

    ####################################################################################################################

    def __get_variables(self):
        """
        Return the list of variables used
        """

        return self._variables

    def __set_variables(self, variables: tuple or list):
        """
        Select the variables to load
        """

        self.__reset_variables()  # Return file list to last state before variables were modified (no effect is this is first variable selection)

        self._variables = variables

        ### Check that all selected ERA5 variables are valid ###
        invalid_variables = [variable for variable in variables if variable not in self._all_variables]
        if len(invalid_variables) > 0:
            raise TypeError(f"The following variables are not valid: {','.join(sorted(invalid_variables))}")

        self._variables_not_in_data = [variable for variable in self._all_variables if variable not in variables]

        ### Remove unwanted variables from the list of files ###
        for variable in self._variables_not_in_data:
            self.era5_files = [file for file in self.era5_files if 'era5_%s_2' % variable not in file]
            self.era5_files_training = [file for file in self.era5_files_training if 'era5_%s_2' % variable not in file]
            self.era5_files_validation = [file for file in self.era5_files_validation if 'era5_%s_2' % variable not in file]
            self.era5_files_test = [file for file in self.era5_files_test if 'era5_%s_2' % variable not in file]

    def __reset_variables(self):
        """
        Reset variables in the ERA5 files
        """

        if not hasattr(self, '_filtered_era5_files_before_variable_selection'):  # If the variables have not been selected yet
            self._filtered_era5_files_before_variable_selection = self.era5_files
            self._filtered_era5_training_files_before_variable_selection = self.era5_files_training
            self._filtered_era5_validation_files_before_variable_selection = self.era5_files_validation
            self._filtered_era5_test_files_before_variable_selection = self.era5_files_test
        else:
            ### Return file lists to last state before variables were modified ###
            self.era5_files = self._filtered_era5_files_before_variable_selection
            self.era5_files_training = self._filtered_era5_training_files_before_variable_selection
            self.era5_files_validation = self._filtered_era5_validation_files_before_variable_selection
            self.era5_files_test = self._filtered_era5_test_files_before_variable_selection

    variables = property(__get_variables, __set_variables)  # Property method for setting variables

    ####################################################################################################################

    def __get_domains(self):
        """
        Return the list of domains used
        """

        return self._domains

    def __set_domains(self, domains: tuple or list):
        """
        Select the domains to load
        """

        self.__reset_domains()  # Return file list to last state before domains were modified (no effect is this is first domain selection)

        self._domains = domains

        ### Check that all selected ERA5 domains are valid ###
        invalid_domains = [domain for domain in domains if domain not in self._all_domains]
        if len(invalid_domains) > 0:
            raise TypeError(f"The following domains are not valid: {','.join(sorted(invalid_domains))}")

        self._domains_not_in_data = [domain for domain in self._all_domains if domain not in domains]

        ### Remove unwanted domains from the list of files ###
        for domain in self._domains_not_in_data:
            self.era5_files = [file for file in self.era5_files if '_%s.nc' % domain not in file]
            self.era5_files_training = [file for file in self.era5_files_training if '_%s.nc' % domain not in file]
            self.era5_files_validation = [file for file in self.era5_files_validation if '_%s.nc' % domain not in file]
            self.era5_files_test = [file for file in self.era5_files_test if '_%s.nc' % domain not in file]

    def __reset_domains(self):
        """
        Reset domains in the ERA5 files
        """

        if not hasattr(self, '_filtered_era5_files_before_domain_selection'):  # If the domains have not been selected yet
            self._filtered_era5_files_before_domain_selection = self.era5_files
            self._filtered_era5_training_files_before_domain_selection = self.era5_files_training
            self._filtered_era5_validation_files_before_domain_selection = self.era5_files_validation
            self._filtered_era5_test_files_before_domain_selection = self.era5_files_test
        else:
            ### Return file lists to last state before domains were modified ###
            self.era5_files = self._filtered_era5_files_before_domain_selection
            self.era5_files_training = self._filtered_era5_training_files_before_domain_selection
            self.era5_files_validation = self._filtered_era5_validation_files_before_domain_selection
            self.era5_files_test = self._filtered_era5_test_files_before_domain_selection

    domains = property(__get_domains, __set_domains)  # Property method for setting domains

    ####################################################################################################################

    def pair_with_fronts(self, front_indir):
        """
        Pair all of the ERA5 timesteps with frontal object files containing matching timesteps

        front_indir: str
            - Directory where the frontal object files are stored.
        """

        self._all_front_files = sorted(glob("%s/*/*/*/FrontObjects*.nc" % front_indir))  # All front files without filtering
        self.front_files = self._all_front_files

        ### Remove unwanted domains from the list of front files ###
        for domain in self._domains_not_in_data:
            self.front_files = [file for file in self.front_files if '_%s.nc' % domain not in file]

        front_files_timestep = []
        era5_timesteps = []  # Timesteps of ERA5/GFS files
        era5_files_by_timestep = []  # List of ERA5/GFS files with each index representing a group of files with a common timestep
        front_files_list = []
        era5_files_list = []

        ### Collect all timesteps in the fronts files ###
        for i in range(len(self.front_files)):
            front_filename_start_index = self.front_files[i].find('FrontObjects_')
            front_files_timestep.append(self.front_files[i][front_filename_start_index + 13:front_filename_start_index + 23])  # Timestep is first 10 characters after 'FrontObjects_'

        ### Find all timesteps in the ERA5 files ###
        for j in range(len(self.era5_files)):
            era5_filename_start_index = self.era5_files[j].find('era5_')
            era5_filename_no_variable_index = self.era5_files[j][era5_filename_start_index:].find('_2') + 1
            era5_timesteps.append(self.era5_files[j][era5_filename_start_index + era5_filename_no_variable_index:era5_filename_start_index + era5_filename_no_variable_index + 10])  # Timestep is the first 10 characters after the pressure level in the filename

        unique_era5_timesteps = list(np.unique(sorted(era5_timesteps)))

        ### Find all ERA5 files for each timestep ###
        for timestep in unique_era5_timesteps:
            files_for_timestep = sorted(filter(lambda filename: timestep in filename, self.era5_files))
            era5_files_by_timestep.append(files_for_timestep)

        ### File matching: only load files with common timesteps ###
        total_era5_files = len(era5_files_by_timestep)
        total_front_files = len(self.front_files)
        for era5_file_no in range(total_era5_files):
            if unique_era5_timesteps[era5_file_no] in front_files_timestep:
                era5_files_list.append(era5_files_by_timestep[era5_file_no])
        for front_file_no in range(total_front_files):
            if front_files_timestep[front_file_no] in unique_era5_timesteps:
                front_files_list.append(self.front_files[front_file_no])

        self.era5_files = era5_files_list
        self.front_files = front_files_list

        ### Iterate through the front files to search for indices within the list to delegate to the lists for training, validation and test datasets ###
        self.era5_files_training, self.front_files_training = [files for files in self.era5_files if any('_%s' % str(year) in files[0] for year in self._training_years)], \
                                                              [file for file in self.front_files if any('_%s' % str(year) in file for year in self._training_years)]
        self.era5_files_validation, self.front_files_validation = [files for files in self.era5_files if any('_%s' % str(year) in files[0] for year in self._validation_years)], \
                                                                  [file for file in self.front_files if any('_%s' % str(year) in file for year in self._validation_years)]
        self.era5_files_test, self.front_files_test = [files for files in self.era5_files if any('_%s' % str(year) in files[0] for year in self._test_years)], \
                                                      [file for file in self.front_files if any('_%s' % str(year) in file for year in self._test_years)]


class GDASfiles:
    """
    Object that loads and manages GDAS netCDF files and datasets
    """
    def __init__(self, gdas_netcdf_indir: str):
        """
        When the GDASfiles object is created, open all GDAS netCDF files

        gdas_netcdf_indir: str
            - Input directory for the GDAS netCDF files.
        """

        self._all_gdas_netcdf_files = sorted(glob("%s/*/*/*/gdas*.nc" % gdas_netcdf_indir))  # All GDAS files without filtering

        ### All available options for specific filters ###
        self._all_domains = ('full', 'conus')
        self._all_variables = ('q', 'r', 'RH', 'sp_z', 'mslp_z', 'T', 'Td', 'theta', 'theta_e', 'theta_v', 'theta_w', 'Tv', 'Tw', 'u', 'v')
        self._all_forecast_hours = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        self._all_years = (2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022)

        self.reset_all_filters()  # Resetting the filters simply creates the GDAS file lists

        ### Current values for the filters used in the files ###
        self._domains = self._all_domains
        self._forecast_hours = self._all_forecast_hours
        self.variables = self._all_variables
        self._training_years = self._all_years[:11]  # default training years: 2006-2016
        self._validation_years = self._all_years[11:14]  # default validation years: 2017-2019
        self._test_years = self._all_years[14:]  # default test years: 2020-2022

    def reset_all_filters(self):
        """
        Reset the lists of GDAS netCDF files and front object files
        """

        self.gdas_files = self._all_gdas_netcdf_files
        self.gdas_files_training = [gdas_file for gdas_file in self._all_gdas_netcdf_files if any('_%s' % str(year) in gdas_file for year in self._all_years[:11])]
        self.gdas_files_validation = [gdas_file for gdas_file in self._all_gdas_netcdf_files if any('_%s' % str(year) in gdas_file for year in self._all_years[11:14])]
        self.gdas_files_test = [gdas_file for gdas_file in self._all_gdas_netcdf_files if any('_%s' % str(year) in gdas_file for year in self._all_years[14:])]

        ### If front files have been loaded to be paired with gdas files, reset the front file lists ###
        if hasattr(self, '_all_front_files'):
            self.front_files = self._all_front_files
            self.front_files_training = [front_file for front_file in self._all_front_files if any('_%s' % str(year) in front_file for year in self._all_years[:11])]
            self.front_files_validation = [front_file for front_file in self._all_front_files if any('_%s' % str(year) in front_file for year in self._all_years[11:14])]
            self.front_files_test = [front_file for front_file in self._all_front_files if any('_%s' % str(year) in front_file for year in self._all_years[14:])]

    ####################################################################################################################

    def __get_training_years(self):
        """
        Return the list of training years used
        """

        return self._training_years

    def __set_training_years(self, training_years: tuple or list):
        """
        Select the training years to load
        """

        self.__reset_training_years()  # Return file list to last state before training years were modified (no effect is this is first training year selection)

        self._training_years = training_years

        ### Check that all selected GDAS training years are valid ###
        invalid_training_years = [year for year in training_years if year not in self._all_years]
        if len(invalid_training_years) > 0:
            raise TypeError(f"The following training years are not valid: {','.join(sorted(invalid_training_years))}")

        self._training_years_not_in_data = [year for year in self._all_years if year not in training_years]

        ### Remove unwanted years from the list of files ###
        for year in self._training_years_not_in_data:
            self.gdas_files_training = [file for file in self.gdas_files_training if '_%s' % str(year) not in file]

    def __reset_training_years(self):
        """
        Reset training years in the GDAS files
        """

        if not hasattr(self, '_filtered_gdas_files_before_training_year_selection'):  # If the training_years have not been selected yet
            self._filtered_gdas_files_before_training_year_selection = self.gdas_files_training
        else:
            self.gdas_files_training = self._filtered_gdas_files_before_training_year_selection  # Return file list to last state before training_years were modified

    training_years = property(__get_training_years, __set_training_years)  # Property method for setting training years

    ####################################################################################################################

    def __get_validation_years(self):
        """
        Return the list of validation years used
        """

        return self._validation_years

    def __set_validation_years(self, validation_years: tuple or list):
        """
        Select the validation years to load
        """

        self.__reset_validation_years()  # Return file list to last state before validation years were modified (no effect is this is first validation year selection)

        self._validation_years = validation_years

        ### Check that all selected GDAS validation years are valid ###
        invalid_validation_years = [year for year in validation_years if year not in self._all_years]
        if len(invalid_validation_years) > 0:
            raise TypeError(f"The following validation years are not valid: {','.join(sorted(invalid_validation_years))}")

        self._validation_years_not_in_data = [year for year in self._all_years if year not in validation_years]

        ### Remove unwanted years from the list of files ###
        for year in self._validation_years_not_in_data:
            self.gdas_files_validation = [file for file in self.gdas_files_validation if '_%s' % str(year) not in file]

    def __reset_validation_years(self):
        """
        Reset validation years in the GDAS files
        """

        if not hasattr(self, '_filtered_gdas_files_before_validation_year_selection'):  # If the validation_years have not been selected yet
            self._filtered_gdas_files_before_validation_year_selection = self.gdas_files_validation
        else:
            self.gdas_files_validation = self._filtered_gdas_files_before_validation_year_selection  # Return file list to last state before validation_years were modified

    validation_years = property(__get_validation_years, __set_validation_years)  # Property method for setting validation years

    ####################################################################################################################

    def __get_test_years(self):
        """
        Return the list of test years used
        """

        return self._test_years

    def __set_test_years(self, test_years: tuple or list):
        """
        Select the test years to load
        """

        self.__reset_test_years()  # Return file list to last state before test years were modified (no effect is this is first test year selection)

        self._test_years = test_years

        ### Check that all selected GDAS test years are valid ###
        invalid_test_years = [year for year in test_years if year not in self._all_years]
        if len(invalid_test_years) > 0:
            raise TypeError(f"The following test years are not valid: {','.join(sorted(invalid_test_years))}")

        self._test_years_not_in_data = [year for year in self._all_years if year not in test_years]

        ### Remove unwanted years from the list of files ###
        for year in self._test_years_not_in_data:
            self.gdas_files_test = [file for file in self.gdas_files_test if '_%s' % str(year) not in file]

    def __reset_test_years(self):
        """
        Reset test years in the GDAS files
        """

        if not hasattr(self, '_filtered_gdas_files_before_test_year_selection'):  # If the test_years have not been selected yet
            self._filtered_gdas_files_before_test_year_selection = self.gdas_files_test
        else:
            self.gdas_files_test = self._filtered_gdas_files_before_test_year_selection  # Return file list to last state before test_years were modified

    test_years = property(__get_test_years, __set_test_years)  # Property method for setting test years

    ####################################################################################################################

    def __get_variables(self):
        """
        Return the list of variables used
        """

        return self._variables

    def __set_variables(self, variables: tuple or list):
        """
        Select the variables to load
        """

        self.__reset_variables()  # Return file list to last state before variables were modified (no effect is this is first variable selection)

        self._variables = variables

        ### Check that all selected GDAS variables are valid ###
        invalid_variables = [variable for variable in variables if variable not in self._all_variables]
        if len(invalid_variables) > 0:
            raise TypeError(f"The following variables are not valid: {','.join(sorted(invalid_variables))}")

        self._variables_not_in_data = [variable for variable in self._all_variables if variable not in variables]

        ### Remove unwanted variables from the list of files ###
        for variable in self._variables_not_in_data:
            self.gdas_files = [file for file in self.gdas_files if 'gdas_%s_2' % variable not in file]
            self.gdas_files_training = [file for file in self.gdas_files_training if 'gdas_%s_2' % variable not in file]
            self.gdas_files_validation = [file for file in self.gdas_files_validation if 'gdas_%s_2' % variable not in file]
            self.gdas_files_test = [file for file in self.gdas_files_test if 'gdas_%s_2' % variable not in file]

    def __reset_variables(self):
        """
        Reset variables in the GDAS files
        """

        if not hasattr(self, '_filtered_gdas_files_before_variable_selection'):  # If the variables have not been selected yet
            self._filtered_gdas_files_before_variable_selection = self.gdas_files
            self._filtered_gdas_training_files_before_variable_selection = self.gdas_files_training
            self._filtered_gdas_validation_files_before_variable_selection = self.gdas_files_validation
            self._filtered_gdas_test_files_before_variable_selection = self.gdas_files_test
        else:
            ### Return file lists to last state before variables were modified ###
            self.gdas_files = self._filtered_gdas_files_before_variable_selection
            self.gdas_files_training = self._filtered_gdas_training_files_before_variable_selection
            self.gdas_files_validation = self._filtered_gdas_validation_files_before_variable_selection
            self.gdas_files_test = self._filtered_gdas_test_files_before_variable_selection

    variables = property(__get_variables, __set_variables)  # Property method for setting variables

    ####################################################################################################################

    def __get_forecast_hours(self):
        """
        Return the list of forecast hours used
        """

        return self._forecast_hours

    def __set_forecast_hours(self, forecast_hours: tuple or list):
        """
        Select the forecast hours to load
        """

        self.__reset_forecast_hours()  # Return file list to last state before forecast hours were modified (no effect is this is first forecast hour selection)

        self._forecast_hours = forecast_hours

        ### Check that all selected GDAS forecast hours are valid ###
        invalid_forecast_hours = [hour for hour in forecast_hours if hour not in self._all_forecast_hours]
        if len(invalid_forecast_hours) > 0:
            raise TypeError(f"The following forecast hours are not valid: {','.join(sorted(str(hour) for hour in invalid_forecast_hours))}")

        self._forecast_hours_not_in_data = [hour for hour in self._all_forecast_hours if hour not in forecast_hours]

        ### Remove unwanted forecast hours from the list of files ###
        for hour in self._forecast_hours_not_in_data:
            self.gdas_files = [file for file in self.gdas_files if '_f%03d_' % hour not in file]
            self.gdas_files_training = [file for file in self.gdas_files_training if '_f%03d_' % hour not in file]
            self.gdas_files_validation = [file for file in self.gdas_files_validation if '_f%03d_' % hour not in file]
            self.gdas_files_test = [file for file in self.gdas_files_test if '_f%03d_' % hour not in file]

    def __reset_forecast_hours(self):
        """
        Reset forecast hours in the GDAS files
        """

        if not hasattr(self, '_filtered_gdas_files_before_forecast_hour_selection'):  # If the forecast hours have not been selected yet
            self._filtered_gdas_files_before_forecast_hour_selection = self.gdas_files
            self._filtered_gdas_training_files_before_forecast_hour_selection = self.gdas_files_training
            self._filtered_gdas_validation_files_before_forecast_hour_selection = self.gdas_files_validation
            self._filtered_gdas_test_files_before_forecast_hour_selection = self.gdas_files_test
        else:
            ### Return file lists to last state before forecast hours were modified ###
            self.gdas_files = self._filtered_gdas_files_before_forecast_hour_selection
            self.gdas_files_training = self._filtered_gdas_training_files_before_forecast_hour_selection
            self.gdas_files_validation = self._filtered_gdas_validation_files_before_forecast_hour_selection
            self.gdas_files_test = self._filtered_gdas_test_files_before_forecast_hour_selection

    forecast_hours = property(__get_forecast_hours, __set_forecast_hours)  # Property method for setting forecast hours

    ####################################################################################################################

    def __get_domains(self):
        """
        Return the list of domains used
        """

        return self._domains

    def __set_domains(self, domains: tuple or list):
        """
        Select the domains to load
        """

        self.__reset_domains()  # Return file list to last state before domains were modified (no effect is this is first domain selection)

        self._domains = domains

        ### Check that all selected GDAS domains are valid ###
        invalid_domains = [domain for domain in domains if domain not in self._all_domains]
        if len(invalid_domains) > 0:
            raise TypeError(f"The following domains are not valid: {','.join(sorted(invalid_domains))}")

        self._domains_not_in_data = [domain for domain in self._all_domains if domain not in domains]

        ### Remove unwanted domains from the list of files ###
        for domain in self._domains_not_in_data:
            self.gdas_files = [file for file in self.gdas_files if '_%s.nc' % domain not in file]
            self.gdas_files_training = [file for file in self.gdas_files_training if '_%s.nc' % domain not in file]
            self.gdas_files_validation = [file for file in self.gdas_files_validation if '_%s.nc' % domain not in file]
            self.gdas_files_test = [file for file in self.gdas_files_test if '_%s.nc' % domain not in file]

    def __reset_domains(self):
        """
        Reset domains in the GDAS files
        """

        if not hasattr(self, '_filtered_gdas_files_before_domain_selection'):  # If the domains have not been selected yet
            self._filtered_gdas_files_before_domain_selection = self.gdas_files
            self._filtered_gdas_training_files_before_domain_selection = self.gdas_files_training
            self._filtered_gdas_validation_files_before_domain_selection = self.gdas_files_validation
            self._filtered_gdas_test_files_before_domain_selection = self.gdas_files_test
        else:
            ### Return file lists to last state before domains were modified ###
            self.gdas_files = self._filtered_gdas_files_before_domain_selection
            self.gdas_files_training = self._filtered_gdas_training_files_before_domain_selection
            self.gdas_files_validation = self._filtered_gdas_validation_files_before_domain_selection
            self.gdas_files_test = self._filtered_gdas_test_files_before_domain_selection

    domains = property(__get_domains, __set_domains)  # Property method for setting domains

    ####################################################################################################################

    def pair_with_fronts(self, front_indir):
        """
        Pair all of the GDAS timesteps with frontal object files containing timesteps for the given forecast hours
        
        front_indir: str
            - Directory where the frontal object files are stored.
        """

        self._all_front_files = sorted(glob("%s/*/*/*/FrontObjects*.nc" % front_indir))  # All front files without filtering
        self.front_files = self._all_front_files

        ### Remove unwanted domains from the list of front files ###
        for domain in self._domains_not_in_data:
            self.front_files = [file for file in self.front_files if '_%s.nc' % domain not in file]

        front_files_timestep = []
        gdas_files_timestep = []
        gdas_timesteps = []  # Timesteps of GDAS/GFS files
        gdas_files_by_timestep = []  # List of GDAS/GFS files with each index representing a group of files with a common timestep
        front_files_list = []
        gdas_files_list = []

        ### Collect all timesteps in the fronts files ###
        for i in range(len(self.front_files)):
            front_filename_start_index = self.front_files[i].find('FrontObjects_')
            front_files_timestep.append(self.front_files[i][front_filename_start_index + 13:front_filename_start_index + 23])  # Timestep is first 10 characters after 'FrontObjects_'

        ### Find all timesteps in the GDAS/GFS files ###
        for j in range(len(self.gdas_files)):
            gdas_filename_start_index = self.gdas_files[j].find('gdas_')
            gdas_filename_no_variable_index = self.gdas_files[j][gdas_filename_start_index:].find('_2') + 1
            gdas_timesteps.append(self.gdas_files[j][gdas_filename_start_index + gdas_filename_no_variable_index:gdas_filename_start_index + gdas_filename_no_variable_index + 10])  # Timestep is the first 10 characters after the pressure level in the filename

        unique_gdas_timesteps = list(np.unique(sorted(gdas_timesteps)))
        unique_forecast_timesteps = []

        for unique_timestep in unique_gdas_timesteps:
            if 'full' in self._domains and any('%02d' % hour in unique_timestep[-2:] for hour in [6, 9, 15, 21]):
                unique_gdas_timesteps.pop(unique_gdas_timesteps.index(unique_timestep))
            else:
                unique_forecast_timesteps.append(data_utils.add_or_subtract_hours_to_timestep(unique_timestep, self._forecast_hours))

        ### Find all GDAS files for each timestep ###
        for timestep in unique_gdas_timesteps:
            files_for_timestep = sorted(filter(lambda filename: timestep in filename, self.gdas_files))
            gdas_files_timestep_temp = []  # Temporary holding folder for GDAS files before they are added to the list of filenames
            for gdas_file in files_for_timestep:
                gdas_filename_start_index = gdas_file.find('gdas_')
                gdas_filename_no_variable_index = gdas_file[gdas_filename_start_index:].find('_2') + 1
                gdas_files_timestep_temp.append(gdas_file[gdas_filename_start_index + gdas_filename_no_variable_index:gdas_filename_start_index + gdas_filename_no_variable_index + 10])
            gdas_files_timestep.append(gdas_files_timestep_temp)
            gdas_files_by_timestep.append(files_for_timestep)

        ### File matching: only load files with common timesteps or the proper timesteps for the given forecast hour ###
        total_gdas_files = len(gdas_files_timestep)
        total_front_files = len(self.front_files)
        for gdas_file_no in range(total_gdas_files):
            if unique_gdas_timesteps[gdas_file_no] in front_files_timestep:
                gdas_files_list.append(gdas_files_by_timestep[gdas_file_no])
        for front_file_no in range(total_front_files):
            if front_files_timestep[front_file_no] in unique_forecast_timesteps:
                front_files_list.append(self.front_files[front_file_no])

        self.gdas_files = gdas_files_list
        self.front_files = front_files_list

        ### Iterate through the front files to search for indices within the list to delegate to the lists for training, validation and test datasets ###
        self.gdas_files_training, self.front_files_training = [files[0] for files in self.gdas_files if any('_%s' % str(year) in files[0] for year in self._training_years)], \
                                                              [file for file in self.front_files if any('_%s' % str(year) in file for year in self._training_years)]
        self.gdas_files_validation, self.front_files_validation = [files[0] for files in self.gdas_files if any('_%s' % str(year) in files[0] for year in self._validation_years)], \
                                                                  [file for file in self.front_files if any('_%s' % str(year) in file for year in self._validation_years)]
        self.gdas_files_test, self.front_files_test = [files[0] for files in self.gdas_files if any('_%s' % str(year) in files[0] for year in self._test_years)], \
                                                      [file for file in self.front_files if any('_%s' % str(year) in file for year in self._test_years)]


def extract_gdas_tarfile(gdas_tar_indir: str, gdas_grib_outdir: str, year: int, month: int, day: int, string_to_find: str = None, remove_tarfile: bool = True):
    """
    Extract members from a GDAS TAR file.

    Parameters
    ----------
    gdas_tar_indir: str
        - Directory for the GDAS TAR files.
    gdas_grib_outdir: str
        - Output directory for the extracted grib files.
    year: int
    month: int
    day: int
    string_to_find: str or None
        - Files containing this string will be extracted from the TAR file and placed into the provided output directory.
        - If left as None, all files within the GDAS TAR file will be extracted.
    remove_tarfile: bool
        - Setting this to True will delete the original TAR file after all members have been extracted.
        - If string_to_find is not None (i.e. members were filtered out of the TAR file), setting this to True will have no effect.
          This is to prevent files that are potentially needed in the future from being removed.

    Raises
    ------
    UserWarning
        - If 'remove_tarfile' is True and 'string_to_find' is not None
    """

    ### Verify that the provided input and output directories are valid ###
    assert os.path.isdir(gdas_tar_indir)  # If this line returns an AssertionError, then the input directory is invalid / does not exist
    assert os.path.isdir(gdas_grib_outdir)  # If this line returns an AssertionError, then the output directory is invalid / does not exist

    tarfile_path = '%s/ncep_global_ssi.%d%02d%02d.tar' % (gdas_tar_indir, year, month, day)

    print(f"Reading GDAS TAR file: {tarfile_path}")
    gdas_tarfile = tarfile.open(tarfile_path, 'r')

    ### Gather files to extract, starting with all files and filtering out files whose names do not contain the provided string (if said string is provided) ###

    files_to_extract = [member for member in gdas_tarfile.getmembers()]  # Find all members within the TAR file
    print("Number of members in GDAS TAR file: %d" % len(files_to_extract))

    if string_to_find is not None:
        files_to_extract = [member for member in gdas_tarfile.getmembers() if string_to_find in member.name]  # Filter files based on given string

    ### Create a folder to hold extracted files for the provided day (if it does not exist)
    gdas_folder_for_provided_day = '%d%02d%02d' % (year, month, day)
    full_gdas_outdir = f'{gdas_grib_outdir}/{gdas_folder_for_provided_day}'
    if not os.path.isdir(full_gdas_outdir):  # Check if the directory exists
        os.mkdir(full_gdas_outdir)  # Create the directory for the day if the directory does not exist
        print("Created directory: %s" % full_gdas_outdir)

    ### Extract the members from the GDAS TAR file and place them into the provided output directory ###
    for member in files_to_extract:
        gdas_tarfile.extract(member, full_gdas_outdir)
    print("Successfully extracted %d members from GDAS TAR file" % len(files_to_extract))

    gdas_tarfile.close()  # Close the GDAS TAR file

    if remove_tarfile:
        if string_to_find is None:
            os.remove(tarfile_path)  # Delete the GDAS TAR file
            print(f"Successfully deleted GDAS TAR file located at: {tarfile_path}")
        else:
            raise UserWarning(f"The GDAS TAR file located at {tarfile_path} will not be removed as some members were not extracted from the TAR file. To delete a TAR file, "
                              "all members must be extracted without a filter. This can be accomplished by leaving the 'string_to_find' parameter as None, which is its default "
                              "value.")


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
            loss_string = 'fractions_skill_score'
        elif loss == 'bss':
            loss_string = 'brier_skill_score'
            loss_function = custom_losses.brier_skill_score
        else:
            loss_string = None
            loss_function = None

        if metric == 'fss':
            fss_mask_size, fss_c = model_properties['fss_mask_c'][0], model_properties['fss_mask_c'][1]  # First element is the mask size, second is the c parameter
            loss_function = custom_losses.make_fractions_skill_score(fss_mask_size, fss_c)
            metric_string = 'fractions_skill_score'
        elif metric == 'bss':
            metric_string = 'brier_skill_score'
            metric_function = custom_losses.brier_skill_score
        else:
            metric_string = None
            metric_function = None

        model_loaded = False  # Boolean flag that will become True once a model successfully loads
        print("Loading model....", end='')
        if loss_string is not None:
            if metric_string is not None:

                ### Older models contain different strings to represent the FSS loss function, so a few conditions ensure that we grab the FSS function if it was used in the model ###
                while model_loaded is False:
                    try:
                        model = lm('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number), custom_objects={loss_string: loss_function, metric_string: metric_function})
                    except ValueError:

                        if 'fractions' in loss_string or 'FSS' in loss_string:
                            if loss_string == 'fractions_skill_score':
                                loss_string = 'FSS_loss_2D'
                            elif loss_string == 'FSS_loss_2D':
                                loss_string = 'FSS_loss_3D'
                            else:
                                print("failed")
                                # Load the model normally to return the ValueError message for unknown loss function
                                model = lm('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number), custom_objects={loss_string: loss_function, metric_string: metric_function})

                        if 'fractions' in metric_string or 'FSS' in metric_string:
                            if metric_string == 'fractions_skill_score':
                                metric_string = 'FSS_loss_2D'
                            elif metric_string == 'FSS_loss_2D':
                                metric_string = 'FSS_loss_3D'
                            else:
                                print("failed")
                                # Load the model normally to return the ValueError message for unknown metric
                                model = lm('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number), custom_objects={loss_string: loss_function, metric_string: metric_function})

                    else:
                        model_loaded = True

            else:

                ### Older models contain different strings to represent the FSS loss function, so a few conditions ensure that we grab the FSS function if it was used in the model ###
                while model_loaded is False:
                    try:
                        model = lm('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number), custom_objects={loss_string: loss_function})
                    except ValueError:

                        if 'fractions' in loss_string or 'FSS' in loss_string:
                            if loss_string == 'fractions_skill_score':
                                loss_string = 'FSS_loss_2D'
                            elif loss_string == 'FSS_loss_2D':
                                loss_string = 'FSS_loss_3D'
                            else:
                                print("failed")
                                # Load the model normally to return the ValueError message for unknown loss function
                                model = lm('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number), custom_objects={loss_string: loss_function})

                    else:
                        model_loaded = True

        else:

            ### Older models contain different strings to represent the FSS loss function, so a few conditions ensure that we grab the FSS function if it was used in the model ###
            while model_loaded is False:
                try:
                    model = lm('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number), custom_objects={metric_string: metric_function})
                except ValueError:
                    if 'fractions' in metric_string or 'FSS' in metric_string:
                        if metric_string == 'fractions_skill_score':
                            metric_string = 'FSS_loss_2D'
                        elif metric_string == 'FSS_loss_2D':
                            metric_string = 'FSS_loss_3D'
                        else:
                            print("failed")
                            # Load the model normally to return the ValueError message for unknown loss function
                            model = lm('%s/model_%d/model_%d.h5' % (model_dir, model_number, model_number), custom_objects={metric_string: metric_function})

                else:
                    model_loaded = True
                    print("done")

    return model


if __name__ == '__main__':
    """
    Warnings
        - Do not use leading zeros when declaring the month, day, and hour in 'date'. (ex: if the day is 2, do not type 02)
        - Longitude values in the 'new_extent' argument must in the 360-degree coordinate system.
    
    Examples
        Example 1. Compress a large set of files into a TAR file.
            > python file_manager.py --compress_files --main_dir ./file_directory --glob_file_string test_files_*_.pkl --tar_filename compressed_files.tar
        
        Example 2. Extract the contents of an existing TAR file.
            > python file_manager.py --extract_tarfile --main_dir ./file_directory --tar_filename compressed_files.tar
        
        Example 3. Remove members from GDAS TAR file.
            
        Example 4. Delete a large group of files using a common string in the filenames.
            > python file_manager.py --delete_grouped_files --main_dir ./file_directory --num_subdir 3 --glob_file_string test_files_*_.pkl
                - Using 3 for 'num_subdir' means that files will be returned from glob that match the following string:
                    ./file_directory/*/*/*/test_files_*_.pkl
                - Using 5 for 'num_subdir':
                    ./file_directory/*/*/*/*/*/test_files_*_.pkl
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--compress_files', action='store_true', help='Compress files')
    parser.add_argument('--delete_grouped_files', action='store_true', help='Delete a set of files')
    parser.add_argument('--extract_gdas_tarfile', action='store_true', help='Remove non-0.25 degree grid files from a GDAS TAR file')
    parser.add_argument('--extract_tarfile', action='store_true', help='Extract a TAR file')
    parser.add_argument('--gdas_tar_indir', type=str, help='Input directory for the GDAS TAR files.')
    parser.add_argument('--gdas_grib_outdir', type=str, help='Output directory for the GDAS grib files.')
    parser.add_argument('--glob_file_string', type=str, help='String of the names of the files to compress or delete.')
    parser.add_argument('--main_dir', type=str, help='Main directory for subdirectory creation or where the files in question are located.')
    parser.add_argument('--num_subdir', type=int, help='Number of subdirectory layers in the main directory.')
    parser.add_argument('--tar_filename', type=str, help='Name of the TAR file.')
    parser.add_argument('--timestep_for_gdas_extraction', type=int, nargs=3, help='Year, month, and day for the GDAS TAR files that will be modified.')
    args = parser.parse_args()
    provided_arguments = vars(args)
    
    if args.compress_files:
        required_arguments = ['main_dir', 'glob_file_string', 'tar_filename']
        check_arguments(provided_arguments, required_arguments)
        compress_files(args.main_dir, args.glob_file_string, args.tar_filename)

    if args.delete_grouped_files:
        required_arguments = ['glob_file_string', 'main_dir', 'num_subdir']
        check_arguments(provided_arguments, required_arguments)
        delete_grouped_files(args.main_dir, args.glob_file_string, args.num_subdir)

    if args.extract_tarfile:
        required_arguments = ['main_dir', 'tar_filename']
        check_arguments(provided_arguments, required_arguments)
        extract_tarfile(args.main_dir, args.tar_filename)

    if args.extract_gdas_tarfile:
        required_arguments = ['gdas_tar_indir', 'gdas_grib_outdir', 'timestep_for_gdas_extraction']
        check_arguments(provided_arguments, required_arguments)
        extract_gdas_tarfile(args.gdas_tar_indir, args.gdas_grib_outdir, args.timestep_for_gdas_extraction[0], args.timestep_for_gdas_extraction[1], args.timestep_for_gdas_extraction[2])
