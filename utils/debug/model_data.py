"""
Debug tools for model data (predictions, statistics, etc.)

Code written by: Andrew Justin (andrewjustinwx@gmail.com)

Last updated: 9/18/2022 9:10 PM CT
"""

import numpy as np
import pandas as pd
import os


def find_missing_statistics(model_dir, model_number, domain, domain_images, domain_trim, variables_data_source, variables_netcdf_indir,
    fronts_netcdf_indir, fronts_xml_indir, dataset=None, years=None, report_outdir=None):
    """
    Search through a folder containing model statistics to find timesteps for which files are missing.
    Information on missing timesteps will be output to a text file.

    Parameters
    ----------
    model_dir: str
        - Main directory for the models.
    model_number: int
        - Slurm job number for the model. This is the number in the model's filename.
    domain: str
        - Domain of the data.
    domain_images: iterable object with 2 ints
        - Number of images along each dimension of the final stitched map (lon lat).
    domain_trim: iterable object with 2 ints
        - Number of pixels to trim each image by along each dimension before taking the maximum of the overlapping pixels (lon lat).
    variables_data_source: str
        - Data used when making the predictions that were used for calculating the statistics.
    variables_netcdf_indir: str
        - Directory for the data used when making the predictions for the statistics.
    fronts_netcdf_indir: str
        - Directory where the front object netcdf files are stored.
    fronts_xml_indir: str
        - Directory where the XML files for front objects are stored.
    dataset: str or None
        - Dataset (years) to analyze. Available options are: 'training', 'validation', 'test', or None.
        - If None, 'years' must be provided.
    years: iterable of ints or None
        - List of years to analyze.
        - If None, 'dataset' must be provided.
    report_outdir: str or None
        - Output directory for the text file.
        - If None, this defaults to the directory for the model (i.e. model_dir/model_<model_number>).

    Raises
    ------
    TypeError
        - If 'dataset' and 'years' are both None.
    """

    assert os.path.isdir(model_dir)  # Assert that the model directory exists
    assert os.path.isdir(variables_netcdf_indir)  # Assert that the directory for variable data exists
    assert os.path.isdir(fronts_netcdf_indir)  # Assert that the directory for the front object netcdf data exists
    assert os.path.isdir(fronts_xml_indir)  # Assert that the directory for the front object XML data exists

    model_folder = '%s/model_%d' % (model_dir, model_number)
    model_properties = pd.read_pickle('%s/model_%d_properties.pkl' % (model_folder, model_number))

    if dataset is not None:
        if type(dataset) != str:
            raise TypeError(f"if 'dataset' is not None, it must be a string. Received type: {type(dataset)}")
        elif years is not None:
            raise TypeError(f"if 'dataset' is not None, 'years' must be None. Received types {type(dataset)} and {type(years)}")
        elif dataset != 'training' and dataset != 'validation' and dataset != 'test':
            raise ValueError(f"{dataset} is not a valid dataset, options are: 'training', 'validation', 'test'")
        else:
            years = model_properties['%s_years' % dataset]

    if report_outdir is None:
        report_outdir = model_folder
    else:
        assert os.path.isdir(report_outdir)  # Check that the output directory for the text file is valid

    report_file_path = f'%s/model_{model_number}_conus_%dx%dimages_%dx%dtrim_statistics_debug.txt' % \
                       (report_outdir, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1])
    report_file = open(report_file_path, 'w')

    stats_folder_to_analyze = '%s/statistics/%s_%dx%dimages_%dx%dtrim' % (model_folder, domain, domain_images[0], domain_images[1], domain_trim[0], domain_trim[1])

    assert os.path.isdir(stats_folder_to_analyze)  # Check that the stats folder is valid

    ### Generate a list of timesteps [[year, 1, 1, 0], [year, ..., ..., ...], [year, 12, 31, 21]] ###
    date_list = []
    days_per_month = dict({str(year): None for year in years})  # Number of days in each month for each year being analyzed
    for year in years:
        if year % 4 == 0:
            month_2_days = 29  # Leap year
        else:
            month_2_days = 28  # Not a leap year

        days_per_month[str(year)] = [31, month_2_days, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        for month in range(1, 13):
            for day in range(1, days_per_month[str(year)][month - 1] + 1):
                for hour in range(0, 24, 3):
                    date_list.append([year, month, day, hour])

    num_timesteps = len(date_list)
    num_missing_stats_files = 0  # Counter for the number of missing statistics files
    missing_indices = dict({str(year): [] for year in years})  # Daily indices representing days where statistics files are missing

    ### Iterate through all of the timesteps to check for statistics files ###
    for timestep in date_list:

        # Boolean flags for signaling whether a specific data file is missing
        missing_prediction_file = False
        missing_variables_file = False
        missing_front_netcdf_file = False
        missing_front_xml_file = False

        stats_file = f'%s/model_{model_number}_%d-%02d-%02d-%02dz_conus_%dx%dimages_%dx%dtrim_statistics.nc' % (stats_folder_to_analyze, timestep[0], timestep[1], timestep[2], timestep[3], domain_images[0], domain_images[1], domain_trim[0], domain_trim[1])
        probs_file = stats_file.replace('statistics', 'probabilities')

        if not os.path.isfile(stats_file):

            # If the statistics file does not exist, check to see if a prediction/probabilities file is missing
            if not os.path.isfile(probs_file):
                missing_prediction_file = True

            # If the statistics file does not exist, check to see if a variable file is missing
            if not os.path.isfile(f'%s/%d/%02d/%02d/{variables_data_source}_T_%d%02d%02d%02d_full.nc' %
                                  (variables_netcdf_indir, timestep[0], timestep[1], timestep[2], timestep[0], timestep[1], timestep[2], timestep[3])):
                missing_variables_file = True

            # If the statistics file does not exist, check to see if a front object netcdf file is missing
            if not os.path.isfile(f'%s/%d/%02d/%02d/FrontObjects_%d%02d%02d%02d_full.nc' %
                                  (fronts_netcdf_indir, timestep[0], timestep[1], timestep[2], timestep[0], timestep[1], timestep[2], timestep[3])):
                missing_front_netcdf_file = True

            # If the statistics file does not exist, check to see if a front object XML file is missing
            if missing_front_netcdf_file:
                if not os.path.isfile(f'%s/pres_pmsl_%d%02d%02d%02df000.xml' % (fronts_xml_indir, timestep[0], timestep[1], timestep[2], timestep[3])):
                    missing_front_xml_file = True

            if missing_variables_file:
                if missing_front_netcdf_file:
                    if missing_front_xml_file:
                        report_file.write(f"\n%d-%02d-%02d-%02dz: missing {variables_data_source.upper()} netcdf, front object netcdf (front XML file found: No)" % (timestep[0], timestep[1], timestep[2], timestep[3]))
                    else:
                        report_file.write(f"\n%d-%02d-%02d-%02dz: missing {variables_data_source.upper()} netcdf, front object netcdf (front XML file found: Yes)" % (timestep[0], timestep[1], timestep[2], timestep[3]))
                else:
                    report_file.write(f"\n%d-%02d-%02d-%02dz: missing {variables_data_source.upper()} netcdf" % (timestep[0], timestep[1], timestep[2], timestep[3]))

            elif missing_front_netcdf_file:
                if missing_front_xml_file:
                    report_file.write(f"\n%d-%02d-%02d-%02dz: missing front object netcdf (front XML file found: No)" % (timestep[0], timestep[1], timestep[2], timestep[3]))
                else:
                    report_file.write(f"\n%d-%02d-%02d-%02dz: missing front object netcdf (front XML file found: Yes)" % (timestep[0], timestep[1], timestep[2], timestep[3]))

            elif missing_prediction_file:
                report_file.write(f"\n%d-%02d-%02d-%02dz: prediction was not completed ({variables_data_source.upper()} netcdf and front object netcdf files were found)" % (timestep[0], timestep[1], timestep[2], timestep[3]))

            else:
                report_file.write(f"\n%d-%02d-%02d-%02dz: statistics calculation was not completed (prediction/probabilities file was found)" % (timestep[0], timestep[1], timestep[2], timestep[3]))

            num_missing_stats_files += 1
            current_year = str(timestep[0])
            current_month = timestep[1]
            current_day = timestep[2]

            index = int(np.sum(days_per_month[current_year][:current_month-1]) + current_day) - 1
            missing_indices[current_year].append(index)

    report_file.write('\n\n\n Summary \n -------------')
    report_file.write(f'\nTimesteps analyzed: {num_timesteps}')
    report_file.write(f'\nNumber of missing statistics files: {num_missing_stats_files}')
    for year in years:
        year = str(year)
        missing_indices[year] = np.unique(missing_indices[year])
        report_file.write(f"\nMissing indices [{year}]: {','.join(missing_indices[year].astype(str))}")
    report_file.close()
