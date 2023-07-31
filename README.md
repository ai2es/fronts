# Operational Fronts Module
This module contains all the code necessary to generate predictions for cold, warm, stationary, occluded and binary fronts using U-Net 3+ models.

**IMPORTANT: AS OF JULY 31, 2023, THE FOLLOWING MODELS ARE NO LONGER SUPPORTED: 6846496, 7236500, 7507525**


## Table of Contents
1. Initial setup
2. Downloading the data
3. Generating predictions
4. Prediction plots
5. XML files
6. Data deletion

## 1. Initial Setup
The steps highlighted in Initial Setup section only need to be completed once.

First, create a directory to keep the models. The models will be contained in folders which are labeled with numbers such that the structure of the folder will be /models/model_N, where N is the number assigned to the model. If any future models are created, add them to the base model directory.

**The models are numbered as follows:**
* 1701: 5-class (Cold, warm, stationary, occluded, dryline)

Next, create a directory for the netCDF / GRIB files. The downloaded data will be stored here, along with the generated netCDF files.

With the provided .yml file, set up the Python environment.

## 2. Downloading the data

**NOTE: {} indicates that an argument needs to be provided**

To download GFS or GDAS data for predictions, you can run the following command from a terminal:

python operational_data_download.py --netcdf_outdir {} --init_time {} --forecast_hours {} --model {} --domain {}

*Required Arguments*

* --netcdf_outdir: Output directory for the netCDF files containing GFS and/or GDAS data. (Ex: --netcdf_outdir /data)
* --init_time: Initialization time for the GFS or GDAS model run. Pass 4 integers to this argument with the year, month, day, and hour. (Ex: --init_time 2023 4 20 18) [18z April 20, 2023]
* --forecast_hours: List of forecast hours to download from the GFS or GDAS model run. (Ex: --forecast_hours 0 3 6 9)
* --model: 'gfs' or 'gdas' (Ex: --model gfs)
* --domain: Domain of the data to download. Can be 'conus' (Continental United States), 'full' (Unified Surface Analysis domain), or 'global'. (Ex: --model full)

This will download GFS or GDAS GRIB files from the AWS server and convert them to netCDF files containing the required data to generate model predictions. The downloaded GRIB files will be deleted after the netCDF have been successfully created.

## 3. Generating predictions

**NOTE: {} indicates that an argument needs to be provided**

To generate predictions with a given model, run the following command in a terminal:

python predict.py --model_dir {} --model_number {} --netcdf_indir {} 
--domain {} --init_time {} --variable_data_source {} --calibration {}

*Required Arguments*

* --model_dir: Base directory where the models are stored. (Ex: /models)
* --model_number: Number assigned to the model. (Ex: --model_number 1701)
* --netcdf_indir: Same as --netcdf_outdir from section 2. (Ex: --netcdf_indir /data)
* --domain: Same as section 1. (Ex: --domain full)
* --init_time: Same as --init_time in section 2. (Ex: --init_time 2023 4 20 18) [18z April 20, 2023]
* --variable_data_source: Same as --model from section 2. (Ex: --variable_data_source gfs)
* --calibration: Calibration distance in kilometers for the frontal probabilities. (Ex: --calibration 100)
  * Can be set to 50, 100, 150, 200, or 250 km. 100 kilometers is recommended to avoid oversaturating the domain with probability contours.

*Optional Arguments*

* --gpu_device: Integer representing which GPU on your machine to use. **If you do not have a GPU, do not add this argument to the terminal as it will return an error during runtime**. (Ex: --gpu_device 0)
* --memory_growth: Boolean flag that runs the following function during runtime: tf.config.experimental.set_memory_growth(device=..., True). See https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth for more information. **If you do not have a GPU, do not add this argument to the terminal as it will return an error during runtime**.

Predictions for all forecast hours at the given initialization time will be downloaded at the same time. The predictions will be output as netcdf files containing probabilities of the respective front types. In each model directory, a subdirectory titled 'predictions' will contain these files. (e.g. /models/model_1701/predictions)

**NOTE: As of July 31, 2023, probabilities will be saved in two separate files: one without calibration and one with calibration.**

## 4. Prediction plots

**NOTE: {} indicates that an argument needs to be provided**

To generate prediction plots, run the following command in a terminal:

python prediction_plot.py --model_dir {} --model_number {} --plot_dir {} --variable_data_source {} --init_time {} 
--forecast_hours {} --domain {}

*Required Arguments*

* --model_dir: (Ex: /models)
* --model_number: (Ex: --model_number 1701)
* --plot_dir: Directory for the prediction plots. (Ex: --plot_dir /plots)
* --variable_data_source: (Ex: --variable_data_source gfs)
* --init_time: (Ex: --init_time 2023 4 20 18) [18z April 20, 2023]
* --forecast_hours: Same as --forecast_hours in Section 2. (Ex: --forecast_hours 0 3 6 9)
* --domain: (Ex: --domain full)

*Optional Arguments*

* --contours: Plot probability contours for each front type at intervals of 10%.
* --splines: Draw splines for the fronts. This can be used with --contours to plot splines on top of probability contours.
* --calibration: Same as section 3. (Ex: --calibration 100)

## 5. XML files

**NOTE: {} indicates that an argument needs to be provided**

To generate XML files with the front splines, run the following command in a terminal:

python create_prediction_xml.py --model_dir {} --model_number {} --xml_dir {} --init_time {} --forecast_hours {} --domain {} --variable_data_source {}

*Required Arguments*

* --model_dir: (Ex: /models)
* --model_number: (Ex: --model_number 1701)
* --xml_dir: Directory for the XML files. (Ex: --plot_dir /plots)
* --init_time: (Ex: --init_time 2023 4 20 18) [18z April 20, 2023]
* --forecast_hours: (Ex: --forecast_hours 0 3 6 9)
* --domain: (Ex: --domain full)
* --variable_data_source: (Ex: --variable_data_source gfs)

## 6. Data deletion

To delete model predictions and/or GFS/GDAS data, you can run the following command in the terminal:

python delete_data.py

You can pass one or both of the following arguments to delete unwanted data:

* --model_dir: (Ex: /models). This will delete all netCDF files stored in the 'predictions' subdirectories in the model folders (e.g. all data in /models/model_1701/predictions will be deleted).
* --netcdf_dir: (Ex: /data). This will delete all netCDF files stored in /data.

Running the command with no arguments will result in no data being deleted.
