# Operational Fronts Module
This module contains all the code necessary to generate predictions for cold, warm, stationary, occluded and binary fronts using U-Net 3+ models from Justin et al. (2022).

## Table of Contents
1. Initial setup
2. Downloading the data
3. Generating predictions
4. Prediction plots
5. Directory cleanup

### 1. Initial Setup
The steps highlighted in Initial Setup section only need to be completed once.

First, create a directory to keep the models. The models will be contained in folders which are labeled with numbers such that the structure of the folder will be /models/model_N, where N is the number assigned to the model. If any future models are created, add them to the base model directory.

Next, create a directory for the netCDF / GRIB files. The downloaded data will be stored here, along with the generated netCDF files.

With the provided .yml file, set up the Python environment.


### 2. Downloading the data

To download GFS or GDAS data for predictions, you can run the following command from a terminal:

python operational_data_download.py --netcdf_outdir {} --init_time {} --forecast_hours {} --model {}

--netcdf_outdir: Output directory for the netCDF files containing GFS and/or GDAS data. (Ex: --netcdf_outdir /data)

--init_time: Initialization time for the GFS or GDAS model run. Pass 4 integers to this argument with the year, month, day, and hour. (Ex: --init_time 2023 4 20 18) [18z April 20, 2023]

--forecast_hours: List of forecast hours to download from the GFS or GDAS model run. (Ex: --forecast_hours 0 6 12 18)

--model: 'gfs' or 'gdas' (Ex: --model gfs)

This will download GFS or GDAS grib files from the AWS server and convert them to netCDF files containing the required data to generate model predictions. The downloaded grib files will be deleted after the netCDF have been successfully created.

### 3. Generating predictions

NOTE: {} indicates that an argument needs to be provided

To generate predictions with a given model, run the following command in a terminal:

python predict.py --model_dir {} --model_number {} --netcdf_indir {} 
--domain {} --init_time {} --variable_data_source {}

--model_dir: Base directory where the models are stored. (Ex: /models)

--model_number: Number assigned to the model. (Ex: --model_number 6846496)

--netcdf_indir: Same as --netcdf_outdir from section 2. (Ex: --netcdf_indir /data)

--domain: Domain over which to make the predictions. Can be 'conus' (Continental United States) or 'full' (Unified Surface Analysis domain). (Ex: --domain full)

--init_time: Same as --init_time in section 2. (Ex: --init_time 2023 4 20 18) [18z April 20, 2023]

*Optional Arguments*

--gpu_device: Integer representing which GPU on your machine to use. **If you do not have a GPU, do not add this argument to the terminal as it will return an error during runtime**. (Ex: --gpu_device 0)

--memory_growth: Boolean flag that runs the following function during runtime: tf.config.experimental.set_memory_growth(device=..., True). See https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth for more information. **If you do not have a GPU, do not add this argument to the terminal as it will return an error during runtime**.

#### References
Justin et al. (2022): [insert link here]
