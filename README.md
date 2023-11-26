# AI2ES Fronts Project Training Guide
The following guide will detail the steps toward successfully train a UNET-style model for frontal boundary predictions.

###### Note that many file-naming conventions and expected directory structures are hard-coded and must be followed for this module to function properly.

## 1. Gathering Data
There are two types of data that need to be gathered for successful model training: front labels (targets) and predictors (inputs). The predictors will normally be sourced from ERA5 data or an NWP model (GFS, ECMWF, etc.). 

#### 1a. Front labels

Front labels are sourced from the National Oceanic and Atmospheric Administration (NOAA) in the form of XML files, or from The Weather Company (TWC) in the form of GML files.

* **NOAA XML file-naming format:** *pres_pmsl_YYYYMMDDHHf000.xml* 
  * Ex: *pres_pmsl_2016062115f000.xml* [15z June 21, 2016 front analysis] 
* **IBM GML file-naming formats and directory structures:** 
  * Current analysis: */YYYYMMDD/HH/rec.sfcanalysis.YYYYMMDDTHH0000Z.NIL.P0D.WORLD@10km.FRONTS.SFC.gml*
    * Ex: */20230827/00/rec.sfcanalysis.20230827T090000Z.NIL.P0D.WORLD@10km.FRONTS.SFC.gml* [9z August 27, 2023 analysis]
  * Forecasted fronts: */YYYYMMDD/HH/rec.sfcanalysis.YYYYMMDDTHH0000Z.YYYYMMDDTHH0000Z.P01D00H00M.WORLD@10km.FRONTS.SFC.gml*
    * Ex: */20230827/00/rec.sfcanalysis.20230828T000000Z.20230827T000000Z.P01D00H00M.WORLD@10km.FRONTS.SFC.gml* [Forecasted fronts for 00z August 28, 2023 (valid time) drawn at 00z August 27, 2023 (init time).]
  * ###### Note that the valid and initialization time strings (e.g. "20230828T000000Z.20230827T000000Z", "20230827T090000Z.NIL") are the only required parts of the base filename outside the .gml suffix. The directory structure must be maintained.

If using TWC front labels, the GML files must be converted to XML files by running the convert_front_gml_to_xml.py script with the command below. Only **required** arguments are shown in the command line. If an argument shows up in the table below the command but is not in the command itself, that argument is **optional**.

    python convert_front_gml_to_xml.py --gml_indir {} --xml_outdir {} --date {} {} {}

| Argument      | Type     | Default | Description                            |
|---------------|----------|---------|----------------------------------------|
| *gml_indir*   | str      | (none)  | Input directory for the TWC GML files. |
| *xml_outdir*  | str      | (none)  | Output directory for the XML files.    |
| *date*        | int (3)  | (none)  | Year, month, and day.                  |

After obtaining the XML files, convert them to netCDF with the command below. *convert_front_xml_to_netcdf.py* will generate a netCDF for each XML file containing an initialization time with the provided date. (i.e. separate netCDF files will be created for 03z 2019-05-20 and 06z 2019-05-20 if the provided date is 2019-05-20)

    python convert_front_xml_to_netcdf.py --xml_indir {} --netcdf_outdir {} --date {}

| Argument        | Type         | Default   | Description                                                                                        |
|-----------------|--------------|-----------|----------------------------------------------------------------------------------------------------|
| *xml_indir*     | str          | (none)    | Input directory for the XML files.                                                                 |
| *netcdf_outdir* | str          | (none)    | Output parent directory for the netCDF files.                                                      |
| *date*          | str          | (none)    | String formatted as YYYY-MM-DD                                                                     |
| *distance*      | int or float | 1         | Interpolation distance for the fronts in kilometers.                                               |
| *domain*        | str          | "full"    | Domain for which to interpolate fronts over. To process IBM fronts, this must be set to 'global'.  |


  * The resulting netCDF files will be placed in subdirectories sorted by month (e.g. */netcdf/201304* contains all netCDF files for April 2013).

#### 1b. Predictor variables
Predictor variables can be obtained from multiple sources, however the main source used in training is ERA5. 

###### NOTE: This module currently does not support ERA5 data outside of NOAA's unified surface analysis domain. Downloaded ERA5 data will automatically be sliced to match this domain. We plan to support global ERA5 data in a future version.

* When downloading ERA5 data, make sure each file contains one year of 3-hourly data. Keep all data in a directory with two folders named *Surface* and *Pressure_Level*. (e.g. /data/era5/Surface, /data/era5/Pressure_Level)
  * At the **surface** level, download air temperature, dewpoint temperature, u-wind and v-wind, and surface pressure, with one file per variable. The base filename for 2-meter (surface) temperature data from 2008 will be *ERA5Global_2008_3hrly_2mT.nc*. Keep all surface files in the *Surface* folder as described above. There should be **five** ERA5 files for each year of surface data.
  * **Pressure level** data is downloaded in the same manner as above, however all pressure levels are contained within a single file. The pressure level variables needed are temperature, u-wind and v-wind, specific humidity, and geopotential height. The base filename for pressure level temperature data from 2008 will be *ERA5Global_PL_2008_3hrly_Q.nc*. Keep all pressure level files in the *Pressure_Level* folder as described above. There should be **five** ERA5 files for each year of pressure level data.
  * After downloading ERA5 data, the data must be sliced and additional variables must be calculated. This is accomplished in the *create_era5_netcdf.py* script:
  

    python create_era5_netcdf.py --netcdf_era5_indir {} --netcdf_outdir {} --date {} {} {}

| Argument               | Type     | Default | Description                                                                  |
|------------------------|----------|---------|------------------------------------------------------------------------------|
| *netcdf_era5_indir*    | str      | (none)  | Input directory for the ERA5 netCDF files.                                   |
| *netcdf_outdir*        | str      | (none)  | Output directory for the sliced ERA5 netCDF files with additional variables. |
| *date*                 | int (3)  | (none)  | Year, month, and day.                                                        |

* ###### All netCDF files will be stored in subdirectories sorted by month (e.g. */netcdf/201304* only contains data with initialization times in April 2013).

* Predictor variables can also be sourced from multiple NWP models using the *download_grib_files.py* script. Supported models include GFS, HRRR, NAM 12km, and the individual NAM nests.
  * Similar to sliced ERA5 netCDF files, downloaded GRIB files will be sorted into monthly directories.


    python download_grib_files.py --grib_outdir {} --model {} --init_time {}

| Argument         | Type       | Default | Description                                                                                                                                                                                                                     |
|------------------|------------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| *grib_outdir*    | str        | (none)  | Output directory for the downloaded GRIB files.                                                                                                                                                                                 |
| *model*          | str        | (none)  | Output directory for the sliced ERA5 netCDF files with additional variables.                                                                                                                                                    |
| *init_time*      | str        | (none)  | Initialization time of the model run, string formatted as YYYY-MM-DD-HH.                                                                                                                                                        |
| *range*          | str (3)    | (none)  | Date range and frequency of the data to download. 3 arguments must be passed: the start and end dates of the range, and the timestep frequency. Reference *download_grib_files.py* for additional information on this argument. |
| *forecast_hours* | int (xN)   | (none)  | List of forecast hours to download for the initialization time(s).                                                                                                                                                              |
| *verbose*        | store_true | N/A     | Print out the status for the GRIB file downloads.                                                                                                                                                                               |
###### Note that one of "init_time", "range" must be passed into the command line.

* After downloading the GRIB files, they must be converted to netCDF format with the *convert_grib_to_netcdf.py*. All forecast hours for a given initialization time are processed at once. The resulting netCDF files are sorted into monthly directories in the same manner as ERA5 files. For GFS and GDAS data, the base filename format is *model_YYYYMMDDHH_fFFF_global.nc* (FFF = forecast_hour). The *_global* string in the base filename is removed for all other models since they have their own specified domains.


      python convert_grib_to_netcdf.py --grib_indir {} --model {} --netcdf_outdir {} --init_time {} {} {} {}

| Argument               | Type       | Default | Description                                                 |
|------------------------|------------|---------|-------------------------------------------------------------|
| *grib_indir*           | str        | (none)  | Input directory for the GRIB files.                         |
| *model*                | str        | (none)  | NWP model from which the GRIB files originated.             |
| *netcdf_outdir*        | str        | (none)  | Output directory for the netCDF files.                      |
| *init_time*            | int (4)    | (none)  | Year, month, day, hour.                                     |
| *overwrite_grib*       | store_true | N/A     | Overwrite split GRIB files if they already exist.           |
| *delete_original_grib* | store_true | N/A     | Delete the downloaded GRIB files after they are split.      |
| *delete_split_grib*    | store_true | N/A     | Delete split GRIB files after they are converted to netCDF. |
| *gpu*                  | store_true | N/A     | Force additional variables to be calculated on a GPU.       |

# 2. TensorFlow datasets

TensorFlow datasets will be the inputs to the model. These datasets will be created from the netCDF files generated in section 1.

#### 2a. Overview

Three datasets will need to be generated: a training dataset, validation dataset, and a testing dataset. The **training dataset**
is used to **train the model**, the **validation dataset** is used to **tune the model's hyperparameters**, and the **testing dataset**
is used to **evaluate the model** with data that the model has never seen. The three datasets are split up by years. For example,
the training dataset may cover years 2008-2017, while the validation and test datasets cover 2018-2019 and 2020, respectively. In this example,
2020 data will not be used to train the model. The inputs to the models are the selected predictor variables, while the model outputs are
probabilities of each of the target front types.

#### 2b. Designing the datasets

There are several steps in the process of building TensorFlow datasets.

1. Choose the years for the dataset. Currently, only years 2008-2020 are supported (will be changed **very soon** to include 2021-2023). The same year cannot be used in multiple datasets (e.g. if 2016 data is used in the training dataset, in cannot be used in validation nor testing sets).
2. Determine the predictor variables (inputs) and front types (targets/labels) that will make up the dataset. Complete lists of variables and front types can be found in appendices 5b and 5c of this guide.
3. Choose what vertical levels to include in the inputs. The list of acceptable vertical levels can be found in appendix 5c.
4. Determine the shape (number of dimensions) of the inputs and targets. (e.g. do you want a model that takes 3D inputs and tries to predict 2D targets?)
5. Choose the domains of the datasets. All available domains can be found in appendix 5a.
6. Select the size of the images for the training/validation datasets and the number of images to extract from each timestep. **Note that by default a timestep will only be used in the final dataset if all requested front types are present in that timestep over the provided domain.**
   1. Determine whether or not you would like to retain timesteps that do not contain all requested front types. For example, if you build a dataset with cold and warm fronts, timesteps with cold fronts that do not also have warm fronts will not be included in the final dataset by default. You can, however, retain a fraction (or even all) of the images that do not have all requested front types.
7. Explore data augmentation and front expansion
   1. Data augmentation is the process of modifying the inputs to the model. Images can be modifying by adding noise and rotating or reflecting the images. 
   2. "Front expansion" refers to the process of expanding the labels identifying the presence of front boundaries at each grid point. Expanding the labels in the training and validation datasets trains the model to output larger frontal regions.
8. Finally, run the convert_netcdf_to_tf.py script to build the dataset.


    
    python convert_netcdf_to_tf.py --variables_netcdf_indir {} --fronts_netcdf_indir {} --tf_outdir {} --year_and_month {} --front_types {}


| Argument                 | Type       | Default | Description                                                                                   |
|--------------------------|------------|---------|-----------------------------------------------------------------------------------------------|
| *variables_netcdf_indir* | str        | (none)  | Input directory for the netCDF files containing variable data.                                |
| *fronts_netcdf_indir*    | str        | (none)  | Input directory for the netCDF files containing front labels.                                 |
| *tf_outdir*              | str        | (none)  | Output directory for the tensorflow datasets.                                                 |
| *year_and_month*         | int (2)    | (none)  | Year and month for which to generate tensorflow datasets.                                     |
| *front_types*            | str (N)    | (none)  | Front types to use as targets (Appendix 5c for options).                                      |
| *variables*              | str (N)    | (none)  | Variables to include in the inputs (Appendix 5b for options).                                 |
| *pressure_levels*        | str (N)    | (none)  | Pressure levels to include in the inputs (Appendix 5d for options).                           |
| *evaluation_dataset*     | store_true | N/A     | Set up the dataset so it can be used for evaluation. See the notes below this table for info. |
| *num_dims*               | int (2)    | 3 3     | Number of dimensions in the input data and front labels, respectively.                        |
| *domain*                 | str        | "conus" | Domain that the dataset will cover (Appendix 5a for options).                                 |
| *images*                 | int (2)    | 9 1     | Number of images to extract from the timestep along the longitude and latitude dimensions.    |
| *image_size*             | int (2)    | 128 128 | Size of the longitude and latitude dimensions of the images (# pixels).                       |
| *front_dilation*         | int        | 0       | Number of pixels to expand the front labels by in all directions.                             |
| *keep_fraction*          | float      | 0.0     | Fraction of timesteps not containing all front types to keep in the dataset. (0 <= x <= 1)    |
| *noise_fraction*         | float      | 0.0     | Fraction of pixels in each image that will contain salt and pepper noise. (0 <= x <= 1)       |
| *rotate_chance*          | float      | 0.0     | Chance that an image will be randomly rotated in 90° intervals. (0 <= x <= 1)                 |
| *flip_chance_lon*        | float      | 0.0     | Chance that an image will have its longitude dimension reversed. (0 <= x <= 1)                |
| *flip_chance_lat*        | float      | 0.0     | Chance that an image will have its latitude dimension reversed. (0 <= x <= 1)                 |
| *overwrite*              | store_true | N/A     | Overwrite the contents of any existing variables and fronts data.                             |
| *verbose*                | store_true | N/A     | Print out the progress of the dataset generation.                                             |
| *gpu_device*             | store_true | N/A     | GPU device numbers.                                                                           |
| *memory_growth*          | store_true | N/A     | Use memory growth on the GPU(s).                                                              |

###### NOTES: 
  * "evaluation_dataset" will override several of the table's arguments. See *convert_netcdf_to_tf.py* for more information.
  * Once the command is executed, a .pkl file and .txt file will be saved to *tf_outdir*. The .pkl file will contain properties of the dataset (values of the table's arguments), and the .txt file is a readable version of the .pkl file. 
  * After the .pkl file is created, the arguments in the .pkl file will be referenced in the future in order to create consistent datasets. For example, if a tensorflow dataset in */my_tf_ds* was initially created with the *num_dims* argument set to *3 2*, passing "--num_dims 3 3" into the command line with the same output directory (i.e. */my_tf_ds*) will have no effect. The .pkl file will be utilized to set critical arguments if the .pkl file exists. See *convert_netcdf_to_tf.py* for more information.

# 3. Model training

    python train_model.py  (too many arguments to list - consult the train_model.py script)

###### NOTES:
* There are many arguments for *train_model.py* and their descriptions and usages are too long to list in this guide. Consult *train_model.py* and read through each argument *carefully* as you will need to use most of the available arguments.
* All model architecture options can be found in Appendix 5e.

# 4. Evaluation

# 5. Appendix

### 5a. Domains
| Argument string | Domain                                             | Extent                         |
|-----------------|----------------------------------------------------|--------------------------------|
| *atlantic*      | Atlantic Ocean                                     | 16-55.75°N, 290-349.75°E       |
| *conus*         | Continental United States                          | 25-56.75°N, 228-299.75°E       |
| *full*          | Unified surface analysis domain used by NOAA       | 0-80°N, 130°E eastward to 10°E |
| *global*        | Global domain                                      | -89.75-90°N, 0-359.75°E        |
| *hrrr*          | High Resolution Rapid Refresh (HRRR) model domain  | non-uniform grid               |
| *nam-12km*      | 12-km North American Model (NAM) domain            | non-uniform grid               |
| *namnest-conus* | CONUS nest of the 3-km NAM                         | non-uniform grid               |
| *pacific*       | Pacific Ocean                                      | 16-55.75°N, 145-234.75°E       |

### 5b. Variables
| Argument string | Variable                                 |
|-----------------|------------------------------------------|
| *q*             | Specific humidity                        |
| *r*             | Mixing ratio                             |
| *RH*            | Relative humidity                        |
| *sp_z*          | Surface pressure and geopotential height |
| *theta*         | Potential temperature                    |
| *theta_e*       | Equivalent potential temperature         |
| *theta_v*       | Virtual potential temperature            |
| *theta_w*       | Wet-bulb potential temperature           |
| *T*             | Air temperature                          |
| *Td*            | Dewpoint temperature                     |
| *Tv*            | Virtual temperature                      |
| *Tw*            | Wet-bulb temperature                     |
| *u*             | U-component of wind                      |
| *v*             | V-component of wind                      |

### 5c. Front types
| Identifier | Argument string | Front type                     |
|------------|-----------------|--------------------------------|
| 1          | *CF*            | Cold front                     |
| 2          | *CF-F*          | Cold front (forming)           |
| 3          | *CF-D*          | Cold front (dissipating)       |
| 4          | *WF*            | Warm front                     |
| 5          | *WF-F*          | Warm front (forming)           |
| 6          | *WF-D*          | Warm front (dissipating)       |
| 7          | *SF*            | Stationary front               |
| 8          | *SF-F*          | Stationary front (forming)     |
| 9          | *SF-D*          | Stationary front (dissipating) |
| 10         | *OF*            | Occluded front                 |
| 11         | *OF-F*          | Occluded front (forming)       |
| 12         | *OF-D*          | Occluded front (dissipating)   |
| 13         | *INST*          | Outflow boundary               |
| 14         | *TROF*          | Trough                         |
| 15         | *TT*            | Tropical trough                |
| 16         | *DL*            | Dryline                        |

### 5d. Vertical levels
| Argument string | Level    |
|-----------------|----------|
| *surface*       | Surface  |
| *1000*          | 1000 hPa |
| *950*           | 950 hPa  |
| *900*           | 900 hPa  |
| *850*           | 850 hPa  |

### 5e. Models

| Argument string | Model          | Reference                                              |
|-----------------|----------------|--------------------------------------------------------|
| unet            | UNET           | https://arxiv.org/pdf/1505.04597.pdf                   |
| unet_plus       | UNET+          | https://arxiv.org/pdf/1912.05074.pdf                   |
| unet_2plus      | UNET++         | https://arxiv.org/pdf/1912.05074.pdf                   |
| unet_3plus      | UNET3+         | https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf |
| unet_ensemble   | UNet ensemble  | https://arxiv.org/pdf/1912.05074.pdf                   |
| attention_unet  | Attention UNET | https://arxiv.org/pdf/1505.04597.pdf                   |
