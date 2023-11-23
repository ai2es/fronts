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

If using TWC front labels, the GML files must be converted to XML files by running the convert_front_gml_to_xml.py script with the following command (all arguments are **required**):

    python convert_front_gml_to_xml.py --gml_indir {} --xml_outdir {} --date {} {} {}

    --gml_indir: Input directory containing the nested directory structure for the GML files.
    --xml_outdir: Output directory for the XML files.
    --date: Three integers representing the date: year, month, day

After obtaining the XML files, convert them to netCDF with the command below. *convert_front_xml_to_netcdf.py* will generate a netCDF for each XML file containing an initialization time with the provided date. (i.e. separate netCDF files will be created for 03z 2019-05-20 and 06z 2019-05-20 if the provided date is 2019-05-20)

    python convert_front_xml_to_netcdf.py --xml_indir {} --netcdf_outdir {} --date {}
    """
    *--xml_indir: Input directory for the XML files.
    *--netcdf_outdir: Output parent directory for the netCDF files. The files will be sorted in monthly directories (Ex: /netcdf/201905/...., where /netcdf is the parent directory)
    *--date: String formatted as YYYY-MM-DD. (Ex: "2019-05-20")
    --distance: Interpolation distance for the fronts in kilometers. (default = 1)
    --domain: Domain for which to interpolate fronts over. To process IBM fronts, this must be set to 'global'. (default = 'full')

    * required arguments
    """

  * The resulting netCDF files will be placed in subdirectories sorted by month (e.g. */netcdf/201304* contains all netCDF files for April 2013).

#### 1b. Predictor variables
Predictor variables can be obtained from multiple sources, however the main source used in training is ERA5. 

###### NOTE: This module currently does not support ERA5 data outside of NOAA's unified surface analysis domain. Downloaded ERA5 data will automatically be sliced to match this domain. We plan to support global ERA5 data in a future version.

* When downloading ERA5 data, make sure each file contains one year of 3-hourly data. Keep all data in a directory with two folders named *Surface* and *Pressure_Level*. (e.g. /data/era5/Surface, /data/era5/Pressure_Level)
  * At the **surface** level, download air temperature, dewpoint temperature, u-wind and v-wind, and surface pressure, with one file per variable. The base filename for 2-meter (surface) temperature data from 2008 will be *ERA5Global_2008_3hrly_2mT.nc*. Keep all surface files in the *Surface* folder as described above. There should be **five** ERA5 files for each year of surface data.
  * **Pressure level** data is downloaded in the same manner as above, however all pressure levels are contained within a single file. The pressure level variables needed are temperature, u-wind and v-wind, specific humidity, and geopotential height. The base filename for pressure level temperature data from 2008 will be *ERA5Global_PL_2008_3hrly_Q.nc*. Keep all pressure level files in the *Pressure_Level* folder as described above. There should be **five** ERA5 files for each year of pressure level data.
  * After downloading ERA5 data, the data must be sliced and additional variables must be calculated. This is accomplished in the *create_era5_netcdf.py* script (all arguments are **required**):

    
    python create_era5_netcdf.py --netcdf_era5_indir {} --netcdf_outdir {} --date {} {} {}

    --netcdf_era5_indir: Input directory for the ERA5 netCDF files (Ex: /data/era5)
    --netcdf_outdir: Output directory for the sliced ERA5 netCDF files with additional variables (Ex: /netcdf)
    --date: Three integers representing the date: year, month, day

###### All netCDF files will be stored in subdirectories sorted by month (e.g. */netcdf/201304* only contains data with initialization times in April 2013).

* Predictor variables can also be sourced from multiple NWP models using the *download_grib_files.py* script. Supported models include GFS, HRRR, NAM 12km, and the individual NAM nests.



## More useful information

#### Domains
* *atlantic*: Atlantic ocean. (16-55.75°N, 290-349.75°E)
* *conus*: Continental United States (CONUS). (25-56.75°N, 228-299.75°E)
* *full*: Unified surface analysis domain used by NOAA. (0-80°N, 130°E eastward to 10°E)
* *global*: Global domain. (-89.75-90°N, 0-359.75°E)
* *hrrr*: Domain for the High Resolution Rapid Refresh (HRRR) model.
* *nam-12km*: 12-km North American Model (NAM) domain.
* *namnest-conus*: CONUS nest of the 3-km NAM.
* *pacific*: Pacific ocean. (16-55.75°N, 145-234.75°E)
