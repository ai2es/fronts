"""
Function that extracts data from a given domain and saves it into a pickle file.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 3/30/2021 11:41 CDT
"""

import argparse
import Plot_ERA5
import xarray as xr
import pickle

def extract_input_variables(lon, lat, year, month, day, hour, netcdf_ERA5_indir):
    """
    Extract surface data for the specified coordinate domain, year, month, day, and hour.

    Parameters
    ----------
    lon: float (x2)
        Two values that specify the longitude domain in degrees in the 360° coordinate system: lon_MIN lon_MAX
    lat: float (x2)
        Two values that specify the latitude domain in degrees: lat_MIN lat_MAX
    year: int
        Year for the surface data.
    month: int
        Month for the surface data.
    day: int
        Day for the surface data.
    hour: int
        Hour for the surface data.
    netcdf_ERA5_indir: str
        Directory where the ERA5 netCDF files are contained.

    Returns
    -------
    xr_pickle: Dataset
        Xarray dataset containing the surface data for the specified domain.
    filename: str
        Filename for the pickle file containing the surface data for the specified domain.
    """

    ds_2mT, ds_2mTd, ds_sp, ds_U10m, ds_V10m, ds_wind, da_theta_w = Plot_ERA5.create_datasets(year, month, day, hour,
                                                                                              netcdf_ERA5_indir)

    ds_2mT = ds_2mT.sel(longitude=slice(lon[0],lon[1]), latitude=slice(lat[1],lat[0]))
    ds_2mTd = ds_2mTd.sel(longitude=slice(lon[0],lon[1]), latitude=slice(lat[1],lat[0]))
    ds_sp = ds_sp.sel(longitude=slice(lon[0],lon[1]), latitude=slice(lat[1],lat[0]))
    ds_U10m = ds_U10m.sel(longitude=slice(lon[0],lon[1]), latitude=slice(lat[1],lat[0]))
    ds_V10m = ds_V10m.sel(longitude=slice(lon[0],lon[1]), latitude=slice(lat[1],lat[0]))
    ds_theta_w = (da_theta_w.sel(longitude=slice(lon[0],lon[1]), latitude=slice(lat[1],lat[0]))).to_dataset(name=
                                                                                                            'theta_w')
    array_2mT = ds_2mT.to_array()

    ds_pickle = [ds_2mT, ds_2mTd, ds_sp, ds_U10m, ds_V10m, ds_theta_w]
    xr_pickle = xr.combine_by_coords(ds_pickle)
    print(type(xr_pickle))

    datestring = "Extracting data for %04d-%02d-%02d %02dz...." % (year, month, day, hour)
    print(datestring,end='')

    filename = "%04d%02d%02d%02d_conus.pkl" % (year, month, day, hour)

    return xr_pickle, filename

def save_data_to_pickle(xr_pickle, filename, pickle_outdir):
    """
    Saves surface domain data to the pickle file.

    Parameters
    ----------
    xr_pickle: Dataset
        Xarray dataset containing the surface data for the specified domain.
    filename: str
        Filename for the pickle file containing the surface data for the specified domain.
    pickle_outdir: str
        Directory where the created pickle files containing the domain data will be stored.
    """

    outfile = open("%s/%s" % (pickle_outdir, filename), 'wb')
    pickle.dump(xr_pickle, outfile)
    outfile.close()

    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--netcdf_ERA5_indir', type=str, required=True, help="input directory for ERA5 netcdf files")
    parser.add_argument('--pickle_outdir', type=str, required=True, help="output directory for pickle files")
    parser.add_argument('--longitude', type=float, nargs=2, help="Longitude domain in degrees: lon_MIN lon_MAX")
    parser.add_argument('--latitude', type=float, nargs=2, help="Latitude domain in degrees: lat_MIN lat_MAX")
    parser.add_argument('--year', type=int, required=True, help="year for the data to be read in")
    parser.add_argument('--month', type=int, required=True, help="month for the data to be read in")
    parser.add_argument('--day', type=int, required=True, help="day for the data to be read in")
    parser.add_argument('--hour', type=int, required=True, help="hour for the data to be read in")
    args = parser.parse_args()

    df, cold_front, warm_front, occluded_front, stationary_front, dryline = Plot_ERA5.read_xml_files_ERA5(args.year,
                                                                                                          args.month,
                                                                                                          args.day,
                                                                                                          args.hour)
    cold_front_lon, cold_front_lat = Plot_ERA5.cold_front_latlon_arrays(df, cold_front)
    warm_front_lon, warm_front_lat = Plot_ERA5.warm_front_latlon_arrays(df, warm_front)

    xr_pickle, filename = extract_input_variables(args.longitude, args.latitude, args.year, args.month, args.day,
                                                    args.hour, args.netcdf_ERA5_indir)

    save_data_to_pickle(xr_pickle, filename, args.pickle_outdir)
