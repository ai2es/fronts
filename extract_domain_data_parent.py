"""
Function that loops through a specific range of years to create pickle files with surface data every 3 hours.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 3/30/2021 11:26 CDT
"""

import argparse
import os

def run_extract_domain_data(lon, lat, start_year, end_year, netcdf_ERA5_indir, pickle_outdir):
    """
    Loops through a specified range of years to extract pressure, temperature, dew point temperature, and wind data from
    the surface.

    Parameters
    ----------
    lon: float (x2)
        Two values that specify the longitude domain in degrees in the 360° coordinate system: lon_MIN lon_MAX
    lat: float (x2)
        Two values that specify the latitude domain in degrees: lat_MIN lat_MAX
    start_year: int
        Starting year for the domain data.
    end_year: int
        End year for the domain data.
    netcdf_ERA5_indir: str
        Directory where the ERA5 netCDF files are contained.
    pickle_outdir: str
        Directory where the created pickle files containing the domain data will be stored.
    """

    for year in range(start_year,end_year+1):
        for month in range(1,13):
            for day in range(1,32):
                for hour in range(0,24,3):
                    os.system("python extract_domain_data.py --longitude %d %d --latitude %d %d --year %d --month %d"
                              " --day %d --hour %d --netcdf_ERA5_indir %s --pickle_outdir %s"
                              % (lon[0], lon[1], lat[0], lat[1], year, month, day, hour, netcdf_ERA5_indir,
                                 pickle_outdir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--netcdf_ERA5_indir', type=str, required=True, help="input directory for ERA5 netcdf files")
    parser.add_argument('--pickle_outdir', type=str, required=True, help="output directory for pickle files")
    parser.add_argument('--longitude', type=float, nargs=2, help="Longitude domain in degrees: lon_MIN lon_MAX")
    parser.add_argument('--latitude', type=float, nargs=2, help="Latitude domain in degrees: lat_MIN lat_MAX")
    parser.add_argument('--start_year', type=int, required=True, help="start year for the data to be read in")
    parser.add_argument('--end_year', type=int, required=True, help="end year for the data to be read in")
    args = parser.parse_args()

    run_extract_domain_data(args.longitude, args.latitude, args.start_year, args.end_year, args.netcdf_ERA5_indir,
                            args.pickle_outdir)
