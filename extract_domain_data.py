"""
Functions in this script create netcdf files containing ERA5, GDAS, or frontal object data.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 9/12/2022 3:37 PM CT
"""

import argparse
import urllib.error
import warnings
import requests
from bs4 import BeautifulSoup
import xarray as xr
import pandas as pd
from utils import data_utils
from errors import check_arguments
import glob
import numpy as np
import xml.etree.ElementTree as ET
import variables
import wget
import os


def check_directory(directory: str, year: int, month: int, day: int):
    """
    Check directory for subdirectories for the given year, month, and day.

    Parameters
    ----------
    directory: str
        - Directory to check
    year: int
    month: int
    day: int
    """

    assert os.path.isdir(directory)  # Check to see if the main directory is valid

    year_dir = '%s/%d' % (directory, year)
    month_dir = '%s/%02d' % (year_dir, month)
    day_dir = '%s/%02d' % (month_dir, day)
    
    ### Check for the correct subdirectories and make them if they do not exist ###
    if not os.path.isdir(year_dir):
        os.mkdir(year_dir)
        os.mkdir(month_dir)
        os.mkdir(day_dir)
    elif not os.path.isdir(month_dir):
        os.mkdir(month_dir)
        os.mkdir(day_dir)
    elif not os.path.isdir(day_dir):
        os.mkdir(day_dir)


def front_xmls_to_netcdf(year, month, day, xml_dir, netcdf_outdir):
    """
    Reads the xml files to pull frontal objects.

    Parameters
    ----------
    year: year
    month: month
    day: day
    xml_dir: str
        Directory where the front xml files are stored.
    netcdf_outdir: str
        Directory where the created netcdf files will be stored.
    """
    files = sorted(glob.glob("%s/*_%04d%02d%02d*f000.xml" % (xml_dir, year, month, day)))

    dss = []  # Dataset with front data organized by front type.
    for filename in files:
        print(filename)
        tree = ET.parse(filename, parser=ET.XMLParser(encoding='utf-8'))
        root = tree.getroot()
        date = filename.split('_')[-1].split('.')[0].split('f')[0]
        fronttype = []
        frontnumber = []
        fronts_number = []  # List array is for the interpolated points' front numbers.
        dates = []
        i = 0  # Define counter for front number.
        lats = []
        lons = []
        point_number = 0
        front_points = []  # List that shows which points are in each front.
        front_types = []
        front_dates = []
        fronts_lon_array = []
        fronts_lat_array = []
        for line in root.iter('Line'):
            i = i + 1
            frontno = i
            frontty = line.get("pgenType")
            if point_number != 0:
                front_points.append(point_number)
            for point in line.iter('Point'):
                point_number = point_number + 1
                dates.append(date)
                frontnumber.append(frontno)
                fronttype.append(frontty)
                lats.append(float(point.get("Lat")))

                # For longitude - we want to convert these values to a 360 system. This will allow us to properly
                # interpolate across the dateline. If we were to use a longitude domain of -180 to 180 rather than
                # 0 to 360, the interpolated points would wrap around the entire globe.
                if float(point.get("Lon")) < 0:
                    lons.append(float(point.get("Lon")) + 360)
                else:
                    lons.append(float(point.get("Lon")))

        # This is the second step in converting longitude to a 360 system. Fronts that cross the prime meridian are
        # only partially corrected due to one side of the front having a negative longitude coordinate. So, we will
        # convert the positive points to the system as well for a consistent adjustment. It is important to note that
        # these fronts that cross the prime meridian (0E/W) will have a range of 0 to 450 rather than 0 to 360.
        # This will NOT affect the interpolation of the fronts.
        for x in range(len(front_points)):
            if x == 0:
                for y in range(0, front_points[x - 1]):
                    if lons[y] < 90 and max(lons[0:front_points[x - 1]]) > 270:
                        lons[y] = lons[y] + 360
            else:
                for y in range(front_points[x - 1], front_points[x]):
                    if lons[y] < 90 and max(lons[front_points[x - 1]:front_points[x]]) > 270:
                        lons[y] = lons[y] + 360

        xs, ys = data_utils.haversine(lons, lats)

        # Reset the front counter.
        i = 0

        for line in root.iter('Line'):
            frontno = i
            frontty = line.get("pgenType")

            if i == 0:
                xs_new = xs[0:front_points[i]]
                ys_new = ys[0:front_points[i]]
            elif i < len(front_points):
                xs_new = xs[front_points[i - 1]:front_points[i]]
                ys_new = ys[front_points[i - 1]:front_points[i]]
            else:
                xs_new = xs[front_points[i - 1]:]
                ys_new = ys[front_points[i - 1]:]

            if len(xs_new) == 1:
                xs_new = np.append(xs_new, xs_new)
                ys_new = np.append(ys_new, ys_new)

            distance = 5  # Cartesian interval distance in kilometers.
            if (max(xs_new) > 100000 or min(xs_new) < -100000 or max(ys_new) > 100000 or min(
                    ys_new) < -100000):
                print(f"{filename}:   -----> Corrupt coordinates for front %d" % (frontno + 1))
            else:
                xy_linestring = data_utils.geometric(xs_new, ys_new)
                xy_vertices = data_utils.redistribute_vertices(xy_linestring, distance)
                x_new, y_new = xy_vertices.xy
                x_new = np.array(x_new)
                y_new = np.array(y_new)
                lon_new, lat_new = data_utils.reverse_haversine(x_new, y_new)
                for c in range(0, len(lon_new)):
                    fronts_lon_array.append(lon_new[c])
                    fronts_lat_array.append(lat_new[c])
                    fronts_number.append(frontno)
                    front_types.append(frontty)
                    front_dates.append(date)

            i += 1

        df = pd.DataFrame(list(zip(front_dates, fronts_number, front_types, fronts_lat_array, fronts_lon_array)),
                          columns=['time', 'Front Number', 'Front Type', 'latitude', 'longitude'])
        df['latitude'] = df.latitude.astype(float)
        df['longitude'] = df.longitude.astype(float)
        df['Front Number'] = df['Front Number'].astype(int)
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H')
        xit = np.linspace(0, 360, 1441)
        yit = np.linspace(80, 0, 321)
        df = df.assign(xit=np.digitize(df['longitude'].values, xit))
        df = df.assign(yit=np.digitize(df['latitude'].values, yit))
        types = df['Front Type'].unique()
        type_da = []
        for i in range(0, len(types)):
            type_df = df[df['Front Type'] == types[i]]
            groups = type_df.groupby(['xit', 'yit'])

            # Create variable "frequency" that describes the number of times a specific front type is present in each
            # gridspace.
            frequency = np.zeros((yit.shape[0] + 1, xit.shape[0] + 1))
            for group in groups:
                frequency[group[1].yit.values[0], group[1].xit.values[0]] += np.where(
                    len(group[1]['Front Number']) >= 1, 1, 0)
            frequency = frequency[1:322, 1:1442]
            ds = xr.Dataset(data_vars={"Frequency": (('latitude', 'longitude'), frequency)},
                            coords={'latitude': yit, 'longitude': xit})
            type_da.append(ds)
        ds = xr.concat(type_da, dim=types)
        ds = ds.rename({'concat_dim': 'type'})
        dss.append(ds)
    timestep = pd.date_range(start=df['time'].dt.strftime('%Y-%m-%d')[0], periods=len(dss), freq='3H')
    dns = xr.concat(dss, dim=timestep)
    dns = dns.rename({'concat_dim': 'time'})

    for hour in range(0, 24, 3):
        try:
            fronts = dns.sel(time='%d-%02d-%02dT%02d:00:00' % (year, month, day, hour))
            fronts = fronts.sel(longitude=np.append(np.arange(130, 360, 0.25), np.arange(0, 10.25, 0.25)))
            fronts['longitude'] = np.arange(130, 370.25, 0.25)
        except KeyError:
            print(f"ERROR: no fronts available for {year}-%02d-%02d-%02dz" % (month, day, hour))
        else:
            print(f"saving fronts for {year}-%02d-%02d-%02dz" % (month, day, hour))
            fronttype = np.empty([len(fronts.latitude), len(fronts.longitude)])
            time = fronts.time
            frequency = fronts.Frequency.values
            types = fronts.type.values
            lats = fronts.latitude.values
            lons = fronts.longitude.values

            for i in range(len(lats)):
                for j in range(len(lons)):
                    for k in range(len(types)):
                        if types[k] == 'COLD_FRONT':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 1
                            else:
                                fronttype[i][j] = 0
                        elif types[k] == 'WARM_FRONT':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 2
                        elif types[k] == 'STATIONARY_FRONT':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 3
                        elif types[k] == 'OCCLUDED_FRONT':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 4

                        elif types[k] == 'COLD_FRONT_FORM':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 5
                        elif types[k] == 'WARM_FRONT_FORM':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 6
                        elif types[k] == 'STATIONARY_FRONT_FORM':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 7
                        elif types[k] == 'OCCLUDED_FRONT_FORM':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 8

                        elif types[k] == 'COLD_FRONT_DISS':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 9
                        elif types[k] == 'WARM_FRONT_DISS':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 10
                        elif types[k] == 'STATIONARY_FRONT_DISS':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 11
                        elif types[k] == 'OCCLUDED_FRONT_DISS':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 12

                        elif types[k] == 'INSTABILITY':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 13
                        elif types[k] == 'TROF':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 14
                        elif types[k] == 'TROPICAL_TROF':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 15
                        elif types[k] == 'DRY_LINE':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 16

        filename_netcdf = "FrontObjects_%04d%02d%02d%02d_full.nc" % (year, month, day, hour)
        fronts_ds = xr.Dataset({"identifier": (('latitude', 'longitude'), fronttype)}, coords={"latitude": lats, "longitude": lons, "time": time})
        fronts_ds.to_netcdf(path="%s/%d/%02d/%02d/%s" % (netcdf_outdir, year, month, day, filename_netcdf), engine='scipy', mode='w')


def create_era5_datasets(year, month, day, netcdf_ERA5_indir, netcdf_outdir):
    """
    Extract ERA5 variable data for the specified year, month, day, and hour.

    Parameters
    ----------
    year: year
    month: month
    day: day
    netcdf_ERA5_indir: Directory where the ERA5 netCDF files are contained.
    netcdf_outdir: Directory where the created netcdf files will be stored.

    Returns
    -------
    xr_netcdf: Xarray dataset containing variable data for the full domain.
    """
    era5_T_sfc_file = 'ERA5Global_%d_3hrly_2mT.nc' % year
    era5_Td_sfc_file = 'ERA5Global_%d_3hrly_2mTd.nc' % year
    era5_sp_file = 'ERA5Global_%d_3hrly_sp.nc' % year
    era5_u_sfc_file = 'ERA5Global_%d_3hrly_U10m.nc' % year
    era5_v_sfc_file = 'ERA5Global_%d_3hrly_V10m.nc' % year

    timestring = "%d-%02d-%02d" % (year, month, day)

    lons = np.append(np.arange(130, 360, 0.25), np.arange(0, 10.25, 0.25))
    lats = np.arange(0, 80.25, 0.25)[::-1]
    lons360 = np.arange(130, 370.25, 0.25)

    T_sfc_full_day = xr.open_mfdataset("%s/%s" % (netcdf_ERA5_indir, era5_T_sfc_file), chunks={'latitude': 721, 'longitude': 1440, 'time': 4}).sel(time=('%s' % timestring), longitude=lons, latitude=lats)
    Td_sfc_full_day = xr.open_mfdataset("%s/%s" % (netcdf_ERA5_indir, era5_Td_sfc_file), chunks={'latitude': 721, 'longitude': 1440, 'time': 4}).sel(time=('%s' % timestring), longitude=lons, latitude=lats)
    sp_full_day = xr.open_mfdataset("%s/%s" % (netcdf_ERA5_indir, era5_sp_file), chunks={'latitude': 721, 'longitude': 1440, 'time': 4}).sel(time=('%s' % timestring), longitude=lons, latitude=lats)
    u_sfc_full_day = xr.open_mfdataset("%s/%s" % (netcdf_ERA5_indir, era5_u_sfc_file), chunks={'latitude': 721, 'longitude': 1440, 'time': 4}).sel(time=('%s' % timestring), longitude=lons, latitude=lats)
    v_sfc_full_day = xr.open_mfdataset("%s/%s" % (netcdf_ERA5_indir, era5_v_sfc_file), chunks={'latitude': 721, 'longitude': 1440, 'time': 4}).sel(time=('%s' % timestring), longitude=lons, latitude=lats)

    PL_data = xr.open_mfdataset(
        paths=('/ourdisk/hpc/ai2es/fronts/era5/Pressure_Level/ERA5Global_PL_%s_3hrly_Q.nc' % year,
               '/ourdisk/hpc/ai2es/fronts/era5/Pressure_Level/ERA5Global_PL_%s_3hrly_T.nc' % year,
               '/ourdisk/hpc/ai2es/fronts/era5/Pressure_Level/ERA5Global_PL_%s_3hrly_U.nc' % year,
               '/ourdisk/hpc/ai2es/fronts/era5/Pressure_Level/ERA5Global_PL_%s_3hrly_V.nc' % year,
               '/ourdisk/hpc/ai2es/fronts/era5/Pressure_Level/ERA5Global_PL_%s_3hrly_Z.nc' % year),
        chunks={'latitude': 721, 'longitude': 1440, 'time': 4}).sel(time=('%s' % timestring), longitude=lons, latitude=lats)

    for hour in range(0, 24, 3):

        print(f"saving ERA5 data for {year}-%02d-%02d-%02dz" % (month, day, hour))

        timestep = '%d-%02d-%02dT%02d:00:00' % (year, month, day, hour)

        PL_850 = PL_data.sel(level=850, time=timestep)
        PL_900 = PL_data.sel(level=900, time=timestep)
        PL_950 = PL_data.sel(level=950, time=timestep)
        PL_1000 = PL_data.sel(level=1000, time=timestep)

        T_sfc = T_sfc_full_day.sel(time=timestep)['t2m'].values
        Td_sfc = Td_sfc_full_day.sel(time=timestep)['d2m'].values
        sp = sp_full_day.sel(time=timestep)['sp'].values
        u_sfc = u_sfc_full_day.sel(time=timestep)['u10'].values
        v_sfc = v_sfc_full_day.sel(time=timestep)['v10'].values

        theta_sfc = variables.potential_temperature(T_sfc, sp)  # Potential temperature
        theta_e_sfc = variables.equivalent_potential_temperature(T_sfc, Td_sfc, sp)  # Equivalent potential temperature
        theta_v_sfc = variables.virtual_temperature_from_dewpoint(T_sfc, Td_sfc, sp)  # Virtual potential temperature
        theta_w_sfc = variables.wet_bulb_potential_temperature(T_sfc, Td_sfc, sp)  # Wet-bulb potential temperature
        r_sfc = variables.mixing_ratio_from_dewpoint(Td_sfc, sp)  # Mixing ratio
        q_sfc = variables.specific_humidity_from_dewpoint(Td_sfc, sp)  # Specific humidity
        RH_sfc = variables.relative_humidity(T_sfc, Td_sfc)  # Relative humidity
        Tv_sfc = variables.virtual_temperature_from_dewpoint(T_sfc, Td_sfc, sp)  # Virtual temperature
        Tw_sfc = variables.wet_bulb_temperature(T_sfc, Td_sfc)  # Wet-bulb temperature

        q_850 = PL_850['q'].values
        q_900 = PL_900['q'].values
        q_950 = PL_950['q'].values
        q_1000 = PL_1000['q'].values
        T_850 = PL_850['t'].values
        T_900 = PL_900['t'].values
        T_950 = PL_950['t'].values
        T_1000 = PL_1000['t'].values
        u_850 = PL_850['u'].values
        u_900 = PL_900['u'].values
        u_950 = PL_950['u'].values
        u_1000 = PL_1000['u'].values
        v_850 = PL_850['v'].values
        v_900 = PL_900['v'].values
        v_950 = PL_950['v'].values
        v_1000 = PL_1000['v'].values
        z_850 = PL_850['z'].values
        z_900 = PL_900['z'].values
        z_950 = PL_950['z'].values
        z_1000 = PL_1000['z'].values

        Td_850 = variables.dewpoint_from_specific_humidity(85000, T_850, q_850)
        Td_900 = variables.dewpoint_from_specific_humidity(90000, T_900, q_900)
        Td_950 = variables.dewpoint_from_specific_humidity(95000, T_950, q_950)
        Td_1000 = variables.dewpoint_from_specific_humidity(100000, T_1000, q_1000)
        r_850 = variables.mixing_ratio_from_dewpoint(Td_850, 85000)
        r_900 = variables.mixing_ratio_from_dewpoint(Td_900, 90000)
        r_950 = variables.mixing_ratio_from_dewpoint(Td_950, 95000)
        r_1000 = variables.mixing_ratio_from_dewpoint(Td_1000, 100000)
        RH_850 = variables.relative_humidity(T_850, Td_850)
        RH_900 = variables.relative_humidity(T_900, Td_900)
        RH_950 = variables.relative_humidity(T_950, Td_950)
        RH_1000 = variables.relative_humidity(T_1000, Td_1000)
        theta_850 = variables.potential_temperature(T_850, 85000)
        theta_900 = variables.potential_temperature(T_900, 90000)
        theta_950 = variables.potential_temperature(T_950, 95000)
        theta_1000 = variables.potential_temperature(T_1000, 100000)
        theta_e_850 = variables.equivalent_potential_temperature(T_850, Td_850, 85000)
        theta_e_900 = variables.equivalent_potential_temperature(T_900, Td_900, 90000)
        theta_e_950 = variables.equivalent_potential_temperature(T_950, Td_950, 95000)
        theta_e_1000 = variables.equivalent_potential_temperature(T_1000, Td_1000, 100000)
        theta_v_850 = variables.virtual_temperature_from_dewpoint(T_850, Td_850, 85000)
        theta_v_900 = variables.virtual_temperature_from_dewpoint(T_900, Td_900, 90000)
        theta_v_950 = variables.virtual_temperature_from_dewpoint(T_950, Td_950, 95000)
        theta_v_1000 = variables.virtual_temperature_from_dewpoint(T_1000, Td_1000, 100000)
        theta_w_850 = variables.wet_bulb_potential_temperature(T_850, Td_850, 85000)
        theta_w_900 = variables.wet_bulb_potential_temperature(T_900, Td_900, 90000)
        theta_w_950 = variables.wet_bulb_potential_temperature(T_950, Td_950, 95000)
        theta_w_1000 = variables.wet_bulb_potential_temperature(T_1000, Td_1000, 100000)
        Tv_850 = variables.virtual_temperature_from_dewpoint(T_850, Td_850, 85000)
        Tv_900 = variables.virtual_temperature_from_dewpoint(T_900, Td_900, 90000)
        Tv_950 = variables.virtual_temperature_from_dewpoint(T_950, Td_950, 95000)
        Tv_1000 = variables.virtual_temperature_from_dewpoint(T_1000, Td_1000, 100000)
        Tw_850 = variables.wet_bulb_temperature(T_850, Td_850)
        Tw_900 = variables.wet_bulb_temperature(T_900, Td_900)
        Tw_950 = variables.wet_bulb_temperature(T_950, Td_950)
        Tw_1000 = variables.wet_bulb_temperature(T_1000, Td_1000)

        pressure_levels = ['surface', 1000, 950, 900, 850]

        T = np.empty(shape=(len(pressure_levels), len(lats), len(lons360)))
        Td = np.empty(shape=(len(pressure_levels), len(lats), len(lons360)))
        Tv = np.empty(shape=(len(pressure_levels), len(lats), len(lons360)))
        Tw = np.empty(shape=(len(pressure_levels), len(lats), len(lons360)))
        theta = np.empty(shape=(len(pressure_levels), len(lats), len(lons360)))
        theta_e = np.empty(shape=(len(pressure_levels), len(lats), len(lons360)))
        theta_v = np.empty(shape=(len(pressure_levels), len(lats), len(lons360)))
        theta_w = np.empty(shape=(len(pressure_levels), len(lats), len(lons360)))
        RH = np.empty(shape=(len(pressure_levels), len(lats), len(lons360)))
        r = np.empty(shape=(len(pressure_levels), len(lats), len(lons360)))
        q = np.empty(shape=(len(pressure_levels), len(lats), len(lons360)))
        u = np.empty(shape=(len(pressure_levels), len(lats), len(lons360)))
        v = np.empty(shape=(len(pressure_levels), len(lats), len(lons360)))
        sp_z = np.empty(shape=(len(pressure_levels), len(lats), len(lons360)))

        T[0, :, :], T[1, :, :], T[2, :, :], T[3, :, :], T[4, :, :] = T_sfc, T_1000, T_950, T_900, T_850
        Td[0, :, :], Td[1, :, :], Td[2, :, :], Td[3, :, :], Td[4, :, :] = Td_sfc, Td_1000, Td_950, Td_900, Td_850
        Tv[0, :, :], Tv[1, :, :], Tv[2, :, :], Tv[3, :, :], Tv[4, :, :] = Tv_sfc, Tv_1000, Tv_950, Tv_900, Tv_850
        Tw[0, :, :], Tw[1, :, :], Tw[2, :, :], Tw[3, :, :], Tw[4, :, :] = Tw_sfc, Tw_1000, Tw_950, Tw_900, Tw_850
        theta[0, :, :], theta[1, :, :], theta[2, :, :], theta[3, :, :], theta[4, :, :] = theta_sfc, theta_1000, theta_950, theta_900, theta_850
        theta_e[0, :, :], theta_e[1, :, :], theta_e[2, :, :], theta_e[3, :, :], theta_e[4, :, :] = theta_e_sfc, theta_e_1000, theta_e_950, theta_e_900, theta_e_850
        theta_v[0, :, :], theta_v[1, :, :], theta_v[2, :, :], theta_v[3, :, :], theta_v[4, :, :] = theta_v_sfc, theta_v_1000, theta_v_950, theta_v_900, theta_v_850
        theta_w[0, :, :], theta_w[1, :, :], theta_w[2, :, :], theta_w[3, :, :], theta_w[4, :, :] = theta_w_sfc, theta_w_1000, theta_w_950, theta_w_900, theta_w_850
        RH[0, :, :], RH[1, :, :], RH[2, :, :], RH[3, :, :], RH[4, :, :] = RH_sfc, RH_1000, RH_950, RH_900, RH_850
        r[0, :, :], r[1, :, :], r[2, :, :], r[3, :, :], r[4, :, :] = r_sfc, r_1000, r_950, r_900, r_850
        q[0, :, :], q[1, :, :], q[2, :, :], q[3, :, :], q[4, :, :] = q_sfc, q_1000, q_950, q_900, q_850
        u[0, :, :], u[1, :, :], u[2, :, :], u[3, :, :], u[4, :, :] = u_sfc, u_1000, u_950, u_900, u_850
        v[0, :, :], v[1, :, :], v[2, :, :], v[3, :, :], v[4, :, :] = v_sfc, v_1000, v_950, v_900, v_850
        sp_z[0, :, :], sp_z[1, :, :], sp_z[2, :, :], sp_z[3, :, :], sp_z[4, :, :] = sp/100, z_1000/98.0665, z_950/98.0665, z_900/98.0665, z_850/98.0665

        full_era5_dataset = xr.Dataset(data_vars=dict(T=(('pressure_level', 'latitude', 'longitude'), T),
                                                      Td=(('pressure_level', 'latitude', 'longitude'), Td),
                                                      Tv=(('pressure_level', 'latitude', 'longitude'), Tv),
                                                      Tw=(('pressure_level', 'latitude', 'longitude'), Tw),
                                                      theta=(('pressure_level', 'latitude', 'longitude'), theta),
                                                      theta_e=(('pressure_level', 'latitude', 'longitude'), theta_e),
                                                      theta_v=(('pressure_level', 'latitude', 'longitude'), theta_v),
                                                      theta_w=(('pressure_level', 'latitude', 'longitude'), theta_w),
                                                      RH=(('pressure_level', 'latitude', 'longitude'), RH),
                                                      r=(('pressure_level', 'latitude', 'longitude'), r * 1000),
                                                      q=(('pressure_level', 'latitude', 'longitude'), q * 1000),
                                                      u=(('pressure_level', 'latitude', 'longitude'), u),
                                                      v=(('pressure_level', 'latitude', 'longitude'), v),
                                                      sp_z=(('pressure_level', 'latitude', 'longitude'), sp_z)),
                                       coords=dict(pressure_level=pressure_levels, latitude=lats, longitude=lons360)).astype('float32')

        full_era5_dataset = full_era5_dataset.expand_dims({'time': np.atleast_1d(timestep)})

        for variable in list(full_era5_dataset.keys()):
            full_era5_dataset[variable].to_netcdf(path='%s/%d/%02d/%02d/era5_%s_%d%02d%02d%02d_full.nc' % (netcdf_outdir, year, month, day, variable, year, month, day, hour), mode='w', engine='scipy')


def download_ncep_has_order(ncep_has_order_number, ncep_has_outdir):
    """
    Download files as part of a HAS order from the NCEP.

    ncep_has_order_number: str
        - HAS order number for the GDAS data.
    ncep_has_outdir: str
        - Directory where the GDAS or GFS file will be saved to.
    """

    print("Connecting")
    url = f'https://www.ncei.noaa.gov/pub/has/model/HAS{ncep_has_order_number}/'
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    files = [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('.tar')]

    for file in files:
        ftp_file = file.replace('https://www', 'ftp://ftp')
        local_filename = file.replace(url, '')
        if not os.path.isfile(f'{ncep_has_outdir}/{local_filename}'):
            try:
                wget.download(ftp_file, out=f"{ncep_has_outdir}/{local_filename}")
            except urllib.error.HTTPError:
                print(f"Error downloading {url.replace('https://www', 'ftp://ftp')}{file}")
                pass


def download_gdas_grib_files(gdas_grib_outdir, year, day_range=(0, 364)):
    """
    Download GDAS grib files for a given timestep from the AWS server.

    gdas_outdir: str
        Main directory for the GDAS grib files.
    """

    if year % 4 == 0:
        month_2_days = 29  # Leap year
    else:
        month_2_days = 28  # Not a leap year

    days_per_month = [31, month_2_days, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    ### Generate a list of dates [[year, 1, 1], [year, ..., ...], [year, 12, 31]] ###
    date_list = []
    for month in range(12):
        for day in range(days_per_month[month]):
            date_list.append([year, month + 1, day + 1])

    for day_index in range(day_range[0], day_range[1] + 1):

        month, day = date_list[day_index][1], date_list[day_index][2]
        hours = np.arange(0, 24, 6)  # Hours that the GDAS model runs forecasts for

        daily_directory = gdas_grib_outdir + '/%d%02d%02d' % (year, month, day)  # Directory for the GDAS grib files for the given day

        # GDAS files have different naming patterns based on which year the data is for
        if year < 2012:
            forecast_hours = np.arange(0, 12, 3)
            files = [f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.{year}%02d%02d/%02d/gdas1.t%02dz.pgrbf%02d.grib2' % (month, day, hour, hour, forecast_hour)
                     for hour in hours
                     for forecast_hour in forecast_hours]
            local_filenames = ['gdas1.t%02dz.pgrbf%02d.grib2' % (hour, forecast_hour)
                               for hour in hours
                               for forecast_hour in forecast_hours]

        elif 2012 < year < 2018:

            forecast_hours = np.arange(0, 10)  # Forecast hours 0-9

            files = [f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.{year}%02d%02d/%02d/gdas1.t%02dz.pgrb2.0p25.f%03d' % (month, day, hour, hour, forecast_hour)
                     for hour in hours
                     for forecast_hour in forecast_hours]
            local_filenames = ['gdas.t%02dz.pgrb2.0p25.f%03d' % (hour, forecast_hour)
                               for hour in hours
                               for forecast_hour in forecast_hours]

        elif year > 2018:

            forecast_hours = np.arange(0, 10)  # Forecast hours 0-9

            files = [f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.{year}%02d%02d/%02d/gdas.t%02dz.pgrb2.0p25.f%03d' % (month, day, hour, hour, forecast_hour)
                     for hour in hours
                     for forecast_hour in forecast_hours]

            local_filenames = ['gdas.t%02dz.pgrb2.0p25.f%03d' % (hour, forecast_hour)
                               for hour in hours
                               for forecast_hour in forecast_hours]

        for file, local_filename in zip(files, local_filenames):

            ### If the directory does not exist, check to see if the file link is valid. If the file link is NOT valid, then the directory will not be created since it will be empty. ###
            if not os.path.isdir(daily_directory):
                if requests.head(file).status_code == requests.codes.ok:
                    os.mkdir(daily_directory)

            full_file_path = f'{daily_directory}/{local_filename}'

            if not os.path.isfile(full_file_path) and os.path.isdir(daily_directory):
                try:
                    wget.download(file, out=full_file_path)
                except urllib.error.HTTPError:
                    print(f"Error downloading {file}")
                    pass

            elif not os.path.isdir(daily_directory):
                warnings.warn(f"Unknown problem encountered when creating the following directory: {daily_directory}, "
                              f"Consider checking the AWS server to make sure that data exists for the given day ({year}-%02d-%02d): "
                              f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/index.html#gdas.{year}%02d%02d" % (date_list[day_index][1], date_list[day_index][2], date_list[day_index][1], date_list[day_index][2]))
                break

            elif os.path.isfile(full_file_path):
                print(f"{full_file_path} already exists, skipping file....")


def download_latest_gdas_data(gdas_grib_outdir):
    """
    Download the latest GDAS grib files from NCEP.

    gdas_outdir: str
        Main directory for the GDAS grib files.
    """

    print(f"Updating GDAS files in directory: {gdas_grib_outdir}")

    print("Connecting to main GDAS source")
    url = f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/'
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    gdas_folders_by_day = [url + node.get('href') for node in soup.find_all('a') if 'gdas' in node.get('href')]
    timesteps_available = [folder[-9:-1] for folder in gdas_folders_by_day if 'prod/gdas' in folder][::-1]

    files_added = 0

    for timestep in timesteps_available:

        year, month, day = timestep[:4], timestep[4:6], timestep[6:]
        daily_directory = '%s/%s' % (gdas_grib_outdir, timestep)

        url = f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gdas.{timestep}/'
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'html.parser')

        gdas_folders_by_hour = [url + node.get('href') for node in soup.find_all('a')]
        hours_available = [folder[-3:-1] for folder in gdas_folders_by_hour if any(hour in folder[-3:-1] for hour in ['00', '06', '12', '18'])][::-1]

        for hour in hours_available:

            print(f"\nSearching for available data: {year}-{month}-{day}-%02dz" % float(hour))

            url = f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gdas.{timestep}/%02d/atmos/' % float(hour)
            page = requests.get(url).text
            soup = BeautifulSoup(page, 'html.parser')
            files = [url + node.get('href') for node in soup.find_all('a') if 'pgrb2.0p25' in node.get('href')]

            if not os.path.isdir(daily_directory):
                os.mkdir(daily_directory)

            for file in files:
                local_filename = file.replace(url, '')
                full_file_path = f'{daily_directory}/{local_filename}'
                if not os.path.isfile(full_file_path):
                    try:
                        wget.download(file, out=full_file_path)
                        files_added += 1
                    except urllib.error.HTTPError:
                        print(f"Error downloading {url}{file}")
                        pass
                else:
                    print(f"{full_file_path} already exists")

    print("GDAS directory update complete, %d files added" % files_added)


def download_latest_gfs_data(gfs_grib_outdir):
    """
    Download the latest GFS grib files from NCEP.

    gfs_outdir: str
        Main directory for the GFS grib files.
    """

    print(f"Updating GFS files in directory: {gfs_grib_outdir}")

    print("Connecting to main GFS source")
    url = f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/'
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    gfs_folders_by_day = [url + node.get('href') for node in soup.find_all('a') if 'gfs' in node.get('href')]
    timesteps_available = [folder[-9:-1] for folder in gfs_folders_by_day if 'prod/gfs' in folder][::-1]

    files_added = 0

    for timestep in timesteps_available:

        year, month, day = timestep[:4], timestep[4:6], timestep[6:]
        daily_directory = '%s/%s' % (gfs_grib_outdir, timestep)

        url = f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{timestep}/'
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'html.parser')

        gfs_folders_by_hour = [url + node.get('href') for node in soup.find_all('a')]
        hours_available = [folder[-3:-1] for folder in gfs_folders_by_hour if any(hour in folder[-3:-1] for hour in ['00', '06', '12', '18'])][::-1]

        for hour in hours_available:

            print(f"\nSearching for available data: {year}-{month}-{day}-%02dz" % float(hour))

            url = f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{timestep}/%02d/atmos/' % float(hour)
            page = requests.get(url).text
            soup = BeautifulSoup(page, 'html.parser')
            files = [url + node.get('href') for node in soup.find_all('a') if 'pgrb2.0p25.f00' in node.get('href')]

            if not os.path.isdir(daily_directory):
                os.mkdir(daily_directory)

            for file in files:
                local_filename = file.replace(url, '')
                full_file_path = f'{daily_directory}/{local_filename}'
                if not os.path.isfile(full_file_path):
                    try:
                        wget.download(file, out=full_file_path)
                        files_added += 1
                    except urllib.error.HTTPError:
                        print(f"Error downloading {url}{file}")
                        pass
                else:
                    print(f"{full_file_path} already exists")

    print("GFS directory update complete, %d files added" % files_added)


def gdas_grib_to_netcdf(year, month, day, hour, gdas_grib_indir, netcdf_outdir, forecast_hour=6):
    """
    Create GDAS netcdf files from the grib files.

    Parameters
    ----------
    year: year
    month: month
    day: day
    hour: hour
    gdas_grib_indir: Directory where the GDAS grib files are stored.
    netcdf_outdir: Directory where the created GDAS or GFS netcdf files will be stored.
    forecast_hour: Hour for the forecast.
    """

    # all lon/lat values in degrees
    start_lon, end_lon = 130, 370  # western boundary, eastern boundary
    start_lat, end_lat = 80, 0  # northern boundary, southern boundary

    # coordinate indices for the unified surface analysis domain in the GDAS datasets
    unified_longitude_indices = np.append(np.arange(start_lon * 4, (end_lon - 10) * 4), np.arange(0, (end_lon - 360) * 4 + 1))
    unified_latitude_indices = np.arange((90 - start_lat) * 4, (90 - end_lat) * 4 + 1)

    gdas_file_formats = ['%s/%d%02d%02d/gdas1.t%02dz.pgrb2.0p25.f%03d' % (gdas_grib_indir, year, month, day, hour, forecast_hour),
                         '%s/%d%02d%02d/gdas.t%02dz.pgrb2.0p25.f%03d' % (gdas_grib_indir, year, month, day, hour, forecast_hour)]

    ###############################################################################################################
    ############################################# Pressure level data #############################################
    ###############################################################################################################

    file_found = False

    attempt_no = 0
    max_attempts = len(gdas_file_formats)

    while file_found is False and attempt_no < max_attempts:
        if not os.path.isfile(gdas_file_formats[attempt_no]):
            attempt_no += 1
            if attempt_no == max_attempts:
                raise FileNotFoundError(f"No GDAS file found for {year}-%02d-%02d-%02dz f%03d" % (month, day, hour, forecast_hour))
            else:
                pass
        else:
            file_found = True
            pressure_level_data = xr.open_dataset(gdas_file_formats[attempt_no], engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}}, chunks={'latitude': 721, 'longitude': 1440})
            surface_data = xr.open_dataset(gdas_file_formats[attempt_no], engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'sigma'}}, chunks={'latitude': 721, 'longitude': 1440})
            raw_pressure_data = xr.open_dataset(gdas_file_formats[attempt_no], engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'stepType': 'instant'}}, chunks={'latitude': 721, 'longitude': 1440})
            mslp_data = xr.open_dataset(gdas_file_formats[attempt_no], engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}}, chunks={'latitude': 721, 'longitude': 1440})

    sp = raw_pressure_data.isel(longitude=unified_longitude_indices, latitude=unified_latitude_indices)['sp'].values / 100  # select unified surface analysis domain
    unified_surface_data = surface_data.isel(longitude=unified_longitude_indices, latitude=unified_latitude_indices)  # select unified surface analysis domain
    unified_mslp_data = mslp_data.isel(longitude=unified_longitude_indices, latitude=unified_latitude_indices)  # select unified surface analysis domain

    print("generating GDAS pressure level data for %d-%02d-%02d-%02dz" % (year, month, day, hour))

    pressure_level_indices = np.array([0, 1, 2, 3, 4, 5, 8])

    unified_pressure_level_data = pressure_level_data.isel(longitude=unified_longitude_indices, latitude=unified_latitude_indices, isobaricInhPa=pressure_level_indices).drop_vars(['valid_time', 'step'])  # unified surface analysis domain
    unified_pressure_level_data['longitude'] = np.arange(start_lon, end_lon + 0.25, 0.25)  # reformat longitude coordinates to a 360 degree system

    # Change data type of the coordinates and pressure levels to reduce the size of the data
    unified_pressure_level_data['longitude'] = unified_pressure_level_data['longitude'].values.astype('float32')
    unified_pressure_level_data['latitude'] = unified_pressure_level_data['latitude'].values.astype('float32')
    unified_pressure_level_data['isobaricInhPa'] = unified_pressure_level_data['isobaricInhPa'].values.astype('float32')

    pressure_levels = unified_pressure_level_data['isobaricInhPa'].values
    P = np.empty(shape=(len(pressure_level_indices), (start_lat - end_lat) * 4 + 1, (end_lon - start_lon) * 4 + 1))  # create 3D array of pressure levels to match the shape of variable arrays
    for pressure_level in range(len(pressure_level_indices)):
        P[pressure_level, :, :] = pressure_levels[pressure_level] * 100

    T_pl = unified_pressure_level_data['t'].values
    RH_pl = unified_pressure_level_data['r'].values / 100
    vap_pres_pl = RH_pl * variables.vapor_pressure(T_pl)
    Td_pl = variables.dewpoint_from_vapor_pressure(vap_pres_pl)
    Tv_pl = variables.virtual_temperature_from_dewpoint(T_pl, Td_pl, P)
    Tw_pl = variables.wet_bulb_temperature(T_pl, Td_pl)
    r_pl = variables.mixing_ratio_from_dewpoint(Td_pl, P) * 1000  # Convert to g/kg
    q_pl = variables.specific_humidity_from_dewpoint(Td_pl, P) * 1000  # Convert to g/kg
    theta_pl = variables.potential_temperature(T_pl, P)
    theta_e_pl = variables.equivalent_potential_temperature(T_pl, Td_pl, P)
    theta_v_pl = variables.virtual_potential_temperature(T_pl, Td_pl, P)
    theta_w_pl = variables.wet_bulb_potential_temperature(T_pl, Td_pl, P)
    z = unified_pressure_level_data['gh'].values / 10  # Convert to dam
    u_pl = unified_pressure_level_data['u'].values
    v_pl = unified_pressure_level_data['v'].values

    ################################################################################################################
    ################################# Surface data (MSLP and sigma coordinate data) ################################
    ################################################################################################################

    print("generating GDAS surface data for %d-%02d-%02d-%02dz" % (year, month, day, hour))

    # Create arrays of coordinates for the surface data
    surface_data_longitudes = np.arange(start_lon, end_lon + 0.25, 0.25).astype('float32')
    surface_data_latitudes = unified_surface_data['latitude'].values.astype('float32')

    mslp = unified_mslp_data['mslet'].values  # mean sea level pressure (eta model reduction)
    T_sigma = unified_surface_data['t'].values
    RH_sigma = unified_surface_data['r'].values / 100
    vap_pres_sigma = RH_sigma * variables.vapor_pressure(T_sigma)
    Td_sigma = variables.dewpoint_from_vapor_pressure(vap_pres_sigma)
    Tv_sigma = variables.virtual_temperature_from_dewpoint(T_sigma, Td_sigma, mslp)
    Tw_sigma = variables.wet_bulb_temperature(T_sigma, Td_sigma)
    r_sigma = variables.mixing_ratio_from_dewpoint(Td_sigma, mslp) * 1000  # Convert to g/kg
    q_sigma = variables.specific_humidity_from_dewpoint(Td_sigma, mslp) * 1000  # Convert to g/kg
    theta_sigma = variables.potential_temperature(T_sigma, mslp)
    theta_e_sigma = variables.equivalent_potential_temperature(T_sigma, Td_sigma, mslp)
    theta_v_sigma = variables.virtual_potential_temperature(T_sigma, Td_sigma, mslp)
    theta_w_sigma = variables.wet_bulb_potential_temperature(T_sigma, Td_sigma, mslp)
    u_sigma = unified_surface_data['u'].values
    v_sigma = unified_surface_data['v'].values

    T = np.empty(shape=(len(pressure_level_indices) + 1, len(surface_data_latitudes), len(surface_data_longitudes)))
    Td = np.empty(shape=(len(pressure_level_indices) + 1, len(surface_data_latitudes), len(surface_data_longitudes)))
    Tv = np.empty(shape=(len(pressure_level_indices) + 1, len(surface_data_latitudes), len(surface_data_longitudes)))
    Tw = np.empty(shape=(len(pressure_level_indices) + 1, len(surface_data_latitudes), len(surface_data_longitudes)))
    theta = np.empty(shape=(len(pressure_level_indices) + 1, len(surface_data_latitudes), len(surface_data_longitudes)))
    theta_e = np.empty(shape=(len(pressure_level_indices) + 1, len(surface_data_latitudes), len(surface_data_longitudes)))
    theta_v = np.empty(shape=(len(pressure_level_indices) + 1, len(surface_data_latitudes), len(surface_data_longitudes)))
    theta_w = np.empty(shape=(len(pressure_level_indices) + 1, len(surface_data_latitudes), len(surface_data_longitudes)))
    RH = np.empty(shape=(len(pressure_level_indices) + 1, len(surface_data_latitudes), len(surface_data_longitudes)))
    r = np.empty(shape=(len(pressure_level_indices) + 1, len(surface_data_latitudes), len(surface_data_longitudes)))
    q = np.empty(shape=(len(pressure_level_indices) + 1, len(surface_data_latitudes), len(surface_data_longitudes)))
    u = np.empty(shape=(len(pressure_level_indices) + 1, len(surface_data_latitudes), len(surface_data_longitudes)))
    v = np.empty(shape=(len(pressure_level_indices) + 1, len(surface_data_latitudes), len(surface_data_longitudes)))
    mslp_z = np.empty(shape=(len(pressure_level_indices) + 1, len(surface_data_latitudes), len(surface_data_longitudes)))
    sp_z = np.empty(shape=(len(pressure_level_indices) + 1, len(surface_data_latitudes), len(surface_data_longitudes)))

    T[0, :, :] = T_sigma
    T[1:, :, :] = T_pl
    Td[0, :, :] = Td_sigma
    Td[1:, :, :] = Td_pl
    Tv[0, :, :] = Tv_sigma
    Tv[1:, :, :] = Tv_pl
    Tw[0, :, :] = Tw_sigma
    Tw[1:, :, :] = Tw_pl
    theta[0, :, :] = theta_sigma
    theta[1:, :, :] = theta_pl
    theta_e[0, :, :] = theta_e_sigma
    theta_e[1:, :, :] = theta_e_pl
    theta_v[0, :, :] = theta_v_sigma
    theta_v[1:, :, :] = theta_v_pl
    theta_w[0, :, :] = theta_w_sigma
    theta_w[1:, :, :] = theta_w_pl
    RH[0, :, :] = RH_sigma
    RH[1:, :, :] = RH_pl
    r[0, :, :] = r_sigma
    r[1:, :, :] = r_pl
    q[0, :, :] = q_sigma
    q[1:, :, :] = q_pl
    u[0, :, :] = u_sigma
    u[1:, :, :] = u_pl
    v[0, :, :] = v_sigma
    v[1:, :, :] = v_pl
    mslp_z[0, :, :] = mslp
    mslp_z[1:, :, :] = z
    sp_z[0, :, :] = sp
    sp_z[1:, :, :] = z

    pressure_levels = ['surface', 1000, 975, 950, 925, 900, 850, 700]

    full_gdas_dataset = xr.Dataset(data_vars=dict(T=(('pressure_level', 'latitude', 'longitude'), T),
                                                  Td=(('pressure_level', 'latitude', 'longitude'), Td),
                                                  Tv=(('pressure_level', 'latitude', 'longitude'), Tv),
                                                  Tw=(('pressure_level', 'latitude', 'longitude'), Tw),
                                                  theta=(('pressure_level', 'latitude', 'longitude'), theta),
                                                  theta_e=(('pressure_level', 'latitude', 'longitude'), theta_e),
                                                  theta_v=(('pressure_level', 'latitude', 'longitude'), theta_v),
                                                  theta_w=(('pressure_level', 'latitude', 'longitude'), theta_w),
                                                  RH=(('pressure_level', 'latitude', 'longitude'), RH),
                                                  r=(('pressure_level', 'latitude', 'longitude'), r),
                                                  q=(('pressure_level', 'latitude', 'longitude'), q),
                                                  u=(('pressure_level', 'latitude', 'longitude'), u),
                                                  v=(('pressure_level', 'latitude', 'longitude'), v),
                                                  mslp_z=(('pressure_level', 'latitude', 'longitude'), mslp_z),
                                                  sp_z=(('pressure_level', 'latitude', 'longitude'), sp_z)),
                                   coords=dict(pressure_level=pressure_levels, latitude=surface_data_latitudes, longitude=surface_data_longitudes)).astype('float32')

    full_gdas_dataset = full_gdas_dataset.expand_dims({'time': np.atleast_1d(unified_pressure_level_data['time'].values),
                                                       'forecast_hour': np.atleast_1d(forecast_hour)})

    for variable in list(full_gdas_dataset.keys()):
        full_gdas_dataset[variable].to_netcdf(path='%s/%d/%02d/%02d/gdas_%s_%d%02d%02d%02d_f%03d_full.nc' % (netcdf_outdir, year, month, day, variable, year, month, day, hour, forecast_hour), mode='w', engine='scipy')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ERA5', action='store_true', help="Generate ERA5 data files")
    parser.add_argument('--netcdf_ERA5_indir', type=str, help="input directory for ERA5 netcdf files")
    parser.add_argument('--fronts', action='store_true', help="Generate front object data files")
    parser.add_argument('--xml_indir', type=str, help="input directory for front XML files")
    parser.add_argument('--download_ncep_has_order', action='store_true', help='Download GDAS tarfiles')
    parser.add_argument('--download_gdas_grib_files', action='store_true', help='Download GDAS grib files for a specific date and time')
    parser.add_argument('--download_latest_gdas_data', action='store_true', help='Download GDAS grib files from NCEP.')
    parser.add_argument('--download_latest_gfs_data', action='store_true', help='Download GFS grib files from NCEP.')
    parser.add_argument('--ncep_has_order_number', type=str, help='HAS order number')
    parser.add_argument('--gdas_grib_to_netcdf', action='store_true', help="Generate GDAS netcdf files")
    parser.add_argument('--gdas_grib_outdir', type=str, help='Output directory for GDAS grib files downloaded from NCEP.')
    parser.add_argument('--gfs_grib_outdir', type=str, help='Output directory for GFS grib files downloaded from NCEP.')
    parser.add_argument('--gdas_grib_indir', type=str, help="input directory for the GDAS grib files")
    parser.add_argument('--ncep_has_outdir', type=str, help='Output directory for downloaded files from the HAS order.')
    parser.add_argument('--netcdf_outdir', type=str, help="output directory for netcdf files")
    parser.add_argument('--day_range', type=int, nargs=2, default=[0, 364], help="Start and end days of the year for grabbing grib files")
    parser.add_argument('--year', type=int, help="year for the data to be read in")
    parser.add_argument('--month', type=int, help="month for the data to be read in")
    parser.add_argument('--day', type=int, help="day for the data to be read in")
    parser.add_argument('--hour', type=int, help='hour for the grib data to be downloaded')
    args = parser.parse_args()
    provided_arguments = vars(args)

    if args.ERA5:
        required_arguments = ['year', 'month', 'day', 'netcdf_ERA5_indir', 'netcdf_outdir']
        check_arguments(provided_arguments, required_arguments)
        check_directory(args.netcdf_outdir, args.year, args.month, args.day)
        create_era5_datasets(args.year, args.month, args.day, args.netcdf_ERA5_indir, args.netcdf_outdir)
    if args.fronts:
        required_arguments = ['year', 'month', 'day', 'xml_indir', 'netcdf_outdir']
        check_arguments(provided_arguments, required_arguments)
        check_directory(args.netcdf_outdir, args.year, args.month, args.day)
        front_xmls_to_netcdf(args.year, args.month, args.day, args.xml_indir, args.netcdf_outdir)
    if args.gdas_grib_to_netcdf:
        required_arguments = ['year', 'month', 'day', 'hour', 'gdas_grib_indir', 'netcdf_outdir']
        check_arguments(provided_arguments, required_arguments)
        check_directory(args.netcdf_outdir, args.year, args.month, args.day)
        for forecast_hour in range(10):
            gdas_grib_to_netcdf(args.year, args.month, args.day, args.hour, args.gdas_grib_indir, args.netcdf_outdir, forecast_hour=forecast_hour)
    if args.download_ncep_has_order:
        required_arguments = ['ncep_has_order_number', 'ncep_has_outdir']
        check_arguments(provided_arguments, required_arguments)
        download_ncep_has_order(args.ncep_has_order_number, args.ncep_has_outdir)
    if args.download_gdas_grib_files:
        required_arguments = ['year', 'day_range', 'gdas_grib_outdir']
        check_arguments(provided_arguments, required_arguments)
        download_gdas_grib_files(args.gdas_grib_outdir, args.year, args.day_range)
    if args.download_latest_gdas_data:
        required_arguments = ['gdas_grib_outdir']
        check_arguments(provided_arguments, required_arguments)
        download_latest_gdas_data(args.gdas_grib_outdir)
    if args.download_latest_gfs_data:
        required_arguments = ['gfs_grib_outdir']
        check_arguments(provided_arguments, required_arguments)
        download_latest_gfs_data(args.gfs_grib_outdir)
