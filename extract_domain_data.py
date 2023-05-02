"""
Functions in this script create netcdf files containing ERA5, GDAS, or frontal object data.

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 4/29/2023 1:45 PM CT

TODO: modify code so GDAS and GFS grib files have the correct units (units are different prior to 2022)
"""

import argparse
import time
import urllib.error
import warnings
import requests
from bs4 import BeautifulSoup
import xarray as xr
import pandas as pd
from utils import data_utils, settings
import glob
import numpy as np
import xml.etree.ElementTree as ET
import variables
import wget
import os
import tensorflow as tf
import itertools


def front_xmls_to_netcdf(year: int, month: int, day: int, xml_dir: str, netcdf_outdir: str):
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

    ######################################### Check the parameters for errors ##########################################
    if not isinstance(year, int):
        raise TypeError(f"year must be an integer, received {type(year)}")
    if not isinstance(month, int):
        raise TypeError(f"year must be an integer, received {type(month)}")
    if not isinstance(day, int):
        raise TypeError(f"year must be an integer, received {type(day)}")
    if not isinstance(xml_dir, str):
        raise TypeError(f"xml_dir must be a string, received {type(xml_dir)}")
    if not isinstance(netcdf_outdir, str):
        raise TypeError(f"netcdf_outdir must be a string, received {type(netcdf_outdir)}")
    ####################################################################################################################

    files = sorted(glob.glob("%s/*_%04d%02d%02d*f000.xml" % (xml_dir, year, month, day)))

    dss = []  # Dataset with front data organized by front type.
    timesteps = []
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
        timesteps.append(pd.to_datetime(df['time'][0], format='%Y%m%d%H'))
    dns = xr.concat(dss, dim=timesteps)
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
            fronts_ds = xr.Dataset({"identifier": (('latitude', 'longitude'), fronttype)}, coords={"latitude": lats, "longitude": lons, "time": time}).astype('float32')

            if not os.path.isdir("%s/%d%02d" % (netcdf_outdir, year, month)):
                os.mkdir("%s/%d%02d" % (netcdf_outdir, year, month))
            fronts_ds.to_netcdf(path="%s/%d%02d/%s" % (netcdf_outdir, year, month, filename_netcdf), engine='netcdf4', mode='w')


def create_era5_datasets(year: int, month: int, day: int, netcdf_ERA5_indir: str, netcdf_outdir: str):
    """
    Extract ERA5 variable data for the specified year, month, day, and hour.

    Parameters
    ----------
    year: year
    month: month
    day: day
    netcdf_ERA5_indir: str
        Directory where the ERA5 netCDF files are contained.
    netcdf_outdir: str
        Directory where the created netcdf files will be stored.

    Returns
    -------
    xr_netcdf: xr.Dataset
        Xarray dataset containing variable data for the full domain.
    """

    ######################################### Check the parameters for errors ##########################################
    if not isinstance(year, int):
        raise TypeError(f"year must be an integer, received {type(year)}")
    if not isinstance(month, int):
        raise TypeError(f"year must be an integer, received {type(month)}")
    if not isinstance(day, int):
        raise TypeError(f"year must be an integer, received {type(day)}")
    if not isinstance(netcdf_ERA5_indir, str):
        raise TypeError(f"netcdf_ERA5_indir must be a string, received {type(netcdf_ERA5_indir)}")
    if not isinstance(netcdf_outdir, str):
        raise TypeError(f"netcdf_outdir must be a string, received {type(netcdf_outdir)}")
    ####################################################################################################################

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

        full_era5_dataset.to_netcdf(path='%s/%d/%02ed/%02d/era5_%d%02d%02d%02d_full.nc' % (netcdf_outdir, year, month, day, year, month, day, hour), mode='w', engine='netcdf4')


def download_ncep_has_order(ncep_has_order_number: str, ncep_has_outdir: str):
    """
    Download files as part of a HAS order from the NCEP.

    ncep_has_order_number: str
        HAS order number for the GDAS data.
    ncep_has_outdir: str
        Directory where the GDAS or GFS file will be saved to.
    """

    ######################################### Check the parameters for errors ##########################################
    if not isinstance(ncep_has_order_number, str):
        raise TypeError(f"ncep_has_order_number must be a string, received {type(ncep_has_order_number)}")
    if not isinstance(ncep_has_outdir, str):
        raise TypeError(f"ncep_has_outdir must be a string, received {type(ncep_has_outdir)}")
    ####################################################################################################################

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


def download_model_grib_files(grib_outdir: str, year: int, model: str = 'gdas', day_range: tuple = (0, 364)):
    """
    Download grib files containing data for a NWP model from the AWS server.

    Parameters
    ----------
    grib_outdir: str
        Output directory for the downloaded GRIB files.
    year: year
    model: str
        Model/source of the data. Current options are: 'gdas', 'gfs'.
    day_range: tuple
        Tuple of two integers marking the start and end indices of the days of the year to download. The end index IS included.
        Example: to download all days (1-365), enter (0, 364).
    """

    ######################################### Check the parameters for errors ##########################################
    if not isinstance(grib_outdir, str):
        raise TypeError(f"grib_outdir must be a string, received {type(grib_outdir)}")
    if not isinstance(year, int):
        raise TypeError(f"year must be an integer, received {type(year)}")
    if not isinstance(model, str):
        raise TypeError(f"model must be a string, received {type(model)}")
    if not isinstance(day_range, tuple):
        raise TypeError(f"day_range must be a tuple, received {type(day_range)}")
    elif len(day_range) != 2:
        raise TypeError(f"Tuple for day_range must be length 2, received length {len(day_range)}")
    ####################################################################################################################

    model_lowercase = model.lower()

    forecast_hours = np.arange(0, 10, 3)

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
        hours = np.arange(0, 24, 6)[::-1]  # Hours that the GDAS model runs forecasts for

        daily_directory = grib_outdir + '/%d%02d%02d' % (year, month, day)  # Directory for the GDAS grib files for the given day

        # GDAS files have different naming patterns based on which year the data is for
        if year < 2014:

            files = [f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.{year}%02d%02d/%02d/gdas1.t%02dz.pgrbf%02d.grib2' % (month, day, hour, hour, forecast_hour)
                     for hour in hours
                     for forecast_hour in forecast_hours]
            local_filenames = ['gdas1.t%02dz.pgrbf%02d.grib2' % (hour, forecast_hour)
                               for hour in hours
                               for forecast_hour in forecast_hours]

        elif year < 2021:

            files = [f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.{year}%02d%02d/%02d/gdas1.t%02dz.pgrb2.0p25.f%03d' % (month, day, hour, hour, forecast_hour)
                     for hour in hours
                     for forecast_hour in forecast_hours]
            local_filenames = ['gdas.t%02dz.pgrb2.0p25.f%03d' % (hour, forecast_hour)
                               for hour in hours
                               for forecast_hour in forecast_hours]

        else:

            files = [f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/{model_lowercase}.{year}%02d%02d/%02d/atmos/{model_lowercase}.t%02dz.pgrb2.0p25.f%03d' % (month, day, hour, hour, forecast_hour)
                     for hour in hours
                     for forecast_hour in forecast_hours]

            local_filenames = [f'{model}.t%02dz.pgrb2.0p25.f%03d' % (hour, forecast_hour)
                               for hour in hours
                               for forecast_hour in forecast_hours]

        for file, local_filename in zip(files, local_filenames):

            ### If the directory does not exist, check to see if the file link is valid. If the file link is NOT valid, then the directory will not be created since it will be empty. ###
            if not os.path.isdir(daily_directory):
                if requests.head(file).status_code == requests.codes.ok or requests.head(file.replace('/atmos', '')).status_code == requests.codes.ok:
                    os.mkdir(daily_directory)

            full_file_path = f'{daily_directory}/{local_filename}'

            if not os.path.isfile(full_file_path) and os.path.isdir(daily_directory):
                try:
                    wget.download(file, out=full_file_path)
                except urllib.error.HTTPError:
                    try:
                        wget.download(file.replace('/atmos', ''), out=full_file_path)
                    except urllib.error.HTTPError:
                        print(f"Error downloading {file}")
                        pass

            elif not os.path.isdir(daily_directory):
                warnings.warn(f"Unknown problem encountered when creating the following directory: {daily_directory}, "
                              f"Consider checking the AWS server to make sure that data exists for the given day ({year}-%02d-%02d): "
                              f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/index.html#{model_lowercase}.{year}%02d%02d" % (date_list[day_index][1], date_list[day_index][2], date_list[day_index][1], date_list[day_index][2]))
                break

            elif os.path.isfile(full_file_path):
                print(f"{full_file_path} already exists, skipping file....")


def download_latest_model_data(grib_outdir: str, model: str = 'gdas'):
    """
    Download the latest model grib files from NCEP.

    Parameters
    ----------
    grib_outdir: str
        Output directory for the downloaded GRIB files.
    model: str
        Model/source of the data. Current options are: 'gdas', 'gfs'.
    """

    ######################################### Check the parameters for errors ##########################################
    if not isinstance(grib_outdir, str):
        raise TypeError(f"grib_outdir must be a string, received {type(grib_outdir)}")
    if not isinstance(model, str):
        raise TypeError(f"model must be a string, received {type(model)}")
    ####################################################################################################################

    model_uppercase = model.upper()
    model_lowercase = model.lower()

    print(f"Updating {model_uppercase} files in directory: {grib_outdir}")

    print(f"Connecting to main {model_uppercase} source")
    url = f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/'
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    gdas_folders_by_day = [url + node.get('href') for node in soup.find_all('a') if model_lowercase in node.get('href')]
    timesteps_available = [folder[-9:-1] for folder in gdas_folders_by_day if f'prod/{model_lowercase}' in folder][::-1]

    files_added = 0

    for timestep in timesteps_available:

        year, month, day = timestep[:4], timestep[4:6], timestep[6:]
        daily_directory = '%s/%s' % (grib_outdir, timestep)

        url = f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/{model_lowercase}.{timestep}/'
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'html.parser')

        gdas_folders_by_hour = [url + node.get('href') for node in soup.find_all('a')]
        hours_available = [folder[-3:-1] for folder in gdas_folders_by_hour if any(hour in folder[-3:-1] for hour in ['00', '06', '12', '18'])][::-1]
        forecast_hours = np.arange(0, 54, 6)

        for hour in hours_available:

            print(f"\nSearching for available data: {year}-{month}-{day}-%02dz" % float(hour))

            url = f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/{model_lowercase}.{timestep}/%02d/atmos/' % float(hour)
            page = requests.get(url).text
            soup = BeautifulSoup(page, 'html.parser')
            files = [url + node.get('href') for node in soup.find_all('a') if any('.pgrb2.0p25.f%03d' % forecast_hour in node.get('href') for forecast_hour in forecast_hours)]

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

    print(f"{model_uppercase} directory update complete, %d files added" % files_added)


def grib_to_netcdf(year: int, month: int, day: int, hour: int, grib_indir: str, netcdf_outdir: str, overwrite_grib: bool = False,
    model: str = 'GDAS', delete_original_grib: bool = False, resolution: int | float = 0.25):
    """
    Create GDAS or GFS netcdf files from the grib files.

    Parameters
    ----------
    year: year
    month: month
    day: day
    hour: hour
    grib_indir: str
        Directory where the GDAS grib files are stored.
    netcdf_outdir: str
        Directory where the created GDAS or GFS netcdf files will be stored.
    overwrite_grib: bool
        Overwrite any existing grib files.
    model: 'GDAS' or 'GFS'
    delete_original_grib: bool
        Delete the original grib files after the netCDF files have been created.
    resolution: int or float
        The resolution of the data in degrees.
    """

    ######################################### Check the parameters for errors ##########################################
    if not isinstance(year, int):
        raise TypeError(f"year must be an integer, received {type(year)}")
    if not isinstance(month, int):
        raise TypeError(f"month must be an integer, received {type(month)}")
    if not isinstance(day, int):
        raise TypeError(f"day must be an integer, received {type(day)}")
    if not isinstance(grib_indir, str):
        raise TypeError(f"grib_indir must be a string, received {type(grib_indir)}")
    if not isinstance(netcdf_outdir, str):
        raise TypeError(f"netcdf_outdir must be a string, received {type(netcdf_outdir)}")
    if not isinstance(overwrite_grib, bool):
        raise TypeError(f"overwrite_grib must be a boolean, received {type(overwrite_grib)}")
    if not isinstance(model, str):
        raise TypeError(f"model must be a string, received {type(model)}")
    if not isinstance(delete_original_grib, bool):
        raise TypeError(f"delete_original_grib must be a boolean, received {type(delete_original_grib)}")
    if not isinstance(resolution, (int, float)):
        raise TypeError(f"resolution must be an integer or a float, received {type(resolution)}")
    ####################################################################################################################

    model = model.lower()

    keys_to_extract = ['gh', 'mslet', 'r', 'sp', 't', 'u', 'v']

    pressure_level_file_indices = [0, 2, 4, 5, 6]
    surface_data_file_indices = [2, 4, 5, 6]
    raw_pressure_data_file_index = 3
    mslp_data_file_index = 1

    # all lon/lat values in degrees
    start_lon, end_lon = 130, 370  # western boundary, eastern boundary
    start_lat, end_lat = 80, 0  # northern boundary, southern boundary
    unified_longitude_indices = np.append(np.arange(start_lon / resolution, (end_lon - 10) / resolution), np.arange(0, (end_lon - 360) / resolution + 1)).astype(int)
    unified_latitude_indices = np.arange((90 - start_lat) / resolution, (90 - end_lat) / resolution + 1).astype(int)
    lon_coords_360 = np.arange(start_lon, end_lon + resolution, resolution)

    domain_indices_isel = {'longitude': unified_longitude_indices,
                           'latitude': unified_latitude_indices}

    isobaricInhPa_isel = [0, 1, 2, 3, 4, 5, 6]  # Dictionary to unpack for selecting indices in the datasets

    chunk_sizes = {'latitude': 321, 'longitude': 961}

    dataset_dimensions = ('forecast_hour', 'pressure_level', 'latitude', 'longitude')

    if year > 2014:
        grib_filename_format = f'%s/%d%02d%02d/{model.lower()}*.t%02dz.pgrb2.0p25.f*' % (grib_indir, year, month, day, hour)
    else:
        grib_filename_format = f'%s/%d%02d%02d/{model.lower()}*1.t%02dz.pgrbf*.grib2' % (grib_indir, year, month, day, hour)

    individual_variable_filename_format = f'%s/%d%02d%02d/{model.lower()}.*.t%02dz.pgrb2.0p25' % (grib_indir, year, month, day, hour)

    ### Split grib files into one file per variable ###
    grib_files = sorted(glob.glob(grib_filename_format))
    grib_files = [file for file in grib_files if 'idx' not in file]

    for key in keys_to_extract:
        output_file = f'%s/%d%02d%02d/{model.lower()}.%s.t%02dz.pgrb2.0p25' % (grib_indir, year, month, day, key, hour)
        if (os.path.isfile(output_file) and overwrite_grib) or not os.path.isfile(output_file):
            os.system(f'grib_copy -w shortName={key} {" ".join(grib_files)} {output_file}')

    if delete_original_grib:
        [os.remove(file) for file in grib_files]

    time.sleep(5)  # Pause the code for 5 seconds to ensure that all contents of the individual files are preserved

    # grib files by variable
    grib_files = sorted(glob.glob(individual_variable_filename_format))

    pressure_level_files = [grib_files[index] for index in pressure_level_file_indices]
    surface_data_files = [grib_files[index] for index in surface_data_file_indices]

    raw_pressure_data_file = grib_files[raw_pressure_data_file_index]
    if 'mslp_data_file_index' in locals():
        mslp_data_file = grib_files[mslp_data_file_index]
        mslp_data = xr.open_dataset(mslp_data_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}}, chunks={'latitude': 721, 'longitude': 1440}).isel(**domain_indices_isel).drop_vars(['step'])

    pressure_levels = [1000, 975, 950, 925, 900, 850, 700]

    # Open the datasets
    pressure_level_data = xr.open_mfdataset(pressure_level_files, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}}, chunks=chunk_sizes, combine='nested').isel(isobaricInhPa=isobaricInhPa_isel, **domain_indices_isel).drop_vars(['step'])
    surface_data = xr.open_mfdataset(surface_data_files, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'sigma'}}, chunks=chunk_sizes).isel(**domain_indices_isel).drop_vars(['step'])
    raw_pressure_data = xr.open_dataset(raw_pressure_data_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'stepType': 'instant'}}, chunks=chunk_sizes).isel(**domain_indices_isel).drop_vars(['step'])

    # Calculate the forecast hours using the surface_data dataset
    try:
        run_time = surface_data['time'].values.astype('int64')
    except KeyError:
        run_time = surface_data['run_time'].values.astype('int64')

    valid_time = surface_data['valid_time'].values.astype('int64')
    forecast_hours = np.array((valid_time - int(run_time)) / 3.6e12, dtype='int32')

    try:
        num_forecast_hours = len(forecast_hours)
    except TypeError:
        num_forecast_hours = 1
        forecast_hours = [forecast_hours, ]

    # Reformat the longitude coordinates to the 360 degree system
    if model in ['gdas', 'gfs']:
        pressure_level_data['longitude'] = lon_coords_360
        surface_data['longitude'] = lon_coords_360
        raw_pressure_data['longitude'] = lon_coords_360
        mslp_data['longitude'] = lon_coords_360

        mslp = mslp_data['mslet'].values  # mean sea level pressure (eta model reduction)
        mslp_z = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
        mslp_z[:, 0, :, :] = mslp / 100  # convert to hectopascals

    P = np.empty(shape=(num_forecast_hours, len(pressure_levels), chunk_sizes['latitude'], chunk_sizes['longitude']))  # create 3D array of pressure levels to match the shape of variable arrays
    for pressure_level_index, pressure_level in enumerate(pressure_levels):
        P[:, pressure_level_index, :, :] = pressure_level * 100

    print("Generating pressure level variables")

    T_pl = pressure_level_data['t'].values
    RH_pl = pressure_level_data['r'].values / 100
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
    z = pressure_level_data['gh'].values / 10  # Convert to dam
    u_pl = pressure_level_data['u'].values
    v_pl = pressure_level_data['v'].values

    if 'mslp_data_file_index' in locals():
        mslp_z[:, 1:, :, :] = z

    # Create arrays of coordinates for the surface data
    surface_data_latitudes = pressure_level_data['latitude'].values

    print("Generating surface variables")

    sp = raw_pressure_data['sp'].values
    T_sigma = surface_data['t'].values
    RH_sigma = surface_data['r'].values / 100
    vap_pres_sigma = RH_sigma * variables.vapor_pressure(T_sigma)
    Td_sigma = variables.dewpoint_from_vapor_pressure(vap_pres_sigma)
    Tv_sigma = variables.virtual_temperature_from_dewpoint(T_sigma, Td_sigma, sp)
    Tw_sigma = variables.wet_bulb_temperature(T_sigma, Td_sigma)
    r_sigma = variables.mixing_ratio_from_dewpoint(Td_sigma, sp) * 1000  # Convert to g/kg
    q_sigma = variables.specific_humidity_from_dewpoint(Td_sigma, sp) * 1000  # Convert to g/kg
    theta_sigma = variables.potential_temperature(T_sigma, sp)
    theta_e_sigma = variables.equivalent_potential_temperature(T_sigma, Td_sigma, sp)
    theta_v_sigma = variables.virtual_potential_temperature(T_sigma, Td_sigma, sp)
    theta_w_sigma = variables.wet_bulb_potential_temperature(T_sigma, Td_sigma, sp)
    u_sigma = surface_data['u'].values
    v_sigma = surface_data['v'].values

    T = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    Td = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    Tv = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    Tw = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    theta = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    theta_e = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    theta_v = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    theta_w = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    RH = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    r = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    q = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    u = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    v = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))
    sp_z = np.empty(shape=(num_forecast_hours, len(pressure_levels) + 1, chunk_sizes['latitude'], chunk_sizes['longitude']))

    T[:, 0, :, :] = T_sigma
    T[:, 1:, :, :] = T_pl
    Td[:, 0, :, :] = Td_sigma
    Td[:, 1:, :, :] = Td_pl
    Tv[:, 0, :, :] = Tv_sigma
    Tv[:, 1:, :, :] = Tv_pl
    Tw[:, 0, :, :] = Tw_sigma
    Tw[:, 1:, :, :] = Tw_pl
    theta[:, 0, :, :] = theta_sigma
    theta[:, 1:, :, :] = theta_pl
    theta_e[:, 0, :, :] = theta_e_sigma
    theta_e[:, 1:, :, :] = theta_e_pl
    theta_v[:, 0, :, :] = theta_v_sigma
    theta_v[:, 1:, :, :] = theta_v_pl
    theta_w[:, 0, :, :] = theta_w_sigma
    theta_w[:, 1:, :, :] = theta_w_pl
    RH[:, 0, :, :] = RH_sigma
    RH[:, 1:, :, :] = RH_pl
    r[:, 0, :, :] = r_sigma
    r[:, 1:, :, :] = r_pl
    q[:, 0, :, :] = q_sigma
    q[:, 1:, :, :] = q_pl
    u[:, 0, :, :] = u_sigma
    u[:, 1:, :, :] = u_pl
    v[:, 0, :, :] = v_sigma
    v[:, 1:, :, :] = v_pl
    sp_z[:, 0, :, :] = sp / 100
    sp_z[:, 1:, :, :] = z

    pressure_levels = ['surface', '1000', '975', '950', '925', '900', '850', '700']

    print("Building final dataset")

    full_dataset_coordinates = dict(forecast_hour=forecast_hours, pressure_level=pressure_levels)

    full_dataset_variables = dict(T=(dataset_dimensions, T),
                                  Td=(dataset_dimensions, Td),
                                  Tv=(dataset_dimensions, Tv),
                                  Tw=(dataset_dimensions, Tw),
                                  theta=(dataset_dimensions, theta),
                                  theta_e=(dataset_dimensions, theta_e),
                                  theta_v=(dataset_dimensions, theta_v),
                                  theta_w=(dataset_dimensions, theta_w),
                                  RH=(dataset_dimensions, RH),
                                  r=(dataset_dimensions, r),
                                  q=(dataset_dimensions, q),
                                  u=(dataset_dimensions, u),
                                  v=(dataset_dimensions, v),
                                  sp_z=(dataset_dimensions, sp_z))

    if 'mslp_data_file_index' in locals():
        full_dataset_variables['mslp_z'] = (('forecast_hour', 'pressure_level', 'latitude', 'longitude'), mslp_z)

    full_dataset_coordinates['latitude'] = surface_data_latitudes
    full_dataset_coordinates['longitude'] = lon_coords_360

    full_grib_dataset = xr.Dataset(data_vars=full_dataset_variables,
                                   coords=full_dataset_coordinates).astype('float32')

    full_grib_dataset = full_grib_dataset.expand_dims({'time': np.atleast_1d(pressure_level_data['time'].values)})

    monthly_dir = '%s/%d%02d' % (netcdf_outdir, year, month)

    if not os.path.isdir(monthly_dir):
        os.mkdir(monthly_dir)

    for fcst_hr_index, forecast_hour in enumerate(forecast_hours):
        full_grib_dataset.isel(forecast_hour=np.atleast_1d(fcst_hr_index)).to_netcdf(path=f'%s/{model.lower()}_%d%02d%02d%02d_f%03d_full.nc' % (monthly_dir, year, month, day, hour, forecast_hour), mode='w', engine='netcdf4')


def netcdf_to_tf(year: int, month: int, era5_netcdf_indir: str, fronts_netcdf_indir: str, tf_outdir: str, era5: bool, fronts: bool,
    front_types: str | list[str] = None, variables: list[str] = None, pressure_levels: list[str] = None, num_dims: tuple[int] | list[int] = (3, 3),
    domain: str = 'conus', images: tuple[int] | list[int] = (9, 1), image_size: tuple[int] | list[int] = (128, 128), keep_all: bool = False,
    shuffle_timesteps: bool = False, shuffle_images: bool = False, front_dilation: int = 1, noise_chance: float = 0.0, rotate_chance: float = 0.0,
    flip_chance_lon: float = 0.0, flip_chance_lat: float = 0.0, status_printout: bool = True):
    """
    Convert matching ERA5 and front object files to tensorflow datasets

    Parameters
    ----------
    year: year
    month: month
    era5_netcdf_indir: str
    fronts_netcdf_indir: str
    tf_outdir: str
    era5: bool
        Generate ERA5 tensorflow datasets
    fronts: bool
        Generate fronts tensorflow datasets
    front_types: str or list (default = None)
        Front types to select in the images. If None, use all types.
    variables: list (default = None)
        List of ERA5 variables to use. Leaving this argument as None will select all available variables.
    pressure_levels: list (default = None)
        List of pressure levels to use. Leaving this argument as None will select all available pressure levels.
    num_dims: tuple or list of 2 ints (default = (3, 3))
        Number of dimensions in the final output maps for ERA5 and front datasets, respectively. (2 or 3)
    domain: str (default = 'conus')
        Domain from which to pull the images.
    images: tuple or list of 2 ints (default = (9, 1))
        Number of images to take along the longitude and latitude dimensions for each timestep. The product of the 2 integers
            will be the total number of images generated per timestep.
    image_size: tuple or list of 2 ints (default = (128, 128))
        Size of the longitude and latitude dimensions in the images.
    keep_all: bool (default = False)
        Keep all of the images, regardless of the presence of frontal boundaries within the images.
    shuffle_timesteps: bool (default = False)
        Shuffle the timesteps when generating the dataset. This is particularly useful when generating very large datasets
            that cannot be shuffled on the fly during training.
    shuffle_images: bool (default = False)
        Shuffle the order of the images in each timestep. This does NOT shuffle the entire dataset for the provided
            month, but rather only the images in each respective timestep. This is particularly useful when generating
            very large datasets that cannot be shuffled on the fly during training.
    front_dilation: int (default = 1)
        Number of pixels to expand the fronts by in all directions.
    noise_chance: float (default = 0.0)
        The probability that noise will be added to a pixel in the image. This can be thought of as the fraction of pixels
            in each image that will contain noise. Can be any float 0 <= x < 1.
    rotate_chance: float (default = 0.0)
        The probability that the current image will be rotated (in any direction, up to 270 degrees). Can be any float
            0 <= rotate_chance <= 1.
    flip_chance_lon: float (default = 0.0)
        The probability that the current image will have its longitude dimension reversed. Can be any float 0 <= flip_chance_lon <= 1.
    flip_chance_lat: float (default = 0.0)
        The probability that the current image will have its latitude dimension reversed. Can be any float 0 <= flip_chance_lat <= 1.
    status_printout: bool  (default = False)
        Print out the progress of the dataset generation.
    """

    ######################################### Check the parameters for errors ##########################################
    if not isinstance(year, int):
        raise TypeError(f"year must be an integer, received {type(year)}")
    if not isinstance(month, int):
        raise TypeError(f"month must be an integer, received {type(month)}")
    if not isinstance(era5_netcdf_indir, str):
        raise TypeError(f"era5_netcdf_indir must be a string, received {type(era5_netcdf_indir)}")
    if not isinstance(fronts_netcdf_indir, str):
        raise TypeError(f"fronts_netcdf_indir must be a string, received {type(fronts_netcdf_indir)}")
    if not isinstance(tf_outdir, str):
        raise TypeError(f"tf_outdir must be a string, received {type(tf_outdir)}")
    if not isinstance(era5, bool):
        raise TypeError(f"era5 must be a boolean, received {type(era5)}")
    if not isinstance(fronts, bool):
        raise TypeError(f"fronts must be a boolean, received {type(fronts)}")
    if not isinstance(variables, list):
        raise TypeError(f"variables must be a list, received {type(variables)}")
    if not isinstance(pressure_levels, list):
        raise TypeError(f"pressure_levels must be a list, received {type(pressure_levels)}")
    if not isinstance(num_dims, (tuple, list)):
        raise TypeError(f"num_dims must be a tuple or list, received {type(num_dims)}")
    if not isinstance(domain, str):
        raise TypeError(f"domain must be a string, received {type(domain)}")
    if not isinstance(images, (tuple, list)):
        raise TypeError(f"images must be a tuple or list, received {type(images)}")
    if not isinstance(image_size, (tuple, list)):
        raise TypeError(f"image_size must be a tuple or list, received {type(image_size)}")
    if not isinstance(shuffle_timesteps, bool):
        raise TypeError(f"shuffle_timesteps must be a boolean, received {type(shuffle_timesteps)}")
    if not isinstance(shuffle_images, bool):
        raise TypeError(f"shuffle_images must be a boolean, received {type(shuffle_images)}")
    if not isinstance(front_dilation, int):
        raise TypeError(f"front_dilation must be an integer, received {type(front_dilation)}")
    if not isinstance(noise_chance, float):
        raise TypeError(f"noise_chance must be a float, received {type(noise_chance)}")
    if not isinstance(rotate_chance, float):
        raise TypeError(f"rotate_chance must be a float, received {type(rotate_chance)}")
    if not isinstance(flip_chance_lon, float):
        raise TypeError(f"flip_chance_lon must be a float, received {type(flip_chance_lon)}")
    if not isinstance(flip_chance_lat, float):
        raise TypeError(f"flip_chance_lat must be a float, received {type(flip_chance_lat)}")
    if not isinstance(status_printout, bool):
        raise TypeError(f"status_printout must be a boolean, received {type(status_printout)}")

    if front_types is not None:
        if not isinstance(front_types, (str, list)):
            raise TypeError(f"front_types must be a string or list, received {type(front_types)}")
    ####################################################################################################################

    print("Generating tensorflow datasets for %d-%02d" % (year, month))

    all_variables = ['T', 'Td', 'sp_z', 'u', 'v', 'theta_w', 'r', 'RH', 'Tv', 'Tw', 'theta_e', 'q']
    all_pressure_levels = ['surface', '1000', '950', '900', '850']

    era5_monthly_directory = '%s/%d%02d' % (era5_netcdf_indir, year, month)
    fronts_monthly_directory = '%s/%d%02d' % (fronts_netcdf_indir, year, month)

    era5_netcdf_files = sorted(glob.glob('%s/era5_%d%02d*_full.nc' % (era5_monthly_directory, year, month)))
    fronts_netcdf_files = sorted(glob.glob('%s/FrontObjects_%d%02d*_full.nc' % (fronts_monthly_directory, year, month)))

    if shuffle_timesteps:
        zipped_list = list(zip(era5_netcdf_files, fronts_netcdf_files))
        np.random.shuffle(zipped_list)
        era5_netcdf_files, fronts_netcdf_files = zip(*zipped_list)

    files_match_flag = all(era5_file[-18:] == fronts_file[-18:] for era5_file, fronts_file in zip(era5_netcdf_files, fronts_netcdf_files))

    isel_kwargs = {'longitude': slice(settings.DEFAULT_DOMAIN_INDICES[domain][0], settings.DEFAULT_DOMAIN_INDICES[domain][1]),
                   'latitude': slice(settings.DEFAULT_DOMAIN_INDICES[domain][2], settings.DEFAULT_DOMAIN_INDICES[domain][3])}

    if not files_match_flag:
        raise OSError("ERA5/fronts files do not match")

    # Normalization parameters are not available for theta_v and theta, so we will only select the following variables
    if variables is None:
        variables_to_use = all_variables
    else:
        variables_to_use = sorted(variables)

    if pressure_levels is None:
        pressure_levels = all_pressure_levels
    else:
        pressure_levels = [lvl for lvl in all_pressure_levels if lvl in pressure_levels]

    images_kept = 0
    images_discarded = 0

    for era5_file, fronts_file, in zip(era5_netcdf_files, fronts_netcdf_files):

        if era5:
            era5_dataset = xr.open_dataset(era5_file, engine='netcdf4')[variables_to_use].isel(**isel_kwargs).sel(pressure_level=pressure_levels).transpose('time', 'longitude', 'latitude', 'pressure_level').astype('float16')
            era5_dataset = data_utils.normalize_variables(era5_dataset).isel(time=0).to_array().transpose('longitude', 'latitude', 'pressure_level', 'variable').astype('float16')

        """ 
        Regardless of whether we want to save the front object datasets, we must load the datasets in. This is so that the
        ERA5 images being saved only correspond to front object images that have the desired front types.
        """
        front_dataset = xr.open_dataset(fronts_file, engine='netcdf4').isel(**isel_kwargs).expand_dims('time', axis=0).astype('float16')
        if front_types is not None:
            front_dataset = data_utils.reformat_fronts(front_dataset, front_types)
            num_front_types = front_dataset.attrs['num_types'] + 1
        else:
            num_front_types = 17

        if front_dilation > 0:
            front_dataset = data_utils.expand_fronts(front_dataset, iterations=front_dilation)  # expand the front labels

        front_dataset = front_dataset.isel(time=0).to_array().transpose('longitude', 'latitude', 'variable')
        front_bins = np.bincount(front_dataset.values.astype('int64').flatten(), minlength=num_front_types)

        if all([front_count > 0 for front_count in front_bins]) > 0 or keep_all:  # If not all front types are present for the current timestep, move to the next timestep

            try:
                start_indices_lon = np.arange(0, settings.DEFAULT_DOMAIN_INDICES[domain][1] - settings.DEFAULT_DOMAIN_INDICES[domain][0] - image_size[0] + 1,
                                              int((settings.DEFAULT_DOMAIN_INDICES[domain][1] - settings.DEFAULT_DOMAIN_INDICES[domain][0] - image_size[0]) / (images[0] - 1)))
            except ZeroDivisionError:
                start_indices_lon = np.zeros([images[0]], dtype=int)

            try:
                start_indices_lat = np.arange(0, settings.DEFAULT_DOMAIN_INDICES[domain][3] - settings.DEFAULT_DOMAIN_INDICES[domain][2] - image_size[1] + 1,
                                              int((settings.DEFAULT_DOMAIN_INDICES[domain][3] - settings.DEFAULT_DOMAIN_INDICES[domain][2] - image_size[1]) / (images[1] - 1)))
            except ZeroDivisionError:
                start_indices_lat = np.zeros([images[1]], dtype=int)

            image_order = list(itertools.product(start_indices_lon, start_indices_lat))  # Every possible combination of longitude and latitude starting points
            if shuffle_images:
                np.random.shuffle(image_order)

            for image_start_indices in image_order:

                start_index_lon = image_start_indices[0]
                end_index_lon = start_index_lon + image_size[0]
                start_index_lat = image_start_indices[1]
                end_index_lat = start_index_lat + image_size[1]

                # boolean flags for rotating and flipping images
                rotate_image = np.random.randint(1, 101) <= rotate_chance * 100
                flip_lon = np.random.randint(1, 101) <= flip_chance_lon * 100
                flip_lat = np.random.randint(1, 101) <= flip_chance_lat * 100

                if rotate_image:
                    rotation_direction = np.random.randint(0, 2)  # 0 = clockwise, 1 = counter-clockwise
                    num_rotations = np.random.randint(1, 4)  # n * 90 degrees

                if era5:

                    era5_tensor = tf.convert_to_tensor(era5_dataset[start_index_lon:end_index_lon, start_index_lat:end_index_lat, :, :], dtype=tf.float16)
                    if flip_lon:
                        era5_tensor = tf.reverse(era5_tensor, axis=[0])  # Reverse values along the longitude dimension
                    if flip_lat:
                        era5_tensor = tf.reverse(era5_tensor, axis=[1])  # Reverse values along the latitude dimension
                    if rotate_image:
                        for rotation in range(num_rotations):
                            era5_tensor = tf.reverse(tf.transpose(era5_tensor, perm=[1, 0, 2, 3]), axis=[rotation_direction])  # Rotate image 90 degrees

                    if noise_chance > 0:
                        ### Add noise to image ###
                        random_values = tf.random.uniform(shape=era5_tensor.shape)
                        era5_tensor = tf.where(random_values < noise_chance / 2, 0.0, era5_tensor)  # add 0s to image
                        era5_tensor = tf.where(random_values > 1 - (noise_chance / 2), 1.0, era5_tensor)  # add 1s to image

                    if num_dims[0] == 2:
                        era5_tensor_shape_3d = era5_tensor.shape
                        # Combine pressure level and variables dimensions, making the images 2D (excluding the final dimension)
                        era5_tensor = tf.reshape(era5_tensor, [era5_tensor_shape_3d[0], era5_tensor_shape_3d[1], era5_tensor_shape_3d[2] * era5_tensor_shape_3d[3]])

                    era5_tensor_for_timestep = tf.data.Dataset.from_tensors(era5_tensor)
                    if 'era5_tensors_for_month' not in locals():
                        era5_tensors_for_month = era5_tensor_for_timestep
                    else:
                        era5_tensors_for_month = era5_tensors_for_month.concatenate(era5_tensor_for_timestep)

                if fronts:

                    front_tensor = tf.convert_to_tensor(front_dataset[start_index_lon:end_index_lon, start_index_lat:end_index_lat, :], dtype=tf.int32)

                    if flip_lon:
                        front_tensor = tf.reverse(front_tensor, axis=[0])  # Reverse values along the longitude dimension
                    if flip_lat:
                        front_tensor = tf.reverse(front_tensor, axis=[1])  # Reverse values along the latitude dimension
                    if rotate_image:
                        for rotation in range(num_rotations):
                            front_tensor = tf.reverse(tf.transpose(front_tensor, perm=[1, 0, 2]), axis=[rotation_direction])  # Rotate image 90 degrees

                    if num_dims[1] == 3:
                        # Make the front object images 3D, with the size of the 3rd dimension equal to the number of pressure levels
                        front_tensor = tf.tile(front_tensor, (1, 1, len(pressure_levels)))
                    else:
                        front_tensor = front_tensor[:, :, 0]

                    front_tensor = tf.cast(tf.one_hot(front_tensor, num_front_types), tf.float16)  # One-hot encode the labels
                    front_tensor_for_timestep = tf.data.Dataset.from_tensors(front_tensor)
                    if 'front_tensors_for_month' not in locals():
                        front_tensors_for_month = front_tensor_for_timestep
                    else:
                        front_tensors_for_month = front_tensors_for_month.concatenate(front_tensor_for_timestep)

            images_kept += (images[0] * images[1])
        else:
            images_discarded += (images[0] * images[1])

        if status_printout:
            print("Images kept/discarded: %d/%d    (Discard rate: %.1f%s)    " % (images_kept, images_discarded, (images_discarded / (images_kept + images_discarded) * 100), '%'), end='\r')

    if status_printout:
        print("Images kept/discarded: %d/%d    (Discard rate: %.1f%s)    " % (images_kept, images_discarded, (images_discarded / (images_kept + images_discarded) * 100), '%'))

    if not os.path.isdir(tf_outdir):
        os.mkdir(tf_outdir)

    if era5:
        tf_dataset_folder = f'%s/era5_%d%02d_tf' % (tf_outdir, year, month)
        try:
            tf.data.Dataset.save(era5_tensors_for_month, path=tf_dataset_folder)
        except UnboundLocalError:
            pass
    if fronts:
        tf_dataset_folder = f'%s/fronts_{"_".join(front_type for front_type in front_types)}_%d%02d_tf' % (tf_outdir, year, month)
        try:
            tf.data.Dataset.save(front_tensors_for_month, path=tf_dataset_folder)
        except UnboundLocalError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--era5', action='store_true', help="Generate ERA5 data files")
    parser.add_argument('--netcdf_ERA5_indir', type=str, help="input directory for ERA5 netcdf files")
    parser.add_argument('--fronts', action='store_true', help="Generate front object data files")
    parser.add_argument('--xml_indir', type=str, help="input directory for front XML files")
    parser.add_argument('--download_ncep_has_order', action='store_true', help='Download GDAS tarfiles')
    parser.add_argument('--download_model_grib_files', action='store_true', help='Download model grib files for a specific date and time')
    parser.add_argument('--download_latest_model_data', action='store_true', help='Download latest model grib files from NCEP.')
    parser.add_argument('--ncep_has_order_number', type=str, help='HAS order number')
    parser.add_argument('--grib_to_netcdf', action='store_true', help="Generate GDAS netcdf files")
    parser.add_argument('--grib_outdir', type=str, help='Output directory for GDAS grib files downloaded from NCEP.')
    parser.add_argument('--grib_indir', type=str, help="input directory for the GDAS grib files")
    parser.add_argument('--model', type=str, help="model of the data for which grib files will be split: 'GDAS' or 'GFS'")
    parser.add_argument('--ncep_has_outdir', type=str, help='Output directory for downloaded files from the HAS order.')
    parser.add_argument('--netcdf_outdir', type=str, help="output directory for netcdf files")
    parser.add_argument('--day_range', type=int, nargs=2, default=[0, 364], help="Start and end days of the year for grabbing grib files")
    parser.add_argument('--year', type=int, help="year for the data to be read in")
    parser.add_argument('--month', type=int, help="month for the data to be read in")
    parser.add_argument('--day', type=int, help="day for the data to be read in")
    parser.add_argument('--hour', type=int, help='hour for the grib data to be downloaded')

    ### Convert netcdf to tensorflow datasets ###
    parser.add_argument('--netcdf_to_tf', action='store_true', help="Convert ERA5 and/or front netcdf files to tensorflow datasets")
    parser.add_argument('--era5_netcdf_indir', type=str, help="input directory for ERA5 netcdf files")
    parser.add_argument('--fronts_netcdf_indir', type=str, help="input directory for ERA5 netcdf files")
    parser.add_argument('--tf_outdir', type=str, help="Output directory for tensors")
    parser.add_argument('--front_types', type=str, nargs='+', help='Front types')
    parser.add_argument('--variables', type=str, nargs='+', help='ERA5 variables to select')
    parser.add_argument('--pressure_levels', type=str, nargs='+', help='ERA5 pressure levels to select')
    parser.add_argument('--num_dims', type=int, nargs=2, default=[3, 3], help='Number of dimensions in the ERA5 and front object images, repsectively.')
    parser.add_argument('--domain', type=str, default='conus', help='Domain from which to pull the images.')
    parser.add_argument('--images', type=int, nargs=2, default=[9, 1],
        help='Number of ERA5/front images along the longitude and latitude dimensions to generate for each timestep. The product of the 2 integers '
             'will be the total number of images generated per timestep.')
    parser.add_argument('--image_size', type=int, nargs=2, default=[128, 128], help='Size of the longitude and latitude dimensions of the images.')
    parser.add_argument('--keep_all', action='store_true', help='Keep all of the images, regardless of the presence of frontal boundaries within the images. ')
    parser.add_argument('--shuffle_timesteps', action='store_true',
        help='Shuffle the timesteps when generating the dataset. This is particularly useful when generating very large '
             'datasets that cannot be shuffled on the fly during training.')
    parser.add_argument('--shuffle_images', action='store_true',
        help='Shuffle the order of the images in each timestep. This does NOT shuffle the entire dataset for the provided '
             'month, but rather only the images in each respective timestep. This is particularly useful when generating '
             'very large datasets that cannot be shuffled on the fly during training.')
    parser.add_argument('--front_dilation', type=int, default=1, help='Number of pixels to expand the fronts by in all directions.')
    parser.add_argument('--noise_chance', type=float, default=0.0,
        help='The probability that noise will be added to a pixel in the image. This can be thought of as the fraction of pixels '
             'in each image that will contain noise. Can be any float 0 <= x < 1.')
    parser.add_argument('--rotate_chance', type=float, default=0.0,
        help='The probability that the current image will be rotated (in any direction, up to 270 degrees). Can be any float 0 <= x <= 1.')
    parser.add_argument('--flip_chance_lon', type=float, default=0.0,
        help='The probability that the current image will have its longitude dimension reversed. Can be any float 0 <= x <= 1.')
    parser.add_argument('--flip_chance_lat', type=float, default=0.0,
        help='The probability that the current image will have its latitude dimension reversed. Can be any float 0 <= x <= 1.')
    parser.add_argument('--status_printout', action='store_true', help='Print out the progress of the dataset generation.')

    args = parser.parse_args()

    if args.era5 and not args.netcdf_to_tf:
        create_era5_datasets(args.year, args.month, args.day, args.netcdf_ERA5_indir, args.netcdf_outdir)

    if args.fronts and not args.netcdf_to_tf:
        front_xmls_to_netcdf(args.year, args.month, args.day, args.xml_indir, args.netcdf_outdir)

    if args.grib_to_netcdf:
        for hour in range(0, 24, 6):
            try:
                grib_to_netcdf(args.year, args.month, args.day, hour, args.grib_indir, args.netcdf_outdir, model=args.model)
            except IndexError:
                pass

    if args.netcdf_to_tf:
        netcdf_to_tf(args.year, args.month, args.era5_netcdf_indir, args.fronts_netcdf_indir, args.tf_outdir, args.era5,
                     args.fronts, args.front_types, args.variables, args.pressure_levels, args.num_dims, args.domain, args.images,
                     args.image_size, args.keep_all, args.shuffle_timesteps, args.shuffle_images, args.front_dilation, args.noise_chance,
                     args.rotate_chance, args.flip_chance_lon, args.flip_chance_lat, args.status_printout)

    if args.download_ncep_has_order:
        download_ncep_has_order(args.ncep_has_order_number, args.ncep_has_outdir)

    if args.download_model_grib_files:
        download_model_grib_files(args.grib_outdir, args.year, args.model, args.day_range)

    if args.download_latest_model_data:
        download_latest_model_data(args.grib_outdir, args.model)
