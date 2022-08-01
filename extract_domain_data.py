"""
Functions in this script create pickle files containing ERA5, GDAS, or frontal object data.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 8/1/2022 2:18 AM CDT
"""

import argparse
import urllib.error
import requests
from bs4 import BeautifulSoup
import xarray as xr
import pickle
import pandas as pd
from utils import data_utils
from errors import check_arguments
import glob
import numpy as np
import xml.etree.ElementTree as ET
import variables
import wget
import os


def front_xmls_to_pickle(year, month, day, xml_dir, pickle_outdir):
    """
    Reads the xml files to pull frontal objects.

    Parameters
    ----------
    year: year
    month: month
    day: day
    xml_dir: str
        Directory where the front xml files are stored.
    pickle_outdir: str
        Directory where the created pickle files will be stored.
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

        fronts_ds = xr.Dataset({"identifier": (('latitude', 'longitude'), fronttype)}, coords={"latitude": lats, "longitude": lons, "time": time})
        fronts_ds['longitude'] = fronts_ds['longitude'].astype('float16')
        fronts_ds['latitude'] = fronts_ds['latitude'].astype('float16')
        fronts_ds['identifier'] = fronts_ds['identifier'].astype('int8')

        filename = "FrontObjects_%04d%02d%02d%02d_full.pkl" % (year, month, day, hour)

        fronts_ds.load()
        outfile = open("%s/%d/%02d/%02d/%s" % (pickle_outdir, year, month, day, filename), 'wb')
        pickle.dump(fronts_ds, outfile)
        outfile.close()


def create_era5_datasets(year, month, day, netcdf_ERA5_indir, pickle_outdir):
    """
    Extract ERA5 variable data for the specified year, month, day, and hour.

    Parameters
    ----------
    year: year
    month: month
    day: day
    netcdf_ERA5_indir: Directory where the ERA5 netCDF files are contained.
    pickle_outdir: Directory where the created pickle files will be stored.

    Returns
    -------
    xr_pickle: Xarray dataset containing variable data for the full domain.
    """
    in_file_name_2mT = 'ERA5Global_%d_3hrly_2mT.nc' % year
    in_file_name_2mTd = 'ERA5Global_%d_3hrly_2mTd.nc' % year
    in_file_name_sp = 'ERA5Global_%d_3hrly_sp.nc' % year
    in_file_name_U10m = 'ERA5Global_%d_3hrly_U10m.nc' % year
    in_file_name_V10m = 'ERA5Global_%d_3hrly_V10m.nc' % year

    timestring = "%d-%02d-%02d" % (year, month, day)

    ds_2mT = xr.open_mfdataset("%s/%s" % (netcdf_ERA5_indir, in_file_name_2mT), chunks={'latitude': 721, 'longitude': 1440, 'time': 4}).sel(time=('%s' % timestring))
    ds_2mTd = xr.open_mfdataset("%s/%s" % (netcdf_ERA5_indir, in_file_name_2mTd), chunks={'latitude': 721, 'longitude': 1440, 'time': 4}).sel(time=('%s' % timestring))
    ds_sp = xr.open_mfdataset("%s/%s" % (netcdf_ERA5_indir, in_file_name_sp), chunks={'latitude': 721, 'longitude': 1440, 'time': 4}).sel(time=('%s' % timestring))
    ds_U10m = xr.open_mfdataset("%s/%s" % (netcdf_ERA5_indir, in_file_name_U10m), chunks={'latitude': 721, 'longitude': 1440, 'time': 4}).sel(time=('%s' % timestring))
    ds_V10m = xr.open_mfdataset("%s/%s" % (netcdf_ERA5_indir, in_file_name_V10m), chunks={'latitude': 721, 'longitude': 1440, 'time': 4}).sel(time=('%s' % timestring))

    ds_2mT = ds_2mT.sel(time='%s' % timestring)
    ds_2mTd = ds_2mTd.sel(time='%s' % timestring)
    ds_sp = ds_sp.sel(time='%s' % timestring)
    ds_U10m = ds_U10m.sel(time='%s' % timestring)
    ds_V10m = ds_V10m.sel(time='%s' % timestring)

    ds = xr.merge((ds_2mT, ds_2mTd, ds_sp, ds_U10m, ds_V10m))
    ds_theta_w = variables.wet_bulb_temperature(ds.t2m, ds.d2m)  # Wet-bulb potential temperature dataset
    ds_mixing_ratio = variables.mixing_ratio_from_dewpoint(ds.d2m, ds.sp)  # Mixing ratio dataset
    ds_RH = variables.relative_humidity(ds.t2m, ds.d2m)  # Relative humidity dataset
    ds_Tv = variables.virtual_temperature_from_dewpoint(ds.t2m, ds.d2m, ds.sp)  # Virtual temperature dataset
    ds_Tw = variables.wet_bulb_temperature(ds.t2m, ds.d2m)  # Wet-bulb temperature dataset

    lons = np.append(np.arange(130, 360, 0.25), np.arange(0, 10.25, 0.25))
    lats = np.arange(0, 80.25, 0.25)
    lons360 = np.arange(130, 370.25, 0.25)

    full_lons = np.arange(0, 360, 0.25)
    full_lats = np.arange(-90, 90.25, 0.25)

    ds_2mT = ds_2mT.sel(longitude=lons, latitude=lats)
    ds_2mTd = ds_2mTd.sel(longitude=lons, latitude=lats)
    ds_sp = ds_sp.sel(longitude=lons, latitude=lats)
    ds_U10m = ds_U10m.sel(longitude=lons, latitude=lats)
    ds_V10m = ds_V10m.sel(longitude=lons, latitude=lats)
    ds_theta_w = ds_theta_w.sel(longitude=lons, latitude=lats).to_dataset(name='theta_w')
    ds_mixing_ratio = ds_mixing_ratio.sel(longitude=lons, latitude=lats).to_dataset(name='mix_ratio')
    ds_RH = ds_RH.sel(longitude=lons, latitude=lats).to_dataset(name='rel_humid')
    ds_Tv = ds_Tv.sel(longitude=lons, latitude=lats).to_dataset(name='virt_temp')
    ds_Tw = ds_Tw.sel(longitude=lons, latitude=lats).to_dataset(name='wet_bulb')
    ds_theta_e = variables.equivalent_potential_temperature(ds_2mT.t2m, ds_2mTd.d2m, ds_sp.sp).to_dataset(name='theta_e')
    ds_theta_v = variables.virtual_potential_temperature(ds_2mT.t2m, ds_2mTd.d2m, ds_sp.sp).to_dataset(name='theta_v')
    ds_q = variables.specific_humidity_from_dewpoint(ds_2mTd.d2m, ds_sp.sp).to_dataset(name='q')

    timestring = "%d-%02d-%02d" % (year, month, day)

    PL_data = xr.open_mfdataset(
        paths=('/ourdisk/hpc/ai2es/fronts/era5/Pressure_Level/ERA5Global_PL_%s_3hrly_Q.nc' % year,
               '/ourdisk/hpc/ai2es/fronts/era5/Pressure_Level/ERA5Global_PL_%s_3hrly_T.nc' % year,
               '/ourdisk/hpc/ai2es/fronts/era5/Pressure_Level/ERA5Global_PL_%s_3hrly_U.nc' % year,
               '/ourdisk/hpc/ai2es/fronts/era5/Pressure_Level/ERA5Global_PL_%s_3hrly_V.nc' % year,
               '/ourdisk/hpc/ai2es/fronts/era5/Pressure_Level/ERA5Global_PL_%s_3hrly_Z.nc' % year),
        chunks={'latitude': 721, 'longitude': 1440, 'time': 4}).sel(time=('%s' % timestring))

    PL_850 = PL_data.sel(level=850)
    PL_900 = PL_data.sel(level=900)
    PL_950 = PL_data.sel(level=950)
    PL_1000 = PL_data.sel(level=1000)

    time = PL_850['time'].values
    q_850 = PL_850['q'].values
    q_900 = PL_900['q'].values
    q_950 = PL_950['q'].values
    q_1000 = PL_1000['q'].values
    t_850 = PL_850['t'].values
    t_900 = PL_900['t'].values
    t_950 = PL_950['t'].values
    t_1000 = PL_1000['t'].values
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

    d_850 = variables.dewpoint_from_specific_humidity(85000, t_850, q_850)
    d_900 = variables.dewpoint_from_specific_humidity(90000, t_900, q_900)
    d_950 = variables.dewpoint_from_specific_humidity(95000, t_950, q_950)
    d_1000 = variables.dewpoint_from_specific_humidity(100000, t_1000, q_1000)
    mix_ratio_850 = variables.mixing_ratio_from_dewpoint(d_850, 85000)
    mix_ratio_900 = variables.mixing_ratio_from_dewpoint(d_900, 90000)
    mix_ratio_950 = variables.mixing_ratio_from_dewpoint(d_950, 95000)
    mix_ratio_1000 = variables.mixing_ratio_from_dewpoint(d_1000, 100000)
    rel_humid_850 = variables.relative_humidity(t_850, d_850)
    rel_humid_900 = variables.relative_humidity(t_900, d_900)
    rel_humid_950 = variables.relative_humidity(t_950, d_950)
    rel_humid_1000 = variables.relative_humidity(t_1000, d_1000)
    theta_e_850 = variables.equivalent_potential_temperature(t_850, d_850, 85000)
    theta_e_900 = variables.equivalent_potential_temperature(t_900, d_900, 90000)
    theta_e_950 = variables.equivalent_potential_temperature(t_950, d_950, 95000)
    theta_e_1000 = variables.equivalent_potential_temperature(t_1000, d_1000, 100000)
    theta_v_850 = variables.virtual_temperature_from_dewpoint(t_850, d_850, 85000)
    theta_v_900 = variables.virtual_temperature_from_dewpoint(t_900, d_900, 90000)
    theta_v_950 = variables.virtual_temperature_from_dewpoint(t_950, d_950, 95000)
    theta_v_1000 = variables.virtual_temperature_from_dewpoint(t_1000, d_1000, 100000)
    theta_w_850 = variables.wet_bulb_potential_temperature(t_850, d_850, 85000)
    theta_w_900 = variables.wet_bulb_potential_temperature(t_900, d_900, 90000)
    theta_w_950 = variables.wet_bulb_potential_temperature(t_950, d_950, 95000)
    theta_w_1000 = variables.wet_bulb_potential_temperature(t_1000, d_1000, 100000)
    virt_temp_850 = variables.virtual_temperature_from_dewpoint(t_850, d_850, 85000)
    virt_temp_900 = variables.virtual_temperature_from_dewpoint(t_900, d_900, 90000)
    virt_temp_950 = variables.virtual_temperature_from_dewpoint(t_950, d_950, 95000)
    virt_temp_1000 = variables.virtual_temperature_from_dewpoint(t_1000, d_1000, 100000)
    wet_bulb_850 = variables.wet_bulb_temperature(t_850, d_850)
    wet_bulb_900 = variables.wet_bulb_temperature(t_900, d_900)
    wet_bulb_950 = variables.wet_bulb_temperature(t_950, d_950)
    wet_bulb_1000 = variables.wet_bulb_temperature(t_1000, d_1000)

    new_850 = xr.Dataset(data_vars=dict(q_850=(['time', 'latitude', 'longitude'], q_850*1000),
                                        t_850=(['time', 'latitude', 'longitude'], t_850),
                                        u_850=(['time', 'latitude', 'longitude'], u_850),
                                        v_850=(['time', 'latitude', 'longitude'], v_850),
                                        z_850=(['time', 'latitude', 'longitude'], z_850/98.0665),
                                        d_850=(['time', 'latitude', 'longitude'], d_850),
                                        mix_ratio_850=(['time', 'latitude', 'longitude'], mix_ratio_850*1000),
                                        rel_humid_850=(['time', 'latitude', 'longitude'], rel_humid_850),
                                        theta_e_850=(['time', 'latitude', 'longitude'], theta_e_850),
                                        theta_v_850=(['time', 'latitude', 'longitude'], theta_v_850),
                                        theta_w_850=(['time', 'latitude', 'longitude'], theta_w_850),
                                        virt_temp_850=(['time', 'latitude', 'longitude'], virt_temp_850),
                                        wet_bulb_850=(['time', 'latitude', 'longitude'], wet_bulb_850)),
                         coords=dict(latitude=full_lats, longitude=full_lons, time=time))
    new_900 = xr.Dataset(data_vars=dict(q_900=(['time', 'latitude', 'longitude'], q_900*1000),
                                        t_900=(['time', 'latitude', 'longitude'], t_900),
                                        u_900=(['time', 'latitude', 'longitude'], u_900),
                                        v_900=(['time', 'latitude', 'longitude'], v_900),
                                        z_900=(['time', 'latitude', 'longitude'], z_900/98.0665),
                                        d_900=(['time', 'latitude', 'longitude'], d_900),
                                        mix_ratio_900=(['time', 'latitude', 'longitude'], mix_ratio_900*1000),
                                        rel_humid_900=(['time', 'latitude', 'longitude'], rel_humid_900),
                                        theta_e_900=(['time', 'latitude', 'longitude'], theta_e_900),
                                        theta_v_900=(['time', 'latitude', 'longitude'], theta_v_900),
                                        theta_w_900=(['time', 'latitude', 'longitude'], theta_w_900),
                                        virt_temp_900=(['time', 'latitude', 'longitude'], virt_temp_900),
                                        wet_bulb_900=(['time', 'latitude', 'longitude'], wet_bulb_900)),
                         coords=dict(latitude=full_lats, longitude=full_lons, time=time))
    new_950 = xr.Dataset(data_vars=dict(q_950=(['time', 'latitude', 'longitude'], q_950*1000),
                                        t_950=(['time', 'latitude', 'longitude'], t_950),
                                        u_950=(['time', 'latitude', 'longitude'], u_950),
                                        v_950=(['time', 'latitude', 'longitude'], v_950),
                                        z_950=(['time', 'latitude', 'longitude'], z_950/98.0665),
                                        d_950=(['time', 'latitude', 'longitude'], d_950),
                                        mix_ratio_950=(['time', 'latitude', 'longitude'], mix_ratio_950*1000),
                                        rel_humid_950=(['time', 'latitude', 'longitude'], rel_humid_950),
                                        theta_e_950=(['time', 'latitude', 'longitude'], theta_e_950),
                                        theta_v_950=(['time', 'latitude', 'longitude'], theta_v_950),
                                        theta_w_950=(['time', 'latitude', 'longitude'], theta_w_950),
                                        virt_temp_950=(['time', 'latitude', 'longitude'], virt_temp_950),
                                        wet_bulb_950=(['time', 'latitude', 'longitude'], wet_bulb_950)),
                         coords=dict(latitude=full_lats, longitude=full_lons, time=time))
    new_1000 = xr.Dataset(data_vars=dict(q_1000=(['time', 'latitude', 'longitude'], q_1000*1000),
                                         t_1000=(['time', 'latitude', 'longitude'], t_1000),
                                         u_1000=(['time', 'latitude', 'longitude'], u_1000),
                                         v_1000=(['time', 'latitude', 'longitude'], v_1000),
                                         z_1000=(['time', 'latitude', 'longitude'], z_1000/98.0665),
                                         d_1000=(['time', 'latitude', 'longitude'], d_1000),
                                         mix_ratio_1000=(['time', 'latitude', 'longitude'], mix_ratio_1000*1000),
                                         rel_humid_1000=(['time', 'latitude', 'longitude'], rel_humid_1000),
                                         theta_e_1000=(['time', 'latitude', 'longitude'], theta_e_1000),
                                         theta_v_1000=(['time', 'latitude', 'longitude'], theta_v_1000),
                                         theta_w_1000=(['time', 'latitude', 'longitude'], theta_w_1000),
                                         virt_temp_1000=(['time', 'latitude', 'longitude'], virt_temp_1000),
                                         wet_bulb_1000=(['time', 'latitude', 'longitude'], wet_bulb_1000)),
                          coords=dict(latitude=full_lats, longitude=full_lons, time=time))

    ds_pickle = [ds_2mT, ds_2mTd, ds_sp, ds_U10m, ds_V10m, ds_theta_w, ds_mixing_ratio, ds_RH, ds_Tv, ds_Tw, ds_theta_e,
                 ds_theta_v, ds_q, new_850, new_900, new_950, new_1000]

    xr_pickle = xr.merge(ds_pickle, combine_attrs='override').sel(longitude=lons, latitude=lats)
    xr_pickle['longitude'] = lons360
    xr_pickle['q'].values = xr_pickle['q'].values*1000
    xr_pickle['t2m'].values = xr_pickle['t2m'].values
    xr_pickle['d2m'].values = xr_pickle['d2m'].values
    xr_pickle['sp'].values = xr_pickle['sp'].values/100  # convert Pa to hPa
    xr_pickle['u10'].values = xr_pickle['u10'].values
    xr_pickle['v10'].values = xr_pickle['v10'].values
    xr_pickle['theta_e'].values = xr_pickle['theta_e'].values
    xr_pickle['theta_v'].values = xr_pickle['theta_v'].values
    xr_pickle['theta_w'].values = xr_pickle['theta_w'].values
    xr_pickle['mix_ratio'].values = xr_pickle['mix_ratio'].values*1000
    xr_pickle['rel_humid'].values = xr_pickle['rel_humid'].values
    xr_pickle['virt_temp'].values = xr_pickle['virt_temp'].values
    xr_pickle['wet_bulb'].values = xr_pickle['wet_bulb'].values

    xr_pickle.t2m.attrs['units'] = 'K'
    xr_pickle.t2m.attrs['long_name'] = '2m AGL Temperature'
    xr_pickle.d2m.attrs['units'] = 'K'
    xr_pickle.d2m.attrs['long_name'] = '2m AGL Dewpoint temperature'
    xr_pickle.sp.attrs['units'] = 'hPa'
    xr_pickle.sp.attrs['long_name'] = 'Surface pressure'
    xr_pickle.u10.attrs['units'] = 'm/s'
    xr_pickle.u10.attrs['long_name'] = '10m AGL Zonal wind velocity'
    xr_pickle.v10.attrs['units'] = 'm/s'
    xr_pickle.v10.attrs['long_name'] = '10m AGL Meridional wind velocity'
    xr_pickle.theta_v.attrs['units'] = 'K'
    xr_pickle.theta_v.attrs['long_name'] = '2m AGL Virtual potential temperature'
    xr_pickle.theta_w.attrs['units'] = 'K'
    xr_pickle.theta_w.attrs['long_name'] = '2m AGL Wet-bulb potential temperature'
    xr_pickle.mix_ratio.attrs['units'] = 'g(H2O)/kg(air)'
    xr_pickle.mix_ratio.attrs['long_name'] = '2m AGL Mixing ratio'
    xr_pickle.rel_humid.attrs['long_name'] = '2m AGL Relative humidity'
    xr_pickle.virt_temp.attrs['units'] = 'K'
    xr_pickle.virt_temp.attrs['long_name'] = '2m AGL Virtual temperature'
    xr_pickle.wet_bulb.attrs['units'] = 'K'
    xr_pickle.wet_bulb.attrs['long_name'] = '2m AGL Wet-bulb temperature'
    xr_pickle.theta_e.attrs['units'] = 'K'
    xr_pickle.theta_e.attrs['long_name'] = '2m AGL Equivalent potential temperature'
    xr_pickle['q'].attrs['units'] = 'g(H20)/kg(air)'
    xr_pickle['q'].attrs['long_name'] = '2m AGL Specific humidity'

    xr_pickle.q_850.attrs['units'] = 'g(H20)/kg(air)'
    xr_pickle.q_850.attrs['long_name'] = '850mb Specific humidity'
    xr_pickle.t_850.attrs['units'] = 'K'
    xr_pickle.t_850.attrs['long_name'] = '850mb Temperature'
    xr_pickle.u_850.attrs['units'] = 'm/s'
    xr_pickle.u_850.attrs['long_name'] = '850mb Zonal wind velocity'
    xr_pickle.v_850.attrs['units'] = 'm/s'
    xr_pickle.v_850.attrs['long_name'] = '850mb Meridional wind velocity'
    xr_pickle.z_850.attrs['units'] = 'dam'
    xr_pickle.z_850.attrs['long_name'] = '850mb Geopotential height'
    xr_pickle.d_850.attrs['units'] = 'K'
    xr_pickle.d_850.attrs['long_name'] = '850mb Dewpoint temperature'
    xr_pickle.mix_ratio_850.attrs['units'] = 'g(H20)/kg(air)'
    xr_pickle.mix_ratio_850.attrs['long_name'] = '850mb Mixing ratio'
    xr_pickle.rel_humid_850.attrs['long_name'] = '850mb Relative humidity'
    xr_pickle.theta_e_850.attrs['units'] = 'K'
    xr_pickle.theta_e_850.attrs['long_name'] = '850mb Equivalent potential temperature'
    xr_pickle.theta_v_850.attrs['units'] = 'K'
    xr_pickle.theta_v_850.attrs['long_name'] = '850mb Virtual potential temperature'
    xr_pickle.theta_w_850.attrs['units'] = 'K'
    xr_pickle.theta_w_850.attrs['long_name'] = '850mb Wet-bulb potential temperature'
    xr_pickle.virt_temp_850.attrs['units'] = 'K'
    xr_pickle.virt_temp_850.attrs['long_name'] = '850mb Virtual temperature'
    xr_pickle.wet_bulb_850.attrs['units'] = 'K'
    xr_pickle.wet_bulb_850.attrs['long_name'] = '850mb Wet-bulb temperature'

    xr_pickle.q_900.attrs['units'] = 'g(H20)/kg(air)'
    xr_pickle.q_900.attrs['long_name'] = '900mb Specific humidity'
    xr_pickle.t_900.attrs['units'] = 'K'
    xr_pickle.t_900.attrs['long_name'] = '900mb Temperature'
    xr_pickle.u_900.attrs['units'] = 'm/s'
    xr_pickle.u_900.attrs['long_name'] = '900mb Zonal wind velocity'
    xr_pickle.v_900.attrs['units'] = 'm/s'
    xr_pickle.v_900.attrs['long_name'] = '900mb Meridional wind velocity'
    xr_pickle.z_900.attrs['units'] = 'dam'
    xr_pickle.z_900.attrs['long_name'] = '900mb Geopotential height'
    xr_pickle.d_900.attrs['units'] = 'K'
    xr_pickle.d_900.attrs['long_name'] = '900mb Dewpoint temperature'
    xr_pickle.mix_ratio_900.attrs['units'] = 'g(H20)/kg(air)'
    xr_pickle.mix_ratio_900.attrs['long_name'] = '900mb Mixing ratio'
    xr_pickle.rel_humid_900.attrs['long_name'] = '900mb Relative humidity'
    xr_pickle.theta_e_900.attrs['units'] = 'K'
    xr_pickle.theta_e_900.attrs['long_name'] = '900mb Equivalent potential temperature'
    xr_pickle.theta_v_900.attrs['units'] = 'K'
    xr_pickle.theta_v_900.attrs['long_name'] = '900mb Virtual potential temperature'
    xr_pickle.theta_w_900.attrs['units'] = 'K'
    xr_pickle.theta_w_900.attrs['long_name'] = '900mb Wet-bulb potential temperature'
    xr_pickle.virt_temp_900.attrs['units'] = 'K'
    xr_pickle.virt_temp_900.attrs['long_name'] = '900mb Virtual temperature'
    xr_pickle.wet_bulb_900.attrs['units'] = 'K'
    xr_pickle.wet_bulb_900.attrs['long_name'] = '900mb Wet-bulb temperature'

    xr_pickle.q_950.attrs['units'] = 'g(H20)/kg(air)'
    xr_pickle.q_950.attrs['long_name'] = '950mb Specific humidity'
    xr_pickle.t_950.attrs['units'] = 'K'
    xr_pickle.t_950.attrs['long_name'] = '950mb Temperature'
    xr_pickle.u_950.attrs['units'] = 'm/s'
    xr_pickle.u_950.attrs['long_name'] = '950mb Zonal wind velocity'
    xr_pickle.v_950.attrs['units'] = 'm/s'
    xr_pickle.v_950.attrs['long_name'] = '950mb Meridional wind velocity'
    xr_pickle.z_950.attrs['units'] = 'dam'
    xr_pickle.z_950.attrs['long_name'] = '950mb Geopotential height'
    xr_pickle.d_950.attrs['units'] = 'K'
    xr_pickle.d_950.attrs['long_name'] = '950mb Dewpoint temperature'
    xr_pickle.mix_ratio_950.attrs['units'] = 'g(H20)/kg(air)'
    xr_pickle.mix_ratio_950.attrs['long_name'] = '950mb Mixing ratio'
    xr_pickle.rel_humid_950.attrs['long_name'] = '950mb Relative humidity'
    xr_pickle.theta_e_950.attrs['units'] = 'K'
    xr_pickle.theta_e_950.attrs['long_name'] = '950mb Equivalent potential temperature'
    xr_pickle.theta_v_950.attrs['units'] = 'K'
    xr_pickle.theta_v_950.attrs['long_name'] = '950mb Virtual potential temperature'
    xr_pickle.theta_w_950.attrs['units'] = 'K'
    xr_pickle.theta_w_950.attrs['long_name'] = '950mb Wet-bulb potential temperature'
    xr_pickle.virt_temp_950.attrs['units'] = 'K'
    xr_pickle.virt_temp_950.attrs['long_name'] = '950mb Virtual temperature'
    xr_pickle.wet_bulb_950.attrs['units'] = 'K'
    xr_pickle.wet_bulb_950.attrs['long_name'] = '950mb Wet-bulb temperature'

    xr_pickle.q_1000.attrs['units'] = 'g(H20)/kg(air)'
    xr_pickle.q_1000.attrs['long_name'] = '1000mb Specific humidity'
    xr_pickle.t_1000.attrs['units'] = 'K'
    xr_pickle.t_1000.attrs['long_name'] = '1000mb Temperature'
    xr_pickle.u_1000.attrs['units'] = 'm/s'
    xr_pickle.u_1000.attrs['long_name'] = '1000mb Zonal wind velocity'
    xr_pickle.v_1000.attrs['units'] = 'm/s'
    xr_pickle.v_1000.attrs['long_name'] = '1000mb Meridional wind velocity'
    xr_pickle.z_1000.attrs['units'] = 'dam'
    xr_pickle.z_1000.attrs['long_name'] = '1000mb Geopotential height'
    xr_pickle.d_1000.attrs['units'] = 'K'
    xr_pickle.d_1000.attrs['long_name'] = '1000mb Dewpoint temperature'
    xr_pickle.mix_ratio_1000.attrs['units'] = 'g(H20)/kg(air)'
    xr_pickle.mix_ratio_1000.attrs['long_name'] = '1000mb Mixing ratio'
    xr_pickle.rel_humid_1000.attrs['long_name'] = '1000mb Relative humidity'
    xr_pickle.theta_e_1000.attrs['units'] = 'K'
    xr_pickle.theta_e_1000.attrs['long_name'] = '1000mb Equivalent potential temperature'
    xr_pickle.theta_v_1000.attrs['units'] = 'K'
    xr_pickle.theta_v_1000.attrs['long_name'] = '1000mb Virtual potential temperature'
    xr_pickle.theta_w_1000.attrs['units'] = 'K'
    xr_pickle.theta_w_1000.attrs['long_name'] = '1000mb Wet-bulb potential temperature'
    xr_pickle.virt_temp_1000.attrs['units'] = 'K'
    xr_pickle.virt_temp_1000.attrs['long_name'] = '1000mb Virtual temperature'
    xr_pickle.wet_bulb_1000.attrs['units'] = 'K'
    xr_pickle.wet_bulb_1000.attrs['long_name'] = '1000mb Wet-bulb temperature'

    lons = np.append(np.arange(130, 360, 0.25), np.arange(0, 10.25, 0.25))
    lats = np.arange(0, 80.25, 0.25)
    lons360 = np.arange(130, 370.25, 0.25)

    xr_pickle = xr_pickle.sel(Longitude=lons, Latitude=lats)
    xr_pickle = xr_pickle.rename(Latitude='latitude', Longitude='longitude', Type='type', Date='time')
    xr_pickle['longitude'] = lons360

    for hour in range(0, 24, 3):

        print(f"saving ERA5 data for {year}-%02d-%02d-%02dz" % (month, day, hour))

        xr_pickle_data = xr_pickle.sel(time='%d-%02d-%02dT%02d:00:00' % (year, month, day, hour))

        xr_pickle_data['longitude'] = xr_pickle_data['longitude'].astype('float16')
        xr_pickle_data['latitude'] = xr_pickle_data['latitude'].astype('float16')
        xr_pickle_data = xr_pickle_data.astype('float16')

        num_variables = len(list(xr_pickle_data.keys()))

        filename = "Data_%dvar_%04d%02d%02d%02d_full.pkl" % (num_variables, year, month, day, hour)

        xr_pickle_data.load()  # prevents bug that can occur from opening the dataset
        outfile = open("%s/%d/%02d/%02d/%s" % (pickle_outdir, year, month, day, filename), 'wb')
        pickle.dump(xr_pickle_data, outfile)
        outfile.close()


def download_gdas_order(gdas_order_number, gdas_outdir):
    """
    Download a set of GDAS files as part of a HAS order from the NCEP.

    gdas_order_number: str
        - HAS order number for the GDAS data.
    gdas_outdir: str
        - Directory where the GDAS or GFS file will be saved to.
    """

    url = f'https://www.ncei.noaa.gov/pub/has/model/HAS{gdas_order_number}/'
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    files = [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('.tar')]

    for file in files:
        local_filename = file.replace(url, '')
        if not os.path.isfile(f'{gdas_outdir}/{local_filename}'):
            try:
                wget.download(file, out=f"{gdas_outdir}/{local_filename}")
            except urllib.error.HTTPError:
                print(f"Error downloading {url}{file}")
                pass


def gdas_grib_to_pickle(year, month, day, gdas_indir, pickle_outdir, forecast_hour=6):
    """
    Create GDAS pickle files from the grib files.

    Parameters
    ----------
    year: year
    month: month
    day: day
    gdas_indir: Directory where the GDAS grib files are stored.
    pickle_outdir: Directory where the created GDAS or GFS pickle files will be stored.
    forecast_hour: Hour for the forecast.
    """

    # all lon/lat values in degrees
    start_lon, end_lon = 130, 370  # western boundary, eastern boundary
    start_lat, end_lat = 80, 0  # northern boundary, southern boundary

    # coordinate indices for the unified surface analysis domain in the GDAS datasets
    unified_longitude_indices = np.append(np.arange(start_lon * 4, (end_lon - 10) * 4), np.arange(0, (end_lon - 360) * 4 + 1))
    unified_latitude_indices = np.arange((90 - start_lat) * 4, (90 - end_lat) * 4 + 1)

    for hour in range(0, 24, 6):

        gdas_file = '%s/%d%02d%02d/gdas.t%02dz.pgrb2.0p25.f%03d' % (gdas_indir, year, month, day, hour, forecast_hour)

        ###############################################################################################################
        ############################################# Pressure level data #############################################
        ###############################################################################################################

        try:
            pressure_level_data = xr.open_dataset(gdas_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}}, chunks={'latitude': 721, 'longitude': 1440})
        except FileNotFoundError:
            pass
        else:
            print("generating GDAS pressure level data for %d-%02d-%02d-%02dz" % (year, month, day, hour))

            unified_pressure_level_data = pressure_level_data.isel(longitude=unified_longitude_indices, latitude=unified_latitude_indices, isobaricInhPa=np.arange(0, 21))  # unified surface analysis domain
            unified_pressure_level_data['longitude'] = np.arange(start_lon, end_lon + 0.25, 0.25)  # reformat longitude coordinates to a 360 degree system

            # Change data type of the coordinates and pressure levels to reduce the size of the data
            unified_pressure_level_data['longitude'] = unified_pressure_level_data['longitude'].values.astype('float16')
            unified_pressure_level_data['latitude'] = unified_pressure_level_data['latitude'].values.astype('float16')
            unified_pressure_level_data['isobaricInhPa'] = unified_pressure_level_data['isobaricInhPa'].values.astype('float32')

            pressure_level_variables = list(unified_pressure_level_data.keys())
            pressure_levels = unified_pressure_level_data['isobaricInhPa'].values
            P = np.empty(shape=(21, (start_lat - end_lat) * 4 + 1, (end_lon - start_lon) * 4 + 1))  # create 3D array of pressure levels to match the shape of variable arrays
            for pressure_level in range(21):
                P[pressure_level, :, :] = pressure_levels[pressure_level] * 100

            T = unified_pressure_level_data['t'].values
            RH = unified_pressure_level_data['r'].values / 100
            vap_pres = RH * variables.vapor_pressure(T)
            Td = variables.dewpoint_from_vapor_pressure(vap_pres)
            Tv = variables.virtual_temperature_from_dewpoint(T, Td, P)
            Tw = variables.wet_bulb_temperature(T, Td)
            r = variables.mixing_ratio_from_dewpoint(Td, P) * 1000  # Convert to g/kg
            q = variables.specific_humidity_from_dewpoint(Td, P) * 1000  # Convert to g/kg
            AH = variables.absolute_humidity(T, Td) * 1000  # Convert to g/kg
            theta = variables.potential_temperature(T, P)
            theta_e = variables.equivalent_potential_temperature(T, Td, P)
            theta_v = variables.virtual_potential_temperature(T, Td, P)
            theta_w = variables.wet_bulb_potential_temperature(T, Td, P)

            unified_pressure_level_data = unified_pressure_level_data.rename(t='T', gh='z', r='RH', isobaricInhPa='pressure_level')
            for variable_to_drop in ['wz', 'absv', 'o3mr', 'q']:
                if variable_to_drop in pressure_level_variables:
                    unified_pressure_level_data = unified_pressure_level_data.drop_vars(variable_to_drop)

            unified_pressure_level_data['RH'] = (('pressure_level', 'latitude', 'longitude'), RH)

            unified_pressure_level_data['Td'] = (('pressure_level', 'latitude', 'longitude'), Td)
            unified_pressure_level_data['Td'].attrs['units'] = 'K'
            unified_pressure_level_data['Td'].attrs['long_name'] = 'Dewpoint temperature'

            unified_pressure_level_data['Tv'] = (('pressure_level', 'latitude', 'longitude'), Tv)
            unified_pressure_level_data['Tv'].attrs['units'] = 'K'
            unified_pressure_level_data['Tv'].attrs['long_name'] = 'Virtual temperature'

            unified_pressure_level_data['Tw'] = (('pressure_level', 'latitude', 'longitude'), Tw)
            unified_pressure_level_data['Tw'].attrs['units'] = 'K'
            unified_pressure_level_data['Tw'].attrs['long_name'] = 'Wet-bulb temperature'

            unified_pressure_level_data['r'] = (('pressure_level', 'latitude', 'longitude'), r)
            unified_pressure_level_data['r'].attrs['units'] = 'g/kg'
            unified_pressure_level_data['r'].attrs['long_name'] = 'Mixing ratio'

            unified_pressure_level_data['q'] = (('pressure_level', 'latitude', 'longitude'), q)
            unified_pressure_level_data['q'].attrs['units'] = 'g/kg'
            unified_pressure_level_data['q'].attrs['long_name'] = 'Specific humidity'

            unified_pressure_level_data['AH'] = (('pressure_level', 'latitude', 'longitude'), AH)
            unified_pressure_level_data['AH'].attrs['units'] = 'percent'
            unified_pressure_level_data['AH'].attrs['long_name'] = 'Mixing ratio'

            unified_pressure_level_data['theta'] = (('pressure_level', 'latitude', 'longitude'), theta)
            unified_pressure_level_data['theta'].attrs['units'] = 'K'
            unified_pressure_level_data['theta'].attrs['long_name'] = 'Potential temperature'

            unified_pressure_level_data['theta_e'] = (('pressure_level', 'latitude', 'longitude'), theta_e)
            unified_pressure_level_data['theta_e'].attrs['units'] = 'K'
            unified_pressure_level_data['theta_e'].attrs['long_name'] = 'Equivalent potential temperature'

            unified_pressure_level_data['theta_v'] = (('pressure_level', 'latitude', 'longitude'), theta_v)
            unified_pressure_level_data['theta_v'].attrs['units'] = 'K'
            unified_pressure_level_data['theta_v'].attrs['long_name'] = 'Virtual potential temperature'

            unified_pressure_level_data['theta_w'] = (('pressure_level', 'latitude', 'longitude'), theta_w)
            unified_pressure_level_data['theta_w'].attrs['units'] = 'K'
            unified_pressure_level_data['theta_w'].attrs['long_name'] = 'Wet-bulb potential temperature'

            variables_in_ds = list(unified_pressure_level_data.keys())
            num_vars = len(variables_in_ds)

            # Convert variables to float16 to conserve storage
            for var in range(num_vars):
                unified_pressure_level_data[variables_in_ds[var]].values = unified_pressure_level_data[variables_in_ds[var]].values.astype('float16')

            # Save pressure level data
            for pressure_level in range(21):
                current_pressure_level = pressure_levels[pressure_level]
                current_pressure_level_ds = unified_pressure_level_data.isel(pressure_level=pressure_level)
                variable_file_path = '%s/%d/%02d/%02d/gdas_%d_%d%02d%02d%02d_f%03d_full.pkl' % (pickle_outdir, year, month, day, current_pressure_level, year, month, day, hour, forecast_hour)
                with open(variable_file_path, 'wb') as gdas_variable_file:
                    pickle.dump(current_pressure_level_ds, gdas_variable_file)

            ################################################################################################################
            ################################# Surface data (MSLP and sigma coordinate data) ################################
            ################################################################################################################

            print("generating GDAS surface data for %d-%02d-%02d-%02dz" % (year, month, day, hour))

            mslp_data = xr.open_dataset(gdas_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}}, chunks={'latitude': 721, 'longitude': 1440})
            unified_mslp_data = mslp_data.isel(longitude=unified_longitude_indices, latitude=unified_latitude_indices)  # select unified surface analysis domain

            surface_data = xr.open_dataset(gdas_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'sigma'}}, chunks={'latitude': 721, 'longitude': 1440})
            unified_surface_data = surface_data.isel(longitude=unified_longitude_indices, latitude=unified_latitude_indices)  # select unified surface analysis domain

            raw_pressure_data = xr.open_dataset(gdas_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'stepType': 'instant'}}, chunks={'latitude': 721, 'longitude': 1440})
            sp = raw_pressure_data.isel(longitude=unified_longitude_indices, latitude=unified_latitude_indices)['sp'].values / 100  # select unified surface analysis domain

            # Create arrays of coordinates for the surface data
            surface_data_longitudes = np.arange(start_lon, end_lon + 0.25, 0.25)
            surface_data_latitudes = unified_surface_data['latitude'].values

            mslp = unified_mslp_data['mslet'].values  # mean sea level pressure (eta model reduction)
            T_sigma = unified_surface_data['t'].values
            RH_sigma = unified_surface_data['r'].values / 100
            vap_pres_sigma = RH_sigma * variables.vapor_pressure(T_sigma)
            Td_sigma = variables.dewpoint_from_vapor_pressure(vap_pres_sigma)
            Tv_sigma = variables.virtual_temperature_from_dewpoint(T_sigma, Td_sigma, mslp)
            Tw_sigma = variables.wet_bulb_temperature(T_sigma, Td_sigma)
            r_sigma = variables.mixing_ratio_from_dewpoint(Td_sigma, mslp) * 1000  # Convert to g/kg
            q_sigma = variables.specific_humidity_from_dewpoint(Td_sigma, mslp) * 1000  # Convert to g/kg
            AH_sigma = variables.absolute_humidity(T_sigma, Td_sigma) * 1000  # Convert to g/kg
            theta_sigma = variables.potential_temperature(T_sigma, mslp)
            theta_e_sigma = variables.equivalent_potential_temperature(T_sigma, Td_sigma, mslp)
            theta_v_sigma = variables.virtual_potential_temperature(T_sigma, Td_sigma, mslp)
            theta_w_sigma = variables.wet_bulb_potential_temperature(T_sigma, Td_sigma, mslp)
            u_sigma = unified_surface_data['u'].values
            v_sigma = unified_surface_data['v'].values

            new_unified_surface_data = xr.Dataset(data_vars=dict(mslp=(('latitude', 'longitude'), mslp / 100),
                                                                 T=(('latitude', 'longitude'), T_sigma),
                                                                 RH=(('latitude', 'longitude'), RH_sigma),
                                                                 Td=(('latitude', 'longitude'), Td_sigma),
                                                                 Tv=(('latitude', 'longitude'), Tv_sigma),
                                                                 Tw=(('latitude', 'longitude'), Tw_sigma),
                                                                 u=(('latitude', 'longitude'), u_sigma),
                                                                 v=(('latitude', 'longitude'), v_sigma),
                                                                 r=(('latitude', 'longitude'), r_sigma),
                                                                 q=(('latitude', 'longitude'), q_sigma),
                                                                 sp=(('latitude', 'longitude'), sp),
                                                                 AH=(('latitude', 'longitude'), AH_sigma),
                                                                 theta=(('latitude', 'longitude'), theta_sigma),
                                                                 theta_e=(('latitude', 'longitude'), theta_e_sigma),
                                                                 theta_v=(('latitude', 'longitude'), theta_v_sigma),
                                                                 theta_w=(('latitude', 'longitude'), theta_w_sigma)),
                                                  coords=dict(latitude=surface_data_latitudes, longitude=surface_data_longitudes)).astype('float16')

            new_unified_surface_data['mslp'].attrs['units'] = 'hPa'
            new_unified_surface_data['mslp'].attrs['long_name'] = 'Mean sea level pressure'

            new_unified_surface_data['T'].attrs['units'] = 'K'
            new_unified_surface_data['T'].attrs['long_name'] = 'Temperature'

            new_unified_surface_data['RH'].attrs['long_name'] = 'Relative humidity'

            new_unified_surface_data['Td'].attrs['units'] = 'K'
            new_unified_surface_data['Td'].attrs['long_name'] = 'Dewpoint temperature'

            new_unified_surface_data['Tv'].attrs['units'] = 'K'
            new_unified_surface_data['Tv'].attrs['long_name'] = 'Virtual temperature'

            new_unified_surface_data['Tw'].attrs['units'] = 'K'
            new_unified_surface_data['Tw'].attrs['long_name'] = 'Wet-bulb temperature'

            new_unified_surface_data['u'].attrs['units'] = 'm/s'
            new_unified_surface_data['u'].attrs['long_name'] = 'Zonal wind velocity'

            new_unified_surface_data['v'].attrs['units'] = 'm/s'
            new_unified_surface_data['v'].attrs['long_name'] = 'Meridional wind velocity'

            new_unified_surface_data['r'].attrs['units'] = 'g/kg'
            new_unified_surface_data['r'].attrs['long_name'] = 'Mixing ratio'

            new_unified_surface_data['q'].attrs['units'] = 'g/kg'
            new_unified_surface_data['q'].attrs['long_name'] = 'Specific humidity'

            new_unified_surface_data['sp'].attrs['units'] = 'hPa'
            new_unified_surface_data['sp'].attrs['long_name'] = 'Surface pressure'

            new_unified_surface_data['AH'].attrs['units'] = 'percent'
            new_unified_surface_data['AH'].attrs['long_name'] = 'Mixing ratio'

            new_unified_surface_data['theta'].attrs['units'] = 'K'
            new_unified_surface_data['theta'].attrs['long_name'] = 'Potential temperature'

            new_unified_surface_data['theta_e'].attrs['units'] = 'K'
            new_unified_surface_data['theta_e'].attrs['long_name'] = 'Equivalent potential temperature'

            new_unified_surface_data['theta_v'].attrs['units'] = 'K'
            new_unified_surface_data['theta_v'].attrs['long_name'] = 'Virtual potential temperature'

            new_unified_surface_data['theta_w'].attrs['units'] = 'K'
            new_unified_surface_data['theta_w'].attrs['long_name'] = 'Wet-bulb potential temperature'

            surface_file_path = '%s/%02d/%02d/%02d/gdas_surface_%d%02d%02d%02d_f%03d_full.pkl' % (pickle_outdir, year, month, day, year, month, day, hour, forecast_hour)
            with open(surface_file_path, 'wb') as surface_file:
                pickle.dump(new_unified_surface_data, surface_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ERA5', action='store_true', required=False, help="Generate ERA5 data files")
    parser.add_argument('--netcdf_ERA5_indir', type=str, required=False, help="input directory for ERA5 netcdf files")
    parser.add_argument('--fronts', action='store_true', required=False, help="Generate front object data files")
    parser.add_argument('--xml_indir', type=str, required=False, help="input directory for front XML files")
    parser.add_argument('--download_gdas', action='store_true', help='Download GDAS tarfiles')
    parser.add_argument('--gdas_order_number', type=str, help='HAS order number for the GDAS data')
    parser.add_argument('--gdas_grib_to_pickle', action='store_true', required=False, help="Generate GDAS pickle files")
    parser.add_argument('--gdas_indir', type=str, required=False, help="input directory for the GDAS grib files")
    parser.add_argument('--gdas_outdir', type=str, help='Output directory for downloaded grib files.')
    parser.add_argument('--pickle_outdir', type=str, help="output directory for pickle files")
    parser.add_argument('--year', type=int, help="year for the data to be read in")
    parser.add_argument('--month', type=int, help="month for the data to be read in")
    parser.add_argument('--day', type=int, help="day for the data to be read in")
    parser.add_argument('--hour', type=int, help='hour for the grib data to be downloaded')
    args = parser.parse_args()
    provided_arguments = vars(args)

    if args.ERA5:
        required_arguments = ['year', 'month', 'day', 'netcdf_ERA5_indir', 'pickle_outdir']
        check_arguments(provided_arguments, required_arguments)
        create_era5_datasets(args.year, args.month, args.day, args.netcdf_ERA5_indir, args.pickle_outdir)
    if args.fronts:
        required_arguments = ['year', 'month', 'day', 'xml_indir', 'pickle_outdir']
        check_arguments(provided_arguments, required_arguments)
        front_xmls_to_pickle(args.year, args.month, args.day, args.xml_indir, args.pickle_outdir)
    if args.gdas_grib_to_pickle:
        required_arguments = ['year', 'month', 'day', 'gdas_indir', 'pickle_outdir']
        check_arguments(provided_arguments, required_arguments)
        gdas_grib_to_pickle(args.year, args.month, args.day, args.gdas_indir, args.pickle_outdir)
    if args.download_gdas:
        required_arguments = ['gdas_order_number', 'gdas_outdir']
        check_arguments(provided_arguments, required_arguments)
        download_gdas_order(args.gdas_order_number, args.gdas_outdir)
