"""
Function that extracts variable and front data from a given domain and saves it into a pickle file.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 7/19/2021 3:16 PM CDT
"""

import argparse
import Plot_ERA5
import xarray as xr
import pickle
import pandas as pd
import create_NC_files as nc
import glob
import numpy as np
import xml.etree.ElementTree as ET
import variables


def read_xml_files_360(year, month, day):
    """
    Reads the xml files to pull frontal objects.

    Parameters
    ----------
    year: int
    month: int
    day: int

    Returns
    -------
    dns: Dataset
        Xarray dataset containing frontal object data organized by date, type, number, and coordinates.
    """
    file_path = "/ourdisk/hpc/ai2es/ajustin/xmls_2006122012-2020061900"
    print(file_path)

    # files = []
    # files.extend(glob.glob("%s/*%04d%02d%02d00f*.xml" % (file_path, year, month, day)))
    # files.extend(glob.glob("%s/*%04d%02d%02d06f*.xml" % (file_path, year, month, day)))
    # files.extend(glob.glob("%s/*%04d%02d%02d12f*.xml" % (file_path, year, month, day)))
    # files.extend(glob.glob("%s/*%04d%02d%02d18f*.xml" % (file_path, year, month, day)))

    files = sorted(glob.glob("%s/*%04d%02d%02d*.xml" % (file_path, year, month, day)))

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

            # Declare "temporary" x and y variables which will be used to add elements to the distance arrays.
            x_km_temp = 0
            y_km_temp = 0

            x_km = []
            y_km = []
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
        for x in range(0, len(front_points)):
            if x == 0:
                for y in range(0, front_points[x - 1]):
                    if lons[y] < 90 and max(lons[0:front_points[x - 1]]) > 270:
                        lons[y] = lons[y] + 360
            else:
                for y in range(front_points[x - 1], front_points[x]):
                    if lons[y] < 90 and max(lons[front_points[x - 1]:front_points[x]]) > 270:
                        lons[y] = lons[y] + 360

        for l in range(1, len(lats)):
            lon1 = lons[l - 1]
            lat1 = lats[l - 1]
            lon2 = lons[l]
            lat2 = lats[l]
            dlon, dlat, dx, dy = nc.haversine(lon1, lat1, lon2, lat2)
            x_km_temp = x_km_temp + dx
            y_km_temp = y_km_temp + dy
            x_km.append(x_km_temp)
            y_km.append(y_km_temp)

        # Reset the front counter.
        i = 0

        for line in root.iter('Line'):
            frontno = i
            frontty = line.get("pgenType")

            # Create new arrays for holding coordinates from separate fronts.
            x_km_new = []
            y_km_new = []

            i = i + 1
            if i == 1:
                x_km_new = x_km[0:front_points[i - 1] - 1]
                y_km_new = y_km[0:front_points[i - 1] - 1]
            elif i < len(front_points):
                x_km_new = x_km[front_points[i - 2] - 1:front_points[i - 1] - 1]
                y_km_new = y_km[front_points[i - 2] - 1:front_points[i - 1] - 1]
            if len(x_km_new) > 1:
                lon_new = []
                lat_new = []
                distance = 25  # Cartesian interval distance in kilometers.
                if (max(x_km_new) > 100000 or min(x_km_new) < -100000 or max(y_km_new) > 100000 or min(
                        y_km_new) < -100000):
                    print("ERROR: Front %d contains corrupt data points, no points will be interpolated from the front."
                          % i)
                else:
                    xy_linestring = nc.geometric(x_km_new, y_km_new)
                    xy_vertices = nc.redistribute_vertices(xy_linestring, distance)
                    x_new, y_new = xy_vertices.xy
                    for a in range(1, len(x_new)):
                        dx = x_new[a] - x_new[a - 1]
                        dy = y_new[a] - y_new[a - 1]
                        lon1, lon2, lat1, lat2 = nc.reverse_haversine(lons, lats, lon_new, lat_new, front_points,
                                                                      i, a, dx, dy)
                        if a == 1:
                            lon_new.append(lon1)
                            lat_new.append(lat1)
                        lon_new.append(lon2)
                        lat_new.append(lat2)
                    for c in range(0, len(lon_new)):
                        fronts_lon_array.append(lon_new[c])
                        fronts_lat_array.append(lat_new[c])
                        fronts_number.append(frontno)
                        front_types.append(frontty)
                        front_dates.append(date)

        df = pd.DataFrame(list(zip(front_dates, fronts_number, front_types, fronts_lat_array, fronts_lon_array)),
                          columns=['Date', 'Front Number', 'Front Type', 'Latitude', 'Longitude'])
        df['Latitude'] = df.Latitude.astype(float)
        df['Longitude'] = df.Longitude.astype(float)
        df['Front Number'] = df['Front Number'].astype(int)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d%H')
        xit = np.linspace(0, 360, 1441)
        yit = np.linspace(80, 0, 321)
        df = df.assign(xit=np.digitize(df['Longitude'].values, xit))
        df = df.assign(yit=np.digitize(df['Latitude'].values, yit))
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
            ds = xr.Dataset(data_vars={"Frequency": (['Latitude', 'Longitude'], frequency)},
                            coords={'Latitude': yit, 'Longitude': xit})

            type_da.append(ds)
        ds = xr.concat(type_da, dim=types)
        ds = ds.rename({'concat_dim': 'Type'})
        dss.append(ds)
    timestep = pd.date_range(start=df['Date'].dt.strftime('%Y-%m-%d')[0], periods=len(dss), freq='3H')
    dns = xr.concat(dss, dim=timestep)
    dns = dns.rename({'concat_dim': 'Date'})
    return dns


def extract_input_variables(lon, lat, year, month, day, netcdf_ERA5_indir):
    """
    Extract variable data for the specified coordinate domain, year, month, day, and hour.

    Parameters
    ----------
    lon: float (x2)
        Two values that specify the longitude domain in degrees in the 360 coordinate system: lon_MIN lon_MAX
    lat: float (x2)
        Two values that specify the latitude domain in degrees: lat_MIN lat_MAX
    year: int
    month: int
    day: int
    netcdf_ERA5_indir: str
        Directory where the ERA5 netCDF files are contained.

    Returns
    -------
    xr_pickle: Dataset
        Xarray dataset containing variable data for the specified domain.
    """
    ds_2mT, ds_2mTd, ds_sp, ds_U10m, ds_V10m, ds_wind, da_theta_w, da_mixing_ratio, da_RH, da_Tv, da_Tw \
        = Plot_ERA5.create_datasets(year, month, day, netcdf_ERA5_indir)

    ds_2mT = ds_2mT.sel(longitude=slice(lon[0], lon[1]), latitude=slice(lat[1], lat[0]))
    ds_2mTd = ds_2mTd.sel(longitude=slice(lon[0], lon[1]), latitude=slice(lat[1], lat[0]))
    ds_sp = ds_sp.sel(longitude=slice(lon[0], lon[1]), latitude=slice(lat[1], lat[0]))
    ds_U10m = ds_U10m.sel(longitude=slice(lon[0], lon[1]), latitude=slice(lat[1], lat[0]))
    ds_V10m = ds_V10m.sel(longitude=slice(lon[0], lon[1]), latitude=slice(lat[1], lat[0]))
    ds_theta_w = (da_theta_w.sel(longitude=slice(lon[0], lon[1]), latitude=slice(lat[1], lat[0]))).to_dataset(
        name='theta_w')
    ds_mixing_ratio = (da_mixing_ratio.sel(longitude=slice(lon[0], lon[1]), latitude=slice(lat[1], lat[0]))
                       ).to_dataset(name='mix_ratio')
    ds_RH = (da_RH.sel(longitude=slice(lon[0], lon[1]), latitude=slice(lat[1], lat[0]))).to_dataset(name='rel_humid')
    ds_Tv = (da_Tv.sel(longitude=slice(lon[0], lon[1]), latitude=slice(lat[1], lat[0]))).to_dataset(name='virt_temp')
    ds_Tw = (da_Tw.sel(longitude=slice(lon[0], lon[1]), latitude=slice(lat[1], lat[0]))).to_dataset(name='wet_bulb')
    ds_theta_e = variables.theta_e(ds_2mT.t2m, ds_2mTd.d2m, ds_sp.sp).to_dataset(name='theta_e')
    ds_q = variables.specific_humidity(ds_2mTd.d2m, ds_sp.sp).to_dataset(name='q')

    timestring = "%d-%02d-%02d" % (year, month, day)

    PL_data = xr.open_mfdataset(
        paths=('/ourdisk/hpc/ai2es/fronts/era5/Pressure_Level/ERA5Global_PL_%s_3hrly_Q.nc' % year,
               '/ourdisk/hpc/ai2es/fronts/era5/Pressure_Level/ERA5Global_PL_%s_3hrly_T.nc' % year,
               '/ourdisk/hpc/ai2es/fronts/era5/Pressure_Level/ERA5Global_PL_%s_3hrly_U.nc' % year,
               '/ourdisk/hpc/ai2es/fronts/era5/Pressure_Level/ERA5Global_PL_%s_3hrly_V.nc' % year,
               '/ourdisk/hpc/ai2es/fronts/era5/Pressure_Level/ERA5Global_PL_%s_3hrly_Z.nc' % year))
    PL_850 = PL_data.sel(level=850, time=('%s' % timestring), longitude=slice(lon[0], lon[1]),
                         latitude=slice(lat[1], lat[0]))
    PL_900 = PL_data.sel(level=900, time=('%s' % timestring), longitude=slice(lon[0], lon[1]),
                         latitude=slice(lat[1], lat[0]))
    PL_950 = PL_data.sel(level=950, time=('%s' % timestring), longitude=slice(lon[0], lon[1]),
                         latitude=slice(lat[1], lat[0]))
    PL_1000 = PL_data.sel(level=1000, time=('%s' % timestring), longitude=slice(lon[0], lon[1]),
                          latitude=slice(lat[1], lat[0]))

    lons = PL_850.longitude.values
    lats = PL_850.latitude.values
    time = PL_850.time.values

    q_850 = PL_850.q.values
    q_900 = PL_900.q.values
    q_950 = PL_950.q.values
    q_1000 = PL_1000.q.values
    t_850 = PL_850.t.values
    t_900 = PL_900.t.values
    t_950 = PL_950.t.values
    t_1000 = PL_1000.t.values
    u_850 = PL_850.u.values
    u_900 = PL_900.u.values
    u_950 = PL_950.u.values
    u_1000 = PL_1000.u.values
    v_850 = PL_850.v.values
    v_900 = PL_900.v.values
    v_950 = PL_950.v.values
    v_1000 = PL_1000.v.values
    z_850 = PL_850.z.values
    z_900 = PL_900.z.values
    z_950 = PL_950.z.values
    z_1000 = PL_1000.z.values

    d_850 = variables.dew_point_from_specific_humidity(85000, PL_850.t.values, PL_850.q.values)
    d_900 = variables.dew_point_from_specific_humidity(90000, PL_900.t.values, PL_900.q.values)
    d_950 = variables.dew_point_from_specific_humidity(95000, PL_950.t.values, PL_950.q.values)
    d_1000 = variables.dew_point_from_specific_humidity(100000, PL_1000.t.values, PL_1000.q.values)
    mix_ratio_850 = variables.mixing_ratio(d_850, 85000)
    mix_ratio_900 = variables.mixing_ratio(d_900, 90000)
    mix_ratio_950 = variables.mixing_ratio(d_950, 95000)
    mix_ratio_1000 = variables.mixing_ratio(d_1000, 100000)
    rel_humid_850 = variables.relative_humidity(PL_850.t.values, d_850)
    rel_humid_900 = variables.relative_humidity(PL_900.t.values, d_900)
    rel_humid_950 = variables.relative_humidity(PL_950.t.values, d_950)
    rel_humid_1000 = variables.relative_humidity(PL_1000.t.values, d_1000)
    theta_e_850 = variables.theta_e(PL_850.t.values, d_850, 85000)
    theta_e_900 = variables.theta_e(PL_900.t.values, d_900, 90000)
    theta_e_950 = variables.theta_e(PL_950.t.values, d_950, 95000)
    theta_e_1000 = variables.theta_e(PL_1000.t.values, d_1000, 100000)
    theta_w_850 = variables.theta_w(PL_850.t.values, d_850, 85000)
    theta_w_900 = variables.theta_w(PL_900.t.values, d_900, 90000)
    theta_w_950 = variables.theta_w(PL_950.t.values, d_950, 95000)
    theta_w_1000 = variables.theta_w(PL_1000.t.values, d_1000, 100000)
    virt_temp_850 = variables.virtual_temperature(PL_850.t.values, d_850, 85000)
    virt_temp_900 = variables.virtual_temperature(PL_900.t.values, d_900, 90000)
    virt_temp_950 = variables.virtual_temperature(PL_950.t.values, d_950, 95000)
    virt_temp_1000 = variables.virtual_temperature(PL_1000.t.values, d_1000, 100000)
    wet_bulb_850 = variables.wet_bulb_temperature(PL_850.t.values, d_850)
    wet_bulb_900 = variables.wet_bulb_temperature(PL_900.t.values, d_900)
    wet_bulb_950 = variables.wet_bulb_temperature(PL_950.t.values, d_950)
    wet_bulb_1000 = variables.wet_bulb_temperature(PL_1000.t.values, d_1000)

    new_850 = xr.Dataset(data_vars=dict(q_850=(['time', 'latitude', 'longitude'], q_850),
                                        t_850=(['time', 'latitude', 'longitude'], t_850),
                                        u_850=(['time', 'latitude', 'longitude'], u_850),
                                        v_850=(['time', 'latitude', 'longitude'], v_850),
                                        z_850=(['time', 'latitude', 'longitude'], z_850),
                                        d_850=(['time', 'latitude', 'longitude'], d_850),
                                        mix_ratio_850=(['time', 'latitude', 'longitude'], mix_ratio_850),
                                        rel_humid_850=(['time', 'latitude', 'longitude'], rel_humid_850),
                                        theta_e_850=(['time', 'latitude', 'longitude'], theta_e_850),
                                        theta_w_850=(['time', 'latitude', 'longitude'], theta_w_850),
                                        virt_temp_850=(['time', 'latitude', 'longitude'], virt_temp_850),
                                        wet_bulb_850=(['time', 'latitude', 'longitude'], wet_bulb_850)),
                         coords=dict(latitude=lats, longitude=lons, time=time))
    new_900 = xr.Dataset(data_vars=dict(q_900=(['time', 'latitude', 'longitude'], q_900),
                                        t_900=(['time', 'latitude', 'longitude'], t_900),
                                        u_900=(['time', 'latitude', 'longitude'], u_900),
                                        v_900=(['time', 'latitude', 'longitude'], v_900),
                                        z_900=(['time', 'latitude', 'longitude'], z_900),
                                        d_900=(['time', 'latitude', 'longitude'], d_900),
                                        mix_ratio_900=(['time', 'latitude', 'longitude'], mix_ratio_900),
                                        rel_humid_900=(['time', 'latitude', 'longitude'], rel_humid_900),
                                        theta_e_900=(['time', 'latitude', 'longitude'], theta_e_900),
                                        theta_w_900=(['time', 'latitude', 'longitude'], theta_w_900),
                                        virt_temp_900=(['time', 'latitude', 'longitude'], virt_temp_900),
                                        wet_bulb_900=(['time', 'latitude', 'longitude'], wet_bulb_900)),
                         coords=dict(latitude=lats, longitude=lons, time=time))
    new_950 = xr.Dataset(data_vars=dict(q_950=(['time', 'latitude', 'longitude'], q_950),
                                        t_950=(['time', 'latitude', 'longitude'], t_950),
                                        u_950=(['time', 'latitude', 'longitude'], u_950),
                                        v_950=(['time', 'latitude', 'longitude'], v_950),
                                        z_950=(['time', 'latitude', 'longitude'], z_950),
                                        d_950=(['time', 'latitude', 'longitude'], d_950),
                                        mix_ratio_950=(['time', 'latitude', 'longitude'], mix_ratio_950),
                                        rel_humid_950=(['time', 'latitude', 'longitude'], rel_humid_950),
                                        theta_e_950=(['time', 'latitude', 'longitude'], theta_e_950),
                                        theta_w_950=(['time', 'latitude', 'longitude'], theta_w_950),
                                        virt_temp_950=(['time', 'latitude', 'longitude'], virt_temp_950),
                                        wet_bulb_950=(['time', 'latitude', 'longitude'], wet_bulb_950)),
                         coords=dict(latitude=lats, longitude=lons, time=time))
    new_1000 = xr.Dataset(data_vars=dict(q_1000=(['time', 'latitude', 'longitude'], q_1000),
                                        t_1000=(['time', 'latitude', 'longitude'], t_1000),
                                        u_1000=(['time', 'latitude', 'longitude'], u_1000),
                                        v_1000=(['time', 'latitude', 'longitude'], v_1000),
                                        z_1000=(['time', 'latitude', 'longitude'], z_1000),
                                        d_1000=(['time', 'latitude', 'longitude'], d_1000),
                                        mix_ratio_1000=(['time', 'latitude', 'longitude'], mix_ratio_1000),
                                        rel_humid_1000=(['time', 'latitude', 'longitude'], rel_humid_1000),
                                        theta_e_1000=(['time', 'latitude', 'longitude'], theta_e_1000),
                                        theta_w_1000=(['time', 'latitude', 'longitude'], theta_w_1000),
                                        virt_temp_1000=(['time', 'latitude', 'longitude'], virt_temp_1000),
                                        wet_bulb_1000=(['time', 'latitude', 'longitude'], wet_bulb_1000)),
                         coords=dict(latitude=lats, longitude=lons, time=time))

    ds_pickle = [ds_2mT, ds_2mTd, ds_sp, ds_U10m, ds_V10m, ds_theta_w, ds_mixing_ratio, ds_RH, ds_Tv, ds_Tw, ds_theta_e,
                 ds_q, new_850, new_900, new_950, new_1000]

    xr_pickle = xr.merge(ds_pickle, combine_attrs='override')
    xr_pickle.q.values = xr_pickle.q.values
    xr_pickle.t2m.values = xr_pickle.t2m.values
    xr_pickle.d2m.values = xr_pickle.d2m.values
    xr_pickle.sp.values = xr_pickle.sp.values
    xr_pickle.u10.values = xr_pickle.u10.values
    xr_pickle.v10.values = xr_pickle.v10.values
    xr_pickle.theta_w.values = xr_pickle.theta_w.values
    xr_pickle.theta_e.values = xr_pickle.theta_e.values
    xr_pickle.mix_ratio.values = xr_pickle.mix_ratio.values
    xr_pickle.rel_humid.values = xr_pickle.rel_humid.values
    xr_pickle.virt_temp.values = xr_pickle.virt_temp.values
    xr_pickle.wet_bulb.values = xr_pickle.wet_bulb.values

    xr_pickle.t2m.attrs['units'] = 'K'
    xr_pickle.t2m.attrs['long_name'] = '2m AGL Temperature'
    xr_pickle.d2m.attrs['units'] = 'K'
    xr_pickle.d2m.attrs['long_name'] = '2m AGL Dewpoint temperature'
    xr_pickle.sp.attrs['units'] = 'Pa'
    xr_pickle.sp.attrs['long_name'] = 'Surface pressure'
    xr_pickle.u10.attrs['units'] = 'm/s'
    xr_pickle.u10.attrs['long_name'] = '10m AGL Zonal wind velocity'
    xr_pickle.v10.attrs['units'] = 'm/s'
    xr_pickle.v10.attrs['long_name'] = '10m AGL Meridional wind velocity'
    xr_pickle.theta_w.attrs['units'] = 'K'
    xr_pickle.theta_w.attrs['long_name'] = '2m AGL Wet-bulb potential temperature'
    xr_pickle.mix_ratio.attrs['units'] = 'g(H2O)/g(air)'
    xr_pickle.mix_ratio.attrs['long_name'] = '2m AGL Mixing ratio'
    xr_pickle.rel_humid.attrs['long_name'] = '2m AGL Relative humidity'
    xr_pickle.virt_temp.attrs['units'] = 'K'
    xr_pickle.virt_temp.attrs['long_name'] = '2m AGL Virtual temperature'
    xr_pickle.wet_bulb.attrs['units'] = 'K'
    xr_pickle.wet_bulb.attrs['long_name'] = '2m AGL Wet-bulb temperature'
    xr_pickle.theta_e.attrs['units'] = 'K'
    xr_pickle.theta_e.attrs['long_name'] = '2m AGL Equivalent potential temperature'
    xr_pickle.q.attrs['units'] = 'g(H20)/g(air)'
    xr_pickle.q.attrs['long_name'] = '2m AGL Specific humidity'

    xr_pickle.q_850.attrs['units'] = 'g(H20)/g(air)'
    xr_pickle.q_850.attrs['long_name'] = '850mb Specific humidity'
    xr_pickle.t_850.attrs['units'] = 'K'
    xr_pickle.t_850.attrs['long_name'] = '850mb Temperature'
    xr_pickle.u_850.attrs['units'] = 'm/s'
    xr_pickle.u_850.attrs['long_name'] = '850mb Zonal wind velocity'
    xr_pickle.v_850.attrs['units'] = 'm/s'
    xr_pickle.v_850.attrs['long_name'] = '850mb Meridional wind velocity'
    xr_pickle.z_850.attrs['units'] = 'm'
    xr_pickle.z_850.attrs['long_name'] = '850mb Altitude'
    xr_pickle.d_850.attrs['units'] = 'K'
    xr_pickle.d_850.attrs['long_name'] = '850mb Dewpoint temperature'
    xr_pickle.mix_ratio_850.attrs['units'] = 'g(H20)/g(air)'
    xr_pickle.mix_ratio_850.attrs['long_name'] = '850mb Mixing ratio'
    xr_pickle.rel_humid_850.attrs['long_name'] = '850mb Relative humidity'
    xr_pickle.theta_e_850.attrs['units'] = 'K'
    xr_pickle.theta_e_850.attrs['long_name'] = '850mb Equivalent potential temperature'
    xr_pickle.theta_w_850.attrs['units'] = 'K'
    xr_pickle.theta_w_850.attrs['long_name'] = '850mb Wet-bulb potential temperature'
    xr_pickle.virt_temp_850.attrs['units'] = 'K'
    xr_pickle.virt_temp_850.attrs['long_name'] = '850mb Virtual potential temperature'
    xr_pickle.wet_bulb_850.attrs['units'] = 'K'
    xr_pickle.wet_bulb_850.attrs['long_name'] = '850mb Wet-bulb temperature'

    xr_pickle.q_900.attrs['units'] = 'g(H20)/g(air)'
    xr_pickle.q_900.attrs['long_name'] = '900mb Specific humidity'
    xr_pickle.t_900.attrs['units'] = 'K'
    xr_pickle.t_900.attrs['long_name'] = '900mb Temperature'
    xr_pickle.u_900.attrs['units'] = 'm/s'
    xr_pickle.u_900.attrs['long_name'] = '900mb Zonal wind velocity'
    xr_pickle.v_900.attrs['units'] = 'm/s'
    xr_pickle.v_900.attrs['long_name'] = '900mb Meridional wind velocity'
    xr_pickle.z_900.attrs['units'] = 'm'
    xr_pickle.z_900.attrs['long_name'] = '900mb Altitude'
    xr_pickle.d_900.attrs['units'] = 'K'
    xr_pickle.d_900.attrs['long_name'] = '900mb Dewpoint temperature'
    xr_pickle.mix_ratio_900.attrs['units'] = 'g(H20)/g(air)'
    xr_pickle.mix_ratio_900.attrs['long_name'] = '900mb Mixing ratio'
    xr_pickle.rel_humid_900.attrs['long_name'] = '900mb Relative humidity'
    xr_pickle.theta_e_900.attrs['units'] = 'K'
    xr_pickle.theta_e_900.attrs['long_name'] = '900mb Equivalent potential temperature'
    xr_pickle.theta_w_900.attrs['units'] = 'K'
    xr_pickle.theta_w_900.attrs['long_name'] = '900mb Wet-bulb potential temperature'
    xr_pickle.virt_temp_900.attrs['units'] = 'K'
    xr_pickle.virt_temp_900.attrs['long_name'] = '900mb Virtual potential temperature'
    xr_pickle.wet_bulb_900.attrs['units'] = 'K'
    xr_pickle.wet_bulb_900.attrs['long_name'] = '900mb Wet-bulb temperature'

    xr_pickle.q_950.attrs['units'] = 'g(H20)/g(air)'
    xr_pickle.q_950.attrs['long_name'] = '950mb Specific humidity'
    xr_pickle.t_950.attrs['units'] = 'K'
    xr_pickle.t_950.attrs['long_name'] = '950mb Temperature'
    xr_pickle.u_950.attrs['units'] = 'm/s'
    xr_pickle.u_950.attrs['long_name'] = '950mb Zonal wind velocity'
    xr_pickle.v_950.attrs['units'] = 'm/s'
    xr_pickle.v_950.attrs['long_name'] = '950mb Meridional wind velocity'
    xr_pickle.z_950.attrs['units'] = 'm'
    xr_pickle.z_950.attrs['long_name'] = '950mb Altitude'
    xr_pickle.d_950.attrs['units'] = 'K'
    xr_pickle.d_950.attrs['long_name'] = '950mb Dewpoint temperature'
    xr_pickle.mix_ratio_950.attrs['units'] = 'g(H20)/g(air)'
    xr_pickle.mix_ratio_950.attrs['long_name'] = '950mb Mixing ratio'
    xr_pickle.rel_humid_950.attrs['long_name'] = '950mb Relative humidity'
    xr_pickle.theta_e_950.attrs['units'] = 'K'
    xr_pickle.theta_e_950.attrs['long_name'] = '950mb Equivalent potential temperature'
    xr_pickle.theta_w_950.attrs['units'] = 'K'
    xr_pickle.theta_w_950.attrs['long_name'] = '950mb Wet-bulb potential temperature'
    xr_pickle.virt_temp_950.attrs['units'] = 'K'
    xr_pickle.virt_temp_950.attrs['long_name'] = '950mb Virtual potential temperature'
    xr_pickle.wet_bulb_950.attrs['units'] = 'K'
    xr_pickle.wet_bulb_950.attrs['long_name'] = '950mb Wet-bulb temperature'

    xr_pickle.q_1000.attrs['units'] = 'g(H20)/g(air)'
    xr_pickle.q_1000.attrs['long_name'] = '1000mb Specific humidity'
    xr_pickle.t_1000.attrs['units'] = 'K'
    xr_pickle.t_1000.attrs['long_name'] = '1000mb Temperature'
    xr_pickle.u_1000.attrs['units'] = 'm/s'
    xr_pickle.u_1000.attrs['long_name'] = '1000mb Zonal wind velocity'
    xr_pickle.v_1000.attrs['units'] = 'm/s'
    xr_pickle.v_1000.attrs['long_name'] = '1000mb Meridional wind velocity'
    xr_pickle.z_1000.attrs['units'] = 'm'
    xr_pickle.z_1000.attrs['long_name'] = '1000mb Altitude'
    xr_pickle.d_1000.attrs['units'] = 'K'
    xr_pickle.d_1000.attrs['long_name'] = '1000mb Dewpoint temperature'
    xr_pickle.mix_ratio_1000.attrs['units'] = 'g(H20)/g(air)'
    xr_pickle.mix_ratio_1000.attrs['long_name'] = '1000mb Mixing ratio'
    xr_pickle.rel_humid_1000.attrs['long_name'] = '1000mb Relative humidity'
    xr_pickle.theta_e_1000.attrs['units'] = 'K'
    xr_pickle.theta_e_1000.attrs['long_name'] = '1000mb Equivalent potential temperature'
    xr_pickle.theta_w_1000.attrs['units'] = 'K'
    xr_pickle.theta_w_1000.attrs['long_name'] = '1000mb Wet-bulb potential temperature'
    xr_pickle.virt_temp_1000.attrs['units'] = 'K'
    xr_pickle.virt_temp_1000.attrs['long_name'] = '1000mb Virtual potential temperature'
    xr_pickle.wet_bulb_1000.attrs['units'] = 'K'
    xr_pickle.wet_bulb_1000.attrs['long_name'] = '1000mb Wet-bulb temperature'

    print(xr_pickle)

    return xr_pickle


def save_variable_data_to_pickle(year, month, day, hour, xr_pickle, pickle_outdir):
    """
    Saves variable data to the pickle file.

    Parameters
    ----------
    year: int
    month: int
    day: int
    hour: int
        Hour in UTC.
    xr_pickle: Dataset
        Xarray dataset containing variable data for the specified domain.
    pickle_outdir: str
        Directory where the created pickle files containing the variable data will be stored.
    """
    xr_pickle_data = xr_pickle.sel(time='%d-%02d-%02dT%02d:00:00' % (year, month, day, hour))

    filename = "Data_60var_%04d%02d%02d%02d_conus_289x129.pkl" % (year, month, day, hour)

    print(filename)

    xr_pickle_data.load()
    outfile = open("%s/%d/%02d/%02d/%s" % (pickle_outdir, year, month, day, filename), 'wb')
    pickle.dump(xr_pickle_data, outfile)
    outfile.close()


def save_fronts_CFWF_to_pickle(ds, year, month, day, hour, pickle_outdir):
    """
    Saves warm front and cold front data to a pickle file.

    Parameters
    ----------
    ds: Dataset
        Xarray dataset containing warm front and cold front data.
    year: int
    month: int
    day: int
    hour: int
        Hour in UTC.
    pickle_outdir: str
        Directory where the created pickle files containing warm front and cold front data will be stored.
    """
    xr_pickle = ds.sel(time='%d-%02d-%02dT%02d:00:00' % (year, month, day, hour))

    fronttype = np.empty([len(xr_pickle.latitude), len(xr_pickle.longitude)])

    time = xr_pickle.time
    frequency = xr_pickle.Frequency.values
    types = xr_pickle.type.values
    lats = xr_pickle.latitude.values
    lons = xr_pickle.longitude.values

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

    xr_pickle_front = xr.Dataset({"identifier": (('latitude', 'longitude'), fronttype)},
                                 coords={"latitude": lats, "longitude": lons, "time": time})

    filename = "FrontObjects_CFWF_%04d%02d%02d%02d_conus_289x129.pkl" % (year, month, day, hour)

    print(filename)

    xr_pickle_front.load()
    outfile = open("%s/%d/%02d/%02d/%s" % (pickle_outdir, year, month, day, filename), 'wb')
    pickle.dump(xr_pickle_front, outfile)
    outfile.close()


def save_fronts_SFOF_to_pickle(ds, year, month, day, hour, pickle_outdir):
    """
    Saves stationary front and occluded front data to a pickle file.

    Parameters
    ----------
    ds: Dataset
        Xarray dataset containing stationary front and occluded front data.
    year: int
    month: int
    day: int
    hour: int
        Hour in UTC.
    pickle_outdir: str
        Directory where the created pickle files containing stationary front and occluded front data will be stored.
    """
    xr_pickle = ds.sel(time='%d-%02d-%02dT%02d:00:00' % (year, month, day, hour))

    fronttype = np.empty([len(xr_pickle.latitude), len(xr_pickle.longitude)])

    time = xr_pickle.time
    frequency = xr_pickle.Frequency.values
    types = xr_pickle.type.values
    lats = xr_pickle.latitude.values
    lons = xr_pickle.longitude.values

    for i in range(len(lats)):
        for j in range(len(lons)):
            for k in range(len(types)):
                if types[k] == 'STATIONARY_FRONT':
                    if frequency[k][i][j] > 0:
                        fronttype[i][j] = 3
                elif types[k] == 'OCCLUDED_FRONT':
                    if frequency[k][i][j] > 0:
                        fronttype[i][j] = 4
                    else:
                        fronttype[i][j] = 0

    xr_pickle_front = xr.Dataset({"identifier": (('latitude', 'longitude'), fronttype)},
                                 coords={"latitude": lats, "longitude": lons, "time": time})

    filename = "FrontObjects_SFOF_%04d%02d%02d%02d_conus_289x129.pkl" % (year, month, day, hour)

    print(filename)

    xr_pickle_front.load()
    outfile = open("%s/%d/%02d/%02d/%s" % (pickle_outdir, year, month, day, filename), 'wb')
    pickle.dump(xr_pickle_front, outfile)
    outfile.close()


def save_fronts_DL_to_pickle(ds, year, month, day, hour, pickle_outdir):
    """
    Saves dryline data to a pickle file.

    Parameters
    ----------
    ds: Dataset
        Xarray dataset containing dryline data.
    year: int
    month: int
    day: int
    hour: int
        Hour in UTC.
    pickle_outdir: str
        Directory where the created pickle files containing dryline data will be stored.
    """
    xr_pickle = ds.sel(time='%d-%02d-%02dT%02d:00:00' % (year, month, day, hour))

    fronttype = np.empty([len(xr_pickle.latitude), len(xr_pickle.longitude)])

    time = xr_pickle.time
    frequency = xr_pickle.Frequency.values
    types = xr_pickle.type.values
    lats = xr_pickle.latitude.values
    lons = xr_pickle.longitude.values

    for i in range(len(lats)):
        for j in range(len(lons)):
            for k in range(len(types)):
                if types[k] == 'DRY_LINE':
                    if frequency[k][i][j] > 0:
                        fronttype[i][j] = 5
                    else:
                        fronttype[i][j] = 0

    xr_pickle_front = xr.Dataset({"identifier": (('latitude', 'longitude'), fronttype)},
                                 coords={"latitude": lats, "longitude": lons, "time": time})

    filename = "FrontObjects_DL_%04d%02d%02d%02d_conus_289x129.pkl" % (year, month, day, hour)

    print(filename)

    xr_pickle_front.load()
    outfile = open("%s/%d/%02d/%02d/%s" % (pickle_outdir, year, month, day, filename), 'wb')
    pickle.dump(xr_pickle_front, outfile)
    outfile.close()


def save_fronts_ALL_to_pickle(ds, year, month, day, hour, pickle_outdir):
    """
    Saves all front data (cold, warm, stationary, occluded, dryline) to a pickle file.

    Parameters
    ----------
    ds: Dataset
        Xarray dataset containing all front data.
    year: int
    month: int
    day: int
    hour: int
        Hour in UTC.
    pickle_outdir: str
        Directory where the created pickle files containing all front data will be stored.
    """
    xr_pickle = ds.sel(time='%d-%02d-%02dT%02d:00:00' % (year, month, day, hour))

    fronttype = np.empty([len(xr_pickle.latitude), len(xr_pickle.longitude)])

    time = xr_pickle.time
    frequency = xr_pickle.Frequency.values
    types = xr_pickle.type.values
    lats = xr_pickle.latitude.values
    lons = xr_pickle.longitude.values

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
                elif types[k] == 'DRY_LINE':
                    if frequency[k][i][j] > 0:
                        fronttype[i][j] = 5

    xr_pickle_front = xr.Dataset({"identifier": (('latitude', 'longitude'), fronttype)},
                                 coords={"latitude": lats, "longitude": lons, "time": time})

    filename = "FrontObjects_ALL_%04d%02d%02d%02d_conus_289x129.pkl" % (year, month, day, hour)

    print(filename)

    xr_pickle_front.load()
    outfile = open("%s/%d/%02d/%02d/%s" % (pickle_outdir, year, month, day, filename), 'wb')
    pickle.dump(xr_pickle_front, outfile)
    outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--netcdf_ERA5_indir', type=str, required=False, help="input directory for ERA5 netcdf files")
    parser.add_argument('--pickle_outdir', type=str, required=False, help="output directory for pickle files")
    parser.add_argument('--longitude', type=float, nargs=2, help="Longitude domain in degrees: lon_MIN lon_MAX")
    parser.add_argument('--latitude', type=float, nargs=2, help="Latitude domain in degrees: lat_MIN lat_MAX")
    parser.add_argument('--year', type=int, required=False, help="year for the data to be read in")
    parser.add_argument('--month', type=int, required=False, help="month for the data to be read in")
    parser.add_argument('--day', type=int, required=False, help="day for the data to be read in")
    args = parser.parse_args()

    xr_pickle = extract_input_variables(args.longitude, args.latitude, args.year, args.month, args.day,
                                        args.netcdf_ERA5_indir)
    # ds = read_xml_files_360(args.year, args.month, args.day)

    for hour in range(0, 24, 3):
        # ds_hour = ds.sel(Longitude=slice(args.longitude[0], args.longitude[1]), Latitude=slice(args.latitude[1],
        #                                                                                       args.latitude[0]))
        # ds_hour = ds_hour.rename(Latitude='latitude', Longitude='longitude', Type='type', Date='time')

        save_variable_data_to_pickle(args.year, args.month, args.day, hour, xr_pickle, args.pickle_outdir)

        # save_fronts_CFWF_to_pickle(ds_hour, args.year, args.month, args.day, hour, args.pickle_outdir)
        # save_fronts_SFOF_to_pickle(ds_hour, args.year, args.month, args.day, hour, args.pickle_outdir)
        # save_fronts_DL_to_pickle(ds_hour, args.year, args.month, args.day, hour, args.pickle_outdir)
        # save_fronts_ALL_to_pickle(ds_hour, args.year, args.month, args.day, hour, args.pickle_outdir)
