"""
Functions used to create ERA5 plots.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 6/30/2021 1:30 PM CDT by Andrew Justin
"""

import matplotlib.pyplot as plt
import xarray as xr
from glob import glob
import argparse
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import create_NC_files
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import variables
plt.switch_backend('agg')


def read_xml_files_ERA5(year, month, day, hour, xml_indir):
    """
    Read the xml files to pull frontal objects.

    Parameters
    ----------
    year: int
    month: int
    day: int
    hour: int
        Hour of the day in UTC.
    xml_indir: str
        Directory for the XML files.

    Returns
    -------
    df: DataFrame
        DataFrame containing frontal data organized by date, type, number, and coordinates.
    cold_front: ndarray
        Numpy array containing indices where cold fronts are present in the DataFrame.
    warm_front: ndarray
        Numpy array containing indices where warm fronts are present in the DataFrame.
    occluded_front: ndarray
        Numpy array containing indices where occluded fronts are present in the DataFrame.
    stationary_front: ndarray
        Numpy array containing indices where stationary fronts are present in the DataFrame.
    dryline: ndarray
        Numpy array containing indices where drylines are present in the DataFrame.
    """
    files = glob("%s/pres_pmsl_%d%02d%02d%02df000.xml" % (xml_indir, year, month, day, hour))

    for filename in files:
        print(filename)
        tree = ET.parse(filename, parser=ET.XMLParser(encoding='utf-8'))
        root = tree.getroot()
        date = filename.split('_')[-1].split('.')[0].split('f')[0]
        fronttype = []
        frontnumber = []
        fronts_number = []  # This array is for the interpolated points' front numbers.
        dates = []
        i = 0  # Define counter for front number.
        lats = []
        lons = []
        point_number = 0
        front_points = []  # Array that shows which points are in each front.
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
            dlon, dlat, dx, dy = create_NC_files.haversine(lon1, lat1, lon2, lat2)
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
                    xy_linestring = create_NC_files.geometric(x_km_new, y_km_new)
                    xy_vertices = create_NC_files.redistribute_vertices(xy_linestring, distance)
                    x_new, y_new = xy_vertices.xy
                    for a in range(1, len(x_new)):
                        dx = x_new[a] - x_new[a - 1]
                        dy = y_new[a] - y_new[a - 1]
                        lon1, lon2, lat1, lat2 = create_NC_files.reverse_haversine(lons, lats, lon_new, lat_new,
                                                                                   front_points, i, a, dx, dy)
                        if a == 1:
                            lon_new.append(lon1)
                            lat_new.append(lat1)
                        lon_new.append(lon2)
                        lat_new.append(lat2)
                    for c in range(0, len(lon_new)):
                        if lon_new[c] > 179.999:
                            fronts_lon_array.append(lon_new[c] - 360)
                        else:
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
        xit = np.linspace(-180, 180, 1441)
        yit = np.linspace(0, 80, 321)
        df = df.assign(xit=np.digitize(df['Longitude'].values, xit))
        df = df.assign(yit=np.digitize(df['Latitude'].values, yit))

        cold_front = np.where(df['Front Type'] == 'COLD_FRONT')[0]
        # cold_front_diss = np.where(df['Front Type'] == 'COLD_FRONT_DISS')[0]
        # cold_front_form = np.where(df['Front Type'] == 'COLD_FRONT_FORM')[0]
        warm_front = np.where(df['Front Type'] == 'WARM_FRONT')[0]
        # warm_front_diss = np.where(df['Front Type'] == 'WARM_FRONT_DISS')[0]
        # warm_front_form = np.where(df['Front Type'] == 'WARM_FRONT_FORM')[0]
        occluded_front = np.where(df['Front Type'] == 'OCCLUDED_FRONT')[0]
        # occluded_front_diss = np.where(df['Front Type'] == 'OCCLUDED_FRONT_DISS')[0]
        # occluded_front_form = np.where(df['Front Type'] == 'OCCLUDED_FRONT_FORM')[0]
        stationary_front = np.where(df['Front Type'] == 'STATIONARY_FRONT')[0]
        # stationary_front_diss = np.where(df['Front Type'] == 'STATIONARY_FRONT_DISS')[0]
        # stationary_front_form = np.where(df['Front Type'] == 'STATIONARY_FRONT_FORM')[0]
        # instability = np.where(df['Front Type'] == 'INSTABILITY')[0]
        # trof = np.where(df['Front Type'] == 'TROF')[0]
        # tropical_trof = np.where(df['Front Type'] == 'TROPICAL_TROF')[0]
        dryline = np.where(df['Front Type'] == 'DRY_LINE')[0]
        return df, cold_front, warm_front, occluded_front, stationary_front, dryline


def cold_front_latlon_arrays(df, cold_front):
    """
    Return coordinates for points where cold fronts are located.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing points, front type, front number, and date.
    cold_front: ndarray
        Numpy array containing indices where cold fronts are located in the dataframe.

    Returns
    -------
    cold_front_lon: ndarray
        Numpy array with points of longitude where cold fronts are located.
    cold_front_lat: ndarray
        Numpy array with points of latitude where cold fronts are located.
    """
    cold_front_lon = []
    cold_front_lat = []
    for a in range(len(cold_front)):
        cold_front_lon.append(df['Longitude'][cold_front[a]])
        cold_front_lat.append(df['Latitude'][cold_front[a]])
    return cold_front_lon, cold_front_lat


def warm_front_latlon_arrays(df, warm_front):
    """
    Return coordinates for points where warm fronts are located.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing points, front type, front number, and date.
    warm_front: ndarray
        Numpy array containing indices where warm fronts are located in the dataframe.

    Returns
    -------
    warm_front_lon: ndarray
        Numpy array with points of longitude where warm fronts are located.
    warm_front_lat: ndarray
        Numpy array with points of latitude where warm fronts are located.
    """
    warm_front_lon = []
    warm_front_lat = []
    for b in range(len(warm_front)):
        warm_front_lon.append(df['Longitude'][warm_front[b]])
        warm_front_lat.append(df['Latitude'][warm_front[b]])
    return warm_front_lon, warm_front_lat


def occluded_front_latlon_arrays(df, occluded_front):
    """
    Return coordinates for points where occluded fronts are located.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing points, front type, front number, and date.
    occluded_front: ndarray
        Numpy array containing indices where occluded fronts are located in the dataframe.

    Returns
    -------
    occluded_front_lon: ndarray
        Numpy array with points of longitude where occluded fronts are located.
    occluded_front_lat: ndarray
        Numpy array with points of latitude where occluded fronts are located.
    """
    occluded_front_lon = []
    occluded_front_lat = []
    for c in range(len(occluded_front)):
        occluded_front_lon.append(df['Longitude'][occluded_front[c]])
        occluded_front_lat.append(df['Latitude'][occluded_front[c]])
    return occluded_front_lon, occluded_front_lat


def stationary_front_latlon_arrays(df, stationary_front):
    """
    Return coordinates for points where stationary fronts are located.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing points, front type, front number, and date.
    stationary_front: ndarray
        Numpy array containing indices where stationary fronts are located in the dataframe.

    Returns
    -------
    stationary_front_lon: ndarray
        Numpy array with points of longitude where stationary fronts are located.
    stationary_front_lat: ndarray
        Numpy array with points of latitude where stationary fronts are located.
    """
    stationary_front_lon = []
    stationary_front_lat = []
    for d in range(len(stationary_front)):
        stationary_front_lon.append(df['Longitude'][stationary_front[d]])
        stationary_front_lat.append(df['Latitude'][stationary_front[d]])
    return stationary_front_lon, stationary_front_lat


def dryline_latlon_arrays(df, dryline):
    """
    Return coordinates for points where drylines are located.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing points, front type, front number, and date.
    dryline: ndarray
        Numpy array containing indices where drylines are located in the dataframe.

    Returns
    -------
    dryline_front_lon: ndarray
        Numpy array with points of longitude where drylines are located.
    dryline_front_lat: ndarray
        Numpy array with points of latitude where drylines are located.
    """
    dryline_lon = []
    dryline_lat = []
    for d in range(len(dryline)):
        dryline_lon.append(df['Longitude'][dryline[d]])
        dryline_lat.append(df['Latitude'][dryline[d]])
    return dryline_lon, dryline_lat


def wind_components_dataset(ds_U10m, ds_V10m):
    """
    Returns xarray dataset containing u (zonal) and v (meridional) wind components.

    Parameters
    ----------
    ds_U10m: Dataset
        Dataset containing u wind components at 10 meters above the surface. Wind speed is in meters per second.
    ds_V10m: Dataset
        Dataset containing v wind components at 10 meters above the surface. Wind speed is in meters per second.

    Returns
    -------
    ds_wind: Dataset
        Dataset containing both u and v wind components at 10 meters above the surface. Wind speed is in meters per
        second.
    """
    ds_wind = xr.merge((ds_U10m, ds_V10m))
    return ds_wind


def plot_background(extent):
    """
    Returns new background for the plot.

    Parameters
    ----------
    extent: ndarray
        Numpy array containing the extent/boundaries of the plot in the format of [min lon, max lon, min lat, max lat].

    Returns
    -------
    ax: GeoAxesSubplot
        New plot background.
    """
    crs = ccrs.LambertConformal(central_longitude=250)
    ax = plt.axes(projection=crs)
    # ax.gridlines()
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    return ax


def create_datasets(year, month, day, netcdf_ERA5_indir):
    """
    Function that loads ERA5 files and creates variable datasets.

    Parameters
    ----------
    year: int
    month: int
    day: int
    netcdf_ERA5_indir: str
        Directory where the ERA5 files are stored.

    Returns
    -------
    ds_2mT: Dataset
        Dataset containing 2-meter AGL temperature data. Temperature has units of kelvin (K).
    ds_2mTd: Dataset
        Dataset containing 2-meter AGL dew point temperature data. Dew point temperature has units of kelvin (K).
    ds_sp: Dataset
        Dataset containing surface pressure data. Surface pressure has units of pascals (Pa).
    ds_U10m: Dataset
        Dataset containing u wind components at 10 meters above the surface. Wind speed is in meters per second.
    ds_V10m: Dataset
        Dataset containing v wind components at 10 meters above the surface. Wind speed is in meters per second.
    ds_wind: Dataset
        Dataset containing both u and v wind components at 10 meters above the surface. Wind speed is in meters per
        second.
    ds_theta_w: Dataset
        Dataset containing 2-meter AGL wet bulb potential temperature data. Wet bulb potential temperature has units of
        kelvin (K).
    ds_mixing_ratio: Dataset
        Dataset containing data for 2-meter AGL mixing ratio. Mixing ratio has units of grams of water per kilogram of
        dry air (g/kg).
    ds_RH: Dataset
        Dataset containing 2-meter AGL relative humidity data.
    ds_Tv: Dataset
        Dataset containing 2-meter AGL virtual temperature data. Virtual temperature has units of kelvin (K).
    ds_Tw: Dataset
        Dataset containing 2-meter AGL wet bulb temperature data. Wet bulb temperature has units of kelvin (K).
    """
    in_file_name_2mT = 'ERA5Global_%d_3hrly_2mT.nc' % year
    in_file_name_2mTd = 'ERA5Global_%d_3hrly_2mTd.nc' % year
    in_file_name_sp = 'ERA5Global_%d_3hrly_sp.nc' % year
    in_file_name_U10m = 'ERA5Global_%d_3hrly_U10m.nc' % year
    in_file_name_V10m = 'ERA5Global_%d_3hrly_V10m.nc' % year

    ds_2mT = xr.open_mfdataset("%s/%s" % (netcdf_ERA5_indir, in_file_name_2mT))
    ds_2mTd = xr.open_mfdataset("%s/%s" % (netcdf_ERA5_indir, in_file_name_2mTd))
    ds_sp = xr.open_mfdataset("%s/%s" % (netcdf_ERA5_indir, in_file_name_sp))
    ds_U10m = xr.open_mfdataset("%s/%s" % (netcdf_ERA5_indir, in_file_name_U10m))
    ds_V10m = xr.open_mfdataset("%s/%s" % (netcdf_ERA5_indir, in_file_name_V10m))

    timestring = "%d-%02d-%02d" % (year, month, day)

    ds_2mT = ds_2mT.sel(time='%s' % timestring)
    ds_2mTd = ds_2mTd.sel(time='%s' % timestring)
    ds_sp = ds_sp.sel(time='%s' % timestring)
    ds_U10m = ds_U10m.sel(time='%s' % timestring)
    ds_V10m = ds_V10m.sel(time='%s' % timestring)
    ds_wind = wind_components_dataset(ds_U10m, ds_V10m)

    ds = xr.merge((ds_2mT, ds_2mTd, ds_sp, ds_U10m, ds_V10m))

    ds_theta_w = variables.theta_w(ds.t2m, ds.d2m, ds.sp)  # Wet-bulb potential temperature dataset
    ds_mixing_ratio = variables.mixing_ratio(ds.d2m, ds.sp)  # Mixing ratio dataset
    ds_RH = variables.relative_humidity(ds.t2m, ds.d2m)  # Relative humidity dataset
    ds_Tv = variables.virtual_temperature(ds.t2m, ds.d2m, ds.sp)  # Virtual temperature dataset
    ds_Tw = variables.wet_bulb_temperature(ds.t2m, ds.d2m)  # Wet-bulb temperature dataset

    return ds_2mT, ds_2mTd, ds_sp, ds_U10m, ds_V10m, ds_wind, ds_theta_w, ds_mixing_ratio, ds_RH, ds_Tv, ds_Tw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create labeled data for the specified day")
    parser.add_argument('--year', type=int, required=True, help="year for the data to be read in")
    parser.add_argument('--month', type=int, required=True, help="month for the data to be read in")
    parser.add_argument('--day', type=int, required=True, help="day for the data to be read in")
    parser.add_argument('--hour', type=int, required=True, help="hour for the data to be read in")
    parser.add_argument('--image_outdir', type=str, required=False, help="output directory for image files")
    parser.add_argument('--xml_indir', type=str, required=True, help="input directory for XML files")
    parser.add_argument('--netcdf_indir', type=str, required=True, help="input directory for ERA5 netcdf files")
    parser.add_argument('--netcdf_ERA5_indir', type=str, required=True, help="input directory for ERA5 netcdf files")
    args = parser.parse_args()

    # Organize coordinates of fronts into arrays based on front type.
    df, cold_front, warm_front, occluded_front, stationary_front, dryline = read_xml_files_ERA5(args.year, args.month, args.day, args.hour, args.xml_indir)
    cold_front_lon, cold_front_lat = cold_front_latlon_arrays(df, cold_front)
    warm_front_lon, warm_front_lat = warm_front_latlon_arrays(df, warm_front)
    occluded_front_lon, occluded_front_lat = occluded_front_latlon_arrays(df, occluded_front)
    stationary_front_lon, stationary_front_lat = stationary_front_latlon_arrays(df, stationary_front)
    dryline_lon, dryline_lat = dryline_latlon_arrays(df, dryline)

    # Create datasets for each netCDF file containing an input variable.
    ds_2mT, ds_2mTd, ds_sp, ds_U10m, ds_V10m, ds_wind, ds_theta_w, ds_mixing_ratio, ds_RH, ds_Tv, ds_Tw \
        = create_datasets(args.year, args.month, args.day, args.netcdf_ERA5_indir)

    extent = [248, 268, 35, 50]
    ax = plot_background(extent)

    timestring = "%d-%02d-%02dT%d:00:00" % (args.year, args.month, args.day, args.hour)

    plotname = "%d-%02d-%02d-%02dz" % (args.year, args.month, args.day, args.hour)
    plt.title("%d-%02d-%02d-%02dz" % (args.year, args.month, args.day, args.hour))

    ds_theta_w = ds_theta_w.to_dataset(name='theta_w')

    # ds_theta_w.theta_w.sel(latitude=slice(extent[3]+10,extent[2]-10), longitude=slice(extent[0]-10,extent[1]+10)).plot(
    #    ax=ax,x='longitude',y='latitude', cmap="terrain_r", transform=ccrs.PlateCarree())

    # ds_2mT.t2m.sel(time=timestring).plot(ax=ax,x='longitude',y='latitude',transform=ccrs.PlateCarree())
    # print(ds_2mTd)
    # ds_theta_w.theta_w.sel(time=timestring).plot(ax=ax,x='longitude',y='latitude', transform=ccrs.PlateCarree())
    # ds_sp.sp.plot(ax=ax,x='longitude',y='latitude',transform=ccrs.PlateCarree())
    # ds.sp.plot(ax=ax,x='longitude',y='latitude',transform=ccrs.PlateCarree())
    ds_U10m.u10.sel(time=timestring).plot(ax=ax,x='longitude',y='latitude',transform=ccrs.PlateCarree())
    # ds.v10.sel(latitude=slice(38.5,35), longitude=slice(240,320)).plot(ax=ax,x='longitude',
    #            y='latitude',transform=ccrs.PlateCarree())

    barbs_longitude = ds_2mT.longitude.values[::4]
    barbs_latitude = ds_2mT.latitude.values[::4]
    barbs_u = ds_U10m.u10.values[::4, ::4]
    barbs_v = ds_V10m.v10.values[::4, ::4]

    # plt.barbs(barbs_longitude, barbs_latitude, u=barbs_u*1.94384, v=barbs_v*1.94384, linewidth=0.3,
    #          color='black', length=4, sizes={'height': 0.5, 'width': 0}, transform=ccrs.PlateCarree())
    plt.scatter(x=cold_front_lon, y=cold_front_lat, s=3, transform=ccrs.PlateCarree(), marker='^', color='blue')
    plt.scatter(x=warm_front_lon, y=warm_front_lat, s=3, transform=ccrs.PlateCarree(), marker='o', color='red')
    plt.scatter(x=occluded_front_lon, y=occluded_front_lat, s=3, transform=ccrs.PlateCarree(), marker='o',
                color='purple')
    plt.scatter(x=dryline_lon, y=dryline_lat, s=3, transform=ccrs.PlateCarree(), marker='o', color='orange')
    plt.scatter(x=stationary_front_lon, y=stationary_front_lat, s=3, transform=ccrs.PlateCarree(), marker='o',
                color='green')

    plt.savefig(os.path.join(args.image_outdir, '%s_ERA5_plot_temp.png' % plotname), bbox_inches='tight', dpi=1000)
