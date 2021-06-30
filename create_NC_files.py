"""
Function used to create netCDF files from the fronts' xml data.

Code started by: Alyssa Woodward (alyssakwoodward@ou.edu)
Code completed by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 6/30/2021 10:43 AM CDT
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xml.etree.ElementTree as ET
import pandas as pd
import glob
import argparse
import os
from shapely.geometry import LineString
plt.switch_backend('agg')


def haversine(lon1, lat1, lon2, lat2):
    """
    Converts lons/lats into x/y and calculates distance between points.

    Parameters
    ----------
    lon1: float
        First longitude point in degrees to be interpolated.
    lat1: float
        First latitude point in degrees to be interpolated.
    lon2: float
        Second longitude point in degrees to be interpolated.
    lat2: float
        Second latitude point in degrees to be interpolated.

    Returns
    -------
    dlon: float
        Longitudinal distance between the two points in degrees.
    dlat: float
        Latitudinal distance between the two points in degrees.
    dx: float
        Longitudinal distance between the two points in kilometers (km).
    dy: float
        Latitudinal distance between the two points in kilometers (km).

    Sources
    -------
    https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    https://stackoverflow.com/questions/24617013/convert-latitude-and-longitude-to-x-and-y-grid-system-using-python
    """
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    dx = (-dlon) * 40075 * math.cos((lat1 + lat2) * math.pi / 360) / 360  # circumference of earth in km = 40075
    dy = (-dlat) * 40075 / 360
    return dlon, dlat, dx, dy


def geometric(x_km_new, y_km_new):
    """
    Turn longitudinal/latitudinal distance (km) lists into LineString for interpolation.

    Parameters
    ----------
    x_km_new: list
        List containing longitude coordinates of fronts in kilometers.
    y_km_new: list
        List containing latitude coordinates of fronts in kilometers.

    Returns
    -------
    xy_linestring: LineString
        LineString object containing coordinates of fronts in kilometers.
    """
    df_xy = pd.DataFrame(list(zip(x_km_new, y_km_new)), columns=['Longitude_km', 'Latitude_km'])
    geometry = [xy for xy in zip(df_xy.Longitude_km, df_xy.Latitude_km)]
    xy_linestring = LineString(geometry)
    return xy_linestring


def redistribute_vertices(xy_linestring, distance):
    """
    Interpolate x/y coordinates at a specified distance.

    Parameters
    ----------
    xy_linestring: LineString
        LineString object containing coordinates of fronts in kilometers.
    distance: int
        Distance at which to interpolate the x/y coordinates.

    Returns
    -------
    xy_vertices: MultiLineString
        Normalized MultiLineString that contains the interpolate coordinates of fronts in kilometers.

    Sources
    -------
    https://stackoverflow.com/questions/34906124/interpolating-every-x-distance-along-multiline-in-shapely/35025274#35025274
    """
    if xy_linestring.geom_type == 'LineString':
        num_vert = int(round(xy_linestring.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [xy_linestring.interpolate(float(n) / num_vert, normalized=True)
             for n in range(num_vert + 1)])
    elif xy_linestring.geom_type == 'MultiLineString':
        parts = [redistribute_vertices(part, distance) for part in xy_linestring]
        return type(xy_linestring)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (xy_linestring.geom_type,))


def reverse_haversine(lons, lats, lon_new, lat_new, front_points, i, a, dx, dy):
    """
    Turns interpolated points from x/y coordinates to lon/lat coordinates.

    Parameters
    ----------
    lons: list
        List containing original longitude coordinates of fronts.
    lats: list
        List containing original latitude coordinates of fronts.
    lon_new: list
        List that will contain interpolated longitude coordinates of fronts.
    lat_new: list
        List that will contain interpolated latitude coordinates of fronts.
    front_points: list
        List that shows which points are in each front using indices.
    i: int
        Current front number.
    a: int
        Current index of the lon_new and lat_new lists.
    dx: float
        Distance between the two selected longitude coordinates in kilometers.
    dy: float
        Distance between the two selected latitude coordinates in kilometers.

    Returns
    -------
    lon1: float
        First longitude point in degrees.
    lat1: float
        First latitude point in degrees.
    lon2: float
        Second longitude point in degrees.
    lat2: float
        Second latitude point in degrees.
    """
    if i == 1:
        if a == 1:
            lon1 = lons[a - 1]
            lat1 = lats[a - 1]
        else:
            lon1 = lon_new[a - 1]
            lat1 = lat_new[a - 1]
    else:
        if a == 1:
            lon1 = lons[front_points[i - 2]]
            lat1 = lats[front_points[i - 2]]
        else:
            lon1 = lon_new[a - 1]
            lat1 = lat_new[a - 1]
    lat2 = lat1 - (dy * (360 / 40075))
    lon2 = lon1 - (dx * (360 / 40075) / (math.cos((math.pi / 360) * (lat1 + lat2))))
    return lon1, lon2, lat1, lat2


def read_xml_files(year, month, day):
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
        Xarray dataset containing front data organized by date, type, number, and coordinates.
    """
    file_path = "/ourdisk/hpc/ai2es/ajustin/xmls_2006122012-2020061900"
    print(file_path)
    files = []
    files.extend(glob.glob("%s/*%04d%02d%02d00f*.xml" % (file_path, year, month, day)))
    files.extend(glob.glob("%s/*%04d%02d%02d06f*.xml" % (file_path, year, month, day)))
    files.extend(glob.glob("%s/*%04d%02d%02d12f*.xml" % (file_path, year, month, day)))
    files.extend(glob.glob("%s/*%04d%02d%02d18f*.xml" % (file_path, year, month, day)))
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

                # For longitude - we want to convert these values to a 360 degree system. This will allow us to properly
                # interpolate across the dateline. If we were to use a longitude domain of -180 degrees to 180 degrees rather than
                # 0 degrees to 360 degrees, the interpolated points would wrap around the entire globe.
                if float(point.get("Lon")) < 0:
                    lons.append(float(point.get("Lon")) + 360)
                else:
                    lons.append(float(point.get("Lon")))

        # This is the second step in converting longitude to a 360 degree system. Fronts that cross the prime meridian are
        # only partially corrected due to one side of the front having a negative longitude coordinate. So, we will
        # convert the positive points to the system as well for a consistent adjustment. It is important to note that
        # these fronts that cross the prime meridian (0 degrees E/W) will have a range of 0 degrees to 450 degrees rather than 0 degrees to 360 degrees.
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
            dlon, dlat, dx, dy = haversine(lon1, lat1, lon2, lat2)
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
                if max(x_km_new) > 100000 or min(x_km_new) < -100000 or max(y_km_new) > 100000 or min(y_km_new) < -100000:
                    print("ERROR: Front %d contains corrupt data points, no points will be interpolated from the front."
                          % i)
                else:
                    xy_linestring = geometric(x_km_new, y_km_new)
                    xy_vertices = redistribute_vertices(xy_linestring, distance)
                    x_new, y_new = xy_vertices.xy
                    for a in range(1, len(x_new)):
                        dx = x_new[a] - x_new[a - 1]
                        dy = y_new[a] - y_new[a - 1]
                        lon1, lon2, lat1, lat2 = reverse_haversine(lons, lats, lon_new, lat_new, front_points, i, a, dx,
                                                                   dy)
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
    timestep = pd.date_range(start=df['Date'].dt.strftime('%Y-%m-%d')[0], periods=len(dss), freq='6H')
    dns = xr.concat(dss, dim=timestep)
    dns = dns.rename({'concat_dim': 'Date'})
    return dns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create labeled data for the specified day")
    parser.add_argument('--year', type=int, required=True, help="year for the data to be read in")
    parser.add_argument('--month', type=int, required=True, help="month for the data to be read in")
    parser.add_argument('--day', type=int, required=True, help="day for the data to be read in")
    parser.add_argument('--image_outdir', type=str, required=False, help="output directory for image files")
    parser.add_argument('--netcdf_outdir', type=str, required=False, help="output directory for netcdf files")
    args = parser.parse_args()

    xmls = read_xml_files(args.year, args.month, args.day)

    netcdf_outtime = str('%04d%02d%02d' % (args.year, args.month, args.day))
    out_file_name = 'FrontalCounts_%s.nc' % netcdf_outtime
    out_file_path = os.path.join(args.netcdf_outdir, out_file_name)
    xmls.to_netcdf(out_file_path)
