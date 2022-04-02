"""
Data tools

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 3/28/2021 12:51 PM CST
"""

import math
import pandas as pd
from shapely.geometry import LineString
import numpy as np


def expand_fronts(ds_fronts):
    """
    Expands fronts by 1 pixel in all directions.

    Parameters
    ----------
    ds_fronts: Dataset that contains frontal objects.

    Returns
    -------
    ds_fronts: Dataset that contains expanded frontal objects.
    """
    lats = ds_fronts.latitude.values
    lons = ds_fronts.longitude.values
    len_lats = len(lats)
    len_lons = len(lons)
    indices = np.where(ds_fronts.identifier.values != 0)
    identifier = ds_fronts.identifier.values

    for i in range(len(indices[0])):
        front_value = identifier[indices[0][i]][indices[1][i]]
        if lats[indices[0][i]] == lats[0]:
            # If the front pixel is at the north end of the domain (max lat), and there is no front directly
            # to the south, expand the front 1 pixel south.
            if identifier[indices[0][i]][indices[1][i]] == 0:
                ds_fronts.identifier.values[indices[0][i]][indices[1][i]] = front_value
            # If the front pixel is at the northwest end of the domain (max/min lat/lon), check the pixels to the
            # southeast and east for fronts, and expand the front there if no other fronts are already present.
            if lons[indices[1][i]] == lons[0]:
                if identifier[indices[0][i] + 1][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] + 1][indices[1][i] + 1] = front_value
                if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] + 1] = front_value
            # If the front pixel is at the northeast end of the domain (max/max lat/lon), check the pixels to the
            # southwest and west for fronts, and expand the front there if no other fronts are already present.
            elif lons[indices[1][i]] == lons[len_lons - 1]:
                if identifier[indices[0][i] + 1][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] + 1][indices[1][i] - 1] = front_value
                if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] - 1] = front_value
            # If the front pixel is at the north end of the domain (max lat), but not at the west or east end (min lon
            # or max lon) check the pixels to the west and east for fronts, and expand the front there if no other
            # fronts are already present.
            else:
                if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] - 1] = front_value
                if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] + 1] = front_value
        elif lats[-1] < lats[indices[0][i]] < lats[len_lats - 1]:
            # If there is no front directly to the south, expand the front 1 pixel south.
            if identifier[indices[0][i] + 1][indices[1][i]] == 0:
                identifier[indices[0][i]][indices[1][i]] = front_value
            # If there is no front directly to the north, expand the front 1 pixel north.
            if identifier[indices[0][i] - 1][indices[1][i]] == 0:
                ds_fronts.identifier.values[indices[0][i] - 1][indices[1][i]] = front_value
            # If the front pixel is at the west end of the domain (min lon), check the pixels to the southeast,
            # east, and northeast for fronts, and expand the front there if no other fronts are already present.
            if lons[indices[1][i]] == lons[0]:
                if identifier[indices[0][i] + 1][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] + 1][indices[1][i] + 1] = front_value
                if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] + 1] = front_value
                if identifier[indices[0][i] - 1][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] - 1][indices[1][i] + 1] = front_value
            # If the front pixel is at the east end of the domain (min lon), check the pixels to the southwest,
            # west, and northwest for fronts, and expand the front there if no other fronts are already present.
            elif lons[indices[1][i]] == lons[len_lons - 1]:
                if identifier[indices[0][i] + 1][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] + 1][indices[1][i] - 1] = front_value
                if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] - 1] = front_value
                if identifier[indices[0][i] - 1][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] - 1][indices[1][i] - 1] = front_value
            # If the front pixel is not at the end of the domain in any direction, check the northeast, east,
            # southeast, northwest, west, and southwest for fronts, and expand the front there if no other fronts
            # are already present.
            else:
                if identifier[indices[0][i] + 1][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] + 1][indices[1][i] + 1] = front_value
                if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] + 1] = front_value
                if identifier[indices[0][i] - 1][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] - 1][indices[1][i] + 1] = front_value
                if identifier[indices[0][i] + 1][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] + 1][indices[1][i] - 1] = front_value
                if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] - 1] = front_value
                if identifier[indices[0][i] - 1][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] - 1][indices[1][i] - 1] = front_value
        else:
            # If the front pixel is at the south end of the domain (max lat), and there is no front directly
            # to the north, expand the front 1 pixel north.
            if identifier[indices[0][i] - 1][indices[1][i]] == 0:
                ds_fronts.identifier.values[indices[0][i] - 1][indices[1][i]] = front_value
            # If the front pixel is at the southwest end of the domain (max/min lat/lon), check the pixels to the
            # northeast and east for fronts, and expand the front there if no other fronts are already present.
            if lons[indices[1][i]] == lons[0]:
                if identifier[indices[0][i] - 1][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] - 1][indices[1][i] + 1] = front_value
                if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] + 1] = front_value
            # If the front pixel is at the southeast end of the domain (max/max lat/lon), check the pixels to the
            # northwest and west for fronts, and expand the front there if no other fronts are already present.
            elif lons[indices[1][i]] == lons[len_lons - 1]:
                if identifier[indices[0][i] - 1][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] - 1][indices[1][i] - 1] = front_value
                if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] - 1] = front_value
            # If the front pixel is at the south end of the domain (max lat), but not at the west or east end (min lon
            # or max lon) check the pixels to the west and east for fronts, and expand the front there if no other
            # fronts are already present.
            else:
                if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] - 1] = front_value
                if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] + 1] = front_value

    return ds_fronts


def haversine(lon1, lat1, lon2, lat2):
    """
    Converts lons/lats into x/y and calculates distance between points.

    Parameters
    ----------
    lon1: First longitude point in degrees to be interpolated.
    lat1: First latitude point in degrees to be interpolated.
    lon2: Second longitude point in degrees to be interpolated.
    lat2: Second latitude point in degrees to be interpolated.

    Returns
    -------
    dlon: Longitudinal distance between the two points in degrees.
    dlat: Latitudinal distance between the two points in degrees.
    dx: Longitudinal distance between the two points in kilometers (km).
    dy: Latitudinal distance between the two points in kilometers (km).

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
    x_km_new: List containing longitude coordinates of fronts in kilometers.
    y_km_new: List containing latitude coordinates of fronts in kilometers.

    Returns
    -------
    xy_linestring: LineString object containing coordinates of fronts in kilometers.
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
    xy_linestring: LineString object containing coordinates of fronts in kilometers.
    distance: Distance at which to interpolate the x/y coordinates.

    Returns
    -------
    xy_vertices: Normalized MultiLineString that contains the interpolate coordinates of fronts in kilometers.

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
    lons: List containing original longitude coordinates of fronts.
    lats: List containing original latitude coordinates of fronts.
    lon_new: List that will contain interpolated longitude coordinates of fronts.
    lat_new: List that will contain interpolated latitude coordinates of fronts.
    front_points: List that shows which points are in each front using indices.
    i: Current front number.
    a: Current index of the lon_new and lat_new lists.
    dx: Distance between the two selected longitude coordinates in kilometers.
    dy: Distance between the two selected latitude coordinates in kilometers.

    Returns
    -------
    lon1: First longitude point in degrees.
    lat1: First latitude point in degrees.
    lon2: Second longitude point in degrees.
    lat2: Second latitude point in degrees.
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
