"""
Various data tools.

References
----------
* Snyder 1987: https://doi.org/10.3133/pp1395

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.9.18
"""

import pandas as pd
from shapely.geometry import LineString
import numpy as np
import xarray as xr
import tensorflow as tf


# Each variable has parameters in the format of [max, min]
normalization_parameters = {'mslp_z_surface': [1050., 960.],
                            'mslp_z_1000': [48., -69.],
                            'mslp_z_950': [86., -27.],
                            'mslp_z_900': [127., 17.],
                            'mslp_z_850': [174., 63.],
                            'q_surface': [24., 0.],
                            'q_1000': [26., 0.],
                            'q_950': [26., 0.],
                            'q_900': [23., 0.],
                            'q_850': [21., 0.],
                            'RH_surface': [1., 0.],
                            'RH_1000': [1., 0.],
                            'RH_950': [1., 0.],
                            'RH_900': [1., 0.],
                            'RH_850': [1., 0.],
                            'r_surface': [25., 0.],
                            'r_1000': [22., 0.],
                            'r_950': [22., 0.],
                            'r_900': [20., 0.],
                            'r_850': [18., 0.],
                            'sp_z_surface': [1075., 620.],
                            'sp_z_1000': [48., -69.],
                            'sp_z_950': [86., -27.],
                            'sp_z_900': [127., 17.],
                            'sp_z_850': [174., 63.],
                            'theta_surface': [331., 213.],
                            'theta_1000': [322., 218.],
                            'theta_950': [323., 219.],
                            'theta_900': [325., 227.],
                            'theta_850': [330., 237.],
                            'theta_e_surface': [375., 213.],
                            'theta_e_1000': [366., 208.],
                            'theta_e_950': [367., 210.],
                            'theta_e_900': [364., 227.],
                            'theta_e_850': [359., 238.],
                            'theta_v_surface': [324., 212.],
                            'theta_v_1000': [323., 218.],
                            'theta_v_950': [319., 215.],
                            'theta_v_900': [315., 220.],
                            'theta_v_850': [316., 227.],
                            'theta_w_surface': [304., 212.],
                            'theta_w_1000': [301., 207.],
                            'theta_w_950': [302., 210.],
                            'theta_w_900': [301., 214.],
                            'theta_w_850': [300., 237.],
                            'T_surface': [323., 212.],
                            'T_1000': [322., 218.],
                            'T_950': [319., 216.],
                            'T_900': [314., 220.],
                            'T_850': [315., 227.],
                            'Td_surface': [304., 207.],
                            'Td_1000': [302., 208.],
                            'Td_950': [301., 210.],
                            'Td_900': [298., 200.],
                            'Td_850': [296., 200.],
                            'Tv_surface': [324., 211.],
                            'Tv_1000': [323., 206.],
                            'Tv_950': [319., 206.],
                            'Tv_900': [316., 220.],
                            'Tv_850': [316., 227.],
                            'Tw_surface': [305., 212.],
                            'Tw_1000': [305., 218.],
                            'Tw_950': [304., 216.],
                            'Tw_900': [301., 219.],
                            'Tw_850': [299., 227.],
                            'u_surface': [36., -35.],
                            'u_1000': [38., -35.],
                            'u_950': [48., -55.],
                            'u_900': [59., -58.],
                            'u_850': [59., -58.],
                            'v_surface': [30., -35.],
                            'v_1000': [35., -38.],
                            'v_950': [55., -56.],
                            'v_900': [58., -59.],
                            'v_850': [58., -59.]}


def expand_fronts(fronts: np.ndarray | tf.Tensor | xr.Dataset | xr.DataArray, iterations: int = 1):
    """
    Expands front labels in all directions.

    Parameters
    ----------
    fronts: array_like of ints of shape (T, M, N) or (M, N)
        2-D or 3-D array of integers that identify the front type at each point. The longitude and latitude dimensions with
            shapes (M,) and (N,) can be in any order, but the time dimension must be the first dimension if it is passed.
    iterations: int
        Integer representing the number of times to expand the fronts in all directions.

    Returns
    -------
    fronts: array_like of ints of shape (T, M, N) or (1, M, N)
        Array of integers for the expanded fronts. If the array_like object passed into the function was 2-D, a third dimension
            will be added to the beginning of the array with size 1.

    Examples
    --------
    * Expanding labels for one front type.
    >>> arr = np.zeros((5, 5))
    >>> arr[2, 2] = 1  # add cold front point
    >>> arr
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])
    >>> expand_fronts(arr, iterations=1)
    array([[[0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 1., 1., 1., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0.]]])

    * Expanding labels for two front types.
    >>> arr = np.zeros((5, 5))
    >>> arr[1, 1] = 1  # add cold front point
    >>> arr[3, 3] = 2  # add warm front point
    >>> arr
    array([[0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 2., 0.],
           [0., 0., 0., 0., 0.]])
    >>> expand_fronts(arr, iterations=1)
    array([[[1., 1., 1., 0., 0.],
            [1., 1., 1., 0., 0.],
            [1., 1., 2., 2., 2.],
            [0., 0., 2., 2., 2.],
            [0., 0., 2., 2., 2.]]])
    """
    if type(fronts) in [xr.Dataset, xr.DataArray]:
        identifier = fronts['identifier'].values if type(fronts) == xr.Dataset else fronts.values
    elif tf.is_tensor(fronts):
        identifier = tf.expand_dims(fronts, axis=0) if len(fronts.shape) == 2 else fronts
    else:
        identifier = np.expand_dims(fronts, axis=0) if len(fronts.shape) == 2 else fronts

    if tf.is_tensor(identifier):
        for _ in range(iterations):
            # 8 tensors representing all directions for the front expansion
            identifier_up_left = tf.Variable(tf.zeros_like(identifier))
            identifier_up_right = tf.Variable(tf.zeros_like(identifier))
            identifier_down_left = tf.Variable(tf.zeros_like(identifier))
            identifier_down_right = tf.Variable(tf.zeros_like(identifier))
            identifier_up = tf.Variable(tf.zeros_like(identifier))
            identifier_down = tf.Variable(tf.zeros_like(identifier))
            identifier_left = tf.Variable(tf.zeros_like(identifier))
            identifier_right = tf.Variable(tf.zeros_like(identifier))

            identifier_down_left[:, 1:, :-1].assign(tf.where((identifier[:, :-1, 1:] > 0) & (identifier[:, 1:, :-1] == 0),
                                                             identifier[:, :-1, 1:], identifier[:, 1:, :-1]))
            identifier_down[:, 1:, :].assign(tf.where((identifier[:, :-1, :] > 0) & (identifier[:, 1:, :] == 0),
                                                      identifier[:, :-1, :], identifier[:, 1:, :]))
            identifier_down_right[:, 1:, 1:].assign(tf.where((identifier[:, :-1, :-1] > 0) & (identifier[:, 1:, 1:] == 0),
                                                             identifier[:, :-1, :-1], identifier[:, 1:, 1:]))
            identifier_up_left[:, :-1, :-1].assign(tf.where((identifier[:, 1:, 1:] > 0) & (identifier[:, :-1, :-1] == 0),
                                                            identifier[:, 1:, 1:], identifier[:, :-1, :-1]))
            identifier_up[:, :-1, :].assign(tf.where((identifier[:, 1:, :] > 0) & (identifier[:, :-1, :] == 0),
                                                     identifier[:, 1:, :], identifier[:, :-1, :]))
            identifier_up_right[:, :-1, 1:].assign(tf.where((identifier[:, 1:, :-1] > 0) & (identifier[:, :-1, 1:] == 0),
                                                            identifier[:, 1:, :-1], identifier[:, :-1, 1:]))
            identifier_left[:, :, :-1].assign(tf.where((identifier[:, :, 1:] > 0) & (identifier[:, :, :-1] == 0),
                                                       identifier[:, :, 1:], identifier[:, :, :-1]))
            identifier_right[:, :, 1:].assign(tf.where((identifier[:, :, :-1] > 0) & (identifier[:, :, 1:] == 0),
                                                       identifier[:, :, :-1], identifier[:, :, 1:]))

            identifier = tf.reduce_max([identifier_up_left, identifier_up, identifier_up_right,
                                        identifier_down_left, identifier_down, identifier_down_right,
                                        identifier_left, identifier_right], axis=0)

    else:
        for _ in range(iterations):
            # 8 arrays representing all directions for the front expansion
            identifier_up_left = np.zeros_like(identifier)
            identifier_up_right = np.zeros_like(identifier)
            identifier_down_left = np.zeros_like(identifier)
            identifier_down_right = np.zeros_like(identifier)
            identifier_up = np.zeros_like(identifier)
            identifier_down = np.zeros_like(identifier)
            identifier_left = np.zeros_like(identifier)
            identifier_right = np.zeros_like(identifier)

            identifier_down_left[:, 1:, :-1] = np.where((identifier[:, :-1, 1:] > 0) & (identifier[:, 1:, :-1] == 0),
                                                        identifier[:, :-1, 1:], identifier[:, 1:, :-1])  # down-left
            identifier_down[:, 1:, :] = np.where((identifier[:, :-1, :] > 0) & (identifier[:, 1:, :] == 0),
                                                 identifier[:, :-1, :], identifier[:, 1:, :])  # down
            identifier_down_right[:, 1:, 1:] = np.where((identifier[:, :-1, :-1] > 0) & (identifier[:, 1:, 1:] == 0),
                                                        identifier[:, :-1, :-1], identifier[:, 1:, 1:])  # down-right
            identifier_up_left[:, :-1, :-1] = np.where((identifier[:, 1:, 1:] > 0) & (identifier[:, :-1, :-1] == 0),
                                                       identifier[:, 1:, 1:], identifier[:, :-1, :-1])  # up-left
            identifier_up[:, :-1, :] = np.where((identifier[:, 1:, :] > 0) & (identifier[:, :-1, :] == 0),
                                                identifier[:, 1:, :], identifier[:, :-1, :])  # up
            identifier_up_right[:, :-1, 1:] = np.where((identifier[:, 1:, :-1] > 0) & (identifier[:, :-1, 1:] == 0),
                                                       identifier[:, 1:, :-1], identifier[:, :-1, 1:])  # up-right
            identifier_left[:, :, :-1] = np.where((identifier[:, :, 1:] > 0) & (identifier[:, :, :-1] == 0),
                                                  identifier[:, :, 1:], identifier[:, :, :-1])  # left
            identifier_right[:, :, 1:] = np.where((identifier[:, :, :-1] > 0) & (identifier[:, :, 1:] == 0),
                                                  identifier[:, :, :-1], identifier[:, :, 1:])  # right

            identifier = np.max([identifier_up_left, identifier_up, identifier_up_right,
                                 identifier_down_left, identifier_down, identifier_down_right,
                                 identifier_left, identifier_right], axis=0)

    if type(fronts) == xr.Dataset:
        fronts['identifier'].values = identifier
    elif type(fronts) == xr.DataArray:
        fronts.values = identifier
    else:
        fronts = identifier

    return fronts


def haversine(lon: np.ndarray | int | float,
              lat: np.ndarray | int | float):
    """
    Haversine formula. Transforms lon/lat points to an x/y cartesian plane.

    Parameters
    ----------
    lon: array_like of shape (N,), int, or float
        Longitude component of the point(s) expressed in degrees.
    lat: array_like of shape (N,), int, or float
        Latitude component of the point(s) expressed in degrees.

    Returns
    -------
    x: array_like of shape (N,) or float
        X component of the transformed points expressed in kilometers.
    y: array_like of shape (N,) or float
        Y component of the transformed points expressed in kilometers.

    Examples
    --------
    >>> lon = -95
    >>> lat = 35
    >>> x, y = haversine(lon, lat)
    >>> x, y
    (-10077.330945462296, 3892.875)

    >>> lon = np.arange(10, 80.1, 10)
    >>> lat = np.arange(10, 80.1, 10)
    >>> x, y = haversine(lon, lat)
    >>> x, y
    (array([1108.01755295, 2190.70484658, 3223.05300087, 4180.69246988,
           5040.20418066, 5779.42053216, 6377.71302882, 6816.26345487]), array([1112.25, 2224.5 , 3336.75, 4449.  , 5561.25, 6673.5 , 7785.75,
           8898.  ]))
    """
    C = 40041  # average circumference of earth in kilometers
    x = lon * C * np.cos(lat * np.pi / 360) / 360
    y = lat * C / 360
    return x, y


def reverse_haversine(x, y):
    """
    Reverse haversine formula. Transforms x/y cartesian coordinates to a lon/lat grid.

    Parameters
    ----------
    x: array_like of shape (N,), int, or float
        X component of the point(s) expressed in kilometers.
    y: array_like of shape (N,), int, or float
        Y component of the point(s) expressed in kilometers.

    Returns
    -------
    lon: array_like of shape (N,) or float
        Longitude component of the transformed point(s) expressed in degrees.
    lat: array_like of shape (N,) or float
        Latitude component of the transformed point(s) expressed in degrees.

    Examples
    --------
    Values pulled from haversine examples.

    >>> x = -10077.330945462296
    >>> y = 3892.875
    >>> lon, lat = reverse_haversine(x, y)
    >>> lon, lat
    (-95.0, 35.0)

    >>> x = np.array([1108.01755295, 2190.70484658, 3223.05300087, 4180.69246988, 5040.20418066, 5779.42053216, 6377.71302882, 6816.26345487])
    >>> y = np.array([1112.25, 2224.5, 3336.75, 4449., 5561.25, 6673.5, 7785.75, 8898.])
    >>> lon, lat = reverse_haversine(x, y)
    >>> lon, lat
    (array([10., 20., 30., 40., 50., 60., 70., 80.]), array([10., 20., 30., 40., 50., 60., 70., 80.]))
    """
    C = 40041  # average circumference of earth in kilometers
    lon = x * 360 / np.cos(y * np.pi / C) / C
    lat = y * 360 / C
    return lon, lat


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
    xy_vertices: Normalized MultiLineString that contains the interpolated coordinates of fronts in kilometers.

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


def reformat_fronts(fronts, front_types):
    """
    Reformat a front dataset, tensor, or array with a given set of front types.

    Parameters
    ----------
    front_types: str or list of strs
        Code(s) that determine how the dataset will be reformatted.
    fronts: xarray Dataset or DataArray, tensor, or np.ndarray
        Dataset containing the front data.
        '''
        Available options for individual front types (cannot be passed with any special codes):

        Code (class #): Front Type
        --------------------------
        CF (1): Cold front
        WF (2): Warm front
        SF (3): Stationary front
        OF (4): Occluded front
        CF-F (5): Cold front (forming)
        WF-F (6): Warm front (forming)
        SF-F (7): Stationary front (forming)
        OF-F (8): Occluded front (forming)
        CF-D (9): Cold front (dissipating)
        WF-D (10): Warm front (dissipating)
        SF-D (11): Stationary front (dissipating)
        OF-D (12): Occluded front (dissipating)
        INST (13): Instability axis
        TROF (14): Trough
        TT (15): Tropical Trough
        DL (16): Dryline


        Special codes (cannot be passed with any individual front codes):
        -----------------------------------------------------------------
        F_BIN (1 class): 1-4, but treat all front types as one type.
            (1): CF, WF, SF, OF

        MERGED-F (4 classes): 1-12, but treat forming and dissipating fronts as standard fronts.
            (1): CF, CF-F, CF-D
            (2): WF, WF-F, WF-D
            (3): SF, SF-F, SF-D
            (4): OF, OF-F, OF-D

        MERGED-F_BIN (1 class): 1-12, but treat all front types and stages as one type. This means that classes 1-12 will all be one class (1).
            (1): CF, CF-F, CF-D, WF, WF-F, WF-D, SF, SF-F, SF-D, OF, OF-F, OF-D

        MERGED-T (1 class): 14-15, but treat troughs and tropical troughs as the same. In other words, TT (15) becomes TROF (14).
            (1): TROF, TT

        MERGED-ALL (7 classes): 1-16, but make the changes in the MERGED-F and MERGED-T codes.
            (1): CF, CF-F, CF-D
            (2): WF, WF-F, WF-D
            (3): SF, SF-F, SF-D
            (4): OF, OF-F, OF-D
            (5): TROF, TT
            (6): INST
            (7): DL

        **** NOTE - Class 0 is always treated as 'no front'.
        '''

    Returns
    -------
    fronts_ds: xr.Dataset
        Reformatted dataset based on the provided code(s).
    """

    if type(front_types) == str:
        front_types = [front_types, ]

    fronts_argument_type = type(fronts)

    if fronts_argument_type == xr.DataArray or fronts_argument_type == xr.Dataset:
        where_function = xr.where
    elif fronts_argument_type == np.ndarray:
        where_function = np.where
    else:
        where_function = tf.where

    front_types_classes = {'CF': 1, 'WF': 2, 'SF': 3, 'OF': 4, 'CF-F': 5, 'WF-F': 6, 'SF-F': 7, 'OF-F': 8, 'CF-D': 9, 'WF-D': 10,
                           'SF-D': 11, 'OF-D': 12, 'INST': 13, 'TROF': 14, 'TT': 15, 'DL': 16}

    if front_types == ['F_BIN', ]:

        fronts = where_function(fronts > 4, 0, fronts)  # Classes 5-16 are removed
        fronts = where_function(fronts > 0, 1, fronts)  # Merge 1-4 into one class

        labels = ['CF-WF-SF-OF', ]
        num_types = 1

    elif front_types == ['MERGED-F']:

        fronts = where_function(fronts == 5, 1, fronts)  # Forming cold front ---> cold front
        fronts = where_function(fronts == 6, 2, fronts)  # Forming warm front ---> warm front
        fronts = where_function(fronts == 7, 3, fronts)  # Forming stationary front ---> stationary front
        fronts = where_function(fronts == 8, 4, fronts)  # Forming occluded front ---> occluded front
        fronts = where_function(fronts == 9, 1, fronts)  # Dying cold front ---> cold front
        fronts = where_function(fronts == 10, 2, fronts)  # Dying warm front ---> warm front
        fronts = where_function(fronts == 11, 3, fronts)  # Dying stationary front ---> stationary front
        fronts = where_function(fronts == 12, 4, fronts)  # Dying occluded front ---> occluded front
        fronts = where_function(fronts > 4, 0, fronts)  # Remove all other fronts

        labels = ['CF_any', 'WF_any', 'SF_any', 'OF_any']
        num_types = 4

    elif front_types == ['MERGED-F_BIN']:

        fronts = where_function(fronts > 12, 0, fronts)  # Classes 13-16 are removed
        fronts = where_function(fronts > 0, 1, fronts)  # Classes 1-12 are merged into one class

        labels = ['CF-WF-SF-OF_any', ]
        num_types = 1

    elif front_types == ['MERGED-T']:

        fronts = where_function(fronts < 14, 0, fronts)  # Remove classes 1-13

        # Merge troughs into one class
        fronts = where_function(fronts == 14, 1, fronts)
        fronts = where_function(fronts == 15, 1, fronts)

        fronts = where_function(fronts == 16, 0, fronts)  # Remove drylines

        labels = ['TR_any', ]
        num_types = 1

    elif front_types == ['MERGED-ALL']:

        fronts = where_function(fronts == 5, 1, fronts)  # Forming cold front ---> cold front
        fronts = where_function(fronts == 6, 2, fronts)  # Forming warm front ---> warm front
        fronts = where_function(fronts == 7, 3, fronts)  # Forming stationary front ---> stationary front
        fronts = where_function(fronts == 8, 4, fronts)  # Forming occluded front ---> occluded front
        fronts = where_function(fronts == 9, 1, fronts)  # Dying cold front ---> cold front
        fronts = where_function(fronts == 10, 2, fronts)  # Dying warm front ---> warm front
        fronts = where_function(fronts == 11, 3, fronts)  # Dying stationary front ---> stationary front
        fronts = where_function(fronts == 12, 4, fronts)  # Dying occluded front ---> occluded front

        # Merge troughs together into class 5
        fronts = where_function(fronts == 14, 5, fronts)
        fronts = where_function(fronts == 15, 5, fronts)

        fronts = where_function(fronts == 13, 6, fronts)  # Move outflow boundaries to class 6
        fronts = where_function(fronts == 16, 7, fronts)  # Move drylines to class 7

        labels = ['CF_any', 'WF_any', 'SF_any', 'OF_any', 'TR_any', 'INST', 'DL']
        num_types = 7

    else:

        # Select the front types that are being used to pull their class identifiers
        filtered_front_types = dict(sorted(dict([(i, front_types_classes[i]) for i in front_types_classes if i in set(front_types)]).items(), key=lambda item: item[1]))
        front_types, num_types = list(filtered_front_types.keys()), len(filtered_front_types.keys())

        for i in range(num_types):
            if i + 1 != front_types_classes[front_types[i]]:
                fronts = where_function(fronts == i + 1, 0, fronts)
                fronts = where_function(fronts == front_types_classes[front_types[i]], i + 1, fronts)  # Reformat front classes

        fronts = where_function(fronts > num_types, 0, fronts)  # Remove unused front types

        labels = front_types

    if fronts_argument_type == xr.Dataset or fronts_argument_type == xr.DataArray:
        fronts.attrs['front_types'] = front_types
        fronts.attrs['num_types'] = num_types
        fronts.attrs['labels'] = labels

    return fronts


def normalize_variables(variable_ds, normalization_parameters=normalization_parameters):
    """
    Function that normalizes thermodynamic variables via min-max normalization.

    Parameters
    ----------
    variable_ds: xr.Dataset
        Dataset containing thermodynamic variable data.
    normalization_parameters: dict
        Dictionary containing parameters for normalization.

    Returns
    -------
    variable_ds: xr.Dataset
        Same as input dataset, but the variables are normalized via min-max normalization.
    """

    # Place pressure levels as the last dimension of the dataset
    original_dim_order = variable_ds.dims
    variable_ds = variable_ds.transpose(*[dim for dim in original_dim_order if dim != 'pressure_level'], 'pressure_level').astype('float64')

    variable_list = list(variable_ds.keys())
    pressure_levels = variable_ds['pressure_level'].values

    for var in variable_list:

        current_variable_values = variable_ds[var].values
        new_variable_values = np.zeros_like(current_variable_values)

        for idx, pressure_level in enumerate(pressure_levels):
            norm_var = '_'.join([var, pressure_level])  # name of the variable as it appears in the normalization parameters dictionary
            max_val, min_val = normalization_parameters[norm_var]
            new_variable_values[..., idx] = np.nan_to_num((variable_ds[var].values[..., idx] - min_val) / (max_val - min_val))

        variable_ds[var].values = new_variable_values  # assign new values for variable

    variable_ds = variable_ds.transpose(*[dim for dim in original_dim_order])
    return variable_ds


def randomize_variables(variable_ds: xr.Dataset, random_variables: list or tuple):
    """
    Scramble the values of specific variables within a given dataset.

    Parameters
    ----------
    variable_ds: xr.Dataset
        Xarray dataset containing thermodynamic variable data.
    random_variables: list or tuple
        List of variables to randomize the values of.

    Returns
    -------
    variable_ds: xr.Dataset
        Same as input, but with the given variables having scrambled values.
    """

    for random_variable in random_variables:
        variable_values = variable_ds[random_variable].values  # DataArray of the current variable
        variable_shape = np.shape(variable_values)
        flattened_variable_values = variable_values.flatten()
        np.random.shuffle(flattened_variable_values)
        variable_ds[random_variable].values = np.reshape(flattened_variable_values, variable_shape)

    return variable_ds


def combine_datasets(input_files: list[str],
                     label_files: list[str] = None):
    """
    Combine many tensorflow datasets into one entire dataset.

    Returns
    -------
    complete_dataset: tf.data.Dataset object
        Concatenated tensorflow dataset.
    """
    inputs = tf.data.Dataset.load(input_files[0])

    if label_files is not None:

        labels = tf.data.Dataset.load(label_files[0])
        for input_file, label_file in zip(input_files[1:], label_files[1:]):
            inputs = inputs.concatenate(tf.data.Dataset.load(input_file))
            labels = labels.concatenate(tf.data.Dataset.load(label_file))

        return tf.data.Dataset.zip((inputs, labels))

    else:

        for input_file in input_files[1:]:
            inputs = inputs.concatenate(tf.data.Dataset.load(input_file))

        return inputs


def lambert_conformal_to_cartesian(
        lon: np.ndarray | tuple | list | int | float,
        lat: np.ndarray | tuple | list | int | float,
        std_parallels: tuple | list = (20., 50.),
        lon_ref: int | float = 0.,
        lat_ref: int | float = 0.):
    """
    Transform points on a Lambert Conformal lat/lon grid to cartesian coordinates.

    Parameters
    ----------
    lon: array_like of shape (N,), int, or float
        Longitude point(s) expressed as degrees.
    lat: array_like of shape (N,), int, or float
        Latitude point(s) expressed as degrees.
    std_parallels: tuple or list of 2 ints or floats
        Standard parallels to use in the coordinate transformation, expressed as degrees.
    lon_ref: int or float
        Reference longitude point expressed as degrees.
    lat_ref: int or float
        Reference latitude point expressed as degrees.

    Returns
    -------
    x: array_like of shape (N,) or float
        X-component of the transformed coordinates, expressed as meters.
    y: array_like of shape (N,) or float
        Y-component of the transformed coordinates, expressed as meters.

    Examples
    --------
    * Using parameters from example on Page 295 of Snyder 1987 (except the output here is expressed as meters):
    >>> x, y = lambert_conformal_to_cartesian(lon=-75, lat=35, std_parallels=(33, 45), lon_ref=-96, lat_ref=23)
    >>> x, y
    (1890206.4076610378, 1568668.1244433122)

    * Same as above but with longitudes expressed from 0 to 360 degrees east:
    >>> x, y = lambert_conformal_to_cartesian(lon=285, lat=35, std_parallels=(33, 45), lon_ref=264, lat_ref=23)
    >>> x, y
    (1890206.4076610343, 1568668.1244433112)

    References
    ----------
    * Snyder 1987: https://doi.org/10.3133/pp1395

    Notes
    -----
    lon and lon_ref must be both expressed in the same longitude range (e.g. -180 to 180 degrees or 0 to 360 degrees)
        to get correct values for x and y.
    """

    R = 6371229  # radius of earth (meters)

    # Points and standard parallels need to be expressed as radians for the transformation formulas
    lon = np.radians(lon)
    lon_ref = np.radians(lon_ref)
    lat = np.radians(lat)
    lat_ref = np.radians(lat_ref)
    std_parallels = np.radians(std_parallels)

    if std_parallels[0] == std_parallels[1]:
        n = np.sin(std_parallels[0])
    else:
        n = np.divide(np.log(np.cos(std_parallels[0]) / np.cos(std_parallels[1])),
                      np.log(np.tan(np.pi/4 + std_parallels[1]/2) / np.tan(np.pi/4 + std_parallels[0]/2)))
    F = np.cos(std_parallels[0]) * np.power(np.tan(np.pi/4 + std_parallels[0]/2), n) / n
    rho = R * F / np.power(np.tan(np.pi/4 + lat/2), n)
    rho0 = R * F / np.power(np.tan(np.pi/4 + lat_ref/2), n)

    x = rho * np.sin(n * (lon - lon_ref))
    y = rho0 - rho * np.cos(n * (lon - lon_ref))

    return x, y
