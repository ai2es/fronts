"""
Data tools

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 3/3/2023 1:23 PM CT
"""

import math
import pandas as pd
from shapely.geometry import LineString
import numpy as np
import xarray as xr
import tensorflow as tf


# Each variable has parameters in the format of [max, min, mean]
normalization_parameters = {'T_surface': [326.3396, 192.2074, 278.8511],
                            'T_1000': [309.2406, 205.7583, 280.1311],
                            'T_950': [309.2406, 205.7583, 278.2461],
                            'T_900': [309.2406, 205.7583, 276.6009],
                            'T_850': [309.2406, 205.7583, 274.6106],
                            'Td_surface': [305.8017, 189.1589, 274.2648],
                            'Td_1000': [313.5051, 170.2352, 281.9077],
                            'Td_950': [311.9054, 169.9939, 280.1014],
                            'Td_900': [305.5049, 169.7402, 277.0051],
                            'Td_850': [299.1380, 169.4729, 273.5636],
                            'r_surface': [30.0701, 0.0007, 4.2927],
                            'r_1000': [50.6989, 0.0000, 7.1011],
                            'r_950': [48.8387, 0.0000, 6.6044],
                            'r_900': [35.4729, 0.0000, 5.6135],
                            'r_850': [25.5920, 0.0000, 4.6430],
                            'RH_surface': [1.07, 0, 0.70],
                            'RH_1000': [1.07, 0, 0.70],
                            'RH_950': [1.07, 0, 0.70],
                            'RH_900': [1.07, 0, 0.70],
                            'RH_850': [1.07, 0, 0.70],
                            'sp_z_surface': [1070.7823, 473.9962, 966.5046],
                            'sp_z_1000': [178.6779, 0.0000, 7.5059],
                            'sp_z_950': [91.7832, 0.0000, 49.7395],
                            'sp_z_900': [135.1058, 0.0000, 93.9318],
                            'sp_z_850': [179.5943, 0.0000, 140.3395],
                            'mslp_z_surface': [1070.7823, 473.9962, 966.5046],
                            'mslp_z_1000': [178.6779, 0.0000, 7.5059],
                            'mslp_z_950': [91.7832, 0.0000, 49.7395],
                            'mslp_z_900': [135.1058, 0.0000, 93.9318],
                            'mslp_z_850': [179.5943, 0.0000, 140.3395],
                            'theta_e_surface': [776.7833, 188.4853, 293.6940],
                            'theta_e_1000': [475.4648, 205.7584, 299.5906],
                            'theta_e_950': [476.3312, 208.7991, 300.6901],
                            'theta_e_900': [435.0011, 212.0529, 300.9420],
                            'theta_e_850': [407.9803, 215.5478, 301.0762],
                            'theta_w_surface': [326.1300, 188.4853, 278.3979],
                            'theta_w_1000': [312.9144, 205.7546, 281.0808],
                            'theta_w_950': [312.9837, 208.7910, 281.5534],
                            'theta_w_900': [309.2340, 212.0367, 281.6605],
                            'theta_w_850': [306.1498, 215.5172, 281.7173],
                            'u_surface': [44.8457, -196.7885, -0.0675],
                            'u_1000': [62.2232, -165.2770, -0.0509],
                            'u_950': [62.4603, -148.5150, 0.3758],
                            'u_900': [89.8531, -165.2056, 0.8397],
                            'u_850': [89.8112, -165.1002, 1.3851],
                            'v_surface': [135.2455, -96.9072, 0.1984],
                            'v_1000': [76.6498, -58.4051, 0.1974],
                            'v_950': [91.0734, -66.1152, 0.2073],
                            'v_900': [91.1151, -64.6468, 0.2014],
                            'v_850': [90.9507, -64.6207, 0.1485],
                            'Tv_surface': [339.7172, 192.2074, 279.5754],
                            'Tv_1000': [318.3087, 205.7583, 281.3315],
                            'Tv_950': [317.9915, 205.7583, 279.3555],
                            'Tv_900': [315.6787, 205.7583, 277.5392],
                            'Tv_850': [313.9301, 205.7583, 275.3819],
                            'Tw_surface': [310.4736, 194.3195, 276.3030],
                            'Tw_1000': [313.3936, 239.7402, 281.3283],
                            'Tw_950': [311.8010, 239.7592, 279.4636],
                            'Tw_900': [306.3249, 239.7782, 276.7549],
                            'Tw_850': [301.8944, 239.7972, 273.8807],
                            'q_surface': [67.4536, 0.0000, 4.2743],
                            'q_1000': [48.1637, 0.0000, 7.0572],
                            'q_950': [46.4895, 0.0000, 6.5669],
                            'q_900': [34.2312, 0.0000, 5.5869],
                            'q_850': [24.9512, 0.0000, 4.6250]}


def expand_fronts(ds_fronts, iterations=1):
    """
    Expands fronts in all directions.

    Parameters
    ----------
    ds_fronts: xr.Dataset
        - Dataset that contains frontal objects.
    iterations: int
        - Number of pixels that the fronts will be expanded by in all directions, determined by the number of iterations across this expansion method.

    Returns
    -------
    ds_fronts: xr.Dataset
        Dataset that contains expanded frontal objects.
    """
    try:
        identifier = ds_fronts['identifier'].values
    except KeyError:
        identifier = ds_fronts.values

    identifier_temp = identifier
    identifier = identifier_temp

    num_dims = len(np.shape(identifier))

    if num_dims == 2:
        identifier = np.expand_dims(identifier, axis=0)

    max_lat_index = np.shape(identifier)[num_dims - 2] - 1
    max_lon_index = np.shape(identifier)[num_dims - 1] - 1
    
    for iteration in range(iterations):
        
        # Indices where fronts are located
        indices = np.where(identifier != 0)
        num_indices = np.shape(indices)[-1]

        timestep_ind = indices[0]
        lats_ind = indices[1]
        lons_ind = indices[2]

        for index in range(num_indices):

            timestep, lat, lon = timestep_ind[index], lats_ind[index], lons_ind[index]
            front_value = identifier[timestep, lat, lon]

            if lat == 0:
                # If the front pixel is at the north end of the domain (max lat), and there is no front directly
                # to the south, expand the front 1 pixel south.
                if identifier[timestep][lat][lon] == 0:
                    identifier[timestep][lat][lon] = front_value
                # If the front pixel is at the northwest end of the domain (max/min lat/lon), check the pixels to the
                # southeast and east for fronts, and expand the front there if no other fronts are already present.
                if lon == 0:
                    if identifier[timestep][lat + 1][lon + 1] == 0:
                        identifier[timestep][lat + 1][lon + 1] = front_value
                    if identifier[timestep][lat][lon + 1] == 0:
                        identifier[timestep][lat][lon + 1] = front_value
                # If the front pixel is at the northeast end of the domain (max/max lat/lon), check the pixels to the
                # southwest and west for fronts, and expand the front there if no other fronts are already present.
                elif lon == max_lon_index:
                    if identifier[timestep][lat + 1][lon - 1] == 0:
                        identifier[timestep][lat + 1][lon - 1] = front_value
                    if identifier[timestep][lat][lon - 1] == 0:
                        identifier[timestep][lat][lon - 1] = front_value
                # If the front pixel is at the north end of the domain (max lat), but not at the west or east end (min lon
                # or max lon) check the pixels to the west and east for fronts, and expand the front there if no other
                # fronts are already present.
                else:
                    if identifier[timestep][lat][lon - 1] == 0:
                        identifier[timestep][lat][lon - 1] = front_value
                    if identifier[timestep][lat][lon + 1] == 0:
                        identifier[timestep][lat][lon + 1] = front_value
            elif 0 < lat < max_lat_index:
                # If there is no front directly to the south, expand the front 1 pixel south.
                if identifier[timestep][lat + 1][lon] == 0:
                    identifier[timestep][lat + 1][lon] = front_value
                # If there is no front directly to the north, expand the front 1 pixel north.
                if identifier[timestep][lat - 1][lon] == 0:
                    identifier[timestep][lat - 1][lon] = front_value
                # If the front pixel is at the west end of the domain (min lon), check the pixels to the southeast,
                # east, and northeast for fronts, and expand the front there if no other fronts are already present.
                if lon == 0:
                    if identifier[timestep][lat + 1][lon + 1] == 0:
                        identifier[timestep][lat + 1][lon + 1] = front_value
                    if identifier[timestep][lat][lon + 1] == 0:
                        identifier[timestep][lat][lon + 1] = front_value
                    if identifier[timestep][lat - 1][lon + 1] == 0:
                        identifier[timestep][lat - 1][lon + 1] = front_value
                # If the front pixel is at the east end of the domain (min lon), check the pixels to the southwest,
                # west, and northwest for fronts, and expand the front there if no other fronts are already present.
                elif lon == max_lon_index:
                    if identifier[timestep][lat + 1][lon - 1] == 0:
                        identifier[timestep][lat + 1][lon - 1] = front_value
                    if identifier[timestep][lat][lon - 1] == 0:
                        identifier[timestep][lat][lon - 1] = front_value
                    if identifier[timestep][lat - 1][lon - 1] == 0:
                        identifier[timestep][lat - 1][lon - 1] = front_value
                # If the front pixel is not at the end of the domain in any direction, check the northeast, east,
                # southeast, northwest, west, and southwest for fronts, and expand the front there if no other fronts
                # are already present.
                else:
                    if identifier[timestep][lat + 1][lon + 1] == 0:
                        identifier[timestep][lat + 1][lon + 1] = front_value
                    if identifier[timestep][lat][lon + 1] == 0:
                        identifier[timestep][lat][lon + 1] = front_value
                    if identifier[timestep][lat - 1][lon + 1] == 0:
                        identifier[timestep][lat - 1][lon + 1] = front_value
                    if identifier[timestep][lat + 1][lon - 1] == 0:
                        identifier[timestep][lat + 1][lon - 1] = front_value
                    if identifier[timestep][lat][lon - 1] == 0:
                        identifier[timestep][lat][lon - 1] = front_value
                    if identifier[timestep][lat - 1][lon - 1] == 0:
                        identifier[timestep][lat - 1][lon - 1] = front_value
            else:
                # If the front pixel is at the south end of the domain (max lat), and there is no front directly
                # to the north, expand the front 1 pixel north.
                if identifier[timestep][lat - 1][lon] == 0:
                    identifier[timestep][lat - 1][lon] = front_value
                # If the front pixel is at the southwest end of the domain (max/min lat/lon), check the pixels to the
                # northeast and east for fronts, and expand the front there if no other fronts are already present.
                if lon == 0:
                    if identifier[timestep][lat - 1][lon + 1] == 0:
                        identifier[timestep][lat - 1][lon + 1] = front_value
                    if identifier[timestep][lat][lon + 1] == 0:
                        identifier[timestep][lat][lon + 1] = front_value
                # If the front pixel is at the southeast end of the domain (max/max lat/lon), check the pixels to the
                # northwest and west for fronts, and expand the front there if no other fronts are already present.
                elif lon == max_lon_index:
                    if identifier[timestep][lat - 1][lon - 1] == 0:
                        identifier[timestep][lat - 1][lon - 1] = front_value
                    if identifier[timestep][lat][lon - 1] == 0:
                        identifier[timestep][lat][lon - 1] = front_value
                # If the front pixel is at the south end of the domain (max lat), but not at the west or east end (min lon
                # or max lon) check the pixels to the west and east for fronts, and expand the front there if no other
                # fronts are already present.
                else:
                    if identifier[timestep][lat][lon - 1] == 0:
                        identifier[timestep][lat][lon - 1] = front_value
                    if identifier[timestep][lat][lon + 1] == 0:
                        identifier[timestep][lat][lon + 1] = front_value

    if num_dims == 2:
        identifier = identifier[0]

    try:
        ds_fronts['identifier'].values = identifier
    except KeyError:
        ds_fronts.values = identifier

    return ds_fronts


def haversine(lons, lats):
    """
    lat/lon ----> cartesian coordinates (km)
    """
    if type(lons) == list:
        lons = np.array(lons)
    if type(lats) == list:
        lats = np.array(lats)

    xs = lons * 40075 * np.cos(lats * math.pi / 360) / 360  # circumference of earth in km = 40075
    ys = lats * 40075 / 360
    return xs, ys


def reverse_haversine(xs, ys):
    """
    Cartesian coordinates (km) ---> lat/lon
    """
    if type(xs) == list:
        xs = np.array(xs)
    if type(ys) == list:
        ys = np.array(ys)

    lons = xs * 360 / np.cos(ys * math.pi / 40075) / 40075
    lats = ys * 360 / 40075
    return lons, lats


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
        - Code(s) that determine how the dataset will be reformatted.
    fronts: xarray Dataset or DataArray, tensor, or np.ndarray
        - Dataset containing the front data.
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
        OFB (13): Outflow boundary
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
            (6): OFB
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
                           'SF-D': 11, 'OF-D': 12, 'OFB': 13, 'TROF': 14, 'TT': 15, 'DL': 16}

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

        labels = ['CF_any', 'WF_any', 'SF_any', 'OF_any', 'TR_any', 'OFB', 'DL']
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


def add_or_subtract_hours_to_timestep(timestep, num_hours):
    """
    Find the valid timestep for a given number of added or subtracted hours.
    
    Parameters
    ----------
    timestep: int, str, tuple
        * Int format for the timestep: YYYYMMDDHH
        * String format for the timestep: YYYY-MM-DD-HH (no dashes)
        * Tuple format for the timestep (all integers): (YYYY, MM, DD, HH)
    num_hours: int
        Number of hours to add or subtract to the new timestep.
    
    Returns
    -------
    New timestep after adding or subtracting the given number of hours. This new timestep will be returned to the same format
    as the input timestep.
    """

    timestep_type = type(timestep)

    if timestep_type == str or timestep_type == np.str_:
        year = int(timestep[:4])
        month = int(timestep[4:6])
        day = int(timestep[6:8])
        hour = int(timestep[8:])
    elif timestep_type == tuple:
        if all(type(timestep_tuple_element) == int for timestep_tuple_element in timestep):
            timestep = tuple([str(timestep_element) for timestep_element in timestep])
        year, month, day, hour = timestep[0], timestep[1], timestep[2], timestep[3]
    else:
        raise TypeError("Timestep must be a string or a tuple with integers in one of the following formats: YYYYMMDDHH or (YYYY, MM, DD, HH)")

    if year % 4 == 0:  # Check if the current year is a leap year
        month_2_days = 29
    else:
        month_2_days = 28

    days_per_month = [31, month_2_days, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    new_year, new_month, new_day, new_hour = year, month, day, hour + num_hours
    if new_hour > 0:
        # If the new timestep is in a future day, this loop will be activated
        while new_hour > 23:
            new_hour -= 24
            new_day += 1

        # If the new timestep is in a future month, this loop will be activated
        while new_day > days_per_month[new_month - 1]:
            new_day -= days_per_month[new_month - 1]
            if new_month < 12:
                new_month += 1
            else:
                new_month = 1
                new_year += 1
    else:
        # If the new timestep is in a past day, this loop will be activated
        while new_hour < 0:
            new_hour += 24
            new_day -= 1

        # If the new timestep is in a past month, this loop will be activated
        while new_day < 1:
            new_day += days_per_month[new_month - 2]
            if new_month > 1:
                new_month -= 1
            else:
                new_month = 12
                new_year -= 1

    return '%d%02d%02d%02d' % (new_year, new_month, new_day, new_hour)


def normalize_variables(variable_ds, normalization_parameters=normalization_parameters):
    """
    Function that normalizes GDAS variables via min-max normalization.

    Parameters
    ----------
    variable_ds: xr.Dataset
        - Dataset containing ERA5, GDAS, or GFS variable data.
    normalization_parameters: dict
        - Dictionary containing parameters for normalization.

    Returns
    -------
    variable_ds: xr.Dataset
        - Same as input dataset, but the variables are normalized via min-max normalization.
    """

    variable_list = list(variable_ds.keys())
    pressure_levels = variable_ds['pressure_level'].values

    new_var_shape = np.shape(variable_ds[variable_list[0]].values)

    for var in variable_list:

        new_values_for_variable = np.empty(shape=new_var_shape)

        for pressure_level_index in range(len(pressure_levels)):

            norm_var = '%s_%s' % (var, pressure_levels[pressure_level_index])

            # Min-max normalization
            if len(np.shape(new_values_for_variable)) == 4:
                new_values_for_variable[:, :, :, pressure_level_index] = np.nan_to_num((variable_ds[var].values[:, :, :, pressure_level_index] - normalization_parameters[norm_var][1]) /
                                                                                       (normalization_parameters[norm_var][0] - normalization_parameters[norm_var][1]))

            elif len(np.shape(new_values_for_variable)) == 5:  # If forecast hours are in the dataset
                new_values_for_variable[:, :, :, :, pressure_level_index] = np.nan_to_num((variable_ds[var].values[:, :, :, :, pressure_level_index] - normalization_parameters[norm_var][1]) /
                                                                                          (normalization_parameters[norm_var][0] - normalization_parameters[norm_var][1]))

        variable_ds[var].values = new_values_for_variable

    return variable_ds


def randomize_variables(variable_ds: xr.Dataset, random_variables: list or tuple):
    """
    Scramble the values of specific variables within a given dataset.

    Parameters
    ----------
    variable_ds: xr.Dataset
        - ERA5 or GDAS variable dataset.
    random_variables: list or tuple
        - List of variables to randomize the values of.

    Returns
    -------
    variable_ds: xr.Dataset
        - Same as input, but with the given variables having scrambled values.
    """

    for random_variable in random_variables:
        variable_values = variable_ds[random_variable].values  # DataArray of the current variable
        variable_shape = np.shape(variable_values)
        flattened_variable_values = variable_values.flatten()
        np.random.shuffle(flattened_variable_values)
        variable_ds[random_variable].values = np.reshape(flattened_variable_values, variable_shape)

    return variable_ds
