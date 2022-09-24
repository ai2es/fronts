"""
Data tools

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 9/24/2022 12:31 PM CT
"""

import math
import pandas as pd
from shapely.geometry import LineString
import numpy as np
import xarray as xr
import file_manager as fm


def expand_fronts(ds_fronts, iterations=1):
    """
    Expands fronts by 1 pixel in all directions.

    Parameters
    ----------
    ds_fronts: xr.Dataset
        Dataset that contains frontal objects.
    iterations: int
        Number of pixels that the fronts will be expanded by in all directions, determined by the number of iterations across this expansion method.

    Returns
    -------
    ds_fronts: xr.Dataset
        Dataset that contains expanded frontal objects.
    """
    lats = ds_fronts.latitude.values
    lons = ds_fronts.longitude.values
    len_lats = len(lats)
    len_lons = len(lons)
    
    variable_name = list(ds_fronts.keys())[0]
    identifier = ds_fronts[variable_name].values

    for iteration in range(iterations):
        indices = np.where(identifier != 0)
        for i in range(len(indices[0])):
            front_value = identifier[indices[0][i]][indices[1][i]]
            if lats[indices[0][i]] == lats[0]:
                # If the front pixel is at the north end of the domain (max lat), and there is no front directly
                # to the south, expand the front 1 pixel south.
                if identifier[indices[0][i]][indices[1][i]] == 0:
                    identifier[indices[0][i]][indices[1][i]] = front_value
                # If the front pixel is at the northwest end of the domain (max/min lat/lon), check the pixels to the
                # southeast and east for fronts, and expand the front there if no other fronts are already present.
                if lons[indices[1][i]] == lons[0]:
                    if identifier[indices[0][i] + 1][indices[1][i] + 1] == 0:
                        identifier[indices[0][i] + 1][indices[1][i] + 1] = front_value
                    if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                        identifier[indices[0][i]][indices[1][i] + 1] = front_value
                # If the front pixel is at the northeast end of the domain (max/max lat/lon), check the pixels to the
                # southwest and west for fronts, and expand the front there if no other fronts are already present.
                elif lons[indices[1][i]] == lons[len_lons - 1]:
                    if identifier[indices[0][i] + 1][indices[1][i] - 1] == 0:
                        identifier[indices[0][i] + 1][indices[1][i] - 1] = front_value
                    if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                        identifier[indices[0][i]][indices[1][i] - 1] = front_value
                # If the front pixel is at the north end of the domain (max lat), but not at the west or east end (min lon
                # or max lon) check the pixels to the west and east for fronts, and expand the front there if no other
                # fronts are already present.
                else:
                    if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                        identifier[indices[0][i]][indices[1][i] - 1] = front_value
                    if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                        identifier[indices[0][i]][indices[1][i] + 1] = front_value
            elif lats[-1] < lats[indices[0][i]] < lats[len_lats - 1]:
                # If there is no front directly to the south, expand the front 1 pixel south.
                if identifier[indices[0][i] + 1][indices[1][i]] == 0:
                    identifier[indices[0][i]][indices[1][i]] = front_value
                # If there is no front directly to the north, expand the front 1 pixel north.
                if identifier[indices[0][i] - 1][indices[1][i]] == 0:
                    identifier[indices[0][i] - 1][indices[1][i]] = front_value
                # If the front pixel is at the west end of the domain (min lon), check the pixels to the southeast,
                # east, and northeast for fronts, and expand the front there if no other fronts are already present.
                if lons[indices[1][i]] == lons[0]:
                    if identifier[indices[0][i] + 1][indices[1][i] + 1] == 0:
                        identifier[indices[0][i] + 1][indices[1][i] + 1] = front_value
                    if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                        identifier[indices[0][i]][indices[1][i] + 1] = front_value
                    if identifier[indices[0][i] - 1][indices[1][i] + 1] == 0:
                        identifier[indices[0][i] - 1][indices[1][i] + 1] = front_value
                # If the front pixel is at the east end of the domain (min lon), check the pixels to the southwest,
                # west, and northwest for fronts, and expand the front there if no other fronts are already present.
                elif lons[indices[1][i]] == lons[len_lons - 1]:
                    if identifier[indices[0][i] + 1][indices[1][i] - 1] == 0:
                        identifier[indices[0][i] + 1][indices[1][i] - 1] = front_value
                    if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                        identifier[indices[0][i]][indices[1][i] - 1] = front_value
                    if identifier[indices[0][i] - 1][indices[1][i] - 1] == 0:
                        identifier[indices[0][i] - 1][indices[1][i] - 1] = front_value
                # If the front pixel is not at the end of the domain in any direction, check the northeast, east,
                # southeast, northwest, west, and southwest for fronts, and expand the front there if no other fronts
                # are already present.
                else:
                    if identifier[indices[0][i] + 1][indices[1][i] + 1] == 0:
                        identifier[indices[0][i] + 1][indices[1][i] + 1] = front_value
                    if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                        identifier[indices[0][i]][indices[1][i] + 1] = front_value
                    if identifier[indices[0][i] - 1][indices[1][i] + 1] == 0:
                        identifier[indices[0][i] - 1][indices[1][i] + 1] = front_value
                    if identifier[indices[0][i] + 1][indices[1][i] - 1] == 0:
                        identifier[indices[0][i] + 1][indices[1][i] - 1] = front_value
                    if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                        identifier[indices[0][i]][indices[1][i] - 1] = front_value
                    if identifier[indices[0][i] - 1][indices[1][i] - 1] == 0:
                        identifier[indices[0][i] - 1][indices[1][i] - 1] = front_value
            else:
                # If the front pixel is at the south end of the domain (max lat), and there is no front directly
                # to the north, expand the front 1 pixel north.
                if identifier[indices[0][i] - 1][indices[1][i]] == 0:
                    identifier[indices[0][i] - 1][indices[1][i]] = front_value
                # If the front pixel is at the southwest end of the domain (max/min lat/lon), check the pixels to the
                # northeast and east for fronts, and expand the front there if no other fronts are already present.
                if lons[indices[1][i]] == lons[0]:
                    if identifier[indices[0][i] - 1][indices[1][i] + 1] == 0:
                        identifier[indices[0][i] - 1][indices[1][i] + 1] = front_value
                    if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                        identifier[indices[0][i]][indices[1][i] + 1] = front_value
                # If the front pixel is at the southeast end of the domain (max/max lat/lon), check the pixels to the
                # northwest and west for fronts, and expand the front there if no other fronts are already present.
                elif lons[indices[1][i]] == lons[len_lons - 1]:
                    if identifier[indices[0][i] - 1][indices[1][i] - 1] == 0:
                        identifier[indices[0][i] - 1][indices[1][i] - 1] = front_value
                    if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                        identifier[indices[0][i]][indices[1][i] - 1] = front_value
                # If the front pixel is at the south end of the domain (max lat), but not at the west or east end (min lon
                # or max lon) check the pixels to the west and east for fronts, and expand the front there if no other
                # fronts are already present.
                else:
                    if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                        identifier[indices[0][i]][indices[1][i] - 1] = front_value
                    if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                        identifier[indices[0][i]][indices[1][i] + 1] = front_value

    ds_fronts[variable_name].values = identifier
    return ds_fronts


def haversine(lons, lats):
    """
    lat/lon ----> cartesian coordinates
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
    Cartesian coordinates ---> lat/lon
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


def reformat_fronts(front_types, fronts_ds=None, return_colors=False, return_names=False):
    """
    Reformat a frontal object dataset along with its respective colors and labels for plotting, or just return the colors and labels
    for the given front types.

    Parameters
    ----------
    front_types: str or list of strs
        - Code(s) that determine how the dataset or array will be reformatted.
    fronts_ds: xr.Dataset or np.array or None
        - Dataset containing the front data.
        - If left as None, no dataset will be returned.
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

        MERGED-F (4 classes): 1-12, but treat forming and dissipating fronts as a standard front.
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
    return_colors: bool
        Setting this to True will return a list of colors that can be used to plot the fronts.
    return_names: bool
        Setting this to True will return a list of names for the classes.

    Returns
    -------
    fronts_ds: xr.Dataset or np.array
        Reformatted dataset or array based on the provided code(s).
    labels: list
        List of labels for the front types that will be used to name the files.
    colors_types: list
        List of colors that can be used to plot the different front types or classes.
    colors_probs: list
        List of colormaps that are used to plot the model predictions for the different front types.
    names: list
        List of names for the front classes.
    """

    front_types_classes = {'CF': 1, 'WF': 2, 'SF': 3, 'OF': 4, 'CF-F': 5, 'WF-F': 6, 'SF-F': 7, 'OF-F': 8, 'CF-D': 9, 'WF-D': 10,
                           'SF-D': 11, 'OF-D': 12, 'OFB': 13, 'TROF': 14, 'TT': 15, 'DL': 16}

    all_color_types = ['blue', 'red', 'limegreen', 'darkviolet', 'darkblue', 'darkred', 'darkgreen', 'darkmagenta', 'lightskyblue',
                       'lightcoral', 'lightgreen', 'violet', 'gold', 'goldenrod', 'orange', 'chocolate']

    all_color_probs = ['Blues', 'Reds', 'Greens', 'Purples', 'Blues', 'Reds', 'Greens', 'Purples', 'Blues', 'Reds', 'Greens',
                       'Purples', 'YlOrBr', 'YlOrRed', 'Oranges', 'copper_r']

    all_names = ['Cold front', 'Warm front', 'Stationary front', 'Occluded front', 'Cold front (forming)', 'Warm front (forming)',
                 'Stationary front (forming)', 'Occluded front (forming)', 'Cold front (dying)', 'Warm front (dying)', 'Stationary front (dying)',
                 'Occluded front (dying)', 'Outflow boundary', 'Trough', 'Tropical trough', 'Dryline']

    if front_types == 'F_BIN':

        if fronts_ds is not None:
            fronts_ds = xr.where(fronts_ds > 4, 0, fronts_ds)  # Classes 5-16 are removed
            fronts_ds = xr.where(fronts_ds > 0, 1, fronts_ds)  # Merge 1-4 into one class

        colors_types = ['tab:red']
        colors_probs = ['Reds']
        names = ['CF, WF, SF, OF', ]
        labels = ['CF-WF-SF-OF', ]

    elif front_types == 'MERGED-F':

        if fronts_ds is not None:
            fronts_ds = xr.where(fronts_ds == 5, 1, fronts_ds)  # Forming cold front ---> cold front
            fronts_ds = xr.where(fronts_ds == 6, 2, fronts_ds)  # Forming warm front ---> warm front
            fronts_ds = xr.where(fronts_ds == 7, 3, fronts_ds)  # Forming stationary front ---> stationary front
            fronts_ds = xr.where(fronts_ds == 8, 4, fronts_ds)  # Forming occluded front ---> occluded front
            fronts_ds = xr.where(fronts_ds == 9, 1, fronts_ds)  # Dying cold front ---> cold front
            fronts_ds = xr.where(fronts_ds == 10, 2, fronts_ds)  # Dying warm front ---> warm front
            fronts_ds = xr.where(fronts_ds == 11, 3, fronts_ds)  # Dying stationary front ---> stationary front
            fronts_ds = xr.where(fronts_ds == 12, 4, fronts_ds)  # Dying occluded front ---> occluded front
            fronts_ds = xr.where(fronts_ds > 4, 0, fronts_ds)  # Remove all other fronts

        colors_types = ['blue', 'red', 'limegreen', 'darkviolet']
        colors_probs = ['Blues', 'Reds', 'Greens', 'Purples']
        names = ['Cold front (any)', 'Warm front (any)', 'Stationary front (any)', 'Occluded front (any)']
        labels = ['CF_any', 'WF_any', 'SF_any', 'OF_any']

    elif front_types == 'MERGED-F_BIN':

        if fronts_ds is not None:
            fronts_ds = xr.where(fronts_ds > 12, 0, fronts_ds)  # Classes 13-16 are removed
            fronts_ds = xr.where(fronts_ds > 0, 1, fronts_ds)  # Classes 1-12 are merged into one class

        colors_types = ['gray']
        colors_probs = ['Greys']
        names = ['CF, WF, SF, OF (any)', ]
        labels = ['CF-WF-SF-OF_any', ]

    elif front_types == 'MERGED-T':

        if fronts_ds is not None:
            fronts_ds = xr.where(fronts_ds < 14, 0, fronts_ds)  # Remove classes 1-13

            # Merge troughs into one class
            fronts_ds = xr.where(fronts_ds == 14, 1, fronts_ds)
            fronts_ds = xr.where(fronts_ds == 15, 1, fronts_ds)

            fronts_ds = xr.where(fronts_ds == 16, 0, fronts_ds)  # Remove drylines

        colors_types = ['brown']
        colors_probs = ['YlOrBr']
        names = ['Trough (any)', ]
        labels = ['TR_any', ]

    elif front_types == 'MERGED-ALL':

        if fronts_ds is not None:
            fronts_ds = xr.where(fronts_ds == 5, 1, fronts_ds)  # Forming cold front ---> cold front
            fronts_ds = xr.where(fronts_ds == 6, 2, fronts_ds)  # Forming warm front ---> warm front
            fronts_ds = xr.where(fronts_ds == 7, 3, fronts_ds)  # Forming stationary front ---> stationary front
            fronts_ds = xr.where(fronts_ds == 8, 4, fronts_ds)  # Forming occluded front ---> occluded front
            fronts_ds = xr.where(fronts_ds == 9, 1, fronts_ds)  # Dying cold front ---> cold front
            fronts_ds = xr.where(fronts_ds == 10, 2, fronts_ds)  # Dying warm front ---> warm front
            fronts_ds = xr.where(fronts_ds == 11, 3, fronts_ds)  # Dying stationary front ---> stationary front
            fronts_ds = xr.where(fronts_ds == 12, 4, fronts_ds)  # Dying occluded front ---> occluded front

            # Merge troughs together into class 5
            fronts_ds = xr.where(fronts_ds == 14, 5, fronts_ds)
            fronts_ds = xr.where(fronts_ds == 15, 5, fronts_ds)

            fronts_ds = xr.where(fronts_ds == 13, 6, fronts_ds)  # Move outflow boundaries to class 6
            fronts_ds = xr.where(fronts_ds == 16, 7, fronts_ds)  # Move drylines to class 7

        colors_types = ['blue', 'red', 'limegreen', 'darkviolet', 'brown', 'gold', 'chocolate']
        colors_probs = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'YlOrBr', 'copper_r']
        names = ['Cold front (any)', 'Warm front (any)', 'Stationary front (any)', 'Occluded front (any)', 'Trough (any)',
                 'Outflow boundary', 'Dryline']
        labels = ['CF_any', 'WF_any', 'SF_any', 'OF_any', 'TR_any', 'OFB', 'DL']

    elif type(front_types) == list:

        names = []
        colors_types = []
        colors_probs = []

        # Select the front types that are being used to pull their class identifiers
        filtered_front_types = dict(sorted(dict([(i, front_types_classes[i]) for i in front_types_classes if i in set(front_types)]).items(), key=lambda item: item[1]))
        front_types, num_types = list(filtered_front_types.keys()), len(filtered_front_types.keys())

        for i in range(num_types):
            if fronts_ds is not None:
                if i + 1 != front_types_classes[front_types[i]]:
                    fronts_ds = xr.where(fronts_ds == i + 1, 0, fronts_ds)
                    fronts_ds = xr.where(fronts_ds == front_types_classes[front_types[i]], i + 1, fronts_ds)  # Reformat front classes

                fronts_ds = xr.where(fronts_ds > num_types, 0, fronts_ds)  # Remove unused front types

            colors_probs.append(all_color_probs[front_types_classes[front_types[i]] - 1])
            colors_types.append(all_color_types[front_types_classes[front_types[i]] - 1])
            names.append(all_names[front_types_classes[front_types[i]] - 1])

        labels = front_types

    else:

        colors_types, colors_probs = all_color_types, all_color_types
        names = all_names
        labels = list(front_types_classes.keys())

    if fronts_ds is not None:
        if return_colors:
            if return_names:
                return fronts_ds, names, labels, colors_types, colors_probs
            else:
                return fronts_ds, colors_types, colors_probs
        elif return_names:
            return fronts_ds, names, labels
        else:
            return fronts_ds
    else:
        if return_colors:
            if return_names:
                return names, labels, colors_types, colors_probs
            else:
                return colors_types, colors_probs
        elif return_names:
            return names, labels
        else:
            pass


def find_era5_normalization_parameters(era5_pickle_indir):
    """
    Parameters
    ----------
    era5_pickle_indir: str
        Input directory for the ERA5 pickle files.
    """

    era5_files = fm.load_era5_pickle_files(era5_pickle_indir, num_variables=60, domain='full')

    num_files = len(era5_files)
    first_dataset = pd.read_pickle(era5_files[0])
    min_data = np.min(first_dataset).to_array().values
    mean_data = np.mean(first_dataset.astype('float32')).to_array().values
    max_data = np.max(first_dataset).to_array().values
    variables_in_dataset = list(first_dataset.keys())
    for file in range(1, num_files):
        print('Calculating parameters....%d/%d' % (file, num_files), end='\r')
        current_dataset = pd.read_pickle(era5_files[file])
        current_dataset_mins = np.min(current_dataset).to_array().values
        current_dataset_means = np.mean(current_dataset.astype('float32')).to_array().values
        current_dataset_maxs = np.max(current_dataset).to_array().values

        min_data = np.minimum(current_dataset_mins, min_data)
        mean_data += current_dataset_means
        max_data = np.maximum(current_dataset_maxs, max_data)
    print('Calculating parameters....%d/%d' % (num_files, num_files))

    for variable, minimum, average, maximum in zip(variables_in_dataset, min_data, mean_data, max_data):
        print(variable, minimum, (average / num_files).astype('float16'), maximum)


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


def normalize_variables(variable_ds):
    """
    Function that normalizes GDAS variables via min-max normalization.

    Parameters
    ----------
    variable_ds: xr.Dataset
        - Dataset containing ERA5 or GDAS variable data.

    Returns
    -------
    variable_ds: xr.Dataset
        - Same as input dataset, but the variables are normalized via min-max normalization.
    """

    norm_params = pd.read_csv('./normalization_parameters.csv', index_col='Variable')

    variable_list = list(variable_ds.keys())
    pressure_levels = variable_ds['pressure_level'].values

    for j in range(len(variable_list)):

        new_values_for_variable = np.empty(shape=np.shape(variable_ds[variable_list[0]].values))
        var = variable_list[j]

        for pressure_level_index in range(len(pressure_levels)):
            norm_var = variable_list[j] + '_' + pressure_levels[pressure_level_index]

            # Min-max normalization
            if len(np.shape(new_values_for_variable)) == 4:
                new_values_for_variable[:, :, :, pressure_level_index] = np.nan_to_num((variable_ds[var].values[:, :, :, pressure_level_index] - norm_params.loc[norm_var, 'Min']) /
                                                                                       (norm_params.loc[norm_var, 'Max'] - norm_params.loc[norm_var, 'Min']))
            elif len(np.shape(new_values_for_variable)) == 5:  # If forecast hours are in the dataset
                new_values_for_variable[:, :, :, :, pressure_level_index] = np.nan_to_num((variable_ds[var].values[:, :, :, :, pressure_level_index] - norm_params.loc[norm_var, 'Min']) /
                                                                                          (norm_params.loc[norm_var, 'Max'] - norm_params.loc[norm_var, 'Min']))

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
