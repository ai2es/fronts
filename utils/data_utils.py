"""
Data tools

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 6/18/2021 5:08 PM CST
"""

import math
import pandas as pd
from shapely.geometry import LineString
import numpy as np
import xarray as xr
import file_manager as fm


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
    
    variable_name = list(ds_fronts.keys())[0]
    indices = np.where(ds_fronts[variable_name].values != 0)
    identifier = ds_fronts[variable_name].values

    for i in range(len(indices[0])):
        front_value = identifier[indices[0][i]][indices[1][i]]
        if lats[indices[0][i]] == lats[0]:
            # If the front pixel is at the north end of the domain (max lat), and there is no front directly
            # to the south, expand the front 1 pixel south.
            if identifier[indices[0][i]][indices[1][i]] == 0:
                ds_fronts[variable_name].values[indices[0][i]][indices[1][i]] = front_value
            # If the front pixel is at the northwest end of the domain (max/min lat/lon), check the pixels to the
            # southeast and east for fronts, and expand the front there if no other fronts are already present.
            if lons[indices[1][i]] == lons[0]:
                if identifier[indices[0][i] + 1][indices[1][i] + 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i] + 1][indices[1][i] + 1] = front_value
                if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i]][indices[1][i] + 1] = front_value
            # If the front pixel is at the northeast end of the domain (max/max lat/lon), check the pixels to the
            # southwest and west for fronts, and expand the front there if no other fronts are already present.
            elif lons[indices[1][i]] == lons[len_lons - 1]:
                if identifier[indices[0][i] + 1][indices[1][i] - 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i] + 1][indices[1][i] - 1] = front_value
                if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i]][indices[1][i] - 1] = front_value
            # If the front pixel is at the north end of the domain (max lat), but not at the west or east end (min lon
            # or max lon) check the pixels to the west and east for fronts, and expand the front there if no other
            # fronts are already present.
            else:
                if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i]][indices[1][i] - 1] = front_value
                if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i]][indices[1][i] + 1] = front_value
        elif lats[-1] < lats[indices[0][i]] < lats[len_lats - 1]:
            # If there is no front directly to the south, expand the front 1 pixel south.
            if identifier[indices[0][i] + 1][indices[1][i]] == 0:
                identifier[indices[0][i]][indices[1][i]] = front_value
            # If there is no front directly to the north, expand the front 1 pixel north.
            if identifier[indices[0][i] - 1][indices[1][i]] == 0:
                ds_fronts[variable_name].values[indices[0][i] - 1][indices[1][i]] = front_value
            # If the front pixel is at the west end of the domain (min lon), check the pixels to the southeast,
            # east, and northeast for fronts, and expand the front there if no other fronts are already present.
            if lons[indices[1][i]] == lons[0]:
                if identifier[indices[0][i] + 1][indices[1][i] + 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i] + 1][indices[1][i] + 1] = front_value
                if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i]][indices[1][i] + 1] = front_value
                if identifier[indices[0][i] - 1][indices[1][i] + 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i] - 1][indices[1][i] + 1] = front_value
            # If the front pixel is at the east end of the domain (min lon), check the pixels to the southwest,
            # west, and northwest for fronts, and expand the front there if no other fronts are already present.
            elif lons[indices[1][i]] == lons[len_lons - 1]:
                if identifier[indices[0][i] + 1][indices[1][i] - 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i] + 1][indices[1][i] - 1] = front_value
                if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i]][indices[1][i] - 1] = front_value
                if identifier[indices[0][i] - 1][indices[1][i] - 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i] - 1][indices[1][i] - 1] = front_value
            # If the front pixel is not at the end of the domain in any direction, check the northeast, east,
            # southeast, northwest, west, and southwest for fronts, and expand the front there if no other fronts
            # are already present.
            else:
                if identifier[indices[0][i] + 1][indices[1][i] + 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i] + 1][indices[1][i] + 1] = front_value
                if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i]][indices[1][i] + 1] = front_value
                if identifier[indices[0][i] - 1][indices[1][i] + 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i] - 1][indices[1][i] + 1] = front_value
                if identifier[indices[0][i] + 1][indices[1][i] - 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i] + 1][indices[1][i] - 1] = front_value
                if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i]][indices[1][i] - 1] = front_value
                if identifier[indices[0][i] - 1][indices[1][i] - 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i] - 1][indices[1][i] - 1] = front_value
        else:
            # If the front pixel is at the south end of the domain (max lat), and there is no front directly
            # to the north, expand the front 1 pixel north.
            if identifier[indices[0][i] - 1][indices[1][i]] == 0:
                ds_fronts[variable_name].values[indices[0][i] - 1][indices[1][i]] = front_value
            # If the front pixel is at the southwest end of the domain (max/min lat/lon), check the pixels to the
            # northeast and east for fronts, and expand the front there if no other fronts are already present.
            if lons[indices[1][i]] == lons[0]:
                if identifier[indices[0][i] - 1][indices[1][i] + 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i] - 1][indices[1][i] + 1] = front_value
                if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i]][indices[1][i] + 1] = front_value
            # If the front pixel is at the southeast end of the domain (max/max lat/lon), check the pixels to the
            # northwest and west for fronts, and expand the front there if no other fronts are already present.
            elif lons[indices[1][i]] == lons[len_lons - 1]:
                if identifier[indices[0][i] - 1][indices[1][i] - 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i] - 1][indices[1][i] - 1] = front_value
                if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i]][indices[1][i] - 1] = front_value
            # If the front pixel is at the south end of the domain (max lat), but not at the west or east end (min lon
            # or max lon) check the pixels to the west and east for fronts, and expand the front there if no other
            # fronts are already present.
            else:
                if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i]][indices[1][i] - 1] = front_value
                if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                    ds_fronts[variable_name].values[indices[0][i]][indices[1][i] + 1] = front_value

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


def reformat_fronts(fronts_ds, front_types, return_colors=False, return_names=False):
    """

    Parameters
    ----------
    fronts_ds: xr.Dataset or np.array
        - Dataset containing the front data.
    front_types: str or list of strs
        - Code(s) that determine how the dataset or array will be reformatted.
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
        F_BIN (1 class): Train on 1-4, but treat all front types as one type.
            (1): CF, WF, SF, OF

        MERGED-F (4 classes): Train on 1-12, but treat forming and dissipating fronts as a standard front.
            (1): CF, CF-F, CF-D
            (2): WF, WF-F, WF-D
            (3): SF, SF-F, SF-D
            (4): OF, OF-F, OF-D

        MERGED-F-BIN (1 class): Train on 1-12, but treat all front types and stages as one type. This means that classes 1-12 will all be one class (1).
            (1): CF, CF-F, CF-D, WF, WF-F, WF-D, SF, SF-F, SF-D, OF, OF-F, OF-D

        MERGED-T (1 class): Train on 14-15, but treat troughs and tropical troughs as the same. In other words, TT (15) becomes TROF (14).
            (1): TROF, TT

        MERGED-ALL (7 classes): Train on all classes (1-16), but make the changes in the MERGED-F and MERGED-T codes.
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
        fronts_ds = xr.where(fronts_ds > 4, 0, fronts_ds)  # Classes 5-16 are removed
        fronts_ds = xr.where(fronts_ds > 0, 1, fronts_ds)  # Merge 1-4 into one class

        colors_types = ['tab:red']
        colors_probs = ['Reds']
        names = ['CF, WF, SF, OF', ]
        labels = ['CF-WF-SF-OF', ]

    elif front_types == 'MERGED-F':
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
        fronts_ds = xr.where(fronts_ds > 12, 0, fronts_ds)  # Classes 13-16 are removed
        fronts_ds = xr.where(fronts_ds > 0, 1, fronts_ds)  # Classes 1-12 are merged into one class

        colors_types = ['gray']
        colors_probs = ['Greys']
        names = ['CF, WF, SF, OF (any)', ]
        labels = ['CF-WF-SF-OF_any', ]

    elif front_types == 'MERGED-T':
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
            if i + 1 != front_types_classes[front_types[i]]:
                fronts_ds = xr.where(fronts_ds == i + 1, 0, fronts_ds)
                fronts_ds = xr.where(fronts_ds == front_types_classes[front_types[i]], i + 1, fronts_ds)  # Reformat front classes

            colors_probs.append(all_color_probs[front_types_classes[front_types[i]] - 1])
            colors_types.append(all_color_types[front_types_classes[front_types[i]] - 1])
            names.append(all_names[front_types_classes[front_types[i]] - 1])

        fronts_ds = xr.where(fronts_ds > num_types, 0, fronts_ds)  # Remove unused front types
        labels = front_types

    else:
        colors_types, colors_probs = all_color_types, all_color_types
        names = all_names
        labels = list(front_types_classes.keys())

    if return_colors:
        if return_names:
            return fronts_ds, names, labels, colors_types, colors_probs
        else:
            return fronts_ds, colors_types, colors_probs
    elif return_names:
        return fronts_ds, names, labels
    else:
        return fronts_ds


def find_era5_normalization_parameters(era5_pickle_indir, fronts_pickle_indir):
    """

    Parameters
    ----------
    era5_pickle_indir: str
        Input directory for the ERA5 pickle files.
    fronts_pickle_indir: str
        Input directory for the front object pickle files.
    """

    _, era5_files = fm.load_era5_pickle_files(era5_pickle_indir, fronts_pickle_indir, num_variables=60, domain='full')

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
