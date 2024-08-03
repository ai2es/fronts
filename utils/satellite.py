"""
General tools for satellite data.

Script version: 2024.8.3
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from utils.plotting import segmented_gradient_colormap, truncated_colormap


def calculate_lat_lon_from_dataset(ds):
    """
    Calculate lat/lon coordinates from an unmodified dataset containing GOES satellite data. This function was pulled directly
        from https://www.star.nesdis.noaa.gov/atmospheric-composition-training/python_abi_lat_lon.php.

    Parameters
    ----------
    ds: xarray or netCDF dataset
        Unmodified GOES satellite dataset.

    Returns
    -------
    abi_lat: np.ndarray
        Latitude in degrees north.
    abi_lon: np.ndarray
        Longitude in degrees east.
    """
    # Read in GOES ABI fixed grid projection variables and constants
    x_coordinate_1d = ds['x'][:]  # E/W scanning angle in radians
    y_coordinate_1d = ds['y'][:]  # N/S elevation angle in radians
    projection_info = ds['goes_imager_projection']
    lon_origin = projection_info.longitude_of_projection_origin
    H = projection_info.perspective_point_height + projection_info.semi_major_axis
    r_eq = projection_info.semi_major_axis
    r_pol = projection_info.semi_minor_axis

    # Create 2D coordinate matrices from 1D coordinate vectors
    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)

    # Equations to calculate latitude and longitude
    lambda_0 = (lon_origin * np.pi) / 180.0
    a_var = np.power(np.sin(x_coordinate_2d), 2.0) + (np.power(np.cos(x_coordinate_2d), 2.0) * (
                np.power(np.cos(y_coordinate_2d), 2.0) + (
                    ((r_eq * r_eq) / (r_pol * r_pol)) * np.power(np.sin(y_coordinate_2d), 2.0))))
    b_var = -2.0 * H * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
    c_var = (H ** 2.0) - (r_eq ** 2.0)
    r_s = (-1.0 * b_var - np.sqrt((b_var ** 2) - (4.0 * a_var * c_var))) / (2.0 * a_var)
    s_x = r_s * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
    s_y = - r_s * np.sin(x_coordinate_2d)
    s_z = r_s * np.cos(x_coordinate_2d) * np.sin(y_coordinate_2d)

    # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
    np.seterr(all='ignore')

    abi_lat = (180.0 / np.pi) * (np.arctan(((r_eq * r_eq) / (r_pol * r_pol)) * (s_z / np.sqrt(((H - s_x) * (H - s_x)) + (s_y * s_y)))))
    abi_lon = (lambda_0 - np.arctan(s_y / (H - s_x))) * (180.0 / np.pi)

    return abi_lat, abi_lon


def get_satellite_colormap(band: int | str):
    """
    Retrieve a colormap for a GOES satellite band.

    Parameters
    ----------
    band: int or 'sandwich'
        Band for which the colormap will be retrieved. Integer must be between 1 and 16.

    Returns
    -------
    cmap: instance of matplotlib.colors.LinearSegmentedColormap
        Colormap for the requested band.
    norm: instance of matplotlib.colors.Normalize
        Normalization for the colormap.
    """

    if isinstance(band, int):

        if 1 <= band <= 6:

            # greyscale colormap normalized from 0 to 1 (reflectance factor)
            return truncated_colormap('Greys_r', minval=0.2), mpl.colors.Normalize(vmin=0, vmax=1)

        elif band == 7:

            # multiple combined colormaps, only -65 to -25 celsius (208 to 248 K) is colored
            n1, n2, n3 = 36, 80, 304

            cmap_1 = truncated_colormap('Greys', minval=0.4, maxval=0.7)(np.linspace(0, 1, n1))
            cmap_2 = truncated_colormap('jet', minval=0.1, maxval=1.0)(np.linspace(0, 1, n2))[::-1]
            cmap_3 = truncated_colormap('Greys', minval=0.4, maxval=0.8)(np.linspace(0, 1, n3))

            levels = np.concatenate([np.linspace(-83, -65, n1), np.linspace(-65, -25, n2), np.linspace(-25, 127, n3)]) + 273.15
            colors = np.vstack((cmap_1, cmap_2, cmap_3))

            cmap, norm = mpl.colors.from_levels_and_colors(levels, colors, extend='max')

            return cmap, norm

        elif 8 <= band <= 10:

            levels = np.array([-100, -75, -40, -25, -16, 0, 0, 7]) + 273.15
            colors = ['#00ffff', '#438525', '#ffffff', '#030275', '#fcfc01', '#ff0000', 'black', 'black']
            ns = [50, 70, 30, 18, 32, 1, 1]
            extend = 'max'

        elif 11 <= band <= 15:

            levels = np.array([-90, -80, -70, -60, -52, -44, -36, -36, 60]) + 273.15
            colors = ['white', 'black', 'red', 'yellow', '#26fe01', '#010370', '#00ffff', '#c6baba', 'black']
            ns = [20, 20, 20, 16, 16, 16, 1, 190]
            extend = 'both'

        elif band == 16:

            levels = np.array([-128, -90, -30, 0, 0, 30, 60, 83, 128]) + 273.15
            colors = ['black', 'white', 'blue', '#704140', '#89140b', 'yellow', 'red', '#818180', 'black']
            ns = [76, 120, 60, 1, 60, 60, 46, 90]
            extend = 'both'

        else:

            raise ValueError(f"Band number must be between 1 and 16, received value:", band)

    elif band == 'sandwich':

        return plt.get_cmap('jet_r'), mpl.colors.Normalize(vmin=-93 + 273.15, vmax=-15 + 273.15)

    else:
        raise ValueError("Unrecognized band:", band)

    cmap, norm = segmented_gradient_colormap(levels, colors, ns, extend)

    return cmap, norm