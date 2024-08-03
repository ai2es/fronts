"""
Plotting tools.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.8.3
"""

import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def plot_background(extent=None, ax=None, linewidth: float | int = 0.5, projection: str = 'PlateCarree'):
    """
    Returns new background for the plot.

    Parameters
    ----------
    extent: Iterable object with 4 integers
        Iterable containing the extent/boundaries of the plot in the format of [min lon, max lon, min lat, max lat] expressed
            in degrees.
    ax: matplotlib.axes.Axes instance or None
        Axis on which the background will be plotted.
    linewidth: float or int
        Thickness of coastlines and the borders of states and countries.
    projection: str
        Coordinate reference system from cartopy.

    Returns
    -------
    ax: matplotlib.axes.Axes instance
        New plot background.
    """

    crs = getattr(ccrs, projection)

    if ax is None:
        ax = plt.axes(projection=crs)
    else:
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=linewidth)
        ax.add_feature(cfeature.BORDERS, linewidth=linewidth)
        ax.add_feature(cfeature.STATES, linewidth=linewidth)
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
    return ax


def segmented_gradient_colormap(levels, colors: list[str], ns, extend='neither'):
    """
    Make a segmented colormap with linear gradients between specified levels.

    Parameters
    ----------
    levels: 1D array of ints or floats with length M
        Levels corresponding to specified colors in the colormap.
    colors: list of strings with length M
        Colors corresponding to specified levels in the colormap.
    ns: 1D array of ints with length M-1
        Number of colors between each pair of specified levels.
    extend: 'neither', 'min', 'max', 'both'
        The behavior when a value falls out of range of the given levels. See matplotlib.axes.Axes.contourf for details.
    """

    len_levels, len_colors, len_ns = len(levels), len(colors), len(ns)

    assert len_levels > 1, "Must specify at least two levels."
    assert len_colors > 1, "Must specify at least two colors."
    assert len_levels == len_colors, "The number of levels and colors must be equal. Received %d levels and %d colors." % (len_levels, len_colors)
    assert len_ns == len_levels - 1, "The length of 'ns' must be equal to the number of levels minus one. Received 'ns' length of %d and %d levels." % (len_ns, len_levels)

    all_levels = np.concatenate([np.linspace(levels[i], levels[i + 1], ns[i]) for i in range(len_levels - 1)])
    all_colors = np.vstack([LinearSegmentedColormap.from_list("", colors[i:i+2])(np.linspace(0, 1, ns[i])) for i in range(len_levels - 1)])

    # matplotlib's extend function has some funky behavior, so we need to modify the colorbar to avoid errors
    if extend == 'both':
        all_colors = np.insert(all_colors, -1, all_colors[-1], axis=0)
    elif extend == 'neither':
        all_colors = np.delete(all_colors, 0, axis=0)
    else:
        pass

    cmap, norm = mpl.colors.from_levels_and_colors(all_levels, all_colors, extend=extend)

    return cmap, norm


def truncated_colormap(cmap: str, minval: float = 0.0, maxval: float = 1.0, n: int = 256):
    """
    Get an instance of a truncated matplotlib.colors.Colormap object.

    Parameters
    ----------
    cmap: str
        Matplotlib colormap to truncate.
    minval: float
        Starting point of the colormap, represented by a float of 0 <= minval < 1.
    maxval: float
        End point of the colormap, represented by a float of 0 < maxval <= 1.
    n: int
        Number of colors for the colormap.

    Returns
    -------
    new_cmap: matplotlib.colors.Colormap instance
        Truncated colormap.
    """
    cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
