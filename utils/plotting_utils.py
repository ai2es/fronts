"""
Plotting tools

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.8.14
"""

import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_background(extent, ax=None, linewidth=0.5):
    """
    Returns new background for the plot.

    Parameters
    ----------
    extent: iterable with 4 ints
        - Iterable containing the extent/boundaries of the plot in the format of [min lon, max lon, min lat, max lat].
    ax: matplotlib.axes.Axes instance or None
        - Axis on which the background will be plotted.
    linewidth: float
        - Thickness of coastlines and the borders of states and countries.

    Returns
    -------
    ax: matplotlib.axes.Axes instance
        - New plot background.
    """
    if ax is None:
        crs = ccrs.LambertConformal(central_longitude=250)
        ax = plt.axes(projection=crs)
    else:
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=linewidth)
        ax.add_feature(cfeature.BORDERS, linewidth=linewidth)
        ax.add_feature(cfeature.STATES, linewidth=linewidth)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    return ax


def truncated_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
