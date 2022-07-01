"""
Plotting tools

Code written by: Andrew Justin (andrewjustinwx@gmail.com)

Last updated: 7/1/2022 2:35 PM CDT
"""

import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
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


def create_colorbar_for_fronts(names, cmap, norm, axis_loc=(0.7765, 0.11, 0.015, 0.77)):
    """
    Create colorbar for given front types.

    Parameters
    ----------
    names: list of strs
        Names of the front types.
    axis_loc: tuple or list of 4 floats
        Location of the new axis for the colorbar: xmin, xmax, ymin, ymax
    cmap: matplotlib.colors.Colormap object
        Colormap for the fronts.
    norm: matplotlib.colors.Normalize
        Colorbar normalization.
    """
    number_of_front_types = len(names)
    cbar_ax = plt.axes(axis_loc)  # Create an axis for the colorbar to the right of the plot
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)  # Create the colorbar
    cbar.set_ticks(np.arange(1, number_of_front_types + 1) + 0.5)  # Place ticks in the middle of each color
    cbar.set_ticklabels(names)  # Label each tick with its respective front type
