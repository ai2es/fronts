"""
Plotting tools

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 4/14/2022 8:54 PM CDT

Known bugs:
- none

Please report any bugs to Andrew Justin: andrewjustinwx@gmail.com
"""

import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


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
