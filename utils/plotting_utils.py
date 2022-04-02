""" Plotting tools """

import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


def plot_background(extent, ax=None, linewidth=0.5):
    """
    Returns new background for the plot.

    Parameters
    ----------
    extent: Numpy array containing the extent/boundaries of the plot in the format of [min lon, max lon, min lat, max lat].
    linewidth: Thickness of coastlines and the borders of states and countries.

    Returns
    -------
    ax: New plot background.
    """
    if ax is None:
        crs = ccrs.LambertConformal(central_longitude=250)
        ax = plt.axes(projection=crs)
    else:
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.25)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25)
        ax.add_feature(cfeature.STATES, linewidth=0.25)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    return ax
