"""
Function that plots each front individually on its own plot. This function can be used for debugging purposes.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 3/9/2021 13:53 CST by Andrew Justin
"""

import matplotlib.pyplot as plt

def plot_front_xy(x_km_new, y_km_new, x_new, y_new, date, image_outdir, i):
    """
    Takes xy coordinates both BEFORE and AFTER interpolation and plots them to check for errors.

    Parameters
    ----------
    x_km_new: list
        List containing longitude coordinates in kilometers BEFORE interpolation.
    y_km_new: list
        List containing latitude coordinates in kilometers BEFORE interpolation.
    x_new: list
        List containing longitude coordinates in kilometers AFTER interpolation.
    y_new: list
        List containing latitude coordinates in kilometers AFTER interpolation.
    date: list
        List containing date and time of the specified fronts.
    image_outdir: str
        Output directory of the plot.
    i: int
        Current front number.
    """

    plt.subplots(2,1)
    plt.figure(figsize=(5,5))

    plt.subplot(2,1,1)
    plt.plot(x_km_new,y_km_new,'ro')
    plt.grid()
    plt.xlabel("Longitudinal distance (km)")
    plt.ylabel("Latitudinal distance (km)")
    plotTitle = "%s - Front #%d (Cartesian) - Before" % (date, i)
    plt.title(plotTitle)

    plt.subplot(2,1,2)
    plt.grid()
    plt.plot(x_new,y_new,'ro')
    plt.xlabel("Longitudinal distance (km)")
    plt.ylabel("Latitudinal distance (km)")
    plotTitle = "%s - Front #%d (Cartesian) - After" % (date, i)
    plt.title(plotTitle)

    plotName = "%s/separated_fronts/%s - Front #%d (Cartesian).png" % (image_outdir, date, i)
    plt.tight_layout(w_pad=3)
    plt.savefig(plotName,bbox_inches='tight')

def plot_front_latlon(lon_new, lat_new, date, image_outdir, i):
    """
    Takes lon/lat coordinates AFTER interpolation and plots them to check for errors.

    Parameters
    ----------
    lon_new: list
        List containing longitude coordinates in degrees AFTER interpolation.
    lat_new: list
        List containing latitude coordinates in degrees AFTER interpolation.
    date: list
        List containing date and time of the specified fronts.
    image_outdir: str
        Output directory of the plot.
    i: int
        Current front number.
    """

    plt.figure(figsize=(5,5))

    plt.plot(lon_new,lat_new,'ro')
    plt.grid()
    plotTitle = "%s - Front #%d (Latitude/Longitude) - After" % (date, i)
    plt.title(plotTitle)
    plt.xlabel("Longitude (degrees)")
    plt.ylabel("Latitude (degrees)")

    plotName = "%s/separated_fronts/%s - Front #%d (LatLon).png" % (image_outdir, date, i)
    plt.savefig(plotName,bbox_inches='tight')

def plot_file_xy(x_km, y_km, date, image_outdir):
    """
    Plots all coordinates of every front in the file in kilometers BEFORE interpolation to check for errors.

    Parameters
    ----------
    x_km: list
        List containing longitude coordinates of every front in the file in kilometers BEFORE interpolation.
    y_km: list
        List containing latitude coordinates of every front in the file in kilometers BEFORE interpolation.
    date: list
        List containing date and time of the specified fronts.
    image_outdir: str
        Output directory of the plot.
    """

    plt.figure(figsize=(5,5))

    plt.plot(x_km,y_km,'ro')
    plt.grid()
    plotTitle = "%s (Cartesian)" % date
    plt.title(plotTitle)
    plt.xlabel("Longitudinal distance (km)")
    plt.ylabel("Latitudinal distance (km)")

    plotName = "%s/separated_fronts/%s (Cartesian).png" % (image_outdir, date)
    plt.savefig(plotName,bbox_inches='tight')
