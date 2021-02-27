# test plot for separated fronts
import matplotlib.pyplot as plt

# Function for a test plot that shows fronts in xy coordinates both BEFORE and AFTER interpolation.
def plot_front_xy(x_km_new, y_km_new, x_new, y_new, date, image_outdir, i):
    # Plot Cartesian (xy) system BEFORE interpolation.
    plt.subplots(2,1)
    plt.figure(figsize=(5,5))
    plt.subplot(2,1,1)
    plt.plot(x_km_new,y_km_new,'ro') # plot front's original points
    plt.grid()
    plt.xlabel("Longitudinal distance (km)")
    plt.ylabel("Latitudinal distance (km)")
    plotTitle = "%s - Front #%d (Cartesian) - Before" % (date, i)
    plt.title(plotTitle)
    # Plot Cartesian (xy) system AFTER interpolation.
    plt.subplot(2,1,2)
    plt.grid()
    plt.plot(x_new,y_new,'ro') # plot new interpolated points
    plt.xlabel("Longitudinal distance (km)")
    plt.ylabel("Latitudinal distance (km)")
    plotTitle = "%s - Front #%d (Cartesian) - After" % (date, i)
    plt.title(plotTitle)
    plotName = "%s/separated_fronts/%s - Front #%d (Cartesian).png" % (image_outdir, date, i)
    plt.tight_layout(w_pad=3)
    plt.savefig(plotName,bbox_inches='tight') # Comment this statement out to prevent plots from saving.

# Function for a test plot for fronts in lat/lon system AFTER interpolation.
def plot_front_latlon(lon_new, lat_new, date, image_outdir, i):
    plt.figure(figsize=(5,5))
    plt.plot(lon_new,lat_new,'ro')
    plt.grid()
    plotTitle = "%s - Front #%d (Latitude/Longitude) - After" % (date, i)
    plt.title(plotTitle)
    plt.xlabel("Longitude (degrees)")
    plt.ylabel("Latitude (degrees)")
    plotName = "%s/separated_fronts/%s - Front #%d (LatLon).png" % (image_outdir, date, i)
    plt.savefig(plotName,bbox_inches='tight')

# Function for a test plot for all points (Cartesian system) of every front in the file BEFORE interpolation.
def plot_file_xy(x_km, y_km, date, image_outdir):
    plt.figure(figsize=(5,5))
    # plot Cartesian system
    plt.plot(x_km,y_km,'ro')
    plt.grid()
    plotTitle = "%s (Cartesian)" % date
    plt.title(plotTitle)
    plt.xlabel("Longitudinal distance (km)")
    plt.ylabel("Latitudinal distance (km)")
    plotName = "%s/separated_fronts/%s (Cartesian).png" % (image_outdir, date)
    plt.savefig(plotName,bbox_inches='tight') # Comment this statement out to prevent plots from saving.
