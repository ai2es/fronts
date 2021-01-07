# Imports
import numpy as np
import xarray as xr
import matplotlib as mpl

#mpl.use('agg', force=True)
import matplotlib.pyplot as plt
import math

plt.switch_backend('agg')
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import xml.etree.ElementTree as ET
import pandas as pd
import glob
import argparse
import os
from scipy.interpolate import interp1d
import time
import sys

def read_xml_files(year, month, day):
    """
    Read in xml files for climatology
    :param year:
    :param month:
    :param day:
    :return:
    """
    file_path = "C:/Users/sling/Documents/GitHub/fronts/xmls_2006122012-2020061900"
    print(file_path)
    # read all the files in the directory
    files = glob.glob("%s/*%04d%02d%02d*.xml" % (file_path, year, month, day))
    dss = []
    # Counter for timer number
    fileNo = 1
    for filename in files:
        z_time = (fileNo-1)*3
        fileNo = fileNo + 1
        print(filename)
        # Load XML Data
        tree = ET.parse(filename, parser=ET.XMLParser(encoding='utf-8'))
        root = tree.getroot()
        # Split Off Date, Could be Useful later
        date = filename.split('_')[-1].split('.')[0].split('f')[0]
        # Haversine Formula: Used to turn latitude and longitude into x and y coordinates.
        def haversine(lon1, lat1, lon2, lat2):
            # print(lon1, lon2, lat1, lat2)
            # haversine formula
            dlon = lon2-lon1
            dlat = lat2-lat1
            a = math.sin(dlat/2)**2+math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            dx = (lon1-lon2)*40075*math.cos((lat1+lat2)*math.pi/360)/360 # circumference of earth in km = 40075
            dy = (lat1-lat2)*40075/360
            c = 2*math.asin(math.sqrt(a))
            km = 6371*c #(radius of earth in km = 6371)
            return dlon, dlat, km, dx, dy
        # Define Holding Lists for Data as Read from XML (Latitude and Longitude declared in lines 77 and 78)
        fronttype = []
        frontnumber = []
        dates = []
        # Define counter for front number.
        i = 0
        lats = []
        lons = []
        # Define counter for point number.
        point_number = 0
        # Create array to assign points to a specific front, or assign it a value if a front has already been analyzed.
        front_points = []
        # Iterate through all Frontal Line objects
        for line in root.iter('Line'):
            # Add to counter for each new front.
            i = i + 1
            # Store relevant info for dataframe.
            frontno = i
            frontty = line.get("pgenType")
            # Declare "temporary" x and y variables which will be used to add elements to the distance arrays.
            x_km_temp = 0
            y_km_temp = 0
            # Create x and y distance arrays
            x_km = []
            y_km = []
            # print("Front %d" % i)
            if point_number != 0:
                front_points.append(point_number)
            # Collect all data points along the front.
            for point in line.iter('Point'):
                point_number = point_number + 1
                dates.append(date)
                frontnumber.append(frontno)
                fronttype.append(frontty)
                lats.append(float(point.get("Lat")))
                lons.append(float(point.get("Lon")))
                # print(lats, lons)
            # Initialize x and y distance values for the haversine function.
            dx = 0
            dy = 0
            # create x and y coordinate arrays
            for l in range(1,len(lats)):
                lon1 = lons[l-1]
                lat1 = lats[l-1]
                lon2 = lons[l]
                lat2 = lats[l]
                dlon, dlat, km, dx, dy = haversine(lon1,lat1,lon2,lat2)
                x_km_temp = x_km_temp + dx
                y_km_temp = y_km_temp + dy
                x_km = np.append(x_km, x_km_temp)
                y_km = np.append(y_km, y_km_temp)
            # print(x_km,y_km)
            # print(front_points)

        # Reset front counter
        i = 0
        # Separate front data
        for line in root.iter('Line'):
            # Create new arrays for separate fronts
            x_km_new = []
            y_km_new = []
            i = i + 1
            # Counters for pulling specific data points
            if i == 1:
                x_km_new = x_km[0:front_points[i-1]]
                y_km_new = y_km[0:front_points[i-1]]
            elif i < len(front_points):
                x_km_new = x_km[front_points[i-2]-1:front_points[i-1]-1]
                y_km_new = y_km[front_points[i-2]-1:front_points[i-1]-1]

            front_status = "Loading "+str(month)+"/"+str(day)+"/"+str(year)+" ("+str(z_time)+"Z): Front #"+str(i)+".........."
            # remove duplicates from arrays
            x_km_removed = []
            for num in x_km_new:
                if num not in x_km_removed:
                    x_km_removed.append(num)
            if i < len(front_points):
                print(front_status)
                # print(x_km_new, y_km_new)
                difference = len(x_km_new)-len(x_km_removed)
                # print(difference)

            # interpolate the arrays
            if difference == 0:
                if (len(x_km_new)>3):
                    f = interp1d(x_km_new, y_km_new, kind='quadratic')
                elif (len(x_km_new)==3):
                    f = interp1d(x_km_new, y_km_new, kind='quadratic')
                elif (len(x_km_new)==2):
                    f = interp1d(x_km_new, y_km_new, kind='linear')
                elif (i < len(front_points)):
                    print("ERROR: There is only one point on this front, therefore it cannot be interpolated.")

            # test plot for separated fronts
            plt.figure(figsize=(8,4))
            plt.plot(x_km_new,y_km_new,'ro')
            # if ((difference == 0) and i < len(front_points)): # Create new interval for interpolated x array and plot the interpolation if there is more than one point.
                # xnew = np.linspace(-40000,40000)
                # plt.plot(xnew,f(xnew),'--')
            plt.grid()
            plotTitle = str(month)+"/"+str(day)+"/"+str(year)+" ("+str(z_time)+"Z) - Front #"+str(i)
            plt.title(plotTitle)
            plt.xlabel("Longitudinal distance (km)")
            plt.ylabel("Latitudinal distance (km)")
            plotName = "C:/Users/sling/PycharmProjects/fronts/separated fronts/" + str(month)+"-"+str(day)+"-"+str(year)+" ("+str(z_time)+"Z) - Front #"+str(i) + '.png'

            #### WARNING: THIS CODE OUTPUTS HUNDREDS OF IMAGES AND MAY IMPACT SYSTEM PERFORMANCE. ####
            plt.savefig(plotName) # Comment this statement out to prevent plots from saving.

            # generate and save plot to assure points are plotting correctly in x and y coordinate system
            # plt.figure(figsize=(8,4))
            # Create new interval for interpolated x array and plot the interpolation if there is more than one point.
            # if (len(x_km)>1):
            #    xnew = np.linspace(min(x_km),max(x_km))
            #    plt.plot(xnew,f(xnew),'--')
            # plt.plot(x_km,y_km,'ro')
            # plt.title("Front example")
            # plt.legend(loc='upper right')
            # plt.grid()
            # plt.xlabel("Longitudinal distance (km)")
            # plt.ylabel("Latitudinal distance (km)")
            # plotName = "C:/Users/sling/PycharmProjects/fronts/test plots/" + " Front" + str(i) + '.png'
            # plt.savefig(plotName)

        # Now Create a Dataframe of the lists using the zip approach.
        df = pd.DataFrame(list(zip(dates, frontnumber, fronttype, lats, lons)),
                          columns=['Date', 'Front Number', 'Front Type', 'Latitude', 'Longitude'])
        # Define Types to Avoid trouble later
        df['Latitude'] = df.Latitude.astype(float)
        df['Longitude'] = df.Longitude.astype(float)
        df['Front Number'] = df['Front Number'].astype(int)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d%H')


        # Define your grid - here I am using 0.25 degree
        # xit = np.linspace(130,370,961)
        xit = np.concatenate((np.linspace(-180, 10, 761), np.linspace(170, 180, 41)[0:40]))
        yit = np.linspace(0, 80, 321)

        # Now Use a Little Trick to Map the data to a New Grid as Pandas Data Columns - in Place.
        # Note that if the values are outside the bins, then these will be placed in 0, len(xit).
        # Remember there is always one less bin than there is a value.
        # df = df.assign(xit=np.digitize(df['Longitude'].values, xit))
        # df = df.assign(xit=np.digitize(df['Longitude'].values, xit))
        df = df.assign(xit=np.digitize(df['Longitude'].values, xit))
        df = df.assign(yit=np.digitize(df['Latitude'].values, yit))

        types = df['Front Type'].unique()

        type_da = []
        for i in range(0, len(types)):
            type_df = df[df['Front Type'] == types[i]]
            groups = type_df.groupby(['xit', 'yit'])

            # Create a Numpy Array to Count Frequency
            frequency = np.zeros((yit.shape[0] + 1, xit.shape[0] + 1))

            for group in groups:
                # Here we count in the appropriate location, and avoid double counting - assign to location
                frequency[group[1].yit.values[0], group[1].xit.values[0]] += np.where(
                    len(group[1]['Front Number']) >= 1, 1, 0)

                # Trim to clean up the final bin where anything outside domain lands
            frequency = frequency[1:322, 1:802]

            # Now turn it into an xarray DataArray
            ds = xr.Dataset(data_vars={"Frequency": (['Latitude', 'Longitude'], frequency)},
                            coords={'Latitude': yit, 'Longitude': xit})
            # ds = xr.DataArray(frequency,dims=('Latitude', 'Longitude'),coords={'Latitude': yit,'Longitude':xit})
            # Append this type of front to the list for concatenation
            type_da.append(ds)
        # Concatenate Along a New Dimension we will label 'Front Type'
        ds = xr.concat(type_da, dim=types)
        ds = ds.rename({'concat_dim': 'Type'})
        dss.append(ds)
    timestep = pd.date_range(start=df['Date'].dt.strftime('%Y-%m-%d')[0], periods=8, freq='3H')
    dns = xr.concat(dss, dim=timestep)
    dns = dns.rename({'concat_dim': 'Date'})
    return dns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create labeled data for the specified day")
    parser.add_argument('--year', type=int, required=True, help="year for the data to be read in")
    parser.add_argument('--month', type=int, required=True, help="month for the data to be read in")
    parser.add_argument('--day', type=int, required=True, help="day for the data to be read in")
    parser.add_argument('--image_outdir', type=str, required=False, help="output directory for image files")
    parser.add_argument('--netcdf_outdir', type=str, required=False, help="output directory for netcdf files")
    args = parser.parse_args()
    #   print(args)
    # read the polygons for the specified day
    #   print("reading the xml")
    xmls = read_xml_files(args.year, args.month, args.day)
    #    print(xmls)

    # One way to output - As a netCDF of just this timestep - the above could then be looped in bash,
    # and xarray used to open the multiple nc files:
    netcdf_outtime = str('%04d%02d%02d' % (args.year, args.month, args.day))
    out_file_name = 'FrontalCounts_%s.nc' % netcdf_outtime
    out_file_path = os.path.join(args.netcdf_outdir, out_file_name)
    xmls.to_netcdf(out_file_path)

#   netcdf_outtime = str('%04d%02d%02d' % (args.year, args.month, args.day))
# xmls.to_netcdf(os.path.join(args.netcdf_outdir, f'FrontalCounts_{netcdf_outtime}.nc'))

# Here is an idea of how you might loop through your list of files to make this have a time dimension as well, using a list of xarrays
# which you then concatenate.
# dss = []
# ds = xr.DataArray(CF_frequency,dims=('Latitude', 'Longitude'),coords={'Latitude': yit,'Longitude':xit})
# dss.append(ds)
# ds = xr.concat(dss, dim=dates, name='Time')

# cartopy map
for i in range(0,8):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=250))
    ax.gridlines()
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.set_extent([130, 370, 0, 80], crs=ccrs.PlateCarree())
    xmls.Frequency.sel(Type='COLD_FRONT').isel(Date=i).plot(x='Longitude', y='Latitude', transform=ccrs.PlateCarree())
    xmls.Frequency.sel(Type='WARM_FRONT').isel(Date=i).plot(x='Longitude', y='Latitude', transform=ccrs.PlateCarree())
    outtime=str(xmls.Frequency.Date[i].dt.strftime('%Y%m%d%H').values)
    print(outtime)
    plt.savefig(os.path.join(args.image_outdir,f'frequencyplot_{outtime}.png'), bbox_inches='tight',dpi=300)
    #plt.show()

# Ryan function to increase number of points in fronts


# plot
# xmls.Frequency.sel(Type= 'COLD_FRONT').isel(Date=i).plot(x='Longitude',y='Latitude',transform=ccrs.PlateCarree())
# xmls.Frequency.sel(Type= 'WARM_FRONT').isel(Date=i).plot(x='Longitude',y='Latitude',transform=ccrs.PlateCarree())
# outtime=str(xmls.Frequency.Date[i].dt.strftime('%Y%m%d%H').values)
# print(outtime)
# plt.savefig(os.path.join(args.image_outdir,f'frequencyplot_{outtime}.png'), bbox_inches='tight',dpi=300)
# plt.show()
