# Imports
import math

# mpl.use('agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

plt.switch_backend('agg')
import xml.etree.ElementTree as ET
import pandas as pd
import glob
import argparse
import os
from shapely.geometry import LineString


# Haversine formula - turns lats/lons into x/y and calculates distance between points
def haversine(lon1, lat1, lon2, lat2):
    # print(lon1, lon2, lat1, lat2)
    # haversine formula for Cartesian system
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = math.sin(dlat/2)**2+math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    dx = (-dlon)*40075*math.cos((lat1+lat2)*math.pi/360)/360 # circumference of earth in km = 40075
    dy = (-dlat)*40075/360
    c = 2*math.asin(math.sqrt(a))
    km = 6371*c # radius of earth in km = 6371
    return dlon, dlat, km, dx, dy

# Create geometric dataframe
def geometric(x_km_new, y_km_new):
    df_xy = pd.DataFrame(list(zip(x_km_new, y_km_new)), columns=['Longitude_km', 'Latitude_km'])
    geometry = [xy for xy in zip(df_xy.Longitude_km,df_xy.Latitude_km)]
    xy_linestring = LineString(geometry)
    return xy_linestring

# Plot x/y points at a specified interval
def redistribute_vertices(xy_linestring, distance):
    if xy_linestring.geom_type == 'LineString':
        num_vert = int(round(xy_linestring.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [xy_linestring.interpolate(float(n) / num_vert, normalized=True)
            for n in range(num_vert + 1)])
    elif xy_linestring.geom_type == 'MultiLineString':
        parts = [redistribute_vertices(part, distance)
            for part in xy_linestring]
        return type(xy_linestring)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (xy_linestring.geom_type,))

# Reverse haversine function - turns x and y coordinates back to latitude and longitude.
def reverse_haversine(lons, lats, lon_new, lat_new, front_points, i, a, dx, dy):
    if i==1:
        if a==1:
            lon1 = lons[a-1]
            lat1 = lats[a-1]
        else:
            lon1 = lon_new[a-1]
            lat1 = lat_new[a-1]
    else:
        if a==1:
            lon1 = lons[front_points[i-2]]
            lat1 = lats[front_points[i-2]]
        else:
            lon1 = lon_new[a-1]
            lat1 = lat_new[a-1]
    lat2 = lat1-(dy*(360/40075))
    lon2 = lon1-(dx*(360/40075)/(math.cos((math.pi/360)*(lat1+lat2))))
    return lon1, lon2, lat1, lat2

def read_xml_files(year, month, day):
    """
    Read in xml files for climatology
    :param year:
    :param month:
    :param day:
    :return:
    """
    global fronts_lon_array, fronts_lat_array
    global period_fronts_lon_array, period_fronts_lat_array
    global period_fronts_type, period_fronts_number_new, period_fronts_date
    file_path = "C:/Users/Andrew/PycharmProjects/fronts/xmls_2006122012-2020061900"
    print(file_path)
    # read all the files in the directory
    # files = glob.glob("%s/*%04d%02d%02d*.xml" % (file_path, year, month, day))
    # files = glob.glob("%s/*%04d%02d*.xml" % (file_path, year, month))
    files = glob.glob("%s/*%04d*.xml" % (file_path, year))
    dss = []
    # Counter for timer number
    fileNo = 1
    # Initialize x and y distance values for the haversine function.
    dx = 0
    dy = 0
    # create arrays for fronts in different periods
    period_fronts_date = []
    period_fronts_type = []
    period_fronts_number_new = []
    period_fronts_lon_array = []
    period_fronts_lat_array = []

    for filename in files:
        z_time = (fileNo-1)*3
        fileNo = fileNo + 1
        print(filename)
        # Load XML Data
        tree = ET.parse(filename, parser=ET.XMLParser(encoding='utf-8'))
        root = tree.getroot()
        # Split Off Date, Could be Useful later
        date = filename.split('_')[-1].split('.')[0].split('f')[0]
        # Define Holding Lists for Data as Read from XML (Latitude and Longitude declared earlier)
        fronttype = []
        frontnumber = []
        fronts_number = [] # this array is for the interpolated points' front type
        dates = []
        # Define counter for front number.
        i = 0
        lats = []
        lons = []
        # Define counter for point number.
        point_number = 0
        # Create array to assign points to a specific front, or assign it a value if a front has already been analyzed.
        front_points = []
        # Create array to assign front types to interpolated points
        front_types = []
        # Create array to assign dates to interpolated points
        front_dates = []
        # Arrays for holding interpolated lat/lon data points
        fronts_lon_array = []
        fronts_lat_array = []
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
            x_km_360_temp = 0
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
            # create x and y coordinate arrays
            for l in range(1,len(lats)):
                lon1 = lons[l-1]
                lat1 = lats[l-1]
                lon2 = lons[l]
                lat2 = lats[l]
                dlon, dlat, km, dx, dy = haversine(lon1,lat1,lon2,lat2)
                # check for errors in the data and correct them
                if (dx>750 or dy>750 or dx<750 or dy<750):
                    if (lon1>0 and lon2<0):
                        lon2 = -lon2
                        dlon, dlat, km, dx, dy = haversine(lon1,lat1,lon2,lat2)
                    if (lon2>0 and lon1<0):
                        lon1 = -lon1
                        dlon, dlat, km, dx, dy = haversine(lon1,lat1,lon2,lat2)
                    if (lat1>0 and lat2<0):
                        lat2 = -lat2
                        dlon, dlat, km, dx, dy = haversine(lon1,lat1,lon2,lat2)
                    if (lat2>0 and lat1<0):
                        lat1 = -lat1
                        dlon, dlat, km, dx, dy = haversine(lon1,lat1,lon2,lat2)
                x_km_temp = x_km_temp + dx
                y_km_temp = y_km_temp + dy
                x_km.append(x_km_temp)
                y_km.append(y_km_temp)

        # Reset front counter
        i = 0
        # Separate front data for Cartesian and 360 coordinate system
        for line in root.iter('Line'):
            # Pull front number and type
            frontno = i
            frontty = line.get("pgenType")
            # Create 360 coordinate array
            x_km_360 = []
            for l in range(1,len(lats)):
                x_km_360.append(x_km[l-1]-min(x_km))
            # Create new arrays for separate fronts
            x_km_new = []
            y_km_new = []
            x_km_360_new = []
            i = i + 1
            # Counters for pulling specific data points
            if i == 1:
                x_km_new = x_km[0:front_points[i-1]]
                y_km_new = y_km[0:front_points[i-1]]
                x_km_360_new = x_km_360[0:front_points[i-1]]
            elif i < len(front_points):
                x_km_new = x_km[front_points[i-2]-1:front_points[i-1]-1]
                y_km_new = y_km[front_points[i-2]-1:front_points[i-1]-1]
                x_km_360_new = x_km_360[front_points[i-2]-1:front_points[i-1]-1]

            # front_status = "Loading "+str(month)+"/"+str(day)+"/"+str(year)+" ("+str(z_time)+"Z): Front #"+str(i)+".........."
            # front_status = "Loading "+str(filename)+": Front #"+str(i)+".........."
            # check for duplicate points in arrays
            x_km_removed = []
            for num in x_km_new:
                if num not in x_km_removed:
                    x_km_removed.append(num)
            if i < len(front_points):
                # print(front_status)
                difference = len(x_km_new)-len(x_km_removed)
                # print(difference)

            # test plot for separated fronts
            # plt.figure(figsize=(10,4))
            # plot Cartesian system
            # plt.subplot(1,2,1)
            # plt.plot(x_km_new,y_km_new,'ro')
            # plt.grid()
            # plotTitle = str(month)+"/"+str(day)+"/"+str(year)+" ("+str(z_time)+"Z) - Front #"+str(i) + " (Cartesian)"
            # plt.title(plotTitle)
            # interpolate Cartesian points at an interval
            if len(x_km_new)>1:
                distance = 25 # Cartesian interval distance in km
                xy_linestring = geometric(x_km_new, y_km_new) # turn Cartesian arrays into LineString
                xy_vertices = redistribute_vertices(xy_linestring, distance) # redistribute Cartesian points every 25km
                x_new,y_new = xy_vertices.xy
                # plt.plot(x_new,y_new,'ro') # plot new interpolated points
            # plt.xlabel("Longitudinal distance (km)")
            # plt.ylabel("Latitudinal distance (km)")
            # plt.subplot(1,2,2)
            # plt.plot(x_km_360_new,y_km_new,'bo')
            # plt.grid()
            # plotTitle = str(month)+"/"+str(day)+"/"+str(year)+" ("+str(z_time)+"Z) - Front #"+str(i) + " (360)"
            # plt.title(plotTitle)
            # if len(x_km_360_new)>1:
                # distance = 25 # 360 interval distance in km
                # xy_360_linestring = geometric(x_km_360_new, y_km_new) # turn 360 arrays into LineString
                # xy_360_vertices = redistribute_vertices(xy_360_linestring, distance) # redistribute 360 points every 25km
                # x_360_new,y_360_new = xy_360_vertices.xy
                # plt.plot(x_360_new,y_360_new,'bo') # plot new interpolated points
            # plt.xlabel("Longitudinal distance (km)")
            # plt.ylabel("Latitudinal distance (km)")
            # plt.subplots_adjust(wspace=0.5)
            # plotName = "C:/Users/sling/PycharmProjects/fronts/separated fronts/" + str(month)+"-"+str(day)+"-"+str(year)+" ("+str(z_time)+"Z) - Front #"+str(i)+".png"
            #### WARNING: THIS CODE OUTPUTS HUNDREDS OF IMAGES AND MAY IMPACT SYSTEM PERFORMANCE. ####
            # plt.savefig(plotName,bbox_inches='tight') # Comment this statement out to prevent plots from saving.

            # Create arrays of interpolated points in lat/lon coordinates
            lon_new = []
            lat_new = []
            for a in range(1,len(x_new)):
                dx = x_new[a]-x_new[a-1]
                dy = y_new[a]-y_new[a-1]
                lon1, lon2, lat1, lat2 = reverse_haversine(lons, lats, lon_new, lat_new, front_points, i, a, dx, dy)
                if a==1:
                    lon_new.append(lon1)
                    lat_new.append(lat1)
                lon_new.append(lon2)
                lat_new.append(lat2)

            # test plot for fronts in latitude/longitude coordinate system
            # plt.figure(figsize=(10,4))
            # plt.plot(lon_new,lat_new,'ro')
            # plt.grid()
            # plotTitle = str(month)+"/"+str(day)+"/"+str(year)+" ("+str(z_time)+"Z) - Front #"+str(i) + " (Latitude/Longitude)"
            # plt.title(plotTitle)
            # plt.xlabel("Longitude (degrees)")
            # plt.ylabel("Latitude (degrees)")
            # plotName = "C:/Users/sling/PycharmProjects/fronts/lat-lon front plots/" + str(month)+"-"+str(day)+"-"+str(year)+" ("+str(z_time)+"Z) - Front #"+str(i)+" latlon.png"
            #### WARNING: THIS CODE OUTPUTS HUNDREDS OF IMAGES AND MAY IMPACT SYSTEM PERFORMANCE. ####
            # plt.savefig(plotName,bbox_inches='tight') # Comment this statement out to prevent plots from saving.

            # test plot for all points of every front in the 3H period
            # plt.figure(figsize=(8,4))
            # plot Cartesian system
            # plt.subplot(1,2,1)
            # plt.plot(x_km,y_km,'ro')
            # plt.grid()
            # plotTitle = str(month)+"/"+str(day)+"/"+str(year)+" ("+str(z_time)+"Z) - " + "(Cartesian)"
            # plt.title(plotTitle)
            # plt.xlabel("Longitudinal distance (km)")
            # plt.ylabel("Latitudinal distance (km)")
            # plt.subplot(1,2,2)
            # plt.plot(x_km_360,y_km,'bo')
            # plt.grid()
            # plotTitle = str(month)+"/"+str(day)+"/"+str(year)+" ("+str(z_time)+"Z) - " + "(360)"
            # plt.title(plotTitle)
            # plt.xlabel("Longitudinal distance (km)")
            # plt.ylabel("Latitudinal distance (km)")
            # plotName = "C:/Users/sling/PycharmProjects/fronts/test plots/" + str(month)+"-"+str(day)+"-"+str(year)+" ("+str(z_time)+"Z)"+kind+".png"
            #### WARNING: THIS CODE OUTPUTS HUNDREDS OF IMAGES AND MAY IMPACT SYSTEM PERFORMANCE. ####
            # plt.savefig(plotName,bbox_inches='tight') # Comment this statement out to prevent plots from saving.

            # add interpolated points to arrays and assign their front type
            for c in range(0,len(lon_new)):
                fronts_lon_array.append(lon_new[c])
                period_fronts_lon_array.append(lon_new[c])
                fronts_lat_array.append(lat_new[c])
                period_fronts_lat_array.append(lat_new[c])
                fronts_number.append(frontno)
                period_fronts_number_new.append(frontno)
                front_types.append(frontty)
                period_fronts_type.append(frontty)
                front_dates.append(date)
                period_fronts_date.append(date)

    # 24H dataframe
    df = pd.DataFrame(list(zip(period_fronts_date, period_fronts_number_new, period_fronts_type, period_fronts_lat_array, period_fronts_lon_array)),
        columns=['Date', 'Front Number', 'Front Type', 'Latitude', 'Longitude'])
    df['Latitude'] = df.Latitude.astype(float)
    df['Longitude'] = df.Longitude.astype(float)
    df['Front Number'] = df['Front Number'].astype(int)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d%H')
    xit = np.concatenate((np.linspace(-180, 0, 721), np.linspace(0, 180, 721)[1:721]))
    yit = np.linspace(0, 80, 321)
    df = df.assign(xit=np.digitize(df['Longitude'].values, xit))
    df = df.assign(yit=np.digitize(df['Latitude'].values, yit))
    types = df['Front Type'].unique()
    type_da = []
    for i in range(0, len(types)):
        type_df = df[df['Front Type'] == types[i]]
        groups = type_df.groupby(['xit', 'yit'])
        frequency = np.zeros((yit.shape[0] + 1, xit.shape[0] + 1))
        for groups in groups:
            frequency[groups[1].yit.values[0], groups[1].xit.values[0]] += np.where(
                len(groups[1]['Front Number']) >= 1, 1, 0)
        frequency = frequency[1:322, 1:1443]
        ds = xr.Dataset(data_vars={"Frequency": (['Latitude', 'Longitude'], frequency)},
                        coords={'Latitude': yit, 'Longitude': xit})
        type_da.append(ds)
    ds = xr.concat(type_da, dim=types)
    ds = ds.rename({'concat_dim': 'Type'})
    dss.append(ds)
    timestep = pd.date_range(start=df['Date'].dt.strftime('%Y-%m-%d')[0], periods=1, freq='24H')
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
    # read the polygons for the specified day
    xmls = read_xml_files(args.year, args.month, args.day)

    # One way to output - As a netCDF of just this timestep - the above could then be looped in bash,
    # and xarray used to open the multiple nc files:
    netcdf_outtime = str('%04d%02d%02d' % (args.year, args.month, args.day))
    out_file_name = 'FrontalCounts_%s.nc' % netcdf_outtime
    out_file_path = os.path.join(args.netcdf_outdir, out_file_name)
    xmls.to_netcdf(out_file_path)
