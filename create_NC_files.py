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

# Haversine function - turns lats/lons into x/y and calculates distance between points
def haversine(lon1, lat1, lon2, lat2):
    # haversine formula for Cartesian system
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = math.sin(dlat/2)**2+math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    dx = (-dlon)*40075*math.cos((lat1+lat2)*math.pi/360)/360 # circumference of earth in km = 40075
    dy = (-dlat)*40075/360
    c = 2*math.asin(math.sqrt(a))
    km = 6371*c # radius of earth in km = 6371
    return dlon, dlat, dx, dy

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
    file_path = "C:/Users/sling/PycharmProjects/fronts/xmls_2006122012-2020061900"
    print(file_path)
    # read all the files in the directory
    # files = glob.glob("%s/*%04d%02d%02d*.xml" % (file_path, year, month, day))

    # code for grabbing files every 6H starting at 0Z #
    files = []
    files.extend(glob.glob("%s/*%04d%02d%02d00f*.xml" % (file_path, year, month, day)))
    files.extend(glob.glob("%s/*%04d%02d%02d06f*.xml" % (file_path, year, month, day)))
    files.extend(glob.glob("%s/*%04d%02d%02d12f*.xml" % (file_path, year, month, day)))
    files.extend(glob.glob("%s/*%04d%02d%02d18f*.xml" % (file_path, year, month, day)))
    # print(files)

    # code for grabbing files every 6H starting at 3Z #
    # files = []
    # files.extend(glob.glob("%s/*%04d%02d%02d03f*.xml" % (file_path, year, month, day)))
    # files.extend(glob.glob("%s/*%04d%02d%02d09f*.xml" % (file_path, year, month, day)))
    # files.extend(glob.glob("%s/*%04d%02d%02d15f*.xml" % (file_path, year, month, day)))
    # files.extend(glob.glob("%s/*%04d%02d%02d21f*.xml" % (file_path, year, month, day)))
    # print(files)

    dss = []
    # Counter for timer number
    fileNo = 1
    # Initialize x and y distance values for the haversine function.
    dx = 0
    dy = 0
    # create arrays for fronts in different periods
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
                if (float(point.get("Lon"))<0):
                    lons.append(float(point.get("Lon"))+360)
                else:
                    lons.append(float(point.get("Lon")))

        # Correct point errors in the fronts. Some points may not have been converted to 360 coordinates in the initial assignments.
        for x in range(0,len(front_points)):
            if x==0:
                for y in range(0,front_points[x-1]):
                    if(lons[y]<90 and max(lons[0:front_points[x-1]])>270):
                        lons[y] = lons[y] + 360
            else:
                for y in range(front_points[x-1],front_points[x]):
                    if(lons[y]<90 and max(lons[front_points[x-1]:front_points[x]])>270):
                        lons[y] = lons[y] + 360

        # create x and y coordinate arrays
        for l in range(1,len(lats)):
            lon1 = lons[l-1]
            lat1 = lats[l-1]
            lon2 = lons[l]
            lat2 = lats[l]
            dlon, dlat, dx, dy = haversine(lon1,lat1,lon2,lat2)
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
            # Create new arrays for separate fronts
            x_km_new = []
            y_km_new = []
            i = i + 1
            if i == 1:
                x_km_new = x_km[0:front_points[i-1]-1]
                y_km_new = y_km[0:front_points[i-1]-1]
            elif i < len(front_points):
                x_km_new = x_km[front_points[i-2]-1:front_points[i-1]-1]
                y_km_new = y_km[front_points[i-2]-1:front_points[i-1]-1]

            # test plot for separated fronts
            #plt.figure(figsize=(10,4))
            # plot Cartesian system
            #plt.plot(x_km_new,y_km_new,'ro')
            #plt.grid()
            #plotTitle = str(month)+"/"+str(day)+"/"+str(year)+" ("+str(z_time)+"Z) - Front #"+str(i) + " (Cartesian)"
            # plt.title(plotTitle)
            # interpolate Cartesian points at an interval
            if len(x_km_new)>1:
                # Create arrays of interpolated points in lat/lon coordinates
                lon_new = []
                lat_new = []
                distance = 25 # Cartesian interval distance in km
                # Check to see whether points in the array are erroneous. Erroneous points can be caused by invalid
                # latitude/longitude coordinates in the data files.
                if (max(x_km_new)>100000 or min(x_km_new)<-100000 or max(y_km_new)>100000 or min(y_km_new)<-100000):
                    print("ERROR: Front %d contains corrupt data points, no points will be interpolated from this front." % i)
                else:
                    xy_linestring = geometric(x_km_new, y_km_new) # turn Cartesian arrays into LineString
                    xy_vertices = redistribute_vertices(xy_linestring, distance) # redistribute Cartesian points every 25km
                    x_new,y_new = xy_vertices.xy
                    # Convert distances back to lat/lon coordinates
                    for a in range(1,len(x_new)):
                        dx = x_new[a]-x_new[a-1]
                        dy = y_new[a]-y_new[a-1]
                        lon1, lon2, lat1, lat2 = reverse_haversine(lons, lats, lon_new, lat_new, front_points, i, a, dx, dy)
                        if a==1:
                            lon_new.append(lon1)
                            lat_new.append(lat1)
                        lon_new.append(lon2)
                        lat_new.append(lat2)
                    # Assign interpolated points to arrays
                    for c in range(0,len(lon_new)):
                        if (lon_new[c]>179.999):
                            fronts_lon_array.append(lon_new[c]-360)
                        else:
                            fronts_lon_array.append(lon_new[c])
                        fronts_lat_array.append(lat_new[c])
                        fronts_number.append(frontno)
                        front_types.append(frontty)
                        front_dates.append(date)

                #    plt.plot(x_new,y_new,'ro') # plot new interpolated points
                # plt.xlabel("Longitudinal distance (km)")
                # plt.ylabel("Latitudinal distance (km)")
                # plotName = "E:/FrontsProjectData/separated_fronts_cartesian/" + str(month)+"-"+str(day)+"-"+str(year)+" ("+str(z_time)+"Z) - Front #"+str(i)+".png"
                #### WARNING: THIS CODE OUTPUTS HUNDREDS OF IMAGES AND MAY IMPACT SYSTEM PERFORMANCE. ####
                # plt.savefig(plotName,bbox_inches='tight') # Comment this statement out to prevent plots from saving.



                # test plot for fronts in latitude/longitude coordinate system
                # plt.figure(figsize=(10,4))
                # plt.plot(lon_new,lat_new,'ro')
                # plt.grid()
                # plotTitle = str(month)+"/"+str(day)+"/"+str(year)+" ("+str(z_time)+"Z) - Front #"+str(i) + " (Latitude/Longitude)"
                # plt.title(plotTitle)
                # plt.xlabel("Longitude (degrees)")
                # plt.ylabel("Latitude (degrees)")
                # plotName = "C:/Users/sling/PycharmProjects/fronts/separated_fronts" + str(month)+"-"+str(day)+"-"+str(year)+" ("+str(z_time)+"Z) - Front #"+str(i)+" latlon.png"
                #### WARNING: THIS CODE OUTPUTS HUNDREDS OF IMAGES AND MAY IMPACT SYSTEM PERFORMANCE. ####
                # plt.savefig(plotName,bbox_inches='tight') # Comment this statement out to prevent plots from saving.

                # test plot for all points of every front in the 3H period
                # plt.figure(figsize=(8,4))
                # plot Cartesian system
                # plt.plot(x_km,y_km,'ro')
                # plt.grid()
                # plotTitle = str(month)+"/"+str(day)+"/"+str(year)+" ("+str(z_time)+"Z) - " + "(Cartesian)"
                # plt.title(plotTitle)
                # plt.xlabel("Longitudinal distance (km)")
                # plt.ylabel("Latitudinal distance (km)")
                # plotName = "C:/Users/sling/PycharmProjects/fronts/3H_plots_cartesian/" + str(month)+"-"+str(day)+"-"+str(year)+" ("+str(z_time)+"Z)"+kind+".png"
                #### WARNING: THIS CODE OUTPUTS HUNDREDS OF IMAGES AND MAY IMPACT SYSTEM PERFORMANCE. ####
                # plt.savefig(plotName,bbox_inches='tight') # Comment this statement out to prevent plots from saving.

                # add interpolated points to arrays and assign their front type, number, and date

        # 3H dataframe
        df = pd.DataFrame(list(zip(front_dates, fronts_number, front_types, fronts_lat_array, fronts_lon_array)),
            columns=['Date', 'Front Number', 'Front Type', 'Latitude', 'Longitude'])
        df['Latitude'] = df.Latitude.astype(float)
        df['Longitude'] = df.Longitude.astype(float)
        df['Front Number'] = df['Front Number'].astype(int)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d%H')
        xit = np.linspace(-180, 180, 1441)
        yit = np.linspace(0, 80, 321)
        df = df.assign(xit=np.digitize(df['Longitude'].values, xit))
        df = df.assign(yit=np.digitize(df['Latitude'].values, yit))
        types = df['Front Type'].unique()
        type_da = []
        for i in range(0, len(types)):
            type_df = df[df['Front Type'] == types[i]]
            groups = type_df.groupby(['xit', 'yit'])
            frequency = np.zeros((yit.shape[0] + 1, xit.shape[0] + 1))
            for group in groups:
                frequency[group[1].yit.values[0], group[1].xit.values[0]] += np.where(
                    len(group[1]['Front Number']) >= 1, 1, 0)
            frequency = frequency[1:322, 1:1442]
            ds = xr.Dataset(data_vars={"Frequency": (['Latitude', 'Longitude'], frequency)},
                            coords={'Latitude': yit, 'Longitude': xit})
            type_da.append(ds)
        ds = xr.concat(type_da, dim=types)
        ds = ds.rename({'concat_dim': 'Type'})
        dss.append(ds)
    timestep = pd.date_range(start=df['Date'].dt.strftime('%Y-%m-%d')[0], periods=len(dss), freq='3H')
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
