"""
Function that extracts data from a given domain and saves it into a pickle file.

Code written by: Andrew Justin (andrewjustin@ou.edu)
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse
import Plot_ERA5
import xarray as xr
import pickle
import pandas as pd
import create_NC_files as nc
import glob
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def read_xml_files_360(year, month, day):
    """
    Reads the xml files to pull frontal objects.

    Parameters
    ----------
    year: int
    month: int
    day: int

    Returns
    -------
    dns: Dataset
        Xarray dataset containing frontal data organized by date, type, number, and coordinates.

    """
    file_path = "E:/FrontsProjectData/xmls_2006122012-2020061900"
    print(file_path)

    #files = []
    #files.extend(glob.glob("%s/*%04d%02d%02d00f*.xml" % (file_path, year, month, day)))
    #files.extend(glob.glob("%s/*%04d%02d%02d06f*.xml" % (file_path, year, month, day)))
    #files.extend(glob.glob("%s/*%04d%02d%02d12f*.xml" % (file_path, year, month, day)))
    #files.extend(glob.glob("%s/*%04d%02d%02d18f*.xml" % (file_path, year, month, day)))

    files = glob.glob("%s/*%04d%02d%02d*.xml" % (file_path, year, month, day))

    dss = [] # Dataset with front data organized by front type.
    for filename in files:
        print(filename)
        tree = ET.parse(filename, parser=ET.XMLParser(encoding='utf-8'))
        root = tree.getroot()
        date = filename.split('_')[-1].split('.')[0].split('f')[0]
        fronttype = []
        frontnumber = []
        fronts_number = [] # List array is for the interpolated points' front numbers.
        dates = []
        i = 0 # Define counter for front number.
        lats = []
        lons = []
        point_number = 0
        front_points = [] # List that shows which points are in each front.
        front_types = []
        front_dates = []
        fronts_lon_array = []
        fronts_lat_array = []
        for line in root.iter('Line'):
            i = i + 1
            frontno = i
            frontty = line.get("pgenType")

            # Declare "temporary" x and y variables which will be used to add elements to the distance arrays.
            x_km_temp = 0
            y_km_temp = 0

            x_km = []
            y_km = []
            if point_number != 0:
                front_points.append(point_number)
            for point in line.iter('Point'):
                point_number = point_number + 1
                dates.append(date)
                frontnumber.append(frontno)
                fronttype.append(frontty)
                lats.append(float(point.get("Lat")))

                # For longitude - we want to convert these values to a 360 system. This will allow us to properly
                # interpolate across the dateline. If we were to use a longitude domain of -180 to 180 rather than
                # 0 to 360, the interpolated points would wrap around the entire globe.
                if (float(point.get("Lon"))<0):
                    lons.append(float(point.get("Lon"))+360)
                else:
                    lons.append(float(point.get("Lon")))

        # This is the second step in converting longitude to a 360 system. Fronts that cross the prime meridian are
        # only partially corrected due to one side of the front having a negative longitude coordinate. So, we will
        # convert the positive points to the system as well for a consistent adjustment. It is important to note that
        # these fronts that cross the prime meridian (0E/W) will have a range of 0 to 450 rather than 0 to 360.
        # This will NOT affect the interpolation of the fronts.
        for x in range(0,len(front_points)):
            if x==0:
                for y in range(0,front_points[x-1]):
                    if(lons[y]<90 and max(lons[0:front_points[x-1]])>270):
                        lons[y] = lons[y] + 360
            else:
                for y in range(front_points[x-1],front_points[x]):
                    if(lons[y]<90 and max(lons[front_points[x-1]:front_points[x]])>270):
                        lons[y] = lons[y] + 360

        for l in range(1,len(lats)):
            lon1 = lons[l-1]
            lat1 = lats[l-1]
            lon2 = lons[l]
            lat2 = lats[l]
            dlon, dlat, dx, dy = nc.haversine(lon1,lat1,lon2,lat2)
            x_km_temp = x_km_temp + dx
            y_km_temp = y_km_temp + dy
            x_km.append(x_km_temp)
            y_km.append(y_km_temp)

        # Reset the front counter.
        i = 0

        for line in root.iter('Line'):
            frontno = i
            frontty = line.get("pgenType")

            # Create new arrays for holding coordinates from separate fronts.
            x_km_new = []
            y_km_new = []

            i = i + 1
            if i == 1:
                x_km_new = x_km[0:front_points[i-1]-1]
                y_km_new = y_km[0:front_points[i-1]-1]
            elif i < len(front_points):
                x_km_new = x_km[front_points[i-2]-1:front_points[i-1]-1]
                y_km_new = y_km[front_points[i-2]-1:front_points[i-1]-1]
            if len(x_km_new)>1:
                lon_new = []
                lat_new = []
                distance = 25 # Cartesian interval distance in kilometers.
                if (max(x_km_new)>100000 or min(x_km_new)<-100000 or max(y_km_new)>100000 or min(y_km_new)<-100000):
                    print("ERROR: Front %d contains corrupt data points, no points will be interpolated from the front."
                          % i)
                else:
                    xy_linestring = nc.geometric(x_km_new, y_km_new)
                    xy_vertices = nc.redistribute_vertices(xy_linestring, distance)
                    x_new,y_new = xy_vertices.xy
                    for a in range(1,len(x_new)):
                        dx = x_new[a]-x_new[a-1]
                        dy = y_new[a]-y_new[a-1]
                        lon1, lon2, lat1, lat2 = nc.reverse_haversine(lons, lats, lon_new, lat_new, front_points, i, a, dx,
                                                                   dy)
                        if a==1:
                            lon_new.append(lon1)
                            lat_new.append(lat1)
                        lon_new.append(lon2)
                        lat_new.append(lat2)
                    for c in range(0,len(lon_new)):
                        fronts_lon_array.append(lon_new[c])
                        fronts_lat_array.append(lat_new[c])
                        fronts_number.append(frontno)
                        front_types.append(frontty)
                        front_dates.append(date)

        df = pd.DataFrame(list(zip(front_dates, fronts_number, front_types, fronts_lat_array, fronts_lon_array)),
            columns=['Date', 'Front Number', 'Front Type', 'Latitude', 'Longitude'])
        df['Latitude'] = df.Latitude.astype(float)
        df['Longitude'] = df.Longitude.astype(float)
        df['Front Number'] = df['Front Number'].astype(int)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d%H')
        xit = np.linspace(0, 360, 1441)
        yit = np.linspace(80, 0, 321)
        df = df.assign(xit=np.digitize(df['Longitude'].values, xit))
        df = df.assign(yit=np.digitize(df['Latitude'].values, yit))
        types = df['Front Type'].unique()
        type_da = []
        for i in range(0, len(types)):
            type_df = df[df['Front Type'] == types[i]]
            groups = type_df.groupby(['xit', 'yit'])

            # Create variable "frequency" that describes the number of times a specific front type is present in each
            # gridspace.
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

def extract_input_variables(lon, lat, year, month, day, netcdf_ERA5_indir):
    """
    Extract surface data for the specified coordinate domain, year, month, day, and hour.

    Parameters
    ----------
    lon: float (x2)
        Two values that specify the longitude domain in degrees in the 360 coordinate system: lon_MIN lon_MAX
    lat: float (x2)
        Two values that specify the latitude domain in degrees: lat_MIN lat_MAX
    year: int
        Year for the surface data.
    month: int
        Month for the surface data.
    day: int
        Day for the surface data.
    hour: int
        Hour for the surface data.
    netcdf_ERA5_indir: str
        Directory where the ERA5 netCDF files are contained.

    Returns
    -------
    xr_pickle: Dataset
        Xarray dataset containing the surface data for the specified domain.
    filename: str
        Filename for the pickle file containing the surface data for the specified domain.
    """

    ds_2mT, ds_2mTd, ds_sp, ds_U10m, ds_V10m, ds_wind, da_theta_w = Plot_ERA5.create_datasets(year, month, day, netcdf_ERA5_indir)

    ds_2mT = ds_2mT.sel(longitude=slice(lon[0],lon[1]), latitude=slice(lat[1],lat[0]))
    ds_2mTd = ds_2mTd.sel(longitude=slice(lon[0],lon[1]), latitude=slice(lat[1],lat[0]))
    ds_sp = ds_sp.sel(longitude=slice(lon[0],lon[1]), latitude=slice(lat[1],lat[0]))
    ds_U10m = ds_U10m.sel(longitude=slice(lon[0],lon[1]), latitude=slice(lat[1],lat[0]))
    ds_V10m = ds_V10m.sel(longitude=slice(lon[0],lon[1]), latitude=slice(lat[1],lat[0]))
    ds_theta_w = (da_theta_w.sel(longitude=slice(lon[0],lon[1]), latitude=slice(lat[1],lat[0]))).to_dataset(name=
                                                                                                            'theta_w')

    ds_pickle = [ds_2mT, ds_2mTd, ds_sp, ds_U10m, ds_V10m, ds_theta_w]
    xr_pickle = xr.combine_by_coords(ds_pickle, combine_attrs='override')

    return xr_pickle

def save_sfcdata_conus_to_pickle(year, month, day, hour, xr_pickle, pickle_outdir):
    """
    Saves surface domain data to the pickle file.

    Parameters
    ----------
    xr_pickle: Dataset
        Xarray dataset containing the surface data for the specified domain.
    filename: str
        Filename for the pickle file containing the surface data for the specified domain.
    pickle_outdir: str
        Directory where the created pickle files containing the domain data will be stored.
    """

    xr_pickle = xr_pickle.sel(time='%d-%02d-%02dT%02d:00:00' % (year, month, day, hour))

    filename = "SurfaceData_%04d%02d%02d%02d_conus.pkl" % (year, month, day, hour)

    print(filename)

    outfile = open("%s/%d/%02d/%02d/%s" % (pickle_outdir, year, month, day, filename), 'wb')
    pickle.dump(xr_pickle, outfile)
    outfile.close()

def save_sfcdata_window_to_pickle(year, month, day, hour, xr_pickle, longitude, latitude, pickle_outdir):

    longitudes = np.linspace(longitude[0],longitude[1],10)
    latitudes = np.linspace(latitude[0],latitude[1],6)

    xr_pickle = xr_pickle.sel(time='%d-%02d-%02dT%02d:00:00' % (year, month, day, hour))

    for i in range(len(longitudes)-1):
        for j in range(len(latitudes)-1):
            xr_pickle_window = xr_pickle.sel(longitude=slice(longitudes[i],longitudes[i+1]),
                                             latitude=slice(latitudes[j+1],latitudes[j]))

            filename = "SurfaceData_%04d%02d%02d%02d_lon(%d_%d)_lat(%d_%d).pkl" % (year, month, day, hour,
                                                                                   longitudes[i], longitudes[i+1],
                                                                                   latitudes[j], latitudes[j+1])

            print(filename)

            outfile = open("%s/%d/%02d/%02d/%s" % (pickle_outdir, year, month, day, filename), 'wb')
            pickle.dump(xr_pickle, outfile)
            outfile.close()

def save_fronts_conus_to_pickle(ds, year, month, day, hour, pickle_outdir):

    xr_pickle = ds.sel(time='%d-%02d-%02dT%02d:00:00' % (year, month, day, hour))

    fronttype = np.empty([len(xr_pickle.latitude),len(xr_pickle.longitude)])

    time = xr_pickle.time
    frequency = xr_pickle.Frequency.values
    types = xr_pickle.type.values
    lats = xr_pickle.latitude.values
    lons = xr_pickle.longitude.values

    for i in range(len(lats)):
        for j in range(len(lons)):
            for k in range(len(types)):
                if types[k]=='COLD_FRONT':
                    if (frequency[k][i][j]>0):
                        fronttype[i][j] = 1
                    else:
                        fronttype[i][j] = 0
                elif types[k]=='WARM_FRONT':
                    if (frequency[k][i][j]>0):
                        fronttype[i][j] = 2

    xr_pickle = xr.Dataset({"identifier": (('latitude','longitude'), fronttype)},
                           coords={"latitude": lats, "longitude": lons, "time": time})

    filename = "FrontObjects_%04d%02d%02d%02d_conus.pkl" % (year, month, day, hour)

    print(filename)

    outfile = open("%s/%d/%02d/%02d/%s" % (pickle_outdir, year, month, day, filename), 'wb')
    pickle.dump(xr_pickle, outfile)
    outfile.close()

def save_fronts_window_to_pickle(ds, year, month, day, hour, longitude, latitude, pickle_outdir):

    longitudes = np.linspace(longitude[0],longitude[1],10)
    latitudes = np.linspace(latitude[0],latitude[1],6)

    xr_pickle = ds.sel(time='%d-%02d-%02dT%02d:00:00' % (year, month, day, hour))

    fronttype = np.empty([len(xr_pickle.latitude),len(xr_pickle.longitude)])

    time = xr_pickle.time
    frequency = xr_pickle.Frequency.values
    types = xr_pickle.type.values
    lats = xr_pickle.latitude.values
    lons = xr_pickle.longitude.values

    for i in range(len(lats)):
        for j in range(len(lons)):
            for k in range(len(types)):
                if types[k]=='COLD_FRONT':
                    if (frequency[k][i][j]>0):
                        fronttype[i][j] = 1
                    else:
                        fronttype[i][j] = 0
                elif types[k]=='WARM_FRONT':
                    if (frequency[k][i][j]>0):
                        fronttype[i][j] = 2

    xr_pickle = xr.Dataset({"identifier": (('latitude','longitude'), fronttype)},
                           coords={"latitude": lats, "longitude": lons, "time": time})

    for i in range(len(longitudes)-1):
        for j in range(len(latitudes)-1):
            xr_pickle_window = xr_pickle.sel(longitude=slice(longitudes[i],longitudes[i+1]),
                                             latitude=slice(latitudes[j+1],latitudes[j]))

            filename = "FrontObjects_%04d%02d%02d%02d_lon(%d_%d)_lat(%d_%d).pkl" % (year, month, day, hour,
                                                                                    longitudes[i], longitudes[i+1],
                                                                                    latitudes[j], latitudes[j+1])

            print(filename)

            outfile = open("%s/%d/%02d/%02d/%s" % (pickle_outdir, year, month, day, filename), 'wb')
            pickle.dump(xr_pickle_window, outfile)
            outfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--netcdf_ERA5_indir', type=str, required=True, help="input directory for ERA5 netcdf files")
    parser.add_argument('--pickle_outdir', type=str, required=True, help="output directory for pickle files")
    parser.add_argument('--longitude', type=float, nargs=2, help="Longitude domain in degrees: lon_MIN lon_MAX")
    parser.add_argument('--latitude', type=float, nargs=2, help="Latitude domain in degrees: lat_MIN lat_MAX")
    parser.add_argument('--year', type=int, required=True, help="year for the data to be read in")
    parser.add_argument('--month', type=int, required=True, help="month for the data to be read in")
    parser.add_argument('--day', type=int, required=True, help="day for the data to be read in")
    args = parser.parse_args()

    xr_pickle = extract_input_variables(args.longitude, args.latitude, args.year, args.month, args.day, args.netcdf_ERA5_indir)
    ds = read_xml_files_360(args.year, args.month, args.day)

    for hour in range(0,24,3):

        ds_hour = ds.sel(Longitude=slice(args.longitude[0],args.longitude[1]), Latitude=slice(args.latitude[1],
                                                                                         args.latitude[0]))
        ds_hour = ds_hour.rename(Latitude='latitude', Longitude='longitude', Type='type', Date='time')

        #save_sfcdata_conus_to_pickle(args.year, args.month, args.day, hour, xr_pickle, args.pickle_outdir)
        #save_sfcdata_window_to_pickle(args.year, args.month, args.day, hour, xr_pickle, args.longitude, args.latitude,
        #                              args.pickle_outdir)

        save_fronts_conus_to_pickle(ds_hour, args.year, args.month, args.day, hour, args.pickle_outdir)
        #save_fronts_window_to_pickle(ds_hour, args.year, args.month, args.day, hour, args.longitude, args.latitude,
        #                             args.pickle_outdir)
