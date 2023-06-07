"""
Convert front XML files to netCDF files.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 6/7/2023 6:40 PM CT
"""

import argparse
import glob
import numpy as np
import os
import pandas as pd
from utils import data_utils
import xarray as xr
import xml.etree.ElementTree as ET


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_indir', type=str, required=True, help="Input directory for front XML filfes.")
    parser.add_argument('--netcdf_outdir', type=str, required=True, help="Output directory for front netCDF files.")
    parser.add_argument('--date', type=int, nargs=3, required=True, help="Date for the data to be read in. (year, month, day)")

    args = vars(parser.parse_args())

    year, month, day = args['date'][0], args['date'][1], args['date'][2]

    files = sorted(glob.glob("%s/*_%04d%02d%02d*f000.xml" % (args['xml_indir'], year, month, day)))

    dss = []  # Dataset with front data organized by front type.
    timesteps = []
    for filename in files:
        print(filename)
        tree = ET.parse(filename, parser=ET.XMLParser(encoding='utf-8'))
        root = tree.getroot()
        date = filename.split('_')[-1].split('.')[0].split('f')[0]
        fronttype = []
        frontnumber = []
        fronts_number = []  # List array is for the interpolated points' front numbers.
        dates = []
        i = 0  # Define counter for front number.
        lats = []
        lons = []
        point_number = 0
        front_points = []  # List that shows which points are in each front.
        front_types = []
        front_dates = []
        fronts_lon_array = []
        fronts_lat_array = []
        for line in root.iter('Line'):
            i = i + 1
            frontno = i
            frontty = line.get("pgenType")
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
                if float(point.get("Lon")) < 0:
                    lons.append(float(point.get("Lon")) + 360)
                else:
                    lons.append(float(point.get("Lon")))

        # This is the second step in converting longitude to a 360 system. Fronts that cross the prime meridian are
        # only partially corrected due to one side of the front having a negative longitude coordinate. So, we will
        # convert the positive points to the system as well for a consistent adjustment. It is important to note that
        # these fronts that cross the prime meridian (0E/W) will have a range of 0 to 450 rather than 0 to 360.
        # This will NOT affect the interpolation of the fronts.
        for x in range(len(front_points)):
            if x == 0:
                for y in range(0, front_points[x - 1]):
                    if lons[y] < 90 and max(lons[0:front_points[x - 1]]) > 270:
                        lons[y] = lons[y] + 360
            else:
                for y in range(front_points[x - 1], front_points[x]):
                    if lons[y] < 90 and max(lons[front_points[x - 1]:front_points[x]]) > 270:
                        lons[y] = lons[y] + 360

        xs, ys = data_utils.haversine(lons, lats)

        # Reset the front counter.
        i = 0

        for line in root.iter('Line'):
            frontno = i
            frontty = line.get("pgenType")

            if i == 0:
                xs_new = xs[0:front_points[i]]
                ys_new = ys[0:front_points[i]]
            elif i < len(front_points):
                xs_new = xs[front_points[i - 1]:front_points[i]]
                ys_new = ys[front_points[i - 1]:front_points[i]]
            else:
                xs_new = xs[front_points[i - 1]:]
                ys_new = ys[front_points[i - 1]:]

            if len(xs_new) == 1:
                xs_new = np.append(xs_new, xs_new)
                ys_new = np.append(ys_new, ys_new)

            distance = 1  # Cartesian interval distance in kilometers.
            if (max(xs_new) > 100000 or min(xs_new) < -100000 or max(ys_new) > 100000 or min(
                    ys_new) < -100000):
                print(f"{filename}:   -----> Corrupt coordinates for front %d" % (frontno + 1))
            else:
                xy_linestring = data_utils.geometric(xs_new, ys_new)
                xy_vertices = data_utils.redistribute_vertices(xy_linestring, distance)
                x_new, y_new = xy_vertices.xy
                x_new = np.array(x_new)
                y_new = np.array(y_new)
                lon_new, lat_new = data_utils.reverse_haversine(x_new, y_new)
                for c in range(0, len(lon_new)):
                    fronts_lon_array.append(lon_new[c])
                    fronts_lat_array.append(lat_new[c])
                    fronts_number.append(frontno)
                    front_types.append(frontty)
                    front_dates.append(date)

            i += 1

        df = pd.DataFrame(list(zip(front_dates, fronts_number, front_types, fronts_lat_array, fronts_lon_array)),
                          columns=['time', 'Front Number', 'Front Type', 'latitude', 'longitude'])
        df['latitude'] = df.latitude.astype(float)
        df['longitude'] = df.longitude.astype(float)
        df['Front Number'] = df['Front Number'].astype(int)
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H')
        xit = np.linspace(0, 360, 1441)
        yit = np.linspace(80, 0, 321)
        df = df.assign(xit=np.digitize(df['longitude'].values, xit))
        df = df.assign(yit=np.digitize(df['latitude'].values, yit))
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
            ds = xr.Dataset(data_vars={"Frequency": (('latitude', 'longitude'), frequency)},
                            coords={'latitude': yit, 'longitude': xit})
            type_da.append(ds)
        try:
            ds = xr.concat(type_da, dim=types)
        except ValueError:
            print("Skipping bad file: %s" % filename)
            continue
        ds = ds.rename({'concat_dim': 'type'})
        dss.append(ds)
        timesteps.append(pd.to_datetime(df['time'][0], format='%Y%m%d%H'))
    dns = xr.concat(dss, dim=timesteps)
    dns = dns.rename({'concat_dim': 'time'})

    for hour in range(0, 24, 3):
        try:
            fronts = dns.sel(time='%d-%02d-%02dT%02d:00:00' % (year, month, day, hour))
            fronts = fronts.sel(longitude=np.append(np.arange(130, 360, 0.25), np.arange(0, 10.25, 0.25)))
            fronts['longitude'] = np.arange(130, 370.25, 0.25)
        except KeyError:
            print(f"ERROR: no fronts available for {year}-%02d-%02d-%02dz" % (month, day, hour))
        else:
            print(f"saving fronts for {year}-%02d-%02d-%02dz" % (month, day, hour))
            fronttype = np.empty([len(fronts.latitude), len(fronts.longitude)])
            time = fronts.time
            frequency = fronts.Frequency.values
            types = fronts.type.values
            lats = fronts.latitude.values
            lons = fronts.longitude.values

            for i in range(len(lats)):
                for j in range(len(lons)):
                    for k in range(len(types)):
                        if types[k] == 'COLD_FRONT':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 1
                            else:
                                fronttype[i][j] = 0
                        elif types[k] == 'WARM_FRONT':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 2
                        elif types[k] == 'STATIONARY_FRONT':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 3
                        elif types[k] == 'OCCLUDED_FRONT':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 4

                        elif types[k] == 'COLD_FRONT_FORM':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 5
                        elif types[k] == 'WARM_FRONT_FORM':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 6
                        elif types[k] == 'STATIONARY_FRONT_FORM':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 7
                        elif types[k] == 'OCCLUDED_FRONT_FORM':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 8

                        elif types[k] == 'COLD_FRONT_DISS':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 9
                        elif types[k] == 'WARM_FRONT_DISS':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 10
                        elif types[k] == 'STATIONARY_FRONT_DISS':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 11
                        elif types[k] == 'OCCLUDED_FRONT_DISS':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 12

                        elif types[k] == 'INSTABILITY':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 13
                        elif types[k] == 'TROF':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 14
                        elif types[k] == 'TROPICAL_TROF':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 15
                        elif types[k] == 'DRY_LINE':
                            if frequency[k][i][j] > 0:
                                fronttype[i][j] = 16

            filename_netcdf = "FrontObjects_%04d%02d%02d%02d_full.nc" % (year, month, day, hour)
            fronts_ds = xr.Dataset({"identifier": (('latitude', 'longitude'), fronttype)}, coords={"latitude": lats, "longitude": lons, "time": time}).astype('float32')

            if not os.path.isdir("%s/%d%02d" % (args['netcdf_outdir'], year, month)):
                os.mkdir("%s/%d%02d" % (args['netcdf_outdir'], year, month))
            fronts_ds.to_netcdf(path="%s/%d%02d/%s" % (args['netcdf_outdir'], year, month, filename_netcdf), engine='netcdf4', mode='w')
