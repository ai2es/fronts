"""
Convert front XML files to netCDF files.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 7/5/2023 8:32 AM CT
"""

import argparse
import glob
import numpy as np
import os
from utils import data_utils
import xarray as xr
import xml.etree.ElementTree as ET

pgenType_identifiers = {'COLD_FRONT': 1, 'WARM_FRONT': 2, 'STATIONARY_FRONT': 3, 'OCCLUDED_FRONT': 4, 'COLD_FRONT_FORM': 5,
                        'WARM_FRONT_FORM': 6, 'STATIONARY_FRONT_FORM': 7, 'OCCLUDED_FRONT_FORM': 8, 'COLD_FRONT_DISS': 9,
                        'WARM_FRONT_DISS': 10, 'STATIONARY_FRONT_DISS': 11, 'OCCLUDED_FRONT_DISS': 12, 'INSTABILITY': 13,
                        'TROF': 14, 'TROPICAL_TROF': 15, 'DRY_LINE': 16}

"""
conus: 132 W to 60.25 W, 57 N to 26.25 N
full: 130 E pointing eastward to 10 E, 80 N to 0.25 N
global: 179.75 W to 180 E, 90 N to 89.75 N
"""
domain_coords = {'conus': {'lons': np.arange(-132, -60, 0.25), 'lats': np.arange(57, 25, -0.25)},
                 'full': {'lons': np.append(np.arange(130, 180.25, 0.25), np.arange(-179.75, 10, 0.25)), 'lats': np.arange(80, 0, -0.25)},
                 'global': {'lons': np.arange(-179.75, 180.25, 0.25), 'lats': np.arange(90, -90, -0.25)}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_indir', type=str, required=True, help="Input directory for front XML files.")
    parser.add_argument('--netcdf_outdir', type=str, required=True, help="Output directory for front netCDF files.")
    parser.add_argument('--date', type=int, nargs=3, required=True, help="Date for the data to be read in. (year, month, day)")
    parser.add_argument('--distance', type=int, default=1, help="Interpolation distance in kilometers.")
    parser.add_argument('--domain', type=str, default='full', help="Domain for which to generate fronts.")

    args = vars(parser.parse_args())

    year, month, day = args['date']

    if args['domain'] == 'global':
        files = sorted(glob.glob("%s/sfcsam*_%04d%02d%02d*.xml" % (args['xml_indir'], year, month, day)))
    else:
        files = sorted(glob.glob("%s/pres*_%04d%02d%02d*f000.xml" % (args['xml_indir'], year, month, day)))

    gridded_lons = domain_coords[args['domain']]['lons'].astype('float32')
    gridded_lats = domain_coords[args['domain']]['lats'].astype('float32')

    for filename in files:

        identifier = np.zeros([len(gridded_lons), len(gridded_lats)]).astype('float32')

        tree = ET.parse(filename, parser=ET.XMLParser(encoding='utf-8'))
        root = tree.getroot()
        date = filename.split('_')[-1].split('.')[0].split('f')[0]  # YYYYMMDDhh

        hour = date[-2:]

        ### Iterate through the individual fronts ###
        for line in root.iter('Line'):
            type_of_front = line.get("pgenType")  # front type
            lons, lats = zip(*[[float(point.get("Lon")), float(point.get("Lat"))] for point in line.iter('Point')])
            lons, lats = np.array(lons), np.array(lats)

            # If the front crosses the dateline or the 180th meridian, its coordinates must be modified for proper interpolation
            front_needs_modification = np.max(np.abs(np.diff(lons))) > 180

            if front_needs_modification:
                lons = np.where(lons < 0, lons + 360, lons)  # convert coordinates to a 360 degree system

            xs, ys = data_utils.haversine(lons, lats)  # x/y coordinates in kilometers
            xy_linestring = data_utils.geometric(xs, ys)  # convert coordinates to a LineString object
            x_new, y_new = data_utils.redistribute_vertices(xy_linestring, args['distance']).xy  # interpolate x/y coordinates
            x_new, y_new = np.array(x_new), np.array(y_new)
            lon_new, lat_new = data_utils.reverse_haversine(x_new, y_new)  # convert interpolated x/y coordinates to lat/lon

            if front_needs_modification:
                lon_new = np.where(lon_new > 180, lon_new - 360, lon_new)  # convert new longitudes to standard -180 to 180 range

            gridded_indices = np.dstack((np.digitize(lon_new, gridded_lons), np.digitize(lat_new, gridded_lats)))[0]  # translate coordinate indices to grid
            gridded_indices_unique = np.unique(gridded_indices, axis=0)  # remove duplicate coordinate indices

            identifier[gridded_indices_unique[:, 0], gridded_indices_unique[:, 1]] = pgenType_identifiers[type_of_front]  # assign labels to the gridded points based on the front type

        date_and_time = np.datetime64('%d-%02d-%02dT%s' % (year, month, day, hour))

        filename_netcdf = "FrontObjects_%s_%s.nc" % (date, args['domain'])
        fronts_ds = xr.Dataset({"identifier": (('longitude', 'latitude'), identifier)},
                               coords={"longitude": gridded_lons, "latitude": gridded_lats}).expand_dims({'time': np.atleast_1d(date_and_time)})

        if not os.path.isdir("%s/%d%02d" % (args['netcdf_outdir'], year, month)):
            os.mkdir("%s/%d%02d" % (args['netcdf_outdir'], year, month))
        fronts_ds.to_netcdf(path="%s/%d%02d/%s" % (args['netcdf_outdir'], year, month, filename_netcdf), engine='netcdf4', mode='w')
