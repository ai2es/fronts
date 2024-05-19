"""
Convert front XML files to netCDF files.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.5.14
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
                 'full': {'lons': np.append(np.arange(-179.75, 10, 0.25), np.arange(130, 180.25, 0.25)), 'lats': np.arange(80, 0, -0.25)},
                 'global': {'lons': np.arange(-179.75, 180.25, 0.25), 'lats': np.arange(90, -90, -0.25)}}
domain_lons_360 = {'conus': np.arange(228, 300, 0.25),
                   'full': np.arange(130, 370, 0.25),
                   'global': np.arange(0, 360, 0.25)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_indir', type=str, required=True, help="Input directory for front XML files.")
    parser.add_argument('--netcdf_outdir', type=str, required=True, help="Output directory for front netCDF files.")
    parser.add_argument('--date', type=str, required=True, help="Date for the data to be read in. YYYY-MM-DD")
    parser.add_argument('--distance', type=float, default=1., help="Interpolation distance in kilometers.")
    parser.add_argument('--domain', type=str, default='full', help="Domain for which to generate fronts.")

    args = vars(parser.parse_args())

    date = np.datetime64(args['date']).astype(object)
    year, month, day = date.year, date.month, date.day

    if args['domain'] == 'global':
        files = sorted(glob.glob("%s/IBM*_%04d%02d%02d*f*.xml" % (args['xml_indir'], year, month, day)))
    else:
        files = sorted(glob.glob("%s/pres*_%04d%02d%02d*f000.xml" % (args['xml_indir'], year, month, day)))

    domain_from_model = args['domain'] not in ['conus', 'full', 'global']

    for filename in files:

        if domain_from_model:

            model_coords_ds = xr.open_dataset('./coordinates/%s.nc' % args['domain'])

            # transform model's coordinates to a cartesian grid
            transform_args = dict(std_parallels=model_coords_ds.attrs['std_parallels'], lon_ref=model_coords_ds.attrs['lon_ref'], lat_ref=model_coords_ds.attrs['lat_ref'])
            gridded_lons = model_coords_ds['longitude'].values.astype('float32')
            gridded_lats = model_coords_ds['latitude'].values.astype('float32')
            model_x_transform, model_y_transform = data_utils.lambert_conformal_to_cartesian(gridded_lons, gridded_lats, **transform_args)
            gridded_x = model_x_transform[0, :]
            gridded_y = model_y_transform[:, 0]
            identifier = np.zeros(np.shape(gridded_lons)).astype('float32')

            # bounds of the model's cartesian domain
            model_x_min, model_x_max = np.min(model_x_transform), np.max(model_x_transform)
            model_y_min, model_y_max = np.min(model_y_transform), np.max(model_y_transform)

        else:

            gridded_lons = domain_coords[args['domain']]['lons'].astype('float32')
            gridded_lats = domain_coords[args['domain']]['lats'].astype('float32')
            identifier = np.zeros([len(gridded_lons), len(gridded_lats)]).astype('float32')

        tree = ET.parse(filename, parser=ET.XMLParser(encoding='utf-8'))
        root = tree.getroot()
        date = os.path.basename(filename).split('_')[-1].split('.')[0].split('f')[0]  # YYYYMMDDhh
        forecast_hour = int(filename.split('f')[-1].split('.')[0])

        hour = date[-2:]

        if hour in ["03", "09", "15", "21"] and args['domain'] == "nam-12km":
            continue

        ### Iterate through the individual fronts ###
        for line in root.iter('Line'):

            type_of_front = line.get("pgenType")  # front type

            lons, lats = zip(*[[float(point.get("Lon")), float(point.get("Lat"))] for point in line.iter('Point')])
            lons, lats = np.array(lons), np.array(lats)

            # If the front crosses the dateline or the 180th meridian, its coordinates must be modified for proper interpolation
            front_needs_modification = np.max(np.abs(np.diff(lons))) > 180

            if front_needs_modification or domain_from_model:
                lons = np.where(lons < 0, lons + 360, lons)  # convert coordinates to a 360 degree system

                if domain_from_model:
                    x_transform_init, y_transform_init = data_utils.lambert_conformal_to_cartesian(lons, lats, **transform_args)

                    # find points outside the model domain
                    points_outside_domain = np.where((x_transform_init < model_x_min) | (x_transform_init > model_x_max) | (y_transform_init < model_y_min) | (y_transform_init > model_y_max))

                    if len(points_outside_domain) > 0:  # remove points outside the model domain
                        lons = np.delete(lons, points_outside_domain)
                        lats = np.delete(lats, points_outside_domain)

            if len(lons) < 2:  # do not generate front if there are not at least two points
                continue

            xs, ys = data_utils.haversine(lons, lats)  # x/y coordinates in kilometers
            xy_linestring = data_utils.geometric(xs, ys)  # convert coordinates to a LineString object
            x_new, y_new = data_utils.redistribute_vertices(xy_linestring, args['distance']).xy  # interpolate x/y coordinates
            x_new, y_new = np.array(x_new), np.array(y_new)
            lon_new, lat_new = data_utils.reverse_haversine(x_new, y_new)  # convert interpolated x/y coordinates to lat/lon

            date_and_time = np.datetime64('%04d-%02d-%02dT%02d' % (year, month, day, int(hour)), 'ns')

            expand_dims_args = {'time': np.atleast_1d(date_and_time)}

            if args['domain'] == 'global' or domain_from_model:
                expand_dims_args['forecast_hour'] = np.atleast_1d(forecast_hour)
                filename_netcdf = "FrontObjects_%s_f%03d_%s.nc" % (date, forecast_hour, args['domain'])
            else:
                filename_netcdf = "FrontObjects_%s_%s.nc" % (date, args['domain'])

            if domain_from_model:

                x_new *= 1000  # convert to meters
                y_new *= 1000  # convert to meters
                x_transform, y_transform = data_utils.lambert_conformal_to_cartesian(lon_new, lat_new, **transform_args)

                gridded_indices = np.dstack((np.digitize(y_transform, gridded_y), np.digitize(x_transform, gridded_x)))[0]  # translate coordinate indices to grid
                gridded_indices_unique = np.unique(gridded_indices, axis=0)  # remove duplicate coordinate indices

                # Remove points outside the domain
                gridded_indices_unique = gridded_indices_unique[np.where(gridded_indices_unique[:, 0] != len(gridded_y))]
                gridded_indices_unique = gridded_indices_unique[np.where(gridded_indices_unique[:, 1] != len(gridded_x))]

                identifier[gridded_indices_unique[:, 0], gridded_indices_unique[:, 1]] = pgenType_identifiers[type_of_front]  # assign labels to the gridded points based on the front type

                fronts_ds = xr.Dataset({"identifier": (('y', 'x'), identifier)}).expand_dims(**expand_dims_args)

            else:

                if front_needs_modification:
                    lon_new = np.where(lon_new > 180, lon_new - 360, lon_new)  # convert new longitudes to standard -180 to 180 range

                gridded_indices = np.dstack((np.digitize(lon_new, gridded_lons), np.digitize(lat_new, gridded_lats)))[0]  # translate coordinate indices to grid
                gridded_indices_unique = np.unique(gridded_indices, axis=0)  # remove duplicate coordinate indices

                # Remove points outside the domain
                gridded_indices_unique = gridded_indices_unique[np.where(gridded_indices_unique[:, 0] != len(gridded_lons))]
                gridded_indices_unique = gridded_indices_unique[np.where(gridded_indices_unique[:, 1] != len(gridded_lats))]

                identifier[gridded_indices_unique[:, 0], gridded_indices_unique[:, 1]] = pgenType_identifiers[type_of_front]  # assign labels to the gridded points based on the front type

                fronts_ds = xr.Dataset({"identifier": (('longitude', 'latitude'), identifier)},
                                       coords={"longitude": gridded_lons, "latitude": gridded_lats}).expand_dims(**expand_dims_args)

                if args['domain'] == 'full':
                    fronts_ds = fronts_ds.sel(longitude=np.append(np.arange(130, 180.25, 0.25), np.arange(-179.75, 10, 0.25)))

                # convert longitudes back to 360-degree system
                fronts_ds["longitude"] = domain_lons_360[args["domain"]]

        if not os.path.isdir("%s/%d%02d" % (args['netcdf_outdir'], year, month)):
            os.mkdir("%s/%d%02d" % (args['netcdf_outdir'], year, month))

        fronts_ds.attrs["domain"] = args["domain"]
        fronts_ds.attrs["interpolation_distance_km"] = args["distance"]
        fronts_ds.attrs["num_front_types"] = 16
        fronts_ds.attrs["front_types"] = "ALL"
        fronts_ds.attrs["xml_to_nc_script_version"] = "2024.5.14"

        fronts_ds.to_netcdf(path="%s/%d%02d/%s" % (args['netcdf_outdir'], year, month, filename_netcdf), engine='netcdf4', mode='w')