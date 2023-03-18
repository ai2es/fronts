"""
Create an XML file containing model predictions in the form of splines.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 3/18/2023 2:35 PM CT
"""


from lxml import etree as ET
from skimage.morphology import skeletonize
import xarray as xr
import numpy as np
import argparse


FRONT_OBJ_THRESHOLDS = {'CF': {'conus': 0.35, 'full': 0.48}, 'WF': {'conus': 0.31, 'full': 0.37},
                        'SF': {'conus': 0.29, 'full': 0.38}, 'OF': {'conus': 0.26, 'full': 0.24},
                        'F_BIN': {'conus': 0.42, 'full': 0.50}, 'DL': {'conus': 0.42, 'full': 0.42}}

XML_FRONT_TYPE = {'CF': 'COLD_FRONT', 'CF-F': 'COLD_FRONT_FORM', 'CF-D': 'COLD_FRONT_DISS',
                  'WF': 'WARM_FRONT', 'WF-F': 'WARM_FRONT_FORM', 'WF-D': 'WARM_FRONT_DISS',
                  'SF': 'STATIONARY_FRONT', 'SF-F': 'STATIONARY_FRONT_FORM', 'SF-D': 'STATIONARY_FRONT_DISS',
                  'OF': 'OCCLUDED_FRONT', 'OF-F': 'OCCLUDED_FRONT_FORM', 'OF-D': 'OCCLUDED_FRONT_DISS',
                  'DL': 'DRY_LINE', 'OFB': 'INSTABILITY', 'T': 'TROF', 'TT': 'TROPICAL_TROF'}

XML_FRONT_COLORS = {'CF': dict(red="0", green="0", blue="255"), 'CF-F': dict(red="0", green="0", blue="255"), 'CF-D': dict(red="0", green="0", blue="255"),
                  'WF': dict(red="255", green="0", blue="0"), 'WF-F': dict(red="255", green="0", blue="0"), 'WF-D': dict(red="255", green="0", blue="0"),
                  'OF': dict(red="145", green="44", blue="238"), 'OF-F': dict(red="145", green="44", blue="238"), 'OF-D': dict(red="145", green="44", blue="238"),
                  'DL': dict(red="255", green="130", blue="71"), 'OFB': dict(red="255", green="0", blue="0"), 'T': dict(red="255", green="130", blue="71"),
                  'TT': dict(red="255", green="130", blue="71")}

LINE_KWARGS = dict(pgenCategory="Front", lineWidth="4", sizeScale="  1.0", smoothFactor="2", closed="false", filled="false",
                   fillPattern="SOLID")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ### Data arguments ###
    parser.add_argument('--init_time', type=int, nargs=4, required=True, help='Date and time of the data. Pass 4 ints in the following order: year, month, day, hour')
    parser.add_argument('--forecast_hours', type=int, nargs='+', required=True, help='Forecast hour for the GDAS/GFS data.')
    parser.add_argument('--domain', type=str, required=True, help='Domain of the data.')
    parser.add_argument('--variable_data_source', type=str, help='Data source for variables used in the predictions.')

    ### Model and directory arguments ###
    parser.add_argument('--model_dir', type=str, required=True, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, required=True, help='Model number.')
    parser.add_argument('--xml_dir', type=str, help='Output directory for the XML files.')

    args = parser.parse_args()

    for forecast_hour in args.forecast_hours:

        root = ET.Element("Product", name="front_predictions", model_number=str(args.model_number),
                          init_time="%d%02d%02d%02d" % (args.init_time[0], args.init_time[1], args.init_time[2], args.init_time[3]),
                          forecast_hour=str(forecast_hour), domain=args.domain, variable_data_source=args.variable_data_source)

        Layer = ET.SubElement(root, "Layer", name="Default", onOff="true", monoColor="false", filled="false")
        ET.SubElement(Layer, "Color", red="255", green="255", blue="0", alpha="255")
        DrawableElement = ET.SubElement(Layer, "DrawableElement")

        probs_ds = xr.open_dataset('%s/model_%d/predictions/model_%d_%d-%02d-%02d-%02dz_%s_f%03d_%s_3x1_probabilities.nc' %
            (args.model_dir, args.model_number, args.model_number, args.init_time[0], args.init_time[1], args.init_time[2],
             args.init_time[3], args.variable_data_source, forecast_hour, args.domain)).isel(time=0)
        lons = probs_ds['longitude'].values
        lats = probs_ds['latitude'].values

        for key in list(probs_ds.keys()):

            splines = dict()

            probs_ds[f'{key}_obj'] = (('longitude', 'latitude'), skeletonize(xr.where(probs_ds[key] > FRONT_OBJ_THRESHOLDS[key][args.domain], 1, 0).values))

            front_point_indices = np.where(probs_ds[f'{key}_obj'].values == 1)
            front_points = []

            for lon, lat in zip(lons[front_point_indices[0]], lats[front_point_indices[1]]):
                front_points.append([lon, lat])
            front_points_copy = front_points.copy()

            splines_made = 0
            points_in_current_spline = []

            for point_no, point in enumerate(front_points):
                front_points_dist = np.abs(np.array(front_points_copy) - np.array(point))
                adjacent_point_indices = np.where((front_points_dist[:, 0] <= 0.25) & (front_points_dist[:, 1] <= 0.25))[0]

                if len(adjacent_point_indices) > 0:
                    if len(adjacent_point_indices) > 1 or len(points_in_current_spline) > 0:
                        for adjacent_point in adjacent_point_indices:
                            if front_points_copy[adjacent_point] not in points_in_current_spline:
                                points_in_current_spline.append(front_points_copy[adjacent_point])
                        for adjacent_point in adjacent_point_indices:
                            front_points_copy[adjacent_point] = [-999, -999]
                else:
                    if len(points_in_current_spline) > 0:
                        splines[str(splines_made + 1)] = points_in_current_spline
                        splines_made += 1
                    points_in_current_spline = []

            for spline in splines.keys():

                Line = ET.SubElement(DrawableElement, "Line", pgenType=XML_FRONT_TYPE[key], **LINE_KWARGS)
                if 'SF' in key:
                    ET.SubElement(Line, "Color", red="255", green="0", blue="0", alpha="255")
                    ET.SubElement(Line, "Color", red="0", green="0", blue="255", alpha="255")
                else:
                    ET.SubElement(Line, "Color", **XML_FRONT_COLORS[key], alpha="255")

                spline_points = splines[spline]
                for spline_point in spline_points:
                    if 180 < spline_point[0]:
                        lon_point = spline_point[0] - 360
                    else:
                        lon_point = spline_point[0]
                    ET.SubElement(Line, "Point", Lat="%.2f" % spline_point[1], Lon="%.2f" % lon_point)

        save_path_file = "%s/model_%d_splines_%d%02d%02d%02df%03d.xml" % \
                         (args.xml_dir, args.model_number, args.init_time[0], args.init_time[1], args.init_time[2], args.init_time[3], forecast_hour)

        ET.indent(root)
        mydata = ET.tostring(root)
        xmlFile = open(save_path_file, "wb")
        xmlFile.write(mydata)
        xmlFile.close()
