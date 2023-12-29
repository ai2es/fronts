"""
Create an XML file containing model predictions in the form of splines.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 8/1/2023 12:36 PM CT
"""
import pandas as pd
from lxml import etree as ET
from skimage.morphology import skeletonize
from itertools import groupby, product, permutations
import xarray as xr
import numpy as np
import argparse


XML_FRONT_TYPE = {'CF': 'COLD_FRONT', 'CF-F': 'COLD_FRONT_FORM', 'CF-D': 'COLD_FRONT_DISS',
                  'WF': 'WARM_FRONT', 'WF-F': 'WARM_FRONT_FORM', 'WF-D': 'WARM_FRONT_DISS',
                  'SF': 'STATIONARY_FRONT', 'SF-F': 'STATIONARY_FRONT_FORM', 'SF-D': 'STATIONARY_FRONT_DISS',
                  'OF': 'OCCLUDED_FRONT', 'OF-F': 'OCCLUDED_FRONT_FORM', 'OF-D': 'OCCLUDED_FRONT_DISS',
                  'DL': 'DRY_LINE', 'OFB': 'INSTABILITY', 'T': 'TROF', 'TT': 'TROPICAL_TROF', 'F_BIN': 'BINARY_FRONT'}

XML_FRONT_COLORS = {'CF': dict(red="0", green="0", blue="255"), 'CF-F': dict(red="0", green="0", blue="255"), 'CF-D': dict(red="0", green="0", blue="255"),
                  'WF': dict(red="255", green="0", blue="0"), 'WF-F': dict(red="255", green="0", blue="0"), 'WF-D': dict(red="255", green="0", blue="0"),
                  'OF': dict(red="145", green="44", blue="238"), 'OF-F': dict(red="145", green="44", blue="238"), 'OF-D': dict(red="145", green="44", blue="238"),
                  'DL': dict(red="255", green="130", blue="71"), 'OFB': dict(red="255", green="0", blue="0"), 'T': dict(red="255", green="130", blue="71"),
                  'TT': dict(red="255", green="130", blue="71"), 'F_BIN': dict(red="128", green="128", blue="128")}

LINE_KWARGS = dict(pgenCategory="Front", lineWidth="4", sizeScale="  1.0", smoothFactor="2", closed="false", filled="false",
                   fillPattern="SOLID")


def _coords_dist(point1, point2):
    """
    Calculate the distance between two points.

    Parameters
    ----------
    point1: tuple or list with 2 ints
    point2: tuple or list with 2 ints

    Returns
    -------
    Distance between point1 and point2.
    """

    dist = np.sqrt(np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1]))
    if dist <= np.sqrt(2):
        print("Point 1: ", point1, "     Point 2:", point2)
    return dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ### Model and directory arguments ###
    parser.add_argument('--model_dir', type=str, required=True, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, required=True, help='Model number.')
    parser.add_argument('--xml_dir', type=str, help='Output directory for the XML files.')

    ### Data arguments ###
    parser.add_argument('--init_time', type=int, nargs=4, required=True, help='Date and time of the data. Pass 4 ints in the following order: year, month, day, hour')
    parser.add_argument('--forecast_hours', type=int, nargs='+', required=True, help='Forecast hour for the GDAS/GFS data.')
    parser.add_argument('--domain', type=str, required=True, help='Domain of the data.')
    parser.add_argument('--data_source', type=str, help='Data source for variables used in the predictions.')
    parser.add_argument('--min_length', type=int, help="Minimum number of points a front must have in order for it to be retained in the output dataset.")
    parser.add_argument('--interval', type=int, default=10, help='Interval of points to save for each front. E.g. 10 = keep every 10th point for each spline')

    args = vars(parser.parse_args())
    
    model_properties = pd.read_pickle(f"{args['model_dir']}/model_{args['model_number']}/model_{args['model_number']}_properties.pkl")
    front_obj_thresholds = model_properties['front_obj_thresholds']

    for forecast_hour in args['forecast_hours']:

        root = ET.Element("Product", name="front_predictions", model_number=str(args['model_number']),
                          init_time="%d%02d%02d%02d" % (args['init_time'][0], args['init_time'][1], args['init_time'][2], args['init_time'][3]),
                          forecast_hour=str(forecast_hour), domain=args['domain'], data_source=args['data_source'])

        Layer = ET.SubElement(root, "Layer", name="Default", onOff="true", monoColor="false", filled="false")
        ET.SubElement(Layer, "Color", red="255", green="255", blue="0", alpha="255")
        DrawableElement = ET.SubElement(Layer, "DrawableElement")

        probs_ds = xr.open_dataset('%s/model_%d/predictions/model_%d_%d-%02d-%02d-%02dz_%s_f%03d_%s_probabilities.nc' %
            (args['model_dir'], args['model_number'], args['model_number'], args['init_time'][0], args['init_time'][1], args['init_time'][2],
             args['init_time'][3], args['data_source'], forecast_hour, args['domain'])).isel(time=0, forecast_hour=0)
        probs_ds = probs_ds.reindex(latitude=list(reversed(probs_ds.latitude)))

        front_types = list(probs_ds.keys())

        if len(front_types) > 1:
            all_possible_front_combinations = permutations(front_types, r=2)
            for combination in all_possible_front_combinations:
                probs_ds[combination[0]].values = np.where(probs_ds[combination[0]].values > probs_ds[combination[1]].values, probs_ds[combination[0]].values, 0)

        lons = probs_ds['longitude'].values
        lats = probs_ds['latitude'].values

        for key in list(probs_ds.keys())[:1]:

            try:
                probs_ds[f'{key}_obj'] = (('longitude', 'latitude'), skeletonize(xr.where(probs_ds[key] > front_obj_thresholds[args['domain']][key]['250'], 1, 0).values.copy(order="C")))
            except KeyError:
                probs_ds[f'{key}_obj'] = (('longitude', 'latitude'), skeletonize(xr.where(probs_ds[key] > front_obj_thresholds['full'][key]['250'], 1, 0).values.copy(order="C")))

            current_probs = probs_ds[key].values.copy(order="C")

            frontal_zones = skeletonize(np.where(current_probs >= 0.1, 1, 0))
            front_point_indices = np.where(frontal_zones == 1)
            front_points = [(coord[0], coord[1]) for coord in zip(front_point_indices[0], front_point_indices[1])]

            main_tuples = [sorted(points) for points in product(front_points, repeat=2) if _coords_dist(*points) <= np.sqrt(2)]
            unique_points = list(sorted(set([point for tup in main_tuples for point in tup])))

            group_dict = {ele: {ele} for ele in unique_points}
            for tup1, tup2 in main_tuples:
                group_dict[tup1] |= group_dict[tup2]
                group_dict[tup2] = group_dict[tup1]

            fronts = [[*next(val)] for key, val in groupby(sorted(group_dict.values(), key = id), id)]
            new_fronts = [sorted(front) for front in fronts]

            for front in new_fronts:

                Line = ET.SubElement(DrawableElement, "Line", pgenType=XML_FRONT_TYPE[key], **LINE_KWARGS)
                if 'SF' in key:
                    ET.SubElement(Line, "Color", red="255", green="0", blue="0", alpha="255")
                    ET.SubElement(Line, "Color", red="0", green="0", blue="255", alpha="255")
                else:
                    ET.SubElement(Line, "Color", **XML_FRONT_COLORS[key], alpha="255")

                for point in front:
                    ET.SubElement(Line, "Point", Lat="%.2f" % lats[point[1]], Lon="%.2f" % lons[point[0]])

        save_path_file = "%s/model_%d_splines_%s_%s_%d%02d%02d%02df%03d.xml" % \
                         (args['xml_dir'], args['model_number'], args['data_source'], args['domain'], args['init_time'][0],
                          args['init_time'][1], args['init_time'][2], args['init_time'][3], forecast_hour)

        el = ET.fromstring(f'<Products xmlns:ns2="http://www.example.org/productType">{ET.tostring(root).decode("utf-8")}</Products>')
        ET.indent(el)
        mydata = ET.tostring(el, xml_declaration=True, encoding='utf-8', standalone='yes', pretty_print=True)
        xmlFile = open(save_path_file, "wb")
        xmlFile.write(mydata)
        xmlFile.close()
