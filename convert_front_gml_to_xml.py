"""
Convert GML files containing IBM/TWC fronts into XML files.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.9.2
"""

import argparse
from lxml import etree as ET
from glob import glob
import os
import numpy as np

XML_FRONT_TYPE = {'Cold Front': 'COLD_FRONT', 'Dissipating Cold Front': 'COLD_FRONT_DISS',
                  'Warm Front': 'WARM_FRONT', 'Stationary Front': 'STATIONARY_FRONT',
                  'Occluded Front': 'OCCLUDED_FRONT', 'Dissipating Occluded Front': 'OCCLUDED_FRONT_DISS',
                  'Dry Line': 'DRY_LINE', 'Trough': 'TROF', 'Squall Line': 'INSTABILITY'}

XML_FRONT_COLORS = {'Cold Front': dict(red="0", green="0", blue="255"), 'Dissipating Cold Front': dict(red="0", green="0", blue="255"),
                    'Warm Front': dict(red="255", green="0", blue="0"), 'Dissipating Warm Front': dict(red="255", green="0", blue="0"),
                    'Occluded Front': dict(red="145", green="44", blue="238"), 'Dissipating Occluded Front': dict(red="145", green="44", blue="238"),
                    'Dry Line': dict(red="255", green="130", blue="71"), 'Trough': dict(red="255", green="130", blue="71"),
                    'Squall Line': dict(red="255", green="0", blue="0")}

LINE_KWARGS = dict(pgenCategory="Front", lineWidth="4", sizeScale="  1.0", smoothFactor="2", closed="false", filled="false",
                   fillPattern="SOLID")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gml_indir', type=str, required=True, help="Input directory for IBM/TWC front GML files.")
    parser.add_argument('--xml_outdir', type=str, required=True, help="Output directory for front XML files.")
    parser.add_argument('--date', type=int, nargs=3, required=True, help="Date for the data to be read in. (year, month, day)")
    args = vars(parser.parse_args())

    year, month, day = args['date']

    gml_files = sorted(glob('%s/%d%02d%02d/*/*%d%02d%02d*.gml' % (args['gml_indir'], year, month, day, year, month, day)))

    for gml_file in gml_files:

        valid_time_str = os.path.basename(gml_file).split('.')[2]
        valid_time_str = valid_time_str[:4] + '-' + valid_time_str[4:6] + '-' + valid_time_str[6:8] + 'T' + valid_time_str[9:11]
        valid_time = np.datetime64(valid_time_str, 'ns')

        init_time_str = os.path.basename(gml_file).split('.')[3]
        if init_time_str != 'NIL':  # an init time of 'NIL' is used to indicate forecast hour 0 (i.e. valid time is same as init time)
            init_time_str = init_time_str[:4] + '-' + init_time_str[4:6] + '-' + init_time_str[6:8] + 'T' + init_time_str[9:11]
            init_time = np.datetime64(init_time_str, 'ns')
        else:
            init_time_str = valid_time_str
            init_time = valid_time

        forecast_hour = int((valid_time - init_time) / np.timedelta64(1, 'h'))

        root_xml = ET.Element("Product", name="IBM_global_fronts", init_time=init_time_str, valid_time=valid_time_str, forecast_hour=str(forecast_hour))
        tree = ET.parse(gml_file, parser=ET.XMLPullParser(encoding='utf-8'))
        root_gml = tree.getroot()

        Layer = ET.SubElement(root_xml, "Layer", name="Default", onOff="true", monoColor="false", filled="false")
        ET.SubElement(Layer, "Color", red="255", green="255", blue="0", alpha="255")
        DrawableElement = ET.SubElement(Layer, "DrawableElement")

        front_elements = [element[0] for element in root_gml if element[0].tag == 'FRONT']

        for element in front_elements:
            front_type = [subelement.text for subelement in element if subelement.tag == 'FRONT_TYPE'][0]
            coords = [subelement for subelement in element if 'lineString' in subelement.tag][0][0][0].text

            Line = ET.SubElement(DrawableElement, "Line", pgenType=XML_FRONT_TYPE[front_type], **LINE_KWARGS)
            if front_type == 'Stationary Front':
                ET.SubElement(Line, "Color", red="255", green="0", blue="0", alpha="255")
                ET.SubElement(Line, "Color", red="0", green="0", blue="255", alpha="255")
            else:
                ET.SubElement(Line, "Color", **XML_FRONT_COLORS[front_type], alpha="255")

            coords = coords.replace('\n', '').split(' ')  # generate coordinate strings
            coords = list(coord_pair.split(',') for coord_pair in coords)  # generate coordinate pairs from the strings

            for coord_pair in coords:
                ET.SubElement(Line, "Point", Lat="%.6f" % float(coord_pair[1]), Lon="%.6f" % float(coord_pair[0]))

        save_path_file = "%s/IBM_fronts_%sf%03d.xml" % (args['xml_outdir'], init_time_str.replace('-', '').replace('T', ''), forecast_hour)

        print(save_path_file)

        ET.indent(root_xml)
        mydata = ET.tostring(root_xml)
        xmlFile = open(save_path_file, "wb")
        xmlFile.write(mydata)
        xmlFile.close()
