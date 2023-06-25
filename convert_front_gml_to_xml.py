from lxml import etree as ET


xml_outdir = 'E:/FrontsProjectData/xmls'

XML_FRONT_TYPE = {'Cold Front': 'COLD_FRONT', 'Warm Front': 'WARM_FRONT', 'Stationary Front': 'STATIONARY_FRONT','Occluded Front': 'OCCLUDED_FRONT',
                  'Dry Line': 'DRY_LINE', 'Trough': 'TROF'}

XML_FRONT_COLORS = {'Cold Front': dict(red="0", green="0", blue="255"), 'Warm Front': dict(red="255", green="0", blue="0"),
                    'Occluded Front': dict(red="145", green="44", blue="238"), 'Dry Line': dict(red="255", green="130", blue="71"),
                    'Trough': dict(red="255", green="130", blue="71")}

LINE_KWARGS = dict(pgenCategory="Front", lineWidth="4", sizeScale="  1.0", smoothFactor="2", closed="false", filled="false",
                   fillPattern="SOLID")

gml_file = 'fronts_mslp_024h_world.gml.xml'

init_time = [2023, 6, 15, 0]
forecast_hour = 12
root_xml = ET.Element("Product", name="IBM_global_fronts", init_time="%d%02d%02d%02d" % (init_time[0], init_time[1], init_time[2], init_time[3]),
                      forecast_hour=str(forecast_hour))
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


save_path_file = "%s/IBMfronts_%d%02d%02d%02df%03d.xml" % (xml_outdir, init_time[0], init_time[1], init_time[2], init_time[3], forecast_hour)

ET.indent(root_xml)
mydata = ET.tostring(root_xml)
xmlFile = open(save_path_file, "wb")
xmlFile.write(mydata)
xmlFile.close()
