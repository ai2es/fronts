"""
TODO: Make sure to change the model directory when xCITE instance is launched
"""
import gc
import branca
import datetime as dt
import folium
import geojsoncontour
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
from folium.features import Element, GeoJsonTooltip
from folium.plugins import TimestampedGeoJson, FloatImage, Fullscreen, ScrollZoomToggler
from glob import glob
from matplotlib import cm, colors

import utils.data_utils

# colormaps of probability contours for front predictions
CONTOUR_CMAPS = {'CF': 'Blues', 'WF': 'Reds', 'SF': 'Greens', 'OF': 'Purples', 'CF-F': 'Blues', 'WF-F': 'Reds', 'SF-F': 'Greens',
                 'OF-F': 'Purples', 'CF-D': 'Blues', 'WF-D': 'Reds', 'SF-D': 'Greens', 'OF-D': 'Purples', 'INST': 'YlOrBr',
                 'TROF': 'YlOrRd', 'TT': 'Oranges', 'DL': 'copper_r', 'MERGED-CF': 'Blues', 'MERGED-WF': 'Reds', 'MERGED-SF': 'Greens',
                 'MERGED-OF': 'Purples', 'MERGED-F': 'Greys', 'MERGED-T': 'YlOrBr', 'F_BIN': 'Greys', 'MERGED-F_BIN': 'Greys'}

# names of front types
FRONT_NAMES = {'CF': 'Cold front', 'WF': 'Warm front', 'SF': 'Stationary front', 'OF': 'Occluded front', 'CF-F': 'Cold front (forming)',
               'WF-F': 'Warm front (forming)', 'SF-F': 'Stationary front (forming)', 'OF-F': 'Occluded front (forming)',
               'CF-D': 'Cold front (dying)', 'WF-D': 'Warm front (dying)', 'SF-D': 'Stationary front (dying)', 'OF-D': 'Occluded front (dying)',
               'INST': 'Outflow boundary', 'TROF': 'Trough', 'TT': 'Tropical trough', 'DL': 'Dryline', 'MERGED-CF': 'Cold front (any)',
               'MERGED-WF': 'Warm front (any)', 'MERGED-SF': 'Stationary front (any)', 'MERGED-OF': 'Occluded front (any)',
               'MERGED-F': 'CF, WF, SF, OF (any)', 'MERGED-T': 'Trough (any)', 'F_BIN': 'Binary front', 'MERGED-F_BIN': 'Binary front (any)'}


# colors for plotted ground truth fronts
FRONT_COLORS = {'CF': 'blue', 'WF': 'red', 'SF': 'limegreen', 'OF': 'darkviolet', 'CF-F': 'darkblue', 'WF-F': 'darkred',
                'SF-F': 'darkgreen', 'OF-F': 'darkmagenta', 'CF-D': 'lightskyblue', 'WF-D': 'lightcoral', 'SF-D': 'lightgreen',
                'OF-D': 'violet', 'INST': 'gold', 'TROF': 'goldenrod', 'TT': 'orange', 'DL': 'chocolate', 'MERGED-CF': 'blue',
                'MERGED-WF': 'red', 'MERGED-SF': 'limegreen', 'MERGED-OF': 'darkviolet', 'MERGED-F': 'gray', 'MERGED-T': 'brown',
                'F_BIN': 'tab:red', 'MERGED-F_BIN': 'tab:red'}

DOMAIN_SELECT_OPTIONS = {'Global': 'global',
                         'Unified Surface Analysis Domain (NOAA)': 'full',
                         'United States (CONUS)': 'conus',
                         'North Atlantic': 'north-atlantic',
                         'North Pacific': 'north-pacific'}

DOMAIN_ZOOMS = {'global': 2,
                'conus': 5,
                'north-atlantic': 4,
                'north-pacific': 4,
                'full': 3}


@st.cache_data
def get_available_data(files):
    dates = []
    domains = []
    nwp_models = []
    forecast_hours = []

    for file in files:
        filename = os.path.basename(file)
        split_filename = filename.split('_')

        timestep = dt.datetime.strptime(split_filename[2], '%Y%m%d%H')

        if timestep not in dates:
            dates.append(timestep)
        if split_filename[3] not in domains:
            domains.append(split_filename[3])
        if split_filename[4] not in nwp_models:
            nwp_models.append(split_filename[4])
        if int(split_filename[5][1:]) not in forecast_hours:
            forecast_hours.append(int(split_filename[5][1:]))

    dates = sorted(set(dates))
    domains = sorted(set(domains))
    nwp_models = set(sorted(nwp_models))
    forecast_hours = np.unique(np.sort(forecast_hours))

    return dates, domains, nwp_models, forecast_hours


st.title('FrontFinder Data Viewer')

refresh = st.button('Refresh Data', on_click=gc.collect)

model_dir = 'E:/FrontsProjectData/models'  # TODO
model_number = 1702
model_properties = pd.read_pickle('%s/model_%d/model_%d_properties.pkl' % (model_dir, model_number, model_number))

available_files = list(sorted(glob('%s/model_%d/predictions/model_%d*prob*.nc' % (model_dir, model_number, model_number))))

dates, _, _, _ = get_available_data(available_files)
init_disabled = len(dates) == 1
init_input = st.selectbox('Select Initialization Time', options=dates, disabled=init_disabled)
timestring = '%d%02d%02d%02d' % (init_input.year, init_input.month, init_input.day, init_input.hour)
files_filtered_init = [file for file in available_files if '_%s_' % timestring in file]

_, _, nwp_models, _ = get_available_data(files_filtered_init)
nwp_disabled = len(nwp_models) == 1
nwp_input = st.selectbox('Select Model', options=nwp_models, disabled=nwp_disabled)
files_filtered_nwp = [file for file in files_filtered_init if '_%s_' % nwp_input in file]

domain_input = st.selectbox('Select Domain', options=list(DOMAIN_SELECT_OPTIONS.keys()))

_, _, _, forecast_hours = get_available_data(files_filtered_nwp)
valid_times = ['F%03d (Valid: %s UTC)' % (fhr, init_input + dt.timedelta(hours=int(fhr))) for fhr in forecast_hours]
fhr_disabled = len(forecast_hours) == 1
fhr_input = st.selectbox('Select Forecast Hour', options=valid_times, disabled=fhr_disabled)
selected_file = [file for file in files_filtered_nwp if '_f%s_' % fhr_input[1:4] in file][0]

map_type = st.radio('Map Type', options=['Interactive Plot (will open in new tab)'], horizontal=True)

show_map = st.button('Show Map', on_click=gc.collect)

if show_map:

    if map_type == 'Interactive Plot (will open in new tab)':
        
        domain_str = DOMAIN_SELECT_OPTIONS[domain_input]
        domain_zoom = DOMAIN_ZOOMS[domain_str]
        extent_slice = utils.data_utils.DOMAIN_EXTENTS[domain_str]
        
        probs_ds = xr.open_dataset(selected_file, engine='netcdf4').isel(time=0, forecast_hour=0).transpose('latitude', 'longitude')
        probs_ds = probs_ds.sel(longitude=slice(extent_slice[0], extent_slice[1]), latitude=slice(extent_slice[3], extent_slice[2]))
        
        if domain_str == 'global':
            probs_ds = probs_ds.reindex(longitude=np.append(np.arange(180, 360, 0.25), np.arange(0, 180, 0.25)).flatten())
            extent_slice = [-180, 180, -90, 90]
        
        # lat/lon coordinates
        lat = probs_ds['latitude'].values
        lon = probs_ds['longitude'].values
        
        if domain_str == 'global':
            lon = np.where(lon >= 180, lon - 360, lon)
        
        Lon, Lat = np.meshgrid(lon, lat)

        geomap_start_coords = [np.median(lat), np.median(lon)]

        geomap = folium.Map(geomap_start_coords, zoom_start=domain_zoom, min_zoom=domain_zoom, max_bounds=True, min_lat=extent_slice[2],
            max_lat=extent_slice[3], min_lon=extent_slice[0], max_lon=extent_slice[1], tiles="Cartodb positron", png_enabled=True,
            position='relative', height='96.2%')
        
        for front_label in ['CF', 'WF', 'SF', 'OF', 'DL']:
            
            front_type = FRONT_NAMES[front_label]
            cmap = CONTOUR_CMAPS[front_label]

            fg = folium.FeatureGroup(front_type)

            cmap_probs, norm_probs = cm.get_cmap(cmap, 20), colors.Normalize(vmin=0, vmax=1.0)

            # calibrate model predictions
            ir_model = model_properties['calibration_models']['conus'][front_label]['100 km']
            original_shape = np.shape(probs_ds[front_label].values)
            probs_ds[front_label].values = ir_model.predict(probs_ds[front_label].values.flatten()).reshape(original_shape)
            
            # st.write("%s max: %.2f" % (front_label, np.nanmax(probs_ds[front_label])))
            
            prob_thresholds = np.append(np.arange(0.2, 1.0, 0.05), np.array([0.99, 1.00])).flatten()
            for prob in prob_thresholds:
                prob = np.round(prob, 2)
                contourf = plt.contourf(Lon, Lat, probs_ds[front_label].values, cmap=cmap_probs, norm=norm_probs, levels=[prob, prob + 0.05], alpha=0.85)
                geojson_object = geojsoncontour.contourf_to_geojson(contourf=contourf, stroke_width=1, geojson_properties={'front-type': FRONT_NAMES[front_label],
                                                                                                                           'probability': '%d%%' % (prob * 100),
                                                                                                                           'outline-color': 'red'})
                
                del contourf
                
                folium.GeoJson(
                    geojson_object,
                    style_function=lambda x: {
                        'color': x['properties']['stroke'],
                        'weight': x['properties']['stroke-width'],
                        'fillColor': x['properties']['fill'],
                        'opacity': 1.0
                    },
                    tooltip=GeoJsonTooltip(fields=['front-type', 'probability'],
                                 aliases=['Front Type', 'Probability'],
                                 labels=True,
                                 localize=True,
                                 style=
                                 f"""
                                 border-color: black;
                                 """)
                ).add_to(fg)
                
                del geojson_object
    
                fg.add_to(geomap)
            
            del fg
            
            index_list = list(range(cmap_probs.N))
            branca_colors = list(tuple(cmap_probs(i)) for i in index_list)
            branca_cmap = branca.colormap.StepColormap(branca_colors, vmin=0.0, vmax=1.0,
                                                       caption=front_type + ' probability')

            branca_cmap.width = 300
        
        folium.LayerControl().add_to(geomap)
        
        ### logos/attributions ###
        Fullscreen(title='Fullscreen', title_cancel='Exit fullscreen', position='topright').add_to(geomap)
        FloatImage('https://pachtml-production.s3-us-west-2.amazonaws.com/www/ou/branding/logo.png', width='50px', height='50px', bottom='88', left='0.2', position='fixed').add_to(geomap)  # OU
        FloatImage('https://arrc.ou.edu/news_images/img_0158.png', width='50px', height='50px', bottom='79', left='0.2', position='fixed').add_to(geomap)  # OU SoM
        FloatImage('https://a.espncdn.com/combiner/i?img=/i/teamlogos/ncaa/500/2117.png&h=200&w=200', width='50px', height='50px', bottom='71', left='0.2', position='fixed').add_to(geomap)  # CMU
        FloatImage('https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/NSF_Official_logo_Med_Res.png/1200px-NSF_Official_logo_Med_Res.png', width='50px', height='50px', bottom='63', left='0.2', position='fixed').add_to(geomap)  # NSF
        FloatImage('https://www.ai2es.org/wp-content/uploads/2020/08/ai2es-logo-01.png', width='50px', height='50px', bottom='55', left='0.2', position='fixed').add_to(geomap)  # AI2ES
        
        valid_time = init_input + dt.timedelta(hours=int(fhr_input[1:4]))
        geomap.get_root().html.add_child(Element('<h3 align="center" style="font-size:16px;"><b>FrontFinder v1.1:</b> GFS {} {}</h3>'.format(init_input, fhr_input)))
        
        geomap.show_in_browser()
        probs_ds.close()
        
        del probs_ds, geomap, branca_cmap
        
        gc.collect()