"""
TODO: Make sure to change the model directory when xCITE instance is launched
"""
import gc
import datetime as dt
import webbrowser
import os
import numpy as np
import streamlit as st
from glob import glob


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


@st.cache_resource
def set_page_title():
    st.title('FrontFinder Data Viewer')


def get_available_files():
    available_files = list(sorted(glob('%s/model_%d*.html' % (folium_dir, model_number))))
    return available_files


@st.cache_data
def get_available_dates(files):
    dates = []
    
    for file in files:
        
        filename = os.path.basename(file)
        split_filename = filename.split('_')
        timestep = dt.datetime.strptime(split_filename[3], '%Y%m%d%H')
        dates.append(timestep)
    
    return sorted(set(dates))


@st.cache_data
def get_available_models(files):
    nwp_models = []
    
    for file in files:
        filename = os.path.basename(file)
        split_filename = filename.split('_')
        nwp_models.append(split_filename[2])

    return sorted(set(nwp_models))


@st.cache_data
def get_available_forecast_hours(files):
    forecast_hours = []
    
    for file in files:
        filename = os.path.basename(file)
        split_filename = filename.split('_')
        forecast_hours.append(int(split_filename[4][1:4]))

    return np.unique(np.sort(forecast_hours))


folium_dir = 'E:/FrontsProjectData/data/folium'

set_page_title()
model_number = 1702

refresh = st.button('Refresh', on_click=gc.collect)
available_files = get_available_files()

dates = get_available_dates(available_files)
init_disabled = len(dates) == 1
init_index = 0 if init_disabled else None
init_input = st.selectbox('Select Initialization Time', options=dates, index=init_index, disabled=init_disabled)

if init_input:
    
    timestring = '%d%02d%02d%02d' % (init_input.year, init_input.month, init_input.day, init_input.hour)
    files_filtered_init = [file for file in available_files if '_%s_' % timestring in file]

    nwp_input = 'gfs'
    
    if nwp_input:
        files_filtered_nwp = [file for file in files_filtered_init if '_%s_' % nwp_input in file]
        forecast_hours = get_available_forecast_hours(files_filtered_nwp)
        valid_times = ['F%03d (Valid: %s UTC)' % (fhr, init_input + dt.timedelta(hours=int(fhr))) for fhr in forecast_hours]
        fhr_disabled = len(forecast_hours) == 1
        fhr_index = 0 if fhr_disabled else None
        fhr_input = st.selectbox('Select Forecast Hour', options=valid_times, index=fhr_index, disabled=fhr_disabled)
            
        if fhr_input:
            selected_file = [file for file in files_filtered_nwp if '_f%s' % fhr_input[1:4] in file][0]
            show_map = st.button('Show Map (opens in new tab)')
            
            if show_map:
                
                webbrowser.open(selected_file)
                
                