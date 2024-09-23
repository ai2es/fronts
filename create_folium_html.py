"""
Generate folium maps for FrontFinder predictions.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.9.20
"""
import argparse
import datetime
import os.path
import ast
import geojsoncontour
from matplotlib import cm, colors
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import folium
from folium.elements import Element
from folium.plugins import GroupedLayerControl, Fullscreen, FloatImage
from utils import variables
import numpy as np
import pandas as pd
import pickle
import xarray as xr
import matplotlib
matplotlib.use('Agg')


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


def _wind_barb_feature_group(u, v, step=5, pole_length=1.25, barb_length=0.5, barb_angle=75, fg_name='10-meter Wind barbs'):
    
    u = u[::step, ::step] * 1.94384  # m/s ---> knots
    v = v[::step, ::step] * 1.94384  # m/s ---> knots
    
    Lon, Lat = np.meshgrid(lon[::step], lat[::step])
    
    lat_rad = np.radians(Lat)
    
    coord_pairs = np.dstack([Lat, Lon])
    
    pole_angles = np.ones_like(u) * np.nan
    pole_angles = np.where((u > 0) & (v > 0), np.arctan(v/u) + np.pi, pole_angles)
    pole_angles = np.where((u < 0) & (v < 0), np.arctan(v/u), pole_angles)
    pole_angles = np.where((u > 0) & (v < 0), np.arctan(v/u) + np.pi, pole_angles)
    pole_angles = np.where(np.isnan(pole_angles), np.arctan(v/u), pole_angles)
    pole_components_scale = np.array([np.sin(pole_angles), np.cos(pole_angles)]).transpose(1, 2, 0)
    
    zero_lat_index = int(360 / step)
    
    barb_angles = np.zeros_like(pole_angles)
    barb_angles[:zero_lat_index, :] = pole_angles[:zero_lat_index, :] - (barb_angle * np.pi / 180)  # barbs in the northern hemisphere are RIGHT of the poles
    barb_angles[zero_lat_index:, :] = pole_angles[zero_lat_index:, :] + (barb_angle * np.pi / 180)  # barbs in the southern hemisphere are LEFT of the poles
    
    flag_top_angles = np.zeros_like(pole_angles)
    flag_top_angles[:zero_lat_index, :] = pole_angles[:zero_lat_index, :] - ((180 - barb_angle) * np.pi / 180)  # flags in the northern hemisphere are RIGHT of the poles
    flag_top_angles[zero_lat_index:, :] = pole_angles[zero_lat_index:, :] + ((180 - barb_angle) * np.pi / 180)  # flags in the southern hemisphere are LEFT of the poles
    flag_base_length = barb_length * np.sin((180 - (2 * barb_angle)) * np.pi / 180) / np.sin(barb_angle * np.pi / 180)
    
    flag_top_components_scale = np.array([np.sin(flag_top_angles), np.cos(flag_top_angles)]).transpose(1, 2, 0)
    barb_components_scale = np.array([np.sin(barb_angles), np.cos(barb_angles)]).transpose(1, 2, 0)
    
    lat_flat = coord_pairs[..., 0].flatten()
    lon_flat = coord_pairs[..., 1].flatten()
    barb_scale_lat = (barb_components_scale[..., 0] * np.cos(lat_rad)).flatten()
    barb_scale_lon = (barb_components_scale[..., 1]).flatten()
    pole_scale_lat = (pole_components_scale[..., 0] * np.cos(lat_rad)).flatten()
    pole_scale_lon = (pole_components_scale[..., 1]).flatten()
    flag_top_scale_lat = (flag_top_components_scale[..., 0] * np.cos(lat_rad)).flatten()
    flag_top_scale_lon = (flag_top_components_scale[..., 1]).flatten()
    u_flat = u.flatten()
    v_flat = v.flatten()
    wspd_flat = np.sqrt(u_flat**2 + v_flat**2)

    fg_barbs = folium.FeatureGroup(fg_name, show=False)
    
    for i, point in enumerate(zip(lat_flat, lon_flat)):
        barb_coord_list = [[point[0], point[1]], [point[0] + (pole_length * pole_scale_lat[i]), point[1] + (pole_length * pole_scale_lon[i])]]
        if 5 <= wspd_flat[i] < 50:
            for j in range(int(wspd_flat[i] / 5)):
                barb_loc_on_pole = (6 - np.floor(j / 2)) / 6 if wspd_flat[i] >= 10 else 5/6
                barb_coord_list.extend([[point[0] + (barb_loc_on_pole * pole_length * pole_scale_lat[i]),
                                         point[1] + (barb_loc_on_pole * pole_length * pole_scale_lon[i])],
                                        [point[0] + (barb_loc_on_pole * pole_length * pole_scale_lat[i]) + (barb_length * barb_scale_lat[i] / (2 - (j % 2))),
                                         point[1] + (barb_loc_on_pole * pole_length * pole_scale_lon[i]) + (barb_length * barb_scale_lon[i] / (2 - (j % 2)))],
                                        [point[0] + (barb_loc_on_pole * pole_length * pole_scale_lat[i]),
                                         point[1] + (barb_loc_on_pole * pole_length * pole_scale_lon[i])]])

            fg_barbs.add_child(folium.PolyLine(barb_coord_list, color='black', weight=1.5))
    
        elif 50 <= wspd_flat[i] < 100:

            flag_polygon = [[point[0] + (pole_length * pole_scale_lat[i]),
                             point[1] + (pole_length * pole_scale_lon[i])],
                            [point[0] + (pole_length * pole_scale_lat[i]) + (barb_length * flag_top_scale_lat[i]),
                             point[1] + (pole_length * pole_scale_lon[i]) + (barb_length * flag_top_scale_lon[i])],
                            [point[0] + ((pole_length - flag_base_length) * pole_scale_lat[i]),
                             point[1] + ((pole_length - flag_base_length) * pole_scale_lon[i])]]

            fg_barbs.add_child(folium.Polygon(flag_polygon, color='black', weight=1.5, fill=True, fillOpacity=1))

            for j in range(int((wspd_flat[i] - 50) / 5)):
                barb_loc_on_pole = ((6 - np.floor(j / 2) - 1) / 6)
                barb_coord_list.extend([[point[0] + (barb_loc_on_pole * (pole_length - flag_base_length) * pole_scale_lat[i]),
                                         point[1] + (barb_loc_on_pole * (pole_length - flag_base_length) * pole_scale_lon[i])],
                                        [point[0] + (barb_loc_on_pole * (pole_length - flag_base_length) * pole_scale_lat[i]) + (barb_length * barb_scale_lat[i] / (2 - (j % 2))),
                                         point[1] + (barb_loc_on_pole * (pole_length - flag_base_length) * pole_scale_lon[i]) + (barb_length * barb_scale_lon[i] / (2 - (j % 2)))],
                                        [point[0] + (barb_loc_on_pole * (pole_length - flag_base_length) * pole_scale_lat[i]),
                                         point[1] + (barb_loc_on_pole * (pole_length - flag_base_length) * pole_scale_lon[i])]])

            fg_barbs.add_child(folium.PolyLine(barb_coord_list, color='black', weight=1.5))
        
        elif 100 <= wspd_flat[i] < 150:

            flag_polygon = [[point[0] + (pole_length * pole_scale_lat[i]),
                             point[1] + (pole_length * pole_scale_lon[i])],
                            [point[0] + (pole_length * pole_scale_lat[i]) + (barb_length * flag_top_scale_lat[i]),
                             point[1] + (pole_length * pole_scale_lon[i]) + (barb_length * flag_top_scale_lon[i])],
                            [point[0] + ((pole_length - flag_base_length) * pole_scale_lat[i]),
                             point[1] + ((pole_length - flag_base_length) * pole_scale_lon[i])]]

            fg_barbs.add_child(folium.Polygon(flag_polygon, color='black', weight=1.5, fill=True, fillOpacity=1))
            
            flag_polygon = [[point[0] + (pole_length - flag_base_length - 1/10) * pole_scale_lat[i],
                             point[1] + (pole_length - flag_base_length - 1/10) * pole_scale_lon[i]],
                            [point[0] + ((pole_length - flag_base_length - 1/10) * pole_scale_lat[i]) + (barb_length * flag_top_scale_lat[i]),
                             point[1] + ((pole_length - flag_base_length - 1/10) * pole_scale_lon[i]) + (barb_length * flag_top_scale_lon[i])],
                            [point[0] + ((pole_length - (2 * flag_base_length) - 1/10) * pole_scale_lat[i]),
                             point[1] + ((pole_length - (2 * flag_base_length) - 1/10) * pole_scale_lon[i])]]

            fg_barbs.add_child(folium.Polygon(flag_polygon, color='black', weight=1.5, fill=True, fillOpacity=1))
            
            for j in range(int((wspd_flat[i] - 30) / 5)):
                barb_loc_on_pole = ((6 - np.floor(j / 2) - 1) / 6)
                barb_coord_list.extend([[point[0] + (barb_loc_on_pole * (pole_length - ((2 * flag_base_length) + (1/8))) * pole_scale_lat[i]),
                                         point[1] + (barb_loc_on_pole * (pole_length - ((2 * flag_base_length) + (1/8))) * pole_scale_lon[i])],
                                        [point[0] + (barb_loc_on_pole * (pole_length - ((2 * flag_base_length) + (1/8))) * pole_scale_lat[i]) + (barb_length * barb_scale_lat[i] / (2 - (j % 2))),
                                         point[1] + (barb_loc_on_pole * (pole_length - ((2 * flag_base_length) + (1/8))) * pole_scale_lon[i]) + (barb_length * barb_scale_lon[i] / (2 - (j % 2)))],
                                        [point[0] + (barb_loc_on_pole * (pole_length - ((2 * flag_base_length) + (1/8))) * pole_scale_lat[i]),
                                         point[1] + (barb_loc_on_pole * (pole_length - ((2 * flag_base_length) + (1/8))) * pole_scale_lon[i])]])
            
            fg_barbs.add_child(folium.PolyLine(barb_coord_list, color='black', weight=1.5))
        
    return fg_barbs


def _potential_temperature_feature_group(T, P, fg_name='2-meter Potential temperature'):
    
    theta = variables.potential_temperature(T, P)
    theta = gaussian_filter(theta, 2)
    
    temp_thresholds = np.arange(200, 340, 2).astype(np.int32)
    
    cmap_probs, norm_probs = cm.get_cmap('jet', len(temp_thresholds)), colors.Normalize(vmin=np.min(temp_thresholds), vmax=np.max(temp_thresholds))
    
    fg_temp = folium.FeatureGroup(fg_name, show=False)
    
    for temp in temp_thresholds:
        contourf = plt.contourf(lon, lat, theta, levels=[temp, temp + 2], cmap=cmap_probs, norm=norm_probs)
        geojson_object = geojsoncontour.contourf_to_geojson(contourf=contourf, stroke_width=1, geojson_properties={'theta': '%d K' % temp})
        
        as_dict = ast.literal_eval(geojson_object)
        style = {'color': as_dict['features'][0]['properties']['stroke'],
                 'weight': as_dict['features'][0]['properties']['stroke-width'],
                 'fillColor': as_dict['features'][0]['properties']['fill'],
                 'fillOpacity': 0.5,
                 'outline-color': None,
                 'opacity': 1.0}
        tooltip_kwargs = dict(labels=True, localize=True, style="border-color: black;")

        fg_temp.add_child(folium.GeoJson(geojson_object, tooltip=folium.GeoJsonTooltip(fields=['theta',], aliases=[fg_name,], **tooltip_kwargs), **style))
    
    return fg_temp


def _isobar_feature_group(mslp, fg_name='MSLP'):
    
    pres_thresholds = np.arange(860, 1080, 4).astype(np.int32)
    
    fg_pres = folium.FeatureGroup(fg_name, show=False)
    
    for pres in pres_thresholds:
        contourf = plt.contourf(lon, lat, mslp, levels=[pres, 99999])
        geojson_object = geojsoncontour.contourf_to_geojson(contourf=contourf, stroke_width=0.8, geojson_properties={'mslp': '%d hPa' % pres})
        
        as_dict = ast.literal_eval(geojson_object)
        style = {'color': 'black', 'weight': as_dict['features'][0]['properties']['stroke-width'],
                 'fillColor': None, 'fill': None, 'fillOpacity': 0, 'outline-color': None, 'opacity': 1.0}
        tooltip_kwargs = dict(labels=True, localize=True, style="border-color: black;")

        fg_pres.add_child(folium.GeoJson(geojson_object, tooltip=folium.GeoJsonTooltip(fields=['mslp',], aliases=[fg_name,], **tooltip_kwargs), **style))
    
    return fg_pres


def _mixing_ratio_feature_group(Td, P, fg_name='2-meter Mixing ratio'):
    
    r = variables.mixing_ratio_from_dewpoint(Td, P) * 1000  # g/g ---> g/kg
    r = gaussian_filter(r, sigma=2)
    r_thresholds = np.arange(6, 30.1, 1).astype(np.int32)

    cmap_probs, norm_probs = cm.get_cmap('terrain_r', len(r_thresholds)), colors.Normalize(vmin=2, vmax=20)
    
    fg_r = folium.FeatureGroup(fg_name, show=False)
    
    for threshold in r_thresholds:
        contourf = plt.contourf(lon, lat, r, levels=[threshold, threshold + 1], cmap=cmap_probs, norm=norm_probs)
        geojson_object = geojsoncontour.contourf_to_geojson(contourf=contourf, stroke_width=0.8, geojson_properties={'r': '%d g/kg' % threshold})
        
        as_dict = ast.literal_eval(geojson_object)
        style = {'color': as_dict['features'][0]['properties']['stroke'],
                 'weight': as_dict['features'][0]['properties']['stroke-width'],
                 'fillColor': as_dict['features'][0]['properties']['fill'],
                 'fillOpacity': 0.5,
                 'outline-color': None,
                 'opacity': 1.0}
        tooltip_kwargs = dict(labels=True, localize=True, style="border-color: black;")

        fg_r.add_child(folium.GeoJson(geojson_object, tooltip=folium.GeoJsonTooltip(fields=['r',], aliases=[fg_name,], **tooltip_kwargs), **style))
    
    return fg_r


def _layer_thickness(thickness, fg_name='1000-850 hPa thickness'):
    
    thickness = gaussian_filter(thickness, sigma=2)
    
    thicknesses = np.arange(1200, 1800.1, 10).astype(np.int32)
    
    fg = folium.FeatureGroup(fg_name, show=False)
    
    for threshold in thicknesses:
        contourf = plt.contourf(lon, lat, thickness, levels=[threshold, 99999])
        geojson_object = geojsoncontour.contourf_to_geojson(contourf=contourf, stroke_width=1.0, geojson_properties={'thickness': '%d m' % threshold})
        
        as_dict = ast.literal_eval(geojson_object)
        color = 'blue' if threshold < 1350 else 'red'
        style = {'color': color, 'weight': as_dict['features'][0]['properties']['stroke-width'],
                 'fillColor': None, 'fill': None, 'fillOpacity': 0, 'outline-color': None, 'opacity': 1.0}
        tooltip_kwargs = dict(labels=True, localize=True, style="border-color: black;")

        fg.add_child(folium.GeoJson(geojson_object, tooltip=folium.GeoJsonTooltip(fields=['thickness',], aliases=[fg_name, ], **tooltip_kwargs), **style))
    
    return fg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grib_indir', type=str, help='Input directory for the NWP grib files.')
    parser.add_argument('--folium_outdir', type=str, help='Output directory for the folium objects.')
    parser.add_argument('--model_dir', type=str, help='Directory for the models.')
    parser.add_argument('--nwp_model', type=str, default='gfs', help='NWP model.')
    parser.add_argument('--model_number', type=int, help='Model number.')
    parser.add_argument('--init_time', type=str, help='Initialization time of the model. Format: YYYY-MM-DD-HH.')
    parser.add_argument('--forecast_hours', type=int, nargs='+', help='Forecast hours for the given initialization time.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing HTML files.')
    args = vars(parser.parse_args())
    
    init_time = pd.date_range(args['init_time'], args['init_time'])[0]
    
    yr, mo, dy, hr = init_time.year, init_time.month, init_time.day, init_time.hour
    timestring = '%d%02d%02d%02d' % (yr, mo, dy, hr)

    model_folder = '%s/model_%d' % (args['model_dir'], args['model_number'])
    model_properties = pd.read_pickle('%s/model_%d_properties.pkl' % (model_folder, args['model_number']))
    os.makedirs('%s/folium' % model_folder, exist_ok=True)

    folium_monthly_dir = '%s/%s' % (args['folium_outdir'], timestring[:6])
    os.makedirs(folium_monthly_dir, exist_ok=True)

    for fhr in args['forecast_hours']:
        
        filepath = args['folium_outdir'] + '/model_%d_%s_%s_f%03d.html' % (args['model_number'], args['nwp_model'], timestring, fhr)
        nwp_output_local_filename = '%s_%s_f%03d_features.pkl' % (args['nwp_model'], timestring, fhr)
        fronts_output_local_filename = 'model_%d_%s_%s_f%03d_features.pkl' % (args['model_number'], timestring, args['nwp_model'], fhr)
        
        if os.path.isfile(filepath):
            if args['overwrite']:
                pass
            else:
                print("File already exists: %s, skipping file......" % filepath)
                continue

        front_feature_groups = {}
        nwp_feature_groups = {}

        nwp_file = '%s/%s/%s_%s_f%03d.grib' % (args['grib_indir'], timestring[:6], args['nwp_model'], timestring, fhr)
        prediction_file = '%s/predictions/model_%d_%d%02d%02d%02d_global_%s_f%03d_probabilities.nc' % (model_folder, args['model_number'], yr, mo, dy, hr, args['nwp_model'], fhr)

        probs_ds = xr.open_dataset(prediction_file, engine='netcdf4').isel(time=0, forecast_hour=0).transpose('latitude', 'longitude')
        probs_ds = probs_ds.reindex(longitude=np.append(np.arange(180, 360, 0.25), np.arange(0, 180, 0.25)).flatten())
        
        lat = probs_ds['latitude'].values
        lon = probs_ds['longitude'].values
        lon = np.where(lon >= 180, lon - 360, lon)
        
        Lon, Lat = np.meshgrid(lon, lat)
        
        for front_label in ['CF', 'WF', 'SF', 'OF', 'DL']:
        
            front_type = FRONT_NAMES[front_label]
            cmap = CONTOUR_CMAPS[front_label]
            
            show_fg = front_label != 'SF'  # disable stationary fronts by default
            fg = folium.FeatureGroup(front_type, show=show_fg)
        
            cmap_probs, norm_probs = cm.get_cmap(cmap, 20), colors.Normalize(vmin=0, vmax=1.0)
        
            # calibrate model predictions
            ir_model = model_properties['calibration_models']['conus'][front_label]['100 km']
            original_shape = np.shape(probs_ds[front_label].values)
            probs_ds[front_label].values = ir_model.predict(probs_ds[front_label].values.flatten()).reshape(original_shape)
        
            prob_thresholds = np.append(np.arange(0.2, 1.0, 0.05), np.array([0.99, 0.995, 0.999, 1.00])).flatten()
            for prob in prob_thresholds:
                prob = np.round(prob, 2)
                contourf = plt.contourf(Lon, Lat, probs_ds[front_label].values, cmap=cmap_probs, norm=norm_probs, levels=[prob, prob + 0.05])
                prob_str = '%d%%' % (prob * 100) if prob not in [0.995, 0.999] else '%.1f%%' % (prob * 100)
                geojson_object = geojsoncontour.contourf_to_geojson(contourf=contourf, stroke_width=1, geojson_properties={'front-type': FRONT_NAMES[front_label],
                                                                                                                           'probability': prob_str})
                plt.close()
                as_dict = ast.literal_eval(geojson_object)
                style = {'color': as_dict['features'][0]['properties']['stroke'],
                         'weight': as_dict['features'][0]['properties']['stroke-width'],
                         'fillColor': as_dict['features'][0]['properties']['fill'],
                         'fillOpacity': 0.8,
                         'outline-color': None,
                         'opacity': 1.0}
            
                fg.add_child(folium.GeoJson(geojson_object,
                                tooltip=folium.GeoJsonTooltip(fields=['front-type', 'probability'],
                                                              aliases=['Front Type', 'Probability'],
                                                              labels=True,
                                                              localize=True,
                                                              style=
                                                              f"""
                                                              border-color: black;
                                                              """),
                                **style))
            
            front_feature_groups[front_label] = fg
        
        reindex_args = dict(longitude=np.append(np.arange(180, 360, 0.25), np.arange(0, 180, 0.25)).flatten())
        
        t2m = xr.open_dataset(nwp_file, engine='cfgrib', filter_by_keys={'cfVarName': 't2m'}).reindex(**reindex_args)['t2m'].values[:-1, :]
        d2m = xr.open_dataset(nwp_file, engine='cfgrib', filter_by_keys={'cfVarName': 'd2m'}).reindex(**reindex_args)['d2m'].values[:-1, :]
        u10 = xr.open_dataset(nwp_file, engine='cfgrib', filter_by_keys={'cfVarName': 'u10'}).reindex(**reindex_args)['u10'].values[:-1, :]
        v10 = xr.open_dataset(nwp_file, engine='cfgrib', filter_by_keys={'cfVarName': 'v10'}).reindex(**reindex_args)['v10'].values[:-1, :]
        sp = xr.open_dataset(nwp_file, engine='cfgrib', filter_by_keys={'cfVarName': 'sp'}).reindex(**reindex_args)['sp'].values[:-1, :]
        mslp = xr.open_dataset(nwp_file, engine='cfgrib', filter_by_keys={'cfVarName': 'prmsl'}).reindex(**reindex_args)['prmsl'].values[:-1, :] / 100  # Pa ---> hPa
        isobaricInhPa = xr.open_dataset(nwp_file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'cfVarName': 'gh'}).reindex(**reindex_args)
        z1000 = isobaricInhPa.sel(isobaricInhPa=1000)['gh'].values[:-1, :]
        z850 = isobaricInhPa.sel(isobaricInhPa=850)['gh'].values[:-1, :]
        
        nwp_feature_groups['mslp'] = _isobar_feature_group(mslp)
        nwp_feature_groups['10m_wind_barbs'] = _wind_barb_feature_group(u10, v10)
        nwp_feature_groups['2m_potential_temperature'] = _potential_temperature_feature_group(t2m, sp)
        nwp_feature_groups['2m_mixing_ratio'] = _mixing_ratio_feature_group(d2m, sp)
        nwp_feature_groups['1000-850hPa_thickness'] = _layer_thickness(z850 - z1000)
        
        geomap = folium.Map(location=[0, 0], zoom_start=2, min_zoom=2, max_bounds=True, min_lat=-90, zoom_control='bottomright',
            max_lat=90, min_lon=-180, max_lon=180, tiles="Cartodb positron", position='relative', height='95.238%', prefer_canvas=True)
        
        for fg in nwp_feature_groups:
            nwp_feature_groups[fg].add_to(geomap)
        
        for fg in front_feature_groups:
            front_feature_groups[fg].add_to(geomap)
        
        GroupedLayerControl(groups={'%s' % args['nwp_model'].upper(): [nwp_feature_groups[fg] for fg in nwp_feature_groups],
                                    'FrontFinder': [front_feature_groups[fg] for fg in front_feature_groups]},
                            exclusive_groups=False, collapsed=True).add_to(geomap)
        
        ### logos/attributions ###
        Fullscreen(title='Fullscreen', title_cancel='Exit fullscreen', position='topright').add_to(geomap)
        # FloatImage('https://pachtml-production.s3-us-west-2.amazonaws.com/www/ou/branding/logo.png', width='50px', height='50px', bottom='88', left='0.2', position='fixed').add_to(geomap)  # OU
        # FloatImage('https://arrc.ou.edu/news_images/img_0158.png', width='50px', height='50px', bottom='79', left='0.2', position='fixed').add_to(geomap)  # OU SoM
        # FloatImage('https://a.espncdn.com/combiner/i?img=/i/teamlogos/ncaa/500/2117.png&h=200&w=200', width='50px', height='50px', bottom='71', left='0.2', position='fixed').add_to(geomap)  # CMU
        # FloatImage('https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/NSF_Official_logo_Med_Res.png/1200px-NSF_Official_logo_Med_Res.png', width='50px', height='50px', bottom='63', left='0.2', position='fixed').add_to(geomap)  # NSF
        # FloatImage('https://www.ai2es.org/wp-content/uploads/2020/08/ai2es-logo-01.png', width='50px', height='50px', bottom='55', left='0.2', position='fixed').add_to(geomap)  # AI2ES
        
        valid_time = init_time + datetime.timedelta(hours=fhr)
        valid_yr, valid_mo, valid_dy, valid_hr = valid_time.year, valid_time.month, valid_time.day, valid_time.hour
        
        header_init_time_str = '%s %d-%02d-%02d %02d:00 UTC F%03d' % (args['nwp_model'].upper(), yr, mo, dy, hr, fhr)
        header_valid_time_str = '(Valid: %d-%02d-%02d %02d:00 UTC)' % (valid_yr, valid_mo, valid_dy, valid_hr)
        
        geomap.get_root().html.add_child(Element('<h3 align="center" style="font-size:16px;"><b>FrontFinder v1.1:</b> {} {}</h3>'.format(header_init_time_str, header_valid_time_str)))
        geomap.get_root().html.add_child(Element('<h3 align="center" style="font-size:14px;">Probability of a front within 100 km of a point</h3>'.format(header_init_time_str, header_valid_time_str)))
        
        # prevent browser from creating a 'focus' box when clicking
        geomap.get_root().html.add_child(Element(
            '''
            <style>
            path.leaflet-interactive:focus {
                outline: none;
            }
            </style>
            '''))

        geomap.save(filepath)
        
        with open('%s/%s' % (folium_monthly_dir, nwp_output_local_filename), 'wb') as f:
            pickle.dump(nwp_feature_groups, f)
    
        with open('%s/folium/%s' % (model_folder, fronts_output_local_filename), 'wb') as f:
            pickle.dump(front_feature_groups, f)