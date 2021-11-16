"""
Function that interpolates surface observations to a 0.25 degree grid and saves datasets for each variable.

Code written by: Andrew Justin and John Allen
Last updated: 11/15/2021 7:20 PM CST by Andrew Justin
"""

import pandas as pd
from metpy.interpolate import interpolate_to_grid, remove_nan_observations
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import pickle
import argparse
from errors import check_arguments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=False)
    parser.add_argument('--month', type=int, required=False)
    parser.add_argument('--day', type=int, required=False)
    parser.add_argument('--hour', type=int, required=False)
    parser.add_argument('--observations_indir', type=str, required=False, help='Path where the file(s) containing observations were saved.')
    parser.add_argument('--pickle_outdir', type=str, required=False, help='Path where the interpolated datasets will be saved.')

    args = parser.parse_args()
    provided_arguments = vars(args)
    required_arguments = ['year','month','day','hour','observations_indir','pickle_outdir']

    print("Checking arguments....",end='')
    check_arguments(provided_arguments, required_arguments)

    timestring = '%d%02d%02d%02d00' % (args.year, args.month, args.day, args.hour)

    df = pd.read_csv("%s/US_sfc_data_%s_%s.txt" % (args.observations_indir, timestring, timestring),
                     skiprows=6, sep=',', low_memory=False,
                     names=['station','valid','lon','lat','tmpf','dwpf','relh','drct','sknt','p01i','alti','mslp','vsby','gust',
                            'skyc1','skyc2','skyc3','skyc4','skyl1','skyl2','skyl3','skyl4','wxcodes','ice_accretion_1hr',
                            'ice_accretion_3hr','ice_accretion_6hr','peak_wind_gust','peak_wind_drct','peak_wind_time',
                            'feel','metar','snowdepth'], na_values='M')

    dt = datetime(args.year, args.month, args.day, args.hour)

    dt_minus = dt - timedelta(minutes=30)
    dt_plus = dt + timedelta(minutes=30)

    curr_data = df[(df.valid >= dt_minus.strftime('%Y-%m-%d %H:%M')) & (df.valid < dt_plus.strftime('%Y-%m-%d %H:%M'))].groupby('station').tail(1)

    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    tmpf = curr_data['tmpf'].values
    dwpf = curr_data['dwpf'].values
    relh = curr_data['relh'].values
    drct = curr_data['drct'].values
    sknt = curr_data['sknt'].values
    p01i = curr_data['p01i'].values
    alti = curr_data['alti'].values
    mslp = curr_data['mslp'].values
    vsby = curr_data['vsby'].values
    gust = curr_data['gust'].values
    skyl1 = curr_data['skyl1'].values
    skyl2 = curr_data['skyl2'].values
    skyl3 = curr_data['skyl3'].values
    skyl4 = curr_data['skyl4'].values
    feel = curr_data['feel'].values
    ice_1hr = curr_data['ice_accretion_1hr'].values
    ice_3hr = curr_data['ice_accretion_3hr'].values
    ice_6hr = curr_data['ice_accretion_6hr'].values
    peak_gust = curr_data['peak_wind_gust'].values
    peak_drct = curr_data['peak_wind_drct'].values

    lon_tmpf, lat_tmpf, tmpf = remove_nan_observations(lon, lat, tmpf)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_dwpf, lat_dwpf, dwpf = remove_nan_observations(lon, lat, dwpf)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_relh, lat_relh, relh = remove_nan_observations(lon, lat, relh)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_drct, lat_drct, drct = remove_nan_observations(lon, lat, drct)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_sknt, lat_sknt, sknt = remove_nan_observations(lon, lat, sknt)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_p01i, lat_p01i, p01i = remove_nan_observations(lon, lat, p01i)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_alti, lat_alti, alti = remove_nan_observations(lon, lat, alti)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_mslp, lat_mslp, mslp = remove_nan_observations(lon, lat, mslp)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_vsby, lat_vsby, vsby = remove_nan_observations(lon, lat, vsby)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_gust, lat_gust, gust = remove_nan_observations(lon, lat, gust)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_skyl1, lat_skyl1, skyl1 = remove_nan_observations(lon, lat, skyl1)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_skyl2, lat_skyl2, skyl2 = remove_nan_observations(lon, lat, skyl2)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_skyl3, lat_skyl3, skyl3 = remove_nan_observations(lon, lat, skyl3)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_skyl4, lat_skyl4, skyl4 = remove_nan_observations(lon, lat, skyl4)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_feel, lat_feel, feel = remove_nan_observations(lon, lat, feel)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_ice_1hr, lat_ice_1hr, ice_1hr = remove_nan_observations(lon, lat, ice_1hr)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_ice_3hr, lat_ice_3hr, ice_3hr = remove_nan_observations(lon, lat, ice_3hr)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_ice_6hr, lat_ice_6hr, ice_6hr = remove_nan_observations(lon, lat, ice_6hr)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_peak_gust, lat_peak_gust, peak_gust = remove_nan_observations(lon, lat, peak_gust)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values
    lon_peak_drct, lat_peak_drct, peak_drct = remove_nan_observations(lon, lat, peak_drct)
    lon = curr_data['lon'].values
    lat = curr_data['lat'].values

    print("Interpolating observations....0/17",end='\r')
    tmpfx, tmpfy, new_tmpf = interpolate_to_grid(lon_tmpf, lat_tmpf, tmpf, interp_type='cressman', hres=0.25)
    print("Interpolating observations....1/17",end='\r')
    dwpfx, dwpfy, new_dwpf = interpolate_to_grid(lon_dwpf, lat_dwpf, dwpf, interp_type='cressman', hres=0.25)
    print("Interpolating observations....2/17",end='\r')
    relhx, relhy, new_relh = interpolate_to_grid(lon_relh, lat_relh, relh, interp_type='cressman', hres=0.25)
    print("Interpolating observations....3/17",end='\r')
    drctx, drcty, new_drct = interpolate_to_grid(lon_drct, lat_drct, drct, interp_type='cressman', hres=0.25)
    print("Interpolating observations....4/17",end='\r')
    skntx, sknty, new_sknt = interpolate_to_grid(lon_sknt, lat_sknt, sknt, interp_type='cressman', hres=0.25)
    print("Interpolating observations....5/17",end='\r')
    p01ix, p01iy, new_p01i = interpolate_to_grid(lon_p01i, lat_p01i, p01i, interp_type='cressman', hres=0.25)
    print("Interpolating observations....6/17",end='\r')
    altix, altiy, new_alti = interpolate_to_grid(lon_alti, lat_alti, alti, interp_type='cressman', hres=0.25)
    print("Interpolating observations....7/17",end='\r')
    mslpx, mslpy, new_mslp = interpolate_to_grid(lon_mslp, lat_mslp, mslp, interp_type='cressman', hres=0.25)
    print("Interpolating observations....8/17",end='\r')
    vsbyx, vsbyy, new_vsby = interpolate_to_grid(lon_vsby, lat_vsby, vsby, interp_type='cressman', hres=0.25)
    print("Interpolating observations....9/17",end='\r')
    gustx, gusty, new_gust = interpolate_to_grid(lon_gust, lat_gust, gust, interp_type='cressman', hres=0.25)
    print("Interpolating observations....10/17",end='\r')
    skyl1x, skyl1y, new_skyl1 = interpolate_to_grid(lon_skyl1, lat_skyl1, skyl1, interp_type='cressman', hres=0.25)
    print("Interpolating observations....11/17",end='\r')
    skyl2x, skyl2y, new_skyl2 = interpolate_to_grid(lon_skyl2, lat_skyl2, skyl2, interp_type='cressman', hres=0.25)
    print("Interpolating observations....12/17",end='\r')
    skyl3x, skyl3y, new_skyl3 = interpolate_to_grid(lon_skyl3, lat_skyl3, skyl3, interp_type='cressman', hres=0.25)
    print("Interpolating observations....13/17",end='\r')
    skyl4x, skyl4y, new_skyl4 = interpolate_to_grid(lon_skyl4, lat_skyl4, skyl4, interp_type='cressman', hres=0.25)
    print("Interpolating observations....14/17",end='\r')
    feelx, feely, new_feel = interpolate_to_grid(lon_feel, lat_feel, feel, interp_type='cressman', hres=0.25)
    print("Interpolating observations....15/17",end='\r')
    peak_gustx, peak_gusty, new_peak_gust = interpolate_to_grid(lon_peak_gust, lat_peak_gust, peak_gust, interp_type='cressman', hres=0.25)
    print("Interpolating observations....16/17",end='\r')
    peak_drctx, peak_drcty, new_peak_drct = interpolate_to_grid(lon_peak_drct, lat_peak_drct, peak_drct, interp_type='cressman', hres=0.25)
    print("Interpolating observations....successful")

    final_tmpf = np.empty(shape=(len(tmpfy[:,0])-1,len(tmpfx[0])-1))
    final_dwpf = np.empty(shape=(len(dwpfy[:,0])-1,len(dwpfx[0])-1))
    final_relh = np.empty(shape=(len(relhy[:,0])-1,len(relhx[0])-1))
    final_drct = np.empty(shape=(len(drcty[:,0])-1,len(drctx[0])-1))
    final_sknt = np.empty(shape=(len(sknty[:,0])-1,len(skntx[0])-1))
    final_p01i = np.empty(shape=(len(p01iy[:,0])-1,len(p01ix[0])-1))
    final_alti = np.empty(shape=(len(altiy[:,0])-1,len(altix[0])-1))
    final_mslp = np.empty(shape=(len(mslpy[:,0])-1,len(mslpx[0])-1))
    final_vsby = np.empty(shape=(len(vsbyy[:,0])-1,len(vsbyx[0])-1))
    final_gust = np.empty(shape=(len(gusty[:,0])-1,len(gustx[0])-1))
    final_skyl1 = np.empty(shape=(len(skyl1y[:,0])-1,len(skyl1x[0])-1))
    final_skyl2 = np.empty(shape=(len(skyl2y[:,0])-1,len(skyl2x[0])-1))
    final_skyl3 = np.empty(shape=(len(skyl3y[:,0])-1,len(skyl3x[0])-1))
    final_skyl4 = np.empty(shape=(len(skyl4y[:,0])-1,len(skyl4x[0])-1))
    final_feel = np.empty(shape=(len(feely[:,0])-1,len(feelx[0])-1))
    final_peak_gust = np.empty(shape=(len(peak_gusty[:,0])-1,len(peak_gustx[0])-1))
    final_peak_drct = np.empty(shape=(len(peak_drcty[:,0])-1,len(peak_drctx[0])-1))

    """ Reformatting coordinates: the x index represents LATITUDE, the y index represents LONGITUDE """
    print("Reformatting coordinates to 0.25 degree grid....0/17",end='\r')
    for x in range(len(tmpfx[:,0])):
        for y in range(len(tmpfx[0,:])):
            tmpfx[x,y] = round(round(tmpfx[x,y] / 0.25) * 0.25, 2)
            tmpfy[x,y] = round(round(tmpfy[x,y] / 0.25) * 0.25, 2)
    print("Reformatting coordinates to 0.25 degree grid....1/17",end='\r')
    for x in range(len(dwpfx[:,0])):
        for y in range(len(dwpfx[0,:])):
            dwpfx[x,y] = round(round(dwpfx[x,y] / 0.25) * 0.25, 2)
            dwpfy[x,y] = round(round(dwpfy[x,y] / 0.25) * 0.25, 2)
    print("Reformatting coordinates to 0.25 degree grid....2/17",end='\r')
    for x in range(len(relhx[:,0])):
        for y in range(len(relhx[0,:])):
            relhx[x,y] = round(round(relhx[x,y] / 0.25) * 0.25, 2)
            relhy[x,y] = round(round(relhy[x,y] / 0.25) * 0.25, 2)
    print("Reformatting coordinates to 0.25 degree grid....3/17",end='\r')
    for x in range(len(drctx[:,0])):
        for y in range(len(drctx[0,:])):
            drctx[x,y] = round(round(drctx[x,y] / 0.25) * 0.25, 2)
            drcty[x,y] = round(round(drcty[x,y] / 0.25) * 0.25, 2)
    print("Reformatting coordinates to 0.25 degree grid....4/17",end='\r')
    for x in range(len(skntx[:,0])):
        for y in range(len(skntx[0,:])):
            skntx[x,y] = round(round(skntx[x,y] / 0.25) * 0.25, 2)
            sknty[x,y] = round(round(sknty[x,y] / 0.25) * 0.25, 2)
    print("Reformatting coordinates to 0.25 degree grid....5/17",end='\r')
    for x in range(len(p01ix[:,0])):
        for y in range(len(p01ix[0,:])):
            p01ix[x,y] = round(round(p01ix[x,y] / 0.25) * 0.25, 2)
            p01iy[x,y] = round(round(p01iy[x,y] / 0.25) * 0.25, 2)
    print("Reformatting coordinates to 0.25 degree grid....6/17",end='\r')
    for x in range(len(altix[:,0])):
        for y in range(len(altix[0,:])):
            altix[x,y] = round(round(altix[x,y] / 0.25) * 0.25, 2)
            altiy[x,y] = round(round(altiy[x,y] / 0.25) * 0.25, 2)
    print("Reformatting coordinates to 0.25 degree grid....7/17",end='\r')
    for x in range(len(mslpx[:,0])):
        for y in range(len(mslpx[0,:])):
            mslpx[x,y] = round(round(mslpx[x,y] / 0.25) * 0.25, 2)
            mslpy[x,y] = round(round(mslpy[x,y] / 0.25) * 0.25, 2)
    print("Reformatting coordinates to 0.25 degree grid....8/17",end='\r')
    for x in range(len(vsbyx[:,0])):
        for y in range(len(vsbyx[0,:])):
            vsbyx[x,y] = round(round(vsbyx[x,y] / 0.25) * 0.25, 2)
            vsbyy[x,y] = round(round(vsbyy[x,y] / 0.25) * 0.25, 2)
    print("Reformatting coordinates to 0.25 degree grid....9/17",end='\r')
    for x in range(len(gustx[:,0])):
        for y in range(len(gustx[0,:])):
            gustx[x,y] = round(round(gustx[x,y] / 0.25) * 0.25, 2)
            gusty[x,y] = round(round(gusty[x,y] / 0.25) * 0.25, 2)
    print("Reformatting coordinates to 0.25 degree grid....10/17",end='\r')
    for x in range(len(skyl1x[:,0])):
        for y in range(len(skyl1x[0,:])):
            skyl1x[x,y] = round(round(skyl1x[x,y] / 0.25) * 0.25, 2)
            skyl1y[x,y] = round(round(skyl1y[x,y] / 0.25) * 0.25, 2)
    print("Reformatting coordinates to 0.25 degree grid....11/17",end='\r')
    for x in range(len(skyl2x[:,0])):
        for y in range(len(skyl2x[0,:])):
            skyl2x[x,y] = round(round(skyl2x[x,y] / 0.25) * 0.25, 2)
            skyl2y[x,y] = round(round(skyl2y[x,y] / 0.25) * 0.25, 2)
    print("Reformatting coordinates to 0.25 degree grid....12/17",end='\r')
    for x in range(len(skyl3x[:,0])):
        for y in range(len(skyl3x[0,:])):
            skyl3x[x,y] = round(round(skyl3x[x,y] / 0.25) * 0.25, 2)
            skyl3y[x,y] = round(round(skyl3y[x,y] / 0.25) * 0.25, 2)
    print("Reformatting coordinates to 0.25 degree grid....13/17",end='\r')
    for x in range(len(skyl4x[:,0])):
        for y in range(len(skyl4x[0,:])):
            skyl4x[x,y] = round(round(skyl4x[x,y] / 0.25) * 0.25, 2)
            skyl4y[x,y] = round(round(skyl4y[x,y] / 0.25) * 0.25, 2)
    print("Reformatting coordinates to 0.25 degree grid....14/17",end='\r')
    for x in range(len(feelx[:,0])):
        for y in range(len(feelx[0,:])):
            feelx[x,y] = round(round(feelx[x,y] / 0.25) * 0.25, 2)
            feely[x,y] = round(round(feely[x,y] / 0.25) * 0.25, 2)
    print("Reformatting coordinates to 0.25 degree grid....15/17",end='\r')
    for x in range(len(peak_gustx[:,0])):
        for y in range(len(peak_gustx[0,:])):
            peak_gustx[x,y] = round(round(peak_gustx[x,y] / 0.25) * 0.25, 2)
            peak_gusty[x,y] = round(round(peak_gusty[x,y] / 0.25) * 0.25, 2)
    print("Reformatting coordinates to 0.25 degree grid....16/17",end='\r')
    for x in range(len(peak_drctx[:,0])):
        for y in range(len(peak_drctx[0,:])):
            peak_drctx[x,y] = round(round(peak_drctx[x,y] / 0.25) * 0.25, 2)
            peak_drcty[x,y] = round(round(peak_drcty[x,y] / 0.25) * 0.25, 2)
    print("Reformatting coordinates to 0.25 degree grid....successful")

    print("Reformatting observations to 0.25 degree grid....0/17",end='\r')
    for lon_point in range(len(final_tmpf[0,:])):
        for lat_point in range(len(final_tmpf[:,0])):
            if new_tmpf[lat_point,lon_point] != float("nan") and new_tmpf[lat_point,lon_point+1] != float("nan") and new_tmpf[lat_point+1,lon_point] != float("nan") \
                    and new_tmpf[lat_point+1,lon_point+1] != float("nan") and abs(tmpfx[lat_point,lon_point] - tmpfx[lat_point,lon_point+1]) == 0.25 \
                    and abs(tmpfy[lat_point,lon_point] - tmpfy[lat_point+1,lon_point]) == 0.25:
                final_tmpf[lat_point,lon_point] = np.mean([new_tmpf[lat_point,lon_point],new_tmpf[lat_point,lon_point+1],new_tmpf[lat_point+1,lon_point],new_tmpf[lat_point+1,lon_point+1]])
    print("Reformatting observations to 0.25 degree grid....1/17",end='\r')
    for lon_point in range(len(final_dwpf[0,:])):
        for lat_point in range(len(final_dwpf[:,0])):
            if new_dwpf[lat_point,lon_point] != float("nan") and new_dwpf[lat_point,lon_point+1] != float("nan") and new_dwpf[lat_point+1,lon_point] != float("nan") \
                    and new_dwpf[lat_point+1,lon_point+1] != float("nan") and abs(dwpfx[lat_point,lon_point] - dwpfx[lat_point,lon_point+1]) == 0.25 \
                    and abs(dwpfy[lat_point,lon_point] - dwpfy[lat_point+1,lon_point]) == 0.25:
                final_dwpf[lat_point,lon_point] = np.mean([new_dwpf[lat_point,lon_point],new_dwpf[lat_point,lon_point+1],new_dwpf[lat_point+1,lon_point],new_dwpf[lat_point+1,lon_point+1]])
    print("Reformatting observations to 0.25 degree grid....2/17",end='\r')
    for lon_point in range(len(final_relh[0,:])):
        for lat_point in range(len(final_relh[:,0])):
            if new_relh[lat_point,lon_point] != float("nan") and new_relh[lat_point,lon_point+1] != float("nan") and new_relh[lat_point+1,lon_point] != float("nan") \
                    and new_relh[lat_point+1,lon_point+1] != float("nan") and abs(relhx[lat_point,lon_point] - relhx[lat_point,lon_point+1]) == 0.25 \
                    and abs(relhy[lat_point,lon_point] - relhy[lat_point+1,lon_point]) == 0.25:
                final_relh[lat_point,lon_point] = np.mean([new_relh[lat_point,lon_point],new_relh[lat_point,lon_point+1],new_relh[lat_point+1,lon_point],new_relh[lat_point+1,lon_point+1]])
    print("Reformatting observations to 0.25 degree grid....3/17",end='\r')
    for lon_point in range(len(final_drct[0,:])):
        for lat_point in range(len(final_drct[:,0])):
            if new_drct[lat_point,lon_point] != float("nan") and new_drct[lat_point,lon_point+1] != float("nan") and new_drct[lat_point+1,lon_point] != float("nan") \
                    and new_drct[lat_point+1,lon_point+1] != float("nan") and abs(drctx[lat_point,lon_point] - drctx[lat_point,lon_point+1]) == 0.25 \
                    and abs(drcty[lat_point,lon_point] - drcty[lat_point+1,lon_point]) == 0.25:
                final_drct[lat_point,lon_point] = np.mean([new_drct[lat_point,lon_point],new_drct[lat_point,lon_point+1],new_drct[lat_point+1,lon_point],new_drct[lat_point+1,lon_point+1]])
    print("Reformatting observations to 0.25 degree grid....4/17",end='\r')
    for lon_point in range(len(final_sknt[0,:])):
        for lat_point in range(len(final_sknt[:,0])):
            if new_sknt[lat_point,lon_point] != float("nan") and new_sknt[lat_point,lon_point+1] != float("nan") and new_sknt[lat_point+1,lon_point] != float("nan") \
                    and new_sknt[lat_point+1,lon_point+1] != float("nan") and abs(skntx[lat_point,lon_point] - skntx[lat_point,lon_point+1]) == 0.25 \
                    and abs(sknty[lat_point,lon_point] - sknty[lat_point+1,lon_point]) == 0.25:
                final_sknt[lat_point,lon_point] = np.mean([new_sknt[lat_point,lon_point],new_sknt[lat_point,lon_point+1],new_sknt[lat_point+1,lon_point],new_sknt[lat_point+1,lon_point+1]])
    print("Reformatting observations to 0.25 degree grid....5/17",end='\r')
    for lon_point in range(len(final_p01i[0,:])):
        for lat_point in range(len(final_p01i[:,0])):
            if new_p01i[lat_point,lon_point] != float("nan") and new_p01i[lat_point,lon_point+1] != float("nan") and new_p01i[lat_point+1,lon_point] != float("nan") \
                    and new_p01i[lat_point+1,lon_point+1] != float("nan") and abs(p01ix[lat_point,lon_point] - p01ix[lat_point,lon_point+1]) == 0.25 \
                    and abs(p01iy[lat_point,lon_point] - p01iy[lat_point+1,lon_point]) == 0.25:
                final_p01i[lat_point,lon_point] = np.mean([new_p01i[lat_point,lon_point],new_p01i[lat_point,lon_point+1],new_p01i[lat_point+1,lon_point],new_p01i[lat_point+1,lon_point+1]])
    print("Reformatting observations to 0.25 degree grid....6/17",end='\r')
    for lon_point in range(len(final_alti[0,:])):
        for lat_point in range(len(final_alti[:,0])):
            if new_alti[lat_point,lon_point] != float("nan") and new_alti[lat_point,lon_point+1] != float("nan") and new_alti[lat_point+1,lon_point] != float("nan") \
                    and new_alti[lat_point+1,lon_point+1] != float("nan") and abs(altix[lat_point,lon_point] - altix[lat_point,lon_point+1]) == 0.25 \
                    and abs(altiy[lat_point,lon_point] - altiy[lat_point+1,lon_point]) == 0.25:
                final_alti[lat_point,lon_point] = np.mean([new_alti[lat_point,lon_point],new_alti[lat_point,lon_point+1],new_alti[lat_point+1,lon_point],new_alti[lat_point+1,lon_point+1]])
    print("Reformatting observations to 0.25 degree grid....7/17",end='\r')
    for lon_point in range(len(final_mslp[0,:])):
        for lat_point in range(len(final_mslp[:,0])):
            if new_mslp[lat_point,lon_point] != float("nan") and new_mslp[lat_point,lon_point+1] != float("nan") and new_mslp[lat_point+1,lon_point] != float("nan") \
                    and new_mslp[lat_point+1,lon_point+1] != float("nan") and abs(mslpx[lat_point,lon_point] - mslpx[lat_point,lon_point+1]) == 0.25 \
                    and abs(mslpy[lat_point,lon_point] - mslpy[lat_point+1,lon_point]) == 0.25:
                final_mslp[lat_point,lon_point] = np.mean([new_mslp[lat_point,lon_point],new_mslp[lat_point,lon_point+1],new_mslp[lat_point+1,lon_point],new_mslp[lat_point+1,lon_point+1]])
    print("Reformatting observations to 0.25 degree grid....8/17",end='\r')
    for lon_point in range(len(final_vsby[0,:])):
        for lat_point in range(len(final_vsby[:,0])):
            if new_vsby[lat_point,lon_point] != float("nan") and new_vsby[lat_point,lon_point+1] != float("nan") and new_vsby[lat_point+1,lon_point] != float("nan") \
                    and new_vsby[lat_point+1,lon_point+1] != float("nan") and abs(vsbyx[lat_point,lon_point] - vsbyx[lat_point,lon_point+1]) == 0.25 \
                    and abs(vsbyy[lat_point,lon_point] - vsbyy[lat_point+1,lon_point]) == 0.25:
                final_vsby[lat_point,lon_point] = np.mean([new_vsby[lat_point,lon_point],new_vsby[lat_point,lon_point+1],new_vsby[lat_point+1,lon_point],new_vsby[lat_point+1,lon_point+1]])
    print("Reformatting observations to 0.25 degree grid....9/17",end='\r')
    for lon_point in range(len(final_gust[0,:])):
        for lat_point in range(len(final_gust[:,0])):
            if new_gust[lat_point,lon_point] != float("nan") and new_gust[lat_point,lon_point+1] != float("nan") and new_gust[lat_point+1,lon_point] != float("nan") \
                    and new_gust[lat_point+1,lon_point+1] != float("nan") and abs(gustx[lat_point,lon_point] - gustx[lat_point,lon_point+1]) == 0.25 \
                    and abs(gusty[lat_point,lon_point] - gusty[lat_point+1,lon_point]) == 0.25:
                final_gust[lat_point,lon_point] = np.mean([new_gust[lat_point,lon_point],new_gust[lat_point,lon_point+1],new_gust[lat_point+1,lon_point],new_gust[lat_point+1,lon_point+1]])
    print("Reformatting observations to 0.25 degree grid....10/17",end='\r')
    for lon_point in range(len(final_skyl1[0,:])):
        for lat_point in range(len(final_skyl1[:,0])):
            if new_skyl1[lat_point,lon_point] != float("nan") and new_skyl1[lat_point,lon_point+1] != float("nan") and new_skyl1[lat_point+1,lon_point] != float("nan") \
                    and new_skyl1[lat_point+1,lon_point+1] != float("nan") and abs(skyl1x[lat_point,lon_point] - skyl1x[lat_point,lon_point+1]) == 0.25 \
                    and abs(skyl1y[lat_point,lon_point] - skyl1y[lat_point+1,lon_point]) == 0.25:
                final_skyl1[lat_point,lon_point] = np.mean([new_skyl1[lat_point,lon_point],new_skyl1[lat_point,lon_point+1],new_skyl1[lat_point+1,lon_point],new_skyl1[lat_point+1,lon_point+1]])
    print("Reformatting observations to 0.25 degree grid....11/17",end='\r')
    for lon_point in range(len(final_skyl2[0,:])):
        for lat_point in range(len(final_skyl2[:,0])):
            if new_skyl2[lat_point,lon_point] != float("nan") and new_skyl2[lat_point,lon_point+1] != float("nan") and new_skyl2[lat_point+1,lon_point] != float("nan") \
                    and new_skyl2[lat_point+1,lon_point+1] != float("nan") and abs(skyl2x[lat_point,lon_point] - skyl2x[lat_point,lon_point+1]) == 0.25 \
                    and abs(skyl2y[lat_point,lon_point] - skyl2y[lat_point+1,lon_point]) == 0.25:
                final_skyl2[lat_point,lon_point] = np.mean([new_skyl2[lat_point,lon_point],new_skyl2[lat_point,lon_point+1],new_skyl2[lat_point+1,lon_point],new_skyl2[lat_point+1,lon_point+1]])
    print("Reformatting observations to 0.25 degree grid....12/17",end='\r')
    for lon_point in range(len(final_skyl3[0,:])):
        for lat_point in range(len(final_skyl3[:,0])):
            if new_skyl3[lat_point,lon_point] != float("nan") and new_skyl3[lat_point,lon_point+1] != float("nan") and new_skyl3[lat_point+1,lon_point] != float("nan") \
                    and new_skyl3[lat_point+1,lon_point+1] != float("nan") and abs(skyl3x[lat_point,lon_point] - skyl3x[lat_point,lon_point+1]) == 0.25 \
                    and abs(skyl3y[lat_point,lon_point] - skyl3y[lat_point+1,lon_point]) == 0.25:
                final_skyl3[lat_point,lon_point] = np.mean([new_skyl3[lat_point,lon_point],new_skyl3[lat_point,lon_point+1],new_skyl3[lat_point+1,lon_point],new_skyl3[lat_point+1,lon_point+1]])
    print("Reformatting observations to 0.25 degree grid....13/17",end='\r')
    for lon_point in range(len(final_skyl4[0,:])):
        for lat_point in range(len(final_skyl4[:,0])):
            if new_skyl4[lat_point,lon_point] != float("nan") and new_skyl4[lat_point,lon_point+1] != float("nan") and new_skyl4[lat_point+1,lon_point] != float("nan") \
                    and new_skyl4[lat_point+1,lon_point+1] != float("nan") and abs(skyl4x[lat_point,lon_point] - skyl4x[lat_point,lon_point+1]) == 0.25 \
                    and abs(skyl4y[lat_point,lon_point] - skyl4y[lat_point+1,lon_point]) == 0.25:
                final_skyl4[lat_point,lon_point] = np.mean([new_skyl4[lat_point,lon_point],new_skyl4[lat_point,lon_point+1],new_skyl4[lat_point+1,lon_point],new_skyl4[lat_point+1,lon_point+1]])
    print("Reformatting observations to 0.25 degree grid....14/17",end='\r')
    for lon_point in range(len(final_feel[0,:])):
        for lat_point in range(len(final_feel[:,0])):
            if new_feel[lat_point,lon_point] != float("nan") and new_feel[lat_point,lon_point+1] != float("nan") and new_feel[lat_point+1,lon_point] != float("nan") \
                    and new_feel[lat_point+1,lon_point+1] != float("nan") and abs(feelx[lat_point,lon_point] - feelx[lat_point,lon_point+1]) == 0.25 \
                    and abs(feely[lat_point,lon_point] - feely[lat_point+1,lon_point]) == 0.25:
                final_feel[lat_point,lon_point] = np.mean([new_feel[lat_point,lon_point],new_feel[lat_point,lon_point+1],new_feel[lat_point+1,lon_point],new_feel[lat_point+1,lon_point+1]])
    print("Reformatting observations to 0.25 degree grid....15/17",end='\r')
    for lon_point in range(len(final_peak_gust[0,:])):
        for lat_point in range(len(final_peak_gust[:,0])):
            if new_peak_gust[lat_point,lon_point] != float("nan") and new_peak_gust[lat_point,lon_point+1] != float("nan") and new_peak_gust[lat_point+1,lon_point] != float("nan") \
                    and new_peak_gust[lat_point+1,lon_point+1] != float("nan") and abs(peak_gustx[lat_point,lon_point] - peak_gustx[lat_point,lon_point+1]) == 0.25 \
                    and abs(peak_gusty[lat_point,lon_point] - peak_gusty[lat_point+1,lon_point]) == 0.25:
                final_peak_gust[lat_point,lon_point] = np.mean([new_peak_gust[lat_point,lon_point],new_peak_gust[lat_point,lon_point+1],new_peak_gust[lat_point+1,lon_point],new_peak_gust[lat_point+1,lon_point+1]])
    print("Reformatting observations to 0.25 degree grid....16/17",end='\r')
    for lon_point in range(len(final_peak_drct[0,:])):
        for lat_point in range(len(final_peak_drct[:,0])):
            if new_peak_drct[lat_point,lon_point] != float("nan") and new_peak_drct[lat_point,lon_point+1] != float("nan") and new_peak_drct[lat_point+1,lon_point] != float("nan") \
                    and new_peak_drct[lat_point+1,lon_point+1] != float("nan") and abs(peak_drctx[lat_point,lon_point] - peak_drctx[lat_point,lon_point+1]) == 0.25 \
                    and abs(peak_drcty[lat_point,lon_point] - peak_drcty[lat_point+1,lon_point]) == 0.25:
                final_peak_drct[lat_point,lon_point] = np.mean([new_peak_drct[lat_point,lon_point],new_peak_drct[lat_point,lon_point+1],new_peak_drct[lat_point+1,lon_point],new_peak_drct[lat_point+1,lon_point+1]])
    print("Reformatting observations to 0.25 degree grid....successful")

    root_filename = 'SurfaceObs_%d%02d%02d%02d' % (args.year, args.month, args.day, args.hour)

    print("Saving datasets....",end='')
    tmpf_ds = xr.Dataset(data_vars={"tmpf": (['latitude', 'longitude'], final_tmpf)}, coords={'latitude': tmpfy[:-1,0], 'longitude': tmpfx[0,:-1]})
    tmpf_ds.load()
    outfile = open("%s/%d/%02d/%02d/%s_tmpf.pkl" % (args.pickle_outdir, args.year, args.month, args.day, root_filename), "wb")
    pickle.dump(tmpf_ds, outfile)
    dwpf_ds = xr.Dataset(data_vars={"dwpf": (['latitude', 'longitude'], final_dwpf)}, coords={'latitude': dwpfy[:-1,0], 'longitude': dwpfx[0,:-1]})
    dwpf_ds.load()
    outfile = open("%s/%d/%02d/%02d/%s_dwpf.pkl" % (args.pickle_outdir, args.year, args.month, args.day, root_filename), "wb")
    pickle.dump(dwpf_ds, outfile)
    relh_ds = xr.Dataset(data_vars={"relh": (['latitude', 'longitude'], final_relh)}, coords={'latitude': relhy[:-1,0], 'longitude': relhx[0,:-1]})
    relh_ds.load()
    outfile = open("%s/%d/%02d/%02d/%s_relh.pkl" % (args.pickle_outdir, args.year, args.month, args.day, root_filename), "wb")
    pickle.dump(relh_ds, outfile)
    drct_ds = xr.Dataset(data_vars={"drct": (['latitude', 'longitude'], final_drct)}, coords={'latitude': drcty[:-1,0], 'longitude': drctx[0,:-1]})
    drct_ds.load()
    outfile = open("%s/%d/%02d/%02d/%s_drct.pkl" % (args.pickle_outdir, args.year, args.month, args.day, root_filename), "wb")
    pickle.dump(drct_ds, outfile)
    sknt_ds = xr.Dataset(data_vars={"sknt": (['latitude', 'longitude'], final_sknt)}, coords={'latitude': sknty[:-1,0], 'longitude': skntx[0,:-1]})
    sknt_ds.load()
    outfile = open("%s/%d/%02d/%02d/%s_sknt.pkl" % (args.pickle_outdir, args.year, args.month, args.day, root_filename), "wb")
    pickle.dump(sknt_ds, outfile)
    p01i_ds = xr.Dataset(data_vars={"p01i": (['latitude', 'longitude'], final_p01i)}, coords={'latitude': p01iy[:-1,0], 'longitude': p01ix[0,:-1]})
    p01i_ds.load()
    outfile = open("%s/%d/%02d/%02d/%s_p01i.pkl" % (args.pickle_outdir, args.year, args.month, args.day, root_filename), "wb")
    pickle.dump(p01i_ds, outfile)
    alti_ds = xr.Dataset(data_vars={"alti": (['latitude', 'longitude'], final_alti)}, coords={'latitude': altiy[:-1,0], 'longitude': altix[0,:-1]})
    alti_ds.load()
    outfile = open("%s/%d/%02d/%02d/%s_alti.pkl" % (args.pickle_outdir, args.year, args.month, args.day, root_filename), "wb")
    pickle.dump(alti_ds, outfile)
    mslp_ds = xr.Dataset(data_vars={"mslp": (['latitude', 'longitude'], final_mslp)}, coords={'latitude': mslpy[:-1,0], 'longitude': mslpx[0,:-1]})
    mslp_ds.load()
    outfile = open("%s/%d/%02d/%02d/%s_mslp.pkl" % (args.pickle_outdir, args.year, args.month, args.day, root_filename), "wb")
    pickle.dump(mslp_ds, outfile)
    vsby_ds = xr.Dataset(data_vars={"vsby": (['latitude', 'longitude'], final_vsby)}, coords={'latitude': vsbyy[:-1,0], 'longitude': vsbyx[0,:-1]})
    vsby_ds.load()
    outfile = open("%s/%d/%02d/%02d/%s_vsby.pkl" % (args.pickle_outdir, args.year, args.month, args.day, root_filename), "wb")
    pickle.dump(vsby_ds, outfile)
    gust_ds = xr.Dataset(data_vars={"gust": (['latitude', 'longitude'], final_gust)}, coords={'latitude': gusty[:-1,0], 'longitude': gustx[0,:-1]})
    gust_ds.load()
    outfile = open("%s/%d/%02d/%02d/%s_gust.pkl" % (args.pickle_outdir, args.year, args.month, args.day, root_filename), "wb")
    pickle.dump(gust_ds, outfile)
    skyl1_ds = xr.Dataset(data_vars={"skyl1": (['latitude', 'longitude'], final_skyl1)}, coords={'latitude': skyl1y[:-1,0], 'longitude': skyl1x[0,:-1]})
    skyl1_ds.load()
    outfile = open("%s/%d/%02d/%02d/%s_skyl1.pkl" % (args.pickle_outdir, args.year, args.month, args.day, root_filename), "wb")
    pickle.dump(skyl1_ds, outfile)
    skyl2_ds = xr.Dataset(data_vars={"skyl2": (['latitude', 'longitude'], final_skyl2)}, coords={'latitude': skyl2y[:-1,0], 'longitude': skyl2x[0,:-1]})
    skyl2_ds.load()
    outfile = open("%s/%d/%02d/%02d/%s_skyl2.pkl" % (args.pickle_outdir, args.year, args.month, args.day, root_filename), "wb")
    pickle.dump(skyl2_ds, outfile)
    skyl3_ds = xr.Dataset(data_vars={"skyl3": (['latitude', 'longitude'], final_skyl3)}, coords={'latitude': skyl3y[:-1,0], 'longitude': skyl3x[0,:-1]})
    skyl3_ds.load()
    outfile = open("%s/%d/%02d/%02d/%s_skyl3.pkl" % (args.pickle_outdir, args.year, args.month, args.day, root_filename), "wb")
    pickle.dump(skyl3_ds, outfile)
    skyl4_ds = xr.Dataset(data_vars={"skyl4": (['latitude', 'longitude'], final_skyl4)}, coords={'latitude': skyl4y[:-1,0], 'longitude': skyl4x[0,:-1]})
    skyl4_ds.load()
    outfile = open("%s/%d/%02d/%02d/%s_skyl4.pkl" % (args.pickle_outdir, args.year, args.month, args.day, root_filename), "wb")
    pickle.dump(skyl4_ds, outfile)
    feel_ds = xr.Dataset(data_vars={"feel": (['latitude', 'longitude'], final_feel)}, coords={'latitude': feely[:-1,0], 'longitude': feelx[0,:-1]})
    feel_ds.load()
    outfile = open("%s/%d/%02d/%02d/%s_feel.pkl" % (args.pickle_outdir, args.year, args.month, args.day, root_filename), "wb")
    pickle.dump(feel_ds, outfile)
    peak_gust_ds = xr.Dataset(data_vars={"peak_gust": (['latitude', 'longitude'], final_peak_gust)}, coords={'latitude': peak_gusty[:-1,0], 'longitude': peak_gustx[0,:-1]})
    peak_gust_ds.load()
    outfile = open("%s/%d/%02d/%02d/%s_peak_gust.pkl" % (args.pickle_outdir, args.year, args.month, args.day, root_filename), "wb")
    pickle.dump(peak_gust_ds, outfile)
    peak_drct_ds = xr.Dataset(data_vars={"peak_drct": (['latitude', 'longitude'], final_peak_drct)}, coords={'latitude': peak_drcty[:-1,0], 'longitude': peak_drctx[0,:-1]})
    peak_drct_ds.load()
    outfile = open("%s/%d/%02d/%02d/%s_peak_drct.pkl" % (args.pickle_outdir, args.year, args.month, args.day, root_filename), "wb")
    pickle.dump(peak_drct_ds, outfile)
    outfile.close()
    print("successful")
