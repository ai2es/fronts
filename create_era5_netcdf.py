"""
Create ERA5 netCDF datasets.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.10.11

TODO:
    * remove hard-coded folder structure for surface and pressure level data
"""
import argparse
import numpy as np
import os
from utils import variables
from glob import glob
import xarray as xr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--netcdf_era5_indir', type=str, required=True, help="Input directory for the global ERA5 netCDF files.")
    parser.add_argument('--netcdf_outdir', type=str, required=True, help="Output directory for front netCDF files.")
    parser.add_argument('--date', type=int, nargs=3, required=True, help="Date for the data to be read in. (year, month, day)")

    args = vars(parser.parse_args())
    
    year, month, day = args['date'][0], args['date'][1], args['date'][2]

    era5_T_sfc_file = 'ERA5Global_%d_3hrly_2mT.nc' % year
    era5_Td_sfc_file = 'ERA5Global_%d_3hrly_2mTd.nc' % year
    era5_sp_file = 'ERA5Global_%d_3hrly_sp.nc' % year
    era5_u_sfc_file = 'ERA5Global_%d_3hrly_U10m.nc' % year
    era5_v_sfc_file = 'ERA5Global_%d_3hrly_V10m.nc' % year

    timestring = "%d-%02d-%02d" % (year, month, day)

    T_sfc_full_day = xr.open_mfdataset("%s/Surface/%s" % (args['netcdf_era5_indir'], era5_T_sfc_file), chunks={'latitude': 721, 'longitude': 1440, 'time': 4}).sel(time=('%s' % timestring))
    Td_sfc_full_day = xr.open_mfdataset("%s/Surface/%s" % (args['netcdf_era5_indir'], era5_Td_sfc_file), chunks={'latitude': 721, 'longitude': 1440, 'time': 4}).sel(time=('%s' % timestring))
    sp_full_day = xr.open_mfdataset("%s/Surface/%s" % (args['netcdf_era5_indir'], era5_sp_file), chunks={'latitude': 721, 'longitude': 1440, 'time': 4}).sel(time=('%s' % timestring))
    u_sfc_full_day = xr.open_mfdataset("%s/Surface/%s" % (args['netcdf_era5_indir'], era5_u_sfc_file), chunks={'latitude': 721, 'longitude': 1440, 'time': 4}).sel(time=('%s' % timestring))
    v_sfc_full_day = xr.open_mfdataset("%s/Surface/%s" % (args['netcdf_era5_indir'], era5_v_sfc_file), chunks={'latitude': 721, 'longitude': 1440, 'time': 4}).sel(time=('%s' % timestring))
    
    lons = T_sfc_full_day['longitude'].values
    lats = T_sfc_full_day['latitude'].values
    
    pressure_level_files = list(sorted(glob('%s/Pressure_Level/ERA5Global_PL_%s_3hrly_*.nc' % (args['netcdf_era5_indir'], year))))

    PL_data = xr.open_mfdataset(pressure_level_files, chunks={'latitude': 721, 'longitude': 1440, 'time': 4}).sel(time=('%s' % timestring))

    if not os.path.isdir('%s/%d%02d' % (args['netcdf_outdir'], year, month)):
        os.mkdir('%s/%d%02d' % (args['netcdf_outdir'], year, month))

    for hour in range(0, 24, 3):

        print(f"saving ERA5 data for {year}-%02d-%02d-%02dz" % (month, day, hour))

        timestep = '%d-%02d-%02dT%02d:00:00' % (year, month, day, hour)

        PL_850 = PL_data.sel(level=850, time=timestep)
        PL_900 = PL_data.sel(level=900, time=timestep)
        PL_950 = PL_data.sel(level=950, time=timestep)
        PL_1000 = PL_data.sel(level=1000, time=timestep)

        T_sfc = T_sfc_full_day.sel(time=timestep)['t2m'].values
        Td_sfc = Td_sfc_full_day.sel(time=timestep)['d2m'].values
        sp = sp_full_day.sel(time=timestep)['sp'].values
        u_sfc = u_sfc_full_day.sel(time=timestep)['u10'].values
        v_sfc = v_sfc_full_day.sel(time=timestep)['v10'].values

        theta_sfc = variables.potential_temperature(T_sfc, sp)  # Potential temperature
        theta_e_sfc = variables.equivalent_potential_temperature(T_sfc, Td_sfc, sp)  # Equivalent potential temperature
        theta_v_sfc = variables.virtual_temperature_from_dewpoint(T_sfc, Td_sfc, sp)  # Virtual potential temperature
        theta_w_sfc = variables.wet_bulb_potential_temperature(T_sfc, Td_sfc, sp)  # Wet-bulb potential temperature
        r_sfc = variables.mixing_ratio_from_dewpoint(Td_sfc, sp)  # Mixing ratio
        q_sfc = variables.specific_humidity_from_dewpoint(Td_sfc, sp)  # Specific humidity
        RH_sfc = variables.relative_humidity_from_dewpoint(T_sfc, Td_sfc)  # Relative humidity
        Tv_sfc = variables.virtual_temperature_from_dewpoint(T_sfc, Td_sfc, sp)  # Virtual temperature
        Tw_sfc = variables.wet_bulb_temperature(T_sfc, Td_sfc)  # Wet-bulb temperature

        q_850 = PL_850['q'].values
        q_900 = PL_900['q'].values
        q_950 = PL_950['q'].values
        q_1000 = PL_1000['q'].values
        T_850 = PL_850['t'].values
        T_900 = PL_900['t'].values
        T_950 = PL_950['t'].values
        T_1000 = PL_1000['t'].values
        u_850 = PL_850['u'].values
        u_900 = PL_900['u'].values
        u_950 = PL_950['u'].values
        u_1000 = PL_1000['u'].values
        v_850 = PL_850['v'].values
        v_900 = PL_900['v'].values
        v_950 = PL_950['v'].values
        v_1000 = PL_1000['v'].values
        z_850 = PL_850['z'].values
        z_900 = PL_900['z'].values
        z_950 = PL_950['z'].values
        z_1000 = PL_1000['z'].values

        Td_850 = variables.dewpoint_from_specific_humidity(85000, T_850, q_850)
        Td_900 = variables.dewpoint_from_specific_humidity(90000, T_900, q_900)
        Td_950 = variables.dewpoint_from_specific_humidity(95000, T_950, q_950)
        Td_1000 = variables.dewpoint_from_specific_humidity(100000, T_1000, q_1000)
        r_850 = variables.mixing_ratio_from_dewpoint(Td_850, 85000)
        r_900 = variables.mixing_ratio_from_dewpoint(Td_900, 90000)
        r_950 = variables.mixing_ratio_from_dewpoint(Td_950, 95000)
        r_1000 = variables.mixing_ratio_from_dewpoint(Td_1000, 100000)
        RH_850 = variables.relative_humidity_from_dewpoint(T_850, Td_850)
        RH_900 = variables.relative_humidity_from_dewpoint(T_900, Td_900)
        RH_950 = variables.relative_humidity_from_dewpoint(T_950, Td_950)
        RH_1000 = variables.relative_humidity_from_dewpoint(T_1000, Td_1000)
        theta_850 = variables.potential_temperature(T_850, 85000)
        theta_900 = variables.potential_temperature(T_900, 90000)
        theta_950 = variables.potential_temperature(T_950, 95000)
        theta_1000 = variables.potential_temperature(T_1000, 100000)
        theta_e_850 = variables.equivalent_potential_temperature(T_850, Td_850, 85000)
        theta_e_900 = variables.equivalent_potential_temperature(T_900, Td_900, 90000)
        theta_e_950 = variables.equivalent_potential_temperature(T_950, Td_950, 95000)
        theta_e_1000 = variables.equivalent_potential_temperature(T_1000, Td_1000, 100000)
        theta_v_850 = variables.virtual_temperature_from_dewpoint(T_850, Td_850, 85000)
        theta_v_900 = variables.virtual_temperature_from_dewpoint(T_900, Td_900, 90000)
        theta_v_950 = variables.virtual_temperature_from_dewpoint(T_950, Td_950, 95000)
        theta_v_1000 = variables.virtual_temperature_from_dewpoint(T_1000, Td_1000, 100000)
        theta_w_850 = variables.wet_bulb_potential_temperature(T_850, Td_850, 85000)
        theta_w_900 = variables.wet_bulb_potential_temperature(T_900, Td_900, 90000)
        theta_w_950 = variables.wet_bulb_potential_temperature(T_950, Td_950, 95000)
        theta_w_1000 = variables.wet_bulb_potential_temperature(T_1000, Td_1000, 100000)
        Tv_850 = variables.virtual_temperature_from_dewpoint(T_850, Td_850, 85000)
        Tv_900 = variables.virtual_temperature_from_dewpoint(T_900, Td_900, 90000)
        Tv_950 = variables.virtual_temperature_from_dewpoint(T_950, Td_950, 95000)
        Tv_1000 = variables.virtual_temperature_from_dewpoint(T_1000, Td_1000, 100000)
        Tw_850 = variables.wet_bulb_temperature(T_850, Td_850)
        Tw_900 = variables.wet_bulb_temperature(T_900, Td_900)
        Tw_950 = variables.wet_bulb_temperature(T_950, Td_950)
        Tw_1000 = variables.wet_bulb_temperature(T_1000, Td_1000)

        pressure_levels = ['surface', 1000, 950, 900, 850]
        
        arr_shape = (len(pressure_levels), len(lats), len(lons))
        T = np.empty(arr_shape)
        Td = np.empty(arr_shape)
        Tv = np.empty(arr_shape)
        Tw = np.empty(arr_shape)
        theta = np.empty(arr_shape)
        theta_e = np.empty(arr_shape)
        theta_v = np.empty(arr_shape)
        theta_w = np.empty(arr_shape)
        RH = np.empty(arr_shape)
        r = np.empty(arr_shape)
        q = np.empty(arr_shape)
        u = np.empty(arr_shape)
        v = np.empty(arr_shape)
        sp_z = np.empty(arr_shape)

        T[0, :, :], T[1, :, :], T[2, :, :], T[3, :, :], T[4, :, :] = T_sfc, T_1000, T_950, T_900, T_850
        Td[0, :, :], Td[1, :, :], Td[2, :, :], Td[3, :, :], Td[4, :, :] = Td_sfc, Td_1000, Td_950, Td_900, Td_850
        Tv[0, :, :], Tv[1, :, :], Tv[2, :, :], Tv[3, :, :], Tv[4, :, :] = Tv_sfc, Tv_1000, Tv_950, Tv_900, Tv_850
        Tw[0, :, :], Tw[1, :, :], Tw[2, :, :], Tw[3, :, :], Tw[4, :, :] = Tw_sfc, Tw_1000, Tw_950, Tw_900, Tw_850
        theta[0, :, :], theta[1, :, :], theta[2, :, :], theta[3, :, :], theta[4, :, :] = theta_sfc, theta_1000, theta_950, theta_900, theta_850
        theta_e[0, :, :], theta_e[1, :, :], theta_e[2, :, :], theta_e[3, :, :], theta_e[4, :, :] = theta_e_sfc, theta_e_1000, theta_e_950, theta_e_900, theta_e_850
        theta_v[0, :, :], theta_v[1, :, :], theta_v[2, :, :], theta_v[3, :, :], theta_v[4, :, :] = theta_v_sfc, theta_v_1000, theta_v_950, theta_v_900, theta_v_850
        theta_w[0, :, :], theta_w[1, :, :], theta_w[2, :, :], theta_w[3, :, :], theta_w[4, :, :] = theta_w_sfc, theta_w_1000, theta_w_950, theta_w_900, theta_w_850
        RH[0, :, :], RH[1, :, :], RH[2, :, :], RH[3, :, :], RH[4, :, :] = RH_sfc, RH_1000, RH_950, RH_900, RH_850
        r[0, :, :], r[1, :, :], r[2, :, :], r[3, :, :], r[4, :, :] = r_sfc, r_1000, r_950, r_900, r_850
        q[0, :, :], q[1, :, :], q[2, :, :], q[3, :, :], q[4, :, :] = q_sfc, q_1000, q_950, q_900, q_850
        u[0, :, :], u[1, :, :], u[2, :, :], u[3, :, :], u[4, :, :] = u_sfc, u_1000, u_950, u_900, u_850
        v[0, :, :], v[1, :, :], v[2, :, :], v[3, :, :], v[4, :, :] = v_sfc, v_1000, v_950, v_900, v_850
        sp_z[0, :, :], sp_z[1, :, :], sp_z[2, :, :], sp_z[3, :, :], sp_z[4, :, :] = sp/100, z_1000/98.0665, z_950/98.0665, z_900/98.0665, z_850/98.0665

        full_era5_dataset = xr.Dataset(data_vars=dict(T=(('pressure_level', 'latitude', 'longitude'), T),
                                                      Td=(('pressure_level', 'latitude', 'longitude'), Td),
                                                      Tv=(('pressure_level', 'latitude', 'longitude'), Tv),
                                                      Tw=(('pressure_level', 'latitude', 'longitude'), Tw),
                                                      theta=(('pressure_level', 'latitude', 'longitude'), theta),
                                                      theta_e=(('pressure_level', 'latitude', 'longitude'), theta_e),
                                                      theta_v=(('pressure_level', 'latitude', 'longitude'), theta_v),
                                                      theta_w=(('pressure_level', 'latitude', 'longitude'), theta_w),
                                                      RH=(('pressure_level', 'latitude', 'longitude'), RH),
                                                      r=(('pressure_level', 'latitude', 'longitude'), r * 1000),
                                                      q=(('pressure_level', 'latitude', 'longitude'), q * 1000),
                                                      u=(('pressure_level', 'latitude', 'longitude'), u),
                                                      v=(('pressure_level', 'latitude', 'longitude'), v),
                                                      sp_z=(('pressure_level', 'latitude', 'longitude'), sp_z)),
                                       coords=dict(pressure_level=pressure_levels, latitude=lats, longitude=lons)).astype('float32')

        full_era5_dataset = full_era5_dataset.expand_dims({'time': np.atleast_1d(timestep)})

        full_era5_dataset.to_netcdf(path='%s/%d%02d/era5_%d%02d%02d%02d_global.nc' % (args['netcdf_outdir'], year, month, year, month, day, hour), mode='w', engine='netcdf4')
