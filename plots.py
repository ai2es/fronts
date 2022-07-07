"""
Code for generating various plots:
    - ERA5/GDAS variable maps with fronts
    - Frequency plots
    - GDAS soundings

TODO:
    * Update file loading functions

Code written by: Andrew Justin (andrewjustin@ou.edu)

Last updated: 7/6/2022 8:30 PM CDT
"""

from matplotlib.colors import Normalize, ListedColormap
from metpy.plots import SkewT
import xarray as xr
from utils import plotting_utils
from utils.data_utils import reformat_fronts, expand_fronts
from glob import glob
import argparse
from errors import check_arguments
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from metpy.units import units
import numpy as np
import pandas as pd

# Default contour levels for geopotential height of a given pressure level ----> {pressure level: geopotential heights (meters)}
# If at the surface, isobars will be plotted instead ----> {'surface': pressure (hectopascals)}
# A single integer (e.g. 10) implies that 10 contour levels will automatically be calculated
ANALYSIS_PRESSURE_CONTOURS = {'surface': np.arange(880, 1100, 4), 925: np.arange(300, 1230, 30), 850: np.arange(900, 1930, 30),
                              700: np.arange(2400, 3630, 30), 500: np.arange(4800, 6360, 60), 300: np.arange(7800, 10320, 120),
                              200: np.arange(9600, 14520, 120)}

# Default contour levels for wind speed at a given pressure level ----> {pressure level: wind speed (knots)}
ANALYSIS_WIND_SPEED_CONTOURS = {925: np.arange(30, 75, 5), 850: np.arange(40, 85, 5), 700: np.arange(40, 85, 5),
                                500: np.arange(50, 200, 25), 300: np.arange(50, 225, 25), 200: np.arange(50, 225, 25)}

# Default contour values for temperature (celsius) at all pressure levels
ANALYSIS_TEMPERATURE_CONTOURS = {'surface': np.arange(-100, 55, 5), 925: np.arange(-80, 80, 2), 850: np.arange(-80, 80, 2),
                                 700: np.arange(-80, 80, 2), 500: np.arange(-80, 80, 2), 300: np.arange(-80, 80, 2),
                                 200: np.arange(-80, 80, 2)}

# Default contour values for temperature at a given pressure level ----> {pressure level: temperature (celsius)}
ANALYSIS_DEWPOINT_CONTOURS = {'surface': np.arange(0, 40, 2), 925: np.arange(12, 80, 2), 850: np.arange(8, 80, 2), 700: np.arange(4, 80, 2)}


def frequency_plots(fronts_pickle_indir, image_outdir):
    """
    Generate plots of the frequencies of different front types.

    Parameters
    ----------
    fronts_pickle_indir: str
        Input directory for pickle files containing frontal objects.
    image_outdir: str
        Output directory for the soundings.
    """
    front_files = sorted(glob(f'{fronts_pickle_indir}/*/*/*/FrontObjects*full.pkl'))

    cold_front_frequency = np.zeros(shape=(321, 961))
    warm_front_frequency = np.zeros(shape=(321, 961))
    stationary_front_frequency = np.zeros(shape=(321, 961))
    occluded_front_frequency = np.zeros(shape=(321, 961))

    cold_front_form_frequency = np.zeros(shape=(321, 961))
    warm_front_form_frequency = np.zeros(shape=(321, 961))
    stationary_front_form_frequency = np.zeros(shape=(321, 961))
    occluded_front_form_frequency = np.zeros(shape=(321, 961))

    cold_front_diss_frequency = np.zeros(shape=(321, 961))
    warm_front_diss_frequency = np.zeros(shape=(321, 961))
    stationary_front_diss_frequency = np.zeros(shape=(321, 961))
    occluded_front_diss_frequency = np.zeros(shape=(321, 961))

    cold_front_merged_frequency = np.zeros(shape=(321, 961))
    warm_front_merged_frequency = np.zeros(shape=(321, 961))
    stationary_front_merged_frequency = np.zeros(shape=(321, 961))
    occluded_front_merged_frequency = np.zeros(shape=(321, 961))

    instability_frequency = np.zeros(shape=(321, 961))
    trough_frequency = np.zeros(shape=(321, 961))
    tropical_trough_frequency = np.zeros(shape=(321, 961))
    dryline_frequency = np.zeros(shape=(321, 961))
    num_files = len(front_files)

    for file_no in range(num_files):
        print(f"File {file_no+1}/{num_files}", end='\r')
        fronts = pd.read_pickle(front_files[file_no])
        cold_front_frequency += np.where(fronts['identifier'].values == 1, 1, 0)
        warm_front_frequency += np.where(fronts['identifier'].values == 2, 1, 0)
        stationary_front_frequency += np.where(fronts['identifier'].values == 3, 1, 0)
        occluded_front_frequency += np.where(fronts['identifier'].values == 4, 1, 0)

        cold_front_form_frequency += np.where(fronts['identifier'].values == 5, 1, 0)
        warm_front_form_frequency += np.where(fronts['identifier'].values == 6, 1, 0)
        stationary_front_form_frequency += np.where(fronts['identifier'].values == 7, 1, 0)
        occluded_front_form_frequency += np.where(fronts['identifier'].values == 8, 1, 0)

        cold_front_diss_frequency += np.where(fronts['identifier'].values == 9, 1, 0)
        warm_front_diss_frequency += np.where(fronts['identifier'].values == 10, 1, 0)
        stationary_front_diss_frequency += np.where(fronts['identifier'].values == 11, 1, 0)
        occluded_front_diss_frequency += np.where(fronts['identifier'].values == 12, 1, 0)

        cold_front_merged_frequency += cold_front_frequency + cold_front_diss_frequency + cold_front_form_frequency
        warm_front_merged_frequency += warm_front_frequency + warm_front_diss_frequency + warm_front_form_frequency
        stationary_front_merged_frequency += stationary_front_frequency + stationary_front_diss_frequency + stationary_front_form_frequency
        occluded_front_merged_frequency += occluded_front_frequency + occluded_front_diss_frequency + occluded_front_form_frequency

        instability_frequency += np.where(fronts['identifier'].values == 13, 1, 0)
        trough_frequency += np.where(fronts['identifier'].values == 14, 1, 0)
        tropical_trough_frequency += np.where(fronts['identifier'].values == 15, 1, 0)
        dryline_frequency += np.where(fronts['identifier'].values == 16, 1, 0)

    print("Calculations done")
    extent = [130, 370, 0, 80]

    crs = ccrs.LambertConformal(central_longitude=250)

    fig, axs = plt.subplots(2, 2, figsize=(12, 7), subplot_kw={'projection': crs})
    axlist = axs.flatten()
    for ax in axlist:
        plotting_utils.plot_background(extent, ax=ax)
    test_file = pd.read_pickle(front_files[0])
    test_file['cold_frequency'] = (['latitude', 'longitude'], cold_front_frequency)
    test_file['warm_frequency'] = (['latitude', 'longitude'], warm_front_frequency)
    test_file['stationary_frequency'] = (['latitude', 'longitude'], stationary_front_frequency)
    test_file['occluded_frequency'] = (['latitude', 'longitude'], occluded_front_frequency)
    test_file = test_file.drop('identifier')

    test_file = test_file.sel(latitude=slice(1.0, 80.0))
    test_file = xr.where(test_file == 0, float("NaN"), test_file)
    test_file['cold_frequency'].plot(ax=axlist[0], cmap='plasma', x='longitude', y='latitude', transform=ccrs.PlateCarree())
    test_file['warm_frequency'].plot(ax=axlist[1], cmap='plasma', x='longitude', y='latitude', transform=ccrs.PlateCarree())
    test_file['stationary_frequency'].plot(ax=axlist[2], cmap='plasma', x='longitude', y='latitude', transform=ccrs.PlateCarree())
    test_file['occluded_frequency'].plot(ax=axlist[3], cmap='plasma', x='longitude', y='latitude', transform=ccrs.PlateCarree())
    axlist[0].set_title("Cold front frequency: 2008-2020")
    axlist[1].set_title("Warm front frequency: 2008-2020")
    axlist[2].set_title("Stationary front frequency: 2008-2020")
    axlist[3].set_title("Occluded front frequency: 2008-2020")
    plt.savefig(f"{image_outdir}/total_front_frequency.png", bbox_inches='tight', dpi=200)
    plt.close()


def gdas_sounding(year, month, day, hour, lat, lon, gdas_pickle_indir, sounding_outdir):
    """
    Plot sounding using GDAS data with a given set of coordinates.

    Parameters
    ----------
    year: year
    month: month
    day: day
    hour: hour
    lat: latitude (degrees)
    lon: longitude (degrees, 0-360 system)
    gdas_pickle_indir: str
        Directory where the pickle files containing GDAS data are stored.
    sounding_outdir: str
        Output directory for the soundings.
    """
    available_files = sorted(glob(f'{gdas_pickle_indir}/{year}/%02d/%02d/gdas_*_{year}%02d%02d%02d_full.pkl' % (month, day, month, day, hour)))

    num_pressure_levels = len(available_files)

    P = np.empty(num_pressure_levels)
    T = np.empty(num_pressure_levels)
    Td = np.empty(num_pressure_levels)
    Tv = np.empty(num_pressure_levels)
    u = np.empty(num_pressure_levels)
    v = np.empty(num_pressure_levels)

    for file in range(num_pressure_levels):
        current_dataset = pd.read_pickle(available_files[file]).sel(latitude=lat, longitude=lon)

        P[file] = current_dataset['pressure_level'].values
        T[file] = current_dataset['T'].values
        Td[file] = current_dataset['Td'].values
        Tv[file] = current_dataset['Tv'].values
        u[file] = current_dataset['u'].values * 1.944
        v[file] = current_dataset['v'].values * 1.944

    P, T, Td, Tv, u, v = zip(*sorted(zip(P, T, Td, Tv, u, v))[::-1])  # sort all arrays so that pressure is in descending order

    # 1000 mb (treating as surface) temperature and dewpoint in fahrenheit
    sfct_f = int((T[0] - 273.15) * 1.8 + 32)
    sfctd_f = int((Td[0] - 273.15) * 1.8 + 32)

    P = P * units['hPa']
    T = T * units['kelvin']
    Td = Td * units['kelvin']
    Tv = Tv * units['kelvin']
    u = u * units['knots']
    v = v * units['knots']

    fig = plt.figure(figsize=(9, 9), dpi=300)
    skew = SkewT(fig, rotation=35)

    skew.plot(P, T, 'r')  # Temperature profile
    skew.plot(P, Td, 'g')  # Dewpoint profile
    skew.plot(P, Tv, color='darkred', linestyle='--', linewidth=0.8)  # Virtual temperature profile
    skew.plot_barbs(P, u, v)  # Wind barbs
    skew.ax.set_ylim(1070, 100)  # Bottom of sounding set to 1070 mb to have space for 1000 mb temperature and dewpoint text
    skew.ax.set_xlim(-40, 50)

    # Isotherms
    skew.ax.axvline(0, color='blue', linestyle='--', linewidth=1, alpha=0.2)  # 0 celsius isotherm
    skew.ax.axvline(-20, color='blue', linestyle='--', linewidth=1, alpha=0.2)  # -20 celsius isotherm

    skew.plot_dry_adiabats(alpha=0.2, linestyles='-', colors='gray')  # Dry adiabats
    skew.plot_moist_adiabats(alpha=0.2, linestyles='--', colors='gray')  # Moist adiabats
    skew.plot_mixing_lines(alpha=0.2)  # Lines of constant mixing ratio

    # 1000 mb temperature and dewpoint text
    plt.text((sfct_f - 32) * 5/9 + 2, 1060, s=f'{sfct_f}F', color='red', bbox=dict(facecolor='white', edgecolor='red', pad=1.5), fontsize=10)
    plt.text((sfctd_f - 32) * 5/9 - 2, 1060, s=f'{sfctd_f}F', color='green', bbox=dict(facecolor='white', edgecolor='green', pad=1.5), fontsize=10)

    plt.xlabel("Temperature (C)")
    plt.ylabel("Pressure (hPa)")
    plt.title(f'GDAS {year}-%02d-%02d-%02dz ({lat}N, {lon}E)' % (month, day, hour))

    sounding_filename = f'{year}-%02d-%02d-%02dz_{lat}_{lon}' % (month, day, hour)
    plt.savefig(f'{sounding_outdir}/sounding_{sounding_filename}.png', bbox_inches='tight')
    plt.close()


def gdas_map(year, month, day, hour, variable, pressure_level, extent, gdas_pickle_indir, fronts_pickle_indir, image_outdir, slice_dataset=False, add_wind_barbs=True):
    """
    Plot variable in GDAS data.

    Parameters
    ----------
    year: year
    month: month
    day: day
    hour: hour
    variable: str
        Variable to plot.
    pressure_level: int
        Pressure level in hPa.
    extent: iterable
        Extent of the plot: [MIN_LON, MAX_LON, MIN_LAT, MAX_LAT]
    gdas_pickle_indir: str
        Input directory for pickle files containing GDAS data.
    fronts_pickle_indir: str
        Input directory for pickle files containing frontal objects.
    image_outdir: str
        Output directory for the soundings.
    slice_dataset: bool
        Slice the ERA5 dataset to only show data within the given extent. If False, data will be plotted across the entire domain,
        but the plot will be zoomed in around the given extent.
    add_wind_barbs: bool
        Plot wind barbs.
    """

    fronts_file = '%s\\%d\\%02d\\%02d\\FrontObjects_%d%02d%02d%02d_full.pkl' % (fronts_pickle_indir, year, month, day, year, month, day, hour)
    gdas_file = '%s\\%d\\%02d\\%02d\\gdas_%d_%d%02d%02d%02d_full.pkl' % (gdas_pickle_indir, year, month, day, pressure_level, year, month, day, hour)

    # Open the files, and slice the datasets if 'slice_dataset' is True
    fronts_dataset = pd.read_pickle(fronts_file)
    gdas_dataset = pd.read_pickle(gdas_file)
    if slice_dataset:
        min_lon, max_lon, min_lat, max_lat = extent[0], extent[1], extent[2], extent[3]
        fronts_dataset = fronts_dataset.isel(latitude=slice(int((80-max_lat)*4), int((80-min_lat)*4)), longitude=slice(int((min_lon-130)*4), int((max_lon-130)*4)))
        gdas_dataset = gdas_dataset.isel(latitude=slice(int((80-max_lat)*4), int((80-min_lat)*4)), longitude=slice(int((min_lon-130)*4), int((max_lon-130)*4)))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), subplot_kw={'projection': ccrs.LambertConformal(central_longitude=250)})
    plotting_utils.plot_background(extent, ax=ax)  # Plot the background with country and state borders
    fronts_dataset, names, _, colors_types, _ = reformat_fronts(fronts_dataset, front_types='ALL', return_colors=True, return_names=True)  # Return colors and names for the front types for making the colorbar
    fronts_dataset = expand_fronts(fronts_dataset)  # Expand the fronts by one pixel in all direction
    fronts_dataset = fronts_dataset['identifier'].where(fronts_dataset['identifier'] > 0)  # Turn all zeros to NaNs so they don't show up in the plot

    """ Add colorbar for fronts """
    cmap_fronts = ListedColormap(colors_types, name='from_list', N=len(names))  # Colormap for fronts
    norm_fronts = Normalize(vmin=1, vmax=len(names) + 1)  # Colorbar normalization
    plotting_utils.create_colorbar_for_fronts(names, cmap_fronts, norm_fronts)  # Create the colorbar

    # Plot wind barbs
    if add_wind_barbs:

        # Interval at which to plot the wind barbs. This value divided by 4 equals the spatial grid on which the barbs are plotted.
        # ex: if step equals 8, the barbs are plotted on an 8/4 degree x 8/4 degree (2 degree x 2 degree) grid.
        step = 16

        lons_barbs = gdas_dataset['longitude'].values[::step]
        lats_barbs = gdas_dataset['latitude'].values[::step]

        # Wind values are converted from meters per second to knots
        # 1.94384 knots = 1 meter per second
        u_wind = gdas_dataset['u'].values[::step, ::step] * 1.94384
        v_wind = gdas_dataset['v'].values[::step, ::step] * 1.94384

        ax.barbs(lons_barbs, lats_barbs, u_wind, v_wind, color='black', length=4, linewidth=0.3, sizes={'height': 0.5, 'width': 0}, transform=ccrs.PlateCarree())

    height_contours = ANALYSIS_PRESSURE_CONTOURS[pressure_level]  # Select height contours for the given pressure level

    gdas_dataset['z'].plot.contour(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree(), linewidths=0.6, levels=height_contours, colors='black')  # Plot height contours
    gdas_dataset[variable].plot.contourf(ax=ax, x='longitude', y='latitude', alpha=0.5, transform=ccrs.PlateCarree(), cbar_kwargs={'orientation': 'horizontal', 'shrink': 0.5})  # Plot GDAS variable contours
    fronts_dataset.plot(ax=ax, x='longitude', y='latitude', cmap=cmap_fronts, norm=norm_fronts, transform=ccrs.PlateCarree(), add_colorbar=False)  # Plot fronts

    ax.set_title('', loc='center')
    ax.set_title(f"GDAS plot: {variable} @ {pressure_level} hPa", loc='left', fontsize=6, pad=3)
    ax.set_title(f"Valid: {year}-%02d-%02d-%02dz" % (month, day, hour), loc='right', fontsize=6, pad=3)
    plt.savefig('%s\\%d\\%02d\\%02d\\%02d\\gdas_%s_%d_%d%02d%02d%02d.png' % (image_outdir, year, month, day, hour, variable, pressure_level, year, month, day, hour), dpi=500, bbox_inches='tight')
    plt.close()


def gdas_analysis(year, month, day, hour, pressure_level, extent, gdas_pickle_indir, fronts_pickle_indir, image_outdir):
    """
    Create an analysis with GDAS data at a given timestep and pressure level.

    Parameters
    ----------
    year: year
    month: month
    day: day
    hour: hour
    pressure_level: int
        Pressure level in hPa.
    extent: iterable
        Extent of the plot: [MIN_LON, MAX_LON, MIN_LAT, MAX_LAT]
    gdas_pickle_indir: str
        Input directory for pickle files containing GDAS data.
    fronts_pickle_indir: str
        Input directory for pickle files containing frontal objects.
    image_outdir: str
        Output directory for the soundings.
    """

    fronts_file = '%s\\%d\\%02d\\%02d\\FrontObjects_%d%02d%02d%02d_full.pkl' % (fronts_pickle_indir, year, month, day, year, month, day, hour)
    gdas_file = '%s\\%d\\%02d\\%02d\\gdas_%s_%d%02d%02d%02d_full.pkl' % (gdas_pickle_indir, year, month, day, pressure_level, year, month, day, hour)

    fronts_dataset = pd.read_pickle(fronts_file)
    gdas_dataset = pd.read_pickle(gdas_file)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), subplot_kw={'projection': ccrs.LambertConformal(central_longitude=250)})
    plotting_utils.plot_background(extent, ax=ax, linewidth=0.2)  # Plot the background with country and state borders
    fronts_dataset, names, _, colors_types, _ = reformat_fronts(fronts_dataset, front_types='ALL', return_colors=True, return_names=True)  # Return colors and names for the front types for making the colorbar
    fronts_dataset = expand_fronts(fronts_dataset)  # Expand the fronts by one pixel in all direction
    fronts_dataset = fronts_dataset['identifier'].where(fronts_dataset['identifier'] > 0)  # Turn all zeros to NaNs so they don't show up in the plot

    """ Add colorbar for fronts """
    cmap_fronts = ListedColormap(colors_types, name='from_list', N=len(names))  # Colormap for fronts
    norm_fronts = Normalize(vmin=1, vmax=len(names) + 1)  # Colorbar normalization

    # Interval at which to plot the wind barbs. This value divided by 4 equals the spatial grid on which the barbs are plotted.
    # ex: if step equals 8, the barbs are plotted on an 8/4 degree x 8/4 degree (2 degree x 2 degree) grid.
    step = 8

    # Wind values are converted from meters per second to knots
    # 1.94384 knots = 1 meter per second
    u_wind = gdas_dataset['u'].values[::step, ::step] * 1.94384
    v_wind = gdas_dataset['v'].values[::step, ::step] * 1.94384
    gdas_dataset['wind_speed'] = (('latitude', 'longitude'), np.sqrt(np.power(gdas_dataset['u'].values, 2) + np.power(gdas_dataset['v'].values, 2)) * 1.94384)

    lons_barbs = gdas_dataset['longitude'].values[::step]
    lats_barbs = gdas_dataset['latitude'].values[::step]

    # Select height and wind contours for the given pressure level
    pressure_contours = ANALYSIS_PRESSURE_CONTOURS[pressure_level]

    # Convert temperature and dewpoint from kelvin to celsius
    gdas_dataset['T'] -= 273.15
    gdas_dataset['Td'] -= 273.15

    if pressure_level == 925 or pressure_level == 850 or pressure_level == 700 or pressure_level == 'surface':
        # Plot temperature contours
        cs_T = gdas_dataset['T'].plot.contour(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree(), linewidths=0.4, linestyles='--', levels=ANALYSIS_TEMPERATURE_CONTOURS[pressure_level], colors='red')
        plt.clabel(cs_T, fontsize=3)

        # Plot dewpoint contours
        cs_Td = gdas_dataset['Td'].plot.contour(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree(), linewidths=0.4, linestyles='--', levels=ANALYSIS_DEWPOINT_CONTOURS[pressure_level], colors='green')
        plt.clabel(cs_Td, fontsize=3)

    if pressure_level == 'surface':

        title = "GDAS analysis: surface"

        plotting_utils.create_colorbar_for_fronts(names, cmap_fronts, norm_fronts, axis_loc=(0.8265, 0.11, 0.015, 0.77))  # Create the colorbar

        # Plot isobars
        cs_mslp = gdas_dataset['mslp'].plot.contour(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree(), linewidths=1, levels=pressure_contours, colors='black')
        plt.clabel(cs_mslp, fontsize=3)

    else:

        title = f"GDAS analysis: {pressure_level} hPa"

        plotting_utils.create_colorbar_for_fronts(names, cmap_fronts, norm_fronts)  # Create the colorbar

        wind_contours = ANALYSIS_WIND_SPEED_CONTOURS[pressure_level]

        # Plot height contours
        cs_z = gdas_dataset['z'].plot.contour(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree(), linewidths=0.4, levels=pressure_contours, colors='black')
        plt.clabel(cs_z, fontsize=3)

        gdas_dataset['wind_speed'].plot.contourf(ax=ax, x='longitude', y='latitude', extend='neither', transform=ccrs.PlateCarree(),
                                                 levels=wind_contours, cmap='brg', alpha=0.5, cbar_kwargs={'orientation': 'horizontal',
                                                                                                           'shrink': 0.5,
                                                                                                           'label': 'Wind Speed (knots)'})

    fronts_dataset.plot(ax=ax, x='longitude', y='latitude', extend='neither', cmap=cmap_fronts, norm=norm_fronts, transform=ccrs.PlateCarree(), add_colorbar=False)  # Plot fronts
    ax.barbs(lons_barbs, lats_barbs, u_wind, v_wind, color='black', length=4, linewidth=0.4, sizes={'height': 0.65, 'width': 0.2}, transform=ccrs.PlateCarree())  # Add wind barbs

    ax.set_title('', loc='center')
    ax.set_title(title, loc='left', fontsize=6, pad=3)
    ax.set_title(f"Valid: {year}-%02d-%02d-%02dz" % (month, day, hour), loc='right', fontsize=6, pad=3)
    plt.savefig('%s\\%d\\%02d\\%02d\\%02d\\gdas_analysis_%s_%d%02d%02d%02d.png' % (image_outdir, year, month, day, hour, pressure_level, year, month, day, hour), dpi=500, bbox_inches='tight')
    plt.close()


def era5_map(year, month, day, hour, variable, extent, era5_pickle_indir, fronts_pickle_indir, image_outdir, slice_dataset=False, add_wind_barbs=False):
    """
    Plot variable in ERA5 data.

    Parameters
    ----------
    year: year
    month: month
    day: day
    hour: hour
    variable: str
        Variable to plot.
    extent: iterable
        Extent of the plot: [MIN_LON, MAX_LON, MIN_LAT, MAX_LAT]
    era5_pickle_indir: str
        Input directory for pickle files containing ERA5 data.
    fronts_pickle_indir: str
        Input directory for pickle files containing frontal objects.
    image_outdir: str
        Output directory for the soundings.
    slice_dataset: bool
        Slice the ERA5 dataset to only show data within the given extent. If False, data will be plotted across the entire domain,
        but the plot will be zoomed in around the given extent.
    add_wind_barbs: bool
        Plot wind barbs.
    """

    fronts_file = '%s\\%d\\%02d\\%02d\\FrontObjects_%d%02d%02d%02d_full.pkl' % (fronts_pickle_indir, year, month, day, year, month, day, hour)
    era5_file = '%s\\%d\\%02d\\%02d\\Data_60var_%d%02d%02d%02d_full.pkl' % (era5_pickle_indir, year, month, day, year, month, day, hour)

    # Open the files, and slice the datasets if 'slice_dataset' is True
    fronts_dataset = pd.read_pickle(fronts_file)
    era5_dataset = pd.read_pickle(era5_file)
    if slice_dataset:
        min_lon, max_lon, min_lat, max_lat = extent[0], extent[1], extent[2], extent[3]
        fronts_dataset = fronts_dataset.isel(latitude=slice(int((80-max_lat)*4), int((80-min_lat)*4)), longitude=slice(int((min_lon-130)*4), int((max_lon-130)*4)))
        era5_dataset = era5_dataset.isel(latitude=slice(int((80-max_lat)*4), int((80-min_lat)*4)), longitude=slice(int((min_lon-130)*4), int((max_lon-130)*4)))

    variable_dataset = era5_dataset[variable]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), subplot_kw={'projection': ccrs.LambertConformal(central_longitude=250)})
    plotting_utils.plot_background(extent, ax=ax, linewidth=0.2)  # Plot the background with country and state borders
    fronts_dataset, names, _, colors_types, _ = reformat_fronts(fronts_dataset, front_types='ALL', return_colors=True, return_names=True)  # Return colors and names for the front types for making the colorbar
    fronts_dataset = expand_fronts(fronts_dataset)  # Expand the fronts by one pixel in all direction
    fronts_dataset = fronts_dataset['identifier'].where(fronts_dataset['identifier'] > 0)  # Turn all zeros to NaNs so they don't show up in the plot

    """ Add colorbar for fronts """
    cmap_fronts = ListedColormap(colors_types, name='from_list', N=len(names))  # Colormap for fronts
    norm_fronts = Normalize(vmin=1, vmax=len(names) + 1)  # Colorbar normalization
    plotting_utils.create_colorbar_for_fronts(names, cmap_fronts, norm_fronts)  # Create the colorbar

    if add_wind_barbs:
        # Interval at which to plot the wind barbs. This value divided by 4 equals the spatial grid on which the barbs are plotted.
        # ex: if step equals 8, the barbs are plotted on an 8/4 degree x 8/4 degree (2 degree x 2 degree) grid.
        step = 16

        # Wind values are converted from meters per second to knots
        # 1.94384 knots = 1 meter per second
        u_wind = era5_dataset['u10'].values[::step, ::step] * 1.94384
        v_wind = era5_dataset['v10'].values[::step, ::step] * 1.94384

        lons_barbs = era5_dataset['longitude'].values[::step]
        lats_barbs = era5_dataset['latitude'].values[::step]

        ax.barbs(lons_barbs, lats_barbs, u_wind, v_wind, color='black', length=4, linewidth=0.3, sizes={'height': 0.5, 'width': 0}, transform=ccrs.PlateCarree())

    variable_dataset.plot.contourf(ax=ax, x='longitude', y='latitude', alpha=0.5, transform=ccrs.PlateCarree(), levels=10, cmap='hot', cbar_kwargs={'orientation': 'horizontal', 'shrink': 0.5})  # Plot ERA5 variable
    fronts_dataset['identifier'].plot(ax=ax, x='longitude', y='latitude',  extend='neither', cmap=cmap_fronts, norm=norm_fronts, transform=ccrs.PlateCarree(), add_colorbar=False)  # Plot fronts

    ax.set_title('', loc='center')
    ax.set_title(f"ERA5 re-analysis: {variable}", loc='left', fontsize=6, pad=3)
    ax.set_title(f"Valid: {year}-%02d-%02d-%02dz" % (month, day, hour), loc='right', fontsize=6, pad=3)
    plt.savefig('%s\\%d\\%02d\\%02d\\%02d\\era5_%s_%d%02d%02d%02d.png' % (image_outdir, year, month, day, hour, variable, year, month, day, hour), dpi=500, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create labeled data for the specified day")
    parser.add_argument('--frequency_plots', action='store_true', help='Generate frequency plots')
    parser.add_argument('--gdas_map', action='store_true', help='Plot GDAS data over a domain')
    parser.add_argument('--gdas_analysis', action='store_true', help='GDAS analysis over a domain')
    parser.add_argument('--pressure_level', help="Pressure level (hPa) or 'surface'.")
    parser.add_argument('--gdas_sounding', action='store_true', help='Plot GDAS sounding at a given point.')
    parser.add_argument('--gdas_pickle_indir', type=str, help='Input directory for the GDAS pickle files.')
    parser.add_argument('--sounding_coords', type=float, nargs=2, help='Coordinates for the GDAS sounding. [Lat, Lon]')
    parser.add_argument('--era5_map', action='store_true', help='Plot ERA5 data over a domain')
    parser.add_argument('--era5_pickle_indir', type=str, help='Input directory for the ERA5 pickle files.')
    parser.add_argument('--fronts_pickle_indir', type=str, help='Input directory for the front object files.')
    parser.add_argument('--variable', type=str, help='Variable to plot')
    parser.add_argument('--extent', type=float, nargs=4, help='Extent of the plot. [MIN_LON, MAX_LON, MIN_LAT, MAX_LAT]')
    parser.add_argument('--station_plot', action='store_true', help='Generate map of ASOS station plots across the CONUS.')
    parser.add_argument('--image_outdir', type=str, help='Output directory for images.')

    parser.add_argument('--year', type=int)
    parser.add_argument('--month', type=int)
    parser.add_argument('--day', type=int)
    parser.add_argument('--hour', type=int)

    args = parser.parse_args()
    provided_arguments = vars(args)

    if args.frequency_plots:
        required_arguments = ['fronts_pickle_indir', 'image_outdir']
        check_arguments(provided_arguments, required_arguments)
        frequency_plots(args.fronts_pickle_indir, args.image_outdir)

    if args.gdas_map:
        required_arguments = ['gdas_pickle_indir', 'fronts_pickle_indir', 'variable', 'pressure_level', 'extent', 'image_outdir', 'year', 'month', 'day', 'hour']
        check_arguments(provided_arguments, required_arguments)
        gdas_map(args.year, args.month, args.day, args.hour, args.variable, args.pressure_level, args.extent, args.gdas_pickle_indir,
                 args.fronts_pickle_indir, args.image_outdir)

    if args.gdas_analysis:
        required_arguments = ['gdas_pickle_indir', 'fronts_pickle_indir', 'pressure_level', 'extent', 'image_outdir', 'year', 'month', 'day', 'hour']
        check_arguments(provided_arguments, required_arguments)
        gdas_analysis(args.year, args.month, args.day, args.hour, args.pressure_level, args.extent, args.gdas_pickle_indir,
                      args.fronts_pickle_indir, args.image_outdir)

    if args.gdas_sounding:
        required_arguments = ['gdas_pickle_indir', 'sounding_coords', 'image_outdir', 'year', 'month', 'day', 'hour']
        check_arguments(provided_arguments, required_arguments)
        lat, lon = args.sounding_coords[0], args.sounding_coords[1]
        gdas_sounding(args.year, args.month, args.day, args.hour, lat, lon, args.gdas_pickle_indir, args.image_outdir)

    if args.era5_map:
        required_arguments = ['era5_pickle_indir', 'fronts_pickle_indir', 'variable', 'extent', 'image_outdir', 'year', 'month', 'day', 'hour']
        check_arguments(provided_arguments, required_arguments)
        era5_map(args.year, args.month, args.day, args.hour, args.variable, args.extent, args.era5_pickle_indir, args.fronts_pickle_indir, args.image_outdir)
