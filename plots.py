"""
Code for generating various plots:
    - ERA5/GDAS variable maps with fronts
    - Frequency plots
    - GDAS soundings

Code written by: Andrew Justin (andrewjustin@ou.edu)

Last updated: 6/25/2022 10:08 PM CDT
"""

from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from metpy.plots import SkewT
import xarray as xr
from utils.plotting_utils import plot_background
from utils.data_utils import reformat_fronts, expand_fronts
from glob import glob
import argparse
from errors import check_arguments
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from metpy.units import units
import numpy as np
import pandas as pd


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
        plot_background(extent, ax=ax)
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

    skew.plot(P, T, 'r')
    skew.plot(P, Td, 'g')
    skew.plot(P, Tv, color='darkred', linestyle='--', linewidth=0.8)
    skew.plot_barbs(P, u, v)
    skew.ax.set_ylim(1070, 100)
    skew.ax.set_xlim(-40, 50)

    skew.ax.axvline(0, color='blue', linestyle='--', linewidth=1, alpha=0.2)
    skew.ax.axvline(-20, color='blue', linestyle='--', linewidth=1, alpha=0.2)

    skew.plot_dry_adiabats(alpha=0.2, linestyles='-', colors='gray')
    skew.plot_moist_adiabats(alpha=0.2, linestyles='--', colors='gray')
    skew.plot_mixing_lines(alpha=0.2)

    plt.text((sfct_f - 32) * 5/9 + 2, 1060, s=f'{sfct_f}F', color='red', bbox=dict(facecolor='white', edgecolor='red', pad=1.5), fontsize=10)
    plt.text((sfctd_f - 32) * 5/9 - 2, 1060, s=f'{sfctd_f}F', color='green', bbox=dict(facecolor='white', edgecolor='green', pad=1.5), fontsize=10)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Pressure (hPa)")
    plt.title(f'GDAS {year}-%02d-%02d-%02dz ({lat}N, {lon}E)' % (month, day, hour))

    sounding_filename = f'{year}-%02d-%02d-%02dz_{lat}_{lon}' % (month, day, hour)
    plt.savefig(f'{sounding_outdir}/sounding_{sounding_filename}.png', bbox_inches='tight')
    plt.close()


def gdas_map(year, month, day, hour, variable, pressure_level, extent, gdas_pickle_indir, fronts_pickle_indir, image_outdir, wind_barbs=False):
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
    wind_barbs: bool
        Plot wind barbs.
    """

    min_lon, max_lon, min_lat, max_lat = extent[0], extent[1], extent[2], extent[3]
    fronts_file = '%s\\%d\\%02d\\%02d\\FrontObjects_%d%02d%02d%02d_full.pkl' % (fronts_pickle_indir, year, month, day, year, month, day, hour)
    gdas_file = '%s\\%d\\%02d\\%02d\\gdas_%d_%d%02d%02d%02d_full.pkl' % (gdas_pickle_indir, year, month, day, pressure_level, year, month, day, hour)

    fronts_dataset = pd.read_pickle(fronts_file).isel(latitude=slice(int((80-max_lat)*4), int((80-min_lat)*4)), longitude=slice(int((min_lon-130)*4), int((max_lon-130)*4)))
    gdas_dataset = pd.read_pickle(gdas_file)[variable].isel(latitude=slice(int((80-max_lat)*4), int((80-min_lat)*4)), longitude=slice(int((min_lon-130)*4), int((max_lon-130)*4)))

    fig = plt.figure(figsize=(12, 6))
    ax = plot_background(extent)
    fronts_dataset, names, _, colors_types, _ = reformat_fronts(fronts_dataset, front_types='ALL', return_colors=True, return_names=True)
    fronts_dataset = expand_fronts(fronts_dataset)
    fronts_dataset = xr.where(fronts_dataset == 0, float("NaN"), fronts_dataset)

    cbar_ax = fig.add_axes([0.7565, 0.11, 0.015, 0.77])
    cmap_fronts = ListedColormap(colors_types, name='from_list', N=len(names))
    norm_fronts = Normalize(vmin=1, vmax=len(names) + 1)
    cbar = plt.colorbar(ScalarMappable(norm=norm_fronts, cmap=cmap_fronts), cax=cbar_ax)
    cbar.set_ticks(np.arange(1, len(names) + 1) + 0.5)
    cbar.set_ticklabels(names)

    gdas_dataset.plot.contourf(ax=ax, x='longitude', y='latitude', alpha=0.5, transform=ccrs.PlateCarree(),
                               levels=20, cbar_kwargs={'orientation': 'horizontal', 'shrink': 0.5})
    fronts_dataset['identifier'].plot(ax=ax, x='longitude', y='latitude', cmap=cmap_fronts, norm=norm_fronts, transform=ccrs.PlateCarree(), add_colorbar=False)
    plt.savefig('%s\\%d\\%02d\\%02d\\%02d\\gdas_%s_%d_%d%02d%02d%02d.png' % (image_outdir, year, month, day, hour, variable, pressure_level, year, month, day, hour), dpi=300, bbox_inches='tight')
    plt.close()


def era5_map(year, month, day, hour, variable, extent, era5_pickle_indir, fronts_pickle_indir, image_outdir, wind_barbs=False):
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
    wind_barbs: bool
        Plot wind barbs.
    """

    min_lon, max_lon, min_lat, max_lat = extent[0], extent[1], extent[2], extent[3]
    fronts_file = '%s\\%d\\%02d\\%02d\\FrontObjects_%d%02d%02d%02d_full.pkl' % (fronts_pickle_indir, year, month, day, year, month, day, hour)
    era5_file = '%s\\%d\\%02d\\%02d\\Data_60var_%d%02d%02d%02d_full.pkl' % (era5_pickle_indir, year, month, day, year, month, day, hour)

    fronts_dataset = pd.read_pickle(fronts_file).isel(latitude=slice(int((80-max_lat)*4), int((80-min_lat)*4)), longitude=slice(int((min_lon-130)*4), int((max_lon-130)*4)))
    era5_dataset = pd.read_pickle(era5_file).isel(latitude=slice(int((80-max_lat)*4), int((80-min_lat)*4)), longitude=slice(int((min_lon-130)*4), int((max_lon-130)*4)))
    variable_dataset = era5_dataset[variable]

    fig = plt.figure(figsize=(12, 6))
    ax = plot_background(extent)
    fronts_dataset, names, _, colors_types, _ = reformat_fronts(fronts_dataset, front_types='ALL', return_colors=True, return_names=True)
    fronts_dataset = expand_fronts(fronts_dataset)
    fronts_dataset = xr.where(fronts_dataset == 0, float("NaN"), fronts_dataset)

    cbar_ax = fig.add_axes([0.7965, 0.11, 0.015, 0.77])
    cmap_fronts = ListedColormap(colors_types, name='from_list', N=len(names))
    norm_fronts = Normalize(vmin=1, vmax=len(names) + 1)
    cbar = plt.colorbar(ScalarMappable(norm=norm_fronts, cmap=cmap_fronts), cax=cbar_ax)
    cbar.set_ticks((np.arange(1, len(names) + 1) + 0.5))
    cbar.set_ticklabels(names)
    cbar.ax.invert_yaxis()

    variable_dataset.plot.contourf(ax=ax, x='longitude', y='latitude', alpha=0.5, transform=ccrs.PlateCarree(),
                               levels=10, cmap='hot', cbar_kwargs={'orientation': 'horizontal', 'shrink': 0.5})
    fronts_dataset['identifier'].plot(ax=ax, x='longitude', y='latitude', cmap=cmap_fronts, norm=norm_fronts, transform=ccrs.PlateCarree(), add_colorbar=False)
    ax.set_title("Equivalent potential temperature and wind barbs (900 hPa) and forecasters' fronts: 2019-03-14-18z", fontsize=8)
    plt.savefig('%s\\%d\\%02d\\%02d\\%02d\\era5_%s_%d%02d%02d%02d.png' % (image_outdir, year, month, day, hour, variable, year, month, day, hour), dpi=500, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create labeled data for the specified day")
    parser.add_argument('--frequency_plots', action='store_true', help='Generate frequency plots')
    parser.add_argument('--gdas_map', action='store_true', help='Plot GDAS data over a domain')
    parser.add_argument('--pressure_level', type=int, help='Pressure level (hPa) of the variable for the GDAS map.')
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

    if args.gdas_sounding:
        required_arguments = ['gdas_pickle_indir', 'sounding_coords', 'image_outdir', 'year', 'month', 'day', 'hour']
        check_arguments(provided_arguments, required_arguments)
        lat, lon = args.sounding_coords[0], args.sounding_coords[1]
        gdas_sounding(args.year, args.month, args.day, args.hour, lat, lon, args.gdas_pickle_indir, args.image_outdir)

    if args.era5_map:
        required_arguments = ['era5_pickle_indir', 'fronts_pickle_indir', 'variable', 'extent', 'image_outdir', 'year', 'month', 'day', 'hour']
        check_arguments(provided_arguments, required_arguments)
        era5_map(args.year, args.month, args.day, args.hour, args.variable, args.extent, args.era5_pickle_indir, args.fronts_pickle_indir, args.image_outdir)
