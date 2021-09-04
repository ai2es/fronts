"""
Function used to plot all variables for a given time using pickle files. Each variable plot is saved in its own file.

Code written by: Andrew Justin (andrewjustin@ou.edu)
"""

import matplotlib.pyplot as plt
import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import matplotlib as mpl
import file_manager as fm


def plot_surface_background(extent):
    """
    Returns new background for the plot.

    Parameters
    ----------
    extent: ndarray
        Numpy array containing the extent/boundaries of the plot in the format of [min lon, max lon, min lat, max lat].

    Returns
    -------
    ax: GeoAxesSubplot
        New plot background.
    """
    crs = ccrs.LambertConformal(central_longitude=250)
    ax = plt.axes(projection=crs)
    # gridlines = ax.gridlines()
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create labeled data for the specified day")
    parser.add_argument('--year', type=int, required=True, help="year for the data to be read in")
    parser.add_argument('--month', type=int, required=True, help="month for the data to be read in")
    parser.add_argument('--day', type=int, required=True, help="day for the data to be read in")
    parser.add_argument('--hour', type=int, required=True, help="hour for the data to be read in")
    parser.add_argument('--front_types', type=str, required=True,
                        help='Front format of the file. If your files contain warm and cold fronts, pass this argument '
                             'as CFWF. If your files contain only drylines, pass this argument as DL. If your files '
                             'contain all fronts, pass this argument as ALL.')
    parser.add_argument('--domain', type=str, required=True, help='Domain of the data. Possible values are: conus')
    parser.add_argument('--extent', type=float, required=True, nargs=4,
                        help='Extent of the plot. Pass 4 integers in the following order: MIN LON, MAX LON, MIN LAT, '
                             'MAX_LAT')
    parser.add_argument('--file_dimensions', type=int, nargs=2, required=True, help='Dimensions of the file size. Two integers'
        ' need to be passed.')
    parser.add_argument('--image_outdir', type=str, required=False, help="output directory for image files")
    args = parser.parse_args()

    extent = [args.extent[0], args.extent[1], args.extent[2], args.extent[3]]

    front_file_list, variable_file_list = fm.load_file_lists(num_variables=60, front_types=args.front_types, domain=args.domain, file_dimensions=args.file_dimensions)

    front_filename_no_dir = 'FrontObjects_%s_%d%02d%02d%02d_%s_%dx%d.pkl' % (args.front_types, args.year, args.month,
        args.day, args.hour, args.domain, args.file_dimensions[0], args.file_dimensions[1])
    variable_filename_no_dir = 'Data_%dvar_%d%02d%02d%02d_%s_%dx%d.pkl' % (60, args.year, args.month, args.day, args.hour,
        args.domain, args.file_dimensions[0], args.file_dimensions[1])

    front_file = [front_filename for front_filename in front_file_list if front_filename_no_dir in front_filename][0]
    variable_file = [variable_filename for variable_filename in variable_file_list if variable_filename_no_dir in variable_filename][0]

    front_ds = pd.read_pickle(front_file).sel(longitude=slice(extent[0]-10, extent[1]+10), latitude=slice(extent[3]+10, extent[2]-10))
    variable_ds = pd.read_pickle(variable_file).sel(longitude=slice(extent[0]-10, extent[1]+10), latitude=slice(extent[3]+10, extent[2]-10))

    variable_list = list(variable_ds.keys())

    for var in variable_list:
        ax = plot_surface_background(extent)
        if var == 't2m' or var == 't_1000' or var == 't_950' or var == 't_900' or var == 't_850':
            cmap = 'jet'
            norm = mpl.colors.Normalize(vmin=np.nanmin(variable_ds[['t2m','t_1000','t_950','t_900','t_850']].to_array()),
                                        vmax=np.nanmax(variable_ds[['t2m','t_1000','t_950','t_900','t_850']].to_array()))
        elif var == 'd2m' or var == 'd_1000' or var == 'd_950' or var == 'd_900' or var == 'd_850':
            cmap = 'terrain_r'
            norm = mpl.colors.Normalize(vmin=np.nanmin(variable_ds[['d2m','d_1000','d_950','d_900','d_850']].to_array()),
                                        vmax=np.nanmax(variable_ds[['d2m','d_1000','d_950','d_900','d_850']].to_array()))
        elif var == 'rel_humid' or var == 'rel_humid_1000' or var == 'rel_humid_950' or var == 'rel_humid_900' or var == 'rel_humid_850':
            cmap = 'terrain_r'
            norm = mpl.colors.Normalize(vmin=0,vmax=1)
        elif var == 'mix_ratio' or var == 'mix_ratio_1000' or var == 'mix_ratio_950' or var == 'mix_ratio_900' or var == 'mix_ratio_850':
            cmap = 'terrain_r'
            norm = mpl.colors.Normalize(vmin=np.nanmin(variable_ds[['mix_ratio','mix_ratio_1000','mix_ratio_950','mix_ratio_900','mix_ratio_850']].to_array()),
                                        vmax=np.nanmax(variable_ds[['mix_ratio','mix_ratio_1000','mix_ratio_950','mix_ratio_900','mix_ratio_850']].to_array()))
        elif var == 'q' or var == 'q_1000' or var == 'q_950' or var == 'q_900' or var == 'q_850':
            cmap = 'terrain_r'
            norm = mpl.colors.Normalize(vmin=np.nanmin(variable_ds[['q','q_1000','q_950','q_900','q_850']].to_array()),
                                        vmax=np.nanmax(variable_ds[['q','q_1000','q_950','q_900','q_850']].to_array()))
        elif var == 'wet_bulb' or var == 'wet_bulb_1000' or var == 'wet_bulb_950' or var == 'wet_bulb_900' or var == 'wet_bulb_850':
            cmap = 'terrain_r'
            norm = mpl.colors.Normalize(vmin=np.nanmin(variable_ds[['wet_bulb','wet_bulb_1000','wet_bulb_950','wet_bulb_900','wet_bulb_850']].to_array()),
                                        vmax=np.nanmax(variable_ds[['wet_bulb','wet_bulb_1000','wet_bulb_950','wet_bulb_900','wet_bulb_850']].to_array()))
        elif var == 'theta_w' or var == 'theta_w_1000' or var == 'theta_w_950' or var == 'theta_w_900' or var == 'theta_w_850':
            cmap = 'terrain_r'
            norm = mpl.colors.Normalize(vmin=np.nanmin(variable_ds[['theta_w','theta_w_1000','theta_w_950','theta_w_900','theta_w_850']].to_array()),
                                        vmax=np.nanmax(variable_ds[['theta_w','theta_w_1000','theta_w_950','theta_w_900','theta_w_850']].to_array()))
        elif var == 'theta_e' or var == 'theta_e_1000' or var == 'theta_e_950' or var == 'theta_e_900' or var == 'theta_e_850':
            cmap = 'terrain_r'
            norm = mpl.colors.Normalize(vmin=np.nanmin(variable_ds[['theta_e','theta_e_1000','theta_e_950','theta_e_900','theta_e_850']].to_array()),
                                        vmax=np.nanmax(variable_ds[['theta_e','theta_e_1000','theta_e_950','theta_e_900','theta_e_850']].to_array()))
        elif var == 'virt_temp' or var == 'virt_temp_1000' or var == 'virt_temp_950' or var == 'virt_temp_900' or var == 'virt_temp_850':
            cmap = 'terrain_r'
            norm = mpl.colors.Normalize(vmin=np.nanmin(variable_ds[['virt_temp','virt_temp_1000','virt_temp_950','virt_temp_900','virt_temp_850']].to_array()),
                                        vmax=np.nanmax(variable_ds[['virt_temp','virt_temp_1000','virt_temp_950','virt_temp_900','virt_temp_850']].to_array()))
        elif var == 'q' or var == 'q_1000' or var == 'q_950' or var == 'q_900' or var == 'q_850':
            cmap = 'terrain_r'
            norm = mpl.colors.Normalize(vmin=np.nanmin(variable_ds[['q','q_1000','q_950','q_900','q_850']].to_array()),
                                        vmax=np.nanmax(variable_ds[['q','q_1000','q_950','q_900','q_850']].to_array()))
        elif var == 'z_1000':
            cmap = 'viridis'
            norm = mpl.colors.Normalize(vmin=1000, vmax=4000)
        elif var == 'z_950':
            cmap = 'viridis'
            norm = mpl.colors.Normalize(vmin=4000, vmax=8000)
        elif var == 'z_900':
            cmap = 'viridis'
            norm = mpl.colors.Normalize(vmin=8000, vmax=12000)
        elif var == 'z_850':
            cmap = 'viridis'
            norm = mpl.colors.Normalize(vmin=12000, vmax=16000)
        elif var == 'sp':
            cmap = 'viridis'
            norm = mpl.colors.Normalize(vmin=np.nanmin(variable_ds['sp']),
                                        vmax=np.nanmax(variable_ds['sp']))
        elif var == 'v10' or var == 'v_1000' or var == 'v_950' or var == 'v_900' or var == 'v_850':
            cmap = 'seismic'
            norm = mpl.colors.Normalize(vmin=-25, vmax=25)
        elif var == 'u10' or var == 'u_1000' or var == 'u_950' or var == 'u_900' or var == 'u_850':
            cmap = 'seismic'
            norm = mpl.colors.Normalize(vmin=-25, vmax=25)

        variable_ds[var].plot(ax=ax, x='longitude', y='latitude', cmap=cmap, norm=norm, alpha=0.5, transform=ccrs.PlateCarree())

        cmap_front = mpl.colors.ListedColormap(["none","blue","red",'green','purple','orange'], name='from_list', N=None)
        norm_front = mpl.colors.Normalize(vmin=0,vmax=6)
        front_ds['identifier'].plot(ax=ax, x='longitude', y='latitude', cmap=cmap_front, norm=norm_front, transform=ccrs.PlateCarree())
        plt.savefig('%s/%d/%02d/%02d/%02d/%d-%02d-%02d-%02dz_%s_plot.png' % (args.image_outdir, args.year, args.month, args.day, args.hour, args.year, args.month, args.day, args.hour, var), bbox_inches='tight', dpi=400)
        plt.close()
