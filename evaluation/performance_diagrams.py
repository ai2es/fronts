"""
Plot performance diagrams for a model.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 6/9/2023 3:32 PM CT
"""
import argparse
import cartopy.crs as ccrs
from matplotlib import colors
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd
import xarray as xr
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # this line allows us to import scripts outside of the current directory
from utils import settings
from utils.plotting_utils import plot_background


if __name__ == '__main__':
    """
    All arguments listed in the examples are listed via argparse in alphabetical order below this comment block.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--confidence_level', type=int, default=95, help="Confidence interval. Options are: 90, 95, 99.")
    parser.add_argument('--dataset', type=str, help="'training', 'validation', or 'test'")
    parser.add_argument('--data_source', type=str, default='era5', help="Source of the variable data (ERA5, GDAS, etc.)")
    parser.add_argument('--domain_images', type=int, nargs=2, help='Number of images for each dimension the final stitched map for predictions: lon, lat')
    parser.add_argument('--domain', type=str, required=True, help='Domain of the data.')
    parser.add_argument('--forecast_hour', type=int, help='Forecast hour for the GDAS or GFS data.')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, required=True, help='Model number.')

    args = vars(parser.parse_args())

    model_properties = pd.read_pickle(f"{args['model_dir']}\\model_{args['model_number']}\\model_{args['model_number']}_properties.pkl")
    front_types = model_properties['dataset_properties']['front_types']

    domain_extent_indices = settings.DEFAULT_DOMAIN_INDICES[args['domain']]

    stats_ds = xr.open_dataset('%s/model_%d/statistics/model_%d_statistics_%s_%s.nc' % (args['model_dir'], args['model_number'], args['model_number'], args['domain'], args['dataset']))

    if type(front_types) == str:
        front_types = [front_types, ]

    for front_no, front_label in enumerate(front_types):

        ################################ CSI and reliability diagrams (panels a and b) #################################
        true_positives_temporal = stats_ds[f'tp_temporal_{front_label}'].values
        false_positives_temporal = stats_ds[f'fp_temporal_{front_label}'].values
        false_negatives_temporal = stats_ds[f'fn_temporal_{front_label}'].values
        spatial_csi_ds = (stats_ds[f'tp_spatial_{front_label}'] / (stats_ds[f'tp_spatial_{front_label}'] + stats_ds[f'fp_spatial_{front_label}'] + stats_ds[f'fn_spatial_{front_label}'])).max('threshold')
        thresholds = stats_ds['threshold'].values

        if args['confidence_level'] != 90:
            CI_low, CI_high = (100 - args['confidence_level']) / 2, 50 + (args['confidence_level'] / 2)
            CI_low, CI_high = '%.1f' % CI_low, '%.1f' % CI_high
        else:
            CI_low, CI_high = 5, 95

        # Confidence intervals for POD and SR
        CI_POD = np.stack((stats_ds[f"POD_{CI_low}_{front_label}"].values, stats_ds[f"POD_{CI_high}_{front_label}"].values), axis=0)
        CI_SR = np.stack((stats_ds[f"SR_{CI_low}_{front_label}"].values, stats_ds[f"SR_{CI_high}_{front_label}"].values), axis=0)
        CI_CSI = np.stack((CI_SR ** -1 + CI_POD ** -1 - 1.) ** -1, axis=0)
        CI_FB = np.stack(CI_POD * (CI_SR ** -1), axis=0)

        # Remove the zeros
        try:
            polygon_stop_index = np.min(np.where(CI_POD == 0)[2])
        except IndexError:
            polygon_stop_index = 100

        ### Statistics with shape (boundary, threshold) after taking the sum along the time axis (axis=0) ###
        true_positives_temporal_sum = np.sum(true_positives_temporal, axis=0)
        false_positives_temporal_sum = np.sum(false_positives_temporal, axis=0)
        false_negatives_temporal_sum = np.sum(false_negatives_temporal, axis=0)

        ### Find the number of true positives and false positives in each probability bin ###
        true_positives_diff = np.abs(np.diff(true_positives_temporal_sum))
        false_positives_diff = np.abs(np.diff(false_positives_temporal_sum))
        observed_relative_frequency = np.divide(true_positives_diff, true_positives_diff + false_positives_diff)

        pod = np.divide(true_positives_temporal_sum, true_positives_temporal_sum + false_negatives_temporal_sum)  # Probability of detection
        sr = np.divide(true_positives_temporal_sum, true_positives_temporal_sum + false_positives_temporal_sum)  # Success ratio

        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        axarr = axs.flatten()

        sr_matrix, pod_matrix = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        csi_matrix = (sr_matrix ** -1 + pod_matrix ** -1 - 1.) ** -1  # CSI coordinates
        fb_matrix = pod_matrix * (sr_matrix ** -1)  # Frequency Bias coordinates
        CSI_LEVELS = np.linspace(0, 1, 11)  # CSI contour levels
        FB_LEVELS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3]  # Frequency Bias levels
        cmap = 'Blues'  # Colormap for the CSI contours
        axis_ticks = np.arange(0, 1.1, 0.1)

        cs = axarr[0].contour(sr_matrix, pod_matrix, fb_matrix, FB_LEVELS, colors='black', linewidths=0.5, linestyles='--')  # Plot FB levels
        axarr[0].clabel(cs, FB_LEVELS, fontsize=8)

        csi_contour = axarr[0].contourf(sr_matrix, pod_matrix, csi_matrix, CSI_LEVELS, cmap=cmap)  # Plot CSI contours in 0.1 increments
        cbar = fig.colorbar(csi_contour, ax=axarr[0], pad=0.02, label='Critical Success Index (CSI)')
        cbar.set_ticks(axis_ticks)

        axarr[1].plot(thresholds, thresholds, color='black', linestyle='--', linewidth=0.5, label='Perfect Reliability')

        cell_text = []  # List of strings that will be used in the table near the bottom of this function

        ### CSI and reliability lines for each boundary ###
        boundary_colors = ['red', 'purple', 'brown', 'darkorange', 'darkgreen']
        max_CSI_scores_by_boundary = np.empty(shape=(5,))
        for boundary, color in enumerate(boundary_colors):
            csi = np.power((1/sr[boundary]) + (1/pod[boundary]) - 1, -1)
            max_CSI_scores_by_boundary[boundary] = np.nanmax(csi)
            max_CSI_index = np.where(csi == max_CSI_scores_by_boundary[boundary])[0]
            max_CSI_threshold = thresholds[max_CSI_index][0]  # Probability threshold where CSI is maximized
            max_CSI_pod = pod[boundary][max_CSI_index][0]  # POD where CSI is maximized
            max_CSI_sr = sr[boundary][max_CSI_index][0]  # SR where CSI is maximized
            max_CSI_fb = max_CSI_pod / max_CSI_sr  # Frequency bias

            cell_text.append([r'$\bf{%.2f}$' % max_CSI_threshold,
                              r'$\bf{%.2f}$' % max_CSI_scores_by_boundary[boundary] + r'$^{%.2f}_{%.2f}$' % (CI_CSI[1, boundary, max_CSI_index], CI_CSI[0, boundary, max_CSI_index]),
                              r'$\bf{%.2f}$' % max_CSI_pod + r'$^{%.2f}_{%.2f}$' % (CI_POD[1, boundary, max_CSI_index], CI_POD[0, boundary, max_CSI_index]),
                              r'$\bf{%.2f}$' % max_CSI_sr + r'$^{%.2f}_{%.2f}$' % (CI_SR[1, boundary, max_CSI_index], CI_SR[0, boundary, max_CSI_index]),
                              r'$\bf{%.2f}$' % (1 - max_CSI_sr) + r'$^{%.2f}_{%.2f}$' % (1 - CI_SR[1, boundary, max_CSI_index], 1 - CI_SR[0, boundary, max_CSI_index]),
                              r'$\bf{%.2f}$' % max_CSI_fb + r'$^{%.2f}_{%.2f}$' % (CI_FB[1, boundary, max_CSI_index], CI_FB[0, boundary, max_CSI_index])])

            # Plot CSI lines
            axarr[0].plot(max_CSI_sr, max_CSI_pod, color=color, marker='*', markersize=10)
            axarr[0].plot(sr[boundary], pod[boundary], color=color, linewidth=1)

            # Plot reliability curve
            axarr[1].plot(thresholds[1:] + 0.005, observed_relative_frequency[boundary], color=color, linewidth=1)

            # Confidence interval
            xs = np.concatenate([CI_SR[0, boundary, :polygon_stop_index], CI_SR[1, boundary, :polygon_stop_index][::-1]])
            ys = np.concatenate([CI_POD[0, boundary, :polygon_stop_index], CI_POD[1, boundary, :polygon_stop_index][::-1]])
            axarr[0].fill(xs, ys, alpha=0.3, color=color)  # Shade the confidence interval

        axarr[0].set_xlabel("Success Ratio (SR = 1 - FAR)")
        axarr[0].set_ylabel("Probability of Detection (POD)")
        axarr[0].set_title('a) CSI diagram')

        axarr[1].set_xlabel("Forecast Probability (uncalibrated)")
        axarr[1].set_ylabel("Observed Relative Frequency")
        axarr[1].set_title('b) Reliability diagram')

        for ax in axarr:
            ax.set_xticks(axis_ticks)
            ax.set_yticks(axis_ticks)
            ax.grid(color='black', alpha=0.1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ################################################################################################################

        ############################################# Data table (panel c) #############################################
        columns = ['Threshold*', 'CSI', 'POD', 'SR', 'FAR', 'FB']  # Column names
        rows = ['50 km', '100 km', '150 km', '200 km', '250 km']  # Row names

        table_axis = plt.axes([0.063, -0.06, 0.4, 0.2])
        table_axis.set_title('c) Data table', x=0.415, y=0.135, pad=-4)
        table_axis.axis('off')
        table_axis.text(0.16, -2.7, '* probability threshold where CSI is maximized')  # Add disclaimer for probability threshold column
        stats_table = table_axis.table(cellText=cell_text, rowLabels=rows, rowColours=boundary_colors, colLabels=columns, cellLoc='center')
        stats_table.scale(1, 3)  # Make the table larger

        ### Shade the cells and make the cell text larger ###
        for cell in stats_table._cells:
            stats_table._cells[cell].set_alpha(.7)
            stats_table._cells[cell].set_text_props(fontproperties=FontProperties(size='xx-large', stretch='expanded'))
        ################################################################################################################

        ########################################## Spatial CSI map (panel d) ###########################################
        # Colorbar keyword arguments
        cbar_kwargs = {'label': 'CSI', 'pad': 0}

        # Adjust the spatial CSI plot based on the domain
        if args['domain'] == 'conus':
            spatial_axis_extent = [0.52, -0.59, 0.48, 0.535]
            spatial_plot_xlabels = [-140, -105, -70]
            spatial_plot_ylabels = [30, 40, 50]
            bottom_labels = False  # Disable longitude labels on the bottom of the subplot
        else:
            spatial_axis_extent = [0.538, -0.6, 0.48, 0.577]
            cbar_kwargs['shrink'] = 0.862
            spatial_plot_xlabels = [-150, -120, -90, -60, -30, 0, 120, 150, 180]
            spatial_plot_ylabels = [0, 20, 40, 60, 80]
            bottom_labels = True  # Longitude labels on the bottom of the subplot

        right_labels = False  # Disable latitude labels on the right side of the subplot
        top_labels = True  # Longitude labels on top of the subplot
        left_labels = True  # Latitude labels on the left side of the subplot

        ## Set up the spatial CSI plot ###
        extent = settings.DEFAULT_DOMAIN_EXTENTS[args['domain']]
        spatial_axis = plt.axes(spatial_axis_extent, projection=ccrs.Miller(central_longitude=250))
        spatial_axis_title_text = '250 km CSI map'
        plot_background(extent=extent, ax=spatial_axis)
        norm_probs = colors.Normalize(vmin=0.1, vmax=1)
        spatial_csi_ds = xr.where(spatial_csi_ds > 0.1, spatial_csi_ds, float("NaN"))
        spatial_csi_ds.isel(boundary=-1).plot(ax=spatial_axis, x='longitude', y='latitude', norm=norm_probs, cmap='gnuplot2', transform=ccrs.PlateCarree(), alpha=0.6, cbar_kwargs=cbar_kwargs)
        spatial_axis.set_title(spatial_axis_title_text)
        gl = spatial_axis.gridlines(draw_labels=True, zorder=0, dms=True, x_inline=False, y_inline=False)
        gl.right_labels = right_labels
        gl.top_labels = top_labels
        gl.left_labels = left_labels
        gl.bottom_labels = bottom_labels
        gl.xlocator = FixedLocator(spatial_plot_xlabels)
        gl.ylocator = FixedLocator(spatial_plot_ylabels)
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 8}
        ################################################################################################################

        ###################################### Generate title for the whole plot #######################################

        ### Front name text ###
        front_text = settings.DEFAULT_FRONT_NAMES[front_label]
        if type(front_types) == list and front_types != ['F_BIN']:
            front_text += 's'
        elif front_types == ['F_BIN']:
            front_text = 'Binary fronts (front / no front)'

        ### Domain text ###
        if args['domain'] == 'conus':
            domain_text = 'CONUS'
        else:
            domain_text = args['domain']
        # domain_text += f' domain ({int(domain_images[0] * domain_images[1])} images per map)'

        # plt.suptitle(f'{num_dimensions}D U-Net 3+ ({kernel_text} kernel): {front_text} over {domain_text}', fontsize=20)  # Create and plot the main title
        ################################################################################################################

        filename = f"%s/model_%d/%s_performance_%s_{args['data_source']}.png" % (args['model_dir'], args['model_number'], front_label, args['dataset'])
        if args['data_source'] != 'era5':
            filename = filename.replace('.png', '_f%03d.png' % args['forecast_hour'])  # Add forecast hour to the end of the filename

        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=500)
        plt.close()
