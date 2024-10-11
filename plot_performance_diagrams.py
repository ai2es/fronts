"""
Plot performance diagrams for a model.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.10.10
"""
import argparse
import cartopy.crs as ccrs
from matplotlib import colors
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd
import pickle
import xarray as xr
from utils import data_utils, plotting


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
    parser.add_argument('--map_neighborhood', type=int, default=250,
        help="Neighborhood for the CSI map in kilometers. Options are: 50, 100, 150, 200, 250")
    parser.add_argument('--model_dir', type=str, required=True, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, required=True, help='Model number.')
    parser.add_argument('--output_type', type=str, default='png', help="Output type for the image file.")

    args = vars(parser.parse_args())

    model_properties_filepath = f"{args['model_dir']}/model_{args['model_number']}/model_{args['model_number']}_properties.pkl"
    model_properties = pd.read_pickle(model_properties_filepath)

    # Some older models do not have the 'dataset_properties' dictionary
    try:
        front_types = model_properties['dataset_properties']['front_types']
    except KeyError:
        front_types = model_properties['front_types']

    stats_ds = xr.open_dataset('%s/model_%d/statistics/model_%d_statistics_%s_%s.nc' % (args['model_dir'], args['model_number'], args['model_number'], args['domain'], args['dataset']))

    if type(front_types) == str:
        front_types = [front_types, ]

    # Probability threshold where CSI is maximized for each front type and domain
    max_csi_thresholds = dict()

    if args['domain'] not in list(max_csi_thresholds.keys()):
        max_csi_thresholds[args['domain']] = dict()

    for front_no, front_label in enumerate(front_types):

        if front_label not in list(max_csi_thresholds[args['domain']].keys()):
            max_csi_thresholds[args['domain']][front_label] = dict()

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

        num_forecasts = true_positives_temporal_sum + false_positives_temporal_sum
        num_forecasts = num_forecasts[0, :]
        total_pixels = true_positives_temporal.shape[0] * len(spatial_csi_ds["latitude"]) * len(spatial_csi_ds["longitude"])
        relative_forecast_fraction = 100 * num_forecasts / total_pixels

        ### Find the number of true positives and false positives in each probability bin ###
        true_positives_diff = np.abs(np.diff(true_positives_temporal_sum))
        false_positives_diff = np.abs(np.diff(false_positives_temporal_sum))
        observed_relative_frequency = np.divide(true_positives_diff, true_positives_diff + false_positives_diff)

        pod = np.divide(true_positives_temporal_sum, true_positives_temporal_sum + false_negatives_temporal_sum)  # Probability of detection
        sr = np.divide(true_positives_temporal_sum, true_positives_temporal_sum + false_positives_temporal_sum)  # Success ratio

        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        axarr = axs.flatten()

        sr_matrix, pod_matrix = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
        csi_matrix = 1 / ((1/sr_matrix) + (1/pod_matrix) - 1)  # CSI coordinates
        fb_matrix = pod_matrix * (sr_matrix ** -1)  # Frequency Bias coordinates
        CSI_LEVELS = np.linspace(0, 1, 11)  # CSI contour levels
        FB_LEVELS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3]  # Frequency Bias levels
        cmap = 'Blues'  # Colormap for the CSI contours
        axis_ticks = np.arange(0, 1.01, 0.1)
        axis_ticklabels = np.arange(0, 100.1, 10).astype(int)

        cs = axarr[0].contour(sr_matrix, pod_matrix, fb_matrix, FB_LEVELS, colors='black', linewidths=0.5, linestyles='--')  # Plot FB levels
        axarr[0].clabel(cs, FB_LEVELS, fontsize=8)

        csi_contour = axarr[0].contourf(sr_matrix, pod_matrix, csi_matrix, CSI_LEVELS, cmap=cmap)  # Plot CSI contours in 0.1 increments
        cbar = fig.colorbar(csi_contour, ax=axarr[0], pad=0.02, label='Critical Success Index (CSI)')
        cbar.set_ticks(axis_ticks)

        axarr1_2 = axarr[1].twinx()
        axarr1_2.set_ylabel("Percentage of Grid Points with Forecasts [bars]")
        axarr1_2.yaxis.set_major_locator(plt.LinearLocator(11))
        axarr1_2.bar(thresholds[:-1], relative_forecast_fraction[1:], color='blue', width=0.005, alpha=0.25)
        axarr[1].plot(thresholds, thresholds, color='black', linestyle='--', linewidth=0.5, label='Perfect Reliability')

        cell_text = []  # List of strings that will be used in the table near the bottom of this function

        ### CSI and reliability lines for each boundary ###
        boundary_colors = ['red', 'purple', 'brown', 'darkorange', 'darkgreen']
        max_CSI_scores_by_boundary = np.zeros(shape=(5,))
        for boundary, color in enumerate(boundary_colors):
            csi = np.power((1/sr[boundary]) + (1/pod[boundary]) - 1, -1)
            max_CSI_scores_by_boundary[boundary] = np.nanmax(csi)
            max_CSI_index = np.where(csi == max_CSI_scores_by_boundary[boundary])[0]
            max_CSI_threshold = thresholds[max_CSI_index][0]  # Probability threshold where CSI is maximized
            max_csi_thresholds[args['domain']][front_label]['%s' % int((boundary + 1) * 50)] = np.round(max_CSI_threshold, 2)
            max_CSI_pod = pod[boundary][max_CSI_index][0]  # POD where CSI is maximized
            max_CSI_sr = sr[boundary][max_CSI_index][0]  # SR where CSI is maximized
            max_CSI_fb = max_CSI_pod / max_CSI_sr  # Frequency bias

            cell_text.append([r'$\bf{%.3f}$' % max_CSI_scores_by_boundary[boundary] + r'$^{%.3f}_{%.3f}$' % (CI_CSI[1, boundary, max_CSI_index][0], CI_CSI[0, boundary, max_CSI_index][0]),
                              r'$\bf{%.1f}$' % (max_CSI_pod * 100) + r'$^{%.1f}_{%.1f}$' % (CI_POD[1, boundary, max_CSI_index][0] * 100, CI_POD[0, boundary, max_CSI_index][0] * 100),
                              r'$\bf{%.1f}$' % ((1 - max_CSI_sr) * 100) + r'$^{%.1f}_{%.1f}$' % ((1 - CI_SR[1, boundary, max_CSI_index][0]) * 100, (1 - CI_SR[0, boundary, max_CSI_index][0]) * 100),
                              r'$\bf{%.3f}$' % max_CSI_fb + r'$^{%.3f}_{%.3f}$' % (CI_FB[1, boundary, max_CSI_index][0], CI_FB[0, boundary, max_CSI_index][0])])

            # Plot CSI lines
            axarr[0].plot(max_CSI_sr, max_CSI_pod, color=color, marker='*', markersize=10)
            axarr[0].plot(sr[boundary], pod[boundary], color=color, linewidth=1)

            # Plot reliability curve
            axarr[1].plot(thresholds[:-1], observed_relative_frequency[boundary], color=color, linewidth=1)

            # Confidence interval
            xs = np.concatenate([CI_SR[0, boundary, :polygon_stop_index], CI_SR[1, boundary, :polygon_stop_index][::-1]])
            ys = np.concatenate([CI_POD[0, boundary, :polygon_stop_index], CI_POD[1, boundary, :polygon_stop_index][::-1]])
            axarr[0].fill(xs, ys, alpha=0.3, color=color)  # Shade the confidence interval

        axarr[0].set_xticklabels(axis_ticklabels[::-1])  # False alarm rate on x-axis means values are reversed
        axarr[0].set_xlabel("False Alarm Rate (FAR; %)")
        axarr[0].set_ylabel("Probability of Detection (POD; %)")
        axarr[0].set_title(r'$\bf{a)}$ $\bf{CSI}$ $\bf{diagram}$ [confidence level = %d%%]' % args['confidence_level'])

        axarr[1].set_xticklabels(axis_ticklabels)
        axarr[1].set_xlabel("Forecast Probability (uncalibrated; %)")
        axarr[1].set_ylabel("Observed Relative Frequency (%) [lines]")
        axarr[1].set_title(r'$\bf{b)}$ $\bf{Reliability}$ $\bf{diagram}$')

        for ax in axarr:
            ax.set_xticks(axis_ticks)
            ax.set_yticks(axis_ticks)
            ax.set_yticklabels(axis_ticklabels)
            ax.grid(color='black', alpha=0.1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ################################################################################################################

        cbar_kwargs = {'label': 'CSI', 'pad': 0}  # Spatial CSI colorbar keyword arguments

        ### Adjust the data table and spatial CSI plot based on the domain ###
        if args['domain'] == 'conus':
            table_axis_extent = [0.063, -0.038, 0.39, 0.239]
            table_scale = (1, 3.3)
            table_title_kwargs = dict(x=0.5, y=0.098, pad=-4)
            spatial_axis_extent = [0.5, -0.582, 0.512, 0.544]
            cbar_kwargs['shrink'] = 1
            spatial_plot_xlabels = [-140, -105, -70]
            spatial_plot_ylabels = [30, 40, 50]
        elif args["domain"] == "full":
            table_axis_extent = [0.063, -0.038, 0.39, 0.229]
            table_scale = (1, 2.8)
            table_title_kwargs = dict(x=0.5, y=0.096, pad=-4)
            spatial_axis_extent = [0.523, -0.5915, 0.48, 0.66]
            cbar_kwargs['shrink'] = 0.675
            spatial_plot_xlabels = [-150, -120, -90, -60, -30, 0, 120, 150, 180]
            spatial_plot_ylabels = [0, 20, 40, 60, 80]
        else:
            raise ValueError("%s domain is currently not supported for performance diagrams." % args["domain"])

        ############################################# Data table (panel c) #############################################
        columns = ['CSI', 'POD %', 'FAR %', 'FB']  # Column names
        rows = ['50 km', '100 km', '150 km', '200 km', '250 km']  # Row names

        table_axis = plt.axes(table_axis_extent)
        table_axis.set_title(r'$\bf{c)}$ $\bf{Data}$ $\bf{table}$ [confidence level = %d%%]' % args['confidence_level'], **table_title_kwargs)
        table_axis.axis('off')
        stats_table = table_axis.table(cellText=cell_text, rowLabels=rows, rowColours=boundary_colors, colLabels=columns, cellLoc='center')
        stats_table.scale(*table_scale)  # Make the table larger

        ### Shade the cells and make the cell text larger ###
        for cell in stats_table._cells:
            stats_table._cells[cell].set_alpha(.7)
            stats_table._cells[cell].set_text_props(fontproperties=FontProperties(size='x-large', stretch='expanded'))
        ################################################################################################################

        ########################################## Spatial CSI map (panel d) ###########################################
        right_labels = False  # Disable latitude labels on the right side of the subplot
        top_labels = False  # Disable longitude labels on top of the subplot
        left_labels = True  # Latitude labels on the left side of the subplot
        bottom_labels = True  # Longitude labels on the bottom of the subplot

        ## Set up the spatial CSI plot ###
        csi_cmap = plotting.truncated_colormap('gnuplot2', maxval=0.9, n=10)
        extent = data_utils.DOMAIN_EXTENTS[args['domain']]
        spatial_axis = plt.axes(spatial_axis_extent, projection=ccrs.Miller(central_longitude=250))
        spatial_axis_title_text = r'$\bf{d)}$ $\bf{%d}$ $\bf{km}$ $\bf{CSI}$ $\bf{map}$' % args['map_neighborhood']
        plotting.plot_background(extent=extent, ax=spatial_axis)
        norm_probs = colors.Normalize(vmin=0.1, vmax=1)
        spatial_csi_ds = xr.where(spatial_csi_ds >= 0.1, spatial_csi_ds, float("NaN"))
        spatial_csi_ds.sel(boundary=args['map_neighborhood']).plot(ax=spatial_axis, x='longitude', y='latitude', norm=norm_probs,
            cmap=csi_cmap, transform=ccrs.PlateCarree(), alpha=0.6, cbar_kwargs=cbar_kwargs)
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

        if args['domain'] == 'conus':
            domain_text = args['domain'].upper()
        else:
            domain_text = args['domain']

        plt.suptitle(f'Five-class model: %ss over %s domain' % (data_utils.FRONT_NAMES[front_label], domain_text), fontsize=20)  # Create and plot the main title

        filename = f"%s/model_%d/performance_%s_%s_%s_{args['data_source']}.{args['output_type']}" % (args['model_dir'], args['model_number'], front_label, args['dataset'], args['domain'])
        if args['data_source'] != 'era5':
            filename = filename.replace(f'.{args["output_type"]}', f'_f%03d.{args["output_type"]}' % args['forecast_hour'])  # Add forecast hour to the end of the filename

        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=500)
        plt.close()

    # Thresholds for creating deterministic splines with different front types and neighborhoods
    model_properties['front_obj_thresholds'] = max_csi_thresholds

    with open(model_properties_filepath, 'wb') as f:
        pickle.dump(model_properties, f)
