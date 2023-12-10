"""
Generate plots for permutation data from a model.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.12.9

TODO:
    * add option for including multi-pass permutation results
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # this line allows us to import scripts outside the current directory
import argparse
import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import settings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='Directory for the models.')
    parser.add_argument('--model_number', type=int, required=True, help='Model number.')
    parser.add_argument('--domain', type=str, required=True, help='Domain for the permutations')

    args = vars(parser.parse_args())

    model_folder = '%s/model_%d' % (args['model_dir'], args['model_number'])

    model_properties = pd.read_pickle('%s/model_%d_properties.pkl' % (model_folder, args['model_number']))
    variables, pressure_levels = model_properties['dataset_properties']['variables'], model_properties['dataset_properties']['pressure_levels']
    front_types = model_properties['dataset_properties']['front_types']
    permutations_dict = pd.read_pickle('%s/permutations_%d_%s.pkl' % (model_folder, args['model_number'], args['domain']))  # contains permutation data

    num_vars = len(variables)  # number of variables
    num_lvls = len(pressure_levels)  # number of pressure levels

    sp = permutations_dict['single_pass']
    sp_vars = np.array([sp[var] for var in variables])  # single-pass results: shuffled variables over all levels
    sp_lvls = np.array([sp[lvl] for lvl in pressure_levels])  # single-pass results: shuffled levels over all variables
    sp_sorting_orders = [np.argsort(sp_vars, axis=0), np.argsort(sp_lvls, axis=0)]  # list of length 2: sorted grouped variables and grouped levels

    for front_num, front_type in enumerate(front_types):
        sp_sorted_data = [sp_vars[:, front_num][sp_sorting_orders[0][:, front_num]], sp_lvls[:, front_num][sp_sorting_orders[1][:, front_num]]]
        sp_sorted_vars_and_lvls = [[variables[ind] for ind in sp_sorting_orders[0][:, front_num]], [pressure_levels[ind] for ind in sp_sorting_orders[1][:, front_num]]]

        y_pos_vars = np.arange(num_vars)  # array marking the y-positions of the variables
        y_pos_lvls = np.arange(num_lvls)  # array marking the y-positions of the pressure levels

        fig = plt.figure(figsize=(12, 9))
        ax1 = plt.subplot(2, 2, 1)  # axis for the subplot containing single-pass variable permutations
        ax2 = plt.subplot(2, 2, 2)  # axis for the subplot containing single-pass level permutations
        ax3 = plt.subplot(2, 1, 2)  # table axis

        # Horizontal bars for the single-pass data
        sp_barh_vars = ax1.barh(y_pos_vars, sp_sorted_data[0], color=settings.DEFAULT_FRONT_COLORS[front_type])
        sp_barh_lvls = ax2.barh(y_pos_lvls, sp_sorted_data[1], color=settings.DEFAULT_FRONT_COLORS[front_type])

        # setting subplot titles, ticks, and ticklabels
        ax1.set_title('a) Grouped variables')
        ax1.set_yticks(y_pos_vars)
        ax1.set_yticklabels(sp_sorted_vars_and_lvls[0])
        ax2.set_title('b) Grouped levels')
        ax2.set_yticks(y_pos_lvls)
        ax2.set_yticklabels(sp_sorted_vars_and_lvls[1])

        # Add labels to the horizontal bars on the subplots
        for pos, data in enumerate(sp_sorted_data[0]):
            ax1.annotate(data, xy=(np.max([data, 0]), pos), va='center')
        for pos, data in enumerate(sp_sorted_data[1]):
            ax2.annotate(data, xy=(np.max([data, 0]), pos), va='center')

        for ax in [ax1, ax2]:
            ax.margins(x=0.12)  # increase the scale of the x-axis so text will not extend past the right side of the subplots
            ax.set_xlabel("Relative importance")
            ax.grid(alpha=0.3, axis='x')

        # importance values for the cells in the single-pass table
        cellValues = np.array([[sp['_'.join([var, lvl])][front_num] for lvl in pressure_levels] for var in variables])
        sorted_indices = np.argsort(-cellValues.flatten())

        # variables at single levels ranked by importance
        ranks = np.zeros_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(sorted_indices)) + 1
        cellText = np.array(['%d (%.3f)' % (rank, val) for rank, val in zip(ranks, cellValues.flatten())]).reshape(cellValues.shape)

        # shade table cells based on the importance values
        max_val = np.max(cellValues)
        cellColorValues = 0.5 + (cellValues / (2 * max_val))
        cellColorValues /= np.max(cellColorValues)
        cmap = mpl.colormaps.get_cmap('bwr_r')
        cellColours = [cmap(val) for val in cellColorValues]

        rowColours = ['gray' for _ in range(len(variables))]  # shade first column containing variable names
        colColours = ['gray' for _ in range(len(pressure_levels))]  # shade header cells containing pressure level names
        ax3.set_title('c) Variables on single levels')
        ax3.axis('off')
        stats_table = ax3.table(cellText=cellText, rowLabels=variables, colLabels=pressure_levels, rowColours=rowColours,
                                colColours=colColours, cellColours=cellColours, cellLoc='center', bbox=[0, 0, 1, 1])

        # bold cells for header and first column, containing variable and level names
        bold_cells = [(var, -1) for var in np.arange(num_vars) + 1]
        bold_cells.extend([(0, lvl) for lvl in np.arange(num_lvls)])

        # Shade the cells and make the cell text larger
        for cell in stats_table._cells:
            stats_table._cells[cell].set_alpha(.7)

        plt.suptitle("Permutations: %s" % settings.DEFAULT_FRONT_NAMES[front_type])
        plt.tight_layout()
        plt.savefig('%s/permutations_%d_%s_%s.png' % (model_folder, args['model_number'], front_type, args['domain']), bbox_inches='tight', dpi=400)
        plt.close()