"""
Generate plots for permutation data from a model.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.11.4
"""

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
    permutations_dict = pd.read_pickle('%s/permutations_%d_%s.pkl' % (model_folder, args['model_number'], args['domain']))

    num_vars = len(variables)  # number of variables
    num_lvls = len(pressure_levels)  # number of pressure levels

    sp = permutations_dict['single_pass']
    sp_vars = np.array([sp[var] for var in variables])  # single-pass results: shuffled variables over all levels
    sp_lvls = np.array([sp[lvl] for lvl in pressure_levels])  # single-pass results: shuffled levels over all variables
    sp_sorting_orders = [np.argsort(sp_vars, axis=0), np.argsort(sp_lvls, axis=0)]  # list of length 2: sorted grouped variables and grouped levels

    for front_num, front_type in enumerate(front_types):
        sp_sorted_data = [sp_vars[:, front_num + 1][sp_sorting_orders[0][:, front_num + 1]], sp_lvls[:, front_num + 1][sp_sorting_orders[1][:, front_num + 1]]]
        sp_sorted_vars_and_lvls = [[variables[ind] for ind in sp_sorting_orders[0][:, front_num + 1]], [pressure_levels[ind] for ind in sp_sorting_orders[1][:, front_num + 1]]]

        mp = permutations_dict['multi_pass']['front_type'][front_type]
        mp_keys = list(mp.keys())
        mp_vars = np.array([mp_keys[var] for var in np.arange(num_vars)])  # multi-pass results: shuffled variables over all levels
        mp_lvls = np.array([mp_keys[lvl] for lvl in np.arange(num_lvls) + num_vars])  # multi-pass results: shuffled levels over all variables
        mp_sorted_data = [[mp[var] for var in mp_keys[:num_vars]][::-1], [mp[lvl] for lvl in mp_keys[num_vars:]][::-1]]

        y_pos_vars = np.arange(num_vars)  # array marking the y-positions of the variables
        y_pos_lvls = np.arange(num_lvls)  # array marking the y-positions of the pressure levels

        fig = plt.figure(figsize=(12, 9))

        ax1 = plt.subplot(3, 2, 1)
        ax2 = plt.subplot(3, 2, 2)
        ax3 = plt.subplot(3, 2, 3)
        ax4 = plt.subplot(3, 2, 4)
        ax5 = plt.subplot(3, 1, 3)

        sp_barh_vars = ax1.barh(y_pos_vars, sp_sorted_data[0], color=settings.DEFAULT_FRONT_COLORS[front_type])
        mp_barh_vars = ax2.barh(y_pos_vars, mp_sorted_data[0], color=settings.DEFAULT_FRONT_COLORS[front_type])
        sp_barh_lvls = ax3.barh(y_pos_lvls, sp_sorted_data[1], color=settings.DEFAULT_FRONT_COLORS[front_type])
        mp_barh_lvls = ax4.barh(y_pos_lvls, mp_sorted_data[1], color=settings.DEFAULT_FRONT_COLORS[front_type])

        ax1.set_title('a) Single-pass: grouped variables')
        ax1.set_yticks(y_pos_vars)
        ax1.set_yticklabels(sp_sorted_vars_and_lvls[0])
        ax2.set_title('b) Multi-pass: grouped variables')
        ax2.set_yticks(y_pos_vars[::-1])
        ax2.set_yticklabels(mp_vars)
        ax3.set_title('c) Single-pass: grouped levels')
        ax3.set_yticks(y_pos_lvls)
        ax3.set_yticklabels(sp_sorted_vars_and_lvls[1])
        ax4.set_title('d) Multi-pass: grouped levels')
        ax4.set_yticks(y_pos_lvls[::-1])
        ax4.set_yticklabels(mp_lvls)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlabel("Relative importance")
            ax.grid(alpha=0.3, axis='x')

        cellValues = np.array([[sp['_'.join([var, lvl])][front_num + 1] for lvl in pressure_levels] for var in variables])
        # cellText = np.array([[str(sp['_'.join([var, lvl])][front_num + 1]) for lvl in pressure_levels] for var in variables])
        sorted_indices = np.argsort(-cellValues.flatten())

        ranks = np.zeros_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(sorted_indices)) + 1
        cellText = np.array(['%d (%.3f)' % (rank, val) for rank, val in zip(ranks, cellValues.flatten())]).reshape(cellValues.shape)

        max_val = np.max(cellValues)
        cellColorValues = 0.5 + (cellValues / (2 * max_val))
        cellColorValues /= np.max(cellColorValues)
        cmap = mpl.colormaps.get_cmap('bwr_r')
        cellColours = [cmap(val) for val in cellColorValues]

        rowColours = ['gray' for _ in range(len(variables))]
        colColours = ['gray' for _ in range(len(pressure_levels))]
        ax5.set_title('e) Single-pass: variables on single levels')
        ax5.axis('off')
        stats_table = ax5.table(cellText=cellText, rowLabels=variables, colLabels=pressure_levels, rowColours=rowColours,
                                colColours=colColours, cellColours=cellColours, cellLoc='center', bbox=[0, 0, 1, 1])
        stats_table.scale(1, 1)  # Make the table larger

        bold_cells = [(var, -1) for var in np.arange(num_vars) + 1]
        bold_cells.extend([(0, lvl) for lvl in np.arange(num_lvls)])

        ### Shade the cells and make the cell text larger ###
        for cell in stats_table._cells:
            stats_table._cells[cell].set_alpha(.7)

        plt.suptitle(settings.DEFAULT_FRONT_NAMES[front_type])
        plt.tight_layout()
        plt.savefig('%s/permutations_%d_%s_%s.png' % (model_folder, args['model_number'], front_type, args['domain']), bbox_inches='tight', dpi=400)
        plt.close()