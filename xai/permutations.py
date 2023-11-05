"""
Calculate permutation importance.

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.11.4

TODO:
    * Generalize and simplify multi-pass importance code
    * Add more documentation
"""

import itertools
import os
import pandas as pd
import custom_losses
from utils.data_utils import combine_datasets
from glob import glob
from file_manager import load_model
import tensorflow as tf
import numpy as np
import argparse
import pickle


def shuffle_inputs(image, labels):
    """
    image:
    """
    if level_nums is None:
        lvl_nums = [None, ]
    elif type(level_nums) == int:
        lvl_nums = [level_nums, ]
    else:
        lvl_nums = level_nums

    if variable_nums is None:
        var_nums = [None, ]
    elif type(variable_nums) == int:
        var_nums = [variable_nums, ]
    else:
        var_nums = variable_nums

    for var_num in var_nums:
        for lvl_num in lvl_nums:

            values_to_shuffle = image[..., lvl_num, var_num]
            num_elements = tf.size(values_to_shuffle)

            lon_indices = tf.random.uniform([num_elements], 0, image.shape[0] - 1, dtype=tf.int32)
            lat_indices = tf.random.uniform([num_elements], 0, image.shape[1] - 1, dtype=tf.int32)

            if lvl_num is None:
                pressure_level_indices = tf.random.uniform([num_elements], 0, image.shape[2] - 1, dtype=tf.int32)
            else:
                pressure_level_indices = tf.cast(tf.fill([num_elements], lvl_num), tf.int32)

            if var_num is None:
                variable_indices = tf.random.uniform([num_elements], 0, image.shape[3] - 1, dtype=tf.int32)
            else:
                variable_indices = tf.cast(tf.fill([num_elements], var_num), tf.int32)

            indices = tf.stack([lon_indices, lat_indices, pressure_level_indices, variable_indices], axis=-1)
            image = tf.tensor_scatter_nd_update(image, indices, tf.reshape(values_to_shuffle, [num_elements]))

    return image, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--tf_indir', type=str, required=True, help='Input directory for the tensorflow dataset(s).')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory where the models are or will be saved to.')
    parser.add_argument('--model_number', type=int, required=True, help='Number that the model will be assigned.')
    parser.add_argument('--baseline', action='store_true', help='Calculate baseline loss values for the permutations.')
    parser.add_argument('--single_pass', action='store_true', help='Perform single-pass permutations.')
    parser.add_argument('--multi_pass', action='store_true', help='Perform multi-pass permutations.')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 2**31 - 1), help='Seed for the random number generators.')

    args = vars(parser.parse_args())

    model = load_model(args['model_number'], args['model_dir'])
    model_properties = pd.read_pickle('%s/model_%d/model_%d_properties.pkl' % (args['model_dir'], args['model_number'], args['model_number']))

    num_classes = model_properties['classes']
    front_types = model_properties['dataset_properties']['front_types']
    variables = model_properties['dataset_properties']['variables']
    pressure_levels = model_properties['dataset_properties']['pressure_levels']
    domain = model_properties['dataset_properties']['domain']
    num_outputs = 4

    try:
        optimizer_args = {arg: model_properties[arg] for arg in ['learning_rate', 'beta_1', 'beta_2']}
        optimizer = getattr(tf.keras.optimizers, model_properties['optimizer'])(**optimizer_args)
    except KeyError:
        optimizer = getattr(tf.keras.optimizers, model_properties['optimizer'][0])(**model_properties['optimizer'][1])

    loss_args = model_properties['loss_args']
    if model_properties['loss_string'] == 'fss_loss':
        loss = custom_losses.fractions_skill_score(**loss_args)

    metric_functions = dict()
    for front_no, front_type in enumerate(front_types):
        class_weights = np.zeros(num_classes)
        class_weights[front_no + 1] = 1
        loss_args['class_weights'] = class_weights
        metric_functions[front_type] = custom_losses.fractions_skill_score(2, class_weights=class_weights)
        metric_functions[front_type]._name = front_type

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric_functions[func] for func in metric_functions])

    X_datasets = list(sorted(glob(args['tf_indir'] + '/era5*tf')))[:1]
    y_datasets = list(sorted(glob(args['tf_indir'] + '/fronts*tf')))[:1]
    X = combine_datasets(X_datasets)
    y = combine_datasets(y_datasets)

    permutations_file = "%s/model_%d/permutations_%d_%s.pkl" % (args['model_dir'], args['model_number'], args['model_number'], domain)
    permutations_dict = dict() if not os.path.isfile(permutations_file) else pd.read_pickle(permutations_file)

    if 'seed' not in permutations_dict:
        permutations_dict['seed'] = args['seed']
    tf.random.set_seed(permutations_dict['seed'])

    print("Seed: %d" % permutations_dict['seed'])
    if 'baseline' not in permutations_dict or args['baseline']:
        print("=== Baselines ===")

        print("--> opening datasets")
        Xy = combine_datasets(X_datasets, y_datasets)
        Xy = Xy.batch(32)

        print("--> generating predictions")
        prediction = np.array(model.evaluate(Xy, verbose=0))

        # overall loss for each front type
        baseline = np.array([np.sum(prediction[np.arange(1 + num_outputs + front_no, num_outputs * num_classes + front_no, num_classes - 1)]) for front_no in range(num_classes - 1)])
        baseline = np.insert(baseline, 0, prediction[0])  # add overall model loss to the front of the array

        print("--> baseline loss:", baseline[0])
        for front_no, front_type in enumerate(front_types):
            print("--> baseline loss (%s): %s" % (front_type, str(baseline[front_no + 1])))

        permutations_dict['baseline'] = baseline

        print("--> saving results to %s" % permutations_file)
        with open(permutations_file, 'wb') as f:
            pickle.dump(permutations_dict, f)

    if 'single_pass' not in permutations_dict or args['single_pass']:

        assert 'baseline' in permutations_dict, "Must calculate baseline values prior to permutations!"
        baseline = permutations_dict['baseline']
        permutations_dict['single_pass'] = dict() if 'single_pass' not in permutations_dict else permutations_dict['single_pass']

        print("\n=== Single-pass permutations ===")

        combinations = list([variable, None] for variable in variables)
        combinations.extend(list([None, level] for level in pressure_levels))
        combinations.extend(list(combo) for combo in itertools.product(variables, pressure_levels))
        
        num_combinations = len(combinations)
        for combination_num, combination in enumerate(combinations):
            variable, level = combination
            variable_nums = variables.index(variable) if variable is not None else None
            level_nums = pressure_levels.index(level) if level is not None else None
            
            printout_str = "(%d/%d) " % (combination_num + 1, num_combinations)
            if variable is None:
                printout_str += "%s, all variables" % level
                permutation_dict_key = level
            elif level is None:
                printout_str += "%s, all levels" % variable
                permutation_dict_key = variable
            else:
                printout_str += "%s_%s" % (variable, level)
                permutation_dict_key = "%s_%s" % (variable, level)

            if permutation_dict_key in permutations_dict['single_pass']:
                printout_str += " [SKIPPING]"
                print(printout_str)
                continue

            print(printout_str)
            Xy = combine_datasets(X_datasets, y_datasets)
            Xy = Xy.map(shuffle_inputs)
            Xy = Xy.batch(32)

            prediction = np.array(model.evaluate(Xy, verbose=0))
            loss = np.array([np.sum(prediction[np.arange(1 + num_outputs + front_no, num_outputs * num_classes + front_no, num_classes - 1)]) for front_no in range(num_classes - 1)])
            loss = np.insert(loss, 0, prediction[0])
            importance = np.round(100 * (loss - baseline) / baseline, 3)
            permutations_dict['single_pass'][permutation_dict_key] = importance

            with open(permutations_file, 'wb') as f:
                pickle.dump(permutations_dict, f)

    if 'multi_pass' not in permutations_dict or args['multi_pass']:

        # Find most important variable (overall and by type) based on the single-pass permutations
        single_pass_results = np.array([permutations_dict['single_pass'][var] for var in variables])
        most_important_variable = np.argmax(single_pass_results, axis=0)

        assert 'baseline' in permutations_dict, "Must calculate baseline values prior to permutations!"
        baseline = permutations_dict['baseline']
        permutations_dict['multi_pass'] = dict() if 'multi_pass' not in permutations_dict else permutations_dict['multi_pass']

        print("\n=== Multi-pass permutations ===")

        print("-- VARIABLES, ALL LEVELS --")
        level_nums = None

        ######################################### Overall variable importance ##########################################

        permutations_dict['multi_pass']['overall'] = dict() if 'overall' not in permutations_dict['multi_pass'] else permutations_dict['multi_pass']['overall']
        permutations_dict['multi_pass']['overall'][variables[most_important_variable[0]]] = np.max(single_pass_results, axis=0)[0]

        # most important variable will be the first variable to be shuffled in the shuffle_inputs function
        shuffled_parameters = np.array([most_important_variable[0]])
        variable_nums = np.array([most_important_variable[0], 0], dtype=np.int32)  # the 0 is a placeholder that will be overwritten iteratively

        variable_shuffle_order = np.arange(0, len(variables))  # variables to shuffle
        variable_shuffle_order = np.delete(variable_shuffle_order, shuffled_parameters)  # remove pre-shuffled indices

        while len(variable_shuffle_order) > 0:

            print("---> shuffling %s" % ', '.join(variables[param] for param in shuffled_parameters))
            temp_importance_list = np.array([])  # list that will be used to temporarily store importance values for the current round

            for var_to_shuffle in variable_shuffle_order:
                variable_nums[-1] = var_to_shuffle

                Xy = combine_datasets(X_datasets, y_datasets)
                Xy = Xy.map(shuffle_inputs)
                Xy = Xy.batch(32)

                loss = np.array(model.evaluate(Xy, verbose=0))[0]
                importance = np.round(100 * (loss - baseline[0]) / baseline[0], 3)
                print("-> %s:" % variables[var_to_shuffle], importance)

                temp_importance_list = np.append(temp_importance_list, importance)

            most_important_variable_for_round = variable_shuffle_order[np.argmax(temp_importance_list)]
            permutations_dict['multi_pass']['overall'][variables[most_important_variable_for_round]] = np.max(temp_importance_list)

            shuffled_parameters = np.append(shuffled_parameters, most_important_variable_for_round)
            variable_nums[-1] = most_important_variable_for_round  # add the round's most important variable to the list of variables to shuffle
            variable_shuffle_order = np.delete(variable_shuffle_order, np.argmax(temp_importance_list))  # remove the round's most important variable for the next round
            variable_nums = np.append(variable_nums, 0)  # add another index for variables in the next round

            with open(permutations_file, 'wb') as f:
                pickle.dump(permutations_dict, f)

        ########################################## Variable importance by type #########################################

        permutations_dict['multi_pass']['front_type'] = dict() if 'front_type' not in permutations_dict['multi_pass'] else permutations_dict['multi_pass']['front_type']

        for front_no, front_type in enumerate(front_types):

            permutations_dict['multi_pass']['front_type'][front_type] = dict() \
                if front_type not in permutations_dict['multi_pass']['front_type'] else permutations_dict['multi_pass']['front_type'][front_type]

            permutations_dict['multi_pass']['front_type'][front_type][variables[most_important_variable[front_no + 1]]] = np.max(single_pass_results, axis=0)[front_no + 1]

            # most important variable will be the first variable to be shuffled in the shuffle_inputs function
            shuffled_parameters = np.array([most_important_variable[front_no + 1]])
            variable_nums = np.array([most_important_variable[front_no + 1], 0], dtype=np.int32)  # the 0 is a placeholder that will be overwritten iteratively

            variable_shuffle_order = np.arange(0, len(variables))  # variables to shuffle
            variable_shuffle_order = np.delete(variable_shuffle_order, shuffled_parameters)  # remove pre-shuffled indices

            while len(variable_shuffle_order) > 0:

                print(f"---> {front_type}: shuffling %s" % ', '.join(variables[param] for param in shuffled_parameters))
                temp_importance_list = np.array([])  # list that will be used to temporarily store importance values for the current round

                for var_to_shuffle in variable_shuffle_order:
                    variable_nums[-1] = var_to_shuffle

                    Xy = combine_datasets(X_datasets, y_datasets)
                    Xy = Xy.map(shuffle_inputs)
                    Xy = Xy.batch(32)

                    losses = np.array(model.evaluate(Xy, verbose=0))
                    loss_by_type = np.array([np.sum(losses[np.arange(1 + num_outputs + front_no, num_outputs * num_classes + front_no, num_classes - 1)]) for front_no in range(num_classes - 1)])
                    importance = np.round(100 * (loss_by_type[front_no] - baseline[front_no + 1]) / baseline[front_no + 1], 3)
                    print("-> %s:" % variables[var_to_shuffle], importance)

                    temp_importance_list = np.append(temp_importance_list, importance)

                most_important_variable_for_round = variable_shuffle_order[np.argmax(temp_importance_list)]
                permutations_dict['multi_pass']['front_type'][front_type][variables[most_important_variable_for_round]] = np.max(temp_importance_list)

                shuffled_parameters = np.append(shuffled_parameters, most_important_variable_for_round)
                variable_nums[-1] = most_important_variable_for_round  # add the round's most important variable to the list of variables to shuffle
                variable_shuffle_order = np.delete(variable_shuffle_order, np.argmax(temp_importance_list))  # remove the round's most important variable for the next round
                variable_nums = np.append(variable_nums, 0)  # add another index for variables in the next round

                with open(permutations_file, 'wb') as f:
                    pickle.dump(permutations_dict, f)

        print("-- LEVELS, ALL VARIABLES --")
        variable_nums = None

        ######################################### Overall level importance ##########################################
        single_pass_results = np.array([permutations_dict['single_pass'][lvl] for lvl in pressure_levels])
        most_important_level = np.argmax(single_pass_results, axis=0)

        permutations_dict['multi_pass']['overall'] = dict() if 'overall' not in permutations_dict['multi_pass'] else permutations_dict['multi_pass']['overall']
        permutations_dict['multi_pass']['overall'][pressure_levels[most_important_level[0]]] = np.max(single_pass_results, axis=0)[0]

        # most important level will be the first level to be shuffled in the shuffle_inputs function
        shuffled_levels = np.array([most_important_level[0]])
        level_nums = np.array([most_important_level[0], 0], dtype=np.int32)  # the 0 is a placeholder that will be overwritten iteratively

        level_shuffle_order = np.arange(0, len(pressure_levels))  # levels to shuffle
        level_shuffle_order = np.delete(level_shuffle_order, shuffled_levels)  # remove pre-shuffled indices

        while len(level_shuffle_order) > 0:

            print("---> shuffling %s" % ', '.join(pressure_levels[param] for param in shuffled_levels))
            temp_importance_list = np.array([])  # list that will be used to temporarily store importance values for the current round

            for level_to_shuffle in level_shuffle_order:
                level_nums[-1] = level_to_shuffle

                Xy = combine_datasets(X_datasets, y_datasets)
                Xy = Xy.map(shuffle_inputs)
                Xy = Xy.batch(32)

                loss = np.array(model.evaluate(Xy, verbose=0))[0]
                importance = np.round(100 * (loss - baseline[0]) / baseline[0], 3)
                print("-> %s:" % pressure_levels[level_to_shuffle], importance)

                temp_importance_list = np.append(temp_importance_list, importance)

            most_important_level_for_round = level_shuffle_order[np.argmax(temp_importance_list)]
            permutations_dict['multi_pass']['overall'][pressure_levels[most_important_level_for_round]] = np.max(temp_importance_list)

            shuffled_levels = np.append(shuffled_levels, most_important_level_for_round)
            level_nums[-1] = most_important_level_for_round  # add the round's most important level to the list of levels to shuffle
            level_shuffle_order = np.delete(level_shuffle_order, np.argmax(temp_importance_list))  # remove the round's most important level for the next round
            level_nums = np.append(level_nums, 0)  # add another index for levels in the next round

            with open(permutations_file, 'wb') as f:
                pickle.dump(permutations_dict, f)

        ######################################### level importance by type #########################################

        permutations_dict['multi_pass']['front_type'] = dict() if 'front_type' not in permutations_dict['multi_pass'] else permutations_dict['multi_pass']['front_type']

        for front_no, front_type in enumerate(front_types):

            permutations_dict['multi_pass']['front_type'][front_type] = dict() \
                if front_type not in permutations_dict['multi_pass']['front_type'] else permutations_dict['multi_pass']['front_type'][front_type]

            permutations_dict['multi_pass']['front_type'][front_type][pressure_levels[most_important_level[front_no + 1]]] = np.max(single_pass_results, axis=0)[front_no + 1]

            # most important level will be the first level to be shuffled in the shuffle_inputs function
            shuffled_levels = np.array([most_important_level[front_no + 1]])
            level_nums = np.array([most_important_level[front_no + 1], 0], dtype=np.int32)  # the 0 is a placeholder that will be overwritten iteratively

            level_shuffle_order = np.arange(0, len(pressure_levels))  # levels to shuffle
            level_shuffle_order = np.delete(level_shuffle_order, shuffled_levels)  # remove pre-shuffled indices

            while len(level_shuffle_order) > 0:

                print(f"---> {front_type}: shuffling %s" % ', '.join(pressure_levels[param] for param in shuffled_levels))
                temp_importance_list = np.array([])  # list that will be used to temporarily store importance values for the current round

                for level_to_shuffle in level_shuffle_order:
                    level_nums[-1] = level_to_shuffle

                    Xy = combine_datasets(X_datasets, y_datasets)
                    Xy = Xy.map(shuffle_inputs)
                    Xy = Xy.batch(32)

                    losses = np.array(model.evaluate(Xy, verbose=0))
                    loss_by_type = np.array([np.sum(losses[np.arange(1 + num_outputs + front_no, num_outputs * num_classes + front_no, num_classes - 1)]) for front_no in range(num_classes - 1)])
                    importance = np.round(100 * (loss_by_type[front_no] - baseline[front_no + 1]) / baseline[front_no + 1], 3)
                    print("-> %s:" % pressure_levels[level_to_shuffle], importance)

                    temp_importance_list = np.append(temp_importance_list, importance)

                most_important_level_for_round = level_shuffle_order[np.argmax(temp_importance_list)]
                permutations_dict['multi_pass']['front_type'][front_type][pressure_levels[most_important_level_for_round]] = np.max(temp_importance_list)

                shuffled_levels = np.append(shuffled_levels, most_important_level_for_round)
                level_nums[-1] = most_important_level_for_round  # add the round's most important level to the list of levels to shuffle
                level_shuffle_order = np.delete(level_shuffle_order, np.argmax(temp_importance_list))  # remove the round's most important level for the next round
                level_nums = np.append(level_nums, 0)  # add another index for levels in the next round

                with open(permutations_file, 'wb') as f:
                    pickle.dump(permutations_dict, f)