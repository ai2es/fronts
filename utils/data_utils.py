"""
Data tools

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 7/30/2023 6:18 PM CT
"""
import numpy as np
import xarray as xr


# Each variable has parameters in the format of [max, min]
normalization_parameters = {'mslp_z_surface': [1050., 960.],
                            'mslp_z_1000': [48., -69.],
                            'mslp_z_950': [86., -27.],
                            'mslp_z_900': [127., 17.],
                            'mslp_z_850': [174., 63.],
                            'q_surface': [24., 0.],
                            'q_1000': [26., 0.],
                            'q_950': [26., 0.],
                            'q_900': [23., 0.],
                            'q_850': [21., 0.],
                            'RH_surface': [1., 0.],
                            'RH_1000': [1., 0.],
                            'RH_950': [1., 0.],
                            'RH_900': [1., 0.],
                            'RH_850': [1., 0.],
                            'r_surface': [25., 0.],
                            'r_1000': [22., 0.],
                            'r_950': [22., 0.],
                            'r_900': [20., 0.],
                            'r_850': [18., 0.],
                            'sp_z_surface': [1075., 620.],
                            'sp_z_1000': [48., -69.],
                            'sp_z_950': [86., -27.],
                            'sp_z_900': [127., 17.],
                            'sp_z_850': [174., 63.],
                            'theta_surface': [331., 213.],
                            'theta_1000': [322., 218.],
                            'theta_950': [323., 219.],
                            'theta_900': [325., 227.],
                            'theta_850': [330., 237.],
                            'theta_e_surface': [375., 213.],
                            'theta_e_1000': [366., 208.],
                            'theta_e_950': [367., 210.],
                            'theta_e_900': [364., 227.],
                            'theta_e_850': [359., 238.],
                            'theta_v_surface': [324., 212.],
                            'theta_v_1000': [323., 218.],
                            'theta_v_950': [319., 215.],
                            'theta_v_900': [315., 220.],
                            'theta_v_850': [316., 227.],
                            'theta_w_surface': [304., 212.],
                            'theta_w_1000': [301., 207.],
                            'theta_w_950': [302., 210.],
                            'theta_w_900': [301., 214.],
                            'theta_w_850': [300., 237.],
                            'T_surface': [323., 212.],
                            'T_1000': [322., 218.],
                            'T_950': [319., 216.],
                            'T_900': [314., 220.],
                            'T_850': [315., 227.],
                            'Td_surface': [304., 207.],
                            'Td_1000': [302., 208.],
                            'Td_950': [301., 210.],
                            'Td_900': [298., 200.],
                            'Td_850': [296., 200.],
                            'Tv_surface': [324., 211.],
                            'Tv_1000': [323., 206.],
                            'Tv_950': [319., 206.],
                            'Tv_900': [316., 220.],
                            'Tv_850': [316., 227.],
                            'Tw_surface': [305., 212.],
                            'Tw_1000': [305., 218.],
                            'Tw_950': [304., 216.],
                            'Tw_900': [301., 219.],
                            'Tw_850': [299., 227.],
                            'u_surface': [36., -35.],
                            'u_1000': [38., -35.],
                            'u_950': [48., -55.],
                            'u_900': [59., -58.],
                            'u_850': [59., -58.],
                            'v_surface': [30., -35.],
                            'v_1000': [35., -38.],
                            'v_950': [55., -56.],
                            'v_900': [58., -59.],
                            'v_850': [58., -59.]}


def add_or_subtract_hours_to_timestep(timestep, num_hours):
    """
    Find the valid timestep for a given number of added or subtracted hours.

    Parameters
    ----------
    timestep: int, str, tuple
        * Int format for the timestep: YYYYMMDDHH
        * String format for the timestep: YYYY-MM-DD-HH (no dashes)
        * Tuple format for the timestep (all integers): (YYYY, MM, DD, HH)
    num_hours: int
        Number of hours to add or subtract to the new timestep.

    Returns
    -------
    New timestep after adding or subtracting the given number of hours. This new timestep will be returned to the same format
    as the input timestep.
    """

    timestep_type = type(timestep)

    if timestep_type == str or timestep_type == np.str_:
        year = int(timestep[:4])
        month = int(timestep[4:6])
        day = int(timestep[6:8])
        hour = int(timestep[8:])
    elif timestep_type == tuple:
        if all(type(timestep_tuple_element) == int for timestep_tuple_element in timestep):
            timestep = tuple([str(timestep_element) for timestep_element in timestep])
        year, month, day, hour = timestep[0], timestep[1], timestep[2], timestep[3]
    else:
        raise TypeError("Timestep must be a string or a tuple with integers in one of the following formats: YYYYMMDDHH or (YYYY, MM, DD, HH)")

    if year % 4 == 0:  # Check if the current year is a leap year
        month_2_days = 29
    else:
        month_2_days = 28

    days_per_month = [31, month_2_days, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    new_year, new_month, new_day, new_hour = year, month, day, hour + num_hours
    if new_hour > 0:
        # If the new timestep is in a future day, this loop will be activated
        while new_hour > 23:
            new_hour -= 24
            new_day += 1

        # If the new timestep is in a future month, this loop will be activated
        while new_day > days_per_month[new_month - 1]:
            new_day -= days_per_month[new_month - 1]
            if new_month < 12:
                new_month += 1
            else:
                new_month = 1
                new_year += 1
    else:
        # If the new timestep is in a past day, this loop will be activated
        while new_hour < 0:
            new_hour += 24
            new_day -= 1

        # If the new timestep is in a past month, this loop will be activated
        while new_day < 1:
            new_day += days_per_month[new_month - 2]
            if new_month > 1:
                new_month -= 1
            else:
                new_month = 12
                new_year -= 1

    return '%d%02d%02d%02d' % (new_year, new_month, new_day, new_hour)


def normalize_variables(variable_ds, normalization_parameters=normalization_parameters):
    """
    Function that normalizes GDAS variables via min-max normalization.

    Parameters
    ----------
    variable_ds: xr.Dataset
        - Dataset containing ERA5, GDAS, or GFS variable data.
    normalization_parameters: dict
        - Dictionary containing parameters for normalization.

    Returns
    -------
    variable_ds: xr.Dataset
        - Same as input dataset, but the variables are normalized via min-max normalization.
    """

    variable_list = list(variable_ds.keys())
    pressure_levels = variable_ds['pressure_level'].values

    new_var_shape = np.shape(variable_ds[variable_list[0]].values)

    for var in variable_list:

        new_values_for_variable = np.empty(shape=new_var_shape)

        for pressure_level_index in range(len(pressure_levels)):

            norm_var = '%s_%s' % (var, pressure_levels[pressure_level_index])

            # Min-max normalization
            if len(np.shape(new_values_for_variable)) == 4:
                new_values_for_variable[:, :, :, pressure_level_index] = np.nan_to_num((variable_ds[var].values[:, :, :, pressure_level_index] - normalization_parameters[norm_var][1]) /
                                                                                       (normalization_parameters[norm_var][0] - normalization_parameters[norm_var][1]))

            elif len(np.shape(new_values_for_variable)) == 5:  # If forecast hours are in the dataset
                new_values_for_variable[:, :, :, :, pressure_level_index] = np.nan_to_num((variable_ds[var].values[:, :, :, :, pressure_level_index] - normalization_parameters[norm_var][1]) /
                                                                                          (normalization_parameters[norm_var][0] - normalization_parameters[norm_var][1]))

        variable_ds[var].values = new_values_for_variable

    return variable_ds
