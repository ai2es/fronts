"""
Miscellaneous tools

Author: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 7/7/2023 12:04 AM CT
"""


def string_arg_to_dict(arg_str: str):
    """
    Function that converts a string argument into a dictionary. Dictionaries cannot be passed through a command line, so
    this function takes a special string argument and converts it to a dictionary so arguments within a function can be
    explicitly called.

    arg_str: str
    arg_dict: dict
    """

    arg_str = arg_str.replace(' ', '')  # Remove all spaces from the string.
    arg_dict = {}  # Dictionary that will contain the arguments and their respective values

    # Iterate through all of the arguments within the string
    while True:

        equals_index = arg_str.find('=')  # Index representing where an equals sign is located, marking the end of the argument name

        ################################# Attempt to see if a tuple or list was passed #################################
        open_parenthesis_index = arg_str.find('(')
        close_parenthesis_index = arg_str.find(')')
        open_bracket_index = arg_str.find('[')
        close_bracket_index = arg_str.find(']')

        if open_parenthesis_index == close_parenthesis_index and open_bracket_index == close_bracket_index:  # These will only be equal when there are no parentheses/brackets in the argument string (i.e. there is no tuple/list)
            comma_index = arg_str.find(',')  # Index representing where the first comma is located within the 'arg' string, essentially representing the end of a argument
        elif open_parenthesis_index == -1 and close_parenthesis_index != -1:
            raise TypeError("An open parenthesis appears to be missing. Check the argument string.")
        elif open_parenthesis_index != -1 and close_parenthesis_index == -1:
            raise TypeError("A closed parenthesis appears to be missing. Check the argument string.")
        elif open_bracket_index == -1 and close_bracket_index != -1:
            raise TypeError("An open bracket appears to be missing. Check the argument string.")
        elif open_bracket_index != -1 and close_bracket_index == -1:
            raise TypeError("A closed bracket appears to be missing. Check the argument string.")
        elif open_parenthesis_index != close_parenthesis_index:
            comma_index = close_parenthesis_index + 1
        else:
            comma_index = close_bracket_index + 1

        current_arg_name = arg_str[:equals_index]

        if comma_index == -1:  # When the final argument is being added to the dictionary, this index will become -1
            current_arg_value = arg_str[equals_index + 1:]
        else:
            current_arg_value = arg_str[equals_index + 1:comma_index]

        ######################################## Convert the argument to a tuple #######################################
        if open_parenthesis_index != close_parenthesis_index:

            arg_dict[current_arg_name] = current_arg_value.replace('(', '').replace(')', '').split(',')

            if '.' in arg_dict[current_arg_name]:  # If the tuple appears to contain a float
                arg_dict[current_arg_name] = tuple([float(val) for val in arg_dict[current_arg_name]])
            else:
                arg_dict[current_arg_name] = tuple([int(val) for val in arg_dict[current_arg_name]])
        ################################################################################################################

        ######################################## Convert the argument to a list ########################################
        elif open_bracket_index != close_bracket_index:

            arg_dict[current_arg_name] = current_arg_value.replace('[', '').replace(']', '').split(',')

            list_values = []
            for val in arg_dict[current_arg_name]:
                if '.' in val:
                    list_values.append(float(val))
                else:
                    try:
                        list_values.append(int(val))
                    except ValueError:
                        list_values.append(val)

            arg_dict[current_arg_name] = list_values
        ################################################################################################################

        else:

            if '.' in current_arg_value:
                arg_dict[current_arg_name] = float(current_arg_value)
            else:
                try:
                    arg_dict[current_arg_name] = int(current_arg_value)
                except ValueError:
                    if current_arg_value == 'True':
                        arg_dict[current_arg_name] = True
                    elif current_arg_value == 'False':
                        arg_dict[current_arg_name] = False
                    else:
                        arg_dict[current_arg_name] = current_arg_value.replace("'", '')

        arg_str = arg_str[comma_index + 1:]  # After the current argument has been added to the dictionary, remove it from the argument string

        if comma_index == -1 or len(arg_str) == 0:
            break

    return arg_dict
