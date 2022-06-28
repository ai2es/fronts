"""
Script that contains custom errors.

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 6/18/2021 2:02 PM CDT
"""


class ArgumentConflictError(TypeError):
    """
    Error raised if two arguments conflict with each other.
    """
    pass


class MissingArgumentError(TypeError):
    """
    Error raised if a necessary argument is missing.
    """
    pass


def check_arguments(provided_arguments, required_arguments):
    """
    Function that checks if the necessary arguments were provided to perform a necessary operation(s) or if certain
    arguments are incompatible with each other.

    Parameters
    ----------
    provided_arguments: list
    required_arguments: list
    """
    missing_arguments = []

    for argument in required_arguments:
        if provided_arguments[argument] is None:
            missing_arguments.append(argument)

    if len(missing_arguments) > 0:
        raise MissingArgumentError(f'%d argument(s) are missing: {", ".join(list(sorted(missing_arguments)))}' % len(missing_arguments))
