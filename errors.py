"""
Script that contains custom errors.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 9/23/2021 7:29 PM CDT
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
        print("error")
        raise MissingArgumentError(f'%d argument(s) are missing: {", ".join(list(sorted(missing_arguments)))}' % len(missing_arguments))
    else:
        print("done")
