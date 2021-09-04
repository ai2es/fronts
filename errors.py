"""
Script that contains custom errors.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 9/4/2021 4:54 PM CDT
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


class ExtraArgumentError(TypeError):
    """
    Error raised if an unwanted argument is passed into the function.
    """
    pass
