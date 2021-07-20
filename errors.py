"""
Script that contains custom errors.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 7/19/2021 8:51 PM CDT
"""


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
