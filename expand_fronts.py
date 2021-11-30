"""
Function that expands frontal objects by 1 pixel in all directions (N, S, E, W, NE, SE, SW, NW).

Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 11/29/2021 7:11 PM CST
"""

import numpy as np


def one_pixel_expansion(ds_fronts):
    """
    Expands fronts by 1 pixel in all directions.

    Parameters
    ----------
    ds_fronts: Dataset that contains frontal objects.

    Returns
    -------
    ds_fronts: Dataset that contains expanded frontal objects.
    """
    lats = ds_fronts.latitude.values
    lons = ds_fronts.longitude.values
    len_lats = len(lats)
    len_lons = len(lons)
    indices = np.where(ds_fronts.identifier.values != 0)
    identifier = ds_fronts.identifier.values

    for i in range(len(indices[0])):
        front_value = identifier[indices[0][i]][indices[1][i]]
        if lats[indices[0][i]] == lats[0]:
            # If the front pixel is at the north end of the domain (max lat), and there is no front directly
            # to the south, expand the front 1 pixel south.
            if identifier[indices[0][i]][indices[1][i]] == 0:
                ds_fronts.identifier.values[indices[0][i]][indices[1][i]] = front_value
            # If the front pixel is at the northwest end of the domain (max/min lat/lon), check the pixels to the
            # southeast and east for fronts, and expand the front there if no other fronts are already present.
            if lons[indices[1][i]] == lons[0]:
                if identifier[indices[0][i] + 1][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] + 1][indices[1][i] + 1] = front_value
                if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] + 1] = front_value
            # If the front pixel is at the northeast end of the domain (max/max lat/lon), check the pixels to the
            # southwest and west for fronts, and expand the front there if no other fronts are already present.
            elif lons[indices[1][i]] == lons[len_lons - 1]:
                if identifier[indices[0][i] + 1][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] + 1][indices[1][i] - 1] = front_value
                if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] - 1] = front_value
            # If the front pixel is at the north end of the domain (max lat), but not at the west or east end (min lon
            # or max lon) check the pixels to the west and east for fronts, and expand the front there if no other
            # fronts are already present.
            else:
                if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] - 1] = front_value
                if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] + 1] = front_value
        elif lats[-1] < lats[indices[0][i]] < lats[len_lats - 1]:
            # If there is no front directly to the south, expand the front 1 pixel south.
            if identifier[indices[0][i] + 1][indices[1][i]] == 0:
                identifier[indices[0][i]][indices[1][i]] = front_value
            # If there is no front directly to the north, expand the front 1 pixel north.
            if identifier[indices[0][i] - 1][indices[1][i]] == 0:
                ds_fronts.identifier.values[indices[0][i] - 1][indices[1][i]] = front_value
            # If the front pixel is at the west end of the domain (min lon), check the pixels to the southeast,
            # east, and northeast for fronts, and expand the front there if no other fronts are already present.
            if lons[indices[1][i]] == lons[0]:
                if identifier[indices[0][i] + 1][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] + 1][indices[1][i] + 1] = front_value
                if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] + 1] = front_value
                if identifier[indices[0][i] - 1][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] - 1][indices[1][i] + 1] = front_value
            # If the front pixel is at the east end of the domain (min lon), check the pixels to the southwest,
            # west, and northwest for fronts, and expand the front there if no other fronts are already present.
            elif lons[indices[1][i]] == lons[len_lons - 1]:
                if identifier[indices[0][i] + 1][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] + 1][indices[1][i] - 1] = front_value
                if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] - 1] = front_value
                if identifier[indices[0][i] - 1][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] - 1][indices[1][i] - 1] = front_value
            # If the front pixel is not at the end of the domain in any direction, check the northeast, east,
            # southeast, northwest, west, and southwest for fronts, and expand the front there if no other fronts
            # are already present.
            else:
                if identifier[indices[0][i] + 1][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] + 1][indices[1][i] + 1] = front_value
                if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] + 1] = front_value
                if identifier[indices[0][i] - 1][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] - 1][indices[1][i] + 1] = front_value
                if identifier[indices[0][i] + 1][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] + 1][indices[1][i] - 1] = front_value
                if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] - 1] = front_value
                if identifier[indices[0][i] - 1][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] - 1][indices[1][i] - 1] = front_value
        else:
            # If the front pixel is at the south end of the domain (max lat), and there is no front directly
            # to the north, expand the front 1 pixel north.
            if identifier[indices[0][i] - 1][indices[1][i]] == 0:
                ds_fronts.identifier.values[indices[0][i] - 1][indices[1][i]] = front_value
            # If the front pixel is at the southwest end of the domain (max/min lat/lon), check the pixels to the
            # northeast and east for fronts, and expand the front there if no other fronts are already present.
            if lons[indices[1][i]] == lons[0]:
                if identifier[indices[0][i] - 1][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] - 1][indices[1][i] + 1] = front_value
                if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] + 1] = front_value
            # If the front pixel is at the southeast end of the domain (max/max lat/lon), check the pixels to the
            # northwest and west for fronts, and expand the front there if no other fronts are already present.
            elif lons[indices[1][i]] == lons[len_lons - 1]:
                if identifier[indices[0][i] - 1][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i] - 1][indices[1][i] - 1] = front_value
                if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] - 1] = front_value
            # If the front pixel is at the south end of the domain (max lat), but not at the west or east end (min lon
            # or max lon) check the pixels to the west and east for fronts, and expand the front there if no other
            # fronts are already present.
            else:
                if identifier[indices[0][i]][indices[1][i] - 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] - 1] = front_value
                if identifier[indices[0][i]][indices[1][i] + 1] == 0:
                    ds_fronts.identifier.values[indices[0][i]][indices[1][i] + 1] = front_value

    return ds_fronts
