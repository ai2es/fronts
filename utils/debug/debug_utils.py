"""
General debugging tools

Code written by: Andrew Justin (andrewjustinwx@gmail.com)

Last updated: 9/23/2022 3:42 PM CT
"""


def generate_date_list(years, include_hours: str = None):
    """
    Generate a nested list of /timesteps for a given year(s).

    Parameters
    ----------
    years: int or iterable of ints
    include_hours: 'synoptic', 'non-synoptic', 'all', or None (strings are case-insensitive)
        - Optional argument that includes hours in the list of dates.
        - If None, no hours will be included.
        - If 'synoptic', hours 0z, 6z, 12z, and 18z will be included.
        - If 'non-synoptic', hours 3z, 9z, 15z, and 21z will be included.
        - If 'all', then both synoptic and non-synoptic hours will be included.

    Returns
    -------
    date_list: nested list of dates/timesteps
        - If parameter 'include_hours' is None, each index of this list will be a list with the following format: [year, month, day]
        - If parameter 'include_hours' is 'synoptic', 'non-synoptic', or 'all', each index of this list will be a list with the following format: [year, month, day, hour]

    Raises
    ------
    TypeError
        - If parameter 'include_hours' is not a string

    ValueError
        - If parameter 'include_hours' is a string but is not any of: 'synoptic', 'non-synoptic', 'all'
    """

    if include_hours is not None:
        include_hours = include_hours.lower()  # make the string lowercase (this way the argument is case-insensitive)
    elif type(include_hours) != str:
        raise TypeError(f"include_hours must be a string or None, received type: {type(include_hours)}")
    else:
        if not any(include_hours == string for string in ['synoptic', 'non-synoptic', 'all']):
            raise ValueError("If include_hours is a string, it must be one of the following: 'synoptic', 'non-synoptic', 'all'")

    date_list = []
    days_per_month = dict({str(year): None for year in years})  # Number of days in each month for each year being analyzed
    for year in years:
        if year % 4 == 0:
            month_2_days = 29  # Leap year
        else:
            month_2_days = 28  # Not a leap year

        days_per_month[str(year)] = [31, month_2_days, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        for month in range(1, 13):
            for day in range(1, days_per_month[str(year)][month - 1] + 1):

                if include_hours == 'synoptic':
                    for hour in range(0, 24, 6):
                        date_list.append([year, month, day, hour])
                elif include_hours == 'non-synoptic':
                    for hour in range(3, 24, 6):
                        date_list.append([year, month, day, hour])
                elif include_hours == 'all':
                    for hour in range(0, 24, 3):
                        date_list.append([year, month, day, hour])
                else:
                    date_list.append([year, month, day])

    return date_list
