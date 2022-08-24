"""
Function that creates new variable datasets.

References
----------
    - Bolton 1980: https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    - Davies-Jones 2008: https://doi.org/10.1175/2007MWR2224.1
    - Stull 2011: https://doi.org/10.1175/JAMC-D-11-0143.1

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
    - Modified for Xarray by: John Allen (allen4jt@cmich.edu)

Last updated: 7/28/2022 11:52 PM CDT by Andrew Justin
"""

import xarray as xr
import numpy as np
from pandas import read_csv

Rd = 287.04  # Gas constant for dry air (J/kg/K)
Rv = 461.5  # Gas constant for water vapor (J/kg/K)
Cpd = 1005.7  # Specific heat of dry air at constant pressure (J/kg/K)
Cpw = 4184  # Specific heat capacity of liquid water (J/kg/K)
kd = Rd / Cpd  # Exponential constant for potential temperature
epsilon = Rd / Rv
P_knot = 1e5  # Pa
e_knot = 611.2  # Pa
Lv = 2.257e6  # Latent heat of vaporization for water vapor (J/kg)


def absolute_humidity(T, Td):
    """
    Calculates absolute humidity for a given dewpoint temperature in kelvin (K).

    Parameters
    ----------
    T: float, iterable, xr.Dataset
        Air temperature in kelvin (K).
    Td: float, iterable, xr.Dataset
        Dewpoint temperature in kelvin (K).

    Returns
    -------
    AH: float, iterable, or xr.Dataset
        Absolute humidity.
    """
    e = vapor_pressure(Td)
    return e / (Rv * T)  # Absolute humidity


def dewpoint_from_vapor_pressure(vapor_pressure):
    """
    Calculates vapor pressure in pascals (Pa) for a given dewpoint temperature in kelvin (K).

    Parameters
    ----------
    vapor_pressure: float, iterable, or xr.Dataset
        Vapor pressure in pascals (Pa).

    Returns
    -------
    Td: float, iterable, or xr.Dataset
        Dewpoint temperature in kelvin (K).
    """
    return (1/273.15 - (np.log(vapor_pressure/e_knot)*(Rv/Lv))) ** -1  # Td


def dewpoint_from_specific_humidity(P, T, q):
    """
    Returns dewpoint temperature in kelvin (K).

    Written for wrapped FORTRAN by John T. Allen - July 2015
    Written in python by Chiara Lepore - April 2019
    Updated to xarray by Chiara and John - August 2019
    Modified for fronts project by Andrew Justin - July 2021
    https://www.vaisala.com/sites/default/files/documents/Humidity_Conversion_Formulas_B210973EN-F.pdf

    Parameters
    ----------
    P: float, iterable, or xr.Dataset
        Air pressure in pascals (Pa).
    T: float, iterable, or xr.Dataset
        Air temperature in kelvin (K).
    q: float, iterable, or xr.Dataset
        Specific humidity in grams of water vapor per gram of dry air (g/g).

    Returns
    -------
    Td: float, iterable, or xr.Dataset
        Dew-point temperature in kelvin (K).
    """

    m = 7.591386
    A = 6.116441
    Tn = 240.7263
    m1 = 9.778707
    A1 = 6.114742
    Tn1 = 273.1466

    E = (P * q) / (0.622 + 0.378 * q)  # (Bolton 1980, Eq. 16)
    E /= 100  # convert to hPa

    # for T>=0
    Td_gt = Tn / ((m / (np.log10(E / A))) - 1)
    Td_lt = Tn1 / ((m1 / (np.log10(E / A1))) - 1)
    Td = xr.where(T >= 0, Td_gt, Td_lt)
    return Td + 273.15


def mixing_ratio_from_dewpoint(Td, P):
    """
    Returns mixing ratio in of grams of water per gram of dry air (g/g).

    Parameters
    ----------
    Td: float, iterable, or xr.Dataset
        Dewpoint temperature in kelvin (K).
    P: float, iterable, or xr.Dataset
        Pressure in pascals (Pa).

    Returns
    -------
    r: float, iterable, or xr.Dataset
        Mixing ratio in grams of water per gram of dry air (g/g).
    """
    e = vapor_pressure(Td)
    return epsilon * e / (P - e)  # Mixing ratio with units of g/g


def potential_temperature(T, P):
    """
    Returns potential temperature in kelvin (K).

    Parameters
    ----------
    T: float, iterable, or xr.Dataset
        Air temperature in kelvin (K).
    P: float, iterable, or xr.Dataset
        Pressure in pascals (Pa).

    Returns
    -------
    theta: float, iterable, or xr.Dataset
        Potential temperature in kelvin (K).
    """
    return T * np.power(1e5 / P, kd)


def relative_humidity(T, Td):
    """
    Returns relative humidity.

    Parameters
    ----------
    T: float, iterable, or xr.Dataset
        Air temperature in kelvin (K).
    Td: float, iterable, or xr.Dataset
        Dewpoint temperature in kelvin (K).

    Returns
    -------
    RH: float, iterable, or xr.Dataset
        Relative humidity.
    """
    e = vapor_pressure(Td)
    es = vapor_pressure(T)  # Saturation vapor pressure, when Td increases to T if T is held constant
    return e / es  # relative humidity


def specific_humidity_from_dewpoint(Td, P):
    """
    Returns specific humidity in grams of water vapor per gram of dry air (g/g).

    Parameters
    ----------
    Td: float, iterable, or xr.Dataset
        Dewpoint temperature in kelvin (K).
    P: float, iterable, or xr.Dataset
        Pressure in pascals (Pa).

    Returns
    -------
    q: float, iterable, or xr.Dataset
        Specific humidity in grams of water vapor per gram of dry air (g/g).
    """
    e = vapor_pressure(Td)
    return epsilon * e / (P - (0.378 * e))  # q: specific humidity


def specific_humidity_from_mixing_ratio(r):
    """
    Returns specific humidity in grams of water vapor per gram of dry air (g/g).

    Parameters
    ----------
    r: float, iterable, or xr.Dataset
        Mixing ratio in in grams of water vapor per gram of dry air (g/g).

    Returns
    -------
    q: float, iterable, or xr.Dataset
        Specific humidity in grams of water vapor per gram of dry air (g/g).
    """
    return r / (1 + r)  # q: specific humidity


def specific_humidity_from_relative_humidity(RH, T, P):
    """
    Returns specific humidity in grams of water vapor per gram of dry air (g/g).

    Parameters
    ----------
    RH: float, iterable, or xr.Dataset
        Relative humidity.
    T: float, iterable, or xr.Dataset
        Temperature in kelvin (K).
    P: float, iterable, or xr.Dataset
        Pressure in pascals (Pa).

    Returns
    -------
    q: float, iterable, or xr.Dataset
        Specific humidity in grams of water vapor per gram of dry air (g/g).
    """
    es = vapor_pressure(T)
    e = RH * es
    w = epsilon * e / (P - e)
    q = w / (w + 1)
    return q


def equivalent_potential_temperature(T, Td, P):
    """
    Returns equivalent potential temperature (theta-e) in kelvin (K).

    Parameters
    ----------
    T: float, iterable, or xr.Dataset
        Air temperature in kelvin (K).
    Td: float, iterable, or xr.Dataset
        Dewpoint temperature in kelvin (K).
    P: float, iterable, or xr.Dataset
        Pressure in pascals (Pa).

    Returns
    -------
    theta_e: float, iterable, or xr.Dataset
        Equivalent potential temperature in kelvin (K).
    """
    RH = relative_humidity(T, Td)
    theta = potential_temperature(T, P)
    rv = mixing_ratio_from_dewpoint(Td, P)
    return theta * np.power(RH, -rv * Rv / Cpd) * np.exp(Lv * rv / (Cpd * T))  # theta-e (https://glossary.ametsoc.org/wiki/Equivalent_potential_temperature)


def wet_bulb_temperature(T, Td):
    """
    Returns wet-bulb temperature with units of kelvin (K).

    Parameters
    ----------
    T: float, iterable, or xr.Dataset
        Air temperature in kelvin (K).
    Td: float, iterable, or xr.Dataset
        Dewpoint temperature in kelvin (K).

    Returns
    -------
    Tw: float, iterable, or xr.Dataset
        Wet-bulb temperature in kelvin (K).
    """
    RH = relative_humidity(T, Td) * 100
    c1 = 0.151977
    c2 = 8.313659
    c3 = 1.676331
    c4 = 0.00391838
    c5 = 0.023101
    c6 = 4.686035

    # Wet-bulb temperature approximation (Stull 2011, Eq. 1)
    Tw = (T - 273.15) * np.arctan(c1 * np.power(RH + c2, 0.5)) + np.arctan(T - 273.15 + RH) - np.arctan(RH - c3) + (
            c4 * np.power(RH, 1.5) * np.arctan(c5 * RH)) - c6 + 273.15

    return Tw


def wet_bulb_potential_temperature(T, Td, P):
    """
    Returns wet-bulb potential temperature (theta-w) in kelvin (K).

    Parameters
    ----------
    T: float, iterable, or xr.Dataset
        Air temperature in kelvin (K).
    Td: float, iterable, or xr.Dataset
        Dewpoint temperature in kelvin (K).
    P: float, iterable, or xr.Dataset
        Pressure in pascals (Pa).

    Returns
    -------
    theta_w: float, iterable, or xr.Dataset
        Wet-bulb potential temperature in kelvin (K).
    """

    e = vapor_pressure(Td)
    r = mixing_ratio_from_dewpoint(Td, P)
    TL = (((1 / (Td - 56)) + (np.log(T / Td) / 800)) ** (-1)) + 56  # LCL temperature (Bolton 1980, Eq. 15)
    theta_L = T * ((P_knot / (P - e)) ** kd) * (
            (T / TL) ** (0.28 * r))  # LCL potential temperature (Bolton 1980, Eq. 24)
    theta_e = theta_L * (np.exp(((3036 / TL) - 1.78) * r * (1 + (0.448 * r))))  # (Bolton 1980, Eq. 39)

    # Wet-bulb potential temperature constants and X variable (Davies-Jones 2008, Section 3).
    a0 = 7.101574
    a1 = -20.68208
    a2 = 16.11182
    a3 = 2.574631
    a4 = -5.205688
    b1 = -3.552497
    b2 = 3.781782
    b3 = -0.6899655
    b4 = -0.592934
    C = 273.15  # K (Davies-Jones 2008, Section 2)
    X = theta_e / C

    # Wet-bulb potential temperature approximation (Davies-Jones 2008, Eq. 3.8).
    theta_wc = theta_e - np.exp(
        (a0 + (a1 * X) + (a2 * np.power(X, 2)) + (a3 * np.power(X, 3)) + (a4 * np.power(X, 4))) /
        (1 + (b1 * X) + (b2 * np.power(X, 2)) + (b3 * np.power(X, 3)) + (b4 * np.power(X, 4))))
    theta_w = xr.where(theta_e > 173.15, theta_wc, theta_e)

    return theta_w


def vapor_pressure(Td):
    """
    Calculates vapor pressure in pascals (Pa) for a given dewpoint temperature in kelvin (K).

    Parameters
    ----------
    Td: float, iterable, or xr.Dataset
        Dewpoint temperature in kelvin (K).

    Returns
    -------
    vapor_pressure: float, iterable, or xr.Dataset
        Vapor pressure in pascals (Pa).
    """
    return e_knot * np.exp((Lv/Rv) * ((1/273.15) - (1/Td)))


def virtual_potential_temperature(T, Td, P):
    """
    Returns virtual potential temperature (theta-v) in kelvin (K).

    Parameters
    ----------
    T: float, iterable, or xr.Dataset
        Air temperature in kelvin (K).
    Td: float, iterable, or xr.Dataset
        Dewpoint temperature in kelvin (K).
    P: float, iterable, or xr.Dataset
        Pressure in pascals (Pa).

    Returns
    -------
    theta_v: float, iterable, or xr.Dataset
        Virtual potential temperature in kelvin (K).
    """
    Tv = virtual_temperature_from_dewpoint(T, Td, P)
    return potential_temperature(Tv, P)


def virtual_temperature_from_mixing_ratio(T, r):
    """
    Calculates virtual temperature.

    Parameters
    ----------
    T: float, iterable, or xr.Dataset
        Air temperature in kelvin (K).
    r: float, iterable, or xr.Dataset
        Mixing ratio in grams of water per gram of dry air (g/g).

    Returns
    -------
    Tv: float, iterable, or xr.Dataset
        Virtual temperature in kelvin (K).
    """
    return T * (1 + (r / epsilon)) / (1 + r)  # Tv: virtual temperature


def virtual_temperature_from_dewpoint(T, Td, P):
    """
    Calculates virtual temperature.

    Parameters
    ----------
    T: float, iterable, or xr.Dataset
        Air temperature in kelvin (K).
    Td: float, iterable, or xr.Dataset
        Dewpoint temperature in kelvin (K).
    P: float, iterable, or xr.Dataset
        Pressure in pascals (Pa).

    Returns
    -------
    Tv: float, iterable, or xr.Dataset
        Virtual temperature in kelvin (K).
    """
    r = mixing_ratio_from_dewpoint(Td, P)
    return virtual_temperature_from_mixing_ratio(T, r)  # virtual temperature
