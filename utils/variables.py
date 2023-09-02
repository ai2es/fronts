"""
Functions for deriving various thermodynamic variables.

References
----------
* Bolton 1980: https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
* Davies-Jones 2008: https://doi.org/10.1175/2007MWR2224.1
* Stull 2011: https://doi.org/10.1175/JAMC-D-11-0143.1
* Vasaila 2013: https://www.vaisala.com/sites/default/files/documents/Humidity_Conversion_Formulas_B210973EN-F.pdf

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.9.2
"""
import numpy as np
from utils import data_utils
import tensorflow as tf

Rd = 287.04  # Gas constant for dry air (J/kg/K)
Rv = 461.5  # Gas constant for water vapor (J/kg/K)
Cpd = 1005.7  # Specific heat of dry air at constant pressure (J/kg/K)
Cpw = 4184  # Specific heat capacity of liquid water (J/kg/K)
kd = Rd / Cpd  # Exponential constant for potential temperature
epsilon = Rd / Rv
P_knot = 1e5  # Pa
e_knot = 611.2  # Pa
Lv = 2.257e6  # Latent heat of vaporization for water vapor (J/kg)
NA = 6.02214076e23  # Avogrado constant (mol^-1)
kB = 1.380649e-23  # Boltzmann constant (J/K)


def absolute_humidity(T, Td):
    """
    Calculates absolute humidity from temperature and dewpoint temperature.

    Parameters
    ----------
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).

    Returns
    -------
    AH: float or iterable object
        Absolute humidity expressed as kilograms of water vapor per cubic meter of air (kg/m^3).
    """
    e = vapor_pressure(Td)  # vapor pressure expressed as pascals
    AH = e / (Rv * T)  # absolute humidity
    return AH


def dewpoint_from_vapor_pressure(vapor_pressure):
    """
    Calculates dewpoint temperature from vapor pressure.

    Parameters
    ----------
    vapor_pressure: float or iterable object
        Vapor pressure expressed as pascals (Pa).

    Returns
    -------
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).
    """
    Td = (1/273.15 - (tf.math.log(vapor_pressure/e_knot)*(Rv/Lv))) ** -1 if tf.is_tensor(vapor_pressure) else \
         (1/273.15 - (np.log(vapor_pressure/e_knot)*(Rv/Lv))) ** -1
    return Td


def dewpoint_from_specific_humidity(P, T, q):
    """
    Calculates dewpoint temperature from specific humidity, pressure, and temperature.

    Parameters
    ----------
    P: float or iterable object
        Air pressure expressed as pascals (Pa).
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    q: float or iterable object
        Specific humidity expressed as grams of water vapor per gram of dry air (unitless; g/g or kg/kg).

    Returns
    -------
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).
    """

    # Constants needed to perform dewpoint calculation (Vasaila 2013)
    m = 7.591386
    A = 6.116441
    Tn = 240.7263
    m1 = 9.778707
    A1 = 6.114742
    Tn1 = 273.1466

    vap_pres = (P * q) / (0.622 + 0.378 * q)  # (Bolton 1980, Eq. 16) expressed as pascals (Pa)
    vap_pres /= 100  # convert to hPa

    # Dewpoint calculation from Vasaila 2013
    Td_gt = Tn / ((m / (np.log10(vap_pres / A))) - 1)
    Td_lt = Tn1 / ((m1 / (np.log10(vap_pres / A1))) - 1)
    Td = np.where(T >= 0, Td_gt, Td_lt)
    return Td + 273.15


def mixing_ratio_from_dewpoint(Td, P):
    """
    Calculates mixing ratio from dewpoint temperature and air pressure.

    Parameters
    ----------
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).
    P: float or iterable object
        Air pressure expressed as pascals (Pa).

    Returns
    -------
    r: float or iterable object
        Mixing ratio expressed as grams of water per gram of dry air (unitless; g/g or kg/kg).
    """
    e = vapor_pressure(Td)
    r = epsilon * e / (P - e)  # mixing ratio
    return r


def potential_temperature(T, P):
    """
    Returns potential temperature expressed as kelvin (K).

    Parameters
    ----------
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    P: float or iterable object
        Air pressure expressed as pascals (Pa).

    Returns
    -------
    theta: float or iterable object
        Potential temperature expressed as kelvin (K).
    """
    theta = T * tf.pow(1e5 / P, kd) if tf.is_tensor(T) and tf.is_tensor(P) else T * np.power(1e5 / P, kd)
    return theta


def relative_humidity(T, Td):
    """
    Returns relative humidity from temperature and dewpoint temperature.

    Parameters
    ----------
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).

    Returns
    -------
    RH: float or iterable object
        Relative humidity.
    """
    e = vapor_pressure(Td)
    es = vapor_pressure(T)  # saturation vapor pressure
    RH = e / es  # relative humidity
    return RH


def specific_humidity_from_dewpoint(Td, P):
    """
    Calculates specific humidity from dewpoint and pressure.

    Parameters
    ----------
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).
    P: float or iterable object
        Air pressure expressed as pascals (Pa).

    Returns
    -------
    q: float or iterable object
        Specific humidity expressed as grams of water vapor per gram of dry air (unitless; g/g or kg/kg).
    """
    e = vapor_pressure(Td)
    return epsilon * e / (P - (0.378 * e))  # q: specific humidity


def specific_humidity_from_mixing_ratio(r):
    """
    Calculates specific humidity from mixing ratio.

    Parameters
    ----------
    r: float or iterable object
        Mixing ratio expressed as grams of water per gram of dry air (unitless; g/g or kg/kg).

    Returns
    -------
    q: float or iterable object
        Specific humidity expressed as grams of water vapor per gram of dry air (unitless; g/g or kg/kg).
    """
    return r / (1 + r)  # q: specific humidity


def specific_humidity_from_relative_humidity(RH, T, P):
    """
    Calculates specific humidity from relative humidity, air temperature, and pressure.

    Parameters
    ----------
    RH: float or iterable object
        Relative humidity.
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    P: float or iterable object
        Air pressure expressed as pascals (Pa).

    Returns
    -------
    q: float or iterable object
        Specific humidity expressed as grams of water vapor per gram of dry air (unitless; g/g or kg/kg).
    """
    es = vapor_pressure(T)
    e = RH * es
    w = epsilon * e / (P - e)
    q = w / (w + 1)
    return q


def equivalent_potential_temperature(T, Td, P):
    """
    Calculates equivalent potential temperature (theta-e) from temperature, dewpoint, and pressure.

    Parameters
    ----------
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).
    P: float or iterable object
        Air pressure expressed as pascals (Pa).

    Returns
    -------
    theta_e: float or iterable object
        Equivalent potential temperature expressed as kelvin (K).
    """
    RH = relative_humidity(T, Td)
    theta = potential_temperature(T, P)
    rv = mixing_ratio_from_dewpoint(Td, P)
    theta_e = theta * tf.pow(RH, -rv * Rv / Cpd) * tf.exp(Lv * rv / (Cpd * T)) if all(tf.is_tensor(var) for var in [T, Td, P]) else \
        theta * np.power(RH, -rv * Rv / Cpd) * np.exp(Lv * rv / (Cpd * T))
    return theta_e


def wet_bulb_temperature(T, Td):
    """
    Calculates wet-bulb temperature from temperature and dewpoint.

    Parameters
    ----------
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).

    Returns
    -------
    Tw: float or iterable object
        Wet-bulb temperature expressed as kelvin (K).
    """
    RH = relative_humidity(T, Td) * 100
    c1 = 0.151977
    c2 = 8.313659
    c3 = 1.676331
    c4 = 0.00391838
    c5 = 0.023101
    c6 = 4.686035

    # Wet-bulb temperature approximation (Stull 2011, Eq. 1)
    if tf.is_tensor(T):
        Tw = (T - 273.15) * tf.atan(c1 * tf.sqrt(RH + c2))
        Tw += tf.atan(T - 273.15 + RH) - tf.atan(RH - c3)
        Tw += (c4 * tf.pow(RH, 1.5) * tf.atan(c5 * RH))
    else:
        Tw = (T - 273.15) * np.arctan(c1 * np.sqrt(RH + c2))
        Tw += np.arctan(T - 273.15 + RH) - np.arctan(RH - c3)
        Tw += (c4 * np.power(RH, 1.5) * np.arctan(c5 * RH))
    Tw += 273.15 - c6

    return Tw


def wet_bulb_potential_temperature(T, Td, P):
    """
    Returns wet-bulb potential temperature (theta-w) in kelvin (K).

    Parameters
    ----------
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).
    P: float or iterable object
        Air pressure expressed as pascals (Pa).

    Returns
    -------
    theta_w: float or iterable object
        Wet-bulb potential temperature expressed as kelvin (K).
    """

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
    C = 273.15

    theta_e = equivalent_potential_temperature(T, Td, P)
    X = theta_e / C

    # Wet-bulb potential temperature approximation (Davies-Jones 2008, Eq. 3.8).
    if all(tf.is_tensor(var) for var in [T, Td, P]):
        theta_wc = theta_e - tf.exp(
            (a0 + (a1 * X) + (a2 * tf.pow(X, 2)) + (a3 * tf.pow(X, 3)) + (a4 * tf.pow(X, 4))) /
            (1 + (b1 * X) + (b2 * tf.pow(X, 2)) + (b3 * tf.pow(X, 3)) + (b4 * tf.pow(X, 4))))
        theta_w = tf.where(theta_e > 173.15, theta_wc, theta_e)
    else:
        theta_wc = theta_e - np.exp(
            (a0 + (a1 * X) + (a2 * np.power(X, 2)) + (a3 * np.power(X, 3)) + (a4 * np.power(X, 4))) /
            (1 + (b1 * X) + (b2 * np.power(X, 2)) + (b3 * np.power(X, 3)) + (b4 * np.power(X, 4))))
        theta_w = np.where(theta_e > 173.15, theta_wc, theta_e)

    return theta_w


def vapor_pressure(Td):
    """
    Calculates vapor pressure in pascals (Pa) for a given Dewpoint temperature expressed as kelvin (K).

    Parameters
    ----------
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).

    Returns
    -------
    vapor_pressure: float or iterable object
        Vapor pressure expressed as pascals (Pa).
    """

    vap_pres = e_knot * tf.exp((Lv/Rv) * ((1/273.15) - (1/Td))) if tf.is_tensor(Td) else e_knot * np.exp((Lv/Rv) * ((1/273.15) - (1/Td)))
    return vap_pres


def virtual_potential_temperature(T, Td, P):
    """
    Calculates virtual potential temperature (theta-v) from temperature, dewpoint, and pressure.

    Parameters
    ----------
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).
    P: float or iterable object
        Air pressure expressed as pascals (Pa).

    Returns
    -------
    theta_v: float or iterable object
        Virtual potential temperature expressed as kelvin (K).
    """
    Tv = virtual_temperature_from_dewpoint(T, Td, P)
    theta_v = potential_temperature(Tv, P)
    return theta_v


def virtual_temperature_from_mixing_ratio(T, r):
    """
    Calculates virtual temperature from temperature and mixing ratio.

    Parameters
    ----------
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    r: float or iterable object
        Mixing ratio expressed as grams of water per gram of dry air (unitless; g/g or kg/kg).

    Returns
    -------
    Tv: float or iterable object
        Virtual temperature expressed as kelvin (K).
    """
    return T * (1 + (r / epsilon)) / (1 + r)


def virtual_temperature_from_dewpoint(T, Td, P):
    """
    Calculates virtual temperature from temperature, dewpoint, and pressure.

    Parameters
    ----------
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).
    P: float or iterable object
        Air pressure expressed as pascals (Pa).

    Returns
    -------
    Tv: float or iterable object
        Virtual temperature expressed as kelvin (K).
    """
    r = mixing_ratio_from_dewpoint(Td, P)
    Tv = virtual_temperature_from_mixing_ratio(T, r)
    return Tv


def advection(field, u, v, lons, lats):
    """
    Calculates advection of a scalar field.

    Parameters
    ----------
    field: Array-like with shape (M, N)
        Scalar field that is being advected.
    u: Array-like with shape (M, N)
        Zonal wind velocities expressed as meters per second (m/s).
    v: Array-like with shape (M, N)
        Meridional wind velocities expressed as meters per second (m/s).
    lons: Array-like with shape (M, )
        Longitude values expressed as degrees east.
    lats: Array-like with shape (N, )
        Latitude values expressed as degrees north.
    """

    advect = np.ones(u.shape) * np.nan

    Lons, Lats = np.meshgrid(lons, lats)
    x, y = data_utils.haversine(Lons, Lats)  # x and y are expressed as kilometers
    x = x.T; y = y.T  # transpose x and y so arrays have shape M x N
    x *= 1e3; y *= 1e3  # convert x and y coordinates to meters

    dfield_dx = np.diff(field, axis=0) / np.diff(x, axis=0)
    dfield_dy = np.diff(field, axis=1) / np.diff(y, axis=1)

    advect[:-1, :-1] = - (u[:-1, :-1] * dfield_dx[:, :-1]) - (v[:-1, :-1] * dfield_dy[:-1, :])

    return advect
