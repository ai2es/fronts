"""
Functions for deriving various thermodynamic variables.

References
----------
* Bolton 1980: https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
* Davies-Jones 2008: https://doi.org/10.1175/2007MWR2224.1
* Stull 2011: https://doi.org/10.1175/JAMC-D-11-0143.1
* Vasaila 2013: https://www.vaisala.com/sites/default/files/documents/Humidity_Conversion_Formulas_B210973EN-F.pdf

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.10.21
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


def absolute_humidity(T: int | float | np.ndarray | tf.Tensor,
                      Td: int | float | np.ndarray | tf.Tensor):
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

    Examples
    --------
    >>> T = 300  # K
    >>> Td = 290  # K
    >>> AH = absolute_humidity(T, Td)  # kg * m^-3
    >>> AH
    0.012493639535490526

    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> Td = np.arange(260, 301, 5)  # K
    >>> Td
    array([260, 265, 270, 275, 280, 285, 290, 295, 300])
    >>> AH = absolute_humidity(T, Td)  # kg * m^-3
    >>> AH
    array([0.00198323, 0.00277676, 0.00383828, 0.00524175, 0.00707688,
           0.00945143, 0.01249364, 0.0163548 , 0.02121195])
    """
    e = vapor_pressure(Td)  # vapor pressure expressed as pascals
    AH = e / (Rv * T)  # absolute humidity
    return AH


def dewpoint_from_vapor_pressure(vapor_pressure: int | float | np.ndarray | tf.Tensor):
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

    Examples
    --------
    >>> vap_pres = 1000  # Pa
    >>> Td = dewpoint_from_vapor_pressure(vap_pres)
    >>> Td
    280.8734119482131

    >>> vap_pres = np.arange(500, 4001, 500)  # Pa
    >>> vap_pres
    array([ 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
    >>> Td = dewpoint_from_vapor_pressure(vap_pres)
    >>> Td
    array([270.12031682, 280.87341195, 287.56990934, 292.51813134,
           296.4751267 , 299.7885851 , 302.64840758, 305.170169  ])
    """
    Td = (1/273.15 - (tf.math.log(vapor_pressure/e_knot)*(Rv/Lv))) ** -1 if tf.is_tensor(vapor_pressure) else \
         (1/273.15 - (np.log(vapor_pressure/e_knot)*(Rv/Lv))) ** -1
    return Td


def dewpoint_from_specific_humidity(
    P: int | float | np.ndarray | tf.Tensor,
    T: int | float | np.ndarray | tf.Tensor,
    q: int | float | np.ndarray | tf.Tensor):
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

    Examples
    --------
    >>> P = 1e5  # Pa
    >>> T = 300  # K
    >>> q = 20 / 1000  # g/kg -> kg/kg
    >>> Td = dewpoint_from_specific_humidity(P, T, q)
    >>> Td
    298.199585429495

    >>> P = np.arange(800, 1001, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> q = np.arange(5, 25.01, 2.5) / 1000  # g/kg -> kg/kg
    >>> q
    array([0.005 , 0.0075, 0.01  , 0.0125, 0.015 , 0.0175, 0.02  , 0.0225,
           0.025 ])
    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> Td = dewpoint_from_specific_humidity(P, T, q)
    >>> Td
    array([273.80033167, 279.97353235, 284.66312436, 288.51045188,
           291.80950762, 294.72063163, 297.34130998, 299.7354364 ,
           301.94726732])
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


def mixing_ratio_from_dewpoint(Td: int | float | np.ndarray | tf.Tensor,
                               P: int | float | np.ndarray | tf.Tensor):
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

    Examples
    --------
    >>> Td = 290  # K
    >>> P = 1e5  # Pa
    >>> r = mixing_ratio_from_dewpoint(Td, P)
    >>> r
    0.010947893449979635

    >>> Td = np.arange(260, 301, 5)  # K
    >>> Td
    array([260, 265, 270, 275, 280, 285, 290, 295, 300])
    >>> P = np.arange(800, 1001, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> r = mixing_ratio_from_dewpoint(Td, P)
    >>> r
    array([0.00192723, 0.0026682 , 0.00365056, 0.00493959, 0.00661507,
           0.00877413, 0.01153478, 0.01504042, 0.01946563])
    """
    e = vapor_pressure(Td)
    r = epsilon * e / (P - e)  # mixing ratio
    return r


def potential_temperature(T: int | float | np.ndarray | tf.Tensor,
                          P: int | float | np.ndarray | tf.Tensor):
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

    Examples
    --------
    >>> T = 275  # K
    >>> P = 9e4  # Pa
    >>> theta = potential_temperature(T, P)
    >>> theta
    283.3951954331142

    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> P = np.arange(800, 1001, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> theta = potential_temperature(T, P)
    >>> theta
    array([287.75518363, 290.52120387, 293.2937428 , 296.07144546,
           298.85311518, 301.63769299, 304.42424008, 307.21192283,
           310.        ])
    """
    theta = T * tf.pow(1e5 / P, kd) if tf.is_tensor(T) and tf.is_tensor(P) else T * np.power(1e5 / P, kd)
    return theta


def relative_humidity_from_dewpoint(T, Td):

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

    Examples
    --------
    >>> T = 300  # K
    >>> Td = 290  # K
    >>> RH = relative_humidity_from_dewpoint(T, Td)
    >>> RH
    0.5699908521249278

    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> Td = np.arange(260, 301, 5)  # K
    >>> Td
    array([260, 265, 270, 275, 280, 285, 290, 295, 300])
    >>> RH = relative_humidity_from_dewpoint(T, Td)
    >>> RH
    array([0.49824518, 0.51115071, 0.52366592, 0.53579872, 0.54755768,
           0.5589519 , 0.56999085, 0.58068426, 0.591042  ])
    """

    exp_func = tf.exp if tf.is_tensor(T) and tf.is_tensor(Td) else np.exp
    RH = exp_func(Lv/Rv * (1/T - 1/Td))

    return RH


def relative_humidity_from_mixing_ratio(T, r, P):
    """
    Returns relative humidity from temperature, mixing ratio, and pressure.

    Parameters
    ----------
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    r: float or iterable object
        Mixing ratio expressed as grams of water per gram of dry air (unitless; g/g or kg/kg).
    P: float or iterable object
        Air pressure expressed as pascals (Pa).

    Returns
    -------
    RH: float or iterable object
        Relative humidity.

    Examples
    --------
    >>> T = 300  # K
    >>> r = 15 / 1000  # g/kg ---> g/g
    >>> P = 1e5  # Pa
    >>> relative_humidity_from_mixing_ratio(T, r, P)
    0.7759915394735595

    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> r = np.arange(1, 9.01, 1) / 1000  # g/kg -> kg/kg
    >>> r
    array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009])
    >>> P = np.arange(800, 1001, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> relative_humidity_from_mixing_ratio(T, r, P)
    array([0.25891395, 0.38355349, 0.4307923 , 0.43453214, 0.41493737,
           0.38391574, 0.3483986 , 0.31231771, 0.27780294])
    """

    exp_func = tf.exp if tf.is_tensor(T) else np.exp
    RH = (P * r / (e_knot * (epsilon + r))) * exp_func(Lv/Rv * (1/T - 1/273.15))

    return RH


def specific_humidity_from_dewpoint(Td: int | float | np.ndarray | tf.Tensor,
                                    P: int | float | np.ndarray | tf.Tensor):
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

    Examples
    --------
    >>> Td = 290  # K
    >>> P = 1e5  # Pa
    >>> q = specific_humidity_from_dewpoint(Td, P)
    >>> q
    0.010829329732443743

    >>> Td = np.arange(260, 301, 5)  # K
    >>> Td
    array([260, 265, 270, 275, 280, 285, 290, 295, 300])
    >>> P = np.arange(800, 1001, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> q = specific_humidity_from_dewpoint(Td, P)
    >>> q
    array([0.00192352, 0.0026611 , 0.00363728, 0.00491531, 0.0065716 ,
           0.00869781, 0.01140324, 0.01481755, 0.01909393])
    """
    e = vapor_pressure(Td)
    return epsilon * e / (P - (0.378 * e))  # q: specific humidity


def mixing_ratio_from_specific_humidity(q: int | float | np.ndarray | tf.Tensor):
    """
    Calculates mixing ratio from specific humidity.

    Parameters
    ----------
    q: float or iterable object
        Specific humidity expressed as grams of water vapor per gram of dry air (unitless; g/g or kg/kg).

    Returns
    -------
    r: float or iterable object
        Mixing ratio expressed as grams of water per gram of dry air (unitless; g/g or kg/kg).

    Examples
    --------
    >>> q = 20 / 1000  # g/kg -> kg/kg
    >>> r = mixing_ratio_from_specific_humidity(q)
    >>> r * 1000  # kg/kg -> g/kg
    20.408163265306126

    >>> q = np.arange(5, 25.01, 2.5) / 1000  # g/kg -> kg/kg
    >>> q
    array([0.005 , 0.0075, 0.01  , 0.0125, 0.015 , 0.0175, 0.02  , 0.0225,
           0.025 ])
    >>> r = mixing_ratio_from_specific_humidity(q)
    >>> r * 1000  # kg/kg -> g/kg
    array([ 5.02512563,  7.55667506, 10.1010101 , 12.65822785, 15.2284264 ,
           17.81170483, 20.40816327, 23.01790281, 25.64102564])
    """
    return q / (1 - q)  # r: mixing ratio


def specific_humidity_from_mixing_ratio(r: int | float | np.ndarray | tf.Tensor):
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

    Examples
    --------
    >>> r = 20 / 1000  # g/kg -> kg/kg
    >>> q = specific_humidity_from_mixing_ratio(r)
    >>> q * 1000  # kg/kg -> g/kg
    19.607843137254903

    >>> r = np.arange(5, 25.01, 2.5) / 1000  # g/kg -> kg/kg
    >>> r
    array([0.005 , 0.0075, 0.01  , 0.0125, 0.015 , 0.0175, 0.02  , 0.0225,
           0.025 ])
    >>> q = specific_humidity_from_mixing_ratio(r)
    >>> q * 1000  # kg/kg -> g/kg
    array([ 4.97512438,  7.44416873,  9.9009901 , 12.34567901, 14.77832512,
           17.1990172 , 19.60784314, 22.00488998, 24.3902439 ])
    """
    return r / (1 + r)  # q: specific humidity


def specific_humidity_from_relative_humidity(
    RH: int | float | np.ndarray | tf.Tensor,
    T: int | float | np.ndarray | tf.Tensor,
    P: int | float | np.ndarray | tf.Tensor):
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

    Examples
    --------
    >>> RH = 0.8
    >>> T = 300  # K
    >>> P = 1e5  # Pa
    >>> q = specific_humidity_from_relative_humidity(RH, T, P)
    >>> q * 1000  # kg/kg -> g/kg
    15.239787946316241

    >>> RH = np.arange(0.5, 0.91, 0.05)
    >>> RH
    array([0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 ])
    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> P = np.arange(800, 1001, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> q = specific_humidity_from_relative_humidity(RH, T, P)
    >>> q * 1000
    array([ 1.93030564,  2.86370132,  4.16882688,  5.96677284,  8.41051444,
           11.6918278 , 16.04970636, 21.78077898, 29.25247145])
    """
    es = vapor_pressure(T)
    e = RH * es
    w = epsilon * e / (P - e)
    q = w / (w + 1)
    return q


def equivalent_potential_temperature(
    T: int | float | np.ndarray | tf.Tensor,
    Td: int | float | np.ndarray | tf.Tensor,
    P: int | float | np.ndarray | tf.Tensor):
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

    Examples
    --------
    >>> T = 300  # K
    >>> Td = 290  # K
    >>> P = 1e5  # Pa
    >>> theta_e = equivalent_potential_temperature(T, Td, P)
    >>> theta_e
    326.52430009577137

    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> Td = np.arange(260, 301, 5)  # K
    >>> Td
    array([260, 265, 270, 275, 280, 285, 290, 295, 300])
    >>> P = np.arange(800, 1001, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> theta_e = equivalent_potential_temperature(T, Td, P)
    >>> theta_e
    array([292.582033  , 297.1606042 , 302.3295548 , 308.2501438 ,
           315.12588597, 323.21500183, 332.84798857, 344.45298499,
           358.59322677])
    """
    RH = relative_humidity_from_dewpoint(T, Td)
    theta = potential_temperature(T, P)
    rv = mixing_ratio_from_dewpoint(Td, P)
    theta_e = theta * tf.pow(RH, -rv * Rv / Cpd) * tf.exp(Lv * rv / (Cpd * T)) if all(tf.is_tensor(var) for var in [T, Td, P]) else \
        theta * np.power(RH, -rv * Rv / Cpd) * np.exp(Lv * rv / (Cpd * T))
    return theta_e


def wet_bulb_temperature(T: int | float | np.ndarray | tf.Tensor,
                         Td: int | float | np.ndarray | tf.Tensor):
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

    Examples
    --------
    >>> T = 300  # K
    >>> Td = 290  # K
    >>> Tw = wet_bulb_temperature(T, Td)
    >>> Tw
    293.8520102695189

    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> Td = np.arange(260, 301, 5)  # K
    >>> Td
    array([260, 265, 270, 275, 280, 285, 290, 295, 300])
    >>> Tw = wet_bulb_temperature(T, Td)
    >>> Tw
    array([266.93692268, 271.30728293, 275.72885706, 280.1976556 ,
           284.70999386, 289.26248252, 293.85201027, 298.47572385,
           303.13100751])
    """
    RH = relative_humidity_from_dewpoint(T, Td) * 100
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


def wet_bulb_potential_temperature(T: int | float | np.ndarray | tf.Tensor,
                                   Td: int | float | np.ndarray | tf.Tensor,
                                   P: int | float | np.ndarray | tf.Tensor):
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

    Examples
    --------
    >>> T = 300  # K
    >>> Td = 290  # K
    >>> P = 1e5  # Pa
    >>> theta_w = wet_bulb_potential_temperature(T, Td, P)
    >>> theta_w
    array(290.65670769)

    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> Td = np.arange(260, 301, 5)  # K
    >>> Td
    array([260, 265, 270, 275, 280, 285, 290, 295, 300])
    >>> P = np.arange(800, 1001, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> theta_w = wet_bulb_potential_temperature(T, Td, P)
    >>> theta_w
    array([277.86226914, 280.00624048, 282.24273162, 284.58957316,
           287.06192678, 289.67108554, 292.42350276, 295.32019538,
           298.35666949])
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


def vapor_pressure(Td: int | float | np.ndarray | tf.Tensor):
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

    Examples
    --------
    >>> Td = 290  # K
    >>> vap_pres = vapor_pressure(Td)
    >>> vap_pres
    1729.7443936886634

    >>> Td = np.arange(260, 301, 5)  # K
    >>> Td
    array([260, 265, 270, 275, 280, 285, 290, 295, 300])
    >>> vap_pres = vapor_pressure(Td)
    >>> vap_pres
    array([ 247.12075845,  352.40493817,  495.98223586,  689.43450819,
            947.13483326, 1286.74161001, 1729.74439369, 2302.0614118 ,
           3034.68799059])
    """

    vap_pres = e_knot * tf.exp((Lv/Rv) * ((1/273.15) - (1/Td))) if tf.is_tensor(Td) else e_knot * np.exp((Lv/Rv) * ((1/273.15) - (1/Td)))
    return vap_pres


def virtual_potential_temperature(T: int | float | np.ndarray | tf.Tensor,
                                  Td: int | float | np.ndarray | tf.Tensor,
                                  P: int | float | np.ndarray | tf.Tensor):
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

    Examples
    --------
    >>> T = 300  # K
    >>> Td = 290  # K
    >>> P = 1e5  # Pa
    >>> theta_v = virtual_potential_temperature(T, Td, P)
    >>> theta_v
    301.9745879930382

    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> Td = np.arange(260, 301, 5)  # K
    >>> Td
    array([260, 265, 270, 275, 280, 285, 290, 295, 300])
    >>> P = np.arange(800, 1001, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> theta_v = virtual_potential_temperature(T, Td, P)
    >>> theta_v
    array([288.09159758, 290.9910892 , 293.94212815, 296.95595223,
           300.04678011, 303.23228364, 306.53413747, 309.97866221,
           313.59758378])
    """
    Tv = virtual_temperature_from_dewpoint(T, Td, P)
    theta_v = potential_temperature(Tv, P)
    return theta_v


def virtual_temperature_from_mixing_ratio(T: int | float | np.ndarray | tf.Tensor,
                                          r: int | float | np.ndarray | tf.Tensor):
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

    Examples
    --------
    >>> T = 300  # K
    >>> r = 20 / 1000  # g/kg -> kg/kg
    >>> Tv = virtual_temperature_from_mixing_ratio(T, r)
    >>> Tv
    303.5752344416027

    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> r = np.arange(5, 25.01, 2.5) / 1000  # g/kg -> kg/kg
    >>> r
    array([0.005 , 0.0075, 0.01  , 0.0125, 0.015 , 0.0175, 0.02  , 0.0225,
           0.025 ])
    >>> Tv = virtual_temperature_from_mixing_ratio(T, r)
    >>> Tv
    array([270.81643413, 276.24423481, 281.68496197, 287.13851986,
           292.60481366, 298.08374951, 303.57523444, 309.07917641,
           314.59548427])
    """
    return T * (1 + (r / epsilon)) / (1 + r)


def virtual_temperature_from_dewpoint(T: int | float | np.ndarray | tf.Tensor,
                                      Td: int | float | np.ndarray | tf.Tensor,
                                      P: int | float | np.ndarray | tf.Tensor):
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

    Examples
    --------
    >>> T = 300  # K
    >>> Td = 290  # K
    >>> P = 1e5  # Pa
    >>> Tv = virtual_temperature_from_dewpoint(T, Td, P)
    >>> Tv
    301.9745879930382

    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> Td = np.arange(260, 301, 5)  # K
    >>> Td
    array([260, 265, 270, 275, 280, 285, 290, 295, 300])
    >>> P = np.arange(800, 1001, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> Tv = virtual_temperature_from_dewpoint(T, Td, P)
    >>> Tv
    array([270.3156564 , 275.44478153, 280.61899683, 285.85143108,
           291.15830423, 296.55950086, 302.07923396, 307.74681888,
           313.59758378])
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
