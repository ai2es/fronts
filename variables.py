"""
Function that creates new variable datasets.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Modified for Xarray by: John Allen (allen4jt@cmich.edu)
Last updated: 7/19/2021 3:17 PM CDT by Andrew Justin

---Sources used in this code---
(Bolton 1980): https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
(Davies-Jones 2008): https://doi.org/10.1175/2007MWR2224.1
(Stull 2011): https://doi.org/10.1175/JAMC-D-11-0143.1
"""

import xarray as xr
import numpy as np


def dew_point_from_specific_humidity(P, T, q):
    """
    Returns dataset containing dew-point temperature data with units of kelvin (K).

    Written for wrapped FORTRAN by John T. Allen - July 2015
    Written in python by Chiara Lepore - April 2019
    Updated to xarray by Chiara and John - August 2019
    Modified for fronts project by Andrew Justin - July 2021
    https://www.vaisala.com/sites/default/files/documents/Humidity_Conversion_Formulas_B210973EN-F.pdf

    Parameters
    ----------
    P: Dataset
        Xarray dataset containing data for pressure. Pressure has units of pascals (Pa).
    T: Dataset
        Xarray dataset containing data for air temperature. Air temperature has units of kelvin (K).
    q: Dataset
        Xarray dataset containing data for specific humidity. Specific humidity has units of grams of water vapor per
        gram of dry air (g/g).

    Returns
    -------
    Td: Dataset
        Xarray dataset containing data for dew-point temperature. Dew-point temperature has units of kelvin (K).
    """
    P = P / 100  # Convert pressure from Pa to hPa (hectopascals).

    m = 7.591386
    A = 6.116441
    Tn = 240.7263
    m1 = 9.778707
    A1 = 6.114742
    Tn1 = 273.1466

    E = (P * q) / (0.622 + 0.378 * q)  # (Bolton 1980, Eq. 16)

    # for T>=0
    Td_gt = Tn / ((m / (np.log10(E / A))) - 1)
    Td_lt = Tn1 / ((m1 / (np.log10(E / A1))) - 1)
    Td = xr.where(T >= 0, Td_gt, Td_lt)
    return Td+273.15


def mixing_ratio(Td, P):
    """
    Returns dataset containing mixing ratio data with units of grams of water per gram of dry air (g/g).

    Parameters
    ----------
    Td: Dataset
        Xarray dataset containing data for dew-point temperature. Dew-point temperature has units of kelvin (K).
    P: Dataset
        Xarray dataset containing data for pressure. Pressure has units of pascals (Pa).

    Returns
    -------
    r: Dataset
        Xarray dataset containing data for mixing ratio. Mixing ratio has units of grams of water per gram of dry air
        (g/g).
    """
    epsilon = 0.622
    P = P / 100  # Convert pressure from Pa to hPa (hectopascals).

    e = 6.112 * np.exp((17.67 * (Td - 273.15)) / ((Td - 273.15) + 243.5))  # (Bolton 1980, Eq. 10)
    r = (epsilon * e) / (P - e)  # Mixing ratio with units of g/g
    return r


def relative_humidity(T, Td):
    """
    Returns dataset containing relative humidity data.

    Parameters
    ----------
    T: Dataset
        Xarray dataset containing data for air temperature. Air temperature has units of kelvin (K).
    Td: Dataset
        Xarray dataset containing data for dew-point temperature. Dew-point temperature has units of kelvin
        (K).

    Returns
    -------
    RH: Dataset
        Xarray dataset containing data for relative humidity. Relative humidity has no units.
    """
    e = 6.112 * np.exp((17.67 * (Td - 273.15)) / ((Td - 273.15) + 243.5))  # (Bolton 1980, Eq. 10)
    es = 6.112 * np.exp((17.67 * (T - 273.15)) / ((T - 273.15) + 243.5))  # (Bolton 1980, Eq. 10)

    RH = e / es
    return RH


def specific_humidity(Td, P):
    """
    Returns dataset containing specific humidity data.

    Parameters
    ----------
    Td: Dataset
        Xarray dataset containing data for dew-point temperature. Dew-point temperature has units of kelvin (K).
    P: Dataset
        Xarray dataset containing data for pressure. Pressure has units of pascals (Pa).

    Returns
    -------
    q: Dataset
        Xarray dataset containing data for specific humidity. Specific humidity has units of grams of water vapor per
        gram of dry air (g/g).
    """
    epsilon = 0.622
    P = P / 100  # Convert pressure from Pa to hPa (hectopascals).

    e = 6.112 * np.exp((17.67 * (Td - 273.15)) / ((Td - 273.15) + 243.5))  # (Bolton 1980, Eq. 10)
    q = (epsilon * e) / (P - (0.378 * e))

    return q


def theta_e(T, Td, P):
    """
    Returns dataset containing wet-bulb potential temperature (theta-w) data with units of kelvin (K).

    Parameters
    ----------
    T: Dataset
        Xarray dataset containing data for air temperature. Air temperature has units of kelvin (K).
    Td: Dataset
        Xarray dataset containing data for dew-point temperature. Dew-point temperature has units of kelvin
        (K).
    P: Dataset
        Xarray dataset containing data for air pressure. Air pressure has units of pascals (Pa).

    Returns
    -------
    theta_w: Dataset
        Xarray dataset containing data for wet-bulb potential temperature. Wet-bulb potential temperature has units of
        kelvin (K).
    """
    epsilon = 0.622
    p_knot = 1000  # hPa
    kd = 0.286
    P = P / 100  # Convert pressure from Pa to hPa (hectopascals).

    e = 6.112 * np.exp((17.67 * (Td - 273.15)) / ((Td - 273.15) + 243.5))  # Vapor pressure (Bolton 1980, Eq. 10)
    r = (epsilon * e) / (P - e)
    TL = (((1 / (Td - 56)) + (np.log(T / Td) / 800)) ** (-1)) + 56  # LCL temperature (Bolton 1980, Eq. 15)
    theta_L = T * ((p_knot / (P - e)) ** kd) * (
            (T / TL) ** (0.28 * r))  # LCL potential temperature (Bolton 1980, Eq. 24)
    theta_e = theta_L * (np.exp(((3036 / TL) - 1.78) * r * (1 + (0.448 * r))))  # (Bolton 1980, Eq. 39)

    return theta_e


def theta_w(T, Td, P):
    """
    Returns dataset containing wet-bulb potential temperature (theta-w) data with units of kelvin (K).

    Parameters
    ----------
    T: Dataset
        Xarray dataset containing data for air temperature. Air temperature has units of kelvin (K).
    Td: Dataset
        Xarray dataset containing data for dew-point temperature. Dew-point temperature has units of kelvin
        (K).
    P: Dataset
        Xarray dataset containing data for pressure. Pressure has units of pascals (Pa).

    Returns
    -------
    theta_w: Dataset
        Xarray dataset containing data for wet-bulb potential temperature. Wet-bulb potential temperature has units of
        kelvin (K).
    """
    epsilon = 0.622
    C = 273.15  # K (Davies-Jones 2008, Section 2)
    p_knot = 1000  # hPa
    kd = 0.286
    P = P / 100  # Convert pressure from Pa to hPa (hectopascals).

    e = 6.112 * np.exp((17.67 * (Td - 273.15)) / ((Td - 273.15) + 243.5))  # Vapor pressure (Bolton 1980, Eq. 10)
    r = (epsilon * e) / (P - e)
    TL = (((1 / (Td - 56)) + (np.log(T / Td) / 800)) ** (-1)) + 56  # LCL temperature (Bolton 1980, Eq. 15)
    theta_L = T * ((p_knot / (P - e)) ** kd) * (
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
    X = theta_e / C

    # Wet-bulb potential temperature approximation (Davies-Jones 2008, Eq. 3.8).
    theta_wc = theta_e - np.exp(
        (a0 + (a1 * X) + (a2 * np.power(X, 2)) + (a3 * np.power(X, 3)) + (a4 * np.power(X, 4))) /
        (1 + (b1 * X) + (b2 * np.power(X, 2)) + (b3 * np.power(X, 3)) + (b4 * np.power(X, 4))))
    theta_w = xr.where(theta_e > 173.15, theta_wc, theta_e)

    return theta_w


def virtual_temperature(T, Td, P):
    """
    Returns dataset containing virtual temperature data with units of kelvin (K).

    Parameters
    ----------
    T: Dataset
        Xarray dataset containing data for air temperature. Air temperature has units of kelvin (K).
    Td: Dataset
        Xarray dataset containing data for dew-point temperature. Dew-point temperature has units of kelvin
        (K).
    P: Dataset
        Xarray dataset containing data for pressure. Pressure has units of pascals (Pa).

    Returns
    -------
    Tv: Dataset
        Xarray dataset containing data for virtual temperature. Virtual temperature has units of kelvin (K).
    """
    epsilon = 0.622
    P = P / 100  # Convert pressure from Pa to hPa (hectopascals).

    e = 6.112 * np.exp((17.67 * (Td - 273.15)) / ((Td - 273.15) + 243.5))  # Vapor pressure (Bolton 1980, Eq. 10)
    r = (epsilon * e) / (P - e)

    Tv = T * (1 + (r / epsilon)) / (1 + r)
    return Tv


def wet_bulb_temperature(T, Td):
    """
    Returns dataset containing wet-bulb temperature data with units of kelvin (K).

    Parameters
    ----------
    T: Dataset
        Xarray dataset containing data for air temperature. Air temperature has units of kelvin (K).
    Td: Dataset
        Xarray dataset containing data for dew-point temperature. Dew-point temperature has units of kelvin
        (K).

    Returns
    -------
    Tw: Dataset
        Xarray dataset containing data for wet-bulb temperature. Wet-bulb temperature has units of kelvin (K).
    """
    e = 6.112 * np.exp((17.67 * (Td - 273.15)) / ((Td - 273.15) + 243.5))  # Vapor pressure (Bolton 1980, Eq. 10)
    es = 6.112 * np.exp((17.67 * (T - 273.15)) / ((T - 273.15) + 243.5))  # Saturation vapor pressure (Bolton 1980, Eq. 10)

    RH = 100 * e / es
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
