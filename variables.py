"""
Function that creates new variable datasets.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Modified for Xarray by: John Allen (allen4jt@cmich.edu)
Last updated: 6/30/2021 1:35 PM CST by Andrew Justin

---Sources used in this code---
(Bolton 1980): https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
(Davies-Jones 2008): https://doi.org/10.1175/2007MWR2224.1
(Stull 2011): https://doi.org/10.1175/JAMC-D-11-0143.1
"""

import xarray as xr
import numpy as np


def mixing_ratio(Td, P):
    """
    Returns dataset containing mixing ratio data with units of grams of water per kilogram of dry air (g/kg).

    Parameters
    ----------
    Td: Dataset
        Xarray dataset containing data for 2-meter AGL dew-point temperature. Dew-point temperature has units of kelvin
        (K).
    P: Dataset
        Xarray dataset containing data for surface pressure. Surface pressure has units of pascals (Pa).

    Returns
    -------
    r: Dataset
        Xarray dataset containing data for 2-meter AGL mixing ratio. Mixing ratio has units of grams of water per
        kilogram of dry air (g/kg).
    """
    L = 2.5 * (10 ** 6)  # Latent heat of vaporization of water (J/kg).
    Rv = 461.5  # Gas constant for water vapor (J/kg/K).
    e_knot = 6.11  # hPa
    T_knot = 273.15  # K
    epsilon = 0.622
    P = P / 100  # Convert surface pressure from Pa to hPa (hectopascals).

    e = e_knot * (np.exp((L / Rv) * ((1 / T_knot) - (1 / Td))))
    r = (epsilon * e) / (P - e) * 1000  # Mixing ratio with units of g/kg
    return r


def relative_humidity(T, Td):
    """
    Returns dataset containing relative humidity data.

    Parameters
    ----------
    T: Dataset
        Xarray dataset containing data for 2-meter AGL air temperature. Air temperature has units of kelvin (K).
    Td: Dataset
        Xarray dataset containing data for 2-meter AGL dew-point temperature. Dew-point temperature has units of kelvin
        (K).

    Returns
    -------
    RH: Dataset
        Xarray dataset containing data for 2-meter AGL relative humidity. Relative humidity has no units.
    """
    L = 2.5 * (10 ** 6)  # Latent heat of vaporization of water (J/kg).
    Rv = 461.5  # Gas constant for water vapor (J/kg/K).
    e_knot = 6.11  # hPa
    T_knot = 273.15  # K

    e = e_knot * (np.exp((L / Rv) * ((1 / T_knot) - (1 / Td))))
    es = e_knot * (np.exp((L / Rv) * ((1 / T_knot) - (1 / T))))

    RH = e / es
    return RH


def theta_e(T, Td, P):
    """
    Returns dataset containing wet-bulb potential temperature (theta-w) data with units of kelvin (K).

    Parameters
    ----------
    T: Dataset
        Xarray dataset containing data for 2-meter AGL air temperature. Air temperature has units of kelvin (K).
    Td: Dataset
        Xarray dataset containing data for 2-meter AGL dew-point temperature. Dew-point temperature has units of kelvin
        (K).
    P: Dataset
        Xarray dataset containing data for surface pressure. Surface pressure has units of pascals (Pa).

    Returns
    -------
    theta_w: Dataset
        Xarray dataset containing data for 2-meter AGL wet-bulb potential temperature. Wet-bulb potential temperature
        has units of kelvin (K).
    """
    L = 2.5 * (10 ** 6)  # Latent heat of vaporization of water (J/kg).
    Rv = 461.5  # Gas constant for water vapor (J/kg/K).
    epsilon = 0.622
    e_knot = 6.11  # hPa
    T_knot = 273.15  # K
    p_knot = 1000  # hPa
    kd = 0.286
    P = P / 100  # Convert surface pressure from Pa to hPa (hectopascals).

    e = e_knot * (np.exp((L / Rv) * ((1 / T_knot) - (1 / Td))))
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
        Xarray dataset containing data for 2-meter AGL air temperature. Air temperature has units of kelvin (K).
    Td: Dataset
        Xarray dataset containing data for 2-meter AGL dew-point temperature. Dew-point temperature has units of kelvin
        (K).
    P: Dataset
        Xarray dataset containing data for surface pressure. Surface pressure has units of pascals (Pa).

    Returns
    -------
    theta_w: Dataset
        Xarray dataset containing data for 2-meter AGL wet-bulb potential temperature. Wet-bulb potential temperature
        has units of kelvin (K).
    """
    L = 2.5 * (10 ** 6)  # Latent heat of vaporization of water (J/kg).
    Rv = 461.5  # Gas constant for water vapor (J/kg/K).
    epsilon = 0.622
    e_knot = 6.11  # hPa
    T_knot = 273.15  # K
    C = 273.15  # K (Davies-Jones 2008, Section 2)
    p_knot = 1000  # hPa
    kd = 0.286
    P = P / 100  # Convert surface pressure from Pa to hPa (hectopascals).

    e = e_knot * (np.exp((L / Rv) * ((1 / T_knot) - (1 / Td))))
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
        Xarray dataset containing data for 2-meter AGL air temperature. Air temperature has units of kelvin (K).
    Td: Dataset
        Xarray dataset containing data for 2-meter AGL dew-point temperature. Dew-point temperature has units of kelvin
        (K).
    P: Dataset
        Xarray dataset containing data for surface pressure. Surface pressure has units of pascals (Pa).

    Returns
    -------
    Tv: Dataset
        Xarray dataset containing data for 2-meter AGL virtual temperature. Virtual temperature has units of kelvin (K).
    """
    L = 2.5 * (10 ** 6)  # Latent heat of vaporization of water (J/kg).
    Rv = 461.5  # Gas constant for water vapor (J/kg/K).
    e_knot = 6.11  # hPa
    T_knot = 273.15  # K
    epsilon = 0.622
    P = P / 100  # Convert surface pressure from Pa to hPa (hectopascals).

    e = e_knot * (np.exp((L / Rv) * ((1 / T_knot) - (1 / Td))))
    r = (epsilon * e) / (P - e)

    Tv = T * (1 + (r / epsilon)) / (1 + r)
    return Tv


def wet_bulb_temperature(T, Td):
    """
    Returns dataset containing wet-bulb temperature data with units of kelvin (K).

    Parameters
    ----------
    T: Dataset
        Xarray dataset containing data for 2-meter AGL air temperature. Air temperature has units of kelvin (K).
    Td: Dataset
        Xarray dataset containing data for 2-meter AGL dew-point temperature. Dew-point temperature has units of kelvin
        (K).

    Returns
    -------
    Tw: Dataset
        Xarray dataset containing data for 2-meter AGL wet-bulb temperature. Wet-bulb temperature has units of kelvin
        (K).
    """
    L = 2.5 * (10 ** 6)  # Latent heat of vaporization of water (J/kg).
    Rv = 461.5  # Gas constant for water vapor (J/kg/K).
    e_knot = 6.11  # hPa
    T_knot = 273.15  # K
    e = e_knot * (np.exp((L / Rv) * ((1 / T_knot) - (1 / Td))))
    es = e_knot * (np.exp((L / Rv) * ((1 / T_knot) - (1 / T))))

    RH = 100 * e / es
    T = T - 273.15  # Convert temperature to celsius
    c1 = 0.151977
    c2 = 8.313659
    c3 = 1.676331
    c4 = 0.00391838
    c5 = 0.023101
    c6 = 4.686035

    # Wet-bulb temperature approximation (Stull 2011, Eq. 1)
    Tw = T * np.arctan(c1 * np.power(RH + c2, 0.5)) + np.arctan(T + RH) - np.arctan(RH - c3) + (
            c4 * np.power(RH, 1.5) * np.arctan(c5 * RH)) - c6 + 273.15

    return Tw
