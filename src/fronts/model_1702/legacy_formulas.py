"""Verbatim numpy port of the legacy AIES thermodynamic formulas.

Source: ``utils/variables.py`` at git commit ``c59ac2b`` (the code that generated model_1702's
training inputs). The formulas and constants are preserved exactly — including the
``dewpoint_from_specific_humidity`` branch on ``T >= 0`` with T in kelvin, whose "warm" branch
always wins — because bit-parity with what model_1702 was trained on matters more than
meteorological refinement. Do not "fix" these against ``fronts.data.derived``, which uses
different constants (e.g. Lv 2.501e6 vs 2.257e6 here) and Bolton-style formulations.

References:
    * Bolton 1980: https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    * Vaisala 2013: https://www.vaisala.com/sites/default/files/documents/Humidity_Conversion_Formulas_B210973EN-F.pdf
"""

import numpy as np

RD = 287.04
RV = 461.5
CPD = 1005.7
KD = RD / CPD
EPSILON = RD / RV
E_KNOT = 611.2
LV = 2.257e6


def vapor_pressure(dewpoint: np.ndarray | float) -> np.ndarray | float:
    """Calculates vapor pressure from dewpoint temperature.

    Args:
        dewpoint: Dewpoint temperature in kelvin.

    Returns:
        Vapor pressure in pascals.
    """
    return E_KNOT * np.exp((LV / RV) * ((1 / 273.15) - (1 / dewpoint)))


def dewpoint_from_specific_humidity(
    pressure: np.ndarray | float, temperature: np.ndarray | float, specific_humidity: np.ndarray | float
) -> np.ndarray | float:
    """Calculates dewpoint temperature from pressure, temperature, and specific humidity.

    Preserves the legacy quirk: the Vaisala warm/cold branch selection compares temperature in
    kelvin against 0, so the warm branch is always selected.

    Args:
        pressure: Air pressure in pascals.
        temperature: Air temperature in kelvin.
        specific_humidity: Specific humidity in kg/kg.

    Returns:
        Dewpoint temperature in kelvin.
    """
    m = 7.591386
    a = 6.116441
    tn = 240.7263
    m1 = 9.778707
    a1 = 6.114742
    tn1 = 273.1466

    vap_pres = (pressure * specific_humidity) / (0.622 + 0.378 * specific_humidity)
    vap_pres /= 100

    dewpoint_warm = tn / ((m / (np.log10(vap_pres / a))) - 1)
    dewpoint_cold = tn1 / ((m1 / (np.log10(vap_pres / a1))) - 1)
    dewpoint = np.where(temperature >= 0, dewpoint_warm, dewpoint_cold)
    return dewpoint + 273.15


def mixing_ratio_from_dewpoint(dewpoint: np.ndarray | float, pressure: np.ndarray | float) -> np.ndarray | float:
    """Calculates mixing ratio from dewpoint temperature and air pressure.

    Args:
        dewpoint: Dewpoint temperature in kelvin.
        pressure: Air pressure in pascals.

    Returns:
        Mixing ratio in kg/kg.
    """
    e = vapor_pressure(dewpoint)
    return EPSILON * e / (pressure - e)


def specific_humidity_from_dewpoint(dewpoint: np.ndarray | float, pressure: np.ndarray | float) -> np.ndarray | float:
    """Calculates specific humidity from dewpoint temperature and air pressure.

    Args:
        dewpoint: Dewpoint temperature in kelvin.
        pressure: Air pressure in pascals.

    Returns:
        Specific humidity in kg/kg.
    """
    e = vapor_pressure(dewpoint)
    return EPSILON * e / (pressure - (0.378 * e))


def relative_humidity(temperature: np.ndarray | float, dewpoint: np.ndarray | float) -> np.ndarray | float:
    """Calculates relative humidity from temperature and dewpoint temperature.

    Args:
        temperature: Air temperature in kelvin.
        dewpoint: Dewpoint temperature in kelvin.

    Returns:
        Relative humidity as a 0-1 fraction.
    """
    return vapor_pressure(dewpoint) / vapor_pressure(temperature)


def potential_temperature(temperature: np.ndarray | float, pressure: np.ndarray | float) -> np.ndarray | float:
    """Calculates potential temperature from temperature and pressure.

    Args:
        temperature: Air temperature in kelvin.
        pressure: Air pressure in pascals.

    Returns:
        Potential temperature in kelvin.
    """
    return temperature * np.power(1e5 / pressure, KD)


def equivalent_potential_temperature(
    temperature: np.ndarray | float, dewpoint: np.ndarray | float, pressure: np.ndarray | float
) -> np.ndarray | float:
    """Calculates equivalent potential temperature from temperature, dewpoint, and pressure.

    Args:
        temperature: Air temperature in kelvin.
        dewpoint: Dewpoint temperature in kelvin.
        pressure: Air pressure in pascals.

    Returns:
        Equivalent potential temperature in kelvin.
    """
    rh = relative_humidity(temperature, dewpoint)
    theta = potential_temperature(temperature, pressure)
    rv = mixing_ratio_from_dewpoint(dewpoint, pressure)
    return theta * np.power(rh, -rv * RV / CPD) * np.exp(LV * rv / (CPD * temperature))


def virtual_temperature_from_mixing_ratio(
    temperature: np.ndarray | float, mixing_ratio: np.ndarray | float
) -> np.ndarray | float:
    """Calculates virtual temperature from temperature and mixing ratio.

    Args:
        temperature: Air temperature in kelvin.
        mixing_ratio: Mixing ratio in kg/kg.

    Returns:
        Virtual temperature in kelvin.
    """
    return temperature * (1 + (mixing_ratio / EPSILON)) / (1 + mixing_ratio)


def virtual_temperature_from_dewpoint(
    temperature: np.ndarray | float, dewpoint: np.ndarray | float, pressure: np.ndarray | float
) -> np.ndarray | float:
    """Calculates virtual temperature from temperature, dewpoint, and pressure.

    Args:
        temperature: Air temperature in kelvin.
        dewpoint: Dewpoint temperature in kelvin.
        pressure: Air pressure in pascals.

    Returns:
        Virtual temperature in kelvin.
    """
    return virtual_temperature_from_mixing_ratio(temperature, mixing_ratio_from_dewpoint(dewpoint, pressure))
