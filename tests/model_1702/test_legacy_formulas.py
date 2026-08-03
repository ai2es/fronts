"""Golden-value tests for the legacy AIES formula port.

Every scalar and array golden value below is copied verbatim from the worked docstring examples
in ``utils/variables.py`` at git commit ``c59ac2b`` — the code that produced model_1702's
training inputs.
"""

import numpy as np
import pytest

from fronts.data import derived
from fronts.model_1702 import legacy_formulas


def test_vapor_pressure_scalar():
    assert legacy_formulas.vapor_pressure(290.0) == pytest.approx(1729.7443936886634, rel=1e-12)


def test_vapor_pressure_array():
    dewpoint = np.arange(260, 301, 5, dtype=float)
    expected = np.array(
        [
            247.12075845,
            352.40493817,
            495.98223586,
            689.43450819,
            947.13483326,
            1286.74161001,
            1729.74439369,
            2302.0614118,
            3034.68799059,
        ]
    )
    np.testing.assert_allclose(legacy_formulas.vapor_pressure(dewpoint), expected, rtol=1e-8)


def test_dewpoint_from_specific_humidity_scalar():
    assert legacy_formulas.dewpoint_from_specific_humidity(1e5, 300.0, 0.02) == pytest.approx(
        298.199585429495, rel=1e-12
    )


def test_dewpoint_from_specific_humidity_array():
    pressure = np.arange(800, 1001, 25, dtype=float) * 100
    specific_humidity = np.arange(5, 25.01, 2.5) / 1000
    temperature = np.arange(270, 311, 5, dtype=float)
    expected = np.array(
        [
            273.80033167,
            279.97353235,
            284.66312436,
            288.51045188,
            291.80950762,
            294.72063163,
            297.34130998,
            299.7354364,
            301.94726732,
        ]
    )
    result = legacy_formulas.dewpoint_from_specific_humidity(pressure, temperature, specific_humidity)
    np.testing.assert_allclose(result, expected, rtol=1e-8)


def test_dewpoint_warm_branch_always_wins_for_kelvin_input():
    cold_kelvin = np.array([200.0, 250.0])
    pressure = np.array([1e5, 1e5])
    specific_humidity = np.array([0.001, 0.001])
    result = legacy_formulas.dewpoint_from_specific_humidity(pressure, cold_kelvin, specific_humidity)
    m = 7.591386
    a = 6.116441
    tn = 240.7263
    vap_pres = (pressure * specific_humidity) / (0.622 + 0.378 * specific_humidity) / 100
    warm_branch = tn / ((m / (np.log10(vap_pres / a))) - 1) + 273.15
    np.testing.assert_allclose(result, warm_branch, rtol=1e-12)


def test_mixing_ratio_from_dewpoint_scalar():
    assert legacy_formulas.mixing_ratio_from_dewpoint(290.0, 1e5) == pytest.approx(0.010947893449979635, rel=1e-12)


def test_mixing_ratio_from_dewpoint_array():
    dewpoint = np.arange(260, 301, 5, dtype=float)
    pressure = np.arange(800, 1001, 25, dtype=float) * 100
    expected = np.array(
        [
            0.00192723,
            0.0026682,
            0.00365056,
            0.00493959,
            0.00661507,
            0.00877413,
            0.01153478,
            0.01504042,
            0.01946563,
        ]
    )
    np.testing.assert_allclose(legacy_formulas.mixing_ratio_from_dewpoint(dewpoint, pressure), expected, rtol=1e-5)


def test_specific_humidity_from_dewpoint_scalar():
    assert legacy_formulas.specific_humidity_from_dewpoint(290.0, 1e5) == pytest.approx(0.010829329732443743, rel=1e-12)


def test_relative_humidity_scalar():
    assert legacy_formulas.relative_humidity(300.0, 290.0) == pytest.approx(0.5699908521249278, rel=1e-12)


def test_relative_humidity_array():
    temperature = np.arange(270, 311, 5, dtype=float)
    dewpoint = np.arange(260, 301, 5, dtype=float)
    expected = np.array(
        [
            0.49824518,
            0.51115071,
            0.52366592,
            0.53579872,
            0.54755768,
            0.5589519,
            0.56999085,
            0.58068426,
            0.591042,
        ]
    )
    np.testing.assert_allclose(legacy_formulas.relative_humidity(temperature, dewpoint), expected, rtol=1e-7)


def test_potential_temperature_scalar():
    assert legacy_formulas.potential_temperature(275.0, 9e4) == pytest.approx(283.3951954331142, rel=1e-12)


def test_equivalent_potential_temperature_scalar():
    assert legacy_formulas.equivalent_potential_temperature(300.0, 290.0, 1e5) == pytest.approx(
        326.52430009577137, rel=1e-12
    )


def test_equivalent_potential_temperature_array():
    temperature = np.arange(270, 311, 5, dtype=float)
    dewpoint = np.arange(260, 301, 5, dtype=float)
    pressure = np.arange(800, 1001, 25, dtype=float) * 100
    expected = np.array(
        [
            292.582033,
            297.1606042,
            302.3295548,
            308.2501438,
            315.12588597,
            323.21500183,
            332.84798857,
            344.45298499,
            358.59322677,
        ]
    )
    result = legacy_formulas.equivalent_potential_temperature(temperature, dewpoint, pressure)
    np.testing.assert_allclose(result, expected, rtol=1e-8)


def test_virtual_temperature_from_mixing_ratio_scalar():
    assert legacy_formulas.virtual_temperature_from_mixing_ratio(300.0, 0.02) == pytest.approx(
        303.5752344416027, rel=1e-12
    )


def test_virtual_temperature_from_dewpoint_scalar():
    assert legacy_formulas.virtual_temperature_from_dewpoint(300.0, 290.0, 1e5) == pytest.approx(
        301.9745879930382, rel=1e-12
    )


def test_legacy_constants_diverge_from_current_derived_module():
    assert legacy_formulas.LV != derived._L_V
    assert legacy_formulas.CPD != derived._C_PD
    assert legacy_formulas.RD != derived._R_D
    assert legacy_formulas.EPSILON != derived._EPSILON
