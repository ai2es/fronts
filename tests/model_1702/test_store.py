"""Tests for the model_1702 side-store derivation."""

import numpy as np
import pytest
import xarray as xr

from fronts.model_1702 import legacy_formulas, store

N_TIME = 2
N_LAT = 3
N_LON = 4


@pytest.fixture
def synthetic_source():
    rng = np.random.default_rng(9)
    times = np.array(["2019-01-01T00", "2019-01-01T03"], dtype="datetime64[ns]")
    lats = np.array([30.0, 30.25, 30.5])
    lons = np.array([250.0, 250.25, 250.5, 250.75])
    levels = np.array(store.PRESSURE_LEVELS_HPA)

    def pressure_field(low, high):
        return (("time", "level", "latitude", "longitude"), rng.uniform(low, high, (N_TIME, len(levels), N_LAT, N_LON)))

    def single_field(low, high):
        return (("time", "latitude", "longitude"), rng.uniform(low, high, (N_TIME, N_LAT, N_LON)))

    temperature = rng.uniform(250.0, 300.0, (N_TIME, len(levels), N_LAT, N_LON))
    target_rh = rng.uniform(0.2, 0.95, temperature.shape)
    pressure_pa = (levels * 100.0)[np.newaxis, :, np.newaxis, np.newaxis]
    saturation_vapor_pressure = legacy_formulas.vapor_pressure(temperature)
    vapor_pres = target_rh * saturation_vapor_pressure
    specific_humidity = legacy_formulas.EPSILON * vapor_pres / (pressure_pa - (0.378 * vapor_pres))

    t2m = rng.uniform(260.0, 305.0, (N_TIME, N_LAT, N_LON))
    d2m = t2m - rng.uniform(0.5, 15.0, t2m.shape)

    return xr.Dataset(
        {
            "geopotential": pressure_field(500.0, 15000.0),
            "temperature": (("time", "level", "latitude", "longitude"), temperature),
            "u_component_of_wind": pressure_field(-30.0, 30.0),
            "v_component_of_wind": pressure_field(-30.0, 30.0),
            "specific_humidity": (("time", "level", "latitude", "longitude"), specific_humidity),
            "surface_pressure": single_field(80000.0, 103000.0),
            "2m_temperature": (("time", "latitude", "longitude"), t2m),
            "2m_dewpoint_temperature": (("time", "latitude", "longitude"), d2m),
            "10m_u_component_of_wind": single_field(-20.0, 20.0),
            "10m_v_component_of_wind": single_field(-20.0, 20.0),
        },
        coords={"time": times, "level": levels, "latitude": lats, "longitude": lons},
    )


class TestBuild1702Dataset:
    def test_variables_levels_and_dtype(self, synthetic_source):
        built = store.build_1702_dataset(synthetic_source)
        assert list(built.data_vars) == ["T", "Td", "Tv", "u", "v", "r", "q", "RH", "sp_z", "theta_e"]
        assert built["level"].values.tolist() == [1013, 1000, 950, 900, 850]
        for variable in built.data_vars:
            assert built[variable].dims == ("time", "level", "latitude", "longitude")
            assert built[variable].dtype == np.float32

    def test_latitude_descending(self, synthetic_source):
        built = store.build_1702_dataset(synthetic_source)
        assert (np.diff(built["latitude"].values) < 0).all()

    def test_surface_level_fields(self, synthetic_source):
        built = store.build_1702_dataset(synthetic_source)
        source = synthetic_source.sortby("latitude", ascending=False)
        surface = built.sel(level=1013)
        np.testing.assert_allclose(surface["T"].values, source["2m_temperature"].values, rtol=1e-6)
        np.testing.assert_allclose(surface["Td"].values, source["2m_dewpoint_temperature"].values, rtol=1e-6)
        np.testing.assert_allclose(surface["u"].values, source["10m_u_component_of_wind"].values, rtol=1e-6)
        np.testing.assert_allclose(surface["sp_z"].values, source["surface_pressure"].values / 100.0, rtol=1e-6)

    def test_pressure_level_sp_z_is_geopotential_in_dam(self, synthetic_source):
        built = store.build_1702_dataset(synthetic_source)
        source = synthetic_source.sortby("latitude", ascending=False)
        np.testing.assert_allclose(
            built["sp_z"].sel(level=850).values,
            source["geopotential"].sel(level=850).values / 98.0665,
            rtol=1e-6,
        )

    def test_humidity_scaled_to_g_per_kg(self, synthetic_source):
        built = store.build_1702_dataset(synthetic_source)
        source = synthetic_source.sortby("latitude", ascending=False)
        np.testing.assert_allclose(
            built["q"].sel(level=1000).values,
            source["specific_humidity"].sel(level=1000).values * 1000.0,
            rtol=1e-6,
        )
        surface_q = legacy_formulas.specific_humidity_from_dewpoint(
            source["2m_dewpoint_temperature"].values, source["surface_pressure"].values
        )
        np.testing.assert_allclose(built["q"].sel(level=1013).values, surface_q * 1000.0, rtol=1e-6)

    def test_derived_values_match_legacy_formulas(self, synthetic_source):
        built = store.build_1702_dataset(synthetic_source)
        source = synthetic_source.sortby("latitude", ascending=False)
        temperature = source["temperature"].sel(level=900).values
        specific_humidity = source["specific_humidity"].sel(level=900).values
        dewpoint = legacy_formulas.dewpoint_from_specific_humidity(90000.0, temperature, specific_humidity)
        np.testing.assert_allclose(built["Td"].sel(level=900).values, dewpoint, rtol=1e-6)
        np.testing.assert_allclose(
            built["RH"].sel(level=900).values,
            legacy_formulas.relative_humidity(temperature, dewpoint),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            built["theta_e"].sel(level=900).values,
            legacy_formulas.equivalent_potential_temperature(temperature, dewpoint, 90000.0),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            built["Tv"].sel(level=900).values,
            legacy_formulas.virtual_temperature_from_dewpoint(temperature, dewpoint, 90000.0),
            rtol=1e-6,
        )

    def test_relative_humidity_physically_plausible(self, synthetic_source):
        built = store.build_1702_dataset(synthetic_source)
        rh = built["RH"].values
        assert (rh > 0.0).all()
        assert (rh < 1.5).all()


def test_arraylake_single_map_extends_core_without_mutation():
    from fronts.data import sources

    assert store.GOOGLE_TO_ARRAYLAKE_SINGLE_1702["surface_pressure"] == "sp"
    assert "surface_pressure" not in sources.GOOGLE_TO_ARRAYLAKE_SINGLE
    core_items = sources.GOOGLE_TO_ARRAYLAKE_SINGLE.items()
    assert all(item in store.GOOGLE_TO_ARRAYLAKE_SINGLE_1702.items() for item in core_items)
