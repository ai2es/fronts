import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fronts.data import sources

_TIME = pd.date_range("2019-01-01", periods=4, freq="6h")
_LEVELS = [1000, 850, 500]
_LAT = np.linspace(45.0, 25.0, 4)
_LON = np.linspace(200.0, 240.0, 4)


@pytest.fixture
def arraylake_pressure_ds() -> xr.Dataset:
    rng = np.random.default_rng(0)
    shape = (len(_TIME), len(_LEVELS), len(_LAT), len(_LON))
    coords = {"valid_time": _TIME, "pressure_level": _LEVELS, "latitude": _LAT, "longitude": _LON}
    dims = ["valid_time", "pressure_level", "latitude", "longitude"]
    return xr.Dataset(
        {
            short: xr.DataArray(rng.standard_normal(shape).astype(np.float32), dims=dims, coords=coords)
            for short in sources.GOOGLE_TO_ARRAYLAKE_PRESSURE.values()
        }
    )


@pytest.fixture
def arraylake_single_ds() -> xr.Dataset:
    rng = np.random.default_rng(1)
    shape = (len(_TIME), len(_LAT), len(_LON))
    coords = {"valid_time": _TIME, "latitude": _LAT, "longitude": _LON}
    dims = ["valid_time", "latitude", "longitude"]
    return xr.Dataset(
        {
            short: xr.DataArray(rng.standard_normal(shape).astype(np.float32), dims=dims, coords=coords)
            for short in sources.GOOGLE_TO_ARRAYLAKE_SINGLE.values()
        }
    )


class TestParseArraylakeRepo:
    def test_strips_prefix(self):
        assert sources.parse_arraylake_repo("arraylake://earthmover-public/era5") == "earthmover-public/era5"

    def test_rejects_non_arraylake_uri(self):
        with pytest.raises(ValueError, match="Not an arraylake URI"):
            sources.parse_arraylake_repo("gs://some-bucket/era5.zarr")


class TestVariableMappings:
    def test_pressure_mapping_contains_required_variables(self):
        required = {
            "geopotential",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "specific_humidity",
            "vertical_velocity",
            "potential_vorticity",
        }
        assert required <= set(sources.GOOGLE_TO_ARRAYLAKE_PRESSURE)

    def test_single_mapping_contains_required_variables(self):
        required = {
            "mean_sea_level_pressure",
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
        }
        assert required <= set(sources.GOOGLE_TO_ARRAYLAKE_SINGLE)

    def test_potential_vorticity_maps_to_pv(self):
        assert sources.GOOGLE_TO_ARRAYLAKE_PRESSURE["potential_vorticity"] == "pv"

    def test_land_sea_mask_not_in_arraylake_mapping(self):
        # land_sea_mask ("lsm") is absent from the live Arraylake single/spatial and
        # single/temporal groups; it is sourced from STATIC_VARIABLE_SOURCES instead
        # (see TestStaticVariableSources) rather than GOOGLE_TO_ARRAYLAKE_SINGLE.
        assert "land_sea_mask" not in sources.GOOGLE_TO_ARRAYLAKE_SINGLE


class TestRenameToGoogle:
    def test_pressure_variables_and_coords_renamed(self, arraylake_pressure_ds: xr.Dataset):
        result = sources._rename_to_google(arraylake_pressure_ds, sources.GOOGLE_TO_ARRAYLAKE_PRESSURE)
        assert set(result.data_vars) == set(sources.GOOGLE_TO_ARRAYLAKE_PRESSURE)
        assert "time" in result.dims
        assert "level" in result.dims
        assert "valid_time" not in result.dims
        assert "pressure_level" not in result.dims

    def test_single_variables_renamed_without_level(self, arraylake_single_ds: xr.Dataset):
        result = sources._rename_to_google(arraylake_single_ds, sources.GOOGLE_TO_ARRAYLAKE_SINGLE)
        assert set(result.data_vars) == set(sources.GOOGLE_TO_ARRAYLAKE_SINGLE)
        assert "time" in result.dims
        assert "level" not in result.dims

    def test_data_unchanged_by_rename(self, arraylake_pressure_ds: xr.Dataset):
        result = sources._rename_to_google(arraylake_pressure_ds, sources.GOOGLE_TO_ARRAYLAKE_PRESSURE)
        np.testing.assert_array_equal(result["temperature"].values, arraylake_pressure_ds["t"].values)

    def test_partial_dataset_renamed(self, arraylake_pressure_ds: xr.Dataset):
        result = sources._rename_to_google(arraylake_pressure_ds[["t", "q"]], sources.GOOGLE_TO_ARRAYLAKE_PRESSURE)
        assert set(result.data_vars) == {"temperature", "specific_humidity"}


class TestMergedGroups:
    def test_pressure_and_single_merge_shares_time(self, arraylake_pressure_ds, arraylake_single_ds):
        pressure = sources._rename_to_google(arraylake_pressure_ds, sources.GOOGLE_TO_ARRAYLAKE_PRESSURE)
        single = sources._rename_to_google(arraylake_single_ds, sources.GOOGLE_TO_ARRAYLAKE_SINGLE)
        merged = xr.merge([pressure, single], join="inner", compat="override")
        assert merged.sizes["time"] == len(_TIME)
        assert merged["temperature"].dims == ("time", "level", "latitude", "longitude")
        assert merged["2m_temperature"].dims == ("time", "latitude", "longitude")


class TestOpenArraylakeEra5Validation:
    def test_unknown_variable_raises(self):
        with pytest.raises(ValueError, match="no known Arraylake mapping"):
            sources.open_arraylake_era5("arraylake://earthmover-public/era5", ["not_a_variable"])


class _FakeSession:
    def __init__(self, store):
        self.store = store


class _FakeRepo:
    def __init__(self, store):
        self._store = store

    def readonly_session(self, branch: str) -> _FakeSession:
        return _FakeSession(self._store)


class _FakeClient:
    def __init__(self, store):
        self._store = store

    def get_repo(self, repo_name: str) -> _FakeRepo:
        return _FakeRepo(self._store)


class TestOpenArraylakeEra5StaleMapping:
    """Reproduces the incident where a mapped short name absent from the live store.

    Raises loudly instead of silently returning an empty dataset (which surfaced
    downstream as a confusing icechunk "no changes to commit" error).
    """

    @pytest.fixture
    def incomplete_single_store(self, tmp_path: pathlib.Path) -> pathlib.Path:
        rng = np.random.default_rng(9)
        ds = xr.Dataset(
            {
                "t2m": xr.DataArray(
                    rng.standard_normal((len(_TIME), len(_LAT), len(_LON))).astype(np.float32),
                    dims=["valid_time", "latitude", "longitude"],
                    coords={"valid_time": _TIME, "latitude": _LAT, "longitude": _LON},
                )
            }
        )
        path = tmp_path / "fake_repo"
        ds.to_zarr(path, group=sources.ARRAYLAKE_SINGLE_GROUP)
        return path

    def test_missing_short_name_raises(self, incomplete_single_store: pathlib.Path, monkeypatch):
        monkeypatch.setattr(sources.arraylake, "Client", lambda: _FakeClient(str(incomplete_single_store)))
        with pytest.raises(ValueError, match="absent from group"):
            sources.open_arraylake_era5("arraylake://fake/repo", ["mean_sea_level_pressure"])

    def test_present_short_name_does_not_raise(self, incomplete_single_store: pathlib.Path, monkeypatch):
        monkeypatch.setattr(sources.arraylake, "Client", lambda: _FakeClient(str(incomplete_single_store)))
        result = sources.open_arraylake_era5("arraylake://fake/repo", ["2m_temperature"])
        assert "2m_temperature" in result.data_vars


class TestStaticVariableSources:
    def test_geopotential_at_surface_registered(self):
        assert "geopotential_at_surface" in sources.STATIC_VARIABLE_SOURCES

    def test_land_sea_mask_registered(self):
        assert "land_sea_mask" in sources.STATIC_VARIABLE_SOURCES

    def test_unregistered_variable_raises(self):
        with pytest.raises(ValueError, match="No static source registered"):
            sources.open_static_era5_variable("not_a_variable")

    def test_unregistered_variable_error_lists_registry(self):
        with pytest.raises(ValueError, match="geopotential_at_surface"):
            sources.open_static_era5_variable("not_a_variable")


class TestOpenStaticEra5Variable:
    @pytest.fixture
    def static_source_zarr(self, make_static_source_zarr) -> pathlib.Path:
        time = pd.date_range("2019-01-01", periods=3, freq="6h")
        return make_static_source_zarr("geopotential_at_surface", _LAT, _LON, time=time, seed=5)

    def test_drops_time_dimension(self, static_source_zarr, monkeypatch):
        monkeypatch.setitem(sources.STATIC_VARIABLE_SOURCES, "geopotential_at_surface", str(static_source_zarr))
        result = sources.open_static_era5_variable("geopotential_at_surface")
        assert "time" not in result.dims
        assert result.dims == ("latitude", "longitude")

    def test_values_match_first_timestep(self, static_source_zarr, monkeypatch):
        monkeypatch.setitem(sources.STATIC_VARIABLE_SOURCES, "geopotential_at_surface", str(static_source_zarr))
        result = sources.open_static_era5_variable("geopotential_at_surface")
        expected = xr.open_zarr(static_source_zarr, chunks=None)["geopotential_at_surface"].isel(time=0)
        np.testing.assert_array_equal(result.values, expected.values)
