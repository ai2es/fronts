import pathlib
from datetime import datetime

import dask.array as dsa
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fronts.data import derived, generate
from fronts.utils import BoundingBox

_LEVELS = [1000, 850, 500]
_LAT = np.linspace(45.0, 25.0, 4)
_LON = np.linspace(-110.0, -70.0, 4)
_TIME = pd.date_range("2019-01-01", periods=3, freq="6h")

_ARCO_VARS = {
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "relative_humidity",
    "vertical_velocity",
}


@pytest.fixture
def base_ds() -> xr.Dataset:
    rng = np.random.default_rng(0)
    shape = (len(_TIME), len(_LEVELS), len(_LAT), len(_LON))
    coords = {"time": _TIME, "level": _LEVELS, "latitude": _LAT, "longitude": _LON}
    dims = ["time", "level", "latitude", "longitude"]

    def _da(seed: int, scale: float = 1.0) -> xr.DataArray:
        rng2 = np.random.default_rng(seed)
        return xr.DataArray(rng2.standard_normal(shape).astype(np.float32) * scale, dims=dims, coords=coords)

    return xr.Dataset(
        {
            "temperature": _da(1, 10.0) + 280.0,
            "specific_humidity": xr.DataArray(
                np.abs(rng.standard_normal(shape).astype(np.float32)) * 0.01, dims=dims, coords=coords
            ),
            "u_component_of_wind": _da(3, 15.0),
            "v_component_of_wind": _da(4, 15.0),
            "geopotential": _da(5, 1000.0) + 50000.0,
        }
    )


@pytest.fixture
def physical_ds() -> xr.Dataset:
    """Dataset with physically consistent temperature and specific humidity.

    Specific humidity is kept well below saturation so derived moisture variables
    satisfy physical bounds (dewpoint <= temperature, RH in [0, 1]).
    """
    shape = (len(_TIME), len(_LEVELS), len(_LAT), len(_LON))
    coords = {"time": _TIME, "level": _LEVELS, "latitude": _LAT, "longitude": _LON}
    dims = ["time", "level", "latitude", "longitude"]
    rng = np.random.default_rng(99)

    t = rng.uniform(240.0, 300.0, shape).astype(np.float32)
    # Saturation q at 500 hPa, 240 K ≈ 0.5 g/kg; keep q at ~10% of that to stay
    # clearly subsaturated across all levels and temperatures in this fixture.
    q = rng.uniform(1e-5, 5e-5, shape).astype(np.float32)

    return xr.Dataset(
        {
            "temperature": xr.DataArray(t, dims=dims, coords=coords),
            "specific_humidity": xr.DataArray(q, dims=dims, coords=coords),
            "u_component_of_wind": xr.DataArray(
                rng.standard_normal(shape).astype(np.float32) * 15.0, dims=dims, coords=coords
            ),
            "v_component_of_wind": xr.DataArray(
                rng.standard_normal(shape).astype(np.float32) * 15.0, dims=dims, coords=coords
            ),
            "geopotential": xr.DataArray(
                rng.uniform(40000.0, 60000.0, shape).astype(np.float32), dims=dims, coords=coords
            ),
        }
    )


@pytest.fixture
def arco_zarr(tmp_path: pathlib.Path, base_ds: xr.Dataset) -> pathlib.Path:
    path = tmp_path / "arco.zarr"
    base_ds.to_zarr(path)
    return path


class TestClassifyVariables:
    def test_all_direct(self):
        direct, derived_vars, static_vars = derived.classify_variables(["temperature", "geopotential"], _ARCO_VARS)
        assert direct == ["temperature", "geopotential"]
        assert derived_vars == []
        assert static_vars == []

    def test_derived_only(self):
        direct, derived_vars, static_vars = derived.classify_variables(["wind_speed"], _ARCO_VARS)
        assert direct == []
        assert derived_vars == ["wind_speed"]
        assert static_vars == []

    def test_static_only(self):
        direct, derived_vars, static_vars = derived.classify_variables(["geopotential_at_surface"], _ARCO_VARS)
        assert direct == []
        assert derived_vars == []
        assert static_vars == ["geopotential_at_surface"]

    def test_mixed(self):
        direct, derived_vars, static_vars = derived.classify_variables(
            ["temperature", "wind_speed", "potential_temperature", "geopotential_at_surface"], _ARCO_VARS
        )
        assert direct == ["temperature"]
        assert set(derived_vars) == {"wind_speed", "potential_temperature"}
        assert static_vars == ["geopotential_at_surface"]

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="no derivation or static function"):
            derived.classify_variables(["nonexistent_variable"], _ARCO_VARS)

    def test_unknown_lists_registry_in_error(self):
        with pytest.raises(ValueError, match="wind_speed"):
            derived.classify_variables(["completely_unknown"], _ARCO_VARS)

    def test_unknown_lists_static_registry_in_error(self):
        with pytest.raises(ValueError, match="geopotential_at_surface"):
            derived.classify_variables(["completely_unknown"], _ARCO_VARS)

    def test_order_preserved_for_direct(self):
        requested = ["geopotential", "temperature", "specific_humidity"]
        direct, _, _ = derived.classify_variables(requested, _ARCO_VARS)
        assert direct == requested


class TestResolveStaticVariables:
    @pytest.fixture
    def static_source_zarr(self, make_static_source_zarr) -> pathlib.Path:
        return make_static_source_zarr("geopotential_at_surface", _LAT, _LON, seed=21)

    def test_returns_requested_variable(self, static_source_zarr, monkeypatch):
        monkeypatch.setitem(derived.sources.STATIC_VARIABLE_SOURCES, "geopotential_at_surface", str(static_source_zarr))
        result = derived.resolve_static_variables(["geopotential_at_surface"])
        assert set(result.data_vars) == {"geopotential_at_surface"}
        assert "time" not in result["geopotential_at_surface"].dims

    def test_empty_request_returns_empty_dataset(self):
        result = derived.resolve_static_variables([])
        assert len(result.data_vars) == 0


class TestResolveDownloadVariables:
    def test_no_derived_unchanged(self):
        direct = ["temperature", "geopotential"]
        result = derived.resolve_download_variables(direct, [])
        assert result == direct

    def test_adds_required_inputs_not_in_direct(self):
        direct = ["geopotential"]
        derived_vars = ["wind_speed"]
        result = derived.resolve_download_variables(direct, derived_vars)
        assert "u_component_of_wind" in result
        assert "v_component_of_wind" in result

    def test_does_not_duplicate_if_already_in_direct(self):
        direct = ["u_component_of_wind", "v_component_of_wind"]
        derived_vars = ["wind_speed"]
        result = derived.resolve_download_variables(direct, derived_vars)
        assert result.count("u_component_of_wind") == 1
        assert result.count("v_component_of_wind") == 1

    def test_multiple_derived_share_input(self):
        direct = ["geopotential"]
        derived_vars = ["potential_temperature", "wind_speed"]
        result = derived.resolve_download_variables(direct, derived_vars)
        assert result.count("temperature") == 1


class TestPressurePa:
    def test_lazy_when_input_is_dask(self, base_ds: xr.Dataset):
        chunked = base_ds["temperature"].chunk({"time": 1})
        result = derived._pressure_pa(chunked)
        assert isinstance(result.data, dsa.Array), "pressure must be dask-backed to avoid 361 GiB materialization"

    def test_not_lazy_without_dask_input_is_fine(self, base_ds: xr.Dataset):
        result = derived._pressure_pa(base_ds["temperature"])
        assert not isinstance(result.data, dsa.Array)

    def test_values_correct(self, base_ds: xr.Dataset):
        result = derived._pressure_pa(base_ds["temperature"])
        expected_pa = np.array(_LEVELS) * 100.0
        np.testing.assert_array_equal(result.isel(time=0, latitude=0, longitude=0).values, expected_pa)

    def test_values_correct_when_dask(self, base_ds: xr.Dataset):
        chunked = base_ds["temperature"].chunk({"time": 1})
        result = derived._pressure_pa(chunked)
        expected_pa = np.array(_LEVELS) * 100.0
        np.testing.assert_array_equal(result.isel(time=0, latitude=0, longitude=0).values, expected_pa)


class TestWindSpeedCompute:
    def test_shape_preserved(self, base_ds: xr.Dataset):
        result = derived.DERIVED_VARIABLE_REGISTRY["wind_speed"].compute(
            base_ds["u_component_of_wind"], base_ds["v_component_of_wind"]
        )
        assert result.shape == base_ds["temperature"].shape
        assert result.dims == base_ds["temperature"].dims

    def test_values_correct(self, base_ds: xr.Dataset):
        result = derived.DERIVED_VARIABLE_REGISTRY["wind_speed"].compute(
            base_ds["u_component_of_wind"], base_ds["v_component_of_wind"]
        )
        u = base_ds["u_component_of_wind"].values
        v = base_ds["v_component_of_wind"].values
        expected = np.sqrt(u**2 + v**2)
        np.testing.assert_allclose(result.values, expected, rtol=1e-5)

    def test_non_negative(self, base_ds: xr.Dataset):
        result = derived.DERIVED_VARIABLE_REGISTRY["wind_speed"].compute(
            base_ds["u_component_of_wind"], base_ds["v_component_of_wind"]
        )
        assert np.all(result.values >= 0)


class TestPotentialTemperatureCompute:
    def test_shape_preserved(self, base_ds: xr.Dataset):
        result = derived.DERIVED_VARIABLE_REGISTRY["potential_temperature"].compute(base_ds["temperature"])
        assert result.shape == base_ds["temperature"].shape

    def test_values_reasonable(self, base_ds: xr.Dataset):
        result = derived.DERIVED_VARIABLE_REGISTRY["potential_temperature"].compute(base_ds["temperature"])
        assert np.all(result.values > 150)
        assert np.all(result.values < 500)


class TestEquivalentPotentialTemperatureCompute:
    def test_shape_preserved(self, base_ds: xr.Dataset):
        result = derived.DERIVED_VARIABLE_REGISTRY["equivalent_potential_temperature"].compute(
            base_ds["temperature"], base_ds["specific_humidity"]
        )
        assert result.shape == base_ds["temperature"].shape

    def test_values_warmer_than_potential_temperature(self, base_ds: xr.Dataset):
        theta = derived.DERIVED_VARIABLE_REGISTRY["potential_temperature"].compute(base_ds["temperature"])
        theta_e = derived.DERIVED_VARIABLE_REGISTRY["equivalent_potential_temperature"].compute(
            base_ds["temperature"], base_ds["specific_humidity"]
        )
        assert np.all(theta_e.values >= theta.values)


class TestVirtualTemperatureCompute:
    def test_shape_preserved(self, base_ds: xr.Dataset):
        result = derived.DERIVED_VARIABLE_REGISTRY["virtual_temperature"].compute(
            base_ds["temperature"], base_ds["specific_humidity"]
        )
        assert result.shape == base_ds["temperature"].shape

    def test_warmer_than_actual_temperature(self, base_ds: xr.Dataset):
        result = derived.DERIVED_VARIABLE_REGISTRY["virtual_temperature"].compute(
            base_ds["temperature"], base_ds["specific_humidity"]
        )
        assert np.all(result.values >= base_ds["temperature"].values)


class TestDewpointTemperatureCompute:
    def test_shape_preserved(self, base_ds: xr.Dataset):
        result = derived.DERIVED_VARIABLE_REGISTRY["dewpoint_temperature"].compute(
            base_ds["temperature"], base_ds["specific_humidity"]
        )
        assert result.shape == base_ds["temperature"].shape

    def test_at_or_below_actual_temperature(self, physical_ds: xr.Dataset):
        result = derived.DERIVED_VARIABLE_REGISTRY["dewpoint_temperature"].compute(
            physical_ds["temperature"], physical_ds["specific_humidity"]
        )
        assert np.all(result.values <= physical_ds["temperature"].values + 1e-3)


class TestRelativeHumidityCompute:
    def test_shape_preserved(self, base_ds: xr.Dataset):
        result = derived.DERIVED_VARIABLE_REGISTRY["relative_humidity"].compute(
            base_ds["temperature"], base_ds["specific_humidity"]
        )
        assert result.shape == base_ds["temperature"].shape

    def test_bounded_between_zero_and_one(self, physical_ds: xr.Dataset):
        result = derived.DERIVED_VARIABLE_REGISTRY["relative_humidity"].compute(
            physical_ds["temperature"], physical_ds["specific_humidity"]
        )
        assert np.all(result.values >= 0)
        assert np.all(result.values <= 1.0 + 1e-6)


class TestNonPositiveSpecificHumidity:
    """ERA5 spectral artifacts produce q <= 0; derived variables must stay finite."""

    @pytest.fixture
    def dry_ds(self, physical_ds: xr.Dataset) -> xr.Dataset:
        ds = physical_ds.copy(deep=True)
        q = ds["specific_humidity"].values
        q[0, 0, 0, 0] = 0.0
        q[0, 0, 0, 1] = -1e-7
        q[0, 0, 0, 2] = 1e-12
        return ds

    @pytest.mark.parametrize(
        "variable",
        ["dewpoint_temperature", "equivalent_potential_temperature", "relative_humidity", "virtual_temperature"],
    )
    def test_finite_for_zero_and_negative_humidity(self, dry_ds: xr.Dataset, variable: str):
        result = derived.DERIVED_VARIABLE_REGISTRY[variable].compute(dry_ds["temperature"], dry_ds["specific_humidity"])
        assert np.isfinite(result.values).all()

    def test_clamp_leaves_normal_values_unchanged(self, physical_ds: xr.Dataset):
        q = physical_ds["specific_humidity"]
        np.testing.assert_array_equal(derived._clamped_specific_humidity(q).values, q.values)

    def test_dewpoint_is_very_cold_for_clamped_humidity(self, dry_ds: xr.Dataset):
        result = derived.DERIVED_VARIABLE_REGISTRY["dewpoint_temperature"].compute(
            dry_ds["temperature"], dry_ds["specific_humidity"]
        )
        assert result.values[0, 0, 0, 0] < 200.0

    def test_relative_humidity_non_negative(self, dry_ds: xr.Dataset):
        result = derived.DERIVED_VARIABLE_REGISTRY["relative_humidity"].compute(
            dry_ds["temperature"], dry_ds["specific_humidity"]
        )
        assert (result.values >= 0).all()


class TestGenerateEra5DownloadData:
    def test_excludes_derived_variable_names(self, arco_zarr: pathlib.Path):
        cfg = generate.ERA5DataLoaderConfig(
            era5_uri=str(arco_zarr),
            variables=["wind_speed"],
            pressure_levels=_LEVELS,
            time_start=datetime(2019, 1, 1),
            time_end=datetime(2019, 1, 1, 12),
            time_resolution="6h",
            coordinates=BoundingBox(lat_min=25.0, lat_max=45.0, lon_min=-110.0, lon_max=-70.0),
            storage_options=None,
            zarr_async_concurrency=10,
            chunks={"time": 1},
        )
        ds = generate.generate_era5_download_data(cfg)
        assert "wind_speed" not in ds.data_vars
        assert "u_component_of_wind" in ds.data_vars
        assert "v_component_of_wind" in ds.data_vars

    def test_direct_and_derived_mix(self, arco_zarr: pathlib.Path):
        cfg = generate.ERA5DataLoaderConfig(
            era5_uri=str(arco_zarr),
            variables=["temperature", "wind_speed"],
            pressure_levels=_LEVELS,
            time_start=datetime(2019, 1, 1),
            time_end=datetime(2019, 1, 1, 12),
            time_resolution="6h",
            coordinates=BoundingBox(lat_min=25.0, lat_max=45.0, lon_min=-110.0, lon_max=-70.0),
            storage_options=None,
            zarr_async_concurrency=10,
            chunks={"time": 1},
        )
        ds = generate.generate_era5_download_data(cfg)
        assert "wind_speed" not in ds.data_vars
        assert set(ds.data_vars) == {"temperature", "u_component_of_wind", "v_component_of_wind"}


class TestGenerateEra5DerivedData:
    def test_returns_only_requested_derived_vars_lazily(self, physical_ds: xr.Dataset):
        source_ds = physical_ds.chunk({"time": 1})
        result = generate.generate_era5_derived_data(["wind_speed", "potential_temperature"], source_ds)

        assert set(result.data_vars) == {"wind_speed", "potential_temperature"}
        assert isinstance(result["wind_speed"].data, dsa.Array)

        expected_wind_speed = derived.DERIVED_VARIABLE_REGISTRY["wind_speed"].compute(
            physical_ds["u_component_of_wind"], physical_ds["v_component_of_wind"]
        )
        np.testing.assert_allclose(result["wind_speed"].compute().values, expected_wind_speed.values, rtol=1e-5)

    def test_returns_empty_dataset_when_no_derived_vars_requested(self, physical_ds: xr.Dataset):
        result = generate.generate_era5_derived_data([], physical_ds.chunk({"time": 1}))
        assert len(result.data_vars) == 0


class TestGenerateEra5DataWithDerived:
    def test_derived_variable_in_output(self, arco_zarr: pathlib.Path):
        cfg = generate.ERA5DataLoaderConfig(
            era5_uri=str(arco_zarr),
            variables=["temperature", "wind_speed"],
            pressure_levels=_LEVELS,
            time_start=datetime(2019, 1, 1),
            time_end=datetime(2019, 1, 1, 12),
            time_resolution="6h",
            coordinates=BoundingBox(lat_min=25.0, lat_max=45.0, lon_min=-110.0, lon_max=-70.0),
            storage_options=None,
            zarr_async_concurrency=10,
            chunks={"time": 1},
        )
        ds = generate.generate_era5_data(cfg)
        assert "wind_speed" in ds.data_vars
        assert "temperature" in ds.data_vars

    def test_intermediate_inputs_dropped(self, arco_zarr: pathlib.Path):
        cfg = generate.ERA5DataLoaderConfig(
            era5_uri=str(arco_zarr),
            variables=["wind_speed"],
            pressure_levels=_LEVELS,
            time_start=datetime(2019, 1, 1),
            time_end=datetime(2019, 1, 1, 12),
            time_resolution="6h",
            coordinates=BoundingBox(lat_min=25.0, lat_max=45.0, lon_min=-110.0, lon_max=-70.0),
            storage_options=None,
            zarr_async_concurrency=10,
            chunks={"time": 1},
        )
        ds = generate.generate_era5_data(cfg)
        assert "wind_speed" in ds.data_vars
        assert "u_component_of_wind" not in ds.data_vars
        assert "v_component_of_wind" not in ds.data_vars

    def test_unknown_variable_raises(self, arco_zarr: pathlib.Path):
        cfg = generate.ERA5DataLoaderConfig(
            era5_uri=str(arco_zarr),
            variables=["nonexistent_variable"],
            pressure_levels=_LEVELS,
            time_start=datetime(2019, 1, 1),
            time_end=datetime(2019, 1, 1, 12),
            time_resolution="6h",
            coordinates=BoundingBox(lat_min=25.0, lat_max=45.0, lon_min=-110.0, lon_max=-70.0),
            storage_options=None,
            zarr_async_concurrency=10,
            chunks={"time": 1},
        )
        with pytest.raises(ValueError, match="no derivation or static function"):
            generate.generate_era5_data(cfg)

    def test_explicit_input_also_stored(self, arco_zarr: pathlib.Path):
        cfg = generate.ERA5DataLoaderConfig(
            era5_uri=str(arco_zarr),
            variables=["u_component_of_wind", "v_component_of_wind", "wind_speed"],
            pressure_levels=_LEVELS,
            time_start=datetime(2019, 1, 1),
            time_end=datetime(2019, 1, 1, 12),
            time_resolution="6h",
            coordinates=BoundingBox(lat_min=25.0, lat_max=45.0, lon_min=-110.0, lon_max=-70.0),
            storage_options=None,
            zarr_async_concurrency=10,
            chunks={"time": 1},
        )
        ds = generate.generate_era5_data(cfg)
        assert "u_component_of_wind" in ds.data_vars
        assert "v_component_of_wind" in ds.data_vars
        assert "wind_speed" in ds.data_vars

    def test_equals_download_plus_derived(self, arco_zarr: pathlib.Path):
        cfg = generate.ERA5DataLoaderConfig(
            era5_uri=str(arco_zarr),
            variables=["temperature", "wind_speed"],
            pressure_levels=_LEVELS,
            time_start=datetime(2019, 1, 1),
            time_end=datetime(2019, 1, 1, 12),
            time_resolution="6h",
            coordinates=BoundingBox(lat_min=25.0, lat_max=45.0, lon_min=-110.0, lon_max=-70.0),
            storage_options=None,
            zarr_async_concurrency=10,
            chunks={"time": 1},
        )
        ds_download = generate.generate_era5_download_data(cfg)
        ds_derived = generate.generate_era5_derived_data(["wind_speed"], ds_download)
        expected = ds_download.assign({str(name): ds_derived[name] for name in ds_derived.data_vars})
        expected = expected.drop_vars([v for v in expected.data_vars if v not in cfg.variables])

        result = generate.generate_era5_data(cfg)
        xr.testing.assert_allclose(result.compute(), expected.compute())
