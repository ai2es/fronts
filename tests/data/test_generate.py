import pathlib
from datetime import datetime

import dacite
import icechunk as ic
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from fronts.data import config, generate
from fronts.utils import BoundingBox

ERA5_VARS = [
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
]
LEVELS = [1000, 950, 900, 850, 700, 500]
LAT = np.linspace(45.0, 25.0, 8)  # descending (N→S) to match ERA5 convention
LON = np.linspace(-110.0, -70.0, 8)

YAML_PATH = pathlib.Path(__file__).parent / "test_generate.yaml"


@pytest.fixture
def time_range() -> pd.DatetimeIndex:
    return pd.date_range("2019-01-01", periods=4, freq="6h")


@pytest.fixture
def minimal_ds(time_range: pd.DatetimeIndex) -> xr.Dataset:
    rng = np.random.default_rng(0)
    data_vars = {
        var: xr.DataArray(
            rng.standard_normal((4, 6, 8, 8)).astype(np.float32),
            dims=["time", "level", "latitude", "longitude"],
            coords={
                "time": time_range,
                "level": LEVELS,
                "latitude": LAT,
                "longitude": LON,
            },
        )
        for var in ERA5_VARS
    }
    return xr.Dataset(data_vars)


@pytest.fixture
def era5_zarr(tmp_path: pathlib.Path, minimal_ds: xr.Dataset) -> pathlib.Path:
    path = tmp_path / "era5.zarr"
    minimal_ds.to_zarr(path)
    return path


@pytest.fixture
def era5_config(era5_zarr: pathlib.Path) -> config.ERA5DataLoaderConfig:
    return config.ERA5DataLoaderConfig(
        era5_uri=str(era5_zarr),
        variables=ERA5_VARS,
        pressure_levels=LEVELS,
        time_start=datetime(2019, 1, 1),
        time_end=datetime(2019, 1, 1, 18),
        time_resolution="6h",
        coordinates=BoundingBox(lat_min=25.0, lat_max=45.0, lon_min=-110.0, lon_max=-70.0),
        storage_options=None,
        chunks={"time": 1},
    )


@pytest.fixture
def storage_config(tmp_path: pathlib.Path) -> config.IcechunkStorageConfig:
    return config.IcechunkStorageConfig(
        store_path=str(tmp_path / "icechunk_store"),
        branch_name="main",
    )


@pytest.fixture
def write_ds(time_range: pd.DatetimeIndex) -> xr.Dataset:
    rng = np.random.default_rng(42)
    return xr.Dataset(
        {
            "temperature": xr.DataArray(
                rng.standard_normal((4, 6, 8, 8)).astype(np.float32),
                dims=["time", "level", "latitude", "longitude"],
                coords={
                    "time": time_range,
                    "level": LEVELS,
                    "latitude": LAT,
                    "longitude": LON,
                },
            )
        }
    )


class TestGenerateEra5Data:
    def test_returns_dataset(self, era5_config):
        result = generate.generate_era5_data(era5_config)
        assert isinstance(result, xr.Dataset)

    def test_variable_subset(self, era5_config):
        subset_vars = ["temperature", "geopotential"]
        era5_config.variables = subset_vars
        result = generate.generate_era5_data(era5_config)
        assert set(result.data_vars) == set(subset_vars)

    def test_all_requested_variables_present(self, era5_config):
        result = generate.generate_era5_data(era5_config)
        for var in era5_config.variables:
            assert var in result.data_vars

    def test_time_range_filter(self, era5_config):
        result = generate.generate_era5_data(era5_config)
        times = pd.DatetimeIndex(result.time.values)
        assert (times >= era5_config.time_start).all()
        assert (times <= era5_config.time_end).all()

    def test_geographic_subset(self, era5_config):
        result = generate.generate_era5_data(era5_config)
        assert result.latitude.min() >= era5_config.coordinates.lat_min
        assert result.latitude.max() <= era5_config.coordinates.lat_max
        assert result.longitude.min() >= era5_config.coordinates.lon_min
        assert result.longitude.max() <= era5_config.coordinates.lon_max


class TestWriteOrAppendIcechunkStore:
    def test_creates_new_store(self, storage_config, write_ds):
        storage = ic.local_filesystem_storage(storage_config.store_path)
        assert not ic.Repository.exists(storage)
        generate.write_or_append_icechunk_store(storage_config, write_ds)
        assert ic.Repository.exists(storage)

    def test_data_roundtrip(self, storage_config, write_ds):
        generate.write_or_append_icechunk_store(storage_config, write_ds)
        storage = ic.local_filesystem_storage(storage_config.store_path)
        repo = ic.Repository.open(storage)
        session = repo.readonly_session("main")
        result = xr.open_zarr(session.store)
        np.testing.assert_array_equal(result["temperature"].values, write_ds["temperature"].values)

    def test_append_increases_time_steps(self, storage_config, write_ds):
        second_time = pd.date_range("2019-01-02", periods=4, freq="6h")
        second_ds = write_ds.assign_coords(time=second_time)

        generate.write_or_append_icechunk_store(storage_config, write_ds)
        generate.write_or_append_icechunk_store(storage_config, second_ds)

        storage = ic.local_filesystem_storage(storage_config.store_path)
        repo = ic.Repository.open(storage)
        session = repo.readonly_session("main")
        result = xr.open_zarr(session.store)
        assert result.sizes["time"] == len(write_ds.time) + len(second_ds.time)


class TestYamlConfigLoading:
    def test_era5_data_config_from_yaml(self):
        with open(YAML_PATH) as f:
            raw = yaml.safe_load(f)

        result = dacite.from_dict(
            data_class=config.ERA5DataLoaderConfig,
            data=raw["era5_data_config"],
            config=dacite.Config(
                type_hooks={BoundingBox: lambda x: BoundingBox(*x)},
                check_types=False,
            ),
        )
        assert result.era5_uri == "/tmp/test_era5.zarr"
        assert result.variables == ["temperature"]
        assert result.pressure_levels == [1000, 850, 500]
        assert result.time_resolution == "6h"
        assert result.storage_options is None
        assert result.coordinates[0] == 20.0  # lat_min
        assert result.coordinates[1] == 50.0  # lat_max

    def test_icechunk_storage_config_from_yaml(self):
        with open(YAML_PATH) as f:
            import yaml

            raw = yaml.safe_load(f)
        import dacite

        result = dacite.from_dict(
            data_class=config.IcechunkStorageConfig,
            data=raw["icechunk_storage_config"],
            config=dacite.Config(cast=[tuple, datetime], check_types=False),
        )
        assert result.store_path == "/tmp/test_store"
        assert result.branch_name == "main"
        assert result.commit_message == "test commit"
