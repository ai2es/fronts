import numpy as np
import pytest
import xarray as xr

from fronts import utils
from fronts.data import datasets

_N_TIME = 5
_N_LAT = 32
_N_LON = 64
_VARIABLES = [
    "geopotential",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
]


@pytest.fixture
def era5_ds() -> xr.Dataset:
    """Raw (pre-channel-stacking) input Dataset, one single-level variable per name."""
    rng = np.random.default_rng(0)
    data_vars = {
        var: xr.DataArray(
            rng.standard_normal((_N_TIME, _N_LAT, _N_LON)).astype(np.float32),
            dims=["time", "latitude", "longitude"],
            coords={"time": np.arange(_N_TIME)},
        )
        for var in _VARIABLES
    }
    return xr.Dataset(data_vars)


@pytest.fixture
def front_da() -> xr.DataArray:
    """Raw (pre-one-hot, pre-remap) integer front-code DataArray."""
    rng = np.random.default_rng(1)
    data = rng.integers(0, 2, size=(_N_TIME, _N_LAT, _N_LON)).astype(np.int32)
    return xr.DataArray(
        data,
        dims=["time", "latitude", "longitude"],
        coords={"time": np.arange(_N_TIME)},
    )


@pytest.fixture
def data_config() -> datasets.DatasetConfig:
    dummy_store = utils.IcechunkStorageConfig(store_path="unused", branch_name="main")
    return datasets.DatasetConfig(
        inputs_icechunk_config=dummy_store,
        targets_icechunk_config=dummy_store,
        variables=_VARIABLES,
        test_years=[],
        val_years=[],
    )
