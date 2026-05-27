import numpy as np
import pytest
import xarray as xr

_N_TIME = 5
_N_LAT = 32
_N_LON = 64
_N_CHANNELS = 30
_N_CLASSES = 6


@pytest.fixture
def era5_da() -> xr.DataArray:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((_N_TIME, _N_LAT, _N_LON, _N_CHANNELS)).astype(np.float32)
    return xr.DataArray(
        data,
        dims=["time", "latitude", "longitude", "channel"],
        coords={"time": np.arange(_N_TIME)},
    )


@pytest.fixture
def front_da() -> xr.DataArray:
    rng = np.random.default_rng(1)
    data = rng.random((_N_TIME, _N_LAT, _N_LON, _N_CLASSES)).astype(np.float32)
    return xr.DataArray(
        data,
        dims=["time", "latitude", "longitude", "class"],
        coords={"time": np.arange(_N_TIME)},
    )
