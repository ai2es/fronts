import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def make_static_source_zarr(tmp_path):
    """Factory fixture building a local zarr store standing in for a static ERA5 source.

    Returns a callable ``(variable_name, lat, lon, time=None, seed=0) -> pathlib.Path``
    that writes a single-variable store with the value broadcast identically across
    time, matching the shape of the real ERA5-invariant public ARCO archive fields
    this substitutes for in tests (see ``sources.STATIC_VARIABLE_SOURCES``).
    """

    def _make(
        variable_name: str, lat: np.ndarray, lon: np.ndarray, time: pd.DatetimeIndex | None = None, seed: int = 0
    ):
        if time is None:
            time = pd.date_range("2019-01-01", periods=2, freq="6h")
        rng = np.random.default_rng(seed)
        values = np.broadcast_to(
            rng.standard_normal((len(lat), len(lon))).astype(np.float32), (len(time), len(lat), len(lon))
        )
        ds = xr.Dataset(
            {
                variable_name: xr.DataArray(
                    values,
                    dims=["time", "latitude", "longitude"],
                    coords={"time": time, "latitude": lat, "longitude": lon},
                )
            }
        )
        path = tmp_path / f"{variable_name}_source.zarr"
        ds.to_zarr(path)
        return path

    return _make
