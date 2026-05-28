import numpy as np
import xarray as xr
import xbatcher

from fronts.data.inputs import (
    collect_norm_sample_from_bgen,
    era5_to_dataarray,
)

N_TIME = 5
N_LAT = 32
N_LON = 64
N_CLASSES = 6

_VARS = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
_LEVELS = [1000, 950, 900, 850, 700, 500]
_N_LEVELS = len(_LEVELS)
_N_VARS = len(_VARS)
N_CHANNELS = _N_LEVELS * _N_VARS


def _make_era5_ds(
    n_time: int = N_TIME,
    n_lat: int = N_LAT,
    n_lon: int = N_LON,
    n_levels: int = _N_LEVELS,
) -> xr.Dataset:
    rng = np.random.default_rng(42)
    ds_vars = {}
    for var in _VARS:
        data = rng.standard_normal((n_time, n_levels, n_lat, n_lon)).astype(np.float32)
        ds_vars[var] = xr.DataArray(
            data,
            dims=["time", "level", "latitude", "longitude"],
            coords={"time": np.arange(n_time), "level": _LEVELS},
        )
    return xr.Dataset(ds_vars)


def _make_bgen(da: xr.DataArray) -> xbatcher.BatchGenerator:
    return xbatcher.BatchGenerator(
        da,
        input_dims={
            "time": 1,
            "latitude": da.sizes["latitude"],
            "longitude": da.sizes["longitude"],
        },
        preload_batch=False,
    )


class TestEra5ToDataarray:
    def test_dims(self):
        ds = _make_era5_ds()
        result = era5_to_dataarray(ds, _VARS)
        assert list(result.dims) == ["time", "latitude", "longitude", "channel"]

    def test_shape(self):
        ds = _make_era5_ds()
        result = era5_to_dataarray(ds, _VARS)
        assert result.shape == (N_TIME, N_LAT, N_LON, N_CHANNELS)

    def test_dtype(self):
        ds = _make_era5_ds()
        result = era5_to_dataarray(ds, _VARS)
        assert result.dtype == np.float32

    def test_time_coord_preserved(self):
        ds = _make_era5_ds()
        result = era5_to_dataarray(ds, _VARS)
        np.testing.assert_array_equal(result.coords["time"].values, ds.time.values)

    def test_values_match_source(self):
        ds = _make_era5_ds()
        result = era5_to_dataarray(ds, _VARS).values
        for var_idx, var in enumerate(_VARS):
            for lev_idx in range(_N_LEVELS):
                channel = lev_idx * _N_VARS + var_idx
                np.testing.assert_array_equal(
                    result[:, :, :, channel],
                    ds[var].values[:, lev_idx, :, :],
                )


class TestCollectNormSampleFromBgen:
    def test_shape(self, era5_da):
        bgen = _make_bgen(era5_da)
        n_samples = 3
        result = collect_norm_sample_from_bgen(bgen, n_samples=n_samples)
        assert result.shape == (n_samples, N_LAT, N_LON, N_CHANNELS)

    def test_dtype(self, era5_da):
        bgen = _make_bgen(era5_da)
        result = collect_norm_sample_from_bgen(bgen, n_samples=2)
        assert result.dtype == np.float32

    def test_capped_at_available_timesteps(self, era5_da):
        bgen = _make_bgen(era5_da)
        result = collect_norm_sample_from_bgen(bgen, n_samples=N_TIME + 100)
        assert len(result) <= N_TIME

    def test_single_sample(self, era5_da):
        bgen = _make_bgen(era5_da)
        result = collect_norm_sample_from_bgen(bgen, n_samples=1)
        assert result.shape == (1, N_LAT, N_LON, N_CHANNELS)
