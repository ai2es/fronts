import numpy as np
import xarray as xr

from fronts.data.inputs import era5_to_dataarray

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


def _make_mixed_ds() -> xr.Dataset:
    ds = _make_era5_ds()
    rng = np.random.default_rng(7)
    ds["2m_temperature"] = xr.DataArray(
        rng.standard_normal((N_TIME, N_LAT, N_LON)).astype(np.float32),
        dims=["time", "latitude", "longitude"],
        coords={"time": ds.time.values},
    )
    ds["land_sea_mask"] = xr.DataArray(
        rng.random((N_LAT, N_LON)).astype(np.float32),
        dims=["latitude", "longitude"],
    )
    return ds


class TestMixedLevelToDataarray:
    def test_channel_count(self):
        ds = _make_mixed_ds()
        result = era5_to_dataarray(ds, [*_VARS, "2m_temperature", "land_sea_mask"])
        assert result.shape == (N_TIME, N_LAT, N_LON, N_CHANNELS + 2)

    def test_level_channels_come_first(self):
        ds = _make_mixed_ds()
        result = era5_to_dataarray(ds, [*_VARS, "2m_temperature"])
        for var_idx, var in enumerate(_VARS):
            for lev_idx in range(_N_LEVELS):
                channel = lev_idx * _N_VARS + var_idx
                np.testing.assert_array_equal(
                    result.values[:, :, :, channel],
                    ds[var].values[:, lev_idx, :, :],
                )

    def test_single_level_values_match(self):
        ds = _make_mixed_ds()
        result = era5_to_dataarray(ds, [*_VARS, "2m_temperature"])
        np.testing.assert_array_equal(result.values[:, :, :, -1], ds["2m_temperature"].values)

    def test_static_variable_broadcast_along_time(self):
        ds = _make_mixed_ds()
        result = era5_to_dataarray(ds, [*_VARS, "land_sea_mask"])
        for t in range(N_TIME):
            np.testing.assert_array_equal(result.values[t, :, :, -1], ds["land_sea_mask"].values)

    def test_channel_labels(self):
        ds = _make_mixed_ds()
        result = era5_to_dataarray(ds, ["temperature", "2m_temperature"])
        labels = list(result.channel.values)
        assert labels == [f"temperature_{level}" for level in _LEVELS] + ["2m_temperature"]

    def test_single_level_only(self):
        ds = _make_mixed_ds()
        result = era5_to_dataarray(ds, ["2m_temperature", "land_sea_mask"])
        assert result.shape == (N_TIME, N_LAT, N_LON, 2)
        assert list(result.dims) == ["time", "latitude", "longitude", "channel"]

    def test_dtype_float32(self):
        ds = _make_mixed_ds()
        result = era5_to_dataarray(ds, [*_VARS, "2m_temperature", "land_sea_mask"])
        assert result.dtype == np.float32
