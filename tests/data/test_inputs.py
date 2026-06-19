import numpy as np
import pytest
import xarray as xr

from fronts.data.inputs import compute_norm_stats, inputs_ds_to_dataarray, load_or_compute_norm_stats

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
        result = inputs_ds_to_dataarray(ds, _VARS)
        assert list(result.dims) == ["time", "latitude", "longitude", "channel"]

    def test_shape(self):
        ds = _make_era5_ds()
        result = inputs_ds_to_dataarray(ds, _VARS)
        assert result.shape == (N_TIME, N_LAT, N_LON, N_CHANNELS)

    def test_dtype(self):
        ds = _make_era5_ds()
        result = inputs_ds_to_dataarray(ds, _VARS)
        assert result.dtype == np.float32

    def test_time_coord_preserved(self):
        ds = _make_era5_ds()
        result = inputs_ds_to_dataarray(ds, _VARS)
        np.testing.assert_array_equal(result.coords["time"].values, ds.time.values)

    def test_values_match_source(self):
        ds = _make_era5_ds()
        result = inputs_ds_to_dataarray(ds, _VARS).values
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
        result = inputs_ds_to_dataarray(ds, [*_VARS, "2m_temperature", "land_sea_mask"])
        assert result.shape == (N_TIME, N_LAT, N_LON, N_CHANNELS + 2)

    def test_level_channels_come_first(self):
        ds = _make_mixed_ds()
        result = inputs_ds_to_dataarray(ds, [*_VARS, "2m_temperature"])
        for var_idx, var in enumerate(_VARS):
            for lev_idx in range(_N_LEVELS):
                channel = lev_idx * _N_VARS + var_idx
                np.testing.assert_array_equal(
                    result.values[:, :, :, channel],
                    ds[var].values[:, lev_idx, :, :],
                )

    def test_single_level_values_match(self):
        ds = _make_mixed_ds()
        result = inputs_ds_to_dataarray(ds, [*_VARS, "2m_temperature"])
        np.testing.assert_array_equal(result.values[:, :, :, -1], ds["2m_temperature"].values)

    def test_static_variable_broadcast_along_time(self):
        ds = _make_mixed_ds()
        result = inputs_ds_to_dataarray(ds, [*_VARS, "land_sea_mask"])
        for t in range(N_TIME):
            np.testing.assert_array_equal(result.values[t, :, :, -1], ds["land_sea_mask"].values)

    def test_channel_labels(self):
        ds = _make_mixed_ds()
        result = inputs_ds_to_dataarray(ds, ["temperature", "2m_temperature"])
        labels = list(result.channel.values)
        assert labels == [f"temperature_{level}" for level in _LEVELS] + ["2m_temperature"]

    def test_single_level_only(self):
        ds = _make_mixed_ds()
        result = inputs_ds_to_dataarray(ds, ["2m_temperature", "land_sea_mask"])
        assert result.shape == (N_TIME, N_LAT, N_LON, 2)
        assert list(result.dims) == ["time", "latitude", "longitude", "channel"]

    def test_dtype_float32(self):
        ds = _make_mixed_ds()
        result = inputs_ds_to_dataarray(ds, [*_VARS, "2m_temperature", "land_sea_mask"])
        assert result.dtype == np.float32


def _make_channel_da(seed: int = 3, with_nan: bool = False) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((N_TIME, N_LAT, N_LON, 4)).astype(np.float32)
    if with_nan:
        data[0, 0, 0, 1] = np.nan
    da = xr.DataArray(
        data,
        dims=["time", "latitude", "longitude", "channel"],
        coords={"time": np.arange(N_TIME), "channel": [f"ch{i}" for i in range(4)]},
    )
    return da.chunk({"time": 2})


class TestComputeNormStats:
    def test_matches_numpy(self):
        da = _make_channel_da()
        mean, variance = compute_norm_stats(da)
        values = da.values
        np.testing.assert_allclose(mean, values.mean(axis=(0, 1, 2)), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(variance, values.var(axis=(0, 1, 2)), rtol=1e-4, atol=1e-6)

    def test_shapes_and_dtype(self):
        da = _make_channel_da()
        mean, variance = compute_norm_stats(da)
        assert mean.shape == (4,)
        assert variance.shape == (4,)
        assert mean.dtype == np.float32
        assert variance.dtype == np.float32

    def test_non_dask_input(self):
        da = _make_channel_da().compute()
        mean, variance = compute_norm_stats(da)
        np.testing.assert_allclose(mean, da.values.mean(axis=(0, 1, 2)), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(variance, da.values.var(axis=(0, 1, 2)), rtol=1e-4, atol=1e-6)

    def test_raises_on_nan(self):
        da = _make_channel_da(with_nan=True)
        with pytest.raises(ValueError, match="NaN in normalization statistics"):
            compute_norm_stats(da)


class TestLoadOrComputeNormStats:
    def test_no_cache_dir_matches_direct_compute(self):
        da = _make_channel_da()
        direct = compute_norm_stats(da)
        cached = load_or_compute_norm_stats(da, None, ("key",))
        np.testing.assert_array_equal(cached[0], direct[0])
        np.testing.assert_array_equal(cached[1], direct[1])

    def test_writes_cache_file(self, tmp_path):
        da = _make_channel_da()
        load_or_compute_norm_stats(da, str(tmp_path), ("snap", "channels", "indices"))
        assert len(list(tmp_path.glob("norm_stats_*.npz"))) == 1

    def test_cache_hit_skips_compute(self, tmp_path):
        da = _make_channel_da()
        key_parts = ("snap", "channels", "indices")
        mean, variance = load_or_compute_norm_stats(da, str(tmp_path), key_parts)
        nan_da = _make_channel_da(with_nan=True)
        cached_mean, cached_variance = load_or_compute_norm_stats(nan_da, str(tmp_path), key_parts)
        np.testing.assert_array_equal(cached_mean, mean)
        np.testing.assert_array_equal(cached_variance, variance)

    def test_different_keys_use_different_files(self, tmp_path):
        da = _make_channel_da()
        load_or_compute_norm_stats(da, str(tmp_path), ("snap-a",))
        load_or_compute_norm_stats(da, str(tmp_path), ("snap-b",))
        assert len(list(tmp_path.glob("norm_stats_*.npz"))) == 2

    def test_creates_missing_cache_dir(self, tmp_path):
        da = _make_channel_da()
        cache_dir = tmp_path / "nested" / "cache"
        mean, variance = load_or_compute_norm_stats(da, str(cache_dir), ("key",))
        assert cache_dir.exists()
        assert mean.shape == (4,)
        assert variance.shape == (4,)
