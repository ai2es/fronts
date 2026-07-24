import numpy as np
import pytest
import xarray as xr

from fronts.data.inputs import (
    compute_norm_stats,
    inputs_ds_to_dataarray,
    inputs_ds_to_volume_dataarray,
    load_or_compute_norm_stats,
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


def _make_volume_da(seed: int = 3, n_levels: int = 3, n_vars: int = 4) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((N_TIME, N_LAT, N_LON, n_levels, n_vars)).astype(np.float32)
    da = xr.DataArray(
        data,
        dims=["time", "latitude", "longitude", "level", "variable"],
        coords={
            "time": np.arange(N_TIME),
            "level": _LEVELS[:n_levels],
            "variable": [f"var{i}" for i in range(n_vars)],
        },
    )
    return da.chunk({"time": 2})


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


class TestVolumeDataarray:
    def test_dims(self):
        ds = _make_era5_ds()
        result = inputs_ds_to_volume_dataarray(ds, _VARS)
        assert list(result.dims) == ["time", "latitude", "longitude", "level", "variable"]

    def test_shape(self):
        ds = _make_era5_ds()
        result = inputs_ds_to_volume_dataarray(ds, _VARS)
        assert result.shape == (N_TIME, N_LAT, N_LON, _N_LEVELS, _N_VARS)

    def test_dtype(self):
        ds = _make_era5_ds()
        result = inputs_ds_to_volume_dataarray(ds, _VARS)
        assert result.dtype == np.float32

    def test_values_match_source(self):
        ds = _make_era5_ds()
        result = inputs_ds_to_volume_dataarray(ds, _VARS).values
        for var_idx, var in enumerate(_VARS):
            for lev_idx in range(_N_LEVELS):
                np.testing.assert_array_equal(
                    result[:, :, :, lev_idx, var_idx],
                    ds[var].values[:, lev_idx, :, :],
                )

    def test_variable_axis_follows_request_order(self):
        ds = _make_era5_ds()
        reordered = list(reversed(_VARS))
        result = inputs_ds_to_volume_dataarray(ds, reordered)
        assert list(result["variable"].values) == reordered

    def test_single_level_variable_broadcast_along_level(self):
        ds = _make_mixed_ds()
        result = inputs_ds_to_volume_dataarray(ds, [*_VARS, "2m_temperature"]).values
        for lev_idx in range(_N_LEVELS):
            np.testing.assert_array_equal(result[:, :, :, lev_idx, -1], ds["2m_temperature"].values)

    def test_static_variable_broadcast_along_time_and_level(self):
        ds = _make_mixed_ds()
        result = inputs_ds_to_volume_dataarray(ds, [*_VARS, "land_sea_mask"]).values
        for t in range(N_TIME):
            for lev_idx in range(_N_LEVELS):
                np.testing.assert_array_equal(result[t, :, :, lev_idx, -1], ds["land_sea_mask"].values)

    def test_empty_variables_raises(self):
        ds = _make_era5_ds()
        with pytest.raises(ValueError, match="No variables requested"):
            inputs_ds_to_volume_dataarray(ds, [])


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

    def test_minmax_matches_numpy(self):
        da = _make_channel_da()
        min_val, max_val = compute_norm_stats(da, method="minmax")
        values = da.values
        np.testing.assert_allclose(min_val, values.min(axis=(0, 1, 2)), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(max_val, values.max(axis=(0, 1, 2)), rtol=1e-4, atol=1e-6)

    def test_volume_da_stats_have_level_variable_shape(self):
        da = _make_volume_da(n_levels=3, n_vars=4)
        mean, variance = compute_norm_stats(da)
        assert mean.shape == (3, 4)
        assert variance.shape == (3, 4)
        values = da.values
        np.testing.assert_allclose(mean, values.mean(axis=(0, 1, 2)), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(variance, values.var(axis=(0, 1, 2)), rtol=1e-4, atol=1e-6)

    def test_unrecognized_method_raises(self):
        da = _make_channel_da()
        with pytest.raises(ValueError, match="Unrecognized normalization method"):
            compute_norm_stats(da, method="bogus")  # type: ignore[arg-type]


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

    def test_minmax_method_round_trips_through_cache(self, tmp_path):
        da = _make_channel_da()
        direct_min, direct_max = compute_norm_stats(da, method="minmax")
        cached_min, cached_max = load_or_compute_norm_stats(da, str(tmp_path), ("key",), method="minmax")
        np.testing.assert_array_equal(cached_min, direct_min)
        np.testing.assert_array_equal(cached_max, direct_max)

    def test_standardization_and_minmax_share_cache_dir_without_colliding(self, tmp_path):
        """A cache dir shared by a standardization run and a minmax run must not collide.

        Two branches pointed at the same norm_stats_cache_dir must not crash when one
        run's cache file is read by the other's method — each method gets its own file.
        """
        da = _make_channel_da()
        key_parts = ("shared-snapshot",)
        mean, variance = load_or_compute_norm_stats(da, str(tmp_path), key_parts, method="standardization")
        min_val, max_val = load_or_compute_norm_stats(da, str(tmp_path), key_parts, method="minmax")

        direct_mean, direct_variance = compute_norm_stats(da, method="standardization")
        direct_min, direct_max = compute_norm_stats(da, method="minmax")
        np.testing.assert_array_equal(mean, direct_mean)
        np.testing.assert_array_equal(variance, direct_variance)
        np.testing.assert_array_equal(min_val, direct_min)
        np.testing.assert_array_equal(max_val, direct_max)
        assert len(list(tmp_path.glob("norm_stats_*.npz"))) == 2

    def test_volume_da_stats_round_trip_through_cache(self, tmp_path):
        da = _make_volume_da()
        direct_mean, direct_variance = compute_norm_stats(da)
        cached_mean, cached_variance = load_or_compute_norm_stats(da, str(tmp_path), ("volume-key",))
        np.testing.assert_array_equal(cached_mean, direct_mean)
        np.testing.assert_array_equal(cached_variance, direct_variance)
        hit_mean, hit_variance = load_or_compute_norm_stats(da, str(tmp_path), ("volume-key",))
        np.testing.assert_array_equal(hit_mean, direct_mean)
        np.testing.assert_array_equal(hit_variance, direct_variance)

    def test_flat_cache_with_same_key_is_not_reused_for_volume_da(self, tmp_path):
        """A (n_channels,) cache entry must be a clean miss for a (level, variable) request."""
        flat_da = _make_channel_da()
        key_parts = ("shared-key",)
        load_or_compute_norm_stats(flat_da, str(tmp_path), key_parts)

        volume_da = _make_volume_da()
        mean, variance = load_or_compute_norm_stats(volume_da, str(tmp_path), key_parts)
        direct_mean, direct_variance = compute_norm_stats(volume_da)
        np.testing.assert_array_equal(mean, direct_mean)
        np.testing.assert_array_equal(variance, direct_variance)

    def test_stale_pre_method_cache_file_is_ignored(self, tmp_path):
        """A cache file from before per-method filenames existed must not crash the loader.

        A file with no 'mean' key, or written under the old bare `norm_stats_{key}.npz`
        name, simply isn't found under the new per-method naming and is recomputed.
        """
        da = _make_channel_da()
        key = "old-format-key"
        import hashlib

        old_cache_key = hashlib.sha256(key.encode()).hexdigest()[:16]
        old_path = tmp_path / f"norm_stats_{old_cache_key}.npz"
        np.savez(old_path, min=np.zeros(4, dtype=np.float32), max=np.ones(4, dtype=np.float32))

        mean, variance = load_or_compute_norm_stats(da, str(tmp_path), (key,), method="standardization")
        direct_mean, direct_variance = compute_norm_stats(da, method="standardization")
        np.testing.assert_array_equal(mean, direct_mean)
        np.testing.assert_array_equal(variance, direct_variance)
