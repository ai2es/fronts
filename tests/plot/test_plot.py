import numpy as np
import pytest
import xarray as xr

from fronts.plot.plot import _normalize_wrapping_longitudes


def _make_da(lats: np.ndarray, lons: np.ndarray) -> xr.DataArray:
    rng = np.random.default_rng(0)
    return xr.DataArray(
        rng.random((len(lats), len(lons))).astype(np.float32),
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
    )


@pytest.mark.parametrize(
    "lon_min,input_lons,expected_lons",
    [
        (
            130.0,
            np.concatenate([np.arange(130.0, 360.0, 0.25), np.arange(0.0, 10.0, 0.25)]),
            np.concatenate([np.arange(130.0, 360.0, 0.25), np.arange(360.0, 370.0, 0.25)]),
        ),
        (
            0.0,
            np.arange(0.0, 360.0, 0.25),
            np.arange(0.0, 360.0, 0.25),
        ),
    ],
)
def test_normalize_wrapping_longitudes(
    lon_min: float,
    input_lons: np.ndarray,
    expected_lons: np.ndarray,
) -> None:
    lats = np.arange(0.25, 81.0, 0.25)
    da = _make_da(lats, input_lons)
    result = _normalize_wrapping_longitudes(da, lon_min)
    np.testing.assert_array_almost_equal(result.longitude.values, expected_lons)


def test_normalize_wrapping_longitudes_monotonic_after_sortby() -> None:
    lats = np.arange(80.0, 0.0, -0.25)
    lons = np.concatenate([np.arange(130.0, 360.0, 0.25), np.arange(0.0, 10.0, 0.25)])
    da = _make_da(lats, lons)

    result = _normalize_wrapping_longitudes(da, lon_min=130.0).sortby("latitude").sortby("longitude")

    assert np.all(np.diff(result.latitude.values) > 0), "latitude not monotonically increasing"
    assert np.all(np.diff(result.longitude.values) > 0), "longitude not monotonically increasing"
