from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from fronts.plot.plot import _normalize_wrapping_longitudes, plot_test_prediction, probability_scale


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


def _cf_probs() -> xr.Dataset:
    lats, lons = np.arange(30.0, 40.25, 1.0), np.arange(250.0, 260.25, 1.0)
    bump = np.exp(-((lons[None, :] - 255.0) ** 2) / 4.0) * np.ones((len(lats), 1))
    return xr.Dataset(
        {"CF": (("latitude", "longitude"), 0.7 * bump.astype(np.float32))},
        coords={"latitude": lats, "longitude": lons},
    )


def test_plot_test_prediction_labels_the_colorbar_with_the_given_scale() -> None:
    probs = _cf_probs()
    fig = plot_test_prediction(
        lats=probs["latitude"].values,
        lons=probs["longitude"].values,
        probs_ds=probs,
        front_types=["CF"],
        filled_contours=True,
        open_contours=False,
        cbar_label="Probability (calibrated - 100 km)",
    )
    try:
        assert "Probability (calibrated - 100 km)" in [ax.get_ylabel() for ax in fig.axes]
    finally:
        plt.close(fig)


def test_probability_scale_leaves_raw_probabilities_alone_without_a_table() -> None:
    probs = _cf_probs()
    scaled, label = probability_scale(probs, calibration_path=None, neighborhood_km=100)
    xr.testing.assert_identical(scaled, probs)
    assert label == "Probability (uncalibrated)"


def test_probability_scale_maps_through_the_calibration_table(tmp_path: Path) -> None:
    table = xr.Dataset(
        {"calibrated_CF": (("neighborhood", "probability"), np.array([[0.1, 0.9], [0.2, 1.0]], dtype=np.float32))},
        coords={"neighborhood": np.array([50, 100]), "probability": np.array([0.0, 1.0])},
    )
    table_path = tmp_path / "calibration.nc"
    table.to_netcdf(table_path)
    probs = _cf_probs()
    scaled, label = probability_scale(probs, calibration_path=str(table_path), neighborhood_km=100)
    np.testing.assert_allclose(scaled["CF"], 0.2 + 0.8 * probs["CF"], rtol=1e-6)
    assert label == "Probability (calibrated - 100 km)"
