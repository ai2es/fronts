import numpy as np
import pytest
import xarray as xr

from fronts.evaluation import global_eval

LATS = np.array([-0.5, 0.0, 0.5])
LONS = np.array([0.0, 1.0])


def _ds() -> xr.Dataset:
    shape = (2, 3, 2)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1.0
    dims = ("time", "latitude", "longitude")
    return xr.Dataset(
        {
            "v_component_of_wind": (dims, base),
            "10m_v_component_of_wind": (dims, base * 2),
            "u_component_of_wind": (dims, base * 3),
        },
        coords={"time": [0, 1], "latitude": LATS, "longitude": LONS},
    )


def test_flip_negates_only_v_and_mirrors_latitude():
    ds = _ds()
    out = global_eval.flip_inputs(ds)
    np.testing.assert_array_equal(out["latitude"].values, LATS)
    np.testing.assert_array_equal(out["u_component_of_wind"].values, ds["u_component_of_wind"].values[:, ::-1])
    np.testing.assert_array_equal(out["v_component_of_wind"].values, -ds["v_component_of_wind"].values[:, ::-1])
    np.testing.assert_array_equal(out["10m_v_component_of_wind"].values, -ds["10m_v_component_of_wind"].values[:, ::-1])


def test_flip_is_an_involution_and_leaves_input_untouched():
    ds = _ds()
    before = ds["v_component_of_wind"].values.copy()
    twice = global_eval.flip_inputs(global_eval.flip_inputs(ds))
    xr.testing.assert_equal(twice, ds)
    np.testing.assert_array_equal(ds["v_component_of_wind"].values, before)


def test_flip_rejects_asymmetric_latitudes():
    ds = _ds().assign_coords(latitude=[0.25, 0.5, 0.75])
    with pytest.raises(ValueError, match="symmetric"):
        global_eval.flip_inputs(ds)


def test_hemisphere_slices_put_equator_in_north():
    rows = global_eval.hemisphere_slices(LATS)
    assert LATS[rows["SH"]].tolist() == [-0.5]
    assert LATS[rows["NH"]].tolist() == [0.0, 0.5]


def test_combine_global_sums_counts_and_stacks_maps():
    def part(lats, value):
        spatial = xr.Dataset({"tp": (("latitude", "longitude"), np.full((len(lats), 2), value))}, coords={"latitude": lats, "longitude": LONS})
        aggregate = xr.Dataset({"tp": (("threshold",), np.array([value, value]))})
        return spatial, aggregate

    spatial, aggregate = global_eval.combine_global({"SH": part([-0.5], 1.0), "NH": part([0.0, 0.5], 2.0)})
    assert spatial["latitude"].values.tolist() == [-0.5, 0.0, 0.5]
    assert spatial["tp"].values[:, 0].tolist() == [1.0, 2.0, 2.0]
    assert aggregate["tp"].values.tolist() == [3.0, 3.0]
