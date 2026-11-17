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


def test_flip_mirrors_descending_latitude_too():
    ds = _ds().isel(latitude=slice(None, None, -1))
    out = global_eval.flip_inputs(ds)
    np.testing.assert_array_equal(out["latitude"].values, LATS[::-1])
    np.testing.assert_array_equal(out["u_component_of_wind"].values, ds["u_component_of_wind"].values[:, ::-1])
    np.testing.assert_array_equal(out["v_component_of_wind"].values, -ds["v_component_of_wind"].values[:, ::-1])


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


def test_hemisphere_slices_descending_axis():
    rows = global_eval.hemisphere_slices(LATS[::-1])
    assert LATS[::-1][rows["NH"]].tolist() == [0.5, 0.0]
    assert LATS[::-1][rows["SH"]].tolist() == [-0.5]


def test_combine_global_sums_counts_and_stacks_maps():
    def part(lats, value):
        grid = np.full((len(lats), 2), value)
        spatial = xr.Dataset({"tp": (("latitude", "longitude"), grid)}, coords={"latitude": lats, "longitude": LONS})
        aggregate = xr.Dataset({"tp": (("threshold",), np.array([value, value]))})
        return spatial, aggregate

    spatial, aggregate = global_eval.combine_global({"SH": part([-0.5], 1.0), "NH": part([0.0, 0.5], 2.0)})
    assert spatial["latitude"].values.tolist() == [-0.5, 0.0, 0.5]
    assert spatial["tp"].values[:, 0].tolist() == [1.0, 2.0, 2.0]
    assert aggregate["tp"].values.tolist() == [3.0, 3.0]


def test_pad_to_multiple_crops_back_to_input_rows():
    import tensorflow as tf

    seen = []

    def fake_model(x, training=False):
        seen.append(x.shape[1])
        return x[..., :2]

    x = tf.constant(np.arange(2 * 5 * 3 * 4, dtype=np.float32).reshape(2, 5, 3, 4))
    out = global_eval.PadToMultiple(fake_model, 4)(x)
    assert seen == [8]
    np.testing.assert_array_equal(out.numpy(), x.numpy()[..., :2])


def test_pad_to_multiple_handles_list_outputs():
    import tensorflow as tf

    def fake_model(x, training=False):
        return [x[..., :1], x[..., 1:2]]

    x = tf.ones((1, 641, 4, 3))
    outs = global_eval.PadToMultiple(fake_model, 32)(x)
    assert [o.shape[1] for o in outs] == [641, 641]


def test_combine_global_passes_through_missing_spatial_maps():
    aggregate = xr.Dataset({"tp": (("threshold",), np.array([1.0, 2.0]))})
    spatial, total = global_eval.combine_global({"SH": (None, aggregate), "NH": (None, aggregate)})
    assert spatial is None
    assert total["tp"].values.tolist() == [2.0, 4.0]


def test_run_scores_only_the_composite_with_spatial_stats(tmp_path, monkeypatch):
    import dataclasses

    import tensorflow as tf

    from fronts import evaluate, utils

    lats = np.arange(-1.5, 1.75, 0.5)  # 7 rows, symmetric: SH rows 0-2, NH rows 3-6
    lons = np.arange(0.0, 4.0, 0.5)
    times = np.array(["2023-06-15T00", "2023-06-15T06"], dtype="datetime64[ns]")
    dims = ("time", "latitude", "longitude")
    coords = {"time": times, "latitude": lats, "longitude": lons}
    inputs = xr.Dataset({"v_component_of_wind": (dims, np.ones((2, 7, 8), np.float32))}, coords=coords)
    codes = np.zeros((2, 7, 8), np.int8)
    codes[:, 5, :] = 1  # a cold front in the NH
    codes[:, 1, :] = 1  # and one in the SH
    fronts = xr.DataArray(codes, dims=dims, coords=coords)

    onehot = np.zeros((2, 7, 8, 9), np.float32)
    onehot[..., 0] = 1.0
    onehot[codes == 1] = [0, 1, 0, 0, 0, 0, 0, 0, 0]

    def fake_predict_batches(model, input_ds, target_da, **kwargs):
        flipped = bool((input_ds["v_component_of_wind"] < 0).all())
        preds = onehot[:, ::-1].copy() if flipped else onehot.copy()  # flipped output is in the mirrored frame
        wrong = slice(0, 4) if flipped else slice(0, 3)  # garbage where the pass is not scored; flipped is mirrored
        preds[:, wrong] = 0.0
        return preds, onehot

    @dataclasses.dataclass
    class FakeDataConfig:
        front_dilation: int = 0
        class_weights: list | None = None

    monkeypatch.setattr(utils, "configure_gpu", lambda device: None)
    monkeypatch.setattr(global_eval, "_open_data", lambda cfg, data_cfg: (inputs, fronts))
    monkeypatch.setattr(tf.keras.models, "load_model", lambda *a, **k: object())
    monkeypatch.setattr(evaluate, "predict_batches", fake_predict_batches)

    cfg = global_eval.GlobalEvalConfig(
        model_path="unused",
        targets_zarr="unused",
        outdir=str(tmp_path),
        coordinates=utils.BoundingBox(-1.5, 1.5, 0.0, 3.5),
        front_types=["CF"],
        front_dilation=0,
        batch_size=2,
        pad_multiple=1,
        chunk_size=1,
        time_start=None,
        time_end=None,
        time_resolution=None,
        write_spatial=False,
        gpu_device=None,
    )
    global_eval.run(cfg, FakeDataConfig())

    composite = tmp_path / "composite"
    newest = max(composite.iterdir(), key=lambda f: f.stat().st_mtime_ns)
    assert newest.stat().st_mtime_ns == (composite / "stats_derived_SH.nc").stat().st_mtime_ns  # the sbatch marker
    assert [p.name for p in tmp_path.iterdir()] == ["composite"]
    written = sorted(p.name for p in (tmp_path / "composite").iterdir())
    assert written == [f"stats_{n}{s}.nc" for n in ("aggregate", "derived") for s in ("", "_NH", "_SH")]
    for suffix in ("", "_NH", "_SH"):
        derived = xr.open_dataset(tmp_path / "composite" / f"stats_derived{suffix}.nc")
        assert "spatial_csi_CF" in derived
        assert derived["spatial_csi_CF"].dims == ("latitude", "longitude", "neighborhood", "threshold")
        aggregate = xr.open_dataset(tmp_path / "composite" / f"stats_aggregate{suffix}.nc")
        assert float(aggregate["fn_CF"].isel(neighborhood=0, threshold=50)) == 0.0
        assert float(aggregate["tp_CF"].isel(neighborhood=0, threshold=50)) > 0.0
