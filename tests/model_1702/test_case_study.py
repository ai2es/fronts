"""Tests for the model_1702 case-study figure module."""

import os

import numpy as np
import pytest
import xarray as xr

from fronts import utils
from fronts.model_1702 import case_study, normalization, store

N_LAT = 6
N_LON = 8
CASE_TIMES = ["2023-12-26T00:00:00", "2023-12-26T12:00:00"]


def _tiny_inputs_ds(times):
    time_values = np.array(times, dtype="datetime64[ns]")
    rng = np.random.default_rng(4)
    data_vars = {
        variable: (
            ("time", "level", "latitude", "longitude"),
            rng.random((len(time_values), 5, N_LAT, N_LON)).astype(np.float32),
        )
        for variable in normalization.VARIABLES
    }
    return xr.Dataset(
        data_vars,
        coords={
            "time": time_values,
            "level": np.array(normalization.LEVEL_COORD),
            "latitude": np.arange(40.0, 40.0 - N_LAT * 0.25, -0.25),
            "longitude": np.arange(250.0, 250.0 + N_LON * 0.25, 0.25),
        },
    )


def _case_cfg(**overrides):
    fields = {
        "model_path": "/nonexistent/model_1702.h5",
        "times": CASE_TIMES,
        "coordinates": utils.BoundingBox(25.0, 56.75, 228.0, 299.75),
        "front_types": ["CF", "WF", "SF", "OF"],
        "era5_uri": "gs://nonexistent",
        "storage_options": None,
        "inputs_cache_path": None,
        "outdir": "/tmp",
        "figure_name": "case.png",
        "gpu_device": None,
    }
    fields.update(overrides)
    return case_study.CaseStudyConfig(**fields)


def test_panel_title_matches_paper_caption_style():
    assert case_study.panel_title(np.datetime64("2023-12-26T00:00:00")) == "0000 UTC 26 Dec 2023"
    assert case_study.panel_title(np.datetime64("2023-12-27T12:00:00")) == "1200 UTC 27 Dec 2023"


def test_config_parses():
    config_path = os.path.join("configs", "model_1702", "case_study_xmas2023.yaml")
    yaml_data = utils.load_yaml(config_path)
    case_cfg = utils.parse_config_section(yaml_data, case_study.CaseStudyConfig, "case_config", utils.YAML_TYPE_HOOKS)
    assert len(case_cfg.times) == 4
    assert case_cfg.front_types == ["CF", "WF", "SF", "OF"]
    assert isinstance(case_cfg.coordinates, utils.BoundingBox)
    assert case_cfg.storage_options == {"token": "anon"}
    assert case_cfg.figure_name.endswith(".png")
    assert case_cfg.use_training_style is False


def test_test_case_config_parses_full_domain_training_style():
    config_path = os.path.join("configs", "model_1702", "case_study_test_2019_01_01.yaml")
    yaml_data = utils.load_yaml(config_path)
    case_cfg = utils.parse_config_section(yaml_data, case_study.CaseStudyConfig, "case_config", utils.YAML_TYPE_HOOKS)
    assert case_cfg.times == ["2019-01-01T00:00:00"]
    assert case_cfg.coordinates == utils.BoundingBox(0.25, 80.0, 130.0, 369.75)
    assert case_cfg.use_training_style is True


class TestLoadCaseInputs:
    def test_cache_hit_selects_requested_times(self, tmp_path):
        cache_path = str(tmp_path / "cache.nc")
        _tiny_inputs_ds([*CASE_TIMES, "2023-12-28T00:00:00"]).to_netcdf(cache_path)
        case_cfg = _case_cfg(inputs_cache_path=cache_path)
        loaded = case_study.load_case_inputs(case_cfg)
        assert loaded.sizes["time"] == len(CASE_TIMES)
        assert list(loaded.data_vars) == list(normalization.VARIABLES)

    def test_cache_missing_time_raises(self, tmp_path):
        cache_path = str(tmp_path / "cache.nc")
        _tiny_inputs_ds(CASE_TIMES[:1]).to_netcdf(cache_path)
        case_cfg = _case_cfg(inputs_cache_path=cache_path)
        with pytest.raises(ValueError, match="lack timesteps"):
            case_study.load_case_inputs(case_cfg)

    def test_derive_unwraps_longitude_for_a_wrap_crossing_domain(self, mocker):
        # A full-domain box like configs/model_1702/case_study_test_2019_01_01.yaml's
        # [0.25, 80.0, 130.0, 369.75] crosses the 360 deg boundary: select_spatial_domain
        # returns longitude ordered [340, 350, 0, 10] (non-monotonic) for a box like
        # [0, 10, 340, 370] below, which load_case_inputs must unwrap to [340, 350, 360, 370]
        # before it reaches store.build_1702_dataset — a non-monotonic axis produces the
        # wraparound/banding artifacts seen when this call was missing.
        times = np.array(["2019-01-01T00:00:00"], dtype="datetime64[ns]")
        lats = np.array([10.0, 0.0])
        lons = np.arange(0.0, 360.0, 10.0)
        levels = np.array(store.PRESSURE_LEVELS_HPA)
        rng = np.random.default_rng(11)
        shape_p = (len(times), len(levels), len(lats), len(lons))
        shape_s = (len(times), len(lats), len(lons))
        source = xr.Dataset(
            {
                "geopotential": (("time", "level", "latitude", "longitude"), rng.uniform(500.0, 15000.0, shape_p)),
                "temperature": (("time", "level", "latitude", "longitude"), rng.uniform(250.0, 300.0, shape_p)),
                "u_component_of_wind": (("time", "level", "latitude", "longitude"), rng.uniform(-30.0, 30.0, shape_p)),
                "v_component_of_wind": (("time", "level", "latitude", "longitude"), rng.uniform(-30.0, 30.0, shape_p)),
                "specific_humidity": (("time", "level", "latitude", "longitude"), rng.uniform(0.0001, 0.02, shape_p)),
                "surface_pressure": (("time", "latitude", "longitude"), rng.uniform(80000.0, 103000.0, shape_s)),
                "2m_temperature": (("time", "latitude", "longitude"), rng.uniform(260.0, 305.0, shape_s)),
                "2m_dewpoint_temperature": (("time", "latitude", "longitude"), rng.uniform(250.0, 300.0, shape_s)),
                "10m_u_component_of_wind": (("time", "latitude", "longitude"), rng.uniform(-20.0, 20.0, shape_s)),
                "10m_v_component_of_wind": (("time", "latitude", "longitude"), rng.uniform(-20.0, 20.0, shape_s)),
            },
            coords={"time": times, "level": levels, "latitude": lats, "longitude": lons},
        )
        mocker.patch.object(case_study.store, "open_source_era5", return_value=source)

        case_cfg = _case_cfg(
            times=["2019-01-01T00:00:00"],
            coordinates=utils.BoundingBox(0.0, 10.0, 340.0, 370.0),
            inputs_cache_path=None,
        )
        built = case_study.load_case_inputs(case_cfg)

        lon_values = built["longitude"].values
        assert np.all(np.diff(lon_values) > 0), f"longitude not monotonic: {lon_values}"
        assert lon_values[0] == pytest.approx(340.0)
        assert lon_values[-1] == pytest.approx(370.0)


def test_predict_case_shape():
    import tensorflow as tf

    def fake_adapter(x, training=False):
        return tf.zeros((tf.shape(x)[0], N_LAT, N_LON, 9))

    preds = case_study.predict_case(fake_adapter, _tiny_inputs_ds(CASE_TIMES))
    assert preds.shape == (len(CASE_TIMES), N_LAT, N_LON, 9)


@pytest.mark.skipif(
    not os.environ.get("MODEL_1702_RENDER_TESTS"),
    reason="set MODEL_1702_RENDER_TESTS=1 to run figure rendering (needs cartopy Natural Earth data)",
)
def test_render_case_figure_writes_file(tmp_path):
    built = _tiny_inputs_ds(CASE_TIMES)
    preds = np.random.default_rng(7).random((len(CASE_TIMES), N_LAT, N_LON, 9)).astype(np.float32)
    out_path = str(tmp_path / "case.png")
    case_study.render_case_figure(
        preds=preds,
        lats=built["latitude"].values,
        lons=built["longitude"].values,
        times=built["time"].values,
        front_types=["CF", "WF", "SF", "OF"],
        out_path=out_path,
    )
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


@pytest.mark.skipif(
    not os.environ.get("MODEL_1702_RENDER_TESTS"),
    reason="set MODEL_1702_RENDER_TESTS=1 to run figure rendering (needs cartopy Natural Earth data)",
)
def test_render_training_style_figure_writes_one_file_per_timestep(tmp_path):
    built = _tiny_inputs_ds(CASE_TIMES)
    preds = np.random.default_rng(7).random((len(CASE_TIMES), N_LAT, N_LON, 9)).astype(np.float32)
    case_study.render_training_style_figure(
        preds=preds,
        lats=built["latitude"].values,
        lons=built["longitude"].values,
        times=built["time"].values,
        front_types=["CF", "WF", "SF", "OF"],
        outdir=str(tmp_path),
        figure_name="case.png",
    )
    written = sorted(tmp_path.iterdir())
    assert len(written) == len(CASE_TIMES)
    for f in written:
        assert f.name.startswith("case_") and f.name.endswith(".png")
        assert f.stat().st_size > 0
