"""End-to-end smoke tests: synthetic side-store data through the real compute_stats engine."""

import glob
import os

import numpy as np
import pytest
import tensorflow as tf
import xarray as xr

from fronts import evaluate, utils
from fronts.data import datasets
from fronts.model_1702 import adapter, normalization, run_eval

N_TIME = 8
N_LAT = 32
N_LON = 64
FRONT_TYPES = ["CF", "WF", "SF", "OF", "DL"]


@pytest.fixture
def synthetic_side_store_ds():
    rng = np.random.default_rng(21)
    times = np.array([f"2019-01-0{d}T00" for d in range(1, N_TIME + 1)], dtype="datetime64[ns]")
    lats = np.arange(40.0, 40.0 - N_LAT * 0.25, -0.25)
    lons = np.arange(250.0, 250.0 + N_LON * 0.25, 0.25)
    maxs, mins = normalization.norm_min_max_tables()
    data_vars = {}
    for var_idx, variable in enumerate(normalization.VARIABLES):
        per_level = [
            rng.uniform(mins[level_idx, var_idx], maxs[level_idx, var_idx], (N_TIME, N_LAT, N_LON))
            for level_idx in range(len(normalization.LEVELS))
        ]
        data_vars[variable] = (
            ("time", "level", "latitude", "longitude"),
            np.stack(per_level, axis=1).astype(np.float32),
        )
    return xr.Dataset(
        data_vars,
        coords={
            "time": times,
            "level": np.array(normalization.LEVEL_COORD),
            "latitude": lats,
            "longitude": lons,
        },
    )


@pytest.fixture
def synthetic_targets(synthetic_side_store_ds):
    rng = np.random.default_rng(33)
    codes = np.zeros((N_TIME, N_LAT, N_LON), dtype=np.int64)
    front_codes = [1, 2, 3, 4, 16, 5, 9, 14]
    for t in range(N_TIME):
        for code in front_codes:
            rows = rng.integers(0, N_LAT, size=12)
            cols = rng.integers(0, N_LON, size=12)
            codes[t, rows, cols] = code
    return xr.DataArray(
        codes,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": synthetic_side_store_ds["time"],
            "latitude": synthetic_side_store_ds["latitude"],
            "longitude": synthetic_side_store_ds["longitude"],
        },
        name="identifier",
    )


@pytest.fixture
def side_store_data_config():
    dummy_store = utils.IcechunkStorageConfig(store_path="/nonexistent", branch_name="main")
    return datasets.DatasetConfig(
        inputs_icechunk_config=dummy_store,
        targets_icechunk_config=dummy_store,
        variables=list(normalization.VARIABLES),
        test_years=[2019],
        val_years=[2018],
        batch_size=4,
        class_weights=[0.0] + [1.0] * 5,
        front_dilation=1,
        volume_inputs=True,
    )


class SoftmaxProbeModel:
    """Fake legacy model: softmax over six channels derived from the surface level."""

    def __call__(self, x, training=False):
        return [tf.nn.softmax(x[:, :, :, 0, :6], axis=-1)]


class TestComputeStatsIntegration:
    def test_1702_adapter_through_compute_stats(
        self, synthetic_side_store_ds, synthetic_targets, side_store_data_config
    ):
        model = adapter.FrontFinder1702Adapter(SoftmaxProbeModel(), lat_ascending=False)
        lats = synthetic_side_store_ds["latitude"].values
        lons = synthetic_side_store_ds["longitude"].values
        spatial_ds, aggregate_ds, derived_ds = evaluate.compute_stats(
            model=model,
            input_ds=synthetic_side_store_ds,
            target_da=synthetic_targets,
            data_config=side_store_data_config,
            front_types=FRONT_TYPES,
            lats=lats,
            lons=lons,
            spatial_mask=None,
            batch_size=4,
            class_weights=side_store_data_config.class_weights,
        )
        for front_type in FRONT_TYPES:
            assert f"csi_{front_type}" in derived_ds
            assert f"tp_{front_type}" in aggregate_ds
            assert f"tp_spatial_{front_type}" in spatial_ds
        assert derived_ds[f"spatial_csi_{FRONT_TYPES[0]}"].dims == (
            "latitude",
            "longitude",
            "neighborhood",
            "threshold",
        )
        tp_plus_fn = aggregate_ds["tp_CF"] + aggregate_ds["fn_CF"]
        assert float(tp_plus_fn.min()) >= 0.0

    def test_region_mask_restricts_statistics(self, synthetic_side_store_ds, synthetic_targets, side_store_data_config):
        model = adapter.FrontFinder1702Adapter(SoftmaxProbeModel(), lat_ascending=False)
        lats = synthetic_side_store_ds["latitude"].values
        lons = synthetic_side_store_ds["longitude"].values
        half_mask = np.zeros((N_LAT, N_LON), dtype=bool)
        half_mask[:, : N_LON // 2] = True
        _, aggregate_full, _ = evaluate.compute_stats(
            model=model,
            input_ds=synthetic_side_store_ds,
            target_da=synthetic_targets,
            data_config=side_store_data_config,
            front_types=["CF"],
            lats=lats,
            lons=lons,
            spatial_mask=None,
            batch_size=4,
            class_weights=side_store_data_config.class_weights,
        )
        _, aggregate_half, _ = evaluate.compute_stats(
            model=model,
            input_ds=synthetic_side_store_ds,
            target_da=synthetic_targets,
            data_config=side_store_data_config,
            front_types=["CF"],
            lats=lats,
            lons=lons,
            spatial_mask=half_mask,
            batch_size=4,
            class_weights=side_store_data_config.class_weights,
        )
        total_full = float((aggregate_full["tp_CF"] + aggregate_full["fp_CF"]).sum())
        total_half = float((aggregate_half["tp_CF"] + aggregate_half["fp_CF"]).sum())
        assert 0.0 < total_half < total_full


class CountingModel:
    """Wraps a model and counts Python-level calls, to verify inference isn't repeated."""

    def __init__(self, base):
        self.base = base
        self.call_count = 0

    def __call__(self, x, training=False):
        self.call_count += 1
        return self.base(x, training=training)


class TestPredictOnceAccumulatePerRegion:
    def test_model_not_reinvoked_across_multiple_regions(
        self, synthetic_side_store_ds, synthetic_targets, side_store_data_config
    ):
        """run_eval.run()'s pattern: predict_batches once, accumulate_stats per region."""
        counting = CountingModel(SoftmaxProbeModel())
        model = adapter.FrontFinder1702Adapter(counting, lat_ascending=False)
        lats = synthetic_side_store_ds["latitude"].values
        lons = synthetic_side_store_ds["longitude"].values

        all_preds, all_targets = evaluate.predict_batches(
            model=model,
            input_ds=synthetic_side_store_ds,
            target_da=synthetic_targets,
            data_config=side_store_data_config,
            batch_size=4,
            class_weights=side_store_data_config.class_weights,
        )
        call_count_after_predict = counting.call_count
        assert call_count_after_predict > 0

        for region in ("full", "land", "ocean"):
            spatial_mask = run_eval.build_region_mask(region, lats, lons)
            evaluate.accumulate_stats(
                all_preds=all_preds,
                all_targets=all_targets,
                front_types=FRONT_TYPES,
                lats=lats,
                lons=lons,
                spatial_mask=spatial_mask,
            )

        assert counting.call_count == call_count_after_predict


class TestConfigParsing:
    @pytest.mark.parametrize(
        "config_name",
        [
            "eval_1702_conus.yaml",
            "eval_1702_conus_6h.yaml",
            "eval_1702_full.yaml",
            "eval_baseline_conus.yaml",
            "eval_baseline_full.yaml",
        ],
    )
    def test_harness_configs_parse(self, config_name):
        config_path = os.path.join("configs", "model_1702", config_name)
        yaml_data = utils.load_yaml(config_path)
        harness_cfg = utils.parse_config_section(
            yaml_data, run_eval.HarnessEvalConfig, "harness_eval_config", utils.YAML_TYPE_HOOKS
        )
        data_cfg = utils.parse_config_section(yaml_data, datasets.DatasetConfig, "data_config", utils.YAML_TYPE_HOOKS)
        assert harness_cfg.model_kind in (run_eval.MODEL_KIND_1702, run_eval.MODEL_KIND_KERAS)
        assert set(harness_cfg.regions) <= set(run_eval.REGION_CHOICES)
        assert harness_cfg.front_types == FRONT_TYPES
        assert isinstance(harness_cfg.coordinates, utils.BoundingBox)
        if harness_cfg.model_kind == run_eval.MODEL_KIND_1702:
            assert data_cfg.variables == list(normalization.VARIABLES)
        assert data_cfg.volume_inputs

    def test_generate_configs_parse(self):
        from fronts.data import generate

        for config_name in ("generate_conus.yaml", "generate_full.yaml"):
            config_path = os.path.join("configs", "model_1702", config_name)
            era5_config = utils.open_config_yaml_as_dataclass(
                config_path, generate.ERA5DataLoaderConfig, config_key="era5_config", type_hooks=utils.YAML_TYPE_HOOKS
            )
            icechunk_config = utils.open_config_yaml_as_dataclass(
                config_path, utils.IcechunkStorageConfig, config_key="icechunk_storage_config"
            )
            assert era5_config.pressure_levels == [1000, 950, 900, 850]
            assert icechunk_config.group_name in ("conus", "full")


def test_stats_file_naming_matches_evaluate_convention(tmp_path):
    ds = xr.Dataset({"csi_CF": (("threshold",), np.zeros(3))}, coords={"threshold": [0.1, 0.2, 0.3]})
    for region in ("full", "land", "WPC"):
        suffix = "" if region == "full" else f"_{region}"
        ds.to_netcdf(os.path.join(tmp_path, f"stats_derived{suffix}.nc"))
    written = sorted(os.path.basename(p) for p in glob.glob(os.path.join(tmp_path, "stats_derived*.nc")))
    assert written == ["stats_derived.nc", "stats_derived_WPC.nc", "stats_derived_land.nc"]
