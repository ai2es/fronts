import dataclasses
import pathlib
import tempfile
from datetime import datetime

import dacite
import icechunk as ic
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from fronts.data import config, generate
from fronts.utils import BoundingBox

ERA5_VARS = [
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
]
LEVELS = [1000, 950, 900, 850, 700, 500]
LAT = np.linspace(45.0, 25.0, 8)  # descending (N→S) to match ERA5 convention
LON = np.linspace(-110.0, -70.0, 8)
LON_WRAP = np.array([330.0, 350.0, 0.0, 20.0])  # wraps past 360: physical domain 330→380

YAML_PATH = pathlib.Path(__file__).parent / "test_generate.yaml"


@pytest.fixture
def time_range() -> pd.DatetimeIndex:
    return pd.date_range("2019-01-01", periods=4, freq="6h")


@pytest.fixture
def minimal_ds(time_range: pd.DatetimeIndex) -> xr.Dataset:
    rng = np.random.default_rng(0)
    data_vars = {
        var: xr.DataArray(
            rng.standard_normal((4, 6, 8, 8)).astype(np.float32),
            dims=["time", "level", "latitude", "longitude"],
            coords={
                "time": time_range,
                "level": LEVELS,
                "latitude": LAT,
                "longitude": LON,
            },
        )
        for var in ERA5_VARS
    }
    return xr.Dataset(data_vars)


@pytest.fixture
def era5_zarr(tmp_path: pathlib.Path, minimal_ds: xr.Dataset) -> pathlib.Path:
    path = tmp_path / "era5.zarr"
    minimal_ds.to_zarr(path)
    return path


@pytest.fixture
def era5_config(era5_zarr: pathlib.Path) -> config.ERA5DataLoaderConfig:
    return config.ERA5DataLoaderConfig(
        era5_uri=str(era5_zarr),
        variables=ERA5_VARS,
        pressure_levels=LEVELS,
        time_start=datetime(2019, 1, 1),
        time_end=datetime(2019, 1, 1, 18),
        time_resolution="6h",
        coordinates=BoundingBox(lat_min=25.0, lat_max=45.0, lon_min=-110.0, lon_max=-70.0),
        storage_options=None,
        chunks={"time": 1},
    )


@pytest.fixture
def storage_config(tmp_path: pathlib.Path) -> config.IcechunkStorageConfig:
    return config.IcechunkStorageConfig(
        store_path=str(tmp_path / "icechunk_store"),
        branch_name="main",
    )


@pytest.fixture
def write_ds(time_range: pd.DatetimeIndex) -> xr.Dataset:
    rng = np.random.default_rng(42)
    return xr.Dataset(
        {
            "temperature": xr.DataArray(
                rng.standard_normal((4, 6, 8, 8)).astype(np.float32),
                dims=["time", "level", "latitude", "longitude"],
                coords={
                    "time": time_range,
                    "level": LEVELS,
                    "latitude": LAT,
                    "longitude": LON,
                },
            )
        }
    )


@pytest.fixture
def populated_store(storage_config: config.IcechunkStorageConfig, write_ds: xr.Dataset) -> config.IcechunkStorageConfig:
    generate.write_or_append_icechunk_store(storage_config, write_ds)
    return storage_config


@pytest.fixture
def time_range_3h() -> pd.DatetimeIndex:
    return pd.date_range("2019-01-01", "2019-01-01 18:00", freq="3h")


@pytest.fixture
def minimal_ds_3h(time_range_3h: pd.DatetimeIndex) -> xr.Dataset:
    rng = np.random.default_rng(2)
    data_vars = {
        var: xr.DataArray(
            rng.standard_normal((len(time_range_3h), 6, 8, 8)).astype(np.float32),
            dims=["time", "level", "latitude", "longitude"],
            coords={"time": time_range_3h, "level": LEVELS, "latitude": LAT, "longitude": LON},
        )
        for var in ERA5_VARS
    }
    return xr.Dataset(data_vars)


@pytest.fixture
def era5_zarr_3h(tmp_path: pathlib.Path, minimal_ds_3h: xr.Dataset) -> pathlib.Path:
    path = tmp_path / "era5_3h.zarr"
    minimal_ds_3h.to_zarr(path)
    return path


@pytest.fixture
def new_variable_ds(time_range: pd.DatetimeIndex) -> xr.Dataset:
    rng = np.random.default_rng(7)
    return xr.Dataset(
        {
            "vertical_velocity": xr.DataArray(
                rng.standard_normal((4, 6, 8, 8)).astype(np.float32),
                dims=["time", "level", "latitude", "longitude"],
                coords={
                    "time": time_range,
                    "level": LEVELS,
                    "latitude": LAT,
                    "longitude": LON,
                },
            )
        }
    )


def _make_store_contents(
    variables: list[str] | None = None,
    times: pd.DatetimeIndex | None = None,
    levels: list[int] | None = None,
    coordinates: BoundingBox | None = None,
) -> generate.StoreContents:
    return generate.StoreContents(
        variables=variables if variables is not None else list(ERA5_VARS),
        times=times if times is not None else pd.date_range("2019-01-01", periods=4, freq="6h"),
        levels=levels if levels is not None else list(LEVELS),
        coordinates=coordinates if coordinates is not None else BoundingBox(25.0, 45.0, -110.0, -70.0),
    )


class TestGenerateEra5Data:
    def test_returns_dataset(self, era5_config):
        result = generate.generate_era5_data(era5_config)
        assert isinstance(result, xr.Dataset)

    def test_variable_subset(self, era5_config):
        subset_vars = ["temperature", "geopotential"]
        era5_config.variables = subset_vars
        result = generate.generate_era5_data(era5_config)
        assert set(result.data_vars) == set(subset_vars)

    def test_all_requested_variables_present(self, era5_config):
        result = generate.generate_era5_data(era5_config)
        for var in era5_config.variables:
            assert var in result.data_vars

    def test_time_range_filter(self, era5_config):
        result = generate.generate_era5_data(era5_config)
        times = pd.DatetimeIndex(result.time.values)
        assert (times >= era5_config.time_start).all()
        assert (times <= era5_config.time_end).all()

    def test_geographic_subset(self, era5_config):
        result = generate.generate_era5_data(era5_config)
        assert result.latitude.min() >= era5_config.coordinates.lat_min
        assert result.latitude.max() <= era5_config.coordinates.lat_max
        assert result.longitude.min() >= era5_config.coordinates.lon_min
        assert result.longitude.max() <= era5_config.coordinates.lon_max


class TestWriteOrAppendIcechunkStore:
    def test_creates_new_store(self, storage_config, write_ds):
        storage = ic.local_filesystem_storage(storage_config.store_path)
        assert not ic.Repository.exists(storage)
        generate.write_or_append_icechunk_store(storage_config, write_ds)
        assert ic.Repository.exists(storage)

    def test_data_roundtrip(self, storage_config, write_ds):
        generate.write_or_append_icechunk_store(storage_config, write_ds)
        storage = ic.local_filesystem_storage(storage_config.store_path)
        repo = ic.Repository.open(storage)
        session = repo.readonly_session("main")
        result = xr.open_zarr(session.store)
        np.testing.assert_array_equal(result["temperature"].values, write_ds["temperature"].values)

    def test_append_increases_time_steps(self, storage_config, write_ds):
        second_time = pd.date_range("2019-01-02", periods=4, freq="6h")
        second_ds = write_ds.assign_coords(time=second_time)

        generate.write_or_append_icechunk_store(storage_config, write_ds)
        generate.write_or_append_icechunk_store(storage_config, second_ds)

        storage = ic.local_filesystem_storage(storage_config.store_path)
        repo = ic.Repository.open(storage)
        session = repo.readonly_session("main")
        result = xr.open_zarr(session.store)
        assert result.sizes["time"] == len(write_ds.time) + len(second_ds.time)


class TestInspectStore:
    def test_returns_none_for_nonexistent_store(self, storage_config):
        assert generate.inspect_store(storage_config) is None

    def test_variables_match_after_write(self, populated_store):
        result = generate.inspect_store(populated_store)
        assert result is not None
        assert set(result.variables) == {"temperature"}

    def test_times_match_after_write(self, populated_store, time_range):
        result = generate.inspect_store(populated_store)
        assert result is not None
        np.testing.assert_array_equal(result.times.astype("datetime64[us]"), time_range.values)

    def test_levels_match_after_write(self, populated_store):
        result = generate.inspect_store(populated_store)
        assert result is not None
        assert result.levels == LEVELS

    def test_coordinates_match_after_write(self, populated_store):
        result = generate.inspect_store(populated_store)
        assert result is not None
        assert result.coordinates.lat_min == pytest.approx(float(LAT.min()))
        assert result.coordinates.lat_max == pytest.approx(float(LAT.max()))
        assert result.coordinates.lon_min == pytest.approx(float(LON.min()))
        assert result.coordinates.lon_max == pytest.approx(float(LON.max()))

    def test_wraparound_longitude_coordinates(self, storage_config, time_range):
        rng = np.random.default_rng(5)
        wrap_ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    rng.standard_normal((4, 6, 4, 4)).astype(np.float32),
                    dims=["time", "level", "latitude", "longitude"],
                    coords={"time": time_range, "level": LEVELS, "latitude": LAT[:4], "longitude": LON_WRAP},
                )
            }
        )
        generate.write_or_append_icechunk_store(storage_config, wrap_ds)
        result = generate.inspect_store(storage_config)
        assert result is not None
        assert result.coordinates.lon_min == pytest.approx(330.0)
        assert result.coordinates.lon_max == pytest.approx(380.0)


class TestDetermineWriteStrategy:
    def _base_config(self, era5_zarr) -> config.ERA5DataLoaderConfig:
        return config.ERA5DataLoaderConfig(
            era5_uri=str(era5_zarr),
            variables=list(ERA5_VARS),
            pressure_levels=list(LEVELS),
            time_start=datetime(2019, 1, 1),
            time_end=datetime(2019, 1, 1, 18),
            time_resolution="6h",
            coordinates=BoundingBox(lat_min=25.0, lat_max=45.0, lon_min=-110.0, lon_max=-70.0),
            storage_options=None,
            chunks={"time": 1},
        )

    def test_no_store_returns_full_write(self, era5_zarr):
        cfg = self._base_config(era5_zarr)
        strategy = generate.determine_write_strategy(cfg, None)
        assert strategy.missing_variables == list(ERA5_VARS)
        assert len(strategy.missing_times) == 4
        assert not strategy.skip_reason
        assert not strategy.error_reason

    def test_all_present_returns_skip(self, era5_zarr):
        cfg = self._base_config(era5_zarr)
        store = _make_store_contents()
        strategy = generate.determine_write_strategy(cfg, store)
        assert strategy.skip_reason
        assert not strategy.error_reason
        assert not strategy.missing_variables
        assert len(strategy.missing_times) == 0

    def test_append_tail_times_returns_missing_times(self, era5_zarr):
        cfg = dataclasses.replace(
            self._base_config(era5_zarr),
            time_end=datetime(2019, 1, 2, 0),
        )
        store = _make_store_contents(times=pd.date_range("2019-01-01", periods=4, freq="6h"))
        strategy = generate.determine_write_strategy(cfg, store)
        assert len(strategy.missing_times) > 0
        assert strategy.missing_times[0] > store.times[-1]
        assert not strategy.error_reason

    def test_resolution_change_requires_merge(self, era5_zarr):
        cfg = dataclasses.replace(self._base_config(era5_zarr), time_resolution="3h")
        store = _make_store_contents()
        strategy = generate.determine_write_strategy(cfg, store)
        assert strategy.merge_required
        assert not strategy.error_reason
        assert not strategy.skip_reason
        assert len(strategy.missing_times) > 0

    def test_append_tail_does_not_require_merge(self, era5_zarr):
        cfg = dataclasses.replace(self._base_config(era5_zarr), time_end=datetime(2019, 1, 2, 0))
        store = _make_store_contents(times=pd.date_range("2019-01-01", periods=4, freq="6h"))
        strategy = generate.determine_write_strategy(cfg, store)
        assert not strategy.merge_required

    def test_level_mismatch_returns_error(self, era5_zarr):
        cfg = dataclasses.replace(self._base_config(era5_zarr), pressure_levels=[1000, 850])
        store = _make_store_contents()
        strategy = generate.determine_write_strategy(cfg, store)
        assert strategy.error_reason

    def test_coordinates_mismatch_returns_error(self, era5_zarr):
        cfg = dataclasses.replace(
            self._base_config(era5_zarr),
            coordinates=BoundingBox(lat_min=30.0, lat_max=50.0, lon_min=-110.0, lon_max=-70.0),
        )
        store = _make_store_contents()
        strategy = generate.determine_write_strategy(cfg, store)
        assert strategy.error_reason

    def test_missing_vars_only_returns_missing_variables(self, era5_zarr):
        cfg = self._base_config(era5_zarr)
        store = _make_store_contents(variables=["temperature"])
        strategy = generate.determine_write_strategy(cfg, store)
        assert set(strategy.missing_variables) == set(ERA5_VARS) - {"temperature"}
        assert len(strategy.missing_times) == 0
        assert not strategy.error_reason

    def test_missing_vars_and_times_returns_error(self, era5_zarr):
        cfg = dataclasses.replace(
            self._base_config(era5_zarr),
            time_end=datetime(2019, 1, 2, 0),
        )
        store = _make_store_contents(variables=["temperature"])
        strategy = generate.determine_write_strategy(cfg, store)
        assert strategy.error_reason


class TestWriteNewVariablesToIcechunkStore:
    def test_new_variable_present_after_write(self, storage_config, write_ds, new_variable_ds):
        generate.write_or_append_icechunk_store(storage_config, write_ds)
        generate.write_new_variables_to_icechunk_store(storage_config, new_variable_ds)

        storage = ic.local_filesystem_storage(storage_config.store_path)
        repo = ic.Repository.open(storage)
        session = repo.readonly_session("main")
        result = xr.open_zarr(session.store, consolidated=False)
        assert "vertical_velocity" in result.data_vars

    def test_existing_variable_preserved_after_new_write(self, storage_config, write_ds, new_variable_ds):
        generate.write_or_append_icechunk_store(storage_config, write_ds)
        generate.write_new_variables_to_icechunk_store(storage_config, new_variable_ds)

        storage = ic.local_filesystem_storage(storage_config.store_path)
        repo = ic.Repository.open(storage)
        session = repo.readonly_session("main")
        result = xr.open_zarr(session.store, consolidated=False)
        assert "temperature" in result.data_vars
        np.testing.assert_array_equal(result["temperature"].values, write_ds["temperature"].values)

    def test_new_variable_data_roundtrip(self, storage_config, write_ds, new_variable_ds):
        generate.write_or_append_icechunk_store(storage_config, write_ds)
        generate.write_new_variables_to_icechunk_store(storage_config, new_variable_ds)

        storage = ic.local_filesystem_storage(storage_config.store_path)
        repo = ic.Repository.open(storage)
        session = repo.readonly_session("main")
        result = xr.open_zarr(session.store, consolidated=False)
        np.testing.assert_array_equal(result["vertical_velocity"].values, new_variable_ds["vertical_velocity"].values)


class TestAttributePreservation:
    def test_write_or_append_preserves_global_attrs(self, storage_config, write_ds):
        write_ds.attrs = {"source": "era5", "version": "1"}
        generate.write_or_append_icechunk_store(storage_config, write_ds)

        storage = ic.local_filesystem_storage(storage_config.store_path)
        repo = ic.Repository.open(storage)
        session = repo.readonly_session("main")
        result = xr.open_zarr(session.store, consolidated=False)
        assert result.attrs.get("source") == "era5"
        assert result.attrs.get("version") == "1"

    def test_write_new_variables_preserves_var_attrs(self, storage_config, write_ds, new_variable_ds):
        new_variable_ds["vertical_velocity"].attrs = {"units": "Pa s-1", "long_name": "Vertical velocity"}
        generate.write_or_append_icechunk_store(storage_config, write_ds)
        generate.write_new_variables_to_icechunk_store(storage_config, new_variable_ds)

        storage = ic.local_filesystem_storage(storage_config.store_path)
        repo = ic.Repository.open(storage)
        session = repo.readonly_session("main")
        result = xr.open_zarr(session.store, consolidated=False)
        assert result["vertical_velocity"].attrs.get("units") == "Pa s-1"
        assert result["vertical_velocity"].attrs.get("long_name") == "Vertical velocity"


class TestTimeResolution:
    def _config(self, era5_zarr: pathlib.Path, resolution: str) -> config.ERA5DataLoaderConfig:
        return config.ERA5DataLoaderConfig(
            era5_uri=str(era5_zarr),
            variables=list(ERA5_VARS),
            pressure_levels=list(LEVELS),
            time_start=datetime(2019, 1, 1),
            time_end=datetime(2019, 1, 1, 18),
            time_resolution=resolution,
            coordinates=BoundingBox(lat_min=25.0, lat_max=45.0, lon_min=-110.0, lon_max=-70.0),
            storage_options=None,
            chunks={"time": 1},
        )

    def test_6h_to_3h_requires_merge(self, era5_zarr_3h):
        store = _make_store_contents(times=pd.date_range("2019-01-01", "2019-01-01 18:00", freq="6h"))
        strategy = generate.determine_write_strategy(self._config(era5_zarr_3h, "3h"), store)
        assert strategy.merge_required
        assert not strategy.error_reason

    def test_6h_to_3h_missing_times_are_gaps(self, era5_zarr_3h):
        store = _make_store_contents(times=pd.date_range("2019-01-01", "2019-01-01 18:00", freq="6h"))
        strategy = generate.determine_write_strategy(self._config(era5_zarr_3h, "3h"), store)
        expected = pd.DatetimeIndex([datetime(2019, 1, 1, 3), datetime(2019, 1, 1, 9), datetime(2019, 1, 1, 15)])
        np.testing.assert_array_equal(strategy.missing_times.astype("datetime64[us]"), expected.values)

    def test_3h_to_6h_is_skip(self, era5_zarr):
        store = _make_store_contents(times=pd.date_range("2019-01-01", "2019-01-01 18:00", freq="3h"))
        strategy = generate.determine_write_strategy(self._config(era5_zarr, "6h"), store)
        assert strategy.skip_reason
        assert not strategy.error_reason

    def test_same_resolution_same_range_is_skip(self, era5_zarr):
        store = _make_store_contents(times=pd.date_range("2019-01-01", "2019-01-01 18:00", freq="6h"))
        strategy = generate.determine_write_strategy(self._config(era5_zarr, "6h"), store)
        assert strategy.skip_reason

    def test_6h_to_3h_end_to_end_has_all_3h_steps(self, storage_config, era5_zarr_3h, minimal_ds):
        generate.write_or_append_icechunk_store(storage_config, minimal_ds)
        cfg = self._config(era5_zarr_3h, "3h")
        strategy = generate.determine_write_strategy(cfg, generate.inspect_store(storage_config))
        strategy.execute(cfg, storage_config)

        result = generate.inspect_store(storage_config)
        assert result is not None
        assert len(result.times) == len(pd.date_range("2019-01-01", "2019-01-01 18:00", freq="3h"))

    def test_6h_to_3h_end_to_end_preserves_existing_data(self, storage_config, era5_zarr_3h, minimal_ds):
        generate.write_or_append_icechunk_store(storage_config, minimal_ds)
        original_times = pd.DatetimeIndex(minimal_ds.time.values)
        cfg = self._config(era5_zarr_3h, "3h")
        strategy = generate.determine_write_strategy(cfg, generate.inspect_store(storage_config))
        strategy.execute(cfg, storage_config)

        storage = ic.local_filesystem_storage(storage_config.store_path)
        repo = ic.Repository.open(storage)
        session = repo.readonly_session("main")
        result = xr.open_zarr(session.store, consolidated=False)
        np.testing.assert_array_equal(
            result.sel(time=original_times.astype("datetime64[ns]"))["temperature"].values,
            minimal_ds["temperature"].values,
        )


class TestYamlConfigLoading:
    def test_era5_data_config_from_yaml(self):
        with open(YAML_PATH) as f:
            raw = yaml.safe_load(f)

        result = dacite.from_dict(
            data_class=config.ERA5DataLoaderConfig,
            data=raw["era5_data_config"],
            config=dacite.Config(
                type_hooks={BoundingBox: lambda x: BoundingBox(*x)},
                check_types=False,
            ),
        )
        assert result.era5_uri == "/tmp/test_era5.zarr"
        assert result.variables == ["temperature"]
        assert result.pressure_levels == [1000, 850, 500]
        assert result.time_resolution == "6h"
        assert result.storage_options is None
        assert result.coordinates[0] == 20.0  # lat_min
        assert result.coordinates[1] == 50.0  # lat_max

    def test_icechunk_storage_config_from_yaml(self):
        with open(YAML_PATH) as f:
            import yaml

            raw = yaml.safe_load(f)
        import dacite

        result = dacite.from_dict(
            data_class=config.IcechunkStorageConfig,
            data=raw["icechunk_storage_config"],
            config=dacite.Config(cast=[tuple, datetime], check_types=False),
        )
        assert result.store_path == "/tmp/test_store"
        assert result.branch_name == "main"
        assert result.commit_message == "test commit"


_BASE_CONFIG = config.ERA5DataLoaderConfig(
    era5_uri="/dev/null",
    variables=list(ERA5_VARS),
    pressure_levels=list(LEVELS),
    time_start=datetime(2019, 1, 1),
    time_end=datetime(2019, 1, 1, 18),
    time_resolution="6h",
    coordinates=BoundingBox(lat_min=25.0, lat_max=45.0, lon_min=-110.0, lon_max=-70.0),
    storage_options=None,
    chunks={"time": 1},
)

_PROP_LEVELS = [1000, 500]
_PROP_LAT = np.linspace(45.0, 25.0, 4)
_PROP_LON = np.linspace(-110.0, -70.0, 4)


@st.composite
def time_index_strategy(draw, min_size=1, max_size=6):
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    freq = draw(st.sampled_from(["3h", "6h", "12h"]))
    # Build start from integers to avoid sub-second precision, which overflows
    # icechunk's CF-convention time encoding when written to zarr.
    year = draw(st.integers(2010, 2017))
    month = draw(st.integers(1, 12))
    day = draw(st.integers(1, 28))
    hour = draw(st.integers(0, 23))
    return pd.date_range(datetime(year, month, day, hour), periods=n, freq=freq)


var_list_strategy = st.lists(st.sampled_from(ERA5_VARS), unique=True, min_size=1, max_size=len(ERA5_VARS))

attr_strategy = st.dictionaries(
    keys=st.text(alphabet=st.characters(whitelist_categories=("L",)), min_size=1, max_size=20),
    values=st.text(max_size=50),
    max_size=5,
)


def _prop_ds(times: pd.DatetimeIndex, var_names: list[str]) -> xr.Dataset:
    rng = np.random.default_rng(0)
    shape = (len(times), len(_PROP_LEVELS), len(_PROP_LAT), len(_PROP_LON))
    return xr.Dataset(
        {
            v: xr.DataArray(
                rng.standard_normal(shape).astype(np.float32),
                dims=["time", "level", "latitude", "longitude"],
                coords={"time": times, "level": _PROP_LEVELS, "latitude": _PROP_LAT, "longitude": _PROP_LON},
            )
            for v in var_names
        }
    )


def _prop_storage(tmp_dir: str) -> config.IcechunkStorageConfig:
    return config.IcechunkStorageConfig(store_path=str(pathlib.Path(tmp_dir) / "store"), branch_name="main")


class TestPropertyBasedStrategy:
    @given(
        n=st.integers(1, 6),
        freq=st.sampled_from(["3h", "6h", "12h"]),
        year=st.integers(2010, 2017),
        month=st.integers(1, 12),
        day=st.integers(1, 28),
        variables=var_list_strategy,
    )
    def test_skip_when_all_present(self, n, freq, year, month, day, variables):
        start = datetime(year, month, day)
        times = pd.date_range(start, periods=n, freq=freq)
        cfg = dataclasses.replace(
            _BASE_CONFIG,
            variables=variables,
            time_start=times[0].to_pydatetime(),
            time_end=times[-1].to_pydatetime(),
            time_resolution=freq,
        )
        store = _make_store_contents(variables=variables, times=times)
        strategy = generate.determine_write_strategy(cfg, store)
        assert strategy.skip_reason
        assert not strategy.error_reason

    @given(
        n=st.integers(2, 5),
        year=st.integers(2010, 2017),
        month=st.integers(1, 12),
        day=st.integers(1, 28),
    )
    def test_merge_required_when_times_interleave(self, n, year, month, day):
        start = datetime(year, month, day)
        # Store has 6h times; request 3h — the 3h gaps are interleaved with existing 6h steps
        existing = pd.date_range(start, periods=n, freq="6h")
        cfg = dataclasses.replace(
            _BASE_CONFIG,
            time_start=existing[0].to_pydatetime(),
            time_end=existing[-1].to_pydatetime(),
            time_resolution="3h",
        )
        store = _make_store_contents(times=existing)
        strategy = generate.determine_write_strategy(cfg, store)
        assert strategy.merge_required
        assert not strategy.error_reason

    @given(
        n_existing=st.integers(1, 4),
        n_extra=st.integers(1, 4),
        year=st.integers(2010, 2017),
        month=st.integers(1, 12),
        day=st.integers(1, 28),
        freq=st.sampled_from(["3h", "6h", "12h"]),
    )
    def test_no_merge_for_strict_tail_append(self, n_existing, n_extra, year, month, day, freq):
        start = datetime(year, month, day)
        existing = pd.date_range(start, periods=n_existing, freq=freq)
        extra = pd.date_range(existing[-1] + pd.Timedelta(freq), periods=n_extra, freq=freq)
        cfg = dataclasses.replace(
            _BASE_CONFIG,
            time_start=existing[0].to_pydatetime(),
            time_end=extra[-1].to_pydatetime(),
            time_resolution=freq,
        )
        store = _make_store_contents(times=existing)
        strategy = generate.determine_write_strategy(cfg, store)
        assert not strategy.merge_required
        assert not strategy.error_reason

    @given(
        levels_a=st.lists(st.integers(1, 1000), unique=True, min_size=1, max_size=8),
        levels_b=st.lists(st.integers(1, 1000), unique=True, min_size=1, max_size=8),
    )
    def test_level_mismatch_always_errors(self, levels_a, levels_b):
        assume(sorted(levels_a) != sorted(levels_b))
        cfg = dataclasses.replace(_BASE_CONFIG, pressure_levels=sorted(levels_a))
        store = _make_store_contents(levels=sorted(levels_b))
        strategy = generate.determine_write_strategy(cfg, store)
        assert strategy.error_reason

    @given(
        variables=st.lists(st.sampled_from(ERA5_VARS), unique=True, min_size=2, max_size=len(ERA5_VARS)),
        n_total=st.integers(2, 6),
        year=st.integers(2010, 2017),
        month=st.integers(1, 12),
        day=st.integers(1, 28),
        freq=st.sampled_from(["3h", "6h", "12h"]),
        data=st.data(),
    )
    def test_missing_vars_and_times_always_errors(self, variables, n_total, year, month, day, freq, data):
        start = datetime(year, month, day)
        all_times = pd.date_range(start, periods=n_total, freq=freq)
        n_stored = data.draw(st.integers(1, n_total - 1))
        subset_vars = data.draw(
            st.lists(st.sampled_from(variables), unique=True, min_size=1, max_size=len(variables) - 1)
        )
        cfg = dataclasses.replace(
            _BASE_CONFIG,
            variables=variables,
            time_start=all_times[0].to_pydatetime(),
            time_end=all_times[-1].to_pydatetime(),
            time_resolution=freq,
        )
        store = _make_store_contents(variables=subset_vars, times=all_times[:n_stored])
        strategy = generate.determine_write_strategy(cfg, store)
        assert strategy.error_reason


class TestPropertyBasedIO:
    @given(times=time_index_strategy())
    @settings(max_examples=20, deadline=None)
    def test_time_count_preserved_after_write(self, times):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sc = _prop_storage(tmp_dir)
            generate.write_or_append_icechunk_store(sc, _prop_ds(times, ["temperature"]))
            result = generate.inspect_store(sc)
        assert result is not None
        assert len(result.times) == len(times)

    @given(times_a=time_index_strategy(), times_b=time_index_strategy())
    @settings(max_examples=20, deadline=None)
    def test_append_accumulates_times(self, times_a, times_b):
        assume(not times_a.isin(times_b).any())
        with tempfile.TemporaryDirectory() as tmp_dir:
            sc = _prop_storage(tmp_dir)
            generate.write_or_append_icechunk_store(sc, _prop_ds(times_a, ["temperature"]))
            generate.write_or_append_icechunk_store(sc, _prop_ds(times_b, ["temperature"]))
            result = generate.inspect_store(sc)
        assert result is not None
        assert len(result.times) == len(times_a) + len(times_b)

    @given(variables=var_list_strategy, times=time_index_strategy())
    @settings(max_examples=20, deadline=None)
    def test_variable_names_roundtrip(self, variables, times):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sc = _prop_storage(tmp_dir)
            generate.write_or_append_icechunk_store(sc, _prop_ds(times, variables))
            result = generate.inspect_store(sc)
        assert result is not None
        assert set(result.variables) == set(variables)

    @given(attrs=attr_strategy, times=time_index_strategy())
    @settings(max_examples=20, deadline=None)
    def test_global_attrs_roundtrip(self, attrs, times):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sc = _prop_storage(tmp_dir)
            ds = _prop_ds(times, ["temperature"])
            ds.attrs = attrs
            generate.write_or_append_icechunk_store(sc, ds)
            storage = ic.local_filesystem_storage(sc.store_path)
            repo = ic.Repository.open(storage)
            result = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
        for key, val in attrs.items():
            assert result.attrs.get(key) == val

    @given(attrs=attr_strategy, times=time_index_strategy())
    @settings(max_examples=20, deadline=None)
    def test_var_attrs_roundtrip(self, attrs, times):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sc = _prop_storage(tmp_dir)
            generate.write_or_append_icechunk_store(sc, _prop_ds(times, ["temperature"]))
            new_ds = _prop_ds(times, ["specific_humidity"])
            new_ds["specific_humidity"].attrs = attrs
            generate.write_new_variables_to_icechunk_store(sc, new_ds)
            storage = ic.local_filesystem_storage(sc.store_path)
            repo = ic.Repository.open(storage)
            result = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
        for key, val in attrs.items():
            assert result["specific_humidity"].attrs.get(key) == val

    @given(
        lat_size_a=st.integers(2, 5),
        lon_size_a=st.integers(2, 5),
        lat_size_b=st.integers(2, 5),
        lon_size_b=st.integers(2, 5),
        times=time_index_strategy(),
    )
    @settings(max_examples=20, deadline=None)
    def test_dimension_mismatch_raises(self, lat_size_a, lon_size_a, lat_size_b, lon_size_b, times):
        assume(lat_size_a != lat_size_b or lon_size_a != lon_size_b)
        lat_a = np.linspace(45.0, 25.0, lat_size_a)
        lon_a = np.linspace(-110.0, -70.0, lon_size_a)
        lat_b = np.linspace(45.0, 25.0, lat_size_b)
        lon_b = np.linspace(-110.0, -70.0, lon_size_b)

        def _ds(lat, lon, var):
            rng = np.random.default_rng(0)
            return xr.Dataset(
                {
                    var: xr.DataArray(
                        rng.standard_normal((len(times), 2, len(lat), len(lon))).astype(np.float32),
                        dims=["time", "level", "latitude", "longitude"],
                        coords={"time": times, "level": _PROP_LEVELS, "latitude": lat, "longitude": lon},
                    )
                }
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            sc = _prop_storage(tmp_dir)
            generate.write_or_append_icechunk_store(sc, _ds(lat_a, lon_a, "temperature"))
            with pytest.raises(ValueError):
                generate.write_new_variables_to_icechunk_store(sc, _ds(lat_b, lon_b, "specific_humidity"))
