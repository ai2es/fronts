import numpy as np
import pytest
import xarray as xr

from fronts import utils


def _make_da(lons: list[float]) -> xr.DataArray:
    data = np.zeros(len(lons))
    return xr.DataArray(data, dims=["longitude"], coords={"longitude": lons})


def _make_ds(lons: list[float]) -> xr.Dataset:
    data = np.zeros(len(lons))
    return xr.Dataset({"var": xr.DataArray(data, dims=["longitude"], coords={"longitude": lons})})


class TestUnwrapLongitude:
    def test_already_monotonic_0_to_360(self):
        lons = [0.0, 90.0, 180.0, 270.0, 359.75]
        da = _make_da(lons)
        result = utils.unwrap_longitude(da)
        np.testing.assert_array_equal(result["longitude"].values, lons)

    def test_already_monotonic_negative_180_to_180(self):
        lons = [-180.0, -90.0, 0.0, 90.0, 180.0]
        da = _make_da(lons)
        result = utils.unwrap_longitude(da)
        np.testing.assert_array_equal(result["longitude"].values, lons)

    def test_wrap_crossing_0_to_360(self):
        lons = [130.0, 200.0, 359.75, 0.0, 9.75]
        da = _make_da(lons)
        result = utils.unwrap_longitude(da)
        expected = [130.0, 200.0, 359.75, 360.0, 369.75]
        np.testing.assert_array_almost_equal(result["longitude"].values, expected)

    def test_wrap_crossing_negative_180_to_180(self):
        lons = [90.0, 150.0, 179.75, -179.75, -90.0]
        da = _make_da(lons)
        result = utils.unwrap_longitude(da)
        expected = [90.0, 150.0, 179.75, 180.25, 270.0]
        np.testing.assert_array_almost_equal(result["longitude"].values, expected)

    def test_wrap_at_first_step(self):
        lons = [359.75, 0.0, 0.25, 0.5]
        da = _make_da(lons)
        result = utils.unwrap_longitude(da)
        expected = [359.75, 360.0, 360.25, 360.5]
        np.testing.assert_array_almost_equal(result["longitude"].values, expected)

    def test_single_element_unchanged(self):
        lons = [45.0]
        da = _make_da(lons)
        result = utils.unwrap_longitude(da)
        np.testing.assert_array_equal(result["longitude"].values, lons)

    def test_two_element_wrap(self):
        lons = [359.75, 0.0]
        da = _make_da(lons)
        result = utils.unwrap_longitude(da)
        expected = [359.75, 360.0]
        np.testing.assert_array_almost_equal(result["longitude"].values, expected)

    def test_two_element_no_wrap(self):
        lons = [0.0, 90.0]
        da = _make_da(lons)
        result = utils.unwrap_longitude(da)
        np.testing.assert_array_equal(result["longitude"].values, lons)

    def test_data_values_unchanged(self):
        lons = [350.0, 355.0, 0.0, 5.0]
        data = np.array([1.0, 2.0, 3.0, 4.0])
        da = xr.DataArray(data, dims=["longitude"], coords={"longitude": lons})
        result = utils.unwrap_longitude(da)
        np.testing.assert_array_equal(result.values, data)

    def test_dataset_wrap_crossing(self):
        lons = [130.0, 200.0, 359.75, 0.0, 9.75]
        ds = _make_ds(lons)
        result = utils.unwrap_longitude(ds)
        expected = [130.0, 200.0, 359.75, 360.0, 369.75]
        np.testing.assert_array_almost_equal(result["longitude"].values, expected)

    def test_dataset_already_monotonic(self):
        lons = [0.0, 90.0, 180.0, 270.0]
        ds = _make_ds(lons)
        result = utils.unwrap_longitude(ds)
        np.testing.assert_array_equal(result["longitude"].values, lons)


class TestEpochsPerFullPass:
    def test_paper_example(self):
        assert utils.epochs_per_full_pass(35_200, 64, 10) == 55

    def test_steps_exceed_full_pass_returns_one(self):
        assert utils.epochs_per_full_pass(100, 4, 999) == 1

    def test_full_pass_per_epoch_returns_one(self):
        assert utils.epochs_per_full_pass(800, 4, 200) == 1

    def test_exact_division(self):
        assert utils.epochs_per_full_pass(800, 4, 50) == 4

    def test_non_divisible_samples_round_up(self):
        assert utils.epochs_per_full_pass(101, 4, 5) == 6

    def test_no_samples_returns_zero(self):
        assert utils.epochs_per_full_pass(0, 4, 10) == 0

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError):
            utils.epochs_per_full_pass(100, 0, 10)

    def test_invalid_steps_per_epoch_raises(self):
        with pytest.raises(ValueError):
            utils.epochs_per_full_pass(100, 4, 0)

    def test_negative_samples_raises(self):
        with pytest.raises(ValueError):
            utils.epochs_per_full_pass(-1, 4, 10)


def _make_times(start: str = "2018-01-01", periods: int = 400, freq_hours: int = 24) -> np.ndarray:
    """Return a datetime64 array at regular hourly intervals spanning multiple seasons."""
    base = np.datetime64(start, "h")
    return base + np.arange(periods) * np.timedelta64(freq_hours, "h")


class TestSplitByYear:
    @pytest.fixture
    def times(self) -> np.ndarray:
        return _make_times(start="2017-01-01", periods=1460, freq_hours=6)

    def test_timesteps_assigned_by_year(self, times):
        train_mask, val_mask, test_mask = utils.split_by_year(times, test_years=[2019], val_years=[2018])
        years = times.astype("datetime64[Y]").astype(int) + 1970
        np.testing.assert_array_equal(test_mask, years == 2019)
        np.testing.assert_array_equal(val_mask, years == 2018)
        np.testing.assert_array_equal(train_mask, ~np.isin(years, [2018, 2019]))

    def test_masks_mutually_exclusive_and_exhaustive(self, times):
        train_mask, val_mask, test_mask = utils.split_by_year(times, test_years=[2019], val_years=[2018])
        stacked = np.stack([train_mask, val_mask, test_mask])
        assert (stacked.sum(axis=0) == 1).all()

    def test_train_is_complement_of_test_and_val(self, times):
        train_mask, val_mask, test_mask = utils.split_by_year(times, test_years=[2019], val_years=[2018])
        np.testing.assert_array_equal(train_mask, ~(val_mask | test_mask))

    def test_overlapping_years_raises(self, times):
        with pytest.raises(ValueError):
            utils.split_by_year(times, test_years=[2018], val_years=[2018])

    def test_empty_years_puts_everything_in_train(self, times):
        train_mask, val_mask, test_mask = utils.split_by_year(times, test_years=[], val_years=[])
        assert train_mask.all()
        assert not val_mask.any()
        assert not test_mask.any()

    def test_year_boundary_handled_correctly(self):
        times = np.array(["2018-12-31T18:00", "2019-01-01T00:00"], dtype="datetime64[h]")
        train_mask, val_mask, test_mask = utils.split_by_year(times, test_years=[2019], val_years=[2018])
        np.testing.assert_array_equal(val_mask, [True, False])
        np.testing.assert_array_equal(test_mask, [False, True])
        np.testing.assert_array_equal(train_mask, [False, False])


class TestSlurmCpuCount:
    def test_uses_slurm_cpus_per_task_when_set(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
        assert utils.slurm_cpu_count() == 8

    def test_falls_back_to_cpu_count_outside_slurm(self, monkeypatch):
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        monkeypatch.setattr("os.cpu_count", lambda: 12)
        assert utils.slurm_cpu_count() == 12

    def test_falls_back_to_one_when_cpu_count_unknown(self, monkeypatch):
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        monkeypatch.setattr("os.cpu_count", lambda: None)
        assert utils.slurm_cpu_count() == 1


class TestParseConfigSectionFloatCoercion:
    def test_yaml_scientific_notation_string_coerces_to_float(self):
        """PyYAML parses 1e-6 (no decimal point) as a string; float fields must still come out numeric."""
        import dataclasses

        from fronts import utils

        @dataclasses.dataclass
        class DecayConfig:
            learning_rate_minimum: float | None = None
            learning_rate_decay_factor: float | None = None

        yaml_data = {"callbacks": {"learning_rate_minimum": "1e-6", "learning_rate_decay_factor": 0.2}}
        cfg = utils.parse_config_section(yaml_data, DecayConfig, "callbacks")
        assert cfg.learning_rate_minimum == pytest.approx(1e-6)
        assert isinstance(cfg.learning_rate_minimum, float)
        assert cfg.learning_rate_decay_factor == pytest.approx(0.2)

    def test_none_values_pass_through(self):
        import dataclasses

        from fronts import utils

        @dataclasses.dataclass
        class DecayConfig:
            learning_rate_minimum: float | None = None

        cfg = utils.parse_config_section({"callbacks": {"learning_rate_minimum": None}}, DecayConfig, "callbacks")
        assert cfg.learning_rate_minimum is None
