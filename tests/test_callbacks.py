"""Tests for fronts.callbacks: W&B metric consolidation and test-set visualization helpers."""

import math
import os
from typing import ClassVar

import numpy as np
import pytest
import xarray as xr

from fronts import constants

fc = pytest.importorskip("fronts.callbacks")


class TestMetricsConsolidationCallback:
    def test_aggregates_hss_and_strips_per_output_keys(self):
        logs = {
            "loss": 1.0,
            "sup1_softmax_hss": 0.1,
            "sup1_softmax_loss": 0.4,
            "sup2_softmax_hss": 0.3,
            "sup2_softmax_loss": 0.2,
            "val_loss": 1.5,
            "val_sup1_softmax_hss": 0.2,
            "val_sup1_softmax_loss": 0.5,
            "val_sup2_softmax_hss": 0.4,
            "val_sup2_softmax_loss": 0.3,
        }
        fc.MetricsConsolidationCallback().on_epoch_end(0, logs)

        assert logs == {
            "loss": 1.0,
            "val_loss": 1.5,
            "hss": pytest.approx(0.2),
            "val_hss": pytest.approx(0.3),
        }

    def test_aggregates_multiple_custom_metrics_independently(self):
        """A second custom metric (e.g. hss_hard) must aggregate to its own key, not hss's."""
        logs = {
            "loss": 1.0,
            "sup1_softmax_hss": 0.1,
            "sup1_softmax_hss_hard": 0.6,
            "sup1_softmax_loss": 0.4,
            "sup2_softmax_hss": 0.3,
            "sup2_softmax_hss_hard": 0.8,
            "sup2_softmax_loss": 0.2,
        }
        fc.MetricsConsolidationCallback().on_epoch_end(0, logs)

        assert logs == {
            "loss": 1.0,
            "hss": pytest.approx(0.2),
            "hss_hard": pytest.approx(0.7),
        }

    def test_noop_on_empty_logs(self):
        logs = {}
        fc.MetricsConsolidationCallback().on_epoch_end(0, logs)
        assert logs == {}

    def test_noop_on_none_logs(self):
        # Should not raise even though Keras can call on_epoch_end with logs=None.
        fc.MetricsConsolidationCallback().on_epoch_end(0, None)


class TestMetricsConsolidationCallbackFrontTypeRenaming:
    """Covers the post-consolidation rename step that groups per-front-type W&B keys."""

    @pytest.mark.parametrize(
        ("raw_key", "renamed_key"),
        [
            ("sup1_softmax_hss_CF", "front/CF/hss"),
            ("sup1_softmax_hss_hard_CF", "front/CF/hss_hard"),
            ("sup1_softmax_csi_DL", "front/DL/csi"),
            ("sup1_softmax_pod_OF", "front/OF/pod"),
            ("sup1_softmax_loss_CF", "front/CF/loss"),
            ("sup1_softmax_loss_none", "front/none/loss"),
            ("val_sup1_softmax_hss_CF", "front/CF/val_hss"),
            ("val_sup1_softmax_loss_none", "front/none/val_loss"),
        ],
    )
    def test_renames_per_front_type_keys(self, raw_key, renamed_key):
        logs = {raw_key: 0.42}
        fc.MetricsConsolidationCallback().on_epoch_end(0, logs)
        assert logs == {renamed_key: pytest.approx(0.42)}

    @pytest.mark.parametrize("key", ["hss", "hss_hard", "loss", "val_loss", "val_hss"])
    def test_aggregate_keys_are_left_alone(self, key):
        logs = {key: 0.5}
        fc.MetricsConsolidationCallback().on_epoch_end(0, logs)
        assert logs == {key: pytest.approx(0.5)}

    def test_key_containing_but_not_ending_with_front_type_token_is_untouched(self):
        """A key containing a front-type token without ending in one must not be renamed."""
        logs = {"CF_hss": 0.5}
        fc.MetricsConsolidationCallback().on_epoch_end(0, logs)
        assert logs == {"CF_hss": pytest.approx(0.5)}

    def test_per_front_type_key_on_only_sup1_survives_with_value_intact(self):
        logs = {"sup1_softmax_hss_CF": 0.75}
        fc.MetricsConsolidationCallback().on_epoch_end(0, logs)
        assert logs == {"front/CF/hss": pytest.approx(0.75)}


class TestCompactProgressCallback:
    """Covers the terminal-width-bounded stdout progress display added to replace verbose=1."""

    _FRONT_TYPES: ClassVar[list[str]] = list(constants.FRONT_TYPE_CLASS_INDEX)

    def _make(self, monkeypatch, is_tty, every_n_batches=10, terminal_width=120, steps=450, epochs=5000):
        monkeypatch.setattr(fc.sys.stdout, "isatty", lambda: is_tty)
        monkeypatch.setattr(
            fc.shutil, "get_terminal_size", lambda fallback=None: os.terminal_size((terminal_width, 24))
        )
        callback = fc.CompactProgressCallback(every_n_batches=every_n_batches)
        callback.set_params({"steps": steps, "epochs": epochs})
        return callback

    def _logs(self, value_fn, with_validation=False):
        logs = {"loss": value_fn(0)}
        for i, front_type in enumerate(self._FRONT_TYPES, start=1):
            logs[f"front/{front_type}/hss"] = value_fn(i)
            logs[f"front/{front_type}/csi"] = value_fn(i + len(self._FRONT_TYPES))
        if with_validation:
            logs["val_loss"] = value_fn(11)
            for i, front_type in enumerate(self._FRONT_TYPES, start=1):
                logs[f"front/{front_type}/val_hss"] = value_fn(i + 12)
                logs[f"front/{front_type}/val_csi"] = value_fn(i + 17)
        return logs

    def _epoch_end_logs(self, hss, csi, val_hss, val_csi, loss=0.0123, val_loss=0.0141):
        logs = {"loss": loss, "val_loss": val_loss}
        for front_type, h, c, vh, vc in zip(self._FRONT_TYPES, hss, csi, val_hss, val_csi, strict=True):
            logs[f"front/{front_type}/hss"] = h
            logs[f"front/{front_type}/csi"] = c
            logs[f"front/{front_type}/val_hss"] = vh
            logs[f"front/{front_type}/val_csi"] = vc
        return logs

    def test_default_batch_row_fits_comfortably_in_80_columns(self, monkeypatch, capsys):
        """The per-batch row (loss + HSS only) is short by construction; sanity-check it fits."""
        callback = self._make(monkeypatch, is_tty=True, every_n_batches=1, terminal_width=80, steps=450)
        hss_values = [0.412, 0.342, 0.272, 0.202, 0.132]
        logs = {"loss": 0.0123}
        for front_type, hss in zip(self._FRONT_TYPES, hss_values, strict=True):
            logs[f"front/{front_type}/hss"] = hss
        callback.on_train_batch_end(311, logs)  # batch_number 312
        row = capsys.readouterr().out.lstrip("\r")
        assert len(row) < 79, "the 80-column safety net truncated a row that should fit by design"
        for value in hss_values:
            assert f"{value:.3f}".lstrip("0") in row

    def test_all_negative_hss_and_csi_fit_in_80_columns_at_epoch_end(self, monkeypatch, capsys):
        """The critical regression guard: sign must not widen a row past what positive values need.

        Fix round 1 made the row fit at 80 columns for positive values only; every field there
        dropped its leading zero but reserved no column for a sign, so two or more negative
        values pushed the row over 80 columns and the safety net silently ate the tail (the
        exact bug fix round 2 addresses). Here every one of the ten HSS/CSI values, train and
        val alike, is negative — the worst case for row width — and must still render in full.
        """
        callback = self._make(monkeypatch, is_tty=True, terminal_width=80)
        hss = [-0.412, -0.342, -0.272, -0.202, -0.132]
        csi = [-0.310, -0.250, -0.190, -0.130, -0.062]
        val_hss = [-0.400, -0.330, -0.260, -0.190, -0.120]
        val_csi = [-0.300, -0.240, -0.180, -0.120, -0.060]
        logs = self._epoch_end_logs(hss, csi, val_hss, val_csi, loss=-0.0123, val_loss=-0.0141)
        callback.on_epoch_end(0, logs)
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line]
        assert len(lines) == 3
        for line in lines:
            assert len(line) < 79, f"line exceeds the 80-column budget: {line!r}"
        hss_line = next(line for line in lines if line.startswith("HSS"))
        csi_line = next(line for line in lines if line.startswith("CSI"))
        for value in hss + val_hss:
            expected = f"{value:.3f}".lstrip("-").lstrip("0")
            assert f"-{expected}" in hss_line, f"HSS value {value} missing from: {hss_line!r}"
        for value in csi + val_csi:
            expected = f"{value:.3f}".lstrip("-").lstrip("0")
            assert f"-{expected}" in csi_line, f"CSI value {value} missing from: {csi_line!r}"

    def test_mixed_sign_rows_are_identical_width_to_all_positive_rows(self, monkeypatch, capsys):
        """Column alignment must not depend on sign: a negative value cannot shift later columns."""
        callback = self._make(monkeypatch, is_tty=True, terminal_width=200)
        pos_hss = [0.412, 0.342, 0.272, 0.202, 0.132]
        pos_csi = [0.310, 0.250, 0.190, 0.130, 0.062]
        mixed_hss = [-0.412, 0.342, -0.272, 0.202, -0.132]
        mixed_csi = [0.310, -0.250, 0.190, -0.130, 0.062]

        callback.on_epoch_end(0, self._epoch_end_logs(pos_hss, pos_csi, pos_hss, pos_csi))
        positive_lines = [line for line in capsys.readouterr().out.splitlines() if line]

        callback.on_epoch_end(1, self._epoch_end_logs(mixed_hss, mixed_csi, mixed_hss, mixed_csi))
        mixed_lines = [line for line in capsys.readouterr().out.splitlines() if line]

        assert len(positive_lines) == len(mixed_lines) == 3
        for positive_line, mixed_line in zip(positive_lines, mixed_lines, strict=True):
            assert len(positive_line) == len(mixed_line), f"sign changed row width: {positive_line!r} vs {mixed_line!r}"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1.0, "1.000"),
            (0.0, " .000"),
            (-0.0, " .000"),
            (math.nan, "  nan"),
            (math.inf, "  inf"),
            (-math.inf, " -inf"),
        ],
    )
    def test_edge_case_values_format_without_crash_or_misreported_sign(self, value, expected):
        rendered = fc._format_value(value, fc._METRIC_FIELD_WIDTH, fc._METRIC_DECIMALS)
        assert rendered == expected
        assert len(rendered) == fc._METRIC_FIELD_WIDTH, "edge-case value did not get a stable width"

    def test_epoch_end_with_nan_and_inf_values_does_not_raise(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, terminal_width=200)
        hss = [math.nan, math.inf, -math.inf, 0.0, -0.0]
        csi = [0.310, 0.250, 0.190, 0.130, 0.062]
        logs = self._epoch_end_logs(hss, csi, hss, csi, loss=math.nan, val_loss=math.inf)
        callback.on_epoch_end(0, logs)  # must not raise
        out = capsys.readouterr().out
        assert "nan" in out
        assert "inf" in out

    def test_every_n_batches_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            fc.CompactProgressCallback(every_n_batches=0)

    def test_every_n_batches_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            fc.CompactProgressCallback(every_n_batches=-3)

    def test_shorter_inplace_write_pads_to_blot_out_longer_previous_write(self, monkeypatch, capsys):
        r"""Covers the residue bug: a bare `\r` moves the cursor but does not clear the line.

        A shorter write must be padded to at least the previous write's length, or that write's
        tail would remain visible, looking like corrupted digits.
        """
        callback = self._make(monkeypatch, is_tty=True, every_n_batches=1, terminal_width=200, steps=450)
        hss_values = [0.412, 0.342, 0.272, 0.202, 0.132]
        long_logs = {"loss": 12.3456}  # loss overflows its reserved width, making this row longer.
        short_logs = {"loss": 0.0123}
        for front_type, hss in zip(self._FRONT_TYPES, hss_values, strict=True):
            long_logs[f"front/{front_type}/hss"] = hss
            short_logs[f"front/{front_type}/hss"] = hss

        callback.on_train_batch_end(0, long_logs)
        first_write = capsys.readouterr().out
        assert first_write.startswith("\r")
        first_content = first_write[1:]

        callback.on_train_batch_end(1, short_logs)
        second_write = capsys.readouterr().out
        assert second_write.startswith("\r")
        second_content = second_write[1:]

        assert len(second_content) == len(first_content), (
            "the second (shorter) write was not padded to blot out the first (longer) write"
        )
        assert second_content.startswith(fc._batch_label(2, 450))  # batch index 1 -> batch number 2
        assert ".0123" in second_content
        assert "12.3456" not in second_content  # no residue from the first write's loss value.

    def test_epoch_end_loss_row_pads_over_longer_batch_row_residue(self, monkeypatch, capsys):
        """The guaranteed-every-epoch case: the epoch-end loss row is shorter than the batch row.

        The batch row it overwrites has an HSS block the loss row lacks, so the loss row must be
        padded or the batch row's HSS values would trail behind it on screen.
        """
        callback = self._make(monkeypatch, is_tty=True, every_n_batches=1, terminal_width=80, steps=450)
        hss_values = [0.412, 0.342, 0.272, 0.202, 0.132]
        csi_values = [0.310, 0.250, 0.190, 0.130, 0.062]
        val_hss_values = [0.400, 0.330, 0.260, 0.190, 0.120]
        val_csi_values = [0.300, 0.240, 0.180, 0.120, 0.060]
        batch_logs = {"loss": 0.0123}
        for front_type, hss in zip(self._FRONT_TYPES, hss_values, strict=True):
            batch_logs[f"front/{front_type}/hss"] = hss

        callback.on_train_batch_end(309, batch_logs)  # batch_number 310
        batch_write = capsys.readouterr().out
        assert batch_write.startswith("\r")
        batch_content = batch_write[1:]
        assert ".412" in batch_content  # sanity: the batch row does carry HSS values.

        epoch_logs = self._epoch_end_logs(hss_values, csi_values, val_hss_values, val_csi_values)
        callback.on_epoch_end(0, epoch_logs)
        epoch_write = capsys.readouterr().out
        first_line, _, _rest = epoch_write.partition("\n")
        assert first_line.startswith("\r")
        first_line_content = first_line[1:]

        unpadded_loss_row = fc._epoch_summary_row("loss", [0.0123], [0.0141], fc._LOSS_FIELD_WIDTH, fc._LOSS_DECIMALS)
        assert first_line_content.startswith(unpadded_loss_row)
        padding = first_line_content[len(unpadded_loss_row) :]
        assert padding == " " * len(padding), f"non-space residue after the loss row: {padding!r}"
        assert len(first_line_content) >= len(batch_content), (
            "epoch-end loss row is shorter than the last batch row and would leave residue"
        )
        assert ".412" not in first_line_content  # the batch row's HSS block must not survive.

    def test_narrow_terminal_safety_net_still_truncates_when_needed(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, terminal_width=40)
        logs = self._logs(lambda i: 12345.6789 + i)
        callback.on_epoch_begin(2, None)
        callback.on_train_batch_end(449, logs)  # final batch (steps=450) -> always updates
        out = capsys.readouterr().out
        for line in out.splitlines():
            assert len(line) <= 39

    def test_tty_batch_update_emits_carriage_return_and_no_newline(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True)
        logs = self._logs(lambda i: 0.01 * i)
        callback.on_train_batch_end(9, logs)  # batch_number 10 -> throttle boundary, fires
        out = capsys.readouterr().out
        assert out.startswith("\r")
        assert "\n" not in out

    def test_batch_update_shows_current_batch_number_not_final(self, monkeypatch, capsys):
        """Regression test for the coordinator's "showed 450/450 instead of 312/450" concern."""
        callback = self._make(monkeypatch, is_tty=True, every_n_batches=1, steps=450, terminal_width=200)
        logs = self._logs(lambda i: 0.01 * i)
        callback.on_train_batch_end(311, logs)  # batch index 311 -> displayed batch number 312
        out = capsys.readouterr().out
        assert "312/450" in out
        assert "450/450" not in out

    def test_non_tty_emits_no_per_batch_output(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=False)
        logs = self._logs(lambda i: 0.01 * i)
        for batch in range(25):
            callback.on_train_batch_end(batch, logs)
        assert capsys.readouterr().out == ""

    def test_throttling_fires_expected_number_of_updates_plus_final(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, every_n_batches=10, steps=25)
        logs = self._logs(lambda i: 0.01 * i)
        for batch in range(25):
            callback.on_train_batch_end(batch, logs)
        out = capsys.readouterr().out
        # Batches 10 and 20 hit the throttle boundary; batch 25 is the epoch's final batch.
        assert out.count("\r") == 3

    def test_header_names_front_types_in_constants_order(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, terminal_width=200)
        callback.on_epoch_begin(2, None)
        out = capsys.readouterr().out
        assert f"fronts: {' '.join(self._FRONT_TYPES)}" in out
        assert out == "Epoch 3/5000  fronts: CF WF SF OF DL\n"

    def test_header_is_printed_on_non_tty_too(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=False, terminal_width=200)
        callback.on_epoch_begin(2, None)
        out = capsys.readouterr().out
        assert "fronts: CF WF SF OF DL" in out

    def test_front_type_values_appear_in_constants_order_within_batch_row(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, terminal_width=200, steps=25)
        values = [0.01 * (i + 1) for i in range(len(self._FRONT_TYPES))]
        logs = {"loss": 0.5}
        for front_type, value in zip(self._FRONT_TYPES, values, strict=True):
            logs[f"front/{front_type}/hss"] = value
        callback.on_train_batch_end(24, logs)  # final batch -> always updates
        out = capsys.readouterr().out
        expected_hss = " ".join(fc._format_value(v, fc._METRIC_FIELD_WIDTH, fc._METRIC_DECIMALS) for v in values)
        assert expected_hss in out

    def test_missing_keys_degrade_gracefully_instead_of_raising(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, terminal_width=200, steps=25)
        callback.on_train_batch_end(24, {})  # no metrics present at all; must not raise
        out = capsys.readouterr().out
        assert "--" in out

    def test_missing_keys_on_epoch_end_do_not_raise(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, terminal_width=200)
        callback.on_epoch_end(0, {})
        out = capsys.readouterr().out
        assert "--" in out

    def test_epoch_end_prints_three_metric_rows_with_train_and_val_side_by_side(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, terminal_width=200)
        logs = self._logs(lambda i: 0.01 * i, with_validation=True)
        callback.on_epoch_end(2, logs)
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line]
        assert len(lines) == 3
        assert lines[0].lstrip("\r").startswith("loss")
        assert lines[1].startswith("HSS")
        assert lines[2].startswith("CSI")
        for line in lines:
            assert "| val" in line

    def test_epoch_end_val_values_are_not_truncated_away(self, monkeypatch, capsys):
        """The core bug fix: at epoch end, every validation value must survive in full."""
        callback = self._make(monkeypatch, is_tty=True, terminal_width=80)
        hss = [0.412, 0.342, 0.272, 0.202, 0.132]
        csi = [0.310, 0.250, 0.190, 0.130, 0.062]
        val_hss = [0.400, 0.330, 0.260, 0.190, 0.120]
        val_csi = [0.300, 0.240, 0.180, 0.120, 0.060]
        logs = self._epoch_end_logs(hss, csi, val_hss, val_csi)
        callback.on_epoch_end(0, logs)
        out = capsys.readouterr().out
        lines = out.splitlines()
        loss_line = next(line for line in lines if line.startswith("loss"))
        hss_line = next(line for line in lines if line.startswith("HSS"))
        csi_line = next(line for line in lines if line.startswith("CSI"))
        assert ".0141" in loss_line
        for value in val_hss:
            assert f"{value:.3f}".lstrip("0") in hss_line, f"val HSS {value} missing from: {hss_line!r}"
        for value in val_csi:
            assert f"{value:.3f}".lstrip("0") in csi_line, f"val CSI {value} missing from: {csi_line!r}"

    def test_non_tty_epoch_end_also_prints_three_rows_with_val_values(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=False, terminal_width=80)
        logs = self._logs(lambda i: 0.01 * i, with_validation=True)
        callback.on_epoch_end(2, logs)
        out = capsys.readouterr().out
        lines = out.splitlines()
        assert len(lines) == 3
        assert "\r" not in out
        assert all("| val" in line for line in lines)

    def test_metric_rows_share_label_width_so_columns_align(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, terminal_width=200)
        logs = self._logs(lambda i: 0.01 * i, with_validation=True)
        callback.on_epoch_end(0, logs)
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line]
        first_value_columns = {line.index(".") for line in lines}
        assert len(first_value_columns) == 1, f"metric rows are not column-aligned: {lines!r}"

    def test_epoch_end_writes_real_newlines_so_next_epoch_starts_fresh(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True)
        logs = self._logs(lambda i: 0.01 * i, with_validation=True)
        callback.on_epoch_end(0, logs)
        out = capsys.readouterr().out
        assert out.endswith("\n")
        assert out.count("\n") == 3

    def test_single_space_separators_and_no_leading_zero_in_representative_batch_row(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, every_n_batches=1, terminal_width=200, steps=450)
        hss_values = [0.412, 0.342, 0.272, 0.202, 0.132]
        logs = {"loss": 0.0123}
        for front_type, hss in zip(self._FRONT_TYPES, hss_values, strict=True):
            logs[f"front/{front_type}/hss"] = hss
        callback.on_train_batch_end(311, logs)
        out = capsys.readouterr().out
        assert "312/450 loss  .0123 HSS  .412  .342  .272  .202  .132" in out


class TestBuildDatasetShapeSummary:
    def test_builds_shape_and_date_range(self):
        times = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]")
        summary = fc.build_dataset_shape_summary(
            split="train", input_shape=(3, 4, 5, 2), target_shape=(3, 4, 5), times=times
        )
        assert summary.split == "train"
        assert summary.input_shape == (3, 4, 5, 2)
        assert summary.target_shape == (3, 4, 5)
        assert summary.date_min == "2020-01-01"
        assert summary.date_max == "2020-01-03"

    def test_out_of_order_times_still_yield_true_min_max(self):
        times = np.array(["2020-03-01", "2020-01-05", "2020-02-10"], dtype="datetime64[D]")
        summary = fc.build_dataset_shape_summary(
            split="val", input_shape=(3, 2, 2), target_shape=(3, 2, 2), times=times
        )
        assert summary.date_min == "2020-01-05"
        assert summary.date_max == "2020-03-01"

    def test_empty_times_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fc.build_dataset_shape_summary(
                split="test", input_shape=(0, 2, 2), target_shape=(0, 2, 2), times=np.array([], dtype="datetime64[D]")
            )


class TestDatasetSummaryCallback:
    def _make_summaries(self) -> list["fc.DatasetShapeSummary"]:
        return [
            fc.DatasetShapeSummary(
                split="train",
                input_shape=(10, 4, 5, 2),
                target_shape=(10, 4, 5),
                date_min="2020-01-01",
                date_max="2020-06-01",
            ),
            fc.DatasetShapeSummary(
                split="val",
                input_shape=(2, 4, 5, 2),
                target_shape=(2, 4, 5),
                date_min="2020-06-02",
                date_max="2020-07-01",
            ),
        ]

    def test_logs_every_split(self, caplog):
        cb = fc.DatasetSummaryCallback(self._make_summaries())
        with caplog.at_level("INFO", logger="fronts.callbacks"):
            cb.on_train_begin()
        assert "train split" in caplog.text
        assert "val split" in caplog.text
        assert "2020-01-01" in caplog.text
        assert "2020-07-01" in caplog.text

    def test_updates_wandb_summary_when_run_active(self, monkeypatch):
        summary_updates = {}
        fake_run = type("FakeRun", (), {"summary": type("FakeSummary", (), {"update": summary_updates.update})()})()
        monkeypatch.setattr(fc.wandb, "run", fake_run)

        cb = fc.DatasetSummaryCallback(self._make_summaries())
        cb.on_train_begin()

        assert summary_updates["data/train"]["input_shape"] == [10, 4, 5, 2]
        assert summary_updates["data/train"]["date_min"] == "2020-01-01"
        assert summary_updates["data/val"]["date_max"] == "2020-07-01"

    def test_no_wandb_call_when_no_run_active(self, monkeypatch):
        monkeypatch.setattr(fc.wandb, "run", None)
        cb = fc.DatasetSummaryCallback(self._make_summaries())
        cb.on_train_begin()  # Must not raise even with no active run.


class TestSelectActiveTestTimestep:
    def test_returns_first_timestep_with_a_front(self):
        data = np.zeros((4, 3, 3), dtype=np.int32)
        data[2, 1, 1] = 1  # CF code at time index 2
        target_da = xr.DataArray(data, dims=["time", "latitude", "longitude"])
        assert fc.select_active_test_timestep(target_da) == 2

    def test_raises_when_no_front_present(self):
        data = np.zeros((4, 3, 3), dtype=np.int32)
        target_da = xr.DataArray(data, dims=["time", "latitude", "longitude"])
        with pytest.raises(ValueError):
            fc.select_active_test_timestep(target_da)


class TestSelectTestSubsample:
    def test_bounded_and_sorted(self):
        idxs = fc.select_test_subsample(n_total=100, sample_size=10, seed=0)
        assert len(idxs) == 10
        assert (np.diff(idxs) > 0).all()
        assert idxs.min() >= 0
        assert idxs.max() < 100

    def test_clamps_to_n_total(self):
        idxs = fc.select_test_subsample(n_total=5, sample_size=200, seed=0)
        assert len(idxs) == 5

    def test_deterministic_for_fixed_seed(self):
        a = fc.select_test_subsample(n_total=50, sample_size=10, seed=42)
        b = fc.select_test_subsample(n_total=50, sample_size=10, seed=42)
        np.testing.assert_array_equal(a, b)


class TestRegionMask:
    def test_whole_domain_is_all_true(self):
        lats = np.array([10.0, 20.0, 30.0])
        lons = np.array([100.0, 200.0])
        mask = fc.region_mask(lats, lons, None)
        assert mask.shape == (3, 2)
        assert mask.all()

    def test_box_restricts_lat_and_lon(self):
        lats = np.array([10.0, 20.0, 30.0, 40.0])
        lons = np.array([100.0, 150.0, 200.0, 250.0])
        region = fc.utils.BoundingBox(lat_min=20.0, lat_max=40.0, lon_min=150.0, lon_max=250.0)
        mask = fc.region_mask(lats, lons, region)
        expected = np.array(
            [
                [False, False, False, False],
                [False, True, True, True],
                [False, True, True, True],
                [False, True, True, True],
            ]
        )
        np.testing.assert_array_equal(mask, expected)


class TestVisualizationCallbackPredict:
    """Tests the viz for predictions in callbacks.

    CallbackPredict must chunk by predict_batch_size rather than calling the model on the full array at once:
    a single unbatched call on e.g. 200 full-resolution test timesteps allocates one huge activation buffer on top
    of training's already resident GPU memory and reliably OOMs (see callbacks.py:on_epoch_end).
    """

    def _make_callback(self, n_samples: int, predict_batch_size: int) -> "fc.TestVisualizationCallback":
        inputs = fc.tf.keras.Input(shape=(2, 2, 1))
        model = fc.tf.keras.Model(inputs, inputs)  # identity: output == input
        cb = fc.TestVisualizationCallback(
            active_day_x=np.zeros((2, 2, 1), dtype=np.float32),
            active_day_y=np.zeros((2, 2, 1), dtype=np.float32),
            active_day_label="active day",
            subsample_x=np.arange(n_samples * 4, dtype=np.float32).reshape(n_samples, 2, 2, 1),
            subsample_y=np.zeros((n_samples, 2, 2, 1), dtype=np.float32),
            lats=np.array([0.0, 1.0]),
            lons=np.array([0.0, 1.0]),
            front_types=["CF"],
            predict_batch_size=predict_batch_size,
        )
        cb.set_model(model)
        return cb

    def test_chunked_prediction_matches_unbatched_input(self):
        # 5 samples with batch_size=2 forces a ragged last chunk (2, 2, 1 samples).
        cb = self._make_callback(n_samples=5, predict_batch_size=2)
        result = cb._predict(cb.subsample_x)
        np.testing.assert_allclose(result, cb.subsample_x)

    def test_never_calls_predict_on_the_full_unchunked_array(self, monkeypatch):
        # Calling model.predict() on the whole subsample at once accumulates every batch's
        # output into one GPU-resident tensor before returning, which is exactly what OOMs on
        # large full-domain subsamples. _predict must call predict() once per
        # predict_batch_size-sized chunk instead (see callbacks.py:_predict) — not
        # predict_on_batch(), which under MirroredStrategy hands its input to
        # distribute_strategy.run() undistributed, so every replica runs the forward pass on
        # the whole chunk and the (duplicate) per-replica outputs get concatenated together,
        # inflating the result to num_replicas x chunk_size rows.
        cb = self._make_callback(n_samples=5, predict_batch_size=2)
        real_predict = cb.model.predict
        call_sizes = []

        def tracking_predict(x, *a, **k):
            call_sizes.append(len(x))
            return real_predict(x, *a, **k)

        monkeypatch.setattr(cb.model, "predict", tracking_predict)
        result = cb._predict(cb.subsample_x)
        np.testing.assert_allclose(result, cb.subsample_x)
        assert call_sizes == [2, 2, 1]

    def test_predict_batch_size_field_is_required(self):
        with pytest.raises(TypeError):
            fc.TestVisualizationCallback(
                active_day_x=np.zeros((2, 2, 1), dtype=np.float32),
                active_day_y=np.zeros((2, 2, 1), dtype=np.float32),
                active_day_label="active day",
                subsample_x=np.zeros((1, 2, 2, 1), dtype=np.float32),
                subsample_y=np.zeros((1, 2, 2, 1), dtype=np.float32),
                lats=np.array([0.0, 1.0]),
                lons=np.array([0.0, 1.0]),
                front_types=["CF"],
            )


class TestVisualizationCallbackOnEpochEnd:
    """On_epoch_end must not pass an explicit `step` to wandb.log: WandbMetricsLogger's.

    Step is the cumulative training batch count, not the epoch number, so a `step=epoch`
    call is always behind the run's current step and gets silently dropped by wandb
    (see callbacks.py:on_epoch_end).
    """

    def _make_callback(self, monkeypatch, every_n_epochs: int) -> "fc.TestVisualizationCallback":
        # CF is class index 1, so 2 channels is the minimum needed to exercise the
        # class-index slicing in on_epoch_end.
        inputs = fc.tf.keras.Input(shape=(2, 2, 2))
        model = fc.tf.keras.Model(inputs, inputs)  # identity
        cb = fc.TestVisualizationCallback(
            active_day_x=np.zeros((2, 2, 2), dtype=np.float32),
            active_day_y=np.zeros((2, 2, 2), dtype=np.float32),
            active_day_label="active day",
            subsample_x=np.zeros((3, 2, 2, 2), dtype=np.float32),
            subsample_y=np.zeros((3, 2, 2, 2), dtype=np.float32),
            lats=np.array([0.0, 1.0]),
            lons=np.array([0.0, 1.0]),
            front_types=["CF"],
            predict_batch_size=2,
            every_n_epochs=every_n_epochs,
        )
        cb.set_model(model)
        # Plotting (cartopy map + table figure) is unrelated to the wandb step bug and
        # would otherwise drag in real map rendering; substitute cheap bare figures.
        monkeypatch.setattr(fc.plot_module, "plot_test_prediction", lambda **_: fc.plot_module.plt.figure())
        monkeypatch.setattr(fc.plot_module, "plot_performance_diagram_lite", lambda **_: fc.plot_module.plt.figure())
        return cb

    def test_logs_one_payload_with_no_explicit_step(self, monkeypatch):
        cb = self._make_callback(monkeypatch, every_n_epochs=1)
        calls = []
        monkeypatch.setattr(fc.wandb, "log", lambda payload, **kwargs: calls.append((payload, kwargs)))

        cb.on_epoch_end(epoch=0)

        assert len(calls) == 1
        payload, kwargs = calls[0]
        assert "step" not in kwargs
        assert "test/prediction" in payload
        assert any(k.startswith("test/performance_diagram/") for k in payload)

    def test_skips_logging_outside_cadence(self, monkeypatch):
        cb = self._make_callback(monkeypatch, every_n_epochs=10)
        calls = []
        monkeypatch.setattr(fc.wandb, "log", lambda payload, **kwargs: calls.append((payload, kwargs)))

        cb.on_epoch_end(epoch=0)

        assert calls == []


class TestAccumulateLiteStats:
    def test_matches_hand_computed_counts(self):
        # (time=1, lat=2, lon=2, n_fronts=1)
        pred = np.array([[[0.9], [0.1]], [[0.4], [0.6]]], dtype=np.float32).reshape(1, 2, 2, 1)
        truth = np.array([[1, 0], [0, 1]], dtype=np.float32).reshape(1, 2, 2, 1)
        weights = np.ones((2, 2), dtype=np.float32)
        thresholds = np.array([0.5], dtype=np.float32)

        tp, fp, tn, fn = fc.accumulate_lite_stats(pred, truth, weights, thresholds)

        assert tp[0, 0] == pytest.approx(2.0)
        assert fp[0, 0] == pytest.approx(0.0)
        assert tn[0, 0] == pytest.approx(2.0)
        assert fn[0, 0] == pytest.approx(0.0)

    def test_zero_weight_excludes_pixel(self):
        pred = np.array([[[0.9], [0.9]], [[0.9], [0.9]]], dtype=np.float32).reshape(1, 2, 2, 1)
        truth = np.ones((1, 2, 2, 1), dtype=np.float32)
        weights = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        thresholds = np.array([0.5], dtype=np.float32)

        tp, fp, tn, fn = fc.accumulate_lite_stats(pred, truth, weights, thresholds)

        # Only the two weight=1 pixels (both true positives) should count.
        assert tp[0, 0] == pytest.approx(2.0)
        assert fp[0, 0] == pytest.approx(0.0)
        assert tn[0, 0] == pytest.approx(0.0)
        assert fn[0, 0] == pytest.approx(0.0)
