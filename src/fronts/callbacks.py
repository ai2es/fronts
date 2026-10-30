"""Keras callbacks for training: resource monitoring, metric cleanup, compact progress, and test-set visualization."""

import collections
import dataclasses
import gc
import logging
import math
import re
import shutil
import sys

import numpy as np
import psutil
import pynvml
import tensorflow as tf
import wandb
import xarray as xr

from fronts import constants, utils
from fronts.plot import plot as plot_module

logger = logging.getLogger(__name__)

# How many predict_batch_size-sized steps model.predict() is allowed to accumulate on GPU
# before TestVisualizationCallback._predict flushes the result to CPU and starts a new call.
_PREDICT_MACRO_CHUNK_MULTIPLIER = 8

_PER_OUTPUT_LOSS_RE = re.compile(r"^sup\d+_.+_loss$")
# Matches any per-output metric key, e.g. "sup1_softmax_hss" or "sup1_softmax_hss_hard" —
# captures everything after "sup{N}_{activation}_" as the metric name, so any custom
# metric passed to model.compile is aggregated the same way without needing a
# metric-name-specific regex (see MetricsConsolidationCallback).
_PER_OUTPUT_METRIC_RE = re.compile(r"^sup\d+_[^_]+_(?P<metric>.+)$")

# Tokens that, as the trailing "_{token}" segment of a consolidated metric key, mark it as
# per-front-type rather than an aggregate (see _rename_front_type_keys).
_FRONT_TYPE_TOKENS = frozenset(constants.FRONT_TYPE_CLASS_INDEX) | {constants.BACKGROUND_CLASS_KEY}


def _strip_val_prefix(key: str) -> str:
    return key[len("val_") :] if key.startswith("val_") else key


def _rename_front_type_keys(logs: dict) -> None:
    """Rewrites keys ending in "_{front_type}" to "front/{front_type}/{remainder}" in place.

    Leaves aggregate keys (e.g. "hss", "val_loss") untouched, since none of them ends in a
    front-type token, and preserves any "val_" prefix on the remainder rather than the
    front-type token (e.g. "val_hss_CF" becomes "front/CF/val_hss", not "front/val_CF/hss").

    Args:
        logs: Mutable Keras logs dict, already consolidated by ``_consolidate``.
    """
    for key in list(logs):
        remainder, separator, token = key.rpartition("_")
        if not separator or not remainder or token not in _FRONT_TYPE_TOKENS:
            continue
        if remainder.startswith("val_"):
            new_key = f"front/{token}/val_{remainder[len('val_') :]}"
        else:
            new_key = f"front/{token}/{remainder}"
        logs[new_key] = logs.pop(key)


@dataclasses.dataclass
class CallbacksConfig:
    """Early-stopping, checkpoint, and periodic test-visualization callback configuration.

    ``patience`` is treated as a floor: training raises the effective early-stopping
    patience to at least the number of epochs in one full training pass (see
    ``utils.epochs_per_full_pass``) so the model sees every training sample before
    training can stop.

    ``test_viz_every_n_epochs`` controls how often (if at all) ``TestVisualizationCallback``
    logs an active-day prediction map and per-office-region performance diagrams to W&B
    from a bounded, seeded random subsample of the (otherwise untouched) sequestered
    test split; None disables this. ``test_viz_sample_size`` bounds that subsample's size.

    Attributes:
        monitor: Metric name to monitor for early stopping.
        patience: Number of epochs with no improvement after which training will be stopped
            or the learning rate will be decayed (if ``learning_rate_decay_factor`` is set).
        learning_rate_decay_factor: Optional factor to multiply the current learning rate by
            when early stopping is triggered. None disables learning-rate decay.
        learning_rate_minimum: Optional lower bound on the learning rate when decaying.
            None disables learning-rate decay.
        min_delta: Smallest monitored-value improvement that resets the patience counter.
            0.0 counts any improvement. Keras's ReduceLROnPlateau default (an absolute
            1e-4) silently freezes training when the monitored loss is itself ~1e-3:
            every epoch reads as a plateau, so the learning rate decays to its floor
            within a dozen epochs regardless of real progress.
        early_stopping_patience: Number of epochs with no improvement before ending the run
            when LR decay is active. Should exceed ``patience`` by a few multiples so LR
            reductions get a chance to rescue a plateau before the run ends. None keeps the
            pre-existing behavior: with LR decay enabled the run has no stop condition and
            continues until ``epochs`` or the job walltime.
        model_checkpoint_path: Optional path to save the best model weights to. None disables
            checkpointing.
        test_viz_every_n_epochs: Optional cadence in epochs for logging test-set visualizations.
            None disables test-set visualization.
        test_viz_sample_size: Maximum number of timesteps to subsample from the test split
            for the performance diagram. Ignored if ``test_viz_every_n_epochs`` is None.
        metrics_csv_path: Optional path to append every epoch's metrics to as CSV, a durable
            local record independent of W&B. None derives ``metrics_epoch.csv`` in the same
            directory as ``model_checkpoint_path``; if that is also None, CSV logging is
            skipped entirely rather than guessing a location. See ``train._build_run_callbacks``.
        compact_progress_every_n_batches: Optional batch throttle for ``CompactProgressCallback``,
            a terminal-width-bounded replacement for Keras's default per-batch progress bar
            (which prints every key in ``logs`` on one line — far wider than a terminal once
            per-front-type metrics are added, so it wraps and floods stdout). None disables
            the compact callback and leaves Keras's default progress bar (``verbose=1``) alone.
            Must be a positive int if set; ``CompactProgressCallback`` raises otherwise.
    """

    monitor: str = "val_loss"
    patience: int = 8
    learning_rate_decay_factor: float | None = None
    learning_rate_minimum: float | None = None
    min_delta: float = 0.0
    early_stopping_patience: int | None = None
    model_checkpoint_path: str | None = None
    test_viz_every_n_epochs: int | None = 10
    test_viz_sample_size: int = 200
    # Defaulted (contrary to the usual no-defaults rule for dataclasses) so the 17 existing
    # YAML configs keep parsing: dacite raises on a missing required field.
    metrics_csv_path: str | None = None
    # Defaulted (contrary to the usual no-defaults rule for dataclasses) so the 17 existing
    # YAML configs keep parsing: dacite raises on a missing required field.
    compact_progress_every_n_batches: int | None = 10


class MetricsConsolidationCallback(tf.keras.callbacks.Callback):
    """Collapses per-deep-supervision-output metrics into single aggregate curves.

    Keras compiles the same loss/metrics for every deep-supervision output, producing
    one ``sup{N}_{activation}_{metric_name}``/``_loss`` key per output for every metric
    passed to ``model.compile`` (e.g. ``hss``, ``hss_hard``, ...). Keras already
    aggregates the per-output losses into a single ``loss``/``val_loss``, so those
    per-output keys are simply dropped; custom metrics have no built-in aggregate, so
    this callback averages each metric's per-output values into ``{metric_name}``/
    ``val_{metric_name}`` before deleting the per-output keys — generically, by
    whatever name Keras assigned the metric, not a hardcoded ``hss``. A metric wrapped
    as a raw ``@tf.function`` reports a fixed ``.name`` from the function it decorates
    (not the reassignable ``__name__``) — use ``tf.keras.metrics.MeanMetricWrapper(fn,
    name=...)`` to give a custom metric a distinct name, or every metric literally named
    ``hss`` collides and Keras silently renames the extras to ``hss_1``, ``hss_2``, etc.

    After that consolidation, keys ending in ``_{front_type}`` (e.g. ``hss_CF``,
    ``loss_none``) are further rewritten to ``front/{front_type}/{metric_name}`` — see
    ``_rename_front_type_keys`` — so ``WandbMetricsLogger``'s ``epoch/`` prefix produces
    ``epoch/front/CF/hss`` and W&B groups per-front-type metrics into one collapsible
    section per front type. Aggregate keys (``hss``, ``val_loss``, ...) do not end in a
    front-type token and are left untouched.

    Must run before ``wandb.keras.WandbMetricsLogger`` in the callbacks list passed to
    ``model.fit`` — Keras shares one mutable ``logs`` dict across every callback's
    ``on_epoch_end``/``on_train_batch_end`` call in list order, so whichever callback
    runs first determines what later callbacks (and W&B) see. ``on_train_batch_end``
    only ever sees unprefixed (training) keys since validation runs once per epoch,
    so the ``val_`` half of the loop below is a no-op there.
    """

    def _consolidate(self, logs: dict | None) -> None:
        if not logs:
            return
        for prefix, is_val_key in (("", False), ("val_", True)):
            keys_by_metric: dict[str, list[str]] = collections.defaultdict(list)
            for key in logs:
                if key.startswith("val_") != is_val_key:
                    continue
                match = _PER_OUTPUT_METRIC_RE.match(_strip_val_prefix(key))
                if match and match["metric"] != "loss":
                    keys_by_metric[match["metric"]].append(key)
            for metric_name, keys in keys_by_metric.items():
                logs[f"{prefix}{metric_name}"] = float(np.mean([logs.pop(k) for k in keys]))

        for key in [k for k in logs if _PER_OUTPUT_LOSS_RE.match(_strip_val_prefix(k))]:
            logs.pop(key)

        _rename_front_type_keys(logs)

    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        """Aggregates per-output hss into hss and strips per-output loss keys in place."""
        self._consolidate(logs)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Aggregates per-output hss into hss/val_hss and strips per-output loss keys in place."""
        self._consolidate(logs)


_LOSS_DECIMALS = 4
_LOSS_FIELD_WIDTH = 6  # "-.0123" / " .0123" — sign column, dot, four decimal digits.
_METRIC_DECIMALS = 3
_METRIC_FIELD_WIDTH = 5  # "-.412" / " .412" — sign column, dot, three decimal digits.
_MISSING_VALUE_PLACEHOLDER = "--"
_EPOCH_END_ROW_LABEL_WIDTH = len("loss")  # Widest of the "loss"/"HSS"/"CSI" row labels.


def _format_value(value: float | None, width: int, decimals: int) -> str:
    """Formats one metric value to a fixed, sign-safe width.

    Every field reserves one column for a sign, so a negative value (``-.412``) occupies
    exactly the same width as a positive one (`` .412``) — column alignment must not depend on
    sign, since HSS is genuinely and commonly negative early in training (a randomly-initialized
    model can be worse than chance), and that is exactly when this display matters most.

    A value in (-1, 1) — true of HSS and CSI always, and of loss in steady state — always has a
    leading zero before the decimal point (``0.412``, ``-0.412``); dropping it (``.412``,
    ``-.412``) recovers the column the sign reservation costs, so the net width matches the
    unsigned, no-reservation format this replaced. ``nan``/``inf``/``-inf`` render as literal
    text; ``-0.0`` is normalized to ``0.0`` first so its sign bit never renders as a spurious
    ``-``. A value at or beyond +/-1.0 (e.g. an early-training loss spike, or a not-quite-possible
    exactly-1.0 HSS) simply renders at its natural, potentially wider length instead of being
    truncated — this is the one case that can still cost the fixed-width guarantee, deliberately
    accepted as a rare edge case rather than being silently mishandled.

    Args:
        value: The value to format, or None if missing from ``logs``.
        width: Target field width, including the reserved sign column. Only a placeholder or an
            in-range value are padded to exactly this width; wider natural-length values (see
            above) are left unpadded.
        decimals: Number of digits after the decimal point.

    Returns:
        The right-justified formatted value, or a right-justified placeholder if ``value`` is
        None.
    """
    if value is None:
        return f"{_MISSING_VALUE_PLACEHOLDER:>{width}}"
    value = float(value)
    if math.isnan(value):
        text = "nan"
    elif math.isinf(value):
        text = "inf" if value > 0 else "-inf"
    else:
        if value == 0:
            value = 0.0  # Normalize -0.0 so it never renders with a spurious "-".
        text = f"{value:.{decimals}f}"
        if text.startswith("0."):
            text = text[1:]
        elif text.startswith("-0."):
            text = "-" + text[2:]
    return f"{text:>{width}}"


def _batch_label(batch: int | None, steps: int | None) -> str:
    """Renders the "batch/steps" position label, digit-padding ``batch`` to ``steps``'s width.

    Padding ``batch`` (not the whole label) keeps the label's width constant across an epoch's
    updates regardless of ``batch``'s own digit count, e.g. ``" 10/450"`` and ``"100/450"`` are
    both 7 characters.

    Args:
        batch: Current 1-indexed batch number, or None if unknown.
        steps: Total batches in the epoch, or None if unknown.

    Returns:
        ``"{batch}/{steps}"`` with ``batch`` padded to ``steps``'s digit count, ``str(batch)``
        alone if ``steps`` is unknown, or ``""`` if ``batch`` is also unknown.
    """
    if steps is not None and batch is not None:
        return f"{batch:>{len(str(steps))}}/{steps}"
    if batch is not None:
        return str(batch)
    return ""


def _epoch_summary_row(
    label: str,
    train_values: list[float | None],
    val_values: list[float | None],
    width: int,
    decimals: int,
) -> str:
    """Renders one "label train_values | val val_values" epoch-summary row.

    Train and validation values for one metric are placed side by side on the same permanent
    row (rather than on two separate rows, as an earlier version of this callback did) so that
    truncating a too-long *line* to the terminal width can never remove validation metrics
    entirely while leaving training metrics intact, or vice versa — both are equally exposed to
    (and, by design, safely clear of) the truncation safety net.

    Args:
        label: Metric name ("loss", "HSS", or "CSI"), left-justified to
            ``_EPOCH_END_ROW_LABEL_WIDTH``.
        train_values: Training value(s) for this metric — a single-element list for loss, or one
            per front type for HSS/CSI, in ``constants.FRONT_TYPE_CLASS_INDEX`` order.
        val_values: Validation value(s), same shape as ``train_values``.
        width: Field width passed to ``_format_value`` for every value in this row.
        decimals: Decimal places passed to ``_format_value`` for every value in this row.

    Returns:
        The rendered row, not yet truncated to the terminal width.
    """
    train_str = " ".join(_format_value(v, width, decimals) for v in train_values)
    val_str = " ".join(_format_value(v, width, decimals) for v in val_values)
    return f"{label:<{_EPOCH_END_ROW_LABEL_WIDTH}} {train_str} | val {val_str}"


def _truncate_to_terminal_width(line: str) -> str:
    r"""Truncates ``line`` to one column short of the terminal width, so ``\r`` always rewinds it.

    A last-resort safety net, not the normal path: every numeric field above reserves a sign
    column and is joined with single spaces, so a full row — even with every HSS/CSI value
    negative — renders at a width set purely by the front-type count, never wider. With the
    nine-class mapping that is ~106 columns for an epoch-summary row, inside the 120-column
    fallback used when stdout is not a terminal (the SLURM log case), so this only engages in an
    interactive terminal narrower than that. See
    ``TestCompactProgressCallback.test_all_negative_hss_and_csi_fit_the_designed_row_width_at_epoch_end``.
    """
    width = shutil.get_terminal_size(fallback=(120, 24)).columns - 1
    return line[:width]


class CompactProgressCallback(tf.keras.callbacks.Callback):
    r"""Prints a small, terminal-width-bounded progress display per epoch instead of Keras's default.

    Keras's default ``ProgbarLogger`` (``verbose=1``) prints every key in ``logs`` on a single
    line. With the ~29 per-front-type metric keys this branch adds, that line is far wider than
    any terminal: it wraps across several visual rows, and Keras's trailing ``\r`` only rewinds
    the last one, so every update strands the wrapped rows above it and stdout degenerates into
    an unreadable wall of text.

    This callback instead renders a fixed, deliberately small set of health-check metrics —
    aggregate ``loss``, and per-front-type ``HSS`` (the soft ``front/{front_type}/hss``) and
    ``CSI`` (``front/{front_type}/csi``), in ``constants.FRONT_TYPE_CLASS_INDEX`` order — using a
    sign-safe fixed-width number format (every field reserves a column for a leading ``-``, so
    column alignment never depends on sign or magnitude), so a full row's width is set purely by
    the front-type count and never grows with the values — ~106 columns for an epoch-summary row
    under the nine-class mapping, inside the 120-column fallback used for a non-TTY stdout, even
    when every value is negative. The rendered row is additionally
    truncated to the actual terminal width before every write as a last-resort safety net (see
    ``_truncate_to_terminal_width``), so ``\r`` always rewinds the whole line even in a narrower
    terminal. Every in-place write is also padded with trailing spaces to at least as long as the
    longest line written in place since the last real newline (see ``_pad_for_inplace``) — a bare
    ``\r`` only moves the cursor to column 0, it does not clear the line, so writing a shorter
    row than its predecessor would otherwise leave that predecessor's tail visible on screen.
    ``hss_hard``, ``pod``, and the per-front-type losses are deliberately omitted here: they
    remain in W&B and metrics_epoch.csv, since stdout here is a health check, not the record.

    A header line naming the front-type column order is printed once per epoch (real newline).
    On a TTY, one train-only, loss-and-HSS-only row (CSI is dropped from this row only, since it
    is short by construction and can never overflow) is then rewritten in place via ``\r``,
    throttled to at most every ``every_n_batches`` batches (plus always the epoch's final batch).
    At ``on_epoch_end``, three permanent rows are printed — one per metric (loss, HSS, CSI) —
    each with that metric's training and validation values side by side (see
    ``_epoch_summary_row``), so validation metrics can never be truncated away while training
    metrics survive, or vice versa. Each row ends in a real newline, so the next epoch's header
    starts fresh. On a non-TTY stdout (e.g. a SLURM log file, where ``\r`` is useless and only
    bloats the file), no per-batch output is emitted at all — only the header and the same three
    epoch-end rows. Whether stdout is a TTY is determined once, at construction, not re-checked
    per batch.

    Must run after ``MetricsConsolidationCallback`` in the callbacks list passed to
    ``model.fit`` — it reads ``front/{front_type}/hss`` and ``front/{front_type}/csi``, which
    only exist in that slash-delimited form after ``MetricsConsolidationCallback`` rewrites the
    shared ``logs`` dict. See ``train._build_run_callbacks``.

    Attributes:
        every_n_batches: Update the in-place train row at most this often, in batches (plus
            always the epoch's final batch). Has no effect on a non-TTY stdout, which never
            updates per batch regardless. Must be a positive int.
    """

    def __init__(self, every_n_batches: int) -> None:
        if every_n_batches <= 0:
            raise ValueError(f"every_n_batches must be a positive int, got {every_n_batches!r}.")
        super().__init__()
        self.every_n_batches = every_n_batches
        self._is_tty = sys.stdout.isatty()
        self._front_types = list(constants.FRONT_TYPE_CLASS_INDEX)
        # Length of the longest line written in place (via `\r`) since the last real newline —
        # see `_pad_for_inplace`.
        self._inplace_written_length = 0

    def _epochs_total(self) -> int | None:
        return (self.params or {}).get("epochs")

    def _steps_total(self) -> int | None:
        return (self.params or {}).get("steps")

    def _pad_for_inplace(self, row: str) -> str:
        r"""Pads ``row`` so it cannot leave stale characters from a previous in-place write.

        A bare ``\r`` only returns the cursor to column 0 — it does not clear the line — so
        writing a shorter string than the previous ``\r``-written line leaves that line's
        trailing characters on screen, looking like corrupted digits (e.g. a short epoch-end
        row overwriting a longer batch row leaves the batch row's tail visible). Padding with
        trailing spaces up to the longest line written in place since the last real newline
        guarantees no such residue survives. Terminal-width truncation must be applied to the
        *result* of this padding, not before, so a padded row still cannot exceed the width
        bound.

        Args:
            row: The not-yet-truncated row about to be written in place.

        Returns:
            ``row`` padded with trailing spaces to at least ``self._inplace_written_length``.
        """
        return row.ljust(self._inplace_written_length)

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        """Prints the epoch header line naming the front-type column order."""
        epochs_total = self._epochs_total()
        header = (
            f"Epoch {epoch + 1}/{epochs_total if epochs_total is not None else '?'}"
            f"  fronts: {' '.join(self._front_types)}"
        )
        sys.stdout.write(_truncate_to_terminal_width(header) + "\n")
        sys.stdout.flush()
        self._inplace_written_length = 0  # A real newline was just written; nothing to blot out.

    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        """Rewrites the in-place train-only, loss-and-HSS-only row, throttled per ``every_n_batches``."""
        if not self._is_tty:
            return
        steps_total = self._steps_total()
        batch_number = batch + 1
        is_final_batch = steps_total is not None and batch_number >= steps_total
        if not is_final_batch and batch_number % self.every_n_batches != 0:
            return
        logs = logs or {}
        label = _batch_label(batch_number, steps_total)
        loss_str = _format_value(logs.get("loss"), _LOSS_FIELD_WIDTH, _LOSS_DECIMALS)
        hss_values = [logs.get(f"front/{ft}/hss") for ft in self._front_types]
        hss_str = " ".join(_format_value(v, _METRIC_FIELD_WIDTH, _METRIC_DECIMALS) for v in hss_values)
        row = f"{label} loss {loss_str} HSS {hss_str}"
        truncated = _truncate_to_terminal_width(self._pad_for_inplace(row))
        sys.stdout.write("\r" + truncated)
        sys.stdout.flush()
        self._inplace_written_length = len(truncated)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Prints the epoch's three permanent train/val summary rows (loss, HSS, CSI)."""
        logs = logs or {}
        loss_row = _epoch_summary_row(
            "loss", [logs.get("loss")], [logs.get("val_loss")], _LOSS_FIELD_WIDTH, _LOSS_DECIMALS
        )
        hss_row = _epoch_summary_row(
            "HSS",
            [logs.get(f"front/{ft}/hss") for ft in self._front_types],
            [logs.get(f"front/{ft}/val_hss") for ft in self._front_types],
            _METRIC_FIELD_WIDTH,
            _METRIC_DECIMALS,
        )
        csi_row = _epoch_summary_row(
            "CSI",
            [logs.get(f"front/{ft}/csi") for ft in self._front_types],
            [logs.get(f"front/{ft}/val_csi") for ft in self._front_types],
            _METRIC_FIELD_WIDTH,
            _METRIC_DECIMALS,
        )
        if self._is_tty:
            # This first write overwrites the last in-place batch row (via `\r`), which may be
            # longer than loss_row — pad it so none of that row's tail survives on screen.
            truncated_loss_row = _truncate_to_terminal_width(self._pad_for_inplace(loss_row))
            sys.stdout.write("\r" + truncated_loss_row + "\n")
        else:
            sys.stdout.write(_truncate_to_terminal_width(loss_row) + "\n")
        self._inplace_written_length = 0  # A real newline was just written; nothing to blot out.
        sys.stdout.write(_truncate_to_terminal_width(hss_row) + "\n")
        sys.stdout.write(_truncate_to_terminal_width(csi_row) + "\n")
        sys.stdout.flush()


class GcCallback(tf.keras.callbacks.Callback):
    """Forces garbage collection and logs RAM/GPU VRAM usage at the end of every epoch."""

    def on_train_begin(self, logs: dict | None = None) -> None:
        """Initializes NVML for the GPU memory queries used in on_epoch_end."""
        pynvml.nvmlInit()

    def on_train_end(self, logs: dict | None = None) -> None:
        """Shuts down NVML."""
        pynvml.nvmlShutdown()

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Collects garbage and logs current RAM/GPU VRAM usage."""
        gc.collect()
        proc = psutil.Process()
        ram_used_gib = proc.memory_info().rss / 2**30
        ram_total_gib = psutil.virtual_memory().total / 2**30
        logger.info("RAM: %.1f%% (%.1f / %.1f GiB)", 100 * ram_used_gib / ram_total_gib, ram_used_gib, ram_total_gib)
        n_gpus = pynvml.nvmlDeviceGetCount()
        for i in range(n_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            logger.info(
                "GPU %d VRAM: %.1f%% (%.1f / %.1f GiB)",
                i,
                100 * mem.used / mem.total,
                mem.used / 2**30,
                mem.total / 2**30,
            )


@dataclasses.dataclass
class DatasetShapeSummary:
    """Array shape and date range for one data split, for training-provenance logging.

    Attributes:
        split: Split name ("train", "val", or "test").
        input_shape: Model input array shape, e.g. (n_samples, latitude, longitude, channel)
            for 2D models or (n_samples, latitude, longitude, level, variable) for volume models.
        target_shape: Raw (non-one-hot) target array shape (n_samples, latitude, longitude).
        date_min: Earliest timestamp in the split (ISO 8601 date).
        date_max: Latest timestamp in the split (ISO 8601 date).
    """

    split: str
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    date_min: str
    date_max: str


def build_dataset_shape_summary(
    split: str,
    input_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    times: np.ndarray,
) -> DatasetShapeSummary:
    """Build a DatasetShapeSummary from a split's array shapes and time coordinate.

    Args:
        split: Split name ("train", "val", or "test").
        input_shape: Model input array shape for this split.
        target_shape: Raw (non-one-hot) target array shape for this split.
        times: 1-D array of ``numpy.datetime64`` timestamps for this split.

    Returns:
        A DatasetShapeSummary with the date range read from ``times``.

    Raises:
        ValueError: If ``times`` is empty.
    """
    if len(times) == 0:
        raise ValueError(f"Cannot summarize an empty '{split}' split (0 timesteps).")
    return DatasetShapeSummary(
        split=split,
        input_shape=tuple(input_shape),
        target_shape=tuple(target_shape),
        date_min=str(np.datetime_as_string(np.min(times), unit="D")),
        date_max=str(np.datetime_as_string(np.max(times), unit="D")),
    )


class DatasetSummaryCallback(tf.keras.callbacks.Callback):
    """Logs each split's input/target array shape and date range once, to logs and W&B.

    Fires on ``on_train_begin`` since shapes and date ranges are fixed for the whole run
    rather than changing per epoch. Always logs through the fronts logger; also updates the
    active W&B run's summary (not a time-series metric — ``wandb.log`` would place the
    string date fields on the run's step-indexed chart axis) when a run is active.

    Attributes:
        summaries: Shape/date-range summaries for each split to log, in log order.
    """

    def __init__(self, summaries: list[DatasetShapeSummary]) -> None:
        super().__init__()
        self.summaries = summaries

    def on_train_begin(self, logs: dict | None = None) -> None:
        """Logs every split's array shape and date range to the fronts logger and W&B."""
        for summary in self.summaries:
            logger.info(
                "%s split: input_shape=%s target_shape=%s date_range=[%s, %s]",
                summary.split,
                summary.input_shape,
                summary.target_shape,
                summary.date_min,
                summary.date_max,
            )
        if wandb.run is not None:
            wandb.run.summary.update(
                {
                    f"data/{summary.split}": {
                        "input_shape": list(summary.input_shape),
                        "target_shape": list(summary.target_shape),
                        "date_min": summary.date_min,
                        "date_max": summary.date_max,
                    }
                    for summary in self.summaries
                }
            )


def select_active_test_timestep(target_da: xr.DataArray) -> int:
    """Return the index of the first test timestep containing any front pixel.

    Args:
        target_da: Raw (non-one-hot) front identifier DataArray, dims (time, latitude, longitude).

    Returns:
        Integer index into ``target_da``'s time axis.

    Raises:
        ValueError: If no timestep in ``target_da`` contains a front pixel.
    """
    front_codes = list(constants.FRONT_CLASS_MAP)
    has_front = target_da.isin(front_codes).any(dim=["latitude", "longitude"]).compute().values
    indices = np.flatnonzero(has_front)
    if len(indices) == 0:
        raise ValueError(
            f"No active (front-containing) timestep found in the test split ({target_da.sizes.get('time', 0)} "
            "timesteps checked). Either the test split is empty or none of its timesteps contain any of the "
            f"front codes {front_codes}."
        )
    return int(indices[0])


def select_test_subsample(n_total: int, sample_size: int, seed: int) -> np.ndarray:
    """Return a sorted, seeded random subsample of timestep indices, bounded by ``n_total``.

    Args:
        n_total: Total number of available timesteps.
        sample_size: Desired subsample size. Clamped to ``n_total``.
        seed: Seed for the sampling RNG.

    Returns:
        Sorted 1-D integer array of selected indices.
    """
    rng = np.random.default_rng(seed)
    size = min(sample_size, n_total)
    return np.sort(rng.choice(n_total, size=size, replace=False))


def region_mask(lats: np.ndarray, lons: np.ndarray, region: utils.BoundingBox | None) -> np.ndarray:
    """Return a (n_lat, n_lon) bool mask — True inside ``region``, all-True if ``region`` is None.

    Args:
        lats: 1-D latitude array.
        lons: 1-D longitude array.
        region: Bounding box to restrict to, or None for the whole domain.

    Returns:
        Boolean mask of shape (len(lats), len(lons)).
    """
    if region is None:
        return np.ones((len(lats), len(lons)), dtype=bool)
    lat_mask = (lats >= region.lat_min) & (lats <= region.lat_max)
    lon_mask = (lons >= region.lon_min) & (lons <= region.lon_max)
    return lat_mask[:, np.newaxis] & lon_mask[np.newaxis, :]


def accumulate_lite_stats(
    pred: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    thresholds: np.ndarray = constants.LITE_THRESHOLDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate weighted TP/FP/TN/FN per front class and threshold, no neighborhood expansion.

    A cheaper alternative to ``evaluation/compute_stats.py``'s multi-neighborhood spatial
    sweep, intended for periodic in-training diagnostics rather than final evaluation.

    Args:
        pred: Predicted probabilities, shape (time, latitude, longitude, n_fronts).
        truth: Binary truth labels, shape (time, latitude, longitude, n_fronts).
        weights: Per-pixel weights, shape (latitude, longitude) — e.g. cosine-latitude
            times a region mask. Pixels with weight 0 do not contribute.
        thresholds: 1-D probability thresholds, shape (T,).

    Returns:
        4-tuple of (tp, fp, tn, fn), each shape (n_fronts, T), summed over time and space.
    """
    n_times, _, _, n_fronts = pred.shape
    n_thresh = len(thresholds)
    w_flat = weights.ravel().astype(np.float32)

    tp = np.zeros((n_fronts, n_thresh), dtype=np.float64)
    fp = np.zeros_like(tp)
    tn = np.zeros_like(tp)
    fn = np.zeros_like(tp)

    for t in range(n_times):
        pred_t = pred[t].reshape(-1, n_fronts).T  # (F, P)
        truth_t = truth[t].reshape(-1, n_fronts).T.astype(bool)  # (F, P)

        above = pred_t[:, :, np.newaxis] >= thresholds  # (F, P, T)
        truth_3d = truth_t[:, :, np.newaxis]

        tp += ((above & truth_3d).astype(np.float32) * w_flat[np.newaxis, :, np.newaxis]).sum(axis=1)
        fp += ((above & ~truth_3d).astype(np.float32) * w_flat[np.newaxis, :, np.newaxis]).sum(axis=1)
        tn += ((~above & ~truth_3d).astype(np.float32) * w_flat[np.newaxis, :, np.newaxis]).sum(axis=1)
        fn += ((~above & truth_3d).astype(np.float32) * w_flat[np.newaxis, :, np.newaxis]).sum(axis=1)

    return tp, fp, tn, fn


@dataclasses.dataclass
class TestVisualizationCallback(tf.keras.callbacks.Callback):
    """Logs an active-day prediction map and per-office-region performance diagrams to W&B.

    Runs every ``every_n_epochs`` epochs. The performance diagram is computed on a
    bounded random subsample of the test split (see ``select_test_subsample``) using
    ``accumulate_lite_stats`` — a coarse threshold grid with no neighborhood expansion,
    cheap enough to run periodically during training.

    Attributes:
        active_day_x: Single-timestep model input, shape (latitude, longitude, channel).
        active_day_y: Single-timestep one-hot truth, shape (latitude, longitude, class).
        active_day_label: Title label for the prediction figure (e.g. the timestamp).
        subsample_x: Subsampled test inputs, shape (time, latitude, longitude, channel).
        subsample_y: Subsampled test one-hot truth, shape (time, latitude, longitude, class).
        lats: 1-D latitude array matching the spatial dims above.
        lons: 1-D longitude array matching the spatial dims above.
        front_types: Front type labels to evaluate, in class order.
        predict_batch_size: Batch size used to chunk ``subsample_x`` inference.
        every_n_epochs: Visualization cadence in epochs.
    """

    active_day_x: np.ndarray
    active_day_y: np.ndarray
    active_day_label: str
    subsample_x: np.ndarray
    subsample_y: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    front_types: list[str]
    predict_batch_size: int
    every_n_epochs: int = 10

    def __post_init__(self) -> None:
        """Initializes the underlying Callback base after dataclass field assignment."""
        super().__init__()

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """Run the model's finest-resolution (first) output, in bounded macro-chunks.

        ``model.predict()`` batches its forward passes via ``batch_size``, but that only
        bounds the per-step compute — it still accumulates every requested sample's output
        into one GPU-resident tensor before returning. At full spatial resolution (e.g.
        full-CONUS-domain runs) that accumulated buffer, on top of training's already-
        resident GPU memory, reliably OOMs for large subsamples (e.g. 200 timesteps).
        Calling ``predict()`` on bounded macro-chunks and moving each one to CPU
        immediately caps how much stays GPU-resident at once.

        ``predict_on_batch`` was tried here first as a per-``predict_batch_size``-chunk
        alternative, but it is not safe under ``tf.distribute.MirroredStrategy``: it does
        not shard/gather a batch across replicas the way ``predict()`` does, and can
        silently return far more rows than requested (observed 4x on a 4-replica batch of
        4). ``predict()`` must be kept for correctness; only the chunk size fed to it
        is bounded here.
        """
        macro_chunk_size = self.predict_batch_size * _PREDICT_MACRO_CHUNK_MULTIPLIER
        outputs: list[np.ndarray] = []
        for start in range(0, x.shape[0], macro_chunk_size):
            pred = self.model.predict(
                x[start : start + macro_chunk_size], batch_size=self.predict_batch_size, verbose=0
            )
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            outputs.append(np.asarray(pred))
        return np.concatenate(outputs, axis=0)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Every ``every_n_epochs`` epochs, logs an active-day prediction map and per-region diagrams."""
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        class_indices = [constants.FRONT_TYPE_CLASS_INDEX[ft] for ft in self.front_types]

        pred_day = self._predict(self.active_day_x[np.newaxis])[0]  # (lat, lon, class)
        probs_ds = xr.Dataset(coords={"latitude": self.lats, "longitude": self.lons})
        for ft, ci in zip(self.front_types, class_indices, strict=True):
            probs_ds[ft] = (["latitude", "longitude"], pred_day[:, :, ci])
        truth_day = np.argmax(self.active_day_y, axis=-1)
        truth_da = xr.DataArray(truth_day, coords={"latitude": self.lats, "longitude": self.lons})

        pred_fig = plot_module.plot_test_prediction(
            lats=self.lats,
            lons=self.lons,
            probs_ds=probs_ds,
            front_types=self.front_types,
            truth_da=truth_da,
            title=self.active_day_label,
        )
        # WandbMetricsLogger tracks the run's step as the cumulative training batch count
        # (regardless of its log_freq), which is always ahead of `epoch` by the time training
        # has run a handful of batches; logging with `step=epoch` is always behind the run's
        # current step and gets silently dropped. Logging everything from this call in one
        # `wandb.log` with no explicit `step` instead lands on the run's actual current step.
        payload = {"test/prediction": wandb.Image(pred_fig)}
        plot_module.plt.close(pred_fig)

        pred_subsample = self._predict(self.subsample_x)[:, :, :, class_indices]
        truth_subsample = self.subsample_y[:, :, :, class_indices] > 0.5
        lat_weights = np.cos(np.deg2rad(self.lats))[:, np.newaxis] * np.ones((1, len(self.lons)), dtype=np.float32)

        regions: dict[str, utils.BoundingBox | None] = {"whole_domain": None, **constants.OFFICE_REGIONS}
        for region_name, region in regions.items():
            weights = lat_weights * region_mask(self.lats, self.lons, region)
            tp, fp, tn, fn = accumulate_lite_stats(pred_subsample, truth_subsample, weights)
            for fi, ft in enumerate(self.front_types):
                fig = plot_module.plot_performance_diagram_lite(
                    front_type=ft,
                    thresholds=constants.LITE_THRESHOLDS,
                    tp=tp[fi],
                    fp=fp[fi],
                    tn=tn[fi],
                    fn=fn[fi],
                    title=region_name,
                )
                payload[f"test/performance_diagram/{region_name}/{ft}"] = wandb.Image(fig)
                plot_module.plt.close(fig)

        wandb.log(payload)
