"""Keras callbacks for training: resource monitoring, W&B metric cleanup, and periodic test-set visualization."""

import collections
import dataclasses
import gc
import logging
import re

import numpy as np
import psutil
import pynvml
import tensorflow as tf
import wandb
import xarray as xr

from fronts import utils
from fronts.data import targets
from fronts.plot import plot as plot_module

logger = logging.getLogger(__name__)

FRONT_TYPE_CLASS_INDEX: dict[str, int] = {"CF": 1, "WF": 2, "SF": 3, "OF": 4, "DL": 5}

# Office-of-responsibility regions for the Unified Surface Analysis (WPC manual, p.25).
# The 30N split and the 140W HFO/NHC boundary come from the manual; WPC vs OPC is
# approximated as a longitude band over the continental US since the real WPC area of
# responsibility is an irregular coastline-following polygon, not a box.
OFFICE_REGIONS: dict[str, utils.BoundingBox] = {
    "OPC_west": utils.BoundingBox(lat_min=30.0, lat_max=80.0, lon_min=130.0, lon_max=220.0),
    "WPC": utils.BoundingBox(lat_min=30.0, lat_max=80.0, lon_min=220.0, lon_max=300.0),
    "OPC_east": utils.BoundingBox(lat_min=30.0, lat_max=80.0, lon_min=300.0, lon_max=369.75),
    "HFO": utils.BoundingBox(lat_min=0.25, lat_max=30.0, lon_min=130.0, lon_max=220.0),
    "NHC": utils.BoundingBox(lat_min=0.25, lat_max=30.0, lon_min=220.0, lon_max=369.75),
}

LITE_THRESHOLDS = np.linspace(0.05, 1.0, 20, dtype=np.float32)

_PER_OUTPUT_LOSS_RE = re.compile(r"^sup\d+_.+_loss$")
# Matches any per-output metric key, e.g. "sup1_softmax_hss" or "sup1_softmax_hss_hard" —
# captures everything after "sup{N}_{activation}_" as the metric name, so any custom
# metric passed to model.compile is aggregated the same way without needing a
# metric-name-specific regex (see MetricsConsolidationCallback).
_PER_OUTPUT_METRIC_RE = re.compile(r"^sup\d+_[^_]+_(?P<metric>.+)$")


def _strip_val_prefix(key: str) -> str:
    return key[len("val_") :] if key.startswith("val_") else key


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

    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        """Aggregates per-output hss into hss and strips per-output loss keys in place."""
        self._consolidate(logs)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Aggregates per-output hss into hss/val_hss and strips per-output loss keys in place."""
        self._consolidate(logs)


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


def select_active_test_timestep(target_da: xr.DataArray) -> int:
    """Return the index of the first test timestep containing any front pixel.

    Args:
        target_da: Raw (non-one-hot) front identifier DataArray, dims (time, latitude, longitude).

    Returns:
        Integer index into ``target_da``'s time axis.

    Raises:
        ValueError: If no timestep in ``target_da`` contains a front pixel.
    """
    front_codes = list(targets.FRONT_CLASS_MAP)
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
    thresholds: np.ndarray = LITE_THRESHOLDS,
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
        """Run the model's finest-resolution (first) output, chunked by ``predict_batch_size``."""
        # model.predict() over the full array batches its forward passes but still accumulates
        # every batch's output into one GPU-resident tensor before returning; at full spatial
        # resolution (e.g. full-CONUS-domain runs) that accumulated buffer, on top of training's
        # already-resident GPU memory, reliably OOMs. Calling predict() once per chunk and moving
        # each result to CPU immediately keeps only one chunk's output on GPU at a time.
        #
        # This must be predict(), not predict_on_batch(): under MirroredStrategy,
        # predict_on_batch() hands its input to distribute_strategy.run() undistributed, so each
        # replica runs the forward pass on the *whole* chunk rather than a shard of it, and the
        # per-replica outputs are then concatenated — silently inflating the result to
        # num_replicas x chunk_size rows. predict() constructs a proper distributed dataset
        # under the hood and returns exactly chunk_size rows.
        outputs: list[np.ndarray] = []
        for start in range(0, x.shape[0], self.predict_batch_size):
            pred = self.model.predict(x[start : start + self.predict_batch_size], verbose=0)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            outputs.append(np.asarray(pred))
        return np.concatenate(outputs, axis=0)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Every ``every_n_epochs`` epochs, logs an active-day prediction map and per-region diagrams."""
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        class_indices = [FRONT_TYPE_CLASS_INDEX[ft] for ft in self.front_types]

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

        regions: dict[str, utils.BoundingBox | None] = {"whole_domain": None, **OFFICE_REGIONS}
        for region_name, region in regions.items():
            weights = lat_weights * region_mask(self.lats, self.lons, region)
            tp, fp, tn, fn = accumulate_lite_stats(pred_subsample, truth_subsample, weights)
            for fi, ft in enumerate(self.front_types):
                fig = plot_module.plot_performance_diagram_lite(
                    front_type=ft,
                    thresholds=LITE_THRESHOLDS,
                    tp=tp[fi],
                    fp=fp[fi],
                    tn=tn[fi],
                    fn=fn[fi],
                    title=region_name,
                )
                payload[f"test/performance_diagram/{region_name}/{ft}"] = wandb.Image(fig)
                plot_module.plt.close(fig)

        wandb.log(payload)
