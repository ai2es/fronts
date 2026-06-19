"""Keras callbacks for training: resource monitoring, W&B metric cleanup, and periodic test-set visualization."""

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

_PER_OUTPUT_HSS_RE = re.compile(r"^sup\d+_.+_hss$")
_PER_OUTPUT_LOSS_RE = re.compile(r"^sup\d+_.+_loss$")


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
    """

    monitor: str = "val_loss"
    patience: int = 8
    model_checkpoint_path: str | None = None
    test_viz_every_n_epochs: int | None = 10
    test_viz_sample_size: int = 200


class MetricsConsolidationCallback(tf.keras.callbacks.Callback):
    """Collapses per-deep-supervision-output metrics into single aggregate curves.

    Keras compiles the same loss/metric for every deep-supervision output, producing
    one ``sup{N}_{activation}_hss``/``_loss`` key per output. Keras already aggregates
    the per-output losses into a single ``loss``/``val_loss``, so those per-output keys
    are simply dropped; HSS has no built-in aggregate, so this callback averages the
    per-output values into ``hss``/``val_hss`` before deleting them.

    Must run before ``wandb.keras.WandbMetricsLogger`` in the callbacks list passed to
    ``model.fit`` — Keras shares one mutable ``logs`` dict across every callback's
    ``on_epoch_end`` call in list order, so whichever callback runs first determines
    what later callbacks (and W&B) see.
    """

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Aggregates per-output hss into hss/val_hss and strips per-output loss keys in place."""
        if not logs:
            return
        for prefix, is_val_key in (("", False), ("val_", True)):
            hss_keys = [
                k for k in logs if k.startswith("val_") == is_val_key and _PER_OUTPUT_HSS_RE.match(_strip_val_prefix(k))
            ]
            if hss_keys:
                logs[f"{prefix}hss"] = float(np.mean([logs.pop(k) for k in hss_keys]))

        for key in [k for k in logs if _PER_OUTPUT_LOSS_RE.match(_strip_val_prefix(k))]:
            logs.pop(key)


class GcCallback(tf.keras.callbacks.Callback):
    """Forces garbage collection and logs RAM/GPU VRAM usage at the end of every epoch."""

    def on_train_begin(self, logs=None):
        """Initializes NVML for the GPU memory queries used in on_epoch_end."""
        pynvml.nvmlInit()

    def on_train_end(self, logs=None):
        """Shuts down NVML."""
        pynvml.nvmlShutdown()

    def on_epoch_end(self, epoch, logs=None):
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
        raise ValueError("No active (front-containing) timestep found in the test split.")
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
    every_n_epochs: int = 10

    def __post_init__(self) -> None:
        """Initializes the underlying Callback base after dataclass field assignment."""
        super().__init__()

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """Run the model's finest-resolution (first) output on a batch of inputs."""
        pred = self.model(x, training=False)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        return np.asarray(pred)

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
        wandb.log({"test/prediction": wandb.Image(pred_fig)}, step=epoch)
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
                wandb.log({f"test/performance_diagram/{region_name}/{ft}": wandb.Image(fig)}, step=epoch)
                plot_module.plt.close(fig)
