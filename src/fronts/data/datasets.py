import dataclasses
import logging
import math
import time

import numpy as np
import tensorflow as tf
import xarray as xr

from fronts import utils
from fronts.data import inputs, targets

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DatasetConfig:
    """Configuration for loading and splitting input and fronts data.

    Attributes:
        inputs_icechunk_config: Icechunk store config for ERA5 input data.
        targets_icechunk_config: Icechunk store config for fronts data.
        variables: ERA5 variable names to load as input channels.
        test_years: Calendar years to hold out as the sequestered test set (never seen
            during training or validation).
        val_years: Calendar years to hold out for validation. Must not overlap test_years.
            All years not in test_years or val_years are used for training.
        batch_size: Number of timesteps per training batch.
        class_weights: Per-class loss weights. None means equal weighting.
        front_dilation: Number of binary dilation iterations applied to each non-background
            front class. 0 means no dilation.
        time_resolution: Optional pandas offset string (e.g. ``"6h"``) used to subsample
            the loaded timesteps. Only timestamps whose hour is already aligned to this
            interval are kept (e.g. ``"6h"`` retains 00, 06, 12, 18 UTC). ``None`` keeps
            all available timesteps.
        norm_stats_cache_dir: Optional directory for caching normalization
            statistics, keyed by store snapshot, channels, and train indices.
            None recomputes the statistics on every run.
        max_queue_size: Maximum number of prefetched batches kept in RAM ahead of the
            training loop (passed to ``tf.keras.utils.PyDataset(max_queue_size=...)``).
    """

    inputs_icechunk_config: utils.IcechunkStorageConfig
    targets_icechunk_config: utils.IcechunkStorageConfig
    variables: list[str]
    test_years: list[int]
    val_years: list[int]
    batch_size: int = 4
    class_weights: list[float] | None = None
    front_dilation: int = 0
    time_resolution: str = "6h"
    norm_stats_cache_dir: str | None = None
    max_queue_size: int = 4


class TrainingDataset(tf.keras.utils.PyDataset):
    """Batches a split's ERA5/fronts DataArrays for training via the PyDataset interface.

    Each ``__getitem__`` call gathers exactly one batch's timesteps with a single
    ``isel(time=idxs)`` take. ``input_ds``/``target_ds`` must already be sliced
    to this split (e.g. ``input_da.isel(time=train_indices)``) and backed by non-dask
    (``chunks=None``) arrays so each take reads directly through the zarr store rather
    than building a dask graph; concurrency across batches comes entirely from
    ``tf.keras.utils.PyDataset``'s own thread pool (``workers``/``max_queue_size``
    passed through ``**kwargs``).

    Yields a single (unreplicated) target per batch — the model's
    ``SharedTargetModel`` (see ``fronts.model``) is responsible for broadcasting it
    across any deep-supervision outputs, not the dataset.

    Attributes:
        input_ds: This split's input DataArray, shape (time, latitude, longitude, channel).
        target_ds: This split's target DataArray, shape (time, latitude, longitude, class).
        batch_size: Number of timesteps per batch.
        shuffle: If True, reshuffles the sample order at the end of every epoch.
    """

    def __init__(
        self,
        input_ds: xr.Dataset,
        target_da: xr.DataArray,
        data_config: DatasetConfig,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if input_ds.sizes["time"] != target_da.sizes["time"]:
            raise ValueError(
                f"Input and target time lengths differ: {input_ds.sizes['time']} vs {target_da.sizes['time']}"
            )
        self.input_ds = input_ds.copy()
        self.target_da = target_da.copy()
        self.data_config = data_config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)
        self._order = self._rng.permutation(self._total) if shuffle else np.arange(self._total)

    @property
    def _total(self) -> int:
        return self.input_ds.sizes["time"]

    @property
    def n_samples(self) -> int:
        """Number of individual timesteps (samples) in this split."""
        return self._total

    def __len__(self) -> int:
        """Returns the number of batches per epoch."""
        return math.ceil(self._total / self.batch_size)

    def on_epoch_end(self) -> None:
        """Reshuffles the sample order for the next epoch, if shuffling is enabled."""
        if self.shuffle:
            self._order = self._rng.permutation(self._total)

    def get_at_indices(self, idxs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns the (input, target) arrays at arbitrary global time indices.

        Unlike ``__getitem__``, ``idxs`` need not be batch-sized or in ``_order``'s
        epoch sequence — used by callers that need specific timesteps directly (e.g. a
        test-set visualization callback selecting one active day or a random subsample).
        """
        x_xarray = self.input_ds.isel(time=idxs)
        y_da = self.target_da.isel(time=idxs)

        # Convert inputs to a DataArray of shape (time, latitude, longitude, channel) and load into memory as float32.
        x = inputs.inputs_ds_to_dataarray(x_xarray, self.data_config.variables).values

        # One-hot encode targets, remap front classes to the configured set, and load into memory as float32.
        # Dilate fronts if > 0
        y_da = targets.one_hot_encode_to_dataarray(targets.remap_fronts(y_da))
        if self.data_config.front_dilation > 0:
            y_da = targets.dilate_fronts(y_da, self.data_config.front_dilation)

        # Convert to numpy arrays in memory. The model's SharedTargetModel is responsible for broadcasting the single
        # target across any deep-supervision outputs, not the dataset.
        y = y_da.values
        return x, y

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the (input, target) batch at ``idx``."""
        local_idxs = self._order[idx * self.batch_size : (idx + 1) * self.batch_size]
        t0 = time.time()
        result = self.get_at_indices(local_idxs)
        elapsed = time.time() - t0
        if elapsed > 30:
            logger.warning(f"Slow batch {idx}: {elapsed:.1f}s")
        return result


class EvalDataset(tf.keras.utils.PyDataset):
    """Yields ERA5 input batches for model.predict() during evaluation.

    Each ``__getitem__`` lazily reads one batch from the zarr-backed DataArray via
    ``isel(time=slice(...))``, matching the access pattern of ``TrainingDataset``.

    Attributes:
        era5_da: ERA5 inputs DataArray of shape (time, latitude, longitude, channel).
        batch_size: Number of timesteps per batch.
    """

    def __init__(self, era5_da: xr.DataArray, batch_size: int, **kwargs):
        super().__init__(**kwargs)
        self._era5_da = era5_da
        self._batch_size = batch_size
        self._n_times = era5_da.sizes["time"]

    def __len__(self) -> int:
        """Returns the number of batches per epoch."""
        return math.ceil(self._n_times / self._batch_size)

    def __getitem__(self, idx: int) -> np.ndarray:
        """Returns the (input, target) batch at ``idx``."""
        slc = slice(idx * self._batch_size, (idx + 1) * self._batch_size)
        return self._era5_da.isel(time=slc).values.astype(np.float32)
