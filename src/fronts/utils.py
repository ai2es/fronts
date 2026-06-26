from __future__ import annotations

import dataclasses
import datetime
import math
import os
import subprocess
from collections import namedtuple
from string import Template
from typing import Any, TypeVar

import dacite
import icechunk as ic
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import zarr
from xarray.core.indexes import IndexSelResult, PandasIndex, _query_slice
from xarray.core.indexing import _expand_slice

T = TypeVar("T")
_XArray = TypeVar("_XArray", xr.Dataset, xr.DataArray)
BoundingBox = namedtuple("BoundingBox", ["lat_min", "lat_max", "lon_min", "lon_max"])


@dataclasses.dataclass
class IcechunkStorageConfig:
    """Configuration for storing generated data in an Icechunk repository.

    Attributes:
        store_path: Path to the Icechunk store.
        branch_name: Name of the branch to write to.
        commit_message: Message for the commit.
        zarr_format: Zarr format version to use when writing the store. Default is 3.
        group_name: Optional group name within the zarr store. Default is None.
        virtual_chunk_local_path: If the store contains virtual chunks referencing local
            netcdf files, set this to the directory those files live in (e.g.
            ``/ourdisk/hpc/ai2es/tman/data/netcdf/``). The ``file://`` URL prefix is
            derived automatically. Leave None for stores with no virtual chunks.
        write_batch_size: Number of time steps to load and commit per write when
            writing or appending. Bounds peak memory without dask: each batch is
            loaded eagerly, appended along time, and committed before the next
            batch is read. None writes the whole dataset in one commit.
    """

    store_path: str
    branch_name: str
    commit_message: str = "add data"
    zarr_format: int = 3
    group_name: str | None = None
    virtual_chunk_local_path: str | None = None
    write_batch_size: int | None = None


class PeriodicBoundaryIndex(PandasIndex):
    """xarray index for a 1-D coordinate that wraps at a period.

    Subclasses PandasIndex and intercepts slice queries so a selection
    that crosses the period boundary (e.g. longitude 350 to 10) is
    returned as two concatenated index arrays rather than an empty
    slice.
    """

    period: float
    _min: float
    _max: float

    __slots__ = ("_max", "_min", "coord_dtype", "dim", "index", "period")

    def __init__(self, *args, period=360, **kwargs):
        super().__init__(*args, **kwargs)
        self.period = period
        self._min = self.index.min()
        self._max = self.index.max()

    @classmethod
    def from_variables(cls, variables, options):
        """Construct index from coordinate variables, reading period from options."""
        obj = super().from_variables(variables, options={})
        obj.period = options.get("period", obj.period)  # pyrefly: ignore[missing-attribute]
        return obj

    def _wrap_periodically(self, label_value: float) -> float:
        # Reduce ``label_value`` into ``[_min, _min + period)``. The
        # earlier formulation used ``label - _max`` which silently
        # shifted in-range labels by ``period - (_max - _min)`` (one
        # grid step on a 0-360 ERA5 axis where ``_max=359.75``).
        # ``label - _min`` is the textbook periodic remap and works
        # for both 0-360 and -180/180 axes.
        return self._min + (label_value - self._min) % self.period

    def _split_slice_across_boundary(self, label: slice) -> np.ndarray:
        """Return concatenated integer indices for a slice that wraps."""
        first_slice = slice(label.start, self._max, label.step)
        second_slice = slice(self._min, label.stop, label.step)

        first_as_index_slice = _query_slice(self.index, first_slice)
        second_as_index_slice = _query_slice(self.index, second_slice)

        first_as_indices = _expand_slice(first_as_index_slice, self.index.size)
        second_as_indices = _expand_slice(second_as_index_slice, self.index.size)

        return np.concatenate([first_as_indices, second_as_indices])

    def sel(self, labels: dict[Any, Any], method=None, tolerance=None) -> IndexSelResult:
        """Remap out-of-range labels back into the index range."""
        assert len(labels) == 1
        coord_name, label = next(iter(labels.items()))

        if isinstance(label, slice):
            start, stop, step = label.start, label.stop, label.step
            if start is None or stop is None:
                return super().sel({coord_name: label})
            if stop < start:
                return super().sel({coord_name: []})

            assert self._min < self._max

            wrapped_start = self._wrap_periodically(label.start)
            wrapped_stop = self._wrap_periodically(label.stop)
            wrapped_label = slice(wrapped_start, wrapped_stop, step)

            if wrapped_start < wrapped_stop:
                return super().sel({coord_name: wrapped_label})
            # Slice crosses the wrap boundary; split in two.
            wrapped_indices = self._split_slice_across_boundary(wrapped_label)
            return IndexSelResult({self.dim: wrapped_indices})

        wrapped_label = self._wrap_periodically(label)  # type: ignore
        return super().sel({coord_name: wrapped_label}, method=method, tolerance=tolerance)

    def __repr__(self) -> str:
        """Return string representation showing the period."""
        return f"PeriodicBoundaryIndex(period={self.period})"


def attach_periodic_lon_index(data: _XArray) -> _XArray:
    """Attach a 360°-period :class:`PeriodicBoundaryIndex` to ``longitude``.

    Replaces the default ``PandasIndex`` so wrap-crossing
    ``.sel(longitude=slice(...))`` queries work, e.g. ``slice(130, 369.75)``
    correctly wraps past 360° to include both 130-359.75 and 0-9.75.

    Args:
        data: Dataset or DataArray with a 1-D longitude coordinate.

    Returns a copy of the input with the longitude index replaced by a
        ``PeriodicBoundaryIndex``.
    """
    return data.drop_indexes("longitude").set_xindex("longitude", index_cls=PeriodicBoundaryIndex, period=360)


def select_spatial_domain(data: xr.Dataset, bb: BoundingBox) -> xr.Dataset:
    """Select latitude and longitude from a Dataset, handling wrap-crossing ranges.

    Attaches a :class:`PeriodicBoundaryIndex` to longitude if one is not already
    present, then performs a single ``.sel()`` so wrap-crossing slices (e.g.
    ``lon_min=350, lon_max=370``) are handled transparently.

    Args:
        data: Dataset with ``latitude`` and ``longitude`` coordinates.
        bb: Bounding box. ``lon_max > 360`` triggers wrap-crossing logic.

    Returns:
        Spatially subsetted Dataset.
    """
    lat_vals = data["latitude"].values
    lat_slice = (
        slice(bb.lat_max, bb.lat_min)
        if len(lat_vals) >= 2 and lat_vals[0] > lat_vals[1]
        else slice(bb.lat_min, bb.lat_max)
    )
    if not isinstance(data.xindexes.get("longitude"), PeriodicBoundaryIndex):
        data = attach_periodic_lon_index(data)
    return data.sel(latitude=lat_slice, longitude=slice(bb.lon_min, bb.lon_max))


def unwrap_longitude(data: _XArray) -> _XArray:
    """Remap a wrap-crossing longitude coordinate to be monotonically increasing.

    After a wrap-crossing ``select_spatial_domain`` the longitude coordinate may
    look like ``[130, ..., 359.75, 0, ..., 9.75]``.  Plotting libraries expect
    monotonically increasing coordinates, so this function shifts the second chunk
    to ``[360, ..., 369.75]``, giving ``[130, ..., 359.75, 360, ..., 369.75]``.

    Args:
        data: Dataset or DataArray with a 1-D ``longitude`` coordinate.

    Returns:
        Input with the longitude coordinate remapped to be monotonically increasing.
        If the coordinate is already monotonic, it is returned unchanged.
    """
    lons = data["longitude"].values
    if len(lons) < 2 or np.all(np.diff(lons) >= 0):
        return data
    wrap_idx = int(np.argmax(np.diff(lons) < 0)) + 1
    new_lons = lons.copy()
    new_lons[wrap_idx:] += 360.0
    return data.assign_coords(longitude=new_lons)


def drop_duplicate_times(data: _XArray) -> _XArray:
    """Drop duplicate timestamps along the time dimension, keeping the first occurrence.

    Args:
        data: DataArray or Dataset with a ``time`` coordinate.

    Returns:
        Input with any duplicate time values removed.
    """
    _, unique_idx = np.unique(data["time"].values, return_index=True)
    return data.isel(time=sorted(unique_idx))


def apply_time_resolution(times: np.ndarray, resolution: str) -> np.ndarray:
    """Return only the elements of ``times`` that fall on the ``resolution`` grid.

    A timestamp is kept when flooring it to ``resolution`` leaves it unchanged,
    meaning its hour (and finer components) are already aligned to the interval.
    For example, ``"6h"`` keeps 00:00, 06:00, 12:00, and 18:00 UTC only.

    Args:
        times: Array of numpy datetime64 values.
        resolution: Pandas offset string, e.g. ``"6h"`` or ``"3h"``.

    Returns:
        Subset of ``times`` whose values are already aligned to ``resolution``.
    """
    idx = pd.DatetimeIndex(times)
    return times[idx == idx.floor(resolution)]


def split_by_year(
    times: np.ndarray, test_years: list[int], val_years: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_mask, val_mask, test_mask) boolean masks based on calendar year.

    Every timestep whose year is in ``test_years``/``val_years`` is assigned to that split;
    all remaining timesteps default to train.

    Args:
        times: 1-D array of numpy datetime64 timestamps.
        test_years: Calendar years to hold out as the sequestered test set.
        val_years: Calendar years to hold out for validation.

    Returns:
        Tuple of boolean arrays, each of shape ``(len(times),)``.

    Raises:
        ValueError: If test_years and val_years overlap.
    """
    if set(test_years) & set(val_years):
        raise ValueError(f"test_years and val_years overlap: {set(test_years) & set(val_years)}")
    years = pd.DatetimeIndex(times).year.to_numpy()
    test_mask = np.zeros(len(times), dtype=bool)
    val_mask = np.zeros(len(times), dtype=bool)
    if test_years:
        test_mask = np.isin(years, test_years)
    if val_years:
        val_mask = np.isin(years, val_years)
    train_mask = ~(test_mask | val_mask)
    return train_mask, val_mask, test_mask


def slurm_cpu_count() -> int:
    """Return the number of CPUs allocated to the current SLURM job, if any.

    Reads ``SLURM_CPUS_PER_TASK`` (set when a job requests ``--cpus-per-task``).
    Falls back to ``os.cpu_count()`` outside of a SLURM allocation, or to 1 if
    that can't be determined either.

    Returns:
        Number of CPUs to use as a default worker/thread count.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        return int(slurm_cpus)
    return os.cpu_count() or 1


def epochs_per_full_pass(n_samples: int, batch_size: int, steps_per_epoch: int) -> int:
    """Return how many subset-epochs of ``steps_per_epoch`` batches cover all samples once.

    The batch pipeline repeats and reshuffles indefinitely, so running ``steps_per_epoch``
    batches per epoch means one full pass over the data spans several epochs. A full pass is
    ``ceil(n_samples / batch_size)`` batches, completed every
    ``ceil(ceil(n_samples / batch_size) / steps_per_epoch)`` epochs. Used to floor the
    early-stopping patience so the model sees every training sample before training can stop.

    Args:
        n_samples: Number of samples in the dataset.
        batch_size: Number of samples per batch.
        steps_per_epoch: Number of batches drawn per epoch.

    Returns:
        Epochs needed for one full pass, or 0 when ``n_samples`` is 0.

    Raises:
        ValueError: If ``n_samples`` is negative, or ``batch_size`` or ``steps_per_epoch`` is
            less than 1.
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if steps_per_epoch < 1:
        raise ValueError(f"steps_per_epoch must be >= 1, got {steps_per_epoch}")
    full_pass_steps = math.ceil(n_samples / batch_size)
    return math.ceil(full_pass_steps / steps_per_epoch)


YAML_TYPE_HOOKS: dict = {
    BoundingBox: lambda d: BoundingBox(*d),
    datetime.datetime: lambda d: datetime.datetime.fromisoformat(str(d)),
}


def _interpolate_yaml_vars(obj: Any, variables: dict[str, str]) -> Any:
    """Recursively substitute ``${key}`` placeholders in YAML string values.

    Top-level scalar keys in the YAML act as the variable namespace. Unrecognized
    placeholders are left unchanged (``Template.safe_substitute`` semantics).
    """
    if isinstance(obj, str):
        return Template(obj).safe_substitute(variables)
    if isinstance(obj, dict):
        return {k: _interpolate_yaml_vars(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_yaml_vars(v, variables) for v in obj]
    return obj


def load_yaml(path: str) -> dict:
    """Load and interpolate a YAML config file into a plain dict.

    Top-level scalar keys (e.g. ``run_name: foo``) are available as ``${run_name}``
    placeholders in any string value throughout the file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Fully interpolated dict from the YAML file.
    """
    with open(path) as f:
        config_yaml = yaml.safe_load(f)
    variables = {k: str(v) for k, v in config_yaml.items() if isinstance(v, (str, int, float))}
    return _interpolate_yaml_vars(config_yaml, variables)


def parse_config_section(
    yaml_data: dict,
    config_class: type[T],
    config_key: str | None = None,
    type_hooks: dict | None = None,
) -> T:
    """Parse a section of a pre-loaded YAML dict into a dataclass instance.

    Args:
        yaml_data: Pre-loaded and interpolated YAML dict (from ``load_yaml``).
        config_class: The dataclass to parse the section into.
        config_key: Key of the sub-dict to parse. None parses the root.
        type_hooks: Optional dictionary of type conversion functions.

    Returns:
        An instance of the specified dataclass.
    """
    section = yaml_data[config_key] if config_key else yaml_data
    return dacite.from_dict(
        data_class=config_class,
        data=section,
        config=dacite.Config(check_types=False, type_hooks=type_hooks or {}),
    )


def open_config_yaml_as_dataclass(
    path: str,
    config_class: type[T],
    config_key: str | None = None,
    type_hooks: dict | None = None,
) -> T:
    """Open a YAML config file and parse it into a dataclass instance.

    Top-level scalar keys (e.g. ``run_name: foo``) are available as ``${run_name}``
    placeholders in any string value throughout the file.

    Args:
        path: Path to the YAML config file.
        config_class: The dataclass to parse the config into.
        config_key: Optional key to extract a sub-dictionary from the YAML.
        type_hooks: Optional dictionary of type conversion functions.

    Returns:
        An instance of the specified dataclass.
    """
    return parse_config_section(load_yaml(path), config_class, config_key, type_hooks)


def get_git_commit() -> str:
    """Return the current HEAD commit hash, or 'unknown' if not in a git repo.

    Returns:
        Full SHA-1 commit hash string, or 'unknown' on failure.
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_icechunk_snapshot_id(
    store_path: str,
    branch: str,
    virtual_chunk_local_path: str | None = None,
) -> str:
    """Return the snapshot ID at the tip of a branch in an icechunk store.

    Args:
        store_path: Path to the icechunk store directory.
        branch: Branch name to read from.
        virtual_chunk_local_path: Local directory containing netcdf files referenced
            by virtual chunks. Leave None for stores with no virtual chunks.

    Returns:
        The snapshot ID string for the branch tip.
    """
    storage = ic.local_filesystem_storage(store_path)
    repo_config = ic.RepositoryConfig.default()
    authorize_virtual_chunk_access = None
    if virtual_chunk_local_path is not None:
        url_prefix = f"file://{virtual_chunk_local_path}"
        repo_config.set_virtual_chunk_container(
            ic.VirtualChunkContainer(
                url_prefix=url_prefix,
                store=ic.local_filesystem_store(virtual_chunk_local_path),
            )
        )
        authorize_virtual_chunk_access = ic.containers_credentials({url_prefix: None})
    repo = ic.Repository.open(
        storage,
        config=repo_config,
        authorize_virtual_chunk_access=authorize_virtual_chunk_access,
    )
    session = repo.readonly_session(branch)
    return session.snapshot_id


def open_readonly_icechunk_store(
    store_path: str,
    branch: str,
    group: str | None = None,
    zarr_format: int = 3,
    virtual_chunk_local_path: str | None = None,
    chunks: Any = "auto",
) -> xr.Dataset:
    """Open a local icechunk store in read-only mode and return it as an xarray datatype.

    Args:
        store_path: Path to the icechunk store directory.
        branch: Branch name to read from.
        group: Optional group name within the zarr store to open.
        zarr_format: Zarr format version to use when opening the store (default is 3).
        virtual_chunk_local_path: Local directory containing the netcdf files referenced
            by virtual chunks (e.g. ``/ourdisk/hpc/data/netcdf/``). When provided,
            registers a VirtualChunkContainer and authorizes access so those chunks can
            be fetched. Leave None for stores with no virtual chunks.
        chunks: Forwarded to ``xr.open_zarr``. ``"auto"`` (default) returns a dask-backed
            Dataset suitable for chunked reductions (e.g. normalization stats). ``None``
            returns a Dataset backed directly by the zarr store with no dask graph, for
            callers that only ever read small, explicit slices themselves.

    Returns:
        An xarray Dataset or DataArray containing the data from the icechunk store.
    """
    storage = ic.local_filesystem_storage(store_path)
    repo_config = ic.RepositoryConfig.default()
    authorize_virtual_chunk_access = None
    if virtual_chunk_local_path is not None:
        url_prefix = f"file://{virtual_chunk_local_path}"
        repo_config.set_virtual_chunk_container(
            ic.VirtualChunkContainer(
                url_prefix=url_prefix,
                store=ic.local_filesystem_store(virtual_chunk_local_path),
            )
        )
        authorize_virtual_chunk_access = ic.containers_credentials({url_prefix: None})
    repo = ic.Repository.open(
        storage,
        config=repo_config,
        authorize_virtual_chunk_access=authorize_virtual_chunk_access,
    )
    session = repo.readonly_session(branch)
    return xr.open_zarr(session.store, group=group, zarr_format=zarr_format, consolidated=False, chunks=chunks)


def configure_gpu(gpu_device: int | None) -> None:
    """Configure TensorFlow GPU visibility before the first TF import.

    Must be called before ``import tensorflow`` in the calling scope.
    Setting ``CUDA_VISIBLE_DEVICES`` after TF has been imported has no effect
    because TF caches device enumeration on first import.

    Args:
        gpu_device: Index of the GPU to use, or None for CPU-only.

    Raises:
        RuntimeError: If ``gpu_device`` is specified but no GPUs are detected.
        IndexError: If ``gpu_device`` is out of range for the available GPUs.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpu_device is None:
        tf.config.set_visible_devices([], "GPU")
        return
    if not gpus:
        raise RuntimeError(f"gpu_device={gpu_device} requested but no GPUs detected.")
    if gpu_device >= len(gpus):
        raise IndexError(f"gpu_device={gpu_device} is out of range — {len(gpus)} GPU(s) available.")
    tf.config.set_visible_devices(gpus[gpu_device], "GPU")
    tf.config.experimental.set_memory_growth(gpus[gpu_device], True)


def limit_workers_for_slurm(max_workers: int = 16) -> int:
    """Determine the number of workers to use for parallelism in a SLURM job.

    This method reads the number of CPUs allocated to the current SLURM job and returns a worker count
    that is the minimum of the allocated CPUs and a specified maximum. This helps to avoid overwhelming
    the data store when loading data and computing normalization stats.

    Args:
        max_workers: The maximum number of workers to use. Default is 16.

    Returns:
        The number of workers to use for parallelism, limited by the allocated CPUs and the specified maximum.
    """
    # We use slurm to run this; this captures the number of CPUs allocated to the job and configures dask to use them
    # for parallelism when loading data and computing normalization stats.
    cpu_count = slurm_cpu_count()
    zarr.config.update({"threading.max_workers": cpu_count})

    # Limit the number of workers to avoid overwhelming the store
    data_workers = min(cpu_count, max_workers)
    return data_workers
