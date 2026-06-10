from __future__ import annotations

import subprocess
from collections import namedtuple
from typing import Any, TypeVar

import dacite
import icechunk as ic
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from xarray.core.indexes import IndexSelResult, PandasIndex, _query_slice
from xarray.core.indexing import _expand_slice

T = TypeVar("T")
_XArray = TypeVar("_XArray", xr.Dataset, xr.DataArray)
BoundingBox = namedtuple("BoundingBox", ["lat_min", "lat_max", "lon_min", "lon_max"])


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


def open_config_yaml_as_dataclass(
    path: str,
    config_class: type[T],
    config_key: str | None = None,
    type_hooks: dict | None = None,
) -> T:
    """Open a YAML config file and parse it into a dataclass instance.

    Args:
        path: Path to the YAML config file.
        config_class: The dataclass to parse the config into.
        config_key: Optional key to extract a sub-dictionary from the YAML.
        type_hooks: Optional dictionary of type conversion functions.

    Returns:
        An instance of the specified dataclass.
    """
    # Open the YAML file and load it as a dictionary
    with open(path) as f:
        config_yaml = yaml.safe_load(f)

    # If a specific key is provided, extract that sub-dictionary for dataclass parsing
    if config_key:
        config_yaml = config_yaml[config_key]

    return dacite.from_dict(
        data_class=config_class,
        data=config_yaml,
        config=dacite.Config(check_types=False, type_hooks=type_hooks or {}),
    )


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
    return xr.open_zarr(session.store, group=group, zarr_format=zarr_format, consolidated=False)


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
