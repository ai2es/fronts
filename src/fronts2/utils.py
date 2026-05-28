from __future__ import annotations

import subprocess
from collections import namedtuple
from collections.abc import Callable
from typing import Any, TypeVar

import dacite
import icechunk as ic
import numpy as np
import xarray as xr
import yaml
from xarray.core.indexes import IndexSelResult, PandasIndex, _query_slice
from xarray.core.indexing import _expand_slice

T = TypeVar("T")
BoundingBox = namedtuple("BoundingBox", ["lat_min", "lat_max", "lon_min", "lon_max"])
XArrayType = xr.Dataset | xr.DataArray
TransformFunc = Callable[[XArrayType], XArrayType]


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


def attach_periodic_lon_index(data: XArrayType) -> XArrayType:
    """Attach a 360°-period :class:`PeriodicBoundaryIndex` to ``longitude``.

    Replaces the default ``PandasIndex`` so wrap-crossing
    ``.sel(longitude=slice(...))`` queries work via
    :func:`_wrap_lon_slice`.

    Args:
        data: Dataset or DataArray with a 1-D longitude coordinate.

    Returns a copy of the input with the longitude index replaced by a
        ``PeriodicBoundaryIndex``.
    """
    return data.drop_indexes("longitude").set_xindex("longitude", index_cls=PeriodicBoundaryIndex, period=360)


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
