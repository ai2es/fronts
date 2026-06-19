import hashlib
import logging
import os
import pathlib
import tempfile

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def inputs_ds_to_dataarray(ds: xr.Dataset, variables: list[str]) -> xr.DataArray:
    """Convert a gridded input Dataset to a lazy 4D DataArray without materializing data.

    Uses xarray's to_array and stack so no data is loaded from disk until
    iteration. Pressure-level variables come first, ordered level-outer,
    variable-inner (all variables at level[0], then all variables at level[1],
    etc.), followed by one channel per single-level variable in the order
    requested. Channel labels are ``{variable}_{level}`` for pressure-level
    variables and the bare variable name for single-level ones. Time-invariant
    variables (e.g. land_sea_mask) are broadcast along the dataset's time
    coordinate.

    Args:
        ds: Dataset containing the requested variables with dims
            (time, level, latitude, longitude) or (time, latitude, longitude);
            time-invariant variables may omit the time dim.
        variables: Variable names to select from ds.

    Returns:
        DataArray of shape (time, latitude, longitude, channel) with dtype float32,
        where channel = n_levels * n_level_variables + n_single_level_variables.
    """
    level_vars = [v for v in variables if "level" in ds[v].dims]
    single_vars = [v for v in variables if "level" not in ds[v].dims]

    pieces = []
    if level_vars:
        stacked = (
            ds[level_vars]
            .to_array(dim="variable")
            .transpose("time", "latitude", "longitude", "level", "variable")
            .stack(channel=("level", "variable"))
        )
        labels = [f"{var}_{int(level)}" for level, var in stacked.indexes["channel"]]
        stacked = stacked.reset_index("channel").drop_vars(["level", "variable"]).assign_coords(channel=labels)
        pieces.append(stacked.reset_coords(drop=True))
    if single_vars:
        broadcast = {}
        for var in single_vars:
            da = ds[var]
            if "time" not in da.dims:
                if "time" not in ds.coords:
                    raise ValueError(
                        f"Variable '{var}' has no time dimension and the dataset has no time "
                        "coordinate to broadcast it along."
                    )
                da = da.expand_dims(time=ds["time"])
            broadcast[var] = da
        single = xr.Dataset(broadcast).to_array(dim="channel").transpose("time", "latitude", "longitude", "channel")
        pieces.append(single.reset_coords(drop=True))

    if not pieces:
        raise ValueError("No variables requested.")
    result = pieces[0] if len(pieces) == 1 else xr.concat(pieces, dim="channel", coords="minimal")
    return result.astype(np.float32)


def compute_norm_stats(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and variance over the full DataArray in one pass.

    Builds both reductions as a single dask graph so each chunk is read from
    disk once and the full array is never materialized in memory. Suitable for
    passing directly to a Keras Normalization layer.

    Args:
        da: DataArray of shape (time, latitude, longitude, channel).

    Returns:
        Tuple of (mean, variance), each of shape (n_channels,) as float32.

    Raises:
        ValueError: If the computed statistics contain NaN values.
    """
    reduction_dims = ["time", "latitude", "longitude"]
    stats = xr.Dataset(
        {
            "mean": da.mean(dim=reduction_dims, skipna=False),
            "variance": da.var(dim=reduction_dims, skipna=False),
        }
    ).compute()
    mean = stats["mean"].values.astype(np.float32)
    variance = stats["variance"].values.astype(np.float32)
    if np.isnan(mean).any() or np.isnan(variance).any():
        raise ValueError(
            "NaN in normalization statistics — ERA5 data contains missing values. "
            "Check the icechunk store for corrupted or incomplete channels."
        )
    return mean, variance


def load_or_compute_norm_stats(
    da: xr.DataArray,
    cache_dir: str | None,
    cache_key_parts: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached normalization statistics or compute and cache them.

    The cache key is a SHA-256 hash of cache_key_parts (e.g. the store snapshot
    ID), so stats are reused only when every part matches a previous run. Cached
    stats whose channel count no longer matches ``da`` are ignored and recomputed.

    Args:
        da: DataArray of shape (time, latitude, longitude, channel).
        cache_dir: Directory for cached stats files. None disables caching.
        cache_key_parts: Strings that uniquely determine the statistics.

    Returns:
        Tuple of (mean, variance), each of shape (n_channels,) as float32.
    """
    snapshot_id = cache_key_parts[0]
    if cache_dir is None:
        logger.info("Normalization cache disabled; computing stats for snapshot %s.", snapshot_id)
        return compute_norm_stats(da)

    key = hashlib.sha256("\x1f".join(cache_key_parts).encode()).hexdigest()[:16]
    cache_path = pathlib.Path(cache_dir) / f"norm_stats_{key}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached["mean"].shape[0] == da.sizes["channel"]:
            logger.info("Reusing cached normalization stats for snapshot %s from %s.", snapshot_id, cache_path)
            return cached["mean"], cached["variance"]
        logger.warning(
            "Cached normalization stats for snapshot %s have %d channels but data has %d; recomputing.",
            snapshot_id,
            cached["mean"].shape[0],
            da.sizes["channel"],
        )
    else:
        logger.info("No cached normalization stats for snapshot %s (key %s); computing.", snapshot_id, key)

    mean, variance = compute_norm_stats(da)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".npz.tmp", delete=False) as tmp:
        np.savez(tmp, mean=mean, variance=variance)
        tmp_path = tmp.name
    os.replace(tmp_path, cache_path)
    logger.info("Saved normalization stats for snapshot %s to %s.", snapshot_id, cache_path)
    return mean, variance
