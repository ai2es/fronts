import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def era5_to_dataarray(ds: xr.Dataset, variables: list[str]) -> xr.DataArray:
    """Convert ERA5 Dataset to a lazy 4D DataArray without materializing data.

    Uses xarray's to_array and stack so no data is loaded from disk until
    iteration. Channel order is level-outer, variable-inner: all variables at
    level[0], then all variables at level[1], etc.

    Args:
        ds: ERA5 Dataset containing the requested variables, each with dims
            (time, level, latitude, longitude).
        variables: Variable names to select from ds.

    Returns:
        DataArray of shape (time, latitude, longitude, channel) with dtype float32,
        where channel = n_levels * len(variables).
    """
    return (
        ds[variables]
        .to_array(dim="variable")
        .transpose("time", "latitude", "longitude", "level", "variable")
        .stack(channel=("level", "variable"))
        .astype(np.float32)
    )


def compute_norm_stats(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and variance over the full DataArray lazily.

    Uses xarray's dask-backed reduction so the full array is never materialized.
    Suitable for passing directly to a Keras Normalization layer.

    Args:
        da: DataArray of shape (time, latitude, longitude, channel).

    Returns:
        Tuple of (mean, variance), each of shape (n_channels,) as float32.
    """
    spatial_dims = ["time", "latitude", "longitude"]
    mean = da.mean(dim=spatial_dims, skipna=False).values.astype(np.float32)
    variance = da.var(dim=spatial_dims, skipna=False).values.astype(np.float32)
    if np.isnan(mean).any() or np.isnan(variance).any():
        raise ValueError(
            "NaN in normalization statistics — ERA5 data contains missing values. "
            "Check the icechunk store for corrupted or incomplete channels."
        )
    return mean, variance
