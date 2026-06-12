import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def era5_to_dataarray(ds: xr.Dataset, variables: list[str]) -> xr.DataArray:
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
        pieces.append(stacked)
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
        pieces.append(single)

    if not pieces:
        raise ValueError("No variables requested.")
    result = pieces[0] if len(pieces) == 1 else xr.concat(pieces, dim="channel")
    return result.astype(np.float32)


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
