import numpy as np
import xarray as xr

# Original front codes → experiment class indices
# 0 = no front (background), 1-4 kept as-is, 15 (dryline) → 5, all others → 0
FRONT_CLASS_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 15: 5}


def filter_timesteps(fronts_da: xr.DataArray, rng: np.random.Generator) -> np.ndarray:
    """Return a boolean keep-mask per timestep using the Justin et al. (2025) sampling rule.

    Retain a timestep unconditionally if every front type is present somewhere in the
    spatial domain; otherwise retain it with 50% probability. This balances class
    frequency without introducing seasonal bias (Justin et al. 2025, section 2b).

    Args:
        fronts_da: Raw identifier DataArray of shape (time, latitude, longitude) with
            original front codes (see ``FRONT_CLASS_MAP``).
        rng: Seeded generator used for the 50% draws.

    Returns:
        Boolean array of shape (time,).
    """
    # Compute any() over space before .compute() so only (n_codes, n_times) booleans
    # are materialised rather than the full spatial array.
    presence = xr.concat(
        [(fronts_da == code).any(dim=["latitude", "longitude"]) for code in FRONT_CLASS_MAP],
        dim="front_type",
    ).compute()
    has_all_types = presence.all(dim="front_type").values
    return has_all_types | (rng.random(len(has_all_types)) < 0.5)


def remap_fronts(da: xr.DataArray) -> xr.DataArray:
    """Map front codes to 6-class experiment labels without loading data.

    Classes: 0=none, 1=CF, 2=WF, 3=SF, 4=OF, 5=dryline. All other original codes map to
        0.

    Returns:
        Lazy int32 DataArray of the same shape as ``da``.
    """
    remapped = xr.full_like(da, 0, dtype=np.int32)
    for orig, new in FRONT_CLASS_MAP.items():
        remapped = xr.where(da == orig, new, remapped)
    return remapped


def _binary_dilate_2d(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Apply binary dilation with a cross-shaped structuring element for ``iterations`` steps.

    Uses scipy.ndimage when available; falls back to a pure-numpy implementation.

    Args:
        mask: Boolean 2-D array of shape (rows, cols).
        iterations: Number of dilation steps.

    Returns:
        Boolean 2-D array of the same shape with dilation applied.
    """
    try:
        import scipy.ndimage

        return scipy.ndimage.binary_dilation(mask, iterations=iterations)
    except ModuleNotFoundError:
        result = mask.copy()
        for _ in range(iterations):
            shifted = (
                np.roll(result, 1, axis=0)
                | np.roll(result, -1, axis=0)
                | np.roll(result, 1, axis=1)
                | np.roll(result, -1, axis=1)
                | result
            )
            result = shifted
        return result


def _dilate_one_timestep(arr: np.ndarray, dilation: int) -> np.ndarray:
    """Apply binary dilation to each non-background class for a single spatial snapshot.

    Args:
        arr: Float32 array of shape (latitude, longitude, class).
        dilation: Number of binary dilation iterations.

    Returns:
        Float32 array of shape (latitude, longitude, class) with dilated front classes
        and recomputed background.
    """
    n_classes = arr.shape[-1]
    result = np.zeros_like(arr)
    for cls in range(1, n_classes):
        dilated = _binary_dilate_2d(arr[..., cls].astype(bool), dilation)
        result[..., cls] = dilated.astype(np.float32)
    result[..., 0] = (1.0 - result[..., 1:].any(axis=-1)).astype(np.float32)
    return result


def dilate_fronts(da: xr.DataArray, dilation: int) -> xr.DataArray:
    """Dilate non-background front classes in a one-hot encoded DataArray.

    Each non-background class (index 1+) is dilated spatially using binary dilation.
    The background class (index 0) is recomputed as the complement of any active front
    class after dilation. Stays lazy via xr.apply_ufunc.

    Args:
        da: One-hot float32 DataArray of shape (time, latitude, longitude, class).
        dilation: Number of binary dilation iterations. Returns ``da`` unchanged when 0.

    Returns:
        Float32 DataArray of the same shape as ``da`` with dilated front classes.
    """
    if dilation == 0:
        return da
    return xr.apply_ufunc(
        _dilate_one_timestep,
        da,
        dilation,
        input_core_dims=[["latitude", "longitude", "class"], []],
        output_core_dims=[["latitude", "longitude", "class"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
    )


def one_hot_encode_to_dataarray(da: xr.DataArray, num_classes: int = 6) -> xr.DataArray:
    """One-hot encode a DataArray of integer class labels without loading data.

    Broadcasts ``da`` against a class axis so no data is materialized until
    xbatcher extracts a patch.

    Args:
        da: Integer DataArray of shape (time, latitude, longitude).
        num_classes: Number of output classes.

    Returns:
        Lazy float32 DataArray of shape (time, latitude, longitude, class).
    """
    classes = xr.DataArray(np.arange(num_classes), dims=["class"])
    return (da == classes).astype(np.float32)
