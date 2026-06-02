import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def plot_background(
    extent: list[float] | None = None,
    ax: plt.Axes | None = None,
    linewidth: float = 0.5,
    crs: ccrs.Projection | None = None,
) -> plt.Axes:
    """Add cartographic background features to a map axis.

    Args:
        extent: Plot boundaries as [lon_min, lon_max, lat_min, lat_max].
        ax: Existing axis to add features to. If None, creates a new axis.
        linewidth: Thickness of coastlines, borders, and state lines.
        crs: Coordinate reference system for the axis.

    Returns:
        The axis with background features added.
    """
    if crs is None:
        crs = ccrs.PlateCarree()
    if ax is None:
        ax = plt.axes(projection=crs)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=linewidth)  # pyrefly: ignore[missing-attribute]
    ax.add_feature(cfeature.BORDERS, linewidth=linewidth)  # pyrefly: ignore[missing-attribute]
    ax.add_feature(cfeature.STATES, linewidth=linewidth)  # pyrefly: ignore[missing-attribute]
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())  # pyrefly: ignore[missing-attribute]
    return ax


def truncated_colormap(
    cmap: str,
    minval: float = 0.0,
    maxval: float = 1.0,
    n: int = 256,
) -> mpl.colors.LinearSegmentedColormap:
    """Return a truncated slice of a named matplotlib colormap.

    Args:
        cmap: Named matplotlib colormap to truncate.
        minval: Start point in [0, 1) of the original colormap.
        maxval: End point in (0, 1] of the original colormap.
        n: Number of discrete colors in the output colormap.

    Returns:
        A new LinearSegmentedColormap covering [minval, maxval] of the source.
    """
    base = plt.get_cmap(cmap)
    return LinearSegmentedColormap.from_list(
        f"trunc({base.name},{minval:.2f},{maxval:.2f})",
        base(np.linspace(minval, maxval, n)),
    )
