r"""Plot model outputs: case study predictions and performance diagrams.

Two subcommands:

    case-study — Run model inference on a single timestep and save a probability map.
    performance-diagrams — Read pre-computed stats NetCDFs and save 4-panel figures.

Usage:
    pixi run -e schooner python src/fronts/plot/plot.py case-study \
        --config_path configs/schooner_train.yaml

    pixi run -e schooner python src/fronts/plot/plot.py performance-diagrams \
        --stats_dir ~/models/fronts/stats --mask land
"""

import argparse
import datetime
import logging
import os
from typing import Any, TypedDict

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import xarray as xr
from matplotlib import cm, colors
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FixedLocator

from fronts import utils
from fronts.data import config, datasets, inputs, targets
from fronts.model import SharedTargetModel
from fronts.plot.utils import plot_background, truncated_colormap

log = logging.getLogger(__name__)

FRONT_COLORS: dict[str, str] = {
    "CF": "blue",
    "WF": "red",
    "SF": "limegreen",
    "OF": "darkviolet",
    "DL": "chocolate",
}

CONTOUR_CMAPS: dict[str, str] = {
    "CF": "Blues",
    "WF": "Reds",
    "SF": "Greens",
    "OF": "Purples",
    "DL": "copper_r",
}

FRONT_NAMES: dict[str, str] = {
    "CF": "Cold front",
    "WF": "Warm front",
    "SF": "Stationary front",
    "OF": "Occluded front",
    "DL": "Dryline",
}

FRONT_TYPE_CLASS_INDEX: dict[str, int] = {"CF": 1, "WF": 2, "SF": 3, "OF": 4, "DL": 5}


class _LayoutConfig(TypedDict):
    table_axis_extent: tuple[float, float, float, float]
    table_scale: tuple[float, float]
    table_title_kwargs: dict[str, Any]
    spatial_axis_extent: tuple[float, float, float, float]
    cbar_kwargs: dict[str, str | float | int]
    spatial_plot_xlabels: list[int]
    spatial_plot_ylabels: list[int]


_DOMAIN_LAYOUT: dict[str, _LayoutConfig] = {
    "conus": {
        "table_axis_extent": (0.063, -0.038, 0.39, 0.239),
        "table_scale": (1.0, 3.3),
        "table_title_kwargs": {"x": 0.5, "y": 0.098, "pad": -4},
        "spatial_axis_extent": (0.5, -0.582, 0.512, 0.544),
        "cbar_kwargs": {"label": "CSI", "pad": 0, "shrink": 1},
        "spatial_plot_xlabels": [-140, -105, -70],
        "spatial_plot_ylabels": [30, 40, 50],
    },
    "full": {
        "table_axis_extent": (0.063, -0.038, 0.39, 0.229),
        "table_scale": (1.0, 2.8),
        "table_title_kwargs": {"x": 0.5, "y": 0.096, "pad": -4},
        "spatial_axis_extent": (0.523, -0.5915, 0.48, 0.66),
        "cbar_kwargs": {"label": "CSI", "pad": 0, "shrink": 0.675},
        "spatial_plot_xlabels": [-150, -120, -90, -60, -30, 0, 120, 150, 180],
        "spatial_plot_ylabels": [0, 20, 40, 60, 80],
    },
}


def _parse_front_types(aggregate_ds: xr.Dataset) -> list[str]:
    """Infer front type labels from variable names (tp_{FT})."""
    return [
        v[3:]
        for v in aggregate_ds.data_vars
        if isinstance(v, str) and v.startswith("tp_") and not v.startswith("tp_spatial_")
    ]


def plot_performance_diagrams(
    front_type: str,
    aggregate_ds: xr.Dataset,
    spatial_ds: xr.Dataset,
    mask: str | None,
    coordinates: utils.BoundingBox,
    map_neighborhood: int,
    output_type: str,
    outdir: str,
) -> None:
    """Generate and save a 4-panel performance figure for one front type.

    Args:
        front_type: Front type key (e.g. "CF").
        aggregate_ds: Aggregated stats Dataset with dims (neighborhood, threshold).
        spatial_ds: Spatial stats Dataset with dims (lat, lon, neighborhood, threshold).
        mask: "land", "ocean", or None — used only for the figure title and filename.
        coordinates: Spatial bounding box as [lat_min, lat_max, lon_min, lon_max].
        map_neighborhood: Neighbourhood radius (km) for the spatial CSI map panel.
        output_type: Output image format (e.g. "png").
        outdir: Directory to write the output file.
    """
    layout_key = "full" if (coordinates.lon_max - coordinates.lon_min) > 150 else "conus"
    layout = _DOMAIN_LAYOUT[layout_key]

    thresholds = aggregate_ds["threshold"].values  # (100,)

    a = aggregate_ds[f"tp_{front_type}"].values  # (neighborhood, threshold)
    b = aggregate_ds[f"fp_{front_type}"].values
    c = aggregate_ds[f"fn_{front_type}"].values
    d = aggregate_ds[f"tn_{front_type}"].values

    pod = np.divide(a, a + c, out=np.zeros_like(a), where=(a + c) > 0)
    sr = np.divide(a, a + b, out=np.zeros_like(a), where=(a + b) > 0)

    sr_matrix, pod_matrix = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
    csi_matrix = 1 / ((1 / sr_matrix) + (1 / pod_matrix) - 1)
    fb_matrix = pod_matrix * (sr_matrix**-1)
    csi_levels = np.linspace(0, 1, 11)
    fb_levels = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3]
    axis_ticks = np.arange(0, 1.01, 0.1)
    axis_ticklabels = np.arange(0, 100.1, 10).astype(int)

    num_forecasts = (a + b)[0, :]
    total_pixels_approx = (a + b + c + d)[0, 0]
    if total_pixels_approx > 0:
        relative_forecast_fraction = 100 * num_forecasts / total_pixels_approx
    else:
        relative_forecast_fraction = num_forecasts * 0

    tp_diff = np.abs(np.diff(a, axis=-1))
    fp_diff = np.abs(np.diff(b, axis=-1))
    denom = tp_diff + fp_diff
    observed_relative_frequency = np.divide(tp_diff, denom, out=np.zeros_like(tp_diff), where=denom > 0)

    tp_sp = spatial_ds[f"tp_spatial_{front_type}"]
    fp_sp = spatial_ds[f"fp_spatial_{front_type}"]
    fn_sp = spatial_ds[f"fn_spatial_{front_type}"]
    spatial_csi_map = (tp_sp / (tp_sp + fp_sp + fn_sp)).max("threshold")

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    ax0, ax1 = axs

    cs = ax0.contour(sr_matrix, pod_matrix, fb_matrix, fb_levels, colors="black", linewidths=0.5, linestyles="--")
    ax0.clabel(cs, fb_levels, fontsize=8)
    csi_contour = ax0.contourf(sr_matrix, pod_matrix, csi_matrix, csi_levels, cmap="Blues")
    cbar = fig.colorbar(csi_contour, ax=ax0, pad=0.02, label="Critical Success Index (CSI)")
    cbar.set_ticks(axis_ticks.tolist())

    ax1_bar = ax1.twinx()
    ax1_bar.set_ylabel("Percentage of Grid Points with Forecasts [bars]")
    ax1_bar.yaxis.set_major_locator(plt.LinearLocator(11))
    ax1_bar.bar(thresholds[:-1], relative_forecast_fraction[1:], color="blue", width=0.005, alpha=0.25)
    ax1.plot(thresholds, thresholds, color="black", linestyle="--", linewidth=0.5, label="Perfect Reliability")

    boundary_colors = ["red", "purple", "brown", "darkorange", "darkgreen"]
    cell_text = []

    for boundary, color in enumerate(boundary_colors):
        csi = np.power((1 / sr[boundary]) + (1 / pod[boundary]) - 1, -1)
        hss = (
            2
            * ((a[boundary] * d[boundary]) - (b[boundary] * c[boundary]))
            / (
                ((a[boundary] + c[boundary]) * (c[boundary] + d[boundary]))
                + ((a[boundary] + b[boundary]) * (b[boundary] + d[boundary]))
            )
        )

        max_csi = np.nanmax(csi)
        max_idx = np.where(csi == max_csi)[0]
        max_csi_pod = pod[boundary][max_idx][0]
        max_csi_sr = sr[boundary][max_idx][0]
        max_csi_fb = max_csi_pod / max_csi_sr if max_csi_sr > 0 else float("nan")
        max_hss = hss[max_idx][0]
        far = 1 - max_csi_sr

        cell_text.append(
            [
                rf"$\bf{{{max_csi:.3f}}}$",
                rf"$\bf{{{max_hss:.3f}}}$",
                rf"$\bf{{{max_csi_pod * 100:.1f}}}$",
                rf"$\bf{{{far * 100:.1f}}}$",
                rf"$\bf{{{max_csi_fb:.3f}}}$",
            ]
        )

        ax0.plot(max_csi_sr, max_csi_pod, color=color, marker="*", markersize=10)
        ax0.plot(sr[boundary], pod[boundary], color=color, linewidth=1)
        ax1.plot(thresholds[:-1], observed_relative_frequency[boundary], color=color, linewidth=1)

    ax0.set_xticklabels(axis_ticklabels[::-1])
    ax0.set_xlabel("False Alarm Rate (FAR; %)")
    ax0.set_ylabel("Probability of Detection (POD; %)")
    ax0.set_title(r"$\bf{a)}$ $\bf{CSI}$ $\bf{diagram}$")
    ax1.set_xticklabels(axis_ticklabels)
    ax1.set_xlabel("Forecast Probability (uncalibrated; %)")
    ax1.set_ylabel("Observed Relative Frequency (%) [lines]")
    ax1.set_title(r"$\bf{b)}$ $\bf{Reliability}$ $\bf{diagram}$")

    for ax in axs:
        ax.set_xticks(axis_ticks)
        ax.set_yticks(axis_ticks)
        ax.set_yticklabels(axis_ticklabels)
        ax.grid(color="black", alpha=0.1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    table_axis = plt.axes(layout["table_axis_extent"])
    table_axis.set_title(r"$\bf{c)}$ $\bf{Data}$ $\bf{table}$", **layout["table_title_kwargs"])
    table_axis.axis("off")
    stats_table = table_axis.table(
        cellText=cell_text,
        rowLabels=["50 km", "100 km", "150 km", "200 km", "250 km"],
        rowColours=boundary_colors,
        colLabels=["CSI", "HSS", "POD %", "FAR %", "FB"],
        cellLoc="center",
    )
    stats_table.scale(*layout["table_scale"])
    for cell in stats_table._cells:  # pyrefly: ignore[missing-attribute]
        stats_table._cells[cell].set_alpha(0.7)  # pyrefly: ignore[missing-attribute]
        stats_table._cells[cell].set_text_props(fontproperties=FontProperties(size="x-large", stretch="expanded"))  # pyrefly: ignore[missing-attribute]

    csi_cmap = truncated_colormap("gnuplot2", maxval=0.9, n=10)
    spatial_axis = plt.axes(layout["spatial_axis_extent"], projection=ccrs.Miller(central_longitude=250))
    plot_background(
        extent=[coordinates.lon_min, coordinates.lon_max, coordinates.lat_min, coordinates.lat_max],
        ax=spatial_axis,
    )

    norm_probs = colors.Normalize(vmin=0.1, vmax=1)
    xr.where(spatial_csi_map >= 0.1, spatial_csi_map, float("nan")).sel(neighborhood=map_neighborhood).plot(
        ax=spatial_axis,
        x="longitude",
        y="latitude",
        norm=norm_probs,
        cmap=csi_cmap,
        transform=ccrs.PlateCarree(),
        alpha=0.6,
        cbar_kwargs=layout["cbar_kwargs"],
    )
    spatial_axis.set_title(rf"$\bf{{d)}}$ $\bf{{{map_neighborhood}}}$ $\bf{{km}}$ $\bf{{CSI}}$ $\bf{{map}}$")
    gl = spatial_axis.gridlines(draw_labels=True, zorder=0, dms=True, x_inline=False, y_inline=False)  # pyrefly: ignore[missing-attribute]
    gl.right_labels = False
    gl.top_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xlocator = FixedLocator(layout["spatial_plot_xlabels"])
    gl.ylocator = FixedLocator(layout["spatial_plot_ylabels"])
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 8}

    mask_label = f" ({mask})" if mask else ""
    plt.suptitle(f"{FRONT_NAMES.get(front_type, front_type)}s{mask_label}", fontsize=20)

    os.makedirs(outdir, exist_ok=True)
    mask_suffix = f"_{mask}" if mask else ""
    filename = os.path.join(outdir, f"performance_{front_type}{mask_suffix}.{output_type}")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=500)
    plt.close()
    print(f"Saved: {filename}")


def _load_prediction(
    model: Any,
    era5_ds: xr.Dataset,
    variables: list[str],
    front_types: list[str],
    init_time: np.datetime64,
) -> xr.Dataset:
    """Run model inference on a single timestep and return per-type probabilities.

    Args:
        model: Loaded Keras model with baked-in normalization.
        era5_ds: ERA5 Dataset from icechunk store.
        variables: ERA5 variable names to pass to ``inputs_ds_to_dataarray``.
        front_types: Front type labels to include in the output Dataset.
        init_time: Target timestep as a numpy datetime64.

    Returns:
        Dataset with one variable per front type, dims (latitude, longitude).
    """
    era5_t = era5_ds.sel(time=[init_time])
    x_np = inputs.inputs_ds_to_dataarray(era5_t, variables).values[0].astype(np.float32)

    pred = model(x_np[np.newaxis], training=False)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    pred_np = pred.numpy()[0].astype(np.float32)  # (lat, lon, n_classes)

    for ft in front_types:
        idx = FRONT_TYPE_CLASS_INDEX[ft]
        log.info("%s (class %d): max prob = %.3f", ft, idx, pred_np[:, :, idx].max())

    lats = era5_t["latitude"].values
    lons = era5_t["longitude"].values
    ds = xr.Dataset(coords={"latitude": lats, "longitude": lons})
    for ft in front_types:
        ds[ft] = (["latitude", "longitude"], pred_np[:, :, FRONT_TYPE_CLASS_INDEX[ft]])
    return ds


def _load_truth(fronts_ds: xr.Dataset, init_time: np.datetime64) -> xr.DataArray:
    """Return remapped integer front labels at a single timestep."""
    return targets.remap_fronts(fronts_ds["identifier"].sel(time=init_time))


def _plot_front_probability_contours(
    fig: Any,
    ax: Any,
    probs_masked: xr.Dataset,
    front_types: list[str],
    levels: np.ndarray,
    prob_mask: float,
    prob_int: float,
    filled_contours: bool,
    open_contours: bool,
) -> None:
    """Draw filled/open per-front-type probability contours plus a shared legend colorbar.

    Args:
        fig: Figure to attach the per-front-type colorbar axes to.
        ax: Cartopy axes to draw the contours on.
        probs_masked: Per-front-type probability Dataset, dims (latitude, longitude).
        front_types: Front type keys to plot (e.g. ["CF", "WF"]).
        levels: Contour levels, e.g. ``np.arange(0, 1 + prob_int, prob_int)``.
        prob_mask: Probabilities at or below this value are not drawn.
        prob_int: Spacing between contour levels.
        filled_contours: Draw filled probability contours with a per-front-type colorbar.
        open_contours: Draw open (line) probability contours.
    """
    n_colors = int(1 / prob_int) + 1
    front_colors = [FRONT_COLORS[ft] for ft in front_types]

    cbar_x_start = 0.85
    cbar_front_labels = []
    cbar_front_ticks = []

    for front_no, ft in enumerate(front_types, start=1):
        if ft not in probs_masked:
            continue

        if filled_contours:
            cmap_probs = truncated_colormap(CONTOUR_CMAPS[ft], minval=0.1, n=n_colors)
            norm_probs = colors.Normalize(vmin=0, vmax=1.01)
            probs_masked[ft].plot.contourf(
                ax=ax,
                x="longitude",
                y="latitude",
                norm=norm_probs,
                levels=levels,
                cmap=cmap_probs,
                transform=ccrs.PlateCarree(),
                add_colorbar=False,
            )
            cbar_ax = fig.add_axes((cbar_x_start + (front_no * 0.015), 0.24, 0.015, 0.64))
            cbar = plt.colorbar(
                cm.ScalarMappable(norm=norm_probs, cmap=cmap_probs),
                cax=cbar_ax,
                boundaries=levels[1:],
            )
            cbar.set_ticklabels([])
            if front_no == len(front_types):
                cbar.set_label("Probability (uncalibrated)", rotation=90)
                tick_vals = np.around(np.arange(prob_mask, 1 + prob_int, prob_int), 2)
                cbar.set_ticks(tick_vals.tolist())
                cbar.set_ticklabels([str(v) for v in tick_vals])

        if open_contours:
            probs_masked[ft].plot.contour(
                ax=ax,
                x="longitude",
                y="latitude",
                levels=levels,
                colors=front_colors[front_no - 1],
                transform=ccrs.PlateCarree(),
                alpha=0.75,
                add_colorbar=False,
            )

        cbar_front_labels.append(FRONT_NAMES.get(ft, ft))
        cbar_front_ticks.append(front_no + 0.5)

    cmap_front = colors.ListedColormap(front_colors, name="from_list", N=len(front_colors))
    norm_front = colors.Normalize(vmin=1, vmax=len(front_colors) + 1)
    cbar_front = plt.colorbar(
        cm.ScalarMappable(norm=norm_front, cmap=cmap_front),
        ax=ax,
        alpha=0.75,
        orientation="horizontal",
        shrink=0.5,
        pad=0.02,
    )
    cbar_front.set_ticks(cbar_front_ticks)
    cbar_front.set_ticklabels(cbar_front_labels)
    cbar_front.set_label(r"$\bf{Front}$ $\bf{type}$")


def plot_test_prediction(
    lats: np.ndarray,
    lons: np.ndarray,
    probs_ds: xr.Dataset,
    front_types: list[str],
    truth_da: xr.DataArray | None = None,
    prob_mask: float = 0.1,
    prob_interval: float = 0.1,
    filled_contours: bool = True,
    open_contours: bool = True,
    title: str = "",
) -> Any:
    """Build a single-timestep probability map figure from in-memory arrays.

    Unlike ``plot_case_study``, this does not open any icechunk store — it plots
    whatever prediction/truth arrays the caller already has in memory (e.g. a single
    active test day evaluated during a training callback).

    Args:
        lats: 1-D latitude array.
        lons: 1-D longitude array.
        probs_ds: Per-front-type predicted probability Dataset, dims (latitude, longitude).
        front_types: Front type keys to plot.
        truth_da: Optional integer truth labels (0=background), dims (latitude, longitude).
        prob_mask: Probabilities at or below this value are not drawn.
        prob_interval: Spacing between contour levels.
        filled_contours: Draw filled probability contours with a per-front-type colorbar.
        open_contours: Draw open (line) probability contours.
        title: Figure title (e.g. the timestep).

    Returns:
        The created Figure.
    """
    levels = np.around(np.arange(0, 1 + prob_interval, prob_interval), 2)
    probs_masked = xr.where(probs_ds > prob_mask, probs_ds, float("nan"))

    central_lon = ((float(lons.min()) + float(lons.max())) / 2) % 360
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(22, 8),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=central_lon)},
    )
    plot_background(
        extent=[float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())],
        ax=ax,
        linewidth=0.5,
    )

    _plot_front_probability_contours(
        fig,
        ax,
        probs_masked,
        front_types,
        levels,
        prob_mask,
        prob_interval,
        filled_contours=filled_contours,
        open_contours=open_contours,
    )

    if truth_da is not None:
        front_colors = [FRONT_COLORS[ft] for ft in front_types]
        cmap_front = colors.ListedColormap(front_colors, name="from_list", N=len(front_colors))
        norm_front = colors.Normalize(vmin=1, vmax=len(front_colors) + 1)
        xr.where(truth_da == 0, float("nan"), truth_da).plot(
            ax=ax,
            x="longitude",
            y="latitude",
            cmap=cmap_front,
            norm=norm_front,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
        )
        ax.set_title("Splines: NOAA fronts", loc="right")

    if title:
        ax.set_title(title, loc="left")

    return fig


def plot_performance_diagram_lite(
    front_type: str,
    thresholds: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    tn: np.ndarray,
    fn: np.ndarray,
    title: str = "",
) -> Any:
    """Build a lightweight 1-panel CSI/POD-vs-SR diagram with a CSI/HSS/POD/FAR table.

    A cheaper alternative to ``plot_performance_diagrams`` for use inside a training
    callback: no spatial CSI map, no reliability diagram, and the caller supplies
    already-accumulated (single-neighborhood) TP/FP/TN/FN counts rather than the full
    multi-neighborhood spatial sweep computed by ``evaluation/compute_stats.py``.

    Args:
        front_type: Front type key, used only for the title.
        thresholds: 1-D probability thresholds, shape (T,).
        tp: True positive counts, shape (T,).
        fp: False positive counts, shape (T,).
        tn: True negative counts, shape (T,).
        fn: False negative counts, shape (T,).
        title: Prefix added to the figure title (e.g. the region name).

    Returns:
        The created Figure.
    """
    pod = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    sr = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    csi = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
    hss = 2 * np.divide(
        (tp * tn) - (fp * fn),
        ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn)),
        out=np.zeros_like(tp),
        where=((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn)) > 0,
    )

    sr_matrix, pod_matrix = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
    csi_matrix = 1 / ((1 / sr_matrix) + (1 / pod_matrix) - 1)
    fb_matrix = pod_matrix * (sr_matrix**-1)
    csi_levels = np.linspace(0, 1, 11)
    fb_levels = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3]

    max_idx = int(np.nanargmax(csi))

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    cs = ax.contour(sr_matrix, pod_matrix, fb_matrix, fb_levels, colors="black", linewidths=0.5, linestyles="--")
    ax.clabel(cs, fb_levels, fontsize=8)
    csi_contour = ax.contourf(sr_matrix, pod_matrix, csi_matrix, csi_levels, cmap="Blues")
    fig.colorbar(csi_contour, ax=ax, pad=0.02, label="Critical Success Index (CSI)")

    ax.plot(sr[max_idx], pod[max_idx], color="red", marker="*", markersize=10)
    ax.plot(sr, pod, color="red", linewidth=1)
    ax.set_xlabel("Success Ratio (1 - FAR)")
    ax.set_ylabel("Probability of Detection (POD)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color="black", alpha=0.1)

    far = 1 - sr[max_idx]
    table_text = [[f"{csi[max_idx]:.3f}", f"{hss[max_idx]:.3f}", f"{pod[max_idx] * 100:.1f}", f"{far * 100:.1f}"]]
    table_ax = fig.add_axes((0.15, -0.05, 0.7, 0.1))
    table_ax.axis("off")
    table_ax.table(
        cellText=table_text,
        colLabels=["CSI", "HSS", "POD %", "FAR %"],
        cellLoc="center",
        loc="center",
    )

    fig.suptitle(f"{title} {FRONT_NAMES.get(front_type, front_type)}".strip())
    fig.tight_layout()
    return fig


def plot_case_study(predict_cfg: config.PredictConfig, data_cfg: datasets.DatasetConfig) -> None:
    """Run inference and generate a single-timestep probability map.

    Args:
        predict_cfg: Prediction configuration (model, coordinates, plot options).
        data_cfg: Data configuration (icechunk store paths, variables).
    """
    utils.configure_gpu(predict_cfg.gpu_device)

    log.info("Loading model from %s …", predict_cfg.model_path)
    model = tf.keras.models.load_model(
        predict_cfg.model_path, compile=False, custom_objects={"SharedTargetModel": SharedTargetModel}
    )

    init_time = np.datetime64(predict_cfg.init_time)

    plot_bb = predict_cfg.coordinates

    ic_era5 = data_cfg.inputs_icechunk_config
    log.info("Opening ERA5 store …")
    era5_ds = utils.open_readonly_icechunk_store(
        ic_era5.store_path,
        ic_era5.branch_name,
        group=ic_era5.group_name,
        zarr_format=ic_era5.zarr_format,
        virtual_chunk_local_path=ic_era5.virtual_chunk_local_path,
    )
    era5_ds = utils.unwrap_longitude(utils.select_spatial_domain(era5_ds, plot_bb))

    probs_ds = _load_prediction(model, era5_ds, data_cfg.variables, predict_cfg.front_types, init_time)

    truth_da = None
    if predict_cfg.targets:
        ic_fronts = data_cfg.targets_icechunk_config
        log.info("Opening fronts store …")
        fronts_ds = utils.open_readonly_icechunk_store(
            ic_fronts.store_path,
            ic_fronts.branch_name,
            group=ic_fronts.group_name,
            zarr_format=ic_fronts.zarr_format,
            virtual_chunk_local_path=ic_fronts.virtual_chunk_local_path,
        )
        truth_da = _load_truth(utils.select_spatial_domain(fronts_ds, plot_bb), init_time).compute()

    front_types = predict_cfg.front_types
    lats = era5_ds["latitude"].values
    lons = era5_ds["longitude"].values

    fig = plot_test_prediction(
        lats=lats,
        lons=lons,
        probs_ds=probs_ds,
        front_types=front_types,
        truth_da=truth_da,
        prob_mask=predict_cfg.prob_mask,
        prob_interval=predict_cfg.prob_interval,
        filled_contours=predict_cfg.filled_contours,
        open_contours=predict_cfg.open_contours,
        title=f"ERA5 {init_time}z",
    )

    os.makedirs(predict_cfg.outdir, exist_ok=True)
    ts = predict_cfg.init_time.strftime("%Y%m%d%H")
    outfile = os.path.join(
        predict_cfg.outdir,
        f"prediction_{ts}.png",
    )
    fig.savefig(outfile, bbox_inches="tight", dpi=500)
    plt.close(fig)
    print(f"Saved: {outfile}")


def main() -> None:
    """Dispatch to the requested plot subcommand."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Plot model outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cs = subparsers.add_parser("case-study", help="Single-timestep probability map.")
    cs.add_argument("--config_path", type=str, required=True, help="Path to config YAML.")

    pd = subparsers.add_parser("performance-diagrams", help="4-panel performance diagram per front type.")
    pd.add_argument(
        "--stats_dir",
        type=str,
        required=True,
        help="Directory containing stats_aggregate_{mask}.nc and stats_spatial_{mask}.nc.",
    )
    pd.add_argument(
        "--mask",
        type=str,
        default=None,
        choices=["land", "ocean"],
        help="'land' or 'ocean' - selects which stats files to load.",
    )
    pd.add_argument(
        "--front_types",
        type=str,
        nargs="+",
        default=None,
        help="Front types to plot (default: all in file).",
    )
    pd.add_argument(
        "--coordinates",
        type=float,
        nargs=4,
        default=[0.25, 80.0, 130.0, 369.75],
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Spatial bounding box as lat_min lat_max lon_min lon_max.",
    )
    pd.add_argument("--map_neighborhood", type=int, default=250, help="Neighbourhood (km) for spatial CSI map.")
    pd.add_argument("--output_type", type=str, default="png", help="Image format (png, pdf, etc.).")
    pd.add_argument("--outdir", type=str, default=None, help="Output directory (defaults to --stats_dir).")

    args = parser.parse_args()

    if args.command == "case-study":
        predict_cfg: config.PredictConfig = utils.open_config_yaml_as_dataclass(
            args.config_path,
            config.PredictConfig,
            config_key="predict_config",
            type_hooks={
                datetime.datetime: lambda d: datetime.datetime.fromisoformat(str(d)),
                utils.BoundingBox: lambda d: utils.BoundingBox(*d),
            },
        )
        data_cfg: datasets.DatasetConfig = utils.open_config_yaml_as_dataclass(
            args.config_path, datasets.DatasetConfig, config_key="data_config"
        )
        plot_case_study(predict_cfg, data_cfg)

    elif args.command == "performance-diagrams":
        mask_suffix = f"_{args.mask}" if args.mask else ""
        aggregate_ds = xr.open_dataset(os.path.join(args.stats_dir, f"stats_aggregate{mask_suffix}.nc"))
        spatial_ds = xr.open_dataset(os.path.join(args.stats_dir, f"stats_spatial{mask_suffix}.nc"))

        front_types = args.front_types or _parse_front_types(aggregate_ds)
        outdir = args.outdir or args.stats_dir

        for ft in front_types:
            print(f"Plotting {ft} …")
            plot_performance_diagrams(
                front_type=ft,
                aggregate_ds=aggregate_ds,
                spatial_ds=spatial_ds,
                mask=args.mask,
                coordinates=utils.BoundingBox(*args.coordinates),
                map_neighborhood=args.map_neighborhood,
                output_type=args.output_type,
                outdir=outdir,
            )


if __name__ == "__main__":
    main()
