"""Multi-panel case-study prediction figures for model_1702, styled after AIES Fig. 14.

Renders one map panel per requested timestep with filled front-probability contours at 10%
intervals (cold = blue, warm = red, stationary = green, occluded = purple), state borders, and
a)/b)/c) panel labels — the presentation used for the 2023 Christmas storm figure in the
FrontFinder paper. Predictions are raw (uncalibrated) sup1 probabilities from ERA5 analysis;
the paper's figure used GFS f000 with neighborhood calibration, so amplitudes read lower here
while the spatial structures are comparable.

Inputs are derived on the fly from a remote ERA5 source (Arraylake or ARCO) via the side-store
derivation, and cached to a local NetCDF so repeat renders and TF-free environments skip the
fetch: with ``inputs_cache_path`` present, no network or data-source access is needed.

Usage:
    python -m fronts.model_1702.case_study --config_path configs/model_1702/case_study_xmas2023.yaml
"""

import argparse
import dataclasses
import logging
import math
import os
import string
import sys

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from fronts import utils
from fronts.data import inputs
from fronts.model_1702 import adapter, loader, store
from fronts.plot import plot
from fronts.plot import utils as plot_utils

logger = logging.getLogger(__name__)

CONTOUR_LEVELS = np.arange(0.1, 1.01, 0.1)
CMAP_TRUNCATION_MIN = 0.35
PANEL_COLUMNS = 2


@dataclasses.dataclass
class CaseStudyConfig:
    """Configuration for a model_1702 case-study figure.

    Attributes:
        model_path: Path to ``model_1702.h5``.
        times: Timesteps to render, one panel each, as ISO datetime strings.
        coordinates: Spatial bounding box as [lat_min, lat_max, lon_min, lon_max].
        front_types: Front type keys to contour (class indices/cmaps from ``fronts.plot.plot``).
        era5_uri: Remote ERA5 source for input derivation (``arraylake://...`` or a zarr URL).
        storage_options: Storage options for non-arraylake zarr sources (e.g. anonymous GCS).
        inputs_cache_path: NetCDF path caching the derived inputs. Loaded instead of the remote
            source when it exists; written after a remote fetch when it does not. None always
            fetches and never caches.
        outdir: Directory to write the figure into.
        figure_name: Output filename (e.g. ``xmas2023_fig14_style.png``).
        gpu_device: GPU index to use. None runs on CPU.
    """

    model_path: str
    times: list[str]
    coordinates: utils.BoundingBox
    front_types: list[str]
    era5_uri: str
    storage_options: dict | None
    inputs_cache_path: str | None
    outdir: str
    figure_name: str
    gpu_device: int | None


def panel_title(time: np.datetime64) -> str:
    """Formats a timestep as in the paper's panel captions, e.g. ``0000 UTC 26 Dec 2023``."""
    return pd.Timestamp(time).strftime("%H%M UTC %d %b %Y")


def load_case_inputs(case_cfg: CaseStudyConfig) -> xr.Dataset:
    """Loads (or derives and caches) the model_1702 input dataset for the case timesteps.

    Args:
        case_cfg: Case-study configuration.

    Returns:
        Derived legacy-variable dataset with dims (time, level, latitude, longitude) covering
        exactly ``case_cfg.times``.

    Raises:
        ValueError: If a cached dataset does not contain all requested timesteps.
    """
    times = np.array(case_cfg.times, dtype="datetime64[ns]")
    if case_cfg.inputs_cache_path and os.path.exists(case_cfg.inputs_cache_path):
        logger.info("Loading cached case inputs from %s", case_cfg.inputs_cache_path)
        cached = xr.open_dataset(case_cfg.inputs_cache_path)
        missing = [str(t) for t in times if t not in cached["time"].values]
        if missing:
            raise ValueError(f"Cached inputs at {case_cfg.inputs_cache_path} lack timesteps {missing}")
        return cached.sel(time=times)

    logger.info("Deriving case inputs from %s", case_cfg.era5_uri)
    source = store.open_source_era5(case_cfg.era5_uri, case_cfg.storage_options)
    source = utils.select_spatial_domain(source, case_cfg.coordinates)
    source = source.sel(time=times, level=store.PRESSURE_LEVELS_HPA).compute()
    built = store.build_1702_dataset(source)
    if case_cfg.inputs_cache_path:
        os.makedirs(os.path.dirname(case_cfg.inputs_cache_path) or ".", exist_ok=True)
        built.to_netcdf(case_cfg.inputs_cache_path)
        logger.info("Cached case inputs to %s", case_cfg.inputs_cache_path)
    return built


def predict_case(model: object, built: xr.Dataset) -> np.ndarray:
    """Runs a wrapped model over the case inputs.

    Args:
        model: Callable following the adapter contract, e.g. ``FrontFinder1702Adapter``.
        built: Derived legacy-variable dataset from :func:`load_case_inputs`.

    Returns:
        Prediction array shaped (time, lat, lon, 6).
    """
    x = inputs.inputs_ds_to_volume_dataarray(built, list(built.data_vars)).values
    return np.asarray(model(x, training=False))


def render_case_figure(
    preds: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    times: np.ndarray,
    front_types: list[str],
    out_path: str,
) -> None:
    """Renders the multi-panel filled-contour figure.

    Args:
        preds: Predictions shaped (time, lat, lon, n_classes), class indices per
            ``fronts.plot.plot.FRONT_TYPE_CLASS_INDEX``.
        lats: Latitude values.
        lons: Longitude values.
        times: Panel timesteps, one per prediction row.
        front_types: Front type keys to contour, drawn in the given order.
        out_path: Output image path.
    """
    n_panels = len(times)
    n_rows = math.ceil(n_panels / PANEL_COLUMNS)
    projection = ccrs.Miller(central_longitude=250)
    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]

    fig, axes = plt.subplots(
        n_rows,
        PANEL_COLUMNS,
        figsize=(6.5 * PANEL_COLUMNS, 4.1 * n_rows),
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for panel, ax in enumerate(axes):
        if panel >= n_panels:
            ax.set_visible(False)
            continue
        plot_utils.plot_background(extent=extent, ax=ax, linewidth=0.4)
        for front_type in front_types:
            class_idx = plot.FRONT_TYPE_CLASS_INDEX[front_type]
            probs = preds[panel, :, :, class_idx]
            masked = np.where(probs >= CONTOUR_LEVELS[0], probs, np.nan)
            ax.contourf(
                lons,
                lats,
                masked,
                levels=CONTOUR_LEVELS,
                cmap=plot_utils.truncated_colormap(plot.CONTOUR_CMAPS[front_type], minval=CMAP_TRUNCATION_MIN),
                transform=ccrs.PlateCarree(),
            )
        ax.text(
            0.015,
            0.955,
            f"{string.ascii_lowercase[panel]})",
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            bbox={"facecolor": "yellow", "edgecolor": "none", "pad": 2},
        )
        ax.set_title(panel_title(times[panel]), fontsize=10)

    legend = ", ".join(f"{plot.FRONT_COLORS[ft]} = {plot.FRONT_NAMES[ft].lower()}" for ft in front_types)
    fig.suptitle(
        "model_1702 predictions from ERA5 analysis — raw (uncalibrated) probabilities, "
        f"filled contours at 10% intervals\n{legend}",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("Figure saved to %s", out_path)


def run(case_cfg: CaseStudyConfig) -> None:
    """Loads data and the model, predicts, and renders the case-study figure.

    Args:
        case_cfg: Case-study configuration.
    """
    utils.configure_gpu(case_cfg.gpu_device)
    built = load_case_inputs(case_cfg)
    model = loader.load_model_1702(case_cfg.model_path)
    lats = built["latitude"].values
    wrapped = adapter.FrontFinder1702Adapter(model, lat_ascending=bool(lats[0] < lats[-1]))
    preds = predict_case(wrapped, built)
    os.makedirs(case_cfg.outdir, exist_ok=True)
    render_case_figure(
        preds=preds,
        lats=lats,
        lons=built["longitude"].values,
        times=built["time"].values,
        front_types=case_cfg.front_types,
        out_path=os.path.join(case_cfg.outdir, case_cfg.figure_name),
    )


def main() -> None:
    """Parses arguments, loads the config, and renders the case study."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Render a model_1702 case-study prediction figure.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the case-study YAML.")
    args = parser.parse_args()

    yaml_data = utils.load_yaml(args.config_path)
    case_cfg: CaseStudyConfig = utils.parse_config_section(
        yaml_data, CaseStudyConfig, "case_config", utils.YAML_TYPE_HOOKS
    )
    run(case_cfg)


if __name__ == "__main__":
    main()
