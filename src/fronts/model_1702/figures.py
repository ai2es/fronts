"""Batch 4-panel figure generation for harness evaluation outputs.

Globs every ``stats_derived*.nc`` in a stats directory (as written by
``fronts.model_1702.run_eval``) and renders the standard 4-panel performance figure for each
front type in each file via ``fronts.plot.plot.plot_performance_diagrams`` — including office
regions the core plot CLI does not expose.

Usage:
    python -m fronts.model_1702.figures \
        --stats_dir /ourdisk/hpc/ai2es/tman/models/stats/model_1702/model_1702_conus/ \
        --coordinates 25.0 56.75 228.0 299.75
"""

import argparse
import glob
import logging
import os

import xarray as xr

from fronts import utils
from fronts.plot import plot

log = logging.getLogger(__name__)

DERIVED_STATS_PREFIX = "stats_derived"


def region_from_filename(path: str) -> str | None:
    """Extracts the region suffix from a stats_derived filename.

    Args:
        path: Path like ``.../stats_derived_WPC.nc`` or ``.../stats_derived.nc``.

    Returns:
        The region name, or None for the unsuffixed full-domain file.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    suffix = stem.removeprefix(DERIVED_STATS_PREFIX)
    return suffix.removeprefix("_") or None


def render_stats_dir(
    stats_dir: str,
    coordinates: utils.BoundingBox,
    map_neighborhood: int,
    output_type: str,
    outdir: str | None,
) -> list[str]:
    """Renders 4-panel figures for every derived-stats file and front type in a directory.

    Args:
        stats_dir: Directory containing ``stats_derived*.nc`` files.
        coordinates: Bounding box the stats were computed over (drives the map panel layout).
        map_neighborhood: Neighbourhood radius (km) for the spatial CSI map panel.
        output_type: Output image format (e.g. "png").
        outdir: Directory to write figures into. None writes next to the stats files.

    Returns:
        Paths of the derived-stats files that were rendered.

    Raises:
        FileNotFoundError: If the directory contains no derived-stats files.
    """
    stats_paths = sorted(glob.glob(os.path.join(stats_dir, f"{DERIVED_STATS_PREFIX}*.nc")))
    if not stats_paths:
        raise FileNotFoundError(f"No {DERIVED_STATS_PREFIX}*.nc files found in {stats_dir}")
    figure_dir = outdir or stats_dir
    os.makedirs(figure_dir, exist_ok=True)

    for stats_path in stats_paths:
        region = region_from_filename(stats_path)
        derived_ds = xr.open_dataset(stats_path)
        front_types = plot._parse_front_types(derived_ds)
        log.info("Rendering %s (region=%s) for front types %s", stats_path, region or "full", front_types)
        for front_type in front_types:
            plot.plot_performance_diagrams(
                front_type=front_type,
                derived_ds=derived_ds,
                mask=region,
                coordinates=coordinates,
                map_neighborhood=map_neighborhood,
                output_type=output_type,
                outdir=figure_dir,
            )
        derived_ds.close()
    return stats_paths


def main() -> None:
    """Parses arguments and renders all figures for a stats directory."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Render 4-panel figures for harness stats outputs.")
    parser.add_argument("--stats_dir", type=str, required=True, help="Directory with stats_derived*.nc files.")
    parser.add_argument(
        "--coordinates",
        type=float,
        nargs=4,
        required=True,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Bounding box the stats were computed over.",
    )
    parser.add_argument("--map_neighborhood", type=int, default=250, help="Radius (km) for the spatial CSI panel.")
    parser.add_argument("--output_type", type=str, default="png", help="Output image format.")
    parser.add_argument("--outdir", type=str, default=None, help="Figure output directory (default: stats_dir).")
    args = parser.parse_args()

    coordinates = utils.BoundingBox(*args.coordinates)
    render_stats_dir(args.stats_dir, coordinates, args.map_neighborhood, args.output_type, args.outdir)


if __name__ == "__main__":
    main()
