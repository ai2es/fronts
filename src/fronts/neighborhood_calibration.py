r"""Neighborhood calibration: turn raw front probabilities into the chance of a front within a radius.

A raw probability answers this question: "is this pixel inside the analyst's dilated band?" Hand-drawn
labels limit raw values to about 0.8. A calibrated value answers a wider question: "is a front of this
type within N km?" The AIES FrontFinder paper drew its maps on this scale (100 km).

This module is a port of the legacy ``evaluation/calibrate_model.py`` and
``evaluation/prediction_plot.py --calibration``. It fits an isotonic regression (a fit that never
decreases) to each neighborhood reliability curve from ``fronts.evaluate``. Then it maps model output
through the fit.

The port has two changes from the legacy script. Each bin is weighted by its pixel count, so a bin with
few pixels does not outweigh a bin with many. Pixels forecast below the lowest eval threshold (0.01)
get their own bin, from ``nbhd_truth_{ft}``. The legacy script gave them the value of the 0.01-0.02 bin,
which tinted whole maps.

Fit on stats from the validation year. Do not use the test year, and do not use a year from training.

Usage:
    python -m fronts.neighborhood_calibration \
        --stats_path /path/to/stats/stats_aggregate.nc --output_path /path/to/neighborhood_calibration.nc
"""

import argparse
import logging

import numpy as np
import xarray as xr

from fronts import constants

log = logging.getLogger(__name__)


def isotonic_fit(y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the non-decreasing sequence closest to ``y`` in weighted least squares (pool adjacent violators).

    Args:
        y: Values in order of increasing x.
        weights: Positive weight for each value.

    Returns:
        Fitted values, same length as ``y``.
    """
    means: list[float] = []
    totals: list[float] = []
    sizes: list[int] = []
    for value, weight in zip(y, weights, strict=True):
        means.append(float(value))
        totals.append(float(weight))
        sizes.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            total = totals[-2] + totals[-1]
            means[-2] = (means[-2] * totals[-2] + means[-1] * totals[-1]) / total
            totals[-2] = total
            sizes[-2] += sizes[-1]
            del means[-1], totals[-1], sizes[-1]
    return np.repeat(means, sizes)


def neighborhood_reliability(aggregate_ds: xr.Dataset, front_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Return per-bin pixel counts and neighborhood hits for one front type.

    Bin 0 holds pixels forecast below the lowest threshold, bin i holds [t_(i-1), t_i), and the last
    bin holds pixels at or above the highest threshold. A hit is a pixel with a front of this type
    within the neighborhood radius.

    Args:
        aggregate_ds: Aggregate stats from ``fronts.evaluate`` with ``tp/fp/tn/fn_{ft}`` (neighborhood,
            threshold) and ``nbhd_truth_{ft}`` (neighborhood,).
        front_type: Front type key, e.g. "CF".

    Returns:
        Tuple of (hits, counts), weighted, each with shape (neighborhood, bin).

    Raises:
        KeyError: If the stats were written before ``nbhd_truth_{ft}`` existed.
    """
    truth_key = f"nbhd_truth_{front_type}"
    if truth_key not in aggregate_ds:
        raise KeyError(f"{truth_key} is missing from these stats; rerun fronts.evaluate to write it.")

    def stat(name: str) -> np.ndarray:
        return aggregate_ds[f"{name}_{front_type}"].transpose("neighborhood", "threshold").values.astype(np.float64)

    tp, fp = stat("tp"), stat("fp")
    total = (tp + fp + stat("tn") + stat("fn"))[:, :1]
    near = aggregate_ds[truth_key].values.astype(np.float64)[:, None]
    zeros = np.zeros_like(total)
    cumulative_hits = np.concatenate([near, tp, zeros], axis=1)
    cumulative_counts = np.concatenate([total, tp + fp, zeros], axis=1)
    counts = np.clip(cumulative_counts[:, :-1] - cumulative_counts[:, 1:], 0.0, None)
    hits = np.clip(cumulative_hits[:, :-1] - cumulative_hits[:, 1:], 0.0, counts)
    return hits, counts


def fit_neighborhood_calibration(aggregate_ds: xr.Dataset, front_types: list[str]) -> xr.Dataset:
    """Fit an isotonic map from raw probability to neighborhood hit rate, per front type and radius.

    Args:
        aggregate_ds: Aggregate stats from ``fronts.evaluate`` (see ``neighborhood_reliability``).
        front_types: Front type keys to fit, e.g. ["CF", "WF"].

    Returns:
        Dataset with one ``calibrated_{ft}`` variable per front type, dims (neighborhood, probability),
        holding the calibrated value at each bin center. Empty bins take values interpolated from
        their neighbors.
    """
    thresholds = aggregate_ds["threshold"].values.astype(np.float64)
    centers = np.concatenate([thresholds[:1] / 2, (thresholds[:-1] + thresholds[1:]) / 2, thresholds[-1:]])
    data_vars = {}
    for ft in front_types:
        hits, counts = neighborhood_reliability(aggregate_ds, ft)
        table = np.empty_like(counts)
        for n in range(counts.shape[0]):
            seen = counts[n] > 0
            fitted = isotonic_fit(hits[n, seen] / counts[n, seen], counts[n, seen])
            table[n] = np.interp(centers, centers[seen], fitted)
        data_vars[f"calibrated_{ft}"] = (("neighborhood", "probability"), table.astype(np.float32))
    return xr.Dataset(
        data_vars,
        coords={"neighborhood": aggregate_ds["neighborhood"].values, "probability": centers},
        attrs={"description": "Chance of a front of each type within each radius (km), by raw probability."},
    )


def calibrate(probs_ds: xr.Dataset, calibration_ds: xr.Dataset, neighborhood_km: int) -> xr.Dataset:
    """Map raw per-front-type probabilities to neighborhood probabilities.

    Values between bin centers are interpolated linearly; values outside the centers take the end
    values.

    Args:
        probs_ds: One variable per front type (e.g. "CF"), any dims.
        calibration_ds: Table from ``fit_neighborhood_calibration``.
        neighborhood_km: Radius to calibrate to; must be one of the table's neighborhoods.

    Returns:
        Dataset shaped like ``probs_ds`` with calibrated values.
    """
    centers = calibration_ds["probability"].values
    return xr.Dataset(
        {
            ft: (
                da.dims,
                np.interp(
                    da.values, centers, calibration_ds[f"calibrated_{ft}"].sel(neighborhood=neighborhood_km).values
                ).astype(np.float32),
            )
            for ft, da in probs_ds.data_vars.items()
        },
        coords=probs_ds.coords,
    )


def main(argv: list[str] | None = None) -> None:
    """Fit neighborhood calibration from an aggregate stats file and save the table.

    Args:
        argv: Command-line arguments. None reads ``sys.argv``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fit neighborhood calibration from fronts.evaluate stats.")
    parser.add_argument("--stats_path", required=True, help="stats_aggregate*.nc written by fronts.evaluate.")
    parser.add_argument("--output_path", required=True, help="Where to write the calibration table (NetCDF).")
    parser.add_argument("--front_types", nargs="+", default=None, help="Front types to fit. Default: all in the stats.")
    args = parser.parse_args(argv)

    with xr.open_dataset(args.stats_path) as aggregate_ds:
        front_types = args.front_types or [ft for ft in constants.FRONT_TYPE_CLASS_INDEX if f"tp_{ft}" in aggregate_ds]
        calibration_ds = fit_neighborhood_calibration(aggregate_ds.load(), front_types)
    calibration_ds.attrs["source_stats"] = args.stats_path
    calibration_ds.to_netcdf(args.output_path)
    log.info("Neighborhood calibration for %s -> %s", ", ".join(front_types), args.output_path)


if __name__ == "__main__":
    main()
