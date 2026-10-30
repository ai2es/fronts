"""Recompute derived variables from a store's own raw inputs and overwrite them in place.

One-off remediation for derived variables written with a bug (e.g. NaNs in
dewpoint_temperature and equivalent_potential_temperature from non-positive
ERA5 specific humidity). Reads the required raw inputs from the icechunk store
itself — no re-download — recomputes the requested derived variables with the
current fronts.data.derived code, and overwrites the existing arrays in a
single commit. Icechunk versioning keeps the previous data reachable in the
prior snapshot.

Usage (on the machine holding the store):
    pixi run python scripts/rewrite_derived_variables.py \
        --config configs/generate_icechunk.yaml \
        --variables dewpoint_temperature equivalent_potential_temperature

Afterwards, verify with:
    pixi run python scripts/scan_nan_channels.py --store-path <store> \
        --variables dewpoint_temperature equivalent_potential_temperature
"""

import argparse
import logging

from fronts import utils
from fronts.data import derived, generate

logger = logging.getLogger(__name__)


def main():
    """Load the generation config and rewrite the requested derived variables."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Recompute and overwrite derived variables in an icechunk store")
    parser.add_argument("--config", required=True, help="Path to the generation YAML config")
    parser.add_argument(
        "--variables",
        nargs="+",
        required=True,
        help=f"Derived variables to rewrite. Choices: {sorted(derived.DERIVED_VARIABLE_REGISTRY)}",
    )
    args = parser.parse_args()

    unknown = [v for v in args.variables if v not in derived.DERIVED_VARIABLE_REGISTRY]
    if unknown:
        parser.error(f"Unknown derived variables: {unknown}. Choices: {sorted(derived.DERIVED_VARIABLE_REGISTRY)}")

    era5_config = utils.open_config_yaml_as_dataclass(
        args.config, generate.ERA5DataLoaderConfig, config_key="era5_config"
    )
    icechunk_config = utils.open_config_yaml_as_dataclass(
        args.config, utils.IcechunkStorageConfig, config_key="icechunk_storage_config"
    )

    logger.info("Rewriting %s in %s", args.variables, icechunk_config.store_path)
    generate.write_derived_variables(era5_config, icechunk_config, args.variables)
    logger.info("Done. Verify with scripts/scan_nan_channels.py before training.")


if __name__ == "__main__":
    main()
