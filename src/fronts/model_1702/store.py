"""Generation of the model_1702 input side-store.

Builds an icechunk store holding model_1702's ten legacy input variables
(``T, Td, Tv, u, v, r, q, RH, sp_z, theta_e``) on its five-level stack (surface + 1000/950/900/
850 hPa) in legacy units, derived from raw ERA5 fields with the verbatim legacy formulas. The
store mimics the 2.0 inputs-store layout — variables with dims (time, level, latitude,
longitude) — so ``fronts.data.datasets.FrontsPyDataset`` consumes it unchanged; the surface
level is stored under the numeric sentinel 1013 (matching the legacy ``*_1013`` aliases).

Run: ``python -m fronts.model_1702.store --config configs/model_1702/generate_conus.yaml``
"""

import argparse
import dataclasses
import logging
import sys

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from fronts import utils
from fronts.data import generate, sources
from fronts.model_1702 import legacy_formulas, normalization

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

PRESSURE_LEVELS_HPA = [1000, 950, 900, 850]
SOURCE_PRESSURE_VARIABLES = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
SOURCE_SINGLE_VARIABLES = [
    "surface_pressure",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
GOOGLE_TO_ARRAYLAKE_SINGLE_1702 = {**sources.GOOGLE_TO_ARRAYLAKE_SINGLE, "surface_pressure": "sp"}
GEOPOTENTIAL_TO_DAM = 98.0665
PA_TO_HPA = 100.0
KG_PER_KG_TO_G_PER_KG = 1000.0


def open_source_era5(era5_uri: str, storage_options: dict | None) -> xr.Dataset:
    """Opens the raw ERA5 source with the variables model_1702's inputs are derived from.

    Args:
        era5_uri: ``arraylake://org/repo`` URI or a zarr store URL (e.g. the Google ARCO ERA5
            store) whose variables already follow Google ARCO naming.
        storage_options: Storage options for non-arraylake zarr sources.

    Returns:
        Lazy dataset with Google-named variables: pressure-level
        ``SOURCE_PRESSURE_VARIABLES`` plus single-level ``SOURCE_SINGLE_VARIABLES``.
    """
    if era5_uri.startswith(sources.ARRAYLAKE_URI_PREFIX):
        return _open_arraylake_with_surface_pressure(era5_uri)
    open_kwargs: dict = {"chunks": None}
    if storage_options:
        open_kwargs["storage_options"] = storage_options
    ds = xr.open_zarr(era5_uri, **open_kwargs)
    return ds[SOURCE_PRESSURE_VARIABLES + SOURCE_SINGLE_VARIABLES]


def _open_arraylake_with_surface_pressure(era5_uri: str) -> xr.Dataset:
    """Arraylake opener variant extending the core single-level map with surface pressure."""
    import arraylake

    repo_name = sources.parse_arraylake_repo(era5_uri)
    client = arraylake.Client()
    repo = client.get_repo(repo_name)
    session = repo.readonly_session("main")

    ds_pressure = xr.open_zarr(session.store, group=sources.ARRAYLAKE_PRESSURE_GROUP, chunks=None)
    pressure_short = [sources.GOOGLE_TO_ARRAYLAKE_PRESSURE[v] for v in SOURCE_PRESSURE_VARIABLES]
    ds_pressure = sources._rename_to_google(ds_pressure[pressure_short], sources.GOOGLE_TO_ARRAYLAKE_PRESSURE)

    ds_single = xr.open_zarr(session.store, group=sources.ARRAYLAKE_SINGLE_GROUP, chunks=None)
    single_short = [GOOGLE_TO_ARRAYLAKE_SINGLE_1702[v] for v in SOURCE_SINGLE_VARIABLES]
    ds_single = sources._rename_to_google(ds_single[single_short], GOOGLE_TO_ARRAYLAKE_SINGLE_1702)

    return xr.merge([ds_pressure, ds_single], join="inner", compat="override")


def _pressure_level_fields(source_ds: xr.Dataset, level_hpa: int) -> dict[str, np.ndarray]:
    level_ds = source_ds.sel(level=level_hpa)
    pressure_pa = level_hpa * 100.0
    temperature = level_ds["temperature"].values
    specific_humidity = level_ds["specific_humidity"].values
    dewpoint = legacy_formulas.dewpoint_from_specific_humidity(pressure_pa, temperature, specific_humidity)
    return {
        "T": temperature,
        "Td": dewpoint,
        "Tv": legacy_formulas.virtual_temperature_from_dewpoint(temperature, dewpoint, pressure_pa),
        "u": level_ds["u_component_of_wind"].values,
        "v": level_ds["v_component_of_wind"].values,
        "r": legacy_formulas.mixing_ratio_from_dewpoint(dewpoint, pressure_pa) * KG_PER_KG_TO_G_PER_KG,
        "q": specific_humidity * KG_PER_KG_TO_G_PER_KG,
        "RH": legacy_formulas.relative_humidity(temperature, dewpoint),
        "sp_z": level_ds["geopotential"].values / GEOPOTENTIAL_TO_DAM,
        "theta_e": legacy_formulas.equivalent_potential_temperature(temperature, dewpoint, pressure_pa),
    }


def _surface_fields(source_ds: xr.Dataset) -> dict[str, np.ndarray]:
    temperature = source_ds["2m_temperature"].values
    dewpoint = source_ds["2m_dewpoint_temperature"].values
    surface_pressure = source_ds["surface_pressure"].values
    return {
        "T": temperature,
        "Td": dewpoint,
        "Tv": legacy_formulas.virtual_temperature_from_dewpoint(temperature, dewpoint, surface_pressure),
        "u": source_ds["10m_u_component_of_wind"].values,
        "v": source_ds["10m_v_component_of_wind"].values,
        "r": legacy_formulas.mixing_ratio_from_dewpoint(dewpoint, surface_pressure) * KG_PER_KG_TO_G_PER_KG,
        "q": legacy_formulas.specific_humidity_from_dewpoint(dewpoint, surface_pressure) * KG_PER_KG_TO_G_PER_KG,
        "RH": legacy_formulas.relative_humidity(temperature, dewpoint),
        "sp_z": surface_pressure / PA_TO_HPA,
        "theta_e": legacy_formulas.equivalent_potential_temperature(temperature, dewpoint, surface_pressure),
    }


def build_1702_dataset(source_ds: xr.Dataset) -> xr.Dataset:
    """Derives the ten legacy input variables on the five-level stack from raw ERA5 fields.

    Args:
        source_ds: Eager dataset with ``SOURCE_PRESSURE_VARIABLES`` on dims
            (time, level, latitude, longitude) covering at least ``PRESSURE_LEVELS_HPA``, and
            ``SOURCE_SINGLE_VARIABLES`` on (time, latitude, longitude), all in native ERA5
            units (K, m/s, kg/kg, Pa, m^2/s^2).

    Returns:
        Dataset with the ten legacy variables as float32 on dims
        (time, level, latitude, longitude), level coordinate ``normalization.LEVEL_COORD``
        (surface sentinel first), latitude descending, values in legacy units.
    """
    source_ds = source_ds.sortby("latitude", ascending=False)
    per_level = [_surface_fields(source_ds)] + [
        _pressure_level_fields(source_ds, level) for level in PRESSURE_LEVELS_HPA
    ]

    coords = {
        "time": source_ds["time"].values,
        "level": np.array(normalization.LEVEL_COORD, dtype=np.int64),
        "latitude": source_ds["latitude"].values,
        "longitude": source_ds["longitude"].values,
    }
    data_vars = {}
    for variable in normalization.VARIABLES:
        stacked = np.stack([fields[variable] for fields in per_level], axis=1)
        data_vars[variable] = xr.DataArray(stacked.astype(np.float32), dims=["time", "level", "latitude", "longitude"])
    return xr.Dataset(data_vars, coords=coords)


def _missing_times(icechunk_config: utils.IcechunkStorageConfig, requested_times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    contents = generate.inspect_store(icechunk_config)
    if contents is None:
        return requested_times
    return requested_times[~requested_times.isin(contents.times)]


def run(era5_config: generate.ERA5DataLoaderConfig, icechunk_config: utils.IcechunkStorageConfig) -> None:
    """Generates and writes all missing side-store time steps, batch by batch.

    Resume-safe: already-stored time steps are skipped, and each batch is committed before the
    next is read so an interrupted run only re-derives the batch it died in.

    Args:
        era5_config: Source configuration; ``variables`` and ``pressure_levels`` are fixed by
            this module and ignored in favor of the model_1702 requirements.
        icechunk_config: Target store configuration; ``write_batch_size`` bounds how many time
            steps are derived and committed at once.
    """
    requested_times = pd.date_range(era5_config.time_start, era5_config.time_end, freq=era5_config.time_resolution)
    missing = _missing_times(icechunk_config, requested_times)
    if missing.empty:
        logger.info("All requested time steps already present in the side store.")
        return
    if not missing.equals(requested_times) and missing[0] <= pd.Timestamp(requested_times[0]):
        logger.info("Resuming side-store generation with %d of %d time steps.", len(missing), len(requested_times))

    source = open_source_era5(era5_config.era5_uri, era5_config.storage_options)
    source = utils.select_spatial_domain(source, era5_config.coordinates)
    source = source.sel(level=PRESSURE_LEVELS_HPA)

    batch_config = dataclasses.replace(icechunk_config, write_batch_size=None)
    batch_size = icechunk_config.write_batch_size or len(missing)
    for start in range(0, len(missing), batch_size):
        batch_times = missing[start : start + batch_size]
        logger.info("Deriving time steps %d-%d of %d missing.", start, start + len(batch_times) - 1, len(missing))
        source_batch = source.sel(time=batch_times).compute()
        built = build_1702_dataset(source_batch)
        generate.write_or_append_icechunk_store(batch_config, built)


def main() -> None:
    """Entry point for side-store generation from a YAML config."""
    parser = argparse.ArgumentParser(description="Generate the model_1702 input side-store")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML generation config")
    args = parser.parse_args()

    era5_config = utils.open_config_yaml_as_dataclass(
        args.config,
        generate.ERA5DataLoaderConfig,
        config_key="era5_config",
        type_hooks=utils.YAML_TYPE_HOOKS,
    )
    icechunk_config = utils.open_config_yaml_as_dataclass(
        args.config, utils.IcechunkStorageConfig, config_key="icechunk_storage_config"
    )
    zarr.config.set({"async.concurrency": era5_config.zarr_async_concurrency})
    logger.info("Side-store generation config loaded: %s", era5_config)
    run(era5_config, icechunk_config)
    logger.info("Side-store generation complete.")


if __name__ == "__main__":
    main()
