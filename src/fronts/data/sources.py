"""Openers for remote ERA5 sources, normalizing names to Google ARCO conventions."""

import logging

import arraylake
import xarray as xr

logger = logging.getLogger(__name__)

ARRAYLAKE_URI_PREFIX = "arraylake://"
ARRAYLAKE_PRESSURE_GROUP = "pressure/spatial"
ARRAYLAKE_SINGLE_GROUP = "single/spatial"

# Note that these mappings are not fixed; we can expand them over time.
# Prefer to use the long names instead of short names in the data
GOOGLE_TO_ARRAYLAKE_PRESSURE = {
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "vertical_velocity": "w",
    "potential_vorticity": "pv",
}

GOOGLE_TO_ARRAYLAKE_SINGLE = {
    "mean_sea_level_pressure": "msl",
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_dewpoint_temperature": "d2m",
    "land_sea_mask": "lsm",
}

ARRAYLAKE_TO_GOOGLE_COORDS = {
    "valid_time": "time",
    "pressure_level": "level",
}


def parse_arraylake_repo(era5_uri: str) -> str:
    """Return the ``org/repo`` part of an ``arraylake://org/repo`` URI.

    Args:
        era5_uri: URI of the form ``arraylake://org/repo``.

    Returns:
        The Arraylake repository name, e.g. ``earthmover-public/era5``.

    Raises:
        ValueError: If the URI does not start with the arraylake prefix.
    """
    if not era5_uri.startswith(ARRAYLAKE_URI_PREFIX):
        raise ValueError(f"Not an arraylake URI: {era5_uri}")
    return era5_uri.removeprefix(ARRAYLAKE_URI_PREFIX)


def _rename_to_google(ds: xr.Dataset, variable_mapping: dict[str, str]) -> xr.Dataset:
    """Rename Arraylake short variable and coordinate names to Google ARCO names."""
    short_to_google = {short: google for google, short in variable_mapping.items() if short in ds}
    coord_renames = {src: dst for src, dst in ARRAYLAKE_TO_GOOGLE_COORDS.items() if src in ds.coords or src in ds.dims}
    return ds.rename({**short_to_google, **coord_renames})


def available_arraylake_variables() -> set[str]:
    """Return all variable names (Google naming) available from the Arraylake ERA5 store."""
    return set(GOOGLE_TO_ARRAYLAKE_PRESSURE) | set(GOOGLE_TO_ARRAYLAKE_SINGLE)


def open_arraylake_era5(era5_uri: str, variables: list[str]) -> xr.Dataset:
    """Open the Arraylake ERA5 store and return requested variables with Google naming.

    Variables are split internally by the mappings: pressure-level variables come
    from the ``pressure/spatial`` group and single-level variables from the
    ``single/spatial`` group. Variables and coordinates are renamed from
    Arraylake short names (e.g. ``t``, ``valid_time``) to Google ARCO
    conventions (e.g. ``temperature``, ``time``) before the groups are merged
    into a single dataset. Single-level variables simply lack the ``level``
    dimension in the merged dataset.

    Args:
        era5_uri: URI of the form ``arraylake://org/repo``.
        variables: Variable names (Google naming) to load. Every name must be a
            key of ``GOOGLE_TO_ARRAYLAKE_PRESSURE`` or ``GOOGLE_TO_ARRAYLAKE_SINGLE``.

    Returns:
        A lazy (non-dask, ``chunks=None``) merged dataset with Google naming.

    Raises:
        ValueError: If a requested variable has no known Arraylake mapping.
    """
    single_level_variables = [v for v in variables if v in GOOGLE_TO_ARRAYLAKE_SINGLE]
    pressure_variables = [v for v in variables if v not in GOOGLE_TO_ARRAYLAKE_SINGLE]
    unknown = [v for v in pressure_variables if v not in GOOGLE_TO_ARRAYLAKE_PRESSURE]
    if unknown:
        raise ValueError(f"Variables with no known Arraylake mapping: {unknown}")

    repo_name = parse_arraylake_repo(era5_uri)
    client = arraylake.Client()
    repo = client.get_repo(repo_name)
    session = repo.readonly_session("main")

    datasets = []
    if pressure_variables:
        ds_pressure = xr.open_zarr(session.store, group=ARRAYLAKE_PRESSURE_GROUP, chunks=None)
        short_names = [GOOGLE_TO_ARRAYLAKE_PRESSURE[v] for v in pressure_variables]
        datasets.append(_rename_to_google(ds_pressure[short_names], GOOGLE_TO_ARRAYLAKE_PRESSURE))
    if single_level_variables:
        ds_single = xr.open_zarr(session.store, group=ARRAYLAKE_SINGLE_GROUP, chunks=None)
        short_names = [GOOGLE_TO_ARRAYLAKE_SINGLE[v] for v in single_level_variables]
        datasets.append(_rename_to_google(ds_single[short_names], GOOGLE_TO_ARRAYLAKE_SINGLE))

    logger.info(
        f"Opened Arraylake repo '{repo_name}' with pressure variables {pressure_variables} "
        f"and single-level variables {single_level_variables}"
    )
    return xr.merge(datasets, join="inner", compat="override")
