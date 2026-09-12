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
}

ARRAYLAKE_TO_GOOGLE_COORDS = {
    "valid_time": "time",
    "pressure_level": "level",
}

ARCO_ERA5_PUBLIC_URI = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Time-invariant fields not carried by the Arraylake mirror, sourced instead from the
# public Google ARCO ERA5 archive (no credentials required). Every timestep in that
# archive holds identical values for these fields; open_static_era5_variable collapses
# the time dimension away.
STATIC_VARIABLE_SOURCES: dict[str, str] = {
    "geopotential_at_surface": ARCO_ERA5_PUBLIC_URI,
    "land_sea_mask": ARCO_ERA5_PUBLIC_URI,
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


def _open_arraylake_group(
    store: object,
    group: str,
    group_variables: list[str],
    mapping: dict[str, str],
    mapping_name: str,
    repo_name: str,
) -> xr.Dataset | None:
    """Open one Arraylake group and return its requested variables renamed to Google naming.

    Args:
        store: Arraylake session store to read from.
        group: Zarr group path within the store (e.g. ``pressure/spatial``).
        group_variables: Google-named variables to select from this group; an empty
            list means the group is not needed for this request.
        mapping: Google-name-to-short-name mapping for this group.
        mapping_name: Name of ``mapping``, for error messages.
        repo_name: Arraylake repository name, for error messages.

    Returns:
        The renamed Dataset, or None if ``group_variables`` is empty.

    Raises:
        ValueError: If a mapped short name is absent from the opened group — a stale
            mapping, since indexing a Dataset by a list of missing names silently
            returns an empty result instead of raising, which otherwise surfaces
            later as a confusing "no changes to commit" error.
    """
    if not group_variables:
        return None
    ds = xr.open_zarr(store, group=group, chunks=None)
    short_names = [mapping[v] for v in group_variables]
    missing = [short for short in short_names if short not in ds.data_vars]
    if missing:
        raise ValueError(
            f"Short names {missing} mapped in {mapping_name} are absent from group '{group}' in {repo_name}."
        )
    return _rename_to_google(ds[short_names], mapping)


# The public ARCO ERA5 archive's time axis spans 1900-2050, but real ERA5 coverage only
# starts 1940-01-01; every timestep before that (including index 0) is a NaN placeholder,
# so a fixed reference time well inside real coverage is used instead of isel(time=0).
_STATIC_REFERENCE_TIME = "2000-01-01"


def open_static_era5_variables(variables: list[str]) -> xr.Dataset:
    """Open time-invariant fields from their registered sources in STATIC_VARIABLE_SOURCES.

    Variables sharing a source URI (e.g. ``geopotential_at_surface`` and
    ``land_sea_mask``, both from the public ARCO archive) are fetched from a single
    open of that source rather than one open per variable.

    Args:
        variables: Google ARCO variable names; each must be a key of STATIC_VARIABLE_SOURCES.

    Returns:
        Dataset of lazy (latitude, longitude) DataArrays with the time dimension
        dropped — every timestep within real ERA5 coverage holds identical values
        for these fields.

    Raises:
        ValueError: If any variable has no registered static source.
    """
    unknown = [v for v in variables if v not in STATIC_VARIABLE_SOURCES]
    if unknown:
        raise ValueError(
            f"No static source registered for variables {unknown}. "
            f"Registered static variables: {sorted(STATIC_VARIABLE_SOURCES)}"
        )

    by_uri: dict[str, list[str]] = {}
    for var in variables:
        by_uri.setdefault(STATIC_VARIABLE_SOURCES[var], []).append(var)

    pieces = []
    for uri, uri_variables in by_uri.items():
        open_kwargs: dict = {"chunks": None}
        if uri.startswith("gs://"):
            open_kwargs["storage_options"] = {"token": "anon"}
        ds = xr.open_zarr(uri, **open_kwargs)
        selected = ds[uri_variables].sel(time=_STATIC_REFERENCE_TIME, method="nearest")
        pieces.append(selected.reset_coords("time", drop=True))
    return pieces[0] if len(pieces) == 1 else xr.merge(pieces, join="inner", compat="override")


def open_static_era5_variable(variable: str) -> xr.DataArray:
    """Open one time-invariant field from its registered source in STATIC_VARIABLE_SOURCES.

    Args:
        variable: Google ARCO variable name; must be a key of STATIC_VARIABLE_SOURCES.

    Returns:
        A lazy (latitude, longitude) DataArray with the time dimension dropped.

    Raises:
        ValueError: If ``variable`` has no registered static source.
    """
    return open_static_era5_variables([variable])[variable]


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
        ValueError: If a requested variable has no known Arraylake mapping, or its
            mapped short name is absent from the opened group — see
            ``_open_arraylake_group``.
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

    datasets = [
        ds
        for ds in (
            _open_arraylake_group(
                session.store,
                ARRAYLAKE_PRESSURE_GROUP,
                pressure_variables,
                GOOGLE_TO_ARRAYLAKE_PRESSURE,
                "GOOGLE_TO_ARRAYLAKE_PRESSURE",
                repo_name,
            ),
            _open_arraylake_group(
                session.store,
                ARRAYLAKE_SINGLE_GROUP,
                single_level_variables,
                GOOGLE_TO_ARRAYLAKE_SINGLE,
                "GOOGLE_TO_ARRAYLAKE_SINGLE",
                repo_name,
            ),
        )
        if ds is not None
    ]

    logger.info(
        f"Opened Arraylake repo '{repo_name}' with pressure variables {pressure_variables} "
        f"and single-level variables {single_level_variables}"
    )
    return xr.merge(datasets, join="inner", compat="override")
