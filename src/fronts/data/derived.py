import dataclasses
from collections.abc import Callable

import xarray as xr
import xarray.ufuncs as xu

_R_D = 287.05  # dry air gas constant, J kg-1 K-1
_C_PD = 1004.0  # specific heat of dry air at constant pressure, J kg-1 K-1
_L_V = 2.501e6  # latent heat of vaporization at 0 °C, J kg-1
_EPSILON = 0.622  # ratio of molar masses of water vapour to dry air
_MIN_SPECIFIC_HUMIDITY = 1e-9  # kg kg-1; ERA5 spectral artifacts yield q <= 0, which makes log(e) NaN


@dataclasses.dataclass
class DerivedVariableSpec:
    """Specification for computing a variable not present in the ARCO ERA5 store.

    Attributes:
        required_inputs: ERA5 variable names that must be downloaded before compute,
            in the order they are passed to ``compute``.
        compute: Callable that accepts one DataArray per entry in ``required_inputs``
            and returns a DataArray with dims (time, level, latitude, longitude).
    """

    required_inputs: list[str]
    compute: Callable[..., xr.DataArray]


def _pressure_pa(da: xr.DataArray) -> xr.DataArray:
    """Broadcast pressure levels (hPa → Pa) with the same dask graph as da."""
    return xr.zeros_like(da) + (da.level * 100.0)


def _saturation_vapour_pressure(temperature: xr.DataArray) -> xr.DataArray:
    """Saturation vapour pressure in hPa via Bolton (1980) eq. 10."""
    return 6.112 * xu.exp(17.67 * (temperature - 273.15) / (temperature - 29.65))


def _clamped_specific_humidity(specific_humidity: xr.DataArray) -> xr.DataArray:
    """Clamp specific humidity to a tiny positive floor.

    ERA5 specific humidity contains zero and slightly negative values in cold,
    dry upper-level air (spectral truncation artifacts); without a floor the
    vapour pressure is non-positive and log-based formulas (dewpoint,
    equivalent potential temperature) produce NaN.
    """
    return specific_humidity.clip(min=_MIN_SPECIFIC_HUMIDITY)


def _vapour_pressure(specific_humidity: xr.DataArray, pressure_hpa: xr.DataArray) -> xr.DataArray:
    """Actual vapour pressure in hPa from specific humidity and pressure (hPa)."""
    q = _clamped_specific_humidity(specific_humidity)
    r = q / (1.0 - q)
    return r / (_EPSILON + r) * pressure_hpa


def _compute_wind_speed(
    u_component_of_wind: xr.DataArray,
    v_component_of_wind: xr.DataArray,
) -> xr.DataArray:
    return xu.hypot(u_component_of_wind, v_component_of_wind)


def _compute_potential_temperature(temperature: xr.DataArray) -> xr.DataArray:
    """Dry potential temperature via Poisson's equation: θ = T (p₀/p)^(R_d/c_pd)."""
    p = _pressure_pa(temperature)
    return temperature * (100000.0 / p) ** (_R_D / _C_PD)


def _compute_equivalent_potential_temperature(
    temperature: xr.DataArray,
    specific_humidity: xr.DataArray,
) -> xr.DataArray:
    """Equivalent potential temperature via Bolton (1980) eq. 43.

    Reference: Bolton, D. (1980). Mon. Wea. Rev., 108, 1046-1053.
    """
    p = _pressure_pa(temperature)
    p_hpa = p / 100.0
    q = _clamped_specific_humidity(specific_humidity)
    e = _vapour_pressure(q, p_hpa)
    r = q / (1.0 - q)
    log_e = xu.log(e / 6.112)
    t_d = 243.5 * log_e / (17.67 - log_e) + 273.15
    t_l = 1.0 / (1.0 / (t_d - 56.0) + xu.log(temperature / t_d) / 800.0) + 56.0
    theta = temperature * (100000.0 / p) ** (_R_D / _C_PD)
    return theta * xu.exp((_L_V * r) / (_C_PD * t_l))


def _compute_virtual_temperature(
    temperature: xr.DataArray,
    specific_humidity: xr.DataArray,
) -> xr.DataArray:
    """Virtual temperature: T_v = T (1 + q/ε) / (1 + q)."""
    q = _clamped_specific_humidity(specific_humidity)
    return temperature * (1.0 + q / _EPSILON) / (1.0 + q)


def _compute_dewpoint_temperature(
    temperature: xr.DataArray,
    specific_humidity: xr.DataArray,
) -> xr.DataArray:
    """Dewpoint temperature via Bolton (1980) eq. 11."""
    p_hpa = _pressure_pa(temperature) / 100.0
    e = _vapour_pressure(_clamped_specific_humidity(specific_humidity), p_hpa)
    log_e = xu.log(e / 6.112)
    return 243.5 * log_e / (17.67 - log_e) + 273.15


def _compute_relative_humidity(
    temperature: xr.DataArray,
    specific_humidity: xr.DataArray,
) -> xr.DataArray:
    """Relative humidity as a fraction (0-1) from specific humidity and pressure."""
    p_hpa = _pressure_pa(temperature) / 100.0
    e = _vapour_pressure(_clamped_specific_humidity(specific_humidity), p_hpa)
    e_s = _saturation_vapour_pressure(temperature)
    return e / e_s


DERIVED_VARIABLE_REGISTRY: dict[str, DerivedVariableSpec] = {
    "wind_speed": DerivedVariableSpec(
        required_inputs=["u_component_of_wind", "v_component_of_wind"],
        compute=_compute_wind_speed,
    ),
    "potential_temperature": DerivedVariableSpec(
        required_inputs=["temperature"],
        compute=_compute_potential_temperature,
    ),
    "equivalent_potential_temperature": DerivedVariableSpec(
        required_inputs=["temperature", "specific_humidity"],
        compute=_compute_equivalent_potential_temperature,
    ),
    "virtual_temperature": DerivedVariableSpec(
        required_inputs=["temperature", "specific_humidity"],
        compute=_compute_virtual_temperature,
    ),
    "dewpoint_temperature": DerivedVariableSpec(
        required_inputs=["temperature", "specific_humidity"],
        compute=_compute_dewpoint_temperature,
    ),
    "relative_humidity": DerivedVariableSpec(
        required_inputs=["temperature", "specific_humidity"],
        compute=_compute_relative_humidity,
    ),
}


def classify_variables(
    requested: list[str],
    arco_available: set[str],
) -> tuple[list[str], list[str]]:
    """Split requested variable names into direct (in ARCO) and derived.

    Args:
        requested: Variable names from the user config.
        arco_available: Variable names present in the ARCO ERA5 Zarr store.

    Returns:
        Tuple of (direct_vars, derived_vars) where direct_vars are available
        in ARCO and derived_vars must be computed via the registry.

    Raises:
        ValueError: If any variable is neither in ``arco_available`` nor in
            ``DERIVED_VARIABLE_REGISTRY``.
    """
    direct_vars: list[str] = []
    derived_vars: list[str] = []
    unknown: list[str] = []

    for var in requested:
        if var in arco_available:
            direct_vars.append(var)
        elif var in DERIVED_VARIABLE_REGISTRY:
            derived_vars.append(var)
        else:
            unknown.append(var)

    if unknown:
        raise ValueError(
            f"Variables not available in ARCO ERA5 and have no derivation function: {unknown}. "
            f"Registered derivable variables: {sorted(DERIVED_VARIABLE_REGISTRY)}"
        )

    return direct_vars, derived_vars


def resolve_download_variables(
    direct_vars: list[str],
    derived_vars: list[str],
) -> list[str]:
    """Return the full list of variables to download from the source.

    Includes all direct variables plus any required inputs for derived variables
    that are not already in the direct list.

    Args:
        direct_vars: Variables available directly in the source.
        derived_vars: Variables that require derivation.

    Returns:
        Deduplicated list of variable names to fetch from the source.
    """
    to_download: list[str] = list(direct_vars)
    seen = set(direct_vars)

    for var in derived_vars:
        spec = DERIVED_VARIABLE_REGISTRY[var]
        for inp in spec.required_inputs:
            if inp not in seen:
                to_download.append(inp)
                seen.add(inp)

    return to_download
