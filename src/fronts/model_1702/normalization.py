"""Input layout and min-max normalization table for model_1702.

The variable and level orderings define the last two axes of model_1702's input tensor and must
never be reordered. ``NORMALIZATION_PARAMETERS`` is embedded verbatim from
``model_1702_properties.txt`` (identical to ``utils/data_utils.py`` at git commit ``c59ac2b``);
a parity test validates the embedded values against the properties file when it is available.
Legacy normalization is ``nan_to_num((x - min) / (max - min))`` applied before the model, unlike
2.0 models which embed normalization as a layer.
"""

import ast

import numpy as np

VARIABLES = ["T", "Td", "Tv", "u", "v", "r", "q", "RH", "sp_z", "theta_e"]
LEVELS = ["surface", "1000", "950", "900", "850"]
LEVEL_COORD = [1013, 1000, 950, 900, 850]

NORMALIZATION_PARAMETERS: dict[str, tuple[float, float]] = {
    "T_surface": (323.0, 212.0),
    "T_1000": (322.0, 218.0),
    "T_950": (319.0, 216.0),
    "T_900": (314.0, 220.0),
    "T_850": (315.0, 227.0),
    "Td_surface": (304.0, 207.0),
    "Td_1000": (302.0, 208.0),
    "Td_950": (301.0, 210.0),
    "Td_900": (298.0, 200.0),
    "Td_850": (296.0, 200.0),
    "Tv_surface": (324.0, 211.0),
    "Tv_1000": (323.0, 206.0),
    "Tv_950": (319.0, 206.0),
    "Tv_900": (316.0, 220.0),
    "Tv_850": (316.0, 227.0),
    "u_surface": (36.0, -35.0),
    "u_1000": (38.0, -35.0),
    "u_950": (48.0, -55.0),
    "u_900": (59.0, -58.0),
    "u_850": (59.0, -58.0),
    "v_surface": (30.0, -35.0),
    "v_1000": (35.0, -38.0),
    "v_950": (55.0, -56.0),
    "v_900": (58.0, -59.0),
    "v_850": (58.0, -59.0),
    "r_surface": (25.0, 0.0),
    "r_1000": (22.0, 0.0),
    "r_950": (22.0, 0.0),
    "r_900": (20.0, 0.0),
    "r_850": (18.0, 0.0),
    "q_surface": (24.0, 0.0),
    "q_1000": (26.0, 0.0),
    "q_950": (26.0, 0.0),
    "q_900": (23.0, 0.0),
    "q_850": (21.0, 0.0),
    "RH_surface": (1.0, 0.0),
    "RH_1000": (1.0, 0.0),
    "RH_950": (1.0, 0.0),
    "RH_900": (1.0, 0.0),
    "RH_850": (1.0, 0.0),
    "sp_z_surface": (1075.0, 620.0),
    "sp_z_1000": (48.0, -69.0),
    "sp_z_950": (86.0, -27.0),
    "sp_z_900": (127.0, 17.0),
    "sp_z_850": (174.0, 63.0),
    "theta_e_surface": (375.0, 213.0),
    "theta_e_1000": (366.0, 208.0),
    "theta_e_950": (367.0, 210.0),
    "theta_e_900": (364.0, 227.0),
    "theta_e_850": (359.0, 238.0),
}


def norm_min_max_tables() -> tuple[np.ndarray, np.ndarray]:
    """Builds max and min lookup arrays aligned with model_1702's input axes.

    Returns:
        Tuple of (maxs, mins), each with shape (len(LEVELS), len(VARIABLES)) so they broadcast
        against input tensors of shape (..., level, variable).
    """
    maxs = np.empty((len(LEVELS), len(VARIABLES)), dtype=np.float32)
    mins = np.empty_like(maxs)
    for level_idx, level in enumerate(LEVELS):
        for var_idx, variable in enumerate(VARIABLES):
            maxs[level_idx, var_idx], mins[level_idx, var_idx] = NORMALIZATION_PARAMETERS[f"{variable}_{level}"]
    return maxs, mins


def parse_properties_normalization(path: str) -> dict[str, tuple[float, float]]:
    """Parses the normalization table from a model properties text file.

    Intended only for the parity test against the embedded table; runtime code uses
    ``NORMALIZATION_PARAMETERS`` directly.

    Args:
        path: Path to a ``model_*_properties.txt`` file containing a
            ``normalization_parameters: {...}`` line with a Python dict literal.

    Returns:
        Mapping of "{variable}_{level}" to (max, min).

    Raises:
        ValueError: If no normalization_parameters line is found.
    """
    prefix = "normalization_parameters:"
    with open(path, encoding="utf-8") as properties_file:
        for line in properties_file:
            if line.strip().startswith(prefix):
                table = ast.literal_eval(line.strip().removeprefix(prefix).strip())
                return {key: (float(value[0]), float(value[1])) for key, value in table.items()}
    raise ValueError(f"No '{prefix}' line found in {path}")
