"""Shared constants for front types, plotting, and evaluation regions.

This module must depend only on the standard library and ``numpy`` — never on anything
else in ``fronts`` — so that dependency-heavy modules (``fronts.callbacks`` imports
``wandb``; ``fronts.plot.plot`` imports ``matplotlib`` and ``cartopy``) can be avoided by
code that only needs the front-type mapping, such as ``fronts.layers.metrics``.
"""

from collections import namedtuple

import numpy as np

BoundingBox = namedtuple("BoundingBox", ["lat_min", "lat_max", "lon_min", "lon_max"])

# Note that FRONT_TYPE_CLASS_INDEX is not the number from the fronts data, but the index
# in the one-hot encoded array (0=background, 1=CF, 2=WF, 3=SF, 4=OF, 5=DL, 6=TROF,
# 7=TT, 8=INST)
FRONT_TYPE_CLASS_INDEX: dict[str, int] = {"CF": 1, "WF": 2, "SF": 3, "OF": 4, "DL": 5, "TROF": 6, "TT": 7, "INST": 8}

# Token used to label the background (class 0) in per-front-type metric names. Must not
# collide with any key in FRONT_TYPE_CLASS_INDEX.
BACKGROUND_CLASS_KEY = "none"

# Front types that must all be present in the domain for targets.filter_timesteps to keep a
# timestep unconditionally (Justin et al. 2025, section 2b). Deliberately the five original
# types rather than every key in FRONT_TYPE_CLASS_INDEX: trough, tropical trough and
# instability axis are sparse enough in the label set that requiring them too would leave the
# rule almost never firing, collapsing the train/val sample to a straight 50% draw and making
# nine-class runs incomparable to the five-class runs they are meant to be measured against.
SAMPLING_REQUIRED_FRONT_TYPES: tuple[str, ...] = ("CF", "WF", "SF", "OF", "DL")

FRONT_NAMES: dict[str, str] = {
    "CF": "Cold front",
    "WF": "Warm front",
    "SF": "Stationary front",
    "OF": "Occluded front",
    "DL": "Dryline",
    "TROF": "Trough",
    "TT": "Tropical trough",
    "INST": "Instability axis",
}

FRONT_COLORS: dict[str, str] = {
    "CF": "blue",
    "WF": "red",
    "SF": "limegreen",
    "OF": "darkviolet",
    "DL": "chocolate",
    "TROF": "goldenrod",
    "TT": "deeppink",
    "INST": "gray",
}

CONTOUR_CMAPS: dict[str, str] = {
    "CF": "Blues",
    "WF": "Reds",
    "SF": "Greens",
    "OF": "Purples",
    "DL": "copper_r",
    "TROF": "YlOrBr",
    "TT": "PuRd",
    "INST": "Greys",
}

# Original front codes → experiment class indices.
# 0 = no front (background), 1-4 kept as-is, forming (5-8) and dissipating (9-12) variants
# collapse into their parent front class, 14=TROF -> 6, 15=TT -> 7, 16=DL -> 5, INST (13) is
# its own class -> 8. All other codes map to 0.
FRONT_CLASS_MAP = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 1,
    6: 2,
    7: 3,
    8: 4,
    9: 1,
    10: 2,
    11: 3,
    12: 4,
    13: 8,
    14: 6,
    15: 7,
    16: 5,
}

# Office-of-responsibility regions for the Unified Surface Analysis (WPC manual, p.25).
# The 30N split and the 140W HFO/NHC boundary come from the manual; WPC vs OPC is
# approximated as a longitude band over the continental US since the real WPC area of
# responsibility is an irregular coastline-following polygon, not a box.
OFFICE_REGIONS: dict[str, BoundingBox] = {
    "OPC_west": BoundingBox(lat_min=30.0, lat_max=80.0, lon_min=130.0, lon_max=220.0),
    "WPC": BoundingBox(lat_min=30.0, lat_max=80.0, lon_min=220.0, lon_max=300.0),
    "OPC_east": BoundingBox(lat_min=30.0, lat_max=80.0, lon_min=300.0, lon_max=369.75),
    "HFO": BoundingBox(lat_min=0.25, lat_max=30.0, lon_min=130.0, lon_max=220.0),
    "NHC": BoundingBox(lat_min=0.25, lat_max=30.0, lon_min=220.0, lon_max=369.75),
}

LITE_THRESHOLDS = np.linspace(0.05, 1.0, 20, dtype=np.float32)
