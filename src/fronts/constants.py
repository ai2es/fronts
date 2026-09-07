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
# in the one-hot encoded array (0=background, 1=CF, 2=WF, 3=SF, 4=OF, 5=DL)
FRONT_TYPE_CLASS_INDEX: dict[str, int] = {"CF": 1, "WF": 2, "SF": 3, "OF": 4, "DL": 5}

# Token used to label the background (class 0) in per-front-type metric names. Must not
# collide with any key in FRONT_TYPE_CLASS_INDEX.
BACKGROUND_CLASS_KEY = "none"

FRONT_NAMES: dict[str, str] = {
    "CF": "Cold front",
    "WF": "Warm front",
    "SF": "Stationary front",
    "OF": "Occluded front",
    "DL": "Dryline",
}

FRONT_COLORS: dict[str, str] = {
    "CF": "blue",
    "WF": "red",
    "SF": "limegreen",
    "OF": "darkviolet",
    "DL": "chocolate",
}

CONTOUR_CMAPS: dict[str, str] = {
    "CF": "Blues",
    "WF": "Reds",
    "SF": "Greens",
    "OF": "Purples",
    "DL": "copper_r",
}

# Original front codes → experiment class indices
# 0 = no front (background), 1-4 kept as-is, 16 (dryline) → 5, all others → 0
FRONT_CLASS_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 16: 5}

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
