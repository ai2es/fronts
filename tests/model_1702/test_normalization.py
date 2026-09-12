"""Tests for the embedded model_1702 normalization table."""

import os

import numpy as np
import pytest

from fronts.model_1702 import normalization

PROPERTIES_PATH = os.path.expanduser("~/data/fronts/model_1702_properties.txt")


def test_axis_orderings():
    assert normalization.VARIABLES == ["T", "Td", "Tv", "u", "v", "r", "q", "RH", "sp_z", "theta_e"]
    assert normalization.LEVELS == ["surface", "1000", "950", "900", "850"]
    assert normalization.LEVEL_COORD == [1013, 1000, 950, 900, 850]
    assert len(normalization.NORMALIZATION_PARAMETERS) == 50


def test_norm_tables_shape_and_placement():
    maxs, mins = normalization.norm_min_max_tables()
    assert maxs.shape == (5, 10)
    assert mins.shape == (5, 10)
    assert maxs.dtype == np.float32
    assert maxs[0, 0] == 323.0
    assert mins[0, 0] == 212.0
    assert maxs[0, 8] == 1075.0
    assert mins[0, 8] == 620.0
    assert maxs[4, 9] == 359.0
    assert mins[4, 9] == 238.0
    assert (maxs > mins).all()


def test_min_max_normalization_hand_computed():
    maxs, mins = normalization.norm_min_max_tables()
    t_surface = 267.4
    normalized = (t_surface - mins[0, 0]) / (maxs[0, 0] - mins[0, 0])
    assert normalized == pytest.approx((267.4 - 212.0) / (323.0 - 212.0))
    sp_surface_hpa = 1013.25
    normalized_sp = (sp_surface_hpa - mins[0, 8]) / (maxs[0, 8] - mins[0, 8])
    assert normalized_sp == pytest.approx((1013.25 - 620.0) / (1075.0 - 620.0))


def test_nan_maps_to_zero_like_legacy():
    values = np.array([np.nan, 300.0])
    maxs, mins = normalization.norm_min_max_tables()
    normalized = np.nan_to_num((values - mins[0, 0]) / (maxs[0, 0] - mins[0, 0]))
    assert normalized[0] == 0.0


@pytest.mark.skipif(not os.path.exists(PROPERTIES_PATH), reason="model_1702_properties.txt not available")
def test_embedded_table_matches_properties_file():
    parsed = normalization.parse_properties_normalization(PROPERTIES_PATH)
    for variable in normalization.VARIABLES:
        for level in normalization.LEVELS:
            key = f"{variable}_{level}"
            assert parsed[key] == normalization.NORMALIZATION_PARAMETERS[key], key
