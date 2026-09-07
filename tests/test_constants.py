"""Tests for fronts.constants: the shared, dependency-light constants module.

These tests double as the guarantee the whole module exists for: importing
``fronts.constants`` must never pull in ``wandb``, ``matplotlib``, or ``tensorflow``, so
that a pure-TensorFlow module (e.g. ``fronts.layers.metrics``) can depend on the front-type
mapping without dragging those in.
"""

import subprocess
import sys

from fronts import constants


def test_front_type_class_index_matches_expected_mapping():
    assert constants.FRONT_TYPE_CLASS_INDEX == {"CF": 1, "WF": 2, "SF": 3, "OF": 4, "DL": 5}


def test_every_front_type_has_a_name_color_and_cmap():
    front_types = set(constants.FRONT_TYPE_CLASS_INDEX)
    assert front_types <= set(constants.FRONT_NAMES)
    assert front_types <= set(constants.FRONT_COLORS)
    assert front_types <= set(constants.CONTOUR_CMAPS)


def test_class_indices_are_unique_contiguous_and_exclude_background():
    indices = sorted(constants.FRONT_TYPE_CLASS_INDEX.values())
    assert len(set(indices)) == len(indices)
    assert 0 not in indices
    assert indices == list(range(1, len(indices) + 1))


def test_background_class_key_does_not_collide_with_a_front_type():
    assert constants.BACKGROUND_CLASS_KEY not in constants.FRONT_TYPE_CLASS_INDEX


def test_front_class_map_values_match_front_type_class_index_values():
    assert set(constants.FRONT_CLASS_MAP.values()) == set(constants.FRONT_TYPE_CLASS_INDEX.values())


def test_importing_constants_does_not_import_heavy_optional_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import fronts.constants; "
            "heavy = {'wandb', 'matplotlib', 'tensorflow'} & set(sys.modules); "
            "assert not heavy, heavy",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
