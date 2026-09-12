"""Tests for harness region masks, including the wrap-crossing longitude fix."""

import numpy as np
import pytest

from fronts.model_1702 import run_eval

FULL_DOMAIN_LATS = np.arange(80.0, 0.0, -0.25)
FULL_DOMAIN_LONS_WRAPPED = np.concatenate([np.arange(130.0, 360.0, 0.25), np.arange(0.0, 10.0, 0.25)])


def test_unwrap_longitudes_restores_monotonicity():
    unwrapped = run_eval.unwrap_longitudes(FULL_DOMAIN_LONS_WRAPPED)
    assert (np.diff(unwrapped) > 0).all()
    assert unwrapped[-1] == 369.75
    np.testing.assert_array_equal(unwrapped[:920], FULL_DOMAIN_LONS_WRAPPED[:920])


def test_unwrap_longitudes_no_op_for_monotonic_domain():
    conus_lons = np.arange(228.0, 300.0, 0.25)
    np.testing.assert_array_equal(run_eval.unwrap_longitudes(conus_lons), conus_lons)


def test_full_region_returns_none():
    assert run_eval.build_region_mask("full", FULL_DOMAIN_LATS, FULL_DOMAIN_LONS_WRAPPED) is None


def test_unknown_region_raises():
    with pytest.raises(ValueError, match="Unknown region"):
        run_eval.build_region_mask("EPO", FULL_DOMAIN_LATS, FULL_DOMAIN_LONS_WRAPPED)


def test_opc_east_includes_wrapped_columns():
    mask = run_eval.build_region_mask("OPC_east", FULL_DOMAIN_LATS, FULL_DOMAIN_LONS_WRAPPED)
    wrapped_columns = mask[:, FULL_DOMAIN_LONS_WRAPPED < 130.0]
    lat_rows_in_band = (FULL_DOMAIN_LATS >= 30.0) & (FULL_DOMAIN_LATS <= 80.0)
    assert wrapped_columns[lat_rows_in_band].all()
    assert not wrapped_columns[~lat_rows_in_band].any()


def test_office_masks_cover_domain_and_overlap_only_at_shared_edges():
    masks = {
        name: run_eval.build_region_mask(name, FULL_DOMAIN_LATS, FULL_DOMAIN_LONS_WRAPPED)
        for name in run_eval.OFFICE_REGIONS
    }
    coverage = np.zeros((len(FULL_DOMAIN_LATS), len(FULL_DOMAIN_LONS_WRAPPED)), dtype=int)
    for mask in masks.values():
        coverage += mask.astype(int)
    assert (coverage >= 1).all()
    interior_lat = ~np.isin(FULL_DOMAIN_LATS, [30.0])
    interior_lon = ~np.isin(run_eval.unwrap_longitudes(FULL_DOMAIN_LONS_WRAPPED), [220.0, 300.0])
    assert (coverage[np.ix_(interior_lat, interior_lon)] == 1).all()


def test_wpc_mask_covers_conus_band():
    mask = run_eval.build_region_mask("WPC", FULL_DOMAIN_LATS, FULL_DOMAIN_LONS_WRAPPED)
    lat_idx = np.where((FULL_DOMAIN_LATS >= 30.0) & (FULL_DOMAIN_LATS <= 56.75))[0]
    lon_idx = np.where((FULL_DOMAIN_LONS_WRAPPED >= 228.0) & (FULL_DOMAIN_LONS_WRAPPED <= 299.75))[0]
    assert mask[np.ix_(lat_idx, lon_idx)].all()


def test_land_ocean_masks_partition_domain():
    lats = np.arange(56.75, 25.0, -0.25)
    lons = np.arange(228.0, 300.0, 0.25)
    land = run_eval.build_region_mask("land", lats, lons)
    ocean = run_eval.build_region_mask("ocean", lats, lons)
    assert land.shape == (len(lats), len(lons))
    assert (land ^ ocean).all()
    assert land.sum() > 0
    assert ocean.sum() > 0
