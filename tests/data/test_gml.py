import numpy as np
import pytest

from fronts.data import gml

LATS = np.arange(-80.0, 80.25, 0.25)
LONS = np.arange(0.0, 360.0, 0.25)

_FILE = "rec.sfcanalysis.20230615T150000Z.NIL.P0D.WORLD@10km.FRONTS.SFC.gml"


def _front(ftype: str, coords: str) -> str:
    return (
        f'<gml:featureMember><FRONT gid="1"><FRONT_TYPE>{ftype}</FRONT_TYPE><gml:lineStringProperty>'
        f'<gml:LineString><gml:coordinates cs="," decimal="." ts=" ">{coords}</gml:coordinates>'
        "</gml:LineString></gml:lineStringProperty></FRONT></gml:featureMember>"
    )


def _write(tmp_path, *members: str):
    path = tmp_path / _FILE
    isobar = (
        '<gml:featureMember><CONTOUR_LINE gid="0"><gml:lineStringProperty><gml:LineString>'
        '<gml:coordinates cs="," decimal="." ts=" ">48.1,-58.0 47.3,-58.4</gml:coordinates>'
        "</gml:LineString></gml:lineStringProperty></CONTOUR_LINE></gml:featureMember>"
    )
    path.write_text(
        f'<?xml version="1.0"?><collection xmlns:gml="http://www.opengis.net/gml">{isobar}{"".join(members)}</collection>'
    )
    return path


def test_valid_time_from_path():
    assert gml.valid_time_from_path(_FILE) == np.datetime64("2023-06-15T15:00:00")


def test_parse_ignores_isobars_and_reads_lon_lat(tmp_path):
    path = _write(tmp_path, _front("Cold Front", "-100.0,40.0 -99.0,41.0"))
    fronts, unmapped = gml.parse_fronts(path)
    assert len(fronts) == 1 and not unmapped
    code, (lon, lat) = fronts[0]
    assert code == 1
    np.testing.assert_array_equal(lon, [-100.0, -99.0])
    np.testing.assert_array_equal(lat, [40.0, 41.0])


def test_parse_counts_unmapped_types(tmp_path):
    path = _write(tmp_path, _front("Mystery Front", "10.0,10.0 11.0,11.0"), _front("Trough", "10.0,10.0 11.0,11.0"))
    fronts, unmapped = gml.parse_fronts(path)
    assert [c for c, _ in fronts] == [14]
    assert unmapped == {"Mystery Front": 1}


def test_parse_rejects_lat_lon_ordered_fronts(tmp_path):
    path = _write(tmp_path, _front("Cold Front", "100.0,-120.0 101.0,-119.0"))
    with pytest.raises(ValueError, match="lon,lat"):
        gml.parse_fronts(path)


def test_parse_clips_small_overshoot_past_the_pole(tmp_path):
    path = _write(tmp_path, _front("Trough", "-163.7,86.6 -161.3,90.92"))
    ((_, (_, lat)),) = gml.parse_fronts(path)[0]
    np.testing.assert_array_equal(lat, [86.6, 90.0])


def test_rasterize_places_cells_on_expected_lat_lon():
    # Horizontal line at 40N from 250E to 251E (given as -110..-109).
    grid = gml.rasterize_fronts([(2, (np.array([-110.0, -109.0]), np.array([40.0, 40.0])))], LATS, LONS)
    rows, cols = np.nonzero(grid)
    assert set(LATS[rows]) == {40.0}
    assert LONS[cols].min() == 250.0 and LONS[cols].max() == 251.0
    assert len(cols) == 5  # 250.0, 250.25, ... 251.0
    assert set(grid[rows, cols]) == {2}


def test_rasterize_crosses_antimeridian_without_spanning_the_map():
    grid = gml.rasterize_fronts([(1, (np.array([178.0, -178.0]), np.array([0.0, 0.0])))], LATS, LONS)
    cols = np.nonzero(grid)[1]
    assert LONS[cols].min() == 178.0 and LONS[cols].max() == 182.0
    assert len(cols) == 17  # 4 degrees at 0.25 spacing, inclusive
    assert not grid[:, 4 * 4 : 170 * 4].any()


def test_rasterize_draw_order_cold_front_wins_over_trough():
    line = (np.array([10.0, 12.0]), np.array([20.0, 20.0]))
    for fronts in ([(1, line), (14, line)], [(14, line), (1, line)]):
        grid = gml.rasterize_fronts(fronts, LATS, LONS)
        assert set(grid[grid > 0]) == {1}


def test_rasterize_drops_points_outside_latitude_range():
    grid = gml.rasterize_fronts([(1, (np.array([10.0, 10.0]), np.array([85.0, 89.0])))], LATS, LONS)
    assert not grid.any()


def test_make_front_dataset_layout():
    grid = np.zeros((len(LATS), len(LONS)), dtype=np.int8)
    ds = gml.make_front_dataset([np.datetime64("2023-06-15T15:00")], [grid], LATS, LONS)
    assert ds["identifier"].dims == ("time", "latitude", "longitude")
    assert ds["identifier"].dtype == np.int8
