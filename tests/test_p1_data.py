"""Tests for P1 modules: geometry, network, labels, candidates."""
import numpy as np
from pyproj import Geod

from pathsnap import schema
from pathsnap.candidates import generate_candidates, points_without_candidates, segment_track
from pathsnap.config import CFG
from pathsnap.geo import LocalProjector, angle_diff_deg, bearing_deg, point_segment_distance
from pathsnap.io import load_trip
from pathsnap.labels import align_points_to_route
from pathsnap.network import RoadNetwork

SMALL_TRIP = "00000046"


def _small():
    trip = load_trip(CFG["paths"]["train_dir"], SMALL_TRIP, with_route=True)
    net = RoadNetwork(trip)
    return trip, net, net.project_track(trip.track)


def test_point_segment_distance():
    d, u = point_segment_distance(np.array([5.0, -3.0, 14.0]), np.array([3.0, 0.0, 0.0]), 0, 0, 10, 0)
    np.testing.assert_allclose(d, [3.0, 3.0, 4.0])
    np.testing.assert_allclose(u, [0.5, 0.0, 1.0])
    d, u = point_segment_distance(1.0, 1.0, 0, 0, 0, 0)  # zero-length segment
    assert np.isclose(d, np.sqrt(2)) and u == 0


def test_bearing_and_angle_diff():
    np.testing.assert_allclose(bearing_deg(0, 0, np.array([0, 1, 0, -1]), np.array([1, 0, -1, 0])), [0, 90, 180, 270])
    assert angle_diff_deg(350, 10) == 20 and angle_diff_deg(0, 180) == 180


def test_projection_matches_ellipsoid_distance():
    proj = LocalProjector(8.6, 49.85)
    lon = np.array([8.55, 8.70]); lat = np.array([49.80, 49.90])
    x, y = proj.to_xy(lon, lat)
    _, _, d = Geod(ellps="WGS84").inv(lon[0], lat[0], lon[1], lat[1])
    assert abs(np.hypot(x[1] - x[0], y[1] - y[0]) - d) < 0.05  # < 5 cm over ~14 km
    lo, la = proj.to_lonlat(x, y)
    np.testing.assert_allclose(lo, lon, atol=1e-9); np.testing.assert_allclose(la, lat, atol=1e-9)


def test_network_connectivity():
    trip, net, _ = _small()
    for a in range(0, net.n_arcs, 37):
        for s in net.successor_arcs(a):
            assert net.arcs[s, 0] == net.arcs[a, 1]
    rev = net.reverse_arc
    has = rev >= 0
    assert np.all(net.arcs[rev[has], 0] == net.arcs[has, 1]) and np.all(net.arcs[rev[has], 1] == net.arcs[has, 0])


def test_query_arcs_sorted_within_radius():
    _, net, xy = _small()
    ids, d, f = net.query_arcs(xy[100, 0], xy[100, 1], 60.0, 8)
    assert 0 < len(ids) <= 8 and np.all(np.diff(d) >= 0) and d.max() <= 60.0 + 1e-6
    # brute force agrees on the nearest arc
    bd, _ = net.arc_distance(xy[100, 0], xy[100, 1], np.arange(net.n_arcs))
    assert np.isclose(d[0], bd.min())


def test_labels_monotone_and_close():
    trip, net, xy = _small()
    lab = align_points_to_route(net, xy, trip.route, 50.0)
    assert lab.shape == (len(xy),) and (lab >= 0).mean() > 0.95
    pos = {a: i for i, a in enumerate(trip.route)}  # route positions must never decrease
    seq = [pos[a] for a in lab if a >= 0]
    assert all(b >= a for a, b in zip(seq, seq[1:])) or len(set(trip.route)) < len(trip.route)


def test_candidates_table_contract():
    trip, net, xy = _small()
    lab = align_points_to_route(net, xy, trip.route, 50.0)
    c = generate_candidates(trip, net, 8, 60.0, lab, track_xy=xy)
    for col in schema.KEY_COLS + schema.CANDIDATE_BASE_COLS + [schema.LABEL_COL]:
        assert col in c.columns
    assert c.groupby("point_idx").size().max() <= 8
    assert set(c[schema.LABEL_COL].unique()) <= {-1, 0, 1}
    assert (c.groupby("point_idx")[schema.LABEL_COL].apply(lambda s: (s == 1).sum()) <= 1).all()
    assert len(points_without_candidates(c, len(xy))) == len(xy) - c.point_idx.nunique()


def test_segment_track():
    np.testing.assert_array_equal(segment_track(np.array([0, 1, 2, 100, 101, 500.0]), 60), [0, 0, 0, 1, 1, 2])
