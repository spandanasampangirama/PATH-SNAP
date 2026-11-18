"""Synthetic P3 tests: directed routing, fractional distances, Viterbi and missing observations."""
import numpy as np
import pandas as pd
from pathsnap.pipeline import PathSnapPredictor

from pathsnap.io import Trip
from pathsnap.network import RoadNetwork
from pathsnap.routing import Router
from pathsnap.hmm import viterbi_decode


def _network():
    nodes = np.array([[0.0, 0.0], [0.0001, 0.0], [0.0002, 0.0], [0.0003, 0.0]])
    arcs = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    track = np.array([[0.00005, 0.0, 0.0], [0.00015, 0.0, 1.0], [0.00025, 0.0, 2.0]])
    trip = Trip("synthetic", track, nodes, arcs)
    return RoadNetwork(trip)


def test_fractional_directed_route_distance_and_intermediate_arcs():
    net = _network()
    router = Router(net)
    expected = 0.5 * net.arc_len_m[0] + net.arc_len_m[1] + 0.5 * net.arc_len_m[2]
    actual = router.route_distance(0, 0.5, 2, 0.5, 100.0)
    assert np.isclose(actual, expected, atol=1e-5)
    assert router.shortest_arc_path(0, 2, 100.0) == [1]
    assert np.isinf(router.route_distance(2, 0.5, 0, 0.5, 100.0))


def test_bounded_route_returns_infinity():
    net = _network()
    router = Router(net)
    assert np.isinf(router.route_distance(0, 0.0, 2, 1.0, 5.0))


def _candidates(arcs_by_point, segments=None):
    rows = []
    for point, arc in enumerate(arcs_by_point):
        if arc is None:
            continue
        rows.append(dict(point_idx=point, segment_id=(segments or [0] * len(arcs_by_point))[point],
                         arc_id=arc, proj_frac=0.5, dist_m=1.0, prob=0.9))
    return pd.DataFrame(rows, columns=["point_idx", "segment_id", "arc_id", "proj_frac", "dist_m", "prob"])


def test_viterbi_follows_connected_sequence():
    net = _network()
    xy = net.project(net.nodes_lonlat[:3, 0], net.nodes_lonlat[:3, 1])
    cands = _candidates([0, 1, 2])
    decoded = viterbi_decode(cands, "prob", Router(net), xy,
                             {"emission": "classifier", "beta_m": 5, "max_route_factor": 4})
    np.testing.assert_array_equal(decoded, [0, 1, 2])


def test_viterbi_preserves_no_candidate_as_minus_one():
    net = _network()
    xy = net.project(net.nodes_lonlat[:3, 0], net.nodes_lonlat[:3, 1])
    decoded = viterbi_decode(_candidates([0, None, 2]), "prob", Router(net), xy, {"emission": "distance"})
    np.testing.assert_array_equal(decoded, [0, -1, 2])


def test_viterbi_splits_at_segment_boundaries():
    net = _network()
    xy = net.project_track(net.nodes_lonlat[:3])
    decoded = viterbi_decode(_candidates([0, 1, 2], [0, 1, 1]), "prob", Router(net), xy,
                             {"emission": "distance"})
    np.testing.assert_array_equal(decoded, [0, 1, 2])


def test_viterbi_restarts_after_dead_end():
    net = _network()
    xy = net.project(net.nodes_lonlat[[0, 2], 0], net.nodes_lonlat[[0, 2], 1])
    rows = pd.DataFrame([
        dict(point_idx=0, segment_id=0, arc_id=0, proj_frac=0.5, dist_m=1.0, prob=0.9),
        dict(point_idx=1, segment_id=0, arc_id=2, proj_frac=0.5, dist_m=1.0, prob=0.8),
    ])
    router = Router(net)
    decoded = viterbi_decode(rows, "prob", router, xy,
                             {"emission": "classifier", "max_route_factor": 0, "max_route_extra_m": 0})
    np.testing.assert_array_equal(decoded, [0, 2])
    assert router.dead_end_restarts == 1


def test_pipeline_loads_committed_p2_baseline_when_final_artifact_is_absent(tmp_path, monkeypatch):
    import pathsnap.pipeline as pipeline_module
    artifacts = tmp_path / "artifacts"
    baseline = artifacts / "untuned" / "logreg.joblib"
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(b"synthetic-test-artifact")
    sentinel_model = object()
    monkeypatch.setattr(pipeline_module.joblib, "load", lambda path: sentinel_model)

    predictor = PathSnapPredictor(str(artifacts))

    assert predictor.model is sentinel_model
    assert predictor.model_name == "logreg-untuned-baseline"
    assert predictor.feature_cols == pipeline_module.FEATURE_COLS
    assert predictor.hmm_cfg["emission"] == "classifier"

def test_viterbi_backpointer_compares_transition_scores_before_current_emission():
    from pathsnap.hmm import _decode_chunk

    class ControlledRouter:
        def route_distance(self, prev_arc, prev_frac, arc, frac, bound):
            return 0.0 if prev_arc == 0 else 0.05

    previous = pd.DataFrame([
        dict(arc_id=0, proj_frac=0.5, dist_m=1.0, prob=0.9),
        dict(arc_id=1, proj_frac=0.5, dist_m=1.0, prob=0.8),
    ])
    current = pd.DataFrame([
        dict(arc_id=2, proj_frac=0.5, dist_m=1.0, prob=0.01),
    ])
    decoded = _decode_chunk(
        [previous, current], [0, 1], ControlledRouter(),
        {"emission": "classifier", "beta_m": 1.0, "max_route_factor": 4.0,
         "max_route_extra_m": 50.0, "_track_xy": np.zeros((2, 2))},
        "prob",
    )
    np.testing.assert_array_equal(decoded, [0, 2])

