"""P3 integration regressions: route filling, dual metrics and bounded inference caching."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pathsnap.evaluate import evaluate_prediction
from pathsnap.pipeline import PathSnapPredictor, PredictionResult


def test_route_fill_bridges_only_within_a_segment():
    class FakeRouter:
        def shortest_arc_path(self, source, target, max_m):
            return [1] if (source, target) == (0, 2) and max_m >= 10 else []

    router = FakeRouter()
    bridged = PathSnapPredictor._continuous_route(
        np.array([0, -1, 2]), np.array([0, 0, 0]), router, 10.0)
    assert bridged == [0, 1, 2]

    separated = PathSnapPredictor._continuous_route(
        np.array([0, -1, 2]), np.array([0, 1, 1]), router, 10.0)
    assert separated == [0, 2]


def test_evaluator_reports_classifier_only_and_hmm_route_metrics(monkeypatch):
    import pathsnap.evaluate as evaluation

    context = {
        "trip": SimpleNamespace(route=np.array([0, 1, 2])),
        "reference_arc": np.array([0, 1, 2]),
        "segments": np.array([0, 0, 0]),
        "candidate_miss_rate": 0.0,
    }
    monkeypatch.setattr(evaluation, "prepare_evaluation", lambda *args, **kwargs: context)

    class FakeNetwork:
        @staticmethod
        def successor_arcs(arc):
            return np.array([arc + 1]) if arc < 2 else np.array([], dtype=int)

    result = PredictionResult(
        trip_id="synthetic",
        n_points=3,
        classifier_arc=np.array([0, 0, 2]),
        hmm_arc=np.array([0, 1, 2]),
        route_arcs=[0, 1, 2],
    )
    metrics = evaluate_prediction(result, "unused", FakeNetwork())

    assert metrics["top1_acc_classifier"] == 2 / 3
    assert metrics["top1_acc_hmm"] == 1.0
    assert metrics["classifier_route_f1"] < metrics["hmm_route_f1"]
    assert metrics["hmm_route_f1"] == metrics["route_f1"] == 1.0
    assert metrics["classifier_route_precision"] == 1.0
    assert metrics["classifier_route_recall"] == 2 / 3


def test_predictor_reuses_one_trip_preprocessing_cache(tmp_path, monkeypatch):
    import pathsnap.pipeline as pipeline

    calls = {"load": 0, "network": 0, "candidates": 0}
    trip = SimpleNamespace(
        track=np.array([[0.0, 0.0, 0.0], [0.0001, 0.0, 1.0]]),
    )

    class FakeNetwork:
        @staticmethod
        def project_track(track):
            return np.zeros((len(track), 2), dtype=float)

    class FakeRouter:
        dead_end_restarts = 0

        @staticmethod
        def shortest_arc_path(source, target, max_m):
            return []

    fake_net, fake_router = FakeNetwork(), FakeRouter()

    def load_trip(*args, **kwargs):
        calls["load"] += 1
        return trip

    def make_network(_trip):
        calls["network"] += 1
        return fake_net

    def make_candidates(*args, **kwargs):
        calls["candidates"] += 1
        return pd.DataFrame([
            {"point_idx": 0, "arc_id": 0, "dist_m": 1.0, "segment_id": 0, "proj_frac": 0.5},
            {"point_idx": 1, "arc_id": 1, "dist_m": 1.0, "segment_id": 0, "proj_frac": 0.5},
        ])

    monkeypatch.setattr(pipeline, "load_trip", load_trip)
    monkeypatch.setattr(pipeline, "RoadNetwork", make_network)
    monkeypatch.setattr(pipeline, "generate_candidates", make_candidates)
    monkeypatch.setattr(pipeline, "Router", lambda _net: fake_router)
    monkeypatch.setattr(pipeline, "viterbi_decode", lambda *args, **kwargs: np.array([0, 1]))
    monkeypatch.setattr(pipeline, "segment_track", lambda *args, **kwargs: np.array([0, 0]))

    predictor = PathSnapPredictor(str(tmp_path))
    first, _ = predictor.predict("synthetic-partition", "synthetic")
    second, _ = predictor.predict("synthetic-partition", "synthetic")

    assert calls == {"load": 1, "network": 1, "candidates": 1}
    assert first.timings_ms["cache_hit"] == 0.0
    assert second.timings_ms["cache_hit"] == 1.0
    np.testing.assert_array_equal(first.hmm_arc, second.hmm_arc)
