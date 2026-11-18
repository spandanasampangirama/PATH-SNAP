"""End-to-end inference for one trip. Ground truth is never loaded by prediction code."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
import threading

import joblib
import numpy as np

from .candidates import generate_candidates, segment_track
from .config import CFG
from .features import FEATURE_COLS, add_features
from .hmm import viterbi_decode
from .io import load_trip
from .network import RoadNetwork
from .routing import Router


@dataclass
class PredictionResult:
    trip_id: str
    n_points: int
    classifier_arc: np.ndarray
    hmm_arc: np.ndarray
    route_arcs: list[int]
    timings_ms: dict = field(default_factory=dict)
    candidate_k: int | None = None
    candidate_radius_m: float | None = None


class PathSnapPredictor:
    """Load the P2 classifier and P3 decoder configuration once.

    A one-trip cache avoids rebuilding the spatial index and feature matrix for every HMM
    parameter trial. It is deliberately bounded to one trip to limit memory on large networks.
    """
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)
        self.model = None
        self.model_name = "distance-baseline"
        self.feature_cols = list(FEATURE_COLS)
        self.candidate_cfg = dict(CFG["candidates"])
        self.hmm_cfg = dict(CFG["hmm"])
        self._predict_lock = threading.RLock()
        self._trip_cache_key = None
        self._trip_cache = None

        meta_path = self.artifacts_dir / "model_meta.json"
        model_path = self.artifacts_dir / "model.joblib"
        baseline_path = self.artifacts_dir / "untuned" / "logreg.joblib"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.model_name = str(meta.get("model_name", self.model_name))
            self.feature_cols = list(meta.get("feature_cols", FEATURE_COLS))
            self.candidate_cfg.update(meta.get("candidates", {}))
        if model_path.exists():
            self.model = joblib.load(model_path)
            if not meta_path.exists():
                self.model_name = "classifier-artifact"
        elif baseline_path.exists():
            self.model = joblib.load(baseline_path)
            self.model_name = "logreg-untuned-baseline"
            self.feature_cols = list(FEATURE_COLS)

        hmm_path = self.artifacts_dir / "hmm_config.json"
        if hmm_path.exists():
            self.hmm_cfg.update(json.loads(hmm_path.read_text(encoding="utf-8")))
        if self.model is None:
            self.hmm_cfg["emission"] = "distance"

    def _score_candidates(self, cands):
        if self.model is None:
            sigma = max(float(self.hmm_cfg.get("sigma_m", 10.0)), 1e-6)
            cands["prob"] = np.exp(-0.5 * np.square(cands["dist_m"].to_numpy(float) / sigma))
            return cands
        enriched = add_features(cands, self._track_xy, self._track_t, self._net)
        absent = [col for col in self.feature_cols if col not in enriched.columns]
        if absent:
            raise ValueError(f"model metadata requires missing feature columns: {absent}")
        matrix = enriched[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("The exported classifier must implement predict_proba().")
        probabilities = self.model.predict_proba(matrix)
        classes = list(getattr(self.model, "classes_", range(probabilities.shape[1])))
        positive = classes.index(1) if 1 in classes else probabilities.shape[1] - 1
        enriched["prob"] = probabilities[:, positive]
        return enriched

    @staticmethod
    def _classifier_top1(cands, n_points):
        out = np.full(n_points, -1, dtype=np.int64)
        for point, rows in cands.groupby("point_idx", sort=False):
            scores = rows["prob"].to_numpy(float) if "prob" in rows else -rows["dist_m"].to_numpy(float)
            out[int(point)] = int(rows.iloc[int(np.argmax(scores))]["arc_id"])
        return out

    @staticmethod
    def _continuous_route(decoded, segment_ids, router, max_m):
        route, previous, previous_segment = [], None, None
        for arc, segment in zip(decoded, segment_ids):
            arc, segment = int(arc), int(segment)
            if arc < 0:
                # Keep the last valid state: a short candidate-free interval may be bridged
                # by a bounded directed shortest path. Long time gaps have distinct segments.
                continue
            if previous is None or previous_segment != segment:
                if not route or route[-1] != arc:
                    route.append(arc)
            elif arc != previous:
                for mid in router.shortest_arc_path(previous, arc, max_m):
                    if not route or route[-1] != mid:
                        route.append(int(mid))
                if not route or route[-1] != arc:
                    route.append(arc)
            previous, previous_segment = arc, segment
        return route

    def predict(self, partition_dir: str, trip_id: str) -> tuple[PredictionResult, RoadNetwork]:
        # The model, feature context and one-trip cache are shared by API requests.
        # Serialize prediction to prevent concurrent requests from mixing trip state.
        with self._predict_lock:
            return self._predict_locked(partition_dir, trip_id)

    def _predict_locked(self, partition_dir: str, trip_id: str) -> tuple[PredictionResult, RoadNetwork]:
        timings, start = {}, time.perf_counter()
        cache_key = (
            str(Path(partition_dir).resolve()), str(trip_id),
            int(self.candidate_cfg["k"]), float(self.candidate_cfg["radius_m"]),
            float(self.hmm_cfg.get("gap_split_s", CFG["gaps"]["split_s"])),
        )
        cache_hit = cache_key == self._trip_cache_key and self._trip_cache is not None
        if cache_hit:
            trip, net, xy, cands = self._trip_cache
            self._net, self._track_xy, self._track_t = net, xy, trip.track[:, 2]
            timings.update({"network": 0.0, "candidates": 0.0, "model": 0.0, "cache_hit": 1.0})
        else:
            t0 = time.perf_counter()
            trip = load_trip(partition_dir, trip_id, with_route=False)
            net = RoadNetwork(trip)
            xy = net.project_track(trip.track)
            timings["network"] = (time.perf_counter() - t0) * 1000
            self._net, self._track_xy, self._track_t = net, xy, trip.track[:, 2]

            t0 = time.perf_counter()
            cands = generate_candidates(
                trip, net, int(self.candidate_cfg["k"]), float(self.candidate_cfg["radius_m"]),
                split_s=float(self.hmm_cfg.get("gap_split_s", CFG["gaps"]["split_s"])), track_xy=xy,
            )
            timings["candidates"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            cands = self._score_candidates(cands)
            timings["model"] = (time.perf_counter() - t0) * 1000
            self._trip_cache_key, self._trip_cache = cache_key, (trip, net, xy, cands)
            timings["cache_hit"] = 0.0

        classifier_arc = self._classifier_top1(cands, len(trip.track))
        t0 = time.perf_counter()
        router = Router(net)
        hmm_arc = viterbi_decode(cands, "prob", router, xy, self.hmm_cfg)
        timings["dead_end_restarts"] = float(getattr(router, "dead_end_restarts", 0))
        timings["viterbi"] = (time.perf_counter() - t0) * 1000
        segment_ids = segment_track(
            trip.track[:, 2], float(self.hmm_cfg.get("gap_split_s", CFG["gaps"]["split_s"])))
        route_arcs = self._continuous_route(
            hmm_arc, segment_ids, router, float(self.hmm_cfg.get("route_fill_max_m", 2000.0)))
        timings["total"] = (time.perf_counter() - start) * 1000
        return PredictionResult(
            trip_id, len(trip.track), classifier_arc, hmm_arc, route_arcs, timings,
            int(self.candidate_cfg["k"]), float(self.candidate_cfg["radius_m"]),
        ), net
