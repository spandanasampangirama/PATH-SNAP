"""Server-side evaluation; this is the only inference-adjacent module that reads .route."""
from __future__ import annotations

import numpy as np

from .candidates import generate_candidates, segment_track
from .config import CFG
from .io import load_trip
from .labels import align_points_to_route
from .pipeline import PredictionResult
from .network import RoadNetwork


def _top1(pred: np.ndarray, ref: np.ndarray) -> float:
    mask = ref >= 0
    return float(np.mean(pred[mask] == ref[mask])) if mask.any() else float("nan")


def _continuity(pred: np.ndarray, segments: np.ndarray, net: RoadNetwork, max_hops: int) -> float:
    pairs = [(int(pred[i]), int(pred[i + 1])) for i in range(len(pred) - 1)
             if segments[i] == segments[i + 1] and pred[i] >= 0 and pred[i + 1] >= 0]
    if not pairs:
        return float("nan")

    def reachable(source, target):
        if source == target:
            return True
        seen, frontier = {source}, [source]
        for _ in range(max(0, int(max_hops))):
            next_frontier = []
            for arc in frontier:
                for successor in net.successor_arcs(arc):
                    successor = int(successor)
                    if successor == target:
                        return True
                    if successor not in seen:
                        seen.add(successor)
                        next_frontier.append(successor)
            frontier = next_frontier
            if not frontier:
                break
        return False

    return float(np.mean([reachable(a, b) for a, b in pairs]))


def _overlap(pred, ref):
    predicted, reference = set(map(int, pred)), set(map(int, ref))
    if not predicted and not reference:
        return 1.0, 1.0, 1.0
    precision = len(predicted & reference) / len(predicted) if predicted else 0.0
    recall = len(predicted & reference) / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _collapse_point_route(pred: np.ndarray, segments: np.ndarray) -> list[int]:
    """Collapse consecutive classifier arc predictions without joining separate time segments."""
    route, previous, previous_segment = [], None, None
    for arc, segment in zip(pred, segments):
        arc, segment = int(arc), int(segment)
        if arc < 0:
            continue
        if previous is None or segment != previous_segment or arc != previous:
            route.append(arc)
        previous, previous_segment = arc, segment
    return route


def prepare_evaluation(partition_dir: str, trip_id: str, net: RoadNetwork,
                       candidate_k: int | None = None, candidate_radius_m: float | None = None) -> dict:
    """Load reference-only evaluation context once; reuse it across HMM parameter trials."""
    trip = load_trip(partition_dir, trip_id, with_route=True)
    xy = net.project_track(trip.track)
    ref = align_points_to_route(net, xy, trip.route, float(CFG["labels"]["max_dist_m"]))
    k = int(candidate_k if candidate_k is not None else CFG["candidates"]["k"])
    radius = float(candidate_radius_m if candidate_radius_m is not None else CFG["candidates"]["radius_m"])
    cands = generate_candidates(
        trip, net, k, radius, split_s=float(CFG["gaps"]["split_s"]), track_xy=xy,
    )
    labelled = ref >= 0
    pairs = set(zip(cands["point_idx"].astype(int), cands["arc_id"].astype(int)))
    present = np.asarray([(i, int(arc)) in pairs if arc >= 0 else False
                          for i, arc in enumerate(ref)], dtype=bool)
    miss = float(1.0 - present[labelled].mean()) if labelled.any() else float("nan")
    return {
        "trip": trip,
        "reference_arc": ref,
        "segments": segment_track(trip.track[:, 2], float(CFG["gaps"]["split_s"])),
        "candidate_miss_rate": miss,
    }


def evaluate_prediction(result: PredictionResult, partition_dir: str, net: RoadNetwork,
                         context: dict | None = None) -> dict:
    """Return classifier-only and HMM metrics for one trip.

    PredictionResult is produced without loading the reference route. This evaluator loads
    the route separately and returns numeric metrics only; reference geometry/IDs are not
    added to PredictionResult or returned to the API/UI.
    """
    context = context or prepare_evaluation(
        partition_dir, result.trip_id, net, result.candidate_k, result.candidate_radius_m)
    trip = context["trip"]
    ref = context["reference_arc"]
    segments = context["segments"]
    miss = float(context["candidate_miss_rate"])

    classifier_route = _collapse_point_route(result.classifier_arc, segments)
    classifier_precision, classifier_recall, classifier_f1 = _overlap(classifier_route, trip.route)
    hmm_precision, hmm_recall, hmm_f1 = _overlap(result.route_arcs, trip.route)

    return {
        "top1_acc_classifier": _top1(result.classifier_arc, ref),
        "top1_acc_hmm": _top1(result.hmm_arc, ref),
        "route_continuity_classifier": _continuity(
            result.classifier_arc, segments, net, CFG["metrics"]["continuity_max_hops"]),
        "route_continuity_hmm": _continuity(
            result.hmm_arc, segments, net, CFG["metrics"]["continuity_max_hops"]),
        "candidate_miss_rate": miss,
        "classifier_route_precision": classifier_precision,
        "classifier_route_recall": classifier_recall,
        "classifier_route_f1": classifier_f1,
        "hmm_route_precision": hmm_precision,
        "hmm_route_recall": hmm_recall,
        "hmm_route_f1": hmm_f1,
        # Backwards-compatible aliases: route_* continues to describe the final HMM route.
        "route_precision": hmm_precision,
        "route_recall": hmm_recall,
        "route_f1": hmm_f1,
        "n_labelled_points": int((ref >= 0).sum()),
        "n_points": int(len(ref)),
    }
