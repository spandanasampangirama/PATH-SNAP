"""Metric definitions — ONE implementation used by training, HMM evaluation and the API.  Owner: P2

Definitions (write these exact definitions into the report):
  candidate-level     accuracy, precision, recall, F1 (positive class), macro-F1 (mean F1 over classes
                      {0,1}), log loss, confusion matrix — on candidate rows with label != -1, threshold 0.5.
  top1_accuracy       per labelled GPS point: highest-probability candidate == reference arc.
                      Points whose reference arc is not among the candidates, or that have no candidates,
                      count as WRONG (denominator = all labelled points, from the points table).
  candidate_miss_rate share of labelled points whose reference arc is absent from their candidates.
  route_continuity    share of consecutive point pairs (same segment, both matched) whose predicted arcs
                      are equal or reachable within `max_hops` directed arc hops.
  route_overlap       sequence level: precision / recall / F1 of the SET of predicted arcs vs reference arcs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score)

from .network import RoadNetwork


def candidate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def top1_from_candidates(cands: pd.DataFrame, prob_col: str = "prob") -> pd.DataFrame:
    """Highest-probability candidate per point -> DataFrame[trip_id, point_idx, arc_id, prob]."""
    best = cands.sort_values(["trip_id", "point_idx", prob_col], ascending=[True, True, False], kind="stable")
    best = best.drop_duplicates(["trip_id", "point_idx"])
    return best[["trip_id", "point_idx", "arc_id", prob_col]].reset_index(drop=True)


def attach_predictions(points: pd.DataFrame, top1: pd.DataFrame, col: str = "pred_arc") -> pd.DataFrame:
    """Left-join per-point predictions onto the points table; points without candidates get -1."""
    out = points.merge(top1[["trip_id", "point_idx", "arc_id"]].rename(columns={"arc_id": col}),
                       on=["trip_id", "point_idx"], how="left")
    out[col] = out[col].fillna(-1).astype(np.int64)
    return out


def top1_accuracy(point_pred_arc: np.ndarray, point_ref_arc: np.ndarray) -> float:
    """Both (N,) arrays of arc_id; points with ref == -1 are ignored."""
    pred, ref = np.asarray(point_pred_arc), np.asarray(point_ref_arc)
    lab = ref >= 0
    return float(np.mean(pred[lab] == ref[lab])) if lab.any() else float("nan")


def candidate_miss_rate(cands: pd.DataFrame, point_ref_arc: np.ndarray) -> float:
    """For ONE trip: share of labelled points whose reference arc is not among their candidate rows."""
    ref = np.asarray(point_ref_arc)
    lab = ref >= 0
    if not lab.any():
        return float("nan")
    pi, ai = cands["point_idx"].to_numpy(), cands["arc_id"].to_numpy()
    hit = np.zeros(len(ref), dtype=bool)
    hit[pi[ref[pi] == ai]] = True
    return float(1.0 - hit[lab].mean())


def _reachable(net: RoadNetwork, a: int, b: int, max_hops: int, cache: dict) -> bool:
    if a == b or net.arcs[a, 1] == net.arcs[b, 0]:
        return True
    key = (a, max_hops)
    if key not in cache:
        seen, frontier = set(), {a}
        for _ in range(max_hops):
            nxt = set()
            for x in frontier:
                for s in net.successor_arcs(x):
                    s = int(s)
                    if s not in seen:
                        seen.add(s)
                        nxt.add(s)
            frontier = nxt
        cache[key] = seen
    return b in cache[key]


def route_continuity(point_pred_arc: np.ndarray, net: RoadNetwork, max_hops: int,
                     segment_id: np.ndarray | None = None) -> float:
    pred = np.asarray(point_pred_arc)
    if len(pred) < 2:
        return float("nan")
    ok = (pred[:-1] >= 0) & (pred[1:] >= 0)
    if segment_id is not None:
        seg = np.asarray(segment_id)
        ok &= seg[:-1] == seg[1:]
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return float("nan")
    cache: dict = {}
    good = sum(_reachable(net, int(pred[i]), int(pred[i + 1]), max_hops, cache) for i in idx)
    return good / len(idx)


def collapse_route(point_arc: np.ndarray) -> np.ndarray:
    """Per-point arcs -> route sequence (drop -1, merge consecutive repeats)."""
    a = np.asarray(point_arc)
    a = a[a >= 0]
    if len(a) == 0:
        return a
    return a[np.r_[True, a[1:] != a[:-1]]]


def route_overlap(pred_route: np.ndarray, ref_route: np.ndarray) -> dict:
    """Arc-set precision / recall / F1 between predicted and reference arc sequences."""
    p, r = set(np.asarray(pred_route).tolist()), set(np.asarray(ref_route).tolist())
    inter = len(p & r)
    prec = inter / len(p) if p else 0.0
    rec = inter / len(r) if r else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}
