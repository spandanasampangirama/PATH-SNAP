"""Candidate generation: for each GPS point, the k nearest arcs within radius.  Owner: P1

Output = the candidate table defined in schema.py:
    KEY_COLS (trip_id, point_idx, t, segment_id, arc_id) + CANDIDATE_BASE_COLS
    (dist_m, proj_frac, arc_len_m, arc_bearing_deg, rank) [+ label]

Convention for points with NO candidate inside the radius: they get NO rows. Use
`points_without_candidates(cands, n_points)` to list them (HMM assigns them -1). This keeps the
training table free of fake "-1 arc" rows.

Note: a two-way street is two arcs with identical geometry (a->b and b->a), so both twins always
appear together with the same dist_m. Only direction (heading feature) can separate them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema
from .io import Trip
from .network import RoadNetwork


def segment_track(t: np.ndarray, split_s: float) -> np.ndarray:
    """(N,) segment_id: increments wherever the time gap to the previous point exceeds split_s."""
    t = np.asarray(t, dtype=float)
    if len(t) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.r_[0, np.cumsum(np.diff(t) > split_s)].astype(np.int64)


def generate_candidates(trip: Trip, net: RoadNetwork, k: int, radius_m: float,
                        point_labels: np.ndarray | None = None, split_s: float = 60.0,
                        track_xy: np.ndarray | None = None) -> pd.DataFrame:
    """Build the candidate table for one trip. If point_labels is given, add the `label` column:
    1 = candidate is the reference arc, 0 = it is not, -1 = point is unlabelled."""
    if track_xy is None:
        track_xy = net.project_track(trip.track)
    t = trip.track[:, 2]
    seg = segment_track(t, split_s)

    pi, ai, dist, frac, rank = net.query_arcs_bulk(track_xy, radius_m, k)
    df = pd.DataFrame({
        "trip_id": trip.trip_id,
        "point_idx": pi.astype(np.int32),
        "t": t[pi],
        "segment_id": seg[pi].astype(np.int32),
        "arc_id": ai.astype(np.int32),
        "dist_m": dist.astype(np.float32),
        "proj_frac": frac.astype(np.float32),
        "arc_len_m": net.arc_len_m[ai].astype(np.float32),
        "arc_bearing_deg": net.arc_bearing[ai].astype(np.float32),
        "rank": rank.astype(np.int16),
    })
    if point_labels is not None:
        ref = np.asarray(point_labels)[pi]
        df[schema.LABEL_COL] = np.where(ref == schema.UNLABELLED, -1, (ref == ai).astype(np.int8)).astype(np.int8)
    return df


def points_without_candidates(cands: pd.DataFrame, n_points: int) -> np.ndarray:
    """point_idx values that have no candidate row."""
    mask = np.ones(n_points, dtype=bool)
    mask[cands["point_idx"].to_numpy()] = False
    return np.flatnonzero(mask)
