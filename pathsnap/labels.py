"""Derive a reference arc for EVERY GPS point from the .route arc sequence.  Owner: P1

Problem: .route lists the arcs travelled IN ORDER, but has far fewer rows than the track
(e.g. 525 arcs vs 1315 points) and carries no timestamps, so there is no given per-point label.

Method: monotone alignment by dynamic programming.
    Assign each point i a route position r_i such that r_0 <= r_1 <= ... <= r_{N-1}
    (the vehicle never goes backwards along its own route) while minimising
        sum_i  min(dist(point_i, route_arc[r_i]), cap)
    Recurrence:  A[i, r] = c[i, r] + min_{r' <= r} A[i-1, r']      (prefix minimum, O(N*R))
    Skipping route arcs is free (at 1 Hz a car can pass several short arcs between two fixes).
    The cap stops a single outlier fix from dragging the alignment.
    Points whose aligned arc is farther than max_dist_m are marked UNLABELLED (-1).

Why monotone and not just "nearest route arc"? Routes revisit places (loops, out-and-back on
the same street in both directions); plain nearest-arc would mix up passes and directions.

Known limitations (state in the report):
  * Labels are derived, not observed; they inherit any error in the reference route
    (route contiguity is 98-100%, not always 100%).
  * A fix right at a junction is equally close to both arcs; the tie is broken toward the later
    route position, so "near-miss" errors at junctions are partly label noise.
Used ONLY for training data and server-side evaluation — never inside prediction.
"""
from __future__ import annotations

import numpy as np

from . import schema
from .network import RoadNetwork


def align_points_to_route(net: RoadNetwork, track_xy: np.ndarray, route: np.ndarray, max_dist_m: float,
                          cap_m: float | None = None, return_dist: bool = False):
    """Returns (N,) int array: reference arc_id per GPS point, or -1 (schema.UNLABELLED).
    With return_dist=True also returns (N,) distance (m) from each point to its aligned arc."""
    route = np.asarray(route, dtype=np.int64)
    N, R = len(track_xy), len(route)
    if N == 0 or R == 0:
        out = np.full(N, schema.UNLABELLED, dtype=np.int64)
        return (out, np.full(N, np.inf)) if return_dist else out
    cap = 2.0 * max_dist_m if cap_m is None else cap_m

    A, B = net.arc_a[route], net.arc_b[route]  # (R, 2) each
    bp_dtype = np.int16 if R < np.iinfo(np.int16).max else np.int32
    back = np.empty((N, R), dtype=bp_dtype)  # back[i, r] = best r' for point i-1 given r_i = r
    ar = np.arange(R)

    def row_cost(i):
        d, _ = _dist_to_route(track_xy[i], A, B)
        return np.minimum(d, cap)

    acc = row_cost(0)
    for i in range(1, N):
        run_min = np.minimum.accumulate(acc)
        # index of the (latest) position attaining the running minimum
        back[i] = np.maximum.accumulate(np.where(acc == run_min, ar, 0))
        acc = row_cost(i) + run_min

    # backtrack
    r_seq = np.empty(N, dtype=np.int64)
    r_seq[-1] = int(np.argmin(acc))
    for i in range(N - 1, 0, -1):
        r_seq[i - 1] = back[i, r_seq[i]]

    arcs = route[r_seq]
    d, _ = net.arc_distance(track_xy[:, 0], track_xy[:, 1], arcs)
    labels = np.where(d <= max_dist_m, arcs, schema.UNLABELLED).astype(np.int64)
    return (labels, d) if return_dist else labels


def _dist_to_route(p, A, B):
    dx, dy = B[:, 0] - A[:, 0], B[:, 1] - A[:, 1]
    len2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        u = ((p[0] - A[:, 0]) * dx + (p[1] - A[:, 1]) * dy) / len2
    u = np.where(len2 > 0, np.clip(u, 0, 1), 0)
    return np.hypot(p[0] - (A[:, 0] + u * dx), p[1] - (A[:, 1] + u * dy)), u
