"""Feature engineering on the candidate table.  Owner: P2

Input : candidate table for ONE trip (P1 columns) + that trip's projected track, timestamps and RoadNetwork.
Output: same rows with every column in schema.FEATURE_COLS added.
Uses only the GPS track and the road network, never the reference route, so the API runs this exact
function at inference time. Scaling is done inside the sklearn Pipeline (models.py), fitted on training
folds only, never here.

Feature groups and why we use them:
  Distance      dist_m, dist_minus_min_m, dist_ratio_min, rank, n_candidates
                -> GPS noise is ~5 m, so the true road is usually (not always) the closest one; *relative*
                   distance to the best candidate matters more than absolute distance in dense areas.
  Direction     heading_diff_deg, heading_align, speed_mps
                -> two-way streets are two arcs with identical geometry; only travel direction separates them.
                   Heading is measured over a +-3 s window (1 s steps are dominated by noise), and
                   heading_align = cos(diff) * min(speed/5, 1) fades out when the vehicle is (nearly) stopped
                   and heading is meaningless.
  Consistency   nbr_dist_mean_m, nbr_dist_max_m
                -> the correct road stays close to the NEIGHBOURING fixes too (+-3 points), whereas a
                   crossing road is close only around the intersection.
  Road context  arc_len_m, endpoint_dist_m, is_oneway, to_out_degree
                -> near a junction (endpoint_dist small) several arcs are equally close; short arcs and
                   high-degree nodes make the choice harder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG
from .geo import angle_diff_deg, bearing_deg
from .network import RoadNetwork

# Constants come from configs/default.yaml (CONTRACTS §1); see the `features:` block there.
HEADING_WINDOW = int(CFG["features"]["heading_window_pts"])  # points on each side
_NW = int(CFG["features"]["neighbour_window_pts"])
NEIGHBOUR_OFFSETS = tuple(o for o in range(-_NW, _NW + 1) if o != 0)
HEADING_FULL_SPEED = float(CFG["features"]["heading_full_speed_mps"])

FEATURE_COLS = [
    "dist_m", "dist_minus_min_m", "dist_ratio_min", "rank", "n_candidates",
    "heading_diff_deg", "heading_align", "speed_mps",
    "nbr_dist_mean_m", "nbr_dist_max_m",
    "arc_len_m", "endpoint_dist_m", "is_oneway", "to_out_degree",
]


def point_motion(track_xy: np.ndarray, t: np.ndarray, w: int = HEADING_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Per-point (speed_mps, heading_deg) from the displacement between points i-w and i+w (clipped at ends)."""
    n = len(track_xy)
    idx = np.arange(n)
    lo, hi = np.clip(idx - w, 0, n - 1), np.clip(idx + w, 0, n - 1)
    d = track_xy[hi] - track_xy[lo]
    span = t[hi] - t[lo]
    dist = np.hypot(d[:, 0], d[:, 1])
    speed = np.divide(dist, span, out=np.zeros(n), where=span > 0)
    heading = bearing_deg(0.0, 0.0, d[:, 0], d[:, 1])
    return speed, heading


def add_features(cands: pd.DataFrame, track_xy: np.ndarray, t: np.ndarray, net: RoadNetwork) -> pd.DataFrame:
    df = cands.copy()
    if df.empty:
        for c in FEATURE_COLS:
            df[c] = pd.Series(dtype=np.float32)
        return df
    n = len(track_xy)
    pi = df["point_idx"].to_numpy()
    ai = df["arc_id"].to_numpy()
    dist = df["dist_m"].to_numpy(dtype=np.float64)

    # distance relative to the best candidate of the same point
    g = df.groupby("point_idx")["dist_m"]
    dmin = g.transform("min").to_numpy(dtype=np.float64)
    df["dist_minus_min_m"] = dist - dmin
    df["dist_ratio_min"] = (dist + 1.0) / (dmin + 1.0)
    df["n_candidates"] = g.transform("size").to_numpy()

    # direction
    speed, heading = point_motion(track_xy, t)
    hd = angle_diff_deg(heading[pi], net.arc_bearing[ai])
    df["speed_mps"] = speed[pi]
    df["heading_diff_deg"] = hd
    df["heading_align"] = np.cos(np.radians(hd)) * np.minimum(speed[pi] / HEADING_FULL_SPEED, 1.0)

    # consistency with neighbouring fixes
    acc = np.zeros(len(df))
    mx = np.zeros(len(df))
    for o in NEIGHBOUR_OFFSETS:
        j = np.clip(pi + o, 0, n - 1)
        dj, _ = net.arc_distance(track_xy[j, 0], track_xy[j, 1], ai)
        acc += dj
        mx = np.maximum(mx, dj)
    df["nbr_dist_mean_m"] = acc / len(NEIGHBOUR_OFFSETS)
    df["nbr_dist_max_m"] = mx

    # road context
    frac = df["proj_frac"].to_numpy(dtype=np.float64)
    df["endpoint_dist_m"] = np.minimum(frac, 1.0 - frac) * net.arc_len_m[ai]
    df["is_oneway"] = (net.reverse_arc[ai] == -1).astype(np.int8)
    df["to_out_degree"] = net.out_degree[net.arcs[ai, 1]]

    for c in FEATURE_COLS:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df
