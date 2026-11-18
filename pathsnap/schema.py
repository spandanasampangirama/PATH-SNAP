"""Shared column names for the CANDIDATE TABLE — the central contract between all modules.

FROZEN after Phase 0. Adding a feature column is fine (append to FEATURE_COLS via P2);
renaming/removing anything here needs agreement from all four members.

Candidate table = pandas DataFrame, one row per (trip_id, point_idx, arc_id):

    trip_id     str     e.g. "00000046"
    point_idx   int     0-based row index into the trip's .track
    t           float   timestamp (s) of that GPS point
    segment_id  int     HMM segment within the trip (track split at time gaps > gaps.split_s)
    arc_id      int     0-based row index into the trip's .arcs  (the candidate road segment)
    dist_m      float   perpendicular distance GPS point -> arc (metres, projected)
    ...features (FEATURE_COLS)
    label       int     1 if arc_id == reference arc for this point, 0 otherwise,
                        -1 if the point is unlabelled (no route / too far). Never present at inference.

Per-point labels (output of labels.py) = np.ndarray[int] of length n_points, value = arc_id or -1.
"""

KEY_COLS = ["trip_id", "point_idx", "t", "segment_id", "arc_id"]
LABEL_COL = "label"
GROUP_COL = "trip_id"  # GroupKFold groups — never split a trip across folds

# Base geometric columns produced by candidates.py (P1).
CANDIDATE_BASE_COLS = ["dist_m", "proj_frac", "arc_len_m", "arc_bearing_deg", "rank"]  # rank: 0 = nearest

# Per-point table (data/processed/points_{part}.parquet), produced by scripts/build_dataset.py (P1):
#   trip_id, point_idx, t, segment_id, x, y (projected m), ref_arc (-1 = unlabelled),
#   ref_dist_m, n_candidates, ref_in_candidates
# Use it for top-1 accuracy / miss rate (points with no candidate rows still appear here).
POINT_COLS = ["trip_id", "point_idx", "t", "segment_id", "x", "y", "ref_arc", "ref_dist_m",
              "n_candidates", "ref_in_candidates"]

# Model inputs produced by features.py (P2). Single source of truth lives in features.FEATURE_COLS.
from .features import FEATURE_COLS  # noqa: E402

UNLABELLED = -1
NO_CANDIDATE = -1
