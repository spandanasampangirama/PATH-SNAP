"""Log-space HMM/Viterbi route reconstruction over per-observation arc candidates (P3).

Each step's (K, J) transition route distances are computed once, used for the reachability
check that decides chunk restarts, and reused by the decoder. The decoder recurrence is
vectorised over candidate pairs; ties still go to the first previous state.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .routing import Router


def _log_emissions(rows: pd.DataFrame, prob_col: str | None, cfg: dict) -> np.ndarray:
    mode = str(cfg.get("emission", "distance")).lower()
    sigma = max(float(cfg.get("sigma_m", 10.0)), 1e-6)
    d = rows["dist_m"].to_numpy(dtype=float)
    log_d = -0.5 * np.square(d / sigma) - np.log(sigma * np.sqrt(2.0 * np.pi))
    if prob_col and prob_col in rows.columns and mode in {"classifier", "product"}:
        p = np.clip(rows[prob_col].to_numpy(dtype=float), 1e-12, 1.0)
        return np.log(p) + log_d if mode == "product" else np.log(p)
    return log_d


def _step_bound(cfg: dict, straight: float) -> float:
    factor, extra = float(cfg.get("max_route_factor", 4.0)), float(cfg.get("max_route_extra_m", 50.0))
    return max(0.0, factor * straight + extra)


def _transitions(router, prev_arcs, prev_fracs, curr_arcs, curr_fracs, bound) -> np.ndarray:
    """(K, J) route distances; falls back to pairwise calls for routers without a matrix API."""
    if hasattr(router, "transition_matrix"):
        return router.transition_matrix(prev_arcs, prev_fracs, curr_arcs, curr_fracs, bound)
    out = np.full((len(prev_arcs), len(curr_arcs)), np.inf)
    for k in range(len(prev_arcs)):
        for j in range(len(curr_arcs)):
            out[k, j] = router.route_distance(int(prev_arcs[k]), prev_fracs[k],
                                              int(curr_arcs[j]), curr_fracs[j], bound)
    return out


def _decode_chunk(groups, points, router, cfg, prob_col, transitions=None):
    """Viterbi over one chunk. ``transitions[i-1]`` is the (K, J) route-distance matrix
    from point i-1 to point i; it is computed here when not supplied."""
    states = [g["arc_id"].to_numpy(dtype=np.int64) for g in groups]
    fracs = [g["proj_frac"].to_numpy(dtype=float) for g in groups]
    emissions = [_log_emissions(g, prob_col, cfg) for g in groups]
    scores = emissions[0].copy()
    backs = [np.full(len(states[0]), -1, dtype=np.int32)]
    beta = max(float(cfg.get("beta_m", 5.0)), 1e-6)
    xy = cfg["_track_xy"]
    for i in range(1, len(points)):
        straight = float(np.linalg.norm(xy[points[i]] - xy[points[i - 1]]))
        if transitions is not None:
            rd = transitions[i - 1]
        else:
            rd = _transitions(router, states[i - 1], fracs[i - 1], states[i], fracs[i],
                              _step_bound(cfg, straight))
        ok = np.isfinite(rd) & np.isfinite(scores)[:, None]
        with np.errstate(invalid="ignore"):
            value = np.where(ok, scores[:, None] - np.abs(rd - straight) / beta, -np.inf)
        back = np.argmax(value, axis=0).astype(np.int32)
        best = value[back, np.arange(value.shape[1])]
        back[~np.isfinite(best)] = -1
        next_scores = best + emissions[i]
        if not np.isfinite(next_scores).any():
            raise RuntimeError("Internal Viterbi error: chunk crosses a dead-end transition")
        scores = next_scores
        backs.append(back)
    state = int(np.argmax(scores))
    decoded = np.empty(len(points), dtype=np.int64)
    for i in range(len(points) - 1, -1, -1):
        decoded[i] = states[i][state]
        if i:
            state = int(backs[i][state])
    return decoded


def viterbi_decode(cands: pd.DataFrame, prob_col: str, router: Router,
                   track_xy: np.ndarray, hmm_cfg: dict) -> np.ndarray:
    """Return an arc id for every GPS point; -1 for observations with no candidate.

    Decoding restarts at segment boundaries, missing-candidate observations, and when no
    transition is reachable from any currently reachable Viterbi state.
    """
    xy = np.asarray(track_xy, dtype=float)
    out = np.full(len(xy), -1, dtype=np.int64)
    router.dead_end_restarts = 0
    if cands.empty or len(xy) == 0:
        return out
    required = {"point_idx", "segment_id", "arc_id", "proj_frac", "dist_m"}
    missing = required.difference(cands.columns)
    if missing:
        raise ValueError(f"candidate table missing HMM columns: {sorted(missing)}")
    if not prob_col or prob_col not in cands.columns:
        prob_col = None
    cfg = dict(hmm_cfg)
    cfg["_track_xy"] = xy
    # Same order as grouping by point_idx and stable-sorting each group by arc_id
    # (multi-key sort_values is a stable lexsort).
    work = cands.sort_values(["point_idx", "arc_id"], kind="stable")
    pidx = work["point_idx"].to_numpy(dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, pidx[1:] != pidx[:-1]])
    ends = np.r_[starts[1:], len(pidx)]
    grouped = {int(pidx[s]): work.iloc[s:e] for s, e in zip(starts, ends)}
    arcs_of = {p: g["arc_id"].to_numpy(dtype=np.int64) for p, g in grouped.items()}
    fracs_of = {p: g["proj_frac"].to_numpy(dtype=float) for p, g in grouped.items()}
    segment_of = {p: int(g["segment_id"].iloc[0]) for p, g in grouped.items()}
    ordered = sorted(p for p in grouped if 0 <= p < len(xy))
    run_points, run_groups, run_trans, run_segment, active = [], [], [], None, None

    def flush():
        if run_points:
            out[np.asarray(run_points, dtype=int)] = _decode_chunk(
                run_groups, run_points, router, cfg, prob_col, run_trans
            )

    for point in ordered:
        group = grouped[point]
        segment = segment_of[point]
        if run_points and (point != run_points[-1] + 1 or segment != run_segment):
            flush()
            run_points, run_groups, run_trans, active = [], [], [], None
        if not run_points:
            run_segment = segment
            run_points, run_groups, run_trans = [point], [group], []
            active = np.ones(len(group), dtype=bool)
            continue
        prev_point = run_points[-1]
        straight = float(np.linalg.norm(xy[point] - xy[prev_point]))
        rd = _transitions(router, arcs_of[prev_point], fracs_of[prev_point],
                          arcs_of[point], fracs_of[point], _step_bound(cfg, straight))
        reachable = (np.isfinite(rd) & active[:, None]).any(axis=0)
        if not reachable.any():
            router.dead_end_restarts += 1
            flush()
            run_points, run_groups, run_trans = [point], [group], []
            run_segment, active = segment, np.ones(len(group), dtype=bool)
        else:
            run_points.append(point)
            run_groups.append(group)
            run_trans.append(rd)
            active = reachable
    flush()
    return out
