"""Build candidate tables + per-point label tables for the train and student-test partitions.  Owner: P1

Outputs (data/processed/, gitignored — regenerate any time):
    candidates_{part}.parquet    schema.py candidate table (+ P2 features once features.py exists) + label
    points_{part}.parquet        one row per GPS point: trip_id, point_idx, t, segment_id, x, y,
                                 ref_arc (-1 = unlabelled), ref_dist_m, n_candidates, ref_in_candidates
Outputs (results/data/, committed — evidence for the report):
    build_stats_{part}.csv       per-trip quality stats (label coverage, miss rate, heading agreement, timings)

Run:
    python -m scripts.build_dataset                       # all 90 trips (train + test)
    python -m scripts.build_dataset --trips 00000046      # quick check on one trip
    python -m scripts.build_dataset --k 10 --radius 80    # override candidate settings

Uses ONLY dataset/train and dataset/test. Never touches evaluator trips.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap import schema
from pathsnap.candidates import generate_candidates, segment_track
from pathsnap.config import CFG
from pathsnap.geo import angle_diff_deg, bearing_deg
from pathsnap.io import list_trips, load_trip
from pathsnap.labels import align_points_to_route
from pathsnap.network import RoadNetwork

try:  # P2's feature module — optional until implemented
    from pathsnap.features import add_features
except ImportError:  # pragma: no cover
    add_features = None


def process_trip(part_dir: str, trip_id: str, k: int, radius_m: float, split_s: float, max_dist_m: float):
    t0 = time.perf_counter()
    trip = load_trip(part_dir, trip_id, with_route=True)
    net = RoadNetwork(trip)
    xy = net.project_track(trip.track)
    t_net = time.perf_counter() - t0

    t1 = time.perf_counter()
    ref, ref_d = align_points_to_route(net, xy, trip.route, max_dist_m, return_dist=True)
    t_lab = time.perf_counter() - t1

    t2 = time.perf_counter()
    cands = generate_candidates(trip, net, k, radius_m, ref, split_s=split_s, track_xy=xy)
    t_cand = time.perf_counter() - t2

    feats_done = False
    if add_features is not None:
        try:
            cands = add_features(cands, xy, trip.track[:, 2], net)
            feats_done = True
        except NotImplementedError:
            pass

    n = len(xy)
    n_cand = np.bincount(cands["point_idx"], minlength=n)
    hit_pts = cands.loc[cands[schema.LABEL_COL] == 1, "point_idx"].to_numpy()
    ref_in = np.zeros(n, dtype=bool)
    ref_in[hit_pts] = True
    seg = segment_track(trip.track[:, 2], split_s)

    points = pd.DataFrame({
        "trip_id": trip_id, "point_idx": np.arange(n, dtype=np.int32), "t": trip.track[:, 2],
        "segment_id": seg.astype(np.int32), "x": xy[:, 0], "y": xy[:, 1],
        "ref_arc": ref.astype(np.int32), "ref_dist_m": ref_d.astype(np.float32),
        "n_candidates": n_cand.astype(np.int16), "ref_in_candidates": ref_in,
    })

    # Label sanity check: while moving (> 3 m/s), GPS heading should agree with labelled arc direction.
    # Heading is measured over a +-3 s window: single 1 s steps are dominated by GPS noise (~5 m).
    dt = np.diff(trip.track[:, 2])
    w = 3
    head_ok = np.array([])
    if n > 2 * w:
        i = np.arange(w, n - w)
        d = xy[i + w] - xy[i - w]
        speed = np.hypot(d[:, 0], d[:, 1]) / np.maximum(trip.track[i + w, 2] - trip.track[i - w, 2], 1e-9)
        moving = (speed > 3.0) & (ref[i] >= 0)
        if moving.any():
            hb = bearing_deg(0.0, 0.0, d[moving, 0], d[moving, 1])
            head_ok = angle_diff_deg(hb, net.arc_bearing[ref[i][moving]]) < 45

    labelled = ref >= 0
    stats = dict(
        trip_id=trip_id, n_points=n, n_nodes=net.n_nodes, n_arcs=net.n_arcs, n_route_arcs=len(trip.route),
        duration_s=float(trip.track[-1, 2] - trip.track[0, 2]), max_gap_s=float(dt.max()) if len(dt) else 0.0,
        n_segments=int(seg.max() + 1),
        oneway_share=float((net.reverse_arc == -1).mean()),
        route_contiguity=float(np.mean(net.arcs[trip.route[:-1], 1] == net.arcs[trip.route[1:], 0])) if len(trip.route) > 1 else np.nan,
        labelled_share=float(labelled.mean()),
        ref_dist_median_m=float(np.median(ref_d[labelled])) if labelled.any() else np.nan,
        ref_dist_p95_m=float(np.percentile(ref_d[labelled], 95)) if labelled.any() else np.nan,
        route_arcs_with_points=float(np.isin(trip.route, ref[labelled]).mean()),
        heading_agreement=float(head_ok.mean()) if len(head_ok) else np.nan,
        points_without_candidates=int((n_cand == 0).sum()),
        mean_candidates=float(n_cand[n_cand > 0].mean()) if (n_cand > 0).any() else 0.0,
        candidate_miss_rate=float(1 - ref_in[labelled].mean()) if labelled.any() else np.nan,
        positive_share=float((cands[schema.LABEL_COL] == 1).mean()),
        t_network_s=t_net, t_labels_s=t_lab, t_candidates_s=t_cand, features=feats_done,
    )
    return cands, points, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["train", "test"], choices=["train", "test"])
    ap.add_argument("--trips", nargs="*", default=None, help="restrict to these trip ids")
    ap.add_argument("--k", type=int, default=CFG["candidates"]["k"])
    ap.add_argument("--radius", type=float, default=CFG["candidates"]["radius_m"])
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--suffix", default="", help="output name suffix, e.g. _k10_r80 for sweeps")
    args = ap.parse_args()

    proc = Path(CFG["paths"]["processed_dir"])
    res = Path(CFG["paths"]["results_dir"]) / "data"
    proc.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    split_s, max_d = CFG["gaps"]["split_s"], CFG["labels"]["max_dist_m"]

    for part in args.parts:
        part_dir = CFG["paths"][f"{part}_dir"]
        trips = list_trips(part_dir)
        if args.trips:
            trips = [t for t in trips if t in set(args.trips)]
        if not trips:
            continue
        t0 = time.perf_counter()
        out = Parallel(n_jobs=args.jobs, verbose=0)(
            delayed(process_trip)(part_dir, t, args.k, args.radius, split_s, max_d) for t in trips)
        stats = pd.DataFrame([o[2] for o in out])
        # Rule-based exclusion (no hardcoded ids): trips whose GPS cannot be matched to their own
        # reference route for most points (systematic offset) would only inject label noise.
        stats["excluded"] = stats["labelled_share"] < CFG["labels"]["min_labelled_share"]
        keep = ~stats["excluded"].to_numpy()
        for tid in stats.loc[~keep, "trip_id"]:
            print(f"   excluding trip {tid}: labelled share "
                  f"{stats.loc[stats.trip_id == tid, 'labelled_share'].item():.3f} < {CFG['labels']['min_labelled_share']}")
        cands = pd.concat([o[0] for o, k in zip(out, keep) if k], ignore_index=True)
        points = pd.concat([o[1] for o, k in zip(out, keep) if k], ignore_index=True)

        sfx = args.suffix if not args.trips else args.suffix + "_subset"
        cands.to_parquet(proc / f"candidates_{part}{sfx}.parquet", index=False)
        points.to_parquet(proc / f"points_{part}{sfx}.parquet", index=False)
        stats.to_csv(res / f"build_stats_{part}{sfx}.csv", index=False)

        lab = points["ref_arc"] >= 0
        print(f"[{part}] {int(keep.sum())}/{len(trips)} trips kept, {len(points):,} points, {len(cands):,} candidate rows "
              f"(k={args.k}, r={args.radius} m) in {time.perf_counter() - t0:.1f}s")
        print(f"   labelled points {lab.mean():.4f} | candidate miss rate {1 - points.loc[lab, 'ref_in_candidates'].mean():.4f} "
              f"| points w/o candidates {(points['n_candidates'] == 0).mean():.4f} "
              f"| positives {(cands[schema.LABEL_COL] == 1).mean():.3f} | features: {bool(stats['features'].all())}")


if __name__ == "__main__":
    main()
