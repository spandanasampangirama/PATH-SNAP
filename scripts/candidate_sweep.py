"""Choose candidate-generation settings (k, radius) on the TRAIN partition only.  Owner: P1

Trade-off: a larger k / radius lowers the candidate miss rate (reference arc absent -> that point
can never be matched correctly), but adds negatives (heavier imbalance, slower model + Viterbi).
Writes results/data/candidate_sweep.csv.

Run: python -m scripts.candidate_sweep
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap.config import CFG
from pathsnap.io import list_trips, load_trip
from pathsnap.labels import align_points_to_route
from pathsnap.network import RoadNetwork

KS = [2, 4, 6, 8, 12, 16]
RADII = [30, 45, 60, 80, 100]


def trip_sweep(part_dir, tid):
    trip = load_trip(part_dir, tid, with_route=True)
    net = RoadNetwork(trip)
    xy = net.project_track(trip.track)
    ref = align_points_to_route(net, xy, trip.route, CFG["labels"]["max_dist_m"])
    if (ref >= 0).mean() < CFG["labels"]["min_labelled_share"]:
        return []
    pi, ai, dist, _, rank = net.query_arcs_bulk(xy, max(RADII), max(KS))
    is_ref = ref[pi] == ai
    lab = ref >= 0
    rows = []
    for k, r in itertools.product(KS, RADII):
        m = (rank < k) & (dist <= r)
        hit = np.zeros(len(xy), bool)
        hit[pi[m & is_ref]] = True
        rows.append(dict(trip_id=tid, k=k, radius_m=r, n_labelled=int(lab.sum()), n_hit=int(hit[lab].sum()),
                         n_rows=int(m.sum()), n_pts_with_cand=len(np.unique(pi[m]))))
    return rows


def main():
    part_dir = CFG["paths"]["train_dir"]
    out = Parallel(n_jobs=-1)(delayed(trip_sweep)(part_dir, t) for t in list_trips(part_dir))
    df = pd.DataFrame([r for rows in out for r in rows])
    g = df.groupby(["k", "radius_m"]).sum(numeric_only=True)
    res = pd.DataFrame({
        "candidate_miss_rate": 1 - g.n_hit / g.n_labelled,
        "rows_per_point": g.n_rows / g.n_pts_with_cand,
        "total_rows": g.n_rows,
    }).reset_index()
    path = Path(CFG["paths"]["results_dir"]) / "data" / "candidate_sweep.csv"
    res.to_csv(path, index=False)
    print(res.pivot(index="k", columns="radius_m", values="candidate_miss_rate").round(4).to_string())
    print()
    print(res.pivot(index="k", columns="radius_m", values="rows_per_point").round(2).to_string())
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
