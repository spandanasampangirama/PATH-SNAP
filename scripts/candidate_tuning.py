"""Tune candidate generation (k, radius) end to end on the TRAIN partition only.  Owner: P1

candidate_sweep.py measures only the miss rate (is the right arc on the shortlist?). A longer
shortlist lowers the miss rate but adds harder negatives for the classifier and k^2 transitions
for the HMM, so the miss rate alone cannot pick k. This script measures what each setting does
to the final answer:

  for each (k, radius):
    1. build train candidates + P2 features (same code as build_dataset.py)
    2. 5-fold GroupKFold by trip: fit logistic regression on 4 folds, score the held-out fold
    3. per held-out trip: classifier top-1 and HMM/Viterbi top-1 (default hmm config)
    4. record miss rate, table size, transitions per HMM step, and timings

Every number comes from training trips scored by a model that never saw them. The student-test
and evaluator trips are not touched. Logistic regression is used because it is fast and is the
committed baseline; the comparison is between candidate settings, not between models.

Writes results/data/candidate_tuning.csv and results/plots/p1_candidate_tuning.png.
Run: python -m scripts.candidate_tuning            (about 2-4 min on a laptop)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from pathsnap import schema
from pathsnap.config import CFG
from pathsnap.hmm import viterbi_decode
from pathsnap.io import load_trip, list_trips
from pathsnap.metrics import attach_predictions, top1_accuracy, top1_from_candidates
from pathsnap.models import build_model
from pathsnap.network import RoadNetwork
from pathsnap.routing import Router
from scripts.build_dataset import process_trip

DEFAULT_GRID = [(4, 60), (6, 60), (8, 45), (8, 60), (12, 60), (16, 60)]


def build(part_dir, trips, k, r):
    split_s, max_d = CFG["gaps"]["split_s"], CFG["labels"]["max_dist_m"]
    t0 = time.perf_counter()
    out = Parallel(n_jobs=-1)(delayed(process_trip)(part_dir, t, k, r, split_s, max_d) for t in trips)
    build_s = time.perf_counter() - t0
    keep = [o for o in out if o[2]["labelled_share"] >= CFG["labels"]["min_labelled_share"]]
    cands = pd.concat([o[0] for o in keep], ignore_index=True)
    points = pd.concat([o[1] for o in keep], ignore_index=True)
    return cands, points, build_s


def hmm_trip(part_dir, tid, cands, hmm_cfg):
    trip = load_trip(part_dir, tid, with_route=False)
    net = RoadNetwork(trip)
    xy = net.project_track(trip.track)
    t0 = time.perf_counter()
    decoded = viterbi_decode(cands, "prob", Router(net), xy, hmm_cfg)
    return tid, decoded, time.perf_counter() - t0


def transitions_per_step(cands):
    """Mean K_prev * K_curr over consecutive points of the same segment (HMM work per step)."""
    n = cands.groupby(["trip_id", "point_idx"], sort=True).agg(n=("arc_id", "size"), seg=("segment_id", "first"))
    n = n.reset_index()
    same = (n.trip_id.values[1:] == n.trip_id.values[:-1]) & (n.seg.values[1:] == n.seg.values[:-1]) \
        & (n.point_idx.values[1:] == n.point_idx.values[:-1] + 1)
    return float((n.n.values[1:] * n.n.values[:-1])[same].mean())


def evaluate(part_dir, k, r, seed, hmm_cfg):
    cands, points, build_s = build(part_dir, list_trips(part_dir), k, r)
    feats, lab = schema.FEATURE_COLS, schema.LABEL_COL
    X, y, groups = cands[feats].to_numpy(np.float32), cands[lab].to_numpy(), cands["trip_id"].to_numpy()
    fit_mask = y != -1
    prob = np.zeros(len(cands))
    t0 = time.perf_counter()
    for tr, va in GroupKFold(n_splits=CFG["cv"]["n_splits"]).split(X, y, groups):
        tr = tr[fit_mask[tr]]
        model = build_model("logreg", None, seed).fit(X[tr], y[tr])
        prob[va] = model.predict_proba(X[va])[:, 1]
    cv_s = time.perf_counter() - t0
    cands = cands.assign(prob=prob)

    cls = attach_predictions(points, top1_from_candidates(cands, "prob"))
    labelled = points["ref_arc"].to_numpy() >= 0
    t0 = time.perf_counter()
    dec = Parallel(n_jobs=-1)(delayed(hmm_trip)(part_dir, tid, g, hmm_cfg)
                              for tid, g in cands.groupby("trip_id", sort=False))
    hmm_wall_s = time.perf_counter() - t0
    hmm_pred = np.full(len(points), -1, dtype=np.int64)
    pos = {tid: np.flatnonzero(points["trip_id"].to_numpy() == tid) for tid, _, _ in dec}
    for tid, decoded, _ in dec:
        idx = pos[tid]
        hmm_pred[idx] = decoded[points["point_idx"].to_numpy()[idx]]
    ref = points["ref_arc"].to_numpy()
    return dict(
        k=k, radius_m=r, n_trips=points.trip_id.nunique(), n_rows=len(cands),
        rows_per_point=len(cands) / max(1, (points["n_candidates"] > 0).sum()),
        candidate_miss_rate=float(1 - points.loc[labelled, "ref_in_candidates"].mean()),
        no_candidate_share=float((points["n_candidates"] == 0).mean()),
        positive_share=float((y == 1).mean()),
        transitions_per_step=transitions_per_step(cands),
        cv_top1_classifier=top1_accuracy(cls["pred_arc"].to_numpy(), ref),
        cv_top1_hmm=top1_accuracy(hmm_pred, ref),
        build_s=build_s, cv_fit_s=cv_s,
        hmm_cpu_s=float(sum(d[2] for d in dec)), hmm_wall_s=hmm_wall_s,
    )


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[df.radius_m == 60].sort_values("k")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(d.k, 100 * d.candidate_miss_rate, "o-", color="#2E7D32")
    ax[0].set(xlabel="k (candidates per point, radius 60 m)", ylabel="candidate miss rate (%)",
              title="Right arc missing from the shortlist")
    ax[1].plot(d.k, 100 * d.cv_top1_classifier, "o-", color="#805AD5", label="classifier only")
    ax[1].plot(d.k, 100 * d.cv_top1_hmm, "o-", color="#E53E3E", label="classifier + HMM")
    ax[1].plot(d.k, 100 * (1 - d.candidate_miss_rate), "--", color="#999999", label="ceiling (1 - miss rate)")
    ax[1].set(xlabel="k (candidates per point, radius 60 m)", ylabel="top-1 accuracy (%)",
              title="5-fold GroupKFold on train trips")
    ax[1].legend(loc="center right")
    for a in ax:
        a.grid(alpha=0.3)
        a.set_xticks(d.k)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", nargs="*", default=None, help="k:radius pairs, e.g. 8:60 12:60")
    args = ap.parse_args()
    grid = [tuple(map(float, g.split(":"))) for g in args.grid] if args.grid else DEFAULT_GRID
    part_dir = CFG["paths"]["train_dir"]
    hmm_cfg = dict(CFG["hmm"])
    rows = []
    for k, r in grid:
        row = evaluate(part_dir, int(k), float(r), CFG["seed"], hmm_cfg)
        rows.append(row)
        print(f"k={row['k']:>2} r={row['radius_m']:>4.0f}  miss {row['candidate_miss_rate']:.4f}  "
              f"rows {row['n_rows']:>9,}  trans/step {row['transitions_per_step']:5.1f}  "
              f"top-1 cls {row['cv_top1_classifier']:.4f}  hmm {row['cv_top1_hmm']:.4f}  "
              f"(hmm cpu {row['hmm_cpu_s']:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    res = Path(CFG["paths"]["results_dir"])
    (res / "data").mkdir(parents=True, exist_ok=True)
    (res / "plots").mkdir(parents=True, exist_ok=True)
    df.to_csv(res / "data" / "candidate_tuning.csv", index=False)
    plot(df, res / "plots" / "p1_candidate_tuning.png")
    print(f"saved {res / 'data' / 'candidate_tuning.csv'} and {res / 'plots' / 'p1_candidate_tuning.png'}")


if __name__ == "__main__":
    main()
