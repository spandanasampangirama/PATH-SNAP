"""Per-model diagnostics on the student-test trips, saved as small CSVs for the per-model plots.  Owner: P2

For every model in artifacts/untuned/ (the refit-on-all-train models from scripts/train_models.py):
    results/model_diagnostics/<model>_calibration.csv   reliability curve: 10 probability bins,
                                                        mean predicted vs observed share of correct arcs
    results/model_diagnostics/<model>_per_trip.csv      top-1 accuracy per student-test trip
    results/model_diagnostics/<model>_importance.csv    permutation importance of each feature
                                                        (increase in log loss when that feature is shuffled)

Permutation importance is model-agnostic, so all models are measured the same way. It uses a fixed
random sample of student-test rows (CFG training.importance_rows) and CFG seed.

Run:  python -m scripts.model_diagnostics [--models lgbm rf]   then   python -m scripts.plot_models
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pathsnap import schema
from pathsnap.config import CFG
from pathsnap.metrics import attach_predictions, candidate_metrics, top1_from_candidates

FEATS = schema.FEATURE_COLS
LABEL = schema.LABEL_COL


def calibration(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        rows.append({"bin_lo": edges[k], "bin_hi": edges[k + 1], "n": int(m.sum()),
                     "mean_pred": float(p[m].mean()) if m.any() else np.nan,
                     "observed": float(y[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def per_trip_top1(c_te: pd.DataFrame, p_te: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    c = c_te[["trip_id", "point_idx", "arc_id"]].copy()
    c["prob"] = prob
    pts = attach_predictions(p_te, top1_from_candidates(c, "prob"))
    nearest = attach_predictions(p_te, c_te.loc[c_te["rank"] == 0, ["trip_id", "point_idx", "arc_id"]], "near")
    pts["near"] = nearest["near"].to_numpy()
    lab = pts[pts.ref_arc >= 0]
    g = lab.groupby("trip_id")
    return pd.DataFrame({"n_points": g.size(),
                         "top1_accuracy": g.apply(lambda d: float((d.pred_arc == d.ref_arc).mean())),
                         "nearest_arc_top1": g.apply(lambda d: float((d.near == d.ref_arc).mean()))}).reset_index()


def permutation_importance(model, X: np.ndarray, y: np.ndarray, seed: int, repeats: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = candidate_metrics(y, model.predict_proba(X)[:, 1])["log_loss"]
    rows = []
    for j, f in enumerate(FEATS):
        deltas = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            deltas.append(candidate_metrics(y, model.predict_proba(Xp)[:, 1])["log_loss"] - base)
        rows.append({"feature": f, "log_loss_increase": float(np.mean(deltas)),
                     "std": float(np.std(deltas)), "base_log_loss": base})
    return pd.DataFrame(rows).sort_values("log_loss_increase", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()
    seed = CFG["seed"]
    tc = CFG["training"]
    art = Path(CFG["paths"]["artifacts_dir"]) / "untuned"
    out = Path(CFG["paths"]["results_dir"]) / "model_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    proc = Path(CFG["paths"]["processed_dir"])
    c_te = pd.read_parquet(proc / "candidates_test.parquet")
    p_te = pd.read_parquet(proc / "points_test.parquet")
    X = c_te[FEATS].to_numpy(np.float32)
    y = c_te[LABEL].to_numpy()
    lab = y != -1
    idx = np.random.default_rng(seed).choice(np.flatnonzero(lab), min(int(tc["importance_rows"]), lab.sum()),
                                             replace=False)
    names = args.models or sorted(p.stem for p in art.glob("*.joblib"))
    for name in names:
        t0 = time.perf_counter()
        model = joblib.load(art / f"{name}.joblib")
        prob = model.predict_proba(X)[:, 1]
        calibration(y[lab], prob[lab]).to_csv(out / f"{name}_calibration.csv", index=False)
        per_trip_top1(c_te, p_te, prob).to_csv(out / f"{name}_per_trip.csv", index=False)
        permutation_importance(model, X[idx], y[idx], seed, int(tc["importance_repeats"])).to_csv(
            out / f"{name}_importance.csv", index=False)
        print(f"{name}: diagnostics written ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
