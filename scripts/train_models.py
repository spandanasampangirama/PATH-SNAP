"""Train + compare models: GroupKFold CV on train, then refit and evaluate on the student-test trips.  Owner: P2

Inputs : data/processed/candidates_{train,test}.parquet, points_{train,test}.parquet
         (run `python -m scripts.build_dataset` first — it adds the features from pathsnap/features.py)
Outputs: results/model_comparison.csv     one row per model (all columns needed for report table 3.1)
         results/cv_folds.csv             per-fold CV metrics (stability across folds)
         results/training_history/        epoch-wise loss for models that have one (MLP only):
                                          mlp_history.csv = per-epoch train AND validation log loss
                                          (80/20 split of train BY TRIP); mlp_final.csv = sklearn loss_curve_
         Plots: run `python -m scripts.plot_models` afterwards (reads only these CSVs).
         artifacts/untuned/<model>.joblib refit-on-all-train model

Run:
    python -m scripts.train_models --models logreg            # baseline
    python -m scripts.train_models                            # all 6 models
    python -m scripts.train_models --models rf lgbm --no-cv   # quick test-set only run

Partitions: CV uses ONLY the training trips (GroupKFold by trip_id, so no trip is split across folds);
the student-test trips are only predicted, never fitted on.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap import schema
from pathsnap.config import CFG
from pathsnap.io import load_route, load_trip
from pathsnap.metrics import (attach_predictions, candidate_metrics, collapse_route, route_continuity,
                              route_overlap, top1_accuracy, top1_from_candidates)
from pathsnap.models import DEFAULT_PARAMS, MAX_TRAIN_ROWS, MODEL_FAMILIES, OBJECTIVES, build_model
from pathsnap.network import RoadNetwork

FEATS = schema.FEATURE_COLS
LABEL = schema.LABEL_COL


def load_tables(proc: Path):
    c_tr = pd.read_parquet(proc / "candidates_train.parquet")
    c_te = pd.read_parquet(proc / "candidates_test.parquet")
    p_tr = pd.read_parquet(proc / "points_train.parquet")
    p_te = pd.read_parquet(proc / "points_test.parquet")
    missing = [f for f in FEATS if f not in c_tr.columns]
    if missing:
        raise SystemExit(f"Feature columns missing {missing}. Run: python -m scripts.build_dataset")
    return c_tr, c_te, p_tr, p_te


def fit_model(name, params, X, y, seed, max_rows):
    if max_rows and len(X) > max_rows:
        idx = np.random.default_rng(seed).choice(len(X), max_rows, replace=False)
        X, y = X[idx], y[idx]
    model = build_model(name, params, seed)
    t0 = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - t0, len(X)


def predict(model, X):
    t0 = time.perf_counter()
    p = model.predict_proba(X)[:, 1]
    return p, time.perf_counter() - t0


def top1_on(cands, prob, points):
    c = cands[["trip_id", "point_idx", "arc_id"]].copy()
    c["prob"] = prob
    pts = attach_predictions(points, top1_from_candidates(c, "prob"))
    return top1_accuracy(pts["pred_arc"].to_numpy(), pts["ref_arc"].to_numpy()), pts


def sequence_metrics(pts: pd.DataFrame, nets: dict, routes: dict, max_hops: int) -> dict:
    """Classifier-only route continuity and route overlap, averaged over trips."""
    cont, f1, prec, rec = [], [], [], []
    for tid, g in pts.groupby("trip_id", sort=False):
        g = g.sort_values("point_idx")
        pred = g["pred_arc"].to_numpy()
        cont.append(route_continuity(pred, nets[tid], max_hops, g["segment_id"].to_numpy()))
        ov = route_overlap(collapse_route(pred), routes[tid])
        f1.append(ov["route_f1"]); prec.append(ov["route_precision"]); rec.append(ov["route_recall"])
    return {"route_continuity": float(np.nanmean(cont)), "route_precision": float(np.mean(prec)),
            "route_recall": float(np.mean(rec)), "route_f1": float(np.mean(f1))}


def reference_baselines(c_tr, c_te, p_tr, p_te) -> pd.DataFrame:
    """Non-ML reference: always pick the nearest candidate arc (rank 0)."""
    rows = []
    for part, c, p in [("train", c_tr, p_tr), ("test", c_te, p_te)]:
        pts = attach_predictions(p, c.loc[c["rank"] == 0, ["trip_id", "point_idx", "arc_id"]])
        rows.append({"baseline": "nearest_arc", "partition": part,
                     "top1_accuracy": top1_accuracy(pts["pred_arc"].to_numpy(), pts["ref_arc"].to_numpy())})
    return pd.DataFrame(rows)


def mlp_epoch_history(X, y, groups, seed: int, out_csv: Path, epochs: int | None = None) -> pd.DataFrame:
    """Genuine per-epoch TRAIN and VALIDATION log loss for the MLP (the only model with an epoch history).

    Train partition only, split 80/20 BY TRIP (GroupShuffleSplit), so validation trips are unseen.
    The scaler is fitted on the 80% part; the MLP is then trained one epoch at a time with partial_fit,
    using the same architecture / regularisation as DEFAULT_PARAMS["mlp"].
    """
    from sklearn.metrics import log_loss
    from sklearn.model_selection import GroupShuffleSplit

    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, groups))
    pipe = build_model("mlp", {"early_stopping": False}, seed)
    scaler, clf = pipe.steps[0][1], pipe.steps[-1][1]
    Xtr, Xva = scaler.fit_transform(X[tr]), scaler.transform(X[va])
    n_ep = epochs or int(DEFAULT_PARAMS["mlp"]["max_iter"])
    rng = np.random.default_rng(seed)
    bs = int(DEFAULT_PARAMS["mlp"]["batch_size"])
    rows = []
    for ep in range(1, n_ep + 1):
        order = rng.permutation(len(Xtr))  # one epoch = one shuffled pass in mini-batches
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            clf.partial_fit(Xtr[b], y[tr][b], classes=np.array([0, 1]))
        rows.append({"epoch": ep,
                     "train_log_loss": log_loss(y[tr], clf.predict_proba(Xtr)[:, 1], labels=[0, 1]),
                     "val_log_loss": log_loss(y[va], clf.predict_proba(Xva)[:, 1], labels=[0, 1]),
                     "n_train_trips": len(np.unique(groups[tr])), "n_val_trips": len(np.unique(groups[va]))})
        print(f"  mlp epoch {ep:3d}: train {rows[-1]['train_log_loss']:.4f}  val {rows[-1]['val_log_loss']:.4f}")
    h = pd.DataFrame(rows)
    h.to_csv(out_csv, index=False)
    return h


def upsert_csv(path: Path, df: pd.DataFrame, keys: list[str]):
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old.set_index(keys).index.isin(df.set_index(keys).index)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_FAMILIES), choices=list(MODEL_FAMILIES))
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--max-train-rows", type=int, default=None, help="cap rows for every model (smoke tests)")
    ap.add_argument("--results-dir", default=CFG["paths"]["results_dir"])
    ap.add_argument("--artifacts-dir", default=CFG["paths"]["artifacts_dir"])
    ap.add_argument("--mlp-history-only", action="store_true",
                    help="only record the MLP per-epoch train/validation log loss (training_history/mlp_history.csv)")
    args = ap.parse_args()

    seed = CFG["seed"]
    res = Path(args.results_dir)
    (res / "training_history").mkdir(parents=True, exist_ok=True)
    art = Path(args.artifacts_dir) / "untuned"
    art.mkdir(parents=True, exist_ok=True)

    c_tr, c_te, p_tr, p_te = load_tables(Path(CFG["paths"]["processed_dir"]))
    fit_mask = c_tr[LABEL].to_numpy() != -1
    X_all, y_all = c_tr[FEATS].to_numpy(np.float32), c_tr[LABEL].to_numpy()
    groups = c_tr["trip_id"].to_numpy()
    X_te = c_te[FEATS].to_numpy(np.float32)
    te_lab = c_te[LABEL].to_numpy() != -1
    print(f"train: {c_tr.trip_id.nunique()} trips, {fit_mask.sum():,} labelled candidate rows | "
          f"test: {c_te.trip_id.nunique()} trips, {len(c_te):,} rows | {len(FEATS)} features")
    if args.mlp_history_only or "mlp" in args.models:
        print("\n=== mlp epoch history (train partition, 80/20 split by trip) ===")
        mlp_epoch_history(X_all[fit_mask], y_all[fit_mask], groups[fit_mask], seed,
                          res / "training_history" / "mlp_history.csv")
        if args.mlp_history_only:
            return

    # networks + reference routes of the student-test trips for sequence-level metrics
    test_dir = CFG["paths"]["test_dir"]
    nets = {t: RoadNetwork(load_trip(test_dir, t)) for t in c_te.trip_id.unique()}
    routes = {t: load_route(test_dir, t) for t in nets}
    max_hops = CFG["metrics"]["continuity_max_hops"]
    miss_rate = float(1 - p_te.loc[p_te.ref_arc >= 0, "ref_in_candidates"].mean())
    ref = reference_baselines(c_tr, c_te, p_tr, p_te)
    ref.to_csv(res / "reference_baselines.csv", index=False)
    print("nearest-arc rule top-1:", ", ".join(f"{r.partition} {r.top1_accuracy:.4f}" for r in ref.itertuples()))

    for name in args.models:
        print(f"\n=== {name} ({MODEL_FAMILIES[name]}) ===")
        max_rows = args.max_train_rows or MAX_TRAIN_ROWS.get(name)
        fold_rows = []
        if not args.no_cv:
            gkf = GroupKFold(n_splits=CFG["cv"]["n_splits"])
            for k, (tr, va) in enumerate(gkf.split(X_all, y_all, groups)):
                tr = tr[fit_mask[tr]]
                model, fit_s, n_fit = fit_model(name, None, X_all[tr], y_all[tr], seed, max_rows)
                prob, pred_s = predict(model, X_all[va])
                va_lab = fit_mask[va]
                m = candidate_metrics(y_all[va][va_lab], prob[va_lab])
                va_trips = np.unique(groups[va])
                t1, _ = top1_on(c_tr.iloc[va], prob, p_tr[p_tr.trip_id.isin(va_trips)])
                fold_rows.append({"model": name, "fold": k, "n_val_trips": len(va_trips), "n_fit_rows": n_fit,
                                  **{k_: m[k_] for k_ in ["accuracy", "precision", "recall", "f1", "macro_f1", "log_loss"]},
                                  "top1_accuracy": t1, "fit_time_s": fit_s, "predict_time_s": pred_s})
                print(f"  fold {k}: macro-F1 {m['macro_f1']:.4f}  top-1 {t1:.4f}  log loss {m['log_loss']:.4f}  ({fit_s:.1f}s)")
            upsert_csv(res / "cv_folds.csv", pd.DataFrame(fold_rows), ["model", "fold"])

        # refit on all training trips, evaluate on student-test trips
        model, fit_s, n_fit = fit_model(name, None, X_all[fit_mask], y_all[fit_mask], seed, max_rows)
        prob, pred_s = predict(model, X_te)
        m = candidate_metrics(c_te[LABEL].to_numpy()[te_lab], prob[te_lab])
        t1, pts = top1_on(c_te, prob, p_te)
        seq = sequence_metrics(pts, nets, routes, max_hops)

        path = art / f"{name}.joblib"
        joblib.dump(model, path, compress=3)
        clf = model.steps[-1][1] if hasattr(model, "steps") else model
        if hasattr(clf, "loss_curve_"):
            hist = pd.DataFrame({"epoch": np.arange(1, len(clf.loss_curve_) + 1), "train_loss": clf.loss_curve_})
            if getattr(clf, "validation_scores_", None) is not None:
                hist["val_accuracy"] = clf.validation_scores_
            hist.to_csv(res / "training_history" / f"{name}_final.csv", index=False)

        cv = pd.DataFrame(fold_rows)
        row = {
            "model": name, "family": MODEL_FAMILIES[name],
            "key_config": json.dumps(DEFAULT_PARAMS[name], default=str), "objective": OBJECTIVES[name],
            "cv_macro_f1_mean": cv.macro_f1.mean() if len(cv) else np.nan,
            "cv_macro_f1_std": cv.macro_f1.std() if len(cv) else np.nan,
            "cv_top1_mean": cv.top1_accuracy.mean() if len(cv) else np.nan,
            "cv_log_loss_mean": cv.log_loss.mean() if len(cv) else np.nan,
            **{f"test_{k}": m[k] for k in ["accuracy", "precision", "recall", "f1", "macro_f1", "log_loss",
                                            "tn", "fp", "fn", "tp"]},
            "test_top1_accuracy": t1,
            **{f"test_{k}": v for k, v in seq.items()},
            "test_candidate_miss_rate": miss_rate,
            "train_time_s": fit_s, "n_train_rows": n_fit,
            "inference_time_s": pred_s, "inference_ms_per_1k_points": 1000 * pred_s / len(p_te) * 1000,
            "model_size_mb": path.stat().st_size / 1e6, "seed": seed,
        }
        upsert_csv(res / "model_comparison.csv", pd.DataFrame([row]), ["model"])
        print(f"  TEST: macro-F1 {m['macro_f1']:.4f}  top-1 {t1:.4f}  log loss {m['log_loss']:.4f}  "
              f"continuity {seq['route_continuity']:.4f}  route-F1 {seq['route_f1']:.4f}  "
              f"| fit {fit_s:.1f}s, predict {pred_s:.2f}s, {row['model_size_mb']:.1f} MB")


if __name__ == "__main__":
    main()
