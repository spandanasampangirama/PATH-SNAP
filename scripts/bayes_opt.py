"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected model.  Owner: P2

Objective: mean GroupKFold (by trip) CV **log loss** on the TRAIN partition only (minimised).
  Why log loss and not macro-F1: the classifier's probabilities are the HMM emission scores, so their
  quality (calibration + ranking) is what the decoder needs. Macro-F1 at a fixed 0.5 threshold ignores
  both; top-1 is a step function of the scores and is noisy to optimise. CV macro-F1 and CV top-1 are
  still computed and logged for every completed trial so they can be reported.

Sampler: TPE (Tree-structured Parzen Estimator), seed CFG["seed"], first `sampler_startup_trials` random.
         Trial 0 is the untuned default configuration, so the search starts from the baseline.
Pruner : MedianPruner — after each fold the running mean is reported; a trial whose running mean is worse
         than the median of earlier trials at the same fold is stopped early (saves time, still recorded).

Outputs:
    results/optuna/study.db             full Optuna study (sqlite) — reload with optuna.load_study
    results/optuna/trials.csv           every trial: params, value, state, CV macro-F1 / top-1, time
    results/optuna/best_params.json     best hyperparameters + search space + objective
    results/tuned_vs_untuned.csv        untuned (DEFAULT_PARAMS) vs tuned, same folds & partitions
    artifacts/model.joblib              tuned model refit on ALL training trips (CONTRACTS §5)
    artifacts/model_meta.json           model_name, feature_cols, params, candidates, versions, cv score, seed

Run:
    python -m scripts.bayes_opt --model lgbm                 # 50 trials (configs/default.yaml optuna.n_trials)
    python -m scripts.bayes_opt --model lgbm --n-trials 5    # quick check
    python -m scripts.bayes_opt --model lgbm --resume        # continue an existing study.db

The evaluator partition is never read. Student-test trips are used only AFTER the search, to report
tuned vs untuned; they never influence the search space, objective or chosen parameters.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap import schema
from pathsnap.config import CFG
from pathsnap.io import load_route, load_trip
from pathsnap.metrics import candidate_metrics
from pathsnap.models import DEFAULT_PARAMS, MAX_TRAIN_ROWS, MODEL_FAMILIES
from pathsnap.network import RoadNetwork
from scripts.train_models import fit_model, load_tables, predict, sequence_metrics, top1_on

FEATS = schema.FEATURE_COLS
LABEL = schema.LABEL_COL


# Search spaces: ranges around the untuned defaults, wide enough to reach both under- and over-fitting.
def search_space(name: str, trial) -> dict:
    if name == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 500, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    if name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 6, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 200, log=True),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-4, 1e2, log=True)}
    if name == "mlp":
        return {
            "hidden_layer_sizes": tuple(json.loads(trial.suggest_categorical(
                "hidden_layer_sizes", ["[32]", "[64]", "[64, 32]", "[128, 64]", "[128, 64, 32]"]))),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        }
    if name == "knn":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 5, 200, log=True)}
    if name == "gnb":
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-2, log=True)}
    raise ValueError(name)


SPACE_DOC = {  # human-readable copy of the ranges, saved with the results (report §4 Q2)
    "lgbm": {"n_estimators": "int 100-1000 (log)", "learning_rate": "float 0.01-0.2 (log)",
             "num_leaves": "int 15-255 (log)", "min_child_samples": "int 10-500 (log)",
             "subsample": "float 0.5-1.0", "colsample_bytree": "float 0.5-1.0",
             "reg_alpha": "float 1e-8-10 (log)", "reg_lambda": "float 1e-8-10 (log)"},
    "rf": {"n_estimators": "int 50-300", "max_depth": "int 6-30", "min_samples_leaf": "int 1-200 (log)",
           "max_features": "{sqrt, log2, 0.5}"},
    "logreg": {"C": "float 1e-4-100 (log)"},
    "mlp": {"hidden_layer_sizes": "{[32], [64], [64,32], [128,64], [128,64,32]}",
            "alpha": "float 1e-6-0.1 (log)", "learning_rate_init": "float 1e-4-0.01 (log)"},
    "knn": {"n_neighbors": "int 5-200 (log)"},
    "gnb": {"var_smoothing": "float 1e-12-1e-2 (log)"},
}


def _default_as_trial_params(name: str) -> dict:
    """DEFAULT_PARAMS restricted to the searched keys, in the encoding the search space uses."""
    out = {}
    for k, v in DEFAULT_PARAMS[name].items():
        if k in SPACE_DOC[name]:
            out[k] = json.dumps(list(v)) if k == "hidden_layer_sizes" else v
    return out


def cv_evaluate(name, params, X, y, groups, fit_mask, c_tr, p_tr, seed, trial=None) -> pd.DataFrame:
    """GroupKFold CV with folds IDENTICAL to scripts/train_models.py. Returns per-fold metrics."""
    gkf = GroupKFold(n_splits=CFG["cv"]["n_splits"])
    rows = []
    for k, (tr, va) in enumerate(gkf.split(X, y, groups)):
        tr = tr[fit_mask[tr]]
        model, fit_s, _ = fit_model(name, params, X[tr], y[tr], seed, MAX_TRAIN_ROWS.get(name))
        prob, _ = predict(model, X[va])
        lab = fit_mask[va]
        m = candidate_metrics(y[va][lab], prob[lab])
        t1, _ = top1_on(c_tr.iloc[va], prob, p_tr[p_tr.trip_id.isin(np.unique(groups[va]))])
        rows.append({"fold": k, "log_loss": m["log_loss"], "macro_f1": m["macro_f1"], "top1": t1, "fit_s": fit_s})
        if trial is not None:
            trial.report(float(np.mean([r["log_loss"] for r in rows])), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return pd.DataFrame(rows)


def test_evaluate(name, params, X_fit, y_fit, c_te, p_te, nets, routes, seed):
    """Refit on all training rows, evaluate on student test. Returns (metrics dict, fitted model)."""
    model, fit_s, n_fit = fit_model(name, params, X_fit, y_fit, seed, MAX_TRAIN_ROWS.get(name))
    prob, pred_s = predict(model, c_te[FEATS].to_numpy(np.float32))
    lab = c_te[LABEL].to_numpy() != -1
    m = candidate_metrics(c_te[LABEL].to_numpy()[lab], prob[lab])
    t1, pts = top1_on(c_te, prob, p_te)
    seq = sequence_metrics(pts, nets, routes, CFG["metrics"]["continuity_max_hops"])
    return {"test_top1_accuracy": t1, "test_macro_f1": m["macro_f1"], "test_precision": m["precision"],
            "test_recall": m["recall"], "test_log_loss": m["log_loss"],
            "test_route_continuity": seq["route_continuity"], "test_route_f1": seq["route_f1"],
            "train_time_s": fit_s, "n_train_rows": n_fit, "inference_time_s": pred_s,
            "inference_ms_per_1k_points": 1e6 * pred_s / len(p_te)}, model


def main():
    oc = CFG["optuna"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_FAMILIES))
    ap.add_argument("--n-trials", type=int, default=int(oc["n_trials"]))
    ap.add_argument("--resume", action="store_true", help="continue results/optuna/study.db")
    args = ap.parse_args()
    if oc["objective"] != "log_loss":
        raise SystemExit("only optuna.objective = log_loss is implemented (see module docstring)")

    seed, name = CFG["seed"], args.model
    res, art = Path(CFG["paths"]["results_dir"]), Path(CFG["paths"]["artifacts_dir"])
    opt_dir = res / "optuna"
    opt_dir.mkdir(parents=True, exist_ok=True)

    c_tr, c_te, p_tr, p_te = load_tables(Path(CFG["paths"]["processed_dir"]))
    fit_mask = c_tr[LABEL].to_numpy() != -1
    X, y, groups = c_tr[FEATS].to_numpy(np.float32), c_tr[LABEL].to_numpy(), c_tr["trip_id"].to_numpy()
    print(f"[{name}] train {c_tr.trip_id.nunique()} trips, {fit_mask.sum():,} labelled rows; "
          f"{args.n_trials} trials, objective = mean {CFG['cv']['n_splits']}-fold GroupKFold log loss")

    # ---------------------------------------------------------------- search (train only)
    storage = f"sqlite:///{opt_dir / 'study.db'}"
    study_name = f"pathsnap_{name}"
    if not args.resume:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except KeyError:
            pass
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="minimize", load_if_exists=args.resume,
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=int(oc["sampler_startup_trials"])),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=int(oc["pruner_warmup_folds"])))
    if not args.resume:
        study.enqueue_trial(_default_as_trial_params(name))  # trial 0 = the untuned configuration

    def objective(trial):
        t0 = time.perf_counter()
        params = search_space(name, trial)
        folds = cv_evaluate(name, params, X, y, groups, fit_mask, c_tr, p_tr, seed, trial)
        trial.set_user_attr("cv_macro_f1", float(folds.macro_f1.mean()))
        trial.set_user_attr("cv_macro_f1_std", float(folds.macro_f1.std()))
        trial.set_user_attr("cv_top1", float(folds.top1.mean()))
        print(f"  trial {trial.number:3d}: log loss {folds.log_loss.mean():.5f}  macro-F1 {folds.macro_f1.mean():.4f}  "
              f"top-1 {folds.top1.mean():.4f}  ({time.perf_counter() - t0:.0f}s)  {params}", flush=True)
        return float(folds.log_loss.mean())

    t_search = time.perf_counter()
    remaining = args.n_trials - (len(study.trials) if args.resume else 0)
    study.optimize(objective, n_trials=max(remaining, 0), gc_after_trial=True)
    search_s = time.perf_counter() - t_search

    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    trials["duration_s"] = trials.pop("duration").dt.total_seconds()
    trials.to_csv(opt_dir / "trials.csv", index=False)
    best_params = search_space(name, optuna.trial.FixedTrial(study.best_params))
    n_done = int((trials.state == "COMPLETE").sum())
    print(f"\nbest trial {study.best_trial.number}: CV log loss {study.best_value:.5f}  params {best_params}")

    # ---------------------------------------------------------------- tuned vs untuned (same folds, same test)
    test_dir = CFG["paths"]["test_dir"]
    nets = {t: RoadNetwork(load_trip(test_dir, t)) for t in c_te.trip_id.unique()}
    routes = {t: load_route(test_dir, t) for t in nets}
    rows, tuned_model = [], None
    for variant, params in [("untuned", None), ("tuned", best_params)]:
        cv = cv_evaluate(name, params, X, y, groups, fit_mask, c_tr, p_tr, seed)
        te, model = test_evaluate(name, params, X[fit_mask], y[fit_mask], c_te, p_te, nets, routes, seed)
        rows.append({"model": name, "variant": variant,
                     "params": json.dumps({**DEFAULT_PARAMS[name], **(params or {})}, default=str),
                     "cv_log_loss": cv.log_loss.mean(), "cv_macro_f1": cv.macro_f1.mean(),
                     "cv_macro_f1_std": cv.macro_f1.std(), "cv_top1": cv.top1.mean(), **te})
        print(f"  {variant:8s} CV log loss {cv.log_loss.mean():.5f}  CV macro-F1 {cv.macro_f1.mean():.4f}  "
              f"test top-1 {te['test_top1_accuracy']:.4f}  test macro-F1 {te['test_macro_f1']:.4f}  "
              f"test log loss {te['test_log_loss']:.4f}", flush=True)
        if variant == "tuned":
            tuned_model = model
    tvu = pd.DataFrame(rows)
    tvu.to_csv(res / "tuned_vs_untuned.csv", index=False)

    # ---------------------------------------------------------------- export artifacts (CONTRACTS §5)
    import lightgbm
    import sklearn

    joblib.dump(tuned_model, art / "model.joblib", compress=3)
    tuned = tvu[tvu.variant == "tuned"].iloc[0]
    meta = {
        "model_name": name, "family": MODEL_FAMILIES[name], "feature_cols": list(FEATS),
        "params": {**DEFAULT_PARAMS[name], **best_params},
        "candidates": {"k": CFG["candidates"]["k"], "radius_m": CFG["candidates"]["radius_m"]},
        "sklearn_version": sklearn.__version__, "lightgbm_version": lightgbm.__version__,
        "trained_on": f"train ({c_tr.trip_id.nunique()} trips after the min_labelled_share rule)",
        "cv_macro_f1": float(tuned.cv_macro_f1), "cv_log_loss": float(tuned.cv_log_loss),
        "cv_top1": float(tuned.cv_top1), "seed": seed,
    }
    (art / "model_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (opt_dir / "best_params.json").write_text(json.dumps({
        "model": name, "objective": "minimise mean GroupKFold CV log loss (train partition)",
        "sampler": f"TPE(seed={seed}, n_startup_trials={oc['sampler_startup_trials']})",
        "pruner": f"MedianPruner(n_startup_trials=5, n_warmup_steps={oc['pruner_warmup_folds']})",
        "n_trials": len(trials), "n_completed": n_done, "n_pruned": int((trials.state == "PRUNED").sum()),
        "search_seconds": search_s, "best_trial": study.best_trial.number, "best_cv_log_loss": study.best_value,
        "best_params": best_params, "search_space": SPACE_DOC[name],
    }, indent=2, default=str))
    print(f"\nwrote {art / 'model.joblib'}, {art / 'model_meta.json'}, {res / 'tuned_vs_untuned.csv'}, "
          f"{opt_dir}/{{study.db, trials.csv, best_params.json}}  (search {search_s / 60:.1f} min)")


if __name__ == "__main__":
    main()
