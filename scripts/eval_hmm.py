"""Tune and evaluate P3 HMM settings without touching evaluator data.

By contract, the student-test partition is the decoder-development/tuning partition.
If tune and eval partitions are both "test", the comparison is a selection-set result,
not an independent holdout estimate. Use --tune-partition train --eval-partition test
when a separate student-test report is desired (train metrics are in-sample for the P2 model).

Examples:
    python -m scripts.eval_hmm
    python -m scripts.eval_hmm --quick
    python -m scripts.eval_hmm --tune-partition train --eval-partition test

The real distance-only Phase-1 check on train trip 00000046 is also run by default.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.config import CFG
from pathsnap.io import list_trips
from pathsnap.pipeline import PathSnapPredictor
from pathsnap.evaluate import evaluate_prediction, prepare_evaluation


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _mean_objective(frame: pd.DataFrame) -> float:
    """Route F1 is primary, with point accuracy and directed continuity as tie-breakers."""
    route_f1 = _mean(frame["hmm_route_f1"])
    top1 = _mean(frame["top1_acc_hmm"])
    continuity = _mean(frame["route_continuity_hmm"])
    if not all(np.isfinite(v) for v in (route_f1, top1, continuity)):
        return float("nan")
    return route_f1 + 0.25 * top1 + 0.10 * continuity


def _parameter_grid(has_model: bool, quick: bool) -> list[dict]:
    emissions = ["distance"] if not has_model else ["distance", "classifier", "product"]
    if quick:
        emissions = ["classifier"] if has_model else ["distance"]
    sigmas = [10.0] if quick else [8.0, 10.0, 15.0]
    betas = [5.0] if quick else [2.0, 5.0, 10.0]
    factors = [4.0] if quick else [2.0, 4.0, 8.0]
    return [
        {
            "emission": emission,
            "sigma_m": sigma,
            "beta_m": beta,
            "max_route_factor": factor,
            "max_route_extra_m": 50.0,
            "gap_split_s": float(CFG["gaps"]["split_s"]),
            "route_fill_max_m": 2000.0,
        }
        for emission, sigma, beta, factor in itertools.product(emissions, sigmas, betas, factors)
    ]


def _trip_ids(partition: str, requested: list[str] | None) -> list[str]:
    directory = CFG["paths"][f"{partition}_dir"]
    available = list_trips(directory)
    if requested:
        wanted = set(requested)
        available = [trip_id for trip_id in available if trip_id in wanted]
    if not available:
        raise SystemExit(f"No trips found in {directory}; requested={requested}")
    return available


def _distance_only_baseline(trip_id: str, artifacts: Path, results: Path) -> None:
    train_dir = CFG["paths"]["train_dir"]
    if trip_id not in list_trips(train_dir):
        raise SystemExit(f"Required P3 baseline trip {trip_id} is missing from {train_dir}")
    predictor = PathSnapPredictor(str(artifacts))
    predictor.model = None
    predictor.model_name = "distance-only-baseline"
    predictor.hmm_cfg.update({"emission": "distance"})
    result, net = predictor.predict(train_dir, trip_id)
    context = prepare_evaluation(train_dir, trip_id, net, result.candidate_k, result.candidate_radius_m)
    metrics = evaluate_prediction(result, train_dir, net, context)
    row = {
        "trip_id": trip_id,
        "model_name": predictor.model_name,
        "emission": "distance",
        **metrics,
        **{f"time_{key}_ms": value for key, value in result.timings_ms.items()},
    }
    path = results / "hmm_distance_baseline.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    print(f"Distance-only HMM baseline on train trip {trip_id}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune-partition", choices=["train", "test"], default="test")
    parser.add_argument("--eval-partition", choices=["train", "test"], default="test")
    parser.add_argument("--tune-trips", nargs="*", default=None)
    parser.add_argument("--eval-trips", nargs="*", default=None)
    parser.add_argument("--trips", nargs="*", default=None,
                        help="Backward-compatible alias for --eval-trips")
    parser.add_argument("--quick", action="store_true", help="single-point smoke-test parameter grid")
    parser.add_argument("--skip-distance-baseline", action="store_true")
    parser.add_argument("--distance-baseline-trip", default="00000046")
    parser.add_argument("--artifacts-dir", default=CFG["paths"]["artifacts_dir"])
    parser.add_argument("--results-dir", default=CFG["paths"]["results_dir"])
    args = parser.parse_args()

    artifacts, results = Path(args.artifacts_dir), Path(args.results_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    if not args.skip_distance_baseline:
        _distance_only_baseline(args.distance_baseline_trip, artifacts, results)

    tune_ids = _trip_ids(args.tune_partition, args.tune_trips)
    eval_requested = args.eval_trips if args.eval_trips is not None else args.trips
    eval_ids = _trip_ids(args.eval_partition, eval_requested)

    probe = PathSnapPredictor(str(artifacts))
    has_model = probe.model is not None
    configs = _parameter_grid(has_model, args.quick)
    if not configs:
        raise SystemExit("No HMM parameter configurations were generated")

    # One predictor and one prepared evaluation context per trip. The predictor's bounded
    # one-trip cache reuses the road index, candidate table and classifier scores across trials.
    trial_metrics: list[list[dict]] = [[] for _ in configs]
    for trip_id in tune_ids:
        predictor = PathSnapPredictor(str(artifacts))
        context = None
        net = None
        for index, config in enumerate(configs):
            predictor.hmm_cfg.update(config)
            result, net = predictor.predict(CFG["paths"][f"{args.tune_partition}_dir"], trip_id)
            if context is None:
                context = prepare_evaluation(
                    CFG["paths"][f"{args.tune_partition}_dir"], trip_id, net,
                    result.candidate_k, result.candidate_radius_m)
            metrics = evaluate_prediction(result, CFG["paths"][f"{args.tune_partition}_dir"], net, context)
            trial_metrics[index].append({"trip_id": trip_id, **metrics})
        print(f"Tuned {len(configs)} HMM configurations for trip {trip_id}")

    search_rows = []
    for index, config in enumerate(configs):
        frame = pd.DataFrame(trial_metrics[index])
        score = _mean_objective(frame)
        search_rows.append({
            "config_id": index,
            **config,
            "objective": score,
            "mean_classifier_route_f1": _mean(frame["classifier_route_f1"]),
            "mean_hmm_route_f1": _mean(frame["hmm_route_f1"]),
            "mean_top1_classifier": _mean(frame["top1_acc_classifier"]),
            "mean_top1_hmm": _mean(frame["top1_acc_hmm"]),
            "mean_continuity_classifier": _mean(frame["route_continuity_classifier"]),
            "mean_continuity_hmm": _mean(frame["route_continuity_hmm"]),
            "n_trips": len(frame),
        })
    search_frame = pd.DataFrame(search_rows)
    valid = search_frame[np.isfinite(search_frame["objective"])]
    if valid.empty:
        raise SystemExit("All HMM configurations produced non-finite objective values")
    best_row = valid.sort_values(["objective", "config_id"], ascending=[False, True]).iloc[0]
    best_cfg = {key: best_row[key].item() if isinstance(best_row[key], np.generic) else best_row[key]
                for key in configs[0]}
    search_frame.to_csv(results / "hmm_param_search.csv", index=False)
    (artifacts / "hmm_config.json").write_text(json.dumps(best_cfg, indent=2) + "\n", encoding="utf-8")

    # Evaluate the selected configuration per trip. If this is the same partition used for
    # selection, metadata explicitly marks it as a tuning/selection-set report.
    comparison = []
    for trip_id in eval_ids:
        predictor = PathSnapPredictor(str(artifacts))
        predictor.hmm_cfg.update(best_cfg)
        result, net = predictor.predict(CFG["paths"][f"{args.eval_partition}_dir"], trip_id)
        context = prepare_evaluation(
            CFG["paths"][f"{args.eval_partition}_dir"], trip_id, net,
            result.candidate_k, result.candidate_radius_m)
        comparison.append({
            "trip_id": trip_id,
            "model_name": predictor.model_name,
            **evaluate_prediction(result, CFG["paths"][f"{args.eval_partition}_dir"], net, context),
            **{f"time_{key}_ms": value for key, value in result.timings_ms.items()},
        })
    comparison_frame = pd.DataFrame(comparison)
    comparison_frame.to_csv(results / "hmm_comparison.csv", index=False)

    metadata = {
        "seed": int(CFG["seed"]),
        "tune_partition": args.tune_partition,
        "tune_trip_ids": tune_ids,
        "eval_partition": args.eval_partition,
        "eval_trip_ids": eval_ids,
        "model_name": probe.model_name,
        "n_configurations": len(configs),
        "selection_objective": "mean(hmm_route_f1) + 0.25*mean(top1_acc_hmm) + 0.10*mean(route_continuity_hmm)",
        "best_objective": float(best_row["objective"]),
        "best_config": best_cfg,
        "evaluation_is_selection_set": args.tune_partition == args.eval_partition
            and set(tune_ids).intersection(eval_ids) == set(eval_ids),
        "evaluator_partition_accessed": False,
        "note": "Student-test tuning/evaluation is development evidence, not an untouched evaluator result."
            if args.tune_partition == "test" else
            "Train-set HMM selection is in-sample for the P2 classifier; student-test metrics are reported separately.",
    }
    (results / "hmm_run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Selected HMM configuration: {best_cfg}")
    print(f"Saved {results / 'hmm_param_search.csv'}")
    print(f"Saved {results / 'hmm_comparison.csv'}")
    print(f"Saved {results / 'hmm_run_metadata.json'}")
    print(f"Saved {artifacts / 'hmm_config.json'}")
    if metadata["evaluation_is_selection_set"]:
        print("WARNING: comparison metrics are from the same student partition used for selection; do not call them independent holdout metrics.")


if __name__ == "__main__":
    main()
