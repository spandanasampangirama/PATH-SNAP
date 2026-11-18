# P3 — HMM and route reconstruction completion notes

## Scope

P3 consumes candidate road arcs and P2 classifier probabilities, then reconstructs a directed, temporally plausible route with log-space Viterbi. Prediction code must call `load_trip(..., with_route=False)`; only the separate evaluation module reads reference `.route` files.

## Implemented behavior

- Directed bounded Dijkstra over the per-trip road graph, including fractional positions along candidate arcs.
- Transition matrices for each pair of consecutive observations.
- Log-space Viterbi with distance, classifier-probability, and product emissions.
- Time-gap segmentation, candidate-free observation handling, and dead-end restarts.
- Bounded shortest-path route filling between decoded arcs inside the same time segment. Separate segments are never joined.
- Classifier-only and HMM pointwise accuracy, route continuity, candidate miss rate, and route-set precision/recall/F1 are emitted per trip.
- A one-trip preprocessing cache reuses the road index, candidates, and classifier probabilities while searching HMM settings. The cache is bounded to one trip to limit memory.
- Prediction calls are serialized with a reentrant lock because the model feature context and bounded cache are shared instance state across API worker threads.
- Reference route loading is isolated in `pathsnap.evaluate`; it does not enter `PredictionResult` or the predictor.

## Reproducible commands

Run from the repository root with the virtual environment activated and the supplied dataset present:

```powershell
# Synthetic unit/regression suite (no dataset required)
python -m pytest -q tests/test_p3_routing_hmm.py tests/test_p3_pipeline_eval.py

# Real Phase-1 distance-only HMM check on training trip 00000046,
# plus a small HMM-grid smoke run
python -m scripts.eval_hmm --quick

# Full HMM parameter grid and per-trip classifier-only vs HMM comparison
python -m scripts.eval_hmm

# Optional: tune on student-test per the project contract, then report on the same set.
# The report is a development/selection-set result, not an independent holdout.
python -m scripts.eval_hmm --tune-partition test --eval-partition test

# Optional: select settings on train and report on student-test.
# The P2 classifier was fitted on train, so train-set selection metrics are in-sample.
python -m scripts.eval_hmm --tune-partition train --eval-partition test

# Real-data end-to-end smoke test (requires dataset/train trip 00000046)
python -m pytest -q tests/test_e2e.py
```

## Outputs

- `artifacts/hmm_config.json`: selected decoder configuration loaded by `PathSnapPredictor`.
- `results/hmm_distance_baseline.csv`: real distance-only HMM metrics for training trip `00000046`.
- `results/hmm_param_search.csv`: every tested configuration and its aggregate selection objective.
- `results/hmm_comparison.csv`: per-trip classifier-only and HMM metrics, including cases where HMM does not improve the result.
- `results/hmm_run_metadata.json`: seed, exact trip IDs/partitions, model name, objective, selected settings, and whether the evaluation set overlaps the selection set.

## Selection objective

`mean(hmm_route_f1) + 0.25 * mean(top1_acc_hmm) + 0.10 * mean(route_continuity_hmm)`.

Route F1 is primary; pointwise accuracy and directed continuity are tie-breakers. The search grid covers emission mode (distance/classifier/product), `sigma_m` (8/10/15 m), `beta_m` (2/5/10 m), and `max_route_factor` (2/4/8), with a 50 m transition allowance, the configured time-gap threshold, and a 2 km route-fill bound. `--quick` reduces this to one configuration for a smoke test.

## Partition and interpretation safeguards

Only `dataset/train` and `dataset/test` are valid CLI partitions. The script never reads `PATHSNAP_EVAL_DIR` or evaluator data. The default contract uses student-test for decoder development; when the same trips are used for selection and reporting, metadata and terminal output explicitly mark the result as a selection-set estimate. It must not be described as an independent holdout result. No evaluator metrics are generated during development.

## Remaining environment-dependent sign-off

The real-trip command and full parameter search require the dataset, committed model artifacts, and a local Python environment. CI runs the synthetic P3 tests; it cannot verify the private/local dataset or produce the real-trip CSVs.
