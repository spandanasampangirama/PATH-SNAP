# P3 implementation notes

## Scope and interfaces

P3 owns routing, HMM/Viterbi, the prediction pipeline, server-side evaluation and the HMM
parameter-search runner. Public call signatures in docs/CONTRACTS.md are preserved.

## Directed routing

Router runs bounded Dijkstra over RoadNetwork.out_arcs and never treats the road network as
undirected. A transition from a fractional position on arc A to a fractional position on arc B
is the remaining fraction of A, plus the shortest directed path from A's end node to B's
start node, plus the traversed fraction of B. Same-arc forward movement is handled directly.
Invalid IDs, unreachable paths and paths beyond the caller's distance bound return infinity.
Dijkstra distance/predecessor trees are cached in a bounded LRU cache. shortest_arc_path
returns only intermediate arcs, excluding both selected endpoint arcs.

## HMM / Viterbi

Each candidate arc is a hidden state for its GPS observation. Emissions are calculated in
log space: distance uses a Gaussian distance likelihood; classifier uses the supplied
positive-class probability; product combines classifier log probability and the Gaussian
distance term. Probabilities are clipped away from zero before taking logarithms.

A transition is allowed only when directed routing returns a finite distance within
max_route_factor * straight_distance + max_route_extra_m. Its log score is
-abs(route_distance - straight_distance) / beta_m. Viterbi maximizes the sum of log emission
and transition scores.

Decoding runs independently across segment_id boundaries, missing-candidate observations,
and dead ends. A point with no candidate remains -1. At a dead end, the decoder closes the
previous run and restarts at the current observation rather than inventing a connection.

## Prediction pipeline and P2 integration

PathSnapPredictor prefers the final model.joblib plus model_meta.json and loads
hmm_config.json when available. Candidate generation settings and feature columns are read
from model metadata. If the final tuned artifact is absent but
artifacts/untuned/logreg.joblib exists, it loads that committed P2 baseline and uses
features.FEATURE_COLS with the YAML candidate settings. The classifier must expose
predict_proba; positive class 1 becomes the candidate probability.

P2 feature engineering, model definitions, training harness and an untuned logistic-regression
artifact are now committed. The distance-only fallback is retained only when no model
artifact is available. The fallback is explicitly named logreg-untuned-baseline so its
results cannot be mistaken for the final tuned model.

The prediction path calls load_trip(..., with_route=False). Reference routes are loaded only
by evaluate_prediction, which returns numeric metrics and never adds reference geometry to
the prediction result.

## Evaluation and tuning

Run python -m scripts.eval_hmm after placing the student-testing partition in dataset/test.
The script writes per-trip classifier-only/HMM metrics to results/hmm_comparison.csv, the
tested parameter grid to results/hmm_param_search.csv, and the selected decoder settings to
artifacts/hmm_config.json. The search uses student-testing trips only, as specified by the
project contract; it must never read the reserved evaluator partition. Run with --quick for
a one-configuration smoke test or --trips ID ... for a restricted run.

Synthetic tests cover directed/fractional routing, distance bounds, connected Viterbi
decoding, missing candidates, segment boundaries, dead-end restarts and loading the P2
baseline artifact. Run pytest -q tests/test_p3_routing_hmm.py and then pytest -q.

## Remaining experiment dependency

The final tuned artifacts/model.joblib and model_meta.json are not yet present; P2's Optuna
script remains a stub. P3 therefore uses the untuned logistic-regression baseline when
available and will automatically prefer the final tuned artifact once exported.

The raw dataset/train and dataset/test partitions are intentionally not committed. Run the
HMM evaluation locally after placing those partitions in the documented paths. It produces
the parameter-search CSV, per-trip classifier-vs-HMM comparison, and selected HMM config.
Do not claim real-trip HMM improvements until those outputs have been generated and reviewed.
The reserved evaluator partition must not be used for tuning. No experimental metrics are
fabricated here.
