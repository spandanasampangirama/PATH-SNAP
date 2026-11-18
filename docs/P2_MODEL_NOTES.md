# P2 model notes: Phase 1 status

## What's ready
| File | What it does |
|---|---|
| `pathsnap/features.py` | 14 features per candidate. `build_dataset` adds them automatically. |
| `pathsnap/metrics.py` | The one shared definition of every metric: candidate metrics, top-1, miss rate, route continuity, route overlap |
| `pathsnap/models.py` | 6 models from 6 families, with their initial hyperparameters and training objectives |
| `scripts/train_models.py` | GroupKFold(5) CV on train, then refit and evaluate on student test. Writes the CSVs and saves the models. |
| `tests/test_p2_models.py` | Tests for metrics, features and all 6 models |

```bash
python -m scripts.build_dataset                  # adds features (~10 s)
python -m scripts.train_models --models logreg   # ~30 s
python -m scripts.train_models                   # all 6 models (Phase 2)
```

## Features (and why)
| Group | Features | Why |
|---|---|---|
| Distance | `dist_m`, `dist_minus_min_m`, `dist_ratio_min`, `rank`, `n_candidates` | The true road is usually the closest one, but in dense areas *relative* distance matters more |
| Direction | `heading_diff_deg`, `heading_align`, `speed_mps` | Two-way streets are two identical arcs, and only direction separates them. Heading uses a ±3 s window. Alignment fades out when the vehicle is almost stopped. |
| Consistency | `nbr_dist_mean_m`, `nbr_dist_max_m` | The correct road stays close to the neighbouring ±3 fixes too, while a crossing road is only close at the junction |
| Road context | `arc_len_m`, `endpoint_dist_m`, `is_oneway`, `to_out_degree` | Junctions and short arcs are where mistakes happen |

Median values on train (correct vs wrong candidate):
- heading difference: 6° vs 103°
- distance beyond the best candidate: 0 m vs 13 m
- neighbour distance: 8 m vs 26 m

## Phase 1 results: baseline (logistic regression, all features)
| Metric | Value | Partition |
|---|---|---|
| CV macro-F1 | 0.905 ± 0.005 (5 folds) | train (78 trips, GroupKFold by trip) |
| CV top-1 accuracy | 0.882 | train |
| Test macro-F1 / log loss | 0.900 / 0.135 | student test (10 trips) |
| **Test top-1 accuracy** | **0.864** | student test |
| Test route continuity (classifier only) | 0.967 | student test |
| Test route F1 (arc set) | 0.854 | student test |
| Nearest-road rule, top-1 (non-ML reference) | 0.536 | student test |

The nearest-road rule is close to a coin flip on two-way streets (both arcs are equally close). The direction features are what move accuracy from 54% to 86%.

All 6 models train and predict end to end (Phase 2 ran five of them; see below). They were smoke-tested on 50k rows (results not saved). Expect the tree and boosting models to beat logistic regression.

## Decisions to be able to defend
- **No class re-weighting** (16% positives). Re-weighting distorts the probabilities, which are the HMM's emission inputs. Top-1 depends only on the ranking within each point.
- **GroupKFold by trip.** All candidates of a trip stay in the same fold, otherwise neighbouring fixes would leak between training and validation.
- **KNN is trained on a 200k-row sample**, because its prediction cost grows with training size. This is recorded in `n_train_rows`.
- **Only the MLP has an epoch-wise loss curve** (`results/training_history/`). The rules forbid artificial curves for the other models.

## Phase 2 (P2): comparison, selection, Bayesian optimisation, export

### Reproduce
```bash
python -m scripts.build_dataset                 # ~45 s, candidate tables + features
python -m scripts.train_models --models logreg gnb knn rf lgbm   # ~40 min (KNN predict and RF fit dominate)
python -m scripts.model_diagnostics             # calibration, per-trip top-1, permutation importance (~15 min, KNN)
python -m scripts.bayes_opt --model lgbm        # 50 Optuna trials, ~60 min -> artifacts/model.joblib
python -m scripts.plot_models                   # every p2_*.png, drawn only from the CSVs
```
**Five models, not six.** The MLP was dropped by team decision once the first five finished (the rules require at least five). Consequently there is no epoch-wise training curve; none of the five models has an epoch history, so none is drawn.

### Model comparison (`results/model_comparison.csv`)
CV = 5-fold GroupKFold by trip on 78 train trips. Test = 10 student-test trips. Seed 42.

| Model | CV macro-F1 (± std) | CV top-1 | CV log loss | Test top-1 | Test macro-F1 | Test log loss | Route continuity | Train | Predict (test) | Size |
|---|---|---|---|---|---|---|---|---|---|---|
| **lgbm** | 0.936 ± 0.007 | **0.898** | 0.088 | **0.896** | **0.934** | **0.086** | **0.975** | 12 s | 0.68 s | 0.6 MB |
| rf | **0.937** ± 0.007 | 0.897 | 0.088 | 0.888 | 0.932 | 0.088 | 0.972 | 205 s | 0.34 s | 34.5 MB |
| knn | 0.924 ± 0.006 | 0.883 | 0.160 | 0.879 | 0.921 | 0.151 | 0.967 | 0.6 s (200k rows) | 58.7 s | 17.9 MB |
| logreg | 0.905 ± 0.005 | 0.882 | 0.140 | 0.863 | 0.900 | 0.135 | 0.967 | 12 s | 0.06 s | <0.1 MB |
| gnb | 0.848 ± 0.007 | 0.865 | 0.561 | 0.844 | 0.835 | 0.603 | 0.960 | 0.9 s | 0.07 s | <0.1 MB |
| nearest-arc rule | – | – | – | 0.533 | – | – | – | – | – | – |

**Over/underfitting** (`results/fit_diagnostics.csv`, train = refit model scored on its own rows): train − CV macro-F1 gap is 0.003 (logreg), 0.002 (gnb), 0.023 (knn), 0.015 (rf), 0.016 (lgbm). Logreg and GNB underfit (train ≈ CV but low); KNN overfits mildly; RF and LightGBM generalise (CV ≈ test).

**Calibration on test** (ECE, `p2_model_<model>.png` panel b): lgbm 0.003, knn 0.007, gnb 0.095 (badly over-confident). Hardest trip for every model: 00000077 (lgbm top-1 0.772, nearest-arc 0.40). Most important feature for lgbm/knn: `heading_align` (permutation importance).

### Selection: LightGBM
- Best student-test top-1 (0.896), macro-F1 (0.934), log loss (0.086) and continuity (0.975).
- Ties RF on CV (0.9363 vs 0.9366, inside the ±0.007 fold spread) but trains 17× faster and is 57× smaller, which made 50 Optuna trials feasible (~1 h vs ~11 h).
- Best-calibrated probabilities (ECE 0.003): exactly what the HMM emissions need.

### Bayesian optimisation (`scripts/bayes_opt.py`, `results/optuna/`)
- **Objective:** minimise mean 5-fold GroupKFold CV **log loss** on train only. Log loss rather than macro-F1 because the probabilities feed the HMM; macro-F1 at 0.5 ignores calibration and within-point ranking. CV macro-F1 and top-1 are logged per trial.
- **Search space:** n_estimators 100–1000 (log), learning_rate 0.01–0.2 (log), num_leaves 15–255 (log), min_child_samples 10–500 (log), subsample 0.5–1, colsample_bytree 0.5–1, reg_alpha and reg_lambda 1e-8–10 (log).
- **Search:** TPE (seed 42, 10 random start-up trials), MedianPruner; trial 0 = untuned defaults. 50 trials: 16 completed, 34 pruned, 58.5 min. Best = trial 20.
- **Best params:** n_estimators 197, learning_rate 0.035, num_leaves 136, min_child_samples 22, subsample 0.795, colsample_bytree 0.777, reg_alpha 0.0069, reg_lambda 3.4e-5.

| (`results/tuned_vs_untuned.csv`) | Untuned | Tuned |
|---|---|---|
| CV log loss | 0.0884 | **0.0871** |
| CV macro-F1 | 0.9363 | **0.9376** |
| CV top-1 | 0.8982 | **0.8995** |
| Test top-1 | 0.8958 | **0.8964** |
| Test macro-F1 | 0.9340 | **0.9350** |
| Test log loss | 0.0861 | **0.0855** |
| Test route F1 | 0.8880 | **0.8909** |
| Inference, ms / 1k points | 30.2 | **21.1** |

Tuning helps slightly on every metric (route continuity flat, 0.9745 vs 0.9742): the defaults were already near the optimum, and the remaining error is ranking between two-way twins, which the HMM must resolve. Note: LightGBM multithreading gives tiny run-to-run differences (trial 0 logged 0.08857 vs 0.08840 when re-evaluated).

### Exported (CONTRACTS §5)
- `artifacts/model.joblib` (1.3 MB): tuned LightGBM refit on all 78 train trips.
- `artifacts/model_meta.json`: model name, the 14 feature columns, params, candidates k=8 / r=60 m, sklearn 1.5.1, lightgbm 4.5.0, CV scores, seed.
- `artifacts/untuned/{logreg,gnb,knn,rf,lgbm}.joblib`: the five comparison models (rf 34.5 MB, under the 50 MB limit).
- Checked: `PathSnapPredictor` loads `lgbm` from these artifacts; trip 00000052 predicts in 0.22 s (classifier top-1 0.973, HMM 0.992).

### Plots (`results/plots/`, all from CSVs via `scripts/plot_models.py`)
`p2_cv_macro_f1`, `p2_test_top1_macro_f1`, `p2_test_log_loss`, `p2_inference_time`, `p2_confusion_<model>` (×5), `p2_model_<model>` (×5: CV per fold, calibration, top-1 per test trip, permutation importance), `p2_optuna_history`, `p2_tuned_vs_untuned`.

### Config keys added (`configs/default.yaml`, no renames)
`features.*` (heading/neighbour windows, heading full speed: previously hardcoded, values unchanged, features verified byte-identical), `training.max_train_rows`, `training.importance_*`, `optuna.*`.
