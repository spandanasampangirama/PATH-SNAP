# Interface contracts (frozen after Phase 0)

These are the only points where one person's code touches another's. If you need to change one, say so in team chat first, get agreement, and make the change in a small PR of its own.

## 1. Conventions
| Thing | Rule |
|---|---|
| Raw coordinates | Files are **lon, lat**. `Trip.track[:, 0]` is lon. |
| API / Folium coordinates | Always **[lat, lon]**. Swap in exactly one place (`RoadNetwork.arc_polyline_lonlat`, `viz.py`). |
| Distances | Metres, in a **per-trip local projection** (`geo.LocalProjector`). Never use Euclidean distance on degrees. Trips span lon −100…137, so a single UTM zone won't work. |
| IDs | `arc_id` = 0-based row of `.arcs`. `node_id` = 0-based row of `.nodes`. `point_idx` = 0-based row of `.track`. `trip_id` = 8-digit string. |
| Missing values | `-1` means "no arc" (unlabelled point or point with no candidate). See `schema.UNLABELLED` and `schema.NO_CANDIDATE`. |
| Constants | Only in `configs/default.yaml`, read via `from pathsnap.config import CFG`. |
| Seeds | `CFG["seed"]` everywhere (models, CV, Optuna). |

## 2. Data partitions (leakage rules)
- `dataset/train` (80 trips): fitting, GroupKFold CV, Optuna.
- `dataset/test` (10 trips, "student testing"): model comparison, decoder tuning.
- Evaluator trips: **never** used in development. On demo day the API reads them from `PATHSNAP_EVAL_DIR`.
- Prediction code calls `load_trip(..., with_route=False)`. `.route` is read only by `labels.py` (training labels) and `evaluate.py` (server-side metrics).

## 3. Candidate table (`pathsnap/schema.py`)
One row per (trip_id, point_idx, candidate arc_id). It is a pandas DataFrame saved as parquet in `data/processed/`.

| column | type | producer |
|---|---|---|
| trip_id, point_idx, t, segment_id, arc_id | key | P1 `candidates.py` |
| dist_m, proj_frac, arc_len_m, arc_bearing_deg | float | P1 `candidates.py` |
| everything in `schema.FEATURE_COLS` | float | P2 `features.py` |
| label | int 1/0/-1 | P1 (from `labels.py`), training only |
| prob | float | added at inference = `model.predict_proba(X)[:, 1]` |

- Points with **no candidate** inside the radius have **no rows**. Use `candidates.points_without_candidates(cands, n)`.
- The per-point table `points_{part}.parquet` (`schema.POINT_COLS`) holds `ref_arc` for every point. Use it for top-1 accuracy and miss rate.
- Train trips with less than 50% labelled points are left out of both tables (rule `labels.min_labelled_share`). See `docs/P1_DATA_NOTES.md`.
- Build both tables with `python -m scripts.build_dataset` (about 10 s). `data/processed/` is gitignored.

## 4. Function signatures (already stubbed, don't change them)
```
io.load_trip(dir, trip_id, with_route=False) -> Trip                       # done
network.RoadNetwork(trip)                                                  # P1
labels.align_points_to_route(net, track_xy, route, max_dist_m) -> (N,) arc # P1
candidates.generate_candidates(trip, net, k, radius_m, labels=None) -> df  # P1
features.add_features(cands, track_xy, t, net) -> df                       # P2
models.build_model(name, params, seed) -> sklearn estimator                # P2
metrics.*                                                                  # P2
routing.Router(net).route_distance(...)                                    # P3
hmm.viterbi_decode(cands, "prob", router, track_xy, hmm_cfg) -> (N,) arc   # P3
pipeline.PathSnapPredictor(artifacts_dir).predict(dir, trip_id) -> (PredictionResult, net)  # P3
evaluate.evaluate_prediction(result, dir, net) -> dict                     # P3
viz.trip_map(...) -> folium.Map                                            # P4
```

## 5. Artifacts (`artifacts/`, committed; produced by P2 and P3, loaded by `PathSnapPredictor`)
```
artifacts/model.joblib       final tuned sklearn Pipeline (scaler inside), joblib.dump(compress=3)
artifacts/model_meta.json    {"model_name", "feature_cols": [...], "params": {...},
                              "candidates": {"k": 8, "radius_m": 60.0},
                              "sklearn_version", "lightgbm_version", "trained_on": "train (80 trips)",
                              "cv_macro_f1", "seed"}
artifacts/hmm_config.json    {"emission", "sigma_m", "beta_m", "max_route_factor", "gap_split_s"}
artifacts/untuned/*.joblib   the 5+ comparison models (skip any over 50 MB; note it in the README)
```
Inference has to use the `candidates` and `feature_cols` stored in `model_meta.json`, not the current YAML. This keeps training and serving consistent.

## 6. HTTP API (`api/schemas.py`)
| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/health` | none | `Health` |
| GET | `/trips` | none | `TripList` (trip_id, n_points), read from `PATHSNAP_EVAL_DIR` |
| POST | `/predict` | `{"trip_id": "..."}` | `PredictResponse`: GPS points, predicted route polyline, optional classifier-only polyline, `Metrics`, timings |

The UI never receives reference-route geometry, only aggregate metrics computed on the server.
