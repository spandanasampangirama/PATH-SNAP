# PathSnap: Phase 1 Review

> Every number in this document comes from our own code: `scripts/build_dataset.py`, `scripts/candidate_sweep.py` and `results/data/*.csv`.
> The partition used for each figure is stated. We never opened the evaluator trips.

---

## 1. The problem in one line
GPS is noisy (it drifts several metres, jumps, and has gaps), so raw points don't sit on roads.
**PathSnap works out which road segment the vehicle was really on at every GPS point, and then joins those into a connected route.**

---

## 2. What we were given

### 2.1 Files per trip
```
dataset/<partition>/<trip_id>/
   <trip_id>.track   lon  lat  t        GPS observations (about 1 Hz)
   <trip_id>.nodes   lon  lat           road-network points      (row index = node_id)
   <trip_id>.arcs    from_node  to_node directed road segments    (row index = arc_id)
   <trip_id>.route   arc_id             reference route: arcs driven, in order
```

How the files relate to each other:
```
 .track (GPS points) ──── NO direct link ────► we must match each point to an arc
                                                       │
 .nodes ◄── from/to ── .arcs ◄── arc_id ── .route (ordered ground-truth arcs)
```
- **Every trip has its own road network.** A GPS row does *not* correspond to a node or arc row.
- **`.route` has far fewer rows than `.track`.** For example, trip 00000001 has 1,315 GPS points but 525 route arcs. **The dataset gives no label per GPS point, so we have to derive one** (see §4, step 5).

### 2.2 Dataset inventory
| | Train | Student test |
|---|---|---|
| Trips | 80 | 10 |
| GPS points (total) | 208,417 | 19,160 |
| Points per trip (min / median / max) | 361 / 1,529 / 19,313 | 262 / 1,420 / 3,958 |
| Nodes per trip (min / median / max) | 299 / 9,608 / 155,003 | 1,770 / 6,596 / 26,635 |
| Arcs per trip (min / median / max) | 649 / 19,526 / **307,392** | 3,577 / 13,145 / 56,027 |
| Route arcs per trip (min / median / max) | 59 / 280 / 2,266 | 95 / 250 / 1,323 |
| Steps exactly 1 s apart | 96.5% | 98.6% |
| Trips with a time gap > 60 s | 22 | 1 |
| Missing (NaN) values / non-increasing timestamps | 0 / 0 | 0 / 0 |
| Longitude range / latitude range | −100.4…137.2 / −34.6…61.3 | −93.4…77.3 / 42.6…61.7 |

The evaluator partition (10 trips) is held by the organisers. We haven't used it.

---

## 3. What we learned from exploring the data
| # | Observation | Why it matters | What we did |
|---|---|---|---|
| 1 | **Trips are spread worldwide** (longitude −100 to 137) | Euclidean distance on degrees is wrong, and one UTM zone can't cover every trip | A local metric projection for each trip (§4, step 2) |
| 2 | **Arcs are directed.** A two-way street is 2 arcs with identical geometry. The one-way share is 0–65% per trip. | Distance alone can't tell the two directions apart | Connectivity respects direction. Heading becomes a key feature for the models. |
| 3 | **Network sizes vary about 470×** (649 to 307,392 arcs) | Brute-force nearest-road search would be too slow, especially in the live API | Spatial index (R-tree) |
| 4 | **Large time gaps.** The largest are 42.5 h (00000090) and 14.7 h (00000078). | No sensible motion can link points across such gaps | The track is split into segments at gaps over 60 s |
| 5 | **Networks aren't fully connected.** Median 20 weakly connected components per train trip. The largest component holds a median of 98% of nodes, and as little as 69%. | Some candidate-to-candidate moves are impossible | The HMM must handle unreachable transitions |
| 6 | **The reference route itself has small breaks** (98.1–100% of consecutive arcs connect) | Labels aren't perfect ground truth | We say so wherever we report accuracy |
| 7 | **Two train trips have systematically shifted GPS.** 00000054 is about 138 m north of its own route and roads. 00000033 is about 115 m south. The shift varies along the trip (std 25–60 m). | Their labels would be mostly wrong | Excluded by a rule, not by ID (§4, step 7) |
| 8 | Many trips are slow (few points above 8 m/s) | Single-step heading at 1 Hz is very noisy | Heading is measured over a ±3 s window |
| 9 | No NaNs, no unordered timestamps, no isolated nodes | No imputation needed | n/a |

---

## 4. Pre-processing pipeline (P1, done)
```
 raw trip ─► 1 load ─► 2 project ─► 3 index network ─► 4 split at gaps
                                                            │
     8 tables ◄─ 7 quality rule ◄─ 6 candidates ◄─ 5 per-point labels
```

| Step | What | How | Why |
|---|---|---|---|
| 1. Load | Read the 4 files | `io.load_trip()`. The route is **only loaded when explicitly asked for**. | Ground truth can't leak into prediction |
| 2. Project | lon/lat → metres | Azimuthal Equidistant projection centred on each trip's GPS centroid (pyproj, WGS84) | Accurate metre distances anywhere in the world. **Checked: error under 1 cm over 62 km** compared with exact ellipsoidal distance. |
| 3. Index | Build the road network | Arc length and bearing, directed successor lists, reverse-arc lookup, shapely STRtree over segments | Instant "roads within R metres" queries. Builds in 0.22 s even for 307k arcs. |
| 4. Segment | Split tracks at gaps | New `segment_id` when Δt > 60 s | Each segment is decoded independently later |
| 5. Label | Reference arc for **every** GPS point | **Monotone dynamic-programming alignment** of points to the ordered route: positions along the route never go backwards, and total (capped) point-to-arc distance is minimised. Points more than 50 m from their arc get −1. | `.route` has no timestamps and fewer rows than the track. Forward-only matching stops loops and out-and-back trips from being mixed up. |
| 6. Candidates | Shortlist per point | Up to **k = 8** arcs within **r = 60 m**, with distance, position along arc, arc length and bearing. Label is 1 for the reference arc, 0 for the others. | Turns the task into scoring a small set of options |
| 7. Quality rule | Drop unusable trips | Exclude a trip if **fewer than 50%** of its points can be labelled | Removes the 2 shifted trips without hardcoding IDs. The next-lowest trip is 88%, so the threshold isn't borderline. |
| 8. Output | Training tables | `candidates_{train,test}.parquet` (one row per point × candidate), `points_{train,test}.parquet` (one row per point), `build_stats_*.csv` (per-trip quality) | Shared input for models (P2) and evaluation (P3) |

All 90 trips build in about **10 seconds**.

### 4.1 Are the derived labels trustworthy?
Measured on train plus student test, with the 2 shifted trips excluded (88 trips):

| Check | Result |
|---|---|
| Distance from a point to its labelled arc (median of per-trip medians) | **≈ 4.6 m**, which is typical GPS noise |
| Direction check: while moving > 3 m/s, does the GPS heading (±3 s window) match the labelled arc's direction within 45°? | **Median 98.6% of points**, minimum 81.4% |
| Points that received a label | Train 98.6%, test 99.9% |

### 4.2 Choosing k and r (train partition only)
Candidate miss rate means the share of labelled points whose true arc isn't in the shortlist. Those points can never be matched correctly.

| k (at r = 60 m) | 4 | 6 | **8** | 12 | 16 |
|---|---|---|---|---|---|
| Miss rate | 2.39% | 0.95% | **0.46%** | 0.19% | 0.11% |
| Candidates per point | 3.6 | 4.9 | **6.0** | 7.7 | 8.8 |

- A radius above 60 m gives no further gain. Labels stop at 50 m anyway.
- **We chose k = 8, r = 60 m.** Going to k = 12 buys only 0.27 points of miss rate, for 27% more rows and about 2.25× the HMM transition work (8² vs 12² pairs per step).

### 4.3 Final training tables
| | Train (78 trips) | Student test (10 trips) |
|---|---|---|
| GPS points | 183,260 | 19,160 |
| Candidate rows | 1,095,448 | 119,602 |
| Candidate miss rate | 0.46% | 0.26% |
| Points with no candidate in 60 m | 0.70% | 0.00% |
| Positive rows (correct arc) | 16.4% | 16.0% |

---

## 5. Overall approach and architecture

### 5.1 How the ML problem is framed
For each GPS point and each of its candidate arcs, the question is: **"Is this the road the vehicle was on?"** That's binary classification (1 or 0).
- Model output: the probability that each candidate is correct.
- Top-1 prediction: the candidate with the highest probability at that point.
- These probabilities then become the **emission probabilities** of the HMM.

### 5.2 End-to-end pipeline
```
             ┌─────────── P1 (done) ──────────┐  ┌──── P2 ────┐   ┌────── P3 ──────┐
 GPS track ─►│ project → candidates (k=8,60m) │─►│ features → │──►│ HMM + Viterbi  │─► route
 road net  ─►│ spatial index, connectivity    │  │ classifier │   │ (connectivity) │
             └────────────────────────────────┘  └────────────┘   └────────────────┘
                     labels (training only) ──────► model training, 5+ models, Bayesian tuning
```
- **Classifier (P2):** at least 5 model families, for example logistic regression, Naive Bayes, KNN, random forest, LightGBM and MLP.
  - Features include distance, heading vs arc direction, speed, distance rank, and so on.
  - Evaluated with **GroupKFold by trip**, so a trip is never split between training and validation.
  - The best model is tuned with Bayesian optimisation (Optuna).
- **HMM (P3):**
  - Hidden states are the candidate arcs of each point.
  - Emissions come from the classifier probabilities.
  - Transitions favour moves where the road distance is close to the straight-line distance, and impossible moves get probability 0.
  - Viterbi picks the best *connected* sequence.
  - Tracks are split at gaps, and the decoder restarts at dead ends.

### 5.3 Baseline classifier (P2, first results)
**Features.** There are 14 per candidate, computed only from the GPS track and road network, never from the route:
| Group | Features | Why |
|---|---|---|
| Distance | distance, distance beyond the best candidate, distance ratio, rank, number of candidates | The true road is usually close, but in dense areas *relative* distance matters most |
| Direction | heading difference (±3 s window), speed-weighted heading alignment, speed | The only way to separate the two directed arcs of a two-way street |
| Consistency | mean and max distance of the neighbouring ±3 fixes to the same arc | The correct road stays close over several fixes, while a crossing road is close only at the junction |
| Road context | arc length, distance to the arc's end, one-way flag, out-degree of the next junction | Junctions and short arcs are where mistakes concentrate |

The features clearly separate the classes. Train medians, correct vs wrong candidate:
- heading difference: **6° vs 103°**
- distance beyond the best candidate: **0 m vs 13 m**
- neighbour distance: **8 m vs 26 m**

**Baseline: logistic regression** (StandardScaler + L2, C = 1, no class re-weighting)
| Metric | Value | Partition |
|---|---|---|
| CV macro-F1 | **0.905 ± 0.005** (folds 0.900–0.912) | train, GroupKFold (5) by trip |
| CV top-1 accuracy | 0.882 | train |
| Test macro-F1 / log loss | 0.900 / 0.135 | student test |
| **Test top-1 accuracy** | **0.864** | student test |
| Test route continuity (classifier only, no HMM yet) | 0.967 | student test |
| Test route F1 (arc set vs reference route) | 0.854 | student test |
| Training / inference time | 2.3 s / 0.01 s for 119,602 candidate rows | |
| *Reference: "always pick the nearest road"* | *top-1 0.536* | *student test* |

**What this shows**
- Picking the nearest road is close to a **coin flip on two-way streets**, because both directed arcs are equally close. The direction features take accuracy from **54% to 86%**.
- The CV folds are stable (± 0.005), so the model isn't overfitting to particular trips.
- Continuity is already 96.7%, but the remaining 3.3% are jumps between unconnected roads. That gap is exactly what the HMM (P3) is meant to close.
- **Next (P2):** all 6 model families with CV (logistic regression, Naive Bayes, KNN, random forest, LightGBM, MLP), comparison plots, and Bayesian tuning of the best one. All 6 already run end to end.

Source: `results/model_comparison.csv`, `results/cv_folds.csv`, `results/reference_baselines.csv`, produced by `python -m scripts.train_models --models logreg`.

### 5.4 Deployment (two laptops)
```
 Laptop 2: UI (Streamlit)                      Laptop 1: API (FastAPI)
 ┌────────────────────────┐  GET /trips        ┌───────────────────────────────────┐
 │ list of 10 eval trips  │ ─────────────────► │ load trip (no route)              │
 │ select trip → Predict  │  POST /predict     │ candidates → model → Viterbi      │
 │ map + metrics          │ ◄───────────────── │ accuracy computed SERVER-SIDE     │
 └────────────────────────┘  route + metrics   │ (route file never sent to the UI) │
                                               │ every request logged in terminal  │
                                               └───────────────────────────────────┘
```

### 5.5 Leakage prevention
- The prediction pipeline loads trips **without** `.route`. Only the label builder and the server-side evaluator read it.
- Train is used for fitting, CV and tuning. Student test is used for comparison. Evaluator trips are never used in development.
- Every selection we made (k, r, the exclusion threshold) was made on **train** only.
- No trip IDs are hardcoded. Exclusions are rule-based.

---

## 6. Maps (P4)
*Show P4's Folium maps here. What to point out on each one:*

| Map | What to point out |
|---|---|
| Raw GPS + road network + reference route (a train trip) | The GPS points drift off the roads. The reference route follows the roads. Every trip ships its own network. |
| Close-up of a two-way street or junction | Two directed arcs sit on top of each other, so heading is needed to tell them apart |
| Shifted trip 00000054 | The whole GPS track sits about 140 m away from every road. This is why the trip is excluded. |
| Trip with a large gap (00000090 or 00000078) | Why we split tracks into segments |

---

## 7. Status
| Part | Owner | State |
|---|---|---|
| Repo structure, interface contracts, tests | Team | ✅ Done |
| Projection, network, labels, candidates, training tables | P1 | ✅ Done and merged |
| EDA notebook and Folium maps | P4 | 🔄 In progress |
| Features, metrics, 6-model setup, baseline (top-1 0.864) | P2 | ✅ Done |
| All 6 models compared, Bayesian tuning, final model export | P2 | ⏳ Next |
| HMM/Viterbi and the inference pipeline | P3 | 🔄 In progress |
| FastAPI and Streamlit UI | P4 | 🔄 In progress |

---

## 8. Likely questions and short answers
**Q: Why not use plain lat/lon distance?**
Degrees aren't metres. A degree of longitude shrinks with latitude. We project each trip into metres (AEQD centred on that trip), and the error is under 1 cm over 62 km.

**Q: Why not one UTM zone?**
The trips span 237° of longitude, and one UTM zone covers only 6°.

**Q: How did you get labels per GPS point when `.route` only lists arcs?**
Monotone DP alignment. Each point is matched to a position on the route that never moves backwards, minimising total distance. We validated it with distance (about 4.6 m) and direction agreement (98.6%).

**Q: Why forward-only and not just "nearest route arc"?**
Routes revisit streets (loops, or there-and-back in opposite directions). Nearest-arc matching would mix up the passes and the directions.

**Q: Isn't excluding trips cheating?**
The rule is on train data only and based on the data, not on IDs: a trip is dropped if fewer than 50% of its points can be matched to its own reference route. Exactly 2 trips fail it, and both have a 100+ m GPS shift. On evaluator trips nothing is excluded, so the model must cope.

**Q: Why k = 8 and 60 m?**
From a sweep on train. It misses the true road for only 0.46% of points, and larger k costs much more compute for little gain.

**Q: What if a point has no candidates?**
That happens for 0.70% of train points. Those points get no rows. The HMM gives them "no match" and continues from the next point.

**Q: Are your labels ground truth?**
They're derived from a reference route that is itself 98–100% connected, so we treat them as strong reference labels, not absolute truth. Points at junctions are naturally ambiguous.

**Q: Why does the model beat "nearest road" so much (86% vs 54%)?**
A two-way street is stored as two directed arcs with identical geometry, so the nearest-road rule ties and guesses the direction. Our heading features compare the vehicle's direction of travel (over ±3 s) with each arc's direction, which resolves the tie.

**Q: Why no class weighting when only 16% of rows are positive?**
The model's probabilities become the HMM's emission probabilities, and re-weighting would distort them. Top-1 accuracy depends only on the ranking of candidates within each point, which weighting barely changes.

**Q: Why GroupKFold?**
Fixes from the same trip are strongly correlated (neighbouring seconds). Splitting by trip makes sure validation trips are completely unseen, which matches the real situation of predicting a new trip.

**Q: Why HMM after a classifier?**
The classifier scores each point independently, so it can jump between roads that don't connect. The HMM adds sequence and road-network connectivity, so the final route is drivable.
