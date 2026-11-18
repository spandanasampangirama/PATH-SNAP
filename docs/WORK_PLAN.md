# PathSnap — Work Plan (4 people)

Assign names to P1–P4 here: **P1 = ___  P2 = ___  P3 = ___  P4 = ___**

## Pipeline and who owns each stage
```
raw trip ──► P1: project + index + candidates + labels ──► candidate table (parquet)
                                                               │
                                       P2: features ──► 5+ models ──► Optuna ──► model.joblib
                                                                                     │ probs
                         P3: routing + HMM/Viterbi ◄─────────────────────────────────┘
                                 │  PathSnapPredictor.predict()
             P4: FastAPI (Laptop 1) ──HTTP──► Streamlit UI (Laptop 2)       P4: EDA + Folium maps
```

## File ownership
Each file has one owner, and only the owner edits it. That is what keeps merges free of conflicts.

| Owner | Role | Files they own | Report sections |
|---|---|---|---|
| **P1** | Data and geometry (critical path) | `pathsnap/config.py, io.py, geo.py, network.py, labels.py, candidates.py`, `scripts/build_dataset.py`, `notebooks/p1_*.ipynb` | §2.1 preprocessing, projection choice, how labels were derived, data limitations |
| **P2** | Features, models, tuning | `pathsnap/features.py, models.py, metrics.py`, `scripts/train_models.py, bayes_opt.py`, `results/model_*`, `results/optuna/`, `artifacts/model*` | §2.2 features, §3 (all 5+ models), §4 Bayesian optimisation |
| **P3** | HMM and inference pipeline | `pathsnap/routing.py, hmm.py, pipeline.py, evaluate.py`, `scripts/eval_hmm.py`, `tests/test_e2e.py`, `artifacts/hmm_config.json` | §5 HMM/Viterbi, classifier-only vs HMM comparison |
| **P4** | EDA, maps, API, UI, deployment | `notebooks/p4_eda.ipynb`, `pathsnap/viz.py`, `scripts/make_maps.py`, `api/*`, `ui/*`, `README.md` | §2 EDA plots, §6 Folium, §7 API/UI screenshots, **putting the final PDF together** |

**Shared files:** `configs/default.yaml`, `requirements.txt`, `pathsnap/schema.py`, `docs/*`.
- Edit these only in a small PR of its own, and say so in chat.
- You can add new keys. Don't rename or remove existing ones.

---

## Phase 0: setup (everyone, about 30 min)
1. One person pushes this scaffold to `main` on GitHub (public repo) and adds the other three as collaborators.
2. Everyone runs:
   ```bash
   git clone <repo> && cd PathSnap
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   # copy the organisers' dataset/ folder into the repo root (it is gitignored)
   pytest -q tests/test_contracts.py        # must print 4 passed
   git checkout -b p1/data    # or p2/models, p3/hmm, p4/app
   ```
3. Read `docs/CONTRACTS.md` together (10 min). After this the contracts are frozen.

## Phase 1: until the first-phase evaluation (before lunch)
Goal for the evaluation: **EDA plus Folium maps, candidates and labels for all 90 trips, a baseline classifier, a distance-only HMM on one trip, and an API that lists trips to the UI.**

| | Tasks, in this order | Unblocks |
|---|---|---|
| **P1** | 1. `geo.py` (per-trip AEQD/UTM projection, point-to-segment distance, bearing). 2. `network.py` (projected nodes, arc lengths and bearings, spatial index, `query_arcs`, `successor_arcs`). **Push to main as soon as these work (around 1 h).** 3. `candidates.py` with `segment_track`. 4. `labels.py` (monotone alignment of points to route arcs). 5. `build_dataset.py` → `candidates_{train,test}.parquet`. Report the **candidate miss rate** for k and radius. | P2 and P3 both need step 2. P2 needs step 5. |
| **P2** | 1. `metrics.py` (pure numpy; test with small hand-made arrays). 2. `models.py` (6 models). 3. `train_models.py` harness with GroupKFold over trip_id, timing, and CSV/plot outputs, tested on a **synthetic DataFrame with the schema columns** until P1's parquet is ready. 4. Once the parquet exists: `features.py` v1, then a logistic regression baseline on real data. | P3 needs a model (it can use distance emissions until then). |
| **P3** | 1. `hmm.py` Viterbi in log space on a **hand-made toy candidate table** (3 points × 3 candidates). 2. `routing.py` bounded Dijkstra over directed arcs (use `scipy.sparse.csgraph` or heapq). 3. Wire the two together with **distance-only emissions** on trip `00000046` using P1's network. 4. Gap splitting, dead-end restart, no-candidate points. | |
| **P4** | 1. `notebooks/p4_eda.ipynb` using `io.py` (ready now) to cover everything in Dataset Exploration §4: inventory, points per trip, dt histogram, large gaps (trip 00000090 has a 42 h gap, 00000078 a 14.7 h gap), extents, network sizes, connected components (`scipy.sparse.csgraph.connected_components`), route contiguity, NaN check. **State the partition for every analysis.** 2. `viz.trip_map` with raw GPS, the nearby network and the reference route (train trips only). 3. API `/health` and `/trips` (already working) plus a UI that lists trips and has the map panel laid out. | |

## Phase 2: after lunch, real models
| | Tasks |
|---|---|
| **P1** | Tune k and radius against candidate miss rate vs table size on **train**. Speed up the 300k-arc trips. Write report §2.1. Then help P2 with features (heading on one-way roads, node degree), but **only through PRs into P2's files that P2 reviews**. |
| **P2** | Train all 6 models. Produce `results/model_comparison.csv` with every column of the report §3.1 table, the CV-F1 / test-F1 / top-1 / log-loss bar charts, confusion matrices, and the MLP loss curve (MLP only, since it's the only model with epoch history). Pick a model and **write down why** (log loss matters for HMM emissions). Run Optuna TPE on train with GroupKFold for at least 50 trials. Export `artifacts/model.joblib` and `model_meta.json`. Write the tuned vs untuned table. |
| **P3** | Plug in the classifier probabilities. Tune sigma, beta and emission mode on **train/test only**. Run `eval_hmm.py` → per-trip classifier-only vs HMM table, including cases where HMM did *not* help. `route_arcs` gap filling. Make `test_e2e.py` pass. Export `hmm_config.json`. |
| **P4** | Wire `PathSnapPredictor` and `evaluate` into `/predict`. Finish the UI (selector, map with legend and layers, metrics, timings). **Two-laptop LAN test** (see the checklist below). Run `make_maps.py` to produce report maps from the real predictions. Write README run instructions. |

## Phase 3: freeze, report, rehearsal
1. **Code freeze:** everything is merged, `pytest -q` passes (contracts and e2e), and the final artifacts are committed.
2. Fresh-clone test: on a laptop that isn't the developer's, follow the README from scratch, start the API, and get predictions for all 10 test trips.
3. Each owner writes their sections into `report/sections/*.md` using **only numbers from committed CSVs**. P4 assembles the PDF. Add an AI-assistance disclosure section.
4. **Teach-back (required, because the viva can ask anyone anything):** each owner spends 10 min explaining their module to the other three. Everyone should be able to explain candidates, labels, features, why the chosen model won, what Optuna searched, emissions and transitions, Viterbi, and the API flow.

---

## Git workflow (so everything merges)
- Branches: `p1/data`, `p2/models`, `p3/hmm`, `p4/app`. **Never commit directly to `main`.**
- Merge into main through a PR at every checkpoint, or earlier when a contract dependency is ready (P1's `network.py`).
  ```bash
  git pull origin main          # bring main into your branch first and resolve anything locally
  pytest -q                     # must pass
  git push && open PR -> one teammate approves -> merge
  ```
- Pull `main` into your branch at least every 1–2 hours.
- Commit small and often **under your own GitHub account**. Organisers look at repository history.
- Notebooks: one owner per notebook (`p1_`, `p4_` prefix) so two people never edit the same `.ipynb`.
- Don't commit: `dataset/`, `data/processed/`, `.env`, or any file over 50 MB.
- Do commit: `results/` (CSVs, plots, maps, Optuna trials). These are the evidence behind every number in the report.

## Checkpoints
| | When | Done means |
|---|---|---|
| C0 | start + 30 min | Scaffold on main, 4 branches, contract tests pass on all laptops |
| C1 | before lunch (phase-1 evaluation) | EDA notebook and Folium maps, parquet for 90 trips, baseline model numbers, distance-HMM on one trip, UI lists trips from the API |
| C2 | mid-afternoon | 5+ models compared, model exported, `/predict` works end to end, two-laptop test passed |
| C3 | before submission | Optuna done, HMM comparison done, e2e test passes, README fresh-clone verified, report assembled, teach-back done |

## Demo-day checklist (Laptop 1 = API, Laptop 2 = UI)
- Both laptops on the same Wi-Fi, with a phone hotspot as backup. On Laptop 1, get the IP with `ipconfig getifaddr en0` (macOS) or `ipconfig` (Windows).
- Laptop 1: `PATHSNAP_EVAL_DIR=<evaluator trips> uvicorn api.main:app --host 0.0.0.0 --port 8000`. Allow port 8000 through the firewall. **Keep the terminal visible, because the evaluators must see the request logs.**
- Laptop 2: `PATHSNAP_API_URL=http://<ip>:8000 streamlit run ui/app.py`.
- Before the evaluators arrive, run all 10 trips once and note the slowest. The largest networks (around 300k arcs) are the risk. Caching road networks at startup is fine. **Never cache or precompute predictions.**
- `pip freeze` must match on the laptop that trained the model and Laptop 1, otherwise the joblib model may fail to load.
