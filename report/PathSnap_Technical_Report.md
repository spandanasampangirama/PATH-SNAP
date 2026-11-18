# PathSnap: Distributed Map-Matching & Route Reconstruction
**Technical Report & System Architecture**  
**Course / Event:** UE24CS352A Machine Learning Hackathon, PES University  
**Date:** November 2026  

---



# Section 2: Dataset Understanding and Exploratory Data Analysis

## 2.1 Dataset Inventory & Data Integrity Verification

The PathSnap dataset comprises vehicle trajectories and directed road networks distributed worldwide. The dataset is partitioned into two developmental sets:
- **Training partition (`dataset/train/`):** 80 trips used for spatial indexing, feature extraction, cross-validation, and model training.
- **Student-Testing partition (`dataset/test/`):** 10 trips held out for developmental evaluation and decoder validation.

Each trip directory $T$ contains four fundamental files:
1. `$T.track`: GPS observations consisting of whitespace-separated `(longitude, latitude, timestamp_seconds)`.
2. `$T.nodes`: Road network vertices consisting of `(longitude, latitude)`, indexed by 0-based row ID (`node_id`).
3. `$T.arcs`: Directed road network edges consisting of `(from_node_id, to_node_id)`, indexed by 0-based row ID (`arc_id`).
4. `$T.route`: Ground-truth chronological sequence of traversed `arc_id` values (available in train and student-test for server-side evaluation).

### Comprehensive Partition Summary
The complete data audit computed across all 90 trips yielded the following verifiable statistics:

| Partition | Trips | Total GPS Observations | Avg Obs / Trip | Min / Max Obs | Avg Road Nodes | Avg Directed Arcs | 1 Hz Steps (%) | Avg Route Contiguity (%) |
|---|---|---|---|---|---|---|---|---|
| **Train** | 80 | 208,417 | 2,605.2 | 269 / 7,201 | 24,168.6 | 51,972.1 | 93.83% | 99.93% |
| **Student-Test** | 10 | 19,160 | 1,916.0 | 728 / 3,601 | 9,068.9 | 19,503.2 | 98.20% | 99.97% |
| **Total / Overall** | **90** | **227,577** | **2,528.6** | **269 / 7,201** | **22,490.8** | **48,364.5** | **94.32%** | **99.94%** |

### Missing Value (NaN / Inf) Audit
Across all 360 individual files (227,577 trajectory points, 2,024,176 road nodes, 4,352,805 road arcs, and 48,120 reference route edges):
- **NaN / Null values:** Exactly **0** detected.
- **Infinite / Corrupt entries:** Exactly **0** detected.
- **File completeness:** All 90 trips possess intact `.track`, `.nodes`, `.arcs`, and `.route` files.

---

## 2.2 Trajectory Sampling Characteristics & Temporal Anomalies

### 1 Hz Sampling Baseline
Nominally, the trajectory logger operates at 1 Hz. As shown in our sampling interval distribution analysis:
- Over **94.3%** of all consecutive timestamps across the dataset exhibit $\Delta t = t_{i} - t_{i-1} = 1.0\text{ s}$ exactly.
- In student-test trips, the regular 1 Hz compliance reaches **98.20%**.

### Extreme Satellite Signal Gaps
Despite nominal 1 Hz logging, several trips exhibit severe tracking blackouts caused by tunnels, urban canyons, receiver power-saving modes, or prolonged vehicle stops:

| Trip ID | Partition | Total GPS Points | Max Single Gap (s) | Max Single Gap (Hours) | Gaps > 60 s |
|---|---|---|---|---|---|
| **`00000090`** | Train | 536 | 153,064.0 s | **42.52 hours** | 1 |
| **`00000078`** | Train | 2,909 | 52,810.0 s | **14.67 hours** | 1 |
| **`00000062`** | Train | 1,441 | 18,342.0 s | **5.10 hours** | 1 |
| **`00000049`** | Train | 2,216 | 7,105.0 s | **1.97 hours** | 1 |
| **`00000017`** | Train | 3,121 | 3,612.0 s | **1.00 hours** | 1 |

### Architectural Implication: Temporal Track Splitting
Attempting to evaluate Hidden Markov Model transitions across a 42-hour blackout will severely distort the Viterbi decoding path, as routing distances between observations recorded days apart lose topological correlation.  
**System Decision:** As defined in `configs/default.yaml` (`gaps.split_s = 60.0`), any gap $\Delta t > 60\text{ s}$ splits the trajectory into independent HMM decoding sub-segments (`segment_id`). Viterbi decoding restarts afresh at the beginning of each contiguous segment.

---

## 2.3 Geographic Coordinate Distribution & Projection Justification

### Worldwide Spatial Dispersion
The 90 trajectories are geographically distributed across disparate locations globally:
- **Longitude Range:** $-100.29^\circ$ to $+137.41^\circ$
- **Latitude Range:** $-34.93^\circ$ to $+62.38^\circ$

Because trajectories occur across North America, Europe, Asia, and the Southern Hemisphere, **no single global UTM zone or planar projection can be chosen**. Applying a single fixed projection (e.g., UTM Zone 32N) would introduce massive distance distortions in distant zones, warping road lengths and perpendicular distance calculations.

### Local Metric Projection Architecture
To ensure millimeter-level Euclidean spatial calculations, every trip is projected into a **per-trip local coordinate system** centered on its trajectory centroid:
$$(\lambda_0, \phi_0) = \left( \frac{1}{N}\sum_{i=1}^N \lambda_i, \; \frac{1}{N}\sum_{i=1}^N \phi_i \right)$$
Using an **Azimuthal Equidistant (AEQD)** or local Transverse Mercator projection centered at $(\lambda_0, \phi_0)$:
- Distances and bearings from the center point are preserved without distortion.
- Local distances in metres represent true ground distances accurately across the entire bounding extent of the trip.
- Raw longitude/latitude coordinates are converted to projected Cartesian coordinates $(x, y) \in \mathbb{R}^2$ prior to candidate querying and feature engineering.

---

## 2.4 Road Network Complexity & Graph Connectivity

### Graph Scale
Road network complexity varies drastically across trips:
- **Smallest network:** Trip `00000046` with **649 directed arcs** and **320 nodes** (ideal for rapid unit testing and benchmarking).
- **Largest network:** Trip `00000084` with **307,432 directed arcs** and **135,119 nodes**.
- **Average network:** 48,364 directed arcs and 22,491 nodes.
- **Edge-to-Node ratio:** $E/V \approx 2.15$, consistent with sparse planar road networks featuring one-way streets and divided multilane highways.

### Directed vs. Undirected Graph Structure
Arcs in `.arcs` represent directed road segments (`from_node -> to_node`). Analysis reveals that the fraction of reversible arcs ($u \to v$ and $v \to u$ both existing) varies between 7% and 100% depending on urban density and highway topology.

### Connected Components
Applying `scipy.sparse.csgraph.connected_components`:
- Most networks contain between 1 and 18 connected components.
- Disconnected components represent isolated service roads, pedestrian paths, or partitioned highway ramps within the extracted bounding box.
- Bounded Dijkstra routing must handle cases where two candidate road segments belong to different connected components by falling back to a penalty transition probability.

---

## 2.5 Reference Route Structure & Ground-Truth Limitations

### Route Contiguity Verification
For all training trips and student-test trips, we checked whether consecutive arcs in `.route` satisfy topological connectivity:
$$\text{to\_node}(arc_i) == \text{from\_node}(arc_{i+1})$$
- **Mean Route Contiguity:** **99.94%** across all trips.
- **Strictly Contiguous Trips:** 83 out of 90 trips exhibit 100.0% contiguity.
- In the remaining 7 trips, isolated non-contiguous transitions occur at complex multigrade junctions or roundabouts where intermediate connector micro-arcs were omitted from the logger.

### Practical Data Limitations
1. **GPS Noise & Urban Multipath:** GPS observations deviate up to 30–50 metres from the road centerline due to atmospheric delay and building reflections.
2. **Parallel Road Ambiguity:** Service lanes, freeway expressways, and frontage roads running parallel within 15 metres require multi-candidate consideration and speed/heading features to differentiate.
3. **Reference Route Inconsistencies:** Because reference routes represent logged device traces rather than physical surveyed alignments, evaluation metrics must measure topological agreement rather than pixel-level coordinate equality.


---


# Section 6: Geographical Visualisation Using Folium

## 6.1 Folium Mapping Architecture & Design Principles

Geographical visualisation is implemented in `pathsnap/viz.py` via the `trip_map()` engine, which interfaces with Folium to produce standalone interactive HTML maps and embedded views for the Streamlit UI.

### Coordinate Order Handling
The raw data files store coordinates as `(longitude, latitude)`. However, Leaflet and Folium require coordinates in `[latitude, longitude]` order. To prevent inversion errors and guarantee strict consistency across the codebase, coordinate swapping is enforced strictly inside `viz.py` and `RoadNetwork.arc_polyline_lonlat()`:
$$\text{Leaflet Coordinates} = [\text{lat}, \text{lon}] = [\text{track}[:, 1], \text{track}[:, 0]]$$

---

## 6.2 Visual Hierarchy & Layer Styling

To ensure clarity during evaluator inspections and technical report review, each data layer is assigned distinct visual attributes, line weights, and color encodings adhering to WCAG contrast standards:

| Layer Name | Visual Element | Color Code | Styling Parameters | Semantic Meaning |
|---|---|---|---|---|
| **Raw GPS Points** | Circular Markers & Polyline | `#4299E1` (Blue) | Radius: 3.5 px, Weight: 1.5 px, Opacity: 0.85, Dashed line | Discrete vehicle GPS observations with tooltips showing index and timestamp |
| **Start / End Markers** | FontAwesome Icons | Green (`play`) & Red (`flag`) | Pulsing marker icons | Marks exact trajectory origin and termination |
| **Road Network** | Thin Polylines | `#718096` (Slate Grey) | Weight: 2.0 px, Opacity: 0.60 | Nearby physical road segments available for candidate matching |
| **Reference Route** | Bold Polyline | `#2E7D32` (Forest Green) | Weight: 5.0 px, Opacity: 0.85 | Ground-truth route traversed by the vehicle (Train & Test evaluation only) |
| **Predicted Route (HMM)** | Bold Polyline | `#E53E3E` (Crimson Red) | Weight: 4.5 px, Opacity: 0.90 | Continuous topologically valid route reconstructed by HMM Viterbi decoding |
| **Classifier-Only Route** | Dashed Polyline | `#805AD5` (Purple) | Weight: 3.5 px, Dash: `6, 6`, Opacity: 0.80 | Pointwise argmax candidate predictions without sequential transition constraints |

---

## 6.3 Interactive Components & Usability Features

1. **Floating Interactive Legend:**  
   Injected into the DOM via a Leaflet `MacroElement` positioned at the bottom right. The legend provides visual color swatches for each active layer and is responsive across varying display resolutions.
2. **Dynamic Extent Framing (`fit_bounds`):**  
   Automatically calculates the collective spatial bounding box across all GPS coordinates and route vertices, applying a 5% margin padding:
   $$\text{Bounds} = \left[ [\phi_{\min} - \delta_\phi, \lambda_{\min} - \delta_\lambda], \; [\phi_{\max} + \delta_\phi, \lambda_{\max} + \delta_\lambda] \right]$$
   This guarantees that opening any trip map immediately zooms to the optimal view without requiring manual panning.
3. **Layer Control Toggle (`folium.LayerControl`):**  
   Placed in the top-right corner, allowing evaluators to toggle individual layers on and off to isolate differences between the classifier-only predictions and the HMM Viterbi route.

---

## 6.4 Key Map Case Studies & Analysis

The batch script `scripts/make_maps.py` generates interactive HTML maps in `results/maps/`. Several notable trip visualisations highlight key problem dynamics:

### 1. Small Benchmark Route: Trip `00000046` (`eda_00000046_baseline.html`)
- **Characteristics:** 1,401 GPS observations matched against an urban network of 649 directed arcs.
- **Visual Observations:** Clean grid topology with 66 reference arcs. GPS observations follow arterial roads with minimal multipath distortion. Serves as our primary benchmark for latency profiling and unit testing.

### 2. Extreme GPS Gap Analysis: Trip `00000078` (`eda_00000078_gap_14h.html`)
- **Characteristics:** 2,909 GPS observations, 6,000 localized road arcs, and an extreme **14.67-hour blackout** ($\Delta t = 52,810\text{ s}$).
- **Visual Observations:** The map shows a dense initial cluster followed by a long straight leap across the city where the device ceased recording overnight. Without gap-splitting, a naive shortest-path router would hallucinate a continuous path between disconnected nighttime and morning locations. The Folium layer reveals that track segmentation correctly decouples the two halves.

### 3. Maximum Gap Analysis: Trip `00000090` (`eda_00000090_gap_42h.html`)
- **Characteristics:** 536 GPS observations, 235 reference arcs, and a massive **42.52-hour gap** ($\Delta t = 153,064\text{ s}$).
- **Visual Observations:** The vehicle sat idle over a weekend before resuming travel on Monday morning. The interactive map cleanly visualizes the start cluster and resume cluster, confirming the necessity of time-gap thresholds in HMM decoding.

### 4. Student-Testing Trajectories: Trips `00000000` & `00000007`
- **Characteristics:** 2,503 and 1,354 GPS points respectively.
- **Visual Observations:** Complex highway interchanges featuring elevated overpasses and cloverleaf ramps. The visual comparison clearly demonstrates that pointwise classifier argmax occasionally jumps to the wrong parallel frontage road, whereas the HMM predicted route adheres strictly to valid highway exit ramps.


---


# Section 7: API, UI and Deployment

## 7.1 Distributed System Architecture

PathSnap is designed as a distributed, decoupled client-server application adhering to the hackathon's two-laptop evaluation protocol:

```
+-------------------------------------------------------------------------+
|                       LAPTOP 1: FASTAPI BACKEND                         |
|                                                                         |
|  [Inference Pipeline]                                                   |
|    PathSnapPredictor (model.joblib + hmm_config.json)                   |
|    Spatial RoadNetwork candidate indexing & Dijkstra shortest path      |
|                                                                         |
|  [REST Endpoints]                                                       |
|    GET  /health   -> {status, model_name, eval_dir, n_trips}            |
|    GET  /trips    -> {partition, trips: [{trip_id, n_points}, ...]}     |
|    POST /predict  -> {trip_id, gps_points, predicted_route, metrics}    |
|                                                                         |
|  [Security Boundary]                                                    |
|    Internal evaluation against .route (server-side only)                |
+-------------------------------------------------------------------------+
                                    ▲
                     HTTP Port 8000 │ (LAN / Wi-Fi)
                                    ▼
+-------------------------------------------------------------------------+
|                      LAPTOP 2: STREAMLIT FRONTEND                       |
|                                                                         |
|  • Dynamic Trip Selector populated from GET /trips                      |
|  • Interactive Folium Map rendered via streamlit_folium                 |
|  • Accuracy & Continuity Metrics Dashboard                              |
|  • Pipeline Latency & Timing Breakdown Display                          |
+-------------------------------------------------------------------------+
```

---

## 7.2 Backend Implementation (`api/main.py` & `api/schemas.py`)

The FastAPI service runs on **Laptop 1** and serves map-matching predictions on port 8000.

### 1. Lifespan Model Initialization
Models and HMM configurations are loaded once into memory during application startup using FastAPI's modern `lifespan` handler. This prevents redundant disk reads and guarantees sub-second response times for subsequent inference queries.

### 2. Request Logging & Observability
Every inbound HTTP request is intercepted by an asynchronous timing middleware:
```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - t0) * 1000
    log.info("%s %s from %s -> %d (%.1f ms)", request.method, request.url.path,
             request.client.host if request.client else "unknown",
             response.status_code, duration_ms)
    return response
```
This guarantees that evaluators seated at Laptop 1 can observe real-time terminal output detailing every prediction request, execution time, and server-side metric.

### 3. Strict Data Leakage Prevention
In strict adherence to competition rules:
- Ground-truth reference routes (`.route`) are never transmitted over the wire or exposed in JSON responses.
- The `/predict` endpoint returns only the vehicle's observed `gps_points`, the reconstructed `predicted_route` (and optional `classifier_route`) polylines, aggregate metrics, and timing metadata.
- Evaluation metrics are computed strictly server-side by loading the reference route internally.

---

## 7.3 Frontend Evaluator Interface (`ui/app.py`)

The user interface on **Laptop 2** is built with Streamlit and communicates exclusively over HTTP with Laptop 1 via the `PATHSNAP_API_URL` environment variable.

### Key UI Capabilities:
1. **Live Health & Connectivity Status:** Automatically pings `GET /health`. If the server is unreachable, a warning badge is displayed with a "Retry Connection" action.
2. **Evaluator Trip Selector:** Queries `GET /trips` and generates a formatted dropdown displaying the Trip ID and observation count (e.g., `Trip 00000000 (2503 pts)`).
3. **Interactive Folium Map View:** Employs `streamlit_folium` to display the map with full layer toggling (GPS points, predicted HMM route, classifier route, start/end pins).
4. **Metrics Dashboard:** Highlighting Top-1 Accuracy, Route Continuity, Precision, Recall, F1 score, and Candidate Miss Rate.
5. **Latency Breakdown:** Displays candidate lookup, feature engineering, model inference, and Viterbi decoding times alongside the total network round-trip time.

---

## 7.4 Demo-Day Two-Laptop Deployment Checklist

To ensure flawless execution during live evaluation:

1. **Local Area Network (LAN) Configuration:**
   - Connect Laptop 1 and Laptop 2 to the same local Wi-Fi network (or a dedicated mobile hotspot).
   - On Laptop 1 (Windows), execute `ipconfig` to discover the IPv4 address (e.g., `192.168.1.45`).
2. **Firewall Rule Verification:**
   - Permit inbound TCP traffic on port `8000` on Laptop 1.
3. **Starting Laptop 1 (API Backend):**
   ```bash
   PATHSNAP_EVAL_DIR=/path/to/evaluation_trips uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   *Note: The terminal window must remain open and visible to evaluators.*
4. **Starting Laptop 2 (UI Frontend):**
   ```bash
   PATHSNAP_API_URL=http://192.168.1.45:8000 streamlit run ui/app.py
   ```
5. **Verification Protocol:**
   - Execute all 10 evaluation trips once before evaluator review to warm up the OS file cache and verify memory stability.


---


# Section 8: Academic Integrity & AI-Assistance Disclosure

## 8.1 Statement of Authenticity & Originality

In accordance with the PES University UE24CS352A Assessment and Academic Integrity Policy, all work submitted in this repository represents the original, independent work of this team:
- All data exploration, spatial indexing, candidate generation, feature design, model training, hyperparameter optimisation, HMM sequential decoding, and API/UI deployments were developed specifically for this project.
- No model weights, code, parameters, or precomputed predictions were shared with or obtained from other participating teams.
- Ground truth from the evaluation set was strictly quarantined and never used to fit models, select features, or tune hyperparameters.

---

## 8.2 AI-Assistance Disclosure

In transparency and compliance with competition guidelines, AI assistance (specifically Google Antigravity) was utilized during the development of this project for the following permitted tasks:
1. **Scaffolding and Syntax Acceleration:** Assisting in formulating boilerplate code for FastAPI endpoints (`api/main.py`), Folium HTML legend injection (`pathsnap/viz.py`), and Streamlit state management (`ui/app.py`).
2. **Exploratory Data Analysis Automation:** Generating exploratory plotting scripts and statistical aggregation routines for `notebooks/p4_eda.ipynb` across all 90 trip files.
3. **Documentation and Markdown Structuring:** Assisting with technical documentation formatting, LaTeX mathematical equation rendering, and markdown report drafting.

All architectural decisions, algorithmic designs (spatial candidate lookup, feature engineering, loss formulation, Viterbi log-likelihood recurrence, and distributed client-server separation) were directed, verified, and validated by the team members.


---
