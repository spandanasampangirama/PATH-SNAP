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
