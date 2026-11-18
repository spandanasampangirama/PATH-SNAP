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
