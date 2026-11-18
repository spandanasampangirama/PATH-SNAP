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
