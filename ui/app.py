"""Evaluator UI — runs on Laptop 2.  Owner: P4

Run:  PATHSNAP_API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py
Talks to the API only (never reads dataset files). Shows the trip list from GET /trips,
sends the chosen trip to POST /predict, draws GPS points + predicted route on a Folium map,
and shows the metrics/timings returned by the API.
"""
from __future__ import annotations

import os
import time

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

API = os.environ.get("PATHSNAP_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="PathSnap — Map-Matching & Route Reconstruction",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polish
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .badge-ok { background-color: #dcfce7; color: #166534; }
    .badge-warn { background-color: #fef9c3; color: #854d0e; }
    .badge-err { background-color: #fee2e2; color: #991b1b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar: API Health & Trip Selection
# ---------------------------------------------------------
with st.sidebar:
    st.title("🛰️ PathSnap")
    st.caption("Distributed Map-Matching Evaluation")
    st.markdown("---")

    # API Connection Check
    health_data = None
    try:
        r_health = requests.get(f"{API}/health", timeout=3.0)
        if r_health.status_code == 200:
            health_data = r_health.json()
    except Exception:
        health_data = None

    if health_data:
        st.markdown(
            f'API Status: <span class="status-badge badge-ok">Online ({API})</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"**Model:** `{health_data.get('model_name') or 'Default'}` | "
            f"**Available Trips:** `{health_data.get('n_trips', 0)}`"
        )
    else:
        st.markdown(
            f'API Status: <span class="status-badge badge-err">Offline ({API})</span>',
            unsafe_allow_html=True,
        )
        st.error(f"Cannot connect to API at {API}. Ensure Laptop 1 has started `api.main:app`.")
        if st.button("Retry Connection"):
            st.rerun()
        st.stop()

    st.markdown("---")

    # Fetch Trips
    try:
        r_trips = requests.get(f"{API}/trips", timeout=10.0)
        r_trips.raise_for_status()
        trips_payload = r_trips.json()
        trips_list = trips_payload.get("trips", [])
    except Exception as e:
        st.error(f"Failed to fetch trips from API: {e}")
        st.stop()

    if not trips_list:
        st.warning("No trips available in evaluation directory.")
        st.stop()

    # Formatted options
    trip_map_dict = {
        f"Trip {t['trip_id']} ({t['n_points']} pts)": t["trip_id"] for t in trips_list
    }
    selected_label = st.selectbox("Select Evaluation Trip", list(trip_map_dict.keys()), index=0)
    selected_trip_id = trip_map_dict[selected_label]

    st.markdown("---")
    st.subheader("Display Layers")
    show_gps = st.checkbox("Show GPS Observations", value=True)
    show_pred = st.checkbox("Show HMM Predicted Route", value=True)
    show_cls = st.checkbox("Show Classifier-Only Route", value=True)
    show_markers = st.checkbox("Start / End Markers", value=True)

    st.markdown("---")
    run_btn = st.button("🚀 Run Map-Matching", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Main Panel: Prediction & Visualisation
# ---------------------------------------------------------
st.header("Trajectory Map-Matching & Route Reconstruction")

# Session state initialization
if "last_pred" not in st.session_state:
    st.session_state["last_pred"] = None
if "current_trip_id" not in st.session_state:
    st.session_state["current_trip_id"] = None

# Automatically run on click or if trip changed
if run_btn or (st.session_state["current_trip_id"] != selected_trip_id and st.session_state["last_pred"] is None):
    with st.spinner(f"Requesting map-matching prediction for trip {selected_trip_id} from {API}..."):
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{API}/predict",
                json={"trip_id": selected_trip_id},
                timeout=60.0,
            )
            roundtrip_ms = (time.perf_counter() - t0) * 1000.0

            if resp.status_code == 200:
                pred_data = resp.json()
                pred_data["roundtrip_ms"] = roundtrip_ms
                st.session_state["last_pred"] = pred_data
                st.session_state["current_trip_id"] = selected_trip_id
                st.success(f"Trip {selected_trip_id} successfully reconstructed in {roundtrip_ms:.1f} ms!")
            elif resp.status_code == 503:
                st.error("API Error 503: Model pipeline not loaded yet on Laptop 1.")
            elif resp.status_code == 501:
                st.warning("API Error 501: Predict endpoint is not implemented on server.")
            else:
                st.error(f"API Error {resp.status_code}: {resp.text}")
        except requests.exceptions.Timeout:
            st.error("Request timed out (60s). Large road network lookup may be in progress.")
        except Exception as e:
            st.error(f"Failed to communicate with API: {e}")

# Display results
pred = st.session_state.get("last_pred")

if pred and pred.get("trip_id") == selected_trip_id:
    n_points = pred.get("n_points", 0)
    gps_points = pred.get("gps_points", [])  # [[lat, lon], ...]
    predicted_route = pred.get("predicted_route", [])  # [[lat, lon], ...]
    classifier_route = pred.get("classifier_route") or []
    metrics = pred.get("metrics", {})
    timings = pred.get("timings_ms", {})

    # Overview metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total GPS Points", n_points)
    with c2:
        pred_arcs = len(predicted_route) // 2 if predicted_route else 0
        st.metric("Predicted Route Vertices", len(predicted_route))
    with c3:
        hmm_acc = metrics.get("top1_acc_hmm")
        st.metric("HMM Top-1 Accuracy", f"{hmm_acc * 100:.1f}%" if hmm_acc is not None else "N/A")
    with c4:
        hmm_cont = metrics.get("route_continuity_hmm")
        st.metric("Route Continuity", f"{hmm_cont * 100:.1f}%" if hmm_cont is not None else "N/A")

    # Layout: Map on left (or full), Metrics on right
    map_col, stats_col = st.columns([7, 3])

    with map_col:
        # Build Folium map for display
        if gps_points:
            center_lat = sum(p[0] for p in gps_points) / len(gps_points)
            center_lon = sum(p[1] for p in gps_points) / len(gps_points)
        elif predicted_route:
            center_lat = sum(p[0] for p in predicted_route) / len(predicted_route)
            center_lon = sum(p[1] for p in predicted_route) / len(predicted_route)
        else:
            center_lat, center_lon = 0.0, 0.0

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles="OpenStreetMap",
            control_scale=True,
        )

        all_lats, all_lons = [], []

        # 1. Classifier-only route
        if show_cls and classifier_route:
            folium.PolyLine(
                locations=classifier_route,
                color="#805AD5",
                weight=3.5,
                dash_array="6, 6",
                opacity=0.8,
                tooltip="Classifier-Only Route (Argmax)",
            ).add_to(m)
            for p in classifier_route:
                all_lats.append(p[0])
                all_lons.append(p[1])

        # 2. Predicted route
        if show_pred and predicted_route:
            folium.PolyLine(
                locations=predicted_route,
                color="#E53E3E",
                weight=5.0,
                opacity=0.9,
                tooltip="Predicted Route (HMM Viterbi)",
            ).add_to(m)
            for p in predicted_route:
                all_lats.append(p[0])
                all_lons.append(p[1])

        # 3. GPS observations
        if show_gps and gps_points:
            folium.PolyLine(
                locations=gps_points,
                color="#3182CE",
                weight=1.5,
                dash_array="3, 4",
                opacity=0.5,
            ).add_to(m)

            for i, pt in enumerate(gps_points):
                all_lats.append(pt[0])
                all_lons.append(pt[1])
                folium.CircleMarker(
                    location=pt,
                    radius=3.5,
                    color="#2B6CB0",
                    fill=True,
                    fill_color="#4299E1",
                    fill_opacity=0.85,
                    weight=1,
                    tooltip=f"GPS #{i} ({pt[0]:.5f}, {pt[1]:.5f})",
                ).add_to(m)

        # 4. Start / End Markers
        if show_markers and gps_points:
            folium.Marker(
                location=gps_points[0],
                popup="<b>Trip Start</b>",
                tooltip="Start",
                icon=folium.Icon(color="green", icon="play", prefix="fa"),
            ).add_to(m)
            folium.Marker(
                location=gps_points[-1],
                popup="<b>Trip End</b>",
                tooltip="End",
                icon=folium.Icon(color="red", icon="flag", prefix="fa"),
            ).add_to(m)

        # Fit bounds
        if all_lats and all_lons:
            m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])

        st_folium(m, width="100%", height=560, returned_objects=[])

    with stats_col:
        st.subheader("Performance & Metrics")
        
        # Server-side Evaluation Metrics
        st.markdown("**Server-Side Reference Comparison:**")
        cls_acc = metrics.get("top1_acc_classifier")
        hmm_acc = metrics.get("top1_acc_hmm")
        cls_cont = metrics.get("route_continuity_classifier")
        hmm_cont = metrics.get("route_continuity_hmm")
        miss_rate = metrics.get("candidate_miss_rate")
        f1 = metrics.get("route_f1")

        st.write(
            {
                "Top-1 Classifier Acc": f"{cls_acc:.3f}" if cls_acc is not None else "N/A",
                "Top-1 HMM Acc": f"{hmm_acc:.3f}" if hmm_acc is not None else "N/A",
                "Continuity (Classifier)": f"{cls_cont:.3f}" if cls_cont is not None else "N/A",
                "Continuity (HMM)": f"{hmm_cont:.3f}" if hmm_cont is not None else "N/A",
                "Candidate Miss Rate": f"{miss_rate:.3f}" if miss_rate is not None else "N/A",
                "Route F1 Score": f"{f1:.3f}" if f1 is not None else "N/A",
            }
        )

        st.markdown("---")
        st.subheader("Latency Breakdown")
        if timings:
            for k, v in timings.items():
                st.text(f"• {k}: {v:.1f} ms")
        if "roundtrip_ms" in pred:
            st.info(f"⚡ Total Round-Trip: {pred['roundtrip_ms']:.1f} ms")
else:
    st.info("Select a trip from the sidebar and click **🚀 Run Map-Matching** to evaluate.")
