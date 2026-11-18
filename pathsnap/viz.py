"""Folium maps (mandatory for the report) and shared plotting helpers.  Owner: P4

Folium uses [lat, lon] — our raw files are lon, lat. Swap carefully.
Required layers (report section 6): raw GPS points, nearby road network, reference route,
predicted route; distinct colours, legend, LayerControl, fit_bounds to the track.
"""
from __future__ import annotations

import branca.element
import folium
import numpy as np


def _to_latlon_list(coords) -> list[tuple[float, float]]:
    """Convert coordinates to list of (lat, lon). Handles numpy array (lon, lat) or existing list."""
    if coords is None:
        return []
    if isinstance(coords, np.ndarray):
        if coords.ndim == 2 and coords.shape[1] >= 2:
            # Files are lon (col 0), lat (col 1) -> swap to lat, lon
            return [(float(row[1]), float(row[0])) for row in coords]
    if isinstance(coords, (list, tuple)):
        out = []
        for pt in coords:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                out.append((float(pt[0]), float(pt[1])))
        return out
    return []


def trip_map(
    track_lonlat,
    network_lines: list[list[tuple[float, float]]] | None = None,
    reference_line: list[tuple[float, float]] | list[list[tuple[float, float]]] | None = None,
    predicted_line: list[tuple[float, float]] | list[list[tuple[float, float]]] | None = None,
    classifier_line: list[tuple[float, float]] | list[list[tuple[float, float]]] | None = None,
    title: str = "",
) -> folium.Map:
    """Return an interactive folium.Map with distinct layers, legend, and layer control.
    
    Parameters:
    -----------
    track_lonlat : np.ndarray | list
        Raw GPS observations with lon as col 0 and lat as col 1 (or list of [lat, lon]).
    network_lines : list of lists of (lat, lon), optional
        Road network segment polylines in (lat, lon) order.
    reference_line : list of (lat, lon) or list of segments, optional
        Ground truth reference route.
    predicted_line : list of (lat, lon) or list of segments, optional
        Final predicted route from HMM / Viterbi decoder.
    classifier_line : list of (lat, lon) or list of segments, optional
        Argmax classifier-only candidate route for comparison.
    title : str, optional
        Map title displayed in heading and legend.
        
    Returns:
    --------
    folium.Map
        Configured Folium map with fit_bounds applied. Save with m.save(filepath).
    """
    # 1. Parse track points to [lat, lon]
    track_pts: list[tuple[float, float]] = []
    timestamps: list[float] | None = None
    if isinstance(track_lonlat, np.ndarray):
        if track_lonlat.ndim == 2 and track_lonlat.shape[1] >= 2:
            track_pts = [(float(r[1]), float(r[0])) for r in track_lonlat]
            if track_lonlat.shape[1] >= 3:
                timestamps = [float(r[2]) for r in track_lonlat]
    elif isinstance(track_lonlat, (list, tuple)):
        for item in track_lonlat:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                track_pts.append((float(item[0]), float(item[1])))

    # Compute bounding box for map initialization and fit_bounds
    all_lats: list[float] = [p[0] for p in track_pts]
    all_lons: list[float] = [p[1] for p in track_pts]

    def _collect_coords(lines):
        if not lines:
            return
        if isinstance(lines, list) and lines and isinstance(lines[0], (list, tuple)):
            if isinstance(lines[0][0], (int, float)):
                # Flat list of (lat, lon)
                for pt in lines:
                    all_lats.append(float(pt[0]))
                    all_lons.append(float(pt[1]))
            else:
                # Nested list of segments
                for seg in lines:
                    for pt in seg:
                        all_lats.append(float(pt[0]))
                        all_lons.append(float(pt[1]))

    _collect_coords(reference_line)
    _collect_coords(predicted_line)
    _collect_coords(classifier_line)

    if all_lats and all_lons:
        center_lat = float(np.mean(all_lats))
        center_lon = float(np.mean(all_lons))
    else:
        center_lat, center_lon = 0.0, 0.0

    # 2. Base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # 3. Layer: Road Network
    if network_lines:
        fg_net = folium.FeatureGroup(name="Road Network (Grey)", show=True)
        for line in network_lines:
            if len(line) >= 2:
                folium.PolyLine(
                    locations=line,
                    color="#718096",
                    weight=2.0,
                    opacity=0.6,
                    tooltip="Road Segment",
                ).add_to(fg_net)
        fg_net.add_to(m)

    # 4. Layer: Reference Route (Ground Truth)
    if reference_line:
        fg_ref = folium.FeatureGroup(name="Reference Route (Green)", show=True)
        # Check if reference_line is flat list of (lat, lon) or list of segments
        if isinstance(reference_line[0], (list, tuple)) and isinstance(reference_line[0][0], (int, float)):
            folium.PolyLine(
                locations=reference_line,
                color="#2E7D32",
                weight=5,
                opacity=0.85,
                tooltip="Ground Truth Route",
            ).add_to(fg_ref)
        else:
            for seg in reference_line:
                if len(seg) >= 2:
                    folium.PolyLine(
                        locations=seg,
                        color="#2E7D32",
                        weight=5,
                        opacity=0.85,
                        tooltip="Ground Truth Route",
                    ).add_to(fg_ref)
        fg_ref.add_to(m)

    # 5. Layer: Classifier-Only Route
    if classifier_line:
        fg_cls = folium.FeatureGroup(name="Classifier-Only Route (Purple dashed)", show=False)
        if isinstance(classifier_line[0], (list, tuple)) and isinstance(classifier_line[0][0], (int, float)):
            folium.PolyLine(
                locations=classifier_line,
                color="#805AD5",
                weight=3.5,
                dash_array="6, 6",
                opacity=0.8,
                tooltip="Classifier-Only Route (Argmax)",
            ).add_to(fg_cls)
        else:
            for seg in classifier_line:
                if len(seg) >= 2:
                    folium.PolyLine(
                        locations=seg,
                        color="#805AD5",
                        weight=3.5,
                        dash_array="6, 6",
                        opacity=0.8,
                        tooltip="Classifier-Only Route (Argmax)",
                    ).add_to(fg_cls)
        fg_cls.add_to(m)

    # 6. Layer: Predicted Route (HMM / Viterbi)
    if predicted_line:
        fg_pred = folium.FeatureGroup(name="Predicted Route - HMM (Red)", show=True)
        if isinstance(predicted_line[0], (list, tuple)) and isinstance(predicted_line[0][0], (int, float)):
            folium.PolyLine(
                locations=predicted_line,
                color="#E53E3E",
                weight=4.5,
                opacity=0.9,
                tooltip="Predicted Route (HMM Viterbi)",
            ).add_to(fg_pred)
        else:
            for seg in predicted_line:
                if len(seg) >= 2:
                    folium.PolyLine(
                        locations=seg,
                        color="#E53E3E",
                        weight=4.5,
                        opacity=0.9,
                        tooltip="Predicted Route (HMM Viterbi)",
                    ).add_to(fg_pred)
        fg_pred.add_to(m)

    # 7. Layer: Raw GPS Observations
    if track_pts:
        fg_gps = folium.FeatureGroup(name="Raw GPS Points (Blue)", show=True)
        # Connect observations with subtle dashed line
        folium.PolyLine(
            locations=track_pts,
            color="#3182CE",
            weight=1.5,
            dash_array="3, 5",
            opacity=0.5,
            tooltip="GPS Trajectory Path",
        ).add_to(fg_gps)

        # Draw circle markers for GPS points
        n_pts = len(track_pts)
        for i, pt in enumerate(track_pts):
            t_str = f" | t={timestamps[i]:.1f}s" if timestamps and i < len(timestamps) else ""
            popup_html = f"<b>Point {i}/{n_pts}</b><br>Lat: {pt[0]:.6f}<br>Lon: {pt[1]:.6f}{t_str}"
            folium.CircleMarker(
                location=pt,
                radius=3.5,
                color="#2B6CB0",
                fill=True,
                fill_color="#4299E1",
                fill_opacity=0.85,
                weight=1,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"GPS #{i}{t_str}",
            ).add_to(fg_gps)

        # Start and End Markers
        folium.Marker(
            location=track_pts[0],
            popup=f"<b>Trip Start</b><br>Lat: {track_pts[0][0]:.5f}, Lon: {track_pts[0][1]:.5f}",
            tooltip="Start Point",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(fg_gps)

        folium.Marker(
            location=track_pts[-1],
            popup=f"<b>Trip End</b><br>Lat: {track_pts[-1][0]:.5f}, Lon: {track_pts[-1][1]:.5f}",
            tooltip="End Point",
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        ).add_to(fg_gps)

        fg_gps.add_to(m)

    # 8. Legend MacroElement
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 25px;
        right: 25px;
        z-index: 9999;
        background-color: white;
        padding: 12px 16px;
        border: 2px solid #CBD5E0;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 12px;
        line-height: 1.6;
        color: #2D3748;
        max-width: 260px;
    ">
        <div style="font-weight: 700; font-size: 13px; margin-bottom: 8px; border-bottom: 1px solid #E2E8F0; padding-bottom: 4px;">
            {title if title else "PathSnap Map Layers"}
        </div>
        <div><span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:#4299E1; border:1px solid #2B6CB0; margin-right:8px;"></span>Raw GPS Point</div>
        <div><span style="display:inline-block; width:16px; height:3px; background:#718096; margin-right:8px; vertical-align:middle;"></span>Road Network</div>
        {"<div><span style='display:inline-block; width:16px; height:4px; background:#2E7D32; margin-right:8px; vertical-align:middle;'></span>Reference Route</div>" if reference_line else ""}
        {"<div><span style='display:inline-block; width:16px; height:4px; background:#E53E3E; margin-right:8px; vertical-align:middle;'></span>Predicted (HMM)</div>" if predicted_line else ""}
        {"<div><span style='display:inline-block; width:16px; height:3px; border-top:2px dashed #805AD5; margin-right:8px; vertical-align:middle;'></span>Classifier-only</div>" if classifier_line else ""}
    </div>
    """
    m.get_root().html.add_child(branca.element.Element(legend_html))

    # 9. Layer Control
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    # 10. Fit bounds
    if all_lats and all_lons:
        min_lat, max_lat = min(all_lats), max(all_lats)
        min_lon, max_lon = min(all_lons), max(all_lons)
        # Pad bounds slightly
        dlat = max(max_lat - min_lat, 0.001) * 0.05
        dlon = max(max_lon - min_lon, 0.001) * 0.05
        m.fit_bounds([[min_lat - dlat, min_lon - dlon], [max_lat + dlat, max_lon + dlon]])

    return m


def network_to_lines(trip, max_arcs: int | None = 5000) -> list[list[tuple[float, float]]]:
    """Extract road network arc segments as [[(lat, lon), (lat, lon)], ...] from a Trip object.
    
    If max_arcs is specified and the network is very large, arcs close to the trajectory are prioritized.
    """
    nodes = trip.nodes  # (V, 2): lon, lat
    arcs = trip.arcs    # (E, 2): from_node, to_node
    
    selected_indices = range(len(arcs))
    if max_arcs is not None and len(arcs) > max_arcs:
        # Spatial bbox filter around track
        track_lon, track_lat = trip.track[:, 0], trip.track[:, 1]
        pad = 0.05  # roughly ~5km
        min_lon, max_lon = track_lon.min() - pad, track_lon.max() + pad
        min_lat, max_lat = track_lat.min() - pad, track_lat.max() + pad
        
        # Filter nodes inside bbox
        in_bbox = (nodes[:, 0] >= min_lon) & (nodes[:, 0] <= max_lon) & \
                  (nodes[:, 1] >= min_lat) & (nodes[:, 1] <= max_lat)
        in_bbox_arcs = in_bbox[arcs[:, 0]] & in_bbox[arcs[:, 1]]
        indices = np.where(in_bbox_arcs)[0]
        if len(indices) > max_arcs:
            indices = indices[:max_arcs]
        selected_indices = indices

    lines = []
    for idx in selected_indices:
        u, v = int(arcs[idx, 0]), int(arcs[idx, 1])
        u_lat, u_lon = float(nodes[u, 1]), float(nodes[u, 0])
        v_lat, v_lon = float(nodes[v, 1]), float(nodes[v, 0])
        lines.append([(u_lat, u_lon), (v_lat, v_lon)])
    return lines


def route_to_lines(trip, route_arc_ids: list[int] | np.ndarray | None = None) -> list[list[tuple[float, float]]]:
    """Convert an ordered sequence of arc IDs to a list of segments in (lat, lon) order."""
    if route_arc_ids is None:
        if trip.route is None:
            return []
        route_arc_ids = trip.route

    nodes = trip.nodes
    arcs = trip.arcs
    lines = []
    for arc_id in route_arc_ids:
        if arc_id < 0 or arc_id >= len(arcs):
            continue
        u, v = int(arcs[arc_id, 0]), int(arcs[arc_id, 1])
        u_lat, u_lon = float(nodes[u, 1]), float(nodes[u, 0])
        v_lat, v_lon = float(nodes[v, 1]), float(nodes[v, 0])
        lines.append([(u_lat, u_lon), (v_lat, v_lon)])
    return lines

