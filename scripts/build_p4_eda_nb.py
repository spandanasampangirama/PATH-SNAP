"""Script to build and execute notebooks/p4_eda.ipynb.  Owner: P4"""
import os
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

repo_root = Path(__file__).resolve().parent.parent
nb = nbf.v4.new_notebook()

cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# PathSnap: Exploratory Data Analysis (EDA)
**Owner:** P4 (EDA, maps, API, UI, deployment)  
**Partition Notice:** Every analysis explicitly states its partition (**Train: 80 trips**, **Student-Test: 10 trips**).  
**Dataset Reference:** Follows *Dataset Exploration §4* guidelines.
"""))

# Cell 1: Setup & Imports
cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure repo root is on sys.path
repo_root = Path("..").resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pathsnap.config import CFG
from pathsnap.io import list_trips, load_trip, load_route
from pathsnap.viz import trip_map, network_to_lines, route_to_lines

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (9, 4.5)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

print("Libraries and PathSnap modules successfully loaded!")
"""))

# Section 1: Inventory & Quality
cells.append(nbf.v4.new_markdown_cell("""## 1. Dataset Inventory & Missing Value (NaN) Analysis
We inspect file presence (`.track`, `.nodes`, `.arcs`, `.route`), row counts, and scan for any NaN / missing values across both partitions:
- **Train partition:** 80 trips (`dataset/train/`)
- **Student-test partition:** 10 trips (`dataset/test/`)
"""))

cells.append(nbf.v4.new_code_cell("""records = []
train_dir = repo_root / CFG["paths"]["train_dir"]
test_dir = repo_root / CFG["paths"]["test_dir"]

for partition, pdir in [("Train (80)", train_dir), ("Student-Test (10)", test_dir)]:
    trips = list_trips(pdir)
    for tid in trips:
        trip = load_trip(pdir, tid, with_route=True)
        nan_track = np.isnan(trip.track).any()
        nan_nodes = np.isnan(trip.nodes).any()
        nan_arcs = np.isnan(trip.arcs).any()
        nan_route = np.isnan(trip.route).any() if trip.route is not None else False
        
        t = trip.track[:, 2]
        dt = np.diff(t)
        
        # Spatial extent
        min_lon, max_lon = float(trip.track[:, 0].min()), float(trip.track[:, 0].max())
        min_lat, max_lat = float(trip.track[:, 1].min()), float(trip.track[:, 1].max())
        
        # Road graph components
        adj = sp.csr_matrix(
            (np.ones(len(trip.arcs)), (trip.arcs[:, 0], trip.arcs[:, 1])),
            shape=(len(trip.nodes), len(trip.nodes))
        )
        n_comp, _ = csgraph.connected_components(adj, directed=False)
        
        # Route contiguity
        contiguity_pct = 100.0
        if trip.route is not None and len(trip.route) > 1:
            r = trip.route
            from_nodes = trip.arcs[r, 0]
            to_nodes = trip.arcs[r, 1]
            contiguity_pct = float((to_nodes[:-1] == from_nodes[1:]).mean() * 100)
            
        records.append({
            "partition": partition,
            "trip_id": tid,
            "n_track": len(trip.track),
            "n_nodes": len(trip.nodes),
            "n_arcs": len(trip.arcs),
            "n_route": len(trip.route) if trip.route is not None else 0,
            "has_nan": bool(nan_track or nan_nodes or nan_arcs or nan_route),
            "min_lon": min_lon,
            "max_lon": max_lon,
            "min_lat": min_lat,
            "max_lat": max_lat,
            "max_gap_s": float(dt.max()) if len(dt) > 0 else 0.0,
            "pct_1s": float((dt == 1.0).mean() * 100) if len(dt) > 0 else 100.0,
            "n_comp": n_comp,
            "contiguity_pct": contiguity_pct,
        })

df_eda = pd.DataFrame(records)

print(f"Total trips examined: {len(df_eda)}")
print(f"Total GPS observations: {df_eda['n_track'].sum():,}")
print(f"Total road nodes: {df_eda['n_nodes'].sum():,}")
print(f"Total road arcs: {df_eda['n_arcs'].sum():,}")
print(f"Any NaN / missing values detected: {df_eda['has_nan'].any()}")

inventory_summary = df_eda.groupby("partition").agg(
    trips=("trip_id", "count"),
    total_gps_obs=("n_track", "sum"),
    avg_gps_obs=("n_track", "mean"),
    min_gps_obs=("n_track", "min"),
    max_gps_obs=("n_track", "max"),
    avg_nodes=("n_nodes", "mean"),
    avg_arcs=("n_arcs", "mean"),
    pct_1hz_dt=("pct_1s", "mean"),
    avg_contiguity=("contiguity_pct", "mean")
).round(2)
inventory_summary
"""))

# Section 2: GPS Points per Trip
cells.append(nbf.v4.new_markdown_cell("""## 2. GPS Observation Count Distribution
Examine the variation in trajectory length across trips in both partitions.
"""))

cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
sns.boxplot(data=df_eda, x="partition", y="n_track", palette=["#3182CE", "#38A169"], width=0.35, ax=ax)
sns.stripplot(data=df_eda, x="partition", y="n_track", color="black", alpha=0.5, jitter=0.2, ax=ax)
ax.set_title("GPS Observations per Trip by Partition")
ax.set_xlabel("Partition")
ax.set_ylabel("Number of GPS Observations")
plt.tight_layout()
plt.show()

df_eda.groupby("partition")["n_track"].describe().round(1)
"""))

# Section 3: Sampling Interval & Large Gaps
cells.append(nbf.v4.new_markdown_cell("""## 3. Sampling Interval Distribution ($\Delta t$) and GPS Gaps
Nominal sampling frequency is 1 Hz ($\Delta t = 1$ s). We verify what fraction of steps are exactly 1 s apart and uncover severe anomalies, including multi-hour satellite signal losses.
"""))

cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

# Percentage of 1s timestamps
sns.histplot(df_eda["pct_1s"], bins=20, color="#3182CE", kde=True, ax=ax1)
ax1.set_title("Distribution of 1 Hz Timestamps ($\Delta t = 1.0$ s)")
ax1.set_xlabel("Percentage of 1 s Steps (%)")
ax1.set_ylabel("Trip Count")

# Extreme gaps (>3 min)
large_gaps = df_eda[df_eda["max_gap_s"] > 180].copy()
large_gaps["max_gap_hr"] = large_gaps["max_gap_s"] / 3600.0
sns.histplot(large_gaps["max_gap_hr"], bins=15, color="#E53E3E", kde=True, ax=ax2)
ax2.set_title("Extreme Time Gaps (> 3 minutes)")
ax2.set_xlabel("Max Gap Duration (Hours)")
ax2.set_ylabel("Trip Count")

plt.tight_layout()
plt.show()

print("Trips with largest GPS time gaps:")
top_gaps = df_eda.sort_values(by="max_gap_s", ascending=False)[["partition", "trip_id", "n_track", "max_gap_s"]].head(5).copy()
top_gaps["max_gap_hr"] = (top_gaps["max_gap_s"] / 3600.0).round(2)
top_gaps
"""))

# Section 4: Spatial Extents
cells.append(nbf.v4.new_markdown_cell("""## 4. Worldwide Geographic Extents & Projection Justification
The trips span longitudes from $-100^{\\circ}$ to $+137^{\\circ}$ and latitudes from $-34^{\\circ}$ to $+62^{\\circ}$.  
**Key Conclusion:** A single global UTM zone or planar projection cannot be used without catastrophic distortion. A **per-trip local metric projection** (such as Azimuthal Equidistant or local UTM centered on the track centroid) is mandatory.
"""))

cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
centroid_lon = (df_eda["min_lon"] + df_eda["max_lon"]) / 2.0
centroid_lat = (df_eda["min_lat"] + df_eda["max_lat"]) / 2.0

colors = df_eda["partition"].map(lambda p: "#3182CE" if "Train" in p else "#E53E3E")
ax.scatter(centroid_lon, centroid_lat, c=colors, s=df_eda["n_arcs"]/1500 + 40, alpha=0.75, edgecolors="black", linewidth=0.5)

ax.set_title("Worldwide Trajectory Centroids (Marker size proportional to Road Network Size)")
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
ax.axvline(0, color="gray", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

print(f"Longitude range: {df_eda['min_lon'].min():.2f}° to {df_eda['max_lon'].max():.2f}°")
print(f"Latitude range:  {df_eda['min_lat'].min():.2f}° to {df_eda['max_lat'].max():.2f}°")
"""))

# Section 5: Road Network Sizes & Connected Components
cells.append(nbf.v4.new_markdown_cell("""## 5. Road Network Complexity & Graph Topology
Road networks range from ~650 arcs to over 300,000 arcs.  
We examine node/arc distributions and connected components using `scipy.sparse.csgraph.connected_components`.
"""))

cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

# Nodes and Arcs log scale
ax1.scatter(df_eda["n_nodes"], df_eda["n_arcs"], alpha=0.7, color="#4A5568", edgecolors="none")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_title("Road Network Arcs vs Nodes (Log-Log Scale)")
ax1.set_xlabel("Nodes ($V$)")
ax1.set_ylabel("Arcs ($E$)")

# Connected components
sns.histplot(df_eda["n_comp"], bins=20, color="#805AD5", ax=ax2)
ax2.set_title("Connected Components in Road Graph")
ax2.set_xlabel("Number of Disconnected Components")
ax2.set_ylabel("Trip Count")

plt.tight_layout()
plt.show()

print("Summary of Road Network Arcs:")
print(df_eda["n_arcs"].describe().round(0))
"""))

# Section 6: Route Contiguity Analysis
cells.append(nbf.v4.new_markdown_cell("""## 6. Reference Route Structure & Contiguity Verification
In `.route` files, we test whether consecutive arcs form a physically connected sequence:
$$\\text{to\\_node}(arc_i) == \\text{from\\_node}(arc_{i+1})$$
Across 90 trips, the mean route contiguity is $>99.9\\%$, validating that reference routes are topological paths along the road network.
"""))

cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(8, 3.5), dpi=150)
sns.histplot(df_eda["contiguity_pct"], bins=20, color="#2E7D32", ax=ax)
ax.set_title("Reference Route Consecutive Arc Contiguity (%)")
ax.set_xlabel("Contiguity Percentage")
ax.set_ylabel("Trip Count")
plt.tight_layout()
plt.show()

print(f"Overall average route contiguity: {df_eda['contiguity_pct'].mean():.3f}%")
print(f"Number of trips with 100% strictly contiguous routes: {(df_eda['contiguity_pct'] == 100.0).sum()}/{len(df_eda)}")
"""))

# Section 7: Interactive Folium Map
cells.append(nbf.v4.new_markdown_cell("""## 7. Interactive Geographical Visualisation with Folium
Using `pathsnap.viz.trip_map`, we render trip `00000046` with:
- Raw GPS points (blue circles)
- Surrounding road network (grey segments)
- Ground-truth reference route (green line)
"""))

cells.append(nbf.v4.new_code_cell("""sample_trip = load_trip(train_dir, "00000046", with_route=True)
net_lines = network_to_lines(sample_trip, max_arcs=2000)
ref_lines = route_to_lines(sample_trip)

m = trip_map(
    track_lonlat=sample_trip.track,
    network_lines=net_lines,
    reference_line=ref_lines,
    title="PathSnap Trip 00000046 (EDA Map)"
)
m
"""))

# Section 8: Modeling Implications
cells.append(nbf.v4.new_markdown_cell("""## 8. Summary of Findings & Pipeline Architecture Decisions
1. **Coordinate Systems:** Worldwide dispersion requires a per-trip local projection (such as Azimuthal Equidistant or local UTM centered on the track centroid).
2. **Spatial Indexing:** Networks with up to ~307,000 arcs mandate a spatial index (`cKDTree` or `STRtree`) for candidate arc queries; naive linear scans would cause catastrophic latency.
3. **Time Gap Handling:** Extreme time gaps (up to 42.5 hours in trip `00000090`) require track splitting at $\Delta t > 60$ s so that HMM state transitions are reset across discontinuous segments.
4. **Data Quality:** Zero NaNs or corrupt rows across all 360 files. Over 94% of points adhere strictly to 1 Hz sampling.
"""))

nb.cells = cells

# Save notebook
out_nb_path = repo_root / "notebooks" / "p4_eda.ipynb"
with open(out_nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote notebook to {out_nb_path}")

# Execute notebook to populate all cell outputs
print("Executing notebook with NotebookClient...")
client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()

with open(out_nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook successfully executed and saved with all outputs rendered!")
