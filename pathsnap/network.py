"""Road network: projected geometry, spatial index, connectivity.  Owner: P1

Facts from the data that drive this design:
  * Every trip ships its OWN network (.nodes/.arcs). Each arc is a straight segment node->node.
  * Arcs are DIRECTED: the share of arcs whose reverse also exists ranges 7%..100% per trip,
    so one-way roads exist and connectivity must respect direction.
  * Networks range from ~650 to ~307k arcs, so nearest-arc lookup uses a spatial index
    (shapely STRtree, an R-tree over segment bounding boxes) instead of brute force.
"""
from __future__ import annotations

import numpy as np
import shapely
from shapely import STRtree

from .geo import LocalProjector, bearing_deg, point_segment_distance
from .io import Trip


class RoadNetwork:
    def __init__(self, trip: Trip):
        """Build from a Trip. The reference route is never used here."""
        self.trip_id = trip.trip_id
        self.nodes_lonlat = trip.nodes
        self.arcs = trip.arcs
        self.n_nodes, self.n_arcs = len(trip.nodes), len(trip.arcs)

        # Projection centred on the GPS track (where accuracy matters most).
        self.proj = LocalProjector.for_track(trip.track)
        x, y = self.proj.to_xy(trip.nodes[:, 0], trip.nodes[:, 1])
        self.node_xy = np.column_stack([x, y])

        a, b = self.arcs[:, 0], self.arcs[:, 1]
        self.arc_a = self.node_xy[a]  # (E, 2) from-node xy
        self.arc_b = self.node_xy[b]  # (E, 2) to-node xy
        d = self.arc_b - self.arc_a
        self.arc_len_m = np.hypot(d[:, 0], d[:, 1])
        self.arc_bearing = bearing_deg(self.arc_a[:, 0], self.arc_a[:, 1], self.arc_b[:, 0], self.arc_b[:, 1])

        # Directed adjacency in CSR form: out_arcs of node v = out_idx[out_ptr[v]:out_ptr[v+1]].
        order = np.argsort(a, kind="stable")
        self.out_idx = order
        self.out_ptr = np.zeros(self.n_nodes + 1, dtype=np.int64)
        np.add.at(self.out_ptr, a + 1, 1)
        self.out_ptr = np.cumsum(self.out_ptr)
        self.out_degree = np.diff(self.out_ptr)
        self.in_degree = np.bincount(b, minlength=self.n_nodes)

        # Reverse arc lookup (twin of a->b is b->a), -1 if the road is one-way.
        key = a.astype(np.int64) * self.n_nodes + b
        rkey = b.astype(np.int64) * self.n_nodes + a
        srt = np.argsort(key)
        pos = np.searchsorted(key[srt], rkey)
        pos = np.clip(pos, 0, len(key) - 1)
        self.reverse_arc = np.where(key[srt][pos] == rkey, srt[pos], -1)

        # Spatial index over arc segments.
        coords = np.stack([self.arc_a, self.arc_b], axis=1)  # (E, 2, 2)
        self._geoms = shapely.linestrings(coords)
        self._tree = STRtree(self._geoms)

    # ------------------------------------------------------------------ projection helpers
    def project(self, lon, lat) -> np.ndarray:
        x, y = self.proj.to_xy(lon, lat)
        return np.column_stack([x, y])

    def project_track(self, track: np.ndarray) -> np.ndarray:
        """(N, 2) projected x/y of a .track array (lon, lat, t)."""
        return self.project(track[:, 0], track[:, 1])

    # ------------------------------------------------------------------ spatial queries
    def arc_distance(self, px, py, arc_ids) -> tuple[np.ndarray, np.ndarray]:
        """Exact distance (m) and proj_frac from points to the given arcs (elementwise / broadcast)."""
        arc_ids = np.asarray(arc_ids)
        A, B = self.arc_a[arc_ids], self.arc_b[arc_ids]
        return point_segment_distance(px, py, A[..., 0], A[..., 1], B[..., 0], B[..., 1])

    def query_arcs_bulk(self, xy: np.ndarray, radius_m: float, k: int):
        """Vectorised k-nearest-arcs-within-radius for many points.

        Returns (point_idx, arc_id, dist_m, proj_frac, rank) flat arrays, sorted by point then distance.
        Points with no arc inside the radius simply produce no rows.
        """
        pts = shapely.points(xy[:, 0], xy[:, 1])
        # R-tree prefilter + exact GEOS distance test: all (point, arc) pairs within radius.
        pi, ai = self._tree.query(pts, predicate="dwithin", distance=radius_m)
        dist, frac = self.arc_distance(xy[pi, 0], xy[pi, 1], ai)
        order = np.lexsort((ai, dist, pi))  # by point, then distance (arc_id breaks ties deterministically)
        pi, ai, dist, frac = pi[order], ai[order], dist[order], frac[order]
        # rank within each point group
        start = np.r_[0, np.flatnonzero(np.diff(pi)) + 1]
        rank = np.arange(len(pi)) - np.repeat(start, np.diff(np.r_[start, len(pi)]))
        keep = rank < k
        return pi[keep], ai[keep], dist[keep], frac[keep], rank[keep]

    def query_arcs(self, x: float, y: float, radius_m: float, k: int):
        """Up to k nearest arcs within radius of one point. Returns (arc_ids, dist_m, proj_frac)."""
        _, ai, d, f, _ = self.query_arcs_bulk(np.array([[x, y]], dtype=float), radius_m, k)
        return ai, d, f

    # ------------------------------------------------------------------ connectivity
    def out_arcs(self, node: int) -> np.ndarray:
        return self.out_idx[self.out_ptr[node]: self.out_ptr[node + 1]]

    def successor_arcs(self, arc_id: int) -> np.ndarray:
        """Arcs that can legally follow arc_id (their from_node == arc_id's to_node). Includes U-turns."""
        return self.out_arcs(int(self.arcs[arc_id, 1]))

    def are_consecutive(self, arc_a: int, arc_b: int) -> bool:
        return bool(self.arcs[arc_a, 1] == self.arcs[arc_b, 0])

    # ------------------------------------------------------------------ geometry output
    def arc_polyline_lonlat(self, arc_ids) -> list[tuple[float, float]]:
        """(lat, lon) vertices for a sequence of arcs, for Folium/API. NOTE: lat first.

        Consecutive arcs share a node, so we emit from-node of the first arc then every to-node.
        If two consecutive arcs are not connected, both endpoints are emitted (a visible jump).
        """
        out: list[tuple[float, float]] = []
        last_node = None
        for aid in arc_ids:
            if aid is None or aid < 0:
                continue
            u, v = int(self.arcs[aid, 0]), int(self.arcs[aid, 1])
            for n in (u, v):
                if n != last_node:
                    lon, lat = self.nodes_lonlat[n]
                    out.append((float(lat), float(lon)))
                    last_node = n
        return out

    def arcs_near_track(self, track_xy: np.ndarray, buffer_m: float = 100.0) -> np.ndarray:
        """Arc ids within buffer of the track — for drawing the 'relevant road network' on maps."""
        _, ai = self._tree.query(shapely.points(track_xy[:, 0], track_xy[:, 1]), predicate="dwithin", distance=buffer_m)
        return np.unique(ai)
