"""Coordinate projection and planar geometry.  Owner: P1

Why a per-trip projection?
  The 90 trips are spread worldwide (lon -100..137, lat -34..62). Euclidean distance on degrees is
  wrong (1 deg of longitude shrinks with cos(lat)), and one fixed UTM zone only covers 6 deg of
  longitude. We therefore project every trip into its own Azimuthal Equidistant (AEQD) plane
  centred on the track centroid. AEQD keeps distances from the centre exact and distortion stays
  far below GPS noise (a few metres) over the extent of a single trip's road network, so after
  projection we can use fast planar geometry (point-to-segment distance, bearings) in metres.
"""
from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer


class LocalProjector:
    """Project lon/lat <-> local x/y metres (x = east, y = north) for one trip."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        crs = CRS.from_proj4(f"+proj=aeqd +lat_0={self.lat0} +lon_0={self.lon0} +datum=WGS84 +units=m +no_defs")
        # always_xy=True -> inputs/outputs are (lon, lat) / (x, y), matching our file column order.
        self._fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self._inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    @classmethod
    def for_track(cls, track: np.ndarray) -> "LocalProjector":
        return cls(np.mean(track[:, 0]), np.mean(track[:, 1]))

    def to_xy(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._fwd.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
        return np.asarray(x), np.asarray(y)

    def to_lonlat(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = self._inv.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return np.asarray(lon), np.asarray(lat)


def point_segment_distance(px, py, ax, ay, bx, by) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised distance from points P to segments AB (all arrays broadcastable, metres).

    Returns (dist_m, proj_frac): proj_frac in [0, 1] is where the closest point lies along AB
    (0 = at A / from-node, 1 = at B / to-node). Zero-length segments are treated as points.
    """
    px, py, ax, ay, bx, by = map(np.asarray, (px, py, ax, ay, bx, by))
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        u = ((px - ax) * dx + (py - ay) * dy) / len2
    u = np.where(len2 > 0, np.clip(u, 0.0, 1.0), 0.0)
    cx, cy = ax + u * dx, ay + u * dy
    return np.hypot(px - cx, py - cy), u


def bearing_deg(ax, ay, bx, by) -> np.ndarray:
    """Planar bearing of vector A->B in degrees [0, 360), 0 = north, 90 = east (clockwise)."""
    return np.degrees(np.arctan2(np.asarray(bx) - ax, np.asarray(by) - ay)) % 360.0


def angle_diff_deg(a, b) -> np.ndarray:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360.0
    return np.minimum(d, 360.0 - d)


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance (m). Used only to sanity-check the projection."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))
