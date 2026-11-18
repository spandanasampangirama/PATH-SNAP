"""Raw file loading.  Owner: P1

Prediction code must call load_trip(..., with_route=False). The reference route is loaded ONLY by
evaluate.py / training-label code, so ground truth can never leak into the prediction pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray  # (N, 3) float: lon, lat, t   (note: LON FIRST)
    nodes: np.ndarray  # (V, 2) float: lon, lat      (row index = node_id)
    arcs: np.ndarray  # (E, 2) int: from_node, to_node (row index = arc_id), directed
    route: np.ndarray | None = None  # (R,) int: ordered reference arc_ids, None at inference


def list_trips(partition_dir: str | Path) -> list[str]:
    d = Path(partition_dir)
    return sorted(p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))


def load_trip(partition_dir: str | Path, trip_id: str, with_route: bool = False) -> Trip:
    base = Path(partition_dir) / trip_id / trip_id
    track = np.loadtxt(f"{base}.track", ndmin=2)
    nodes = np.loadtxt(f"{base}.nodes", ndmin=2)
    arcs = np.loadtxt(f"{base}.arcs", dtype=np.int64, ndmin=2)
    route = load_route(partition_dir, trip_id) if with_route else None
    return Trip(trip_id, track, nodes, arcs, route)


def load_route(partition_dir: str | Path, trip_id: str) -> np.ndarray:
    base = Path(partition_dir) / trip_id / trip_id
    return np.loadtxt(f"{base}.route", dtype=np.int64, ndmin=1)
