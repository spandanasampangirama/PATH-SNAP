"""API request/response contract between Laptop 1 (API) and Laptop 2 (UI).  Owner: P4 (frozen after Phase 0)

All coordinates are [lat, lon] (Folium order). Reference-route geometry is NEVER returned.
"""
from __future__ import annotations

from pydantic import BaseModel


class TripInfo(BaseModel):
    trip_id: str
    n_points: int


class TripList(BaseModel):
    partition: str
    trips: list[TripInfo]


class PredictRequest(BaseModel):
    trip_id: str


class Metrics(BaseModel):
    top1_acc_classifier: float | None = None
    top1_acc_hmm: float | None = None
    route_continuity_classifier: float | None = None
    route_continuity_hmm: float | None = None
    candidate_miss_rate: float | None = None
    route_precision: float | None = None
    route_recall: float | None = None
    route_f1: float | None = None


class PredictResponse(BaseModel):
    trip_id: str
    n_points: int
    model_name: str
    gps_points: list[list[float]]  # [[lat, lon], ...] raw observations (may be downsampled for display)
    predicted_route: list[list[float]]  # [[lat, lon], ...] polyline of the HMM route
    classifier_route: list[list[float]] | None = None  # optional: classifier-only polyline for comparison
    metrics: Metrics  # computed server-side from internal reference data
    timings_ms: dict[str, float]


class Health(BaseModel):
    status: str
    model_name: str | None
    eval_dir: str
    n_trips: int
