"""Per-stage inference latency of the full pipeline on the student-test trips.  Owner: P1

Runs PathSnapPredictor.predict (the exact code the API calls) once per trip, single process,
and records the per-stage timings it reports. Ground truth is not loaded.

Writes results/data/inference_latency.csv.
Run: python -m scripts.benchmark_latency [--partition test] [--label after_router_speedup]
"""
from __future__ import annotations

import argparse
import platform
import time
import warnings
from pathlib import Path

import pandas as pd

from pathsnap.config import CFG
from pathsnap.io import list_trips
from pathsnap.pipeline import PathSnapPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="test", choices=["train", "test"])
    ap.add_argument("--label", default="current")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    part_dir = CFG["paths"][f"{args.partition}_dir"]
    predictor = PathSnapPredictor(CFG["paths"]["artifacts_dir"])
    rows = []
    for tid in list_trips(part_dir):
        t0 = time.perf_counter()
        result, net = predictor.predict(part_dir, tid)
        wall = (time.perf_counter() - t0) * 1000
        rows.append({"label": args.label, "partition": args.partition, "trip_id": tid,
                     "n_points": result.n_points, "n_arcs": net.n_arcs, "model": predictor.model_name,
                     **{f"{k}_ms": v for k, v in result.timings_ms.items() if k != "dead_end_restarts"},
                     "dead_end_restarts": result.timings_ms.get("dead_end_restarts"), "wall_ms": wall,
                     "machine": platform.machine(), "python": platform.python_version()})
        print(f"{tid}  arcs {net.n_arcs:>7,}  points {result.n_points:>5}  total {wall:7.0f} ms  "
              f"(viterbi {result.timings_ms['viterbi']:.0f} ms)", flush=True)
    df = pd.DataFrame(rows)
    path = Path(CFG["paths"]["results_dir"]) / "data" / "inference_latency.csv"
    if path.exists():
        old = pd.read_csv(path, dtype={"trip_id": str})
        df = pd.concat([old[old.label != args.label], df], ignore_index=True)
    df.to_csv(path, index=False)
    cur = df[df.label == args.label]
    print(f"\n{args.label}: mean {cur.wall_ms.mean():.0f} ms, max {cur.wall_ms.max():.0f} ms per trip -> {path}")


if __name__ == "__main__":
    main()
