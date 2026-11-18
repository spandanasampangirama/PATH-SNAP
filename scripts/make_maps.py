"""Generate Folium maps for the report into results/maps/*.html.  Owner: P4

EDA maps (raw GPS + network + reference route) and prediction maps (GPS, reference, classifier-only,
HMM route) for selected TRAIN/STUDENT-TEST trips. Never for evaluator trips.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pathsnap.config import CFG
from pathsnap.io import load_trip
from pathsnap.viz import network_to_lines, route_to_lines, trip_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.maps")


def make_eda_map(partition_dir: str, trip_id: str, output_path: Path, with_route: bool = True) -> Path:
    """Generate and save an EDA Folium map showing GPS, network and reference route."""
    log.info("Loading trip %s from %s for EDA map...", trip_id, partition_dir)
    trip = load_trip(partition_dir, trip_id, with_route=with_route)
    
    net_lines = network_to_lines(trip, max_arcs=6000)
    ref_lines = route_to_lines(trip) if with_route and trip.route is not None else None
    
    title = f"PathSnap Trip {trip_id} (EDA: GPS + Network + Reference)"
    m = trip_map(
        track_lonlat=trip.track,
        network_lines=net_lines,
        reference_line=ref_lines,
        title=title,
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    log.info("Saved EDA map to %s (%d GPS pts, %d net lines, %d ref lines)",
             output_path, len(trip.track), len(net_lines), len(ref_lines) if ref_lines else 0)
    return output_path


def make_prediction_map(partition_dir: str, trip_id: str, output_path: Path, predictor=None) -> Path:
    """Generate and save a prediction comparison map (GPS + predicted route + reference route)."""
    log.info("Generating prediction map for %s...", trip_id)
    trip = load_trip(partition_dir, trip_id, with_route=True)
    ref_lines = route_to_lines(trip) if trip.route is not None else None
    net_lines = network_to_lines(trip, max_arcs=5000)

    pred_lines = None
    cls_lines = None

    if predictor is not None:
        try:
            result, net = predictor.predict(partition_dir, trip_id)
            if hasattr(net, "arc_polyline_lonlat") and result.route_arcs:
                pred_lines = net.arc_polyline_lonlat(result.route_arcs)
            if hasattr(result, "classifier_arc") and result.classifier_arc is not None:
                valid = [int(a) for a in result.classifier_arc if a >= 0]
                if hasattr(net, "arc_polyline_lonlat") and valid:
                    cls_lines = net.arc_polyline_lonlat(valid)
        except Exception as e:
            log.warning("Predictor failed on %s: %s", trip_id, e)

    title = f"PathSnap Trip {trip_id} Prediction Map"
    m = trip_map(
        track_lonlat=trip.track,
        network_lines=net_lines,
        reference_line=ref_lines,
        predicted_line=pred_lines,
        classifier_line=cls_lines,
        title=title,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    log.info("Saved prediction map to %s", output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate Folium maps for PathSnap report")
    parser.add_argument("--outdir", default=CFG["paths"]["results_dir"] + "/maps",
                        help="Output directory for generated HTML maps")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = CFG["paths"]["train_dir"]
    test_dir = CFG["paths"]["test_dir"]

    # 1. Representative EDA maps from training partition
    # Trip 00000046: Small benchmark trip (649 arcs, clean urban route)
    make_eda_map(train_dir, "00000046", out_dir / "eda_00000046_baseline.html", with_route=True)

    # Trip 00000078: Long trip with massive 14.7h GPS gap
    make_eda_map(train_dir, "00000078", out_dir / "eda_00000078_gap_14h.html", with_route=True)

    # Trip 00000090: Long trip with massive 42h GPS gap
    make_eda_map(train_dir, "00000090", out_dir / "eda_00000090_gap_42h.html", with_route=True)

    # 2. Representative EDA maps from student test partition
    make_eda_map(test_dir, "00000000", out_dir / "eda_00000000_test.html", with_route=True)
    make_eda_map(test_dir, "00000007", out_dir / "eda_00000007_test.html", with_route=True)

    # 3. Check if predictor is available for prediction comparison maps
    try:
        from pathsnap.pipeline import PathSnapPredictor
        predictor = PathSnapPredictor(CFG["paths"]["artifacts_dir"])
        make_prediction_map(train_dir, "00000046", out_dir / "pred_00000046.html", predictor=predictor)
        make_prediction_map(test_dir, "00000000", out_dir / "pred_00000000.html", predictor=predictor)
    except Exception as e:
        log.info("PathSnapPredictor not active yet (%s); skipping model prediction overlays", e)

    log.info("All maps generated successfully in %s", out_dir)


if __name__ == "__main__":
    main()
