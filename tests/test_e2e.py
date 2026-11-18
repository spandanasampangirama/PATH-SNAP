"""End-to-end gate: becomes mandatory once Phase 1 is merged (expected to fail before that)."""
import numpy as np

from pathsnap.config import CFG
from pathsnap.pipeline import PathSnapPredictor

SMALL_TRIP = "00000046"


def test_predict_small_trip():
    p = PathSnapPredictor(CFG["paths"]["artifacts_dir"])
    res, net = p.predict(CFG["paths"]["train_dir"], SMALL_TRIP)
    assert len(res.hmm_arc) == res.n_points == len(res.classifier_arc)
    valid = res.hmm_arc[res.hmm_arc >= 0]
    assert valid.size > 0 and valid.max() < len(net.arcs)
    assert isinstance(res.route_arcs, list)
