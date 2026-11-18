"""Tests for P2 modules: metrics, features, model zoo."""
import numpy as np
import pandas as pd
import pytest

from pathsnap import schema
from pathsnap.candidates import generate_candidates
from pathsnap.config import CFG
from pathsnap.features import FEATURE_COLS, add_features
from pathsnap.io import load_trip
from pathsnap.labels import align_points_to_route
from pathsnap.metrics import (attach_predictions, candidate_metrics, collapse_route, route_continuity,
                              route_overlap, top1_accuracy, top1_from_candidates)
from pathsnap.models import MODEL_FAMILIES, build_model
from pathsnap.network import RoadNetwork


def test_candidate_metrics_perfect_and_counts():
    m = candidate_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.9, 0.8]))
    assert m["macro_f1"] == 1.0 and m["tp"] == 2 and m["tn"] == 2 and m["log_loss"] < 0.25


def test_top1_and_attach():
    cands = pd.DataFrame({"trip_id": "a", "point_idx": [0, 0, 1, 1], "arc_id": [5, 6, 7, 8], "prob": [0.2, 0.7, 0.9, 0.1]})
    top = top1_from_candidates(cands)
    assert top.arc_id.tolist() == [6, 7]
    pts = pd.DataFrame({"trip_id": "a", "point_idx": [0, 1, 2], "ref_arc": [6, 8, 9]})
    out = attach_predictions(pts, top)
    assert out.pred_arc.tolist() == [6, 7, -1]  # point 2 has no candidate -> -1, counted wrong
    assert top1_accuracy(out.pred_arc, out.ref_arc) == pytest.approx(1 / 3)
    assert top1_accuracy(np.array([1, 2]), np.array([1, -1])) == 1.0  # unlabelled ignored


def test_route_helpers():
    np.testing.assert_array_equal(collapse_route(np.array([3, 3, -1, 4, 4, 3])), [3, 4, 3])
    ov = route_overlap(np.array([1, 2, 3]), np.array([2, 3, 4, 5]))
    assert ov["route_precision"] == pytest.approx(2 / 3) and ov["route_recall"] == 0.5


def _small():
    trip = load_trip(CFG["paths"]["train_dir"], "00000046", with_route=True)
    net = RoadNetwork(trip)
    xy = net.project_track(trip.track)
    lab = align_points_to_route(net, xy, trip.route, 50.0)
    return trip, net, xy, lab


def test_continuity_of_reference_labels_is_high():
    trip, net, xy, lab = _small()
    assert route_continuity(lab, net, 3) > 0.95


def test_features_complete_and_finite():
    trip, net, xy, lab = _small()
    c = generate_candidates(trip, net, 8, 60.0, lab, track_xy=xy)
    f = add_features(c, xy, trip.track[:, 2], net)
    assert list(schema.FEATURE_COLS) == FEATURE_COLS
    assert all(col in f.columns for col in FEATURE_COLS) and len(f) == len(c)
    assert np.isfinite(f[FEATURE_COLS].to_numpy(dtype=float)).all()
    assert f["heading_diff_deg"].between(0, 180).all()


@pytest.mark.parametrize("name", list(MODEL_FAMILIES))
def test_every_model_gives_probabilities(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(FEATURE_COLS))).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.normal(size=400) > 0).astype(int)
    p = build_model(name, {"n_neighbors": 5} if name == "knn" else None).fit(X, y).predict_proba(X)[:, 1]
    assert p.shape == (400,) and np.all((p >= 0) & (p <= 1))


# ---------------------------------------------------------------- hand-computed metrics (WORK_PLAN P2 task 1)
def test_candidate_metrics_hand_computed():
    y = np.array([1, 0, 0, 1, 0])
    p = np.array([0.9, 0.2, 0.6, 0.4, 0.1])
    m = candidate_metrics(y, p)
    # threshold 0.5 -> pred [1,0,1,0,0]: tp=1 fp=1 fn=1 tn=2
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    f1_neg = 2 * (2 / 3) * (2 / 3) / (4 / 3)  # class 0: precision 2/3, recall 2/3
    assert m["macro_f1"] == pytest.approx((0.5 + f1_neg) / 2)
    assert m["log_loss"] == pytest.approx(-np.mean(np.log([0.9, 0.8, 0.4, 0.4, 0.9])))


def test_candidate_miss_rate_hand_computed():
    from pathsnap.metrics import candidate_miss_rate
    c = pd.DataFrame({"point_idx": [0, 0, 1, 2], "arc_id": [10, 11, 12, 13]})
    # p0 ref 11 found; p1 ref 99 missed; p2 ref 13 found; p3 ref 5 has no candidates -> missed; p4 unlabelled
    assert candidate_miss_rate(c, np.array([11, 99, 13, 5, -1])) == pytest.approx(2 / 4)


class _ToyNet:
    """0 -a0-> 1 -a1-> 2 -a2-> 3, side road a3: 1 -> 4, and a4: 5 -> 6 (disconnected)."""
    arcs = np.array([[0, 1], [1, 2], [2, 3], [1, 4], [5, 6]])

    def successor_arcs(self, a):
        return np.flatnonzero(self.arcs[:, 0] == self.arcs[a, 1])


def test_route_continuity_hand_computed():
    net = _ToyNet()
    # pairs: (0,0) same ok, (0,1) succ ok, (1,2) succ ok, (2,4) disconnected -> 3/4
    assert route_continuity(np.array([0, 0, 1, 2, 4]), net, 3) == pytest.approx(3 / 4)
    # 0 -> 2 needs 2 hops
    assert route_continuity(np.array([0, 2]), net, 3) == 1.0
    assert route_continuity(np.array([0, 2]), net, 1) == 0.0
    # -1 points and pairs across a segment boundary are skipped
    # [0,-1,4,1]: only the pair (4,1) has two valid arcs; 4 -> 1 is unreachable -> 0.0
    assert route_continuity(np.array([0, -1, 4, 1]), net, 3) == 0.0
    # ...and that pair crosses a segment boundary, so with segments there is nothing to score
    assert np.isnan(route_continuity(np.array([0, -1, 4, 1]), net, 3, np.array([0, 0, 0, 1])))
    assert np.isnan(route_continuity(np.array([0, 4]), net, 3, np.array([0, 1])))


# ---------------------------------------------------------------- harness on a synthetic table (P2 task 3)
def _synthetic_tables(seed=0, n_trips=6, n_pts=40, k=4):
    """FAKE candidate + point tables with the schema columns. Exercises code paths only."""
    rng = np.random.default_rng(seed)
    cand, pts = [], []
    for t in range(n_trips):
        tid = f"{t:08d}"
        for p in range(n_pts):
            ref = p * 10 + int(rng.integers(0, k)) if p % 13 else -1  # every 13th point unlabelled
            for c in range(k):
                arc = p * 10 + c
                row = {"trip_id": tid, "point_idx": p, "arc_id": arc, "rank": c,
                       "label": -1 if ref == -1 else int(arc == ref)}
                for j, f in enumerate(FEATURE_COLS):
                    row[f] = rng.normal(2.0 if arc == ref else 0.0, 1.0)
                row["rank"] = c
                cand.append(row)
            pts.append({"trip_id": tid, "point_idx": p, "ref_arc": ref})
    return pd.DataFrame(cand), pd.DataFrame(pts)


def test_harness_on_synthetic_table(tmp_path):
    from sklearn.model_selection import GroupKFold

    from scripts.train_models import fit_model, predict, reference_baselines, top1_on, upsert_csv
    c, p = _synthetic_tables()
    lab = c["label"].to_numpy() != -1
    X, y, g = c[FEATURE_COLS].to_numpy(np.float32), c["label"].to_numpy(), c["trip_id"].to_numpy()
    for tr, va in GroupKFold(n_splits=3).split(X, y, g):
        assert not set(g[tr]) & set(g[va])  # no trip in both train and validation
        tr = tr[lab[tr]]
        model, fit_s, n_fit = fit_model("logreg", None, X[tr], y[tr], 0, max_rows=100)
        assert n_fit == 100 and fit_s >= 0  # row cap respected
        prob, _ = predict(model, X[va])
        t1, pts = top1_on(c.iloc[va], prob, p[p.trip_id.isin(np.unique(g[va]))])
        assert 0.5 < t1 <= 1.0  # strong synthetic signal -> well above chance (0.25)
        assert (pts["pred_arc"] >= 0).all()
    ref = reference_baselines(c, c, p, p)
    assert set(ref.partition) == {"train", "test"} and ref.top1_accuracy.between(0, 1).all()
    # upsert replaces rows with the same key and keeps the others
    path = tmp_path / "x.csv"
    upsert_csv(path, pd.DataFrame({"model": ["a", "b"], "v": [1, 2]}), ["model"])
    upsert_csv(path, pd.DataFrame({"model": ["a"], "v": [9]}), ["model"])
    assert pd.read_csv(path).set_index("model")["v"].to_dict() == {"a": 9, "b": 2}
