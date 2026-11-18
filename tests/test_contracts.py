"""Must ALWAYS pass on main. Run before every merge:  pytest -q"""
from pathsnap import schema
from pathsnap.config import CFG
from pathsnap.io import list_trips, load_trip

SMALL_TRIP = "00000046"  # train, 649 arcs — fastest trip for smoke tests


def test_partitions():
    assert len(list_trips(CFG["paths"]["train_dir"])) == 80
    assert len(list_trips(CFG["paths"]["test_dir"])) == 10


def test_load_trip_no_route_by_default():
    trip = load_trip(CFG["paths"]["train_dir"], SMALL_TRIP)
    assert trip.route is None
    assert trip.track.shape[1] == 3 and trip.nodes.shape[1] == 2 and trip.arcs.shape[1] == 2
    assert trip.arcs.max() < len(trip.nodes)


def test_schema_feature_cols_unique():
    assert len(set(schema.FEATURE_COLS)) == len(schema.FEATURE_COLS)


def test_api_imports():
    import api.main  # noqa: F401
