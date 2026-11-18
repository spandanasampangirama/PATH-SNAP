"""The fast Router must give exactly what an independent shortest-path solver gives.  Owner: P1

The reference is scipy's Dijkstra on the same directed graph (no shared code with routing.py).
Covers: transition_matrix == pairwise route_distance, bound handling, cache reuse across
different bounds, and predecessor paths that add up to the reported distance.
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from pathsnap.io import Trip
from pathsnap.network import RoadNetwork
from pathsnap.routing import Router


def _grid_network(n=12, spacing_deg=0.0004, oneway_share=0.3, seed=0):
    """n x n street grid near Bengaluru; some streets one-way, others two-way."""
    rng = np.random.default_rng(seed)
    lon0, lat0 = 77.59, 12.97
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    jitter = rng.normal(0, spacing_deg * 0.05, size=(n * n, 2))
    nodes = np.c_[lon0 + jj.ravel() * spacing_deg, lat0 + ii.ravel() * spacing_deg] + jitter
    arcs = []
    for i in range(n):
        for j in range(n):
            u = i * n + j
            for v in ([u + 1] if j + 1 < n else []) + ([u + n] if i + 1 < n else []):
                r = rng.random()
                if r < oneway_share / 2:
                    arcs.append((u, v))
                elif r < oneway_share:
                    arcs.append((v, u))
                else:
                    arcs += [(u, v), (v, u)]
    arcs = np.array(arcs, dtype=np.int64)
    track = np.c_[nodes[:3], np.arange(3.0)]
    return RoadNetwork(Trip("grid", track, nodes, arcs))


def _reference_distance(net, all_pairs, a, fa, b, fb, max_m):
    """route_distance semantics, computed with scipy's all-pairs node distances."""
    fa, fb = float(np.clip(fa, 0, 1)), float(np.clip(fb, 0, 1))
    L = net.arc_len_m
    if a == b and fb >= fa:
        d = (fb - fa) * L[a]
        return d if d <= max_m else np.inf
    d = (1 - fa) * L[a] + all_pairs[net.arcs[a, 1], net.arcs[b, 0]] + fb * L[b]
    return d if d <= max_m else np.inf


@pytest.fixture(scope="module")
def grid():
    net = _grid_network()
    g = csr_matrix((net.arc_len_m, (net.arcs[:, 0], net.arcs[:, 1])), shape=(net.n_nodes, net.n_nodes))
    return net, dijkstra(g, directed=True)


def test_transition_matrix_matches_independent_dijkstra(grid):
    net, all_pairs = grid
    rng = np.random.default_rng(1)
    router = Router(net)
    for bound in [0.0, 30.0, 80.0, 150.0, 400.0, 1e5]:
        pa, ca = rng.integers(0, net.n_arcs, 8), rng.integers(0, net.n_arcs, 8)
        pf, cf = rng.random(8), rng.random(8)
        pa[0], ca[0], pf[0], cf[0] = 5, 5, 0.2, 0.7          # same arc, forward
        pa[1], ca[1], pf[1], cf[1] = 7, 7, 0.9, 0.1          # same arc, backward -> loop or inf
        m = router.transition_matrix(pa, pf, ca, cf, bound)
        for k in range(8):
            for j in range(8):
                exp = _reference_distance(net, all_pairs, pa[k], pf[k], ca[j], cf[j], bound)
                assert m[k, j] == pytest.approx(exp, rel=1e-9, abs=1e-6), (bound, k, j)
                assert m[k, j] == pytest.approx(router.route_distance(pa[k], pf[k], ca[j], cf[j], bound))


def test_cached_search_reused_for_smaller_and_larger_bounds(grid):
    """A search cached with a big bound must not leak far nodes into a small-bound query,
    and a later larger bound must trigger a wider search."""
    net, all_pairs = grid
    router = Router(net, min_bound_m=0.0)
    a, b = 0, net.n_arcs - 1
    far = _reference_distance(net, all_pairs, a, 0.5, b, 0.5, 1e9)
    assert np.isfinite(far)
    assert np.isinf(router.route_distance(a, 0.5, b, 0.5, far * 0.5))        # small search first
    assert router.route_distance(a, 0.5, b, 0.5, far * 1.01) == pytest.approx(far)  # grows
    assert np.isinf(router.route_distance(a, 0.5, b, 0.5, far * 0.99))       # reuse big search, still bounded


def test_shortest_arc_path_is_connected_and_adds_up(grid):
    net, all_pairs = grid
    router = Router(net)
    rng = np.random.default_rng(2)
    checked = 0
    for a, b in rng.integers(0, net.n_arcs, size=(60, 2)):
        path = router.shortest_arc_path(a, b, 5000.0)
        s, t = net.arcs[a, 1], net.arcs[b, 0]
        if a == b or s == t or not np.isfinite(all_pairs[s, t]):
            continue
        chain = [a, *path, b]
        assert all(net.arcs[x, 1] == net.arcs[y, 0] for x, y in zip(chain, chain[1:]))
        assert net.arc_len_m[path].sum() == pytest.approx(all_pairs[s, t], abs=1e-6)
        checked += 1
    assert checked > 20
