"""Bounded shortest paths on the directed road graph (P3).

Performance notes (P1 phase 2):
- Dijkstra runs on plain Python lists built once from the RoadNetwork CSR arrays, instead
  of calling ``net.out_arcs`` and indexing numpy scalars inside the inner loop.
- The cache is keyed by source node only. A search run with bound B gives exact distances
  and the same predecessor tree for every node within any smaller bound b <= B: nodes beyond
  b are only popped after all nodes within b, so they cannot change them. A cached result is
  reused for any request with bound <= B and only re-run when a larger bound is needed.
- ``transition_matrix`` returns every candidate-pair route distance for one HMM step in a
  single call. Each value equals ``route_distance`` for that pair.
"""
from __future__ import annotations
from collections import OrderedDict
import heapq
import numpy as np
from .network import RoadNetwork

_INF = float("inf")


class Router:
    def __init__(self, net: RoadNetwork, cache_size: int = 4096, min_bound_m: float = 150.0):
        self.net = net
        self.cache_size = max(1, int(cache_size))
        # Searches are run with at least this bound so that later HMM steps with slightly
        # larger bounds can reuse them. This never changes results, only how much is cached.
        self.min_bound_m = float(min_bound_m)
        self._cache: OrderedDict[int, tuple[float, dict[int, float], dict[int, tuple[int, int]]]] = OrderedDict()
        self._ptr = net.out_ptr.tolist()
        self._idx = net.out_idx.tolist()
        self._tail = net.arcs[:, 0].tolist()
        self._head = net.arcs[:, 1].tolist()
        self._len = net.arc_len_m.tolist()
        self._n_arcs = int(net.n_arcs)

    def _bounded_dijkstra(self, source: int, max_m: float):
        """Return ``(dist, prev, bound)`` for a search covering at least ``max_m`` metres.

        ``dist`` may contain nodes farther than ``max_m``; callers must compare against
        their own bound.
        """
        source, need = int(source), max(0.0, float(max_m))
        cached = self._cache.get(source)
        if cached is not None and cached[0] >= need:
            self._cache.move_to_end(source)
            return cached[1], cached[2], cached[0]
        bound = max(need, self.min_bound_m)
        ptr, idx, head, length = self._ptr, self._idx, self._head, self._len
        dist, prev = {source: 0.0}, {}
        queue = [(0.0, source)]
        pop, push = heapq.heappop, heapq.heappush
        while queue:
            cost, node = pop(queue)
            if cost != dist.get(node) or cost > bound:
                continue
            for p in range(ptr[node], ptr[node + 1]):
                aid = idx[p]
                nxt = head[aid]
                nd = cost + length[aid]
                if nd <= bound and nd < dist.get(nxt, _INF):
                    dist[nxt], prev[nxt] = nd, (node, aid)
                    push(queue, (nd, nxt))
        self._cache[source] = (bound, dist, prev)
        self._cache.move_to_end(source)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return dist, prev, bound

    @staticmethod
    def _reconstruct(source: int, target: int, previous: dict[int, tuple[int, int]]) -> list[int]:
        path, node = [], int(target)
        while node != int(source):
            item = previous.get(node)
            if item is None:
                return []
            parent, aid = item
            path.append(aid)
            node = parent
        return list(reversed(path))

    def _pair(self, a: int, fa: float, b: int, fb: float, max_m: float) -> float:
        """Route distance for one pair with already-validated ids and clipped fractions."""
        length = self._len
        if a == b and fb >= fa:
            d = (fb - fa) * length[a]
            return d if d <= max_m else _INF
        prefix = (1.0 - fa) * length[a]
        suffix = fb * length[b]
        remaining = max_m - prefix - suffix
        if remaining < 0:
            return _INF
        source, target = self._head[a], self._tail[b]
        if source == target:
            return prefix + suffix
        dist, _, _ = self._bounded_dijkstra(source, remaining)
        middle = dist.get(target)
        if middle is None or middle > remaining:
            return _INF
        return prefix + middle + suffix

    def route_distance(self, arc_a: int, frac_a: float, arc_b: int, frac_b: float, max_m: float) -> float:
        """Shortest directed distance between fractional positions; inf if unreachable/out of bound."""
        a, b = int(arc_a), int(arc_b)
        if min(a, b) < 0 or max(a, b) >= self._n_arcs:
            return _INF
        fa, fb = float(np.clip(frac_a, 0, 1)), float(np.clip(frac_b, 0, 1))
        return float(self._pair(a, fa, b, fb, float(max_m)))

    def transition_matrix(self, prev_arcs, prev_fracs, curr_arcs, curr_fracs, max_m: float) -> np.ndarray:
        """(K, J) route distances from every previous candidate to every current candidate.

        Entry [k, j] equals ``route_distance(prev_arcs[k], prev_fracs[k], curr_arcs[j],
        curr_fracs[j], max_m)``. One Dijkstra per distinct source node covers the whole row.
        """
        pa = [int(x) for x in prev_arcs]
        ca = [int(x) for x in curr_arcs]
        pf = np.clip(np.asarray(prev_fracs, dtype=float), 0, 1).tolist()
        cf = np.clip(np.asarray(curr_fracs, dtype=float), 0, 1).tolist()
        max_m = float(max_m)
        out = np.full((len(pa), len(ca)), _INF)
        n = self._n_arcs
        valid_j = [j for j, b in enumerate(ca) if 0 <= b < n]
        for k, a in enumerate(pa):
            if not 0 <= a < n:
                continue
            fa = pf[k]
            row = out[k]
            for j in valid_j:
                row[j] = self._pair(a, fa, ca[j], cf[j], max_m)
        return out

    def shortest_arc_path(self, arc_a: int, arc_b: int, max_m: float) -> list[int]:
        """Intermediate arcs from arc_a to arc_b (exclusive), excluding both endpoint arcs."""
        a, b = int(arc_a), int(arc_b)
        if a == b or min(a, b) < 0 or max(a, b) >= self._n_arcs:
            return []
        source, target = self._head[a], self._tail[b]
        if source == target:
            return []
        dist, prev, _ = self._bounded_dijkstra(source, max_m)
        d = dist.get(target)
        return self._reconstruct(source, target, prev) if d is not None and d <= max_m else []
