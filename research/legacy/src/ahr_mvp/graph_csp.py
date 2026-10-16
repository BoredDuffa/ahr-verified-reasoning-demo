#!/usr/bin/env python3
"""Exact small-graph search and verification for the AHR MVP harness."""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

Edge = Tuple[int, int]


@dataclass(frozen=True)
class GraphSearchResult:
    vertices: int
    checked: int
    solution: Tuple[Edge, ...] | None
    verification: dict | None
    conclusion: str

    @property
    def passed(self) -> bool:
        return self.solution is not None


def all_possible_edges(vertices: int) -> Tuple[Edge, ...]:
    return tuple((i, j) for i in range(vertices) for j in range(i + 1, vertices))


def normalize_edges(edges: Iterable[Sequence[int]]) -> Tuple[Edge, ...]:
    normalized = []
    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"edge must have two endpoints: {edge}")
        a, b = int(edge[0]), int(edge[1])
        if a == b:
            raise ValueError(f"self-loop is not allowed: {edge}")
        normalized.append((min(a, b), max(a, b)))
    return tuple(sorted(set(normalized)))


def adjacency(vertices: int, edges: Iterable[Edge]) -> list[set[int]]:
    adj = [set() for _ in range(vertices)]
    for a, b in edges:
        if a < 0 or b < 0 or a >= vertices or b >= vertices:
            raise ValueError(f"edge endpoint out of range: {(a, b)}")
        adj[a].add(b)
        adj[b].add(a)
    return adj


def is_connected(vertices: int, edges: Iterable[Edge]) -> bool:
    if vertices == 0:
        return True
    adj = adjacency(vertices, edges)
    seen = {0}
    q = deque([0])
    while q:
        node = q.popleft()
        for nxt in adj[node]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == vertices


def graph_diameter(vertices: int, edges: Iterable[Edge]) -> int | None:
    if not is_connected(vertices, edges):
        return None
    adj = adjacency(vertices, edges)
    diameter = 0
    for start in range(vertices):
        dist = [-1] * vertices
        dist[start] = 0
        q = deque([start])
        while q:
            node = q.popleft()
            for nxt in adj[node]:
                if dist[nxt] == -1:
                    dist[nxt] = dist[node] + 1
                    q.append(nxt)
        diameter = max(diameter, max(dist))
    return diameter


def edge_removal_disconnects(vertices: int, edges: Tuple[Edge, ...]) -> bool:
    return all(not is_connected(vertices, tuple(edge for edge in edges if edge != removed)) for removed in edges)


def triangle_count(vertices: int, edges: Iterable[Edge]) -> int:
    edge_set = set(edges)
    count = 0
    for a, b, c in itertools.combinations(range(vertices), 3):
        if (a, b) in edge_set and (a, c) in edge_set and (b, c) in edge_set:
            count += 1
    return count


def verify_graph(
    vertices: int,
    edges: Iterable[Sequence[int]],
    diameter: int | None = None,
    edge_critical_connected: bool = False,
    triangles: int | None = None,
) -> dict:
    edge_tuple = normalize_edges(edges)
    connected = is_connected(vertices, edge_tuple)
    actual_diameter = graph_diameter(vertices, edge_tuple)
    critical = edge_removal_disconnects(vertices, edge_tuple) if connected else False
    tri_count = triangle_count(vertices, edge_tuple)
    checks = {
        "connected": connected,
        "diameter": actual_diameter,
        "edge_critical_connected": critical,
        "triangle_count": tri_count,
    }
    valid = connected
    if diameter is not None:
        valid = valid and actual_diameter == diameter
    if edge_critical_connected:
        valid = valid and critical
    if triangles is not None:
        valid = valid and tri_count == triangles
    return {
        "vertices": vertices,
        "edges": [list(edge) for edge in edge_tuple],
        "adjacency": {str(i): sorted(adjacency(vertices, edge_tuple)[i]) for i in range(vertices)},
        "checks": checks,
        "valid": valid,
    }


def search_graph(
    vertices: int,
    diameter: int | None = None,
    edge_critical_connected: bool = False,
    triangles: int | None = None,
    max_masks: int | None = None,
) -> GraphSearchResult:
    possible = all_possible_edges(vertices)
    if len(possible) > 22:
        raise ValueError("exact graph search is capped at 22 possible edges for this MVP")
    checked = 0
    for mask in range(1 << len(possible)):
        if max_masks is not None and checked >= max_masks:
            break
        checked += 1
        edges = tuple(possible[i] for i in range(len(possible)) if (mask >> i) & 1)
        verification = verify_graph(vertices, edges, diameter, edge_critical_connected, triangles)
        if verification["valid"]:
            return GraphSearchResult(vertices, checked, edges, verification, "sat")
    conclusion = "unsat_under_exhaustive_simple_graph_search"
    return GraphSearchResult(vertices, checked, None, None, conclusion)

