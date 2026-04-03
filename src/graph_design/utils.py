from __future__ import annotations

from itertools import combinations
from math import comb

import networkx as nx

from graph_design.types import DesignProblem


def validate_problem(problem: DesignProblem) -> None:
    if problem.n < 2:
        raise ValueError("n must be >= 2")
    min_edges = problem.n - 1
    max_edges = comb(problem.n, 2)
    if problem.m < min_edges or problem.m > max_edges:
        raise ValueError(
            f"m must satisfy {min_edges} <= m <= {max_edges}, got {problem.m}"
        )


def normalized_density(n: int, m: int) -> float:
    min_edges = n - 1
    max_edges = comb(n, 2)
    denom = max_edges - min_edges
    if denom <= 0:
        return 1.0
    return (m - min_edges) / denom


def edge_budget_from_density(n: int, rho: float) -> int:
    min_edges = n - 1
    max_edges = comb(n, 2)
    denom = max_edges - min_edges
    if denom <= 0:
        return max_edges
    rho = max(0.0, min(1.0, rho))
    return min_edges + int(round(rho * denom))


def relabel_to_integers(graph: nx.Graph) -> nx.Graph:
    mapping = {node: idx for idx, node in enumerate(sorted(graph.nodes()))}
    return nx.relabel_nodes(graph, mapping, copy=True)


def iter_missing_edges(graph: nx.Graph):
    nodes = list(graph.nodes())
    for u, v in combinations(nodes, 2):
        if not graph.has_edge(u, v):
            yield u, v
