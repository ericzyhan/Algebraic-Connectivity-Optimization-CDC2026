from __future__ import annotations

from itertools import combinations
import random

import networkx as nx

from graph_design.completion.base import CompletionPolicy
from graph_design.spectral import fiedler_vector


class FVGCompletionPolicy(CompletionPolicy):
    def complete(
        self,
        graph: nx.Graph,
        target_edges: int,
        rng: random.Random,
        max_steps: int | None = None,
    ) -> nx.Graph:
        del rng
        output = graph.copy()
        if output.number_of_edges() > target_edges:
            raise ValueError(
                "Add-only completion cannot start with more edges than target budget."
            )

        steps = 0
        while output.number_of_edges() < target_edges:
            if max_steps is not None and steps >= max_steps:
                raise RuntimeError("FVG completion reached max_steps before hitting target.")
            edge = self._best_non_edge(output)
            if edge is None:
                raise RuntimeError("No available non-edge remains before reaching target.")
            output.add_edge(*edge)
            steps += 1

        return output

    @staticmethod
    def _best_non_edge(graph: nx.Graph) -> tuple[int, int] | None:
        nodes = list(sorted(graph.nodes()))
        n = len(nodes)
        if n < 2:
            return None
        vec = fiedler_vector(graph)
        if len(vec) != n:
            return None

        best_score = -1.0
        best_edge: tuple[int, int] | None = None
        for i, j in combinations(range(n), 2):
            u = nodes[i]
            v = nodes[j]
            if graph.has_edge(u, v):
                continue
            score = float((vec[i] - vec[j]) ** 2)
            if score > best_score:
                best_score = score
                best_edge = (u, v)
        return best_edge
