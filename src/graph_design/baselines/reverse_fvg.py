from __future__ import annotations

import random
import time

import networkx as nx

from graph_design.baselines.base import BaselineSolver
from graph_design.config import DesignConfig
from graph_design.spectral import algebraic_connectivity, fiedler_vector
from graph_design.types import BaselineResult, DesignProblem


class ReverseFVGBaseSolver(BaselineSolver):
    name = "reverse_fvg"

    def solve(
        self,
        problem: DesignProblem,
        config: DesignConfig,
        rng: random.Random,
    ) -> BaselineResult:
        start = time.perf_counter()
        graph = nx.complete_graph(problem.n)

        steps = 0
        while graph.number_of_edges() > problem.m:
            if config.fvg_max_steps is not None and steps >= config.fvg_max_steps:
                raise RuntimeError("Reverse FVG reached max_steps before hitting target.")
            edge = self._worst_edge_to_remove(graph, rng)
            if edge is None:
                raise RuntimeError(
                    "No removable non-bridge edge found before reaching target budget."
                )
            graph.remove_edge(*edge)
            steps += 1

        l2 = algebraic_connectivity(graph)
        runtime = time.perf_counter() - start
        return BaselineResult(
            name=self.name,
            problem=problem,
            graph=graph,
            final_lambda2=l2,
            runtime_sec=runtime,
            metadata={
                "initial_graph": "complete_graph",
                "initial_edges": problem.max_edges,
                "final_edges": graph.number_of_edges(),
                "deletion_steps": steps,
            },
        )

    @staticmethod
    def _worst_edge_to_remove(
        graph: nx.Graph,
        rng: random.Random,
    ) -> tuple[int, int] | None:
        nodes = list(sorted(graph.nodes()))
        if len(nodes) < 2:
            return None
        vec = fiedler_vector(graph)
        if len(vec) != len(nodes):
            return None
        node_index = {node: idx for idx, node in enumerate(nodes)}

        bridges = {tuple(sorted(edge)) for edge in nx.bridges(graph)}
        min_score = float("inf")
        candidates: list[tuple[int, int]] = []
        eps = 1e-12

        for u, v in graph.edges():
            edge = (u, v) if u < v else (v, u)
            if edge in bridges:
                continue
            i = node_index[u]
            j = node_index[v]
            score = float((vec[i] - vec[j]) ** 2)
            if score + eps < min_score:
                min_score = score
                candidates = [edge]
            elif abs(score - min_score) <= eps:
                candidates.append(edge)

        if not candidates:
            return None
        return candidates[rng.randrange(len(candidates))]
