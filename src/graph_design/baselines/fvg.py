from __future__ import annotations

import random
import time
from dataclasses import dataclass

import networkx as nx

from graph_design.baselines.base import BaselineSolver
from graph_design.completion.fvg import FVGCompletionPolicy
from graph_design.config import DesignConfig
from graph_design.spectral import algebraic_connectivity
from graph_design.types import BaselineResult, DesignProblem


@dataclass
class _FVGTrajectoryCache:
    n: int
    max_steps: int | None
    graph: nx.Graph
    added_edges: list[tuple[int, int]]
    l2_cache: dict[int, float]
    l2_runtime_by_edges: dict[int, float]
    build_runtime_by_edges: dict[int, float]
    total_build_runtime: float

    @property
    def initial_edges(self) -> int:
        return self.n - 1

    def ensure_to_target(self, target_edges: int) -> None:
        policy = FVGCompletionPolicy()
        while self.graph.number_of_edges() < target_edges:
            if self.max_steps is not None and len(self.added_edges) >= self.max_steps:
                raise RuntimeError("FVG cache reached max_steps before hitting target.")
            step_start = time.perf_counter()
            edge = policy._best_non_edge(self.graph)
            if edge is None:
                raise RuntimeError("No non-edge remained while extending FVG cache.")
            self.graph.add_edge(*edge)
            self.added_edges.append(edge)
            self.total_build_runtime += time.perf_counter() - step_start
            self.build_runtime_by_edges[self.graph.number_of_edges()] = self.total_build_runtime

    def graph_for_edges(self, target_edges: int) -> nx.Graph:
        extra = max(0, target_edges - self.initial_edges)
        graph = nx.path_graph(self.n)
        if extra > 0:
            graph.add_edges_from(self.added_edges[:extra])
        return graph

    def runtime_to_edges(self, target_edges: int) -> float:
        build_runtime = self.build_runtime_by_edges.get(target_edges)
        if build_runtime is None:
            raise RuntimeError(f"Missing cached build runtime for edge count {target_edges}.")
        l2_runtime = self.l2_runtime_by_edges.get(target_edges, 0.0)
        return build_runtime + l2_runtime


class FVGBaseSolver(BaselineSolver):
    name = "fvg"
    _cache: dict[tuple[int, int | None], _FVGTrajectoryCache] = {}

    def __init__(self) -> None:
        self._completion = FVGCompletionPolicy()

    def solve(
        self,
        problem: DesignProblem,
        config: DesignConfig,
        rng: random.Random,
    ) -> BaselineResult:
        del rng
        if config.fvg_reuse_cache:
            key = (problem.n, config.fvg_max_steps)
            cache = self._cache.get(key)
            if cache is None:
                cache = _FVGTrajectoryCache(
                    n=problem.n,
                    max_steps=config.fvg_max_steps,
                    graph=nx.path_graph(problem.n),
                    added_edges=[],
                    l2_cache={},
                    l2_runtime_by_edges={},
                    build_runtime_by_edges={problem.n - 1: 0.0},
                    total_build_runtime=0.0,
                )
                self._cache[key] = cache
            cache.ensure_to_target(problem.m)
            graph = cache.graph_for_edges(problem.m)
            l2 = cache.l2_cache.get(problem.m)
            if l2 is None:
                l2_start = time.perf_counter()
                l2 = algebraic_connectivity(graph)
                cache.l2_cache[problem.m] = l2
                cache.l2_runtime_by_edges[problem.m] = time.perf_counter() - l2_start
            runtime = cache.runtime_to_edges(problem.m)
            reused = True
            runtime_mode = "cumulative_to_m"
        else:
            start = time.perf_counter()
            graph = nx.path_graph(problem.n)
            graph = self._completion.complete(
                graph=graph,
                target_edges=problem.m,
                rng=random.Random(problem.seed),
                max_steps=config.fvg_max_steps,
            )
            l2 = algebraic_connectivity(graph)
            runtime = time.perf_counter() - start
            reused = False
            runtime_mode = "single_run"
        return BaselineResult(
            name=self.name,
            problem=problem,
            graph=graph,
            final_lambda2=l2,
            runtime_sec=runtime,
            metadata={
                "initial_graph": "path_tree",
                "initial_edges": problem.n - 1,
                "final_edges": graph.number_of_edges(),
                "cache_reused": reused,
                "runtime_mode": runtime_mode,
            },
        )
