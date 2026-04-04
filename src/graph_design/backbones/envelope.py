from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx

from graph_design.backbones.base import BackboneGenerator
from graph_design.config import DesignConfig
from graph_design.types import BackboneCandidate, DesignProblem


def _partition_for_level(n: int, k: int) -> list[int]:
    if k == n:
        return [1] * n
    part_size = n - k
    q, b = divmod(n, part_size)
    parts = [part_size] * q
    if b > 0:
        parts.append(b)
    parts.sort()
    return parts


def _m_env(n: int, k: int) -> int:
    if k == n:
        return n * (n - 1) // 2
    part_size = n - k
    q, b = divmod(n, part_size)
    value = (n * n - q * (part_size * part_size) - b * b) // 2
    return int(value)


def _max_feasible_level(n: int, m_target: int) -> int:
    # m_env(k) is monotone nondecreasing in k on [1, n-2].
    lo, hi = 1, n - 2
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if _m_env(n, mid) <= m_target:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


@dataclass
class EnvelopeBackboneGenerator(BackboneGenerator):
    family: str = "envelope"

    def generate(
        self,
        problem: DesignProblem,
        config: DesignConfig,
        rng: random.Random,
    ) -> BackboneCandidate | None:
        del config, rng
        n = problem.n
        m_target = problem.m

        max_edges = n * (n - 1) // 2
        if m_target >= max_edges:
            graph = nx.complete_graph(n)
            return BackboneCandidate(
                family=self.family,
                graph=graph,
                backbone_lambda2=float(n),
                metadata={
                    "k_level": n,
                    "parts": [1] * n,
                    "m_env": max_edges,
                },
            )

        if n <= 2:
            return None

        k = _max_feasible_level(n, m_target)
        edges = _m_env(n, k)
        parts = _partition_for_level(n, k)
        graph = nx.complete_multipartite_graph(*parts)
        return BackboneCandidate(
            family=self.family,
            graph=graph,
            backbone_lambda2=float(k),
            metadata={
                "k_level": k,
                "parts": parts,
                "m_env": edges,
            },
        )
