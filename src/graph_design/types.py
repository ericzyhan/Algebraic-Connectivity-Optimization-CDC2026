from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Any

import networkx as nx


@dataclass(frozen=True)
class DesignProblem:
    n: int
    m: int
    seed: int | None = None

    @property
    def max_edges(self) -> int:
        return comb(self.n, 2)

    @property
    def density(self) -> float:
        min_edges = self.n - 1
        denom = self.max_edges - min_edges
        if denom <= 0:
            return 1.0
        return (self.m - min_edges) / denom


@dataclass
class BackboneCandidate:
    family: str
    graph: nx.Graph
    backbone_lambda2: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateEvaluation:
    family: str
    route_label: str
    backbone_edges: int
    backbone_lambda2: float
    final_graph: nx.Graph
    final_lambda2: float
    completion_gain: float
    runtime_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DesignResult:
    problem: DesignProblem
    route_label: str
    selected_family: str
    graph: nx.Graph
    final_lambda2: float
    backbone_lambda2: float
    completion_gain: float
    runtime_sec: float
    fallback_used: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselineResult:
    name: str
    problem: DesignProblem
    graph: nx.Graph
    final_lambda2: float
    runtime_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)
