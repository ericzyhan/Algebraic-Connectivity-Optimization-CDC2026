from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .graph_core import non_edges
from .incremental_linear_algebra import IncrementalState

CandidateMode = Literal["er", "fiedler"]


@dataclass
class CandidateSet:
    edges: np.ndarray  # [K,2] int64
    er_scores: np.ndarray  # [K] float64


class CandidateGenerator:
    def __init__(self, top_k: int = 128, fraction_cap: float = 1.0, mode: CandidateMode = "er") -> None:
        self.top_k = int(top_k)
        self.fraction_cap = float(fraction_cap)
        mode = str(mode).strip().lower()
        if mode not in ("er", "fiedler"):
            raise ValueError(f"Unsupported candidate mode: {mode}. Supported: er, fiedler")
        self.mode: CandidateMode = mode  # type: ignore[assignment]

    def generate(self, state: IncrementalState) -> CandidateSet:
        all_non_edges = non_edges(state.adj)
        if not all_non_edges:
            return CandidateSet(
                edges=np.zeros((0, 2), dtype=np.int64),
                er_scores=np.zeros((0,), dtype=np.float64),
            )

        ers = np.asarray(
            [state.effective_resistance(u, v) for (u, v) in all_non_edges],
            dtype=np.float64,
        )
        if self.mode == "fiedler":
            _, phi2, _ = state.spectral_chart(force=False)
            fiedler_scores = np.asarray(
                [(float(phi2[u]) - float(phi2[v])) ** 2 for (u, v) in all_non_edges],
                dtype=np.float64,
            )
            order = np.argsort(-fiedler_scores)
        else:
            order = np.argsort(-ers)
        max_by_frac = max(1, int(round(len(all_non_edges) * self.fraction_cap)))
        k = min(self.top_k, len(all_non_edges), max_by_frac)
        chosen = order[:k]
        chosen_edges = np.asarray([all_non_edges[idx] for idx in chosen], dtype=np.int64)
        chosen_ers = ers[chosen]
        return CandidateSet(edges=chosen_edges, er_scores=chosen_ers)
