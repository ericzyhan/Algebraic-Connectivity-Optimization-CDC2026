from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import spectral
from .graph_core import add_edge, edge_count


@dataclass
class IncrementalStateConfig:
    refresh_every_steps: int = 1
    exact_reset_every_steps: int = 32
    eigsh_cutoff_n: int = 96


class IncrementalState:
    """
    Maintains:
    - adjacency
    - grounded Laplacian inverse updated by Sherman-Morrison rank-1 updates
    - lazily refreshed spectral chart (lambda2, phi2, phi3)
    """

    def __init__(
        self,
        adj: np.ndarray,
        config: Optional[IncrementalStateConfig] = None,
        ground: int = 0,
    ) -> None:
        self.adj = adj.astype(bool, copy=True)
        self.n = self.adj.shape[0]
        self.ground = int(ground)
        self.config = config or IncrementalStateConfig()
        self.steps_since_exact_reset = 0
        self.step_count = 0
        self._inv_lg = spectral.grounded_laplacian_inverse(self.adj, self.ground)
        self._lam2_cache: Optional[float] = None
        self._phi2_cache: Optional[np.ndarray] = None
        self._phi3_cache: Optional[np.ndarray] = None
        self._spectral_age = 10**9

    def clone(self) -> "IncrementalState":
        out = IncrementalState(self.adj.copy(), self.config, self.ground)
        out.steps_since_exact_reset = self.steps_since_exact_reset
        out.step_count = self.step_count
        out._inv_lg = self._inv_lg.copy()
        out._lam2_cache = self._lam2_cache
        out._phi2_cache = None if self._phi2_cache is None else self._phi2_cache.copy()
        out._phi3_cache = None if self._phi3_cache is None else self._phi3_cache.copy()
        out._spectral_age = self._spectral_age
        return out

    @property
    def e(self) -> int:
        return edge_count(self.adj)

    @property
    def degrees(self) -> np.ndarray:
        return self.adj.sum(axis=1).astype(np.int64)

    def add_edge(self, u: int, v: int) -> None:
        add_edge(self.adj, u, v)
        self._sherman_morrison_update(u, v)
        self.step_count += 1
        self.steps_since_exact_reset += 1
        self._spectral_age += 1
        if self.steps_since_exact_reset >= self.config.exact_reset_every_steps:
            self._recompute_exact_inverse()
            self.steps_since_exact_reset = 0

    def _recompute_exact_inverse(self) -> None:
        self._inv_lg = spectral.grounded_laplacian_inverse(self.adj, self.ground)

    def _sherman_morrison_update(self, u: int, v: int) -> None:
        g = spectral.grounded_incidence_vector(u, v, self.n, self.ground)
        x = self._inv_lg @ g
        denom = 1.0 + float(g.T @ x)
        if denom <= 1e-12 or not np.isfinite(denom):
            self._recompute_exact_inverse()
            return
        self._inv_lg = self._inv_lg - np.outer(x, x) / denom

    def effective_resistance(self, u: int, v: int) -> float:
        return spectral.effective_resistance_from_grounded_inv(
            self._inv_lg, u, v, self.n, self.ground
        )

    def spectral_chart(self, force: bool = False) -> Tuple[float, np.ndarray, np.ndarray]:
        stale = self._spectral_age >= self.config.refresh_every_steps
        if force or self._lam2_cache is None or stale:
            warm = self._phi2_cache
            lam2, phi2, phi3 = spectral.two_smallest_nontrivial(
                self.adj, warm_start=warm, eigsh_cutoff_n=self.config.eigsh_cutoff_n
            )
            self._lam2_cache = float(lam2)
            self._phi2_cache = phi2
            self._phi3_cache = phi3
            self._spectral_age = 0
        return (
            float(self._lam2_cache),
            self._phi2_cache.copy(),
            self._phi3_cache.copy(),
        )
