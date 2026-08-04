from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .graph_math import (
    complete_edge_count,
    edge_count,
    laplacian,
    normalized_density,
)


def _grounded_incidence_vector(
    u: int, v: int, n: int, ground: int = 0
) -> np.ndarray:
    g = np.zeros(n - 1, dtype=np.float64)
    for node, sign in ((u, 1.0), (v, -1.0)):
        if node == ground:
            continue
        idx = node if node < ground else node - 1
        g[idx] += sign
    return g


def _lambda2_multiplicity(
    evals: np.ndarray,
    eps_abs: float = 1e-12,
    eps_rel: float = 1e-10,
) -> int:
    lambda2 = float(evals[1])
    tol = max(eps_abs, eps_rel * max(1.0, abs(lambda2)))
    mult_mask = np.abs(evals - lambda2) <= tol
    mult_mask[0] = False
    return int(np.sum(mult_mask))


class SpectralTracker:
    """Incrementally tracks Laplacian spectral properties after edge additions.

    Tier 1 — Sherman-Morrison on grounded inverse M = L_g^{-1} (O(n²) per step)
    Tier 2 — exact R_G via trace identity + first-order eigenvalue estimates
    Tier 3 — full eigendecomposition every K steps for drift correction (O(n³))

    Mathematical identities
    -----------------------
    Edge (i,j):  L_new = L + v v^T   where  v = e_i - e_j   (rank-1).

    Pseudoinverse update (Sherman-Morrison, valid since v ⟂ 1):
        L⁺_new = L⁺ - (L⁺v)(L⁺v)^T / (1 + v^T L⁺ v)

    Total effective resistance (exact, from trace identity):
        R_G = n · trace(L⁺)
        ΔR_G = -n · ‖L⁺v‖² / (1 + v^T L⁺ v)

    First-order eigenvalue estimate:
        λ_k' ≈ λ_k + (φ_k[i] - φ_k[j])²

    Note: the Fiedler gap score H(i,j) = (φ₂[i] - φ₂[j])² is precisely the
    first-order estimate of Δλ₂.  Eigenvectors are kept from the last exact
    recomputation and not updated between Tier 3 resets (they change slowly
    under rank-1 perturbations).
    """

    def __init__(
        self,
        adj: np.ndarray,
        ground: int = 0,
        exact_reset_every: int = 32,
        eps_abs: float = 1e-12,
        eps_rel: float = 1e-10,
    ):
        self.n = int(adj.shape[0])
        if self.n < 2:
            raise ValueError("SpectralTracker requires n >= 2")
        self.ground = int(ground)
        if not (0 <= self.ground < self.n):
            raise ValueError(f"ground out of range: {self.ground}")

        self.adj = adj.astype(np.uint8, copy=True)
        self.exact_reset_every = int(exact_reset_every)
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)
        self.steps_since_exact = 0

        self._init_grounded_inverse()
        self._recompute_exact_eigh()

        self.lambda2 = float(self._evals[1])
        self.lambda3 = float(self._evals[2]) if self.n >= 3 else 0.0
        self.lambda4 = float(self._evals[3]) if self.n >= 4 else 0.0
        self.phi2 = self._evecs[:, 1].copy()
        self.phi3 = self._evecs[:, 2].copy() if self.n >= 3 else np.zeros(self.n)
        self.phi4 = self._evecs[:, 3].copy() if self.n >= 4 else np.zeros(self.n)
        self.RG = float(self.n) * float(np.sum(1.0 / np.maximum(self._evals[1:], 1e-14)))
        self.multiplicity = _lambda2_multiplicity(self._evals, self.eps_abs, self.eps_rel)
        self.P_min = self._compute_P_min()

    # ------------------------------------------------------------------ #
    #  Effective resistance from grounded inverse
    # ------------------------------------------------------------------ #

    def _eff_resistance(self, u: int, v: int) -> float:
        """O(1) effective resistance between nodes u and v from M."""
        if u == v:
            return 0.0
        if u == self.ground:
            vi = v if v < self.ground else v - 1
            return float(max(self._M[vi, vi], 0.0))
        if v == self.ground:
            ui = u if u < self.ground else u - 1
            return float(max(self._M[ui, ui], 0.0))
        ui = u if u < self.ground else u - 1
        vi = v if v < self.ground else v - 1
        val = self._M[ui, ui] + self._M[vi, vi] - 2.0 * self._M[ui, vi]
        return float(max(val, 0.0))

    def _compute_P_min(self) -> float:
        """Compute min resistance curvature P_min = min_i p_i.

        p_i = 1 - (1/2) Σ_{j~i} ω_{ij}
        where ω_{ij} = R_eff(i,j) is the effective resistance on edge (i,j).

        Cost: O(m) = O(n·d_avg) effective resistance lookups.
        """
        p = np.ones(self.n, dtype=np.float64)
        for node in range(self.n):
            neighbors = np.flatnonzero(self.adj[node] > 0)
            if len(neighbors) == 0:
                continue
            res_sum = sum(self._eff_resistance(node, int(nbr)) for nbr in neighbors)
            p[node] = 1.0 - 0.5 * res_sum
        return float(np.min(p))

    # ------------------------------------------------------------------ #
    #  Tier 1 & 3
    # ------------------------------------------------------------------ #

    def _init_grounded_inverse(self) -> None:
        L = laplacian(self.adj)
        keep = [i for i in range(self.n) if i != self.ground]
        Lg = L[np.ix_(keep, keep)]
        ridge = 1e-10 * np.eye(Lg.shape[0], dtype=np.float64)
        self._M = np.linalg.inv(Lg + ridge)

    def _recompute_exact_eigh(self) -> None:
        L = laplacian(self.adj)
        vals, vecs = np.linalg.eigh(L)
        order = np.argsort(vals)
        self._evals = np.asarray(vals[order], dtype=np.float64)
        self._evecs = np.asarray(vecs[:, order], dtype=np.float64)

    def force_exact(self) -> None:
        self._init_grounded_inverse()
        self._recompute_exact_eigh()
        self.steps_since_exact = 0
        self.lambda2 = float(self._evals[1])
        self.lambda3 = float(self._evals[2]) if self.n >= 3 else 0.0
        self.lambda4 = float(self._evals[3]) if self.n >= 4 else 0.0
        self.phi2 = self._evecs[:, 1].copy()
        self.phi3 = self._evecs[:, 2].copy() if self.n >= 3 else np.zeros(self.n)
        self.phi4 = self._evecs[:, 3].copy() if self.n >= 4 else np.zeros(self.n)
        self.RG = float(self.n) * float(np.sum(1.0 / np.maximum(self._evals[1:], 1e-14)))
        self.multiplicity = _lambda2_multiplicity(self._evals, self.eps_abs, self.eps_rel)
        self.P_min = self._compute_P_min()

    # ------------------------------------------------------------------ #
    #  Tier 2: add_edge
    # ------------------------------------------------------------------ #

    def add_edge(self, i: int, j: int) -> None:
        if i == j:
            raise ValueError(f"Self-loop not allowed: ({i},{j})")
        if self.adj[i, j] != 0:
            raise ValueError(f"Edge ({i},{j}) already present")

        self.adj[i, j] = 1
        self.adj[j, i] = 1

        # ---- 1. Sherman-Morrison on grounded inverse ----
        g = _grounded_incidence_vector(i, j, self.n, self.ground)
        x = self._M @ g                     # M_old @ g
        denom = 1.0 + float(g @ x)          # 1 + v^T L⁺_old v
        if denom <= 1e-14 or not np.isfinite(denom):
            self.force_exact()
            return

        # ---- 2. Exact R_G update via trace identity (uses L⁺_old) ----
        b = self.adj[self.ground].astype(np.float64, copy=True)
        b = np.delete(b, self.ground)
        s_old = self._M @ b                 # M_old @ b
        alpha = float(x.sum())
        t_old = float(s_old.sum())
        if abs(1.0 - t_old) > 1e-12:
            wp_old = -alpha / (1.0 - t_old)
        else:
            wp_old = -alpha
        wg_old = x - s_old * wp_old
        norm_Lplus_old_v_sq = float(wg_old @ wg_old) + wp_old * wp_old
        self.RG += -float(self.n) * norm_Lplus_old_v_sq / denom

        # ---- 3. Apply SM: M ← M - x x^T / denom ----
        self._M -= np.outer(x, x) / denom

        # ---- 4. First-order eigenvalue estimates ----
        #  λ_k' ≈ λ_k + (φ_k[i] - φ_k[j])²
        gap2 = float((self.phi2[i] - self.phi2[j]) ** 2)
        gap3 = float((self.phi3[i] - self.phi3[j]) ** 2)
        gap4 = float((self.phi4[i] - self.phi4[j]) ** 2)

        self.lambda2 += gap2
        self.lambda3 += gap3
        self.lambda4 += gap4

        # ---- 5. Multiplicity: λ₂ does NOT shift if multiplicity > 1 ----
        #  (the shifted eigenvalue goes UP, floor stays at λ₂)
        tol = max(self.eps_abs, self.eps_rel * max(1.0, abs(self.lambda2 - gap2)))
        if abs((self.lambda2 - gap2) - (self.lambda3 - gap3)) <= tol and self.multiplicity > 1:
            self.lambda2 -= gap2  # revert: the cluster floor doesn't move

        # ---- 6. Update P_min (min resistance curvature) ----
        self.P_min = self._compute_P_min()

        # ---- 7. Periodic exact reset ----
        self.steps_since_exact += 1
        if self.steps_since_exact >= self.exact_reset_every:
            self.force_exact()

    # ------------------------------------------------------------------ #
    #  Accessors
    # ------------------------------------------------------------------ #

    def get_spectral_features(self) -> Tuple[
        float, float, float,
        np.ndarray, np.ndarray, np.ndarray,
        float, int, np.ndarray,
    ]:
        return (
            float(self.lambda2),
            float(self.lambda3),
            float(self.lambda4),
            self.phi2.copy().astype(np.float64),
            self.phi3.copy().astype(np.float64),
            self.phi4.copy().astype(np.float64),
            float(self.RG),
            int(self.multiplicity),
            self._evecs.copy().astype(np.float64),
        )

    @property
    def m(self) -> int:
        return edge_count(self.adj)

    @property
    def rho_current(self) -> float:
        return normalized_density(self.n, self.m)

    @property
    def m_max(self) -> int:
        return complete_edge_count(self.n)

    def get_P_min(self) -> float:
        """Minimum resistance curvature (updated after each add_edge)."""
        return float(self.P_min)

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        return {
            "adj": self.adj.copy(),
            "n": self.n,
            "ground": self.ground,
            "exact_reset_every": self.exact_reset_every,
            "eps_abs": self.eps_abs,
            "eps_rel": self.eps_rel,
            "steps_since_exact": self.steps_since_exact,
            "M": self._M.copy(),
            "evals": self._evals.copy(),
            "evecs": self._evecs.copy(),
            "lambda2": self.lambda2,
            "lambda3": self.lambda3,
            "lambda4": self.lambda4,
            "phi2": self.phi2.copy(),
            "phi3": self.phi3.copy(),
            "phi4": self.phi4.copy(),
            "RG": self.RG,
            "multiplicity": self.multiplicity,
            "P_min": self.P_min,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "SpectralTracker":
        obj = cls.__new__(cls)
        obj.n = int(state["n"])
        obj.ground = int(state["ground"])
        obj.exact_reset_every = int(state["exact_reset_every"])
        obj.eps_abs = float(state["eps_abs"])
        obj.eps_rel = float(state["eps_rel"])
        obj.steps_since_exact = int(state["steps_since_exact"])
        obj.adj = state["adj"].copy()
        obj._M = state["M"].copy()
        obj._evals = state["evals"].copy()
        obj._evecs = state["evecs"].copy()
        obj.lambda2 = float(state["lambda2"])
        obj.lambda3 = float(state["lambda3"])
        obj.lambda4 = float(state["lambda4"])
        obj.phi2 = state["phi2"].copy()
        obj.phi3 = state["phi3"].copy()
        obj.phi4 = state["phi4"].copy()
        obj.RG = float(state["RG"])
        obj.multiplicity = int(state["multiplicity"])
        obj.P_min = float(state.get("P_min", 0.0))
        return obj
