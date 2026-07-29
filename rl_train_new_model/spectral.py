from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from .graph_math import complete_edge_count, edge_count, laplacian, normalized_density


class SpectralTracker:
    """Incrementally tracks Laplacian spectral properties after edge additions.

    Implements the pipeline in docs/algorithm.md and docs/theory.md:

    - L+ and K := (L+)^2 maintained exactly via Sherman-Morrison / its rank-2
      square (b = e_i - e_j is always perp to 1 = ker L, so no special-casing
      is ever needed -- docs/CLAUDE.md invariant #3).
    - Bottom invariant subspace U (q = r + oversample columns) refreshed each
      step by warm-started block power iteration (Rayleigh-Ritz) on L+, with
      a periodic full eigh as the drift-correcting anchor.
    - Hard multiplicity r and the soft-degenerate band, per algorithm.md S3.1.
    - Exact two-sided secular bracket every step (needs only beta_2, delta');
      the secular ROOT itself is exact immediately after a full recompute and
      an approximation (tail lumped at the last tracked pole) otherwise --
      marked APPROX below, per docs/CLAUDE.md S7 style rule.
    - Z_accum / sigma_r(Z_accum) bookkeeping for the potential-based shaping
      term in shaping.py (docs/algorithm.md S5.2).
    """

    def __init__(
        self,
        adj: np.ndarray,
        oversample: int = 3,
        exact_reset_every: int = 50,
        power_iters: int = 4,
        eps_abs: float = 1e-12,
        eps_rel: float = 1e-10,
        drift_threshold: float = 1e-8,
        soft_band_rel: float = 1e-2,
    ):
        self.n = int(adj.shape[0])
        if self.n < 2:
            raise ValueError("SpectralTracker requires n >= 2")

        self.adj = adj.astype(np.uint8, copy=True)
        self.oversample = int(oversample)
        self.exact_reset_every = int(exact_reset_every)
        self.power_iters = int(power_iters)
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)
        self.drift_threshold = float(drift_threshold)
        self.soft_band_rel = float(soft_band_rel)

        # Populated by _full_recompute / _detect_multiplicity below.
        self.multiplicity = 1
        self._evals_full: Optional[np.ndarray] = None
        self._evecs_full: Optional[np.ndarray] = None
        self.Lplus: Optional[np.ndarray] = None
        self.K: Optional[np.ndarray] = None
        self.U: Optional[np.ndarray] = None
        self.subspace_evals: Optional[np.ndarray] = None
        self.RG: float = 0.0
        self.P_min: float = 0.0
        self.steps_since_exact = 0
        self.drift_residual = 0.0
        self.mu2 = 0.0
        self.mu3 = float("inf")
        self.soft_degenerate = False

        # Per-edge secular diagnostics (also feed the lightweight certificates).
        self.last_secular_delta: Optional[float] = None
        self.last_secular_bracket: Optional[Tuple[float, float]] = None
        self.last_secular_exact: bool = False
        self.last_beta2: float = 0.0
        self.last_delta_prime: float = 0.0

        self._z_accum_cols: List[np.ndarray] = []
        self._z_accum_U: Optional[np.ndarray] = None
        self._z_accum_r: int = 1

        self._full_recompute()
        self._detect_multiplicity()
        self._reset_z_accum()

    # ------------------------------------------------------------------ #
    #  Sizing
    # ------------------------------------------------------------------ #

    def _q(self, r: int) -> int:
        return int(min(max(r + self.oversample, 1), self.n - 1))

    # ------------------------------------------------------------------ #
    #  Full recompute (drift-correcting anchor; also gives the exact
    #  full spectrum used for one exact secular solve right afterward)
    # ------------------------------------------------------------------ #

    def _full_recompute(self) -> None:
        L = laplacian(self.adj)
        evals, evecs = np.linalg.eigh(L)
        order = np.argsort(evals)
        evals = np.asarray(evals[order], dtype=np.float64)
        evecs = np.asarray(evecs[:, order], dtype=np.float64)
        self._evals_full = evals
        self._evecs_full = evecs

        n = self.n
        inv = np.zeros(n, dtype=np.float64)
        inv[1:] = 1.0 / np.maximum(evals[1:], 1e-14)
        self.Lplus = (evecs * inv) @ evecs.T
        self.K = (evecs * (inv * inv)) @ evecs.T

        q = self._q(self.multiplicity)
        self.U = evecs[:, 1 : 1 + q].copy()
        self.subspace_evals = evals[1 : 1 + q].copy()

        self.RG = float(n) * float(np.sum(1.0 / np.maximum(evals[1:], 1e-14)))
        self.P_min = self._compute_P_min_from_Lplus()
        self.steps_since_exact = 0
        self.drift_residual = 0.0

    # ------------------------------------------------------------------ #
    #  Block power iteration (Rayleigh-Ritz warm start) on L+
    # ------------------------------------------------------------------ #

    def _refine_subspace(self) -> None:
        Q = self.U
        for _ in range(self.power_iters):
            Y = self.Lplus @ Q
            # Re-project out the all-ones component every sweep (numerical hygiene).
            Y = Y - Y.mean(axis=0, keepdims=True)
            Q, _ = np.linalg.qr(Y)
        T = Q.T @ self.Lplus @ Q
        w, Vt = np.linalg.eigh(T)
        order = np.argsort(w)[::-1]  # descending L+ eigenvalues -> ascending L eigenvalues
        w = w[order]
        Vt = Vt[:, order]
        Q = Q @ Vt
        w_safe = np.maximum(w, 1e-14)
        self.U = Q
        self.subspace_evals = 1.0 / w_safe

    # ------------------------------------------------------------------ #
    #  Multiplicity detection (algorithm.md S3.1)
    # ------------------------------------------------------------------ #

    def _detect_multiplicity(self) -> int:
        q_avail = int(self.subspace_evals.shape[0])
        mu2 = float(self.subspace_evals[0])
        tol = max(self.eps_abs, self.eps_rel * max(1.0, abs(mu2)))
        mask = np.abs(self.subspace_evals - mu2) <= tol
        r = int(np.sum(mask))
        r = max(1, min(r, q_avail))

        self.multiplicity = r
        self.mu2 = mu2
        self.mu3 = float(self.subspace_evals[r]) if r < q_avail else float("inf")
        denom = max(self.mu2, 1e-12)
        self.soft_degenerate = bool(
            np.isfinite(self.mu3) and (self.mu3 - self.mu2) < self.soft_band_rel * denom
        )

        if r >= q_avail and q_avail < self.n - 1:
            # Tracked window too narrow to bound the true multiplicity -- force an
            # exact refresh (with a wider q informed by this r) on the next edge.
            self.steps_since_exact = self.exact_reset_every
        return r

    # ------------------------------------------------------------------ #
    #  Z_accum / sigma_r bookkeeping for potential-based shaping
    # ------------------------------------------------------------------ #

    def _reset_z_accum(self) -> None:
        r = self.multiplicity
        self._z_accum_r = r
        self._z_accum_U = self.U[:, :r].copy()
        self._z_accum_cols = []

    def sigma_r_accum(self) -> float:
        if not self._z_accum_cols:
            return 0.0
        r = self._z_accum_r
        Z = np.stack(self._z_accum_cols, axis=1)  # (r, count)
        if Z.shape[1] < r:
            return 0.0
        svals = np.linalg.svd(Z, compute_uv=False)
        return float(svals[-1])

    # ------------------------------------------------------------------ #
    #  Resistance curvature (Devriendt-Lambiotte)
    # ------------------------------------------------------------------ #

    def _compute_P_min_from_Lplus(self) -> float:
        d = np.diag(self.Lplus)
        p = np.ones(self.n, dtype=np.float64)
        adj_bool = self.adj > 0
        for node in range(self.n):
            neighbors = np.flatnonzero(adj_bool[node])
            if neighbors.size == 0:
                continue
            Rij = d[node] + d[neighbors] - 2.0 * self.Lplus[node, neighbors]
            p[node] = 1.0 - 0.5 * float(np.sum(np.maximum(Rij, 0.0)))
        return float(np.min(p))

    def get_P_min(self) -> float:
        return float(self.P_min)

    # ------------------------------------------------------------------ #
    #  Numerical hygiene (docs/algorithm.md S7)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _recenter(M: np.ndarray) -> np.ndarray:
        # Double-centering J M J via row-mean then column-mean subtraction
        # (equivalent for symmetric M, standard classical-MDS trick).
        M = M - M.mean(axis=1, keepdims=True)
        M = M - M.mean(axis=0, keepdims=True)
        return M

    def _drift_residual(self) -> float:
        # NOTE: O(n^3). Fine at the current curriculum's n (<=64); if n grows
        # into the "low thousands" this should move to a periodic-only check.
        L = laplacian(self.adj)
        resid = L @ self.Lplus @ L - L
        denom = float(np.linalg.norm(L, ord="fro"))
        if denom <= 1e-14:
            return 0.0
        return float(np.linalg.norm(resid, ord="fro") / denom)

    # ------------------------------------------------------------------ #
    #  Secular equation (theory.md S4): exact bracket every step, exact
    #  root immediately after a full recompute, tail-lumped APPROX root
    #  otherwise.
    # ------------------------------------------------------------------ #

    def _solve_secular_for_edge(self, i: int, j: int) -> None:
        if self.multiplicity != 1:
            self.last_secular_delta = None
            self.last_secular_bracket = None
            self.last_secular_exact = False
            return

        mu2 = self.mu2
        mu3 = self.mu3 if np.isfinite(self.mu3) else mu2

        exact = self.steps_since_exact == 0 and self._evecs_full is not None
        if exact:
            evecs = self._evecs_full
            beta = evecs[i, 1:] - evecs[j, 1:]
            mu_tail = self._evals_full[1:]
        else:
            beta = self.U[i, :] - self.U[j, :]
            mu_tail = self.subspace_evals

        beta2 = float(beta[0]) if beta.shape[0] > 0 else 0.0
        delta_prime = float(mu3 - mu2)

        # Exact two-sided bracket -- valid regardless of tail truncation.
        if (delta_prime + 2.0) > 0:
            lo = (delta_prime / (delta_prime + 2.0)) * (beta2 * beta2)
        else:
            lo = 0.0
        hi = beta2 * beta2
        self.last_secular_bracket = (float(lo), float(hi))
        self.last_beta2 = beta2
        self.last_delta_prime = delta_prime

        if beta2 == 0.0 or delta_prime <= 0.0 or not np.isfinite(mu3):
            self.last_secular_delta = 0.5 * (lo + hi)
            self.last_secular_exact = False
            return

        rest_mu = mu_tail[1:]
        rest_beta2 = beta[1:] ** 2

        # APPROX: beyond the tracked window, lump the remaining Parseval mass
        # (sum_{j>=2} beta_j^2 = ||b||^2 = 2, exact) at a single representative
        # pole (the last tracked eigenvalue). Error shrinks as oversample grows;
        # bracket above does not depend on this approximation.
        tail_mass = 0.0
        tail_pole = mu3
        if not exact:
            tail_mass = max(2.0 - beta2 * beta2 - float(np.sum(rest_beta2)), 0.0)
            tail_pole = float(mu_tail[-1]) if mu_tail.shape[0] > 0 else mu3

        def w_theta(theta: float) -> float:
            val = 1.0 + (beta2 * beta2) / (mu2 - theta)
            if rest_mu.shape[0] > 0:
                val += float(np.sum(rest_beta2 / (rest_mu - theta)))
            if tail_mass > 0.0 and (tail_pole - theta) != 0.0:
                val += tail_mass / (tail_pole - theta)
            return val

        eps = 1e-9 * max(1.0, delta_prime)
        lo_theta = mu2 + eps
        hi_theta = mu3 - eps
        if lo_theta >= hi_theta:
            self.last_secular_delta = 0.5 * (lo + hi)
            self.last_secular_exact = False
            return

        try:
            theta2 = brentq(w_theta, lo_theta, hi_theta, xtol=1e-13, maxiter=100)
            delta = float(theta2 - mu2)
        except Exception:
            delta = 0.5 * (lo + hi)
            exact = False

        self.last_secular_delta = delta
        self.last_secular_exact = bool(exact)

    # ------------------------------------------------------------------ #
    #  Tier 2 (r>1): RRQR ranking only -- no local swap refinement, per
    #  project decision: the GNN policy keeps the final pick among the
    #  RRQR-ranked survivors instead of a deterministic macro-action.
    # ------------------------------------------------------------------ #

    def rrqr_rank(self, Z: np.ndarray) -> np.ndarray:
        """Column-pivoted QR ranking of Z (r x k), most-important column first.

        No swap refinement is applied (see docs/CLAUDE.md decision log) --
        this is the base RRQR ranking only, handed to candidacy.py to shortlist
        survivors that the policy then chooses among.
        """
        if Z.shape[1] == 0:
            return np.zeros((0,), dtype=np.int64)
        # scipy/numpy column-pivoted QR: piv[k] is the k-th selected original column.
        from scipy.linalg import qr as scipy_qr

        _, _, piv = scipy_qr(Z, mode="economic", pivoting=True)
        return np.asarray(piv, dtype=np.int64)

    # ------------------------------------------------------------------ #
    #  Certificates (lightweight, logging-only -- optimality-gap.md S1, S2)
    # ------------------------------------------------------------------ #

    def interlacing_bound(self, k_remaining: int) -> float:
        # NOTE: uses the spectrum from the last full recompute; staleness is
        # bounded by exact_reset_every. Logging-only, not a paper-grade cert.
        evals = self._evals_full
        if evals is None or evals.shape[0] == 0:
            return float("inf")
        idx = 1 + max(0, int(k_remaining))
        if idx >= evals.shape[0]:
            return float(evals[-1])
        return float(evals[idx])

    def resistance_sandwich(self) -> Tuple[float, float, float]:
        d = np.diag(self.Lplus)
        R = d[:, None] + d[None, :] - 2.0 * self.Lplus
        np.fill_diagonal(R, -np.inf)
        R_max = float(np.max(R)) if self.n > 1 else 0.0
        n = float(self.n)
        lower = n / max(self.RG, 1e-12)
        upper = 2.0 / R_max if R_max > 0 else float("inf")
        loose_upper = n * (n - 1.0) / max(self.RG, 1e-12)
        return lower, upper, loose_upper

    # ------------------------------------------------------------------ #
    #  Tier 2: incremental Delta P_min estimate on survivors
    #  (algorithm.md S4 Tier 2 formula sheet)
    # ------------------------------------------------------------------ #

    def delta_p_min_estimate(self, i_idx: np.ndarray, j_idx: np.ndarray) -> np.ndarray:
        """Approximate Delta p_i contribution for candidate edges (i,j).

        Delta p_i = (1/(2 gamma)) sum_{k~i} (w_i - w_k)^2, restricted to the
        two endpoints (new-neighbour term folded in via the edge itself).
        This is a cheap O(1)-per-candidate proxy fed as a Tier-2 feature, not
        the reward (the reward uses the exact post-hoc P_min recompute).
        """
        Lplus = self.Lplus
        w = Lplus[:, i_idx] - Lplus[:, j_idx]  # (n, k)
        Re = w[i_idx, np.arange(i_idx.shape[0])] - w[j_idx, np.arange(i_idx.shape[0])]
        gamma = 1.0 + Re
        gamma = np.where(np.abs(gamma) < 1e-10, 1e-10, gamma)
        adj_bool = self.adj > 0
        out = np.zeros(i_idx.shape[0], dtype=np.float64)
        for idx in range(i_idx.shape[0]):
            i = int(i_idx[idx])
            j = int(j_idx[idx])
            wcol = w[:, idx]
            acc = 0.0
            for node in (i, j):
                neighbors = np.flatnonzero(adj_bool[node])
                if neighbors.size == 0:
                    continue
                diff = wcol[node] - wcol[neighbors]
                acc += float(np.sum(diff * diff))
            out[idx] = acc / (2.0 * gamma[idx])
        return out

    # ------------------------------------------------------------------ #
    #  Edge addition (Tier-0 exact updates, S1-S2 of algorithm.md)
    # ------------------------------------------------------------------ #

    def add_edge(self, i: int, j: int) -> None:
        if i == j:
            raise ValueError(f"Self-loop not allowed: ({i},{j})")
        if self.adj[i, j] != 0:
            raise ValueError(f"Edge ({i},{j}) already present")

        w = self.Lplus[:, i] - self.Lplus[:, j]
        Re = float(w[i] - w[j])
        gamma = 1.0 + Re
        if gamma <= 1e-10 or not np.isfinite(gamma):
            self.adj[i, j] = 1
            self.adj[j, i] = 1
            self._full_recompute()
            self._detect_multiplicity()
            self._reset_z_accum()
            return

        s = self.K[:, i] - self.K[:, j]
        norm_Lplus_b_sq = float(w @ w)
        delta_RG = -float(self.n) * norm_Lplus_b_sq / gamma

        r_before = self.multiplicity
        Ur_before = self._z_accum_U
        z_before = Ur_before[i, :] - Ur_before[j, :]

        self._solve_secular_for_edge(i, j)

        Lplus_new = self.Lplus - np.outer(w, w) / gamma
        K_new = (
            self.K
            - (np.outer(s, w) + np.outer(w, s)) / gamma
            + (norm_Lplus_b_sq / (gamma * gamma)) * np.outer(w, w)
        )
        Lplus_new = self._recenter(Lplus_new)
        K_new = self._recenter(K_new)

        self.adj[i, j] = 1
        self.adj[j, i] = 1
        self.Lplus = Lplus_new
        self.K = K_new
        self.RG += delta_RG

        self._z_accum_cols.append(z_before)

        self.steps_since_exact += 1
        if self.steps_since_exact >= self.exact_reset_every:
            self._full_recompute()
        else:
            drift = self._drift_residual()
            self.drift_residual = drift
            if drift > self.drift_threshold:
                self._full_recompute()
            else:
                self._refine_subspace()

        r_after = self._detect_multiplicity()
        if r_after < r_before:
            self._reset_z_accum()

        self.P_min = self._compute_P_min_from_Lplus()

    def force_exact(self) -> None:
        self._full_recompute()
        self._detect_multiplicity()

    # ------------------------------------------------------------------ #
    #  Accessors (kept close to the old incremental_spectral.py shape;
    #  the trailing element is now the (n, q) bottom subspace U, not a
    #  full n x n eigenvector matrix -- see candidacy.py)
    # ------------------------------------------------------------------ #

    def get_spectral_features(self) -> Tuple[
        float, float, float,
        np.ndarray, np.ndarray, np.ndarray,
        float, int, np.ndarray,
    ]:
        q = self.U.shape[1]
        phi2 = self.U[:, 0].copy()
        phi3 = self.U[:, 1].copy() if q > 1 else np.zeros(self.n, dtype=np.float64)
        phi4 = self.U[:, 2].copy() if q > 2 else np.zeros(self.n, dtype=np.float64)
        lambda2 = float(self.subspace_evals[0])
        lambda3 = float(self.subspace_evals[1]) if q > 1 else lambda2
        lambda4 = float(self.subspace_evals[2]) if q > 2 else lambda2
        return (
            lambda2,
            lambda3,
            lambda4,
            phi2,
            phi3,
            phi4,
            float(self.RG),
            int(self.multiplicity),
            self.U.copy(),
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

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        return {
            "adj": self.adj.copy(),
            "n": self.n,
            "oversample": self.oversample,
            "exact_reset_every": self.exact_reset_every,
            "power_iters": self.power_iters,
            "eps_abs": self.eps_abs,
            "eps_rel": self.eps_rel,
            "drift_threshold": self.drift_threshold,
            "soft_band_rel": self.soft_band_rel,
            "steps_since_exact": self.steps_since_exact,
            "drift_residual": self.drift_residual,
            "Lplus": self.Lplus.copy(),
            "K": self.K.copy(),
            "evals_full": self._evals_full.copy() if self._evals_full is not None else None,
            "evecs_full": self._evecs_full.copy() if self._evecs_full is not None else None,
            "U": self.U.copy(),
            "subspace_evals": self.subspace_evals.copy(),
            "multiplicity": self.multiplicity,
            "mu2": self.mu2,
            "mu3": self.mu3,
            "soft_degenerate": self.soft_degenerate,
            "RG": self.RG,
            "P_min": self.P_min,
            "z_accum_cols": [c.copy() for c in self._z_accum_cols],
            "z_accum_U": self._z_accum_U.copy() if self._z_accum_U is not None else None,
            "z_accum_r": self._z_accum_r,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "SpectralTracker":
        obj = cls.__new__(cls)
        obj.n = int(state["n"])
        obj.oversample = int(state["oversample"])
        obj.exact_reset_every = int(state["exact_reset_every"])
        obj.power_iters = int(state["power_iters"])
        obj.eps_abs = float(state["eps_abs"])
        obj.eps_rel = float(state["eps_rel"])
        obj.drift_threshold = float(state["drift_threshold"])
        obj.soft_band_rel = float(state["soft_band_rel"])
        obj.steps_since_exact = int(state["steps_since_exact"])
        obj.drift_residual = float(state["drift_residual"])
        obj.adj = state["adj"].copy()
        obj.Lplus = state["Lplus"].copy()
        obj.K = state["K"].copy()
        obj._evals_full = state["evals_full"].copy() if state["evals_full"] is not None else None
        obj._evecs_full = state["evecs_full"].copy() if state["evecs_full"] is not None else None
        obj.U = state["U"].copy()
        obj.subspace_evals = state["subspace_evals"].copy()
        obj.multiplicity = int(state["multiplicity"])
        obj.mu2 = float(state["mu2"])
        obj.mu3 = float(state["mu3"])
        obj.soft_degenerate = bool(state["soft_degenerate"])
        obj.RG = float(state["RG"])
        obj.P_min = float(state["P_min"])
        obj._z_accum_cols = [c.copy() for c in state.get("z_accum_cols", [])]
        obj._z_accum_U = (
            state["z_accum_U"].copy() if state.get("z_accum_U") is not None else None
        )
        obj._z_accum_r = int(state.get("z_accum_r", obj.multiplicity))
        obj.last_secular_delta = None
        obj.last_secular_bracket = None
        obj.last_secular_exact = False
        obj.last_beta2 = 0.0
        obj.last_delta_prime = 0.0
        return obj
