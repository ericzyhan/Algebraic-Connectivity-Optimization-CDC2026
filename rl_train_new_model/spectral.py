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
        degeneracy_probe_rel: float = 1e-5,
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
        self.degeneracy_probe_rel = float(degeneracy_probe_rel)

        # Populated by _full_recompute / _detect_multiplicity below.
        self.multiplicity = 1
        self._evals_full: Optional[np.ndarray] = None
        self._evecs_full: Optional[np.ndarray] = None
        self.Lplus: Optional[np.ndarray] = None
        self.K: Optional[np.ndarray] = None
        self.U: Optional[np.ndarray] = None
        self.subspace_evals: Optional[np.ndarray] = None
        self.RG: float = 0.0
        # p is carried as the full vector, not just its min: the closed-form
        # Delta p update (identity 5 below) is incremental, and Foster's
        # sum_i p_i = 1 then comes for free as a drift certificate.
        self.p: Optional[np.ndarray] = None
        self.P_min: float = 0.0
        self.steps_since_exact = 0
        self.drift_residual = 0.0
        self.mu2 = 0.0
        self.mu3 = float("inf")
        self.soft_degenerate = False
        self.unresolved_degeneracy = False

        # Per-edge secular diagnostics (also feed the lightweight certificates).
        self.last_secular_delta: Optional[float] = None
        self.last_secular_bracket: Optional[Tuple[float, float]] = None
        self.last_secular_exact: bool = False
        self.last_beta2: float = 0.0
        self.last_delta_prime: float = 0.0

        # Cheap per-step drift certificates (replace the O(n^3) residual).
        self.cert_foster: float = 0.0
        self.cert_trace: float = 0.0
        self.cert_k_sync: float = 0.0

        self._z_accum_cols: List[np.ndarray] = []
        self._z_accum_U: Optional[np.ndarray] = None
        self._z_accum_r: int = 1
        self._z_accum_epoch: int = 0

        # Deterministic r-1 deflation batch (step 2 of the algorithm).
        self._batch_active: bool = False
        self._batch_remaining: int = 0
        self._batch_expected_r: Optional[int] = None
        self._deferred_exact: bool = False
        self.last_deflation_ok: bool = True

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
        self.p = self._compute_p_from_Lplus()
        self.P_min = float(np.min(self.p))
        self.steps_since_exact = 0
        self.drift_residual = 0.0
        self.cert_foster = 0.0
        self.cert_trace = 0.0
        self.cert_k_sync = 0.0

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
        gap = self.mu3 - self.mu2
        self.soft_degenerate = bool(np.isfinite(self.mu3) and gap < self.soft_band_rel * denom)
        # Separate, much tighter band: below it the tracked gap is smaller than
        # what the block power iteration can actually resolve (~1e-6 relative),
        # so r is not decidable from these eigenvalues at the 1e-10 tolerance
        # above. add_edge escalates to an exact spectrum rather than guessing --
        # without that, a true plateau reads as r=1 and is only ever visible
        # immediately after a periodic anchor.
        self.unresolved_degeneracy = bool(
            np.isfinite(self.mu3) and gap <= self.degeneracy_probe_rel * denom
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
        self._z_accum_epoch += 1

    @property
    def z_accum_epoch(self) -> int:
        """Bumped every time the accumulator basis is re-drawn.

        A caller holding z columns it computed against an earlier basis (the
        deflation batch planner is the only one) compares this against the
        value it saw at begin_deflation_batch to tell that its columns went
        stale mid-batch -- which happens whenever an exact anchor lands
        between planning and commit.
        """
        return self._z_accum_epoch

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

    def _compute_p_from_Lplus(self) -> np.ndarray:
        """Reference O(sum deg) curvature vector, used only at the exact anchor.

        Note there is deliberately no clamp on Rij here: a negative effective
        resistance is drift, and clamping it would silently repair the Foster
        invariant (sum_i p_i = 1) that _cheap_drift relies on as a certificate.
        """
        d = np.diag(self.Lplus)
        p = np.ones(self.n, dtype=np.float64)
        adj_bool = self.adj > 0
        for node in range(self.n):
            neighbors = np.flatnonzero(adj_bool[node])
            if neighbors.size == 0:
                continue
            Rij = d[node] + d[neighbors] - 2.0 * self.Lplus[node, neighbors]
            p[node] = 1.0 - 0.5 * float(np.sum(Rij))
        return p

    def _delta_p(self, w: np.ndarray, i: int, j: int, Re: float, gamma: float) -> np.ndarray:
        """Exact Delta p for ALL n nodes in one matvec.

        Delta p_k = -1/2 sum_{j~k} Delta R_kj  with  Delta R_kj = -(w_k-w_j)^2/gamma, so

            Delta p_k = (1/2g)[deg_k w_k^2 - 2 w_k (Aw)_k + (A w^2)_k].

        Because L L+ = J and b = e_i - e_j is centered, L w = b exactly, hence
        Aw = deg*w - b -- which removes the (Aw) matvec and leaves only A(w^2).
        The two endpoints additionally pick up -R_e'/2 for the newly created
        neighbour, and R_e' = R_e/gamma.
        """
        deg = self.adj.sum(axis=1).astype(np.float64)  # pre-edge degrees
        w2 = w * w
        dp = (self.adj @ w2 - deg * w2) / (2.0 * gamma)
        # + 2*w*b/(2g), and b = e_i - e_j touches only these two entries.
        dp[i] += w[i] / gamma
        dp[j] -= w[j] / gamma
        # New-neighbour term at the endpoints: -R_e'/2 = -R_e/(2 gamma).
        half = 0.5 * Re / gamma
        dp[i] -= half
        dp[j] -= half
        return dp

    def get_P_min(self) -> float:
        return float(self.P_min)

    def get_p(self) -> np.ndarray:
        return self.p.copy()

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
        # NOTE: O(n^3). No longer called per edge -- reserved for the periodic
        # exact anchor as a ground-truth audit of the cheap certificates below.
        L = laplacian(self.adj)
        resid = L @ self.Lplus @ L - L
        denom = float(np.linalg.norm(L, ord="fro"))
        if denom <= 1e-14:
            return 0.0
        return float(np.linalg.norm(resid, ord="fro") / denom)

    def _cheap_drift(self, k_sync: float = 0.0) -> float:
        """Per-step drift signal from three O(n)/O(1) certificates.

        Replaces the O(n^3) residual, which defeated the "one pseudoinverse per
        step" amortization. Measured on a 1e-6 corruption of L+, Foster fires at
        4.9e-6 against the residual's 1.0e-6 -- i.e. it is the more sensitive of
        the two, so substituting it errs toward re-anchoring early.

          Foster : |sum_i p_i - 1|              (exact invariant, free from p)
          trace  : |R_G - n tr(L+)| / R_G       (accumulated R_G vs the matrix)
          k_sync : |w'w - (K_ii+K_jj-2K_ij)|    (L+ / K desync, passed in)
        """
        self.cert_foster = abs(float(np.sum(self.p)) - 1.0)
        tr = float(self.n) * float(np.trace(self.Lplus))
        self.cert_trace = abs(self.RG - tr) / max(abs(self.RG), 1e-12)
        self.cert_k_sync = float(k_sync)
        return max(self.cert_foster, self.cert_trace, self.cert_k_sync)

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
    #  Tier 2 (r>1): RRQR ranking, then MaxVol refinement. RRQR alone is
    #  greedy and only guarantees a rank-revealing ordering; MaxVol polishes
    #  the chosen r-subset to (near-)maximal volume, which is what makes the
    #  deterministic r-1 deflation batch well posed.
    # ------------------------------------------------------------------ #

    def rrqr_rank(self, Z: np.ndarray) -> np.ndarray:
        """Column-pivoted QR ranking of Z (r x k), most-important column first.

        This is the base ordering only. It is still used to rank the tail of
        the survivor list (everything MaxVol did not select), and as the
        fallback ordering when the pool is too small for MaxVol to run.
        """
        if Z.shape[1] == 0:
            return np.zeros((0,), dtype=np.int64)
        # scipy/numpy column-pivoted QR: piv[k] is the k-th selected original column.
        from scipy.linalg import qr as scipy_qr

        _, _, piv = scipy_qr(Z, mode="economic", pivoting=True)
        return np.asarray(piv, dtype=np.int64)

    # ------------------------------------------------------------------ #
    #  Deterministic r-1 deflation batch (step 2 of the algorithm)
    # ------------------------------------------------------------------ #

    @staticmethod
    def deflation_pool_conditioning(Z: np.ndarray) -> float:
        """cond(R) from the thin QR Z^T = QR -- the one precondition MaxVol needs.

        Selecting rows of Q by volume is equivalent to selecting rows of Z^T
        because det((Z^T)_I) = det(Q_I) det(R), which is exactly why R may be
        tossed -- but only while R is nonsingular. Q has r orthonormal columns,
        so once rank(Z) = r a nonsingular r x r submatrix is guaranteed to exist
        and MaxVol's dominance property certifies the one it returns is
        well-conditioned. Independence is therefore a property of the
        construction; what has to be checked is that the screened pool B'
        actually spans the eigenspace, and cond(R) is that check, for free,
        on an r x r matrix already in hand.
        """
        if Z.ndim != 2 or Z.shape[0] == 0 or Z.shape[1] < Z.shape[0]:
            return float("inf")
        _, R = np.linalg.qr(Z.T)
        return float(np.linalg.cond(R))

    @staticmethod
    def maxvol_warm_start(Q: np.ndarray) -> np.ndarray:
        """LU-with-partial-pivoting warm start for MaxVol.

        Partial pivoting already hoists large-magnitude rows to the top, so the
        first r pivots of P Q = L U are a cheap near-dominant starting set and
        MaxVol only has to polish it. Q = P L U, hence row m of P^T Q is row
        argmax(P[:, m]) of Q -- that argmax over the first r columns of P is
        exactly the "indices where a 1 appears in the first r rows" of the
        algorithm write-up.
        """
        from scipy.linalg import lu

        k, r = Q.shape
        if k < r:
            raise ValueError(f"MaxVol warm start needs k >= r, got Q with shape {Q.shape}")
        P, _, _ = lu(Q)
        return np.argmax(P[:, :r], axis=0).astype(np.int64)

    @staticmethod
    def maxvol(
        Q: np.ndarray,
        I_init: Optional[np.ndarray] = None,
        e: float = 1.02,
        max_iter: int = 100,
    ) -> np.ndarray:
        """Goreinov-Tyrtyshnikov MaxVol: r rows of Q (k x r) of near-maximal volume.

        Returns I with every entry of B = Q Q[I]^-1 satisfying |B_ab| <= e. That
        dominance property is the certificate: it bounds how far Q[I] is from the
        true maximum-volume submatrix, and hence bounds its conditioning. Since
        det((Z^T)_I) = det(Q_I) det(R), maximizing the volume of Q's rows
        maximizes the volume of Z^T's rows -- which is precisely why R may be
        tossed after the thin QR (see deflation_pool_conditioning).

        B is recomputed by a solve each sweep rather than rank-1 updated: the
        pool here is the screened shortlist (k <= 32) and r is the multiplicity
        (small), so O(k r^2) per sweep is already negligible and a fresh solve
        is the better-conditioned of the two.
        """
        Q = np.asarray(Q, dtype=np.float64)
        if Q.ndim != 2:
            raise ValueError(f"MaxVol expects a 2-D Q, got shape {Q.shape}")
        k, r = Q.shape
        if r == 0 or k < r:
            return np.arange(min(k, r), dtype=np.int64)

        if I_init is None:
            I = SpectralTracker.maxvol_warm_start(Q)
        else:
            I = np.asarray(I_init, dtype=np.int64).copy()
            if I.shape != (r,):
                raise ValueError(f"I_init must have shape ({r},), got {I.shape}")

        for _ in range(int(max_iter)):
            try:
                # B = Q Q[I]^-1, solved rather than inverted.
                B = np.linalg.solve(Q[I, :].T, Q.T).T
            except np.linalg.LinAlgError:
                # Singular starting minor: the pool cannot span the eigenspace,
                # which deflation_pool_conditioning is the designated check for.
                break
            row, col = np.unravel_index(int(np.argmax(np.abs(B))), B.shape)
            if abs(float(B[row, col])) <= e:
                break
            I[col] = row
        return I

    def maxvol_rank(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Order the columns of Z (r x k) MaxVol-selected-first, RRQR for the tail.

        Returns (order, selected), where `selected` is the MaxVol r-subset and
        `order` is a full ranking of all k columns with those r at the head.
        Falls back to the plain RRQR ordering when k < r, i.e. when the pool is
        too small for a volume-maximal r-subset to exist at all.

        MaxVol is warm-started from the RRQR pivots rather than its own LU
        default: the pivots are already in hand here for the tail ordering, and
        every MaxVol swap multiplies |det| by a factor > e > 1, so starting from
        RRQR makes the result provably no worse than RRQR. The LU warm start
        remains the default for a standalone `maxvol` call, where no pivoted QR
        has been paid for -- but it is only a local optimum, and on random pools
        it does sometimes finish below the greedy RRQR pick.
        """
        r, k = Z.shape
        piv = self.rrqr_rank(Z)
        if k < r or r == 0:
            return piv, piv[: min(k, r)]
        Q, _ = np.linalg.qr(Z.T)  # thin QR: Q is (k, r); R is tossed
        selected = self.maxvol(Q, I_init=piv[:r])
        tail = piv[~np.isin(piv, selected)]
        return np.concatenate([selected, tail]).astype(np.int64), selected.astype(np.int64)

    def begin_deflation_batch(self, size: int) -> None:
        """Freeze the lambda_2 eigenbasis for the duration of an r-1 edge batch.

        U is defined only up to an orthogonal transform within the degenerate
        block, and both eigh and the QR inside _refine_subspace return an
        arbitrary one. sigma_r is invariant under a rotation applied
        consistently but meaningless across mixed bases, so every z column of a
        batch must be taken in one frozen basis.
        """
        if size < 1:
            return
        self._batch_active = True
        self._batch_remaining = int(size)
        self._batch_expected_r = int(self.multiplicity)
        self.last_deflation_ok = True
        self._reset_z_accum()

    @property
    def batch_active(self) -> bool:
        return self._batch_active

    def end_deflation_batch(self) -> None:
        self._batch_active = False
        self._batch_remaining = 0
        self._batch_expected_r = None
        if self._deferred_exact:
            # An anchor came due mid-batch and was held back so it could not
            # re-draw the frozen basis; take it now.
            self._deferred_exact = False
            self._full_recompute()
            self._detect_multiplicity()
            self._reset_z_accum()

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

    def add_edge(
        self,
        i: int,
        j: int,
        *,
        cached: Optional[dict] = None,
        z: Optional[np.ndarray] = None,
    ) -> None:
        """Commit edge (i,j) and refresh every quantity the reward consumes.

        `cached` optionally carries the per-candidate scalars the screen already
        computed for this edge (`Re`, `norm_Lplus_b_sq`, `delta_RG`); `z` carries
        its column of Z = U^T B' in the frozen batch basis. Both are pure reuse
        -- omitting them recomputes the same values.
        """
        if i == j:
            raise ValueError(f"Self-loop not allowed: ({i},{j})")
        if self.adj[i, j] != 0:
            raise ValueError(f"Edge ({i},{j}) already present")

        w = self.Lplus[:, i] - self.Lplus[:, j]
        # Centering the O(n) update vector keeps every outer product built from
        # it centered, so the two O(n^2) double-centerings of L+ and K are not
        # needed (w = L+ b is centered exactly; this is roundoff hygiene only).
        w = w - w.mean()
        Re = float(cached["Re"]) if cached and "Re" in cached else float(w[i] - w[j])
        gamma = 1.0 + Re
        if gamma <= 1e-10 or not np.isfinite(gamma):
            self.adj[i, j] = 1
            self.adj[j, i] = 1
            self._full_recompute()
            self._detect_multiplicity()
            self._reset_z_accum()
            return

        s = self.K[:, i] - self.K[:, j]
        s = s - s.mean()

        # ||L+ b||^2 is an O(1) lookup in K; w'w is the same number the long way
        # round, so their disagreement is a free L+/K desync probe.
        norm_from_K = float(self.K[i, i] + self.K[j, j] - 2.0 * self.K[i, j])
        norm_from_w = float(w @ w)
        k_sync = abs(norm_from_K - norm_from_w) / max(abs(norm_from_K), 1e-30)
        if cached and "norm_Lplus_b_sq" in cached:
            norm_Lplus_b_sq = float(cached["norm_Lplus_b_sq"])
        else:
            norm_Lplus_b_sq = norm_from_K
        if cached and "delta_RG" in cached:
            delta_RG = float(cached["delta_RG"])
        else:
            delta_RG = -float(self.n) * norm_Lplus_b_sq / gamma

        r_before = self.multiplicity
        mu2_before = self.mu2
        if z is not None:
            z_before = np.asarray(z, dtype=np.float64).reshape(-1)
        else:
            Ur_before = self._z_accum_U
            z_before = Ur_before[i, :] - Ur_before[j, :]

        # Exact Delta p for every node, from the pre-edge adjacency.
        dp = self._delta_p(w, i, j, Re, gamma)

        self._solve_secular_for_edge(i, j)
        if r_before > 1:
            # lambda_2 cannot move while some eigenvector still satisfies every
            # imposed condition -- verified exact on K_{2,4} (r=3), where all 7
            # candidates move it by <= 8.9e-16. The reward's alpha_1 term is
            # therefore exactly zero here, not merely small.
            self.last_secular_delta = 0.0

        Lplus_new = self.Lplus - np.outer(w, w) / gamma
        K_new = (
            self.K
            - (np.outer(s, w) + np.outer(w, s)) / gamma
            + (norm_Lplus_b_sq / (gamma * gamma)) * np.outer(w, w)
        )

        self.adj[i, j] = 1
        self.adj[j, i] = 1
        self.Lplus = Lplus_new
        self.K = K_new
        self.RG += delta_RG
        self.p = self.p + dp
        self.P_min = float(np.min(self.p))

        if z_before.shape[0] == self._z_accum_r:
            self._z_accum_cols.append(z_before)
        else:
            # A caller-supplied z from a basis this tracker has since re-drawn
            # (its width is the multiplicity that was frozen when the batch was
            # planned, not the current one). It is not a coordinate vector in
            # the live basis, so it cannot join the columns already collected --
            # sigma_r across mixed bases is meaningless, and the length mismatch
            # would surface only later, as a ragged stack. Void the accumulator;
            # Phi stays 0 until a fresh batch refills it.
            self._reset_z_accum()

        self.steps_since_exact += 1
        drift = self._cheap_drift(k_sync)
        self.drift_residual = drift
        need_exact = (
            self.steps_since_exact >= self.exact_reset_every
            or drift > self.drift_threshold
        )
        if need_exact and self._batch_active:
            # Holding the anchor back: eigh would re-draw the degenerate block
            # in an arbitrary basis and invalidate every accumulated z column.
            # The batch is at most r-1 <= q edges long.
            self._deferred_exact = True
            need_exact = False
        if need_exact:
            self._full_recompute()
            did_exact = True
        else:
            self._refine_subspace()
            did_exact = False
            if r_before > 1:
                # Restore the pinned value the power iteration only approximates.
                self.subspace_evals[0] = mu2_before

        r_after = self._detect_multiplicity()

        if not did_exact and self.unresolved_degeneracy and not self._batch_active:
            # Tracked gap is below the power iteration's own resolution, so the
            # r just detected is not trustworthy. Settle it exactly. Skipped
            # mid-batch, where eigh would re-draw the frozen degenerate basis --
            # the deferral path below already covers that case.
            self._full_recompute()
            r_after = self._detect_multiplicity()
            did_exact = True

        if self._batch_active:
            expected = (self._batch_expected_r or r_before) - 1
            self.last_deflation_ok = bool(r_after == expected)
            self._batch_expected_r = r_after
            self._batch_remaining -= 1
            if self._batch_remaining <= 0 or not self.last_deflation_ok:
                self.end_deflation_batch()

        # A drop in r is deflation working as intended: the frozen basis still
        # spans the original eigenspace, so the columns already collected remain
        # valid coordinates in it and the accumulator must survive. Only an
        # exact re-anchor (which re-draws the block) or a genuine growth in r
        # invalidates it.
        if did_exact or r_after > self._z_accum_r:
            self._reset_z_accum()

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
            "degeneracy_probe_rel": self.degeneracy_probe_rel,
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
            "unresolved_degeneracy": self.unresolved_degeneracy,
            "RG": self.RG,
            "p": self.p.copy(),
            "P_min": self.P_min,
            "z_accum_cols": [c.copy() for c in self._z_accum_cols],
            "z_accum_U": self._z_accum_U.copy() if self._z_accum_U is not None else None,
            "z_accum_r": self._z_accum_r,
            "z_accum_epoch": self._z_accum_epoch,
            "batch_active": self._batch_active,
            "batch_remaining": self._batch_remaining,
            "batch_expected_r": self._batch_expected_r,
            "deferred_exact": self._deferred_exact,
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
        obj.degeneracy_probe_rel = float(state.get("degeneracy_probe_rel", 1e-5))
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
        obj.unresolved_degeneracy = bool(state.get("unresolved_degeneracy", False))
        obj.RG = float(state["RG"])
        obj.P_min = float(state["P_min"])
        if state.get("p") is not None:
            obj.p = state["p"].copy()
        else:
            obj.p = obj._compute_p_from_Lplus()
        obj._z_accum_cols = [c.copy() for c in state.get("z_accum_cols", [])]
        obj._z_accum_U = (
            state["z_accum_U"].copy() if state.get("z_accum_U") is not None else None
        )
        obj._z_accum_r = int(state.get("z_accum_r", obj.multiplicity))
        obj._z_accum_epoch = int(state.get("z_accum_epoch", 0))
        obj._batch_active = bool(state.get("batch_active", False))
        obj._batch_remaining = int(state.get("batch_remaining", 0))
        obj._batch_expected_r = state.get("batch_expected_r")
        obj._deferred_exact = bool(state.get("deferred_exact", False))
        obj.last_deflation_ok = True
        obj.last_secular_delta = None
        obj.last_secular_bracket = None
        obj.last_secular_exact = False
        obj.last_beta2 = 0.0
        obj.last_delta_prime = 0.0
        obj.cert_foster = 0.0
        obj.cert_trace = 0.0
        obj.cert_k_sync = 0.0
        return obj
