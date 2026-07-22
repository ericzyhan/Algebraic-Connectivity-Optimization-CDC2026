from __future__ import annotations
from collections import deque
from typing import List, Tuple

import numpy as np

try:
    from scipy.sparse.csgraph import shortest_path as scipy_shortest_path
except Exception:  # pragma: no cover - optional dependency
    scipy_shortest_path = None


def build_path_adjacency(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    return adj


def complete_edge_count(n: int) -> int:
    return (n * (n - 1)) // 2


def normalized_density(n: int, m: int) -> float:
    min_m = n - 1
    max_m = complete_edge_count(n)
    denom = max_m - min_m
    if denom <= 0:
        return 0.0
    return float(m - min_m) / float(denom)


def m_target_from_rho(n: int, rho: float) -> int:
    rho = max(0.0, min(1.0, rho))
    min_m = n - 1
    max_m = complete_edge_count(n)
    m = min_m + rho * (max_m - min_m)
    m_rounded = int(round(m))
    return max(min_m, min(max_m, m_rounded))


def edge_count(adj: np.ndarray) -> int:
    return int(np.sum(adj) // 2)


def laplacian(adj: np.ndarray) -> np.ndarray:
    # Cast to signed/float before subtraction.
    # If we keep uint8, off-diagonal -1 entries underflow to huge positive values.
    adj_f = adj.astype(np.float64, copy=False)
    deg = np.sum(adj_f, axis=1, dtype=np.float64)
    return np.diag(deg) - adj_f


def spectral_features(
    adj: np.ndarray,
    *,
    eps_abs: float = 1e-12,
    eps_rel: float = 1e-10,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, float, int, np.ndarray, np.ndarray]:
    """Compute spectral features, total effective resistance, and λ₂ multiplicity.

    Returns
    -------
    lambda2 : float
    lambda3 : float
    lambda4 : float
    phi2 : ndarray (n,)
    phi3 : ndarray (n,)
    phi4 : ndarray (n,)
    rg : float
        Total effective (graph) resistance R_G = n * Σ 1/λ_i for i=2..n.
    multiplicity : int
        Number of eigenvalues clustered around λ₂ (within tolerance), excluding λ₁=0.
    evecs : ndarray (n, n)
        Full eigenvector matrix from np.linalg.eigh (columns are eigenvectors).
    evals : ndarray (n,)
        Eigenvalues of the Laplacian in ascending order.
    """
    L = laplacian(adj).astype(np.float64)
    evals, evecs = np.linalg.eigh(L)

    n = adj.shape[0]
    if n >= 3:
        lambda2 = float(evals[1])
        lambda3 = float(evals[2])
        lambda4 = float(evals[3]) if n >= 4 else 0.0
        phi2 = evecs[:, 1]
        phi3 = evecs[:, 2]
        phi4 = evecs[:, 3] if n >= 4 else evecs[:, 2]
    else:
        lambda2 = 0.0
        lambda3 = 0.0
        lambda4 = 0.0
        phi2 = np.zeros((n,), dtype=np.float64)
        phi3 = np.zeros((n,), dtype=np.float64)
        phi4 = np.zeros((n,), dtype=np.float64)

    # Total effective resistance: R_G = n * Σ_{i=2}^n 1/λ_i
    safe_evals = np.maximum(evals[1:], 1e-10)
    rg = float(n) * float(np.sum(1.0 / safe_evals))

    # Multiplicity of λ₂: count eigenvalues within tolerance of λ₂ (exclude λ₁=0)
    tol = max(eps_abs, eps_rel * max(1.0, abs(lambda2)))
    mult_mask = (np.abs(evals - lambda2) <= tol)
    mult_mask[0] = False
    multiplicity = int(np.sum(mult_mask))

    return lambda2, lambda3, lambda4, phi2.astype(np.float64), phi3.astype(np.float64), phi4.astype(np.float64), rg, multiplicity, evecs.astype(np.float64), evals.astype(np.float64)


def node_degrees(adj: np.ndarray) -> np.ndarray:
    return np.sum(adj, axis=1).astype(np.float64)


def two_hop_neighbor_counts(adj: np.ndarray) -> np.ndarray:
    # Avoid uint8 overflow in adjacency multiplication.
    adj_bool = adj > 0
    adj_i = adj_bool.astype(np.int64, copy=False)
    a2 = (adj_i @ adj_i) > 0
    np.fill_diagonal(a2, False)
    two_hop_only = np.logical_and(a2, np.logical_not(adj_bool))
    return np.sum(two_hop_only, axis=1, dtype=np.float64)


def clustering_coefficients(adj: np.ndarray) -> np.ndarray:
    adj_f = adj.astype(np.float64, copy=False)
    deg = np.sum(adj_f, axis=1, dtype=np.float64)
    # For simple undirected graphs: (A^3)_{ii} = 2 * (# triangles touching node i).
    a3_diag = np.diag(adj_f @ adj_f @ adj_f)
    denom = deg * np.maximum(1.0, deg - 1.0)
    coeffs = np.divide(
        a3_diag,
        denom,
        out=np.zeros_like(a3_diag, dtype=np.float64),
        where=deg >= 2.0,
    )
    return coeffs


def non_edges(adj: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.triu(adj == 0, k=1)
    rows, cols = np.where(mask)
    return list(zip(rows.tolist(), cols.tolist()))


def edge_index_from_adj(adj: np.ndarray) -> np.ndarray:
    rows, cols = np.where(adj > 0)
    return np.vstack([rows, cols]).astype(np.int64)


def truncated_all_pairs_shortest_path(adj: np.ndarray, cap: int) -> np.ndarray:
    if scipy_shortest_path is not None:
        dist = scipy_shortest_path(
            adj.astype(np.float64, copy=False),
            directed=False,
            unweighted=True,
        )
        dist = np.minimum(dist, float(cap))
        return dist.astype(np.int64, copy=False)

    n = adj.shape[0]
    dist = np.full((n, n), fill_value=cap + 1, dtype=np.int64)
    for src in range(n):
        dist[src, src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            if dist[src, u] >= cap:
                continue
            neighbors = np.where(adj[u] > 0)[0]
            for v in neighbors:
                if dist[src, v] > dist[src, u] + 1:
                    dist[src, v] = dist[src, u] + 1
                    q.append(v)
    dist = np.minimum(dist, cap)
    return dist


def jaccard_overlap(adj: np.ndarray, i: int, j: int) -> float:
    ni = set(np.where(adj[i] > 0)[0].tolist())
    nj = set(np.where(adj[j] > 0)[0].tolist())
    union = ni.union(nj)
    if not union:
        return 0.0
    inter = ni.intersection(nj)
    return float(len(inter)) / float(len(union))


def common_neighbors_norm(adj: np.ndarray, i: int, j: int) -> float:
    n = adj.shape[0]
    denom = max(1, n - 2)
    common = np.sum((adj[i] > 0) & (adj[j] > 0))
    return float(common) / float(denom)


def fiedler_scores_for_pairs(phi2: np.ndarray, pairs: List[Tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.zeros((0,), dtype=np.float64)
    arr = np.asarray(pairs, dtype=np.int64)
    diff = phi2[arr[:, 0]] - phi2[arr[:, 1]]
    return diff * diff

def multispectral_scores_for_all_pairs(phi2: np.ndarray, phi3: np.ndarray, phi4: np.ndarray, pairs: List[Tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.zeros((0,), dtype=np.float64)
    arr = np.asarray(pairs, dtype=np.int64)
    diff2 = phi2[arr[:, 0]] - phi2[arr[:, 1]]
    diff3 = phi3[arr[:, 0]] - phi3[arr[:, 1]]
    diff4 = phi4[arr[:, 0]] - phi4[arr[:, 1]]
    return (diff2 * diff2) + (diff3 * diff3) + (diff4 * diff4)


def total_effective_resistance(evals: np.ndarray, n: int, eps: float = 1e-10) -> float:
    """Compute R_G = n * Σ_{i=2}^n 1/λ_i from the full eigenvalue array."""
    if n < 2:
        return 0.0
    safe = np.maximum(evals[1:], eps)
    return float(n) * float(np.sum(1.0 / safe))


def lambda2_eigenspace_scores(
    evals: np.ndarray,
    evecs: np.ndarray,
    pairs: np.ndarray,
    *,
    eps_abs: float = 1e-12,
    eps_rel: float = 1e-10,
) -> np.ndarray:
    """Score each candidate pair by effective-resistance-weighted λ₂ eigenspace gap.

    Detects the multiplicity d of λ₂, extracts the d eigenvectors spanning the
    λ₂ eigenspace, and computes:

        score(u,v) = (1/λ₂) * Σ_{k=1}^{d} (φ_{k+1}[u] - φ_{k+1}[v])²

    where φ₂,...,φ_{d+1} are the eigenvectors in the λ₂ eigenspace.
    Dividing by λ₂ gives the ER-weighted spectral gap.
    """
    if pairs.size == 0:
        return np.zeros((0,), dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)
    n = int(evals.shape[0])
    if n < 2:
        return np.zeros((pairs.shape[0],), dtype=np.float64)

    lambda2 = float(evals[1])
    tol = max(eps_abs, eps_rel * max(1.0, abs(lambda2)))
    mult_mask = (np.abs(evals - lambda2) <= tol)
    mult_mask[0] = False
    d = int(np.sum(mult_mask))

    if d <= 0:
        # Fallback: just use φ₂ with zero weight
        diff = evecs[pairs[:, 0], 1] - evecs[pairs[:, 1], 1]
        return diff * diff

    # Extract d eigenvectors in the λ₂ eigenspace
    idx = np.flatnonzero(mult_mask)  # shape (d,), e.g. [1] or [1,2] or [1,2,3]
    phi_stack = evecs[:, idx].astype(np.float64)  # (n, d)

    # Compute sum of squared differences across the d eigenvectors
    diff = phi_stack[pairs[:, 0], :] - phi_stack[pairs[:, 1], :]  # (p, d)
    scores = np.sum(diff * diff, axis=1)  # (p,)

    # Normalize by λ₂ (all d eigenvalues ≈ λ₂ within tolerance)
    safe_l2 = max(float(abs(lambda2)), 1e-10)
    return scores / safe_l2


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if len(scores) <= k:
        return np.arange(len(scores), dtype=np.int64)
    order = np.argsort(-scores, kind="mergesort")
    return order[:k].astype(np.int64)


def pairwise_effective_resistance(
    evals: np.ndarray,
    evecs: np.ndarray,
    pairs: np.ndarray,
    *,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute effective resistance ω_ij for each pair (i,j).

    ω_ij = Σ_{k=2}^{n} (φ_k(i) − φ_k(j))² / λ_k

    Uses the full eigendecomposition (already available from spectral_features).
    """
    if pairs.size == 0:
        return np.zeros((0,), dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)
    n = int(evals.shape[0])
    if n < 2:
        return np.zeros((pairs.shape[0],), dtype=np.float64)
    safe_evals = np.maximum(evals[1:], eps)
    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]
    phi = evecs[:, 1:].astype(np.float64)  # (n, n-1), skip λ₁=0 eigenvector
    diff = phi[i_idx, :] - phi[j_idx, :]  # (p, n-1)
    weighted = diff * diff / safe_evals[np.newaxis, :]  # (p, n-1)
    return np.sum(weighted, axis=1)  # (p,)


def resistance_curvature(
    adj: np.ndarray,
    evals: np.ndarray,
    evecs: np.ndarray,
    *,
    eps: float = 1e-10,
) -> tuple[np.ndarray, float, float, float]:
    """Compute per-vertex resistance curvature p_i and aggregates.

    p_i = 1 − ½ · Σ_{j~i} ω_ij

    Returns
    -------
    p : ndarray (n,) — per-vertex curvature
    P_min : float — minimum curvature
    P_mean : float — mean curvature
    P_var : float — variance of curvature
    """
    n = adj.shape[0]
    if n < 2:
        p = np.zeros((n,), dtype=np.float64)
        return p, 0.0, 0.0, 0.0

    rows, cols = np.where(np.triu(adj > 0, k=1))
    edge_pairs = np.stack([rows, cols], axis=1).astype(np.int64)
    if edge_pairs.size == 0:
        p = np.ones((n,), dtype=np.float64)
        return p, 0.0 if n == 0 else 1.0, 0.0 if n == 0 else 1.0, 0.0

    omega_edges = pairwise_effective_resistance(evals, evecs, edge_pairs, eps=eps)

    # Accumulate Σ_{j~i} ω_ij for each vertex
    neighbor_sum = np.zeros((n,), dtype=np.float64)
    for idx in range(edge_pairs.shape[0]):
        i = int(edge_pairs[idx, 0])
        j = int(edge_pairs[idx, 1])
        w = float(omega_edges[idx])
        neighbor_sum[i] += w
        neighbor_sum[j] += w

    p = 1.0 - 0.5 * neighbor_sum
    P_mean = float(np.mean(p))
    P_min = float(np.min(p))
    P_var = float(np.var(p))
    return p.astype(np.float64), P_min, P_mean, P_var


class IncrementalSpectralState:
    """Maintains spectral quantities via rank-1 Sherman-Morrison updates.

    Instead of recomputing the full eigendecomposition O(n³) on every edge
    addition, this class:

    1.  Computes the grounded Laplacian inverse L_g^{-1} once at init (O(n³)).
    2.  Updates L_g^{-1} via Sherman-Morrison on each edge addition (O(n²)).
    3.  Periodically recomputes L_g^{-1} exactly to prevent numerical drift.
    4.  Lazy-refreshes λ₂, φ₂, φ₃ via eigsh with warm-start.
    5.  Provides O(1) effective resistance ω_ij for any pair via L_g^{-1}.
    6.  Caches full eigenvalues/eigenvectors from the most recent refresh.

    For training graphs with n ≤ 128 the O(n²) per-step cost is negligible
    compared to O(n³) per-step full eigh.
    """

    def __init__(
        self,
        adj: np.ndarray,
        *,
        ground: int = 0,
        exact_reset_every: int = 32,
        spectral_refresh_every: int = 1,
        eigsh_cutoff_n: int = 96,
        eps: float = 1e-10,
    ) -> None:
        self.n = adj.shape[0]
        self.ground = int(ground)
        self.exact_reset_every = int(exact_reset_every)
        self.spectral_refresh_every = int(spectral_refresh_every)
        self.eigsh_cutoff_n = int(eigsh_cutoff_n)
        self.eps = float(eps)

        self._steps_since_exact = 0
        self._steps_since_spectral = 10**9

        # Grounded Laplacian inverse (n-1 × n-1)
        self._inv_lg: np.ndarray = self._compute_grounded_inverse(adj)
        # Cache the adjacency (bool) for edge lookups
        self._adj_bool: np.ndarray = (adj > 0)

        # Spectral cache
        self._lam2: float = 0.0
        self._lam3: float = 0.0
        self._lam4: float = 0.0
        self._phi2: np.ndarray = np.zeros((self.n,), dtype=np.float64)
        self._phi3: np.ndarray = np.zeros((self.n,), dtype=np.float64)
        self._phi4: np.ndarray = np.zeros((self.n,), dtype=np.float64)
        self._rg: float = 0.0
        self._mult: int = 1
        self._evals: np.ndarray = np.zeros((self.n,), dtype=np.float64)
        self._evecs: np.ndarray = np.zeros((self.n, self.n), dtype=np.float64)
        self._spectral_valid: bool = False

        # Curvature cache
        self._p: np.ndarray = np.zeros((self.n,), dtype=np.float64)
        self._P_min: float = 0.0
        self._P_mean: float = 0.0
        self._P_var: float = 0.0
        self._curvature_valid: bool = False

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def lam2(self) -> float:
        return self._lam2

    @property
    def lam3(self) -> float:
        return self._lam3

    @property
    def lam4(self) -> float:
        return self._lam4

    @property
    def phi2(self) -> np.ndarray:
        return self._phi2

    @property
    def phi3(self) -> np.ndarray:
        return self._phi3

    @property
    def phi4(self) -> np.ndarray:
        return self._phi4

    @property
    def rg(self) -> float:
        return self._rg

    @property
    def multiplicity(self) -> int:
        return self._mult

    @property
    def evals(self) -> np.ndarray:
        return self._evals

    @property
    def evecs(self) -> np.ndarray:
        return self._evecs

    @property
    def curvature(self) -> np.ndarray:
        if not self._curvature_valid:
            self._recompute_curvature()
        return self._p

    @property
    def P_min(self) -> float:
        if not self._curvature_valid:
            self._recompute_curvature()
        return self._P_min

    @property
    def P_mean(self) -> float:
        if not self._curvature_valid:
            self._recompute_curvature()
        return self._P_mean

    @property
    def P_var(self) -> float:
        if not self._curvature_valid:
            self._recompute_curvature()
        return self._P_var

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _compute_grounded_inverse(self, adj: np.ndarray) -> np.ndarray:
        """Compute (n-1)×(n-1) grounded Laplacian inverse."""
        adj_f = adj.astype(np.float64, copy=False)
        deg = np.sum(adj_f, axis=1, dtype=np.float64)
        L = np.diag(deg) - adj_f
        keep = [i for i in range(self.n) if i != self.ground]
        Lg = L[np.ix_(keep, keep)]
        ridge = 1e-10 * np.eye(Lg.shape[0], dtype=np.float64)
        return np.linalg.inv(Lg + ridge)

    def _grounded_incidence(self, u: int, v: int) -> np.ndarray:
        """Incidence vector e_u - e_v in the grounded (n-1)-dim space."""
        g = np.zeros(self.n - 1, dtype=np.float64)
        for node, sign in ((u, 1.0), (v, -1.0)):
            if node == self.ground:
                continue
            idx = node if node < self.ground else node - 1
            g[idx] += sign
        return g

    # ------------------------------------------------------------------
    # Edge addition
    # ------------------------------------------------------------------
    def add_edge(self, u: int, v: int) -> None:
        """Add edge (u,v) and update all incremental state."""
        if self._adj_bool[u, v]:
            return  # already present
        self._adj_bool[u, v] = True
        self._adj_bool[v, u] = True

        # Sherman-Morrison rank-1 update to grounded inverse
        self._sherman_morrison_update(u, v)

        self._steps_since_exact += 1
        self._steps_since_spectral += 1
        self._curvature_valid = False
        # NOTE: _spectral_valid is NOT set to False here — refresh_spectral()
        # uses _steps_since_spectral vs spectral_refresh_every to decide when
        # to re-run the full eigendecomposition. This allows lazy refresh
        # across multiple edge additions (e.g. only recompute every 4 steps).

        # Periodic exact recompute to prevent drift
        if self._steps_since_exact >= self.exact_reset_every:
            self._refresh_exact_inverse()
            self._steps_since_exact = 0

    def _sherman_morrison_update(self, u: int, v: int) -> None:
        g = self._grounded_incidence(u, v)
        x = self._inv_lg @ g
        denom = 1.0 + float(g.T @ x)
        if denom <= 1e-12 or not np.isfinite(denom):
            self._refresh_exact_inverse()
            return
        self._inv_lg -= np.outer(x, x) / denom

    def _refresh_exact_inverse(self) -> None:
        adj = self._adj_bool.astype(np.uint8)
        self._inv_lg = self._compute_grounded_inverse(adj)

    # ------------------------------------------------------------------
    # Effective resistance (O(1))
    # ------------------------------------------------------------------
    def effective_resistance(self, u: int, v: int) -> float:
        """Effective resistance ω_uv computed from grounded inverse."""
        if u == v:
            return 0.0
        if u == self.ground:
            vi = v if v < self.ground else v - 1
            return float(max(self._inv_lg[vi, vi], 0.0))
        if v == self.ground:
            ui = u if u < self.ground else u - 1
            return float(max(self._inv_lg[ui, ui], 0.0))
        ui = u if u < self.ground else u - 1
        vi = v if v < self.ground else v - 1
        val = self._inv_lg[ui, ui] + self._inv_lg[vi, vi] - 2.0 * self._inv_lg[ui, vi]
        return float(max(val, 0.0))

    def effective_resistance_batch(self, pairs: np.ndarray) -> np.ndarray:
        """Compute ω_uv for a batch of pairs.  pairs shape: (p, 2)."""
        if pairs.size == 0:
            return np.zeros((0,), dtype=np.float64)
        pairs = np.asarray(pairs, dtype=np.int64)
        out = np.empty((pairs.shape[0],), dtype=np.float64)
        for idx in range(pairs.shape[0]):
            out[idx] = self.effective_resistance(
                int(pairs[idx, 0]), int(pairs[idx, 1])
            )
        return out

    # ------------------------------------------------------------------
    # Curvature (recomputed from cached ω_ij)
    # ------------------------------------------------------------------
    def _recompute_curvature(self) -> None:
        n = self.n
        neighbor_sum = np.zeros((n,), dtype=np.float64)
        rows, cols = np.where(np.triu(self._adj_bool, k=1))
        for r, c in zip(rows.tolist(), cols.tolist()):
            w = self.effective_resistance(r, c)
            neighbor_sum[r] += w
            neighbor_sum[c] += w
        self._p = 1.0 - 0.5 * neighbor_sum
        self._P_min = float(np.min(self._p))
        self._P_mean = float(np.mean(self._p))
        self._P_var = float(np.var(self._p))
        self._curvature_valid = True

    # ------------------------------------------------------------------
    # Spectral refresh (lazy, with warm-start)
    # ------------------------------------------------------------------
    def refresh_spectral(self, force: bool = False) -> None:
        """Recompute λ₂,λ₃,λ₄, φ₂,φ₃,φ₄, R_G, multiplicity, evals, evecs."""
        stale = self._steps_since_spectral >= self.spectral_refresh_every
        if not force and self._spectral_valid and not stale:
            return

        n = self.n
        adj_u8 = self._adj_bool.astype(np.uint8)
        if n <= self.eigsh_cutoff_n:
            # Dense path: full eigh
            adj_f = adj_u8.astype(np.float64, copy=False)
            deg = np.sum(adj_f, axis=1)
            L = np.diag(deg) - adj_f
            evals, evecs = np.linalg.eigh(L)
            self._evals = evals.astype(np.float64)
            self._evecs = evecs.astype(np.float64)
        else:
            # Sparse path: eigsh with warm-start
            try:
                from scipy.sparse import csr_matrix
                from scipy.sparse.linalg import eigsh
                adj_f = adj_u8.astype(np.float64, copy=False)
                deg = np.sum(adj_f, axis=1)
                L = np.diag(deg) - adj_f
                Ls = csr_matrix(L)
                v0 = self._phi2 if self._spectral_valid else None
                # Request enough eigenvalues to capture λ₂ cluster
                k_req = min(n - 1, max(4, self._mult + 4))
                vals, vecs = eigsh(Ls, k=k_req, which="SM", v0=v0, tol=1e-5, maxiter=3000)
                order = np.argsort(vals)
                self._evals = vals[order].astype(np.float64)
                self._evecs = vecs[:, order].astype(np.float64)
            except Exception:
                # Fallback to dense
                adj_f = adj_u8.astype(np.float64, copy=False)
                deg = np.sum(adj_f, axis=1)
                L = np.diag(deg) - adj_f
                evals, evecs = np.linalg.eigh(L)
                self._evals = evals.astype(np.float64)
                self._evecs = evecs.astype(np.float64)

        evals = self._evals
        evecs = self._evecs

        if n >= 3:
            self._lam2 = float(evals[1])
            self._lam3 = float(evals[2])
            self._lam4 = float(evals[3]) if n >= 4 else 0.0
            self._phi2 = evecs[:, 1].astype(np.float64).copy()
            self._phi3 = evecs[:, 2].astype(np.float64).copy()
            self._phi4 = evecs[:, 3].astype(np.float64).copy() if n >= 4 else evecs[:, 2].astype(np.float64).copy()
        else:
            self._lam2 = 0.0
            self._lam3 = 0.0
            self._lam4 = 0.0
            self._phi2 = np.zeros((n,), dtype=np.float64)
            self._phi3 = np.zeros((n,), dtype=np.float64)
            self._phi4 = np.zeros((n,), dtype=np.float64)

        # R_G = n * Σ_{k=2}^n 1/λ_k
        safe = np.maximum(evals[1:], 1e-10)
        self._rg = float(n) * float(np.sum(1.0 / safe))

        # Multiplicity
        tol = max(1e-12, 1e-10 * max(1.0, abs(self._lam2)))
        mult_mask = np.abs(evals - self._lam2) <= tol
        mult_mask[0] = False
        self._mult = int(np.sum(mult_mask))

        self._spectral_valid = True
        self._steps_since_spectral = 0

    # ------------------------------------------------------------------
    # Convenience: return tuple matching spectral_features() signature
    # ------------------------------------------------------------------
    def spectral_tuple(self):
        """Return (lam2, lam3, lam4, phi2, phi3, phi4, rg, mult, evecs, evals)."""
        self.refresh_spectral()
        return (
            self._lam2, self._lam3, self._lam4,
            self._phi2.copy(), self._phi3.copy(), self._phi4.copy(),
            self._rg, self._mult,
            self._evecs.copy(), self._evals.copy(),
        )
