from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import sparse

from .candidates import CandidateGenerator
from .graph_core import (
    complement_graph,
    Edge,
    edge_count,
    is_connected,
    max_edges,
    non_edges,
    path_graph,
    star_graph,
)
from .incremental_linear_algebra import IncrementalState, IncrementalStateConfig
from .spectral import (
    algebraic_connectivity,
    lambda2_subspace_edge_scores,
    laplacian,
    two_smallest_nontrivial,
)


def _bfs_distances(adj: np.ndarray, src: int) -> np.ndarray:
    n = int(adj.shape[0])
    dist = -np.ones(n, dtype=np.int64)
    q: deque[int] = deque()
    dist[int(src)] = 0
    q.append(int(src))
    while q:
        u = int(q.popleft())
        du = int(dist[u])
        nbrs = np.flatnonzero(adj[u])
        for v in nbrs.tolist():
            v = int(v)
            if dist[v] >= 0:
                continue
            dist[v] = du + 1
            q.append(v)
    return dist


def _choose_with_ec(
    candidates: np.ndarray,
    ec: np.ndarray | None,
    *,
    random_tie: bool,
    rng: np.random.Generator | None,
) -> int:
    cand = np.asarray(candidates, dtype=np.int64).reshape(-1)
    if cand.size == 0:
        raise RuntimeError("empty candidate set")
    if ec is None:
        best = cand
    else:
        vals = np.asarray(ec[cand], dtype=np.float64)
        mn = float(vals.min())
        best = cand[np.flatnonzero(vals <= (mn + 1e-12))]
    if best.size == 1:
        return int(best[0])
    if bool(random_tie) and rng is not None:
        return int(best[int(rng.integers(0, int(best.size)))])
    return int(best.min())


def mdmd_complete(
    adj: np.ndarray,
    m_target: int,
    *,
    use_ec: bool = True,
    random_tie: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """
    Iterative MDMD edge-addition heuristic (paper-inspired single-edge MACP repeated).

    At each step:
    1) choose a min-degree vertex i0 (with EC tie-break if use_ec=True),
    2) BFS from i0, collect farthest vertices,
    3) choose j0 among farthest (with EC tie-break if use_ec=True),
    4) add edge (i0, j0).
    """
    work = adj.copy()
    n = int(work.shape[0])
    m_hi = int(max_edges(n))
    target = int(min(int(m_target), m_hi))
    cur_edges = int(edge_count(work))
    if cur_edges >= target:
        return work

    rng = np.random.default_rng(int(seed)) if bool(random_tie) else None
    while cur_edges < target:
        deg = work.sum(axis=1).astype(np.int64)
        ec: np.ndarray | None = None
        if bool(use_ec):
            # Extendibility centrality: sum of neighbor degrees.
            ec = (work.astype(np.int64) @ deg).astype(np.int64)

        mdeg = int(deg.min())
        M = np.flatnonzero(deg == mdeg).astype(np.int64)
        i0 = _choose_with_ec(M, ec, random_tie=bool(random_tie), rng=rng)

        dist = _bfs_distances(work, i0)
        non_i0 = np.flatnonzero(~work[i0]).astype(np.int64)
        non_i0 = non_i0[non_i0 != int(i0)]
        if non_i0.size == 0:
            break

        d_non = dist[non_i0]
        d_max = int(d_non.max())
        far_non = non_i0[np.flatnonzero(d_non == d_max)].astype(np.int64)
        if far_non.size == 0:
            break
        j0 = _choose_with_ec(far_non, ec, random_tie=bool(random_tie), rng=rng)

        if bool(work[i0, j0]):
            # Defensive fallback: this should not happen because j0 is picked
            # from non-neighbors of i0. If it does, choose any remaining non-edge.
            iu, iv = np.where(np.triu(~work, k=1))
            if int(iu.size) == 0:
                break
            if bool(random_tie) and rng is not None:
                k = int(rng.integers(0, int(iu.size)))
            else:
                k = 0
            i0, j0 = int(iu[k]), int(iv[k])
        work[i0, j0] = True
        work[j0, i0] = True
        cur_edges += 1
    return work


def _add_edge_if_absent(adj: np.ndarray, u: int, v: int) -> None:
    if u == v:
        return
    if adj[u, v]:
        return
    adj[u, v] = True
    adj[v, u] = True


def min_laplacian_energy_graph(n: int, m: int) -> np.ndarray:
    """
    Construct a connected minimal-Laplacian-energy graph (Algorithm 1 in
    "Fast Consensus Topology Design via Minimizing Laplacian Energy", CDC 2024).

    The construction is defined for connected regimes m >= n-1.
    """
    n = int(n)
    m = int(m)
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if m < (n - 1):
        raise ValueError(f"Algorithm 1 assumes connected regime m>=n-1, got n={n}, m={m}")
    if m > max_edges(n):
        raise ValueError(f"m must be <= n(n-1)/2, got n={n}, m={m}")

    adj = np.zeros((n, n), dtype=bool)
    two_m = int(2 * m)
    k = int(two_m // n)
    l = int(two_m - n * k)  # 0 <= l < n

    # Case 1: k is even.
    if (k % 2) == 0:
        half = int(k // 2)
        for i in range(n):
            for s in range(1, half + 1):
                _add_edge_if_absent(adj, i, (i + s) % n)
        # If 2m = nk + l, l is even here.
        if l > 0:
            for i in range(int(l // 2)):
                _add_edge_if_absent(adj, i, i + (n // 2))
    else:
        # Case 2: k is odd.
        if (n % 2) == 0:
            # Case 2(1): n even, 2m = nk.
            half = int((k - 1) // 2)
            for i in range(n):
                for s in range(1, half + 1):
                    _add_edge_if_absent(adj, i, (i + s) % n)
                _add_edge_if_absent(adj, i, (i + (n // 2)) % n)

            # Case 2(2): n even, 2m = nk + l, l even.
            if l > 0:
                shift = int((n - 2) // 2)
                for i in range(int(l // 2)):
                    _add_edge_if_absent(adj, i, i + shift)
        else:
            # Case 2(3): n odd, 2m = nk + 1.
            half = int((k - 1) // 2)
            for i in range(n):
                for s in range(1, half + 1):
                    _add_edge_if_absent(adj, i, (i + s) % n)

            shift = int((n - 1) // 2)
            for i in range(int((n + 1) // 2)):
                _add_edge_if_absent(adj, i, i + shift)

            # Case 2(4): n odd, 2m = nk + 1 + l_even, l_even even.
            l_even = int(l - 1)
            if l_even > 0:
                # 1-based i in {(n+3)/2, ..., (n+1+l_even)/2}.
                i1_start = int((n + 3) // 2)
                i1_end = int((n + 1 + l_even) // 2)
                for i1 in range(i1_start, i1_end + 1):
                    u = int(i1 - 1)  # convert to 0-based index
                    v = int((u + shift) % n)
                    _add_edge_if_absent(adj, u, v)

    # Defensive checks: Algorithm-1 should be exact on edge count and connected.
    m_now = edge_count(adj)
    if m_now != m:
        raise RuntimeError(
            f"Algorithm-1 construction mismatch: expected m={m}, built m={m_now} "
            f"(n={n}, k={k}, l={l})"
        )
    if not bool(is_connected(adj)):
        raise RuntimeError(f"Algorithm-1 construction is disconnected for n={n}, m={m}")
    return adj


def min_laplacian_energy_complete(adj: np.ndarray, m_target: int) -> np.ndarray:
    """
    Completion wrapper used by density-sweep pipeline.
    This method is direct-by-(n,m) and does not depend on the input init graph.
    """
    n = int(adj.shape[0])
    return min_laplacian_energy_graph(n=n, m=int(m_target))


def _union_complete_components_graph(
    n: int,
    component_sizes: Sequence[int],
) -> np.ndarray:
    sizes = [int(s) for s in component_sizes if int(s) > 0]
    used = int(sum(sizes))
    if used > int(n):
        raise ValueError(f"component sizes exceed n: used={used}, n={n}, sizes={sizes}")
    # Remaining vertices are isolated K1 components.
    if used < int(n):
        sizes = sizes + [1] * (int(n) - used)

    adj = np.zeros((int(n), int(n)), dtype=bool)
    start = 0
    for s in sizes:
        if s >= 2:
            blk = np.ones((s, s), dtype=bool)
            np.fill_diagonal(blk, False)
            adj[start : start + s, start : start + s] = blk
        start += s
    return adj


def _lgm_paper_component_sizes(
    n: int,
    m_desired: int,
) -> Tuple[List[int], int]:
    """
    Implement Algorithm 1 from:
      "Algebraic Connectivity: Local and Global Maximizer Graphs" (2023)
    for constructing a union-of-complete-components graph.

    Returns:
      (component_sizes, m_actual), with m_actual <= m_desired.
    """
    n_i_rem = int(n)
    m_i_rem = int(m_desired)
    q = 2
    sizes: List[int] = []

    def c2(x: int) -> int:
        return int(x * (x - 1) // 2)

    while (m_i_rem > 0) and (n_i_rem > 0) and (q <= n_i_rem):
        l_max = int(n_i_rem // q)
        if m_i_rem > c2(n_i_rem):
            # "m_rem too high for any complete graph" branch from Algorithm 1.
            q_use = int(n_i_rem)
            sizes.append(q_use)
            n_i_rem -= q_use
            m_i_rem -= c2(q_use)
            q = 2
            continue

        cap = int(l_max * c2(q))
        if m_i_rem <= cap:
            e_q = int(c2(q))
            l = int(min(l_max, (m_i_rem // e_q) if e_q > 0 else 0))

            # Corner case in the paper pseudocode: if no q-size component fits,
            # reduce q by one and retry once.
            if l <= 0:
                if q > 2:
                    q -= 1
                    l_max = int(n_i_rem // q)
                    e_q = int(c2(q))
                    l = int(min(l_max, (m_i_rem // e_q) if e_q > 0 else 0))
                if l <= 0:
                    q += 1
                    continue

            sizes.extend([int(q)] * int(l))
            n_i_rem -= int(q * l)
            m_i_rem -= int(l * e_q)
            q = 2
        else:
            q += 1

    m_actual = int(m_desired - m_i_rem)
    return sizes, m_actual


def _fill_edges_balancing_degree(
    adj: np.ndarray,
    m_target: int,
) -> np.ndarray:
    """
    Add remaining edges while preferring low-degree endpoints.
    Used only when the paper's exact union-of-cliques construction
    under-fills the desired budget.
    """
    work = adj.copy()
    target = int(m_target)
    while edge_count(work) < target:
        cand = non_edges(work)
        if not cand:
            break
        uv = np.asarray(cand, dtype=np.int64)
        deg = work.sum(axis=1).astype(np.int64)
        u = uv[:, 0]
        v = uv[:, 1]
        s1 = np.maximum(deg[u], deg[v])
        best = np.flatnonzero(s1 == int(s1.min()))
        if int(best.size) > 1:
            s2 = deg[u[best]] + deg[v[best]]
            best = best[np.flatnonzero(s2 == int(s2.min()))]
        idx = int(best[0])
        a = int(uv[idx, 0])
        b = int(uv[idx, 1])
        work[a, b] = True
        work[b, a] = True
    return work


def local_global_maximizer_complete(
    adj: np.ndarray,
    m_target: int,
    *,
    exact_budget: bool = True,
) -> np.ndarray:
    """
    Baseline from "Algebraic Connectivity: Local and Global Maximizer Graphs".

    Construct complement graph via Algorithm 1 (union of complete components
    minimizing Laplacian largest eigenvalue in their local/global sense), then
    return its complement as the ACM candidate graph.

    Notes:
    - The paper algorithm may return m_actual < m_desired for some (n,m).
    - If exact_budget=True, we fill the remaining complement-edge budget using
      a low-degree balancing rule to keep final edge count exact.
    """
    n = int(adj.shape[0])
    m = int(m_target)
    m_comp_target = int(max_edges(n) - m)
    if m_comp_target < 0:
        raise ValueError(f"Invalid target: n={n}, m_target={m_target}")

    comp_sizes, m_comp_actual = _lgm_paper_component_sizes(n=n, m_desired=m_comp_target)
    comp_adj = _union_complete_components_graph(n=n, component_sizes=comp_sizes)
    if edge_count(comp_adj) != int(m_comp_actual):
        raise RuntimeError(
            f"LGM construction mismatch: expected m_actual={m_comp_actual}, "
            f"built={edge_count(comp_adj)}"
        )

    if bool(exact_budget) and int(m_comp_actual) < int(m_comp_target):
        comp_adj = _fill_edges_balancing_degree(comp_adj, int(m_comp_target))

    out = complement_graph(comp_adj)
    return out


def _top_k_binary_vector(scores: np.ndarray, k: int) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    p = int(s.size)
    out = np.zeros(p, dtype=np.float64)
    kk = int(k)
    if kk <= 0 or p == 0:
        return out
    if kk >= p:
        out.fill(1.0)
        return out
    # Deterministic top-k under exact ties:
    # - all scores strictly above cutoff are always selected
    # - exact-cutoff ties are selected by increasing index
    part = np.argpartition(-s, kk - 1)[:kk]
    cutoff = float(s[part].min())
    strict_idx = np.flatnonzero(s > cutoff)
    need = int(kk - int(strict_idx.size))
    if need <= 0:
        idx = strict_idx[:kk]
    else:
        tie_idx = np.flatnonzero(s == cutoff)
        idx = np.concatenate([strict_idx, tie_idx[:need]])
    out[idx] = 1.0
    return out


def _fiedler_from_laplacian(L: np.ndarray) -> Tuple[float, np.ndarray]:
    vals, vecs = np.linalg.eigh(np.asarray(L, dtype=np.float64))
    n = int(L.shape[0])
    if vals.size < 2:
        return 0.0, np.zeros(n, dtype=np.float64)
    lam2 = float(max(vals[1], 0.0))
    y = np.asarray(vecs[:, 1], dtype=np.float64)
    return lam2, y


def _laplacian_with_fractional_candidates(
    base_lap: np.ndarray,
    candidate_uv: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    L = np.asarray(base_lap, dtype=np.float64).copy()
    if candidate_uv.size == 0:
        return L
    uv = np.asarray(candidate_uv, dtype=np.int64)
    w = np.asarray(x, dtype=np.float64).reshape(-1)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"candidate_uv must have shape (p,2), got {uv.shape}")
    if int(w.size) != int(uv.shape[0]):
        raise ValueError(f"x length mismatch: len(x)={w.size}, p={uv.shape[0]}")
    u = uv[:, 0]
    v = uv[:, 1]
    np.add.at(L, (u, u), w)
    np.add.at(L, (v, v), w)
    np.add.at(L, (u, v), -w)
    np.add.at(L, (v, u), -w)
    return L


def mac_complete(
    adj: np.ndarray,
    m_target: int,
    *,
    fw_iters: int = 20,
    duality_gap_tol: float = 1e-6,
    init_mode: str = "fiedler_topk",
    seed: int = 0,
) -> np.ndarray:
    """
    MAC (Maximizing Algebraic Connectivity) via Boolean relaxation + rounding.

    Paper-inspired implementation (Doherty et al., 2022):
    - Candidate set: all non-edges of the fixed base graph.
    - Solve relaxed selection x in [0,1]^p, 1^T x = K with Frank-Wolfe.
    - Round by selecting top-K entries of x.
    """
    base = adj.copy()
    q = int(m_target - edge_count(base))
    if q <= 0:
        return base

    cand = non_edges(base)
    p = int(len(cand))
    if p == 0:
        return base
    uv = np.asarray(cand, dtype=np.int64)
    if q >= p:
        out = base.copy()
        out[uv[:, 0], uv[:, 1]] = True
        out[uv[:, 1], uv[:, 0]] = True
        return out

    base_lap = laplacian(base)
    mode = str(init_mode).strip().lower()
    if mode not in ("fiedler_topk", "uniform", "random"):
        raise ValueError(
            f"Unsupported mac init_mode: {init_mode}. "
            "Use fiedler_topk|uniform|random."
        )

    if mode == "uniform":
        x = np.full(p, float(q) / float(p), dtype=np.float64)
    elif mode == "random":
        rng = np.random.default_rng(int(seed))
        idx = rng.permutation(p)[:q]
        x = np.zeros(p, dtype=np.float64)
        x[idx] = 1.0
    else:
        _, y0 = _fiedler_from_laplacian(base_lap)
        score0 = (y0[uv[:, 0]] - y0[uv[:, 1]]) ** 2
        x = _top_k_binary_vector(score0, q)

    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    ssum = float(x.sum())
    if ssum > 0.0 and abs(ssum - float(q)) > 1e-12:
        x = x * (float(q) / ssum)
        x = np.clip(x, 0.0, 1.0)

    max_iters = max(0, int(fw_iters))
    tol = float(duality_gap_tol)
    for t in range(max_iters):
        Lx = _laplacian_with_fractional_candidates(base_lap, uv, x)
        _, y = _fiedler_from_laplacian(Lx)
        grad = (y[uv[:, 0]] - y[uv[:, 1]]) ** 2

        s = _top_k_binary_vector(grad, q)
        gap = float(np.dot(grad, (s - x)))
        if tol > 0.0 and gap <= tol:
            break

        alpha = float(2.0 / (2.0 + float(t)))
        x = x + alpha * (s - x)
        x = np.clip(x, 0.0, 1.0)

    select = _top_k_binary_vector(x, q)
    sel_idx = np.flatnonzero(select > 0.5)
    out = base.copy()
    if int(sel_idx.size) > 0:
        sel_uv = uv[sel_idx]
        out[sel_uv[:, 0], sel_uv[:, 1]] = True
        out[sel_uv[:, 1], sel_uv[:, 0]] = True
    return out


def er_greedy_complete(
    adj: np.ndarray,
    m_target: int,
    top_k: int = 1,
    state_config: IncrementalStateConfig | None = None,
) -> np.ndarray:
    state = IncrementalState(adj, config=state_config or IncrementalStateConfig())
    cand_gen = CandidateGenerator(top_k=max(1, top_k), fraction_cap=1.0)
    while edge_count(state.adj) < m_target:
        cand = cand_gen.generate(state)
        if cand.edges.shape[0] == 0:
            break
        u, v = map(int, cand.edges[0])
        state.add_edge(u, v)
    return state.adj


def fiedler_greedy_complete(adj: np.ndarray, m_target: int) -> np.ndarray:
    work = adj.copy()
    target = int(m_target)
    cur_edges = int(edge_count(work))
    if cur_edges >= target:
        return work

    n = int(work.shape[0])
    iu, iv = np.triu_indices(n, k=1)
    available = ~work[iu, iv]
    while cur_edges < target:
        cand_idx = np.flatnonzero(available)
        if int(cand_idx.size) == 0:
            break
        uv = np.stack([iu[cand_idx], iv[cand_idx]], axis=1).astype(np.int64, copy=False)
        _, scores = lambda2_subspace_edge_scores(work, uv)
        pick_local = int(np.argmax(scores))
        pick = int(cand_idx[pick_local])
        u = int(iu[pick])
        v = int(iv[pick])
        work[u, v] = True
        work[v, u] = True
        available[pick] = False
        cur_edges += 1
    return work


def fiedler_greedy_complete_singlevec(
    adj: np.ndarray,
    m_target: int,
    *,
    eigsh_cutoff_n: int = 96,
) -> np.ndarray:
    """
    Original FVG: score non-edge (u,v) by single-vector Fiedler score
    (phi2[u]-phi2[v])^2 using one representative eigenvector for lambda2.
    """
    work = adj.copy()
    target = int(m_target)
    cur_edges = int(edge_count(work))
    if cur_edges >= target:
        return work

    n = int(work.shape[0])
    iu, iv = np.triu_indices(n, k=1)
    available = ~work[iu, iv]
    while cur_edges < target:
        cand_idx = np.flatnonzero(available)
        if int(cand_idx.size) == 0:
            break
        uv = np.stack([iu[cand_idx], iv[cand_idx]], axis=1).astype(np.int64, copy=False)
        _, phi2, _ = two_smallest_nontrivial(work, eigsh_cutoff_n=int(eigsh_cutoff_n))
        scores = (phi2[uv[:, 0]] - phi2[uv[:, 1]]) ** 2
        pick_local = int(np.argmax(scores))
        pick = int(cand_idx[pick_local])
        u = int(iu[pick])
        v = int(iv[pick])
        work[u, v] = True
        work[v, u] = True
        available[pick] = False
        cur_edges += 1
    return work


def _bisection_best_edge_add(
    adj: np.ndarray,
    *,
    bisect_eps: float = 1e-4,
    bisect_sign_tol: float = 1e-12,
    bisect_max_iters: int = 0,
    bisect_final_eval_mode: str = "exact",
    bisect_final_eval_cap: int = 64,
    eigsh_cutoff_n: int = 96,
) -> Tuple[int, int]:
    """
    Select one edge via Kim (2010)-style secular-equation bisection.

    For each candidate non-edge (i,j), define
      f_ij(mu) = 1 + sum_{k=2..n} (v_k[i]-v_k[j])^2 / (lambda_k - mu),
    and bisection on mu in (lambda2, lambda3). Candidates with f_ij(mu)<=0
    have lambda2(G+e_ij) >= mu, so they remain in the contender set.
    """
    cand = non_edges(adj)
    if not cand:
        raise RuntimeError("no non-edge left to add")

    uv = np.asarray(cand, dtype=np.int64)
    p = int(uv.shape[0])
    if p == 1:
        return int(uv[0, 0]), int(uv[0, 1])

    mode = str(bisect_final_eval_mode).strip().lower()
    if mode not in ("exact", "fvg"):
        raise ValueError(
            f"Unsupported bisect_final_eval_mode: {bisect_final_eval_mode}. "
            "Use exact|fvg."
        )

    def _pick_by_fvg(cand_idx: np.ndarray) -> Tuple[int, int]:
        _, scores = lambda2_subspace_edge_scores(adj, uv[cand_idx], eigsh_cutoff_n=int(eigsh_cutoff_n))
        local_idx = int(np.argmax(scores))
        idx = int(cand_idx[local_idx])
        return int(uv[idx, 0]), int(uv[idx, 1])

    def _pick_by_exact(cand_idx: np.ndarray) -> Tuple[int, int]:
        best_idx = int(cand_idx[0])
        best_lam = -np.inf
        work = adj.copy()
        tol = 1e-12
        for idx in cand_idx.tolist():
            u = int(uv[int(idx), 0])
            v = int(uv[int(idx), 1])
            work[u, v] = True
            work[v, u] = True
            lam = float(algebraic_connectivity(work))
            work[u, v] = False
            work[v, u] = False
            if lam > (best_lam + tol):
                best_lam = lam
                best_idx = int(idx)
        return int(uv[best_idx, 0]), int(uv[best_idx, 1])

    # Full spectrum is needed for secular-equation evaluations.
    vals, vecs = np.linalg.eigh(laplacian(adj).astype(np.float64, copy=False))
    vals = np.asarray(vals, dtype=np.float64)
    vecs = np.asarray(vecs, dtype=np.float64)
    if vals.size < 3:
        return _pick_by_fvg(np.arange(p, dtype=np.int64))

    lam2 = float(vals[1])
    gap_tol = max(1e-12, 1e-10 * max(1.0, abs(lam2)))
    mult = int(np.count_nonzero(np.abs(vals[1:] - lam2) <= gap_tol))
    # Paper assumption is lambda2(G) != lambda3(G). When multiplicity is >1,
    # use robust fallback.
    if mult != 1:
        cand_idx = np.arange(p, dtype=np.int64)
        if mode == "fvg" or int(cand_idx.size) > int(max(1, int(bisect_final_eval_cap))):
            return _pick_by_fvg(cand_idx)
        return _pick_by_exact(cand_idx)
    idx3 = -1
    for k in range(2, int(vals.size)):
        if float(vals[k]) > (lam2 + gap_tol):
            idx3 = int(k)
            break
    # Kim's bisection assumes lambda2 < lambda3. Fall back when multiplicity.
    if idx3 < 0:
        return _pick_by_fvg(np.arange(p, dtype=np.int64))

    lam3 = float(vals[idx3])
    eps = max(1e-12, float(bisect_eps))
    if lam3 <= (lam2 + eps):
        return _pick_by_fvg(np.arange(p, dtype=np.int64))

    lam_tail = vals[1:]           # lambda_2 .. lambda_n
    vec_tail = vecs[:, 1:]        # v_2 .. v_n

    S = np.arange(p, dtype=np.int64)
    Lb = float(lam2)
    Ub = float(lam3)
    if int(bisect_max_iters) > 0:
        max_iters = int(bisect_max_iters)
    else:
        max_iters = int(np.ceil(np.log2(max((Ub - Lb) / eps, 1.0)))) + 2
        max_iters = max(1, min(max_iters, 64))

    for _ in range(max_iters):
        if int(S.size) <= 1:
            break
        if (Ub - Lb) <= eps:
            break
        M = 0.5 * (Lb + Ub)

        denom = lam_tail - M
        denom = np.where(np.abs(denom) < 1e-14, np.sign(denom) * 1e-14 + (denom == 0.0) * 1e-14, denom)
        diff = vec_tail[uv[S, 0], :] - vec_tail[uv[S, 1], :]
        fvals = 1.0 + np.sum((diff * diff) / denom[None, :], axis=1)

        keep = S[np.flatnonzero(fvals <= float(bisect_sign_tol))]
        if int(keep.size) > 0:
            # Some candidates have lambda2(G+e) >= M; discard weaker ones.
            Lb = M
            S = keep
        else:
            # No candidate reaches M, so M is an upper bound for the best edge.
            Ub = M

    if int(S.size) <= 0:
        S = np.arange(p, dtype=np.int64)

    if mode == "fvg" or int(S.size) > int(max(1, int(bisect_final_eval_cap))):
        return _pick_by_fvg(S)

    return _pick_by_exact(S)


def bisection_greedy_complete(
    adj: np.ndarray,
    m_target: int,
    *,
    bisect_eps: float = 1e-4,
    bisect_sign_tol: float = 1e-12,
    bisect_max_iters: int = 0,
    bisect_final_eval_mode: str = "exact",
    bisect_final_eval_cap: int = 64,
    eigsh_cutoff_n: int = 96,
) -> np.ndarray:
    """
    Sequential one-edge greedy completion via secular-equation bisection
    (Kim, IEEE TAC 2010).
    """
    work = adj.copy()
    while edge_count(work) < int(m_target):
        non = non_edges(work)
        if not non:
            break
        u, v = _bisection_best_edge_add(
            work,
            bisect_eps=float(bisect_eps),
            bisect_sign_tol=float(bisect_sign_tol),
            bisect_max_iters=int(bisect_max_iters),
            bisect_final_eval_mode=str(bisect_final_eval_mode).strip().lower(),
            bisect_final_eval_cap=int(bisect_final_eval_cap),
            eigsh_cutoff_n=int(eigsh_cutoff_n),
        )
        work[u, v] = True
        work[v, u] = True
    return work


def _adj_edges_array(adj: np.ndarray) -> np.ndarray:
    iu, iv = np.where(np.triu(adj, k=1))
    if iu.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack([iu.astype(np.int64), iv.astype(np.int64)], axis=1)


def _rewire_one_step_by_fiedler(
    adj: np.ndarray,
    *,
    order: str = "remove_then_add",
    eigsh_cutoff_n: int = 96,
) -> bool:
    """
    One Fiedler-score rewiring move from:
      "Optimizing algebraic connectivity by edge rewiring" (2013).

    Let a_ij = |u2[i] - u2[j]| where u2 is Fiedler vector of current graph.
    - Remove edge with smallest a_ij (least harmful)
    - Add non-edge with largest a_ij (most beneficial)
    with connectivity preservation.
    """
    work = adj
    mode = str(order).strip().lower()
    if mode not in ("remove_then_add", "add_then_remove"):
        raise ValueError(
            f"Unsupported rewire order: {order}. Use remove_then_add|add_then_remove."
        )

    edges = _adj_edges_array(work)
    missing = non_edges(work)
    if int(edges.shape[0]) == 0 or len(missing) == 0:
        return False

    uv_non = np.asarray(missing, dtype=np.int64)
    _, phi2, _ = two_smallest_nontrivial(work, eigsh_cutoff_n=int(eigsh_cutoff_n))
    score_e = np.abs(phi2[edges[:, 0]] - phi2[edges[:, 1]])
    score_n = np.abs(phi2[uv_non[:, 0]] - phi2[uv_non[:, 1]])

    rem_order = np.argsort(score_e, kind="mergesort")            # smallest first
    add_order = np.argsort(score_n, kind="mergesort")[::-1]      # largest first

    if mode == "remove_then_add":
        for rloc in rem_order.tolist():
            ru, rv = int(edges[rloc, 0]), int(edges[rloc, 1])
            work[ru, rv] = False
            work[rv, ru] = False
            if not bool(is_connected(work)):
                work[ru, rv] = True
                work[rv, ru] = True
                continue

            for aloc in add_order.tolist():
                au, av = int(uv_non[aloc, 0]), int(uv_non[aloc, 1])
                if bool(work[au, av]):
                    continue
                work[au, av] = True
                work[av, au] = True
                return True

            # No add candidate remained valid; rollback removal.
            work[ru, rv] = True
            work[rv, ru] = True
        return False

    # add_then_remove
    for aloc in add_order.tolist():
        au, av = int(uv_non[aloc, 0]), int(uv_non[aloc, 1])
        if bool(work[au, av]):
            continue
        work[au, av] = True
        work[av, au] = True

        for rloc in rem_order.tolist():
            ru, rv = int(edges[rloc, 0]), int(edges[rloc, 1])
            if not bool(work[ru, rv]):
                continue
            work[ru, rv] = False
            work[rv, ru] = False
            if bool(is_connected(work)):
                return True
            work[ru, rv] = True
            work[rv, ru] = True

        # No removable edge preserved connectivity for this add candidate.
        work[au, av] = False
        work[av, au] = False
    return False


def edge_rewiring_complete(
    adj: np.ndarray,
    m_target: int,
    *,
    rewire_fraction: float = 0.07,
    rewire_steps: int = 0,
    rewire_order: str = "remove_then_add",
    prefill_mode: str = "fvg",
    seed: int = 0,
    eigsh_cutoff_n: int = 96,
) -> np.ndarray:
    """
    Rewire edges to improve lambda2 while keeping edge count fixed.

    Pipeline:
    1) If init graph has fewer than m_target edges, prefill to m_target.
    2) Run a sequence of one-edge swaps using Fiedler-score rules:
         remove min-|u2[i]-u2[j]| edge, add max-|u2[i]-u2[j]| non-edge.
    """
    work = adj.copy()
    m_now = int(edge_count(work))
    m_tar = int(m_target)
    if m_now > m_tar:
        # Not expected in current usage; keep current graph unchanged.
        return work

    mode = str(prefill_mode).strip().lower()
    if mode not in ("fvg", "random"):
        raise ValueError(f"Unsupported rewire prefill_mode: {prefill_mode}. Use fvg|random.")
    if m_now < m_tar:
        if mode == "fvg":
            work = fiedler_greedy_complete(work, m_tar)
        else:
            rng = np.random.default_rng(int(seed))
            work = random_complete(work, m_tar, rng)

    m_cur = int(edge_count(work))
    if m_cur <= 0:
        return work

    if int(rewire_steps) > 0:
        steps = int(rewire_steps)
    else:
        frac = max(0.0, float(rewire_fraction))
        steps = int(round(frac * float(m_cur)))
        if frac > 0.0 and steps <= 0:
            steps = 1
    steps = max(0, min(steps, m_cur))

    for _ in range(steps):
        moved = _rewire_one_step_by_fiedler(
            work,
            order=str(rewire_order).strip().lower(),
            eigsh_cutoff_n=int(eigsh_cutoff_n),
        )
        if not moved:
            break
    return work


def lambda2_greedy_complete(
    adj: np.ndarray,
    m_target: int,
    *,
    candidate_cap: int = 0,
    eval_mode: str = "exact",
    eigsh_cutoff_n: int = 96,
    tie_tol: float = 1e-12,
    tie_break: str = "fvg_among_best",
) -> np.ndarray:
    """
    Greedy solver that picks the edge with the largest true lambda2 gain.

    At each step, evaluate candidate edges by the resulting lambda2 after
    adding that edge, then choose the best.

    Runtime controls:
    - candidate_cap > 0: prefilter candidates by Fiedler score and evaluate
      only top-k edges.
    - eval_mode:
      * "exact": dense eigenvalue computation (most faithful, slowest)
      * "fast": warm-started two_smallest_nontrivial() (faster, approximate)
      * "auto": exact for n<=64, otherwise fast
    - tie_tol / tie_break:
      * tie_tol defines a near-best set:
        candidates with lambda2_new >= (max_lambda2_new - tie_tol).
      * tie_break decides selection within/around that near-best set.
    """
    work = adj.copy()
    mode = str(eval_mode).strip().lower()
    if mode not in ("exact", "fast", "auto"):
        raise ValueError(f"Unsupported eval_mode: {eval_mode}. Use exact|fast|auto.")
    if mode == "auto":
        mode = "exact" if int(work.shape[0]) <= 64 else "fast"
    tie_mode = str(tie_break).strip().lower()
    if tie_mode not in ("fvg_among_best", "fvg_if_all_tied", "first"):
        raise ValueError(
            f"Unsupported tie_break: {tie_break}. "
            "Use fvg_among_best|fvg_if_all_tied|first."
        )

    cap = int(candidate_cap)
    tol = float(tie_tol)
    if tol < 0.0:
        raise ValueError(f"tie_tol must be >= 0, got {tie_tol}")
    while edge_count(work) < m_target:
        non = non_edges(work)
        if not non:
            break

        cand_edges = non
        if cap > 0 and len(non) > cap:
            uv = np.asarray(non, dtype=np.int64)
            _, scores = lambda2_subspace_edge_scores(work, uv, eigsh_cutoff_n=int(eigsh_cutoff_n))
            top_idx = np.argpartition(-scores, cap - 1)[:cap]
            top_idx = top_idx[np.argsort(-scores[top_idx], kind="mergesort")]
            cand_edges = [non[int(i)] for i in top_idx.tolist()]

        cand_uv: List[Tuple[int, int]] = []
        cand_lam2: List[float] = []
        warm: np.ndarray | None = None
        if mode == "fast":
            _, warm, _ = two_smallest_nontrivial(work, eigsh_cutoff_n=int(eigsh_cutoff_n))
        for u, v in cand_edges:
            work[u, v] = True
            work[v, u] = True
            if mode == "exact":
                lam2_new = float(algebraic_connectivity(work))
            else:
                lam2_new, _, _ = two_smallest_nontrivial(
                    work,
                    warm_start=warm,
                    eigsh_cutoff_n=int(eigsh_cutoff_n),
                )
                lam2_new = float(lam2_new)
            work[u, v] = False
            work[v, u] = False

            cand_uv.append((int(u), int(v)))
            cand_lam2.append(lam2_new)

        if not cand_uv:
            break
        lam_arr = np.asarray(cand_lam2, dtype=np.float64)
        lam_max = float(lam_arr.max())
        lam_min = float(lam_arr.min())
        near_best_idx = np.flatnonzero(lam_arr >= (lam_max - tol))
        if near_best_idx.size == 0:
            near_best_idx = np.asarray([int(np.argmax(lam_arr))], dtype=np.int64)

        use_fv = False
        if tie_mode == "first":
            best_idx = int(near_best_idx[0])
        elif tie_mode == "fvg_if_all_tied":
            if float(lam_max - lam_min) <= tol:
                use_fv = True
            else:
                best_idx = int(near_best_idx[0])
        else:  # fvg_among_best
            if int(near_best_idx.size) <= 1:
                best_idx = int(near_best_idx[0])
            else:
                use_fv = True

        if use_fv:
            uv = np.asarray(cand_uv, dtype=np.int64)[near_best_idx]
            _, fv_scores = lambda2_subspace_edge_scores(work, uv, eigsh_cutoff_n=int(eigsh_cutoff_n))
            local_idx = int(np.argmax(fv_scores))
            best_idx = int(near_best_idx[local_idx])

        u_best, v_best = cand_uv[best_idx]
        work[u_best, v_best] = True
        work[v_best, u_best] = True

    return work


def _fiedler_seed_candidate_indices(
    adj: np.ndarray,
    candidate_uv: np.ndarray,
    k_add: int,
) -> List[int]:
    """Pick k candidate-edge indices using the same Fiedler greedy score."""
    work = adj.copy()
    p = int(candidate_uv.shape[0])
    available = np.ones(p, dtype=bool)
    selected: List[int] = []

    for _ in range(max(0, int(k_add))):
        idx_avail = np.flatnonzero(available)
        if idx_avail.size == 0:
            break
        uv = candidate_uv[idx_avail]
        _, scores = lambda2_subspace_edge_scores(work, uv)
        idx = int(idx_avail[int(np.argmax(scores))])
        selected.append(idx)
        available[idx] = False
        u, v = int(candidate_uv[idx, 0]), int(candidate_uv[idx, 1])
        work[u, v] = True
        work[v, u] = True
    return selected


def _round_top_k_indices(x: np.ndarray, k_add: int) -> np.ndarray:
    k = int(k_add)
    idx = np.argpartition(-x, k - 1)[:k]
    # Stable order for reproducibility under ties.
    return idx[np.argsort(-x[idx], kind="mergesort")]


def _apply_candidate_edge_indices(
    adj: np.ndarray,
    candidate_uv: np.ndarray,
    selected_idx: np.ndarray,
) -> np.ndarray:
    out = adj.copy()
    for sel in selected_idx.tolist():
        uu, vv = int(candidate_uv[sel, 0]), int(candidate_uv[sel, 1])
        out[uu, vv] = True
        out[vv, uu] = True
    return out


def _fiedler_rank_indices(
    adj: np.ndarray,
    candidate_uv: np.ndarray,
    *,
    descending: bool = True,
) -> np.ndarray:
    _, scores = lambda2_subspace_edge_scores(adj, candidate_uv)
    order = np.argsort(scores, kind="mergesort")
    return order[::-1] if descending else order


def _choose_edge_combos(
    pool: Sequence[int],
    k: int,
    max_combos: int,
    rng: np.random.Generator,
) -> List[Tuple[int, ...]]:
    if k <= 0:
        return []
    if len(pool) < k:
        return []
    all_combos = list(combinations(pool, k))
    if max_combos <= 0 or len(all_combos) <= max_combos:
        return all_combos
    idx = rng.choice(len(all_combos), size=max_combos, replace=False)
    idx.sort()
    return [all_combos[int(i)] for i in idx.tolist()]


def kopt_exchange_complete(
    adj: np.ndarray,
    m_target: int,
    *,
    k: int = 1,
    m: int = 20,
    max_rounds: int = 20,
    combo_cap_add: int = 300,
    combo_cap_del: int = 300,
    seed: int = 0,
) -> np.ndarray:
    """
    Greedy k-opt exchange heuristic for augmentation from a fixed base graph.

    - Base graph edges are immutable.
    - A fixed number q of augment edges is maintained.
    - Each round: add k augment edges from top-ranked candidates and remove k
      augment edges from low-ranked selected edges, accepting improving swaps.
    """
    base = adj.copy()
    q = int(m_target - edge_count(base))
    if q <= 0:
        return base

    cand = non_edges(base)
    p = len(cand)
    if p == 0:
        return base
    if q >= p:
        out = base.copy()
        for uu, vv in cand:
            out[uu, vv] = True
            out[vv, uu] = True
        return out

    k_eff = max(1, int(k))
    pool_m = max(1, int(m))
    rng = np.random.default_rng(int(seed))
    candidate_uv = np.asarray(cand, dtype=np.int64)

    # Stage-1 initial feasible augmentation: exact sequential Fiedler-greedy
    # (same construction as the baseline path_fvg completion).
    selected = _fiedler_seed_candidate_indices(base, candidate_uv, q)
    if len(selected) < q:
        # Defensive fallback for degenerate numeric/tie cases.
        remaining = np.setdiff1d(np.arange(p, dtype=np.int64), np.asarray(selected, dtype=np.int64))
        fill = min(int(q - len(selected)), int(remaining.size))
        if fill > 0:
            extra = remaining[:fill]
            selected.extend(int(x) for x in extra.tolist())

    selected_set = set(selected)
    current_adj = _apply_candidate_edge_indices(base, candidate_uv, np.asarray(selected, dtype=np.int64))
    current_lam2 = float(algebraic_connectivity(current_adj))

    for _ in range(max(0, int(max_rounds))):
        add_available = [idx for idx in range(p) if idx not in selected_set]
        if len(add_available) < k_eff:
            break

        # Add side ranking from current graph using Fiedler score.
        add_scores_order = _fiedler_rank_indices(
            current_adj,
            candidate_uv[np.asarray(add_available, dtype=np.int64)],
            descending=True,
        )
        add_pool = [add_available[int(i)] for i in add_scores_order[: min(pool_m, len(add_available))].tolist()]
        add_combos = _choose_edge_combos(add_pool, k_eff, int(combo_cap_add), rng)
        if not add_combos:
            break

        best_lam2 = current_lam2
        best_selected: List[int] | None = None

        for add_combo in add_combos:
            plus_set = set(selected_set)
            plus_set.update(int(x) for x in add_combo)
            plus_list = sorted(plus_set)
            plus_adj = _apply_candidate_edge_indices(base, candidate_uv, np.asarray(plus_list, dtype=np.int64))

            # Remove side ranking from augmented graph; choose low-score selected edges.
            plus_uv = candidate_uv[np.asarray(plus_list, dtype=np.int64)]
            rem_order_local = _fiedler_rank_indices(plus_adj, plus_uv, descending=False)
            rem_pool_local = [plus_list[int(i)] for i in rem_order_local[: min(pool_m, len(plus_list))].tolist()]
            rem_combos = _choose_edge_combos(rem_pool_local, k_eff, int(combo_cap_del), rng)
            if not rem_combos:
                continue

            for rem_combo in rem_combos:
                new_set = set(plus_set)
                for ridx in rem_combo:
                    new_set.discard(int(ridx))
                if len(new_set) != q:
                    continue
                new_list = sorted(new_set)
                new_adj = _apply_candidate_edge_indices(base, candidate_uv, np.asarray(new_list, dtype=np.int64))
                lam2_new = float(algebraic_connectivity(new_adj))
                if lam2_new > (best_lam2 + 1e-12):
                    best_lam2 = lam2_new
                    best_selected = new_list

        if best_selected is None:
            break
        selected = best_selected
        selected_set = set(selected)
        current_adj = _apply_candidate_edge_indices(base, candidate_uv, np.asarray(selected, dtype=np.int64))
        current_lam2 = best_lam2

    return current_adj


def _mch_prepare_weights(n: int, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        w = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(w, 0.0)
        return w
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n, n):
        raise ValueError(f"weights must have shape {(n, n)}, got {w.shape}")
    if np.any(w < 0.0):
        raise ValueError("weights must be nonnegative")
    w = 0.5 * (w + w.T)
    np.fill_diagonal(w, 0.0)
    return w


def _mch_rank_central_nodes(w: np.ndarray, k: int) -> List[int]:
    """
    Rank central-node candidates by sum of top-(n-k) incident weights.
    """
    n = int(w.shape[0])
    take = max(1, min(n - 1, n - int(k)))
    s = np.empty(n, dtype=np.float64)
    for i in range(n):
        row = w[i, np.arange(n) != i]
        if take >= row.size:
            s[i] = float(np.sum(row))
        else:
            idx = np.argpartition(-row, take - 1)[:take]
            s[i] = float(np.sum(row[idx]))
    return np.argsort(-s, kind="mergesort").astype(np.int64).tolist()


def _mch_edge(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _mch_priority_edges_for_central(
    n: int,
    central: int,
    k: int,
    h2: int,
    w: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Build Algorithm-2 style edge priority order O*_le for one central node.
    """
    k_eff = max(1, min(n - 1, int(k)))
    top_h2 = max(1, int(h2))

    neigh = [j for j in range(n) if j != central]
    neigh_sorted = sorted(neigh, key=lambda j: (-float(w[central, j]), int(j)))

    # Top (n-k) spokes incident to central.
    spoke_count = max(1, min(n - 1, n - k_eff))
    spoke_nodes = neigh_sorted[:spoke_count]

    # Bottom (k-1) neighbors are treated as "leaf" nodes for O*_le.
    leaf_count = max(0, min(len(neigh_sorted), k_eff - 1))
    leaf_nodes = neigh_sorted[-leaf_count:] if leaf_count > 0 else []

    star = np.zeros((n, n), dtype=bool)
    for j in range(n):
        if j == central:
            continue
        star[central, j] = True
        star[j, central] = True
    _, v_star, _ = two_smallest_nontrivial(star)

    priority_edges: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()

    def push_edge(u: int, v: int) -> None:
        if u == v:
            return
        e = _mch_edge(int(u), int(v))
        if e in seen:
            return
        seen.add(e)
        priority_edges.append(e)

    # Stage 1: central spokes.
    for j in spoke_nodes:
        push_edge(central, int(j))

    # Stage 2: O*_le ranking for selected leaf nodes.
    for leaf in leaf_nodes:
        # The paper scores edges (i, j) where i is connected to central.
        leaf_cands = [i for i in spoke_nodes if i != leaf]
        if not leaf_cands:
            continue
        scores = np.asarray(
            [float(w[leaf, i]) * float((v_star[leaf] - v_star[i]) ** 2) for i in leaf_cands],
            dtype=np.float64,
        )
        order = np.argsort(scores, kind="mergesort")[::-1]
        for idx in order[: min(top_h2, int(order.size))].tolist():
            push_edge(leaf, int(leaf_cands[int(idx)]))

    # Stage 3: complete deterministic ranking to keep feasibility under any budget.
    all_edges: List[Tuple[int, int]] = []
    all_scores: List[float] = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
            if u == central or v == central:
                all_scores.append(float(w[u, v]))
            else:
                all_scores.append(float(w[u, v]) * float((v_star[u] - v_star[v]) ** 2))
    order_all = np.argsort(np.asarray(all_scores, dtype=np.float64), kind="mergesort")[::-1]
    for idx in order_all.tolist():
        uu, vv = all_edges[int(idx)]
        push_edge(int(uu), int(vv))

    return priority_edges


def _mch_complete_for_central(
    base: np.ndarray,
    m_target: int,
    central: int,
    h2: int,
    *,
    k: int,
    w: np.ndarray,
) -> np.ndarray:
    """
    Construct one MCH candidate for a fixed central node via direct ranking fill.
    """
    n = int(base.shape[0])
    priority_edges = _mch_priority_edges_for_central(
        n=n,
        central=int(central),
        k=int(k),
        h2=int(h2),
        w=w,
    )
    work = base.copy()
    cur_m = int(edge_count(work))
    for u, v in priority_edges:
        if cur_m >= int(m_target):
            break
        if not work[u, v]:
            work[u, v] = True
            work[v, u] = True
            cur_m += 1
    return work


def _mch_oa_complete_for_central(
    base: np.ndarray,
    m_target: int,
    central: int,
    h2: int,
    *,
    k: int,
    w: np.ndarray,
    oa_pool_extra: int,
    oa_max_iters: int,
    oa_tol: float,
    oa_solver: str,
    oa_verbose: bool,
) -> np.ndarray:
    """
    Solve ranking-restricted MCH subproblem via OA MILP with eigenvector cuts.
    """
    try:
        import cvxpy as cp  # type: ignore
    except Exception as exc:
        raise RuntimeError("paper-faithful MCH-OA requires cvxpy. Install with: pip install cvxpy") from exc

    base_adj = base.copy()
    k_add = int(m_target - edge_count(base_adj))
    if k_add <= 0:
        return base_adj

    n = int(base_adj.shape[0])
    priority = _mch_priority_edges_for_central(
        n=n,
        central=int(central),
        k=int(k),
        h2=int(h2),
        w=w,
    )
    non_set = set(non_edges(base_adj))
    cand_ranked = [e for e in priority if e in non_set]
    if not cand_ranked:
        return base_adj

    if int(oa_pool_extra) < 0:
        p_cap = len(cand_ranked)
    else:
        p_cap = min(len(cand_ranked), max(k_add, k_add + int(oa_pool_extra)))
    candidate_uv = np.asarray(cand_ranked[:p_cap], dtype=np.int64)
    p = int(candidate_uv.shape[0])
    if p <= 0:
        return base_adj
    if k_add >= p:
        return _apply_candidate_edge_indices(base_adj, candidate_uv, np.arange(p, dtype=np.int64))

    L0 = laplacian(base_adj).astype(np.float64, copy=False)
    J = np.eye(n, dtype=np.float64) - (np.ones((n, n), dtype=np.float64) / float(n))

    cuts_alpha: List[np.ndarray] = []
    cuts_beta: List[float] = []

    def add_cut(v: np.ndarray) -> bool:
        vv = np.asarray(v, dtype=np.float64).reshape(-1)
        if vv.shape[0] != n:
            return False
        denom = float(vv @ (J @ vv))
        if denom <= 1e-12:
            return False
        alpha = ((vv[candidate_uv[:, 0]] - vv[candidate_uv[:, 1]]) ** 2) / denom
        beta = float(vv @ (L0 @ vv)) / denom
        cuts_alpha.append(np.asarray(alpha, dtype=np.float64))
        cuts_beta.append(float(beta))
        return True

    # Initial cut from base Fiedler vector.
    _, phi0, _ = two_smallest_nontrivial(base_adj)
    if not add_cut(phi0):
        v = np.linspace(-1.0, 1.0, num=n, dtype=np.float64)
        v -= float(v.mean())
        if not add_cut(v):
            raise RuntimeError("MCH-OA failed to generate an initial eigenvector cut.")

    installed = {s.upper() for s in cp.installed_solvers()}
    solver_upper = str(oa_solver).strip().upper()
    if solver_upper in ("", "AUTO"):
        for name in ("HIGHS", "SCIPY"):
            if name in installed:
                solver_upper = name
                break
        else:
            raise RuntimeError(f"No MILP-capable cvxpy solver found. Installed: {sorted(installed)}")
    if solver_upper not in installed:
        raise RuntimeError(
            f"Requested MCH-OA solver '{solver_upper}' is not installed. Installed: {sorted(installed)}"
        )

    best_adj: np.ndarray | None = None
    best_lam2 = -np.inf
    last_adj: np.ndarray | None = None

    for _ in range(max(1, int(oa_max_iters))):
        x = cp.Variable(p, boolean=True)
        gamma = cp.Variable()

        # Degree lower-bound structure from Fkl for the chosen center:
        # deg_center(final) >= n-k.
        # In augmentation mode, this can be infeasible for small residual budget;
        # enforce the strongest reachable target to retain feasibility.
        center = int(central)
        base_deg_center = int(base_adj[center].sum())
        cand_incident = ((candidate_uv[:, 0] == center) | (candidate_uv[:, 1] == center)).astype(np.float64)
        avail_incident = int(np.count_nonzero(cand_incident))
        reachable_target = base_deg_center + min(int(k_add), avail_incident)
        target_center_deg = min(int(n - int(k)), reachable_target)

        constraints = [
            cp.sum(x) == float(k_add),
            gamma >= 0.0,
            gamma <= float(n),
            base_deg_center + cand_incident @ x >= float(target_center_deg),
        ]
        for alpha, beta in zip(cuts_alpha, cuts_beta):
            constraints.append(beta + alpha @ x >= gamma)

        prob = cp.Problem(cp.Maximize(gamma), constraints)
        prob.solve(solver=getattr(cp, solver_upper), verbose=bool(oa_verbose))
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"MCH-OA MILP failed with status={prob.status}")
        if x.value is None:
            raise RuntimeError("MCH-OA MILP returned no x solution.")

        x_val = np.asarray(x.value, dtype=np.float64).reshape(-1)
        x_val = np.nan_to_num(x_val, nan=0.0, posinf=1.0, neginf=0.0)
        idx = _round_top_k_indices(x_val, k_add)
        cand_adj = _apply_candidate_edge_indices(base_adj, candidate_uv, idx)
        last_adj = cand_adj

        lam2, phi2, _ = two_smallest_nontrivial(cand_adj)
        lam2_f = float(lam2)
        if lam2_f > (best_lam2 + 1e-12):
            best_lam2 = lam2_f
            best_adj = cand_adj

        gamma_val = float(gamma.value) if gamma.value is not None else np.inf
        if gamma_val <= (lam2_f + float(oa_tol)):
            break

        if not add_cut(phi2):
            break

    if best_adj is not None:
        return best_adj
    if last_adj is not None:
        return last_adj
    return base_adj


def mch_complete(
    adj: np.ndarray,
    m_target: int,
    *,
    k: int = 3,
    h1: int = 5,
    h2: int = 3,
    weights: np.ndarray | None = None,
    oa_pool_extra: int = 40,
    oa_max_iters: int = 25,
    oa_tol: float = 1e-6,
    oa_solver: str = "HIGHS",
    oa_verbose: bool = False,
) -> np.ndarray:
    """
    Maximum Cost Heuristic (MCH) with paper-style restricted OA optimization.

    For each of top-h1 central candidates from O*_cn, this builds O*_le ranking
    and solves a ranking-restricted OA-MILP (eigenvector cuts) over a capped
    candidate pool. The best feasible graph across those central-node choices
    is returned.
    """
    base = adj.copy()
    q = int(m_target - edge_count(base))
    if q <= 0:
        return base
    if not non_edges(base):
        return base

    n = int(base.shape[0])
    kk = int(max(1, min(n - 1, int(k))))
    top_h1 = max(1, min(n, int(h1)))
    w = _mch_prepare_weights(n, weights)

    central_order = _mch_rank_central_nodes(w, kk)[:top_h1]
    best_adj: np.ndarray | None = None
    best_lam2 = -np.inf
    for c in central_order:
        cand_adj = _mch_oa_complete_for_central(
            base,
            m_target,
            int(c),
            int(h2),
            k=kk,
            w=w,
            oa_pool_extra=int(oa_pool_extra),
            oa_max_iters=int(oa_max_iters),
            oa_tol=float(oa_tol),
            oa_solver=str(oa_solver).strip(),
            oa_verbose=bool(oa_verbose),
        )
        lam2 = float(algebraic_connectivity(cand_adj))
        if best_adj is None or lam2 > (best_lam2 + 1e-12):
            best_lam2 = lam2
            best_adj = cand_adj
    return base if best_adj is None else best_adj


def oa_exact_complete(
    adj: np.ndarray,
    m_target: int,
    *,
    solver: str = "HIGHS",
    max_iters: int = 60,
    tol: float = 1e-6,
    verbose: bool = False,
    warm_start_fvg: bool = True,
) -> np.ndarray:
    """
    Exact OA solver (for small instances) via iterative eigenvector cuts.

    Decision variables:
      - x_e in {0,1} for each non-edge of the base graph
      - gamma (lower-bounded objective surrogate for lambda2)

    MILP at each OA iteration:
      maximize gamma
      s.t. sum_e x_e = k_add
           beta_t + alpha_t^T x >= gamma   for all cuts t

    where each cut is generated from a violated eigenvector at an integer
    solution (x), following the standard OA/eigenvector-cut framework.
    """
    try:
        import cvxpy as cp  # type: ignore
    except Exception as exc:
        raise RuntimeError("oa_exact solver requires cvxpy. Install with: pip install cvxpy") from exc

    base = adj.copy()
    k_add = int(m_target - edge_count(base))
    if k_add <= 0:
        return base

    cand = non_edges(base)
    p = len(cand)
    if p == 0:
        return base
    if k_add >= p:
        out = base.copy()
        for uu, vv in cand:
            out[uu, vv] = True
            out[vv, uu] = True
        return out

    n = int(base.shape[0])
    candidate_uv = np.asarray(cand, dtype=np.int64)
    L0 = laplacian(base).astype(np.float64, copy=False)
    J = np.eye(n, dtype=np.float64) - (np.ones((n, n), dtype=np.float64) / float(n))

    cuts_alpha: List[np.ndarray] = []
    cuts_beta: List[float] = []

    def add_cut(v: np.ndarray) -> bool:
        vv = np.asarray(v, dtype=np.float64).reshape(-1)
        if vv.shape[0] != n:
            return False
        denom = float(vv @ (J @ vv))
        if denom <= 1e-12:
            return False
        alpha = ((vv[candidate_uv[:, 0]] - vv[candidate_uv[:, 1]]) ** 2) / denom
        beta = float(vv @ (L0 @ vv)) / denom
        cuts_alpha.append(np.asarray(alpha, dtype=np.float64))
        cuts_beta.append(float(beta))
        return True

    _, phi0, _ = two_smallest_nontrivial(base)
    if not add_cut(phi0):
        v = np.linspace(-1.0, 1.0, num=n, dtype=np.float64)
        v -= float(v.mean())
        if not add_cut(v):
            raise RuntimeError("oa_exact failed to initialize eigenvector cuts.")

    installed = {s.upper() for s in cp.installed_solvers()}
    solver_upper = str(solver).strip().upper()
    if solver_upper in ("", "AUTO"):
        for name in ("HIGHS", "SCIPY"):
            if name in installed:
                solver_upper = name
                break
        else:
            raise RuntimeError(f"No MILP-capable cvxpy solver found. Installed: {sorted(installed)}")
    if solver_upper not in installed:
        raise RuntimeError(
            f"Requested oa_exact solver '{solver_upper}' is not installed. Installed: {sorted(installed)}"
        )

    best_adj: np.ndarray | None = None
    best_lam2 = -np.inf
    if bool(warm_start_fvg):
        warm_adj = fiedler_greedy_complete(base.copy(), m_target)
        best_adj = warm_adj
        best_lam2 = float(algebraic_connectivity(warm_adj))

    last_adj: np.ndarray | None = None
    for _ in range(max(1, int(max_iters))):
        x = cp.Variable(p, boolean=True)
        gamma = cp.Variable()
        constraints = [
            cp.sum(x) == float(k_add),
            gamma >= 0.0,
            gamma <= float(n),
        ]
        for alpha, beta in zip(cuts_alpha, cuts_beta):
            constraints.append(beta + alpha @ x >= gamma)

        prob = cp.Problem(cp.Maximize(gamma), constraints)
        prob.solve(solver=getattr(cp, solver_upper), verbose=bool(verbose))
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"oa_exact MILP failed with status={prob.status}")
        if x.value is None:
            raise RuntimeError("oa_exact MILP returned no x solution.")

        x_val = np.asarray(x.value, dtype=np.float64).reshape(-1)
        x_val = np.nan_to_num(x_val, nan=0.0, posinf=1.0, neginf=0.0)
        idx = np.flatnonzero(x_val > 0.5).astype(np.int64)
        if int(idx.size) != int(k_add):
            idx = _round_top_k_indices(x_val, k_add).astype(np.int64)

        cand_adj = _apply_candidate_edge_indices(base, candidate_uv, idx)
        last_adj = cand_adj
        lam2, phi2, _ = two_smallest_nontrivial(cand_adj)
        lam2_f = float(lam2)
        if best_adj is None or lam2_f > (best_lam2 + 1e-12):
            best_lam2 = lam2_f
            best_adj = cand_adj

        ub = float(gamma.value) if gamma.value is not None else np.inf
        if ub <= (best_lam2 + float(tol)):
            break
        if not add_cut(phi2):
            break

    if best_adj is not None:
        return best_adj
    if last_adj is not None:
        return last_adj
    return base


def _connected_components_nodes(adj: np.ndarray) -> List[List[int]]:
    n = int(adj.shape[0])
    seen = np.zeros(n, dtype=bool)
    comps: List[List[int]] = []
    for s in range(n):
        if bool(seen[s]):
            continue
        q: deque[int] = deque([int(s)])
        seen[s] = True
        comp: List[int] = []
        while q:
            u = int(q.popleft())
            comp.append(u)
            nbrs = np.flatnonzero(adj[u])
            for v in nbrs.tolist():
                vv = int(v)
                if bool(seen[vv]):
                    continue
                seen[vv] = True
                q.append(vv)
        comps.append(comp)
    return comps


def oa_exact_global(
    n: int,
    m_target: int,
    *,
    solver: str = "HIGHS",
    max_iters: int = 120,
    tol: float = 1e-6,
    verbose: bool = False,
    warm_start_fvg: bool = True,
) -> np.ndarray:
    """
    Global OA exact solver over all m-edge connected graphs on n nodes.

    Decision variables are on all complete-graph edges, not augmentations
    from a fixed base graph. Connectivity is enforced via cut separation.
    """
    try:
        import cvxpy as cp  # type: ignore
    except Exception as exc:
        raise RuntimeError("oa_exact_global solver requires cvxpy. Install with: pip install cvxpy") from exc

    n_i = int(n)
    m_tar = int(m_target)
    m_hi = int(max_edges(n_i))
    if n_i < 2:
        raise ValueError(f"n must be >= 2, got {n_i}")
    if m_tar < int(n_i - 1):
        raise ValueError(
            f"oa_exact_global assumes connected regime m>=n-1, got n={n_i}, m_target={m_tar}"
        )
    if m_tar > m_hi:
        raise ValueError(f"m_target exceeds complete graph edges: n={n_i}, m_target={m_tar}, m_max={m_hi}")
    if m_tar == m_hi:
        out = np.ones((n_i, n_i), dtype=bool)
        np.fill_diagonal(out, False)
        return out

    cand = list(combinations(range(n_i), 2))
    candidate_uv = np.asarray(cand, dtype=np.int64)
    p = int(candidate_uv.shape[0])

    J = np.eye(n_i, dtype=np.float64) - (np.ones((n_i, n_i), dtype=np.float64) / float(n_i))
    cuts_alpha: List[np.ndarray] = []
    # No base graph term here; beta is always 0 for global edge-selection.
    cuts_beta: List[float] = []

    def add_cut(v: np.ndarray) -> bool:
        vv = np.asarray(v, dtype=np.float64).reshape(-1)
        if vv.shape[0] != n_i:
            return False
        denom = float(vv @ (J @ vv))
        if denom <= 1e-12:
            return False
        alpha = ((vv[candidate_uv[:, 0]] - vv[candidate_uv[:, 1]]) ** 2) / denom
        cuts_alpha.append(np.asarray(alpha, dtype=np.float64))
        cuts_beta.append(0.0)
        return True

    # Seed OA with a non-constant vector.
    v0 = np.linspace(-1.0, 1.0, num=n_i, dtype=np.float64)
    v0 -= float(v0.mean())
    if not add_cut(v0):
        raise RuntimeError("oa_exact_global failed to initialize eigenvector cuts.")

    installed = {s.upper() for s in cp.installed_solvers()}
    solver_upper = str(solver).strip().upper()
    if solver_upper in ("", "AUTO"):
        for name in ("HIGHS", "SCIPY"):
            if name in installed:
                solver_upper = name
                break
        else:
            raise RuntimeError(f"No MILP-capable cvxpy solver found. Installed: {sorted(installed)}")
    if solver_upper not in installed:
        raise RuntimeError(
            f"Requested oa_exact_global solver '{solver_upper}' is not installed. Installed: {sorted(installed)}"
        )

    topol_cuts: List[np.ndarray] = []
    topol_cut_keys: set[Tuple[int, ...]] = set()

    best_adj: np.ndarray | None = None
    best_lam2 = -np.inf
    if bool(warm_start_fvg):
        warm1 = fiedler_greedy_complete(path_graph(n_i), m_tar)
        lam1 = float(algebraic_connectivity(warm1))
        warm2 = fiedler_greedy_complete(star_graph(n_i), m_tar)
        lam2 = float(algebraic_connectivity(warm2))
        if lam2 > lam1:
            best_adj = warm2
            best_lam2 = lam2
        else:
            best_adj = warm1
            best_lam2 = lam1

    last_adj: np.ndarray | None = None
    for _ in range(max(1, int(max_iters))):
        x = cp.Variable(p, boolean=True)
        gamma = cp.Variable()
        constraints = [
            cp.sum(x) == float(m_tar),
            gamma >= 0.0,
            gamma <= float(n_i),
        ]
        for alpha, beta in zip(cuts_alpha, cuts_beta):
            constraints.append(beta + alpha @ x >= gamma)
        for idx in topol_cuts:
            constraints.append(cp.sum(x[idx]) >= 1.0)

        prob = cp.Problem(cp.Maximize(gamma), constraints)
        prob.solve(solver=getattr(cp, solver_upper), verbose=bool(verbose))
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"oa_exact_global MILP failed with status={prob.status}")
        if x.value is None:
            raise RuntimeError("oa_exact_global MILP returned no x solution.")

        x_val = np.asarray(x.value, dtype=np.float64).reshape(-1)
        x_val = np.nan_to_num(x_val, nan=0.0, posinf=1.0, neginf=0.0)
        idx = np.flatnonzero(x_val > 0.5).astype(np.int64)
        if int(idx.size) != int(m_tar):
            idx = _round_top_k_indices(x_val, m_tar).astype(np.int64)

        cand_adj = np.zeros((n_i, n_i), dtype=bool)
        for sel in idx.tolist():
            uu = int(candidate_uv[int(sel), 0])
            vv = int(candidate_uv[int(sel), 1])
            cand_adj[uu, vv] = True
            cand_adj[vv, uu] = True
        last_adj = cand_adj

        # Separate connectivity cuts first: sum_{delta(S)} x >= 1 for each
        # disconnected component S in incumbent.
        if not bool(is_connected(cand_adj)):
            comps = _connected_components_nodes(cand_adj)
            added_topol = 0
            for comp in comps:
                s = tuple(sorted(int(v) for v in comp))
                if len(s) == 0 or len(s) == n_i:
                    continue
                comp_set = set(s)
                comp_bar = tuple(i for i in range(n_i) if i not in comp_set)
                key = min(s, comp_bar)
                if key in topol_cut_keys:
                    continue
                in_s_u = np.isin(candidate_uv[:, 0], np.asarray(s, dtype=np.int64))
                in_s_v = np.isin(candidate_uv[:, 1], np.asarray(s, dtype=np.int64))
                cross = np.flatnonzero(np.logical_xor(in_s_u, in_s_v)).astype(np.int64)
                if int(cross.size) <= 0:
                    continue
                topol_cuts.append(cross)
                topol_cut_keys.add(key)
                added_topol += 1
            if added_topol <= 0:
                break
            continue

        lam2_f, phi2, _ = two_smallest_nontrivial(cand_adj)
        lam2 = float(lam2_f)
        if best_adj is None or lam2 > (best_lam2 + 1e-12):
            best_lam2 = lam2
            best_adj = cand_adj

        ub = float(gamma.value) if gamma.value is not None else np.inf
        if ub <= (best_lam2 + float(tol)):
            break
        if not add_cut(phi2):
            break

    if best_adj is not None:
        return best_adj
    if last_adj is not None and bool(is_connected(last_adj)):
        return last_adj
    # Fallback should be connected and feasible in connected regime.
    return fiedler_greedy_complete(path_graph(n_i), m_tar)


def _laplacian_affine_operator(
    n: int,
    candidate_uv: np.ndarray,
) -> sparse.csc_matrix:
    """
    Build sparse A such that vec(L(x)) = vec(L0) + A @ x for candidate edges.
    """
    p = int(candidate_uv.shape[0])
    rows = np.empty(4 * p, dtype=np.int64)
    cols = np.empty(4 * p, dtype=np.int64)
    data = np.empty(4 * p, dtype=np.float64)
    for e in range(p):
        u = int(candidate_uv[e, 0])
        v = int(candidate_uv[e, 1])
        b = 4 * e
        rows[b : b + 4] = (
            u * n + u,
            v * n + v,
            u * n + v,
            v * n + u,
        )
        cols[b : b + 4] = (e, e, e, e)
        data[b : b + 4] = (1.0, 1.0, -1.0, -1.0)
    return sparse.csc_matrix((data, (rows, cols)), shape=(n * n, p))


def _solve_relaxed_sdp_x(
    adj: np.ndarray,
    candidate_uv: np.ndarray,
    k_add: int,
    *,
    solver: str = "SCS",
    max_iters: int = 10_000,
    eps: float = 1e-5,
    verbose: bool = False,
) -> np.ndarray:
    """
    Solve relaxed SDP and return the relaxed edge-selection vector x.
    """
    try:
        import cvxpy as cp  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "sdp_greedy completion requires cvxpy. Install with: pip install cvxpy"
        ) from exc

    p = int(candidate_uv.shape[0])
    if p <= 0:
        return np.zeros(0, dtype=np.float64)
    if k_add <= 0:
        return np.zeros(p, dtype=np.float64)
    if k_add >= p:
        return np.ones(p, dtype=np.float64)

    n = int(adj.shape[0])
    L0 = laplacian(adj).astype(np.float64, copy=False)
    J = np.eye(n, dtype=np.float64) - (np.ones((n, n), dtype=np.float64) / float(n))
    A = _laplacian_affine_operator(n, candidate_uv)

    x = cp.Variable(p)
    h = cp.Variable()
    vec_l0 = L0.reshape(n * n, order="C")
    vec_lx = vec_l0 + A @ x
    Lx = cp.reshape(vec_lx, (n, n), order="C")

    constraints = [
        x >= 0.0,
        x <= 1.0,
        cp.sum(x) == float(k_add),
        (Lx - h * J) >> 0,
    ]
    prob = cp.Problem(cp.Maximize(h), constraints)

    installed = {s.upper() for s in cp.installed_solvers()}
    solver_upper = str(solver).strip().upper()
    if solver_upper in ("", "AUTO"):
        for name in ("MOSEK", "CVXOPT", "SCS"):
            if name in installed:
                solver_upper = name
                break
        else:
            solver_upper = "SCS"
    if solver_upper not in installed:
        raise RuntimeError(
            f"Requested SDP solver '{solver_upper}' is not installed. "
            f"Installed: {sorted(installed)}"
        )

    solve_kwargs = {"verbose": bool(verbose)}
    if solver_upper == "SCS":
        solve_kwargs["max_iters"] = int(max_iters)
        solve_kwargs["eps"] = float(eps)

    prob.solve(solver=getattr(cp, solver_upper), **solve_kwargs)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"SDP solve failed with status={prob.status}")
    if x.value is None:
        raise RuntimeError("SDP solve returned no x solution.")

    x_val = np.asarray(x.value, dtype=np.float64).reshape(-1)
    x_val = np.nan_to_num(x_val, nan=0.0, posinf=1.0, neginf=0.0)
    x_val = np.clip(x_val, 0.0, 1.0)
    return x_val


def sdp_greedy_rounding_complete(
    adj: np.ndarray,
    m_target: int,
    *,
    solver: str = "SCS",
    max_iters: int = 10_000,
    eps: float = 1e-5,
    verbose: bool = False,
) -> np.ndarray:
    """
    Paper-faithful relaxed SDP solve + greedy rounding.

    Solve:
      maximize h
      s.t. L0 + sum_e x_e b_e b_e^T - h * (I - 11^T/n) >= 0
           sum_e x_e = k, 0 <= x_e <= 1
    then round by selecting top-k values of x.
    """
    k_add = int(m_target - edge_count(adj))
    if k_add <= 0:
        return adj.copy()

    cand = non_edges(adj)
    p = len(cand)
    if p == 0:
        return adj.copy()
    if k_add >= p:
        out = adj.copy()
        for uu, vv in cand:
            out[uu, vv] = True
            out[vv, uu] = True
        return out

    candidate_uv = np.asarray(cand, dtype=np.int64)
    x_val = _solve_relaxed_sdp_x(
        adj,
        candidate_uv,
        k_add,
        solver=solver,
        max_iters=max_iters,
        eps=eps,
        verbose=verbose,
    )
    idx = _round_top_k_indices(x_val, k_add)
    return _apply_candidate_edge_indices(adj, candidate_uv, idx)


def sdp_step_rounding_complete(
    adj: np.ndarray,
    m_target: int,
    *,
    solver: str = "SCS",
    max_iters: int = 10_000,
    eps: float = 1e-5,
    verbose: bool = False,
) -> np.ndarray:
    """
    Paper-faithful SDP step-by-step rounding:
    solve relaxed SDP, add one edge with largest x, and repeat.
    """
    work = adj.copy()
    while edge_count(work) < m_target:
        cand = non_edges(work)
        p = len(cand)
        if p == 0:
            break
        k_rem = int(m_target - edge_count(work))
        if k_rem <= 0:
            break
        candidate_uv = np.asarray(cand, dtype=np.int64)
        if k_rem >= p:
            for uu, vv in cand:
                work[uu, vv] = True
                work[vv, uu] = True
            break

        x_val = _solve_relaxed_sdp_x(
            work,
            candidate_uv,
            k_rem,
            solver=solver,
            max_iters=max_iters,
            eps=eps,
            verbose=verbose,
        )
        idx = int(np.argmax(x_val))
        uu, vv = int(candidate_uv[idx, 0]), int(candidate_uv[idx, 1])
        work[uu, vv] = True
        work[vv, uu] = True
    return work


def wts_complete(
    adj: np.ndarray,
    m_target: int,
    *,
    seed: int = 0,
    iterations: int = 20,
    tabu_tenure: int = 16,
    max_old_edges_per_iter: int = 8,
    neighbor_sample_per_old: int = 3,
    random_jump_per_old: int = 1,
    init_mode: str = "fvg",
) -> np.ndarray:
    """
    Weighted Tabu Search-style completion for unweighted adjacency.

    This solver optimizes the set of k residual added edges jointly by one-edge
    replacement moves with tabu memory and aspiration.
    """
    k_add = int(m_target - edge_count(adj))
    if k_add <= 0:
        return adj.copy()

    cand = non_edges(adj)
    p = len(cand)
    if p == 0:
        return adj.copy()
    candidate_uv = np.asarray(cand, dtype=np.int64)
    if k_add >= p:
        out = adj.copy()
        out[candidate_uv[:, 0], candidate_uv[:, 1]] = True
        out[candidate_uv[:, 1], candidate_uv[:, 0]] = True
        return out

    n = int(adj.shape[0])
    rng = np.random.default_rng(int(seed))

    mode = str(init_mode).strip().lower()
    if mode not in ("fvg", "random"):
        raise ValueError(f"Unsupported wts init_mode: {init_mode}. Use fvg|random.")

    if mode == "fvg":
        selected = _fiedler_seed_candidate_indices(adj, candidate_uv, k_add)
        if len(selected) < k_add:
            remaining = np.setdiff1d(np.arange(p, dtype=np.int64), np.asarray(selected, dtype=np.int64))
            extra = rng.choice(remaining, size=k_add - len(selected), replace=False)
            selected.extend(int(x) for x in extra.tolist())
    else:
        selected = [int(x) for x in rng.choice(p, size=k_add, replace=False).tolist()]

    work = adj.copy()
    if selected:
        sel_uv0 = candidate_uv[np.asarray(selected, dtype=np.int64)]
        work[sel_uv0[:, 0], sel_uv0[:, 1]] = True
        work[sel_uv0[:, 1], sel_uv0[:, 0]] = True

    # Candidate-edge incidence map for local neighborhood proposals.
    incident: List[List[int]] = [[] for _ in range(n)]
    for idx in range(p):
        u, v = int(candidate_uv[idx, 0]), int(candidate_uv[idx, 1])
        incident[u].append(idx)
        incident[v].append(idx)

    selected_set = set(selected)
    idx_to_pos = {idx: pos for pos, idx in enumerate(selected)}
    current_lam2 = float(algebraic_connectivity(work))
    best_lam2 = current_lam2
    best_selected = selected.copy()

    tabu_q: deque[Tuple[int, int]] = deque()
    tabu_s: set[Tuple[int, int]] = set()
    tenure = max(1, int(tabu_tenure))
    tol = 1e-12

    for _ in range(max(0, int(iterations))):
        if not selected:
            break
        old_count = min(len(selected), max(1, int(max_old_edges_per_iter)))
        if old_count == len(selected):
            old_candidates = selected.copy()
        else:
            old_candidates = [
                int(x)
                for x in rng.choice(np.asarray(selected, dtype=np.int64), size=old_count, replace=False).tolist()
            ]

        best_move: Tuple[int, int] | None = None
        best_move_lam2 = -np.inf

        for old_idx in old_candidates:
            u_old, v_old = int(candidate_uv[old_idx, 0]), int(candidate_uv[old_idx, 1])
            local_pool: set[int] = set()
            for node in (u_old, v_old):
                nbrs = incident[node]
                if not nbrs:
                    continue
                take = min(len(nbrs), max(1, int(neighbor_sample_per_old)))
                if take == len(nbrs):
                    sampled = nbrs
                else:
                    sampled = rng.choice(
                        np.asarray(nbrs, dtype=np.int64),
                        size=take,
                        replace=False,
                    ).tolist()
                for idx in sampled:
                    local_pool.add(int(idx))

            for _ in range(max(0, int(random_jump_per_old))):
                local_pool.add(int(rng.integers(0, p)))

            for new_idx in local_pool:
                if new_idx == old_idx or new_idx in selected_set:
                    continue
                move = (old_idx, new_idx)

                u_new, v_new = int(candidate_uv[new_idx, 0]), int(candidate_uv[new_idx, 1])
                work[u_old, v_old] = False
                work[v_old, u_old] = False
                work[u_new, v_new] = True
                work[v_new, u_new] = True
                lam2 = float(algebraic_connectivity(work))
                work[u_new, v_new] = False
                work[v_new, u_new] = False
                work[u_old, v_old] = True
                work[v_old, u_old] = True

                is_tabu = move in tabu_s
                if is_tabu and lam2 <= (best_lam2 + tol):
                    continue
                if lam2 > (best_move_lam2 + tol):
                    best_move = move
                    best_move_lam2 = lam2

        if best_move is None:
            break

        old_idx, new_idx = best_move
        u_old, v_old = int(candidate_uv[old_idx, 0]), int(candidate_uv[old_idx, 1])
        u_new, v_new = int(candidate_uv[new_idx, 0]), int(candidate_uv[new_idx, 1])
        work[u_old, v_old] = False
        work[v_old, u_old] = False
        work[u_new, v_new] = True
        work[v_new, u_new] = True

        pos = idx_to_pos.pop(old_idx)
        selected[pos] = new_idx
        idx_to_pos[new_idx] = pos
        selected_set.remove(old_idx)
        selected_set.add(new_idx)
        current_lam2 = best_move_lam2
        if current_lam2 > (best_lam2 + tol):
            best_lam2 = current_lam2
            best_selected = selected.copy()

        reverse_move = (new_idx, old_idx)
        if len(tabu_q) >= tenure:
            expired = tabu_q.popleft()
            tabu_s.discard(expired)
        tabu_q.append(reverse_move)
        tabu_s.add(reverse_move)

    out = adj.copy()
    if best_selected:
        best_uv = candidate_uv[np.asarray(best_selected, dtype=np.int64)]
        out[best_uv[:, 0], best_uv[:, 1]] = True
        out[best_uv[:, 1], best_uv[:, 0]] = True
    return out


def random_complete(adj: np.ndarray, m_target: int, rng: np.random.Generator) -> np.ndarray:
    work = adj.copy()
    while edge_count(work) < m_target:
        non = non_edges(work)
        if not non:
            break
        idx = int(rng.integers(0, len(non)))
        u, v = non[idx]
        work[u, v] = True
        work[v, u] = True
    return work


def evaluate_constructor(
    constructor,
    instances: Sequence[Tuple[int, int]],
    init_graph_builder,
) -> Dict[Tuple[int, int], float]:
    out: Dict[Tuple[int, int], float] = {}
    for n, m in instances:
        init_adj = init_graph_builder(n, m)
        final = constructor(init_adj, m)
        out[(n, m)] = algebraic_connectivity(final)
    return out
