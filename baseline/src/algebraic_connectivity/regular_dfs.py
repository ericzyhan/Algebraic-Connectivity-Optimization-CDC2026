from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Sequence, Tuple

import numpy as np

from .graph_core import complement_graph, is_connected
from .spectral import algebraic_connectivity, laplacian


@dataclass
class RegularDFSConfig:
    depth_ratio: float = 0.7
    use_graphic_prune: bool = True
    use_bound_prune: bool = True
    use_complement_for_dense: bool = True
    collect_all_optima: bool = False
    time_limit_s: float | None = None
    max_nodes: int | None = None
    improve_tol: float = 1e-12


@dataclass
class RegularDFSResult:
    n: int
    k: int
    mode: str
    best_score: float
    best_adj: np.ndarray | None
    elapsed_s: float
    visited_nodes: int
    leaves: int
    feasible_leaves: int
    pruned_degree: int
    pruned_graphic: int
    pruned_bound: int
    stopped_by_time: bool
    stopped_by_nodes: bool
    num_optima: int
    all_optima: List[np.ndarray] | None


def _largest_laplacian_eig(adj: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(laplacian(adj))
    if vals.size == 0:
        return 0.0
    return float(vals[-1])


def _is_graphical_erdos_gallai(deg: Sequence[int]) -> bool:
    d = np.asarray([int(x) for x in deg], dtype=np.int64)
    if d.size == 0:
        return True
    if np.any(d < 0):
        return False
    s = int(d.sum())
    if s % 2 != 0:
        return False
    d = np.sort(d)[::-1]
    n = int(d.size)
    if d[0] >= n:
        return False
    csum = np.zeros(n + 1, dtype=np.int64)
    csum[1:] = np.cumsum(d)
    for k in range(1, n + 1):
        left = int(csum[k])
        right = k * (k - 1)
        if k < n:
            dk = int(d[k - 1])
            j = np.searchsorted(-d, -k, side="right")
            j = max(j, k)
            right += int((j - k) * k)
            if j < n:
                right += int(csum[n] - csum[j])
        if left > right:
            return False
    return True


def _fixed_anchor_setup(n: int, degree: int) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Build anchored search setup:
    - Vertex 0 is forced adjacent to vertices 1..degree.
    - Edges (0, degree+1..n-1) are forced absent.
    - Search variables are edges among vertices 1..n-1.
    """
    base = np.zeros((n, n), dtype=bool)
    for v in range(1, degree + 1):
        base[0, v] = True
        base[v, 0] = True

    target = np.full(n - 1, int(degree), dtype=np.int64)
    if degree > 0:
        target[:degree] -= 1

    edges: List[Tuple[int, int]] = []
    for u in range(1, n):
        for v in range(u + 1, n):
            edges.append((u, v))
    return base, target, edges


def _search_regular(
    n: int,
    degree: int,
    objective_mode: str,
    cfg: RegularDFSConfig,
) -> RegularDFSResult:
    if objective_mode not in ("lambda2", "n_minus_lmax"):
        raise ValueError(f"Unsupported objective_mode: {objective_mode}")

    base, target_sub, edge_list = _fixed_anchor_setup(n, degree)
    m = len(edge_list)
    bound_depth = int(max(0, min(m, round(float(cfg.depth_ratio) * float(m)))))

    lower = base.copy()
    upper = base.copy()
    # Undecided subgraph starts as complete on vertices 1..n-1.
    upper[1:, 1:] = True
    np.fill_diagonal(upper, False)

    sub_n = n - 1
    deg_low = np.zeros(sub_n, dtype=np.int64)
    deg_up = np.full(sub_n, sub_n - 1, dtype=np.int64)
    best = -np.inf
    best_adj: np.ndarray | None = None
    all_optima: List[np.ndarray] = []

    visited_nodes = 0
    leaves = 0
    feasible_leaves = 0
    pruned_degree = 0
    pruned_graphic = 0
    pruned_bound = 0
    stopped_by_time = False
    stopped_by_nodes = False
    t0 = time.perf_counter()

    def objective(adj: np.ndarray) -> float:
        if objective_mode == "lambda2":
            return float(algebraic_connectivity(adj))
        return float(n - _largest_laplacian_eig(adj))

    def ub_from_state(depth: int) -> float:
        if objective_mode == "lambda2":
            return float(algebraic_connectivity(upper))
        # For complement-mode objective, descendants are supergraphs of lower.
        # Since lambda_max is monotone nondecreasing by edge addition,
        # n-lambda_max(lower) is an upper bound of descendants' objective.
        return float(n - _largest_laplacian_eig(lower))

    def should_stop() -> bool:
        nonlocal stopped_by_time, stopped_by_nodes
        if cfg.time_limit_s is not None:
            if (time.perf_counter() - t0) >= float(cfg.time_limit_s):
                stopped_by_time = True
                return True
        if cfg.max_nodes is not None:
            if visited_nodes >= int(cfg.max_nodes):
                stopped_by_nodes = True
                return True
        return False

    def dfs(depth: int) -> None:
        nonlocal visited_nodes, leaves, feasible_leaves
        nonlocal pruned_degree, pruned_graphic, pruned_bound
        nonlocal best, best_adj, all_optima

        if should_stop():
            return
        visited_nodes += 1

        # Degree feasibility for exact target degrees.
        if np.any(deg_low > target_sub) or np.any(deg_up < target_sub):
            pruned_degree += 1
            return
        deficit = target_sub - deg_low
        rem_cap = deg_up - deg_low
        if np.any(deficit > rem_cap):
            pruned_degree += 1
            return
        rem_edges = int(m - depth)
        dsum = int(deficit.sum())
        if (dsum % 2) != 0 or dsum > 2 * rem_edges:
            pruned_degree += 1
            return

        if cfg.use_graphic_prune:
            if not _is_graphical_erdos_gallai(deficit.tolist()):
                pruned_graphic += 1
                return

        if cfg.use_bound_prune and depth >= bound_depth and np.isfinite(best):
            ub = ub_from_state(depth)
            if ub <= (best + float(cfg.improve_tol)):
                pruned_bound += 1
                return

        if depth == m:
            leaves += 1
            if np.all(deficit == 0):
                feasible_leaves += 1
                sc = objective(lower)
                if sc > (best + float(cfg.improve_tol)):
                    best = float(sc)
                    best_adj = lower.copy()
                    if cfg.collect_all_optima:
                        all_optima = [best_adj.copy()]
                elif cfg.collect_all_optima and abs(sc - best) <= float(cfg.improve_tol):
                    all_optima.append(lower.copy())
            return

        u, v = edge_list[depth]
        us = u - 1
        vs = v - 1

        # Include branch.
        if deg_low[us] < target_sub[us] and deg_low[vs] < target_sub[vs]:
            lower[u, v] = True
            lower[v, u] = True
            deg_low[us] += 1
            deg_low[vs] += 1
            dfs(depth + 1)
            deg_low[us] -= 1
            deg_low[vs] -= 1
            lower[u, v] = False
            lower[v, u] = False
            if should_stop():
                return

        # Exclude branch.
        if (deg_up[us] - 1) >= target_sub[us] and (deg_up[vs] - 1) >= target_sub[vs]:
            upper[u, v] = False
            upper[v, u] = False
            deg_up[us] -= 1
            deg_up[vs] -= 1
            dfs(depth + 1)
            deg_up[us] += 1
            deg_up[vs] += 1
            upper[u, v] = True
            upper[v, u] = True

    dfs(0)
    elapsed = float(time.perf_counter() - t0)
    return RegularDFSResult(
        n=int(n),
        k=int(degree),
        mode=str(objective_mode),
        best_score=float(best if np.isfinite(best) else 0.0),
        best_adj=None if best_adj is None else best_adj.copy(),
        elapsed_s=elapsed,
        visited_nodes=int(visited_nodes),
        leaves=int(leaves),
        feasible_leaves=int(feasible_leaves),
        pruned_degree=int(pruned_degree),
        pruned_graphic=int(pruned_graphic),
        pruned_bound=int(pruned_bound),
        stopped_by_time=bool(stopped_by_time),
        stopped_by_nodes=bool(stopped_by_nodes),
        num_optima=int(len(all_optima) if cfg.collect_all_optima else (1 if best_adj is not None else 0)),
        all_optima=[x.copy() for x in all_optima] if cfg.collect_all_optima else None,
    )


def solve_acm_regular_graph(
    n: int,
    k: int,
    *,
    cfg: RegularDFSConfig | None = None,
) -> RegularDFSResult:
    """
    Solve (small-scale) ACM regular graph search for fixed (n,k) using DFS+pruning.

    If cfg.use_complement_for_dense and k > n/2, this searches complement degree
    q = n-k-1 with objective n-lambda_max(·), then returns complement graph.
    """
    n = int(n)
    k = int(k)
    c = cfg if cfg is not None else RegularDFSConfig()
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if k < 0 or k >= n:
        raise ValueError(f"k must be in [0,n-1], got n={n}, k={k}")
    if (n * k) % 2 != 0:
        raise ValueError(f"n*k must be even for k-regular graphs, got n={n}, k={k}")

    # Trivial cases.
    if k == 0:
        adj = np.zeros((n, n), dtype=bool)
        return RegularDFSResult(
            n=n,
            k=k,
            mode="direct",
            best_score=0.0,
            best_adj=adj,
            elapsed_s=0.0,
            visited_nodes=0,
            leaves=1,
            feasible_leaves=1,
            pruned_degree=0,
            pruned_graphic=0,
            pruned_bound=0,
            stopped_by_time=False,
            stopped_by_nodes=False,
            num_optima=1,
            all_optima=[adj.copy()] if c.collect_all_optima else None,
        )
    if k == n - 1:
        adj = np.ones((n, n), dtype=bool)
        np.fill_diagonal(adj, False)
        return RegularDFSResult(
            n=n,
            k=k,
            mode="direct",
            best_score=float(algebraic_connectivity(adj)),
            best_adj=adj,
            elapsed_s=0.0,
            visited_nodes=0,
            leaves=1,
            feasible_leaves=1,
            pruned_degree=0,
            pruned_graphic=0,
            pruned_bound=0,
            stopped_by_time=False,
            stopped_by_nodes=False,
            num_optima=1,
            all_optima=[adj.copy()] if c.collect_all_optima else None,
        )

    use_complement = bool(c.use_complement_for_dense and (k > (n // 2)))
    if not use_complement:
        res = _search_regular(n=n, degree=k, objective_mode="lambda2", cfg=c)
        res.mode = "direct"
        return res

    q = n - k - 1
    res_c = _search_regular(n=n, degree=q, objective_mode="n_minus_lmax", cfg=c)
    out_adj: np.ndarray | None = None
    out_score = float(res_c.best_score)
    if res_c.best_adj is not None:
        out_adj = complement_graph(res_c.best_adj)
        out_score = float(algebraic_connectivity(out_adj))
    return RegularDFSResult(
        n=n,
        k=k,
        mode="complement",
        best_score=float(out_score),
        best_adj=out_adj,
        elapsed_s=float(res_c.elapsed_s),
        visited_nodes=int(res_c.visited_nodes),
        leaves=int(res_c.leaves),
        feasible_leaves=int(res_c.feasible_leaves),
        pruned_degree=int(res_c.pruned_degree),
        pruned_graphic=int(res_c.pruned_graphic),
        pruned_bound=int(res_c.pruned_bound),
        stopped_by_time=bool(res_c.stopped_by_time),
        stopped_by_nodes=bool(res_c.stopped_by_nodes),
        num_optima=int(res_c.num_optima),
        all_optima=[complement_graph(x) for x in res_c.all_optima] if res_c.all_optima is not None else None,
    )

