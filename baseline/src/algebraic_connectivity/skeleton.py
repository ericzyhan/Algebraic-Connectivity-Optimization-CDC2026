from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from math import cos, gcd, pi, sin, sqrt
from typing import Dict, List, Tuple

import numpy as np

from .graph_core import (
    add_edge,
    complement_graph,
    complete_r_partite_graph,
    edge_count,
    is_connected,
    max_edges,
    multipartite_density_from_sizes,
    multipartite_edge_count_from_sizes,
    multipartite_lambda2_closed_form,
    path_graph,
    target_density,
    validate_nm,
)
from .spectral import algebraic_connectivity


@dataclass
class WarmStartMeta:
    pocket: str
    rho: float
    backbone_degree: int
    residual_edges: int
    init_mode: str


def density_pocket(rho: float, thresholds: Tuple[float, float]) -> str:
    lo, hi = thresholds
    if rho < lo:
        return "sparse"
    if rho < hi:
        return "mid"
    return "dense"


def _rpartite_extremal_sizes_fixed_dominant(n: int, r: int, s: int) -> List[int] | None:
    """
    For fixed (n, r, dominant part size s), construct a complete r-partite split
    minimizing edge count under constraints:
      - remaining r-1 parts sum to n-s
      - each remaining part in [1, s]
    The minimizer is maximally uneven: many s's, then at most one residual part,
    then ones.
    """
    if r < 2 or r > n or s < 1:
        return None

    t = r - 1
    T = n - s
    if T < t or T > t * s:
        return None

    extra = T - t
    step = s - 1

    if step == 0:
        if extra != 0:
            return None
        parts = [1] * t
    else:
        q, rem = divmod(extra, step)
        if q > t:
            return None
        parts = [s] * q
        if rem > 0:
            if len(parts) >= t:
                return None
            parts.append(1 + rem)
        parts.extend([1] * (t - len(parts)))

    sizes = parts + [s]
    if sum(sizes) != n:
        return None
    return sizes


def _rpartite_approx_branch_range(r: int) -> Tuple[float, float]:
    lo = (r - 2) / (r - 1) if r >= 3 else 0.0
    hi = (r - 1) / r
    return float(lo), float(hi)


def _rpartite_acm_delta(sizes: List[int]) -> float:
    n = int(sum(int(x) for x in sizes))
    if n <= 0:
        return 0.0
    s_dom = int(max(int(x) for x in sizes))
    sum_sq = int(sum(int(x) * int(x) for x in sizes))
    return float(s_dom) - (float(sum_sq) / float(n))


def _rpartite_points(
    n: int,
    r_max: int,
    use_approx_filter: bool,
) -> List[Tuple[int, float, float, List[int], int, float, int]]:
    """
    Returns points as tuples:
      (m, rho, lambda2, part_sizes, r, acm_delta, acm_cert)
    """
    pts: List[Tuple[int, float, float, List[int], int, float, int]] = []
    for r in range(2, min(n, max(2, int(r_max))) + 1):
        rho_lo, rho_hi = _rpartite_approx_branch_range(r)
        s_lo = (n + r - 1) // r  # ceil(n/r)
        s_hi = n - (r - 1)       # from n-s >= r-1
        for s in range(max(1, s_lo), max(1, s_hi) + 1):
            sizes = _rpartite_extremal_sizes_fixed_dominant(n, r, s)
            if sizes is None:
                continue
            m = multipartite_edge_count_from_sizes(sizes)
            rho = multipartite_density_from_sizes(sizes)
            if use_approx_filter and not (rho_lo - 1e-12 <= rho <= rho_hi + 1e-12):
                continue
            lam = multipartite_lambda2_closed_form(sizes)
            acm_delta = _rpartite_acm_delta(sizes)
            acm_cert = int(acm_delta < 1.0)
            pts.append((int(m), float(rho), float(lam), sizes, int(r), float(acm_delta), int(acm_cert)))
    pts.sort(key=lambda x: (x[0], -x[2], x[4]))
    return pts


def _rpartite_upper_envelope(
    pts: List[Tuple[int, float, float, List[int], int, float, int]]
) -> List[Tuple[int, float, float, List[int], int, float, int]]:
    """
    Keep points that improve running max lambda2 as m increases.
    """
    env: List[Tuple[int, float, float, List[int], int, float, int]] = []
    best = -1e100
    for p in pts:
        lam = p[2]
        if lam > best + 1e-12:
            env.append(p)
            best = lam
    return env


@lru_cache(maxsize=256)
def _rpartite_envelope_cache(
    n: int,
    r_max: int,
    use_approx_filter: bool,
) -> Tuple[
    Tuple[Tuple[int, float, float, Tuple[int, ...], int, float, int], ...],
    Tuple[int, ...],
    Tuple[Tuple[int, float, float, Tuple[int, ...], int, float, int], ...],
    Tuple[int, ...],
]:
    pts = _rpartite_points(n=n, r_max=r_max, use_approx_filter=use_approx_filter)
    env = _rpartite_upper_envelope(pts)

    env_t = tuple(
        (
            int(p[0]),
            float(p[1]),
            float(p[2]),
            tuple(int(x) for x in p[3]),
            int(p[4]),
            float(p[5]),
            int(p[6]),
        )
        for p in env
    )
    env_m = tuple(int(p[0]) for p in env_t)
    cert_t = tuple(p for p in env_t if int(p[6]) == 1)
    cert_m = tuple(int(p[0]) for p in cert_t)
    return env_t, env_m, cert_t, cert_m


def deterministic_warm_start_rpartite_envelope(
    n: int,
    m: int,
    thresholds: Tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
    r_max: int = 16,
    envelope_rank: int = 0,
    use_approx_filter: bool = True,
    require_acm_cert: bool = False,
    allow_uncertified_fallback: bool = True,
) -> Tuple[np.ndarray, WarmStartMeta]:
    """
    Warm start from complete r-partite upper-envelope:
    1) build envelope points (m_env, lambda2_env) with m_env <= target m,
    2) choose closest predecessor by m (rank=0), or next predecessor (rank=1), ...
    3) use that complete r-partite graph as initialization.

    If approximate-branch filtering produces no usable point, retries without it.
    """
    validate_nm(n, m)
    rho = target_density(n, m)
    pocket = density_pocket(rho, thresholds)

    if m == n - 1:
        adj = path_graph(n)
        return (
            adj,
            WarmStartMeta(
                pocket=pocket,
                rho=rho,
                backbone_degree=1,
                residual_edges=0,
                init_mode="path",
            ),
        )

    rank = max(0, int(envelope_rank))

    def _select_pick(
        env_pts: Tuple[Tuple[int, float, float, Tuple[int, ...], int, float, int], ...],
        env_m: Tuple[int, ...],
        cert_pts: Tuple[Tuple[int, float, float, Tuple[int, ...], int, float, int], ...],
        cert_m: Tuple[int, ...],
    ) -> Tuple[Tuple[int, float, float, Tuple[int, ...], int, float, int] | None, str]:
        if not env_pts:
            return None, "empty_env"

        hi = bisect_right(env_m, int(m)) - 1  # predecessor in all envelope points
        if hi < 0:
            return None, "no_predecessor"

        if not require_acm_cert:
            idx = hi - rank
            if idx >= 0:
                return env_pts[idx], "all"
            return None, "all_rank_oob"

        if cert_pts:
            cert_hi = bisect_right(cert_m, int(m)) - 1
            cert_idx = cert_hi - rank
            if cert_idx >= 0:
                return cert_pts[cert_idx], "cert"
            return None, "cert_rank_oob"

        if allow_uncertified_fallback:
            idx = hi - rank
            if idx >= 0:
                return env_pts[idx], "fallback_uncert"
            return None, "fallback_rank_oob"
        return None, "cert_only_no_candidate"

    env, env_m, cert_pts, cert_m = _rpartite_envelope_cache(
        n=int(n), r_max=int(r_max), use_approx_filter=bool(use_approx_filter)
    )
    pick, cand_tag = _select_pick(env, env_m, cert_pts, cert_m)

    # Fallback: disable filter if current filtered branch produced no predecessor.
    if pick is None and use_approx_filter:
        env, env_m, cert_pts, cert_m = _rpartite_envelope_cache(
            n=int(n), r_max=int(r_max), use_approx_filter=False
        )
        pick, cand_tag = _select_pick(env, env_m, cert_pts, cert_m)

    if pick is None:
        # Defensive fallback.
        adj = path_graph(n)
        res = m - edge_count(adj)
        return (
            adj,
            WarmStartMeta(
                pocket=pocket,
                rho=rho,
                backbone_degree=1,
                residual_edges=res,
                init_mode="path_fallback_rpartite_empty",
            ),
        )

    m0, _rho0, _lam0, sizes_t, r_sel, _acm_delta, acm_cert = pick
    sizes = list(int(x) for x in sizes_t)

    try:
        adj = complete_r_partite_graph(sizes)
        if edge_count(adj) != m0:
            raise RuntimeError("constructed edge count mismatch")
    except Exception:
        adj = path_graph(n)
        res = m - edge_count(adj)
        return (
            adj,
            WarmStartMeta(
                pocket=pocket,
                rho=rho,
                backbone_degree=1,
                residual_edges=res,
                init_mode="path_fallback_rpartite_builder_error",
            ),
        )

    if not is_connected(adj):
        adj = path_graph(n)
        res = m - edge_count(adj)
        return (
            adj,
            WarmStartMeta(
                pocket=pocket,
                rho=rho,
                backbone_degree=1,
                residual_edges=res,
                init_mode="path_fallback_rpartite_disconnected",
            ),
        )

    res = m - edge_count(adj)
    if res < 0:
        adj = path_graph(n)
        res = m - edge_count(adj)
        return (
            adj,
            WarmStartMeta(
                pocket=pocket,
                rho=rho,
                backbone_degree=1,
                residual_edges=res,
                init_mode="path_fallback_rpartite_negative_residual",
            ),
        )

    k_est = int(np.floor((2.0 * edge_count(adj)) / n))
    mode_tag = "approx" if use_approx_filter else "full"
    cert_tag = "cert" if int(acm_cert) == 1 else "uncert"
    return (
        adj,
        WarmStartMeta(
            pocket=pocket,
            rho=rho,
            backbone_degree=max(1, k_est),
            residual_edges=res,
            init_mode=f"rpartite_env_r{r_sel}_rank{rank}_{mode_tag}_{cand_tag}_{cert_tag}",
        ),
    )


def deterministic_warm_start(
    n: int,
    m: int,
    thresholds: Tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
) -> Tuple[np.ndarray, WarmStartMeta]:
    """
    Deterministic phase-1 warm start by density pocket:
    - sparse/mid: circulant-style near-regular backbone
    - dense: build sparse missing-edge regular graph, then complement
    """
    validate_nm(n, m)
    rho = target_density(n, m)
    pocket = density_pocket(rho, thresholds)

    if m == n - 1:
        adj = path_graph(n)
        meta = WarmStartMeta(
            pocket=pocket,
            rho=rho,
            backbone_degree=2 if n > 2 else 1,
            residual_edges=0,
            init_mode="path",
        )
        return adj, meta

    k_raw = int((2 * m) // n)
    k = min(k_raw, n - 1)
    if (n * k) % 2 == 1:
        k -= 1
    if k < 2:
        adj = path_graph(n)
        res = m - edge_count(adj)
        meta = WarmStartMeta(
            pocket=pocket,
            rho=rho,
            backbone_degree=1,
            residual_edges=res,
            init_mode="path",
        )
        return adj, meta

    try:
        if pocket == "dense":
            q = n - 1 - k
            missing = _build_regular_circulant(n, q, pocket="sparse")
            adj = complement_graph(missing)
            init_mode = "dense_complement"
        else:
            adj = _build_regular_circulant(n, k, pocket=pocket)
            init_mode = f"{pocket}_circulant"
    except Exception:
        adj = path_graph(n)
        init_mode = "path_fallback_builder_error"
        k = 1

    if not is_connected(adj):
        # Conservative fallback for guaranteed connectivity.
        adj = path_graph(n)
        init_mode = "path_fallback"
        k = 1

    res = m - edge_count(adj)
    if res < 0:
        # Extremely defensive guard; should not happen with k=floor(2m/n).
        adj = path_graph(n)
        init_mode = "path_fallback_negative_residual"
        res = m - edge_count(adj)
        k = 1

    meta = WarmStartMeta(
        pocket=pocket,
        rho=rho,
        backbone_degree=k,
        residual_edges=res,
        init_mode=init_mode,
    )
    return adj, meta


def deterministic_warm_start_expander(
    n: int,
    m: int,
    thresholds: Tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
    comm_penalty_scale: float = 1.0,
) -> Tuple[np.ndarray, WarmStartMeta]:
    """
    Deterministic warm start guided by:
    - geometric spacing (maximize chordal distance on ring)
    - cycle resonance suppression (penalize commensurate short offsets)
    - near-regular degree targets

    Complexity target: O(n^2) candidate generation + sorting with linear passes.
    """
    validate_nm(n, m)
    rho = target_density(n, m)
    pocket = density_pocket(rho, thresholds)

    if m == n - 1:
        adj = path_graph(n)
        return (
            adj,
            WarmStartMeta(
                pocket=pocket,
                rho=rho,
                backbone_degree=1,
                residual_edges=0,
                init_mode="path",
            ),
        )

    k_raw = int((2 * m) // n)
    k = min(k_raw, n - 1)
    if (n * k) % 2 == 1:
        k -= 1
    if k < 2:
        adj = path_graph(n)
        res = m - edge_count(adj)
        return (
            adj,
            WarmStartMeta(
                pocket=pocket,
                rho=rho,
                backbone_degree=1,
                residual_edges=res,
                init_mode="path",
            ),
        )

    e_target = n * k // 2
    if pocket == "dense":
        e_missing = max_edges(n) - e_target
        missing = _build_expander_like_graph(
            n=n,
            e_target=e_missing,
            pocket="sparse",
            require_connected=False,
            comm_penalty_scale=comm_penalty_scale,
        )
        adj = complement_graph(missing)
        init_mode = "dense_expander_complement"
    else:
        adj = _build_expander_like_graph(
            n=n,
            e_target=e_target,
            pocket=pocket,
            require_connected=True,
            comm_penalty_scale=comm_penalty_scale,
        )
        init_mode = f"{pocket}_expander"

    if not is_connected(adj):
        adj = path_graph(n)
        init_mode = "path_fallback"
        k = 1

    res = m - edge_count(adj)
    if res < 0:
        adj = path_graph(n)
        init_mode = "path_fallback_negative_residual"
        res = m - edge_count(adj)
        k = 1

    return (
        adj,
        WarmStartMeta(
            pocket=pocket,
            rho=rho,
            backbone_degree=k,
            residual_edges=res,
            init_mode=init_mode,
        ),
    )


def deterministic_warm_start_smallworld(
    n: int,
    m: int,
    thresholds: Tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
    num_samples: int = 1,
) -> Tuple[np.ndarray, WarmStartMeta]:
    """
    Deterministic small-world-like warm start:
    - base ring-lattice connectivity with short-range neighborhoods
    - deterministic rewiring for occasional long-range shortcuts
    - exact edge-budget fill with near-regular preference

    Dense pocket uses complement of a sparse small-world missing-edge graph.
    """
    validate_nm(n, m)
    rho = target_density(n, m)
    pocket = density_pocket(rho, thresholds)
    num_samples = max(1, int(num_samples))

    if m == n - 1:
        adj = path_graph(n)
        return (
            adj,
            WarmStartMeta(
                pocket=pocket,
                rho=rho,
                backbone_degree=1,
                residual_edges=0,
                init_mode="path",
            ),
        )

    k_raw = int((2 * m) // n)
    k = min(k_raw, n - 1)
    if (n * k) % 2 == 1:
        k -= 1
    if k < 2:
        adj = path_graph(n)
        res = m - edge_count(adj)
        return (
            adj,
            WarmStartMeta(
                pocket=pocket,
                rho=rho,
                backbone_degree=1,
                residual_edges=res,
                init_mode="path",
            ),
        )

    e_target = n * k // 2
    if pocket == "dense":
        e_missing = max_edges(n) - e_target
        adj = _best_smallworld_dense_by_multistart(
            n=n,
            e_missing=e_missing,
            num_samples=num_samples,
        )
        init_mode = "dense_smallworld_complement"
    else:
        adj = _best_smallworld_sparsemid_by_multistart(
            n=n,
            e_target=e_target,
            pocket=pocket,
            num_samples=num_samples,
        )
        init_mode = f"{pocket}_smallworld"

    if not is_connected(adj):
        adj = path_graph(n)
        init_mode = "path_fallback"
        k = 1

    res = m - edge_count(adj)
    if res < 0:
        adj = path_graph(n)
        init_mode = "path_fallback_negative_residual"
        res = m - edge_count(adj)
        k = 1

    return (
        adj,
        WarmStartMeta(
            pocket=pocket,
            rho=rho,
            backbone_degree=k,
            residual_edges=res,
            init_mode=init_mode,
        ),
    )


def deterministic_warm_start_fib_hybrid(
    n: int,
    m: int,
    thresholds: Tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
    extra_fraction_by_pocket: Dict[str, float] | None = None,
    fib_geo_weight_scale: float = 1.0,
    fib_comm_penalty_scale: float = 1.0,
    fib_geo_weight_override: float | None = None,
    fib_comm_weight_override: float | None = None,
) -> Tuple[np.ndarray, WarmStartMeta]:
    """
    Deterministic warm start with conservative edge commitment:
    - start from connected path
    - add only a small fraction of residual edges
    - choose added edges by Fibonacci-sphere geometric spacing + degree balancing
      while discouraging early triangle overuse
    """
    validate_nm(n, m)
    rho = target_density(n, m)
    pocket = density_pocket(rho, thresholds)

    base_adj = path_graph(n)
    base_edges = edge_count(base_adj)
    if m <= base_edges:
        return (
            base_adj,
            WarmStartMeta(
                pocket=pocket,
                rho=rho,
                backbone_degree=1,
                residual_edges=0,
                init_mode="path",
            ),
        )

    fractions = {"sparse": 0.20, "mid": 0.18, "dense": 0.10}
    if extra_fraction_by_pocket is not None:
        fractions.update(extra_fraction_by_pocket)
    beta = float(np.clip(fractions.get(pocket, 0.15), 0.0, 1.0))
    residual_total = m - base_edges
    warm_extra = int(round(beta * residual_total))
    target_init_edges = min(m, base_edges + max(0, warm_extra))

    if target_init_edges <= base_edges:
        adj = base_adj
        init_mode = "path"
    else:
        try:
            adj = _build_fib_hybrid_graph(
                n,
                target_init_edges,
                pocket,
                fib_geo_weight_scale=fib_geo_weight_scale,
                fib_comm_penalty_scale=fib_comm_penalty_scale,
                fib_geo_weight_override=fib_geo_weight_override,
                fib_comm_weight_override=fib_comm_weight_override,
            )
            init_mode = f"{pocket}_fib_hybrid"
        except Exception:
            adj = base_adj
            init_mode = "path_fallback_builder_error"

    if not is_connected(adj):
        adj = base_adj
        init_mode = "path_fallback"

    res = m - edge_count(adj)
    if res < 0:
        adj = base_adj
        init_mode = "path_fallback_negative_residual"
        res = m - edge_count(adj)

    k_est = max(1, int((2 * edge_count(adj)) // n))
    return (
        adj,
        WarmStartMeta(
            pocket=pocket,
            rho=rho,
            backbone_degree=k_est,
            residual_edges=res,
            init_mode=init_mode,
        ),
    )


def deterministic_warm_start_fib_s3_hybrid(
    n: int,
    m: int,
    thresholds: Tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
    extra_fraction_by_pocket: Dict[str, float] | None = None,
    fib_geo_weight_scale: float = 1.0,
    fib_comm_penalty_scale: float = 1.0,
    fib_geo_weight_override: float | None = None,
    fib_comm_weight_override: float | None = None,
) -> Tuple[np.ndarray, WarmStartMeta]:
    """
    Deterministic warm start using quasi-uniform points on S^3:
    - start from connected path
    - add a controlled fraction of residual edges
    - rank candidate edges by 4D chordal distance with index-commensurability penalty
    - preserve near-regular degree targets and avoid early triangle overuse
    """
    validate_nm(n, m)
    rho = target_density(n, m)
    pocket = density_pocket(rho, thresholds)

    base_adj = path_graph(n)
    base_edges = edge_count(base_adj)
    if m <= base_edges:
        return (
            base_adj,
            WarmStartMeta(
                pocket=pocket,
                rho=rho,
                backbone_degree=1,
                residual_edges=0,
                init_mode="path",
            ),
        )

    fractions = {"sparse": 0.18, "mid": 0.14, "dense": 0.08}
    if extra_fraction_by_pocket is not None:
        fractions.update(extra_fraction_by_pocket)
    beta = float(np.clip(fractions.get(pocket, 0.12), 0.0, 1.0))
    residual_total = m - base_edges
    warm_extra = int(round(beta * residual_total))
    target_init_edges = min(m, base_edges + max(0, warm_extra))

    if target_init_edges <= base_edges:
        adj = base_adj
        init_mode = "path"
    else:
        try:
            adj = _build_fib_s3_hybrid_graph(
                n,
                target_init_edges,
                pocket,
                fib_geo_weight_scale=fib_geo_weight_scale,
                fib_comm_penalty_scale=fib_comm_penalty_scale,
                fib_geo_weight_override=fib_geo_weight_override,
                fib_comm_weight_override=fib_comm_weight_override,
            )
            init_mode = f"{pocket}_fib_s3_hybrid"
        except Exception:
            adj = base_adj
            init_mode = "path_fallback_builder_error"

    if not is_connected(adj):
        adj = base_adj
        init_mode = "path_fallback"

    res = m - edge_count(adj)
    if res < 0:
        adj = base_adj
        init_mode = "path_fallback_negative_residual"
        res = m - edge_count(adj)

    k_est = max(1, int((2 * edge_count(adj)) // n))
    return (
        adj,
        WarmStartMeta(
            pocket=pocket,
            rho=rho,
            backbone_degree=k_est,
            residual_edges=res,
            init_mode=init_mode,
        ),
    )


def _build_regular_circulant(n: int, k: int, pocket: str) -> np.ndarray:
    if k == 0:
        return np.zeros((n, n), dtype=bool)
    if k >= n:
        raise ValueError(f"k must be < n, got n={n}, k={k}")
    if (n * k) % 2 == 1:
        raise ValueError(f"n*k must be even for k-regular graph, got n={n}, k={k}")

    half = n // 2 if (n % 2 == 0) else None
    need_one_degree = (k % 2 == 1)
    offsets: List[int] = []
    rem = k
    if need_one_degree:
        if half is None:
            raise ValueError(f"odd k requires even n in circulant construction: n={n}, k={k}")
        offsets.append(half)
        rem -= 1

    need_pairs = rem // 2
    nonhalf = list(range(1, (n // 2) + 1))
    if half is not None and half in nonhalf:
        nonhalf.remove(half)
    ordered = _order_offsets(nonhalf, pocket)
    if need_pairs > len(ordered):
        raise ValueError(
            f"not enough offsets for n={n}, k={k}, pocket={pocket}, need={need_pairs}"
        )
    offsets.extend(ordered[:need_pairs])
    adj = _circulant_from_offsets(n, offsets)
    deg = adj.sum(axis=1).astype(np.int64)
    e = edge_count(adj)
    expected_e = n * k // 2
    if int(deg.min()) != k or int(deg.max()) != k or e != expected_e:
        raise RuntimeError(
            f"invalid circulant build for n={n}, k={k}, pocket={pocket}: "
            f"deg_range=({int(deg.min())},{int(deg.max())}), edges={e}, expected_edges={expected_e}"
        )
    return adj


def _order_offsets(cands: List[int], pocket: str) -> List[int]:
    if not cands:
        return []
    cands = sorted(set(cands))
    if pocket == "sparse":
        if 1 in cands:
            return [1] + sorted([x for x in cands if x != 1], reverse=True)
        return sorted(cands, reverse=True)
    if pocket == "mid":
        # Greedy max-min spread over offset space, anchored at 1 for connectivity.
        pool = cands.copy()
        out: List[int] = []
        if 1 in pool:
            out.append(1)
            pool.remove(1)
        while pool:
            if not out:
                pick = pool.pop(len(pool) // 2)
                out.append(pick)
                continue
            best = max(pool, key=lambda x: min(abs(x - y) for y in out))
            out.append(best)
            pool.remove(best)
        return out
    # dense is primarily handled by complement trick; fall back to sparse-like ordering.
    return [1] + sorted([x for x in cands if x != 1], reverse=True) if 1 in cands else sorted(
        cands, reverse=True
    )


def _circulant_from_offsets(n: int, offsets: List[int]) -> np.ndarray:
    adj = np.zeros((n, n), dtype=bool)
    for s in offsets:
        if s == 0:
            continue
        if n % 2 == 0 and s == n // 2:
            for i in range(n // 2):
                add_edge(adj, i, i + s)
            continue
        # For non-half offsets, add all wrap-around edges exactly once.
        seen = set()
        for i in range(n):
            j = (i + s) % n
            u, v = (i, j) if i < j else (j, i)
            if u == v or (u, v) in seen:
                continue
            seen.add((u, v))
            add_edge(adj, u, v)
    return adj


def _build_expander_like_graph(
    n: int,
    e_target: int,
    pocket: str,
    require_connected: bool,
    comm_penalty_scale: float = 1.0,
) -> np.ndarray:
    if not (0 <= e_target <= max_edges(n)):
        raise ValueError(f"invalid edge target e={e_target} for n={n}")

    # Connected scaffold to avoid disconnected warm starts in sparse/mid.
    if require_connected:
        if e_target < n - 1:
            raise ValueError(f"connected scaffold requires e_target >= n-1, got {e_target}")
        if e_target == n - 1:
            return path_graph(n)
        adj = _cycle_graph(n)
    else:
        adj = np.zeros((n, n), dtype=bool)

    e_cur = edge_count(adj)
    if e_cur >= e_target:
        return adj

    target_deg = _near_regular_degree_targets(n, e_target)
    deg = adj.sum(axis=1).astype(np.int64)
    scored = _scored_pairs(n, pocket, comm_penalty_scale=comm_penalty_scale)

    # Phase 1: enforce near-regular target strongly.
    for mode in (0, 1, 2):
        if e_cur >= e_target:
            break
        for _, u, v in scored:
            if adj[u, v]:
                continue
            du, dv = int(deg[u]), int(deg[v])
            tu, tv = int(target_deg[u]), int(target_deg[v])
            if mode == 0 and not (du < tu and dv < tv):
                continue
            if mode == 1 and not (du < tu or dv < tv):
                continue
            add_edge(adj, u, v)
            deg[u] += 1
            deg[v] += 1
            e_cur += 1
            if e_cur >= e_target:
                break

    # Final defensive fill by score order.
    if e_cur < e_target:
        for _, u, v in scored:
            if adj[u, v]:
                continue
            add_edge(adj, u, v)
            e_cur += 1
            if e_cur >= e_target:
                break

    return adj


def _build_smallworld_like_graph(
    n: int,
    e_target: int,
    pocket: str,
    require_connected: bool,
    sample_index: int = 0,
) -> np.ndarray:
    if not (0 <= e_target <= max_edges(n)):
        raise ValueError(f"invalid edge target e={e_target} for n={n}")

    if require_connected:
        if e_target < n - 1:
            raise ValueError(f"connected scaffold requires e_target >= n-1, got {e_target}")
        if e_target == n - 1:
            return path_graph(n)

    # Use a deterministic seed so construction is fully reproducible.
    pocket_id = {"sparse": 11, "mid": 23, "dense": 37}.get(pocket, 17)
    seed = int(
        (
            n * 1000003
            + e_target * 9176
            + pocket_id * 131
            + (1 if require_connected else 0)
            + int(sample_index) * 2654435761
        )
        % (2**32)
    )
    rng = np.random.default_rng(seed)

    k = int((2 * e_target) // n)
    k = min(k, n - 1)
    if k % 2 == 1:
        k -= 1
    if require_connected and k < 2:
        k = 2
    if k >= n:
        k = n - 1 if (n - 1) % 2 == 0 else n - 2

    if k >= 2:
        adj = _ring_lattice_graph(n, k)
        beta_map = {"sparse": 0.22, "mid": 0.14, "dense": 0.08}
        beta = float(beta_map.get(pocket, 0.14))
        _smallworld_rewire(adj, k=k, beta=beta, rng=rng)
    else:
        adj = np.zeros((n, n), dtype=bool)
        if require_connected:
            adj = path_graph(n)

    e_cur = edge_count(adj)
    if e_cur >= e_target:
        return adj

    target_deg = _near_regular_degree_targets(n, e_target)
    deg = adj.sum(axis=1).astype(np.int64)
    scored = _smallworld_scored_pairs(n, pocket)

    # Prefer near-regular endpoints first; then relax.
    for mode in (0, 1, 2):
        if e_cur >= e_target:
            break
        for _, u, v in scored:
            if adj[u, v]:
                continue
            du, dv = int(deg[u]), int(deg[v])
            tu, tv = int(target_deg[u]), int(target_deg[v])
            if mode == 0 and not (du < tu and dv < tv):
                continue
            if mode == 1 and not (du < tu or dv < tv):
                continue
            add_edge(adj, u, v)
            deg[u] += 1
            deg[v] += 1
            e_cur += 1
            if e_cur >= e_target:
                break

    return adj


def _best_smallworld_sparsemid_by_multistart(
    n: int,
    e_target: int,
    pocket: str,
    num_samples: int,
) -> np.ndarray:
    best_adj = None
    best_lam2 = -1.0
    for s in range(num_samples):
        adj = _build_smallworld_like_graph(
            n=n,
            e_target=e_target,
            pocket=pocket,
            require_connected=True,
            sample_index=s,
        )
        lam2 = float(algebraic_connectivity(adj))
        if (best_adj is None) or (lam2 > best_lam2):
            best_adj = adj
            best_lam2 = lam2
    if best_adj is None:
        return path_graph(n)
    return best_adj


def _best_smallworld_dense_by_multistart(
    n: int,
    e_missing: int,
    num_samples: int,
) -> np.ndarray:
    best_adj = None
    best_lam2 = -1.0
    for s in range(num_samples):
        missing = _build_smallworld_like_graph(
            n=n,
            e_target=e_missing,
            pocket="sparse",
            require_connected=False,
            sample_index=s,
        )
        adj = complement_graph(missing)
        lam2 = float(algebraic_connectivity(adj))
        if (best_adj is None) or (lam2 > best_lam2):
            best_adj = adj
            best_lam2 = lam2
    if best_adj is None:
        return path_graph(n)
    return best_adj


def _ring_lattice_graph(n: int, k: int) -> np.ndarray:
    if k % 2 == 1:
        raise ValueError(f"ring lattice requires even k, got {k}")
    if k < 0 or k >= n:
        raise ValueError(f"invalid ring-lattice degree k={k} for n={n}")
    adj = np.zeros((n, n), dtype=bool)
    h = k // 2
    for u in range(n):
        for s in range(1, h + 1):
            v = (u + s) % n
            a, b = (u, v) if u < v else (v, u)
            if not adj[a, b]:
                add_edge(adj, a, b)
    return adj


def _smallworld_rewire(
    adj: np.ndarray,
    k: int,
    beta: float,
    rng: np.random.Generator,
) -> None:
    if beta <= 0.0 or k < 4:
        return
    n = adj.shape[0]
    h = k // 2
    if h <= 1:
        return

    # Keep nearest-neighbor ring edges (shift=1) untouched for robust connectivity.
    candidates: List[Tuple[int, int]] = []
    for u in range(n):
        for s in range(2, h + 1):
            v = (u + s) % n
            a, b = (u, v) if u < v else (v, u)
            if adj[a, b]:
                candidates.append((a, b))

    for a, b in candidates:
        if not adj[a, b]:
            continue
        if float(rng.random()) >= beta:
            continue
        u = a
        v = b
        # Rewire edge (u,v) -> (u,w), preserving edge count.
        adj[u, v] = False
        adj[v, u] = False

        w = -1
        order = rng.permutation(n)
        for cand in order:
            c = int(cand)
            if c == u or adj[u, c]:
                continue
            w = c
            break

        if w < 0:
            # Restore original edge if no feasible rewiring target.
            adj[u, v] = True
            adj[v, u] = True
            continue
        add_edge(adj, u, w)


def _smallworld_scored_pairs(n: int, pocket: str) -> List[Tuple[float, int, int]]:
    if pocket == "sparse":
        w_local, w_long, w_jit = 1.00, 0.08, 0.05
    elif pocket == "mid":
        w_local, w_long, w_jit = 0.86, 0.15, 0.07
    else:
        w_local, w_long, w_jit = 0.76, 0.22, 0.09

    scored: List[Tuple[float, int, int]] = []
    half = max(1.0, (n / 2.0))
    for u in range(n):
        for v in range(u + 1, n):
            d = min(v - u, n - (v - u))
            local = 1.0 / (1.0 + float(d))
            long_range = float(d) / half
            jitter = _det_pair_noise(u, v, n)
            score = (w_local * local) + (w_long * long_range) + (w_jit * jitter)
            scored.append((score, u, v))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    return scored


def _det_pair_noise(u: int, v: int, n: int) -> float:
    x = (u + 1) * 73856093
    y = (v + 1) * 19349663
    z = (n + 1) * 83492791
    mix = (x ^ y ^ z) & 0xFFFFFFFF
    return (float(mix) / 2147483647.5) - 1.0


def _near_regular_degree_targets(n: int, e_target: int) -> np.ndarray:
    total_deg = 2 * e_target
    base = total_deg // n
    rem = total_deg - base * n
    out = np.full(n, base, dtype=np.int64)
    if rem > 0:
        idx = np.floor(np.arange(rem) * n / rem).astype(np.int64)
        out[idx] += 1
    return out


def _scored_pairs(
    n: int,
    pocket: str,
    comm_penalty_scale: float = 1.0,
) -> List[Tuple[float, int, int]]:
    # Weight by density pocket.
    if pocket == "sparse":
        w_chord, w_res, w_comm = 1.00, 0.35, 0.20
    elif pocket == "mid":
        w_chord, w_res, w_comm = 0.85, 0.22, 0.12
    else:
        w_chord, w_res, w_comm = 0.75, 0.18, 0.10

    w_comm = float(w_comm) * float(comm_penalty_scale)
    scored: List[Tuple[float, int, int]] = []
    for u in range(n):
        for v in range(u + 1, n):
            d = min(v - u, n - (v - u))
            chord = 2.0 * sin(pi * d / n)
            resonance = 1.0 / max(1.0, float(d))
            comm = 1.0 if gcd(d, n) > 1 else 0.0
            score = (w_chord * chord) - (w_res * resonance) - (w_comm * comm)
            scored.append((score, u, v))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    return scored


def _cycle_graph(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        j = (i + 1) % n
        if i < j:
            add_edge(adj, i, j)
        else:
            if not adj[j, i]:
                add_edge(adj, j, i)
    return adj


def _build_fib_hybrid_graph(
    n: int,
    e_target: int,
    pocket: str,
    fib_geo_weight_scale: float = 1.0,
    fib_comm_penalty_scale: float = 1.0,
    fib_geo_weight_override: float | None = None,
    fib_comm_weight_override: float | None = None,
) -> np.ndarray:
    if e_target < n - 1:
        raise ValueError(f"fib-hybrid requires e_target >= n-1, got {e_target}")
    if e_target > max_edges(n):
        raise ValueError(f"fib-hybrid e_target exceeds max edges: n={n}, e={e_target}")
    if e_target == n - 1:
        return path_graph(n)

    adj = path_graph(n)
    e_cur = edge_count(adj)
    if e_cur >= e_target:
        return adj

    target_deg = _near_regular_degree_targets(n, e_target)
    deg = adj.sum(axis=1).astype(np.int64)
    ordered_pairs = _fibonacci_pair_order(
        n,
        pocket,
        fib_geo_weight_scale=fib_geo_weight_scale,
        fib_comm_penalty_scale=fib_comm_penalty_scale,
        fib_geo_weight_override=fib_geo_weight_override,
        fib_comm_weight_override=fib_comm_weight_override,
    )

    # Multi-phase fill: strict degree-deficit + anti-triangle, then relax.
    for phase in (0, 1, 2, 3):
        if e_cur >= e_target:
            break
        for _, u, v in ordered_pairs:
            if e_cur >= e_target:
                break
            if adj[u, v]:
                continue
            du, dv = int(deg[u]), int(deg[v])
            tu, tv = int(target_deg[u]), int(target_deg[v])
            tri = int(np.count_nonzero(adj[u] & adj[v]))
            if phase == 0:
                if not (du < tu and dv < tv):
                    continue
                if tri > 0:
                    continue
            elif phase == 1:
                if not (du < tu or dv < tv):
                    continue
                if tri > 0:
                    continue
            elif phase == 2:
                if not (du < tu or dv < tv):
                    continue
                if tri > 1:
                    continue
            add_edge(adj, u, v)
            deg[u] += 1
            deg[v] += 1
            e_cur += 1
    return adj


def _build_fib_s3_hybrid_graph(
    n: int,
    e_target: int,
    pocket: str,
    fib_geo_weight_scale: float = 1.0,
    fib_comm_penalty_scale: float = 1.0,
    fib_geo_weight_override: float | None = None,
    fib_comm_weight_override: float | None = None,
) -> np.ndarray:
    if e_target < n - 1:
        raise ValueError(f"fib-s3-hybrid requires e_target >= n-1, got {e_target}")
    if e_target > max_edges(n):
        raise ValueError(f"fib-s3-hybrid e_target exceeds max edges: n={n}, e={e_target}")
    if e_target == n - 1:
        return path_graph(n)

    adj = path_graph(n)
    e_cur = edge_count(adj)
    if e_cur >= e_target:
        return adj

    target_deg = _near_regular_degree_targets(n, e_target)
    deg = adj.sum(axis=1).astype(np.int64)
    ordered_pairs = _fibonacci_s3_pair_order(
        n,
        pocket,
        fib_geo_weight_scale=fib_geo_weight_scale,
        fib_comm_penalty_scale=fib_comm_penalty_scale,
        fib_geo_weight_override=fib_geo_weight_override,
        fib_comm_weight_override=fib_comm_weight_override,
    )

    for phase in (0, 1, 2, 3):
        if e_cur >= e_target:
            break
        for _, u, v in ordered_pairs:
            if e_cur >= e_target:
                break
            if adj[u, v]:
                continue
            du, dv = int(deg[u]), int(deg[v])
            tu, tv = int(target_deg[u]), int(target_deg[v])
            tri = int(np.count_nonzero(adj[u] & adj[v]))
            if phase == 0:
                if not (du < tu and dv < tv):
                    continue
                if tri > 0:
                    continue
            elif phase == 1:
                if not (du < tu or dv < tv):
                    continue
                if tri > 0:
                    continue
            elif phase == 2:
                if not (du < tu or dv < tv):
                    continue
                if tri > 1:
                    continue
            add_edge(adj, u, v)
            deg[u] += 1
            deg[v] += 1
            e_cur += 1
    return adj


def _fib_pair_base_weights(pocket: str) -> Tuple[float, float]:
    if pocket == "sparse":
        return 1.00, 0.16
    if pocket == "mid":
        return 1.10, 0.12
    return 0.90, 0.10


def _resolve_fib_pair_weights(
    pocket: str,
    fib_geo_weight_scale: float = 1.0,
    fib_comm_penalty_scale: float = 1.0,
    fib_geo_weight_override: float | None = None,
    fib_comm_weight_override: float | None = None,
) -> Tuple[float, float]:
    base_geo, base_comm = _fib_pair_base_weights(pocket)
    if fib_geo_weight_override is None:
        w_geo = float(base_geo) * float(fib_geo_weight_scale)
    else:
        w_geo = float(fib_geo_weight_override)
    if fib_comm_weight_override is None:
        w_comm = float(base_comm) * float(fib_comm_penalty_scale)
    else:
        w_comm = float(fib_comm_weight_override)
    return max(0.0, w_geo), max(0.0, w_comm)


def _fibonacci_pair_order(
    n: int,
    pocket: str,
    fib_geo_weight_scale: float = 1.0,
    fib_comm_penalty_scale: float = 1.0,
    fib_geo_weight_override: float | None = None,
    fib_comm_weight_override: float | None = None,
) -> List[Tuple[float, int, int]]:
    pts = _fibonacci_sphere_points(n)
    w_geo, w_comm = _resolve_fib_pair_weights(
        pocket,
        fib_geo_weight_scale=fib_geo_weight_scale,
        fib_comm_penalty_scale=fib_comm_penalty_scale,
        fib_geo_weight_override=fib_geo_weight_override,
        fib_comm_weight_override=fib_comm_weight_override,
    )
    scored: List[Tuple[float, int, int]] = []
    for u in range(n):
        pu = pts[u]
        for v in range(u + 1, n):
            pv = pts[v]
            dot = float(np.clip(np.dot(pu, pv), -1.0, 1.0))
            # Chord length in [0,2], normalize to [0,1].
            geo = sqrt(max(0.0, 2.0 - 2.0 * dot)) * 0.5
            gap = min(v - u, n - (v - u))
            comm = 1.0 if gcd(gap, n) > 1 else 0.0
            score = (w_geo * geo) - (w_comm * comm)
            scored.append((score, u, v))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    return scored


def _fibonacci_s3_pair_order(
    n: int,
    pocket: str,
    fib_geo_weight_scale: float = 1.0,
    fib_comm_penalty_scale: float = 1.0,
    fib_geo_weight_override: float | None = None,
    fib_comm_weight_override: float | None = None,
) -> List[Tuple[float, int, int]]:
    pts = _fibonacci_s3_points(n)
    w_geo, w_comm = _resolve_fib_pair_weights(
        pocket,
        fib_geo_weight_scale=fib_geo_weight_scale,
        fib_comm_penalty_scale=fib_comm_penalty_scale,
        fib_geo_weight_override=fib_geo_weight_override,
        fib_comm_weight_override=fib_comm_weight_override,
    )
    scored: List[Tuple[float, int, int]] = []
    for u in range(n):
        pu = pts[u]
        for v in range(u + 1, n):
            pv = pts[v]
            dot = float(np.clip(np.dot(pu, pv), -1.0, 1.0))
            geo = sqrt(max(0.0, 2.0 - 2.0 * dot)) * 0.5
            gap = min(v - u, n - (v - u))
            comm = 1.0 if gcd(gap, n) > 1 else 0.0
            score = (w_geo * geo) - (w_comm * comm)
            scored.append((score, u, v))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    return scored


def _fibonacci_sphere_points(n: int) -> np.ndarray:
    pts = np.zeros((n, 3), dtype=np.float64)
    ga = pi * (3.0 - sqrt(5.0))
    for i in range(n):
        z = 1.0 - (2.0 * (i + 0.5) / n)
        r = sqrt(max(0.0, 1.0 - z * z))
        theta = ga * i
        pts[i, 0] = r * cos(theta)
        pts[i, 1] = r * sin(theta)
        pts[i, 2] = z
    return pts


def _fibonacci_s3_points(n: int) -> np.ndarray:
    """
    Deterministic low-discrepancy points on S^3 (R^4 unit sphere).
    Uses additive irrational recurrences mapped through Hopf-style coordinates.
    """
    pts = np.zeros((n, 4), dtype=np.float64)
    phi = (1.0 + sqrt(5.0)) * 0.5
    a1 = 1.0 / phi
    a2 = 1.0 / (phi * phi)
    a3 = 1.0 / (phi * phi * phi)
    for i in range(n):
        t = float(i + 1)
        u1 = (0.5 + t * a1) % 1.0
        u2 = (0.5 + t * a2) % 1.0
        u3 = (0.5 + t * a3) % 1.0
        theta = 2.0 * pi * u1
        psi = 2.0 * pi * u2
        # u3 controls relative radius split across two orthogonal 2D planes.
        r1 = sqrt(max(0.0, 1.0 - u3))
        r2 = sqrt(max(0.0, u3))
        pts[i, 0] = r1 * cos(theta)
        pts[i, 1] = r1 * sin(theta)
        pts[i, 2] = r2 * cos(psi)
        pts[i, 3] = r2 * sin(psi)
    return pts
