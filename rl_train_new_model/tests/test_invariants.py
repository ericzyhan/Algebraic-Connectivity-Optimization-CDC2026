"""Property tests for the spectral tracker (docs/CLAUDE.md S5).

No pytest in this environment -- these are plain assert-based functions,
runnable standalone:

    python -m rl_train_new_model.tests.test_invariants

Each test_* function raises AssertionError on failure; main() runs them all
and reports a PASS/FAIL summary, exiting 1 if anything failed.
"""
from __future__ import annotations

import traceback

import numpy as np

from ..graph_math import build_path_adjacency, laplacian
from ..spectral import SpectralTracker


def _build_cycle_adjacency(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        adj[i, (i + 1) % n] = 1
        adj[(i + 1) % n, i] = 1
    return adj


def _non_edges(adj: np.ndarray):
    n = adj.shape[0]
    return [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 0]


def _random_walk_tracker(n: int, steps: int, seed: int) -> SpectralTracker:
    adj = build_path_adjacency(n)
    tracker = SpectralTracker(adj, oversample=3, exact_reset_every=7)
    rng = np.random.RandomState(seed)
    remaining = _non_edges(adj)
    for _ in range(steps):
        if not remaining:
            break
        idx = rng.randint(len(remaining))
        i, j = remaining.pop(idx)
        tracker.add_edge(i, j)
    return tracker


def _p_vector(tracker: SpectralTracker) -> np.ndarray:
    """Recompute the full resistance-curvature vector p_i (not just P_min),
    independently of the tracker's internal _compute_P_min_from_Lplus, for
    an implementation-independent Foster check."""
    d = np.diag(tracker.Lplus)
    p = np.ones(tracker.n, dtype=np.float64)
    adj_bool = tracker.adj > 0
    for node in range(tracker.n):
        neighbors = np.flatnonzero(adj_bool[node])
        if neighbors.size == 0:
            continue
        Rij = d[node] + d[neighbors] - 2.0 * tracker.Lplus[node, neighbors]
        p[node] = 1.0 - 0.5 * float(np.sum(Rij))
    return p


def test_moore_penrose():
    tracker = _random_walk_tracker(n=16, steps=15, seed=1)
    L = laplacian(tracker.adj)
    resid = np.linalg.norm(L @ tracker.Lplus @ L - L, ord="fro")
    denom = np.linalg.norm(L, ord="fro")
    rel = resid / denom
    assert rel < 1e-10, f"Moore-Penrose residual too large: {rel}"


def test_centering():
    tracker = _random_walk_tracker(n=16, steps=15, seed=2)
    ones = np.ones(tracker.n)
    assert np.max(np.abs(tracker.Lplus @ ones)) < 1e-8, "L+ @ 1 != 0"
    assert np.max(np.abs(ones @ tracker.Lplus)) < 1e-8, "1^T @ L+ != 0"


def test_foster_node_form():
    tracker = _random_walk_tracker(n=14, steps=12, seed=3)
    p = _p_vector(tracker)
    total = float(np.sum(p))
    assert abs(total - 1.0) < 1e-8, f"sum_i p_i != 1: got {total}"


def test_foster_edge_form():
    tracker = _random_walk_tracker(n=14, steps=12, seed=4)
    n = tracker.n
    d = np.diag(tracker.Lplus)
    adj_bool = tracker.adj > 0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if adj_bool[i, j]:
                Rij = d[i] + d[j] - 2.0 * tracker.Lplus[i, j]
                total += Rij
    assert abs(total - (n - 1)) < 1e-8, f"sum_(i,j) in E R_ij != n-1: got {total}"


def test_delta_rg_exactness():
    adj = build_path_adjacency(14)
    tracker = SpectralTracker(adj, oversample=3, exact_reset_every=50)
    remaining = _non_edges(adj)
    rng = np.random.RandomState(5)
    for _ in range(10):
        idx = rng.randint(len(remaining))
        i, j = remaining.pop(idx)
        rg_before = tracker.RG
        tracker.add_edge(i, j)
        rg_after_incremental = tracker.RG

        L_full = laplacian(tracker.adj)
        evals_full = np.linalg.eigvalsh(L_full)
        rg_recomputed = float(tracker.n) * float(np.sum(1.0 / np.maximum(evals_full[1:], 1e-14)))

        rel = abs(rg_after_incremental - rg_recomputed) / max(abs(rg_recomputed), 1e-12)
        assert rel < 1e-8, (
            f"Delta R_G incremental value mismatches full recompute: "
            f"incremental={rg_after_incremental} recompute={rg_recomputed} rel={rel}"
        )
        assert rg_after_incremental < rg_before, "R_G did not decrease on edge addition"


def test_k_consistency():
    tracker = _random_walk_tracker(n=16, steps=6, seed=6)
    remaining = _non_edges(tracker.adj)
    rng = np.random.RandomState(7)
    for _ in range(8):
        if not remaining:
            break
        idx = rng.randint(len(remaining))
        i, j = remaining.pop(idx)
        tracker.add_edge(i, j)
        resid = np.linalg.norm(tracker.K - tracker.Lplus @ tracker.Lplus, ord="fro")
        denom = max(np.linalg.norm(tracker.K, ord="fro"), 1e-12)
        assert resid / denom < 1e-6, f"K != (L+)^2 after add_edge: rel resid {resid/denom}"


def test_deflation():
    n = 12
    adj = _build_cycle_adjacency(n)
    tracker = SpectralTracker(adj, oversample=3, exact_reset_every=50)
    assert tracker.multiplicity == 2, f"expected C_12 to have lambda2 multiplicity 2, got {tracker.multiplicity}"
    mu2_before = tracker.mu2

    tracker.add_edge(0, 5)

    assert tracker.multiplicity == 1, f"expected multiplicity to drop by exactly 1, got {tracker.multiplicity}"
    assert abs(tracker.mu2 - mu2_before) < 1e-10, (
        f"lambda2 should be exactly unchanged after a single edge with r=2: "
        f"before={mu2_before} after={tracker.mu2}"
    )


def test_secular_bracket():
    adj = build_path_adjacency(14)
    tracker = SpectralTracker(adj, oversample=3, exact_reset_every=50)
    remaining = _non_edges(adj)
    rng = np.random.RandomState(8)
    checked = 0
    for _ in range(10):
        if not remaining:
            break
        idx = rng.randint(len(remaining))
        i, j = remaining.pop(idx)
        tracker.add_edge(i, j)
        if tracker.last_secular_bracket is not None and tracker.last_secular_delta is not None:
            lo, hi = tracker.last_secular_bracket
            delta = tracker.last_secular_delta
            tol = 1e-9 * max(1.0, abs(hi))
            assert lo - tol <= delta <= hi + tol, (
                f"secular root {delta} outside exact bracket [{lo}, {hi}]"
            )
            checked += 1
    assert checked > 0, "no r=1 secular solves were exercised"


def test_rank_one_interlacing():
    n = 14
    adj = build_path_adjacency(n)
    remaining = _non_edges(adj)
    rng = np.random.RandomState(9)
    for _ in range(6):
        idx = rng.randint(len(remaining))
        i, j = remaining.pop(idx)

        L_before = laplacian(adj)
        evals_before = np.sort(np.linalg.eigvalsh(L_before))

        adj = adj.copy()
        adj[i, j] = 1
        adj[j, i] = 1
        L_after = laplacian(adj)
        evals_after = np.sort(np.linalg.eigvalsh(L_after))

        for k in range(n - 1):
            assert evals_before[k] - 1e-9 <= evals_after[k], (
                f"interlacing violated (lower): lambda_{k}(G)={evals_before[k]} > lambda_{k}(G+e)={evals_after[k]}"
            )
            assert evals_after[k] <= evals_before[k + 1] + 1e-9, (
                f"interlacing violated (upper): lambda_{k}(G+e)={evals_after[k]} > lambda_{k+1}(G)={evals_before[k+1]}"
            )


def test_monotonicity_rg():
    tracker = _random_walk_tracker(n=16, steps=1, seed=10)
    adj = tracker.adj.copy()
    tracker = SpectralTracker(adj, oversample=3, exact_reset_every=50)
    remaining = _non_edges(adj)
    rng = np.random.RandomState(11)
    prev_rg = tracker.RG
    for _ in range(10):
        if not remaining:
            break
        idx = rng.randint(len(remaining))
        i, j = remaining.pop(idx)
        tracker.add_edge(i, j)
        assert tracker.RG < prev_rg, f"R_G did not strictly decrease: {prev_rg} -> {tracker.RG}"
        prev_rg = tracker.RG


ALL_TESTS = [
    test_moore_penrose,
    test_centering,
    test_foster_node_form,
    test_foster_edge_form,
    test_delta_rg_exactness,
    test_k_consistency,
    test_deflation,
    test_secular_bracket,
    test_rank_one_interlacing,
    test_monotonicity_rg,
]


def main() -> int:
    failures = 0
    for test_fn in ALL_TESTS:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
        except Exception:
            failures += 1
            print(f"ERROR {name}:")
            traceback.print_exc()
    total = len(ALL_TESTS)
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
