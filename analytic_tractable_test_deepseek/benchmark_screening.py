"""Scale check for the O(n³) -> O(n²) Stage-1 screening optimization.

Confirms, at n = 32 / 64 / 128:
  1. the optimized Stage-1 scores (R_e from L⁺, F from U) are identical to the
     previous full-projection ψ = QᵀB formulation;
  2. the optimized screening is faster and scales closer to O(n²), while the
     old O(n·k) ψ pass scales closer to O(n³) (the empirical log-log exponent).

Run from the project root:

    python -m analytic_tractable_test_deepseek.benchmark_screening
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from analytic_tractable_test_deepseek.analytic_state import AnalyticState
from analytic_tractable_test_deepseek.graph_math import build_path_adjacency, non_edges


def _old_stage1(Q, evals, U, i_idx, j_idx, r):
    """Pre-optimization Stage 1: form full ψ = QᵀB over all non-edges."""
    diff = Q[i_idx, :] - Q[j_idx, :]   # (k, n)
    psi = diff.T                       # (n, k)
    inv = np.zeros(evals.shape[0])
    inv[1:] = 1.0 / np.maximum(evals[1:], 1e-14)
    r_e = np.einsum("i,ij->j", inv, psi * psi)
    f_score = np.sum(psi[1 : 1 + r, :] ** 2, axis=0)
    return r_e, f_score


def _new_stage1(L_plus, U, i_idx, j_idx, r):
    """Optimized Stage 1: R_e from L⁺ (O(1)/edge), F from U (O(r)/edge)."""
    r_e = L_plus[i_idx, i_idx] + L_plus[j_idx, j_idx] - 2.0 * L_plus[i_idx, j_idx]
    f_score = np.sum((U[i_idx] - U[j_idx]) ** 2, axis=1)
    return r_e, f_score


def _best_time(fn, reps: int) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    sizes = [32, 64, 128]
    rows = []
    exponents_new = []
    exponents_old = []

    print("=" * 88)
    print("  Stage-1 screening scale check  (path graphs)")
    print("=" * 88)
    header = (
        f"{'n':>5}  {'k(non-edges)':>12}  {'old ψ O(n·k) [s]':>18}  "
        f"{'new O(n²) [s]':>16}  {'speedup':>9}"
    )
    print(header)
    print("-" * len(header))

    prev = None
    for n in sizes:
        adj = build_path_adjacency(n).astype(np.uint8)
        st = AnalyticState(adj)
        ne = np.asarray(non_edges(adj), dtype=np.int64)
        i_idx, j_idx = ne[:, 0], ne[:, 1]
        r = st.r

        # 1) equivalence check
        r_e_old, f_old = _old_stage1(st.Q, st.evals, st.U, i_idx, j_idx, r)
        r_e_new, f_new = _new_stage1(st.L_plus, st.U, i_idx, j_idx, r)
        assert np.allclose(r_e_old, r_e_new, rtol=1e-10, atol=1e-12), (
            f"n={n}: R_e mismatch between ψ and L⁺ formulations"
        )
        assert np.allclose(f_old, f_new, rtol=1e-10, atol=1e-12), (
            f"n={n}: F mismatch between ψ and U formulations"
        )

        # 2) timing (repeated calls so timings are measurable)
        old_t = _best_time(lambda: _old_stage1(st.Q, st.evals, st.U, i_idx, j_idx, r), 30)
        new_t = _best_time(lambda: _new_stage1(st.L_plus, st.U, i_idx, j_idx, r), 300)
        speedup = old_t / max(new_t, 1e-12)
        rows.append((n, i_idx.size, old_t, new_t, speedup))

        # 3) empirical exponent from consecutive sizes
        if prev is not None:
            dn = np.log(float(n) / float(prev[0]))
            e_new = np.log(rows[-1][3] / prev[2]) / dn
            e_old = np.log(rows[-1][2] / prev[1]) / dn
            exponents_new.append(e_new)
            exponents_old.append(e_old)
        prev = (n, rows[-1][2], rows[-1][3])

        print(
            f"{n:>5}  {i_idx.size:>12}  {old_t:>18.6e}  {new_t:>16.6e}  {speedup:>9.2f}x"
        )

    print("-" * len(header))
    if exponents_new:
        print(f"  empirical log-log slope (n: 64->128):  new = {exponents_new[-1]:.2f}  "
              f"old = {exponents_old[-1]:.2f}   (expect ~2 vs ~3)")

    # Final assertions: new is faster at n=128 and scales better.
    n128 = [r for r in rows if r[0] == 128][0]
    assert n128[4] > 1.0, "optimized Stage-1 should be faster than the ψ pass at n=128"
    if exponents_new:
        assert exponents_new[-1] < exponents_old[-1], (
            "optimized screening should scale better (lower empirical exponent)"
        )
    print("\nSCREENING OPTIMIZATION SCALE CHECK PASSED")


if __name__ == "__main__":
    main()
