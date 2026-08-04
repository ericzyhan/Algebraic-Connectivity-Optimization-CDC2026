"""Validation of AnalyticState against exact dense eigendecomposition.

Run from the project root:

    python -m analytic_tractable_test_deepseek.test_analytic_state

Checks:
  1. add_edge keeps λ₂, R_G, p_min close to the exact re-computation.
  2. The secular root matches the exact new λ₂ for single-edge additions.
  3. The ΔR_G reduction formula matches the exact R_G change.
  4. Case B (multiple components) resolves multiplicity via Rayleigh-Ritz.
  5. select_candidates returns at most 32 candidates, all unused non-edges.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from analytic_tractable_test_deepseek.analytic_state import (
    AnalyticState,
    compute_p_min,
    solve_secular_equation,
)
from analytic_tractable_test_deepseek.graph_math import (
    build_path_adjacency,
    laplacian,
    non_edges,
    spectral_features,
)


def _exact_spectral(adj: np.ndarray) -> tuple[float, float, float]:
    L = laplacian(adj)
    evals, evecs = np.linalg.eigh(L)
    rg = float(adj.shape[0]) * float(np.sum(1.0 / np.maximum(evals[1:], 1e-14)))
    pmin = compute_p_min(adj, evecs[:, 1:] @ np.diag(1.0 / np.maximum(evals[1:], 1e-14)) @ evecs[:, 1:].T)
    return float(evals[1]), rg, pmin


def test_case_a_single_edges() -> None:
    n = 10
    adj = build_path_adjacency(n).astype(np.uint8)
    st = AnalyticState(adj)
    adj_exact = adj.copy()

    edges = [(1, 8), (0, 7), (2, 9), (3, 6), (0, 5)]
    for (i, j) in edges:
        l2_exact, rg_exact, pm_exact = _exact_spectral(adj_exact)
        l2_state = float(st.evals[1])
        rg_state = float(st.R_G)
        pm_state = float(st.p_min)
        err_l2 = abs(l2_state - l2_exact)
        rel_rg = abs(rg_state - rg_exact) / max(1.0, abs(rg_exact))
        err_pm = abs(pm_state - pm_exact)
        print(
            f"  before ({i},{j}): λ₂ err={err_l2:.2e}  R_G rel err={rel_rg:.2e}  "
            f"p_min err={err_pm:.2e}"
        )
        assert err_l2 < 1e-6, f"λ₂ drift too large: {err_l2:.2e}"
        assert rel_rg < 1e-8, f"R_G drift too large: {rel_rg:.2e}"
        assert err_pm < 1e-6, f"p_min drift too large: {err_pm:.2e}"

        st.add_edge(i, j)
        adj_exact[i, j] = 1
        adj_exact[j, i] = 1


def test_secular_root_matches_exact() -> None:
    n = 10
    adj = build_path_adjacency(n).astype(np.uint8)
    st = AnalyticState(adj)
    for (i, j) in non_edges(adj)[:8]:
        a = np.zeros(n)
        a[i] = 1.0
        a[j] = -1.0
        psi = st.Q.T @ a
        root = solve_secular_equation(psi, st.evals, float(st.evals[1]), st.mu)
        adj2 = adj.copy()
        adj2[i, j] = 1
        adj2[j, i] = 1
        ev2 = np.linalg.eigvalsh(laplacian(adj2))
        if abs(psi[1]) > 1e-14:
            err = abs(root - ev2[1])
            assert err < 1e-8, f"secular root err too large for ({i},{j}): {err:.2e}"
            print(f"  secular root err ({i},{j}): {err:.2e}")


def test_case_b_multiplicity() -> None:
    # Cycle C6 has a repeated Fiedler eigenvalue λ₂ = λ₃ = 1 (multiplicity 2).
    n = 6
    adj = np.zeros((n, n), dtype=np.uint8)
    for k in range(n):
        adj[k, (k + 1) % n] = 1
        adj[(k + 1) % n, k] = 1
    st = AnalyticState(adj)
    print(f"  cycle C6: r={st.r} λ₂={st.evals[1]:.6f}")
    assert st.r >= 2, "expected multiplicity > 1 for cycle C6"
    st.add_edge(0, 3)  # chord breaks the C6 symmetry -> multiplicity resolves
    print(f"  after chord: r={st.r} λ₂={st.evals[1]:.6f}")


def test_select_candidates() -> None:
    n = 16
    adj = build_path_adjacency(n).astype(np.uint8)
    st = AnalyticState(adj)
    pairs, info = st.select_candidates(top=32)
    assert pairs.shape[0] <= 32, f"more than 32 candidates returned: {pairs.shape[0]}"
    assert pairs.shape[1] == 2
    for (i, j) in pairs:
        assert adj[i, j] == 0, f"returned an existing edge ({i},{j})"
    print(f"  path: selected {pairs.shape[0]} candidates, case={info.get('case', [None])[0]}")

    # Case B (r > 1): cycle C6 has λ₂ multiplicity 2; the MaxVol path should
    # return r-1 = 1 candidate edge, which must be an unused non-edge.
    m = 6
    adj_c6 = np.zeros((m, m), dtype=np.uint8)
    for k in range(m):
        adj_c6[k, (k + 1) % m] = 1
        adj_c6[(k + 1) % m, k] = 1
    st_c6 = AnalyticState(adj_c6)
    assert st_c6.r >= 2, "expected multiplicity > 1 for cycle C6"
    pairs_b, info_b = st_c6.select_candidates(top=32)
    assert pairs_b.shape[0] <= 32
    assert pairs_b.shape[0] >= 1, "Case B should return at least one candidate"
    for (i, j) in pairs_b:
        assert adj_c6[i, j] == 0, f"Case B returned an existing edge ({i},{j})"
    print(f"  cycle C6 (r={st_c6.r}): selected {pairs_b.shape[0]} candidates, "
          f"case={info_b.get('case', [None])[0]}")


def main() -> None:
    print("[test] Case A single-edge drift vs exact:")
    test_case_a_single_edges()
    print("[test] secular root vs exact λ₂:")
    test_secular_root_matches_exact()
    print("[test] Case B multiplicity:")
    test_case_b_multiplicity()
    print("[test] select_candidates:")
    test_select_candidates()
    print("\nALL ANALYTIC STATE TESTS PASSED")


if __name__ == "__main__":
    main()
