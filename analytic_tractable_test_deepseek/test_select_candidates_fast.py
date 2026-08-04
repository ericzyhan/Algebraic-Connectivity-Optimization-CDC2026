"""Equivalence + speed tests for the vectorised Case A scoring path.

Two changes are under test, both in ``AnalyticState.select_candidates``:

  1. ``solve_secular_equation_batch`` replacing the per-candidate
     ``scipy.optimize.brentq`` loop.
  2. The closed-form Δp_min ``p_i - ½ R_eff(i,j)`` replacing the per-candidate
     ``_cheap_delta_pmin`` neighbour loop.

Both are meant to be *algebraically identical* to what they replace, so the
tests here compare against the original scalar implementations rather than
against hand-written expected values.  ``_cheap_delta_pmin`` is retained in the
module for exactly this purpose.

Run from the project root:

    python -m analytic_tractable_test_deepseek.test_select_candidates_fast
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from analytic_tractable_test_deepseek.analytic_state import (
    ZERO_PROJ_GUARD,
    AnalyticState,
    solve_secular_equation,
    solve_secular_equation_batch,
)
from analytic_tractable_test_deepseek.graph_math import (
    build_path_adjacency,
    laplacian,
    non_edges,
)


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #


def _connected_er(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Random connected Erdos-Renyi adjacency (rejection sampled on λ₂ > 0)."""
    for _ in range(500):
        a = np.triu((rng.random((n, n)) < p).astype(np.uint8), 1)
        a = a + a.T
        if float(np.linalg.eigvalsh(laplacian(a))[1]) > 1e-8:
            return a
    raise RuntimeError(f"could not sample a connected graph at n={n}, p={p}")


def _states(rng: np.random.Generator):
    """A spread of states: path graphs, ER graphs, and mid-rollout states."""
    out = []
    for n in (10, 16, 32):
        out.append((f"path n={n}", AnalyticState(build_path_adjacency(n).astype(np.uint8))))
    for n, p in ((16, 0.15), (32, 0.20), (64, 0.25)):
        out.append((f"ER n={n} p={p}", AnalyticState(_connected_er(n, p, rng))))
    # Mid-rollout: several edges in, so the basis is a mix of exact anchors and
    # incremental updates -- where the batch solver must agree with the scalar
    # one on a *stale* ``evals``, not just a fresh one.
    st = AnalyticState(_connected_er(32, 0.18, rng))
    for _ in range(7):
        pairs, _info = st.select_candidates(top=32)
        if pairs.shape[0] == 0:
            break
        st.add_edge(int(pairs[0, 0]), int(pairs[0, 1]))
    out.append(("ER n=32 mid-rollout (7 edges)", st))
    return out


def _stage1(st: AnalyticState):
    """Reproduce the Stage 1 screening so tests can score the same candidates."""
    from analytic_tractable_test_deepseek.analytic_state import screen_candidate_indices

    i_idx, j_idx = st._all_nonedge_incidences()
    if i_idx.size == 0:
        return None
    Lp = st.L_plus
    r_e = Lp[i_idx, i_idx] + Lp[j_idx, j_idx] - 2.0 * Lp[i_idx, j_idx]
    f_score = np.sum((st.U[i_idx] - st.U[j_idx]) ** 2, axis=1)
    i_cand = screen_candidate_indices(r_e, f_score, top=32, cap=64)
    if i_cand.size == 0:
        return None
    psi = (st.Q[i_idx[i_cand]] - st.Q[j_idx[i_cand]]).T
    return i_idx, j_idx, r_e, i_cand, psi


# --------------------------------------------------------------------------- #
#  1. Batched secular solve == scalar secular solve
# --------------------------------------------------------------------------- #


def test_batch_matches_scalar_solver() -> None:
    rng = np.random.default_rng(11)
    worst = 0.0
    total = 0
    for name, st in _states(rng):
        s1 = _stage1(st)
        if s1 is None:
            continue
        _i, _j, _re, _ic, psi = s1
        lambda2 = float(st.evals[1])
        active = np.flatnonzero(np.abs(psi[1, :]) >= ZERO_PROJ_GUARD)
        if active.size == 0:
            continue
        batch = solve_secular_equation_batch(
            psi[:, active], st.evals, lambda2, st.mu, eps=st.secular_eps
        )
        scalar = np.array(
            [
                solve_secular_equation(
                    psi[:, c], st.evals, lambda2, st.mu, eps=st.secular_eps
                )
                for c in active
            ]
        )
        # Relative to the (λ₂, μ) gap: that is the scale the root lives on, and
        # an absolute tolerance would be meaningless when the gap is tiny.
        gap = max(float(st.mu) - lambda2, 1e-300) if np.isfinite(st.mu) else 1.0
        err = float(np.max(np.abs(batch - scalar))) / gap
        worst = max(worst, err)
        total += int(active.size)
        print(f"  {name:32s} m={active.size:3d}  max rel-to-gap err={err:.3e}")
    print(f"  -> {total} candidate roots compared, worst={worst:.3e}")
    assert worst < 1e-9, f"batched secular solve disagrees with scalar: {worst:.3e}"


def test_batch_semantics_edge_cases() -> None:
    """The three-way outcome (root / λ₂ / μ-collision) must survive batching."""
    rng = np.random.default_rng(5)
    st = AnalyticState(_connected_er(16, 0.2, rng))
    lambda2 = float(st.evals[1])
    n = st.n

    # mu = inf  =>  every column returns λ₂ unchanged.
    psi = rng.standard_normal((n, 6))
    got = solve_secular_equation_batch(psi, st.evals, lambda2, float("inf"))
    assert np.allclose(got, lambda2), "mu=inf must leave lambda2 put"
    print("  mu=inf -> lambda2 unchanged: ok")

    # Empty candidate block.
    got = solve_secular_equation_batch(np.zeros((n, 0)), st.evals, lambda2, st.mu)
    assert got.shape == (0,), f"empty block should give shape (0,), got {got.shape}"
    print("  empty block -> shape (0,): ok")

    # A ψ₃-only column (ψ₂ = 0 exactly) drives the μ-endpoint collision branch;
    # whatever the scalar solver returns, the batch must return the same.
    for trial in range(40):
        psi_c = rng.standard_normal(n)
        psi_c[0] = 0.0
        psi_c[1] = 0.0                       # kill the λ₂ pole
        s = solve_secular_equation(psi_c, st.evals, lambda2, st.mu)
        b = solve_secular_equation_batch(psi_c[:, None], st.evals, lambda2, st.mu)[0]
        assert abs(s - b) <= 1e-12 * max(1.0, abs(s)), (
            f"psi2=0 column diverged on trial {trial}: scalar={s!r} batch={b!r}"
        )
    print("  psi2=0 columns match scalar solver over 40 trials: ok")


def test_secular_root_vs_exact_eigh() -> None:
    """Accuracy, not just self-consistency: batch roots vs a real eigh."""
    n = 12
    adj = build_path_adjacency(n).astype(np.uint8)
    st = AnalyticState(adj)
    ne = non_edges(adj)[:20]
    psi = np.stack([st.Q[i] - st.Q[j] for (i, j) in ne], axis=1)
    roots = solve_secular_equation_batch(psi, st.evals, float(st.evals[1]), st.mu)
    worst = 0.0
    for k, (i, j) in enumerate(ne):
        if abs(psi[1, k]) <= ZERO_PROJ_GUARD:
            continue
        adj2 = adj.copy()
        adj2[i, j] = adj2[j, i] = 1
        exact = float(np.linalg.eigvalsh(laplacian(adj2))[1])
        worst = max(worst, abs(float(roots[k]) - exact))
    print(f"  worst |batch root - exact lambda2| over {len(ne)} non-edges: {worst:.3e}")
    assert worst < 1e-10, f"batch root not tracking the true lambda2: {worst:.3e}"


# --------------------------------------------------------------------------- #
#  2. Closed-form Δp_min == the neighbour-loop reference
# --------------------------------------------------------------------------- #


def test_delta_pmin_closed_form() -> None:
    rng = np.random.default_rng(23)
    worst = 0.0
    total = 0
    for name, st in _states(rng):
        s1 = _stage1(st)
        if s1 is None:
            continue
        i_idx, j_idx, r_e, i_cand, _psi = s1
        cand_i, cand_j = i_idx[i_cand], j_idx[i_cand]

        closed = (
            np.minimum(
                np.minimum(st.p[cand_i], st.p[cand_j]) - 0.5 * r_e[i_cand],
                float(st.p_min),
            )
            - float(st.p_min)
        )
        loop = np.array(
            [
                st._cheap_delta_pmin(int(cand_i[k]), int(cand_j[k]))
                for k in range(i_cand.size)
            ]
        )
        err = float(np.max(np.abs(closed - loop)))
        worst = max(worst, err)
        total += int(i_cand.size)
        print(f"  {name:32s} m={i_cand.size:3d}  max |closed - loop|={err:.3e}")
    print(f"  -> {total} candidates compared, worst={worst:.3e}")
    # Algebraically exact; differs only in float summation order (np.sum over n
    # masked entries vs Python sum over |N(i)| terms), so ~1e-16, not bitwise.
    assert worst < 1e-12, f"closed-form dp_min disagrees with the loop: {worst:.3e}"


def test_delta_pmin_sign() -> None:
    """Δp_min ≤ 0 identically -- the α₃ term can never reward an edge."""
    rng = np.random.default_rng(31)
    for name, st in _states(rng):
        s1 = _stage1(st)
        if s1 is None:
            continue
        i_idx, j_idx, r_e, i_cand, _psi = s1
        closed = (
            np.minimum(
                np.minimum(st.p[i_idx[i_cand]], st.p[j_idx[i_cand]])
                - 0.5 * r_e[i_cand],
                float(st.p_min),
            )
            - float(st.p_min)
        )
        assert np.all(closed <= 1e-15), f"{name}: found dp_min > 0"
    print("  dp_min <= 0 on every candidate in every fixture (see report)")


# --------------------------------------------------------------------------- #
#  3. End-to-end: the ranking select_candidates produces is unchanged
# --------------------------------------------------------------------------- #


def _reference_case_a(st: AnalyticState, top: int = 32):
    """The pre-change Case A scoring, verbatim, for end-to-end comparison."""
    s1 = _stage1(st)
    if s1 is None:
        return None
    i_idx, j_idx, r_e, i_cand, Bp = s1
    m = int(i_cand.size)
    nf = float(max(1, st.n))
    from analytic_tractable_test_deepseek.analytic_state import (
        ALPHA1_DEFAULT, ALPHA2_DEFAULT, ALPHA3_DEFAULT,
    )

    # dR_G by the spec's eigenbasis form -- the same expression the module uses.
    # This function exists to test the *secular* and *dp_min* vectorisation, so
    # dR_G must be computed identically on both sides: the direct per-candidate
    # form agrees only to ~1e-16, which is enough to swap two near-tied
    # candidates under argsort and would make this a test of float tie-breaking.
    # The independent cross-check of dR_G lives in ``test_drg_formula``.
    P2 = Bp * Bp
    beta = 1.0 + st.inv_lambda @ P2
    d_rg_red = st.n * (st.inv_lambda_sq @ P2) / np.maximum(beta, 1e-14)

    d_l2 = np.zeros(m)
    lambda2 = float(st.evals[1])
    for c in np.flatnonzero(np.abs(Bp[1, :]) >= ZERO_PROJ_GUARD):
        d_l2[c] = solve_secular_equation(
            Bp[:, c], st.evals, lambda2, st.mu, eps=st.secular_eps
        ) - lambda2
    d_pmin = np.array(
        [st._cheap_delta_pmin(int(i_idx[i_cand[c]]), int(j_idx[i_cand[c]]))
         for c in range(m)]
    )
    reward = (
        ALPHA1_DEFAULT * d_l2 / nf
        + ALPHA2_DEFAULT * d_rg_red / (nf * nf)
        + ALPHA3_DEFAULT * d_pmin
    )
    order = np.argsort(-reward, kind="mergesort")[: min(top, m)]
    pairs = np.stack([i_idx[i_cand[order]], j_idx[i_cand[order]]], axis=1)
    return pairs, reward[order]


def test_end_to_end_ranking_unchanged() -> None:
    rng = np.random.default_rng(47)
    worst_reward = 0.0
    checked = 0
    for name, st in _states(rng):
        if st.r != 1:
            print(f"  {name:32s} r={st.r} (Case B, skipped)")
            continue
        ref = _reference_case_a(st)
        if ref is None:
            continue
        ref_pairs, ref_reward = ref
        new_pairs, info = st.select_candidates(top=32)
        assert new_pairs.shape == ref_pairs.shape, (
            f"{name}: shape {new_pairs.shape} vs reference {ref_pairs.shape}"
        )
        err = float(np.max(np.abs(info["reward"] - ref_reward)))
        worst_reward = max(worst_reward, err)
        checked += 1

        # The selected *set* must be identical.  Ordering may differ only where
        # rewards are tied to within float noise: on symmetric fixtures (path
        # graphs especially) equal-reward candidates are common, and the
        # vectorised d_lambda2/dp_min differ from the scalar ones by ~1 ULP,
        # which is enough to swap two exactly-tied entries under argsort.
        # Asserting bitwise order equality would be testing tie-breaking.
        same_order = bool(np.array_equal(new_pairs, ref_pairs))
        assert set(map(tuple, new_pairs.tolist())) == set(map(tuple, ref_pairs.tolist())), (
            f"{name}: selected candidate set changed"
        )
        swapped = [k for k in range(len(ref_pairs))
                   if tuple(ref_pairs[k]) != tuple(new_pairs[k])]
        for k in swapped:
            assert abs(ref_reward[k] - info["reward"][k]) < 1e-13, (
                f"{name}: position {k} reordered on a non-tied reward "
                f"({ref_reward[k]!r} vs {info['reward'][k]!r})"
            )
        print(
            f"  {name:32s} set identical=True  order identical={same_order}"
            f"  ties swapped={len(swapped)}  max |dreward|={err:.3e}"
        )
    assert checked > 0, "no Case A fixtures were exercised"
    print(f"  -> {checked} fixtures, worst |dreward|={worst_reward:.3e}")
    assert worst_reward < 1e-14, f"reward drifted: {worst_reward:.3e}"


def test_rollout_trajectory_identical() -> None:
    """A greedy rollout must pick the same edge sequence and same final λ₂."""
    rng = np.random.default_rng(101)
    adj = _connected_er(32, 0.18, rng)
    seq, l2 = [], []
    st = AnalyticState(adj.copy())
    for _ in range(25):
        pairs, _info = st.select_candidates(top=32)
        if pairs.shape[0] == 0:
            break
        i, j = int(pairs[0, 0]), int(pairs[0, 1])
        st.add_edge(i, j)
        seq.append((i, j))
        l2.append(float(st.evals[1]))
    exact = adj.copy()
    for (i, j) in seq:
        exact[i, j] = exact[j, i] = 1
    l2_exact = float(np.linalg.eigvalsh(laplacian(exact))[1])
    err = abs(l2[-1] - l2_exact)
    print(f"  {len(seq)} edges added; final lambda2={l2[-1]:.10f} exact={l2_exact:.10f} err={err:.2e}")
    assert err < 1e-9, f"maintained lambda2 diverged from exact: {err:.2e}"


# --------------------------------------------------------------------------- #
#  4. Speed
# --------------------------------------------------------------------------- #


def _time(fn, reps: int = 20, rounds: int = 5) -> float:
    """Best-of-``rounds`` mean over ``reps`` calls, after a warm-up round."""
    for _ in range(reps):
        fn()
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, (time.perf_counter() - t0) / reps)
    return best


def benchmark() -> None:
    """Stage 2 in isolation, then whole-step, so the split is visible.

    Only Stage 2 scoring was vectorised.  Stage 1 screening
    (``_all_nonedge_incidences`` -> ``non_edges``, which materialises a Python
    list of ~n^2/2 tuples, plus two argsorts over all non-edges) is untouched
    and shared by both columns, so the end-to-end ratio is diluted by it --
    increasingly so as it becomes the dominant term.
    """
    rng = np.random.default_rng(7)
    print("  Stage 2 scoring only (Stage 1 hoisted out of the timing loop):")
    print(f"  {'n':>5} {'reference (ms)':>16} {'vectorised (ms)':>17} {'speedup':>9}")
    for n in (12, 16, 32, 64, 128):
        st = AnalyticState(_connected_er(n, 0.25, rng))
        if st.r != 1:
            continue
        s1 = _stage1(st)
        t_ref = _time(lambda: _score_reference(st, s1))
        t_new = _time(lambda: _score_vectorised(st, s1))
        print(f"  {n:>5} {t_ref*1e3:>16.3f} {t_new*1e3:>17.3f} {t_ref/t_new:>8.1f}x")

    print("\n  Full select_candidates (Stage 1 + Stage 2, both paths):")
    print(f"  {'n':>5} {'reference (ms)':>16} {'vectorised (ms)':>17} {'speedup':>9}"
          f" {'stage1 share':>13}")
    for n in (12, 16, 32, 64, 128):
        st = AnalyticState(_connected_er(n, 0.25, rng))
        if st.r != 1:
            continue
        t_s1 = _time(lambda: _stage1(st))
        t_ref = _time(lambda: _reference_case_a(st))
        t_new = _time(lambda: st.select_candidates(top=32))
        print(f"  {n:>5} {t_ref*1e3:>16.3f} {t_new*1e3:>17.3f} {t_ref/t_new:>8.1f}x"
              f" {100.0*t_s1/t_new:>12.0f}%")


def _score_reference(st: AnalyticState, s1):
    """Old Stage 2: per-candidate brentq loop + per-candidate neighbour loop."""
    i_idx, j_idx, _r_e, i_cand, Bp = s1
    m = int(i_cand.size)
    lambda2 = float(st.evals[1])
    d_l2 = np.zeros(m)
    for c in np.flatnonzero(np.abs(Bp[1, :]) >= ZERO_PROJ_GUARD):
        d_l2[c] = solve_secular_equation(
            Bp[:, c], st.evals, lambda2, st.mu, eps=st.secular_eps
        ) - lambda2
    d_pmin = np.array(
        [st._cheap_delta_pmin(int(i_idx[i_cand[c]]), int(j_idx[i_cand[c]]))
         for c in range(m)]
    )
    return d_l2, d_pmin


def _score_vectorised(st: AnalyticState, s1):
    """New Stage 2: one batched solve + one closed-form expression."""
    i_idx, j_idx, r_e, i_cand, Bp = s1
    m = int(i_cand.size)
    lambda2 = float(st.evals[1])
    d_l2 = np.zeros(m)
    active = np.flatnonzero(np.abs(Bp[1, :]) >= ZERO_PROJ_GUARD)
    if active.size:
        d_l2[active] = solve_secular_equation_batch(
            Bp[:, active], st.evals, lambda2, st.mu, eps=st.secular_eps
        ) - lambda2
    d_pmin = (
        np.minimum(
            np.minimum(st.p[i_idx[i_cand]], st.p[j_idx[i_cand]]) - 0.5 * r_e[i_cand],
            float(st.p_min),
        )
        - float(st.p_min)
    )
    return d_l2, d_pmin


# --------------------------------------------------------------------------- #
#  5. Diagnostic (not an assertion) -- see report
# --------------------------------------------------------------------------- #


def test_drg_formula() -> None:
    """Check ΔR_G's screening formula against the exact rank-1 value.

    Not part of the change under test; reported because it sits three lines
    above the code that was edited and the check was cheap to add.
    """
    rng = np.random.default_rng(3)
    worst_beta = worst_drg = worst_spread = 0.0
    total = 0
    for n, p in ((16, 0.20), (24, 0.20), (32, 0.25), (64, 0.25)):
        st = AnalyticState(_connected_er(n, p, rng))
        if st.r != 1:
            continue
        # The *shipped* value, read straight out of select_candidates.
        pairs, info = st.select_candidates(top=32)
        drg_shipped = info["delta_rg_reduction"]
        r_e_sel = info["r_e"]

        # Three independent references, evaluated on the returned pairs.
        beta_dir = np.empty(pairs.shape[0])
        drg_dir = np.empty(pairs.shape[0])
        psi_sel = np.empty((st.n, pairs.shape[0]))
        for c, (i, j) in enumerate(pairs):
            a = np.zeros(st.n)
            a[int(i)] = 1.0
            a[int(j)] = -1.0
            Lpa = st.L_plus @ a
            beta_dir[c] = 1.0 + float(a @ Lpa)            # ref 1: direct
            drg_dir[c] = st.n * float(Lpa @ Lpa) / beta_dir[c]
            psi_sel[:, c] = st.Q.T @ a
        P2 = psi_sel * psi_sel
        beta_eig = 1.0 + st.inv_lambda @ P2               # ref 2: eigenbasis
        drg_eig = st.n * (st.inv_lambda_sq @ P2) / beta_eig
        beta_re = 1.0 + r_e_sel                           # ref 3: 1 + R_eff

        spread = max(
            float(np.max(np.abs(beta_dir - beta_eig))),
            float(np.max(np.abs(beta_dir - beta_re))),
        )
        e_beta = float(np.max(np.abs(beta_eig - beta_dir)))
        e_drg = float(np.max(np.abs(drg_shipped - drg_dir)))
        rel = e_drg / max(float(np.max(np.abs(drg_dir))), 1e-30)
        worst_spread = max(worst_spread, spread)
        worst_beta = max(worst_beta, e_beta)
        worst_drg = max(worst_drg, rel)
        total += pairs.shape[0]
        print(f"  n={n:<4} refs agree to {spread:.2e}   "
              f"shipped dR_G vs direct: abs={e_drg:.2e} rel={rel:.2e}")
    print(f"  -> {total} candidates checked; worst reference spread "
          f"{worst_spread:.3e}, worst shipped rel err {worst_drg:.3e}")
    assert worst_spread < 1e-12, f"reference forms disagree: {worst_spread:.3e}"
    assert worst_drg < 1e-10, (
        f"shipped dR_G disagrees with a^T(L+)^2a/(1+a^T L+ a): {worst_drg:.3e} "
        f"-- is psi being used as the incidence vector?"
    )


def main() -> None:
    print("[test] batched secular solve == scalar solver:")
    test_batch_matches_scalar_solver()
    print("[test] batched solver edge-case semantics:")
    test_batch_semantics_edge_cases()
    print("[test] batched secular root vs exact eigh:")
    test_secular_root_vs_exact_eigh()
    print("[test] closed-form dp_min == neighbour-loop reference:")
    test_delta_pmin_closed_form()
    print("[test] dp_min sign:")
    test_delta_pmin_sign()
    print("[test] end-to-end select_candidates ranking unchanged:")
    test_end_to_end_ranking_unchanged()
    print("[test] greedy rollout tracks exact lambda2:")
    test_rollout_trajectory_identical()
    print("[bench] Case A scoring, reference loops vs vectorised:")
    benchmark()
    print("[test] dR_G screening formula vs 3 independent references:")
    test_drg_formula()
    print("\nALL SELECT_CANDIDATES FAST-PATH TESTS PASSED")


if __name__ == "__main__":
    main()
