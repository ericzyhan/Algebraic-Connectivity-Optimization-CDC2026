"""Verification for step 3 (`SpectralTracker.add_edge`).

Covers the closed forms the algorithm relies on:

  1. differential test of Delta lambda_2 / Delta R_G / Delta p against dense truth
  2. the multiplicity plateau, asserted exact rather than merely small
  3. the deterministic r-1 deflation batch and its z-accumulator
  4. long-horizon drift with the exact anchor suppressed
  5. cost scaling

Run:  python -m rl_train_new_model.tests.test_step3
"""

from __future__ import annotations

import itertools
import time

import numpy as np

from ..graph_math import laplacian
from ..spectral import SpectralTracker


# --------------------------------------------------------------------------- #
#  Dense reference implementations
# --------------------------------------------------------------------------- #

def dense_pinv(adj):
    return np.linalg.pinv(laplacian(adj).astype(np.float64))


def dense_p(adj):
    Lp = dense_pinv(adj)
    d = np.diag(Lp)
    n = adj.shape[0]
    p = np.ones(n)
    for k in range(n):
        nb = np.flatnonzero(adj[k] > 0)
        if nb.size:
            p[k] = 1.0 - 0.5 * np.sum(d[k] + d[nb] - 2.0 * Lp[k, nb])
    return p


def dense_rg(adj):
    ev = np.linalg.eigvalsh(laplacian(adj).astype(np.float64))
    return float(adj.shape[0]) * float(np.sum(1.0 / ev[1:]))


def dense_lambda2(adj):
    return float(np.linalg.eigvalsh(laplacian(adj).astype(np.float64))[1])


def random_connected(n, pr, rng):
    while True:
        A = (rng.random((n, n)) < pr).astype(np.uint8)
        A = np.triu(A, 1)
        A = A + A.T
        if np.linalg.eigvalsh(laplacian(A).astype(np.float64))[1] > 1e-8:
            return A


def k24():
    A = np.zeros((6, 6), dtype=np.uint8)
    for a in (0, 1):
        for c in (2, 3, 4, 5):
            A[a, c] = A[c, a] = 1
    return A


# --------------------------------------------------------------------------- #
#  1. Differential test against dense truth
# --------------------------------------------------------------------------- #

def test_delta_p_exact():
    """Closed-form Delta p must match a dense pinv recompute for every non-edge."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for n in (8, 14, 20, 32):
        A = random_connected(n, 0.3, rng)
        for (i, j) in itertools.combinations(range(n), 2):
            if A[i, j]:
                continue
            tr = SpectralTracker(A, exact_reset_every=10**9)
            before = tr.get_p()
            tr.add_edge(i, j)
            got = tr.get_p() - before
            A2 = A.copy()
            A2[i, j] = A2[j, i] = 1
            want = dense_p(A2) - dense_p(A)
            worst = max(worst, float(np.abs(got - want).max()))
    assert worst < 1e-11, f"Delta p error {worst:.3e}"
    return f"max |Delta p - dense| = {worst:.2e}"


def test_delta_rg_and_pmin_exact():
    rng = np.random.default_rng(1)
    worst_rg = worst_pm = 0.0
    for n in (8, 14, 20, 32):
        A = random_connected(n, 0.3, rng)
        for (i, j) in itertools.combinations(range(n), 2):
            if A[i, j]:
                continue
            tr = SpectralTracker(A, exact_reset_every=10**9)
            tr.add_edge(i, j)
            A2 = A.copy()
            A2[i, j] = A2[j, i] = 1
            worst_rg = max(worst_rg, abs(tr.RG - dense_rg(A2)) / dense_rg(A2))
            worst_pm = max(worst_pm, abs(tr.get_P_min() - dense_p(A2).min()))
    assert worst_rg < 1e-11, f"R_G rel error {worst_rg:.3e}"
    assert worst_pm < 1e-11, f"P_min error {worst_pm:.3e}"
    return f"R_G rel {worst_rg:.2e}, P_min {worst_pm:.2e}"


def test_foster_invariant_holds():
    """sum_i p_i = 1 must survive a run of incremental updates."""
    rng = np.random.default_rng(2)
    A = random_connected(24, 0.25, rng)
    tr = SpectralTracker(A, exact_reset_every=10**9)
    worst = abs(float(tr.get_p().sum()) - 1.0)
    for _ in range(40):
        nz = [(i, j) for i, j in itertools.combinations(range(24), 2) if not tr.adj[i, j]]
        i, j = nz[rng.integers(len(nz))]
        tr.add_edge(i, j)
        worst = max(worst, abs(float(tr.get_p().sum()) - 1.0))
    assert worst < 1e-10, f"Foster drift {worst:.3e}"
    return f"max |sum p - 1| over 40 edges = {worst:.2e}"


def test_secular_delta_lambda2():
    """Delta lambda_2 from the secular solve, against dense eigvalsh."""
    rng = np.random.default_rng(3)
    worst = 0.0
    n = 20
    A = random_connected(n, 0.3, rng)
    for (i, j) in itertools.combinations(range(n), 2):
        if A[i, j]:
            continue
        tr = SpectralTracker(A, exact_reset_every=10**9)
        if tr.multiplicity != 1:
            continue
        l2_before = dense_lambda2(A)
        tr._solve_secular_for_edge(i, j)
        if tr.last_secular_delta is None:
            continue
        A2 = A.copy()
        A2[i, j] = A2[j, i] = 1
        want = dense_lambda2(A2) - l2_before
        worst = max(worst, abs(tr.last_secular_delta - want))
    assert worst < 1e-9, f"Delta lambda_2 error {worst:.3e}"
    return f"max |Delta lambda_2 - dense| = {worst:.2e}"


# --------------------------------------------------------------------------- #
#  2. Plateau, asserted exact
# --------------------------------------------------------------------------- #

def test_plateau_exact():
    A = k24()
    base = SpectralTracker(A)
    assert base.multiplicity == 3, f"expected r=3 on K_2,4, got {base.multiplicity}"
    for (i, j) in itertools.combinations(range(6), 2):
        if A[i, j]:
            continue
        tr = SpectralTracker(A)
        l2_before = tr.get_spectral_features()[0]
        tr.add_edge(i, j)
        l2_after = tr.get_spectral_features()[0]
        assert tr.last_secular_delta == 0.0, (
            f"edge ({i},{j}): last_secular_delta={tr.last_secular_delta}, want exact 0.0"
        )
        assert l2_after == l2_before, (
            f"edge ({i},{j}): lambda_2 moved by {l2_after - l2_before:.3e}, want exactly 0"
        )
    return "r=3, all 7 candidates give Delta lambda_2 == 0.0 exactly"


# --------------------------------------------------------------------------- #
#  3. Deflation batch + z-accumulator
# --------------------------------------------------------------------------- #

def test_batch_deflation_cascade():
    """r-1 RRQR-selected edges cascade r -> 1, and Z_accum survives."""
    A = k24()
    tr = SpectralTracker(A)
    r = tr.multiplicity
    U = tr.U[:, :r]
    nz = [(i, j) for i, j in itertools.combinations(range(6), 2) if not A[i, j]]
    Z = np.array([U[i] - U[j] for i, j in nz]).T

    cond = SpectralTracker.deflation_pool_conditioning(Z)
    assert np.isfinite(cond), f"full pool should be well conditioned, got cond(R)={cond}"

    order = tr.rrqr_rank(Z)
    sel = [nz[t] for t in order[: r - 1]]

    tr.begin_deflation_batch(len(sel))
    seen = [r]
    for (i, j) in sel:
        tr.add_edge(i, j)
        seen.append(tr.multiplicity)
    want = list(range(r, r - len(sel) - 1, -1))  # one drop per edge, ending at 1
    assert seen == want, f"multiplicity path {seen}, want {want}"
    assert tr.last_deflation_ok, "deflation post-condition failed"
    assert len(tr._z_accum_cols) == r - 1, (
        f"Z_accum has {len(tr._z_accum_cols)} columns, want {r-1} "
        "(reset-on-drop would have wiped it)"
    )
    # The r-th edge then moves lambda_2 through the simple-case secular path.
    l2_before = tr.get_spectral_features()[0]
    rest = [e for e in nz if e not in sel]
    moved = []
    for (i, j) in rest:
        t2 = SpectralTracker.from_state_dict(tr.state_dict())
        t2.add_edge(i, j)
        moved.append(t2.get_spectral_features()[0] - l2_before)
    assert max(moved) > 0.5, f"no r-th edge moved lambda_2: {moved}"
    return (
        f"r cascade {seen}, Z_accum kept {len(tr._z_accum_cols)} cols, "
        f"best r-th edge Delta lambda_2 = {max(moved):+.6f}"
    )


def test_batch_basis_is_frozen():
    """sigma_r is rotation-invariant only in a consistent basis; assert we froze one."""
    A = k24()
    tr = SpectralTracker(A)
    r = tr.multiplicity
    nz = [(i, j) for i, j in itertools.combinations(range(6), 2) if not A[i, j]]
    Z = np.array([U for U in (tr.U[:, :r][i] - tr.U[:, :r][j] for i, j in nz)]).T
    sel = [nz[t] for t in tr.rrqr_rank(Z)[: r - 1]]

    tr.begin_deflation_batch(len(sel))
    frozen = tr._z_accum_U.copy()
    for (i, j) in sel:
        tr.add_edge(i, j)
        assert np.array_equal(tr._z_accum_U, frozen), "basis rotated mid-batch"

    Zacc = np.stack(tr._z_accum_cols, axis=1)
    s_here = np.linalg.svd(Zacc, compute_uv=False)[-1]

    rng = np.random.default_rng(11)
    Q, _ = np.linalg.qr(rng.standard_normal((r, r)))
    s_rot = np.linalg.svd(Q.T @ Zacc, compute_uv=False)[-1]
    assert abs(s_here - s_rot) < 1e-10, (
        f"sigma_min not rotation-invariant: {s_here:.6f} vs {s_rot:.6f}"
    )
    return f"basis frozen across batch; sigma_min {s_here:.6f} invariant under rotation"


def test_maxvol_precondition():
    """cond(R) catches a rank-deficient pool; MaxVol excludes the ||z||=0 edge."""
    A = k24()
    tr = SpectralTracker(A)
    r = tr.multiplicity
    U = tr.U[:, :r]
    nz = [(i, j) for i, j in itertools.combinations(range(6), 2) if not A[i, j]]
    Z = np.array([U[i] - U[j] for i, j in nz]).T

    zero_rows = [t for t in range(Z.shape[1]) if np.linalg.norm(Z[:, t]) < 1e-9]
    assert zero_rows, "expected K_2,4 to contain a ||z||=0 candidate"

    good = SpectralTracker.deflation_pool_conditioning(Z)
    bad = SpectralTracker.deflation_pool_conditioning(
        np.column_stack([Z[:, 1], Z[:, 1] * 2.0, Z[:, 1] * -0.5])
    )
    assert np.isfinite(good), f"full pool cond(R)={good}"
    assert not np.isfinite(bad) or bad > 1e12, f"rank-1 pool cond(R)={bad}, want blow-up"

    _, selected = tr.maxvol_rank(Z)
    assert not (set(selected.tolist()) & set(zero_rows)), "MaxVol picked a zero-z candidate"
    return f"cond(R) full={good:.2e} rank-1={bad:.2e}; zero-z candidate excluded"


def test_maxvol_dominance():
    """MaxVol must return a dominant subset: max |Q Q[I]^-1| <= e."""
    rng = np.random.default_rng(23)
    e = 1.02
    worst_ratio = 0.0
    gains = []
    for (k, r) in ((32, 3), (32, 5), (16, 2), (64, 7)):
        for _ in range(20):
            Q, _ = np.linalg.qr(rng.standard_normal((k, r)))
            I_warm = SpectralTracker.maxvol_warm_start(Q)
            I = SpectralTracker.maxvol(Q, e=e)
            assert I.shape == (r,), f"MaxVol returned shape {I.shape}, want ({r},)"
            assert len(set(I.tolist())) == r, f"MaxVol returned duplicate rows: {I}"

            B = np.linalg.solve(Q[I, :].T, Q.T).T
            worst_ratio = max(worst_ratio, float(np.abs(B).max()))
            # ...and the volume must not be worse than the LU warm start it began from.
            v_warm = abs(float(np.linalg.det(Q[I_warm, :])))
            v_max = abs(float(np.linalg.det(Q[I, :])))
            assert v_max >= v_warm - 1e-12, f"MaxVol lost volume: {v_warm:.3e} -> {v_max:.3e}"
            if v_warm > 1e-12:
                gains.append(v_max / v_warm)
    assert worst_ratio <= e + 1e-9, f"dominance violated: max|B| = {worst_ratio:.6f} > {e}"
    return (
        f"max|Q Q[I]^-1| = {worst_ratio:.4f} <= {e}; "
        f"mean volume gain over LU warm start = {float(np.mean(gains)):.3f}x"
    )


def test_maxvol_beats_rrqr_volume():
    """maxvol_rank warm-starts from RRQR, so its subset never has less volume.

    Also pins the reason that warm start exists: the standalone LU-started
    MaxVol is only locally dominant and does sometimes finish below RRQR.
    """
    rng = np.random.default_rng(31)
    tr = SpectralTracker(k24())
    ratios = []
    lu_losses = 0
    r, k = 4, 24
    for _ in range(200):
        Z = rng.standard_normal((r, k))
        Q, _ = np.linalg.qr(Z.T)
        v_rrqr = abs(float(np.linalg.det(Q[tr.rrqr_rank(Z)[:r], :])))

        _, selected = tr.maxvol_rank(Z)
        v_mv = abs(float(np.linalg.det(Q[selected, :])))
        assert v_mv >= v_rrqr - 1e-10, f"maxvol_rank below RRQR: {v_mv:.4e} < {v_rrqr:.4e}"
        if v_rrqr > 1e-12:
            ratios.append(v_mv / v_rrqr)
        if abs(float(np.linalg.det(Q[SpectralTracker.maxvol(Q), :]))) < v_rrqr - 1e-10:
            lu_losses += 1
    return (
        f"200 pools, maxvol_rank/RRQR volume: mean {float(np.mean(ratios)):.3f}x, "
        f"max {max(ratios):.3f}x, min {min(ratios):.3f}x; "
        f"LU-warm-started MaxVol would have lost to RRQR {lu_losses}/200 times"
    )


def test_stale_z_column_cannot_go_ragged():
    """A z column from a basis that has since been re-drawn must be refused.

    Regression: the deflation planner hands add_edge a z whose width is the
    multiplicity frozen at plan time. If the basis is re-drawn before that
    column is committed -- an aborted batch takes its deferred anchor
    immediately, at a smaller r -- the width no longer matches the accumulator.
    Appending it anyway left _z_accum_cols ragged, which surfaced only on the
    next ordinary edge as a ValueError inside np.stack in sigma_r_accum.
    """
    A = k24()
    tr = SpectralTracker(A)
    r = tr.multiplicity
    U = tr.U[:, :r]
    nz = [(i, j) for i, j in itertools.combinations(range(6), 2) if not A[i, j]]
    Z = np.array([U[i] - U[j] for i, j in nz]).T
    order = tr.rrqr_rank(Z)
    sel = [nz[t] for t in order[: r - 1]]
    # A column of the frozen r=3 basis that the batch itself does not consume.
    stale_z = Z[:, order[r - 1]].copy()

    tr.begin_deflation_batch(len(sel))
    for (i, j) in sel:
        tr.add_edge(i, j)
    assert tr.multiplicity == 1, f"batch did not reach r=1 (r={tr.multiplicity})"

    tr._reset_z_accum()  # what an anchor does: re-draw the basis at the new r
    assert tr._z_accum_r == 1 and stale_z.shape[0] == r, "test fixture is not stale"

    rest = [e for e in nz if e not in sel]
    tr.add_edge(*rest[0], z=stale_z)  # stale width r, accumulator is width 1
    tr.add_edge(*rest[1])             # ordinary edge, width _z_accum_r

    widths = {int(c.shape[0]) for c in tr._z_accum_cols}
    assert len(widths) <= 1, f"accumulator went ragged: widths {sorted(widths)}"
    assert widths <= {tr._z_accum_r}, (
        f"columns of width {sorted(widths)} in a width-{tr._z_accum_r} accumulator"
    )
    sigma = tr.sigma_r_accum()  # must not raise
    assert np.isfinite(sigma), f"sigma_r_accum returned {sigma}"
    return (
        f"stale width-{r} z refused by a width-{tr._z_accum_r} accumulator; "
        f"sigma_r = {sigma:.6f}"
    )


def test_batch_defers_exact_anchor():
    """A mid-batch anchor must be held back so the frozen basis survives."""
    A = k24()
    tr = SpectralTracker(A, exact_reset_every=1)  # anchor due every step
    r = tr.multiplicity
    U = tr.U[:, :r]
    nz = [(i, j) for i, j in itertools.combinations(range(6), 2) if not A[i, j]]
    Z = np.array([U[i] - U[j] for i, j in nz]).T
    sel = [nz[t] for t in tr.rrqr_rank(Z)[: r - 1]]

    tr.begin_deflation_batch(len(sel))
    frozen = tr._z_accum_U.copy()
    tr.add_edge(*sel[0])
    assert tr._deferred_exact, "anchor should have been deferred inside the batch"
    assert np.array_equal(tr._z_accum_U, frozen), "deferred anchor still rotated the basis"
    tr.add_edge(*sel[1])
    assert not tr._batch_active, "batch should have closed"
    assert not tr._deferred_exact, "deferred anchor should have been taken at batch end"
    return "anchor deferred through batch, taken at batch end"


# --------------------------------------------------------------------------- #
#  4. Long-horizon drift with the anchor suppressed
# --------------------------------------------------------------------------- #

def test_long_horizon_drift():
    rng = np.random.default_rng(7)
    n = 40
    A = random_connected(n, 0.12, rng)
    tr = SpectralTracker(A, exact_reset_every=10**9, drift_threshold=1e9)
    rows = []
    for step in range(200):
        nz = [(i, j) for i, j in itertools.combinations(range(n), 2) if not tr.adj[i, j]]
        if not nz:
            break
        i, j = nz[rng.integers(len(nz))]
        tr.add_edge(i, j)
        if (step + 1) % 50 == 0:
            rows.append((step + 1, tr.cert_foster, tr.cert_trace,
                         tr.cert_k_sync, tr._drift_residual()))
    lines = ["    step   foster     trace      k_sync     O(n^3) resid"]
    for st, f, t, k, d in rows:
        lines.append(f"    {st:5d}  {f:.3e}  {t:.3e}  {k:.3e}  {d:.3e}")
    final = rows[-1]
    assert final[4] < 1e-8, f"true residual drifted to {final[4]:.3e}"
    assert final[1] < 1e-8, f"Foster drifted to {final[1]:.3e}"
    return "200 edges, no anchor:\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
#  5. Cost
# --------------------------------------------------------------------------- #

def test_cost_scaling():
    """add_edge cost, against the O(n^3) drift check it no longer pays per step."""
    rng = np.random.default_rng(9)
    lines = ["    n    ms/add_edge   ms/O(n^3) check   would have been"]
    for n in (32, 64, 128, 256):
        A = random_connected(n, 0.1, rng)
        tr = SpectralTracker(A, exact_reset_every=10**9, drift_threshold=1e9)
        nz = [(i, j) for i, j in itertools.combinations(range(n), 2) if not tr.adj[i, j]]
        rng.shuffle(nz)
        picks = nz[:30]
        t0 = time.perf_counter()
        for (i, j) in picks:
            tr.add_edge(i, j)
        dt = (time.perf_counter() - t0) / len(picks) * 1e3

        t0 = time.perf_counter()
        for _ in range(10):
            tr._drift_residual()
        dr = (time.perf_counter() - t0) / 10 * 1e3
        lines.append(f"    {n:4d}  {dt:9.3f}   {dr:13.3f}   {dt + dr:14.3f}")
    return "\n".join(lines)


TESTS = [
    test_delta_p_exact,
    test_delta_rg_and_pmin_exact,
    test_foster_invariant_holds,
    test_secular_delta_lambda2,
    test_plateau_exact,
    test_batch_deflation_cascade,
    test_batch_basis_is_frozen,
    test_stale_z_column_cannot_go_ragged,
    test_maxvol_precondition,
    test_maxvol_dominance,
    test_maxvol_beats_rrqr_volume,
    test_batch_defers_exact_anchor,
    test_long_horizon_drift,
    test_cost_scaling,
]


def main() -> int:
    passed = 0
    for fn in TESTS:
        try:
            detail = fn()
        except AssertionError as exc:
            print(f"FAIL  {fn.__name__}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
            continue
        passed += 1
        print(f"PASS  {fn.__name__}" + (f"  --  {detail}" if detail else ""))
    print(f"\n{passed}/{len(TESTS)} passed")
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
