"""Analytic edge-selection state implementing ``algorithm.md`` exactly.

This module is a from-scratch implementation of the "Complete Edge Addition
Algorithm" specification (``algorithm.md``).  It replaces the previous
``analytic_spectral.py`` module and provides the deterministic candidate
narrowing used inside the RL loop:

  * Section "Preliminaries"  - Laplacian, eigenvalues/eigenvectors, non-edge
    incidence vectors, effective resistance ``R_e``, total effective
    resistance ``R_G = n * tr(L^+)``.
  * Section "Initialisation" - full eigendecomposition once, λ₂-multiplicity
    ``r`` (ε_mult = 1e-12, absolute), ``U``, ``mu = λ_{r+1}``, pseudoinverse
    ``L^+``, ``inv``/``inv2`` vectors, initial ``R_G`` and ``p_min``, drift
    control (τ_drift = 1e-6, max 32 incremental steps).
  * Section "Main Loop / Candidate Screening (Stage 1)" - ``R_e`` and the
    multispectral Fiedler score ``F`` over *every* unused non-edge; candidate
    set = unique(top-32 by R_e ∪ top-32 by F), ≤ 64.
  * Section "Second-Stage Selection":
      - Case A (r = 1): per candidate Δλ₂ (secular equation, Brent; zero
        projection guard |ψ₂| < 1e-14 ⇒ Δλ₂ = 0), ΔR_G via the fast
        Sherman-Morrison pseudoinverse update (reported as the positive
        reduction R_G^old - R_G^new), and a *cheap first-order* Δp_min.  The
        reward  R_c = α₁ Δλ₂/n + α₂ η_R ΔR_G/n² + α₃ η_p Δp_min  ranks the
        candidates; the top ``top`` (default 32) are returned for the policy.
      - Case B (r > 1): ``Z_mult = Uᵀ B'``, column-norm percentile filter,
        MaxVol row selection (tolerance 1e-6, max 200 iterations) → r edges,
        discard one → r-1 edges, returned as the candidate set for the policy.
  * Section "Single-Edge Update" / Case B - rank-1 Laplacian and pseudoinverse
    update (Sherman-Morrison + centering projection), exact Fiedler-vector
    secular update, Rayleigh-Ritz eigenspace refinement, ``mu`` refresh via
    ARPACK ``eigsh``, drift monitoring with full recomputation.

Conventions fixed with the caller (see the Q&A in the task):

  * ε_mult = 1e-12 absolute (spec).
  * Secular bracket margin = max(1e-12, 1e-8·gap) (existing implementation).
  * MaxVol max iterations = 200 (existing implementation).
  * ``mu`` refresh uses ARPACK ``eigsh``.
  * Max incremental steps between forced full recomputations = 32.
  * ΔR_G is expressed as the positive reduction R_G^old - R_G^new.
  * p_min = min_i [1 - (1/2) Σ_{j~i} R_eff(i,j)]  (resistance curvature).
  * Cheap Δp_min is used only for candidate scoring (Case A); the exact
    p_min recomputation happens after an edge is actually added.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from scipy.linalg import lu as _scipy_lu
except Exception:
    _scipy_lu = None

try:  # pragma: no cover - optional dependency
    from scipy.optimize import brentq as _brentq
except Exception:
    _brentq = None

from .graph_math import laplacian, non_edges

# --------------------------------------------------------------------------- #
#  Constants from the spec / confirmed choices
# --------------------------------------------------------------------------- #

EPS_MULT_DEFAULT = 1e-12      # λ₂ multiplicity tolerance (absolute, spec §2)
SECULAR_EPS_DEFAULT = 1e-12   # secular bracket lower margin epsilon
ZERO_PROJ_GUARD = 1e-14       # |ψ₂| below this ⇒ Fiedler vector does not move
TAU_DRIFT_DEFAULT = 1e-6      # drift residual threshold
MAX_STEPS_WITHOUT_RESET_DEFAULT = 32
MAXVOL_EPS_DEFAULT = 1e-6     # MaxVol stopping tolerance
MAXVOL_ITER_DEFAULT = 200     # MaxVol max iterations (confirmed)

# Default selection-reward weights (convex combination, confirmed).
ALPHA1_DEFAULT = 0.10
ALPHA2_DEFAULT = 0.20
ALPHA3_DEFAULT = 1.0 - ALPHA1_DEFAULT - ALPHA2_DEFAULT  # 0.70


# --------------------------------------------------------------------------- #
#  Pure spectral helpers
# --------------------------------------------------------------------------- #


def lambda2_multiplicity(evals: np.ndarray, eps_mult: float = EPS_MULT_DEFAULT) -> int:
    """Multiplicity ``r`` of λ₂, excluding the trivial λ₁ = 0.

    Spec §2: ``r = #{λ_i | |λ_i - λ₂| < ε_mult}`` (absolute tolerance).
    """
    evals = np.asarray(evals, dtype=np.float64)
    n = evals.shape[0]
    if n < 2:
        return 1
    lambda2 = float(evals[1])
    mask = np.abs(evals - lambda2) < eps_mult
    mask[0] = False
    return int(np.sum(mask))


def compute_p(adj: np.ndarray, L_plus: np.ndarray) -> np.ndarray:
    """Resistance-curvature vector  p_i = 1 - (1/2) Σ_{j~i} R_eff(i,j).

    ``R_eff(i,j) = aᵀ L⁺ a`` with ``a = e_i - e_j``, i.e.
    ``R_eff(i,j) = L⁺[i,i] + L⁺[j,j] - 2 L⁺[i,j]``.  Only existing edges
    contribute to the sum over neighbours.

    Vectorised: the whole all-pairs resistance matrix is one outer-sum, and
    masking it by the adjacency picks out exactly the neighbour terms. The
    previous version was a Python double loop over nodes x neighbours, which
    made this O(|E|) *interpreted* operations on every edge addition.

    Returns the full vector rather than just its minimum because the caller
    needs Σ_i p_i for the Foster drift certificate -- it was already being
    built and thrown away.

    No clamp on a negative R_eff: that is drift, and clamping it would
    silently repair the Foster invariant the certificate relies on.
    """
    adj = np.asarray(adj)
    Lp = np.asarray(L_plus, dtype=np.float64)
    d = np.diag(Lp)
    R = d[:, None] + d[None, :] - 2.0 * Lp
    return 1.0 - 0.5 * np.sum(np.where(adj > 0, R, 0.0), axis=1)


def compute_p_min(adj: np.ndarray, L_plus: np.ndarray) -> float:
    """min_i p_i.  Thin wrapper over :func:`compute_p`."""
    return float(np.min(compute_p(adj, L_plus)))


def p_min_for_adj(adj: np.ndarray, eps: float = 1e-14) -> float:
    """Exact p_min for an adjacency matrix (via full eigendecomposition).

    Used by the calibration script and validation tests.
    """
    L = laplacian(adj)
    evals, evecs = np.linalg.eigh(L)
    inv = np.zeros_like(evals, dtype=np.float64)
    inv[1:] = 1.0 / np.maximum(evals[1:], eps)
    L_plus = evecs[:, 1:] @ np.diag(inv[1:]) @ evecs[:, 1:].T
    return compute_p_min(adj, L_plus)


# --------------------------------------------------------------------------- #
#  Secular equation (Case A)
# --------------------------------------------------------------------------- #


def _secular_f(lam: float, psi: np.ndarray, lambdas: np.ndarray) -> float:
    """f(λ) = 1 + Σ_{i=2}^n ψ_i²/(λ_i - λ).  Index 0 contributes 0 (a ⟂ 1)."""
    return 1.0 + float(np.sum(psi * psi / (lambdas - lam)))


def secular_upper_endpoint(
    lambdas: np.ndarray,
    lambda2: float,
    mu: float,
    eps_mult: float = EPS_MULT_DEFAULT,
) -> float:
    """Pole-free upper end of the secular bracket: ``min(mu, next stored λ)``.

    ``f(λ) = 1 + Σ ψ_i²/(λ_i - λ)`` has its poles at the *stored* ``lambdas``,
    so a valid bracket may not contain one.  ``mu`` alone is not enough: it
    comes from ``_refresh_mu``, which runs a dense solve on the **current** L,
    while ``lambdas`` is the basis carried since the last exact anchor.  Once
    those disagree -- i.e. on every step after the first past an anchor -- a
    stale ``lambdas[2]`` can sit strictly inside (λ₂, μ).  f is then
    non-monotonic on the bracket with two roots in it, and the root-finder
    converges to whichever one it stumbles into: measured at n=32, 7 edges in,
    ``brentq`` returned 2.790662 and a safeguarded Newton 2.790498 for the same
    edge whose exact λ₂' is 2.790509.

    Interlacing pins the answer down: for the rank-1 PSD update L' = L + aaᵀ,
    λ_k ≤ λ'_k ≤ λ_{k+1}, so λ₂' lies below the next stored eigenvalue and the
    first pole-free interval above λ₂ is the one that contains it.

    When the basis is exact -- always true immediately after
    ``_recompute_exact``, where r = 1 gives ``mu = evals[2]`` -- this returns
    ``mu`` unchanged, so the common path is bit-identical to the old behaviour.
    """
    lambdas = np.asarray(lambdas, dtype=np.float64)
    above = lambdas[lambdas > float(lambda2) + float(eps_mult)]
    if above.size == 0:
        return float(mu)
    return min(float(mu), float(np.min(above)))


def _brent_bracket(
    psi: np.ndarray,
    lambdas: np.ndarray,
    lambda2: float,
    mu: float,
    eps: float = SECULAR_EPS_DEFAULT,
) -> Tuple[Optional[float], Optional[float]]:
    """Bracket (a, b) inside (λ₂, μ) with f(a)·f(b) < 0, or (None, None).

    The bracket margin is ``max(eps, 1e-8 * gap)`` (existing implementation).
    """
    mu = secular_upper_endpoint(lambdas, lambda2, mu)
    gap = float(mu) - float(lambda2)
    if not np.isfinite(gap) or gap <= 0.0:
        return None, None
    margin = max(eps, 1e-8 * gap)
    a = float(lambda2) + margin
    b = float(mu) - margin
    if not (b > a):
        return None, None
    fa = _secular_f(a, psi, lambdas)
    fb = _secular_f(b, psi, lambdas)
    if not (np.isfinite(fa) and np.isfinite(fb)):
        return None, None
    if fa * fb > 0.0:
        return None, None
    return a, b


def _bisection(f, a: float, b: float, tol: float = 1e-14, max_iter: int = 300) -> float:
    fa = f(a)
    fb = f(b)
    if fa * fb > 0.0:
        raise ValueError("bisection requires a sign change")
    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = f(c)
        if abs(fc) < tol or abs(b - a) < tol:
            return c
        if fa * fc < 0.0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5 * (a + b)


def solve_secular_equation(
    psi: np.ndarray,
    lambdas: np.ndarray,
    lambda2: float,
    mu: float,
    eps: float = SECULAR_EPS_DEFAULT,
    tol: float = 1e-13,
) -> float:
    """New λ₂' in (λ₂, μ) solving ``1 + Σ ψ_i²/(λ_i - λ) = 0``.

    Uses Brent's method (scipy) with a bisection fallback.  If there is no
    sign change inside the open bracket (λ₂, μ):

      * ψ₂ ≈ 0 (Fiedler projection guard) ⇒ returns ``lambda2`` unchanged
        (Δλ₂ = 0);
      * the new λ₂ collides exactly with the old ``mu`` (typically ψ₃ ≈ 0, so
        f stays negative up to μ and the root sits on the μ endpoint) ⇒
        returns ``mu`` so the state tracks the true λ₂.

    The μ-endpoint collision is a known gap in the open-bracket formulation of
    the spec (the previous analytic implementation silently reported Δλ₂ = 0
    for such edges); it is handled here and documented in the final report.
    """
    psi = np.asarray(psi, dtype=np.float64)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    if not np.isfinite(float(mu)):
        # No eigenvalue above λ₂ (``_refresh_mu`` returns inf). There is no
        # upper bracket, so the secular root is not defined -- leave λ₂ put.
        return float(lambda2)
    # Poles of f sit at the *stored* ``lambdas``; cap the endpoint so the
    # bracket cannot contain one (see ``secular_upper_endpoint``).
    mu = secular_upper_endpoint(lambdas, lambda2, mu)
    a, b = _brent_bracket(psi, lambdas, float(lambda2), float(mu), eps=eps)
    if a is None or b is None:
        if abs(float(psi[1])) >= ZERO_PROJ_GUARD and float(mu) > float(lambda2):
            gap = float(mu) - float(lambda2)
            boundary = float(mu) - max(eps, 1e-8 * gap)
            fb = _secular_f(boundary, psi, lambdas)
            if np.isfinite(fb) and fb < 0.0:
                # Root sits on the upper endpoint μ (λ₂' = μ exactly).
                return float(mu)
        return float(lambda2)

    f = lambda t: _secular_f(t, psi, lambdas)
    if _brentq is not None:
        try:
            return float(_brentq(f, a, b, xtol=tol, rtol=4e-15, maxiter=200))
        except Exception:  # pragma: no cover - fall back to bisection
            pass
    return float(_bisection(f, a, b, tol=tol))


def solve_secular_equation_batch(
    Psi: np.ndarray,
    lambdas: np.ndarray,
    lambda2: float,
    mu: float,
    eps: float = SECULAR_EPS_DEFAULT,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> np.ndarray:
    """Vectorised :func:`solve_secular_equation` over many candidate edges.

    ``Psi`` is (n, m): column ``c`` is the projection ψ = Qᵀa_c for candidate
    ``c``.  Returns the (m,) vector of roots, with the *same* three-way
    semantics as the scalar solver (root / λ₂ unchanged / μ-endpoint collision).

    Why this exists
    ---------------
    The scalar path drives ``scipy.optimize.brentq`` with a Python callable,
    once per candidate.  At the sizes here that is ~58 candidates x ~12.6
    iterations = ~730 interpreter round-trips per environment step, each doing
    64 flops behind ~3.3us of NumPy dispatch -- a ~150:1 overhead ratio, and it
    dominated the whole module.  Every candidate is an *independent scalar* root
    find over the same ``lambdas``, which is exactly the shape that vectorises:
    one (n, m) array op replaces m callbacks.  ``psi*psi`` is also loop
    invariant and was being rebuilt on every Brent iteration; here it is formed
    once.

    Method
    ------
    Safeguarded Newton, not plain Newton and not Brent:

      * f'(λ) = Σ ψ_i²/(λ_i - λ)² > 0, so f is *strictly increasing* between
        consecutive poles.  On (λ₂, μ) it runs -inf -> +inf, giving exactly one
        root and a guaranteed bracket -- and the sign of f alone says which
        side of the root we are on, which is what makes the bracket update
        vectorise (``f < 0`` => raise the floor, ``f > 0`` => lower the ceiling).
      * f''(λ) = Σ 2ψ_i²/(λ_i - λ)³ is *not* single-signed on (λ₂, μ): the pole
        below contributes negatively, the poles above positively.  So plain
        Newton is not unconditionally convergent here (this is why the
        literature -- and ``algorithm.md`` -- reach for BNS rational
        interpolation).  Keeping the bracket and falling back to its midpoint
        whenever a Newton step leaves it restores unconditional convergence
        without the extra machinery.

    A non-finite f or f' (a stale ``evals`` entry landing inside the bracket)
    fails the ``ok`` test and bisects, so it degrades rather than diverging.
    """
    Psi = np.asarray(Psi, dtype=np.float64)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    if Psi.ndim != 2:
        raise ValueError(f"Psi must be (n, m), got shape {Psi.shape}")
    m = int(Psi.shape[1])
    lambda2 = float(lambda2)
    mu = float(mu)

    out = np.full(m, lambda2, dtype=np.float64)
    if m == 0 or not np.isfinite(mu):
        return out

    # Bracket endpoints depend only on (λ₂, μ, eps), so they are shared by all
    # columns -- same margin rule and same pole-free cap as the scalar path.
    mu = secular_upper_endpoint(lambdas, lambda2, mu)
    gap = mu - lambda2
    if not np.isfinite(gap) or gap <= 0.0:
        return out
    margin = max(float(eps), 1e-8 * gap)
    a = lambda2 + margin
    b = mu - margin
    if not (b > a):
        return out

    Psi2 = Psi * Psi                                  # (n, m), loop invariant

    with np.errstate(divide="ignore", invalid="ignore"):
        fa = 1.0 + np.sum(Psi2 / (lambdas[:, None] - a), axis=0)
        fb = 1.0 + np.sum(Psi2 / (lambdas[:, None] - b), axis=0)

    # ``_brent_bracket`` rejects only on ``fa*fb > 0``, so an exact endpoint
    # root (product == 0) still counts as bracketed.
    ok_ends = np.isfinite(fa) & np.isfinite(fb)
    bracketed = ok_ends & ~(fa * fb > 0.0)

    # μ-endpoint collision: no sign change, ψ₂ not degenerate, and f still
    # negative at the top of the bracket => the root sits on μ itself.
    collided = (
        (~bracketed)
        & (np.abs(Psi[1, :]) >= ZERO_PROJ_GUARD)
        & np.isfinite(fb)
        & (fb < 0.0)
    )
    out[collided] = mu
    # Remaining unbracketed columns keep λ₂ (``out`` is pre-filled with it).

    idx = np.flatnonzero(bracketed)
    if idx.size == 0:
        return out

    # ---- Iterate on the pole-free reformulation, not on f itself ----
    #
    # Newton applied directly to f stalls: near the λ₂ pole f is almost
    # vertical, every Newton step lands outside the bracket, and the safeguard
    # falls back to bisection -- which needs ~40 halvings to reach ``tol`` on a
    # gap of O(1).  Measured at n=64: median 5 iterations but a maximum of 40,
    # and at n=16 a maximum of 67.
    #
    # Substituting η = λ - λ₂ and splitting the λ₂ pole out of the sum,
    #
    #     f = 1 - ψ₂²/η + g(λ),      g(λ) = Σ_{i≠1} ψ_i²/(λ_i - λ)
    #
    # and multiplying through by η > 0 gives an equivalent equation with no
    # pole in it at all:
    #
    #     h(η) = η·(1 + g) - ψ₂² = 0,    h'(η) = 1 + g + η·g'
    #
    # ``h = η·f`` with η > 0, so h and f have identical signs -- the bracketing
    # decided above from f transfers unchanged, and the same monotonicity
    # argument applies.  On a pole-free interval every surviving term of g has
    # λ_i ≥ μ > λ (index 0 contributes nothing, ψ₀ = 0), so g > 0 and g' > 0,
    # hence h' > 1 > 0: h is strictly increasing and well scaled exactly where
    # f is not.  This drops the worst-case iteration count from 40-67 to 5-12.
    P2 = Psi2[:, idx]                                 # (n, k)
    psi2_sq = P2[1, :].copy()                         # the λ₂ pole numerator
    P_rest = P2.copy()
    P_rest[1, :] = 0.0                                # g omits the λ₂ term

    # ``act`` indexes the columns still iterating.  Convergence is uneven even
    # after the reformulation, and the (n, |act|) work per iteration shrinks
    # with the active set, so compaction makes the cost track the mean
    # iteration count rather than the maximum.
    act = np.arange(idx.size)
    lo = np.full(idx.size, a - lambda2, dtype=np.float64)
    hi = np.full(idx.size, b - lambda2, dtype=np.float64)
    eta = 0.5 * (lo + hi)

    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(int(max_iter)):
            Pa = P_rest[:, act]
            eta_a, lo_a, hi_a = eta[act], lo[act], hi[act]

            D = lambdas[:, None] - (lambda2 + eta_a)[None, :]   # (n, |act|)
            T = Pa / D
            g = T.sum(axis=0)
            g_prime = (T / D).sum(axis=0)

            hv = eta_a * (1.0 + g) - psi2_sq[act]
            hp = 1.0 + g + eta_a * g_prime

            # h strictly increasing => its sign places η relative to the root.
            lo_a = np.where(hv < 0.0, eta_a, lo_a)
            hi_a = np.where(hv > 0.0, eta_a, hi_a)

            step = eta_a - hv / hp
            accept = np.isfinite(step) & (step > lo_a) & (step < hi_a)
            eta_new = np.where(accept, step, 0.5 * (lo_a + hi_a))

            # Terminate on the *step size*, not the bracket width: Newton
            # closes on this root from one side, so only one of lo/hi moves and
            # ``hi - lo`` stalls at the distance to the far endpoint -- a width
            # test never fires and burns every iteration on a converged answer.
            delta = np.abs(eta_new - eta_a)

            eta[act], lo[act], hi[act] = eta_new, lo_a, hi_a
            act = act[delta > tol * np.maximum(1.0, np.abs(lambda2 + eta_new))]
            if act.size == 0:
                break

    out[idx] = lambda2 + eta
    return out


def pole_collision_index(
    lambdas: np.ndarray,
    lambda2_prime: float,
    psi: np.ndarray,
    tol: float = 1e-13,
) -> Optional[int]:
    """Index of the stored eigenvalue λ₂' has landed on, or ``None``.

    On a μ-endpoint collision the solvers return the bracket's *upper endpoint*
    rather than an interior root, and ``secular_upper_endpoint`` may have set
    that endpoint to a stored ``lambdas[k]`` -- i.e. to a pole of f.  The
    secular eigenvector sum then divides by exactly zero at ``k``.

    Ties (a repeated stored eigenvalue) are broken towards the smallest |ψ|,
    which is the direction that actually survives the update: ψ_k = q_kᵀa ≈ 0
    is what put the root on the endpoint in the first place.
    """
    lambdas = np.asarray(lambdas, dtype=np.float64)
    lam = float(lambda2_prime)
    if not np.isfinite(lam):
        return None
    thresh = float(tol) * max(1.0, abs(lam))
    hit = np.flatnonzero(np.abs(lambdas - lam) <= thresh)
    hit = hit[hit >= 1]                      # index 0 is the null vector
    if hit.size == 0:
        return None
    psi = np.asarray(psi, dtype=np.float64)
    return int(hit[np.argmin(np.abs(psi[hit]))])


def update_fiedler_vector(
    psi: np.ndarray,
    lambdas: np.ndarray,
    lambda2_prime: float,
    Q: np.ndarray,
) -> np.ndarray:
    """Exact new Fiedler vector: u₂' = Σ_{i=2}^n ψ_i/(λ_i - λ₂') q_i, normalised."""
    psi = np.asarray(psi, dtype=np.float64)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    # λ₂' sitting on a stored eigenvalue makes the sum 0/0 at that term.  The
    # limit is that eigenvector itself: with ψ_k = q_kᵀa ≈ 0,
    # L'q_k = Lq_k + a(aᵀq_k) = λ_k q_k + ψ_k a ≈ λ_k q_k, so q_k is already an
    # eigenvector of the updated L at λ₂'.  Callers are expected to catch the
    # collision before getting here; this keeps the function total either way.
    k = pole_collision_index(lambdas, lambda2_prime, psi)
    if k is not None:
        u = Q[:, k].copy()
        nrm = float(np.linalg.norm(u))
        return u / nrm if nrm > 1e-14 else u

    coeffs = psi / (lambdas - float(lambda2_prime))
    u = Q @ coeffs
    norm = float(np.linalg.norm(u))
    # ``norm`` can be inf (a near-pole coefficient overflowed) or nan (±inf
    # cancelling inside the matvec); both must fall back rather than divide.
    if np.isfinite(norm) and norm > 1e-14 and np.isfinite(u).all():
        u = u / norm
    else:
        u = Q[:, 1].copy()
    return u


# --------------------------------------------------------------------------- #
#  Case B - MaxVol selection helpers
# --------------------------------------------------------------------------- #


def screen_candidate_indices(
    r_e: np.ndarray,
    f_score: np.ndarray,
    top: int = 32,
    cap: int = 64,
) -> np.ndarray:
    """I_cand = unique(top-``top`` by R_e ∪ top-``top`` by F), truncated to ``cap``.

    ``top`` = 32 and ``cap`` = 64 per spec Stage 1.  Because each top-32 set
    has at most 32 elements, the union has at most 64 elements, so the cap is
    effectively a safety bound; when fewer non-edges remain, all are kept.
    """
    r_e = np.asarray(r_e, dtype=np.float64)
    f_score = np.asarray(f_score, dtype=np.float64)
    if r_e.size == 0:
        return np.zeros((0,), dtype=np.int64)

    def _topk(scores: np.ndarray, k: int) -> np.ndarray:
        if k >= scores.size:
            return np.arange(scores.size, dtype=np.int64)
        return np.argsort(-scores, kind="mergesort")[:k].astype(np.int64, copy=False)

    i_r = _topk(r_e, top)
    i_f = _topk(f_score, top)
    union = np.unique(np.concatenate([i_r, i_f]))
    if union.size > cap:
        union = union[:cap]
    return union.astype(np.int64, copy=False)


def filter_high_potential_columns(Z_mult: np.ndarray, r: int) -> np.ndarray:
    """Keep columns of Z_mult with column norm ≥ τ (τ = 75th percentile).

    If fewer than ``r`` columns survive, lower τ to the 50th then 25th
    percentile until at least ``r`` columns remain (spec Case B step 2).
    """
    Z_mult = np.asarray(Z_mult, dtype=np.float64)
    m = Z_mult.shape[1]
    if m == 0:
        return np.zeros((0,), dtype=np.int64)
    if m <= r:
        return np.arange(m, dtype=np.int64)

    norms = np.linalg.norm(Z_mult, axis=0)
    for q in (75, 50, 25):
        tau = float(np.percentile(norms, q))
        keep = np.flatnonzero(norms >= tau).astype(np.int64, copy=False)
        if keep.size >= r:
            return keep
    # Fallback: keep the r columns with the largest norms.
    return np.argsort(-norms)[:r].astype(np.int64, copy=False)


def maxvol_selection(
    Q_R: np.ndarray,
    eps: float = MAXVOL_EPS_DEFAULT,
    max_iter: int = MAXVOL_ITER_DEFAULT,
) -> np.ndarray:
    """MaxVol row selection (spec Case B step 3).

    ``Q_R`` is the thin-QR factor of ``Z_keepᵀ``, shape (m, r) with
    orthonormal columns.  Returns the indices (size r) of the selected rows
    of ``Q_R``, which correspond to selected columns of ``Z_keep``.

    Initial active set comes from LU with row pivoting:  ``P Q_R = L U`` and
    row ``i`` of ``P Q_R`` is row ``p[i]`` of ``Q_R``;  ``I_init = {p[0..r-1]}``.
    """
    Q_R = np.asarray(Q_R, dtype=np.float64)
    m, r = Q_R.shape
    if r <= 0 or m == 0:
        return np.zeros((0,), dtype=np.int64)
    if m <= r:
        return np.arange(m, dtype=np.int64)

    # ---- Initial active set via LU with row pivoting ----
    if _scipy_lu is not None:
        try:
            P, _, _ = _scipy_lu(Q_R)
            # Row i of (P Q_R) is row p[i] of Q_R, so p[i] = argmax_j P[i, j].
            p = np.argmax(P, axis=1)
            I = [int(x) for x in p[:r]]
        except Exception:  # pragma: no cover
            I = list(range(r))
    else:  # pragma: no cover - fallback without scipy
        Q = Q_R.copy()
        cols = list(range(r))
        I = []
        for _ in range(r):
            norms = np.linalg.norm(Q, axis=0)
            c = int(np.argmax(norms))
            v = Q[:, c].copy()
            v /= max(float(np.linalg.norm(v)), 1e-30)
            I.append(c)
            cols.remove(c)
            Q = Q - np.outer(v, v @ Q)
            Q = np.delete(Q, c, axis=1)

    # ---- MaxVol iteration ----
    for _ in range(max_iter):
        try:
            QI_inv = np.linalg.inv(Q_R[I, :])
        except np.linalg.LinAlgError:  # pragma: no cover
            break
        C = Q_R @ QI_inv
        absC = np.abs(C)
        notI = [i for i in range(m) if i not in I]
        if not notI:
            break
        # Columns of C are positions 0..r-1 within the active set I.
        sub = absC[np.ix_(notI, list(range(r)))]
        amax = float(np.max(sub))
        if amax <= 1.0 + eps:
            break
        loc = int(np.argmax(sub))
        i_loc, j_loc = np.unravel_index(loc, sub.shape)
        i_max = notI[i_loc]
        j_max = I[j_loc]
        I[I.index(j_max)] = i_max

    return np.asarray(I, dtype=np.int64)


# --------------------------------------------------------------------------- #
#  Analytic state
# --------------------------------------------------------------------------- #


class AnalyticState:
    """Maintains the Laplacian spectral state following algorithm.md.

    Keeps an approximate full basis ``Q`` / ``evals`` that is updated with
    the exact rank-1 (secular) formulas (Case A) or Rayleigh-Ritz refinement
    (Case B), and triggers a full dense recomputation when the drift residual
    exceeds ``tau_drift`` or after ``max_steps_without_reset`` steps.
    """

    def __init__(
        self,
        adj: np.ndarray,
        tau_drift: float = TAU_DRIFT_DEFAULT,
        max_steps_without_reset: int = MAX_STEPS_WITHOUT_RESET_DEFAULT,
        eps_mult: float = EPS_MULT_DEFAULT,
        secular_eps: float = SECULAR_EPS_DEFAULT,
        maxvol_eps: float = MAXVOL_EPS_DEFAULT,
        maxvol_iter: int = MAXVOL_ITER_DEFAULT,
    ):
        self.n = int(adj.shape[0])
        if self.n < 2:
            raise ValueError("AnalyticState requires n >= 2")
        self.tau_drift = float(tau_drift)
        self.max_steps_without_reset = int(max_steps_without_reset)
        self.eps_mult = float(eps_mult)
        self.secular_eps = float(secular_eps)
        self.maxvol_eps = float(maxvol_eps)
        self.maxvol_iter = int(maxvol_iter)

        self.adj = adj.astype(np.uint8, copy=True)
        self.steps_since_exact = 0

        # Drift certificates (see _cheap_drift) and the secular bracket guard
        # (see _check_secular_bracket). Set by _recompute_exact / per step.
        self.p: np.ndarray = np.zeros(self.n, dtype=np.float64)
        self.cert_foster: float = 0.0
        self.cert_trace: float = 0.0
        self.drift_residual: float = 0.0
        self.last_secular_bracket: Optional[Tuple[float, float]] = None
        self.bracket_violations: int = 0

        self._recompute_exact()

    # ------------------------------------------------------------------ #
    #  Accessors (compatible with the previous SpectralTracker interface)
    # ------------------------------------------------------------------ #

    def get_spectral_features(self) -> Tuple[
        float, float, float,
        np.ndarray, np.ndarray, np.ndarray,
        float, int, np.ndarray,
    ]:
        lambda2 = float(self.evals[1])
        lambda3 = float(self.evals[2]) if self.n >= 3 else 0.0
        lambda4 = float(self.evals[3]) if self.n >= 4 else 0.0
        phi2 = self.Q[:, 1].copy()
        phi3 = self.Q[:, 2].copy() if self.n >= 3 else np.zeros(self.n, dtype=np.float64)
        phi4 = self.Q[:, 3].copy() if self.n >= 4 else np.zeros(self.n, dtype=np.float64)
        return (
            lambda2,
            lambda3,
            lambda4,
            phi2.astype(np.float64, copy=False),
            phi3.astype(np.float64, copy=False),
            phi4.astype(np.float64, copy=False),
            float(self.R_G),
            int(self.r),
            self.Q.copy().astype(np.float64),
        )

    def get_P_min(self) -> float:
        return float(self.p_min)

    @property
    def multiplicity(self) -> int:
        return int(self.r)

    def get_evals(self) -> np.ndarray:
        return self.evals.copy()

    def get_U(self) -> np.ndarray:
        return self.U.copy()

    # ------------------------------------------------------------------ #
    #  Initialisation (spec §2) / full recomputation
    # ------------------------------------------------------------------ #

    def _refresh_mu(self, lambda2: float) -> float:
        """First eigenvalue strictly greater than λ₂ (spec §2 step 3 / §3.2).

        Dense ``eigvalsh``. This used to run ARPACK ``eigsh`` (Lanczos) first and
        fall back to the dense spectrum, but Lanczos is the wrong tool at these
        sizes: it is an iterative method whose cost is dominated by per-iteration
        Python/SciPy overhead, and it was measured at ~87 iterations per edge on a
        32x32 matrix -- roughly 6x the cost of the LAPACK dense solve it was
        avoiding, and ~78% of all time spent in ``add_edge``. The dense path is
        also the exact answer rather than a Ritz estimate, which matters because
        ``mu`` is not purely a bracket endpoint here (see ``add_edge``: it is
        returned as λ₂' on a μ-collision and compared against a 1e-10 threshold).

        Returns ``inf`` when no eigenvalue lies above λ₂. The old code returned
        λ_max in that case, which is not a conservative bracket but a bracket
        spanning the whole spectrum -- every intervening pole included -- so
        Brent could converge to a root in the wrong interval. ``inf`` makes the
        caller skip the secular solve instead.
        """
        if self.n < 2:
            return float("inf")
        # Same tolerance the multiplicity test uses. These were 1e-9 here vs
        # eps_mult=1e-12 there, leaving a band in which an eigenvalue counted as
        # outside the λ₂ cluster yet not above λ₂ -- so μ skipped it and
        # overshot the pole at the true λ₃.
        tol = self.eps_mult
        vals = np.sort(np.linalg.eigvalsh(self.L))
        above = vals[vals > float(lambda2) + tol]
        return float(above[0]) if above.size > 0 else float("inf")

    def _recompute_exact(self) -> None:
        """Full dense eigendecomposition (spec §2)."""
        L = laplacian(self.adj)
        evals, evecs = np.linalg.eigh(L)

        self.L = L.astype(np.float64, copy=True)
        self.evals = evals.astype(np.float64, copy=True)
        self.Q = evecs.astype(np.float64, copy=True)

        self.r = int(lambda2_multiplicity(self.evals, self.eps_mult))
        self.U = self.Q[:, 1 : 1 + self.r].copy()

        # mu = λ_{r+1}: the first eigenvalue strictly greater than λ₂.
        if self.r + 1 < self.n:
            self.mu = float(self.evals[self.r + 1])
        else:
            self.mu = self._refresh_mu(float(self.evals[1]))

        self.inv_lambda = np.zeros(self.n, dtype=np.float64)
        self.inv_lambda_sq = np.zeros(self.n, dtype=np.float64)
        safe = np.maximum(self.evals[1:], 1e-14)
        self.inv_lambda[1:] = 1.0 / safe
        self.inv_lambda_sq[1:] = 1.0 / (safe * safe)

        self.L_plus = self.Q[:, 1:] @ np.diag(self.inv_lambda[1:]) @ self.Q[:, 1:].T
        self.R_G = float(self.n) * float(np.sum(self.inv_lambda[1:]))
        self.p = compute_p(self.adj, self.L_plus)
        self.p_min = float(np.min(self.p))
        self.steps_since_exact = 0
        self.cert_foster = 0.0
        self.cert_trace = 0.0

    def force_exact(self) -> None:
        self._recompute_exact()

    # ------------------------------------------------------------------ #
    #  Cheap first-order Δp_min for candidate scoring (Case A, spec step 3)
    # ------------------------------------------------------------------ #

    def _cheap_delta_pmin(self, i: int, j: int) -> float:
        """First-order approximation of Δp_min for a hypothetical edge (i, j).

        Only the new resistance R_eff(i, j) is added to p_i and p_j; all other
        effective resistances are taken from the *current* L⁺ (the L⁺ update
        is ignored).  Returns ``min(p_i^new, p_j^new, p_min) - p_min``.

        This is used only for candidate scoring in Case A, never for the exact
        p_min update after an edge is actually added.

        **Reference implementation only.**  ``select_candidates`` no longer
        calls this: because ``self.p`` already holds
        ``p_i = 1 - ½ Σ_{b~i} R_eff(i,b)`` for the current ``adj``/``L⁺`` (both
        ``_recompute_exact`` and ``add_edge`` refresh it in lockstep with
        ``L_plus``), and this rule holds ``L⁺`` fixed while appending exactly
        one term, the whole neighbour loop collapses to
        ``p_i - ½ R_eff(i,j)``.  The loop form ran ~1575 interpreted ``eff()``
        calls per step to rederive numbers the object already had.  Kept so the
        closed form has something to be tested against -- see
        ``test_delta_pmin_closed_form``.
        """
        Lp = self.L_plus
        nbrs_i = np.flatnonzero(self.adj[i] > 0)
        nbrs_j = np.flatnonzero(self.adj[j] > 0)

        def eff(u: int, v: int) -> float:
            return float(Lp[u, u] + Lp[v, v] - 2.0 * Lp[u, v])

        res_i = sum(eff(i, int(nb)) for nb in nbrs_i) + eff(i, j)
        res_j = sum(eff(j, int(nb)) for nb in nbrs_j) + eff(i, j)
        p_i_new = 1.0 - 0.5 * res_i
        p_j_new = 1.0 - 0.5 * res_j
        return min(p_i_new, p_j_new, float(self.p_min)) - float(self.p_min)

    # ------------------------------------------------------------------ #
    #  Candidate screening + second-stage selection (spec §3.1 / §3.2)
    # ------------------------------------------------------------------ #

    def _all_nonedge_incidences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (i_idx, j_idx) for every unused non-edge."""
        ne = non_edges(self.adj)
        if not ne:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
            )
        arr = np.asarray(ne, dtype=np.int64)
        i_idx = arr[:, 0]
        j_idx = arr[:, 1]
        return i_idx, j_idx

    def select_candidates(
        self,
        top: int = 32,
        *,
        alpha1: Optional[float] = None,
        alpha2: Optional[float] = None,
        alpha3: Optional[float] = None,
        eta_R: Optional[float] = None,
        eta_p: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Return the candidate edge pairs for the policy (≤ ``top`` pairs).

        Implements Stage 1 screening over all non-edges and Stage 2:

          * Case A (r = 1): rank candidates by the selection reward
            ``R_c = α₁ Δλ₂/n + α₂ η_R ΔR_G/n² + α₃ η_p Δp_min`` and keep the
            top ``top`` (default 32).
          * Case B (r > 1): the ``r-1`` MaxVol-selected edges are returned as
            the candidate set (the policy chooses one per environment step).

        Returns ``(pairs, info)`` where ``pairs`` is an (m, 2) int array of
        node indices and ``info`` carries per-candidate diagnostics.
        """
        i_idx, j_idx = self._all_nonedge_incidences()
        k = i_idx.size
        if k == 0:
            return np.zeros((0, 2), dtype=np.int64), {}

        # ---- Stage 1: R_e and multispectral Fiedler score ----
        # R_e is the effective resistance, read O(1) per non-edge from the
        # maintained L⁺ (identical to inv·(ψ⊙ψ)).  F lives in the λ₂
        # eigenspace U.  Both avoid forming the full n×k projection ψ, so the
        # screening is O(n²) rather than O(n·k) = O(n³).
        Lp = self.L_plus
        r_e = Lp[i_idx, i_idx] + Lp[j_idx, j_idx] - 2.0 * Lp[i_idx, j_idx]
        U_rows = self.U
        f_score = np.sum((U_rows[i_idx] - U_rows[j_idx]) ** 2, axis=1)
        i_cand = screen_candidate_indices(r_e, f_score, top=32, cap=64)
        m = int(i_cand.size)
        if m == 0:
            return np.zeros((0, 2), dtype=np.int64), {}

        # Full n-dimensional projection only for the ≤64 survivors (Stage 2).
        # ``i_cand`` indexes the non-edge list, so gather node rows of Q via
        # ``i_idx[i_cand]`` / ``j_idx[i_cand]``.
        psi = (self.Q[i_idx[i_cand]] - self.Q[j_idx[i_cand]]).T  # (n, m)

        nf = float(max(1, self.n))
        a1 = ALPHA1_DEFAULT if alpha1 is None else float(alpha1)
        a2 = ALPHA2_DEFAULT if alpha2 is None else float(alpha2)
        a3 = ALPHA3_DEFAULT if alpha3 is None else float(alpha3)
        eta_r = 1.0 if eta_R is None else float(eta_R)
        eta_p = 1.0 if eta_p is None else float(eta_p)

        if self.r == 1:
            # ---- Case A: reward-ranked top-``top`` candidates ----
            Bp = psi                             # (n, m)
            # ΔR_G in the eigenbasis, exactly as algorithm.md writes it:
            #   β    = 1 + Σ_i ψ_i²/λ_i          (= 1 + aᵀL⁺a = 1 + R_eff(i,j))
            #   ΔR_G = n · (Σ_i ψ_i²/λ_i²) / β   (= n · aᵀ(L⁺)²a / β)
            #
            # The previous form was ``V = L_plus @ Bp; beta = 1 + Σ Bp·V``,
            # which uses ψ = Qᵀa as though it were the incidence vector a and
            # so computes ψᵀL⁺ψ = aᵀ(Q L⁺ Qᵀ)a, not aᵀL⁺a.  Measured at n=24
            # that put β at 1.477 where all three correct forms give 1.753, and
            # left only 3 of the top-8 candidates by ΔR_G in common with the
            # true ordering.  ``inv_lambda``/``inv_lambda_sq`` are already
            # maintained for this, carry the λ₁ = 0 term as 0, and drop the
            # (n, m) matmul as a side effect.
            P2 = Bp * Bp                         # (n, m)
            beta = 1.0 + self.inv_lambda @ P2
            d_rg_red = (
                self.n * (self.inv_lambda_sq @ P2) / np.maximum(beta, 1e-14)
            )  # positive reduction R_G^old - R_G^new

            # Δλ₂ for every candidate in one batched safeguarded-Newton solve.
            # Guarded columns (|ψ₂| < 1e-14, spec "zero-projection guard") keep
            # Δλ₂ = 0 and are excluded rather than handed to the solver.
            lambda2 = float(self.evals[1])
            d_l2 = np.zeros(m, dtype=np.float64)
            active = np.flatnonzero(np.abs(Bp[1, :]) >= ZERO_PROJ_GUARD)
            if active.size > 0:
                roots = solve_secular_equation_batch(
                    Bp[:, active], self.evals, lambda2, self.mu, eps=self.secular_eps
                )
                d_l2[active] = roots - lambda2

            # Δp_min in closed form.  p_i = 1 - ½ Σ_{b~i} R_eff(i,b) is already
            # maintained in ``self.p``, and the first-order rule only appends
            # the one new term R_eff(i,j) while holding L⁺ fixed, so
            #     p_i^new = p_i - ½ R_eff(i,j),   p_j^new = p_j - ½ R_eff(i,j)
            # -- the same subtracted term on both endpoints, with R_eff(i,j)
            # already computed for every non-edge as ``r_e`` in Stage 1.  This
            # is ``_cheap_delta_pmin`` (kept below as the reference form)
            # rederived without recomputing the neighbour sums it already has.
            cand_i = i_idx[i_cand]
            cand_j = j_idx[i_cand]
            p_endpoint = (
                np.minimum(self.p[cand_i], self.p[cand_j]) - 0.5 * r_e[i_cand]
            )
            d_pmin = np.minimum(p_endpoint, float(self.p_min)) - float(self.p_min)

            reward = (
                a1 * d_l2 / nf
                + a2 * eta_r * d_rg_red / (nf * nf)
                + a3 * eta_p * d_pmin
            )
            order = np.argsort(-reward, kind="mergesort")
            n_sel = int(min(top, m))
            sel_local = order[:n_sel]
            sel_global = i_cand[sel_local]
            pairs = np.stack([i_idx[sel_global], j_idx[sel_global]], axis=1)
            info = {
                "reward": reward[sel_local],
                "delta_lambda2": d_l2[sel_local],
                "delta_rg_reduction": d_rg_red[sel_local],
                "delta_pmin": d_pmin[sel_local],
                "r_e": r_e[sel_global],
                "f_score": f_score[sel_global],
                "case": np.full(n_sel, 1, dtype=np.int64),
            }
            return pairs.astype(np.int64, copy=False), info

        # ---- Case B: MaxVol-selected r-1 edges as the candidate set ----
        Bp = psi                                 # (n, m)
        Z_mult = self.U.T @ Bp                  # (r, m)
        J_keep = filter_high_potential_columns(Z_mult, self.r)
        if J_keep.size < self.r:
            sel_local = J_keep
        else:
            Z_keep = Z_mult[:, J_keep]          # (r, m')
            Q_R, _ = np.linalg.qr(Z_keep.T, mode="reduced")  # (m', r)
            I_opt = maxvol_selection(Q_R, eps=self.maxvol_eps, max_iter=self.maxvol_iter)
            sel_local = J_keep[I_opt]
            if sel_local.size > 1:
                sel_local = sel_local[:-1]      # discard one edge (the last)
        n_sel = int(min(top, sel_local.size))
        sel_local = sel_local[:n_sel]
        sel_global = i_cand[sel_local]
        pairs = np.stack([i_idx[sel_global], j_idx[sel_global]], axis=1)
        info = {
            "r_e": r_e[sel_global],
            "f_score": f_score[sel_global],
            "col_norm": np.linalg.norm(Z_mult[:, sel_local], axis=0),
            "case": np.full(n_sel, self.r, dtype=np.int64),
        }
        return pairs.astype(np.int64, copy=False), info

    # ------------------------------------------------------------------ #
    #  Edge addition (spec Single-Edge Update + Case B sequential step)
    # ------------------------------------------------------------------ #

    def _rank1_update(self, i: int, j: int) -> np.ndarray:
        """Sherman-Morrison pseudoinverse + Laplacian update.

        Returns the incidence vector ``a``.  Also returns nothing else; the
        exact ΔR_G reduction is computed by the caller via ``L⁺ a``.
        """
        a = np.zeros(self.n, dtype=np.float64)
        a[i] = 1.0
        a[j] = -1.0

        Lpa = self.L_plus @ a
        beta = 1.0 + float(a @ Lpa)
        if not np.isfinite(beta) or beta <= 1e-14:
            self._recompute_exact()
            return a

        # Centering the O(n) update vector replaces double-centering the n x n
        # matrix. L⁺·1 = 0, so 1ᵀ(L⁺a) = (L⁺1)ᵀa = 0 -- Lpa has zero sum
        # exactly, hence outer(Lpa, Lpa) is already centered and so is the
        # downdated L⁺. The spec's (I - J) L⁺ (I - J) is therefore roundoff
        # hygiene only, and it cost two dense n x n matmuls (~2n³ flops) per
        # edge to achieve what one O(n) mean subtraction does.
        Lpa = Lpa - Lpa.mean()

        self.L_plus = self.L_plus - np.outer(Lpa, Lpa) / beta
        self.L = self.L + np.outer(a, a)
        return a

    def _rayleigh_ritz_update(self) -> None:
        """Rayleigh-Ritz eigenspace refinement after a rank-1 edge addition
        (spec Case B step 5d).  Updates U, r, λ₂ and the stored basis."""
        H = self.U.T @ self.L @ self.U          # (r, r)
        Theta, V = np.linalg.eigh(H)
        U_new = self.U @ V                      # (n, r)

        theta_min = float(Theta[0])
        cluster = np.flatnonzero(np.abs(Theta - theta_min) < self.eps_mult)
        r_new = int(cluster.size)

        self.evals[1 : 1 + r_new] = theta_min
        self.Q[:, 1 : 1 + r_new] = U_new[:, :r_new]
        self.U = U_new[:, :r_new].copy()
        self.r = r_new

    def _drift_residual(self, u2: np.ndarray, lambda2: float) -> float:
        """Eigenpair residual ‖L u₂ - λ₂ u₂‖ (spec §3.3 step 7).

        O(n²) -- one matvec -- and measured at 0.1-1.5% of a dense ``eigh``
        across n=32..256, so it is not a meaningful cost. It is also the *only*
        signal here that sees λ₂/eigenvector drift: the secular update writes
        ``evals[1]``/``Q[:,1]`` with no self-correcting step, so nothing else
        would notice it going stale.
        """
        return float(np.linalg.norm(self.L @ u2 - lambda2 * u2))

    def _cheap_drift(self) -> float:
        """Two O(n) identities that catch drift the eigenpair residual cannot.

        Both are *exact* invariants of a correct state, so a nonzero value is
        drift by definition rather than approximation error:

          Foster : |Σ_i p_i - 1|
              Foster's theorem gives Σ_{(i,j)∈E} R_eff(i,j) = n - 1, so
              Σ_i p_i = n - (n - 1) = 1 identically. Free -- p is already
              computed every step for the reward.

          trace  : |R_G - n·tr(L⁺)| / |R_G|
              R_G is carried as a running accumulator while L⁺ is carried as a
              matrix; R_G = n·tr(L⁺) ties them together.

        These cover the pseudoinverse and the R_G accumulator; the residual
        covers the eigenpair. They are complementary, not substitutes -- an
        earlier revision replaced the residual with these and λ₂ silently
        drifted to 8e-2 within three edges, because neither certificate can
        observe the Fiedler pair at all.
        """
        self.cert_foster = abs(float(np.sum(self.p)) - 1.0)
        tr = float(self.n) * float(np.trace(self.L_plus))
        self.cert_trace = abs(float(self.R_G) - tr) / max(abs(float(self.R_G)), 1e-12)
        return max(self.cert_foster, self.cert_trace)

    def _drift_check(self, u2: np.ndarray, lambda2: float) -> None:
        """Drift monitoring, with a forced full recomputation every
        ``max_steps_without_reset`` steps.

        Fires on the eigenpair residual (λ₂ accuracy) or either cheap
        certificate (L⁺ / R_G accuracy), whichever trips first.
        """
        self.steps_since_exact += 1
        residual = self._drift_residual(u2, lambda2)
        drift = max(residual, self._cheap_drift())
        self.drift_residual = residual
        # A non-finite certificate must FAIL CLOSED.  ``nan > tau_drift`` is
        # False, so a NaN residual used to sail through the threshold and leave
        # the corrupt state in place, where it reached the policy as NaN logits
        # instead of being re-anchored here.
        if (
            not np.isfinite(drift)
            or drift > self.tau_drift
            or self.steps_since_exact >= self.max_steps_without_reset
        ):
            self._recompute_exact()

    def _check_secular_bracket(
        self, lambda2_prime: float, old_lambda2: float, psi2: float
    ) -> bool:
        """Certify the exact secular root against a closed-form two-sided bound.

        For the rank-1 update L' = L + aaᵀ with a = e_i - e_j, writing
        ψ₂ = q₂ᵀa and δ' = μ - λ₂ for the gap above λ₂:

            (δ'/(δ' + 2))·ψ₂²  ≤  Δλ₂  ≤  ψ₂²

        Both bounds are O(1) -- ψ is already formed for the secular solve -- and
        the inequality was checked against exact eigendecompositions on 90
        random (graph, edge) pairs with zero violations, so it is sound when the
        basis is exact. This does **not** replace the exact root: the root is
        still what gets used. The bracket only certifies it.

        **Diagnostic only -- it does not force a recomputation.** It was
        originally written to re-anchor on violation, but measurement showed
        that buys nothing: over 40-edge runs at n=12/32/64/128 the final λ₂
        error is bit-identical (3.55e-15, 2.22e-16, 4.83e-15, 5.69e-16) whether
        the guard re-anchors or merely counts. It fires ~10 times per 40 edges,
        and those firings are benign -- Case A leaves ``Q[:,2:]``/``evals[2:]``
        stale between anchors, so the bracket is evaluated against a basis that
        is partly out of date, while the eigenpair residual in ``_drift_check``
        already catches every case that actually moves λ₂. Re-anchoring here
        would have spent ~10 extra dense eigendecompositions per 40 edges to
        change no digit of the answer.

        Returns True when the root is inside the bracket (or the check does not
        apply). Records violations in ``bracket_violations`` and the interval in
        ``last_secular_bracket``.
        """
        delta = float(lambda2_prime) - float(old_lambda2)
        mu = float(self.mu)
        # No upper eigenvalue => no secular solve happened; nothing to certify.
        if not np.isfinite(mu):
            return True
        delta_prime = mu - float(old_lambda2)
        if delta_prime <= 0.0 or not np.isfinite(delta_prime):
            return True
        hi = float(psi2) * float(psi2)
        lo = (delta_prime / (delta_prime + 2.0)) * hi
        self.last_secular_bracket = (lo, hi)
        # Absolute slack for the root-finder's own tolerance, scaled to the
        # magnitudes involved so it stays meaningful as λ₂ grows.
        tol = 1e-8 * max(1.0, abs(hi), abs(delta))
        if (lo - tol) <= delta <= (hi + tol):
            return True
        self.bracket_violations += 1
        return False

    def add_edge(self, i: int, j: int) -> None:
        """Add edge (i, j) and update the spectral state per the spec."""
        if i == j:
            raise ValueError(f"Self-loop not allowed: ({i},{j})")
        if self.adj[i, j] != 0:
            raise ValueError(f"Edge ({i},{j}) already present")

        old_lambda2 = float(self.evals[1])
        old_p_min = float(self.p_min)

        # Projection of the incidence vector onto the *current* basis.
        a = np.zeros(self.n, dtype=np.float64)
        a[i] = 1.0
        a[j] = -1.0
        psi = self.Q.T @ a                      # (n,)  psi[0] == 0 because a ⟂ 1

        # Exact ΔR_G as a positive reduction via the fast SM pseudoinverse
        # update (no eigenbasis needed):  v = L⁺ a,  ΔR_G_red = n (vᵀv)/β.
        Lpa = self.L_plus @ a
        beta = 1.0 + float(a @ Lpa)
        if not np.isfinite(beta) or beta <= 1e-14:
            self._recompute_exact()
            return
        delta_rg_red = float(self.n) * float(Lpa @ Lpa) / beta
        self.R_G = self.R_G - delta_rg_red

        self.adj[i, j] = 1
        self.adj[j, i] = 1
        self._rank1_update(i, j)

        # Refresh p before any drift check: _cheap_drift's Foster certificate
        # reads self.p, and a p left over from the pre-edge adjacency would
        # register as drift that isn't there.
        self.p = compute_p(self.adj, self.L_plus)
        self.p_min = float(np.min(self.p))

        if self.r == 1:
            # ---- Case A / Single-Edge Update ----
            lambda2_prime = solve_secular_equation(
                psi, self.evals, old_lambda2, self.mu, eps=self.secular_eps
            )
            self._check_secular_bracket(lambda2_prime, old_lambda2, float(psi[1]))
            # Endpoint collision (ψ at the upper pole ≈ 0): the solver returns
            # the bracket's *upper endpoint* instead of an interior root.  Test
            # against the endpoint the solver actually used --
            # ``secular_upper_endpoint`` clamps μ to the next stored eigenvalue
            # -- not against ``self.mu``; comparing only to ``self.mu`` misses
            # every collision resolved by the stored bound.
            endpoint = secular_upper_endpoint(
                self.evals, old_lambda2, self.mu, eps_mult=self.eps_mult
            )
            collided = lambda2_prime > old_lambda2 and (
                abs(lambda2_prime - endpoint) < 1e-10
                or abs(lambda2_prime - float(self.mu)) < 1e-10
            )

            if collided:
                # λ₂' = endpoint follows from interlacing (λ₂ ≤ λ₂' ≤ λ₃) only
                # while the stored spectrum is exact -- but Case A writes just
                # evals[1]/Q[:,1], so evals[2:] is stale between anchors and the
                # clamped endpoint can be a stale eigenvalue that is *not* the
                # new λ₂.  Neither incremental repair is safe:
                #
                #   * the secular eigenvector sum divides by zero, because the
                #     endpoint is a pole of f (this is the NaN that reached the
                #     policy as NaN logits);
                #   * taking q_k instead is worse than it looks -- (q_k, λ_k) is
                #     a genuine eigenpair of L' = L + aaᵀ when ψ_k ≈ 0, just not
                #     the *second* one, so the residual certificate reads ~0 and
                #     the drift check silently keeps a λ₂ that was off by up to
                #     2.0 in measurement.
                #
                # A wrong-index eigenpair is invisible to every certificate we
                # carry, so the only sound response is to re-anchor exactly.
                # Collisions are rare and ``_refresh_mu`` already spends a dense
                # eigvalsh every step, so this costs little.
                self._recompute_exact()
            elif lambda2_prime > old_lambda2:
                u2 = update_fiedler_vector(psi, self.evals, lambda2_prime, self.Q)
                self.Q[:, 1] = u2
                self.evals[1] = lambda2_prime
                self._drift_check(u2, lambda2_prime)
            else:
                # Edge does not raise λ₂ (projection on u₂ ≈ 0).  The Fiedler
                # pair is unchanged; still run drift monitoring (the residual
                # is ~|ψ₂|·√2 < τ here) so the periodic full recomputation
                # fires even during long no-raise streaks.
                self._drift_check(self.Q[:, 1].copy(), float(self.evals[1]))
        else:
            # ---- Case B: Rayleigh-Ritz eigenspace update ----
            self._rayleigh_ritz_update()
            self._drift_check(self.U[:, 0], float(self.evals[1]))

        self.mu = self._refresh_mu(float(self.evals[1]))

        self._delta_l2_last = float(self.evals[1]) - old_lambda2
        self._delta_rg_last = float(delta_rg_red)
        self._delta_pmin_last = float(self.p_min - old_p_min)

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def state_dict(self) -> Dict:
        return {
            "n": self.n,
            "tau_drift": self.tau_drift,
            "max_steps_without_reset": self.max_steps_without_reset,
            "eps_mult": self.eps_mult,
            "secular_eps": self.secular_eps,
            "maxvol_eps": self.maxvol_eps,
            "maxvol_iter": self.maxvol_iter,
            "adj": self.adj.copy(),
            "steps_since_exact": self.steps_since_exact,
            "L": self.L.copy(),
            "Q": self.Q.copy(),
            "evals": self.evals.copy(),
            "r": int(self.r),
            "U": self.U.copy(),
            "mu": float(self.mu),
            "L_plus": self.L_plus.copy(),
            "R_G": float(self.R_G),
            "p_min": float(self.p_min),
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "AnalyticState":
        obj = cls.__new__(cls)
        obj.n = int(state["n"])
        obj.tau_drift = float(state["tau_drift"])
        obj.max_steps_without_reset = int(state["max_steps_without_reset"])
        obj.eps_mult = float(state["eps_mult"])
        obj.secular_eps = float(state["secular_eps"])
        obj.maxvol_eps = float(state["maxvol_eps"])
        obj.maxvol_iter = int(state["maxvol_iter"])
        obj.adj = state["adj"].copy()
        obj.steps_since_exact = int(state["steps_since_exact"])
        obj.L = state["L"].copy()
        obj.Q = state["Q"].copy()
        obj.evals = state["evals"].copy()
        obj.r = int(state["r"])
        obj.U = state["U"].copy()
        obj.mu = float(state["mu"])
        obj.L_plus = state["L_plus"].copy()
        obj.R_G = float(state["R_G"])
        obj.p_min = float(state["p_min"])
        obj.inv_lambda = np.zeros(obj.n, dtype=np.float64)
        obj.inv_lambda_sq = np.zeros(obj.n, dtype=np.float64)
        safe = np.maximum(obj.evals[1:], 1e-14)
        obj.inv_lambda[1:] = 1.0 / safe
        obj.inv_lambda_sq[1:] = 1.0 / (safe * safe)
        # Derived, so not carried in the payload: p is a pure function of
        # (adj, L⁺) and _cheap_drift reads it on the first step after a
        # restore. Diagnostics restart at zero by design.
        obj.p = compute_p(obj.adj, obj.L_plus)
        obj.cert_foster = 0.0
        obj.cert_trace = 0.0
        obj.drift_residual = 0.0
        obj.last_secular_bracket = None
        obj.bracket_violations = 0
        return obj
