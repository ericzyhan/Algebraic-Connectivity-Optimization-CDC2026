from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .spectral import SpectralTracker


@dataclass
class CandidacyResult:
    candidate_pairs: np.ndarray  # (K, 2) int64
    pair_features: np.ndarray  # (K, F) float32
    tier0_count: int
    tier1_count: int
    tier2_survivor_count: int
    multiplicity: int
    soft_degenerate: bool


@dataclass
class DeflationPlan:
    """The deterministic r-1 edge batch that collapses mult(lambda_2) to 1."""

    edges: np.ndarray  # (r-1, 2) int64
    z_columns: np.ndarray  # (r-1, r) float64, rows are z_e in the pre-batch basis
    r: int
    cond_R: float
    pool_size: int
    used_full_pool: bool


PAIR_FEATURE_DIM = 10


def _empty_result(multiplicity: int, soft_degenerate: bool) -> CandidacyResult:
    return CandidacyResult(
        candidate_pairs=np.zeros((0, 2), dtype=np.int64),
        pair_features=np.zeros((0, PAIR_FEATURE_DIM), dtype=np.float32),
        tier0_count=0,
        tier1_count=0,
        tier2_survivor_count=0,
        multiplicity=multiplicity,
        soft_degenerate=soft_degenerate,
    )


def _normalize(x: np.ndarray) -> np.ndarray:
    denom = float(np.max(np.abs(x))) if x.size > 0 else 0.0
    if denom <= 1e-10:
        return np.zeros_like(x, dtype=np.float32)
    return (x / denom).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
#  Tier 0 / Tier 1, shared by the candidacy pipeline and the deflation planner
# --------------------------------------------------------------------------- #


def _tier0_scores(
    tracker: SpectralTracker,
    i_all: np.ndarray,
    j_all: np.ndarray,
    r: int,
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """O(1)-per-candidate scores for every non-edge: matrix-entry lookups only."""
    Lplus = tracker.Lplus
    Lplus_diag = np.diag(Lplus)
    R_e = Lplus_diag[i_all] + Lplus_diag[j_all] - 2.0 * Lplus[i_all, j_all]
    gamma_e = 1.0 + R_e

    K_diag = np.diag(tracker.K)
    Kb = K_diag[i_all] + K_diag[j_all] - 2.0 * tracker.K[i_all, j_all]  # ||L+ b||^2, exact
    delta_RG_e = -float(n) * Kb / np.maximum(gamma_e, 1e-10)  # exact, <= 0

    Ur = tracker.U[:, :r]
    diff = Ur[i_all, :] - Ur[j_all, :]  # (k, r)
    spectral_score = np.sum(diff * diff, axis=1)  # beta_2^2 when r==1, ||z_e||^2 otherwise
    return R_e, delta_RG_e, spectral_score, diff


def _tier1_shortlist(
    delta_RG_e: np.ndarray,
    spectral_score: np.ndarray,
    topk_dr: int,
    topk_spectral: int,
) -> np.ndarray:
    """Union (not intersection) of the two top-k screens."""
    count = int(delta_RG_e.shape[0])
    order_dr = np.argsort(-np.abs(delta_RG_e))[: min(topk_dr, count)]
    order_sp = np.argsort(-spectral_score)[: min(topk_spectral, count)]
    return np.unique(np.concatenate([order_dr, order_sp]))


# --------------------------------------------------------------------------- #
#  Deterministic r-1 deflation batch (algorithm S2, mult(lambda_2) > 1 branch)
# --------------------------------------------------------------------------- #


def plan_deflation_batch(
    tracker: SpectralTracker,
    *,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_is_nonedge: np.ndarray,
    n: int,
    tier1_topk_dr: int = 32,
    tier1_topk_spectral: int = 32,
    cond_max: float = 1e10,
) -> Optional[DeflationPlan]:
    """Pick the r-1 edges that cascade mult(lambda_2) from r down to 1.

    Screened pool B' first, as the algorithm specifies. B' is a *screened* pool,
    so it is not guaranteed to span the lambda_2 eigenspace -- cond(R) from the
    thin QR is the designated check, and a failure falls back to the full
    non-edge set rather than aborting. Returns None only when even the full pool
    is rank deficient (a genuinely degenerate graph), in which case the caller
    leaves the pick to the policy.
    """
    r = int(tracker.multiplicity)
    if r <= 1:
        return None

    nonedge_idx = np.flatnonzero(pair_is_nonedge)
    if nonedge_idx.size < r:
        return None

    i_all = pair_i[nonedge_idx]
    j_all = pair_j[nonedge_idx]
    _, delta_RG_e, spectral_score, diff = _tier0_scores(tracker, i_all, j_all, r, n)

    shortlist = _tier1_shortlist(delta_RG_e, spectral_score, tier1_topk_dr, tier1_topk_spectral)
    full = np.arange(i_all.shape[0], dtype=np.int64)

    chosen_pool: Optional[np.ndarray] = None
    cond_R = float("inf")
    used_full_pool = False
    for pool, is_full in ((shortlist, False), (full, True)):
        if pool.size < r:
            continue
        Z_try = diff[pool, :].T  # (r, |pool|)
        cond_try = SpectralTracker.deflation_pool_conditioning(Z_try)
        if np.isfinite(cond_try) and cond_try <= cond_max:
            chosen_pool, cond_R, used_full_pool = pool, float(cond_try), is_full
            break
        if is_full:
            # Record the full pool's conditioning for the caller's diagnostics
            # even when it fails, so a rejection is distinguishable from r <= 1.
            cond_R = float(cond_try)
    if chosen_pool is None:
        return None

    i_pool = i_all[chosen_pool]
    j_pool = j_all[chosen_pool]
    Z = diff[chosen_pool, :].T  # (r, |pool|)

    _, selected = tracker.maxvol_rank(Z)
    if selected.size < r:
        return None

    # MaxVol returns the volume-maximal r-subset; deflation consumes r-1 of them.
    # Volume is a property of the set, so any r-1 of the r still span an
    # (r-1)-dimensional slice of the eigenspace -- the tiebreak keeps the
    # largest-norm z columns, which are the ones furthest from the ||z||=0
    # candidates that cannot deflate at all.
    z_sel = Z[:, selected].T  # (r, r)
    keep = np.argsort(-np.linalg.norm(z_sel, axis=1))[: r - 1]
    sel = selected[keep]

    return DeflationPlan(
        edges=np.stack([i_pool[sel], j_pool[sel]], axis=1).astype(np.int64, copy=False),
        z_columns=Z[:, sel].T.astype(np.float64, copy=False),
        r=r,
        cond_R=cond_R,
        pool_size=int(chosen_pool.size),
        used_full_pool=used_full_pool,
    )


# --------------------------------------------------------------------------- #
#  Tiered candidacy pipeline
# --------------------------------------------------------------------------- #


def run_candidacy_pipeline(
    tracker: SpectralTracker,
    *,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_is_nonedge: np.ndarray,
    dist_matrix: Optional[np.ndarray],
    deg: np.ndarray,
    adj: np.ndarray,
    n: int,
    dist_cap: int,
    tier1_topk_dr: int = 32,
    tier1_topk_spectral: int = 32,
    tier2_survivor_size: int = 32,
    tier3_top: int = 24,
    tier3_random: int = 8,
    rng: Optional[np.random.RandomState] = None,
) -> CandidacyResult:
    """Tiered edge-candidacy pipeline.

    Tier 1 screens to <= 64 (top 32 by |Delta R_G| union top 32 by the
    multispectral Fiedler score), Tier 2 refines to <= 32, Tier 3 hands the
    policy those 32 as its action set.

    The r>1 Tier-2 branch is the fallback path: when the deflation macro-action
    is enabled the env collapses r to 1 before the policy ever observes a
    degenerate state, so this branch is reached only when deflation was declined
    (rank-deficient pool) or disabled.
    """
    r = tracker.multiplicity
    soft_degenerate = tracker.soft_degenerate

    nonedge_idx = np.flatnonzero(pair_is_nonedge)
    if nonedge_idx.size == 0:
        return _empty_result(r, soft_degenerate)

    i_all = pair_i[nonedge_idx]
    j_all = pair_j[nonedge_idx]

    # ---- Tier 0: score every non-edge, O(1) each (matrix-entry lookups) ----
    R_e, delta_RG_e, spectral_score, diff = _tier0_scores(tracker, i_all, j_all, r, n)
    tier0_count = int(nonedge_idx.size)

    # ---- Tier 1: shortlist by union, not intersection ----
    shortlist_local = _tier1_shortlist(
        delta_RG_e, spectral_score, tier1_topk_dr, tier1_topk_spectral
    )
    tier1_count = int(shortlist_local.size)

    i_short = i_all[shortlist_local]
    j_short = j_all[shortlist_local]
    R_e_short = R_e[shortlist_local]
    delta_RG_short = delta_RG_e[shortlist_local]
    spectral_short = spectral_score[shortlist_local]
    diff_short = diff[shortlist_local, :]

    # ---- Tier 2: refine on the shortlist ----
    if r == 1:
        delta_prime = float(tracker.mu3 - tracker.mu2) if np.isfinite(tracker.mu3) else 0.0
        beta2_sq = spectral_short
        if delta_prime > 0.0:
            lo = (delta_prime / (delta_prime + 2.0)) * beta2_sq
        else:
            lo = np.zeros_like(beta2_sq)
        hi = beta2_sq
        # APPROX: bracket midpoint stands in for the exact secular root here
        # (avoids a per-candidate brentq call on the whole shortlist); the
        # bracket bounds themselves are exact given beta_2, delta'.
        refined_score = 0.5 * (lo + hi)
        bracket_width = hi - lo
        survivor_size = min(tier2_survivor_size, tier1_count)
        survivor_local = np.argsort(-refined_score)[:survivor_size]
    else:
        Z = diff_short.T  # (r, tier1_count)
        order, _ = tracker.maxvol_rank(Z)
        survivor_size = min(tier2_survivor_size, tier1_count)
        survivor_local = order[:survivor_size]
        refined_score = spectral_short
        bracket_width = np.zeros_like(spectral_short)

    tier2_survivor_count = int(survivor_local.size)
    if tier2_survivor_count == 0:
        return _empty_result(r, soft_degenerate)

    i_surv = i_short[survivor_local]
    j_surv = j_short[survivor_local]
    R_e_surv = R_e_short[survivor_local]
    delta_RG_surv = delta_RG_short[survivor_local]
    spectral_surv = spectral_short[survivor_local]
    refined_surv = refined_score[survivor_local]
    bracket_width_surv = bracket_width[survivor_local]
    delta_p_min_surv = tracker.delta_p_min_estimate(i_surv, j_surv)

    # ---- Tier 3: top-scored + a uniform sample of the remainder ----
    rank3 = np.argsort(-refined_surv)
    top_n = min(tier3_top, tier2_survivor_count)
    top_idx = rank3[:top_n]
    remainder_idx = rank3[top_n:]
    n_random = min(tier3_random, remainder_idx.size)
    if n_random > 0:
        rng_ = rng if rng is not None else np.random.RandomState()
        random_idx = rng_.choice(remainder_idx, size=n_random, replace=False)
    else:
        random_idx = np.zeros((0,), dtype=np.int64)
    final_idx = np.concatenate([top_idx, random_idx])

    i_final = i_surv[final_idx]
    j_final = j_surv[final_idx]

    # ---- Auxiliary topology features (kept from the original candidate builder) ----
    common_counts = np.sum(
        np.logical_and(adj[i_final] > 0, adj[j_final] > 0), axis=1, dtype=np.float64
    )
    cn_norm = common_counts / float(max(1, n - 2))
    union = deg[i_final] + deg[j_final] - common_counts
    jac = np.divide(
        common_counts, union, out=np.zeros_like(common_counts, dtype=np.float64), where=union > 0
    )
    if dist_matrix is not None:
        clipped_dist = np.minimum(dist_matrix[i_final, j_final], dist_cap)
        dist_norm = clipped_dist.astype(np.float64) / float(max(1, dist_cap))
    else:
        dist_norm = np.zeros_like(cn_norm)

    spectral_final = spectral_surv[final_idx]
    refined_final = refined_surv[final_idx]
    R_e_final = R_e_surv[final_idx]
    delta_RG_final = delta_RG_surv[final_idx]
    bracket_width_final = bracket_width_surv[final_idx]
    delta_p_min_final = delta_p_min_surv[final_idx]

    rg_gain = -delta_RG_final  # flip sign so "larger is better", matches other terms

    pair_features = np.stack(
        [
            spectral_final.astype(np.float32),
            _normalize(spectral_final),
            R_e_final.astype(np.float32),
            _normalize(rg_gain),
            refined_final.astype(np.float32),
            _normalize(bracket_width_final),
            _normalize(delta_p_min_final),
            cn_norm.astype(np.float32),
            jac.astype(np.float32),
            dist_norm.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    candidate_pairs = np.stack([i_final, j_final], axis=1).astype(np.int64, copy=False)

    return CandidacyResult(
        candidate_pairs=candidate_pairs,
        pair_features=pair_features,
        tier0_count=tier0_count,
        tier1_count=tier1_count,
        tier2_survivor_count=tier2_survivor_count,
        multiplicity=r,
        soft_degenerate=soft_degenerate,
    )
