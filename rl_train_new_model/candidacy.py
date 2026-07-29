from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    tier1_topk_dr: int = 192,
    tier1_topk_spectral: int = 64,
    tier2_survivor_size: int = 64,
    tier3_top: int = 48,
    tier3_random: int = 16,
    rng: Optional[np.random.RandomState] = None,
) -> CandidacyResult:
    """Tiered edge-candidacy pipeline (docs/algorithm.md S4, with the Tier-2
    r>1 branch trimmed of local swap refinement / macro-actions per project
    decision: RRQR only ranks/shortlists, the GNN policy makes the final pick).
    """
    r = tracker.multiplicity
    soft_degenerate = tracker.soft_degenerate

    nonedge_idx = np.flatnonzero(pair_is_nonedge)
    if nonedge_idx.size == 0:
        return _empty_result(r, soft_degenerate)

    i_all = pair_i[nonedge_idx]
    j_all = pair_j[nonedge_idx]

    # ---- Tier 0: score every non-edge, O(1) each (matrix-entry lookups) ----
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

    tier0_count = int(nonedge_idx.size)

    # ---- Tier 1: shortlist by union, not intersection ----
    k_dr = min(tier1_topk_dr, tier0_count)
    k_sp = min(tier1_topk_spectral, tier0_count)
    order_dr = np.argsort(-np.abs(delta_RG_e))[:k_dr]
    order_sp = np.argsort(-spectral_score)[:k_sp]
    shortlist_local = np.unique(np.concatenate([order_dr, order_sp]))
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
        piv = tracker.rrqr_rank(Z)
        survivor_size = min(tier2_survivor_size, tier1_count)
        survivor_local = piv[:survivor_size]
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

    # ---- Tier 3: 48 top-scored + 16 sampled uniformly from the remainder ----
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
