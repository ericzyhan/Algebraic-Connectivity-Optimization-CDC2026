from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..graph_math import (
    clustering_coefficients,
    edge_index_from_adj,
    node_degrees,
    non_edges,
    top_k_indices,
    truncated_all_pairs_shortest_path,
    two_hop_neighbor_counts,
)


def build_node_features(
    *,
    adj: np.ndarray,
    n: int,
    phi2: np.ndarray,
    phi3: np.ndarray,
    incremental_observation: bool,
    deg_cache: Optional[np.ndarray],
    a2_counts: Optional[np.ndarray],
    triangles_per_node: Optional[np.ndarray],
    eye_mask: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if (
        incremental_observation
        and deg_cache is not None
        and a2_counts is not None
        and triangles_per_node is not None
        and eye_mask is not None
    ):
        deg = deg_cache
        adj_bool = adj > 0
        two_hop_mask = (a2_counts > 0) & (~adj_bool) & (~eye_mask)
        two_hop = np.sum(two_hop_mask, axis=1, dtype=np.float64)
        cluster_denom = deg * np.maximum(1.0, deg - 1.0)
        cluster = np.divide(
            2.0 * triangles_per_node,
            cluster_denom,
            out=np.zeros_like(deg, dtype=np.float64),
            where=deg >= 2.0,
        )
    else:
        deg = node_degrees(adj)
        two_hop = two_hop_neighbor_counts(adj)
        cluster = clustering_coefficients(adj)

    denom = max(1, n - 1)
    node_features = np.stack(
        [
            deg / denom,
            two_hop / denom,
            phi2,
            phi3,
            cluster,
        ],
        axis=1,
    ).astype(np.float32)
    return node_features, deg


def build_global_features(
    *,
    n: int,
    rho_target: float,
    rho_current: float,
    lambda2: float,
    lambda3: float,
    lambda4: float,
) -> np.ndarray:
    return np.array(
        [
            float(n),
            float(rho_target),
            float(rho_current),
            float(lambda2 / max(1, n)),
            float(lambda3 / max(1, n)),
            float(lambda4 / max(1, n)),
        ],
        dtype=np.float32,
    )


def _lambda2_eigenspace_gap(
    evecs: np.ndarray,
    multiplicity: int,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
) -> np.ndarray:
    """Compute H(i,j) = Σ_{k=2}^{d} (evecs[i,k] - evecs[j,k])²

    where d = multiplicity of λ₂ (the algebraic connectivity).  Sums squared
    eigenvector differences across all eigenvectors in the λ₂ eigenspace.
    For d=1 this reduces to the standard Fiedler gap.
    """
    d = int(max(1, multiplicity))
    # d eigenvectors starting at column index 1 (skip λ₁=0 eigenvector)
    phi_stack = evecs[:, 1 : 1 + d].astype(np.float64)  # (n, d)
    diff = phi_stack[i_idx, :] - phi_stack[j_idx, :]  # (p, d)
    return np.sum(diff * diff, axis=1)  # (p,)


def build_candidate_features(
    *,
    adj: np.ndarray,
    deg: np.ndarray,
    n: int,
    phi2: np.ndarray,
    phi3: np.ndarray,
    phi4: np.ndarray,
    evecs: np.ndarray,
    multiplicity: int,
    top_k: int,
    dist_cap: int,
    incremental_observation: bool,
    pair_i: Optional[np.ndarray],
    pair_j: Optional[np.ndarray],
    pair_is_nonedge: Optional[np.ndarray],
    dist_matrix: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if (
        incremental_observation
        and pair_i is not None
        and pair_j is not None
        and pair_is_nonedge is not None
        and dist_matrix is not None
    ):
        nonedge_pair_idx = np.flatnonzero(pair_is_nonedge)
        if nonedge_pair_idx.size == 0:
            return (
                np.zeros((0, 2), dtype=np.int64),
                np.zeros((0, 7), dtype=np.float32),
            )

        # λ₂ eigenspace gap: H(i,j) = Σ_{k=2}^{d} (φ_k(i) - φ_k(j))²
        all_h = _lambda2_eigenspace_gap(evecs, multiplicity, pair_i, pair_j)
        selected_local = top_k_indices(all_h[nonedge_pair_idx], top_k)
        selected_pair_idx = nonedge_pair_idx[selected_local]
        candidate_pairs = np.stack(
            [pair_i[selected_pair_idx], pair_j[selected_pair_idx]],
            axis=1,
        ).astype(np.int64, copy=False)
    else:
        all_non_edges = non_edges(adj)
        if not all_non_edges:
            return (
                np.zeros((0, 2), dtype=np.int64),
                np.zeros((0, 7), dtype=np.float32),
            )
        # λ₂ eigenspace gap for all non-edges
        arr = np.asarray(all_non_edges, dtype=np.int64)
        i_all = arr[:, 0]
        j_all = arr[:, 1]
        all_h = _lambda2_eigenspace_gap(evecs, multiplicity, i_all, j_all)
        selected = top_k_indices(all_h, top_k)
        candidate_pairs_list = [all_non_edges[idx] for idx in selected]
        candidate_pairs = np.array(candidate_pairs_list, dtype=np.int64)

    i_idx = candidate_pairs[:, 0]
    j_idx = candidate_pairs[:, 1]

    # λ₂ eigenspace gap as a pair feature (replaces plain Fiedler gap)
    h_score = _lambda2_eigenspace_gap(evecs, multiplicity, i_idx, j_idx)
    phi3_gap = np.abs(phi3[i_idx] - phi3[j_idx])
    phi4_gap = np.abs(phi4[i_idx] - phi4[j_idx])

    common_counts = np.sum(
        np.logical_and(adj[i_idx] > 0, adj[j_idx] > 0),
        axis=1,
        dtype=np.float64,
    )
    cn_norm = common_counts / float(max(1, n - 2))

    union = deg[i_idx] + deg[j_idx] - common_counts
    jac = np.divide(
        common_counts,
        union,
        out=np.zeros_like(common_counts, dtype=np.float64),
        where=union > 0,
    )

    if dist_matrix is not None:
        clipped_dist = np.minimum(dist_matrix[i_idx, j_idx], dist_cap)
        dist_norm = clipped_dist.astype(np.float64) / float(max(1, dist_cap))
    else:
        all_dist = truncated_all_pairs_shortest_path(adj, dist_cap)
        dist_norm = all_dist[i_idx, j_idx].astype(np.float64) / float(max(1, dist_cap))

    h_norm = h_score / max(float(np.max(h_score)), 1e-10)

    pair_features = np.stack(
        [h_score, phi3_gap, phi4_gap, cn_norm, jac, dist_norm, h_norm],
        axis=1,
    ).astype(np.float32, copy=False)
    return candidate_pairs, pair_features


def build_edge_index(adj: np.ndarray) -> np.ndarray:
    return edge_index_from_adj(adj)
