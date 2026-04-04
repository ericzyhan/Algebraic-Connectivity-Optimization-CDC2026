from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .graph_core import edge_count, target_density


TargetDegreeMode = Literal["floor", "round", "ceil"]


@dataclass
class FeaturePack:
    node_features: np.ndarray  # [n,5]
    global_features: np.ndarray  # [5]
    edge_features: np.ndarray  # [k,11]


def compute_target_degree(n: int, m_target: int, mode: TargetDegreeMode) -> int:
    avg = 2.0 * m_target / n
    if mode == "floor":
        return int(np.floor(avg))
    if mode == "ceil":
        return int(np.ceil(avg))
    return int(np.round(avg))


def node_features(adj: np.ndarray, phi2: np.ndarray, phi3: np.ndarray) -> np.ndarray:
    deg = adj.sum(axis=1).astype(np.float64)
    max_deg = max(float(deg.max()), 1.0)
    deg_norm = deg / max_deg

    z_real = phi2.astype(np.float64)
    z_imag = phi3.astype(np.float64)
    return np.stack([deg_norm, z_real, z_imag, phi2, phi3], axis=1).astype(np.float32)


def global_features(n: int, m_target: int, e_cur: int, lam2: float) -> np.ndarray:
    rho = target_density(n, m_target)
    denom = max(m_target - (n - 1), 1)
    progress = float(np.clip((e_cur - (n - 1)) / denom, 0.0, 1.0))
    regular_feasible = 1.0 if ((2 * m_target) % n == 0) else 0.0
    out = np.asarray(
        [np.log1p(n), rho, progress, float(max(lam2, 0.0)), regular_feasible],
        dtype=np.float32,
    )
    return out


def edge_features(
    adj: np.ndarray,
    candidates: np.ndarray,
    er_scores: np.ndarray,
    phi2: np.ndarray,
    phi3: np.ndarray,
    gfeat: np.ndarray,
    m_target: int,
    target_degree_mode: TargetDegreeMode = "floor",
) -> np.ndarray:
    n = adj.shape[0]
    if candidates.size == 0:
        return np.zeros((0, 11), dtype=np.float32)

    deg = adj.sum(axis=1).astype(np.float64)
    max_deg = max(float(deg.max()), 1.0)
    deg_norm = deg / max_deg
    z = phi2 + 1j * phi3
    progress = float(gfeat[2])
    rho = float(gfeat[1])
    lam2 = float(gfeat[3])

    k_tgt = compute_target_degree(n, m_target, target_degree_mode)
    mu = float(deg.mean())
    out = np.zeros((candidates.shape[0], 11), dtype=np.float64)
    n_ov_denom = float(max(n - 2, 1))

    for idx, (u, v) in enumerate(candidates):
        u = int(u)
        v = int(v)
        du = float(deg[u])
        dv = float(deg[v])
        delta_u = max(0.0, k_tgt - du)
        delta_v = max(0.0, k_tgt - dv)
        delta_var = ((2.0 * (du + dv) + 2.0) / n) - (4.0 * mu / n) + (4.0 / (n * n))
        overlap = float(np.count_nonzero(adj[u] & adj[v])) / n_ov_denom
        out[idx] = np.asarray(
            [
                abs(z[u] - z[v]),
                deg_norm[u],
                deg_norm[v],
                progress,
                rho,
                lam2,
                float(er_scores[idx]),
                delta_u,
                delta_v,
                delta_var,
                overlap,
            ],
            dtype=np.float64,
        )

    return out.astype(np.float32)
