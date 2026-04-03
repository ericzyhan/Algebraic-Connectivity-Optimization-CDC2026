from __future__ import annotations

import numpy as np
import networkx as nx
import scipy.sparse.linalg as spla


def _dense_laplacian_eig(graph: nx.Graph):
    lap = nx.laplacian_matrix(graph).astype(float).toarray()
    vals, vecs = np.linalg.eigh(lap)
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]


def algebraic_connectivity(graph: nx.Graph) -> float:
    n = graph.number_of_nodes()
    if n < 2:
        return 0.0
    if n <= 3:
        vals, _ = _dense_laplacian_eig(graph)
        return float(vals[1])
    lap = nx.laplacian_matrix(graph).astype(float)
    try:
        vals = spla.eigsh(
            lap,
            k=2,
            which="SM",
            return_eigenvectors=False,
            tol=1e-6,
        )
        vals = np.sort(np.real(vals))
        return float(vals[1])
    except Exception:
        vals, _ = _dense_laplacian_eig(graph)
        return float(vals[1])


def fiedler_vector(graph: nx.Graph) -> np.ndarray:
    n = graph.number_of_nodes()
    if n < 2:
        return np.zeros(n, dtype=float)
    if n <= 3:
        _, vecs = _dense_laplacian_eig(graph)
        return np.asarray(vecs[:, 1], dtype=float)
    lap = nx.laplacian_matrix(graph).astype(float)
    try:
        vals, vecs = spla.eigsh(
            lap,
            k=2,
            which="SM",
            return_eigenvectors=True,
            tol=1e-6,
        )
        order = np.argsort(np.real(vals))
        return np.asarray(np.real(vecs[:, order[1]]), dtype=float)
    except Exception:
        _, vecs = _dense_laplacian_eig(graph)
        return np.asarray(vecs[:, 1], dtype=float)
