from __future__ import annotations

from collections import deque
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .backbone_init import default_path_init_metadata, normalize_init_metadata
from .candidacy import PAIR_FEATURE_DIM, run_candidacy_pipeline
from .certificates import log_diagnostics
from .curriculum import CurriculumScheduler
from .features.full import (
    build_global_features as build_global_features_full,
    build_node_features as build_node_features_full,
)
from .features.lite import (
    build_candidate_features as build_candidate_features_lite,
    build_edge_index,
    build_global_features as build_global_features_lite,
    build_node_features as build_node_features_lite,
)
from .graph_math import (
    build_path_adjacency,
    edge_count,
    m_target_from_rho,
    normalized_density,
    spectral_features,
    total_effective_resistance,
    truncated_all_pairs_shortest_path,
)
from .shaping import PotentialShaper
from .spectral import SpectralTracker

InitAdjBuilder = Callable[[int, float, int], Tuple[np.ndarray, Dict[str, Any]]]


@dataclass
class GraphObservation:
    node_features: np.ndarray
    edge_index: np.ndarray
    candidate_pairs: np.ndarray
    pair_features: np.ndarray
    global_features: np.ndarray
    lambda2_norm: float
    n: int
    rho_target: float
    rho_current: float

    def to_dict(self) -> Dict:
        return {
            "node_features": self.node_features,
            "edge_index": self.edge_index,
            "candidate_pairs": self.candidate_pairs,
            "pair_features": self.pair_features,
            "global_features": self.global_features,
            "lambda2_norm": self.lambda2_norm,
            "n": self.n,
            "rho_target": self.rho_target,
            "rho_current": self.rho_current,
        }

    @staticmethod
    def from_dict(data: Dict) -> "GraphObservation":
        return GraphObservation(
            node_features=data["node_features"],
            edge_index=data["edge_index"],
            candidate_pairs=data["candidate_pairs"],
            pair_features=data["pair_features"],
            global_features=data["global_features"],
            lambda2_norm=float(data["lambda2_norm"]),
            n=int(data["n"]),
            rho_target=float(data["rho_target"]),
            rho_current=float(data["rho_current"]),
        )


class GraphEnv:
    def __init__(
        self,
        env_id: int,
        scheduler: CurriculumScheduler,
        top_k: int,
        dist_cap: int,
        terminal_bonus_coef: float,
        seed: int,
        rl_variant: str = "full",
        compute_spectral_each_step: bool = True,
        incremental_observation: bool = True,
        reward_alpha: float = 0.5,
        reward_eta: float = 1.0,
        reward_alpha_1: float = 0.15,
        reward_alpha_2: float = 0.70,
        reward_alpha_3: float = 0.15,
        reward_eta_p: float = 1.0,
        shaping_gamma: float = 0.99,
        spectral_oversample: int = 3,
        spectral_exact_reset_every: int = 50,
        spectral_power_iters: int = 4,
        spectral_drift_threshold: float = 1e-8,
        spectral_soft_band_rel: float = 1e-2,
        tier1_topk_dr: int = 192,
        tier1_topk_spectral: int = 64,
        tier2_survivor_size: int = 64,
        tier3_top: int = 48,
        tier3_random: int = 16,
    ):
        self.env_id = env_id
        self.scheduler = scheduler
        self.top_k = top_k
        self.dist_cap = dist_cap
        self.terminal_bonus_coef = terminal_bonus_coef
        self.reward_alpha = float(reward_alpha)
        self.reward_eta = float(reward_eta)
        self.reward_alpha_1 = float(reward_alpha_1)
        self.reward_alpha_2 = float(reward_alpha_2)
        self.reward_alpha_3 = float(reward_alpha_3)
        self.reward_eta_p = float(reward_eta_p)
        self.shaping_gamma = float(shaping_gamma)
        self.spectral_oversample = int(spectral_oversample)
        self.spectral_exact_reset_every = int(spectral_exact_reset_every)
        self.spectral_power_iters = int(spectral_power_iters)
        self.spectral_drift_threshold = float(spectral_drift_threshold)
        self.spectral_soft_band_rel = float(spectral_soft_band_rel)
        self.tier1_topk_dr = int(tier1_topk_dr)
        self.tier1_topk_spectral = int(tier1_topk_spectral)
        self.tier2_survivor_size = int(tier2_survivor_size)
        self.tier3_top = int(tier3_top)
        self.tier3_random = int(tier3_random)
        self.rl_variant = str(rl_variant)
        self.compute_spectral_each_step = bool(compute_spectral_each_step)
        self.incremental_observation = bool(incremental_observation)
        if self.rl_variant not in {"full", "lite_v2"}:
            raise ValueError(f"Unsupported rl_variant: {self.rl_variant}")

        self.py_rng = random.Random(seed + 10007 * env_id)
        self.np_rng = np.random.RandomState(seed + 20011 * env_id)
        self._init_seed_base = int(seed + 30013 * env_id)

        self.n: int = 0
        self.adj: Optional[np.ndarray] = None
        self.rho_target: float = 0.0
        self.m_target: int = 0
        self.episode_return: float = 0.0
        self.episode_len: int = 0
        self.current_lambda2: float = 0.0
        self.current_lambda3: float = 0.0
        self.current_lambda4: float = 0.0
        self.current_rg: float = 0.0
        self.current_multiplicity: int = 1
        self.current_phi2: Optional[np.ndarray] = None
        self.current_phi3: Optional[np.ndarray] = None
        self.current_phi4: Optional[np.ndarray] = None
        self.current_evecs: Optional[np.ndarray] = None
        self.current_P_min: float = 0.0
        self._deg_cache: Optional[np.ndarray] = None
        self._a2_counts: Optional[np.ndarray] = None
        self._triangles_per_node: Optional[np.ndarray] = None
        self._dist_matrix: Optional[np.ndarray] = None
        self._pair_i: Optional[np.ndarray] = None
        self._pair_j: Optional[np.ndarray] = None
        self._pair_is_nonedge: Optional[np.ndarray] = None
        self._pair_to_index: Optional[np.ndarray] = None
        self._eye_mask: Optional[np.ndarray] = None
        self._spectral_tracker: Optional[SpectralTracker] = None
        self._shaper: Optional[PotentialShaper] = None
        self._last_diagnostics: Dict[str, Any] = {}
        self._episode_init_metadata: Dict[str, Any] = default_path_init_metadata(0)

    # Empirical η(n) calibration from estimate_eta.py
    _ETA_TABLE_NS = np.array([8, 12, 16, 24, 32, 48, 64], dtype=np.float64)
    _ETA_TABLE_VALS = np.array(
        [0.803010, 0.368719, 0.104369, 0.019933, 0.006965, 0.001646, 0.000596],
        dtype=np.float64,
    )

    def _effective_eta(self) -> float:
        """Return η for the current graph size n.

        If reward_eta > 0 it is used directly (config override).
        Otherwise η is obtained by linear interpolation of the
        empirically calibrated table (see estimate_eta.py).
        """
        if float(self.reward_eta) > 0.0:
            return float(self.reward_eta)
        n_f = float(max(2, self.n))
        # Clip to the table range and interpolate
        ns = type(self)._ETA_TABLE_NS
        vs = type(self)._ETA_TABLE_VALS
        if n_f <= float(ns[0]):
            return float(vs[0])
        if n_f >= float(ns[-1]):
            return float(vs[-1])
        return float(np.interp(n_f, ns, vs))

    def _init_spectral_tracker(self) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, float, int, np.ndarray]:
        """Create a fresh SpectralTracker from the current adjacency.

        Returns the initial spectral features (same tuple as spectral_features).
        """
        if self.adj is None:
            raise RuntimeError("adjacency not set")
        self._spectral_tracker = SpectralTracker(
            self.adj,
            oversample=self.spectral_oversample,
            exact_reset_every=self.spectral_exact_reset_every,
            power_iters=self.spectral_power_iters,
            drift_threshold=self.spectral_drift_threshold,
            soft_band_rel=self.spectral_soft_band_rel,
        )
        self._shaper = PotentialShaper(alpha1=self.reward_alpha_1, gamma=self.shaping_gamma)
        self._shaper.reset(self._spectral_tracker)
        return self._spectral_tracker.get_spectral_features()

    def _compute_all_pairs_shortest_path(self, adj: np.ndarray) -> np.ndarray:
        n = adj.shape[0]
        dist = np.full((n, n), fill_value=n + 1, dtype=np.int64)
        for src in range(n):
            dist[src, src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                du = dist[src, u]
                for v in np.where(adj[u] > 0)[0]:
                    if dist[src, v] > du + 1:
                        dist[src, v] = du + 1
                        q.append(int(v))
        return dist

    def _get_pair_arrays(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return (pair_i, pair_j, pair_is_nonedge, dist_matrix), using the
        incrementally-maintained cache when available and falling back to a
        fresh O(n^2) computation otherwise."""
        if (
            self._pair_i is not None
            and self._pair_j is not None
            and self._pair_is_nonedge is not None
        ):
            return self._pair_i, self._pair_j, self._pair_is_nonedge, self._dist_matrix
        n = int(self.adj.shape[0])
        rows, cols = np.triu_indices(n, k=1)
        pair_i = rows.astype(np.int64, copy=False)
        pair_j = cols.astype(np.int64, copy=False)
        pair_is_nonedge = self.adj[pair_i, pair_j] == 0
        dist_matrix = truncated_all_pairs_shortest_path(self.adj, self.dist_cap)
        return pair_i, pair_j, pair_is_nonedge, dist_matrix

    def _initialize_incremental_state(self) -> None:
        if self.adj is None:
            self._deg_cache = None
            self._a2_counts = None
            self._triangles_per_node = None
            self._dist_matrix = None
            self._pair_i = None
            self._pair_j = None
            self._pair_is_nonedge = None
            self._pair_to_index = None
            self._eye_mask = None
            return

        n = self.adj.shape[0]
        adj_bool = self.adj > 0
        adj_i = adj_bool.astype(np.int64, copy=False)

        self._deg_cache = np.sum(adj_bool, axis=1, dtype=np.float64)
        self._a2_counts = adj_i @ adj_i
        # Each triangle touching node i contributes 2 to (A^3)_{ii}.
        self._triangles_per_node = (np.diag(adj_i @ adj_i @ adj_i).astype(np.float64) / 2.0)
        self._dist_matrix = self._compute_all_pairs_shortest_path(self.adj)

        rows, cols = np.triu_indices(n, k=1)
        self._pair_i = rows.astype(np.int64, copy=False)
        self._pair_j = cols.astype(np.int64, copy=False)
        self._pair_is_nonedge = (self.adj[self._pair_i, self._pair_j] == 0)

        pair_to_index = np.full((n, n), -1, dtype=np.int64)
        pair_idx = np.arange(self._pair_i.size, dtype=np.int64)
        pair_to_index[self._pair_i, self._pair_j] = pair_idx
        pair_to_index[self._pair_j, self._pair_i] = pair_idx
        self._pair_to_index = pair_to_index
        self._eye_mask = np.eye(n, dtype=bool)

    def _update_incremental_state_after_add(
        self,
        i: int,
        j: int,
        adj_row_i_before: np.ndarray,
        adj_row_j_before: np.ndarray,
        common_neighbors: np.ndarray,
    ) -> None:
        if (
            self._deg_cache is None
            or self._a2_counts is None
            or self._triangles_per_node is None
            or self._dist_matrix is None
            or self._pair_is_nonedge is None
            or self._pair_to_index is None
        ):
            self._initialize_incremental_state()
            return

        pair_idx = int(self._pair_to_index[i, j])
        if pair_idx >= 0:
            self._pair_is_nonedge[pair_idx] = False

        self._deg_cache[i] += 1.0
        self._deg_cache[j] += 1.0

        c = float(common_neighbors.size)
        if c > 0.0:
            self._triangles_per_node[i] += c
            self._triangles_per_node[j] += c
            self._triangles_per_node[common_neighbors] += 1.0

        # Exact rank-2 update for A^2 after adding edge (i, j).
        self._a2_counts[:, j] += adj_row_i_before
        self._a2_counts[:, i] += adj_row_j_before
        self._a2_counts[j, :] += adj_row_i_before
        self._a2_counts[i, :] += adj_row_j_before
        self._a2_counts[i, i] += 1
        self._a2_counts[j, j] += 1

        old_dist = self._dist_matrix
        via_ij = old_dist[:, i : i + 1] + 1 + old_dist[j : j + 1, :]
        via_ji = old_dist[:, j : j + 1] + 1 + old_dist[i : i + 1, :]
        self._dist_matrix = np.minimum(old_dist, np.minimum(via_ij, via_ji)).astype(
            np.int64, copy=False
        )

    def reset(
        self,
        global_env_step: int,
        *,
        init_mode: str = "path",
        initial_adj_builder: Optional[InitAdjBuilder] = None,
    ) -> GraphObservation:
        self.n = self.scheduler.sample_n(global_env_step, self.py_rng)
        self.rho_target = self.py_rng.random()
        self.m_target = m_target_from_rho(self.n, self.rho_target)
        # Avoid zero-decision episodes in training when rho maps to m=n-1.
        if self.n > 2 and self.m_target <= self.n - 1:
            self.m_target = self.n

        if init_mode == "backbone":
            if initial_adj_builder is None:
                raise ValueError("init_mode='backbone' requires initial_adj_builder.")
            init_seed = self._init_seed_base
            init_adj, init_meta = initial_adj_builder(
                int(self.n),
                float(self.rho_target),
                int(init_seed),
            )
            adj = np.asarray(init_adj, dtype=np.uint8)
            if adj.shape != (self.n, self.n):
                raise ValueError(
                    f"initial_adj shape mismatch: got {adj.shape}, expected {(self.n, self.n)}"
                )
            if not np.array_equal(adj, adj.T):
                raise ValueError("initial_adj must be symmetric for undirected graph.")
            if np.any(np.diag(adj) != 0):
                raise ValueError("initial_adj diagonal must be zero.")

            self.adj = adj.copy()
            self._episode_init_metadata = normalize_init_metadata(
                init_meta,
                n=self.n,
                init_mode="backbone",
            )
            if edge_count(self.adj) > self.m_target:
                raise RuntimeError(
                    "Initialization edge budget violation: "
                    f"m_init={edge_count(self.adj)} > m_target={self.m_target}"
                )
            self.episode_return = 0.0
            self.episode_len = 0
            if self.incremental_observation:
                self._initialize_incremental_state()
            spectral_tuple = self._init_spectral_tracker()
            lambda2, lambda3, lambda4, phi2, phi3, phi4, rg, mult, evecs = spectral_tuple
            self.current_rg = float(rg)
            self.current_multiplicity = int(mult)
            self.current_evecs = evecs.copy()
            self.current_P_min = (
                self._spectral_tracker.get_P_min()
                if self._spectral_tracker is not None
                else 0.0
            )
            return self._build_observation(spectral_cache=(lambda2, lambda3, lambda4, phi2, phi3, phi4, evecs))

        self.adj = build_path_adjacency(self.n)
        self._episode_init_metadata = normalize_init_metadata(
            default_path_init_metadata(self.n),
            n=self.n,
            init_mode="path",
        )
        self.episode_return = 0.0
        self.episode_len = 0
        if self.incremental_observation:
            self._initialize_incremental_state()
        spectral_tuple = self._init_spectral_tracker()
        lambda2, lambda3, lambda4, phi2, phi3, phi4, rg, mult, evecs = spectral_tuple
        self.current_rg = float(rg)
        self.current_multiplicity = int(mult)
        self.current_evecs = evecs.copy()
        # Seed P_min from the tracker, as the backbone branch above does.
        # Left at 0.0 the first step of the episode would score
        # Delta P_min = P_min_new - 0 instead of a true delta.
        self.current_P_min = (
            self._spectral_tracker.get_P_min()
            if self._spectral_tracker is not None
            else 0.0
        )
        return self._build_observation(spectral_cache=(lambda2, lambda3, lambda4, phi2, phi3, phi4, evecs))

    def reset_with_target(
        self,
        n: int,
        rho_target: float,
        initial_adj: Optional[np.ndarray] = None,
        init_mode: str = "path",
        init_metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphObservation:
        self.n = int(n)
        self.rho_target = float(rho_target)
        self.m_target = m_target_from_rho(self.n, self.rho_target)
        if initial_adj is None:
            self.adj = build_path_adjacency(self.n)
            self._episode_init_metadata = normalize_init_metadata(
                init_metadata if init_metadata is not None else default_path_init_metadata(self.n),
                n=self.n,
                init_mode="path",
            )
        else:
            adj = np.asarray(initial_adj, dtype=np.uint8)
            if adj.shape != (self.n, self.n):
                raise ValueError(
                    f"initial_adj shape mismatch: got {adj.shape}, expected {(self.n, self.n)}"
                )
            if not np.array_equal(adj, adj.T):
                raise ValueError("initial_adj must be symmetric for undirected graph.")
            if np.any(np.diag(adj) != 0):
                raise ValueError("initial_adj diagonal must be zero.")
            self.adj = adj.copy()
            self._episode_init_metadata = normalize_init_metadata(
                init_metadata,
                n=self.n,
                init_mode=init_mode,
            )
            if edge_count(self.adj) > self.m_target:
                raise RuntimeError(
                    "Initialization edge budget violation: "
                    f"m_init={edge_count(self.adj)} > m_target={self.m_target}"
                )
        self.episode_return = 0.0
        self.episode_len = 0
        if self.incremental_observation:
            self._initialize_incremental_state()
        spectral_tuple = self._init_spectral_tracker()
        lambda2, lambda3, lambda4, phi2, phi3, phi4, rg, mult, evecs = spectral_tuple
        self.current_rg = float(rg)
        self.current_multiplicity = int(mult)
        self.current_evecs = evecs.copy()
        self.current_P_min = (
            self._spectral_tracker.get_P_min()
            if self._spectral_tracker is not None
            else 0.0
        )
        return self._build_observation(spectral_cache=(lambda2, lambda3, lambda4, phi2, phi3, phi4, evecs))

    def _build_observation(
        self,
        *,
        spectral_cache: Optional[Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> GraphObservation:
        if self.adj is None:
            raise RuntimeError("Environment not initialized")

        n = self.n
        m = edge_count(self.adj)
        rho_current = normalized_density(n, m)

        use_spectral = (self.rl_variant == "full") or self.compute_spectral_each_step
        if spectral_cache is not None:
            lambda2, lambda3, lambda4, phi2, phi3, phi4, evecs = spectral_cache
            self.current_lambda2 = float(lambda2)
            self.current_lambda3 = float(lambda3)
            self.current_lambda4 = float(lambda4)
            self.current_phi2 = np.asarray(phi2, dtype=np.float64).copy()
            self.current_phi3 = np.asarray(phi3, dtype=np.float64).copy()
            self.current_phi4 = np.asarray(phi4, dtype=np.float64).copy()
            self.current_evecs = evecs.copy()
        elif use_spectral:
            lambda2, lambda3, lambda4, phi2, phi3, phi4, rg, mult, evecs = spectral_features(self.adj)
            self.current_lambda2 = float(lambda2)
            self.current_lambda3 = float(lambda3)
            self.current_lambda4 = float(lambda4)
            self.current_rg = float(rg)
            self.current_multiplicity = int(mult)
            self.current_phi2 = np.asarray(phi2, dtype=np.float64).copy()
            self.current_phi3 = np.asarray(phi3, dtype=np.float64).copy()
            self.current_phi4 = np.asarray(phi4, dtype=np.float64).copy()
            self.current_evecs = evecs.copy()
        else:
            lambda2 = float(self.current_lambda2)
            lambda3 = float(self.current_lambda3)
            lambda4 = float(self.current_lambda4)
            phi2 = (
                np.asarray(self.current_phi2, dtype=np.float64)
                if self.current_phi2 is not None
                else np.zeros((n,), dtype=np.float64)
            )
            phi3 = (
                np.asarray(self.current_phi3, dtype=np.float64)
                if self.current_phi3 is not None
                else np.zeros((n,), dtype=np.float64)
            )
            phi4 = (
                np.asarray(self.current_phi4, dtype=np.float64)
                if self.current_phi4 is not None
                else np.zeros((n,), dtype=np.float64)
            )

        if self.rl_variant == "full":
            node_features, deg = build_node_features_full(
                adj=self.adj,
                n=n,
                phi2=phi2,
                phi3=phi3,
                incremental_observation=self.incremental_observation,
                deg_cache=self._deg_cache,
                a2_counts=self._a2_counts,
                triangles_per_node=self._triangles_per_node,
                eye_mask=self._eye_mask,
            )
            global_features = build_global_features_full(
                n=n,
                rho_target=self.rho_target,
                rho_current=rho_current,
                lambda2=lambda2,
                lambda3=lambda3,
                lambda4=lambda4,
            )
            if self._spectral_tracker is None:
                raise RuntimeError("Spectral tracker not available for full variant observation")
            pair_i, pair_j, pair_is_nonedge, dist_matrix = self._get_pair_arrays()
            candidacy = run_candidacy_pipeline(
                tracker=self._spectral_tracker,
                pair_i=pair_i,
                pair_j=pair_j,
                pair_is_nonedge=pair_is_nonedge,
                dist_matrix=dist_matrix,
                deg=deg,
                adj=self.adj,
                n=n,
                dist_cap=self.dist_cap,
                tier1_topk_dr=self.tier1_topk_dr,
                tier1_topk_spectral=self.tier1_topk_spectral,
                tier2_survivor_size=self.tier2_survivor_size,
                tier3_top=self.tier3_top,
                tier3_random=self.tier3_random,
                rng=self.np_rng,
            )
            candidate_pairs = candidacy.candidate_pairs
            pair_features = candidacy.pair_features
            self._last_diagnostics.update(
                {
                    "tier0_count": candidacy.tier0_count,
                    "tier1_count": candidacy.tier1_count,
                    "tier2_survivor_count": candidacy.tier2_survivor_count,
                    "candidacy_multiplicity": candidacy.multiplicity,
                    "candidacy_soft_degenerate": candidacy.soft_degenerate,
                }
            )
        elif self.rl_variant == "lite_v2":
            node_features, deg = build_node_features_lite(
                adj=self.adj,
                n=n,
                incremental_observation=self.incremental_observation,
                deg_cache=self._deg_cache,
                a2_counts=self._a2_counts,
                triangles_per_node=self._triangles_per_node,
                eye_mask=self._eye_mask,
            )
            global_features = build_global_features_lite(
                n=n,
                rho_target=self.rho_target,
                rho_current=rho_current,
            )
            candidate_pairs, pair_features = build_candidate_features_lite(
                adj=self.adj,
                deg=deg,
                n=n,
                dist_cap=self.dist_cap,
                incremental_observation=self.incremental_observation,
                pair_i=self._pair_i,
                pair_j=self._pair_j,
                pair_is_nonedge=self._pair_is_nonedge,
                dist_matrix=self._dist_matrix,
            )
        else:  # pragma: no cover
            raise ValueError(f"Unsupported rl_variant: {self.rl_variant}")
        edge_index = build_edge_index(self.adj)

        return GraphObservation(
            node_features=node_features,
            edge_index=edge_index,
            candidate_pairs=candidate_pairs,
            pair_features=pair_features,
            global_features=global_features,
            lambda2_norm=float(lambda2 / max(1, n)),
            n=n,
            rho_target=self.rho_target,
            rho_current=rho_current,
        )

    def step(
        self,
        action_pair: Tuple[int, int],
        extra_reward: float = 0.0,
    ) -> Tuple[GraphObservation, float, bool, Dict]:
        if self.adj is None:
            raise RuntimeError("Environment not initialized")

        i, j = action_pair
        if i == j or i < 0 or j < 0 or i >= self.n or j >= self.n:
            raise ValueError(f"Invalid action pair {action_pair}")
        if self.adj[i, j] == 1:
            raise ValueError(f"Action pair already an edge: {action_pair}")

        old_lambda2 = self.current_lambda2
        old_P_min = self.current_P_min
        adj_row_i_before = self.adj[i].astype(np.int64, copy=True)
        adj_row_j_before = self.adj[j].astype(np.int64, copy=True)
        common_neighbors = np.flatnonzero(
            np.logical_and(adj_row_i_before > 0, adj_row_j_before > 0)
        ).astype(np.int64, copy=False)

        self.adj[i, j] = 1
        self.adj[j, i] = 1
        self.episode_len += 1
        if self.incremental_observation:
            self._update_incremental_state_after_add(
                i=i,
                j=j,
                adj_row_i_before=adj_row_i_before,
                adj_row_j_before=adj_row_j_before,
                common_neighbors=common_neighbors,
            )

        use_spectral = (self.rl_variant == "full") or self.compute_spectral_each_step
        done = edge_count(self.adj) >= self.m_target
        reward = 0.0

        if use_spectral:
            old_rg = float(self.current_rg)
            # Incremental rank-1 update via SpectralTracker (Tier 2)
            if self._spectral_tracker is not None:
                self._spectral_tracker.add_edge(i, j)
                spectral_tuple = self._spectral_tracker.get_spectral_features()
            else:
                spectral_tuple = spectral_features(self.adj)
            new_lambda2, new_lambda3, new_lambda4, phi2, phi3, phi4, new_rg, new_mult, new_evecs = spectral_tuple
            self.current_lambda2 = float(new_lambda2)
            self.current_lambda3 = float(new_lambda3)
            self.current_lambda4 = float(new_lambda4)
            self.current_rg = float(new_rg)
            self.current_multiplicity = int(new_mult)
            self.current_evecs = new_evecs.copy()
            self.current_phi2 = np.asarray(phi2, dtype=np.float64).copy()
            self.current_phi3 = np.asarray(phi3, dtype=np.float64).copy()
            self.current_phi4 = np.asarray(phi4, dtype=np.float64).copy()

            # Blended reward: r_t = α₁·Δλ₂/n + α₂·η_R·ΔR_G/n² + α₃·η_P·ΔP_min
            # ΔR_G, ΔP_min positive when property improves.
            nf = float(max(1, self.n))
            delta_l2 = float(new_lambda2 - old_lambda2)
            delta_rg = float(old_rg - new_rg)

            # P_min from tracker if available
            if self._spectral_tracker is not None:
                new_P_min = self._spectral_tracker.get_P_min()
            else:
                new_P_min = old_P_min
            delta_P_min = float(new_P_min - old_P_min)
            self.current_P_min = new_P_min

            eta_R = self._effective_eta()
            eta_P = float(self.reward_eta_p)

            alpha1 = float(self.reward_alpha_1)
            alpha2 = float(self.reward_alpha_2)
            alpha3 = float(self.reward_alpha_3)

            reward_l2 = delta_l2 / nf
            reward_rg = eta_R * delta_rg / (nf * nf)
            reward_pmin = eta_P * delta_P_min

            reward = alpha1 * reward_l2 + alpha2 * reward_rg + alpha3 * reward_pmin

            shaping_reward = 0.0
            if self._shaper is not None and self._spectral_tracker is not None:
                shaping_reward = self._shaper.step(self._spectral_tracker)
                reward += shaping_reward

            if self._spectral_tracker is not None:
                k_remaining = max(0, self.m_target - edge_count(self.adj))
                self._last_diagnostics.update(log_diagnostics(self._spectral_tracker, k_remaining=k_remaining))
                self._last_diagnostics["shaping_reward"] = float(shaping_reward)

            if done:
                # Terminal bonus based on final λ₂
                reward += self.terminal_bonus_coef * (new_lambda2 / nf)
            obs = self._build_observation(
                spectral_cache=(new_lambda2, new_lambda3, new_lambda4, phi2, phi3, phi4, new_evecs)
            )
            terminal_lambda2_norm = (new_lambda2 / nf) if done else None
            terminal_lambda2 = float(new_lambda2) if done else None
        else:
            # Fast inference path for lite_v2: avoid per-step spectral decomposition.
            if done:
                new_lambda2, new_lambda3, new_lambda4, phi2, phi3, phi4, new_rg, new_mult, new_evecs = spectral_features(self.adj)
                self.current_lambda2 = float(new_lambda2)
                self.current_lambda3 = float(new_lambda3)
                self.current_lambda4 = float(new_lambda4)
                self.current_rg = float(new_rg)
                self.current_multiplicity = int(new_mult)
                self.current_evecs = new_evecs.copy()
                self.current_phi2 = np.asarray(phi2, dtype=np.float64).copy()
                self.current_phi3 = np.asarray(phi3, dtype=np.float64).copy()
                self.current_phi4 = np.asarray(phi4, dtype=np.float64).copy()
                terminal_lambda2_norm = new_lambda2 / float(max(1, self.n))
                terminal_lambda2 = float(new_lambda2)
            else:
                terminal_lambda2_norm = None
                terminal_lambda2 = None
            obs = self._build_observation()

        reward += float(extra_reward)
        self.episode_return += reward

        info: Dict = {
            "episode_done": done,
            "episode_return": self.episode_return if done else None,
            "episode_len": self.episode_len if done else None,
            "terminal_lambda2_norm": terminal_lambda2_norm,
            "terminal_lambda2": terminal_lambda2,
            "rho_target": self.rho_target,
            "n": self.n,
        }
        if self.rl_variant == "full":
            info.update(self._last_diagnostics)
        if done:
            info.update(self._episode_init_metadata)
        return obs, float(reward), done, info

    def state_dict(self) -> Dict:
        return {
            "env_id": self.env_id,
            "rl_variant": self.rl_variant,
            "compute_spectral_each_step": self.compute_spectral_each_step,
            "incremental_observation": self.incremental_observation,
            "n": self.n,
            "adj": self.adj,
            "rho_target": self.rho_target,
            "m_target": self.m_target,
            "episode_return": self.episode_return,
            "episode_len": self.episode_len,
            "current_lambda2": self.current_lambda2,
            "current_lambda3": self.current_lambda3,
            "current_lambda4": self.current_lambda4,
            "current_rg": self.current_rg,
            "current_multiplicity": self.current_multiplicity,
            "current_P_min": self.current_P_min,
            "current_phi2": self.current_phi2,
            "current_phi3": self.current_phi3,
            "current_phi4": self.current_phi4,
            "current_evecs": self.current_evecs,
            "episode_init_metadata": dict(self._episode_init_metadata),
            "spectral_tracker": (
                self._spectral_tracker.state_dict()
                if self._spectral_tracker is not None
                else None
            ),
            "shaper": self._shaper.state_dict() if self._shaper is not None else None,
            "py_rng_state": self.py_rng.getstate(),
            "np_rng_state": self.np_rng.get_state(),
        }

    def load_state_dict(self, state: Dict) -> None:
        self.rl_variant = str(state.get("rl_variant", self.rl_variant))
        self.compute_spectral_each_step = bool(
            state.get("compute_spectral_each_step", self.compute_spectral_each_step)
        )
        self.n = int(state["n"])
        self.adj = state["adj"].copy() if state["adj"] is not None else None
        self.rho_target = float(state["rho_target"])
        self.m_target = int(state["m_target"])
        self.episode_return = float(state["episode_return"])
        self.episode_len = int(state["episode_len"])
        self.current_lambda2 = float(state["current_lambda2"])
        self.current_lambda3 = float(state["current_lambda3"])
        self.current_lambda4 = float(state.get("current_lambda4", 0.0))
        self.current_rg = float(state.get("current_rg", 0.0))
        self.current_multiplicity = int(state.get("current_multiplicity", 1))
        self.current_P_min = float(state.get("current_P_min", 0.0))
        raw_phi2 = state.get("current_phi2")
        raw_phi3 = state.get("current_phi3")
        raw_phi4 = state.get("current_phi4")
        raw_evecs = state.get("current_evecs")
        self.current_phi2 = (
            np.asarray(raw_phi2, dtype=np.float64).copy()
            if raw_phi2 is not None
            else None
        )
        self.current_phi3 = (
            np.asarray(raw_phi3, dtype=np.float64).copy()
            if raw_phi3 is not None
            else None
        )
        self.current_phi4 = (
            np.asarray(raw_phi4, dtype=np.float64).copy()
            if raw_phi4 is not None
            else None
        )
        self.current_evecs = (
            np.asarray(raw_evecs, dtype=np.float64).copy()
            if raw_evecs is not None
            else None
        )
        raw_meta = state.get("episode_init_metadata")
        raw_mode = "path"
        if isinstance(raw_meta, dict):
            raw_mode = str(raw_meta.get("init_mode", "path"))
        self._episode_init_metadata = normalize_init_metadata(
            raw_meta if isinstance(raw_meta, dict) else None,
            n=max(0, self.n),
            init_mode=raw_mode,
        )
        self.py_rng.setstate(state["py_rng_state"])
        self.np_rng.set_state(state["np_rng_state"])
        self.incremental_observation = bool(state.get("incremental_observation", True))
        if self.incremental_observation:
            self._initialize_incremental_state()
        tracker_state = state.get("spectral_tracker")
        if tracker_state is not None:
            self._spectral_tracker = SpectralTracker.from_state_dict(tracker_state)
        else:
            self._spectral_tracker = None

        shaper_state = state.get("shaper")
        if shaper_state is not None:
            self._shaper = PotentialShaper.from_state_dict(shaper_state)
        else:
            self._shaper = PotentialShaper(alpha1=self.reward_alpha_1, gamma=self.shaping_gamma)


class VectorGraphEnvManager:
    def __init__(
        self,
        num_envs: int,
        scheduler: CurriculumScheduler,
        top_k: int,
        dist_cap: int,
        terminal_bonus_coef: float,
        base_seed: int,
        rl_variant: str = "full",
        compute_spectral_each_step: bool = True,
        incremental_observation: bool = True,
        init_mode: str = "path",
        initial_adj_builder: Optional[InitAdjBuilder] = None,
        reward_alpha: float = 0.5,
        reward_eta: float = 1.0,
        reward_alpha_1: float = 0.15,
        reward_alpha_2: float = 0.70,
        reward_alpha_3: float = 0.15,
        reward_eta_p: float = 1.0,
        shaping_gamma: float = 0.99,
        spectral_oversample: int = 3,
        spectral_exact_reset_every: int = 50,
        spectral_power_iters: int = 4,
        spectral_drift_threshold: float = 1e-8,
        spectral_soft_band_rel: float = 1e-2,
        tier1_topk_dr: int = 192,
        tier1_topk_spectral: int = 64,
        tier2_survivor_size: int = 64,
        tier3_top: int = 48,
        tier3_random: int = 16,
    ):
        if init_mode not in {"path", "backbone"}:
            raise ValueError(f"Unsupported init_mode: {init_mode}")
        if init_mode == "backbone" and initial_adj_builder is None:
            raise ValueError("init_mode='backbone' requires initial_adj_builder.")

        self.init_mode = init_mode
        self.rl_variant = str(rl_variant)
        self.compute_spectral_each_step = bool(compute_spectral_each_step)
        self.reward_alpha = float(reward_alpha)
        self.reward_eta = float(reward_eta)
        self.reward_alpha_1 = float(reward_alpha_1)
        self.reward_alpha_2 = float(reward_alpha_2)
        self.reward_alpha_3 = float(reward_alpha_3)
        self.reward_eta_p = float(reward_eta_p)
        self.initial_adj_builder = initial_adj_builder
        self.envs: List[GraphEnv] = [
            GraphEnv(
                env_id=i,
                scheduler=scheduler,
                top_k=top_k,
                dist_cap=dist_cap,
                terminal_bonus_coef=terminal_bonus_coef,
                seed=base_seed,
                rl_variant=self.rl_variant,
                compute_spectral_each_step=self.compute_spectral_each_step,
                incremental_observation=incremental_observation,
                reward_alpha=self.reward_alpha,
                reward_eta=self.reward_eta,
                reward_alpha_1=self.reward_alpha_1,
                reward_alpha_2=self.reward_alpha_2,
                reward_alpha_3=self.reward_alpha_3,
                reward_eta_p=self.reward_eta_p,
                shaping_gamma=shaping_gamma,
                spectral_oversample=spectral_oversample,
                spectral_exact_reset_every=spectral_exact_reset_every,
                spectral_power_iters=spectral_power_iters,
                spectral_drift_threshold=spectral_drift_threshold,
                spectral_soft_band_rel=spectral_soft_band_rel,
                tier1_topk_dr=tier1_topk_dr,
                tier1_topk_spectral=tier1_topk_spectral,
                tier2_survivor_size=tier2_survivor_size,
                tier3_top=tier3_top,
                tier3_random=tier3_random,
            )
            for i in range(num_envs)
        ]
        self.current_obs: List[Optional[GraphObservation]] = [None for _ in range(num_envs)]

    def _reset_until_actionable(self, env: GraphEnv, global_env_step: int) -> GraphObservation:
        # In backbone mode it is possible to initialize directly at target edge budget
        # (m_init == m_target). Training should skip such zero-completion episodes.
        max_attempts = 256
        for _ in range(max_attempts):
            obs = env.reset(
                global_env_step,
                init_mode=self.init_mode,
                initial_adj_builder=self.initial_adj_builder,
            )
            if env.adj is None:
                raise RuntimeError("Environment adjacency missing after reset")
            if edge_count(env.adj) < env.m_target and obs.candidate_pairs.shape[0] > 0:
                return obs
        raise RuntimeError(
            "Unable to sample actionable episode after many attempts. "
            "This usually means initialization frequently saturates m_target."
        )

    def ensure_actionable_observations(self, global_env_step: int) -> None:
        for i, env in enumerate(self.envs):
            obs = self.current_obs[i]
            needs_reset = True
            if obs is not None and env.adj is not None:
                if edge_count(env.adj) < env.m_target and obs.candidate_pairs.shape[0] > 0:
                    needs_reset = False
            if needs_reset:
                self.current_obs[i] = self._reset_until_actionable(env, global_env_step)

    def reset_all(self, global_env_step: int) -> List[GraphObservation]:
        out = []
        for i, env in enumerate(self.envs):
            obs = self._reset_until_actionable(env, global_env_step)
            self.current_obs[i] = obs
            out.append(obs)
        return out

    def step(
        self,
        action_pairs: Sequence[Tuple[int, int]],
        global_env_step_after_batch: int,
        extra_rewards: Optional[Sequence[float]] = None,
    ) -> Tuple[List[GraphObservation], List[float], List[bool], List[Dict]]:
        next_obs: List[GraphObservation] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict] = []

        if extra_rewards is None:
            extra_rewards = [0.0 for _ in range(len(self.envs))]

        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step(action_pairs[i], extra_reward=float(extra_rewards[i]))
            if done:
                reset_obs = self._reset_until_actionable(env, global_env_step_after_batch)
                next_obs.append(reset_obs)
                self.current_obs[i] = reset_obs
            else:
                next_obs.append(obs)
                self.current_obs[i] = obs
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return next_obs, rewards, dones, infos

    def state_dict(self) -> Dict:
        return {
            "init_mode": self.init_mode,
            "rl_variant": self.rl_variant,
            "compute_spectral_each_step": self.compute_spectral_each_step,
            "env_states": [env.state_dict() for env in self.envs],
            "current_obs": [obs.to_dict() if obs is not None else None for obs in self.current_obs],
        }

    def load_state_dict(self, state: Dict) -> None:
        checkpoint_init_mode = str(state.get("init_mode", self.init_mode))
        if checkpoint_init_mode != self.init_mode:
            raise ValueError(
                f"Mismatched init_mode in checkpoint: {checkpoint_init_mode} != {self.init_mode}"
            )
        checkpoint_variant = str(state.get("rl_variant", self.rl_variant))
        if checkpoint_variant != self.rl_variant:
            raise ValueError(
                f"Mismatched rl_variant in checkpoint: {checkpoint_variant} != {self.rl_variant}"
            )
        checkpoint_compute_spectral = bool(
            state.get("compute_spectral_each_step", self.compute_spectral_each_step)
        )
        if checkpoint_compute_spectral != self.compute_spectral_each_step:
            raise ValueError(
                "Mismatched compute_spectral_each_step in checkpoint: "
                f"{checkpoint_compute_spectral} != {self.compute_spectral_each_step}"
            )
        env_states = state["env_states"]
        if len(env_states) != len(self.envs):
            raise ValueError("Mismatched number of environments in checkpoint")

        for env, env_state in zip(self.envs, env_states):
            env.load_state_dict(env_state)

        loaded_obs = state["current_obs"]
        self.current_obs = [
            GraphObservation.from_dict(obs_dict) if obs_dict is not None else None
            for obs_dict in loaded_obs
        ]

    def get_observations(self) -> List[GraphObservation]:
        obs = []
        for item in self.current_obs:
            if item is None:
                raise RuntimeError("Environment observations are not initialized")
            obs.append(item)
        return obs
