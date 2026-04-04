from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import numpy as np

from .candidates import CandidateGenerator
from .features import edge_features, global_features, node_features
from .graph_core import (
    deterministic_tree_graph,
    edge_count,
    graph_edge_index,
    path_graph,
    star_graph,
    validate_nm,
)
from .incremental_linear_algebra import IncrementalState, IncrementalStateConfig
from .skeleton import (
    WarmStartMeta,
    deterministic_warm_start,
    deterministic_warm_start_expander,
    deterministic_warm_start_fib_hybrid,
    deterministic_warm_start_fib_s3_hybrid,
    deterministic_warm_start_rpartite_envelope,
    deterministic_warm_start_smallworld,
)


class RolloutPolicy(Protocol):
    name: str

    def select_action(self, obs: Dict[str, np.ndarray]) -> int:
        ...


@dataclass
class EnvConfig:
    init_mode: str = "deterministic_warm_start_expander"
    density_thresholds: Tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    target_degree_mode: str = "floor"
    expander_comm_penalty_scale: float = 1.0
    fib_geo_weight_scale: float = 1.0
    fib_comm_penalty_scale: float = 1.0
    fib_geo_weight_override: Optional[float] = None
    fib_comm_weight_override: Optional[float] = None
    smallworld_num_samples: int = 1
    fib_hybrid_fractions: Optional[Dict[str, float]] = None
    rpartite_r_max: int = 16
    rpartite_envelope_rank: int = 0
    rpartite_use_approx_filter: bool = True
    rpartite_require_acm_cert: bool = False
    rpartite_allow_uncertified_fallback: bool = True


class GraphAugmentationEnv:
    """
    Environment for sequentially adding edges until target m.
    """

    def __init__(
        self,
        n: int,
        m_target: int,
        candidate_generator: CandidateGenerator,
        state_config: Optional[IncrementalStateConfig] = None,
        env_config: Optional[EnvConfig] = None,
    ) -> None:
        validate_nm(n, m_target)
        self.n = n
        self.m_target = m_target
        self.candidate_generator = candidate_generator
        self.state_config = state_config or IncrementalStateConfig()
        self.env_config = env_config or EnvConfig()
        self.state: Optional[IncrementalState] = None
        self.initial_adj: Optional[np.ndarray] = None
        self.meta: Optional[WarmStartMeta] = None

    @property
    def done(self) -> bool:
        if self.state is None:
            return False
        return edge_count(self.state.adj) >= self.m_target

    def reset(self) -> Dict[str, np.ndarray]:
        if self.env_config.init_mode == "path":
            adj = path_graph(self.n)
            self.meta = WarmStartMeta(
                pocket="sparse",
                rho=0.0,
                backbone_degree=1,
                residual_edges=max(self.m_target - edge_count(adj), 0),
                init_mode="path",
            )
        elif self.env_config.init_mode == "star":
            adj = star_graph(self.n)
            self.meta = WarmStartMeta(
                pocket="sparse",
                rho=0.0,
                backbone_degree=max(1, self.n - 1),
                residual_edges=max(self.m_target - edge_count(adj), 0),
                init_mode="star",
            )
        elif self.env_config.init_mode == "tree":
            adj = deterministic_tree_graph(self.n, seed=int(self.n * 1009))
            self.meta = WarmStartMeta(
                pocket="sparse",
                rho=0.0,
                backbone_degree=max(1, int(adj.sum(axis=1).max())),
                residual_edges=max(self.m_target - edge_count(adj), 0),
                init_mode="tree",
            )
        elif self.env_config.init_mode == "deterministic_warm_start":
            adj, self.meta = deterministic_warm_start(
                self.n, self.m_target, thresholds=self.env_config.density_thresholds
            )
        elif self.env_config.init_mode == "deterministic_warm_start_expander":
            adj, self.meta = deterministic_warm_start_expander(
                self.n,
                self.m_target,
                thresholds=self.env_config.density_thresholds,
                comm_penalty_scale=self.env_config.expander_comm_penalty_scale,
            )
        elif self.env_config.init_mode == "deterministic_warm_start_smallworld":
            adj, self.meta = deterministic_warm_start_smallworld(
                self.n,
                self.m_target,
                thresholds=self.env_config.density_thresholds,
                num_samples=self.env_config.smallworld_num_samples,
            )
        elif self.env_config.init_mode == "deterministic_warm_start_fib_hybrid":
            adj, self.meta = deterministic_warm_start_fib_hybrid(
                self.n,
                self.m_target,
                thresholds=self.env_config.density_thresholds,
                extra_fraction_by_pocket=self.env_config.fib_hybrid_fractions,
                fib_geo_weight_scale=self.env_config.fib_geo_weight_scale,
                fib_comm_penalty_scale=self.env_config.fib_comm_penalty_scale,
                fib_geo_weight_override=self.env_config.fib_geo_weight_override,
                fib_comm_weight_override=self.env_config.fib_comm_weight_override,
            )
        elif self.env_config.init_mode == "deterministic_warm_start_fib_s3_hybrid":
            adj, self.meta = deterministic_warm_start_fib_s3_hybrid(
                self.n,
                self.m_target,
                thresholds=self.env_config.density_thresholds,
                extra_fraction_by_pocket=self.env_config.fib_hybrid_fractions,
                fib_geo_weight_scale=self.env_config.fib_geo_weight_scale,
                fib_comm_penalty_scale=self.env_config.fib_comm_penalty_scale,
                fib_geo_weight_override=self.env_config.fib_geo_weight_override,
                fib_comm_weight_override=self.env_config.fib_comm_weight_override,
            )
        elif self.env_config.init_mode == "deterministic_warm_start_rpartite_envelope":
            adj, self.meta = deterministic_warm_start_rpartite_envelope(
                self.n,
                self.m_target,
                thresholds=self.env_config.density_thresholds,
                r_max=self.env_config.rpartite_r_max,
                envelope_rank=self.env_config.rpartite_envelope_rank,
                use_approx_filter=self.env_config.rpartite_use_approx_filter,
                require_acm_cert=self.env_config.rpartite_require_acm_cert,
                allow_uncertified_fallback=self.env_config.rpartite_allow_uncertified_fallback,
            )
        else:
            raise ValueError(f"unknown init mode: {self.env_config.init_mode}")

        self.initial_adj = adj.copy()
        self.state = IncrementalState(adj, config=self.state_config)
        return self.observe()

    def observe(self) -> Dict[str, np.ndarray]:
        if self.state is None:
            raise RuntimeError("reset() must be called before observe()")
        return self._build_observation_for_state(self.state)

    def step(
        self, action: int, obs: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[Optional[Dict[str, np.ndarray]], bool, Dict[str, float]]:
        if self.state is None:
            raise RuntimeError("reset() must be called before step()")
        if obs is None:
            obs = self.observe()
        cands = obs["candidate_edges"]
        if cands.shape[0] == 0:
            return None, True, {"applied": 0.0}
        if action < 0 or action >= cands.shape[0]:
            raise IndexError(f"action out of range: {action} / {cands.shape[0]}")
        u, v = map(int, cands[action])
        self.state.add_edge(u, v)
        done = self.done
        nxt = None if done else self.observe()
        return nxt, done, {"applied": 1.0, "u": float(u), "v": float(v)}

    def _build_observation_for_state(self, state: IncrementalState) -> Dict[str, np.ndarray]:
        lam2, phi2, phi3 = state.spectral_chart(force=False)
        cand = self.candidate_generator.generate(state)
        nfeat = node_features(state.adj, phi2, phi3)
        gfeat = global_features(self.n, self.m_target, edge_count(state.adj), lam2)
        efeat = edge_features(
            state.adj,
            cand.edges,
            cand.er_scores,
            phi2,
            phi3,
            gfeat,
            self.m_target,
            target_degree_mode=self.env_config.target_degree_mode,  # type: ignore[arg-type]
        )

        return {
            "node_features": nfeat.astype(np.float32),
            "global_features": gfeat.astype(np.float32),
            "edge_features": efeat.astype(np.float32),
            "candidate_edges": cand.edges.astype(np.int64),
            "er_scores": cand.er_scores.astype(np.float32),
            "graph_edges": graph_edge_index(state.adj).astype(np.int64),
        }

    def rollout_completion(
        self,
        state: IncrementalState,
        policy: RolloutPolicy,
    ) -> float:
        sim = state.clone()
        while edge_count(sim.adj) < self.m_target:
            obs = self._build_observation_for_state(sim)
            if obs["candidate_edges"].shape[0] == 0:
                break
            action = int(policy.select_action(obs))
            action = max(0, min(action, obs["candidate_edges"].shape[0] - 1))
            u, v = map(int, obs["candidate_edges"][action])
            sim.add_edge(u, v)
        lam2, _, _ = sim.spectral_chart(force=True)
        return float(lam2)
