from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .baseline_manager import RolloutPolicy
from .config import RewardConfig
from .env import GraphAugmentationEnv
from .graph_core import edge_count
from .incremental_linear_algebra import IncrementalState


@dataclass
class StepReward:
    reward: float
    ratio: float
    regularity_bonus: float
    q_target_chosen: float
    q_target_baseline: float
    baseline_action: int


class RewardComputer:
    def __init__(self, cfg: RewardConfig) -> None:
        self.cfg = cfg

    def compute_step_reward(
        self,
        env: GraphAugmentationEnv,
        state_before: IncrementalState,
        obs: Dict[str, np.ndarray],
        action: int,
        baseline_policy: RolloutPolicy,
    ) -> StepReward:
        cands = obs["candidate_edges"]
        if cands.shape[0] == 0:
            return StepReward(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        action = int(np.clip(action, 0, cands.shape[0] - 1))
        baseline_action = int(np.clip(baseline_policy.select_action(obs), 0, cands.shape[0] - 1))
        cu, cv = map(int, cands[action])
        bu, bv = map(int, cands[baseline_action])

        q_target_chosen = 0.0
        q_target_baseline = 0.0
        ratio = 0.0
        if self.cfg.use_step_shaping:
            chosen_state = state_before.clone()
            chosen_state.add_edge(cu, cv)
            base_state = state_before.clone()
            base_state.add_edge(bu, bv)
            q_target_chosen = env.rollout_completion(chosen_state, baseline_policy)
            q_target_baseline = env.rollout_completion(base_state, baseline_policy)
            ratio = (q_target_chosen - q_target_baseline) / max(
                self.cfg.eps_denom, abs(q_target_baseline)
            )

        clipped = float(np.clip(ratio, -self.cfg.step_reward_clip, self.cfg.step_reward_clip))
        reg_bonus = self._regularity_bonus(state_before, (cu, cv), env.m_target)
        reward = self.cfg.shaping_scale * clipped + reg_bonus
        return StepReward(
            reward=float(reward),
            ratio=float(ratio),
            regularity_bonus=float(reg_bonus),
            q_target_chosen=float(q_target_chosen),
            q_target_baseline=float(q_target_baseline),
            baseline_action=baseline_action,
        )

    def compute_terminal_reward(
        self,
        env: GraphAugmentationEnv,
        initial_state: IncrementalState,
        final_lambda2: float,
        baseline_policy: RolloutPolicy,
    ) -> float:
        if self.cfg.terminal_weight <= 0.0:
            return 0.0
        base_lam2 = env.rollout_completion(initial_state.clone(), baseline_policy)
        rel = (final_lambda2 - base_lam2) / max(self.cfg.eps_denom, abs(base_lam2))
        return float(self.cfg.terminal_weight * rel)

    def _regularity_bonus(
        self,
        state_before: IncrementalState,
        edge: tuple[int, int],
        m_target: int,
    ) -> float:
        if self.cfg.regularity_bonus_coef <= 0.0:
            return 0.0
        n = state_before.n
        if (2 * m_target) % n != 0:
            return 0.0
        deg = state_before.degrees.astype(np.float64)
        var0 = float(np.var(deg))
        if var0 <= 1e-12:
            return 0.0
        u, v = edge
        deg1 = deg.copy()
        deg1[u] += 1.0
        deg1[v] += 1.0
        var1 = float(np.var(deg1))
        rel = (var0 - var1) / var0
        rel = float(np.clip(rel, -1.0, 1.0))
        return float(self.cfg.regularity_bonus_coef * rel)
