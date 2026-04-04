from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol

import copy
import numpy as np
import torch

from .config import BaselineConfig, PromotionConfig


class RolloutPolicy(Protocol):
    name: str

    def select_action(self, obs: Dict[str, np.ndarray]) -> int:
        ...


@dataclass
class PromotionStats:
    episode: int
    base_score: float
    candidate_score: float
    promoted: bool


class ERTop1Policy:
    name = "er_top1"

    def select_action(self, obs: Dict[str, np.ndarray]) -> int:
        # Candidates are generated in descending effective resistance order.
        if obs["candidate_edges"].shape[0] == 0:
            return 0
        return 0


class EROnCandidatesPolicy:
    name = "er_on_candidates"

    def select_action(self, obs: Dict[str, np.ndarray]) -> int:
        cands = obs["candidate_edges"]
        if cands.shape[0] == 0:
            return 0
        ers = obs.get("er_scores")
        if ers is None or ers.shape[0] != cands.shape[0]:
            return 0
        return int(np.argmax(ers))


class FiedlerOnCandidatesPolicy:
    name = "fiedler_on_candidates"

    def select_action(self, obs: Dict[str, np.ndarray]) -> int:
        cands = obs["candidate_edges"]
        if cands.shape[0] == 0:
            return 0

        # Use the Fiedler score (phi2 difference squared) on the current candidate set.
        node_feat = obs.get("node_features")
        if node_feat is not None and node_feat.ndim == 2 and node_feat.shape[1] >= 4:
            u = cands[:, 0].astype(np.int64, copy=False)
            v = cands[:, 1].astype(np.int64, copy=False)
            phi2 = node_feat[:, 3].astype(np.float64, copy=False)
            scores = (phi2[u] - phi2[v]) ** 2
            return int(np.argmax(scores))

        # Fallback: edge feature column 0 is |z_u - z_v| from spectral embedding.
        edge_feat = obs.get("edge_features")
        if edge_feat is not None and edge_feat.ndim == 2 and edge_feat.shape[1] > 0:
            return int(np.argmax(edge_feat[:, 0]))
        return 0


class FrozenModelPolicy:
    name = "frozen_model"

    def __init__(self, model: torch.nn.Module, device: str = "cpu") -> None:
        from .model import obs_to_torch

        self._obs_to_torch = obs_to_torch
        self.model = copy.deepcopy(model).eval()
        self.device = device
        self.model.to(device)

    def select_action(self, obs: Dict[str, np.ndarray]) -> int:
        if obs["candidate_edges"].shape[0] == 0:
            return 0
        with torch.no_grad():
            tout = self.model(self._obs_to_torch(obs, self.device))
            logits = tout["logits"]
            return int(torch.argmax(logits).item())


class LiveModelGreedyPolicy:
    name = "live_model"

    def __init__(self, model: torch.nn.Module, device: str = "cpu") -> None:
        from .model import obs_to_torch

        self._obs_to_torch = obs_to_torch
        self.model = model
        self.device = device

    def select_action(self, obs: Dict[str, np.ndarray]) -> int:
        if obs["candidate_edges"].shape[0] == 0:
            return 0
        with torch.no_grad():
            tout = self.model(self._obs_to_torch(obs, self.device))
            logits = tout["logits"]
            return int(torch.argmax(logits).item())


class BaselineManager:
    def __init__(self, promotion_cfg: PromotionConfig, baseline_cfg: BaselineConfig) -> None:
        self.cfg = promotion_cfg
        init_name = str(baseline_cfg.init_policy).strip().lower()
        if init_name == "er_top1":
            self.policy: RolloutPolicy = ERTop1Policy()
        elif init_name == "er_on_candidates":
            self.policy = EROnCandidatesPolicy()
        elif init_name == "fiedler_on_candidates":
            self.policy = FiedlerOnCandidatesPolicy()
        else:
            raise ValueError(
                f"Unsupported baseline.init_policy='{baseline_cfg.init_policy}'. "
                "Supported: er_top1, er_on_candidates, fiedler_on_candidates."
            )
        self.last_promotion_episode: int = 0

    def current_policy(self) -> RolloutPolicy:
        return self.policy

    def maybe_promote(
        self,
        episode: int,
        candidate_policy: RolloutPolicy,
        evaluator: Callable[[RolloutPolicy], float],
    ) -> Optional[PromotionStats]:
        if not self.cfg.enabled:
            return None
        if episode % self.cfg.eval_interval_episodes != 0:
            return None
        if (episode - self.last_promotion_episode) < self.cfg.cooldown_episodes:
            return None

        base_score = float(evaluator(self.policy))
        cand_score = float(evaluator(candidate_policy))
        promoted = cand_score >= (base_score + self.cfg.margin)
        if promoted:
            self.policy = candidate_policy
            self.last_promotion_episode = episode
        return PromotionStats(
            episode=episode,
            base_score=base_score,
            candidate_score=cand_score,
            promoted=promoted,
        )
