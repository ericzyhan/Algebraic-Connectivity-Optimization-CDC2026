from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import GraphObservation
from .model import GraphPolicyNetwork


@dataclass
class ReinforceTransitionRecord:
    obs: Dict
    action_idx: int
    old_logprob: float
    reward: float
    done: bool

    def to_dict(self) -> Dict:
        return {
            "obs": self.obs,
            "action_idx": self.action_idx,
            "old_logprob": self.old_logprob,
            "reward": self.reward,
            "done": self.done,
        }

    @staticmethod
    def from_dict(data: Dict) -> "ReinforceTransitionRecord":
        return ReinforceTransitionRecord(
            obs=data["obs"],
            action_idx=int(data["action_idx"]),
            old_logprob=float(data["old_logprob"]),
            reward=float(data["reward"]),
            done=bool(data["done"]),
        )


@dataclass
class ReinforceSample:
    obs: Dict
    action_idx: int
    old_logprob: float
    return_t: float


class ReinforceRolloutBuffer:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.pending_by_env: List[List[ReinforceTransitionRecord]] = [[] for _ in range(num_envs)]
        self.completed_episodes: List[List[ReinforceTransitionRecord]] = []
        self.completed_transitions: int = 0

    def add(self, env_idx: int, record: ReinforceTransitionRecord) -> None:
        self.pending_by_env[env_idx].append(record)
        if record.done:
            episode = self.pending_by_env[env_idx]
            self.completed_episodes.append(episode)
            self.completed_transitions += len(episode)
            self.pending_by_env[env_idx] = []

    def __len__(self) -> int:
        return self.completed_transitions

    def clear_completed(self) -> None:
        self.completed_episodes = []
        self.completed_transitions = 0

    def state_dict(self) -> Dict:
        return {
            "num_envs": self.num_envs,
            "completed_transitions": self.completed_transitions,
            "pending_by_env": [
                [record.to_dict() for record in env_records]
                for env_records in self.pending_by_env
            ],
            "completed_episodes": [
                [record.to_dict() for record in episode]
                for episode in self.completed_episodes
            ],
        }

    def load_state_dict(self, state: Dict) -> None:
        if int(state["num_envs"]) != self.num_envs:
            raise ValueError("Mismatched num_envs in reinforce rollout buffer state")

        self.completed_transitions = int(state["completed_transitions"])
        self.pending_by_env = [
            [ReinforceTransitionRecord.from_dict(record_dict) for record_dict in env_records]
            for env_records in state["pending_by_env"]
        ]
        self.completed_episodes = [
            [ReinforceTransitionRecord.from_dict(record_dict) for record_dict in episode]
            for episode in state["completed_episodes"]
        ]

    def build_samples(self, gamma: float) -> List[ReinforceSample]:
        samples: List[ReinforceSample] = []
        for episode in self.completed_episodes:
            if not episode:
                continue
            episode_samples_rev: List[ReinforceSample] = []
            g = 0.0
            for record in reversed(episode):
                g = float(record.reward) + float(gamma) * g
                episode_samples_rev.append(
                    ReinforceSample(
                        obs=record.obs,
                        action_idx=record.action_idx,
                        old_logprob=record.old_logprob,
                        return_t=g,
                    )
                )
            samples.extend(reversed(episode_samples_rev))
        return samples


def reinforce_update(
    model: GraphPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    buffer: ReinforceRolloutBuffer,
    gamma: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    update_epochs: int,
    minibatch_size: int,
    normalize_advantage: bool,
    device: torch.device,
) -> Dict[str, float]:
    samples = buffer.build_samples(gamma=gamma)
    if not samples:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

    num_samples = len(samples)
    returns = np.array([s.return_t for s in samples], dtype=np.float32)
    baselines = np.zeros(num_samples, dtype=np.float32)

    with torch.no_grad():
        for idx, sample in enumerate(samples):
            obs = GraphObservation.from_dict(sample.obs)
            output = model.forward_observation(obs, device=device)
            baselines[idx] = float(output.value.item())

    advantages = returns - baselines
    if normalize_advantage:
        advantages = (advantages - float(np.mean(advantages))) / float(np.std(advantages) + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_batches = 0

    for _ in range(update_epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for start in range(0, num_samples, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            if len(batch_idx) == 0:
                continue

            policy_losses = []
            value_losses = []
            entropies = []
            kls = []

            for idx in batch_idx:
                sample = samples[int(idx)]
                obs = GraphObservation.from_dict(sample.obs)
                output = model.forward_observation(obs, device=device)
                dist = Categorical(logits=output.logits)

                action_t = torch.tensor(sample.action_idx, dtype=torch.long, device=device)
                old_logprob_t = torch.tensor(sample.old_logprob, dtype=torch.float32, device=device)
                return_t = torch.tensor(sample.return_t, dtype=torch.float32, device=device)
                advantage_t = torch.tensor(float(advantages[int(idx)]), dtype=torch.float32, device=device)

                new_logprob = dist.log_prob(action_t)
                entropy = dist.entropy()

                policy_loss = -(new_logprob * advantage_t)
                value_loss = F.mse_loss(output.value, return_t)
                kl = old_logprob_t - new_logprob

                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy)
                kls.append(kl)

            mean_policy_loss = torch.stack(policy_losses).mean()
            mean_value_loss = torch.stack(value_losses).mean()
            mean_entropy = torch.stack(entropies).mean()
            mean_kl = torch.stack(kls).mean()

            loss = mean_policy_loss + value_coef * mean_value_loss - entropy_coef * mean_entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += float(mean_policy_loss.detach().cpu().item())
            total_value_loss += float(mean_value_loss.detach().cpu().item())
            total_entropy += float(mean_entropy.detach().cpu().item())
            total_kl += float(mean_kl.detach().cpu().item())
            total_batches += 1

    scheduler.step()

    denom = max(1, total_batches)
    return {
        "policy_loss": total_policy_loss / denom,
        "value_loss": total_value_loss / denom,
        "entropy": total_entropy / denom,
        "approx_kl": total_kl / denom,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }
