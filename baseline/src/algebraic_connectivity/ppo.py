from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .config import PPOConfig
from .model import obs_to_torch


@dataclass
class Transition:
    obs: Dict[str, np.ndarray]
    action: int
    old_log_prob: float
    old_value: float
    reward: float
    done: bool
    q_target_chosen: float
    q_target_baseline: float
    baseline_action: int


def _compute_gae(transitions: List[Transition], gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray([t.reward for t in transitions], dtype=np.float64)
    values = np.asarray([t.old_value for t in transitions], dtype=np.float64)
    dones = np.asarray([t.done for t in transitions], dtype=np.float64)
    T = len(transitions)
    adv = np.zeros(T, dtype=np.float64)
    last = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
        next_value = values[t]
    ret = adv + values
    return adv, ret


class PPOTrainer:
    def __init__(self, model: nn.Module, cfg: PPOConfig, device: str = "cpu") -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.cur_lr = cfg.lr

    def update(self, transitions: List[Transition]) -> Dict[str, float]:
        if not transitions:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "q_loss": 0.0,
                "entropy": 0.0,
                "gate_entropy": 0.0,
                "gate_balance": 0.0,
                "approx_kl": 0.0,
                "lr": float(self.cur_lr),
            }

        adv, ret = _compute_gae(transitions, self.cfg.gamma, self.cfg.gae_lambda)
        adv = (adv - adv.mean()) / max(adv.std(), 1e-8)

        old_logps = np.asarray([t.old_log_prob for t in transitions], dtype=np.float32)
        old_vals = np.asarray([t.old_value for t in transitions], dtype=np.float32)
        returns = ret.astype(np.float32)
        advs = adv.astype(np.float32)
        idxs = np.arange(len(transitions))

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "q_loss": 0.0,
            "entropy": 0.0,
            "gate_entropy": 0.0,
            "gate_balance": 0.0,
            "approx_kl": 0.0,
        }
        n_batches = 0

        for _ in range(self.cfg.train_epochs):
            np.random.shuffle(idxs)
            epoch_kls: List[float] = []
            for start in range(0, len(idxs), self.cfg.minibatch_size):
                mb = idxs[start : start + self.cfg.minibatch_size]
                if len(mb) == 0:
                    continue

                pol_terms: List[torch.Tensor] = []
                val_terms: List[torch.Tensor] = []
                q_terms: List[torch.Tensor] = []
                ent_terms: List[torch.Tensor] = []
                gate_ent_terms: List[torch.Tensor] = []
                gate_prob_terms: List[torch.Tensor] = []
                kl_terms: List[torch.Tensor] = []

                for i in mb:
                    tr = transitions[int(i)]
                    out = self.model(obs_to_torch(tr.obs, self.device))
                    logits = out["logits"]
                    if logits.numel() == 0:
                        continue
                    action = int(np.clip(tr.action, 0, logits.shape[0] - 1))
                    dist = Categorical(logits=logits)
                    logp = dist.log_prob(torch.tensor(action, device=logits.device))
                    entropy = dist.entropy().mean()
                    old_logp = torch.tensor(old_logps[i], device=logits.device)
                    ratio = torch.exp(logp - old_logp)
                    adv_t = torch.tensor(advs[i], device=logits.device)
                    s1 = ratio * adv_t
                    s2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * adv_t
                    pol_terms.append(-torch.min(s1, s2))

                    v = out["value"]
                    ret_t = torch.tensor(returns[i], device=logits.device)
                    old_v = torch.tensor(old_vals[i], device=logits.device)
                    v_clip = old_v + torch.clamp(v - old_v, -self.cfg.value_clip, self.cfg.value_clip)
                    v1 = (v - ret_t) ** 2
                    v2 = (v_clip - ret_t) ** 2
                    val_terms.append(0.5 * torch.max(v1, v2))

                    qvals = out["q_values"]
                    if qvals.numel() > 0:
                        qsel = qvals[action]
                        tq = torch.tensor(tr.q_target_chosen, device=logits.device, dtype=torch.float32)
                        q_terms.append((qsel - tq) ** 2)
                        ba = int(np.clip(tr.baseline_action, 0, qvals.shape[0] - 1))
                        qb = qvals[ba]
                        tb = torch.tensor(tr.q_target_baseline, device=logits.device, dtype=torch.float32)
                        q_terms.append((qb - tb) ** 2)

                    ent_terms.append(entropy)
                    g_probs = out.get("gate_probs", None)
                    if g_probs is not None and g_probs.numel() > 0:
                        gp = g_probs.float()
                        gate_ent = -(gp * torch.log(torch.clamp(gp, min=1e-8))).sum()
                        gate_ent_terms.append(gate_ent)
                        gate_prob_terms.append(gp)
                    kl_terms.append(old_logp - logp)

                if not pol_terms:
                    continue

                policy_loss = torch.stack(pol_terms).mean()
                value_loss = torch.stack(val_terms).mean() if val_terms else torch.tensor(0.0, device=self.device)
                q_loss = torch.stack(q_terms).mean() if q_terms else torch.tensor(0.0, device=self.device)
                entropy = torch.stack(ent_terms).mean() if ent_terms else torch.tensor(0.0, device=self.device)
                gate_entropy = (
                    torch.stack(gate_ent_terms).mean()
                    if gate_ent_terms
                    else torch.tensor(0.0, device=self.device)
                )
                if gate_prob_terms:
                    gate_mean = torch.stack(gate_prob_terms, dim=0).mean(dim=0)
                    target = torch.full_like(gate_mean, 1.0 / max(gate_mean.numel(), 1))
                    gate_balance = torch.mean((gate_mean - target) ** 2)
                else:
                    gate_balance = torch.tensor(0.0, device=self.device)
                approx_kl = torch.stack(kl_terms).mean() if kl_terms else torch.tensor(0.0, device=self.device)

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    + self.cfg.q_coef * q_loss
                    - self.cfg.entropy_coef * entropy
                    - self.cfg.gate_entropy_coef * gate_entropy
                    + self.cfg.gate_balance_coef * gate_balance
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["q_loss"] += float(q_loss.item())
                metrics["entropy"] += float(entropy.item())
                metrics["gate_entropy"] += float(gate_entropy.item())
                metrics["gate_balance"] += float(gate_balance.item())
                metrics["approx_kl"] += float(approx_kl.item())
                n_batches += 1
                epoch_kls.append(float(approx_kl.item()))

            mean_epoch_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
            if mean_epoch_kl > self.cfg.target_kl:
                break

        if n_batches > 0:
            for k in metrics:
                metrics[k] /= n_batches

        self._adapt_lr(metrics["approx_kl"])
        metrics["lr"] = self.cur_lr
        return metrics

    def _adapt_lr(self, approx_kl: float) -> None:
        if approx_kl > self.cfg.target_kl:
            self.cur_lr *= self.cfg.kl_lr_down_factor
        elif approx_kl < (self.cfg.target_kl * self.cfg.kl_low_threshold):
            self.cur_lr *= self.cfg.kl_lr_up_factor
        self.cur_lr = float(np.clip(self.cur_lr, 1e-6, 1e-2))
        for g in self.optimizer.param_groups:
            g["lr"] = self.cur_lr
