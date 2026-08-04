from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import GraphObservation


def adaptive_top_k(
    num_candidates: int,
    ratio: float,
    k_min: int,
    k_max: int,
) -> int:
    if num_candidates <= 0:
        return 0
    k_raw = int(math.ceil(float(ratio) * float(num_candidates)))
    k = max(int(k_min), min(int(k_raw), int(k_max)))
    return min(int(num_candidates), int(k))


def stable_topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or scores.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if k >= scores.size:
        return np.argsort(-scores, kind="mergesort").astype(np.int64, copy=False)
    order = np.argsort(-scores, kind="mergesort")
    return order[:k].astype(np.int64, copy=False)


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return np.zeros((0,), dtype=np.float64)
    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    if not np.isfinite(s_min) or not np.isfinite(s_max) or (s_max - s_min) <= 1e-12:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - s_min) / (s_max - s_min)


def normalized_spectral_scores_for_pairs(
    candidate_pairs: np.ndarray,
    phi2: np.ndarray,
    phi3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    i_idx = candidate_pairs[:, 0]
    j_idx = candidate_pairs[:, 1]
    s2 = (phi2[i_idx] - phi2[j_idx]) ** 2
    s3 = (phi3[i_idx] - phi3[j_idx]) ** 2
    s2n = minmax_normalize(s2.astype(np.float64, copy=False))
    s3n = minmax_normalize(s3.astype(np.float64, copy=False))
    return s2n, s3n


def teacher_scores_for_pairs(
    candidate_pairs: np.ndarray,
    phi2: np.ndarray,
    phi3: np.ndarray,
    w2: float,
    w3: float,
) -> np.ndarray:
    s2n, s3n = normalized_spectral_scores_for_pairs(candidate_pairs, phi2, phi3)
    return (float(w2) * s2n) + (float(w3) * s3n)


def er_teacher_scores_for_pairs(
    candidate_pairs: np.ndarray,
    phi2: np.ndarray,
    phi3: np.ndarray,
    phi4: np.ndarray,
    lambda2: float,
    lambda3: float,
    lambda4: float,
) -> np.ndarray:
    """Score candidates by their effective-resistance-weighted spectral gap.

    score = Σ_{k=2}^4 (Δφ_k)² / λ_k   (ER-weighted, then min-max normalized).

    This weights each eigenvector contribution by 1/λ_k, so low eigenvalues
    (where connectivity is weakest) get more influence — a direct reflection
    of the total effective resistance R_G = n · Σ 1/λ_k.
    """
    i_idx = candidate_pairs[:, 0]
    j_idx = candidate_pairs[:, 1]
    safe_l2 = max(float(abs(lambda2)), 1e-10)
    safe_l3 = max(float(abs(lambda3)), 1e-10)
    safe_l4 = max(float(abs(lambda4)), 1e-10)

    raw = (
        (phi2[i_idx] - phi2[j_idx]) ** 2 / safe_l2
        + (phi3[i_idx] - phi3[j_idx]) ** 2 / safe_l3
        + (phi4[i_idx] - phi4[j_idx]) ** 2 / safe_l4
    )
    return minmax_normalize(raw.astype(np.float64, copy=False))


def slice_observation(obs: GraphObservation, candidate_indices: np.ndarray) -> GraphObservation:
    return GraphObservation(
        node_features=obs.node_features,
        edge_index=obs.edge_index,
        candidate_pairs=obs.candidate_pairs[candidate_indices],
        pair_features=obs.pair_features[candidate_indices],
        global_features=obs.global_features,
        lambda2_norm=obs.lambda2_norm,
        n=obs.n,
        rho_target=obs.rho_target,
        rho_current=obs.rho_current,
    )


@dataclass
class SelectorTrainingSample:
    obs: Dict
    target_mask: np.ndarray
    k: int

    def to_dict(self) -> Dict:
        return {
            "obs": self.obs,
            "target_mask": self.target_mask.astype(np.float32).tolist(),
            "k": int(self.k),
        }

    @staticmethod
    def from_dict(data: Dict) -> "SelectorTrainingSample":
        return SelectorTrainingSample(
            obs=data["obs"],
            target_mask=np.asarray(data["target_mask"], dtype=np.float32),
            k=int(data["k"]),
        )


class SelectorBuffer:
    def __init__(self) -> None:
        self.samples: List[SelectorTrainingSample] = []

    def add(self, sample: SelectorTrainingSample) -> None:
        self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def clear(self) -> None:
        self.samples = []

    def state_dict(self) -> Dict:
        return {"samples": [s.to_dict() for s in self.samples]}

    def load_state_dict(self, state: Dict) -> None:
        self.samples = [SelectorTrainingSample.from_dict(s) for s in state.get("samples", [])]


class PairSelectorNetwork(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        pair_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int,
        layers: int,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("Selector layers must be >= 1.")

        in_dim = (2 * node_feature_dim) + pair_feature_dim + global_feature_dim
        mlp_layers: List[nn.Module] = []
        dim = in_dim
        for _ in range(layers):
            mlp_layers.append(nn.Linear(dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            dim = hidden_dim
        mlp_layers.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward_observation(self, obs: GraphObservation, device: torch.device) -> torch.Tensor:
        x = torch.as_tensor(obs.node_features, dtype=torch.float32, device=device)
        candidate_pairs = torch.as_tensor(obs.candidate_pairs, dtype=torch.long, device=device)
        pair_features = torch.as_tensor(obs.pair_features, dtype=torch.float32, device=device)
        global_features = torch.as_tensor(obs.global_features, dtype=torch.float32, device=device)

        if candidate_pairs.numel() == 0:
            raise RuntimeError("No candidate actions available for selector.")

        i_idx = candidate_pairs[:, 0]
        j_idx = candidate_pairs[:, 1]
        x_i = x[i_idx]
        x_j = x[j_idx]
        rep = torch.cat(
            [
                torch.abs(x_i - x_j),
                x_i * x_j,
                pair_features,
                global_features.unsqueeze(0).expand(candidate_pairs.shape[0], -1),
            ],
            dim=1,
        )
        return self.mlp(rep).squeeze(-1)


def selector_update(
    selector: PairSelectorNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    buffer: SelectorBuffer,
    max_grad_norm: float,
    minibatch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    if len(buffer) == 0:
        learning_rate = float(optimizer.param_groups[0]["lr"])
        return {
            "selector_loss": 0.0,
            "selector_recall_at_k": 0.0,
            "selector_k_mean": 0.0,
            "selector_learning_rate": learning_rate,
        }

    indices = np.arange(len(buffer.samples), dtype=np.int64)
    np.random.shuffle(indices)

    total_loss = 0.0
    total_recall = 0.0
    total_k = 0.0
    total_batches = 0

    for start in range(0, len(indices), max(1, int(minibatch_size))):
        batch_ids = indices[start : start + max(1, int(minibatch_size))]
        if batch_ids.size == 0:
            continue

        losses: List[torch.Tensor] = []
        batch_recall = 0.0
        batch_k = 0.0

        for idx in batch_ids.tolist():
            sample = buffer.samples[int(idx)]
            obs = GraphObservation.from_dict(sample.obs)
            logits = selector.forward_observation(obs, device=device)

            targets = torch.as_tensor(sample.target_mask, dtype=torch.float32, device=device)
            if targets.shape != logits.shape:
                raise RuntimeError("Selector target/logit shape mismatch.")

            n = int(targets.numel())
            k = int(max(1, sample.k))
            pos_weight = float(max(0, n - k)) / float(k)
            pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight_t)
            losses.append(loss)

            pred_idx = stable_topk_indices(logits.detach().cpu().numpy(), k)
            true_idx = np.flatnonzero(sample.target_mask > 0.5).astype(np.int64, copy=False)
            if true_idx.size == 0:
                recall = 1.0
            else:
                recall = float(np.intersect1d(pred_idx, true_idx).size) / float(k)
            batch_recall += recall
            batch_k += float(k)

        mean_loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        mean_loss.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += float(mean_loss.detach().cpu().item())
        total_recall += batch_recall / float(batch_ids.size)
        total_k += batch_k / float(batch_ids.size)
        total_batches += 1

    if scheduler is not None:
        scheduler.step()

    denom = max(1, total_batches)
    learning_rate = float(optimizer.param_groups[0]["lr"])
    return {
        "selector_loss": total_loss / denom,
        "selector_recall_at_k": total_recall / denom,
        "selector_k_mean": total_k / denom,
        "selector_learning_rate": learning_rate,
    }
