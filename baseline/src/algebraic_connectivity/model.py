from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F


@dataclass
class ModelConfig:
    node_dim: int = 5
    edge_dim: int = 11
    global_dim: int = 5
    hidden_dim: int = 96
    gat_layers: int = 2
    dropout: float = 0.0
    negative_slope: float = 0.2
    num_experts: int = 4
    gate_hidden_dim: int = 64


def obs_to_torch(obs: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, val in obs.items():
        if key in ("candidate_edges", "graph_edges"):
            out[key] = torch.as_tensor(val, dtype=torch.long, device=device)
        else:
            out[key] = torch.as_tensor(val, dtype=torch.float32, device=device)
    return out


class DenseGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(out_dim))
        self.attn_dst = nn.Parameter(torch.empty(out_dim))
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.normal_(self.attn_src, std=0.02)
        nn.init.normal_(self.attn_dst, std=0.02)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        h = self.lin(x)

        src = (h * self.attn_src).sum(dim=-1).unsqueeze(1)  # [n,1]
        dst = (h * self.attn_dst).sum(dim=-1).unsqueeze(0)  # [1,n]
        e = F.leaky_relu(src + dst, negative_slope=self.negative_slope)  # [n,n]

        mask = torch.zeros((n, n), dtype=torch.bool, device=x.device)
        if edge_index.numel() > 0:
            u = edge_index[:, 0]
            v = edge_index[:, 1]
            mask[u, v] = True
            mask[v, u] = True
        mask.fill_diagonal_(True)

        e = e.masked_fill(~mask, -1e9)
        alpha = F.softmax(e, dim=1)
        out = alpha @ h
        return F.elu(out)


class GraphEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        layers = []
        in_dim = cfg.node_dim
        for _ in range(cfg.gat_layers):
            layers.append(DenseGATLayer(in_dim, cfg.hidden_dim, cfg.negative_slope))
            in_dim = cfg.hidden_dim
        self.layers = nn.ModuleList(layers)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class PolicyValueNet(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_experts = max(1, int(cfg.num_experts))
        self.encoder = GraphEncoder(cfg)
        self.query = nn.Parameter(torch.empty(cfg.hidden_dim))
        nn.init.normal_(self.query, std=0.02)

        cand_in = (cfg.hidden_dim * 3) + cfg.edge_dim + cfg.global_dim
        self.policy_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cand_in, cfg.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(cfg.hidden_dim, 1),
                )
                for _ in range(self.num_experts)
            ]
        )
        self.q_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cand_in, cfg.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(cfg.hidden_dim, 1),
                )
                for _ in range(self.num_experts)
            ]
        )
        val_in = cfg.hidden_dim + cfg.global_dim
        self.value_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(val_in, cfg.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(cfg.hidden_dim, 1),
                )
                for _ in range(self.num_experts)
            ]
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(val_in, cfg.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.gate_hidden_dim, self.num_experts),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        node_feats = obs["node_features"]  # [n,5]
        g_edges = obs["graph_edges"]  # [E,2]
        gfeat = obs["global_features"]  # [5]
        c_edges = obs["candidate_edges"]  # [k,2]
        e_feat = obs["edge_features"]  # [k,11]

        h = self.encoder(node_feats, g_edges)  # [n,h]
        attn_logits = h @ self.query
        attn = F.softmax(attn_logits, dim=0)
        context = torch.sum(attn.unsqueeze(-1) * h, dim=0)  # [h]

        value_in = torch.cat([context, gfeat], dim=0)  # [h+g]
        gate_logits = self.gate_mlp(value_in)
        gate_log_probs = F.log_softmax(gate_logits, dim=0)
        gate_probs = gate_log_probs.exp()

        value_terms = torch.stack([head(value_in).squeeze(-1) for head in self.value_experts], dim=0)
        value = torch.sum(gate_probs * value_terms)

        if c_edges.numel() == 0:
            return {
                "logits": torch.zeros((0,), device=h.device),
                "q_values": torch.zeros((0,), device=h.device),
                "value": value,
                "context": context,
                "gate_probs": gate_probs,
                "gate_logits": gate_logits,
            }

        u = c_edges[:, 0]
        v = c_edges[:, 1]
        hu = h[u]
        hv = h[v]
        k = c_edges.shape[0]
        cxt = context.unsqueeze(0).expand(k, -1)
        g = gfeat.unsqueeze(0).expand(k, -1)
        cand_in = torch.cat([hu, hv, cxt, e_feat, g], dim=1)

        # Expert-specific candidate logits/Q-values, then soft-gated mixture.
        expert_logits = torch.stack(
            [head(cand_in).squeeze(-1) for head in self.policy_experts],
            dim=0,
        )  # [K,k]
        expert_q = torch.stack(
            [head(cand_in).squeeze(-1) for head in self.q_experts],
            dim=0,
        )  # [K,k]
        expert_log_probs = F.log_softmax(expert_logits, dim=1)
        mixed_log_probs = torch.logsumexp(
            gate_log_probs.unsqueeze(1) + expert_log_probs,
            dim=0,
        )  # [k]
        q_values = torch.sum(gate_probs.unsqueeze(1) * expert_q, dim=0)  # [k]
        return {
            "logits": mixed_log_probs,
            "q_values": q_values,
            "value": value,
            "context": context,
            "gate_probs": gate_probs,
            "gate_logits": gate_logits,
        }

    def act(
        self,
        obs: Dict[str, np.ndarray],
        device: str,
        deterministic: bool = False,
    ) -> Dict[str, float | int]:
        t_obs = obs_to_torch(obs, device)
        out = self.forward(t_obs)
        logits = out["logits"]
        if logits.numel() == 0:
            return {
                "action": 0,
                "log_prob": 0.0,
                "entropy": 0.0,
                "value": float(out["value"].item()),
            }
        dist = Categorical(logits=logits)
        if deterministic:
            action = int(torch.argmax(logits).item())
        else:
            action = int(dist.sample().item())
        log_prob = float(dist.log_prob(torch.tensor(action, device=logits.device)).item())
        entropy = float(dist.entropy().mean().item())
        return {
            "action": action,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": float(out["value"].item()),
        }
