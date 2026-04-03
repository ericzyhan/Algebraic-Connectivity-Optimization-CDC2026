from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required. Install dependencies from requirements-rl.txt"
    ) from exc

from .env import GraphObservation


@dataclass
class ForwardOutput:
    logits: torch.Tensor
    value: torch.Tensor


def feature_dims_for_variant(variant_name: str) -> tuple[int, int, int]:
    if variant_name == "full":
        return 5, 5, 5
    if variant_name == "lite_v2":
        return 3, 5, 3
    raise ValueError(f"Unsupported variant: {variant_name}")


class GraphPolicyNetwork(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        pair_feature_dim: int,
        global_feature_dim: int,
        gat_hidden_dim: int,
        gat_heads: int,
        gat_layers: int,
        edge_mlp_hidden_dim: int,
        value_mlp_hidden_dim: int,
    ):
        super().__init__()

        if gat_layers < 1:
            raise ValueError("gat_layers must be >= 1")

        self.gat_layers = nn.ModuleList()
        in_dim = node_feature_dim
        for _ in range(gat_layers):
            layer = GATConv(in_dim, gat_hidden_dim, heads=gat_heads, concat=False)
            self.gat_layers.append(layer)
            in_dim = gat_hidden_dim

        rep_dim = (2 * gat_hidden_dim) + (2 * node_feature_dim) + pair_feature_dim + global_feature_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(rep_dim, edge_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_mlp_hidden_dim, edge_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_mlp_hidden_dim, 1),
        )

        self.value_mlp = nn.Sequential(
            nn.Linear(gat_hidden_dim + global_feature_dim, value_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_mlp_hidden_dim, value_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_mlp_hidden_dim, 1),
        )

    def forward_observation(self, obs: GraphObservation, device: torch.device) -> ForwardOutput:
        x = torch.as_tensor(obs.node_features, dtype=torch.float32, device=device)
        edge_index = torch.as_tensor(obs.edge_index, dtype=torch.long, device=device)
        candidate_pairs = torch.as_tensor(obs.candidate_pairs, dtype=torch.long, device=device)
        pair_features = torch.as_tensor(obs.pair_features, dtype=torch.float32, device=device)
        global_features = torch.as_tensor(obs.global_features, dtype=torch.float32, device=device)

        if candidate_pairs.numel() == 0:
            raise RuntimeError("No candidate actions available for the current graph state")

        h = x
        for layer in self.gat_layers:
            h = layer(h, edge_index)
            h = F.elu(h)

        i_idx = candidate_pairs[:, 0]
        j_idx = candidate_pairs[:, 1]

        h_i = h[i_idx]
        h_j = h[j_idx]
        x_i = x[i_idx]
        x_j = x[j_idx]

        rep = torch.cat(
            [
                torch.abs(h_i - h_j),
                h_i * h_j,
                torch.abs(x_i - x_j),
                x_i * x_j,
                pair_features,
                global_features.unsqueeze(0).expand(candidate_pairs.shape[0], -1),
            ],
            dim=1,
        )

        logits = self.edge_mlp(rep).squeeze(-1)
        pooled_h = torch.mean(h, dim=0)
        value_in = torch.cat([pooled_h, global_features], dim=0)
        value = self.value_mlp(value_in).squeeze(-1)

        return ForwardOutput(logits=logits, value=value)
