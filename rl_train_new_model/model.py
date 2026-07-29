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
        from .candidacy import PAIR_FEATURE_DIM

        return 5, PAIR_FEATURE_DIM, 6
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

    def forward_batched(
        self,
        observations: list[GraphObservation],
        device: torch.device,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Batched forward pass for multiple observations with potentially
        different graph sizes (n_i) and candidate counts (K_i).

        Returns
        -------
        logits_list : list[Tensor]
            Per-graph logits, each shape (K_i,).
        values : Tensor
            Shape (B,), one scalar value per graph.
        """
        B = len(observations)
        if B == 0:
            return [], torch.empty(0, device=device)

        # --- Convert all observations to tensors ---
        xs: list[torch.Tensor] = []
        edge_indices: list[torch.Tensor] = []
        cand_pairs: list[torch.Tensor] = []
        cand_feats: list[torch.Tensor] = []
        global_feats: list[torch.Tensor] = []
        ns: list[int] = []
        ks: list[int] = []

        for obs in observations:
            x = torch.as_tensor(obs.node_features, dtype=torch.float32, device=device)
            ei = torch.as_tensor(obs.edge_index, dtype=torch.long, device=device)
            cp = torch.as_tensor(obs.candidate_pairs, dtype=torch.long, device=device)
            cf = torch.as_tensor(obs.pair_features, dtype=torch.float32, device=device)
            gf = torch.as_tensor(obs.global_features, dtype=torch.float32, device=device)

            xs.append(x)
            edge_indices.append(ei)
            cand_pairs.append(cp)
            cand_feats.append(cf)
            global_feats.append(gf)
            ns.append(x.shape[0])
            ks.append(cp.shape[0])

        # --- Build the combined disconnected graph ---
        cum_n: list[int] = [0]
        for n in ns:
            cum_n.append(cum_n[-1] + n)
        total_n = cum_n[-1]
        max_K = max(ks) if ks else 0

        x_batch = torch.cat(xs, dim=0)  # (total_n, node_feature_dim)

        # Offset edge indices so each graph's indices point into the combined nodes
        edge_index_parts: list[torch.Tensor] = []
        for i, ei in enumerate(edge_indices):
            edge_index_parts.append(ei + cum_n[i])
        edge_index_batch = torch.cat(edge_index_parts, dim=1)  # (2, total_E)

        # --- Pad candidates to max_K ---
        pair_dim = cand_feats[0].shape[1] if cand_feats else 0
        cand_pairs_batch = torch.full(
            (B, max_K, 2), fill_value=-1, dtype=torch.long, device=device
        )
        cand_feats_batch = torch.zeros(
            (B, max_K, pair_dim), dtype=torch.float32, device=device
        )
        cand_mask = torch.zeros((B, max_K), dtype=torch.bool, device=device)

        for i in range(B):
            k = ks[i]
            if k > 0:
                cand_pairs_batch[i, :k] = cand_pairs[i]
                cand_feats_batch[i, :k] = cand_feats[i]
                cand_mask[i, :k] = True

        global_batch = torch.stack(global_feats, dim=0)  # (B, global_feature_dim)

        # --- GAT forward on the combined graph ---
        h = x_batch
        for layer in self.gat_layers:
            h = layer(h, edge_index_batch)
            h = F.elu(h)

        # --- Gather per-candidate representations (handle -1 sentinel) ---
        ci = cand_pairs_batch               # (B, max_K, 2)
        ci_clamped = ci.clamp(min=0)        # clamp -1 -> 0 for safe indexing

        h_i = h[ci_clamped[:, :, 0]]  # (B, max_K, hidden_dim)
        h_j = h[ci_clamped[:, :, 1]]
        x_i = x_batch[ci_clamped[:, :, 0]]  # (B, max_K, node_feature_dim)
        x_j = x_batch[ci_clamped[:, :, 1]]

        # Zero out gathered values for invalid (padded) candidates
        mask_f = cand_mask.unsqueeze(-1).float()
        h_i = h_i * mask_f
        h_j = h_j * mask_f
        x_i = x_i * mask_f
        x_j = x_j * mask_f

        # --- Build per-candidate representations ---
        rep = torch.cat(
            [
                torch.abs(h_i - h_j),
                h_i * h_j,
                torch.abs(x_i - x_j),
                x_i * x_j,
                cand_feats_batch,
                global_batch.unsqueeze(1).expand(-1, max_K, -1),
            ],
            dim=-1,
        )  # (B, max_K, rep_dim)

        # --- Edge MLP (batched) ---
        rep_flat = rep.view(B * max_K, -1)
        logits_flat = self.edge_mlp(rep_flat).squeeze(-1)  # (B * max_K,)
        logits = logits_flat.view(B, max_K)  # (B, max_K)
        logits[~cand_mask] = -float("inf")   # mask invalid candidates

        # --- Value head (batched) ---
        h_graphs = [h[cum_n[i] : cum_n[i + 1]] for i in range(B)]
        pooled = torch.stack(
            [hg.mean(dim=0) for hg in h_graphs], dim=0
        )  # (B, hidden_dim)
        value_in = torch.cat([pooled, global_batch], dim=-1)
        values = self.value_mlp(value_in).squeeze(-1)  # (B,)

        # --- Return per-graph logits at original lengths ---
        logits_list = [logits[i, : ks[i]] for i in range(B)]
        return logits_list, values
