"""Integration test: policy forward pass over analytic candidates + state round-trip.

Requires torch_geometric (as the full-variant policy uses GATConv).  Verifies:

  1. the policy can score/sample from the analytic top-32 candidate set;
  2. stepping with the sampled action keeps the loop healthy;
  3. AnalyticState.state_dict / from_state_dict round-trips exactly.

Run from the project root:

    python -m analytic_tractable_test_deepseek.test_policy_integration
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from analytic_tractable_test_deepseek.config import load_config
from analytic_tractable_test_deepseek.curriculum import CurriculumScheduler
from analytic_tractable_test_deepseek.env import GraphEnv
from analytic_tractable_test_deepseek.model import GraphPolicyNetwork, feature_dims_for_variant
from analytic_tractable_test_deepseek.analytic_state import AnalyticState
from analytic_tractable_test_deepseek.graph_math import build_path_adjacency


def main() -> None:
    cfg = load_config(str(Path(_project_root) / "configs" / "rl_cpu_standard.yaml"))
    scheduler = CurriculumScheduler(cfg.curriculum)

    env = GraphEnv(
        env_id=0,
        scheduler=scheduler,
        top_k=cfg.env.top_k,
        dist_cap=cfg.env.dist_cap,
        terminal_bonus_coef=cfg.env.terminal_bonus_coef,
        seed=7,
        rl_variant="full",
        compute_spectral_each_step=True,
        incremental_observation=True,
        reward_alpha=cfg.env.reward_alpha,
        reward_eta=cfg.env.reward_eta,
        reward_alpha_1=cfg.env.reward_alpha_1,
        reward_alpha_2=cfg.env.reward_alpha_2,
        reward_alpha_3=cfg.env.reward_alpha_3,
        reward_eta_p=cfg.env.reward_eta_p,
    )

    obs = env.reset_with_target(n=16, rho_target=0.6)
    node_dim, pair_dim, global_dim = feature_dims_for_variant("full")
    model = GraphPolicyNetwork(
        node_feature_dim=node_dim,
        pair_feature_dim=pair_dim,
        global_feature_dim=global_dim,
        gat_hidden_dim=cfg.model.gat_hidden_dim,
        gat_heads=cfg.model.gat_heads,
        gat_layers=cfg.model.gat_layers,
        edge_mlp_hidden_dim=cfg.model.edge_mlp_hidden_dim,
        value_mlp_hidden_dim=cfg.model.value_mlp_hidden_dim,
    )
    model.eval()
    device = torch.device("cpu")

    # Policy forward pass over the analytic candidate set.
    output = model.forward_observation(obs, device=device)
    assert output.logits.shape[0] == obs.candidate_pairs.shape[0], "logits/candidates mismatch"
    dist = Categorical(logits=output.logits)
    action_idx = int(dist.sample().item())
    action_pair = tuple(int(x) for x in obs.candidate_pairs[action_idx])
    print(f"policy scored {obs.candidate_pairs.shape[0]} analytic candidates, "
          f"picked {action_pair}")

    # A few healthy steps.
    for _ in range(5):
        obs, reward, done, info = env.step(action_pair)
        assert np.isfinite(float(reward))
        if done:
            break
        output = model.forward_observation(obs, device=device)
        dist = Categorical(logits=output.logits)
        action_idx = int(dist.sample().item())
        action_pair = tuple(int(x) for x in obs.candidate_pairs[action_idx])
    print("policy + analytic candidates loop OK")

    # State-dict round-trip.
    st = AnalyticState(build_path_adjacency(10).astype(np.uint8))
    st.add_edge(1, 8)
    st.add_edge(0, 7)
    sd = st.state_dict()
    st2 = AnalyticState.from_state_dict(sd)
    assert np.allclose(st2.evals, st.evals)
    assert np.allclose(st2.Q, st.Q)
    assert np.allclose(st2.L_plus, st.L_plus)
    assert abs(st2.R_G - st.R_G) < 1e-12
    assert abs(st2.p_min - st.p_min) < 1e-12
    assert st2.r == st.r
    print("AnalyticState state_dict round-trip OK")

    print("\nPOLICY + ANALYTIC INTEGRATION TEST PASSED")


if __name__ == "__main__":
    main()
