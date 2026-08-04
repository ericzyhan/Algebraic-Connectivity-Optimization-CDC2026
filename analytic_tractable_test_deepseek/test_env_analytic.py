"""End-to-end smoke test for the analytic selection wired into the RL env.

Runs a full episode in ``GraphEnv`` (full variant) where the "policy" greedily
picks the analytic algorithm's top-ranked candidate at every step, and checks:

  1. the observation candidate set has at most 32 pairs (all non-edges);
  2. every step reward is finite;
  3. the episode terminates exactly at m_target;
  4. the analytic state tracks the exact λ₂, R_G and p_min of the graph
     (the state's internal ``evals``/``R_G``/``p_min`` vs a fresh dense
     eigendecomposition) within tolerance.

Run from the project root:

    python -m analytic_tractable_test_deepseek.test_env_analytic
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from analytic_tractable_test_deepseek.config import load_config
from analytic_tractable_test_deepseek.curriculum import CurriculumScheduler
from analytic_tractable_test_deepseek.env import GraphEnv
from analytic_tractable_test_deepseek.graph_math import edge_count, laplacian, m_target_from_rho
from analytic_tractable_test_deepseek.analytic_state import compute_p_min


def main() -> None:
    cfg = load_config(str(Path(_project_root) / "configs" / "rl_cpu_standard.yaml"))
    scheduler = CurriculumScheduler(cfg.curriculum)

    env = GraphEnv(
        env_id=0,
        scheduler=scheduler,
        top_k=cfg.env.top_k,
        dist_cap=cfg.env.dist_cap,
        terminal_bonus_coef=cfg.env.terminal_bonus_coef,
        seed=1234,
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

    for (n, rho) in [(16, 0.5), (32, 0.4), (12, 0.8)]:
        obs = env.reset_with_target(n=n, rho_target=rho)
        m_target = m_target_from_rho(n, rho)
        m_init = edge_count(env.adj)
        steps = 0
        max_pairs = 0
        total_reward = 0.0

        while True:
            k = int(obs.candidate_pairs.shape[0])
            max_pairs = max(max_pairs, k)
            assert k <= 32, f"more than 32 candidates: {k}"
            for (i, j) in obs.candidate_pairs:
                assert env.adj[i, j] == 0, f"candidate ({i},{j}) is already an edge"
            if k == 0:
                break
            action = tuple(int(x) for x in obs.candidate_pairs[0])  # greedy top-1
            obs, reward, done, info = env.step(action)
            assert np.isfinite(float(reward)), f"non-finite reward: {reward}"
            total_reward += float(reward)
            steps += 1
            if done:
                break

        m_final = edge_count(env.adj)
        assert m_final == m_target, f"terminated at m={m_final} != m_target={m_target}"
        assert steps == m_target - m_init, f"episode length {steps} != {m_target - m_init}"

        # ---- Validate the analytic state against exact dense eigendecomposition ----
        evals2, evecs2 = np.linalg.eigh(laplacian(env.adj))
        n = int(n)
        exact_l2 = float(evals2[1])
        exact_rg = float(n) * float(np.sum(1.0 / np.maximum(evals2[1:], 1e-14)))
        inv = np.zeros(n, dtype=np.float64)
        inv[1:] = 1.0 / np.maximum(evals2[1:], 1e-14)
        L_plus_exact = evecs2[:, 1:] @ np.diag(inv[1:]) @ evecs2[:, 1:].T
        exact_pmin = compute_p_min(env.adj, L_plus_exact)

        state_l2 = float(env.current_lambda2)
        state_rg = float(env.current_rg)
        state_pmin = float(env.current_P_min)

        err_l2 = abs(state_l2 - exact_l2)
        rel_rg = abs(state_rg - exact_rg) / max(1.0, abs(exact_rg))
        err_pm = abs(state_pmin - exact_pmin)

        print(
            f"n={n} rho={rho}: steps={steps} m={m_final} max_pairs={max_pairs} "
            f"reward={total_reward:+.4f}"
        )
        print(f"  λ₂  state={state_l2:.6f} exact={exact_l2:.6f} err={err_l2:.2e}")
        print(f"  R_G state={state_rg:.6f} exact={exact_rg:.6f} rel err={rel_rg:.2e}")
        print(f"  p_min state={state_pmin:.6f} exact={exact_pmin:.6f} err={err_pm:.2e}")

        assert err_l2 < 2e-6, f"λ₂ drift too large: {err_l2:.2e}"
        assert rel_rg < 1e-8, f"R_G drift too large: {rel_rg:.2e}"
        assert err_pm < 1e-6, f"p_min drift too large: {err_pm:.2e}"

    print("\nENV + ANALYTIC SELECTION SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
