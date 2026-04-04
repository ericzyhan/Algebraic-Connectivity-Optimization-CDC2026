from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .baseline_manager import BaselineManager, FrozenModelPolicy, RolloutPolicy
from .candidates import CandidateGenerator
from .config import TrainConfig
from .env import EnvConfig, GraphAugmentationEnv
from .graph_core import max_edges
from .incremental_linear_algebra import IncrementalStateConfig
from .model import ModelConfig, PolicyValueNet
from .ppo import PPOTrainer, Transition
from .reward import RewardComputer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_nm(rng: np.random.Generator, n_min: int, n_max: int) -> Tuple[int, int]:
    n = int(rng.integers(n_min, n_max + 1))
    m_lo = n - 1
    m_hi = max_edges(n)
    rho = float(rng.random())
    m = int(round(m_lo + rho * (m_hi - m_lo)))
    m = int(np.clip(m, m_lo, m_hi))
    return n, m


def run_policy_episode(
    n: int,
    m: int,
    policy: RolloutPolicy,
    cfg: TrainConfig,
) -> float:
    cand = CandidateGenerator(cfg.candidate.top_k, cfg.candidate.fraction_cap, cfg.candidate.mode)
    s_cfg = IncrementalStateConfig(
        refresh_every_steps=cfg.spectral.refresh_every_steps,
        exact_reset_every_steps=cfg.spectral.exact_reset_every_steps,
        eigsh_cutoff_n=cfg.spectral.eigsh_cutoff_n,
    )
    e_cfg = EnvConfig(
        init_mode=cfg.init_mode,
        density_thresholds=cfg.features.density_thresholds,
        target_degree_mode=cfg.features.target_degree_mode,
        expander_comm_penalty_scale=cfg.features.expander_comm_penalty_scale,
        fib_geo_weight_scale=cfg.features.fib_geo_weight_scale,
        fib_comm_penalty_scale=cfg.features.fib_comm_penalty_scale,
        fib_geo_weight_override=cfg.features.fib_geo_weight_override,
        fib_comm_weight_override=cfg.features.fib_comm_weight_override,
        smallworld_num_samples=cfg.features.smallworld_num_samples,
        rpartite_r_max=cfg.features.rpartite_r_max,
        rpartite_envelope_rank=cfg.features.rpartite_envelope_rank,
        rpartite_use_approx_filter=cfg.features.rpartite_use_approx_filter,
        rpartite_require_acm_cert=cfg.features.rpartite_require_acm_cert,
        rpartite_allow_uncertified_fallback=cfg.features.rpartite_allow_uncertified_fallback,
        fib_hybrid_fractions=cfg.features.fib_hybrid_fractions,
    )
    env = GraphAugmentationEnv(n, m, cand, state_config=s_cfg, env_config=e_cfg)
    obs = env.reset()
    done = env.done
    while not done:
        action = int(policy.select_action(obs))
        action = int(np.clip(action, 0, max(obs["candidate_edges"].shape[0] - 1, 0)))
        nxt, done, _ = env.step(action, obs)
        if done:
            break
        obs = nxt  # type: ignore[assignment]
    lam2, _, _ = env.state.spectral_chart(force=True)  # type: ignore[union-attr]
    return float(lam2)


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _infer_resume_state(log_path: Path) -> Tuple[int, float]:
    """
    Returns:
      - start_episode (next episode index)
      - historical_best_ma (best logged reward_mean_50)
    """
    rows = _load_jsonl_rows(log_path)
    if not rows:
        return 1, -1e18
    last_ep = 0
    best_ma = -1e18
    for row in rows:
        ep = int(row.get("episode", 0))
        last_ep = max(last_ep, ep)
        ma = row.get("reward_mean_50", None)
        if isinstance(ma, (int, float)):
            best_ma = max(best_ma, float(ma))
    return last_ep + 1, best_ma


def _load_init_checkpoint(
    model: PolicyValueNet,
    ppo: PPOTrainer,
    checkpoint_path: str,
    device: str,
) -> Dict[str, Any]:
    def _load_matching_state_dict(state_dict: Dict[str, Any]) -> Dict[str, int]:
        model_state = model.state_dict()
        matched: Dict[str, Any] = {}
        skipped = 0
        for k, v in state_dict.items():
            if k in model_state and hasattr(v, "shape") and model_state[k].shape == v.shape:
                matched[k] = v
            else:
                skipped += 1
        missing, unexpected = model.load_state_dict(matched, strict=False)
        return {
            "loaded": int(len(matched)),
            "skipped": int(skipped),
            "missing_after_load": int(len(missing)),
            "unexpected_after_load": int(len(unexpected)),
        }

    ckpt = torch.load(checkpoint_path, map_location=device)
    meta: Dict[str, Any] = {"loaded": True, "path": checkpoint_path}
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        meta["model_load_stats"] = _load_matching_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                ppo.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass
        if "cur_lr" in ckpt:
            try:
                ppo.cur_lr = float(ckpt["cur_lr"])
                for g in ppo.optimizer.param_groups:
                    g["lr"] = ppo.cur_lr
            except Exception:
                pass
        for k in ("best_ma", "episode"):
            if k in ckpt:
                meta[k] = ckpt[k]
        return meta

    if isinstance(ckpt, dict):
        # Assume raw model state_dict.
        meta["model_load_stats"] = _load_matching_state_dict(ckpt)
        return meta

    raise ValueError(
        f"Unsupported checkpoint format at {checkpoint_path}. "
        "Expected model state_dict or dict with model_state_dict."
    )


def _save_training_state(
    path: Path,
    model: PolicyValueNet,
    ppo: PPOTrainer,
    episode: int,
    best_ma: float,
) -> None:
    payload = {
        "episode": int(episode),
        "best_ma": float(best_ma),
        "cur_lr": float(ppo.cur_lr),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": ppo.optimizer.state_dict(),
    }
    torch.save(payload, path)


def _collect_rollout(
    model: PolicyValueNet,
    env: GraphAugmentationEnv,
    reward_comp: RewardComputer,
    baseline_policy: RolloutPolicy,
    device: str,
) -> Tuple[List[Transition], Dict[str, Any]]:
    obs = env.reset()
    initial_state = env.state.clone()  # type: ignore[union-attr]
    init_edges = int(initial_state.e)
    residual_budget = int(env.m_target - init_edges)
    done = env.done

    transitions: List[Transition] = []
    ep_reward = 0.0
    shaping_sum = 0.0
    ratio_sum = 0.0
    reg_sum = 0.0
    step_count = 0

    while not done:
        act = model.act(obs, device=device, deterministic=False)
        action = int(act["action"])
        state_before = env.state.clone()  # type: ignore[union-attr]
        step_r = reward_comp.compute_step_reward(
            env=env,
            state_before=state_before,
            obs=obs,
            action=action,
            baseline_policy=baseline_policy,
        )
        nxt, done, _ = env.step(action, obs)
        tr = Transition(
            obs={k: v.copy() for k, v in obs.items()},
            action=action,
            old_log_prob=float(act["log_prob"]),
            old_value=float(act["value"]),
            reward=float(step_r.reward),
            done=bool(done),
            q_target_chosen=float(step_r.q_target_chosen),
            q_target_baseline=float(step_r.q_target_baseline),
            baseline_action=int(step_r.baseline_action),
        )
        transitions.append(tr)
        ep_reward += step_r.reward
        shaping_sum += step_r.reward
        ratio_sum += step_r.ratio
        reg_sum += step_r.regularity_bonus
        step_count += 1
        if not done:
            obs = nxt  # type: ignore[assignment]

    final_lam2, _, _ = env.state.spectral_chart(force=True)  # type: ignore[union-attr]
    terminal_bonus = reward_comp.compute_terminal_reward(
        env=env,
        initial_state=initial_state,
        final_lambda2=float(final_lam2),
        baseline_policy=baseline_policy,
    )
    if transitions:
        transitions[-1].reward += terminal_bonus
    ep_reward += terminal_bonus

    summary: Dict[str, Any] = {
        "n": env.n,
        "m": env.m_target,
        "init_edges": init_edges,
        "residual_budget": residual_budget,
        "steps": step_count,
        "residual_mismatch": int(step_count - residual_budget),
        "episode_reward": float(ep_reward),
        "terminal_bonus": float(terminal_bonus),
        "final_lambda2": float(final_lam2),
        "shaping_sum": float(shaping_sum),
        "ratio_sum": float(ratio_sum),
        "regularity_bonus_sum": float(reg_sum),
    }
    if env.meta is not None:
        summary["warm_start"] = {
            "init_mode": env.meta.init_mode,
            "pocket": env.meta.pocket,
            "rho": env.meta.rho,
            "backbone_degree": env.meta.backbone_degree,
            "meta_residual_edges": env.meta.residual_edges,
        }
    return transitions, summary


def train(
    cfg: TrainConfig,
    device: str = "cpu",
    init_checkpoint: Optional[str] = None,
    append_log: bool = False,
) -> PolicyValueNet:
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg.log_jsonl_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if append_log:
        Path(checkpoint_dir / "config_snapshot_resume_latest.json").write_text(
            json.dumps(asdict(cfg), indent=2), encoding="utf-8"
        )
    else:
        Path(checkpoint_dir / "config_snapshot.json").write_text(
            json.dumps(asdict(cfg), indent=2), encoding="utf-8"
        )

    model = PolicyValueNet(ModelConfig()).to(device)
    ppo = PPOTrainer(model, cfg.ppo, device=device)
    if init_checkpoint is not None:
        _load_init_checkpoint(model, ppo, init_checkpoint, device)

    reward_comp = RewardComputer(cfg.reward)
    baseline_mgr = BaselineManager(cfg.promotion, cfg.baseline)

    candidate_gen = CandidateGenerator(cfg.candidate.top_k, cfg.candidate.fraction_cap, cfg.candidate.mode)
    state_cfg = IncrementalStateConfig(
        refresh_every_steps=cfg.spectral.refresh_every_steps,
        exact_reset_every_steps=cfg.spectral.exact_reset_every_steps,
        eigsh_cutoff_n=cfg.spectral.eigsh_cutoff_n,
    )
    env_cfg = EnvConfig(
        init_mode=cfg.init_mode,
        density_thresholds=cfg.features.density_thresholds,
        target_degree_mode=cfg.features.target_degree_mode,
        expander_comm_penalty_scale=cfg.features.expander_comm_penalty_scale,
        fib_geo_weight_scale=cfg.features.fib_geo_weight_scale,
        fib_comm_penalty_scale=cfg.features.fib_comm_penalty_scale,
        fib_geo_weight_override=cfg.features.fib_geo_weight_override,
        fib_comm_weight_override=cfg.features.fib_comm_weight_override,
        smallworld_num_samples=cfg.features.smallworld_num_samples,
        rpartite_r_max=cfg.features.rpartite_r_max,
        rpartite_envelope_rank=cfg.features.rpartite_envelope_rank,
        rpartite_use_approx_filter=cfg.features.rpartite_use_approx_filter,
        rpartite_require_acm_cert=cfg.features.rpartite_require_acm_cert,
        rpartite_allow_uncertified_fallback=cfg.features.rpartite_allow_uncertified_fallback,
        fib_hybrid_fractions=cfg.features.fib_hybrid_fractions,
    )

    ep_rewards: List[float] = []
    start_episode = 1
    best_ma = -1e18
    if append_log:
        start_episode, best_ma = _infer_resume_state(log_path)
    best_path = checkpoint_dir / "best_model.pt"
    train_state_path = checkpoint_dir / "latest_train_state.pt"
    collapse_count = 0

    def evaluator(pol: RolloutPolicy) -> float:
        vals = [run_policy_episode(n, m, pol, cfg) for (n, m) in cfg.promotion.validation_instances]
        return float(np.mean(vals)) if vals else 0.0

    log_mode = "a" if append_log else "w"
    with log_path.open(log_mode, encoding="utf-8") as f_log:
        b = max(1, int(cfg.graphs_per_update))
        r = max(1, int(cfg.rollouts_per_graph))
        for ep in range(start_episode, start_episode + cfg.ppo.episodes):
            baseline_policy = baseline_mgr.current_policy()
            transitions: List[Transition] = []
            rollout_summaries: List[Dict[str, Any]] = []

            for _ in range(b):
                n, m = sample_nm(rng, cfg.n_min, cfg.n_max)
                for _ in range(r):
                    env = GraphAugmentationEnv(
                        n=n,
                        m_target=m,
                        candidate_generator=candidate_gen,
                        state_config=state_cfg,
                        env_config=env_cfg,
                    )
                    traj, summary = _collect_rollout(
                        model=model,
                        env=env,
                        reward_comp=reward_comp,
                        baseline_policy=baseline_policy,
                        device=device,
                    )
                    transitions.extend(traj)
                    rollout_summaries.append(summary)

            update_stats = ppo.update(transitions)

            if rollout_summaries:
                rewards = [float(x["episode_reward"]) for x in rollout_summaries]
                lambdas = [float(x["final_lambda2"]) for x in rollout_summaries]
                steps = [int(x["steps"]) for x in rollout_summaries]
                residuals = [int(x["residual_budget"]) for x in rollout_summaries]
                mismatches = [abs(int(x["residual_mismatch"])) for x in rollout_summaries]
                term_bonuses = [float(x["terminal_bonus"]) for x in rollout_summaries]
                shaping_vals = [float(x["shaping_sum"]) for x in rollout_summaries]
                ratio_vals = [float(x["ratio_sum"]) for x in rollout_summaries]
                reg_vals = [float(x["regularity_bonus_sum"]) for x in rollout_summaries]
                batch_reward_sum = float(np.sum(rewards))
                batch_reward_mean = float(np.mean(rewards))
                batch_lambda_mean = float(np.mean(lambdas))
                batch_lambda_min = float(np.min(lambdas))
                batch_lambda_max = float(np.max(lambdas))
                batch_steps_sum = int(np.sum(steps))
                batch_steps_mean = float(np.mean(steps))
                batch_residual_mean = float(np.mean(residuals))
                batch_mismatch_max = int(np.max(mismatches))
                terminal_bonus_sum = float(np.sum(term_bonuses))
                shaping_sum = float(np.sum(shaping_vals))
                ratio_sum = float(np.sum(ratio_vals))
                reg_sum = float(np.sum(reg_vals))
            else:
                batch_reward_sum = 0.0
                batch_reward_mean = 0.0
                batch_lambda_mean = 0.0
                batch_lambda_min = 0.0
                batch_lambda_max = 0.0
                batch_steps_sum = 0
                batch_steps_mean = 0.0
                batch_residual_mean = 0.0
                batch_mismatch_max = 0
                terminal_bonus_sum = 0.0
                shaping_sum = 0.0
                ratio_sum = 0.0
                reg_sum = 0.0

            ep_rewards.append(batch_reward_mean)

            ma = float(np.mean(ep_rewards[-50:]))
            if ma > best_ma:
                best_ma = ma
                torch.save(model.state_dict(), best_path)
                collapse_count = 0
            elif ma < (best_ma - 0.2):
                collapse_count += 1
            else:
                collapse_count = max(0, collapse_count - 1)

            # Collapse guard: restore best known model and damp LR / increase entropy.
            if collapse_count >= 5 and best_path.exists():
                model.load_state_dict(torch.load(best_path, map_location=device))
                ppo.cur_lr = max(ppo.cur_lr * 0.7, 1e-6)
                for g in ppo.optimizer.param_groups:
                    g["lr"] = ppo.cur_lr
                cfg.ppo.entropy_coef = min(cfg.ppo.entropy_coef * 1.1, 0.2)
                collapse_count = 0

            promotion = baseline_mgr.maybe_promote(
                episode=ep,
                candidate_policy=FrozenModelPolicy(model, device=device),
                evaluator=evaluator,
            )

            line = {
                "episode": ep,
                "graphs_per_update": b,
                "rollouts_per_graph": r,
                "rollouts_in_update": len(rollout_summaries),
                # Compatibility key: use mean reward across rollouts in this update.
                "episode_reward": batch_reward_mean,
                "episode_reward_sum": batch_reward_sum,
                "steps": batch_steps_sum,
                "steps_mean": batch_steps_mean,
                "residual_budget_mean": batch_residual_mean,
                "residual_mismatch_max": batch_mismatch_max,
                "terminal_bonus_sum": terminal_bonus_sum,
                "final_lambda2_mean": batch_lambda_mean,
                "final_lambda2_min": batch_lambda_min,
                "final_lambda2_max": batch_lambda_max,
                "baseline_policy": baseline_mgr.current_policy().name,
                "reward_mean_50": ma,
                "shaping_sum": shaping_sum,
                "ratio_sum": ratio_sum,
                "regularity_bonus_sum": reg_sum,
                "update": update_stats,
                "rollouts": rollout_summaries,
            }
            if promotion is not None:
                line["promotion"] = {
                    "promoted": promotion.promoted,
                    "base_score": promotion.base_score,
                    "candidate_score": promotion.candidate_score,
                }
            f_log.write(json.dumps(line) + "\n")
            f_log.flush()

            _save_training_state(
                path=train_state_path,
                model=model,
                ppo=ppo,
                episode=ep,
                best_ma=best_ma,
            )
            if ep % 100 == 0:
                torch.save(model.state_dict(), checkpoint_dir / f"model_ep{ep}.pt")

    return model
