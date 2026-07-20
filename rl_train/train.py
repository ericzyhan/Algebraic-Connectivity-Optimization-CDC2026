from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.distributions import Categorical

from .backbone_init import BackboneInitializer
from .checkpoint import load_checkpoint, prune_checkpoints, save_checkpoint, write_run_state
from .config import Config, load_config
from .curriculum import CurriculumScheduler
from .determinism import apply_determinism, get_rng_state, set_rng_state
from .env import GraphObservation, GraphEnv, VectorGraphEnvManager
from .model import GraphPolicyNetwork, feature_dims_for_variant
from .ppo import RolloutBuffer, TransitionRecord, ppo_update
from .reinforce import ReinforceRolloutBuffer, ReinforceTransitionRecord, reinforce_update
from .selector import (
    PairSelectorNetwork,
    SelectorBuffer,
    SelectorTrainingSample,
    adaptive_top_k,
    er_teacher_scores_for_pairs,
    normalized_spectral_scores_for_pairs,
    selector_update,
    slice_observation,
    stable_topk_indices,
)
from .tracker import ProgressTracker
from .utils import safe_mean, state_dict_hash


@dataclass
class TrainerState:
    global_env_steps: int = 0
    update_idx: int = 0
    best_eval_metric: float = -1e18
    last_log_env_step: int = 0
    last_eval_env_step: int = 0
    next_checkpoint_env_step: int = 0
    elapsed_sec_before_resume: float = 0.0
    last_checkpoint_path: str = ""
    latest_policy_loss: float = 0.0
    latest_value_loss: float = 0.0
    latest_entropy: float = 0.0
    latest_approx_kl: float = 0.0
    latest_learning_rate: float = 0.0
    latest_selector_loss: float = 0.0
    latest_selector_recall_at_k: float = 0.0
    latest_selector_k_mean: float = 0.0
    latest_selector_learning_rate: float = 0.0
    selector_update_idx: int = 0
    last_selector_update_env_step: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict) -> "TrainerState":
        return TrainerState(**data)


def _linear_lr_lambda(current_update: int, total_updates: int) -> float:
    if total_updates <= 0:
        return 1.0
    frac = 1.0 - (float(current_update) / float(total_updates))
    return max(0.0, frac)


def _build_model(cfg: Config, device: torch.device) -> GraphPolicyNetwork:
    node_dim, pair_dim, global_dim = feature_dims_for_variant(cfg.variant.name)
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
    return model.to(device)


def _build_selector(cfg: Config, device: torch.device) -> PairSelectorNetwork:
    node_dim, pair_dim, global_dim = feature_dims_for_variant(cfg.variant.name)
    selector = PairSelectorNetwork(
        node_feature_dim=node_dim,
        pair_feature_dim=pair_dim,
        global_feature_dim=global_dim,
        hidden_dim=cfg.lite_v2.selector_hidden_dim,
        layers=cfg.lite_v2.selector_layers,
    )
    return selector.to(device)


def _select_actions(
    model: GraphPolicyNetwork,
    observations: List[GraphObservation],
    device: torch.device,
) -> Tuple[List[int], List[Tuple[int, int]], List[float], List[float]]:
    action_indices: List[int] = []
    action_pairs: List[Tuple[int, int]] = []
    logprobs: List[float] = []
    values: List[float] = []

    with torch.no_grad():
        logits_list, values_tensor = model.forward_batched(observations, device)
        for i, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            action_idx_t = dist.sample()

            action_idx = int(action_idx_t.item())
            pair = tuple(int(x) for x in observations[i].candidate_pairs[action_idx])

            action_indices.append(action_idx)
            action_pairs.append(pair)
            logprobs.append(float(dist.log_prob(action_idx_t).item()))
            values.append(float(values_tensor[i].item()))

    return action_indices, action_pairs, logprobs, values


@dataclass
class SelectionBatch:
    policy_observations: List[GraphObservation]
    action_indices: List[int]
    action_pairs: List[Tuple[int, int]]
    logprobs: List[float]
    values: List[float]
    extra_rewards: List[float]


def _select_actions_with_lite_selector(
    cfg: Config,
    model: GraphPolicyNetwork,
    selector: PairSelectorNetwork,
    selector_buffer: SelectorBuffer,
    observations: List[GraphObservation],
    envs: List[GraphEnv],
    device: torch.device,
) -> SelectionBatch:
    policy_observations: List[GraphObservation] = []
    action_indices: List[int] = []
    action_pairs: List[Tuple[int, int]] = []
    logprobs: List[float] = []
    values: List[float] = []
    extra_rewards: List[float] = []

    w_teacher_l2 = float(cfg.lite_v2.teacher_weight_lambda2)
    w_teacher_l3 = float(cfg.lite_v2.teacher_weight_lambda3)
    reward_coef = float(cfg.lite_v2.reward_shaping_coef)

    with torch.no_grad():
        for env, obs in zip(envs, observations):
            num_candidates = int(obs.candidate_pairs.shape[0])
            if num_candidates <= 0:
                raise RuntimeError("No candidate actions available for the current graph state")

            phi2 = env.current_phi2
            phi3 = env.current_phi3
            phi4 = env.current_phi4
            if phi2 is None or phi3 is None:
                raise RuntimeError("Missing spectral vectors for lite selector teacher.")

            k = adaptive_top_k(
                num_candidates=num_candidates,
                ratio=cfg.lite_v2.actor_topk_ratio,
                k_min=cfg.lite_v2.actor_topk_min,
                k_max=cfg.lite_v2.actor_topk_max,
            )
            if k <= 0:
                raise RuntimeError("Adaptive top-k produced an empty candidate set.")

            selector_logits = selector.forward_observation(obs, device=device)
            selector_scores = selector_logits.detach().cpu().numpy()
            selected_idx = stable_topk_indices(selector_scores, k)

            # Teacher scores — ER-weighted or traditional spectral gap
            use_er = bool(getattr(cfg.lite_v2, 'use_er_teacher', False))
            lambda2 = env.current_lambda2 if hasattr(env, 'current_lambda2') else 0.0
            lambda3 = env.current_lambda3 if hasattr(env, 'current_lambda3') else 0.0
            lambda4 = env.current_lambda4 if hasattr(env, 'current_lambda4') else 0.0

            if use_er:
                teacher_scores = er_teacher_scores_for_pairs(
                    candidate_pairs=obs.candidate_pairs,
                    phi2=phi2,
                    phi3=phi3,
                    phi4=phi4,
                    lambda2=lambda2,
                    lambda3=lambda3,
                    lambda4=lambda4,
                )
            else:
                s2n, s3n = normalized_spectral_scores_for_pairs(
                    candidate_pairs=obs.candidate_pairs,
                    phi2=phi2,
                    phi3=phi3,
                )
                teacher_scores = (float(w_teacher_l2) * s2n) + (float(w_teacher_l3) * s3n)

            teacher_top_idx = stable_topk_indices(teacher_scores, k)
            teacher_mask = np.zeros((num_candidates,), dtype=np.float32)
            teacher_mask[teacher_top_idx] = 1.0

            selector_buffer.add(
                SelectorTrainingSample(
                    obs=obs.to_dict(),
                    target_mask=teacher_mask,
                    k=k,
                )
            )

            policy_obs = slice_observation(obs, selected_idx)
            output = model.forward_observation(policy_obs, device=device)
            dist = Categorical(logits=output.logits)
            action_idx_t = dist.sample()
            action_idx = int(action_idx_t.item())
            action_pair = tuple(int(x) for x in policy_obs.candidate_pairs[action_idx])
            global_candidate_idx = int(selected_idx[action_idx])

            # Shaped reward from teacher score
            shaped_reward = reward_coef * float(teacher_scores[global_candidate_idx])

            policy_observations.append(policy_obs)
            action_indices.append(action_idx)
            action_pairs.append(action_pair)
            logprobs.append(float(dist.log_prob(action_idx_t).item()))
            values.append(float(output.value.item()))
            extra_rewards.append(float(shaped_reward))

    return SelectionBatch(
        policy_observations=policy_observations,
        action_indices=action_indices,
        action_pairs=action_pairs,
        logprobs=logprobs,
        values=values,
        extra_rewards=extra_rewards,
    )


def _compute_bootstrap_values(
    model: GraphPolicyNetwork,
    observations: List[GraphObservation],
    device: torch.device,
) -> List[float]:
    with torch.no_grad():
        _, values_tensor = model.forward_batched(observations, device)
    return [float(v) for v in values_tensor]


def _evaluate_policy(
    cfg: Config,
    model: GraphPolicyNetwork,
    selector: PairSelectorNetwork | None,
    scheduler: CurriculumScheduler,
    device: torch.device,
    global_env_steps: int,
    init_mode: str = "path",
    backbone_initializer: BackboneInitializer | None = None,
) -> float:
    model.eval()
    if selector is not None:
        selector.eval()
    eval_env = GraphEnv(
        env_id=999,
        scheduler=scheduler,
        top_k=cfg.env.top_k,
        dist_cap=cfg.env.dist_cap,
        terminal_bonus_coef=cfg.env.terminal_bonus_coef,
        seed=cfg.determinism.seed + 99991,
        rl_variant=cfg.variant.name,
        compute_spectral_each_step=True,
    )

    terminal_l2: List[float] = []
    episodes = max(1, cfg.evaluation.episodes)

    initial_adj_builder = (
        backbone_initializer.get_initial_adj
        if (init_mode == "backbone" and backbone_initializer is not None)
        else None
    )
    for _ in range(episodes):
        obs = eval_env.reset(
            global_env_steps,
            init_mode=init_mode,
            initial_adj_builder=initial_adj_builder,
        )
        done = False
        while not done:
            with torch.no_grad():
                if cfg.variant.name == "lite_v2":
                    if selector is None:
                        raise RuntimeError("lite_v2 evaluation requires selector model.")
                    num_candidates = int(obs.candidate_pairs.shape[0])
                    k = adaptive_top_k(
                        num_candidates=num_candidates,
                        ratio=cfg.lite_v2.actor_topk_ratio,
                        k_min=cfg.lite_v2.actor_topk_min,
                        k_max=cfg.lite_v2.actor_topk_max,
                    )
                    selector_logits = selector.forward_observation(obs, device=device)
                    selector_scores = selector_logits.detach().cpu().numpy()
                    selected_idx = stable_topk_indices(selector_scores, k)
                    policy_obs = slice_observation(obs, selected_idx)
                    output = model.forward_observation(policy_obs, device=device)
                    action_idx = int(torch.argmax(output.logits).item())
                    action_pair = tuple(int(x) for x in policy_obs.candidate_pairs[action_idx])
                else:
                    output = model.forward_observation(obs, device=device)
                    action_idx = int(torch.argmax(output.logits).item())
                    action_pair = tuple(int(x) for x in obs.candidate_pairs[action_idx])
            obs, _, done, info = eval_env.step(action_pair)
            if done and info.get("terminal_lambda2_norm") is not None:
                terminal_l2.append(float(info["terminal_lambda2_norm"]))

    model.train()
    if selector is not None:
        selector.train()
    return safe_mean(terminal_l2)


def _prepare_dirs(cfg: Config) -> None:
    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint.dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.run_state_path).parent.mkdir(parents=True, exist_ok=True)


def _build_checkpoint_payload(
    cfg: Config,
    model: GraphPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler_obj: torch.optim.lr_scheduler._LRScheduler,
    selector: PairSelectorNetwork | None,
    selector_optimizer: torch.optim.Optimizer | None,
    selector_scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    selector_buffer: SelectorBuffer | None,
    backbone_initializer: BackboneInitializer | None,
    env_manager: VectorGraphEnvManager,
    rollout_buffer: Union[RolloutBuffer, ReinforceRolloutBuffer],
    trainer_state: TrainerState,
    recent_returns: deque,
    recent_terminal_l2: deque,
    recent_init_edges: deque,
    recent_init_lambda2: deque,
    recent_init_build_runtime: deque,
    recent_init_cache_hit: deque,
) -> Dict:
    return {
        "algo": cfg.trainer.algo,
        "variant": cfg.variant.name,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": scheduler_obj.state_dict(),
        "selector_state": selector.state_dict() if selector is not None else None,
        "selector_optimizer_state": (
            selector_optimizer.state_dict() if selector_optimizer is not None else None
        ),
        "selector_lr_scheduler_state": (
            selector_scheduler.state_dict() if selector_scheduler is not None else None
        ),
        "selector_buffer_state": selector_buffer.state_dict() if selector_buffer is not None else None,
        "backbone_initializer_state": (
            backbone_initializer.state_dict() if backbone_initializer is not None else None
        ),
        "env_manager_state": env_manager.state_dict(),
        "rollout_buffer_state": rollout_buffer.state_dict(),
        "trainer_state": trainer_state.to_dict(),
        "recent_returns": list(recent_returns),
        "recent_terminal_l2": list(recent_terminal_l2),
        "recent_init_edges": list(recent_init_edges),
        "recent_init_lambda2": list(recent_init_lambda2),
        "recent_init_build_runtime": list(recent_init_build_runtime),
        "recent_init_cache_hit": list(recent_init_cache_hit),
        "rng_state": get_rng_state(),
        "config_snapshot": {
            "curriculum": [asdict(p) for p in cfg.curriculum.phases],
            "env": asdict(cfg.env),
            "model": asdict(cfg.model),
            "trainer": asdict(cfg.trainer),
            "ppo": asdict(cfg.ppo),
            "reinforce": asdict(cfg.reinforce),
            "logging": asdict(cfg.logging),
            "checkpoint": asdict(cfg.checkpoint),
            "determinism": asdict(cfg.determinism),
            "evaluation": asdict(cfg.evaluation),
            "train": asdict(cfg.train),
            "variant": asdict(cfg.variant),
            "lite_v2": asdict(cfg.lite_v2),
            "init": asdict(cfg.init),
            "backbone_init": asdict(cfg.backbone_init),
        },
    }


def run_training(
    cfg: Config,
    resume_from: Optional[str] = None,
    device: str = "auto",
    quiet: bool = False,
) -> Dict:
    _prepare_dirs(cfg)
    apply_determinism(cfg.determinism)

    if cfg.train.max_env_steps % cfg.env.num_envs != 0:
        raise ValueError(
            "max_env_steps must be divisible by num_envs for exact deterministic stepping"
        )
    if cfg.checkpoint.every_env_steps % cfg.env.num_envs != 0:
        raise ValueError(
            "checkpoint.every_env_steps must be divisible by num_envs"
        )

    # Auto-detect device: use CUDA if available, otherwise CPU
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: device='{device}' requested but CUDA not available; falling back to CPU")
        device = "cpu"

    device_obj = torch.device(device)
    scheduler = CurriculumScheduler(cfg.curriculum)
    algo = cfg.trainer.algo
    rl_variant = cfg.variant.name
    init_mode = cfg.init.mode
    backbone_initializer = BackboneInitializer(cfg.backbone_init) if init_mode == "backbone" else None
    initial_adj_builder = (
        backbone_initializer.get_initial_adj if backbone_initializer is not None else None
    )

    env_manager = VectorGraphEnvManager(
        num_envs=cfg.env.num_envs,
        scheduler=scheduler,
        top_k=cfg.env.top_k,
        dist_cap=cfg.env.dist_cap,
        terminal_bonus_coef=cfg.env.terminal_bonus_coef,
        base_seed=cfg.determinism.seed,
        rl_variant=rl_variant,
        compute_spectral_each_step=True,
        init_mode=init_mode,
        initial_adj_builder=initial_adj_builder,
        reward_alpha=cfg.env.reward_alpha,
        reward_eta=cfg.env.reward_eta,
        reward_alpha_l2=cfg.env.reward_alpha_l2,
        reward_alpha_rg=cfg.env.reward_alpha_rg,
        reward_eta_pmin=cfg.env.reward_eta_pmin,
    )

    model = _build_model(cfg, device_obj)
    if algo == "ppo":
        learning_rate = cfg.ppo.learning_rate
        rollout_env_steps = cfg.ppo.rollout_env_steps
    elif algo == "reinforce":
        learning_rate = cfg.reinforce.learning_rate
        rollout_env_steps = cfg.reinforce.rollout_env_steps
    else:  # pragma: no cover
        raise ValueError(f"Unsupported trainer.algo: {algo}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_updates = max(1, cfg.train.max_env_steps // max(1, rollout_env_steps))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda update_idx: _linear_lr_lambda(update_idx, total_updates),
    )

    selector: PairSelectorNetwork | None = None
    selector_optimizer: torch.optim.Optimizer | None = None
    selector_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
    selector_buffer: SelectorBuffer | None = None
    if rl_variant == "lite_v2":
        selector = _build_selector(cfg, device_obj)
        selector_optimizer = torch.optim.Adam(
            selector.parameters(),
            lr=cfg.lite_v2.selector_learning_rate,
            weight_decay=cfg.lite_v2.selector_weight_decay,
        )
        selector_buffer = SelectorBuffer()

    rollout_buffer: Union[RolloutBuffer, ReinforceRolloutBuffer]
    if algo == "ppo":
        rollout_buffer = RolloutBuffer(num_envs=cfg.env.num_envs)
    else:
        rollout_buffer = ReinforceRolloutBuffer(num_envs=cfg.env.num_envs)
    trainer_state = TrainerState(
        next_checkpoint_env_step=cfg.checkpoint.every_env_steps,
        latest_learning_rate=learning_rate,
    )
    if selector_optimizer is not None:
        trainer_state.latest_selector_learning_rate = float(
            selector_optimizer.param_groups[0]["lr"]
        )

    recent_returns = deque(maxlen=200)
    recent_terminal_l2 = deque(maxlen=200)
    recent_init_edges = deque(maxlen=200)
    recent_init_lambda2 = deque(maxlen=200)
    recent_init_build_runtime = deque(maxlen=200)
    recent_init_cache_hit = deque(maxlen=200)

    if resume_from:
        # Always load checkpoint payload on CPU so RNG byte tensors remain CPU-compatible.
        checkpoint = load_checkpoint(resume_from, map_location="cpu")
        checkpoint_algo = str(checkpoint.get("algo", "ppo"))
        checkpoint_variant = str(checkpoint.get("variant", "full"))
        if checkpoint_algo != algo:
            raise ValueError(
                f"Checkpoint algo mismatch: checkpoint={checkpoint_algo} config={algo}"
            )
        if checkpoint_variant != rl_variant:
            raise ValueError(
                f"Checkpoint variant mismatch: checkpoint={checkpoint_variant} config={rl_variant}"
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])
        if rl_variant == "lite_v2":
            if selector is None or selector_optimizer is None or selector_buffer is None:
                raise RuntimeError("lite_v2 resume requires selector components.")
            selector_state = checkpoint.get("selector_state")
            selector_optimizer_state = checkpoint.get("selector_optimizer_state")
            selector_lr_scheduler_state = checkpoint.get("selector_lr_scheduler_state")
            selector_buffer_state = checkpoint.get("selector_buffer_state")
            if selector_state is None or selector_optimizer_state is None:
                raise ValueError("Missing selector state in lite_v2 checkpoint.")
            selector.load_state_dict(selector_state)
            selector_optimizer.load_state_dict(selector_optimizer_state)
            if selector_scheduler is not None and selector_lr_scheduler_state is not None:
                selector_scheduler.load_state_dict(selector_lr_scheduler_state)
            if selector_buffer_state is not None:
                selector_buffer.load_state_dict(selector_buffer_state)
        if backbone_initializer is not None:
            bb_state = checkpoint.get("backbone_initializer_state")
            if bb_state is not None:
                backbone_initializer.load_state_dict(bb_state)

        env_manager.load_state_dict(checkpoint["env_manager_state"])
        rollout_buffer.load_state_dict(checkpoint["rollout_buffer_state"])
        trainer_state = TrainerState.from_dict(checkpoint["trainer_state"])
        recent_returns.extend(float(x) for x in checkpoint.get("recent_returns", []))
        recent_terminal_l2.extend(float(x) for x in checkpoint.get("recent_terminal_l2", []))
        recent_init_edges.extend(float(x) for x in checkpoint.get("recent_init_edges", []))
        recent_init_lambda2.extend(float(x) for x in checkpoint.get("recent_init_lambda2", []))
        recent_init_build_runtime.extend(
            float(x) for x in checkpoint.get("recent_init_build_runtime", [])
        )
        recent_init_cache_hit.extend(float(x) for x in checkpoint.get("recent_init_cache_hit", []))
        set_rng_state(checkpoint["rng_state"])
        env_manager.ensure_actionable_observations(trainer_state.global_env_steps)

        if not quiet:
            print(f"[train] resumed from checkpoint: {resume_from} (algo={algo})", flush=True)
    else:
        env_manager.reset_all(trainer_state.global_env_steps)

    tracker = ProgressTracker(
        max_env_steps=cfg.train.max_env_steps,
        metrics_csv_path=cfg.logging.metrics_path,
        tensorboard_dir=cfg.logging.tensorboard_dir,
        resumed_elapsed_sec=trainer_state.elapsed_sec_before_resume,
    )

    try:
        while trainer_state.global_env_steps < cfg.train.max_env_steps:
            observations = env_manager.get_observations()
            if rl_variant == "lite_v2":
                if selector is None or selector_buffer is None:
                    raise RuntimeError("lite_v2 requires selector components.")
                selection_batch = _select_actions_with_lite_selector(
                    cfg=cfg,
                    model=model,
                    selector=selector,
                    selector_buffer=selector_buffer,
                    observations=observations,
                    envs=env_manager.envs,
                    device=device_obj,
                )
            else:
                action_indices, action_pairs, logprobs, values = _select_actions(
                    model=model,
                    observations=observations,
                    device=device_obj,
                )
                selection_batch = SelectionBatch(
                    policy_observations=observations,
                    action_indices=action_indices,
                    action_pairs=action_pairs,
                    logprobs=logprobs,
                    values=values,
                    extra_rewards=[0.0 for _ in range(len(observations))],
                )

            post_step = trainer_state.global_env_steps + cfg.env.num_envs
            next_obs, rewards, dones, infos = env_manager.step(
                action_pairs=selection_batch.action_pairs,
                global_env_step_after_batch=post_step,
                extra_rewards=selection_batch.extra_rewards,
            )

            for env_idx in range(cfg.env.num_envs):
                if algo == "ppo":
                    record = TransitionRecord(
                        obs=selection_batch.policy_observations[env_idx].to_dict(),
                        action_idx=selection_batch.action_indices[env_idx],
                        old_logprob=selection_batch.logprobs[env_idx],
                        old_value=selection_batch.values[env_idx],
                        reward=float(rewards[env_idx]),
                        done=bool(dones[env_idx]),
                    )
                    casted_buffer = rollout_buffer
                    if not isinstance(casted_buffer, RolloutBuffer):
                        raise RuntimeError("Expected PPO rollout buffer")
                    casted_buffer.add(env_idx, record)
                else:
                    record = ReinforceTransitionRecord(
                        obs=selection_batch.policy_observations[env_idx].to_dict(),
                        action_idx=selection_batch.action_indices[env_idx],
                        old_logprob=selection_batch.logprobs[env_idx],
                        reward=float(rewards[env_idx]),
                        done=bool(dones[env_idx]),
                    )
                    casted_buffer = rollout_buffer
                    if not isinstance(casted_buffer, ReinforceRolloutBuffer):
                        raise RuntimeError("Expected REINFORCE rollout buffer")
                    casted_buffer.add(env_idx, record)

                info = infos[env_idx]
                if info.get("episode_done"):
                    recent_returns.append(float(info["episode_return"]))
                    recent_terminal_l2.append(float(info["terminal_lambda2_norm"]))
                    recent_init_edges.append(float(info.get("init_edges", 0.0)))
                    recent_init_lambda2.append(float(info.get("init_lambda2", 0.0)))
                    recent_init_build_runtime.append(float(info.get("init_build_runtime_sec", 0.0)))
                    recent_init_cache_hit.append(1.0 if bool(info.get("init_cache_hit", False)) else 0.0)

            trainer_state.global_env_steps = post_step

            if rl_variant == "lite_v2":
                if selector is None or selector_optimizer is None or selector_buffer is None:
                    raise RuntimeError("lite_v2 requires selector components.")
                if (
                    trainer_state.global_env_steps - trainer_state.last_selector_update_env_step
                    >= cfg.lite_v2.selector_update_every_env_steps
                ):
                    selector_metrics = selector_update(
                        selector=selector,
                        optimizer=selector_optimizer,
                        scheduler=selector_scheduler,
                        buffer=selector_buffer,
                        max_grad_norm=cfg.lite_v2.selector_max_grad_norm,
                        minibatch_size=cfg.env.num_envs,
                        device=device_obj,
                    )
                    selector_buffer.clear()
                    trainer_state.selector_update_idx += 1
                    trainer_state.last_selector_update_env_step = trainer_state.global_env_steps
                    trainer_state.latest_selector_loss = selector_metrics["selector_loss"]
                    trainer_state.latest_selector_recall_at_k = selector_metrics[
                        "selector_recall_at_k"
                    ]
                    trainer_state.latest_selector_k_mean = selector_metrics["selector_k_mean"]
                    trainer_state.latest_selector_learning_rate = selector_metrics[
                        "selector_learning_rate"
                    ]

            if algo == "ppo" and len(rollout_buffer) >= cfg.ppo.rollout_env_steps:
                bootstrap_values = _compute_bootstrap_values(
                    model=model,
                    observations=next_obs,
                    device=device_obj,
                )

                ppo_buffer = rollout_buffer
                if not isinstance(ppo_buffer, RolloutBuffer):
                    raise RuntimeError("Expected PPO rollout buffer")
                ppo_metrics = ppo_update(
                    model=model,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    buffer=ppo_buffer,
                    next_values=bootstrap_values,
                    gamma=cfg.ppo.gamma,
                    gae_lambda=cfg.ppo.gae_lambda,
                    clip_ratio=cfg.ppo.clip_ratio,
                    entropy_coef=cfg.ppo.entropy_coef,
                    value_coef=cfg.ppo.value_coef,
                    max_grad_norm=cfg.ppo.max_grad_norm,
                    update_epochs=cfg.ppo.update_epochs,
                    minibatch_size=cfg.ppo.minibatch_size,
                    device=device_obj,
                )
                ppo_buffer.clear()

                trainer_state.update_idx += 1
                trainer_state.latest_policy_loss = ppo_metrics["policy_loss"]
                trainer_state.latest_value_loss = ppo_metrics["value_loss"]
                trainer_state.latest_entropy = ppo_metrics["entropy"]
                trainer_state.latest_approx_kl = ppo_metrics["approx_kl"]
                trainer_state.latest_learning_rate = ppo_metrics["learning_rate"]
            elif algo == "reinforce" and len(rollout_buffer) >= cfg.reinforce.rollout_env_steps:
                reinforce_buffer = rollout_buffer
                if not isinstance(reinforce_buffer, ReinforceRolloutBuffer):
                    raise RuntimeError("Expected REINFORCE rollout buffer")
                reinforce_metrics = reinforce_update(
                    model=model,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    buffer=reinforce_buffer,
                    gamma=cfg.reinforce.gamma,
                    entropy_coef=cfg.reinforce.entropy_coef,
                    value_coef=cfg.reinforce.value_coef,
                    max_grad_norm=cfg.reinforce.max_grad_norm,
                    update_epochs=cfg.reinforce.update_epochs,
                    minibatch_size=cfg.reinforce.minibatch_size,
                    normalize_advantage=cfg.reinforce.normalize_advantage,
                    device=device_obj,
                )
                reinforce_buffer.clear_completed()

                trainer_state.update_idx += 1
                trainer_state.latest_policy_loss = reinforce_metrics["policy_loss"]
                trainer_state.latest_value_loss = reinforce_metrics["value_loss"]
                trainer_state.latest_entropy = reinforce_metrics["entropy"]
                trainer_state.latest_approx_kl = reinforce_metrics["approx_kl"]
                trainer_state.latest_learning_rate = reinforce_metrics["learning_rate"]

            if trainer_state.global_env_steps - trainer_state.last_eval_env_step >= cfg.logging.eval_every_env_steps:
                eval_metric = _evaluate_policy(
                    cfg=cfg,
                    model=model,
                    selector=selector,
                    scheduler=scheduler,
                    device=device_obj,
                    global_env_steps=trainer_state.global_env_steps,
                    init_mode=init_mode,
                    backbone_initializer=backbone_initializer,
                )
                trainer_state.last_eval_env_step = trainer_state.global_env_steps

                if eval_metric > trainer_state.best_eval_metric:
                    trainer_state.best_eval_metric = eval_metric
                    best_payload = _build_checkpoint_payload(
                        cfg=cfg,
                        model=model,
                        optimizer=optimizer,
                        scheduler_obj=lr_scheduler,
                        selector=selector,
                        selector_optimizer=selector_optimizer,
                        selector_scheduler=selector_scheduler,
                        selector_buffer=selector_buffer,
                        backbone_initializer=backbone_initializer,
                        env_manager=env_manager,
                        rollout_buffer=rollout_buffer,
                        trainer_state=trainer_state,
                        recent_returns=recent_returns,
                        recent_terminal_l2=recent_terminal_l2,
                        recent_init_edges=recent_init_edges,
                        recent_init_lambda2=recent_init_lambda2,
                        recent_init_build_runtime=recent_init_build_runtime,
                        recent_init_cache_hit=recent_init_cache_hit,
                    )
                    best_ckpt_path = str(Path(cfg.checkpoint.dir) / "best.pt")
                    save_checkpoint(best_ckpt_path, best_payload)

            if trainer_state.global_env_steps - trainer_state.last_log_env_step >= cfg.logging.console_log_every_env_steps:
                elapsed = tracker.elapsed_sec()
                sps = float(trainer_state.global_env_steps) / max(1e-6, elapsed)
                remaining = max(0, cfg.train.max_env_steps - trainer_state.global_env_steps)
                eta = float(remaining) / max(1e-6, sps)
                phase_name = scheduler.phase_name(trainer_state.global_env_steps)

                row = {
                    "global_env_steps": trainer_state.global_env_steps,
                    "update_idx": trainer_state.update_idx,
                    "phase": phase_name,
                    "rl_variant": rl_variant,
                    "init_mode": init_mode,
                    "steps_per_sec": sps,
                    "elapsed_sec": elapsed,
                    "eta_sec": eta,
                    "mean_episode_return": safe_mean(recent_returns),
                    "mean_terminal_lambda2_norm": safe_mean(recent_terminal_l2),
                    "mean_init_edges": safe_mean(recent_init_edges),
                    "mean_init_lambda2": safe_mean(recent_init_lambda2),
                    "mean_init_build_runtime_sec": safe_mean(recent_init_build_runtime),
                    "mean_init_cache_hit": safe_mean(recent_init_cache_hit),
                    "policy_loss": trainer_state.latest_policy_loss,
                    "value_loss": trainer_state.latest_value_loss,
                    "entropy": trainer_state.latest_entropy,
                    "approx_kl": trainer_state.latest_approx_kl,
                    "learning_rate": trainer_state.latest_learning_rate,
                    "selector_loss": trainer_state.latest_selector_loss,
                    "selector_recall_at_k": trainer_state.latest_selector_recall_at_k,
                    "selector_k_mean": trainer_state.latest_selector_k_mean,
                    "selector_learning_rate": trainer_state.latest_selector_learning_rate,
                    "best_eval_metric": trainer_state.best_eval_metric,
                }

                tracker.log(row)
                if not quiet:
                    tracker.print_console(row)
                trainer_state.last_log_env_step = trainer_state.global_env_steps

                run_state = {
                    "global_env_steps": trainer_state.global_env_steps,
                    "phase": phase_name,
                    "algo": algo,
                    "rl_variant": rl_variant,
                    "init_mode": init_mode,
                    "best_eval_metric": trainer_state.best_eval_metric,
                    "last_checkpoint_path": trainer_state.last_checkpoint_path,
                    "elapsed_sec": elapsed,
                    "update_idx": trainer_state.update_idx,
                    "selector_update_idx": trainer_state.selector_update_idx,
                }
                write_run_state(cfg.logging.run_state_path, run_state)

            while trainer_state.global_env_steps >= trainer_state.next_checkpoint_env_step:
                trainer_state.elapsed_sec_before_resume = tracker.elapsed_sec()
                payload = _build_checkpoint_payload(
                    cfg=cfg,
                    model=model,
                    optimizer=optimizer,
                    scheduler_obj=lr_scheduler,
                    selector=selector,
                    selector_optimizer=selector_optimizer,
                    selector_scheduler=selector_scheduler,
                    selector_buffer=selector_buffer,
                    backbone_initializer=backbone_initializer,
                    env_manager=env_manager,
                    rollout_buffer=rollout_buffer,
                    trainer_state=trainer_state,
                    recent_returns=recent_returns,
                    recent_terminal_l2=recent_terminal_l2,
                    recent_init_edges=recent_init_edges,
                    recent_init_lambda2=recent_init_lambda2,
                    recent_init_build_runtime=recent_init_build_runtime,
                    recent_init_cache_hit=recent_init_cache_hit,
                )
                ckpt_path = str(Path(cfg.checkpoint.dir) / f"step_{trainer_state.next_checkpoint_env_step}.pt")
                save_checkpoint(ckpt_path, payload)
                trainer_state.last_checkpoint_path = ckpt_path

                last_path = str(Path(cfg.checkpoint.dir) / "last.pt")
                save_checkpoint(last_path, payload)
                prune_checkpoints(cfg.checkpoint.dir, cfg.checkpoint.keep_last_k)

                trainer_state.next_checkpoint_env_step += cfg.checkpoint.every_env_steps

        trainer_state.elapsed_sec_before_resume = tracker.elapsed_sec()
        final_payload = _build_checkpoint_payload(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            scheduler_obj=lr_scheduler,
            selector=selector,
            selector_optimizer=selector_optimizer,
            selector_scheduler=selector_scheduler,
            selector_buffer=selector_buffer,
            backbone_initializer=backbone_initializer,
            env_manager=env_manager,
            rollout_buffer=rollout_buffer,
            trainer_state=trainer_state,
            recent_returns=recent_returns,
            recent_terminal_l2=recent_terminal_l2,
            recent_init_edges=recent_init_edges,
            recent_init_lambda2=recent_init_lambda2,
            recent_init_build_runtime=recent_init_build_runtime,
            recent_init_cache_hit=recent_init_cache_hit,
        )
        final_path = str(Path(cfg.checkpoint.dir) / "last.pt")
        save_checkpoint(final_path, final_payload)
        trainer_state.last_checkpoint_path = final_path

        # Always persist final run state, even when no periodic log tick occurred
        # (for example, very short smoke runs).
        elapsed = tracker.elapsed_sec()
        phase_name = scheduler.phase_name(trainer_state.global_env_steps)
        final_run_state = {
            "global_env_steps": trainer_state.global_env_steps,
            "phase": phase_name,
            "algo": algo,
            "rl_variant": rl_variant,
            "init_mode": init_mode,
            "best_eval_metric": trainer_state.best_eval_metric,
            "last_checkpoint_path": trainer_state.last_checkpoint_path,
            "elapsed_sec": elapsed,
            "update_idx": trainer_state.update_idx,
            "selector_update_idx": trainer_state.selector_update_idx,
        }
        write_run_state(cfg.logging.run_state_path, final_run_state)

        model_hash = state_dict_hash(model.state_dict())
        optim_hash = state_dict_hash(optimizer.state_dict())
        selector_hash = state_dict_hash(selector.state_dict()) if selector is not None else ""

        summary = {
            "algo": algo,
            "rl_variant": rl_variant,
            "init_mode": init_mode,
            "final_checkpoint": final_path,
            "global_env_steps": trainer_state.global_env_steps,
            "update_idx": trainer_state.update_idx,
            "model_hash": model_hash,
            "optimizer_hash": optim_hash,
            "selector_hash": selector_hash,
            "best_eval_metric": trainer_state.best_eval_metric,
        }

        if not quiet:
            print(f"[train] finished: {summary}", flush=True)
        return summary
    finally:
        tracker.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL graph-construction policy")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--max-env-steps",
        type=int,
        default=None,
        help="Override max environment steps",
    )
    parser.add_argument(
        "--checkpoint-every-env-steps",
        type=int,
        default=None,
        help="Override checkpoint cadence in environment steps",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint path to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: cpu, cuda, or auto (auto-detect)",
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        choices=["path", "backbone"],
        default=None,
        help="Override initialization mode (path or backbone).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    overrides: Dict = {}
    if args.max_env_steps is not None:
        overrides.setdefault("train", {})["max_env_steps"] = int(args.max_env_steps)
    if args.checkpoint_every_env_steps is not None:
        overrides.setdefault("checkpoint", {})["every_env_steps"] = int(args.checkpoint_every_env_steps)
    if args.init_mode is not None:
        overrides.setdefault("init", {})["mode"] = str(args.init_mode)

    cfg = load_config(args.config, overrides=overrides if overrides else None)

    run_training(
        cfg=cfg,
        resume_from=args.resume_from,
        device=args.device,
        quiet=False,
    )


if __name__ == "__main__":
    main()
