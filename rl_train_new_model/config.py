from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class CurriculumPhaseConfig:
    name: str
    start_env_step: int
    end_env_step: int
    n_min: int
    n_max: int


@dataclass
class CurriculumConfig:
    phases: List[CurriculumPhaseConfig] = field(default_factory=list)


@dataclass
class EnvConfig:
    num_envs: int = 8
    top_k: int = 128
    dist_cap: int = 4
    terminal_bonus_coef: float = 0.2
    reward_alpha: float = 0.5
    reward_eta: float = 0.0  # 0 = auto (calibrated per n); set >0 to override
    # alpha_2 >> alpha_1, alpha_3: -R_G is monotone submodular (optimality-gap.md S6)
    # and Delta lambda_2 is nondifferentiable at multiplicity (theory.md S3), so the
    # resistance term is weighted most heavily. This is a fixed ratio, not a learned
    # ablation harness (see docs/open-questions.md Q4).
    reward_alpha_1: float = 0.15
    reward_alpha_2: float = 0.70
    reward_alpha_3: float = 0.15
    reward_eta_p: float = 1.0
    # Spectral tracker (docs/algorithm.md S1, S3, S7)
    spectral_oversample: int = 3
    spectral_exact_reset_every: int = 50
    spectral_power_iters: int = 4
    spectral_drift_threshold: float = 1e-8
    spectral_soft_band_rel: float = 1e-2
    # Tiered candidacy pipeline (docs/algorithm.md S4)
    tier1_topk_dr: int = 192
    tier1_topk_spectral: int = 64
    tier2_survivor_size: int = 64
    tier3_top: int = 48
    tier3_random: int = 16


@dataclass
class ModelConfig:
    gat_hidden_dim: int = 64
    gat_heads: int = 4
    gat_layers: int = 2
    edge_mlp_hidden_dim: int = 128
    value_mlp_hidden_dim: int = 128


@dataclass
class TrainerConfig:
    algo: str = "ppo"


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 64
    rollout_env_steps: int = 2048


@dataclass
class ReinforceConfig:
    gamma: float = 0.99
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    rollout_env_steps: int = 2048
    minibatch_size: int = 64
    update_epochs: int = 1
    normalize_advantage: bool = True


@dataclass
class LoggingConfig:
    console_log_every_env_steps: int = 200
    eval_every_env_steps: int = 2000
    metrics_path: str = "experiments/rl/runs/ppo_path_init/metrics.csv"
    tensorboard_dir: str = "experiments/rl/runs/ppo_path_init/tensorboard"
    run_state_path: str = "experiments/rl/runs/ppo_path_init/run_state.json"


@dataclass
class CheckpointConfig:
    every_env_steps: int = 2000
    keep_last_k: int = 5
    dir: str = "experiments/rl/runs/ppo_path_init/checkpoints"


@dataclass
class DeterminismConfig:
    strict: bool = True
    seed: int = 1234
    num_threads: int = 1


@dataclass
class EvalConfig:
    episodes: int = 8


@dataclass
class TrainConfig:
    max_env_steps: int = 1_200_000
    output_dir: str = "experiments/rl/runs/ppo_path_init"


@dataclass
class VariantConfig:
    name: str = "full"


@dataclass
class LiteV2Config:
    actor_topk_ratio: float = 0.12
    actor_topk_min: int = 32
    actor_topk_max: int = 160
    selector_hidden_dim: int = 64
    selector_layers: int = 2
    selector_learning_rate: float = 3e-4
    selector_weight_decay: float = 0.0
    selector_max_grad_norm: float = 1.0
    selector_update_every_env_steps: int = 8
    teacher_weight_lambda2: float = 0.7
    teacher_weight_lambda3: float = 0.3
    reward_shaping_coef: float = 0.02
    reward_weight_lambda2: float = 0.7
    reward_weight_lambda3: float = 0.3
    fast_inference: bool = True
    use_er_teacher: bool = False  # Use ER-weighted teacher (True) or weighted spectral-gap teacher (False)


@dataclass
class InitConfig:
    mode: str = "path"  # path, backbone


@dataclass
class BackboneInitConfig:
    routing_mode: str = "window"  # window, always_both, always_envelope, always_cayley
    rho_very_low: float = 0.05
    rho_low_mix_end: float = 0.10
    rho_low: float = 0.5
    rho_high: float = 0.75
    cayley_samples: int = 500
    cayley_degree_slack: int = 0
    cayley_multi_index_overlap: bool = False
    cayley_enable_index4: bool = True
    cayley_index2_overlap_high: float = 0.6
    cayley_index3_overlap_high: float = 2.0 / 3.0
    cayley_index4_overlap_low: float = 2.0 / 3.0
    cayley_spectral_backend: str = "cpu"  # cpu, cuda
    cayley_eval_batch_size: int = 128
    cayley_eval_mode: str = "dense"  # dense, character
    force_connected_backbone: bool = True


@dataclass
class Config:
    curriculum: CurriculumConfig
    env: EnvConfig
    model: ModelConfig
    trainer: TrainerConfig
    ppo: PPOConfig
    reinforce: ReinforceConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig
    determinism: DeterminismConfig
    evaluation: EvalConfig
    train: TrainConfig
    variant: VariantConfig
    lite_v2: LiteV2Config
    init: InitConfig
    backbone_init: BackboneInitConfig


DEFAULT_CONFIG_DICT: Dict[str, Any] = {
    "curriculum": {
        "phases": [
            {
                "name": "phase_a_12_16",
                "start_env_step": 0,
                "end_env_step": 400000,
                "n_min": 12,
                "n_max": 16,
            },
            {
                "name": "phase_b_16_32",
                "start_env_step": 400000,
                "end_env_step": 800000,
                "n_min": 16,
                "n_max": 32,
            },
            {
                "name": "phase_c_16_64",
                "start_env_step": 800000,
                "end_env_step": 1200000,
                "n_min": 16,
                "n_max": 64,
            },
        ]
    },
    "env": {
        "num_envs": 8,
        "top_k": 128,
        "dist_cap": 4,
        "terminal_bonus_coef": 0.2,
        "reward_alpha_1": 0.15,
        "reward_alpha_2": 0.70,
        "reward_alpha_3": 0.15,
        "spectral_oversample": 3,
        "spectral_exact_reset_every": 50,
        "spectral_power_iters": 4,
        "spectral_drift_threshold": 1e-8,
        "spectral_soft_band_rel": 1e-2,
        "tier1_topk_dr": 192,
        "tier1_topk_spectral": 64,
        "tier2_survivor_size": 64,
        "tier3_top": 48,
        "tier3_random": 16,
    },
    "model": {
        "gat_hidden_dim": 64,
        "gat_heads": 4,
        "gat_layers": 2,
        "edge_mlp_hidden_dim": 128,
        "value_mlp_hidden_dim": 128,
    },
    "trainer": {
        "algo": "ppo",
    },
    "ppo": {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "learning_rate": 3e-4,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 1.0,
        "update_epochs": 4,
        "minibatch_size": 64,
        "rollout_env_steps": 2048,
    },
    "reinforce": {
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 1.0,
        "rollout_env_steps": 2048,
        "minibatch_size": 64,
        "update_epochs": 1,
        "normalize_advantage": True,
    },
    "logging": {
        "console_log_every_env_steps": 200,
        "eval_every_env_steps": 2000,
        "metrics_path": "experiments/rl/runs/ppo_path_init/metrics.csv",
        "tensorboard_dir": "experiments/rl/runs/ppo_path_init/tensorboard",
        "run_state_path": "experiments/rl/runs/ppo_path_init/run_state.json",
    },
    "checkpoint": {
        "every_env_steps": 2000,
        "keep_last_k": 5,
        "dir": "experiments/rl/runs/ppo_path_init/checkpoints",
    },
    "determinism": {
        "strict": True,
        "seed": 1234,
        "num_threads": 1,
    },
    "evaluation": {
        "episodes": 8,
    },
    "train": {
        "max_env_steps": 1200000,
        "output_dir": "experiments/rl/runs/ppo_path_init",
    },
    "variant": {
        "name": "full",
    },
    "lite_v2": {
        "actor_topk_ratio": 0.12,
        "actor_topk_min": 32,
        "actor_topk_max": 160,
        "selector_hidden_dim": 64,
        "selector_layers": 2,
        "selector_learning_rate": 3e-4,
        "selector_weight_decay": 0.0,
        "selector_max_grad_norm": 1.0,
        "selector_update_every_env_steps": 8,
        "teacher_weight_lambda2": 0.7,
        "teacher_weight_lambda3": 0.3,
        "reward_shaping_coef": 0.02,
        "reward_weight_lambda2": 0.7,
        "reward_weight_lambda3": 0.3,
        "fast_inference": True,
    },
    "init": {
        "mode": "path",
    },
    "backbone_init": {
        "routing_mode": "window",
        "rho_very_low": 0.05,
        "rho_low_mix_end": 0.10,
        "rho_low": 0.5,
        "rho_high": 0.75,
        "cayley_samples": 500,
        "cayley_degree_slack": 0,
        "cayley_multi_index_overlap": False,
        "cayley_enable_index4": True,
        "cayley_index2_overlap_high": 0.6,
        "cayley_index3_overlap_high": 2.0 / 3.0,
        "cayley_index4_overlap_low": 2.0 / 3.0,
        "cayley_spectral_backend": "cpu",
        "cayley_eval_batch_size": 128,
        "cayley_eval_mode": "dense",
        "force_connected_backbone": True,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_curriculum(curriculum_dict: Dict[str, Any]) -> CurriculumConfig:
    phases = [CurriculumPhaseConfig(**phase_dict) for phase_dict in curriculum_dict["phases"]]
    phases = sorted(phases, key=lambda p: p.start_env_step)
    return CurriculumConfig(phases=phases)


def _resolve_path(base_path: Path, maybe_relative: str) -> str:
    path = Path(maybe_relative)
    if path.is_absolute():
        return str(path)
    return str((base_path / path).resolve())


def load_config(config_path: str, overrides: Dict[str, Any] | None = None) -> Config:
    cfg_path = Path(config_path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    merged = _deep_merge(DEFAULT_CONFIG_DICT, loaded)
    if overrides:
        merged = _deep_merge(merged, overrides)

    base_dir = cfg_path.parent.parent

    logging_cfg = LoggingConfig(**merged["logging"])
    checkpoint_cfg = CheckpointConfig(**merged["checkpoint"])
    train_cfg = TrainConfig(**merged["train"])
    trainer_cfg = TrainerConfig(**merged["trainer"])
    reinforce_cfg = ReinforceConfig(**merged["reinforce"])
    variant_cfg = VariantConfig(**merged["variant"])
    lite_v2_cfg = LiteV2Config(**merged["lite_v2"])
    init_cfg = InitConfig(**merged["init"])
    backbone_init_cfg = BackboneInitConfig(**merged["backbone_init"])

    if trainer_cfg.algo not in {"ppo", "reinforce"}:
        raise ValueError(f"Unsupported trainer.algo: {trainer_cfg.algo}")
    if not str(variant_cfg.name).strip():
        raise ValueError("variant.name must be a non-empty string.")
    if variant_cfg.name not in {"full", "lite_v2"}:
        raise ValueError(f"Unsupported variant.name: {variant_cfg.name}")
    if lite_v2_cfg.actor_topk_ratio <= 0.0:
        raise ValueError("lite_v2.actor_topk_ratio must be > 0.")
    if lite_v2_cfg.actor_topk_min < 1:
        raise ValueError("lite_v2.actor_topk_min must be >= 1.")
    if lite_v2_cfg.actor_topk_max < lite_v2_cfg.actor_topk_min:
        raise ValueError("lite_v2.actor_topk_max must be >= actor_topk_min.")
    if lite_v2_cfg.selector_layers < 1:
        raise ValueError("lite_v2.selector_layers must be >= 1.")
    if lite_v2_cfg.selector_update_every_env_steps < 1:
        raise ValueError("lite_v2.selector_update_every_env_steps must be >= 1.")
    if init_cfg.mode not in {"path", "backbone"}:
        raise ValueError(f"Unsupported init.mode: {init_cfg.mode}")

    logging_cfg.metrics_path = _resolve_path(base_dir, logging_cfg.metrics_path)
    logging_cfg.tensorboard_dir = _resolve_path(base_dir, logging_cfg.tensorboard_dir)
    logging_cfg.run_state_path = _resolve_path(base_dir, logging_cfg.run_state_path)
    checkpoint_cfg.dir = _resolve_path(base_dir, checkpoint_cfg.dir)
    train_cfg.output_dir = _resolve_path(base_dir, train_cfg.output_dir)

    return Config(
        curriculum=_parse_curriculum(merged["curriculum"]),
        env=EnvConfig(**merged["env"]),
        model=ModelConfig(**merged["model"]),
        trainer=trainer_cfg,
        ppo=PPOConfig(**merged["ppo"]),
        reinforce=reinforce_cfg,
        logging=logging_cfg,
        checkpoint=checkpoint_cfg,
        determinism=DeterminismConfig(**merged["determinism"]),
        evaluation=EvalConfig(**merged["evaluation"]),
        train=train_cfg,
        variant=variant_cfg,
        lite_v2=lite_v2_cfg,
        init=init_cfg,
        backbone_init=backbone_init_cfg,
    )


