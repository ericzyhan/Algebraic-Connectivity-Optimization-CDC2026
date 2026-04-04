from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class CandidateConfig:
    top_k: int = 128
    fraction_cap: float = 1.0
    mode: str = "er"  # er | fiedler


@dataclass
class SpectralConfig:
    refresh_every_steps: int = 1
    exact_reset_every_steps: int = 32
    warm_start_eig: bool = True
    eigsh_cutoff_n: int = 96


@dataclass
class FeatureConfig:
    density_thresholds: Tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    target_degree_mode: str = "floor"  # floor | round | ceil
    expander_comm_penalty_scale: float = 1.0
    fib_geo_weight_scale: float = 1.0
    fib_comm_penalty_scale: float = 1.0
    fib_geo_weight_override: float | None = None
    fib_comm_weight_override: float | None = None
    smallworld_num_samples: int = 1
    rpartite_r_max: int = 16
    rpartite_envelope_rank: int = 0
    rpartite_use_approx_filter: bool = True
    rpartite_require_acm_cert: bool = False
    rpartite_allow_uncertified_fallback: bool = True
    fib_hybrid_fractions: Dict[str, float] = field(
        default_factory=lambda: {"sparse": 0.20, "mid": 0.18, "dense": 0.10}
    )


@dataclass
class RewardConfig:
    use_step_shaping: bool = True
    step_reward_clip: float = 2.0
    shaping_scale: float = 1.0
    terminal_weight: float = 0.10
    regularity_bonus_coef: float = 0.02
    eps_denom: float = 1e-8


@dataclass
class BaselineConfig:
    init_policy: str = "er_top1"  # er_top1 | er_on_candidates | fiedler_on_candidates


@dataclass
class PromotionConfig:
    enabled: bool = True
    eval_interval_episodes: int = 50
    cooldown_episodes: int = 100
    margin: float = 0.02
    validation_instances: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (24, 60),
            (24, 90),
            (32, 90),
            (32, 140),
        ]
    )


@dataclass
class PPOConfig:
    episodes: int = 1000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    q_coef: float = 0.2
    lr: float = 3e-4
    train_epochs: int = 4
    minibatch_size: int = 32
    target_kl: float = 0.02
    max_grad_norm: float = 1.0
    kl_lr_down_factor: float = 0.5
    kl_lr_up_factor: float = 1.03
    kl_low_threshold: float = 0.5
    gate_entropy_coef: float = 0.005
    gate_balance_coef: float = 0.05


@dataclass
class TrainConfig:
    seed: int = 7
    n_min: int = 16
    n_max: int = 32
    graphs_per_update: int = 8
    rollouts_per_graph: int = 2
    init_mode: str = (
        "deterministic_warm_start_expander"
    )  # deterministic_warm_start_expander | deterministic_warm_start_smallworld | deterministic_warm_start_fib_hybrid | deterministic_warm_start_fib_s3_hybrid | deterministic_warm_start_rpartite_envelope | deterministic_warm_start | path | star | tree
    checkpoint_dir: str = "checkpoints"
    log_jsonl_path: str = "train_log.jsonl"
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)


def _update_dataclass(dc: Any, payload: Dict[str, Any]) -> Any:
    for key, val in payload.items():
        if not hasattr(dc, key):
            continue
        cur = getattr(dc, key)
        if hasattr(cur, "__dataclass_fields__") and isinstance(val, dict):
            _update_dataclass(cur, val)
        else:
            setattr(dc, key, val)
    return dc


def load_config(path: str | None) -> TrainConfig:
    cfg = TrainConfig()
    if path is None:
        return cfg
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return _update_dataclass(cfg, data)


def dump_config(cfg: TrainConfig, path: str) -> None:
    Path(path).write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
