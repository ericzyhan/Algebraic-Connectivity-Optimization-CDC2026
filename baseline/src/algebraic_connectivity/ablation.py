from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Dict, List, Sequence, Tuple

from .config import TrainConfig
from .eval import evaluate_instances
from .train import train


def run_basic_ablations(
    base_cfg: TrainConfig,
    instances: Sequence[Tuple[int, int]],
    device: str = "cpu",
) -> List[Dict[str, object]]:
    """
    Minimal ablation set:
    1) full method
    2) no step shaping
    3) no regularity bonus
    4) path initialization
    """
    variants: List[tuple[str, TrainConfig]] = []

    cfg_full = copy.deepcopy(base_cfg)
    variants.append(("full", cfg_full))

    cfg_no_shape = copy.deepcopy(base_cfg)
    cfg_no_shape.reward.use_step_shaping = False
    variants.append(("no_step_shaping", cfg_no_shape))

    cfg_no_reg = copy.deepcopy(base_cfg)
    cfg_no_reg.reward.regularity_bonus_coef = 0.0
    variants.append(("no_regularity_bonus", cfg_no_reg))

    cfg_path = copy.deepcopy(base_cfg)
    cfg_path.init_mode = "path"
    variants.append(("path_init", cfg_path))

    rows: List[Dict[str, object]] = []
    for tag, cfg in variants:
        model = train(cfg, device=device)
        ckpt = f"{cfg.checkpoint_dir}/best_model.pt"
        eval_rows = evaluate_instances(instances=instances, cfg=cfg, model_checkpoint=ckpt, device=device)
        rows.append({"tag": tag, "config": asdict(cfg), "eval": eval_rows})
    return rows
