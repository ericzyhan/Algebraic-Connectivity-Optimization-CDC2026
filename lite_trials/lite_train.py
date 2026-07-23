"""Lite training run (~20K steps) to test the new 3-term reward heuristic.

Usage:
    python -m lite_trials.lite_train          # from project root
    python lite_trials/lite_train.py          # direct
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from rl_train.config import load_config
from rl_train.train import run_training


def main() -> None:
    config_path = _project_root / "lite_trials" / "lite_config.yaml"
    print(f"Loading config from: {config_path}")
    cfg = load_config(str(config_path))

    print("Starting lite training...")
    print(f"  max_env_steps: {cfg.train.max_env_steps}")
    print(f"  n range: {cfg.curriculum.phases[0].n_min}-{cfg.curriculum.phases[0].n_max}")
    print(f"  reward: alpha1={cfg.env.reward_alpha_1:.2f} "
          f"alpha2={cfg.env.reward_alpha_2:.2f} "
          f"alpha3={cfg.env.reward_alpha_3:.2f} "
          f"eta_P={cfg.env.reward_eta_p:.2f}")

    run_training(
        cfg=cfg,
        resume_from=None,
        device="cpu",
        quiet=False,
    )

    print("\nLite training complete!")
    print(f"  Checkpoints: {_project_root / 'lite_trials' / 'checkpoints'}")
    print(f"  Metrics: {_project_root / 'lite_trials' / 'metrics.csv'}")


if __name__ == "__main__":
    main()
