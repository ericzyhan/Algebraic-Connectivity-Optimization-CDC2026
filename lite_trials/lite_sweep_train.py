"""Train one lite model per (alpha_1, alpha_2, alpha_3) reward-blend combo.

Each combo gets its own output directory under lite_trials/sweep/<tag>/ so
checkpoints, metrics, tensorboard logs, and run state never collide between
combos. A manifest.json is written (and updated after every combo, so a
crash partway through the sweep doesn't lose completed runs) that downstream
stages (lite_sweep_experiment.py, lite_sweep_visualize.py) read to find each
trained checkpoint and its alpha triple.

Usage (from project root):
    python -m lite_trials.lite_sweep_train
    python -m lite_trials.lite_sweep_train --alphas "l2_only:1,0,0;balanced:0.34,0.33,0.33"
    python -m lite_trials.lite_sweep_train --max-env-steps 20000 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from rl_train.config import load_config
from rl_train.train import run_training

from lite_trials.alpha_sweep_config import parse_alpha_combos

_BASE_CONFIG = _project_root / "lite_trials" / "lite_config.yaml"
_SWEEP_ROOT = _project_root / "lite_trials" / "sweep"
_MANIFEST_PATH = _SWEEP_ROOT / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alphas", type=str, default=None,
        help="Semicolon-separated 'tag:a1,a2,a3' combos. Defaults to the preset sweep "
             "in alpha_sweep_config.DEFAULT_ALPHA_COMBOS.",
    )
    parser.add_argument(
        "--max-env-steps", type=int, default=None,
        help="Override train.max_env_steps for every combo (defaults to lite_config.yaml's value).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def _rel(path: Path) -> str:
    return str(path.relative_to(_project_root)).replace("\\", "/")


def main() -> None:
    args = parse_args()
    combos = parse_alpha_combos(args.alphas)
    _SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for i, combo in enumerate(combos):
        total = combo.alpha_1 + combo.alpha_2 + combo.alpha_3
        if abs(total - 1.0) > 1e-6:
            print(
                f"WARNING: alpha combo '{combo.tag}' sums to {total:.4f}, not 1.0 "
                f"(reward is a weighted sum regardless, just not a strict convex combination)."
            )

        run_dir = _SWEEP_ROOT / combo.tag
        overrides: dict[str, object] = {
            "env": {
                "reward_alpha_1": combo.alpha_1,
                "reward_alpha_2": combo.alpha_2,
                "reward_alpha_3": combo.alpha_3,
            },
            "train": {"output_dir": _rel(run_dir)},
            "checkpoint": {"dir": _rel(run_dir / "checkpoints")},
            "logging": {
                "metrics_path": _rel(run_dir / "metrics.csv"),
                "tensorboard_dir": _rel(run_dir / "tensorboard"),
                "run_state_path": _rel(run_dir / "run_state.json"),
            },
        }
        if args.max_env_steps is not None:
            overrides["train"] = dict(overrides["train"], max_env_steps=int(args.max_env_steps))

        cfg = load_config(str(_BASE_CONFIG), overrides=overrides)

        print(
            f"\n=== [{i + 1}/{len(combos)}] Training combo '{combo.tag}' "
            f"(alpha1={combo.alpha_1:.2f}, alpha2={combo.alpha_2:.2f}, alpha3={combo.alpha_3:.2f}) ==="
        )
        print(f"  output: {run_dir}")

        run_training(cfg=cfg, resume_from=None, device=args.device, quiet=False)

        best_ckpt = run_dir / "checkpoints" / "best.pt"
        manifest.append(
            {
                "tag": combo.tag,
                "alpha_1": combo.alpha_1,
                "alpha_2": combo.alpha_2,
                "alpha_3": combo.alpha_3,
                "config_path": _rel(_BASE_CONFIG),
                "run_dir": _rel(run_dir),
                "checkpoint_best": _rel(best_ckpt),
                "metrics_path": _rel(run_dir / "metrics.csv"),
            }
        )
        _MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nSweep training complete. Manifest written to: {_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
