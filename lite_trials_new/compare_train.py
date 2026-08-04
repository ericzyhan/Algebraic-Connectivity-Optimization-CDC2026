"""Train one lite model per arm (new analytic model vs old model).

Each arm gets a real, complete training loop -- PPO updates, curriculum,
periodic evaluation, checkpointing -- just capped at a small number of
environment steps (20K by default) so the whole comparison runs in minutes.

Usage (from project root):
    python -m lite_trials_new.compare_train
    python -m lite_trials_new.compare_train --arms old_model,new_model_blend
    python -m lite_trials_new.compare_train --max-env-steps 4000   # smoke test
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from lite_trials_new.arms import (  # noqa: E402
    LITE_ROOT,
    MAX_ENV_STEPS,
    Arm,
    base_config,
    select_arms,
)

MANIFEST_PATH = LITE_ROOT / "manifest.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", type=str, default=None, help="Comma list of arm tags (default: all)")
    p.add_argument("--max-env-steps", type=int, default=MAX_ENV_STEPS)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--skip-trained",
        action="store_true",
        help="Skip arms that already have a best.pt checkpoint.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue each arm from its checkpoints/last.pt when one exists "
             "(optimizer, buffers and RNG included). Arms without one train "
             "from scratch. An arm already at --max-env-steps exits at once, so "
             "raise it to actually train further.",
    )
    return p.parse_args()


def write_config(arm: Arm, max_env_steps: int) -> Path:
    cfg = base_config(arm, max_env_steps=max_env_steps)
    arm.config_path.parent.mkdir(parents=True, exist_ok=True)
    with arm.config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return arm.config_path


def train_arm(arm: Arm, config_path: Path, device: str, resume: bool = False) -> float:
    if not arm.repo_root.exists():
        raise FileNotFoundError(
            f"Arm '{arm.tag}' repo root not found: {arm.repo_root}. "
            "Set LITE_TRIALS_OLD_MODEL_ROOT if the old model lives elsewhere."
        )
    cmd = [
        sys.executable, "-m", f"{arm.package}.train",
        "--config", str(config_path),
        "--device", device,
    ]
    # All three packages take --resume-from; each restores optimizer, scheduler,
    # env manager, rollout buffer and RNG alongside the weights, so a resumed
    # leg continues the same run rather than starting a fresh one at step 0.
    resuming = resume and arm.last_checkpoint.exists()
    if resuming:
        cmd.extend(["--resume-from", str(arm.last_checkpoint)])
    print(f"\n=== Training arm '{arm.tag}' ({arm.label}) ===")
    print(f"  package : {arm.package}")
    print(f"  cwd     : {arm.repo_root}")
    print(f"  config  : {config_path}")
    if resume:
        print(f"  resume  : {arm.last_checkpoint if resuming else 'no last.pt -- from scratch'}")
    if arm.env_overrides:
        print(f"  reward  : {arm.env_overrides}")
    t0 = time.time()
    subprocess.run(cmd, cwd=str(arm.repo_root), check=True)
    return time.time() - t0


def main() -> None:
    args = parse_args()
    arms = select_arms(args.arms)

    manifest: list[dict[str, object]] = []
    for arm in arms:
        config_path = write_config(arm, args.max_env_steps)
        if args.skip_trained and arm.best_checkpoint.exists():
            print(f"\n=== Skipping arm '{arm.tag}' (checkpoint exists) ===")
            elapsed = float("nan")
        else:
            elapsed = train_arm(arm, config_path, args.device, resume=args.resume)
            print(f"  done in {elapsed:.1f}s")

        manifest.append(
            {
                "tag": arm.tag,
                "label": arm.label,
                "package": arm.package,
                "repo_root": str(arm.repo_root),
                "config_path": str(arm.config_path),
                "run_dir": str(arm.run_dir),
                "checkpoint_best": str(arm.best_checkpoint),
                "metrics_path": str(arm.metrics_path),
                "max_env_steps": int(args.max_env_steps),
                "env_overrides": arm.env_overrides,
                "train_wall_sec": elapsed,
                "notes": arm.notes,
            }
        )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
