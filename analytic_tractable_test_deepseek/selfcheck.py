from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Dict, List

from .config import Config, load_config
from .train import run_training


def _clone_cfg(base: Config) -> Config:
    return copy.deepcopy(base)


def _set_run_paths(cfg: Config, run_root: Path) -> None:
    cfg.train.output_dir = str(run_root)
    cfg.checkpoint.dir = str(run_root / "checkpoints")
    cfg.logging.metrics_path = str(run_root / "metrics.csv")
    cfg.logging.tensorboard_dir = str(run_root / "tensorboard")
    cfg.logging.run_state_path = str(run_root / "run_state.json")


def _read_rows(path: Path, min_step_exclusive: int) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(float(row["global_env_steps"]))
            if step > min_step_exclusive:
                rows.append(
                    {
                        "global_env_steps": row["global_env_steps"],
                        "update_idx": row["update_idx"],
                        "phase": row["phase"],
                        "mean_episode_return": row["mean_episode_return"],
                        "mean_terminal_lambda2_norm": row["mean_terminal_lambda2_norm"],
                        "policy_loss": row["policy_loss"],
                        "value_loss": row["value_loss"],
                        "entropy": row["entropy"],
                        "approx_kl": row["approx_kl"],
                        "learning_rate": row["learning_rate"],
                        "selector_loss": row.get("selector_loss", "0.0"),
                        "selector_recall_at_k": row.get("selector_recall_at_k", "0.0"),
                        "selector_k_mean": row.get("selector_k_mean", "0.0"),
                        "selector_learning_rate": row.get("selector_learning_rate", "0.0"),
                        "best_eval_metric": row["best_eval_metric"],
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic resume selfcheck")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--total-env-steps", type=int, default=4000)
    parser.add_argument("--split-env-steps", type=int, default=2000)
    parser.add_argument("--root-dir", type=str, default="experiments/rl/runs/selfcheck/ppo")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json-out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)

    if args.split_env_steps >= args.total_env_steps:
        raise ValueError("split-env-steps must be smaller than total-env-steps")

    root_dir = Path(args.root_dir).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    continuous_dir = root_dir / "continuous"
    segmented_dir = root_dir / "segmented"

    continuous_cfg = _clone_cfg(base_cfg)
    _set_run_paths(continuous_cfg, continuous_dir)
    continuous_cfg.train.max_env_steps = int(args.total_env_steps)
    continuous_cfg.checkpoint.every_env_steps = int(args.split_env_steps)
    continuous_cfg.logging.console_log_every_env_steps = max(1, args.split_env_steps // 4)
    continuous_cfg.logging.eval_every_env_steps = int(args.split_env_steps)

    segmented_cfg_1 = _clone_cfg(base_cfg)
    _set_run_paths(segmented_cfg_1, segmented_dir)
    segmented_cfg_1.train.max_env_steps = int(args.split_env_steps)
    segmented_cfg_1.checkpoint.every_env_steps = int(args.split_env_steps)
    segmented_cfg_1.logging.console_log_every_env_steps = max(1, args.split_env_steps // 4)
    segmented_cfg_1.logging.eval_every_env_steps = int(args.split_env_steps)

    segmented_cfg_2 = _clone_cfg(base_cfg)
    _set_run_paths(segmented_cfg_2, segmented_dir)
    segmented_cfg_2.train.max_env_steps = int(args.total_env_steps)
    segmented_cfg_2.checkpoint.every_env_steps = int(args.split_env_steps)
    segmented_cfg_2.logging.console_log_every_env_steps = max(1, args.split_env_steps // 4)
    segmented_cfg_2.logging.eval_every_env_steps = int(args.split_env_steps)

    print("[selfcheck] running continuous training", flush=True)
    continuous_summary = run_training(
        cfg=continuous_cfg,
        resume_from=None,
        device=args.device,
        quiet=True,
    )

    print("[selfcheck] running segmented training part 1", flush=True)
    segmented_summary_1 = run_training(
        cfg=segmented_cfg_1,
        resume_from=None,
        device=args.device,
        quiet=True,
    )

    print("[selfcheck] running segmented training part 2 (resume)", flush=True)
    segmented_summary_2 = run_training(
        cfg=segmented_cfg_2,
        resume_from=segmented_summary_1["final_checkpoint"],
        device=args.device,
        quiet=True,
    )

    same_model_hash = continuous_summary["model_hash"] == segmented_summary_2["model_hash"]
    same_optimizer_hash = continuous_summary["optimizer_hash"] == segmented_summary_2["optimizer_hash"]
    same_selector_hash = continuous_summary.get("selector_hash", "") == segmented_summary_2.get(
        "selector_hash", ""
    )

    cont_rows = _read_rows(Path(continuous_cfg.logging.metrics_path), args.split_env_steps)
    seg_rows = _read_rows(Path(segmented_cfg_2.logging.metrics_path), args.split_env_steps)
    same_metric_rows = cont_rows == seg_rows

    passed = same_model_hash and same_optimizer_hash and same_selector_hash and same_metric_rows

    report = {
        "passed": passed,
        "same_model_hash": same_model_hash,
        "same_optimizer_hash": same_optimizer_hash,
        "same_selector_hash": same_selector_hash,
        "same_metric_rows_after_split": same_metric_rows,
        "continuous_summary": continuous_summary,
        "segmented_summary": segmented_summary_2,
        "continuous_metrics_rows_after_split": len(cont_rows),
        "segmented_metrics_rows_after_split": len(seg_rows),
    }

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print("[selfcheck] report:")
    print(json.dumps(report, indent=2))

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
