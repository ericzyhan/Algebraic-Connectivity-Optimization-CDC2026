"""Lite experiment: evaluate trained policy on path vs backbone completion.

Usage (from project root):
    python lite_trials/lite_experiment.py

Uses the full pre-trained checkpoint (trained up to n=64) and evaluates
at n=24, 32, 91 with 10 density points and 3 repeats.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent


def main() -> None:
    # Use the lite-trained checkpoint and lite config
    checkpoint = _project_root / "lite_trials" / "checkpoints" / "best.pt"
    config = _project_root / "lite_trials" / "lite_config.yaml"

    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        sys.exit(1)

    out_dir = _project_root / "lite_trials" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "rl_train.run_path_vs_backbone_experiment",
        "--config", str(config),
        "--checkpoint", str(checkpoint),
        "--n-values", "24,32,91",
        "--points-per-n", "10",
        "--repeats", "3",
        "--seed-start", "0",
        "--device", "cpu",
        "--progress-every", "3",
        "--out-dir", str(out_dir),
        "--out-csv", "raw.csv",
        "--out-meta", "raw.meta.json",
        "--out-cost-csv", "summary_cost.csv",
        "--out-quality-csv", "summary_quality.csv",
    ]

    print(f"Running experiment for n=24,32,91...")
    print(f"  Checkpoint: {checkpoint}")
    subprocess.run(cmd, cwd=str(_project_root), check=True)
    print(f"\nResults written to: {out_dir}")


if __name__ == "__main__":
    main()
