"""Run the whole new-vs-old lite comparison: train -> evaluate -> visualize.

Usage (from project root):
    python -m lite_trials_new.run_all
    python -m lite_trials_new.run_all --max-env-steps 4000 --points-per-n 5 --repeats 2
    python -m lite_trials_new.run_all --skip-train        # re-evaluate existing checkpoints
    python -m lite_trials_new.run_all --resume            # continue an interrupted run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from lite_trials_new.arms import MAX_ENV_STEPS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", type=str, default=None)
    p.add_argument("--max-env-steps", type=int, default=MAX_ENV_STEPS)
    p.add_argument("--n-values", type=str, default="12,16,32,64")
    p.add_argument("--points-per-n", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--baseline", type=str, default="old_model")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Pick up where an interrupted run stopped: continue each arm from "
             "its last.pt, and reuse evaluation shards that already match the "
             "requested grid.",
    )
    return p.parse_args()


def _run(module: str, extra: list[str]) -> None:
    cmd = [sys.executable, "-m", module] + extra
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(_project_root), check=True)


def main() -> None:
    args = parse_args()
    arm_args = ["--arms", args.arms] if args.arms else []

    resume_args = ["--resume"] if args.resume else []

    if not args.skip_train:
        _run(
            "lite_trials_new.compare_train",
            arm_args + resume_args
            + ["--max-env-steps", str(args.max_env_steps), "--device", args.device],
        )
    if not args.skip_eval:
        _run(
            "lite_trials_new.compare_experiment",
            arm_args + resume_args + [
                "--n-values", args.n_values,
                "--points-per-n", str(args.points_per_n),
                "--repeats", str(args.repeats),
                "--device", args.device,
            ],
        )
    _run("lite_trials_new.compare_visualize", ["--baseline", args.baseline])

    print("\nAll done. See lite_trials_new/comparison/")


if __name__ == "__main__":
    main()
