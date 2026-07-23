"""Run a comprehensive path-vs-backbone evaluation for every trained combo in
lite_trials/sweep/manifest.json, then average algebraic connectivity over
density (rho) buckets so combos can be compared at "specific densities"
(sparse / medium / dense) rather than only pointwise across the raw sweep.

The default --n-values spans: a couple in-distribution sizes (lite trains on
n=8-12), out-of-distribution power-of-two sizes (16/32/64/128) where the
Cayley backbone construction (src/graph_design/backbones/cayley.py) gets its
easiest structure (even n needs no lift node, and n>=16 powers of two get a
bonus semidihedral generator set), even-but-not-power-of-2 sizes (24/48/96)
as a middle ground, and 91 = 7*13 -- an odd semiprime with no small-index
subgroup, which costs the Cayley construction its one real structural
penalty (an odd n needs a 1-node "lift and delete" pad). None of this makes
the backbone construction fail outright (it always falls back to the
envelope/complete-multipartite family), but it's the closest thing this
codebase has to graph sizes without an "easy" backbone.

Usage (from project root, after lite_sweep_train.py):
    python -m lite_trials.lite_sweep_experiment
    python -m lite_trials.lite_sweep_experiment --n-values 8,12,16,24 --points-per-n 15 --repeats 5
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from lite_trials.alpha_sweep_config import bucket_for_rho

_SWEEP_ROOT = _project_root / "lite_trials" / "sweep"
_MANIFEST_PATH = _SWEEP_ROOT / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-values", type=str, default="12,16,24,32,48,64,91,96,128",
        help="Graph sizes to evaluate. Default spans in-distribution sizes (lite trains on "
             "n=8-12) through out-of-distribution sizes up to 128, mixing power-of-two n "
             "(16/32/64/128, the Cayley backbone's easiest case), even non-power-of-2 n "
             "(24/48/96), and 91=7*13 (an odd semiprime -- the Cayley backbone's one real "
             "structural weak point; see module docstring).",
    )
    parser.add_argument("--points-per-n", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--manifest", type=str, default=str(_MANIFEST_PATH))
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _compute_density_bucket_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Average terminal lambda2 (raw -- algebraic connectivity is an absolute
    measure, not one that should be divided by n) over repeats within each
    (method, n, bucket, density point), then average those point-means across
    density points in the bucket -- algebraic connectivity averaged over a
    density regime rather than a single point estimate.
    """
    by_point: dict[tuple[str, int, str, float], list[float]] = {}
    for r in rows:
        method = str(r["method"])
        n = int(float(r["n"]))
        rho = float(r["rho_final"])
        bucket = bucket_for_rho(rho)
        lam2 = float(r["terminal_lambda2"])
        key = (method, n, bucket, round(rho, 6))
        by_point.setdefault(key, []).append(lam2)

    by_bucket: dict[tuple[str, int, str], list[float]] = {}
    for (method, n, bucket, _rho), vals in by_point.items():
        by_bucket.setdefault((method, n, bucket), []).append(float(np.mean(vals)))

    out: list[dict[str, object]] = []
    for (method, n, bucket), point_means in sorted(
        by_bucket.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])
    ):
        arr = np.asarray(point_means, dtype=np.float64)
        out.append(
            {
                "method": method,
                "n": n,
                "density_bucket": bucket,
                "lambda2_mean": float(np.mean(arr)),
                "lambda2_std": float(np.std(arr)),
                "density_points_in_bucket": int(arr.size),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}. Run lite_sweep_train.py first.")
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for i, entry in enumerate(manifest):
        tag = entry["tag"]
        checkpoint = _project_root / entry["checkpoint_best"]
        config = _project_root / entry["config_path"]
        if not checkpoint.exists():
            print(f"SKIP '{tag}': checkpoint not found at {checkpoint}")
            continue

        out_dir = _project_root / entry["run_dir"] / "results"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== [{i + 1}/{len(manifest)}] Evaluating combo '{tag}' ===")
        cmd = [
            sys.executable, "-m", "rl_train.run_path_vs_backbone_experiment",
            "--config", str(config),
            "--checkpoint", str(checkpoint),
            "--n-values", args.n_values,
            "--points-per-n", str(args.points_per_n),
            "--repeats", str(args.repeats),
            "--seed-start", str(args.seed_start),
            "--device", args.device,
            "--progress-every", str(args.progress_every),
            "--out-dir", str(out_dir),
            "--out-csv", "raw.csv",
            "--out-meta", "raw.meta.json",
            "--out-cost-csv", "summary_cost.csv",
            "--out-quality-csv", "summary_quality.csv",
        ]
        subprocess.run(cmd, cwd=str(_project_root), check=True)

        raw_csv = out_dir / "raw.csv"
        rows = _read_csv(raw_csv)
        bucket_rows = _compute_density_bucket_summary(rows)
        _write_csv(
            out_dir / "density_bucket_summary.csv",
            bucket_rows,
            [
                "method",
                "n",
                "density_bucket",
                "lambda2_mean",
                "lambda2_std",
                "density_points_in_bucket",
            ],
        )
        print(f"  Wrote: {out_dir / 'density_bucket_summary.csv'}")

    print("\nSweep experiment complete.")


if __name__ == "__main__":
    main()
