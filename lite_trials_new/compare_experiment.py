"""Path-vs-backbone evaluation of every trained arm, on one shared grid.

This mirrors ``<package>/run_path_vs_backbone_experiment.py`` (same non-trivial
density grid, same per-(n, init_mode) calls into ``<package>/evaluate.py``, same
raw/cost/quality summaries) but drives the evaluation itself, for two reasons:

1. ``run_path_vs_backbone_experiment.py`` hard-codes ``-m rl_train.evaluate``.
   Under ``analytic_tractable_test_deepseek`` that silently evaluates the
   *wrong* package, because this repo also contains an ``rl_train/``.
2. Each arm has to run with its own repo root as cwd, which that script cannot
   express (it derives the root from its own file location).

``evaluate.py`` is byte-identical between ``old_model/rl_train`` and
``analytic_tractable_test_deepseek``. ``rl_train_new_model``'s copy differs by
exactly 16 lines, all of them forwarding its extra tracker/candidacy config into
the env constructor -- the protocol itself (rollout loop, density grid, metrics,
CSV schema) is unchanged. So the evaluation is genuinely shared across all three
arms; only the env/model/features being evaluated differ.

Usage (from project root, after compare_train.py):
    python -m lite_trials_new.compare_experiment
    python -m lite_trials_new.compare_experiment --n-values 12,16,24 --points-per-n 8 --repeats 3
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from lite_trials_new.arms import LITE_ROOT, Arm, bucket_for_rho, select_arms  # noqa: E402

MANIFEST_PATH = LITE_ROOT / "manifest.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", type=str, default=None, help="Comma list of arm tags (default: all)")
    p.add_argument(
        "--n-values", type=str, default="12,16,32,64",
        help="Graph sizes. Training is on n=8-12, so 12 is in-distribution and "
             "16/32/64 probe generalization. The grid runs to 64 deliberately: "
             "the spectral trackers are closest at n<=32, so a grid that stops "
             "there understates the differences between them.",
    )
    p.add_argument("--points-per-n", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reuse per-(n, init_mode) shards already on disk that match the "
             "requested grid and are newer than the arm's checkpoint; re-run "
             "only the rest. Without it every shard is recomputed.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    return p.parse_args()


# --- density grid (matches run_path_vs_backbone_experiment.py) --------------


def _parse_int_list(raw: str) -> list[int]:
    return [int(tok.strip()) for tok in str(raw).split(",") if tok.strip()]


def _nontrivial_m_points(n: int, target_points: int) -> list[int]:
    """Edge counts strictly between a spanning tree (n-1) and K_n."""
    m_lo = int(n) - 1
    m_hi = (int(n) * (int(n) - 1)) // 2
    all_nontrivial = list(range(m_lo + 1, m_hi))
    if len(all_nontrivial) <= int(target_points):
        return all_nontrivial
    idx = np.linspace(0, len(all_nontrivial) - 1, int(target_points), dtype=np.float64)
    idx_u = sorted(set(int(i) for i in np.round(idx).astype(np.int64).tolist()))
    return [int(all_nontrivial[i]) for i in idx_u]


def _rho_conn_from_m(n: int, m: int) -> float:
    m_lo = int(n) - 1
    m_hi = (int(n) * (int(n) - 1)) // 2
    den = m_hi - m_lo
    return float((int(m) - m_lo) / float(den)) if den > 0 else 0.0


# --- csv helpers -----------------------------------------------------------


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# --- evaluation ------------------------------------------------------------

_BACKBONE_ARGS = [
    "--no-cayley-multi-index-overlap",
    "--cayley-enable-index4",
    "--cayley-index2-overlap-high", "0.6",
    "--cayley-index3-overlap-high", str(2.0 / 3.0),
    "--cayley-index4-overlap-low", str(2.0 / 3.0),
    "--cayley-eval-mode", "dense",
]


def _run_eval(
    arm: Arm,
    *,
    n: int,
    densities: list[float],
    seeds: list[int],
    device: str,
    progress_every: int,
    init_mode: str,
    out_csv: Path,
) -> None:
    cmd = [
        sys.executable, "-m", f"{arm.package}.evaluate",
        "--config", str(arm.config_path),
        "--checkpoint", f"best={arm.best_checkpoint}",
        "--n-values", str(int(n)),
        "--densities", ",".join(f"{rho:.12f}" for rho in densities),
        "--seeds", ",".join(str(int(s)) for s in seeds),
        "--device", str(device),
        "--progress-every", str(int(progress_every)),
        "--init-mode", str(init_mode),
        "--out", str(out_csv),
    ]
    if init_mode == "backbone":
        cmd.extend(_BACKBONE_ARGS)
    subprocess.run(cmd, cwd=str(arm.repo_root), check=True)


def _shard_reusable(
    arm: Arm,
    *,
    n: int,
    densities: list[float],
    seeds: list[int],
    init_mode: str,
    out_csv: Path,
) -> tuple[bool, str]:
    """Can an existing per-(n, init_mode) shard stand in for re-running it?

    evaluate.py writes its CSV only after every rollout in the shard succeeds,
    so a shard that exists is complete for the grid it was called with -- but
    not necessarily for the grid being asked for now. The check is therefore
    against the sidecar evaluate.py drops next to the CSV, which records that
    grid verbatim; a leftover from a different --points-per-n or --repeats has
    the right shape for the wrong points. The mtime comparison covers the other
    stale case, which no grid check can see: retraining an arm rewrites best.pt
    in place, leaving its path unchanged.
    """
    meta_path = out_csv.with_suffix(".meta.json")
    if not out_csv.exists():
        return False, "no csv"
    if not meta_path.exists():
        return False, "no meta sidecar"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"unreadable meta ({type(exc).__name__})"

    if [int(v) for v in meta.get("n_values", [])] != [int(n)]:
        return False, "n mismatch"
    if [int(v) for v in meta.get("seeds", [])] != [int(s) for s in seeds]:
        return False, "seeds mismatch"
    if str(meta.get("init_mode", "")) != init_mode:
        return False, "init_mode mismatch"
    # _run_eval formats the grid to 12 dp, so compare at the same precision.
    if [round(float(r), 12) for r in meta.get("densities", [])] != [
        round(float(r), 12) for r in densities
    ]:
        return False, "density grid mismatch"

    want_rows = len(densities) * len(seeds)
    got_rows = len(_read_csv(out_csv))
    if got_rows != want_rows:
        return False, f"{got_rows} rows, want {want_rows}"
    if (
        arm.best_checkpoint.exists()
        and out_csv.stat().st_mtime < arm.best_checkpoint.stat().st_mtime
    ):
        return False, "checkpoint is newer than the shard"
    return True, f"{got_rows} rows"


# --- summaries -------------------------------------------------------------


#: Runtime channels kept in the cost summary.
#:   rl_runtime_sec        -- the policy's own rollout time. This is the number
#:                            that actually separates the two code bases.
#:   init_build_runtime_sec-- backbone construction, shared identical code.
#:   runtime_total_sec     -- rl + init.
_RUNTIME_COLS = ("rl_runtime_sec", "init_build_runtime_sec", "runtime_total_sec")


def _cost_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Wall-clock per episode, averaged over repeats then over density points.

    Also reports per-edge-decision cost (rollout time / episode length), which
    is the density-independent way to compare the two agents' step cost.
    """
    by_point: dict[tuple[str, int, int], dict[str, list[float]]] = {}
    for r in rows:
        key = (str(r["method"]), int(r["n"]), int(r["m_target"]))
        slot = by_point.setdefault(key, {c: [] for c in (*_RUNTIME_COLS, "per_step")})
        for col in _RUNTIME_COLS:
            slot[col].append(float(r[col]))
        steps = max(1.0, float(r["episode_len"]))
        slot["per_step"].append(float(r["rl_runtime_sec"]) / steps)

    by_method_n: dict[tuple[str, int], dict[str, list[float]]] = {}
    for (method, n, _m), slot in by_point.items():
        agg = by_method_n.setdefault((method, n), {c: [] for c in (*_RUNTIME_COLS, "per_step")})
        for col, vals in slot.items():
            agg[col].append(float(np.mean(vals)))

    out: list[dict[str, object]] = []
    for (method, n), agg in sorted(by_method_n.items()):
        total = np.asarray(agg["runtime_total_sec"], dtype=np.float64)
        rl = np.asarray(agg["rl_runtime_sec"], dtype=np.float64)
        init = np.asarray(agg["init_build_runtime_sec"], dtype=np.float64)
        per_step = np.asarray(agg["per_step"], dtype=np.float64)
        out.append(
            {
                "method": method,
                "n": int(n),
                "runtime_mean_over_density_s": float(np.mean(total)),
                "runtime_std_over_density_s": float(np.std(total)),
                "rl_runtime_mean_over_density_s": float(np.mean(rl)),
                "rl_runtime_std_over_density_s": float(np.std(rl)),
                "init_runtime_mean_over_density_s": float(np.mean(init)),
                "rl_runtime_per_step_mean_s": float(np.mean(per_step)),
                "density_points": int(total.size),
            }
        )
    return out


def _bucket_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Raw terminal lambda_2 averaged over repeats, then over the density points
    that fall in each sparse/medium/dense bucket.

    lambda_2 is reported raw -- algebraic connectivity is an absolute measure,
    not something to divide by n.
    """
    by_point: dict[tuple[str, int, str, float], list[float]] = {}
    for r in rows:
        rho = float(r["rho_final"])
        key = (str(r["method"]), int(r["n"]), bucket_for_rho(rho), round(rho, 6))
        by_point.setdefault(key, []).append(float(r["terminal_lambda2"]))

    by_bucket: dict[tuple[str, int, str], list[float]] = {}
    for (method, n, bucket, _rho), vals in by_point.items():
        by_bucket.setdefault((method, n, bucket), []).append(float(np.mean(vals)))

    out: list[dict[str, object]] = []
    for (method, n, bucket), point_means in sorted(by_bucket.items()):
        arr = np.asarray(point_means, dtype=np.float64)
        out.append(
            {
                "method": method,
                "n": int(n),
                "density_bucket": bucket,
                "lambda2_mean": float(np.mean(arr)),
                "lambda2_std": float(np.std(arr)),
                "density_points_in_bucket": int(arr.size),
            }
        )
    return out


def _quality_summary(rows: list[dict[str, object]], tie_tol: float = 1e-10) -> list[dict[str, object]]:
    """Backbone-vs-path delta within an arm, paired by (n, m_target, seed)."""
    path_l2: dict[tuple[int, int, int], float] = {}
    for r in rows:
        if str(r["method"]) == "RL+Path":
            path_l2[(int(r["n"]), int(r["m_target"]), int(r["repeat_idx"]))] = float(
                r["terminal_lambda2"]
            )

    delta_by_point: dict[tuple[str, int, int], list[float]] = {}
    delta_pct_by_point: dict[tuple[str, int, int], list[float]] = {}
    win: dict[tuple[str, int], int] = {}
    tie: dict[tuple[str, int], int] = {}
    total: dict[tuple[str, int], int] = {}

    for r in rows:
        method, n, m, rep = str(r["method"]), int(r["n"]), int(r["m_target"]), int(r["repeat_idx"])
        base = path_l2.get((n, m, rep))
        if base is None:
            continue
        d = float(r["terminal_lambda2"]) - base
        delta_by_point.setdefault((method, n, m), []).append(d)
        delta_pct_by_point.setdefault((method, n, m), []).append(
            d / max(1e-15, abs(base)) * 100.0
        )
        total[(method, n)] = total.get((method, n), 0) + 1
        if abs(d) <= tie_tol:
            tie[(method, n)] = tie.get((method, n), 0) + 1
        elif d > tie_tol:
            win[(method, n)] = win.get((method, n), 0) + 1

    d_by_method_n: dict[tuple[str, int], list[float]] = {}
    p_by_method_n: dict[tuple[str, int], list[float]] = {}
    for (method, n, m), vals in delta_by_point.items():
        d_by_method_n.setdefault((method, n), []).append(float(np.mean(vals)))
        p_by_method_n.setdefault((method, n), []).append(
            float(np.mean(delta_pct_by_point[(method, n, m)]))
        )

    out: list[dict[str, object]] = []
    for method, n in sorted(d_by_method_n):
        d_arr = np.asarray(d_by_method_n[(method, n)], dtype=np.float64)
        p_arr = np.asarray(p_by_method_n[(method, n)], dtype=np.float64)
        tot = total.get((method, n), 0)
        out.append(
            {
                "method": method,
                "n": int(n),
                "delta_lambda2_mean_over_density": float(np.mean(d_arr)),
                "delta_lambda2_std_over_density": float(np.std(d_arr)),
                "delta_lambda2_pct_mean_over_density": float(np.mean(p_arr)),
                "win_pct_vs_rl_path_over_all_points": float(100.0 * win.get((method, n), 0) / tot) if tot else 0.0,
                "tie_pct_vs_rl_path_over_all_points": float(100.0 * tie.get((method, n), 0) / tot) if tot else 0.0,
                "total_points": int(tot),
                "density_points": int(d_arr.size),
            }
        )
    return out


# --- driver ----------------------------------------------------------------


def evaluate_arm(arm: Arm, args: argparse.Namespace, n_values: list[int], seeds: list[int]) -> None:
    out_dir = arm.results_dir
    per_mode_dir = out_dir / "per_n_mode"
    per_mode_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []
    fields: list[str] | None = None
    t0 = time.time()

    for n in n_values:
        m_points = _nontrivial_m_points(n, args.points_per_n)
        rho_values = [_rho_conn_from_m(n, m) for m in m_points]
        print(f"[{arm.tag}] n={n}, density_points={len(rho_values)}, repeats={len(seeds)}", flush=True)
        for init_mode in ("path", "backbone"):
            out_csv = per_mode_dir / f"eval_{init_mode}_n{n}.csv"
            reuse, why = (
                _shard_reusable(
                    arm,
                    n=n,
                    densities=rho_values,
                    seeds=seeds,
                    init_mode=init_mode,
                    out_csv=out_csv,
                )
                if args.resume
                else (False, "")
            )
            if reuse:
                print(f"[{arm.tag}] resume: reusing {out_csv.name} ({why})", flush=True)
            else:
                if args.resume:
                    print(f"[{arm.tag}] resume: re-running {out_csv.name} ({why})", flush=True)
                _run_eval(
                    arm,
                    n=n,
                    densities=rho_values,
                    seeds=seeds,
                    device=args.device,
                    progress_every=args.progress_every,
                    init_mode=init_mode,
                    out_csv=out_csv,
                )
            mode_rows = _read_csv(out_csv)
            for row in mode_rows:
                enriched: dict[str, object] = dict(row)
                enriched["method"] = "RL+Backbone" if init_mode == "backbone" else "RL+Path"
                enriched["repeat_idx"] = int(float(row["seed"]))
                enriched["arm"] = arm.tag
                enriched["arm_label"] = arm.label
                raw_rows.append(enriched)
            if fields is None and mode_rows:
                fields = list(mode_rows[0].keys())

    if fields is None:
        raise RuntimeError(f"No rows produced for arm '{arm.tag}'.")
    for extra in ("method", "repeat_idx", "arm", "arm_label"):
        if extra not in fields:
            fields.append(extra)

    _write_csv(out_dir / "raw.csv", raw_rows, fields)
    _write_csv(
        out_dir / "density_bucket_summary.csv",
        _bucket_summary(raw_rows),
        ["method", "n", "density_bucket", "lambda2_mean", "lambda2_std", "density_points_in_bucket"],
    )
    _write_csv(
        out_dir / "summary_cost.csv",
        _cost_summary(raw_rows),
        [
            "method", "n",
            "runtime_mean_over_density_s", "runtime_std_over_density_s",
            "rl_runtime_mean_over_density_s", "rl_runtime_std_over_density_s",
            "init_runtime_mean_over_density_s", "rl_runtime_per_step_mean_s",
            "density_points",
        ],
    )
    _write_csv(
        out_dir / "summary_quality.csv",
        _quality_summary(raw_rows),
        [
            "method", "n",
            "delta_lambda2_mean_over_density", "delta_lambda2_std_over_density",
            "delta_lambda2_pct_mean_over_density",
            "win_pct_vs_rl_path_over_all_points", "tie_pct_vs_rl_path_over_all_points",
            "total_points", "density_points",
        ],
    )
    (out_dir / "raw.meta.json").write_text(
        json.dumps(
            {
                "arm": arm.tag,
                "label": arm.label,
                "package": arm.package,
                "repo_root": str(arm.repo_root),
                "config": str(arm.config_path),
                "checkpoint": str(arm.best_checkpoint),
                "n_values": n_values,
                "points_per_n": int(args.points_per_n),
                "repeats": int(args.repeats),
                "seeds": seeds,
                "total_rows": len(raw_rows),
                "eval_wall_sec": time.time() - t0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[{arm.tag}] wrote {len(raw_rows)} rows to {out_dir} ({time.time() - t0:.1f}s)")


def main() -> None:
    args = parse_args()
    arms = select_arms(args.arms)
    n_values = _parse_int_list(args.n_values)
    if not n_values:
        raise ValueError("--n-values is empty.")
    seeds = [int(args.seed_start) + i for i in range(max(1, int(args.repeats)))]

    jobs_per_arm = sum(len(_nontrivial_m_points(n, args.points_per_n)) for n in n_values) * len(seeds) * 2
    print(
        json.dumps(
            {
                "arms": [a.tag for a in arms],
                "n_values": n_values,
                "points_per_n": args.points_per_n,
                "seeds": seeds,
                "rollouts_per_arm": jobs_per_arm,
                "rollouts_total": jobs_per_arm * len(arms),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

    missing = [a.tag for a in arms if not a.best_checkpoint.exists()]
    if missing:
        print(f"ERROR: no checkpoint for arm(s) {missing}. Run compare_train.py first.")
        sys.exit(1)

    for arm in arms:
        print(f"\n=== Evaluating arm '{arm.tag}' ({arm.label}) ===")
        evaluate_arm(arm, args, n_values, seeds)

    print("\nComparison experiment complete.")


if __name__ == "__main__":
    main()
