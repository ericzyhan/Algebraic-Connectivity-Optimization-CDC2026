from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _parse_int_list(raw: str) -> list[int]:
    return [int(tok.strip()) for tok in str(raw).split(",") if tok.strip()]


def _nontrivial_m_points(n: int, target_points: int) -> list[int]:
    n_i = int(n)
    m_lo = int(n_i - 1)
    m_hi = int((n_i * (n_i - 1)) // 2)
    all_nontrivial = list(range(m_lo + 1, m_hi))
    if len(all_nontrivial) <= int(target_points):
        return all_nontrivial

    idx = np.linspace(0, len(all_nontrivial) - 1, int(target_points), dtype=np.float64)
    idx_i = np.round(idx).astype(np.int64)
    idx_u = sorted(set(int(i) for i in idx_i.tolist()))
    if len(idx_u) < int(target_points):
        used = set(idx_u)
        for i in range(len(all_nontrivial)):
            if i not in used:
                idx_u.append(i)
                used.add(i)
            if len(idx_u) >= int(target_points):
                break
        idx_u = sorted(idx_u[: int(target_points)])
    return [int(all_nontrivial[i]) for i in idx_u]


def _rho_conn_from_m(n: int, m: int) -> float:
    n_i = int(n)
    m_lo = int(n_i - 1)
    m_hi = int((n_i * (n_i - 1)) // 2)
    den = int(m_hi - m_lo)
    if den <= 0:
        return 0.0
    return float((int(m) - m_lo) / float(den))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _to_int(v: str | int | float) -> int:
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    return int(float(v))


def _to_float(v: str | int | float) -> float:
    if isinstance(v, float):
        return v
    if isinstance(v, int):
        return float(v)
    return float(v)


def _compute_cost_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    # 1) average over repeats per (method,n,m_target)
    rt_by_point: dict[tuple[str, int, int], list[float]] = {}
    for r in rows:
        k = (str(r["method"]), int(r["n"]), int(r["m_target"]))
        rt_by_point.setdefault(k, []).append(float(r["runtime_total_sec"]))

    # 2) average those point means over density points
    rt_point_mean: dict[tuple[str, int], list[float]] = {}
    repeats_by_point: dict[tuple[str, int], list[int]] = {}
    for (method, n, _m), vv in rt_by_point.items():
        rt_point_mean.setdefault((method, n), []).append(float(np.mean(np.asarray(vv, dtype=np.float64))))
        repeats_by_point.setdefault((method, n), []).append(len(vv))

    out: list[dict[str, object]] = []
    for (method, n), vals in sorted(rt_point_mean.items(), key=lambda x: (x[0][0], x[0][1])):
        arr = np.asarray(vals, dtype=np.float64)
        reps = repeats_by_point[(method, n)]
        out.append(
            {
                "method": method,
                "n": int(n),
                "runtime_mean_over_density_s": float(np.mean(arr)),
                "runtime_std_over_density_s": float(np.std(arr)),
                "density_points": int(arr.size),
                "avg_repeats_per_point": float(np.mean(np.asarray(reps, dtype=np.float64))),
                "min_repeats_per_point": int(min(reps)),
                "max_repeats_per_point": int(max(reps)),
            }
        )
    return out


def _compute_quality_summary(rows: list[dict[str, object]], tie_tol: float = 1e-10) -> list[dict[str, object]]:
    # Path mode as baseline by (n,m_target,repeat_idx)
    path_l2: dict[tuple[int, int, int], float] = {}
    for r in rows:
        if str(r["method"]) == "RL+Path":
            key = (int(r["n"]), int(r["m_target"]), int(r["repeat_idx"]))
            path_l2[key] = float(r["terminal_lambda2"])

    delta_by_point: dict[tuple[str, int, int], list[float]] = {}
    delta_pct_by_point: dict[tuple[str, int, int], list[float]] = {}
    win_count: dict[tuple[str, int], int] = {}
    tie_count: dict[tuple[str, int], int] = {}
    total_count: dict[tuple[str, int], int] = {}
    repeats_count_by_method_n: dict[tuple[str, int], list[int]] = {}

    for r in rows:
        method = str(r["method"])
        n = int(r["n"])
        m = int(r["m_target"])
        rep = int(r["repeat_idx"])
        lam = float(r["terminal_lambda2"])

        base = path_l2.get((n, m, rep), None)
        if base is None:
            continue
        d = float(lam - base)
        d_pct = float(d / max(1e-15, abs(base)) * 100.0)

        k_point = (method, n, m)
        delta_by_point.setdefault(k_point, []).append(d)
        delta_pct_by_point.setdefault(k_point, []).append(d_pct)
        repeats_count_by_method_n.setdefault((method, n), []).append(rep)

        k_method_n = (method, n)
        total_count[k_method_n] = int(total_count.get(k_method_n, 0) + 1)
        if abs(d) <= float(tie_tol):
            tie_count[k_method_n] = int(tie_count.get(k_method_n, 0) + 1)
        elif d > float(tie_tol):
            win_count[k_method_n] = int(win_count.get(k_method_n, 0) + 1)

    # Average over repeats first, then over density points.
    q_delta_mean_by_method_n: dict[tuple[str, int], list[float]] = {}
    q_delta_pct_mean_by_method_n: dict[tuple[str, int], list[float]] = {}
    rep_counts_by_method_n: dict[tuple[str, int], list[int]] = {}
    for (method, n, m), vv in delta_by_point.items():
        q_delta_mean_by_method_n.setdefault((method, n), []).append(float(np.mean(np.asarray(vv, dtype=np.float64))))
        q_delta_pct_mean_by_method_n.setdefault((method, n), []).append(
            float(np.mean(np.asarray(delta_pct_by_point[(method, n, m)], dtype=np.float64)))
        )
        rep_counts_by_method_n.setdefault((method, n), []).append(len(vv))

    out: list[dict[str, object]] = []
    for method, n in sorted(set(q_delta_mean_by_method_n.keys()), key=lambda x: (x[0], x[1])):
        d_arr = np.asarray(q_delta_mean_by_method_n[(method, n)], dtype=np.float64)
        p_arr = np.asarray(q_delta_pct_mean_by_method_n[(method, n)], dtype=np.float64)
        total = int(total_count.get((method, n), 0))
        win = int(win_count.get((method, n), 0))
        tie = int(tie_count.get((method, n), 0))
        reps = rep_counts_by_method_n[(method, n)]
        out.append(
            {
                "method": method,
                "n": int(n),
                "delta_lambda2_mean_over_density": float(np.mean(d_arr)),
                "delta_lambda2_std_over_density": float(np.std(d_arr)),
                "delta_lambda2_pct_mean_over_density": float(np.mean(p_arr)),
                "delta_lambda2_pct_std_over_density": float(np.std(p_arr)),
                "win_pct_vs_rl_path_over_all_points": float(100.0 * win / total) if total > 0 else 0.0,
                "tie_pct_vs_rl_path_over_all_points": float(100.0 * tie / total) if total > 0 else 0.0,
                "win_count": int(win),
                "tie_count": int(tie),
                "total_points": int(total),
                "density_points": int(d_arr.size),
                "avg_repeats_per_point": float(np.mean(np.asarray(reps, dtype=np.float64))),
            }
        )
    return out


def _run_eval(
    *,
    project_root: Path,
    config: Path,
    checkpoint: Path,
    n: int,
    densities: list[float],
    seeds: list[int],
    device: str,
    progress_every: int,
    init_mode: str,
    out_csv: Path,
    backbone_args: dict[str, object],
) -> None:
    densities_arg = ",".join(f"{rho:.12f}" for rho in densities)
    seeds_arg = ",".join(str(int(s)) for s in seeds)

    cmd: list[str] = [
        sys.executable,
        "-m",
        "rl_train.evaluate",
        "--config",
        str(config),
        "--checkpoint",
        f"best={checkpoint}",
        "--n-values",
        str(int(n)),
        "--densities",
        densities_arg,
        "--seeds",
        seeds_arg,
        "--device",
        str(device),
        "--progress-every",
        str(int(progress_every)),
        "--init-mode",
        str(init_mode),
        "--out",
        str(out_csv),
    ]

    if init_mode == "backbone":
        if bool(backbone_args["cayley_multi_index_overlap"]):
            cmd.append("--cayley-multi-index-overlap")
        else:
            cmd.append("--no-cayley-multi-index-overlap")
        if bool(backbone_args["cayley_enable_index4"]):
            cmd.append("--cayley-enable-index4")
        else:
            cmd.append("--no-cayley-enable-index4")
        cmd.extend(
            [
                "--cayley-index2-overlap-high",
                str(float(backbone_args["cayley_index2_overlap_high"])),
                "--cayley-index3-overlap-high",
                str(float(backbone_args["cayley_index3_overlap_high"])),
                "--cayley-index4-overlap-low",
                str(float(backbone_args["cayley_index4_overlap_low"])),
                "--cayley-eval-mode",
                str(backbone_args["cayley_eval_mode"]),
            ]
        )

    subprocess.run(cmd, cwd=str(project_root), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RL+Path vs RL+Backbone experiment with baseline-style non-trivial density sweeps."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-values", type=str, default="8,16,32,64,128")
    parser.add_argument("--points-per-n", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--out-csv", type=str, default="raw_rl_path_vs_backbone.csv")
    parser.add_argument("--out-meta", type=str, default="raw_rl_path_vs_backbone.meta.json")
    parser.add_argument("--out-cost-csv", type=str, default="summary_cost.csv")
    parser.add_argument("--out-quality-csv", type=str, default="summary_quality.csv")

    parser.add_argument("--cayley-index2-overlap-high", type=float, default=0.6)
    parser.add_argument("--cayley-index3-overlap-high", type=float, default=2.0 / 3.0)
    parser.add_argument("--cayley-index4-overlap-low", type=float, default=2.0 / 3.0)
    parser.add_argument(
        "--cayley-multi-index-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cayley-enable-index4",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cayley-eval-mode",
        type=str,
        default="dense",
        choices=["dense", "character"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    n_values = _parse_int_list(args.n_values)
    if not n_values:
        raise ValueError("--n-values is empty.")
    points_per_n = max(1, int(args.points_per_n))
    repeats = max(1, int(args.repeats))
    seeds = [int(args.seed_start) + i for i in range(repeats)]

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "per_n_mode"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / str(args.out_csv)
    out_meta = out_dir / str(args.out_meta)
    out_cost_csv = out_dir / str(args.out_cost_csv)
    out_quality_csv = out_dir / str(args.out_quality_csv)

    backbone_args: dict[str, object] = {
        "cayley_index2_overlap_high": float(args.cayley_index2_overlap_high),
        "cayley_index3_overlap_high": float(args.cayley_index3_overlap_high),
        "cayley_index4_overlap_low": float(args.cayley_index4_overlap_low),
        "cayley_multi_index_overlap": bool(args.cayley_multi_index_overlap),
        "cayley_enable_index4": bool(args.cayley_enable_index4),
        "cayley_eval_mode": str(args.cayley_eval_mode),
    }

    plan: dict[int, dict[str, object]] = {}
    total_jobs = 0
    for n in n_values:
        m_points = _nontrivial_m_points(n, points_per_n)
        rho_values = [_rho_conn_from_m(n, m) for m in m_points]
        jobs_n = len(rho_values) * len(seeds) * 2
        total_jobs += jobs_n
        plan[int(n)] = {
            "n": int(n),
            "m_points": len(m_points),
            "m_min": int(min(m_points)) if m_points else None,
            "m_max": int(max(m_points)) if m_points else None,
            "rho_min": float(min(rho_values)) if rho_values else None,
            "rho_max": float(max(rho_values)) if rho_values else None,
            "jobs_n": int(jobs_n),
        }

    preview = {
        "name": "rl_path_vs_backbone",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "n_values": n_values,
        "points_per_n": points_per_n,
        "repeats": repeats,
        "seed_start": int(args.seed_start),
        "seeds": seeds,
        "total_jobs": int(total_jobs),
        "dry_run": bool(args.dry_run),
        "backbone": backbone_args,
        "outputs": {
            "out_csv": str(out_csv),
            "out_meta": str(out_meta),
            "out_cost_csv": str(out_cost_csv),
            "out_quality_csv": str(out_quality_csv),
        },
        "plan": plan,
    }
    print("RL Path-vs-Backbone Plan")
    print(json.dumps(preview, indent=2))
    if args.dry_run:
        return

    raw_rows: list[dict[str, object]] = []
    first_fields: list[str] | None = None
    for n in n_values:
        m_points = _nontrivial_m_points(n, points_per_n)
        rho_values = [_rho_conn_from_m(n, m) for m in m_points]
        print(
            f"[run] n={n}, nontrivial_points={len(rho_values)}, repeats={repeats}",
            flush=True,
        )
        for init_mode in ("path", "backbone"):
            out_csv_mode_n = tmp_dir / f"eval_{init_mode}_n{n}.csv"
            print(
                f"[run] n={n}, init_mode={init_mode}, out={out_csv_mode_n}",
                flush=True,
            )
            _run_eval(
                project_root=project_root,
                config=config,
                checkpoint=checkpoint,
                n=n,
                densities=rho_values,
                seeds=seeds,
                device=args.device,
                progress_every=int(args.progress_every),
                init_mode=init_mode,
                out_csv=out_csv_mode_n,
                backbone_args=backbone_args,
            )
            rows_mode_n = _read_csv_rows(out_csv_mode_n)
            for row in rows_mode_n:
                enriched: dict[str, object] = dict(row)
                enriched["method"] = "RL+Backbone" if init_mode == "backbone" else "RL+Path"
                enriched["repeat_idx"] = int(_to_int(row["seed"]))
                raw_rows.append(enriched)
            if first_fields is None and rows_mode_n:
                first_fields = list(rows_mode_n[0].keys())

    if first_fields is None:
        raise RuntimeError("No rows produced by evaluation.")

    raw_fields = list(first_fields)
    if "method" not in raw_fields:
        raw_fields.append("method")
    if "repeat_idx" not in raw_fields:
        raw_fields.append("repeat_idx")
    _write_csv(out_csv, raw_rows, raw_fields)

    cost_rows = _compute_cost_summary(raw_rows)
    _write_csv(
        out_cost_csv,
        cost_rows,
        [
            "method",
            "n",
            "runtime_mean_over_density_s",
            "runtime_std_over_density_s",
            "density_points",
            "avg_repeats_per_point",
            "min_repeats_per_point",
            "max_repeats_per_point",
        ],
    )

    quality_rows = _compute_quality_summary(raw_rows)
    _write_csv(
        out_quality_csv,
        quality_rows,
        [
            "method",
            "n",
            "delta_lambda2_mean_over_density",
            "delta_lambda2_std_over_density",
            "delta_lambda2_pct_mean_over_density",
            "delta_lambda2_pct_std_over_density",
            "win_pct_vs_rl_path_over_all_points",
            "tie_pct_vs_rl_path_over_all_points",
            "win_count",
            "tie_count",
            "total_points",
            "density_points",
            "avg_repeats_per_point",
        ],
    )

    meta = {
        "name": "rl_path_vs_backbone",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "n_values": n_values,
        "points_per_n_requested": points_per_n,
        "repeats": repeats,
        "seed_start": int(args.seed_start),
        "seeds": seeds,
        "backbone": backbone_args,
        "total_rows": len(raw_rows),
        "plan": plan,
        "outputs": {
            "out_csv": str(out_csv),
            "out_meta": str(out_meta),
            "out_cost_csv": str(out_cost_csv),
            "out_quality_csv": str(out_quality_csv),
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {out_csv}", flush=True)
    print(f"Wrote: {out_cost_csv}", flush=True)
    print(f"Wrote: {out_quality_csv}", flush=True)
    print(f"Wrote: {out_meta}", flush=True)


if __name__ == "__main__":
    main()

