"""Head-to-head comparison of the old model against the analytic model.

Both runs produce the same 34-column raw.csv (the 32 evaluator columns plus
``method`` / ``repeat_idx``), so they can be compared directly. The two runs do
not share a density grid, though: the old run used ``_nontrivial_m_points`` while
the analytic run swept ``rho = linspace(0, 1, 100)``. Exact ``m_target`` overlap
is only 20/94/50/48/44 points at n = 8/16/32/64/128.

So two views are emitted:

* ``head_to_head_matched.csv`` -- the strict view. Restricted to (n, m_target)
  pairs present in BOTH runs, paired on (method, n, m_target, repeat_idx). This
  is the defensible answer to "is the analytic model better on the same problem
  instance?".
* ``head_to_head_all_points.csv`` -- context. Each run aggregated over its own
  full grid, so the means differ in which densities they cover.

Both follow the repo's aggregation convention (mean over repeats per density
point, then mean/std over density points), matching summary_cost.csv /
summary_quality.csv.

Usage (from the project root):
    python compare_old_vs_analytic.py \
        --old-csv "../../Old Model/Algebraic-Connectivity-Optimization-CDC2026/results/rl/raw.csv" \
        --new-csv results/rl_analytic/raw.csv \
        --out-dir results/head_to_head_old_vs_analytic
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analytic_tractable_test_deepseek.run_path_vs_backbone_experiment import (
    _to_float,
    _to_int,
    _write_csv,
)

_DEFAULT_OLD_CSV = (
    "../../Old Model/Algebraic-Connectivity-Optimization-CDC2026/results/rl/raw.csv"
)

_MATCHED_FIELDS = [
    "method",
    "n",
    "matched_density_points",
    "matched_paired_points",
    "lambda2_old_mean_over_density",
    "lambda2_new_mean_over_density",
    "delta_lambda2_mean_over_density",
    "delta_lambda2_std_over_density",
    "delta_lambda2_pct_mean_over_density",
    "delta_lambda2_pct_std_over_density",
    "win_pct_new_vs_old",
    "tie_pct_new_vs_old",
    "loss_pct_new_vs_old",
    "win_count",
    "tie_count",
    "loss_count",
    "runtime_old_mean_over_density_s",
    "runtime_new_mean_over_density_s",
    "runtime_speedup_old_over_new",
]

_ALL_POINTS_FIELDS = [
    "method",
    "n",
    "density_points_old",
    "density_points_new",
    "lambda2_old_mean_over_density",
    "lambda2_new_mean_over_density",
    "runtime_old_mean_over_density_s",
    "runtime_old_std_over_density_s",
    "runtime_new_mean_over_density_s",
    "runtime_new_std_over_density_s",
    "runtime_speedup_old_over_new",
]

_METHOD_ORDER = ["RL+Backbone", "RL+Path"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-csv", type=str, default=_DEFAULT_OLD_CSV)
    p.add_argument("--new-csv", type=str, default="results/rl_analytic/raw.csv")
    p.add_argument("--out-dir", type=str, default="results/head_to_head_old_vs_analytic")
    p.add_argument("--out-matched-csv", type=str, default="head_to_head_matched.csv")
    p.add_argument("--out-all-points-csv", type=str, default="head_to_head_all_points.csv")
    p.add_argument("--out-meta", type=str, default="head_to_head.meta.json")
    p.add_argument(
        "--tie-tol",
        type=float,
        default=1e-10,
        help="Tie threshold on absolute terminal_lambda2 difference (new vs old).",
    )
    return p.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    # utf-8-sig, not utf-8: the old-model raw.csv carries a BOM, the new one does not.
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _index(rows: list[dict[str, str]]) -> dict[tuple[str, int, int, int], dict[str, float]]:
    """Collapse to one record per (method, n, m_target, repeat_idx).

    The analytic n=8 shards map ~4.5 rho values onto each m_target, so duplicate
    keys exist there; averaging within the key is the neutral way to collapse
    them (their terminal_lambda2 values are identical, only runtimes vary).
    """
    acc: dict[tuple[str, int, int, int], dict[str, list[float]]] = {}
    for r in rows:
        key = (
            str(r["method"]),
            _to_int(r["n"]),
            _to_int(r["m_target"]),
            _to_int(r["repeat_idx"]),
        )
        slot = acc.setdefault(key, {"lambda2": [], "runtime": []})
        slot["lambda2"].append(_to_float(r["terminal_lambda2"]))
        slot["runtime"].append(_to_float(r["runtime_total_sec"]))
    return {
        k: {
            "lambda2": float(np.mean(np.asarray(v["lambda2"], dtype=np.float64))),
            "runtime": float(np.mean(np.asarray(v["runtime"], dtype=np.float64))),
        }
        for k, v in acc.items()
    }


def _mean_over_density(per_point: dict[tuple[str, int, int], list[float]]) -> dict[tuple[str, int], list[float]]:
    """Mean over repeats per density point, then bucket by (method, n)."""
    out: dict[tuple[str, int], list[float]] = {}
    for (method, n, _m), vals in per_point.items():
        out.setdefault((method, n), []).append(
            float(np.mean(np.asarray(vals, dtype=np.float64)))
        )
    return out


def _compute_matched(
    old_idx: dict[tuple[str, int, int, int], dict[str, float]],
    new_idx: dict[tuple[str, int, int, int], dict[str, float]],
    tie_tol: float,
) -> list[dict[str, object]]:
    shared = sorted(set(old_idx) & set(new_idx))

    d_by_point: dict[tuple[str, int, int], list[float]] = {}
    d_pct_by_point: dict[tuple[str, int, int], list[float]] = {}
    l2_old_by_point: dict[tuple[str, int, int], list[float]] = {}
    l2_new_by_point: dict[tuple[str, int, int], list[float]] = {}
    rt_old_by_point: dict[tuple[str, int, int], list[float]] = {}
    rt_new_by_point: dict[tuple[str, int, int], list[float]] = {}
    win: dict[tuple[str, int], int] = {}
    tie: dict[tuple[str, int], int] = {}
    loss: dict[tuple[str, int], int] = {}
    total: dict[tuple[str, int], int] = {}

    for key in shared:
        method, n, m, _rep = key
        o = old_idx[key]
        v = new_idx[key]
        d = float(v["lambda2"] - o["lambda2"])
        d_pct = float(d / max(1e-15, abs(o["lambda2"])) * 100.0)

        k_point = (method, n, m)
        d_by_point.setdefault(k_point, []).append(d)
        d_pct_by_point.setdefault(k_point, []).append(d_pct)
        l2_old_by_point.setdefault(k_point, []).append(o["lambda2"])
        l2_new_by_point.setdefault(k_point, []).append(v["lambda2"])
        rt_old_by_point.setdefault(k_point, []).append(o["runtime"])
        rt_new_by_point.setdefault(k_point, []).append(v["runtime"])

        k_mn = (method, n)
        total[k_mn] = total.get(k_mn, 0) + 1
        if abs(d) <= float(tie_tol):
            tie[k_mn] = tie.get(k_mn, 0) + 1
        elif d > float(tie_tol):
            win[k_mn] = win.get(k_mn, 0) + 1
        else:
            loss[k_mn] = loss.get(k_mn, 0) + 1

    d_mn = _mean_over_density(d_by_point)
    d_pct_mn = _mean_over_density(d_pct_by_point)
    l2_old_mn = _mean_over_density(l2_old_by_point)
    l2_new_mn = _mean_over_density(l2_new_by_point)
    rt_old_mn = _mean_over_density(rt_old_by_point)
    rt_new_mn = _mean_over_density(rt_new_by_point)

    out: list[dict[str, object]] = []
    for method, n in sorted(d_mn, key=lambda x: (_METHOD_ORDER.index(x[0]) if x[0] in _METHOD_ORDER else 99, x[1])):
        k = (method, n)
        d_arr = np.asarray(d_mn[k], dtype=np.float64)
        p_arr = np.asarray(d_pct_mn[k], dtype=np.float64)
        rt_o = float(np.mean(np.asarray(rt_old_mn[k], dtype=np.float64)))
        rt_n = float(np.mean(np.asarray(rt_new_mn[k], dtype=np.float64)))
        tot = int(total.get(k, 0))
        w = int(win.get(k, 0))
        t = int(tie.get(k, 0))
        l = int(loss.get(k, 0))
        out.append(
            {
                "method": method,
                "n": int(n),
                "matched_density_points": int(d_arr.size),
                "matched_paired_points": tot,
                "lambda2_old_mean_over_density": float(np.mean(np.asarray(l2_old_mn[k], dtype=np.float64))),
                "lambda2_new_mean_over_density": float(np.mean(np.asarray(l2_new_mn[k], dtype=np.float64))),
                "delta_lambda2_mean_over_density": float(np.mean(d_arr)),
                "delta_lambda2_std_over_density": float(np.std(d_arr)),
                "delta_lambda2_pct_mean_over_density": float(np.mean(p_arr)),
                "delta_lambda2_pct_std_over_density": float(np.std(p_arr)),
                "win_pct_new_vs_old": float(w / max(1, tot) * 100.0),
                "tie_pct_new_vs_old": float(t / max(1, tot) * 100.0),
                "loss_pct_new_vs_old": float(l / max(1, tot) * 100.0),
                "win_count": w,
                "tie_count": t,
                "loss_count": l,
                "runtime_old_mean_over_density_s": rt_o,
                "runtime_new_mean_over_density_s": rt_n,
                "runtime_speedup_old_over_new": float(rt_o / max(1e-15, rt_n)),
            }
        )
    return out


def _compute_all_points(
    old_idx: dict[tuple[str, int, int, int], dict[str, float]],
    new_idx: dict[tuple[str, int, int, int], dict[str, float]],
) -> list[dict[str, object]]:
    def _fold(idx):
        l2: dict[tuple[str, int, int], list[float]] = {}
        rt: dict[tuple[str, int, int], list[float]] = {}
        for (method, n, m, _rep), v in idx.items():
            l2.setdefault((method, n, m), []).append(v["lambda2"])
            rt.setdefault((method, n, m), []).append(v["runtime"])
        return _mean_over_density(l2), _mean_over_density(rt)

    l2_o, rt_o = _fold(old_idx)
    l2_n, rt_n = _fold(new_idx)

    out: list[dict[str, object]] = []
    keys = sorted(
        set(l2_o) | set(l2_n),
        key=lambda x: (_METHOD_ORDER.index(x[0]) if x[0] in _METHOD_ORDER else 99, x[1]),
    )
    for k in keys:
        method, n = k
        rt_o_arr = np.asarray(rt_o.get(k, []), dtype=np.float64)
        rt_n_arr = np.asarray(rt_n.get(k, []), dtype=np.float64)
        rt_o_mean = float(np.mean(rt_o_arr)) if rt_o_arr.size else float("nan")
        rt_n_mean = float(np.mean(rt_n_arr)) if rt_n_arr.size else float("nan")
        out.append(
            {
                "method": method,
                "n": int(n),
                "density_points_old": int(len(l2_o.get(k, []))),
                "density_points_new": int(len(l2_n.get(k, []))),
                "lambda2_old_mean_over_density": float(np.mean(np.asarray(l2_o[k], dtype=np.float64))) if k in l2_o else float("nan"),
                "lambda2_new_mean_over_density": float(np.mean(np.asarray(l2_n[k], dtype=np.float64))) if k in l2_n else float("nan"),
                "runtime_old_mean_over_density_s": rt_o_mean,
                "runtime_old_std_over_density_s": float(np.std(rt_o_arr)) if rt_o_arr.size else float("nan"),
                "runtime_new_mean_over_density_s": rt_n_mean,
                "runtime_new_std_over_density_s": float(np.std(rt_n_arr)) if rt_n_arr.size else float("nan"),
                "runtime_speedup_old_over_new": float(rt_o_mean / max(1e-15, rt_n_mean)),
            }
        )
    return out


def _print_table(matched: list[dict[str, object]]) -> None:
    print()
    print("Matched-point head-to-head (analytic vs old, same n / m_target / seed)")
    print(
        f"{'method':<13}{'n':>5}{'pts':>6}{'lam2_old':>11}{'lam2_new':>11}"
        f"{'delta':>10}{'delta%':>9}{'win%':>7}{'tie%':>7}{'loss%':>7}"
        f"{'rt_old_s':>11}{'rt_new_s':>11}{'speedup':>9}"
    )
    for r in matched:
        print(
            f"{r['method']:<13}{r['n']:>5}{r['matched_density_points']:>6}"
            f"{r['lambda2_old_mean_over_density']:>11.4f}{r['lambda2_new_mean_over_density']:>11.4f}"
            f"{r['delta_lambda2_mean_over_density']:>+10.4f}{r['delta_lambda2_pct_mean_over_density']:>+9.1f}"
            f"{r['win_pct_new_vs_old']:>7.1f}{r['tie_pct_new_vs_old']:>7.1f}{r['loss_pct_new_vs_old']:>7.1f}"
            f"{r['runtime_old_mean_over_density_s']:>11.4f}{r['runtime_new_mean_over_density_s']:>11.4f}"
            f"{r['runtime_speedup_old_over_new']:>9.2f}"
        )
    print()


def main() -> None:
    args = parse_args()
    old_csv = Path(args.old_csv).resolve()
    new_csv = Path(args.new_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    old_rows = _read_rows(old_csv)
    new_rows = _read_rows(new_csv)
    if not old_rows or not new_rows:
        raise RuntimeError(f"Empty input: old={len(old_rows)} rows, new={len(new_rows)} rows.")
    print(f"[compare] old: {len(old_rows)} rows from {old_csv}")
    print(f"[compare] new: {len(new_rows)} rows from {new_csv}")

    old_idx = _index(old_rows)
    new_idx = _index(new_rows)
    shared = set(old_idx) & set(new_idx)
    print(f"[compare] matched (method, n, m_target, repeat_idx) keys: {len(shared)}")

    matched = _compute_matched(old_idx, new_idx, float(args.tie_tol))
    all_points = _compute_all_points(old_idx, new_idx)

    out_matched = out_dir / args.out_matched_csv
    out_all = out_dir / args.out_all_points_csv
    out_meta = out_dir / args.out_meta

    _write_csv(out_matched, matched, _MATCHED_FIELDS)
    _write_csv(out_all, all_points, _ALL_POINTS_FIELDS)

    def _provenance(rows: list[dict[str, str]]) -> dict[str, object]:
        return {
            "rows": len(rows),
            "checkpoint_global_env_steps": sorted({r["checkpoint_global_env_steps"] for r in rows}),
            "checkpoint_path": sorted({r["checkpoint_path"] for r in rows}),
            "config_path": sorted({r["config_path"] for r in rows}),
            "rl_variant": sorted({r["rl_variant"] for r in rows}),
            "seeds": sorted({r["seed"] for r in rows}),
        }

    meta = {
        "name": "head_to_head_old_vs_analytic",
        "old_csv": str(old_csv),
        "new_csv": str(new_csv),
        "tie_tol": float(args.tie_tol),
        "matched_keys": len(shared),
        "matched_density_points_by_method_n": {
            f"{r['method']}|n={r['n']}": r["matched_density_points"] for r in matched
        },
        "old": _provenance(old_rows),
        "new": _provenance(new_rows),
        "outputs": {
            "out_matched_csv": str(out_matched),
            "out_all_points_csv": str(out_all),
            "out_meta": str(out_meta),
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _print_table(matched)
    print(f"Wrote: {out_matched}")
    print(f"Wrote: {out_all}")
    print(f"Wrote: {out_meta}")


if __name__ == "__main__":
    main()
