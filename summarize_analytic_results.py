"""Summarize analytic path-vs-backbone evaluation shards into the same outputs
the old-model README produced (raw.csv / summary_cost.csv / summary_quality.csv
/ raw.meta.json).

The old-model README's RL section ran ``run_path_vs_backbone_experiment``, whose
main() both evaluates AND summarizes. The analytic package's copy of that driver
hard-codes ``-m rl_train.evaluate``, which would evaluate the wrong package in
this repo (there is also an ``rl_train/``). So the evaluation is normally driven
by direct ``analytic_tractable_test_deepseek.evaluate`` calls (one per
n, init_mode -> results/rl_analytic/per_n_mode/eval_<mode>_n<N>.csv), and this
script performs the summarization half: it reads those shards and reproduces the
driver's raw.csv / summary_cost.csv / summary_quality.csv / raw.meta.json.

Usage (from the project root):
    python summarize_analytic_results.py \
        --in-dir results/rl_analytic/per_n_mode \
        --out-dir results/rl_analytic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analytic_tractable_test_deepseek.run_path_vs_backbone_experiment import (
    _compute_cost_summary,
    _compute_quality_summary,
    _read_csv_rows,
    _to_int,
    _write_csv,
)

_COST_FIELDS = [
    "method",
    "n",
    "runtime_mean_over_density_s",
    "runtime_std_over_density_s",
    "density_points",
    "avg_repeats_per_point",
    "min_repeats_per_point",
    "max_repeats_per_point",
]

_QUALITY_FIELDS = [
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
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=str, default="results/rl_analytic/per_n_mode")
    p.add_argument("--out-dir", type=str, default="results/rl_analytic")
    p.add_argument("--out-csv", type=str, default="raw.csv")
    p.add_argument("--out-meta", type=str, default="raw.meta.json")
    p.add_argument("--out-cost-csv", type=str, default="summary_cost.csv")
    p.add_argument("--out-quality-csv", type=str, default="summary_quality.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []
    first_fields: list[str] | None = None

    # Mirror run_path_vs_backbone_experiment.main(): shards named eval_{mode}_n{N}.csv
    for shard in sorted(in_dir.glob("eval_*.csv")):
        rows = _read_csv_rows(shard)
        if not rows:
            print(f"[summarize] {shard.name}: 0 rows, skipping")
            continue
        if shard.name.startswith("eval_backbone_"):
            method = "RL+Backbone"
        elif shard.name.startswith("eval_path_"):
            method = "RL+Path"
        else:
            print(f"[summarize] {shard.name}: unrecognized shard name, skipping")
            continue
        for row in rows:
            enriched: dict[str, object] = dict(row)
            enriched["method"] = method
            enriched["repeat_idx"] = int(_to_int(row["seed"]))
            raw_rows.append(enriched)
        if first_fields is None:
            first_fields = list(rows[0].keys())
        print(f"[summarize] {shard.name}: {len(rows)} rows -> method={method}")

    if first_fields is None or not raw_rows:
        raise RuntimeError(f"No shard rows found in {in_dir}.")

    raw_fields = list(first_fields)
    for col in ("method", "repeat_idx"):
        if col not in raw_fields:
            raw_fields.append(col)

    out_csv = out_dir / args.out_csv
    out_cost_csv = out_dir / args.out_cost_csv
    out_quality_csv = out_dir / args.out_quality_csv
    out_meta = out_dir / args.out_meta

    _write_csv(out_csv, raw_rows, raw_fields)

    cost_rows = _compute_cost_summary(raw_rows)
    _write_csv(out_cost_csv, cost_rows, _COST_FIELDS)

    quality_rows = _compute_quality_summary(raw_rows)
    _write_csv(out_quality_csv, quality_rows, _QUALITY_FIELDS)

    n_values = sorted({int(str(r["n"])) for r in raw_rows})
    meta = {
        "name": "rl_path_vs_backbone",
        "in_dir": str(in_dir),
        "n_values": n_values,
        "total_rows": len(raw_rows),
        "shards": sorted(str(p.name) for p in in_dir.glob("eval_*.csv")),
        "outputs": {
            "out_csv": str(out_csv),
            "out_meta": str(out_meta),
            "out_cost_csv": str(out_cost_csv),
            "out_quality_csv": str(out_quality_csv),
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_cost_csv}")
    print(f"Wrote: {out_quality_csv}")
    print(f"Wrote: {out_meta}")


if __name__ == "__main__":
    main()
