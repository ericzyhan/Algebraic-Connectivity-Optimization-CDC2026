from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import random
import time
from typing import Iterable

from graph_design.backbones.cayley import (
    CayleyBackboneGenerator,
    RandomCayleyBackboneGenerator,
)
from graph_design.config import DesignConfig
from graph_design.experiments import build_problem_grid
from graph_design.types import DesignProblem


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _linspace(start: float, end: float, count: int) -> list[float]:
    if count <= 0:
        raise ValueError("density-count must be positive.")
    if count == 1:
        return [float(start)]
    step = (end - start) / float(count - 1)
    return [float(start + i * step) for i in range(count)]


def _run_mode_once(
    *,
    mode: str,
    generator,
    problem: DesignProblem,
    cfg: DesignConfig,
) -> dict[str, str | int | float]:
    seed = int(problem.seed if problem.seed is not None else 0)
    start = time.perf_counter()
    candidate = generator.generate(problem, cfg, random.Random(seed))
    elapsed = float(time.perf_counter() - start)

    base: dict[str, str | int | float] = {
        "mode": mode,
        "n": int(problem.n),
        "m_target": int(problem.m),
        "rho": float(problem.density),
        "seed": seed,
        "runtime_sec": elapsed,
        "candidate_found": 0,
        "backbone_edges": "",
        "backbone_lambda2": "",
        "selected_index": "",
        "group_kind": "",
        "generation_mode": mode,
        "k_max_selected": "",
        "k_feasible_cap": "",
    }
    if candidate is None:
        return base

    meta = candidate.metadata or {}
    base.update(
        {
            "candidate_found": 1,
            "backbone_edges": int(candidate.graph.number_of_edges()),
            "backbone_lambda2": float(candidate.backbone_lambda2),
            "selected_index": str(meta.get("selected_index", "")),
            "group_kind": str(meta.get("group_kind", "")),
            "generation_mode": str(meta.get("generation_mode", mode)),
            "k_max_selected": str(meta.get("k_max_selected", "")),
            "k_feasible_cap": str(meta.get("k_feasible_cap", "")),
        }
    )
    return base


def _pair_key(row: dict[str, str | int | float]) -> tuple[int, int, int]:
    return int(row["n"]), int(row["m_target"]), int(row["seed"])


def _build_summary(
    rows: list[dict[str, str | int | float]],
) -> list[dict[str, str | int | float]]:
    coset = {_pair_key(r): r for r in rows if str(r["mode"]) == "coset"}
    random_mode = {_pair_key(r): r for r in rows if str(r["mode"]) == "random"}
    keys = sorted(set(coset.keys()) | set(random_mode.keys()))

    by_n: dict[int, list[tuple[dict[str, str | int | float] | None, dict[str, str | int | float] | None]]] = {}
    for key in keys:
        n = key[0]
        by_n.setdefault(n, []).append((coset.get(key), random_mode.get(key)))

    summary: list[dict[str, str | int | float]] = []
    for n in sorted(by_n.keys()):
        pairs = by_n[n]
        points = len(pairs)
        coset_missing = sum(1 for c, _ in pairs if not c or int(c["candidate_found"]) == 0)
        random_missing = sum(1 for _, r in pairs if not r or int(r["candidate_found"]) == 0)
        valid = [
            (c, r)
            for c, r in pairs
            if c is not None
            and r is not None
            and int(c["candidate_found"]) == 1
            and int(r["candidate_found"]) == 1
        ]
        paired_found = len(valid)

        if paired_found > 0:
            coset_mean_l2 = sum(float(c["backbone_lambda2"]) for c, _ in valid) / paired_found
            random_mean_l2 = sum(float(r["backbone_lambda2"]) for _, r in valid) / paired_found
            coset_mean_rt = sum(float(c["runtime_sec"]) for c, _ in valid) / paired_found
            random_mean_rt = sum(float(r["runtime_sec"]) for _, r in valid) / paired_found
            l2_coset_wins = sum(
                1 for c, r in valid if float(c["backbone_lambda2"]) > float(r["backbone_lambda2"])
            )
            l2_random_wins = sum(
                1 for c, r in valid if float(r["backbone_lambda2"]) > float(c["backbone_lambda2"])
            )
            l2_ties = paired_found - l2_coset_wins - l2_random_wins
            rt_coset_faster = sum(
                1 for c, r in valid if float(c["runtime_sec"]) < float(r["runtime_sec"])
            )
            rt_random_faster = sum(
                1 for c, r in valid if float(r["runtime_sec"]) < float(c["runtime_sec"])
            )
            rt_ties = paired_found - rt_coset_faster - rt_random_faster
        else:
            coset_mean_l2 = ""
            random_mean_l2 = ""
            coset_mean_rt = ""
            random_mean_rt = ""
            l2_coset_wins = 0
            l2_random_wins = 0
            l2_ties = 0
            rt_coset_faster = 0
            rt_random_faster = 0
            rt_ties = 0

        summary.append(
            {
                "n": n,
                "points": points,
                "paired_found": paired_found,
                "coset_no_candidate": coset_missing,
                "random_no_candidate": random_missing,
                "coset_mean_lambda2": coset_mean_l2,
                "random_mean_lambda2": random_mean_l2,
                "coset_mean_runtime_sec": coset_mean_rt,
                "random_mean_runtime_sec": random_mean_rt,
                "lambda2_coset_wins": l2_coset_wins,
                "lambda2_random_wins": l2_random_wins,
                "lambda2_ties": l2_ties,
                "runtime_coset_faster": rt_coset_faster,
                "runtime_random_faster": rt_random_faster,
                "runtime_ties": rt_ties,
            }
        )
    return summary


def _write_csv(
    *,
    path: Path,
    rows: list[dict[str, str | int | float]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_results(
    *,
    rows: list[dict[str, str | int | float]],
    out_dir: Path,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    written: list[Path] = []
    ns = sorted({int(r["n"]) for r in rows})
    for n in ns:
        sub = [r for r in rows if int(r["n"]) == n and int(r["candidate_found"]) == 1]
        if not sub:
            continue

        by_mode_rho_l2: dict[str, dict[float, list[float]]] = {"coset": {}, "random": {}}
        by_mode_rho_rt: dict[str, dict[float, list[float]]] = {"coset": {}, "random": {}}
        for row in sub:
            mode = str(row["mode"])
            rho = float(row["rho"])
            by_mode_rho_l2[mode].setdefault(rho, []).append(float(row["backbone_lambda2"]))
            by_mode_rho_rt[mode].setdefault(rho, []).append(float(row["runtime_sec"]))

        plt.figure(figsize=(8, 5))
        for mode, color in (("coset", "#1f77b4"), ("random", "#d62728")):
            items = sorted(by_mode_rho_l2[mode].items())
            if not items:
                continue
            xs = [k for k, _ in items]
            ys = [sum(v) / len(v) for _, v in items]
            plt.plot(xs, ys, label=mode, color=color, linewidth=1.8)
        plt.xlabel("Density")
        plt.ylabel("Backbone lambda2")
        plt.title(f"Cayley mode comparison (n={n})")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        out_l2 = out_dir / f"lambda2_compare_n{n}.png"
        plt.savefig(out_l2, dpi=160)
        plt.close()
        written.append(out_l2)

        plt.figure(figsize=(8, 5))
        for mode, color in (("coset", "#1f77b4"), ("random", "#d62728")):
            items = sorted(by_mode_rho_rt[mode].items())
            if not items:
                continue
            xs = [k for k, _ in items]
            ys = [sum(v) / len(v) for _, v in items]
            plt.plot(xs, ys, label=mode, color=color, linewidth=1.8)
        plt.xlabel("Density")
        plt.ylabel("Runtime (sec)")
        plt.title(f"Cayley mode runtime comparison (n={n})")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        out_rt = out_dir / f"runtime_compare_n{n}.png"
        plt.savefig(out_rt, dpi=160)
        plt.close()
        written.append(out_rt)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare coset-based vs random Cayley backbone generators."
    )
    parser.add_argument("--n-values", type=str, required=True, help="Comma list, e.g. 32,64")
    parser.add_argument("--densities", type=str, default=None, help="Comma list of density values")
    parser.add_argument("--density-count", type=int, default=100)
    parser.add_argument("--density-min", type=float, default=0.0)
    parser.add_argument("--density-max", type=float, default=1.0)
    parser.add_argument("--seeds", type=str, default="1234", help="Comma list of seeds")
    parser.add_argument("--cayley-samples", type=int, default=500)
    parser.add_argument("--cayley-degree-slack", type=int, default=0)
    parser.add_argument("--cayley-multi-index-overlap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cayley-index2-overlap-high", type=float, default=0.6)
    parser.add_argument("--cayley-eval-batch-size", type=int, default=128)
    parser.add_argument("--cayley-spectral-backend", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--force-connected-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "experiments" / "cayley_compare",
        help="Output directory for compare CSVs and plots.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag in output file names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_values = _parse_int_list(args.n_values)
    seeds = _parse_int_list(args.seeds)
    if args.densities is not None:
        densities = _parse_float_list(args.densities)
    else:
        densities = _linspace(args.density_min, args.density_max, args.density_count)

    base_cfg = DesignConfig(
        routing_mode="always_cayley",
        cayley_samples=int(args.cayley_samples),
        cayley_degree_slack=int(args.cayley_degree_slack),
        cayley_multi_index_overlap=bool(args.cayley_multi_index_overlap),
        cayley_index2_overlap_high=float(args.cayley_index2_overlap_high),
        cayley_eval_batch_size=int(args.cayley_eval_batch_size),
        cayley_spectral_backend=str(args.cayley_spectral_backend),
        cayley_generation_mode="coset",
        force_connected_backbone=bool(args.force_connected_backbone),
    )

    problems = build_problem_grid(n_values=n_values, densities=densities, seeds=seeds)
    coset_generator = CayleyBackboneGenerator(generation_mode="coset")
    random_generator = RandomCayleyBackboneGenerator()
    rows: list[dict[str, str | int | float]] = []

    for problem in problems:
        coset_cfg = replace(base_cfg, cayley_generation_mode="coset")
        random_cfg = replace(base_cfg, cayley_generation_mode="random")
        rows.append(
            _run_mode_once(
                mode="coset",
                generator=coset_generator,
                problem=problem,
                cfg=coset_cfg,
            )
        )
        rows.append(
            _run_mode_once(
                mode="random",
                generator=random_generator,
                problem=problem,
                cfg=random_cfg,
            )
        )

    summary_rows = _build_summary(rows)
    out_dir = args.out_dir.resolve()
    tag = args.tag or f"n{'_'.join(str(v) for v in n_values)}_d{len(densities)}_s{'_'.join(str(v) for v in seeds)}"
    points_csv = out_dir / f"cayley_mode_compare_points_{tag}.csv"
    summary_csv = out_dir / f"cayley_mode_compare_summary_{tag}.csv"
    plot_dir = out_dir / f"plots_{tag}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        path=points_csv,
        rows=rows,
        fieldnames=[
            "mode",
            "n",
            "m_target",
            "rho",
            "seed",
            "runtime_sec",
            "candidate_found",
            "backbone_edges",
            "backbone_lambda2",
            "selected_index",
            "group_kind",
            "generation_mode",
            "k_max_selected",
            "k_feasible_cap",
        ],
    )
    _write_csv(
        path=summary_csv,
        rows=summary_rows,
        fieldnames=[
            "n",
            "points",
            "paired_found",
            "coset_no_candidate",
            "random_no_candidate",
            "coset_mean_lambda2",
            "random_mean_lambda2",
            "coset_mean_runtime_sec",
            "random_mean_runtime_sec",
            "lambda2_coset_wins",
            "lambda2_random_wins",
            "lambda2_ties",
            "runtime_coset_faster",
            "runtime_random_faster",
            "runtime_ties",
        ],
    )
    plot_paths = _plot_results(rows=rows, out_dir=plot_dir)

    print(f"[compare_cayley_generators] wrote points: {points_csv}")
    print(f"[compare_cayley_generators] wrote summary: {summary_csv}")
    for path in plot_paths:
        print(f"[compare_cayley_generators] wrote plot: {path}")
    print(
        "[compare_cayley_generators] "
        f"problems={len(problems)} rows={len(rows)} summary_rows={len(summary_rows)}"
    )


if __name__ == "__main__":
    main()
