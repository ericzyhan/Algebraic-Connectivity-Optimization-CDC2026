from __future__ import annotations

import argparse
import csv
from math import comb
from pathlib import Path
import random
import time
from typing import Callable

import numpy as np

from graph_design.backbones.cayley import (
    _InverseClosedSampler,
    _LiftDeleteBatchEvaluator,
    _build_group,
    _is_degree_feasible,
    _sample_generators,
    _split_cosets,
)
from graph_design.utils import normalized_density


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _evaluate_degree_mode(
    *,
    group,
    delete_count: int,
    degree: int,
    sample_count: int,
    feasible_fn: Callable[[int], bool],
    sample_fn: Callable[[int, random.Random], set[int] | None],
    expected_size: int,
    rng: random.Random,
    eval_backend: str,
    batch_size: int,
    n_target: int,
) -> dict[str, float | int]:
    start = time.perf_counter()
    if not feasible_fn(degree):
        return {
            "candidate_found": 0,
            "samples_used": 0,
            "best_lambda2": 0.0,
            "mean_lambda2": 0.0,
            "best_m_after": 0,
            "mean_m_after": 0.0,
            "best_density": 0.0,
            "mean_density": 0.0,
            "runtime_sec": float(time.perf_counter() - start),
        }

    arrays: list[np.ndarray] = []
    for _ in range(max(1, sample_count)):
        try:
            generators = sample_fn(degree, rng)
        except ValueError:
            continue
        if not generators:
            continue
        arr = np.fromiter(
            sorted(generators),
            dtype=np.int32,
            count=len(generators),
        )
        if arr.size == expected_size:
            arrays.append(arr)

    if not arrays:
        return {
            "candidate_found": 0,
            "samples_used": 0,
            "best_lambda2": 0.0,
            "mean_lambda2": 0.0,
            "best_m_after": 0,
            "mean_m_after": 0.0,
            "best_density": 0.0,
            "mean_density": 0.0,
            "runtime_sec": float(time.perf_counter() - start),
        }

    evaluator = _LiftDeleteBatchEvaluator(
        group=group,
        delete_count=delete_count,
        backend=eval_backend,
    )
    l2_list: list[np.ndarray] = []
    m_list: list[np.ndarray] = []
    for i in range(0, len(arrays), batch_size):
        chunk = arrays[i : i + batch_size]
        mat = np.stack(chunk, axis=0)
        l2_vals, edge_vals = evaluator.lambda2_edges_from_index_matrix(mat)
        l2_list.append(l2_vals)
        m_list.append(edge_vals)
    l2 = np.concatenate(l2_list, axis=0)
    m_after = np.concatenate(m_list, axis=0)
    if l2.size == 0:
        return {
            "candidate_found": 0,
            "samples_used": 0,
            "best_lambda2": 0.0,
            "mean_lambda2": 0.0,
            "best_m_after": 0,
            "mean_m_after": 0.0,
            "best_density": 0.0,
            "mean_density": 0.0,
            "runtime_sec": float(time.perf_counter() - start),
        }

    best_idx = int(np.argmax(l2))
    best_l2 = float(l2[best_idx])
    best_m = int(m_after[best_idx])
    mean_l2 = float(np.mean(l2))
    mean_m = float(np.mean(m_after))
    best_rho = float(normalized_density(n_target, best_m))
    mean_rho = float(np.mean([normalized_density(n_target, int(x)) for x in m_after]))

    return {
        "candidate_found": 1,
        "samples_used": int(l2.size),
        "best_lambda2": best_l2,
        "mean_lambda2": mean_l2,
        "best_m_after": best_m,
        "mean_m_after": mean_m,
        "best_density": best_rho,
        "mean_density": mean_rho,
        "runtime_sec": float(time.perf_counter() - start),
    }


def _plot_degree_curves(
    *,
    rows: list[dict[str, str | int | float]],
    out_dir: Path,
    show_markers: bool = True,
    scatter_only: bool = False,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    from matplotlib.lines import Line2D

    written: list[Path] = []
    label_fs = 24
    tick_fs = 20
    legend_fs = 16
    modes = ("coset", "random")
    degrees = sorted({int(r["degree"]) for r in rows})
    index_colors = {
        2: "#d62728",  # red
        3: "#2ca02c",  # green
        4: "#1f77b4",  # blue
    }
    mode_line = {
        "coset": {"color": "#d62728", "marker": "o"},  # red
        "random": {"color": "#1f77b4", "marker": "^"},  # blue
    }

    best_rows: list[dict[str, str | int | float]] = []
    for mode in modes:
        mode_rows = [r for r in rows if str(r["mode"]) == mode]
        for degree in degrees:
            cands = [
                r
                for r in mode_rows
                if int(r["degree"]) == degree and int(r["candidate_found"]) == 1
            ]
            if not cands:
                best_rows.append(
                    {
                        "mode": mode,
                        "degree": degree,
                        "candidate_found": 0,
                        "best_index": "",
                        "best_lambda2": 0.0,
                        "best_density": 0.0,
                        "runtime_sec": 0.0,
                    }
                )
                continue
            best = max(cands, key=lambda r: (float(r["best_lambda2"]), -float(r["runtime_sec"])))
            best_rows.append(
                {
                    "mode": mode,
                    "degree": degree,
                    "candidate_found": 1,
                    "best_index": int(best["index"]),
                    "best_lambda2": float(best["best_lambda2"]),
                    "best_density": float(best["best_density"]),
                    "runtime_sec": float(best["runtime_sec"]),
                }
            )

    # Plot only connected-range points: remove degree<=0 and negative connected-range density.
    best_rows = [
        r
        for r in best_rows
        if int(r["degree"]) > 0 and float(r["best_density"]) >= 0.0
    ]

    plt.figure(figsize=(9, 5.5))
    for mode in modes:
        sub = [
            r
            for r in best_rows
            if str(r["mode"]) == mode and int(r["candidate_found"]) == 1
        ]
        if not sub:
            continue
        sub = sorted(sub, key=lambda x: float(x["best_density"]))
        xs = [float(r["best_density"]) for r in sub]
        ys = [float(r["best_lambda2"]) for r in sub]
        if scatter_only:
            plt.scatter(xs, ys, s=18, color=mode_line[mode]["color"])
        else:
            plt.plot(xs, ys, linewidth=1.8, color=mode_line[mode]["color"])
        if (show_markers and not scatter_only) and mode == "coset":
            for idx in sorted({int(r["best_index"]) for r in sub}):
                idx_rows = [r for r in sub if int(r["best_index"]) == idx]
                plt.scatter(
                    [float(r["best_density"]) for r in idx_rows],
                    [float(r["best_lambda2"]) for r in idx_rows],
                    s=14,
                    marker=mode_line[mode]["marker"],
                    color=index_colors.get(idx, "#9467bd"),
                    zorder=3,
                )

    mode_handles = [
        Line2D(
            [0],
            [0],
            color=mode_line[m]["color"],
            linewidth=1.8,
            label=f"{m} best-over-indices",
        )
        for m in modes
    ]
    index_handles = (
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=index_colors[i],
                markersize=7,
                label=f"coset winner index={i}",
            )
            for i in sorted(index_colors)
        ]
        if show_markers
        else []
    )
    plt.xlabel("Density", fontsize=label_fs)
    plt.ylabel("Algebraic Connectivity", fontsize=label_fs)
    plt.xticks(fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)
    plt.grid(alpha=0.25)
    plt.legend(handles=mode_handles + index_handles, fontsize=legend_fs)
    plt.tight_layout()
    lambda2_name = (
        "lambda2_by_degree_best_over_indices_scatter.png"
        if scatter_only
        else "lambda2_by_degree_best_over_indices.png"
    )
    out_path = out_dir / lambda2_name
    plt.savefig(out_path, dpi=170)
    plt.close()
    written.append(out_path)

    plt.figure(figsize=(9, 5.5))
    for mode in modes:
        sub = [
            r
            for r in best_rows
            if str(r["mode"]) == mode and int(r["candidate_found"]) == 1
        ]
        if not sub:
            continue
        sub = sorted(sub, key=lambda x: float(x["best_density"]))
        xs = [float(r["best_density"]) for r in sub]
        ys = [float(r["runtime_sec"]) for r in sub]
        if scatter_only:
            plt.scatter(xs, ys, s=18, color=mode_line[mode]["color"])
        else:
            plt.plot(xs, ys, linewidth=1.8, color=mode_line[mode]["color"])
        if (show_markers and not scatter_only) and mode == "coset":
            for idx in sorted({int(r["best_index"]) for r in sub}):
                idx_rows = [r for r in sub if int(r["best_index"]) == idx]
                plt.scatter(
                    [float(r["best_density"]) for r in idx_rows],
                    [float(r["runtime_sec"]) for r in idx_rows],
                    s=14,
                    marker=mode_line[mode]["marker"],
                    color=index_colors.get(idx, "#9467bd"),
                    zorder=3,
                )
    plt.xlabel("Density", fontsize=label_fs)
    plt.ylabel("Runtime (sec)", fontsize=label_fs)
    plt.xticks(fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)
    plt.grid(alpha=0.25)
    plt.legend(handles=mode_handles + index_handles, fontsize=legend_fs)
    plt.tight_layout()
    runtime_name = (
        "runtime_by_degree_best_over_indices_scatter.png"
        if scatter_only
        else "runtime_by_degree_best_over_indices.png"
    )
    out_path = out_dir / runtime_name
    plt.savefig(out_path, dpi=170)
    plt.close()
    written.append(out_path)

    return written


def _degree_to_density(n: int, degree: int) -> float:
    m = int(round((n * degree) / 2.0))
    m = max(0, min(comb(n, 2), m))
    return float(normalized_density(n, m))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare coset vs random Cayley sampling over all generator-count points "
            "(degree sweep) for fixed n."
        )
    )
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--indices", type=str, default="2,3,4")
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help=(
            "Sample budget per degree point, per mode, per index "
            "(not split across degree points)."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "experiments" / "cayley_compare",
    )
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument(
        "--no-marker",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable markers on curves (line-only plots).",
    )
    parser.add_argument(
        "--scatter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use scatter points instead of line curves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indices = _parse_int_list(args.indices)
    n = int(args.n)
    seed = int(args.seed)
    samples_per_degree = int(args.samples)
    out_dir = args.out_dir.resolve()
    tag = args.tag or f"n{n}_s{samples_per_degree}_idx{'_'.join(str(i) for i in indices)}"

    rows: list[dict[str, str | int | float]] = []
    rng_root = random.Random(seed)
    for index in indices:
        lift_nodes = (index - (n % index)) % index
        group = _build_group(target_n=n, index=index, lift_nodes=lift_nodes)
        subgroup, nontrivial_union = _split_cosets(group, index)
        subgroup_no_identity = [g for g in subgroup if g != 0]
        coset_nontrivial_sampler = _InverseClosedSampler(group, nontrivial_union)
        coset_subgroup_sampler = _InverseClosedSampler(group, subgroup_no_identity)
        random_full_sampler = _InverseClosedSampler(group, [g for g in range(group.order) if g != 0])

        degrees = list(range(1, group.order))
        coset_feasible = []
        random_feasible = []
        for d in degrees:
            c_ok = _is_degree_feasible(
                desired_degree=d,
                nontrivial_size=len(nontrivial_union),
                nontrivial_sampler=coset_nontrivial_sampler,
                subgroup_sampler=coset_subgroup_sampler,
            )
            if c_ok:
                coset_feasible.append(d)
            if random_full_sampler.feasible(d):
                random_feasible.append(d)

        coset_feasible_set = set(coset_feasible)
        random_feasible_set = set(random_feasible)

        for mode in ("coset", "random"):
            for degree in degrees:
                local_seed = rng_root.randint(0, 2**31 - 1)
                rng = random.Random(local_seed)
                if mode == "coset":
                    def feasible_fn(k: int) -> bool:
                        return _is_degree_feasible(
                            desired_degree=k,
                            nontrivial_size=len(nontrivial_union),
                            nontrivial_sampler=coset_nontrivial_sampler,
                            subgroup_sampler=coset_subgroup_sampler,
                        )

                    def sample_fn(k: int, local_rng: random.Random) -> set[int] | None:
                        return _sample_generators(
                            nontrivial_union=nontrivial_union,
                            subgroup_no_identity=subgroup_no_identity,
                            desired_degree=k,
                            rng=local_rng,
                            nontrivial_sampler=coset_nontrivial_sampler,
                            subgroup_sampler=coset_subgroup_sampler,
                        )

                    sample_count = int(samples_per_degree if degree in coset_feasible_set else 0)
                else:
                    def feasible_fn(k: int) -> bool:
                        return random_full_sampler.feasible(k)

                    def sample_fn(k: int, local_rng: random.Random) -> set[int] | None:
                        return random_full_sampler.sample(k, local_rng)

                    sample_count = int(samples_per_degree if degree in random_feasible_set else 0)

                stats = _evaluate_degree_mode(
                    group=group,
                    delete_count=lift_nodes,
                    degree=degree,
                    sample_count=sample_count,
                    feasible_fn=feasible_fn,
                    sample_fn=sample_fn,
                    expected_size=degree,
                    rng=rng,
                    eval_backend=args.backend,
                    batch_size=args.eval_batch_size,
                    n_target=n,
                )
                rows.append(
                    {
                        "n": n,
                        "index": index,
                        "mode": mode,
                        "degree": degree,
                        "candidate_found": int(stats["candidate_found"]),
                        "samples_used": int(stats["samples_used"]),
                        "best_lambda2": float(stats["best_lambda2"]),
                        "mean_lambda2": float(stats["mean_lambda2"]),
                        "best_m_after": int(stats["best_m_after"]),
                        "mean_m_after": float(stats["mean_m_after"]),
                        "best_density": float(stats["best_density"]),
                        "mean_density": float(stats["mean_density"]),
                        "runtime_sec": float(stats["runtime_sec"]),
                    }
                )

    points_csv = out_dir / f"cayley_degree_compare_points_{tag}.csv"
    points_csv.parent.mkdir(parents=True, exist_ok=True)
    with points_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n",
                "index",
                "mode",
                "degree",
                "candidate_found",
                "samples_used",
                "best_lambda2",
                "mean_lambda2",
                "best_m_after",
                "mean_m_after",
                "best_density",
                "mean_density",
                "runtime_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, str | int | float]] = []
    for index in sorted(set(indices)):
        sub = [r for r in rows if int(r["index"]) == index]
        by_mode_degree = {
            mode: {int(r["degree"]): r for r in sub if str(r["mode"]) == mode}
            for mode in ("coset", "random")
        }
        common = sorted(set(by_mode_degree["coset"]) & set(by_mode_degree["random"]))
        valid = [
            (by_mode_degree["coset"][d], by_mode_degree["random"][d])
            for d in common
            if int(by_mode_degree["coset"][d]["candidate_found"]) == 1
            and int(by_mode_degree["random"][d]["candidate_found"]) == 1
        ]
        if valid:
            coset_mean = float(np.mean([float(c["best_lambda2"]) for c, _ in valid]))
            random_mean = float(np.mean([float(r["best_lambda2"]) for _, r in valid]))
            coset_rt = float(np.mean([float(c["runtime_sec"]) for c, _ in valid]))
            random_rt = float(np.mean([float(r["runtime_sec"]) for _, r in valid]))
            coset_wins = sum(1 for c, r in valid if float(c["best_lambda2"]) > float(r["best_lambda2"]))
            random_wins = sum(1 for c, r in valid if float(r["best_lambda2"]) > float(c["best_lambda2"]))
            ties = len(valid) - coset_wins - random_wins
        else:
            coset_mean = 0.0
            random_mean = 0.0
            coset_rt = 0.0
            random_rt = 0.0
            coset_wins = 0
            random_wins = 0
            ties = 0
        summary_rows.append(
            {
                "n": n,
                "index": index,
                "degree_points": len(common),
                "paired_found": len(valid),
                "coset_mean_best_lambda2": coset_mean,
                "random_mean_best_lambda2": random_mean,
                "coset_mean_runtime_sec": coset_rt,
                "random_mean_runtime_sec": random_rt,
                "lambda2_coset_wins": coset_wins,
                "lambda2_random_wins": random_wins,
                "lambda2_ties": ties,
            }
        )

    summary_csv = out_dir / f"cayley_degree_compare_summary_{tag}.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n",
                "index",
                "degree_points",
                "paired_found",
                "coset_mean_best_lambda2",
                "random_mean_best_lambda2",
                "coset_mean_runtime_sec",
                "random_mean_runtime_sec",
                "lambda2_coset_wins",
                "lambda2_random_wins",
                "lambda2_ties",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    best_by_mode_degree: dict[str, dict[int, dict[str, str | int | float]]] = {
        "coset": {},
        "random": {},
    }
    for mode in ("coset", "random"):
        sub_mode = [r for r in rows if str(r["mode"]) == mode and int(r["candidate_found"]) == 1]
        for degree in sorted({int(r["degree"]) for r in sub_mode}):
            cands = [r for r in sub_mode if int(r["degree"]) == degree]
            if not cands:
                continue
            best_by_mode_degree[mode][degree] = max(
                cands,
                key=lambda r: (float(r["best_lambda2"]), -float(r["runtime_sec"])),
            )

    common_best_degrees = sorted(
        set(best_by_mode_degree["coset"].keys()) & set(best_by_mode_degree["random"].keys())
    )
    if common_best_degrees:
        pairs = [
            (best_by_mode_degree["coset"][d], best_by_mode_degree["random"][d])
            for d in common_best_degrees
        ]
        coset_mean_l2 = float(np.mean([float(c["best_lambda2"]) for c, _ in pairs]))
        random_mean_l2 = float(np.mean([float(r["best_lambda2"]) for _, r in pairs]))
        coset_mean_rt = float(np.mean([float(c["runtime_sec"]) for c, _ in pairs]))
        random_mean_rt = float(np.mean([float(r["runtime_sec"]) for _, r in pairs]))
        coset_wins = sum(1 for c, r in pairs if float(c["best_lambda2"]) > float(r["best_lambda2"]))
        random_wins = sum(1 for c, r in pairs if float(r["best_lambda2"]) > float(c["best_lambda2"]))
        ties = len(pairs) - coset_wins - random_wins
    else:
        coset_mean_l2 = 0.0
        random_mean_l2 = 0.0
        coset_mean_rt = 0.0
        random_mean_rt = 0.0
        coset_wins = 0
        random_wins = 0
        ties = 0

    combined_summary_csv = out_dir / f"cayley_degree_compare_summary_best_over_indices_{tag}.csv"
    with combined_summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n",
                "indices",
                "degree_points",
                "paired_found",
                "coset_mean_best_lambda2",
                "random_mean_best_lambda2",
                "coset_mean_runtime_sec",
                "random_mean_runtime_sec",
                "lambda2_coset_wins",
                "lambda2_random_wins",
                "lambda2_ties",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "n": n,
                "indices": ",".join(str(i) for i in indices),
                "degree_points": len(common_best_degrees),
                "paired_found": len(common_best_degrees),
                "coset_mean_best_lambda2": coset_mean_l2,
                "random_mean_best_lambda2": random_mean_l2,
                "coset_mean_runtime_sec": coset_mean_rt,
                "random_mean_runtime_sec": random_mean_rt,
                "lambda2_coset_wins": coset_wins,
                "lambda2_random_wins": random_wins,
                "lambda2_ties": ties,
            }
        )

    direct_rows: list[dict[str, str | int | float]] = []
    for degree in common_best_degrees:
        cos_row = best_by_mode_degree["coset"][degree]
        rnd_row = best_by_mode_degree["random"][degree]
        rho_deg = _degree_to_density(n=n, degree=degree)
        if rho_deg < 0.0:
            continue
        c_l2 = float(cos_row["best_lambda2"])
        r_l2 = float(rnd_row["best_lambda2"])
        direct_rows.append(
            {
                "n": n,
                "degree": int(degree),
                "density_from_degree": rho_deg,
                "coset_lambda2": c_l2,
                "random_lambda2": r_l2,
                "delta_coset_minus_random": c_l2 - r_l2,
                "coset_runtime_sec": float(cos_row["runtime_sec"]),
                "random_runtime_sec": float(rnd_row["runtime_sec"]),
            }
        )

    direct_csv = out_dir / f"cayley_degree_direct_compare_{tag}.csv"
    with direct_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n",
                "degree",
                "density_from_degree",
                "coset_lambda2",
                "random_lambda2",
                "delta_coset_minus_random",
                "coset_runtime_sec",
                "random_runtime_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(direct_rows)

    plot_dir = out_dir / f"plots_degree_{tag}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = _plot_degree_curves(
        rows=rows,
        out_dir=plot_dir,
        show_markers=not bool(args.no_marker),
        scatter_only=bool(args.scatter),
    )

    if direct_rows:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            pass
        else:
            dr = sorted(direct_rows, key=lambda r: float(r["density_from_degree"]))
            xs = [float(r["density_from_degree"]) for r in dr]
            cys = [float(r["coset_lambda2"]) for r in dr]
            rys = [float(r["random_lambda2"]) for r in dr]
            dys = [float(r["delta_coset_minus_random"]) for r in dr]

            plt.figure(figsize=(9, 5.5))
            if bool(args.scatter):
                plt.scatter(xs, cys, s=18, color="#d62728", label="coset")
                plt.scatter(xs, rys, s=18, color="#1f77b4", label="random")
            else:
                plt.plot(xs, cys, linewidth=1.8, color="#d62728", label="coset")
                plt.plot(xs, rys, linewidth=1.8, color="#1f77b4", label="random")
            plt.xlabel("Density", fontsize=24)
            plt.ylabel("Algebraic Connectivity", fontsize=24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.grid(alpha=0.25)
            plt.legend(fontsize=16)
            plt.tight_layout()
            p1_name = (
                "lambda2_direct_compare_by_degree_density_scatter.png"
                if bool(args.scatter)
                else "lambda2_direct_compare_by_degree_density.png"
            )
            p1 = plot_dir / p1_name
            plt.savefig(p1, dpi=170)
            plt.close()
            plot_paths.append(p1)

            plt.figure(figsize=(9, 4.8))
            if bool(args.scatter):
                plt.scatter(xs, dys, s=18, color="#2ca02c", label="coset - random")
            else:
                plt.plot(xs, dys, linewidth=1.8, color="#2ca02c", label="coset - random")
            plt.axhline(0.0, color="black", linewidth=1.0)
            plt.xlabel("Density", fontsize=24)
            plt.ylabel("Delta Algebraic Connectivity", fontsize=24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.grid(alpha=0.25)
            plt.legend(fontsize=16)
            plt.tight_layout()
            p2_name = (
                "lambda2_direct_delta_by_degree_density_scatter.png"
                if bool(args.scatter)
                else "lambda2_direct_delta_by_degree_density.png"
            )
            p2 = plot_dir / p2_name
            plt.savefig(p2, dpi=170)
            plt.close()
            plot_paths.append(p2)

    print(f"[compare_cayley_generators_by_degree] wrote points: {points_csv}")
    print(f"[compare_cayley_generators_by_degree] wrote summary: {summary_csv}")
    print(f"[compare_cayley_generators_by_degree] wrote combined summary: {combined_summary_csv}")
    print(f"[compare_cayley_generators_by_degree] wrote direct compare: {direct_csv}")
    for path in plot_paths:
        print(f"[compare_cayley_generators_by_degree] wrote plot: {path}")
    print(
        "[compare_cayley_generators_by_degree] "
        f"n={n} indices={indices} rows={len(rows)} summary_rows={len(summary_rows)}"
    )


if __name__ == "__main__":
    main()
