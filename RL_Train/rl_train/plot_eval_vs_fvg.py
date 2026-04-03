from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
EXPERIMENTS_RL_ROOT = PROJECT_ROOT / "experiments" / "rl"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_design.baselines.registry import create_baseline
from graph_design.config import DesignConfig
from graph_design.types import DesignProblem


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _aggregate(rows: list[dict], metric_key: str) -> dict[tuple[str, int], list[tuple[float, float]]]:
    grouped: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for row in rows:
        solver = row["solver_name"]
        n = int(row["n"])
        rho = float(row["rho_target"])
        grouped[(solver, n, rho)].append(float(row[metric_key]))

    out: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for (solver, n, rho), values in grouped.items():
        out[(solver, n)].append((rho, sum(values) / len(values)))
    for key in out:
        out[key] = sorted(out[key], key=lambda x: x[0])
    return out


def _solver_order(name: str) -> tuple[int, str]:
    if name == "fvg":
        return (0, name)
    if name == "best":
        return (1, name)
    if name == "last":
        return (2, name)
    return (10, name)


def _solver_style(name: str) -> dict:
    if name == "fvg":
        return {"color": "#2ca02c", "linewidth": 1.8}
    if name == "best":
        return {"color": "#1f77b4", "linewidth": 1.8}
    if name == "last":
        return {"color": "#ff7f0e", "linewidth": 1.8}
    return {"color": None, "linewidth": 1.6}


def _plot_per_n(
    data: dict[tuple[str, int], list[tuple[float, float]]],
    solver_names: list[str],
    n_values: list[int],
    y_label: str,
    y_scale: str,
    out_prefix: Path,
) -> list[Path]:
    saved: list[Path] = []
    for n in n_values:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        has_any = False
        for solver in solver_names:
            points = data.get((solver, n), [])
            if not points:
                continue
            xs = [x for x, _ in points]
            ys = [y for _, y in points]
            style = _solver_style(solver)
            ax.plot(
                xs,
                ys,
                label=solver,
                color=style["color"],
                linewidth=style["linewidth"],
            )
            has_any = True

        if y_scale == "exponential":
            y_values = [y for solver in solver_names for _, y in data.get((solver, n), [])]
            if any(v <= 0 for v in y_values):
                raise ValueError(
                    f"Cannot use exponential runtime scale for n={n}: non-positive values found."
                )
            ax.set_yscale("log")

        ax.set_title(f"n={n}")
        ax.set_xlabel("Density (rho)")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)
        if has_any:
            ax.legend()
        fig.tight_layout()

        out_path = out_prefix.with_name(f"{out_prefix.stem}_n{n}{out_prefix.suffix}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        saved.append(out_path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot RL eval CSV and compare against FVG baseline on same targets."
    )
    parser.add_argument("--eval-csv", required=True, help="Path to rl_train.evaluate CSV output")
    parser.add_argument(
        "--checkpoint-labels",
        default=None,
        help="Comma list of checkpoint labels to plot (e.g. best,last). Default: all labels in CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for combined CSV and figures. Default: <eval_csv_stem>_vs_fvg",
    )
    parser.add_argument(
        "--runtime-y-scale",
        default="linear",
        choices=["linear", "exponential"],
        help="Y-axis scale for runtime plot. 'exponential' uses log-scale.",
    )
    parser.add_argument(
        "--runtime-source",
        default="auto",
        choices=["auto", "runtime_sec", "runtime_total_sec"],
        help=(
            "Which eval CSV runtime column to use for RL curves. "
            "'auto' prefers runtime_total_sec when present."
        ),
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_csv).resolve()
    rows = _read_rows(eval_path)
    if not rows:
        raise RuntimeError("Eval CSV has no rows.")

    all_labels = sorted({row["checkpoint_label"] for row in rows})
    if args.checkpoint_labels:
        selected_labels = {item.strip() for item in args.checkpoint_labels.split(",") if item.strip()}
    else:
        selected_labels = set(all_labels)
    rl_rows = [row for row in rows if row["checkpoint_label"] in selected_labels]
    if not rl_rows:
        raise RuntimeError("No rows left after checkpoint label filtering.")

    csv_fields = set(rl_rows[0].keys())
    if args.runtime_source == "runtime_total_sec":
        if "runtime_total_sec" not in csv_fields:
            raise RuntimeError("runtime_total_sec not found in eval CSV.")
        runtime_key = "runtime_total_sec"
    elif args.runtime_source == "runtime_sec":
        runtime_key = "runtime_sec"
    else:
        runtime_key = "runtime_total_sec" if "runtime_total_sec" in csv_fields else "runtime_sec"

    if args.out_dir:
        root_dir = Path(args.out_dir).resolve()
        data_dir = root_dir / "data"
        plot_dir = root_dir / "plots"
    else:
        tag = f"{eval_path.stem}_vs_fvg"
        data_dir = EXPERIMENTS_RL_ROOT / "data" / "compare" / tag
        plot_dir = EXPERIMENTS_RL_ROOT / "plots" / tag
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Build unique target points (independent of checkpoint label).
    unique_targets: dict[tuple[int, int, int, float], dict] = {}
    for row in rl_rows:
        n = int(row["n"])
        m = int(row["m_target"])
        seed = int(row["seed"])
        rho = float(row["rho_target"])
        unique_targets[(n, m, seed, rho)] = row

    # Compute FVG once per (n, m), then reuse across seeds/rhos mapping.
    fvg_solver = create_baseline("fvg")
    fvg_cfg = DesignConfig(fvg_reuse_cache=True)
    fvg_cache: dict[tuple[int, int], tuple[float, float]] = {}
    for n, m, _, _ in sorted(unique_targets.keys()):
        key = (n, m)
        if key in fvg_cache:
            continue
        result = fvg_solver.solve(
            DesignProblem(n=n, m=m, seed=0),
            fvg_cfg,
            random.Random(0),
        )
        fvg_cache[key] = (float(result.final_lambda2), float(result.runtime_sec))

    combined_rows: list[dict] = []
    for row in rl_rows:
        combined_rows.append(
            {
                "solver_name": row["checkpoint_label"],
                "n": int(row["n"]),
                "rho_target": float(row["rho_target"]),
                "seed": int(row["seed"]),
                "m_target": int(row["m_target"]),
                "final_lambda2": float(row["terminal_lambda2"]),
                "runtime_sec": float(row[runtime_key]),
            }
        )

    # One FVG row per unique (n, rho, seed) target.
    for n, m, seed, rho in sorted(unique_targets.keys()):
        l2, rt = fvg_cache[(n, m)]
        combined_rows.append(
            {
                "solver_name": "fvg",
                "n": n,
                "rho_target": rho,
                "seed": seed,
                "m_target": m,
                "final_lambda2": l2,
                "runtime_sec": rt,
            }
        )

    combined_csv = data_dir / "combined_eval_vs_fvg.csv"
    with combined_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "solver_name",
                "n",
                "rho_target",
                "seed",
                "m_target",
                "final_lambda2",
                "runtime_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(combined_rows)

    n_values = sorted({int(row["n"]) for row in combined_rows})
    solver_names = sorted({row["solver_name"] for row in combined_rows}, key=_solver_order)
    lambda_data = _aggregate(combined_rows, "final_lambda2")
    runtime_data = _aggregate(combined_rows, "runtime_sec")

    lambda_paths = _plot_per_n(
        data=lambda_data,
        solver_names=solver_names,
        n_values=n_values,
        y_label=r"$\lambda_2$",
        y_scale="linear",
        out_prefix=plot_dir / "lambda2_vs_density.png",
    )
    runtime_paths = _plot_per_n(
        data=runtime_data,
        solver_names=solver_names,
        n_values=n_values,
        y_label="Runtime (sec)",
        y_scale=args.runtime_y_scale,
        out_prefix=plot_dir / "runtime_vs_density.png",
    )

    print(f"Wrote combined CSV: {combined_csv}")
    for p in lambda_paths + runtime_paths:
        print(f"Saved plot: {p}")


if __name__ == "__main__":
    main()
