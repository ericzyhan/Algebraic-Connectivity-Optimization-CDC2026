from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_RL_ROOT = PROJECT_ROOT / "experiments" / "rl"


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_label_set(value: str | None, all_labels: set[str]) -> set[str]:
    if value is None:
        return {"best"} if "best" in all_labels else set(all_labels)
    out = {item.strip() for item in value.split(",") if item.strip()}
    if not out:
        raise ValueError("checkpoint-labels is empty after parsing.")
    return out


def _normalized_density(n: int, m: int) -> float:
    min_m = n - 1
    max_m = comb(n, 2)
    denom = max_m - min_m
    if denom <= 0:
        return 0.0
    return float(m - min_m) / float(denom)


def _aggregate(rows: list[dict], metric_key: str) -> dict[tuple[str, int], list[tuple[float, float]]]:
    grouped: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for row in rows:
        solver = row["solver_name"]
        n = int(row["n"])
        rho = float(row["rho"])
        grouped[(solver, n, rho)].append(float(row[metric_key]))

    out: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for (solver, n, rho), values in grouped.items():
        out[(solver, n)].append((rho, sum(values) / len(values)))
    for key in out:
        out[key] = sorted(out[key], key=lambda x: x[0])
    return out


def _solver_order(name: str) -> tuple[int, str]:
    if name == "backbone_fvg":
        return (0, name)
    if name == "backbone_rl_best":
        return (1, name)
    if name == "backbone_rl_last":
        return (2, name)
    return (10, name)


def _solver_style(name: str) -> dict:
    if name == "backbone_fvg":
        return {"color": "#2ca02c", "linewidth": 1.9}
    if name == "backbone_rl_best":
        return {"color": "#1f77b4", "linewidth": 1.9}
    if name == "backbone_rl_last":
        return {"color": "#ff7f0e", "linewidth": 1.7}
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
                    f"Cannot use exponential scale for n={n}: non-positive values found."
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


def _compute_pairwise_summary(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple[int, int, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (int(row["n"]), int(row["m"]), int(row["seed"]))
        by_key[key][row["solver_name"]] = row

    out: list[dict] = []
    for (n, m, seed), group in sorted(by_key.items()):
        fvg = group.get("backbone_fvg")
        if fvg is None:
            continue
        for solver_name, solver_row in group.items():
            if solver_name == "backbone_fvg":
                continue
            out.append(
                {
                    "n": n,
                    "m": m,
                    "seed": seed,
                    "rho": _normalized_density(n, m),
                    "solver_name": solver_name,
                    "delta_lambda2_vs_backbone_fvg": float(solver_row["final_lambda2"])
                    - float(fvg["final_lambda2"]),
                    "delta_runtime_sec_vs_backbone_fvg": float(solver_row["runtime_sec"])
                    - float(fvg["runtime_sec"]),
                    "runtime_ratio_vs_backbone_fvg": float(solver_row["runtime_sec"])
                    / float(fvg["runtime_sec"])
                    if float(fvg["runtime_sec"]) > 0
                    else float("inf"),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare backbone+FVG (hybrid) vs backbone-initialized RL eval."
    )
    parser.add_argument("--rl-eval-csv", required=True, help="Path to rl_train.evaluate CSV.")
    parser.add_argument(
        "--hybrid-csv",
        required=True,
        help="Path to experiments run_benchmarks results.csv (contains hybrid rows).",
    )
    parser.add_argument(
        "--checkpoint-labels",
        default=None,
        help="Comma list of RL checkpoint labels to compare (default: best if present, else all).",
    )
    parser.add_argument(
        "--runtime-source",
        default="auto",
        choices=["auto", "runtime_sec", "runtime_total_sec"],
        help=(
            "Runtime column for RL rows. "
            "'auto' prefers runtime_total_sec when present."
        ),
    )
    parser.add_argument(
        "--runtime-y-scale",
        default="linear",
        choices=["linear", "exponential"],
        help="Y-axis scale for runtime plots.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for combined CSV and plots.",
    )
    args = parser.parse_args()

    rl_path = Path(args.rl_eval_csv).resolve()
    hybrid_path = Path(args.hybrid_csv).resolve()
    rl_rows_all = _read_rows(rl_path)
    hybrid_rows_all = _read_rows(hybrid_path)

    if not rl_rows_all:
        raise RuntimeError("RL eval CSV has no rows.")
    if not hybrid_rows_all:
        raise RuntimeError("Hybrid CSV has no rows.")

    all_labels = {row["checkpoint_label"] for row in rl_rows_all}
    labels = _parse_label_set(args.checkpoint_labels, all_labels)
    rl_rows = [row for row in rl_rows_all if row["checkpoint_label"] in labels]
    if not rl_rows:
        raise RuntimeError("No RL rows left after checkpoint label filtering.")

    rl_fields = set(rl_rows[0].keys())
    if args.runtime_source == "runtime_total_sec":
        if "runtime_total_sec" not in rl_fields:
            raise RuntimeError("runtime_total_sec not found in RL eval CSV.")
        rl_runtime_key = "runtime_total_sec"
    elif args.runtime_source == "runtime_sec":
        rl_runtime_key = "runtime_sec"
    else:
        rl_runtime_key = "runtime_total_sec" if "runtime_total_sec" in rl_fields else "runtime_sec"

    hybrid_rows = [row for row in hybrid_rows_all if row.get("method") == "hybrid"]
    if not hybrid_rows:
        raise RuntimeError("No hybrid rows found in hybrid CSV.")

    if args.out_dir:
        root_dir = Path(args.out_dir).resolve()
        data_dir = root_dir / "data"
        plot_dir = root_dir / "plots"
    else:
        tag = f"{rl_path.stem}_vs_backbone_fvg"
        data_dir = EXPERIMENTS_RL_ROOT / "data" / "compare" / tag
        plot_dir = EXPERIMENTS_RL_ROOT / "plots" / tag
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Build map for hybrid by exact target.
    hybrid_by_key: dict[tuple[int, int, int], dict] = {}
    for row in hybrid_rows:
        key = (int(row["n"]), int(row["m"]), int(row["seed"]))
        hybrid_by_key[key] = row

    combined_rows: list[dict] = []

    # Add hybrid rows.
    for row in hybrid_rows:
        n = int(row["n"])
        m = int(row["m"])
        seed = int(row["seed"])
        combined_rows.append(
            {
                "solver_name": "backbone_fvg",
                "n": n,
                "m": m,
                "seed": seed,
                "rho": float(row["rho"]),
                "final_lambda2": float(row["final_lambda2"]),
                "runtime_sec": float(row["runtime_sec"]),
            }
        )

    # Add RL rows matched by (n,m,seed).
    missing = 0
    for row in rl_rows:
        n = int(row["n"])
        m = int(row["m_target"])
        seed = int(row["seed"])
        key = (n, m, seed)
        if key not in hybrid_by_key:
            missing += 1
            continue
        label = row["checkpoint_label"].strip().lower()
        solver_name = f"backbone_rl_{label}"
        combined_rows.append(
            {
                "solver_name": solver_name,
                "n": n,
                "m": m,
                "seed": seed,
                "rho": _normalized_density(n, m),
                "final_lambda2": float(row["terminal_lambda2"]),
                "runtime_sec": float(row[rl_runtime_key]),
            }
        )
    if missing > 0:
        print(f"[compare] warning: {missing} RL rows had no matching hybrid (n,m,seed).")

    combined_csv = data_dir / "combined_backbone_fvg_vs_backbone_rl.csv"
    with combined_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "solver_name",
                "n",
                "m",
                "seed",
                "rho",
                "final_lambda2",
                "runtime_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(combined_rows)

    summary_rows = _compute_pairwise_summary(combined_rows)
    summary_csv = data_dir / "summary_vs_backbone_fvg.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n",
                "m",
                "seed",
                "rho",
                "solver_name",
                "delta_lambda2_vs_backbone_fvg",
                "delta_runtime_sec_vs_backbone_fvg",
                "runtime_ratio_vs_backbone_fvg",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

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
    print(f"Wrote summary CSV: {summary_csv}")
    for path in lambda_paths + runtime_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
