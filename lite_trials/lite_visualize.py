"""Visualize lite experiment results.

Usage:
    python lite_trials/lite_visualize.py

Produces:
    lite_trials/results/lambda2_vs_density.png  — RL+Path vs RL+Backbone overlaid
    lite_trials/results/training_curves.png     — (if metrics.csv exists)
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
_results_dir = _project_root / "lite_trials" / "results"
_metrics_path = _project_root / "lite_trials" / "metrics.csv"

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    try:
        plt.style.use("ggplot")
    except Exception:
        pass


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def plot_lambda2_vs_density() -> None:
    """Overlay RL+Path and RL+Backbone on the same scatter plot."""
    raw_csv = _results_dir / "raw.csv"
    if not raw_csv.exists():
        print(f"Raw CSV not found: {raw_csv}")
        return

    rows = _read_csv(raw_csv)
    if not rows:
        print("No data rows found.")
        return

    # Collect: method -> n -> [(rho, lambda2), ...]
    data: dict[str, dict[int, list[tuple[float, float]]]] = {}
    for row in rows:
        method = row.get("method", "unknown")
        n = int(float(row["n"]))
        rho = float(row["rho_target"])
        lam2 = float(row["terminal_lambda2"])
        data.setdefault(method, {}).setdefault(n, []).append((rho, lam2))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Style: method = color, n = marker
    method_styles = {
        "RL+Path":     {"color": "C0", "marker": "o", "label": "RL+Path"},
        "RL+Backbone": {"color": "C1", "marker": "s", "label": "RL+Backbone"},
    }

    # Collect legend handles (one per (method, n) combo)
    handles = []

    for method in sorted(data):
        style = method_styles.get(method, {"color": "gray", "marker": "x", "label": method})
        for n_val in sorted(data[method]):
            points = sorted(data[method][n_val])
            rhos = [p[0] for p in points]
            lams = [p[1] for p in points]
            h = ax.scatter(
                rhos, lams,
                color=style["color"],
                marker=style["marker"],
                alpha=0.7, s=50, edgecolors="black", linewidth=0.3,
                label=f"{style['label']}  n={n_val}",
            )
            handles.append(h)

    ax.set_xlabel("Density (rho)")
    ax.set_ylabel("Algebraic Connectivity (lambda2)")
    ax.set_title("RL+Path vs RL+Backbone: lambda2 vs Density")
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = _results_dir / "lambda2_vs_density.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_training_curves() -> None:
    """Plot training metrics from metrics.csv."""
    if not _metrics_path.exists():
        return

    rows = _read_csv(_metrics_path)
    if not rows:
        return

    env_steps = [int(float(r["global_env_steps"])) for r in rows]
    eval_l2 = [float(r.get("eval_terminal_lambda2_mean", 0) or 0) for r in rows]
    episode_return = [float(r.get("mean_episode_return", 0) or 0) for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(env_steps, eval_l2, "b-", alpha=0.7)
    ax1.set_ylabel("Eval lambda2 (raw)")
    ax1.set_title("Evaluation: Terminal lambda2")
    ax1.grid(True, alpha=0.3)

    ax2.plot(env_steps, episode_return, "r-", alpha=0.7)
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Mean Episode Return")
    ax2.set_title("Training: Mean Episode Return")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Training Curves (Lite Trial)", fontsize=14)
    fig.tight_layout()
    out_path = _results_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    _results_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves()
    plot_lambda2_vs_density()
    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
