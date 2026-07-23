"""Compare every trained combo in lite_trials/sweep/manifest.json.

Reads, per combo: metrics.csv (training curves), results/raw.csv (density
sweep), results/density_bucket_summary.csv (density-bucket averages), and
results/summary_quality.csv (win-rate vs RL+Path). Produces a comparison
figure set plus a master summary_by_alpha.csv under lite_trials/sweep/comparison/.

Usage (from project root, after lite_sweep_train.py + lite_sweep_experiment.py):
    python -m lite_trials.lite_sweep_visualize
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

_SWEEP_ROOT = _project_root / "lite_trials" / "sweep"
_MANIFEST_PATH = _SWEEP_ROOT / "manifest.json"
_COMPARISON_DIR = _SWEEP_ROOT / "comparison"

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    try:
        plt.style.use("ggplot")
    except Exception:
        pass

# Validated categorical palette (fixed hue order -- never cycled/reassigned
# by filtering). Assigned to combos by their fixed position in manifest.json.
_CATEGORICAL_HUES = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
_SEQUENTIAL_BLUE_STEPS = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]
_SEQUENTIAL_BLUE_CMAP = LinearSegmentedColormap.from_list("seq_blue", _SEQUENTIAL_BLUE_STEPS)

_GRID_KW = dict(alpha=0.3, linewidth=0.6)
_METHOD_LINESTYLE = {"RL+Path": "-", "RL+Backbone": "--"}
_METHOD_MARKER = {"RL+Path": "o", "RL+Backbone": "s"}
_DENSITY_BUCKET_ORDER = ["sparse", "medium", "dense"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _combo_color(idx: int) -> str:
    return _CATEGORICAL_HUES[idx % len(_CATEGORICAL_HUES)]


def _load_manifest() -> list[dict[str, object]]:
    if not _MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found: {_MANIFEST_PATH}. Run lite_sweep_train.py first.")
        sys.exit(1)
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def _load_combo_data(entry: dict[str, object]) -> dict[str, list[dict[str, str]]]:
    run_dir = _project_root / str(entry["run_dir"])
    return {
        "metrics": _read_csv(run_dir / "metrics.csv"),
        "raw": _read_csv(run_dir / "results" / "raw.csv"),
        "buckets": _read_csv(run_dir / "results" / "density_bucket_summary.csv"),
        "quality": _read_csv(run_dir / "results" / "summary_quality.csv"),
    }


def _combo_label(entry: dict[str, object]) -> str:
    return (
        f"{entry['tag']} "
        f"(a1={float(entry['alpha_1']):.2f}, "
        f"a2={float(entry['alpha_2']):.2f}, "
        f"a3={float(entry['alpha_3']):.2f})"
    )


def plot_training_curves(manifest: list[dict[str, object]], data: list[dict[str, list]]) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    any_data = False
    for i, (entry, d) in enumerate(zip(manifest, data)):
        rows = d["metrics"]
        if not rows:
            continue
        any_data = True
        color = _combo_color(i)
        steps = [int(float(r["global_env_steps"])) for r in rows]
        eval_l2 = [float(r.get("eval_terminal_lambda2_mean", 0) or 0) for r in rows]
        episode_return = [float(r.get("mean_episode_return", 0) or 0) for r in rows]
        label = _combo_label(entry)
        ax1.plot(steps, eval_l2, color=color, linewidth=2, alpha=0.9, label=label)
        ax2.plot(steps, episode_return, color=color, linewidth=2, alpha=0.9, label=label)

    if not any_data:
        plt.close(fig)
        print("No metrics.csv data found across combos; skipping training_curves_comparison.png")
        return

    ax1.set_ylabel("Eval lambda2 (raw)")
    ax1.set_title("Evaluation: terminal lambda2 by reward-blend combo")
    ax1.grid(True, **_GRID_KW)
    ax1.legend(fontsize=8, loc="lower right")

    ax2.set_xlabel("Environment steps")
    ax2.set_ylabel("Mean episode return")
    ax2.set_title("Training: mean episode return by reward-blend combo")
    ax2.grid(True, **_GRID_KW)
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle("Training curves across alpha combos", fontsize=14)
    fig.tight_layout()
    out_path = _COMPARISON_DIR / "training_curves_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _binned_mean_std(points: list[tuple[float, float]], n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, means, stds = [], [], []
    rhos = np.asarray([p[0] for p in points], dtype=np.float64)
    lams = np.asarray([p[1] for p in points], dtype=np.float64)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (rhos >= lo) & (rhos < hi) if hi < 1.0 else (rhos >= lo) & (rhos <= hi)
        if not np.any(mask):
            continue
        centers.append(0.5 * (lo + hi))
        means.append(float(np.mean(lams[mask])))
        stds.append(float(np.std(lams[mask])))
    return np.asarray(centers), np.asarray(means), np.asarray(stds)


def plot_lambda2_vs_density(manifest: list[dict[str, object]], data: list[dict[str, list]]) -> None:
    n_values: set[int] = set()
    for d in data:
        for r in d["raw"]:
            n_values.add(int(float(r["n"])))
    if not n_values:
        print("No raw.csv data found across combos; skipping lambda2_vs_density_by_alpha.png")
        return
    n_sorted = sorted(n_values)

    ncols = min(3, len(n_sorted))
    nrows = int(np.ceil(len(n_sorted) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)

    for panel_idx, n_val in enumerate(n_sorted):
        ax = axes[panel_idx // ncols][panel_idx % ncols]
        for i, (entry, d) in enumerate(zip(manifest, data)):
            color = _combo_color(i)
            by_method: dict[str, list[tuple[float, float]]] = {}
            for r in d["raw"]:
                if int(float(r["n"])) != n_val:
                    continue
                method = r.get("method", "unknown")
                by_method.setdefault(method, []).append(
                    (float(r["rho_final"]), float(r["terminal_lambda2"]))
                )
            for method, points in by_method.items():
                if len(points) < 2:
                    continue
                centers, means, stds = _binned_mean_std(points)
                if centers.size == 0:
                    continue
                ls = _METHOD_LINESTYLE.get(method, "-")
                ax.plot(centers, means, color=color, linestyle=ls, linewidth=1.8, alpha=0.9)
                ax.fill_between(centers, means - stds, means + stds, color=color, alpha=0.12, linewidth=0)
        ax.set_title(f"n = {n_val}", fontsize=10)
        ax.set_xlabel("Density (rho)")
        ax.set_ylabel("lambda2")
        ax.grid(True, **_GRID_KW)

    for panel_idx in range(len(n_sorted), nrows * ncols):
        axes[panel_idx // ncols][panel_idx % ncols].axis("off")

    combo_handles = [
        Patch(color=_combo_color(i), label=entry["tag"]) for i, entry in enumerate(manifest)
    ]
    method_handles = [
        plt.Line2D([0], [0], color="black", linestyle=ls, label=method)
        for method, ls in _METHOD_LINESTYLE.items()
    ]
    fig.legend(
        handles=combo_handles + method_handles,
        loc="lower center",
        ncol=min(len(combo_handles) + len(method_handles), 5),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("lambda2 vs density, binned mean +/- std, by alpha combo", fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out_path = _COMPARISON_DIR / "lambda2_vs_density_by_alpha.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_density_bucket_bars(manifest: list[dict[str, object]], data: list[dict[str, list]]) -> None:
    methods: set[str] = set()
    n_values: set[int] = set()
    for d in data:
        for r in d["buckets"]:
            methods.add(r["method"])
            n_values.add(int(float(r["n"])))
    if not methods or not n_values:
        print("No density_bucket_summary.csv data found; skipping density_bucket_bars.png")
        return
    methods_sorted = sorted(methods)
    n_sorted = sorted(n_values)

    nrows, ncols = len(n_sorted), len(methods_sorted)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.2 * nrows), squeeze=False)

    n_combos = len(manifest)
    bar_width = 0.8 / max(1, n_combos)
    bucket_x = np.arange(len(_DENSITY_BUCKET_ORDER))

    for row, n_val in enumerate(n_sorted):
        for col, method in enumerate(methods_sorted):
            ax = axes[row][col]
            for i, (entry, d) in enumerate(zip(manifest, data)):
                color = _combo_color(i)
                lookup = {
                    (r["density_bucket"]): (float(r["lambda2_mean"]), float(r["lambda2_std"]))
                    for r in d["buckets"]
                    if r["method"] == method and int(float(r["n"])) == n_val
                }
                means = [lookup.get(b, (0.0, 0.0))[0] for b in _DENSITY_BUCKET_ORDER]
                stds = [lookup.get(b, (0.0, 0.0))[1] for b in _DENSITY_BUCKET_ORDER]
                present = [b in lookup for b in _DENSITY_BUCKET_ORDER]
                offsets = bucket_x + (i - (n_combos - 1) / 2.0) * bar_width
                means_masked = [m if p else np.nan for m, p in zip(means, present)]
                stds_masked = [s if p else 0.0 for s, p in zip(stds, present)]
                ax.bar(
                    offsets, means_masked, width=bar_width * 0.92, color=color,
                    yerr=stds_masked, capsize=2, error_kw={"linewidth": 0.8, "alpha": 0.6},
                )
            ax.set_xticks(bucket_x)
            ax.set_xticklabels(_DENSITY_BUCKET_ORDER, fontsize=8)
            ax.set_title(f"{method}, n={n_val}", fontsize=9)
            ax.set_ylabel("lambda2", fontsize=8)
            ax.grid(True, axis="y", **_GRID_KW)

    combo_handles = [
        Patch(color=_combo_color(i), label=entry["tag"]) for i, entry in enumerate(manifest)
    ]
    fig.legend(
        handles=combo_handles, loc="lower center", ncol=min(len(combo_handles), 5),
        fontsize=8, bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("Algebraic connectivity averaged over density buckets, by alpha combo", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out_path = _COMPARISON_DIR / "density_bucket_bars.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_alpha_simplex(manifest: list[dict[str, object]], data: list[dict[str, list]], scores: list[float | None]) -> None:
    valid = [(e, s) for e, s in zip(manifest, scores) if s is not None]
    if len(valid) < 2:
        print("Fewer than 2 combos have benchmark scores; skipping alpha_simplex_summary.png")
        return

    fig, ax = plt.subplots(figsize=(6.5, 6))
    a1 = np.array([float(e["alpha_1"]) for e, _ in valid])
    a3 = np.array([float(e["alpha_3"]) for e, _ in valid])
    vals = np.array([s for _, s in valid])
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmax - vmin < 1e-9:
        vmax = vmin + 1e-9

    ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color="#c3c2b7", linewidth=1.2, zorder=1)
    sc = ax.scatter(
        a1, a3, c=vals, cmap=_SEQUENTIAL_BLUE_CMAP, vmin=vmin, vmax=vmax,
        s=180, edgecolors="#0b0b0b", linewidths=0.8, zorder=3,
    )
    for (e, _), x, y in zip(valid, a1, a3):
        ax.annotate(
            str(e["tag"]), (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, color="#0b0b0b"
        )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Mean lambda2 (RL+Path, all n & density buckets)", fontsize=9)
    cbar.formatter = matplotlib.ticker.FormatStrFormatter("%.3f")
    cbar.update_ticks()

    ax.set_xlabel("alpha1 (delta-lambda2 weight)")
    ax.set_ylabel("alpha3 (delta-P_min weight)")
    ax.set_title("Reward-blend simplex: alpha2 = 1 - alpha1 - alpha3", fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, **_GRID_KW)

    fig.tight_layout()
    out_path = _COMPARISON_DIR / "alpha_simplex_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _overall_score(bucket_rows: list[dict[str, str]], method: str = "RL+Path") -> float | None:
    vals = [float(r["lambda2_mean"]) for r in bucket_rows if r["method"] == method]
    if not vals:
        return None
    return float(np.mean(vals))


def write_summary_csv(manifest: list[dict[str, object]], data: list[dict[str, list]]) -> list[float | None]:
    _COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tag", "alpha_1", "alpha_2", "alpha_3",
        "mean_lambda2_rl_path", "mean_lambda2_rl_backbone",
        "win_pct_vs_rl_path_avg", "best_eval_lambda2_from_training_log", "final_mean_episode_return",
        "density_points", "n_values_covered",
    ]
    rows: list[dict[str, object]] = []
    scores: list[float | None] = []
    for entry, d in zip(manifest, data):
        score_path = _overall_score(d["buckets"], "RL+Path")
        score_bb = _overall_score(d["buckets"], "RL+Backbone")
        scores.append(score_path)

        win_pcts = [float(r["win_pct_vs_rl_path_over_all_points"]) for r in d["quality"] if r["method"] == "RL+Backbone"]
        eval_vals = [float(r.get("eval_terminal_lambda2_mean", 0) or 0) for r in d["metrics"]]
        return_vals = [float(r.get("mean_episode_return", 0) or 0) for r in d["metrics"]]
        n_covered = sorted({int(float(r["n"])) for r in d["raw"]})

        rows.append(
            {
                "tag": entry["tag"],
                "alpha_1": entry["alpha_1"],
                "alpha_2": entry["alpha_2"],
                "alpha_3": entry["alpha_3"],
                "mean_lambda2_rl_path": score_path if score_path is not None else "",
                "mean_lambda2_rl_backbone": score_bb if score_bb is not None else "",
                "win_pct_vs_rl_path_avg": float(np.mean(win_pcts)) if win_pcts else "",
                "best_eval_lambda2_from_training_log": max(eval_vals) if eval_vals else "",
                "final_mean_episode_return": return_vals[-1] if return_vals else "",
                "density_points": len(d["raw"]),
                "n_values_covered": ";".join(str(n) for n in n_covered),
            }
        )

    out_path = _COMPARISON_DIR / "summary_by_alpha.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {out_path}")
    return scores


def main() -> None:
    manifest = _load_manifest()
    if not manifest:
        print("Manifest is empty; nothing to visualize.")
        return
    _COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    data = [_load_combo_data(entry) for entry in manifest]

    scores = write_summary_csv(manifest, data)
    plot_training_curves(manifest, data)
    plot_lambda2_vs_density(manifest, data)
    plot_density_bucket_bars(manifest, data)
    plot_alpha_simplex(manifest, data, scores)

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
