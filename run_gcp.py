#!/usr/bin/env python3
"""
Standalone runner for Backbone+RL evaluation and figure generation.
Designed for headless execution on a GCP VM.

Usage:
    python run_gcp.py --checkpoint /path/to/best.pt \\
                      --config configs/rl_cpu_standard.yaml \\
                      --n-values 8,16,32,64,128 \\
                      --points-per-n 100 --repeats 5 \\
                      --out-dir results/rl --fig-dir figures

    # Backbone+RL only (skip Path+RL baseline):
    python run_gcp.py --checkpoint ... --backbone-only

    # Quick test:
    python run_gcp.py --checkpoint ... --quick-test
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────
# 1. Path & dependency setup
# ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def _check_dependencies() -> None:
    """Verify core imports work before launching subprocesses."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: PyTorch not found. Install with:")
        print("  pip install -r requirements-rl.txt")
        sys.exit(1)
    try:
        from rl_train.evaluate import main as eval_main  # noqa: F401
    except ImportError as e:
        print(f"ERROR: Cannot import rl_train.evaluate: {e}")
        print(f"Make sure you're running from the project root: {_PROJECT_ROOT}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────
# 2. Density helpers (mirrors run_path_vs_backbone_experiment.py)
# ─────────────────────────────────────────────────────────────────────
def _nontrivial_m_points(n: int, target_points: int) -> list[int]:
    n_i = int(n)
    m_lo = n_i - 1
    m_hi = (n_i * (n_i - 1)) // 2
    all_nt = list(range(m_lo + 1, m_hi))
    if len(all_nt) <= target_points:
        return all_nt
    idx = np.linspace(0, len(all_nt) - 1, target_points)
    idx_u = sorted(set(int(round(i)) for i in idx))
    if len(idx_u) < target_points:
        used = set(idx_u)
        for i in range(len(all_nt)):
            if i not in used:
                idx_u.append(i)
                used.add(i)
            if len(idx_u) >= target_points:
                break
        idx_u = sorted(idx_u[:target_points])
    return [all_nt[i] for i in idx_u]


def _rho_from_m(n: int, m: int) -> float:
    m_lo = n - 1
    m_hi = (n * (n - 1)) // 2
    den = m_hi - m_lo
    return 0.0 if den <= 0 else (m - m_lo) / den


# ─────────────────────────────────────────────────────────────────────
# 3. Evaluate subprocess launcher
# ─────────────────────────────────────────────────────────────────────
def _run_eval_job(
    *,
    config: Path,
    checkpoint: Path,
    n: int,
    densities: list[float],
    seeds: list[int],
    device: str,
    init_mode: str,
    out_csv: Path,
    backbone_args: dict,
) -> tuple[int, str, bool, str]:
    """Run evaluate.py for one (n, init_mode). Returns (n, mode, ok, csv_path)."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "rl_train.evaluate",
        "--config", str(config),
        "--checkpoint", f"best={checkpoint}",
        "--n-values", str(n),
        "--densities", ",".join(f"{r:.12f}" for r in densities),
        "--seeds", ",".join(str(s) for s in seeds),
        "--device", device,
        "--progress-every", "9999",
        "--quiet",
        "--init-mode", init_mode,
        "--out", str(out_csv),
    ]
    if init_mode == "backbone":
        ba = backbone_args
        cmd += ["--cayley-multi-index-overlap"] if ba["cayley_multi_index_overlap"] else ["--no-cayley-multi-index-overlap"]
        cmd += ["--cayley-enable-index4"] if ba["cayley_enable_index4"] else ["--no-cayley-enable-index4"]
        cmd += ["--cayley-index2-overlap-high", str(ba["cayley_index2_overlap_high"])]
        cmd += ["--cayley-index3-overlap-high", str(ba["cayley_index3_overlap_high"])]
        cmd += ["--cayley-index4-overlap-low", str(ba["cayley_index4_overlap_low"])]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    if ok:
        print(f"  [OK] n={n}, {init_mode}, {elapsed:.1f}s")
    else:
        print(f"  [FAIL] n={n}, {init_mode}, {elapsed:.1f}s")
        if result.stderr:
            print(f"    stderr: {result.stderr[:500]}")
    return n, init_mode, ok, str(out_csv)


# ─────────────────────────────────────────────────────────────────────
# 4. Summary computation (mirrors run_path_vs_backbone_experiment.py)
# ─────────────────────────────────────────────────────────────────────
def _compute_cost_summary(rows: list[dict]) -> list[dict]:
    rt_by_point: dict[tuple[str, int, int], list[float]] = {}
    for r in rows:
        k = (str(r["method"]), int(r["n"]), int(r["m_target"]))
        rt_by_point.setdefault(k, []).append(float(r["runtime_total_sec"]))
    rt_point_mean: dict[tuple[str, int], list[float]] = {}
    repeats_by_point: dict[tuple[str, int], list[int]] = {}
    for (method, n, _m), vv in rt_by_point.items():
        rt_point_mean.setdefault((method, n), []).append(float(np.mean(vv)))
        repeats_by_point.setdefault((method, n), []).append(len(vv))
    out = []
    for (method, n), vals in sorted(rt_point_mean.items(), key=lambda x: (x[0][0], x[0][1])):
        arr = np.asarray(vals, dtype=np.float64)
        reps = repeats_by_point[(method, n)]
        out.append({"method": method, "n": int(n),
                     "runtime_mean_over_density_s": float(np.mean(arr)),
                     "runtime_std_over_density_s": float(np.std(arr)),
                     "density_points": int(arr.size),
                     "avg_repeats_per_point": float(np.mean(reps)),
                     "min_repeats_per_point": int(min(reps)),
                     "max_repeats_per_point": int(max(reps))})
    return out


def _compute_quality_summary(rows: list[dict], tie_tol: float = 1e-10) -> list[dict]:
    path_l2: dict[tuple[int, int, int], float] = {}
    for r in rows:
        if str(r["method"]) == "RL+Path":
            path_l2[(int(r["n"]), int(r["m_target"]), int(r["repeat_idx"]))] = float(r["terminal_lambda2"])
    delta_by_point: dict[tuple[str, int, int], list[float]] = {}
    delta_pct_by_point: dict[tuple[str, int, int], list[float]] = {}
    win_count: dict[tuple[str, int], int] = {}
    tie_count: dict[tuple[str, int], int] = {}
    total_count: dict[tuple[str, int], int] = {}
    for r in rows:
        method, n, m_target, rep = str(r["method"]), int(r["n"]), int(r["m_target"]), int(r["repeat_idx"])
        lam = float(r["terminal_lambda2"])
        base = path_l2.get((n, m_target, rep))
        if base is None:
            continue
        d = lam - base
        d_pct = d / max(1e-15, abs(base)) * 100.0
        delta_by_point.setdefault((method, n, m_target), []).append(d)
        delta_pct_by_point.setdefault((method, n, m_target), []).append(d_pct)
        total_count[(method, n)] = total_count.get((method, n), 0) + 1
        if abs(d) <= tie_tol:
            tie_count[(method, n)] = tie_count.get((method, n), 0) + 1
        elif d > tie_tol:
            win_count[(method, n)] = win_count.get((method, n), 0) + 1
    q_delta_mean: dict[tuple[str, int], list[float]] = {}
    q_delta_pct_mean: dict[tuple[str, int], list[float]] = {}
    rep_counts: dict[tuple[str, int], list[int]] = {}
    for (method, n, m_target), vv in delta_by_point.items():
        q_delta_mean.setdefault((method, n), []).append(float(np.mean(vv)))
        q_delta_pct_mean.setdefault((method, n), []).append(float(np.mean(delta_pct_by_point[(method, n, m_target)])))
        rep_counts.setdefault((method, n), []).append(len(vv))
    out = []
    for (method, n) in sorted(set(q_delta_mean.keys()), key=lambda x: (x[0], x[1])):
        d_arr = np.asarray(q_delta_mean[(method, n)], dtype=np.float64)
        p_arr = np.asarray(q_delta_pct_mean[(method, n)], dtype=np.float64)
        total = total_count.get((method, n), 0)
        out.append({"method": method, "n": int(n),
                     "delta_lambda2_mean_over_density": float(np.mean(d_arr)),
                     "delta_lambda2_std_over_density": float(np.std(d_arr)),
                     "delta_lambda2_pct_mean_over_density": float(np.mean(p_arr)),
                     "delta_lambda2_pct_std_over_density": float(np.std(p_arr)),
                     "win_pct_vs_rl_path_over_all_points": float(100.0 * win_count.get((method, n), 0) / total) if total else 0.0,
                     "tie_pct_vs_rl_path_over_all_points": float(100.0 * tie_count.get((method, n), 0) / total) if total else 0.0,
                     "win_count": win_count.get((method, n), 0),
                     "tie_count": tie_count.get((method, n), 0),
                     "total_points": total,
                     "density_points": int(d_arr.size),
                     "avg_repeats_per_point": float(np.mean(rep_counts.get((method, n), [0])))})
    return out


# ─────────────────────────────────────────────────────────────────────
# 5. Figure generation (standalone, no notebook required)
# ─────────────────────────────────────────────────────────────────────
def _generate_figures(raw: "pd.DataFrame", summary_c: "pd.DataFrame | None",
                       summary_q: "pd.DataFrame | None", fig_dir: Path) -> None:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 11, "axes.titlesize": 13,
        "axes.labelsize": 12, "legend.fontsize": 10, "lines.linewidth": 1.5,
    })

    bb = raw[raw["method"] == "RL+Backbone"].copy()
    path_only = raw[raw["method"] == "RL+Path"].copy()
    n_order = sorted(raw["n"].unique())
    colors = {8: "#4C72B0", 16: "#DD8452", 32: "#55A868", 64: "#C44E52", 128: "#8172B2"}
    markers = {8: "o", 16: "s", 32: "D", 64: "^", 128: "v"}

    def group_agg(df, xcol, ycol, group_cols):
        g = df.groupby(group_cols).agg(mean=(ycol, "mean"), std=(ycol, "std"), count=(ycol, "count")).reset_index()
        return g

    # ── Fig 1: λ₂ vs ρ per n ──
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes_flat = axes.flatten()
    for idx, n in enumerate(n_order):
        ax = axes_flat[idx]
        sub = bb[bb["n"] == n]
        grp = group_agg(sub, "rho_target", "terminal_lambda2", ["rho_target"])
        ax.errorbar(grp["rho_target"], grp["mean"], yerr=grp["std"],
                    fmt=f"-{markers[n]}", color=colors[n], capsize=3, markersize=5, label=f"n={n}")
        ax.set_xlabel(r"Target density $\rho$")
        ax.set_ylabel(r"$\lambda_2$")
        ax.set_title(f"n = {n}")
        ax.legend(); ax.grid(True, alpha=0.3)
    axes_flat[-1].set_visible(False)
    fig.suptitle(r"Backbone+RL: $\lambda_2$ vs Target Density", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_lambda2_vs_rho.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] fig1_lambda2_vs_rho.png")

    # ── Fig 2: Normalized λ₂/n ──
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for n in n_order:
        sub = bb[bb["n"] == n]
        grp = group_agg(sub, "rho_target", "terminal_lambda2_norm", ["rho_target"])
        ax.errorbar(grp["rho_target"], grp["mean"], yerr=grp["std"],
                    fmt=f"-{markers[n]}", color=colors[n], capsize=3, markersize=5, label=f"n={n}")
    ax.set_xlabel(r"Target density $\rho$")
    ax.set_ylabel(r"$\lambda_2 / n$")
    ax.set_title(r"Backbone+RL: Normalized $\lambda_2$ vs Density")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_lambda2norm_vs_rho.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] fig2_lambda2norm_vs_rho.png")

    # ── Fig 3: Improvement ──
    bb_plot = bb.copy()
    bb_plot["lambda2_improvement"] = bb_plot["terminal_lambda2"] - bb_plot["init_lambda2"]
    bb_plot["lambda2_improvement_pct"] = (
        (bb_plot["terminal_lambda2"] - bb_plot["init_lambda2"]) / bb_plot["init_lambda2"].clip(lower=1e-10) * 100)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax_idx, (col, ylabel, title) in enumerate([
        ("lambda2_improvement", r"$\Delta\lambda_2$", "Absolute Improvement"),
        ("lambda2_improvement_pct", r"$\Delta\lambda_2 / \lambda_2^{\text{init}}$ (%)", "Relative Improvement"),
    ]):
        ax = axes[ax_idx]
        for n in n_order:
            sub = bb_plot[bb_plot["n"] == n]
            grp = group_agg(sub, "rho_target", col, ["rho_target"])
            ax.errorbar(grp["rho_target"], grp["mean"], yerr=grp["std"],
                        fmt=f"-{markers[n]}", color=colors[n], capsize=3, markersize=5, label=f"n={n}")
        ax.set_xlabel(r"Target density $\rho$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(r"Backbone+RL: $\lambda_2$ Improvement over Initial Backbone", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_lambda2_improvement.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] fig3_lambda2_improvement.png")

    # ── Fig 4: BB+RL vs Path+RL ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    for idx, n in enumerate(n_order):
        ax = axes_flat[idx]
        for label, grp_df, color in [("RL+Path", path_only, "#888888"), ("RL+Backbone", bb, colors[n])]:
            sub = grp_df[grp_df["n"] == n]
            grp = group_agg(sub, "rho_target", "terminal_lambda2", ["rho_target"])
            ax.plot(grp["rho_target"], grp["mean"],
                    marker=markers[n] if label == "RL+Backbone" else ".", color=color, label=label, alpha=0.85)
        ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$\lambda_2$"); ax.set_title(f"n={n}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes_flat[-1].set_visible(False)
    fig.suptitle(r"$\lambda_2$: Backbone+RL vs Path+RL", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_bb_vs_path_lambda2.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] fig4_bb_vs_path_lambda2.png")

    # ── Fig 5: Heatmap ──
    bb_plot = bb.copy()
    bb_plot["rho_bin"] = pd.cut(bb_plot["rho_target"], bins=20, labels=False) / 20
    bin_edges = pd.cut(bb["rho_target"], bins=20).cat.categories
    bin_centers = [(iv.left + iv.right) / 2 for iv in bin_edges]
    bb_pivot = bb_plot.pivot_table(index="n", columns="rho_bin", values="terminal_lambda2", aggfunc="mean")
    bb_pivot = bb_pivot.reindex(n_order)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(bb_pivot.values, aspect="auto", cmap="viridis", origin="lower", interpolation="nearest")
    tick_step = max(1, len(bin_centers) // 10)
    ax.set_xticks(range(0, len(bin_centers), tick_step))
    ax.set_xticklabels([f"{bin_centers[i]:.2f}" for i in range(0, len(bin_centers), tick_step)])
    ax.set_yticks(range(len(n_order)))
    ax.set_yticklabels([f"n={int(n)}" for n in n_order])
    ax.set_xlabel(r"Target density $\rho$")
    ax.set_title(r"Mean $\lambda_2$ across $(n, \rho)$ — Backbone+RL")
    fig.colorbar(im, ax=ax, shrink=0.8).set_label(r"$\lambda_2$")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_heatmap_lambda2.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] fig5_heatmap_lambda2.png")

    # ── Fig 6: Runtime ──
    if summary_c is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for _, row in summary_c.iterrows():
            n = int(row["n"])
            if row["method"] == "RL+Path":
                continue
            ax.errorbar(n, row["runtime_mean_over_density_s"],
                        yerr=row["runtime_std_over_density_s"],
                        fmt=markers.get(n, "o"), color=colors.get(n, "#333"),
                        capsize=3, markersize=8, label=f"n={n}")
        ax.set_xlabel("Graph size $n$"); ax.set_ylabel("Runtime (s)")
        ax.set_title("Backbone+RL: Runtime per Episode vs $n$")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(list(n_order)); ax.set_xticklabels([str(v) for v in n_order])
        ax.legend(); ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig6_runtime.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  [FIG] fig6_runtime.png")

    # ── Fig 7: By family ──
    family_colors = {"cayley": "#E24A33", "envelope": "#348ABD"}
    family_markers = {"cayley": "^", "envelope": "s"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    for idx, n in enumerate(n_order):
        ax = axes_flat[idx]
        sub = bb[bb["n"] == n]
        for fam in ["cayley", "envelope"]:
            fam_df = sub[sub["init_family"] == fam]
            if fam_df.empty:
                continue
            grp = group_agg(fam_df, "rho_target", "terminal_lambda2", ["rho_target"])
            ax.errorbar(grp["rho_target"], grp["mean"], yerr=grp["std"],
                        fmt=f"-{family_markers[fam]}", color=family_colors[fam],
                        capsize=3, markersize=5, label=f"{fam}")
        ax.set_xlabel(r"Target density $\rho$"); ax.set_ylabel(r"$\lambda_2$")
        ax.set_title(f"n = {n}"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes_flat[-1].set_visible(False)
    fig.suptitle(r"BB+RL: $\lambda_2$ by Backbone Family", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig7_lambda2_by_family.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] fig7_lambda2_by_family.png")

    # ── Fig 8: Summary panel ──
    fig = plt.figure(figsize=(16, 12))
    def plot_metric(ax, ycol, ylabel, title):
        for n in n_order:
            sub = bb[bb["n"] == n].copy()
            ycol_use = ycol
            if ycol not in sub.columns and ycol == "lambda2_improvement_pct":
                sub["lambda2_improvement_pct"] = (sub["terminal_lambda2"] - sub["init_lambda2"]) / sub["init_lambda2"].clip(lower=1e-10) * 100
                ycol_use = "lambda2_improvement_pct"
            grp = group_agg(sub, "rho_target", ycol_use, ["rho_target"])
            ax.plot(grp["rho_target"], grp["mean"], marker=markers[n], color=colors[n], label=f"n={n}")
        ax.set_xlabel(r"$\rho$"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    panels = [
        (1, "terminal_lambda2", r"$\lambda_2$", r"$\lambda_2$ vs $\rho$"),
        (2, "terminal_lambda2_norm", r"$\lambda_2 / n$", r"Normalized $\lambda_2$"),
        (3, "episode_len", "Edges added", "Episode Length"),
        (4, "init_lambda2", r"$\lambda_2^{\text{init}}$", "Initial Backbone $\lambda_2$"),
        (5, "lambda2_improvement_pct", r"$\Delta\lambda_2 / \lambda_2^{\text{init}}$ (%)", "Relative Improvement"),
        (6, "rl_runtime_sec", "Runtime (s)", "RL Runtime"),
    ]
    for pos, col, ylbl, title in panels:
        ax = fig.add_subplot(2, 3, pos)
        plot_metric(ax, col, ylbl, title)
    fig.suptitle("Backbone+RL Performance Summary", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig8_summary_panel.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] fig8_summary_panel.png")

    print(f"  All figures saved to {fig_dir}/")


# ─────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backbone+RL: Run evaluation and generate figures on GCP VM."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--config", default=str(_PROJECT_ROOT / "configs" / "rl_cpu_standard.yaml"),
                        help="Path to YAML config")
    parser.add_argument("--n-values", default="8,16,32,64,128", help="Comma-separated graph sizes")
    parser.add_argument("--points-per-n", type=int, default=100, help="Density points per n")
    parser.add_argument("--repeats", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--out-dir", default="results/rl", help="Output directory for CSVs")
    parser.add_argument("--fig-dir", default="figures", help="Output directory for figures")
    parser.add_argument("--backbone-only", action="store_true",
                        help="Skip Path+RL baseline (only run Backbone+RL)")
    parser.add_argument("--path-only", action="store_true",
                        help="Skip Backbone+RL (only run Path+RL baseline)")
    parser.add_argument("--quick-test", action="store_true",
                        help="Small grid for testing: n=8,16,32, points=10, repeats=2")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation, only generate figures from existing CSVs")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max parallel worker processes (default: os.cpu_count())")
    # Backbone args
    parser.add_argument("--cayley-index2-overlap-high", type=float, default=0.6)
    parser.add_argument("--cayley-index3-overlap-high", type=float, default=2.0 / 3.0)
    parser.add_argument("--cayley-index4-overlap-low", type=float, default=2.0 / 3.0)
    parser.add_argument("--cayley-multi-index-overlap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cayley-enable-index4", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick_test:
        n_values = [8, 16, 32]
        points_per_n = 10
        repeats = 2
    else:
        n_values = [int(v.strip()) for v in args.n_values.split(",") if v.strip()]
        points_per_n = args.points_per_n
        repeats = args.repeats

    seeds = list(range(args.seed_start, args.seed_start + repeats))
    out_dir = Path(args.out_dir).resolve()
    fig_dir = Path(args.fig_dir).resolve()
    config = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    max_workers = args.max_workers or os.cpu_count() or 4

    backbone_args = {
        "cayley_index2_overlap_high": args.cayley_index2_overlap_high,
        "cayley_index3_overlap_high": args.cayley_index3_overlap_high,
        "cayley_index4_overlap_low": args.cayley_index4_overlap_low,
        "cayley_multi_index_overlap": args.cayley_multi_index_overlap,
        "cayley_enable_index4": args.cayley_enable_index4,
    }

    modes = []
    if not args.path_only:
        modes.append("backbone")
    if not args.backbone_only:
        modes.append("path")

    # ── Header ──
    print("=" * 72)
    print(f"Backbone+RL Evaluation — GCP VM Runner")
    print(f"  Config:     {config}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  n values:   {n_values}")
    print(f"  Points/n:   {points_per_n}, Repeats: {repeats}")
    print(f"  Modes:      {modes}")
    print(f"  Workers:    {max_workers}")
    print(f"  Out dir:    {out_dir}")
    print(f"  Fig dir:    {fig_dir}")
    print("=" * 72)

    # ── Check ──
    _check_dependencies()
    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        sys.exit(1)
    if not config.exists():
        print(f"ERROR: Config not found: {config}")
        sys.exit(1)

    # ── Evaluate ──
    if not args.skip_eval:
        jobs = [(n, mode) for n in n_values for mode in modes]
        total_episodes = sum(len(_nontrivial_m_points(n, points_per_n)) * repeats * len(modes) for n in n_values)
        print(f"\nTotal episodes: {total_episodes} ({len(jobs)} parallel jobs)")
        t_start = time.time()

        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for n in n_values:
                m_points = _nontrivial_m_points(n, points_per_n)
                rho_values = [_rho_from_m(n, m) for m in m_points]
                for mode in modes:
                    out_csv = out_dir / "per_n_mode" / f"eval_{mode}_n{n}.csv"
                    fut = pool.submit(_run_eval_job, config=config, checkpoint=checkpoint,
                                       n=n, densities=rho_values, seeds=seeds, device=args.device,
                                       init_mode=mode, out_csv=out_csv, backbone_args=backbone_args)
                    futures[fut] = (n, mode)
            for future in concurrent.futures.as_completed(futures):
                n, mode, ok, csv_path = future.result()
                results[(n, mode)] = (ok, csv_path)

        elapsed = time.time() - t_start
        print(f"\nEvaluation completed in {elapsed:.1f}s")

        # ── Merge results ──
        raw_rows = []
        first_fields = None
        for n in n_values:
            for mode in modes:
                ok, csv_path = results.get((n, mode), (False, ""))
                if not ok or not Path(csv_path).exists():
                    print(f"  WARNING: Missing results for n={n}, mode={mode}")
                    continue
                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        enriched = dict(row)
                        enriched["method"] = "RL+Backbone" if mode == "backbone" else "RL+Path"
                        enriched["repeat_idx"] = int(float(row["seed"]))
                        raw_rows.append(enriched)
                        if first_fields is None:
                            first_fields = list(row.keys())

        out_dir.mkdir(parents=True, exist_ok=True)
        raw_fields = list(first_fields or [])
        for col in ("method", "repeat_idx"):
            if col not in raw_fields:
                raw_fields.append(col)
        with open(out_dir / "raw.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=raw_fields)
            w.writeheader()
            w.writerows(raw_rows)
        print(f"Combined {len(raw_rows)} rows -> {out_dir / 'raw.csv'}")

        # Summary CSVs
        if "path" in modes and "backbone" in modes:
            cost_rows = _compute_cost_summary(raw_rows)
            qual_rows = _compute_quality_summary(raw_rows)
            with open(out_dir / "summary_cost.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(cost_rows[0].keys()))
                w.writeheader(); w.writerows(cost_rows)
            with open(out_dir / "summary_quality.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(qual_rows[0].keys()))
                w.writeheader(); w.writerows(qual_rows)
            print(f"Summary CSVs written ({len(cost_rows)} cost, {len(qual_rows)} quality)")

        with open(out_dir / "raw.meta.json", "w") as f:
            json.dump({"checkpoint": str(checkpoint), "n_values": n_values,
                       "points_per_n": points_per_n, "repeats": repeats,
                       "total_rows": len(raw_rows), "elapsed_sec": round(elapsed, 1)}, f, indent=2)
    else:
        print("\nSkipping evaluation (--skip-eval). Loading existing CSVs...")

    # ── Generate figures ──
    print("\n" + "=" * 72)
    print("GENERATING FIGURES")
    print("=" * 72)

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not installed. Figures cannot be generated.")
        print("Install: pip install pandas matplotlib")
        sys.exit(1)

    raw_path = out_dir / "raw.csv"
    if not raw_path.exists():
        # Fallback to results_prev
        fallback = _PROJECT_ROOT / "results_prev" / "rl" / "raw.csv"
        if fallback.exists():
            print(f"Results not found at {raw_path}, using fallback: {fallback}")
            raw_path = fallback
        else:
            print(f"ERROR: No results found at {raw_path} or {fallback}")
            sys.exit(1)

    raw = pd.read_csv(raw_path)
    summary_c = None
    summary_q = None
    for p in [out_dir / "summary_cost.csv", _PROJECT_ROOT / "results_prev" / "rl" / "summary_cost.csv"]:
        if p.exists():
            summary_c = pd.read_csv(p)
            break
    for p in [out_dir / "summary_quality.csv", _PROJECT_ROOT / "results_prev" / "rl" / "summary_quality.csv"]:
        if p.exists():
            summary_q = pd.read_csv(p)
            break

    print(f"Loaded {len(raw)} rows, methods={raw['method'].unique().tolist()}")
    _generate_figures(raw, summary_c, summary_q, fig_dir)

    # ── Summary ──
    if summary_q is not None:
        print("\n" + "=" * 72)
        print("KEY FINDINGS")
        print("=" * 72)
        bb_summary = summary_q[summary_q["method"] == "RL+Backbone"]
        if len(bb_summary) > 0:
            print(f"{'n':>4}  {'Δλ₂ mean':>10}  {'Δλ₂ %':>8}  {'Win%':>6}  {'Points':>7}")
            print("-" * 42)
            for _, row in bb_summary.iterrows():
                print(f"{int(row['n']):>4}  {row['delta_lambda2_mean_over_density']:>10.4f}"
                      f"  {row['delta_lambda2_pct_mean_over_density']:>7.1f}%"
                      f"  {row['win_pct_vs_rl_path_over_all_points']:>5.1f}%"
                      f"  {int(row['total_points']):>7}")

    print("\nDone.")


if __name__ == "__main__":
    main()
