"""
Generate publication-quality figures for the Backbone+RL algebraic connectivity paper.

Design philosophy:
  - All figures have transparent backgrounds (readable on dark backgrounds).
  - A single "main result" figure also has a transparent background but is styled
    for readability on a white background (dark text/grid).
  - Clean, minimal, no chartjunk.
  - Font size 28 base, consistent across all scripts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figure_config as cfg

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent / "results-prev" / "rl"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
raw = pd.read_csv(RESULTS_DIR / "raw.csv")
summary_q = pd.read_csv(RESULTS_DIR / "summary_quality.csv")
summary_c = pd.read_csv(RESULTS_DIR / "summary_cost.csv")

bb = raw[raw["method"] == "RL+Backbone"].copy()
path_only = raw[raw["method"] == "RL+Path"].copy()

print(f"Data: {len(raw)} rows | {raw['method'].unique()}")
print(f"n values: {sorted(raw['n'].unique())}")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _group_agg(df: pd.DataFrame, xcol: str, ycol: str, group_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(group_cols).agg(mean=(ycol, "mean"), std=(ycol, "std"), count=(ycol, "count")).reset_index()
    return g


def _set_transparent(fig: plt.Figure, ax: plt.Axes | None = None) -> None:
    fig.patch.set_alpha(0.0)
    if ax is None:
        for a in fig.axes:
            a.patch.set_alpha(0.0)
    else:
        ax.patch.set_alpha(0.0)

# ═════════════════════════════════════════════════════════════════════════════
# DARK-BACKGROUND FIGURES  (readable on black)
# ═════════════════════════════════════════════════════════════════════════════


# ── FIGURE 1 — λ₂ vs ρ (BB+RL, all n overlaid, single clean panel) ──
with plt.rc_context(cfg.DARK_RC):
    fig, ax = plt.subplots(figsize=(cfg.SINGLE_W, cfg.SINGLE_H))
    _set_transparent(fig, ax)

    for n in cfg.N_ORDER:
        sub = bb[bb["n"] == n]
        grp = _group_agg(sub, "rho_target", "terminal_lambda2", ["rho_target"])
        ax.errorbar(grp["rho_target"], grp["mean"], yerr=grp["std"],
                    fmt=f"-{cfg.MARKERS[n]}", color=cfg.COLORS[n],
                    capsize=3, markersize=5, label=f"$n={n}$")

    ax.set_xlabel(r"Target density $\rho$")
    ax.set_ylabel(r"$\lambda_2$ (algebraic connectivity)")
    ax.set_title(r"Backbone+RL: $\lambda_2$ vs Target Density")
    ax.legend(ncol=3, loc="lower right", framealpha=0.0)
    ax.grid(True, alpha=cfg.DARK_RC["grid.alpha"])
    fig.tight_layout()
    cfg.save_fig(fig, OUTPUT_DIR / "fig1_lambda2_vs_rho", bbox_inches="tight", transparent=True)
    print("[OK] Fig 1 — λ₂ vs ρ (dark)")

# ── FIGURE 2 — Normalized λ₂/n vs ρ (BB+RL, all n overlaid) ──
with plt.rc_context(cfg.DARK_RC):
    fig, ax = plt.subplots(figsize=(cfg.SINGLE_W, cfg.SINGLE_H))
    _set_transparent(fig, ax)

    for n in cfg.N_ORDER:
        sub = bb[bb["n"] == n]
        grp = _group_agg(sub, "rho_target", "terminal_lambda2_norm", ["rho_target"])
        ax.errorbar(grp["rho_target"], grp["mean"], yerr=grp["std"],
                    fmt=f"-{cfg.MARKERS[n]}", color=cfg.COLORS[n],
                    capsize=3, markersize=5, label=f"$n={n}$")

    ax.set_xlabel(r"Target density $\rho$")
    ax.set_ylabel(r"$\lambda_2 / n$ (normalized)")
    ax.set_title(r"Backbone+RL: Normalized $\lambda_2$ vs Density")
    ax.legend(ncol=3, loc="lower right", framealpha=0.0)
    ax.grid(True, alpha=cfg.DARK_RC["grid.alpha"])
    fig.tight_layout()
    cfg.save_fig(fig, OUTPUT_DIR / "fig2_lambda2norm_vs_rho", bbox_inches="tight", transparent=True)
    print("[OK] Fig 2 — Normalized λ₂ (dark)")

# ── FIGURE 3 — BB+RL vs Path+RL: λ₂ comparison (all n, side-by-side) ──
with plt.rc_context(cfg.DARK_RC):
    fig, axes = plt.subplots(2, 3, figsize=(cfg.PANEL_W, cfg.PANEL_H))
    _set_transparent(fig)
    axes_flat = axes.flatten()

    for idx, n in enumerate(cfg.N_ORDER):
        ax = axes_flat[idx]

        # Path+RL (gray, dashed)
        sub_p = path_only[path_only["n"] == n]
        grp_p = _group_agg(sub_p, "rho_target", "terminal_lambda2", ["rho_target"])
        ax.plot(grp_p["rho_target"], grp_p["mean"], "--", color=cfg.THEME_LIGHT_GRAY, linewidth=1.2, label="Path+RL")

        # BB+RL (coloured, solid, with error band)
        sub_b = bb[bb["n"] == n]
        grp_b = _group_agg(sub_b, "rho_target", "terminal_lambda2", ["rho_target"])
        ax.errorbar(grp_b["rho_target"], grp_b["mean"], yerr=grp_b["std"],
                    fmt=f"-{cfg.MARKERS[n]}", color=cfg.COLORS[n],
                    capsize=3, markersize=5, label="Backbone+RL")

        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$\lambda_2$")
        ax.set_title(f"$n={n}$")
        ax.legend(fontsize=18, framealpha=0.0)
        ax.grid(True, alpha=cfg.DARK_RC["grid.alpha"])

    axes_flat[-1].set_visible(False)
    fig.suptitle(r"$\lambda_2$: Backbone+RL vs Path+RL", fontsize=34, y=1.02)
    fig.tight_layout()
    cfg.save_fig(fig, OUTPUT_DIR / "fig3_bb_vs_path", bbox_inches="tight", transparent=True)
    print("[OK] Fig 3 — BB+RL vs Path+RL (dark)")

# ── FIGURE 4 — λ₂ by backbone family (Cayley vs Envelope) ──
with plt.rc_context(cfg.DARK_RC):
    fig, axes = plt.subplots(2, 3, figsize=(cfg.PANEL_W, cfg.PANEL_H))
    _set_transparent(fig)
    axes_flat = axes.flatten()

    for idx, n in enumerate(cfg.N_ORDER):
        ax = axes_flat[idx]
        for fam in ("cayley", "envelope"):
            sub = bb[(bb["n"] == n) & (bb["init_family"] == fam)]
            if sub.empty:
                continue
            grp = _group_agg(sub, "rho_target", "terminal_lambda2", ["rho_target"])
            ax.errorbar(grp["rho_target"], grp["mean"], yerr=grp["std"],
                        fmt=f"-{cfg.FAMILY_MARKERS[fam]}", color=cfg.FAMILY_COLORS[fam],
                        capsize=3, markersize=5, label=fam.capitalize())
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$\lambda_2$")
        ax.set_title(f"$n={n}$")
        ax.legend(fontsize=18, framealpha=0.0)
        ax.grid(True, alpha=cfg.DARK_RC["grid.alpha"])

    axes_flat[-1].set_visible(False)
    fig.suptitle(r"Backbone+RL: $\lambda_2$ by Backbone Family", fontsize=34, y=1.02)
    fig.tight_layout()
    cfg.save_fig(fig, OUTPUT_DIR / "fig4_by_family", bbox_inches="tight", transparent=True)
    print("[OK] Fig 4 — By family (dark)")

# ── FIGURE 5 — Runtime scaling (BB+RL only, log-log) ──
with plt.rc_context(cfg.DARK_RC):
    fig, ax = plt.subplots(figsize=(cfg.SMALL_W, cfg.SMALL_H))
    _set_transparent(fig, ax)

    sc = summary_c[summary_c["method"] == "RL+Backbone"]
    ax.errorbar(sc["n"].values, sc["runtime_mean_over_density_s"].values,
                yerr=sc["runtime_std_over_density_s"].values,
                fmt="o-", color=cfg.THEME_CYAN, capsize=4, markersize=8, linewidth=2)
    for _, row in sc.iterrows():
        ax.annotate(f"$n={int(row['n'])}$", (row["n"], row["runtime_mean_over_density_s"]),
                    textcoords="offset points", xytext=(8, 8), fontsize=18,
                    color=cfg.DARK_RC["text.color"])

    ax.set_xlabel("Graph size $n$")
    ax.set_ylabel("Mean runtime per episode (s)")
    ax.set_title("Backbone+RL: Runtime Scaling")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(cfg.N_ORDER)
    ax.set_xticklabels([str(v) for v in cfg.N_ORDER])
    ax.grid(True, alpha=cfg.DARK_RC["grid.alpha"], which="both")

    fig.tight_layout()
    cfg.save_fig(fig, OUTPUT_DIR / "fig5_runtime", bbox_inches="tight", transparent=True)
    print("[OK] Fig 5 — Runtime (dark)")

# ── FIGURE 6 — λ₂ heatmap across (n, ρ) (BB+RL) ──
with plt.rc_context(cfg.DARK_RC):
    bb_plot = bb.copy()
    bb_plot["rho_bin"] = pd.cut(bb_plot["rho_target"], bins=20, labels=False) / 20
    bin_edges = pd.cut(bb["rho_target"], bins=20).cat.categories
    bin_centers = [(iv.left + iv.right) / 2 for iv in bin_edges]

    bb_pivot = bb_plot.pivot_table(index="n", columns="rho_bin", values="terminal_lambda2", aggfunc="mean")
    bb_pivot = bb_pivot.reindex(cfg.N_ORDER)

    fig, ax = plt.subplots(figsize=(cfg.HEATMAP_W, cfg.HEATMAP_H))
    _set_transparent(fig, ax)

    im = ax.imshow(bb_pivot.values, aspect="auto", cmap="plasma", origin="lower", interpolation="nearest")

    tick_step = max(1, len(bin_centers) // 10)
    ax.set_xticks(range(0, len(bin_centers), tick_step))
    ax.set_xticklabels([f"{bin_centers[i]:.2f}" for i in range(0, len(bin_centers), tick_step)])
    ax.set_yticks(range(len(cfg.N_ORDER)))
    ax.set_yticklabels([f"$n={int(n)}$" for n in cfg.N_ORDER])
    ax.set_xlabel(r"Target density $\rho$")
    ax.set_title(r"Mean $\lambda_2$ — Backbone+RL")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r"$\lambda_2$")
    cbar.ax.yaxis.set_tick_params(color=cfg.DARK_RC["xtick.color"])
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color=cfg.DARK_RC["xtick.color"])

    fig.tight_layout()
    cfg.save_fig(fig, OUTPUT_DIR / "fig6_heatmap", bbox_inches="tight", transparent=True)
    print("[OK] Fig 6 — Heatmap (dark)")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN RESULT  (dark-background readable, transparent bg)
# ═════════════════════════════════════════════════════════════════════════════
with plt.rc_context(cfg.DARK_RC):
    fig, ax = plt.subplots(figsize=(cfg.MAIN_W, cfg.MAIN_H))
    _set_transparent(fig, ax)

    # ── BB+RL: solid coloured lines with error bands ──
    for n in cfg.N_ORDER:
        sub = bb[bb["n"] == n]
        grp = _group_agg(sub, "rho_target", "terminal_lambda2", ["rho_target"])
        ax.errorbar(grp["rho_target"], grp["mean"], yerr=grp["std"],
                    fmt=f"-{cfg.MARKERS[n]}", color=cfg.COLORS[n],
                    capsize=3, markersize=6, linewidth=2, label=f"BB+RL $n={n}$")

    # ── Path+RL: dashed gray across all n for context ──
    for n in [8, 16, 32, 64, 128]:
        sub_p = path_only[path_only["n"] == n]
        if sub_p.empty:
            continue
        grp_p = _group_agg(sub_p, "rho_target", "terminal_lambda2", ["rho_target"])
        ax.plot(grp_p["rho_target"], grp_p["mean"], "--", color=cfg.THEME_LIGHT_GRAY, linewidth=1.0, alpha=0.55,
                label=f"Path+RL $n={n}$" if n == 8 else None)

    proxy_line = Line2D([0], [0], linestyle="--", color=cfg.THEME_LIGHT_GRAY, linewidth=1.0, alpha=0.55)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(proxy_line)
    labels.append("Path+RL (all $n$)")

    ax.set_xlabel(r"Target density $\rho$")
    ax.set_ylabel(r"Algebraic connectivity $\lambda_2$")
    ax.set_title(r"Backbone+RL vs Path+RL: $\lambda_2$ across Graph Sizes and Densities")
    ax.legend(handles=handles, labels=labels, ncol=2, loc="lower right",
              fontsize=20, framealpha=0.0, edgecolor="none")
    ax.grid(True, alpha=cfg.DARK_RC["grid.alpha"])

    fig.tight_layout()
    cfg.save_fig(fig, OUTPUT_DIR / "fig_main_result", bbox_inches="tight", transparent=True, dpi=200)
    print("[OK] MAIN RESULT — Fig main_result (white-background readable)")


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════
print("\n── All figures saved to:", OUTPUT_DIR)
print("   Dark-background:  fig1–fig6")
print("   Dark-background:  fig_main_result.svg")
