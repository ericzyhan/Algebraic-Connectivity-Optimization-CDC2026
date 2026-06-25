"""
Generate figures analyzing algebraic connectivity (λ₂) of graphs produced
by the Backbone + RL method.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent / "results_prev" / "rl"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
raw = pd.read_csv(RESULTS_DIR / "raw.csv")
summary_q = pd.read_csv(RESULTS_DIR / "summary_quality.csv")
summary_c = pd.read_csv(RESULTS_DIR / "summary_cost.csv")

print(f"Raw rows: {len(raw)}")
print(f"Methods: {raw['method'].unique()}")
print(f"n values: {sorted(raw['n'].unique())}")

# Filter to Backbone+RL only (user request: focus on BB+RL)
bb = raw[raw["method"] == "RL+Backbone"].copy()
path = raw[raw["method"] == "RL+Path"].copy()

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 1.5,
})

N_ORDER = [8, 16, 32, 64, 128]
COLORS = {8: "#4C72B0", 16: "#DD8452", 32: "#55A868", 64: "#C44E52", 128: "#8172B2"}
MARKERS = {8: "o", 16: "s", 32: "D", 64: "^", 128: "v"}

def _group_agg(df, xcol, ycol, group_cols):
    """Group and aggregate with mean±std."""
    g = df.groupby(group_cols).agg(
        mean=(ycol, "mean"),
        std=(ycol, "std"),
        count=(ycol, "count"),
    ).reset_index()
    return g


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1:  λ₂ vs ρ_target  (Backbone+RL only, per n)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=False)
axes_flat = axes.flatten()
for idx, n in enumerate(N_ORDER):
    ax = axes_flat[idx]
    sub = bb[bb["n"] == n].copy()
    grp = _group_agg(sub, "rho_target", "terminal_lambda2", ["rho_target"])
    ax.errorbar(
        grp["rho_target"], grp["mean"], yerr=grp["std"],
        fmt=f"-{MARKERS[n]}", color=COLORS[n], capsize=3, markersize=5,
        label=f"n={n}",
    )
    ax.set_xlabel(r"Target density $\rho$")
    ax.set_ylabel(r"$\lambda_2$ (algebraic connectivity)")
    ax.set_title(f"n = {n}")
    ax.legend()
    ax.grid(True, alpha=0.3)

axes_flat[-1].set_visible(False)
fig.suptitle("Backbone+RL: Algebraic Connectivity $\lambda_2$ vs Target Density", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig1_lambda2_vs_rho.png", bbox_inches="tight")
print("[OK] Figure 1 saved")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2:  Normalized λ₂/n vs ρ_target  (Backbone+RL)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5.5))
for n in N_ORDER:
    sub = bb[bb["n"] == n].copy()
    grp = _group_agg(sub, "rho_target", "terminal_lambda2_norm", ["rho_target"])
    ax.errorbar(
        grp["rho_target"], grp["mean"], yerr=grp["std"],
        fmt=f"-{MARKERS[n]}", color=COLORS[n], capsize=3, markersize=5,
        label=f"n={n}",
    )
ax.set_xlabel(r"Target density $\rho$")
ax.set_ylabel(r"$\lambda_2 / n$ (normalized algebraic connectivity)")
ax.set_title("Backbone+RL: Normalized $\lambda_2$ vs Target Density")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig2_lambda2norm_vs_rho.png", bbox_inches="tight")
print("[OK] Figure 2 saved")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3:  λ₂ improvement (final - init)  — backbone+RL only
# ═══════════════════════════════════════════════════════════════════════════
bb["lambda2_improvement"] = bb["terminal_lambda2"] - bb["init_lambda2"]
bb["lambda2_improvement_pct"] = (
    (bb["terminal_lambda2"] - bb["init_lambda2"]) / bb["init_lambda2"].clip(lower=1e-10) * 100
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: absolute improvement
ax = axes[0]
for n in N_ORDER:
    sub = bb[bb["n"] == n].copy()
    grp = _group_agg(sub, "rho_target", "lambda2_improvement", ["rho_target"])
    ax.errorbar(
        grp["rho_target"], grp["mean"], yerr=grp["std"],
        fmt=f"-{MARKERS[n]}", color=COLORS[n], capsize=3, markersize=5,
        label=f"n={n}",
    )
ax.set_xlabel(r"Target density $\rho$")
ax.set_ylabel(r"$\Delta\lambda_2 = \lambda_2^{\text{final}} - \lambda_2^{\text{init}}$")
ax.set_title("Absolute $\lambda_2$ Improvement (Backbone+RL)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Right: relative (percentage) improvement
ax = axes[1]
for n in N_ORDER:
    sub = bb[bb["n"] == n].copy()
    grp = _group_agg(sub, "rho_target", "lambda2_improvement_pct", ["rho_target"])
    ax.errorbar(
        grp["rho_target"], grp["mean"], yerr=grp["std"],
        fmt=f"-{MARKERS[n]}", color=COLORS[n], capsize=3, markersize=5,
        label=f"n={n}",
    )
ax.set_xlabel(r"Target density $\rho$")
ax.set_ylabel(r"$\Delta\lambda_2$ / $\lambda_2^{\text{init}}$ (%)")
ax.set_title("Relative $\lambda_2$ Improvement (Backbone+RL)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("Backbone+RL: $\lambda_2$ Improvement over Initial Backbone", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig3_lambda2_improvement.png", bbox_inches="tight")
print("[OK] Figure 3 saved")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4:  Side-by-side: λ₂ Backbone+RL vs RL+Path (for reference)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes_flat = axes.flatten()
for idx, n in enumerate(N_ORDER):
    ax = axes_flat[idx]
    for label, grp_df, color_theme in [
        ("RL+Path", path, "#888888"),
        ("RL+Backbone", bb, COLORS[n]),
    ]:
        sub = grp_df[grp_df["n"] == n]
        grp = _group_agg(sub, "rho_target", "terminal_lambda2", ["rho_target"])
        ax.plot(
            grp["rho_target"], grp["mean"],
            marker=MARKERS[n] if label == "RL+Backbone" else ".",
            color=color_theme, label=label, alpha=0.85,
        )
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\lambda_2$")
    ax.set_title(f"n={n}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes_flat[-1].set_visible(False)
fig.suptitle(r"$\lambda_2$ Comparison: Backbone+RL vs Path+RL", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig4_bb_vs_path_lambda2.png", bbox_inches="tight")
print("[OK] Figure 4 saved")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5:  λ₂ heatmap (Backbone+RL) across (n, ρ) — binned for clarity
# ═══════════════════════════════════════════════════════════════════════════
n_unique = sorted(bb["n"].unique())
# Bin rho into ~20 equal-width bins for readability
bb_plot = bb.copy()
bb_plot["rho_bin"] = pd.cut(bb_plot["rho_target"], bins=20, labels=False) / 20
# Take bin centers for labeling
bin_edges = pd.cut(bb["rho_target"], bins=20).cat.categories
bin_centers = [ (interval.left + interval.right) / 2 for interval in bin_edges ]

bb_pivot = bb_plot.pivot_table(
    index="n", columns="rho_bin", values="terminal_lambda2", aggfunc="mean"
)
# Ensure rows are sorted by n
bb_pivot = bb_pivot.reindex(n_unique)

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(bb_pivot.values, aspect="auto", cmap="viridis",
               origin="lower", interpolation="nearest")
# Label every 5th bin to avoid crowding
tick_step = max(1, len(bin_centers) // 10)
ax.set_xticks(range(0, len(bin_centers), tick_step))
ax.set_xticklabels([f"{bin_centers[i]:.2f}" for i in range(0, len(bin_centers), tick_step)])
ax.set_yticks(range(len(n_unique)))
ax.set_yticklabels([f"n={int(n)}" for n in n_unique])
ax.set_xlabel(r"Target density $\rho$")
ax.set_title(r"Mean $\lambda_2$ across (n, $\rho$) — Backbone+RL")
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label(r"$\lambda_2$")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig5_heatmap_lambda2.png", bbox_inches="tight")
print("[OK] Figure 5 saved")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6:  Runtime cost comparison
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
for _, row in summary_c.iterrows():
    n = int(row["n"])
    meth = row["method"]
    if meth == "RL+Path":
        continue  # skip path for now since user said no benchmarking
    ax.errorbar(
        n, row["runtime_mean_over_density_s"],
        yerr=row["runtime_std_over_density_s"],
        fmt=MARKERS.get(n, "o"), color=COLORS.get(n, "#333"),
        capsize=3, markersize=8, label=f"n={n}",
    )
ax.set_xlabel("Graph size $n$")
ax.set_ylabel("Runtime (s)")
ax.set_title("Backbone+RL: Runtime per Episode vs $n$")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(N_ORDER)
ax.set_xticklabels([str(v) for v in N_ORDER])
ax.legend()
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig6_runtime.png", bbox_inches="tight")
print("[OK] Figure 6 saved")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7:  λ₂ by backbone family (Cayley vs Envelope) for BB+RL
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes_flat = axes.flatten()
FAMILY_COLORS = {"cayley": "#E24A33", "envelope": "#348ABD"}
FAMILY_MARKERS = {"cayley": "^", "envelope": "s"}

for idx, n in enumerate(N_ORDER):
    ax = axes_flat[idx]
    sub = bb[bb["n"] == n].copy()
    for fam in ["cayley", "envelope"]:
        fam_df = sub[sub["init_family"] == fam]
        if fam_df.empty:
            continue
        grp = _group_agg(fam_df, "rho_target", "terminal_lambda2", ["rho_target"])
        ax.errorbar(
            grp["rho_target"], grp["mean"], yerr=grp["std"],
            fmt=f"-{FAMILY_MARKERS[fam]}", color=FAMILY_COLORS[fam],
            capsize=3, markersize=5, label=f"{fam} backbone",
        )
    ax.set_xlabel(r"Target density $\rho$")
    ax.set_ylabel(r"$\lambda_2$")
    ax.set_title(f"n = {n}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes_flat[-1].set_visible(False)
fig.suptitle(r"BB+RL: $\lambda_2$ by Backbone Family (Cayley vs Envelope)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig7_lambda2_by_family.png", bbox_inches="tight")
print("[OK] Figure 7 saved")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8:  Summary panel — stacked multi-panel overview for BB+RL
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 12))

# 8a: λ₂ vs ρ (all n, one panel)
ax1 = fig.add_subplot(2, 3, 1)
for n in N_ORDER:
    sub = bb[bb["n"] == n]
    grp = _group_agg(sub, "rho_target", "terminal_lambda2", ["rho_target"])
    ax1.plot(grp["rho_target"], grp["mean"],
             marker=MARKERS[n], color=COLORS[n], label=f"n={n}")
ax1.set_xlabel(r"$\rho$")
ax1.set_ylabel(r"$\lambda_2$")
ax1.set_title(r"$\lambda_2$ vs $\rho$")
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)

# 8b: Normalized λ₂
ax2 = fig.add_subplot(2, 3, 2)
for n in N_ORDER:
    sub = bb[bb["n"] == n]
    grp = _group_agg(sub, "rho_target", "terminal_lambda2_norm", ["rho_target"])
    ax2.plot(grp["rho_target"], grp["mean"],
             marker=MARKERS[n], color=COLORS[n], label=f"n={n}")
ax2.set_xlabel(r"$\rho$")
ax2.set_ylabel(r"$\lambda_2 / n$")
ax2.set_title(r"Normalized $\lambda_2$")
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

# 8c: λ₂ improvement %
ax3 = fig.add_subplot(2, 3, 3)
for n in N_ORDER:
    sub = bb[bb["n"] == n]
    sub = sub.copy()
    sub["imp_pct"] = (sub["terminal_lambda2"] - sub["init_lambda2"]) / sub["init_lambda2"].clip(lower=1e-10) * 100
    grp = _group_agg(sub, "rho_target", "imp_pct", ["rho_target"])
    ax3.plot(grp["rho_target"], grp["mean"],
             marker=MARKERS[n], color=COLORS[n], label=f"n={n}")
ax3.set_xlabel(r"$\rho$")
ax3.set_ylabel(r"$\Delta\lambda_2 / \lambda_2^{\text{init}}$ (%)")
ax3.set_title(r"Relative $\lambda_2$ Improvement")
ax3.legend(fontsize=7)
ax3.grid(True, alpha=0.3)

# 8d: Episode length (# edges added)
ax4 = fig.add_subplot(2, 3, 4)
for n in N_ORDER:
    sub = bb[bb["n"] == n]
    grp = _group_agg(sub, "rho_target", "episode_len", ["rho_target"])
    ax4.plot(grp["rho_target"], grp["mean"],
             marker=MARKERS[n], color=COLORS[n], label=f"n={n}")
ax4.set_xlabel(r"$\rho$")
ax4.set_ylabel("Edges added by RL")
ax4.set_title("Episode Length (# RL steps)")
ax4.legend(fontsize=7)
ax4.grid(True, alpha=0.3)

# 8e: Initial λ₂ from backbone
ax5 = fig.add_subplot(2, 3, 5)
for n in N_ORDER:
    sub = bb[bb["n"] == n]
    grp = _group_agg(sub, "rho_target", "init_lambda2", ["rho_target"])
    ax5.plot(grp["rho_target"], grp["mean"],
             marker=MARKERS[n], color=COLORS[n], label=f"n={n}")
ax5.set_xlabel(r"$\rho$")
ax5.set_ylabel(r"$\lambda_2^{\text{init}}$")
ax5.set_title("Initial Backbone $\lambda_2$")
ax5.legend(fontsize=7)
ax5.grid(True, alpha=0.3)

# 8f: Initial edges count
ax6 = fig.add_subplot(2, 3, 6)
for n in N_ORDER:
    sub = bb[bb["n"] == n]
    grp = _group_agg(sub, "rho_target", "init_edges", ["rho_target"])
    ax6.plot(grp["rho_target"], grp["mean"],
             marker=MARKERS[n], color=COLORS[n], label=f"n={n}")
ax6.set_xlabel(r"$\rho$")
ax6.set_ylabel("Backbone edges")
ax6.set_title("Initial Edge Count from Backbone")
ax6.legend(fontsize=7)
ax6.grid(True, alpha=0.3)

fig.suptitle("Backbone+RL Performance Summary", fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig8_summary_panel.png", bbox_inches="tight")
print("[OK] Figure 8 saved")


print("\n── All figures saved to:", OUTPUT_DIR)
