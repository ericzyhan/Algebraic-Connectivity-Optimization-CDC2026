"""Figures for the old-model vs analytic-model head-to-head.

Encoding is composite rather than four flat colors: **hue carries the model**
(blue = old, orange = analytic) and **line style + marker carry the init mode**
(solid/circle = backbone, dashed/triangle = path). Two hues instead of four keeps
the palette inside the colorblind-safe gate and, more importantly, matches the
data's actual 2x2 structure.

Figures produced:
  fig1_lambda2_vs_density  -- the headline: terminal lambda2 vs connectivity
                              density, one panel per n, all four arms.
  fig2_delta_vs_density    -- paired delta (analytic - old) vs density on matched
                              (n, m_target, seed) points, diverging about zero.
  fig3_runtime_vs_n        -- runtime scaling, log-log.
  fig4_win_tie_loss        -- diverging stacked bars, analytic vs old per n.

Usage (from the project root):
    python plot_head_to_head.py                # light figures
    python plot_head_to_head.py --mode dark    # dark-surface variants
    python plot_head_to_head.py --mode both
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analytic_tractable_test_deepseek.run_path_vs_backbone_experiment import (
    _to_float,
    _to_int,
)

_DEFAULT_OLD_CSV = (
    "../../Old Model/Algebraic-Connectivity-Optimization-CDC2026/results/rl/raw.csv"
)

N_ORDER = [8, 16, 32, 64, 128]
METHODS = ["RL+Backbone", "RL+Path"]

# Reference palette (references/palette.md). Light and dark are selected steps of
# the same hues, not an automatic flip.
THEME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "old": "#2a78d6",   # categorical slot 1, blue
        "new": "#eb6834",   # categorical slot 2, orange
        "pos": "#2a78d6",   # diverging warm/cool poles
        "neg": "#e34948",
        "neutral": "#f0efec",
        "neutral_edge": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "old": "#3987e5",
        "new": "#d95926",
        "pos": "#3987e5",
        "neg": "#e66767",
        "neutral": "#383835",
        "neutral_edge": "#52514e",
    },
}

# Init mode -> line style / marker. This is the secondary channel.
STYLE = {
    "RL+Backbone": {"ls": "-", "marker": "o", "fill": "full"},
    "RL+Path": {"ls": "--", "marker": "^", "fill": "none"},
}
SHORT = {"RL+Backbone": "Backbone", "RL+Path": "Path"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-csv", type=str, default=_DEFAULT_OLD_CSV)
    p.add_argument("--new-csv", type=str, default="results/rl_analytic/raw.csv")
    p.add_argument(
        "--out-dir", type=str, default="results/head_to_head_old_vs_analytic/figures"
    )
    p.add_argument("--mode", type=str, default="both", choices=["light", "dark", "both"])
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    # utf-8-sig: the old-model raw.csv carries a BOM, the new one does not.
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _rc(t: dict[str, str]) -> dict[str, object]:
    return {
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "text.color": t["ink"],
        "axes.labelcolor": t["ink2"],
        "axes.edgecolor": t["axis"],
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"],
        "grid.color": t["grid"],
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "legend.frameon": False,
    }


# ── data shaping ─────────────────────────────────────────────────────────────


def _curve(rows: list[dict[str, str]], method: str, n: int):
    """Mean and std of terminal_lambda2 over seeds, as a function of density."""
    by_rho: dict[float, list[float]] = {}
    for r in rows:
        if str(r["method"]) != method or _to_int(r["n"]) != n:
            continue
        by_rho.setdefault(round(_to_float(r["rho_target"]), 9), []).append(
            _to_float(r["terminal_lambda2"])
        )
    xs = np.asarray(sorted(by_rho), dtype=np.float64)
    mean = np.asarray([np.mean(by_rho[float(x)]) for x in xs], dtype=np.float64)
    std = np.asarray([np.std(by_rho[float(x)]) for x in xs], dtype=np.float64)
    return xs, mean, std


def _index(rows: list[dict[str, str]]):
    """(method, n, m_target, seed) -> record, collapsing duplicate keys by mean.

    The analytic n=8 shards map several rho values onto one m_target, so
    duplicates exist there; their lambda2 values are identical and only runtimes
    vary, so a mean is the neutral collapse.
    """
    acc: dict[tuple[str, int, int, int], dict[str, list[float]]] = {}
    for r in rows:
        k = (str(r["method"]), _to_int(r["n"]), _to_int(r["m_target"]), _to_int(r["seed"]))
        slot = acc.setdefault(k, {"l2": [], "rt": [], "rho": []})
        slot["l2"].append(_to_float(r["terminal_lambda2"]))
        slot["rt"].append(_to_float(r["runtime_total_sec"]))
        slot["rho"].append(_to_float(r["rho_target"]))
    return {
        k: {kk: float(np.mean(np.asarray(vv, dtype=np.float64))) for kk, vv in v.items()}
        for k, v in acc.items()
    }


def _matched_delta(old_idx, new_idx, method: str, n: int):
    """Paired analytic-minus-old lambda2, averaged over seeds, vs density.

    x uses the old run's rho_target: the two runs agree on the (n, m_target) ->
    rho mapping to within 0.023 (pure grid quantization), and the old grid has no
    duplicate m per rho.
    """
    by_m: dict[int, list[float]] = {}
    rho_of_m: dict[int, float] = {}
    for k, o in old_idx.items():
        if k[0] != method or k[1] != n or k not in new_idx:
            continue
        by_m.setdefault(k[2], []).append(new_idx[k]["l2"] - o["l2"])
        rho_of_m[k[2]] = o["rho"]
    ms = sorted(by_m)
    xs = np.asarray([rho_of_m[m] for m in ms], dtype=np.float64)
    d = np.asarray([np.mean(by_m[m]) for m in ms], dtype=np.float64)
    return xs, d


def _runtime_by_n(idx, method: str):
    """Mean over seeds per density point, then mean/std over density points."""
    means, stds = [], []
    for n in N_ORDER:
        per_point: dict[int, list[float]] = {}
        for k, v in idx.items():
            if k[0] == method and k[1] == n:
                per_point.setdefault(k[2], []).append(v["rt"])
        pts = np.asarray(
            [np.mean(np.asarray(v, dtype=np.float64)) for v in per_point.values()],
            dtype=np.float64,
        )
        means.append(float(np.mean(pts)))
        stds.append(float(np.std(pts)))
    return np.asarray(means), np.asarray(stds)


def _read_matched_summary(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    out: dict[tuple[str, int], dict[str, float]] = {}
    for r in _read_rows(path):
        out[(str(r["method"]), _to_int(r["n"]))] = {
            "win": _to_float(r["win_pct_new_vs_old"]),
            "tie": _to_float(r["tie_pct_new_vs_old"]),
            "loss": _to_float(r["loss_pct_new_vs_old"]),
        }
    return out


# ── figures ──────────────────────────────────────────────────────────────────


def _model_legend(t, extra=()):
    handles = [
        Line2D([], [], color=t["old"], lw=2.4, label="Old model"),
        Line2D([], [], color=t["new"], lw=2.4, label="Analytic model"),
        Line2D([], [], color=t["muted"], lw=2.0, ls="-", marker="o", ms=7,
               label="Backbone init"),
        Line2D([], [], color=t["muted"], lw=2.0, ls="--", marker="^", ms=7,
               markerfacecolor="none", label="Path init"),
    ]
    return list(handles) + list(extra)


def _label_max_separation(ax, old, new, t, n: int, method: str) -> None:
    """Direct-label both models where their curves are furthest apart."""
    xo, mo, _ = _curve(old, method, n)
    xn, mn, _ = _curve(new, method, n)
    if xo.size == 0 or xn.size == 0:
        return
    mn_on_xo = np.interp(xo, xn, mn)
    j = int(np.argmax(np.abs(mn_on_xo - mo)))
    x = float(xo[j])
    y_old, y_new = float(mo[j]), float(mn_on_xo[j])
    for label, y, key in (("Old", y_old, "old"), ("Analytic", y_new, "new")):
        dy = 15 if y >= (y_old + y_new) / 2 else -15
        ax.annotate(
            label, xy=(x, y), xytext=(14, dy), textcoords="offset points",
            color=t[key], fontsize=10.5, fontweight="bold", ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", color=t[key], lw=1.0,
                            shrinkA=0, shrinkB=3, alpha=0.7),
        )


def fig1_lambda2_vs_density(old, new, t, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    flat = axes.ravel()

    for i, n in enumerate(N_ORDER):
        ax = flat[i]
        for rows, key in ((old, "old"), (new, "new")):
            for method in METHODS:
                xs, mean, std = _curve(rows, method, n)
                if xs.size == 0:
                    continue
                s = STYLE[method]
                ax.fill_between(xs, mean - std, mean + std, color=t[key], alpha=0.13,
                                linewidth=0)
                ax.plot(
                    xs, mean, color=t[key], ls=s["ls"], lw=2.0,
                    marker=s["marker"], markersize=8, markevery=max(1, xs.size // 6),
                    markerfacecolor=(t["surface"] if s["fill"] == "none" else t[key]),
                    markeredgecolor=t[key], markeredgewidth=1.6,
                    solid_capstyle="round",
                )
        ax.set_title(f"$n = {n}$", color=t["ink"], loc="left", pad=8)
        ax.set_xlabel(r"connectivity density  $\rho$")
        ax.set_ylabel(r"terminal  $\lambda_2$")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.9)
        ax.set_axisbelow(True)

    # Direct labels so identity never rests on color alone. Anchor them at the
    # density where the two models separate most, otherwise they land on top of
    # each other -- the curves coincide almost everywhere.
    _label_max_separation(flat[1], old, new, t, n=16, method="RL+Backbone")

    legend_ax = flat[5]
    legend_ax.axis("off")
    legend_ax.legend(
        handles=_model_legend(t), loc="center left", fontsize=12,
        labelcolor=t["ink2"], handlelength=2.8, borderaxespad=0,
    )
    legend_ax.text(
        0.0, 0.93,
        "Mean over 5 seeds; band is $\\pm1$ s.d.\nEach run is shown on its own\ndensity grid.",
        transform=legend_ax.transAxes, color=t["muted"], fontsize=10.5, va="top",
    )

    fig.suptitle(
        r"Terminal $\lambda_2$ vs connectivity density — old model vs analytic model",
        color=t["ink"], fontsize=15, x=0.011, ha="left", y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fig2_delta_vs_density(old_idx, new_idx, t, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(17.5, 7.2), sharex=True, sharey="row")

    # One y-scale per row so panels within a row are directly comparable; the two
    # rows differ by ~4x in delta magnitude, so a single shared scale would flatten
    # the backbone row into a line.
    row_lim = []
    for method in METHODS:
        peak = max(
            (float(np.max(np.abs(d))) for _x, d in
             (_matched_delta(old_idx, new_idx, method, n) for n in N_ORDER) if d.size),
            default=1.0,
        )
        row_lim.append(peak * 1.15)

    for row, method in enumerate(METHODS):
        for col, n in enumerate(N_ORDER):
            ax = axes[row, col]
            xs, d = _matched_delta(old_idx, new_idx, method, n)
            ax.axhline(0, color=t["axis"], lw=1.2, zorder=1)
            if xs.size:
                ax.fill_between(xs, 0, d, where=(d >= 0), color=t["pos"], alpha=0.30,
                                interpolate=True, linewidth=0, zorder=2)
                ax.fill_between(xs, 0, d, where=(d < 0), color=t["neg"], alpha=0.30,
                                interpolate=True, linewidth=0, zorder=2)
                ax.plot(xs, d, color=t["ink2"], lw=1.4, zorder=3)
            ax.set_ylim(-row_lim[row], row_lim[row])
            if row == 0:
                ax.set_title(f"$n = {n}$", color=t["ink"], loc="left", pad=8)
            if row == 1:
                ax.set_xlabel(r"density  $\rho$")
            if col == 0:
                ax.set_ylabel(f"{SHORT[method]} init\n" r"$\Delta\lambda_2$", color=t["ink"])
            ax.set_xlim(0, 1)
            ax.set_axisbelow(True)

    handles = [
        Patch(facecolor=t["pos"], alpha=0.30, label="analytic higher $\\lambda_2$"),
        Patch(facecolor=t["neg"], alpha=0.30, label="old higher $\\lambda_2$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11.5,
               labelcolor=t["ink2"], bbox_to_anchor=(0.5, -0.012))
    fig.suptitle(
        r"Paired $\Delta\lambda_2$ (analytic $-$ old) on matched $(n, m, \mathrm{seed})$ points",
        color=t["ink"], fontsize=15, x=0.011, ha="left", y=0.985,
    )
    fig.text(
        0.011, 0.935,
        "Matched density points per $n$: 20 / 94 / 50 / 48 / 44. Mean over 5 seeds.",
        color=t["muted"], fontsize=10.5, ha="left",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.925))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fig3_runtime_vs_n(old_idx, new_idx, t, out: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ns = np.asarray(N_ORDER, dtype=np.float64)

    for idx, key in ((old_idx, "old"), (new_idx, "new")):
        for method in METHODS:
            mean, std = _runtime_by_n(idx, method)
            s = STYLE[method]
            # Clip the lower whisker: on a log axis a symmetric +/-1 s.d. bar runs
            # off the bottom wherever the s.d. approaches the mean.
            lower = np.minimum(std, mean * 0.85)
            ax.errorbar(
                ns, mean, yerr=np.vstack([lower, std]), color=t[key], ls=s["ls"], lw=2.0,
                marker=s["marker"], markersize=9,
                markerfacecolor=(t["surface"] if s["fill"] == "none" else t[key]),
                markeredgecolor=t[key], markeredgewidth=1.8,
                capsize=4, elinewidth=1.2, alpha=0.98,
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(N_ORDER)
    ax.set_xticklabels([str(v) for v in N_ORDER])
    ax.set_xlabel("graph size  $n$")
    ax.set_ylabel("runtime per density point  (s)")
    ax.grid(True, which="both", alpha=0.9)
    ax.set_axisbelow(True)
    ax.legend(handles=_model_legend(t), loc="upper left", fontsize=11,
              labelcolor=t["ink2"], handlelength=2.8)

    # State the actual measured ratio rather than a round claim.
    bb, _ = _runtime_by_n(new_idx, "RL+Backbone")
    pa, _ = _runtime_by_n(new_idx, "RL+Path")
    lo, hi = pa[0] / bb[0], pa[-1] / bb[-1]
    ax.set_title(
        f"Runtime scaling — backbone init is {lo:.0f}x cheaper at $n=8$, "
        f"{hi:.0f}x at $n=128$",
        color=t["ink"], loc="left", pad=12, fontsize=14,
    )
    fig.text(0.005, 0.005,
             "Mean over density points; bars are $\\pm1$ s.d. across points "
             "(lower whisker clipped at the axis).",
             color=t["muted"], fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fig4_win_tie_loss(summary, t, out: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    labels, y = [], []
    pos = 0.0
    for method in METHODS:
        for n in N_ORDER:
            labels.append(f"$n={n}$")
            y.append(pos)
            pos += 1.0
        pos += 0.9  # gap between the two init-mode groups

    for i, (method, n) in enumerate([(m, n) for m in METHODS for n in N_ORDER]):
        rec = summary.get((method, n))
        if rec is None:
            continue
        half = rec["tie"] / 2.0
        ax.barh(y[i], rec["tie"], left=-half, height=0.62, color=t["neutral"],
                edgecolor=t["neutral_edge"], linewidth=1.0, zorder=2)
        ax.barh(y[i], rec["win"], left=half, height=0.62, color=t["pos"], zorder=2)
        ax.barh(y[i], -rec["loss"], left=-half, height=0.62, color=t["neg"], zorder=2)

        if rec["win"] >= 6:
            ax.text(half + rec["win"] / 2, y[i], f"{rec['win']:.0f}", ha="center",
                    va="center", color=t["surface"], fontsize=10, fontweight="bold",
                    zorder=3)
        if rec["loss"] >= 6:
            ax.text(-half - rec["loss"] / 2, y[i], f"{rec['loss']:.0f}", ha="center",
                    va="center", color=t["surface"], fontsize=10, fontweight="bold",
                    zorder=3)
        if rec["tie"] >= 10:
            ax.text(0, y[i], f"{rec['tie']:.0f}", ha="center", va="center",
                    color=t["ink2"], fontsize=10, fontweight="bold", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color=t["axis"], lw=1.2, zorder=1)
    ax.set_xlabel("share of matched points  (%)")
    ax.xaxis.set_major_formatter(lambda v, _p: f"{abs(v):.0f}")
    ax.grid(True, axis="x", alpha=0.9)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)

    for method, anchor in zip(METHODS, (y[0], y[len(N_ORDER)])):
        ax.text(-0.135, anchor - 0.75, f"{SHORT[method]} init", transform=ax.get_yaxis_transform(),
                color=t["ink"], fontsize=12, fontweight="bold", ha="left")

    handles = [
        Patch(facecolor=t["neg"], label="old wins"),
        Patch(facecolor=t["neutral"], edgecolor=t["neutral_edge"], label="tie"),
        Patch(facecolor=t["pos"], label="analytic wins"),
    ]
    # Legend below the plot: inside the axes it lands on the n=128 path bar.
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11,
               labelcolor=t["ink2"], bbox_to_anchor=(0.5, -0.02))
    ax.set_title("Analytic vs old, point by point", color=t["ink"], loc="left",
                 pad=26, fontsize=14)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    old_csv = Path(args.old_csv).resolve()
    new_csv = Path(args.new_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    old = _read_rows(old_csv)
    new = _read_rows(new_csv)
    print(f"[plot] old: {len(old)} rows from {old_csv}")
    print(f"[plot] new: {len(new)} rows from {new_csv}")

    old_idx = _index(old)
    new_idx = _index(new)

    summary_path = out_dir.parent / "head_to_head_matched.csv"
    summary = _read_matched_summary(summary_path) if summary_path.exists() else {}
    if not summary:
        print(f"[plot] {summary_path} not found -- skipping fig4. "
              f"Run compare_old_vs_analytic.py first.")

    modes = ["light", "dark"] if args.mode == "both" else [args.mode]
    for mode in modes:
        t = THEME[mode]
        suffix = "" if mode == "light" else "_dark"
        with plt.rc_context(_rc(t)):
            f1 = out_dir / f"fig1_lambda2_vs_density{suffix}.png"
            fig1_lambda2_vs_density(old, new, t, f1, args.dpi)
            print(f"Wrote: {f1}")

            f2 = out_dir / f"fig2_delta_vs_density{suffix}.png"
            fig2_delta_vs_density(old_idx, new_idx, t, f2, args.dpi)
            print(f"Wrote: {f2}")

            f3 = out_dir / f"fig3_runtime_vs_n{suffix}.png"
            fig3_runtime_vs_n(old_idx, new_idx, t, f3, args.dpi)
            print(f"Wrote: {f3}")

            if summary:
                f4 = out_dir / f"fig4_win_tie_loss{suffix}.png"
                fig4_win_tie_loss(summary, t, f4, args.dpi)
                print(f"Wrote: {f4}")


if __name__ == "__main__":
    main()
