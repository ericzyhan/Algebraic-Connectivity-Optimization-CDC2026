"""Comparison figures + summary CSVs for the new-model vs old-model lite trials.

Reads, per arm: metrics.csv (training curves) and results/{raw,
density_bucket_summary, summary_cost, summary_quality}.csv, then writes a
figure set and two summary tables under lite_trials_new/comparison/.

The head-to-head tables pair arms exactly: same n, same target edge count, same
evaluation seed, same init mode. Because evaluate.py is byte-identical across
the two code bases and every arm shares one config template, a paired delta is
attributable to the env/model/reward differences and nothing else.

lambda_2 is reported raw throughout -- algebraic connectivity is an absolute
measure, not something to divide by n.

Usage (from project root, after compare_train.py + compare_experiment.py):
    python -m lite_trials_new.compare_visualize
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from lite_trials_new.arms import LITE_ROOT  # noqa: E402

MANIFEST_PATH = LITE_ROOT / "manifest.json"
COMPARISON_DIR = LITE_ROOT / "comparison"

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    try:
        plt.style.use("ggplot")
    except Exception:
        pass

# Categorical hues in fixed order, assigned by an arm's position in
# manifest.json -- never cycled or reassigned by filtering. Same validated set
# used by lite_trials/lite_sweep_visualize.py.
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
# Secondary encoding, so arm identity never rests on color alone.
_ARM_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

_GRID_KW = dict(alpha=0.3, linewidth=0.6)
_INK = "#2b2b2b"
_MUTED = "#6b6b6b"
_METHOD_LINESTYLE = {"RL+Path": "-", "RL+Backbone": "--"}
_BUCKET_ORDER = ["sparse", "medium", "dense"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline", type=str, default="old_model",
        help="Arm tag used as the head-to-head baseline (default: old_model).",
    )
    return p.parse_args()


def _arm_color(idx: int) -> str:
    return _CATEGORICAL_HUES[idx % len(_CATEGORICAL_HUES)]


def _arm_marker(idx: int) -> str:
    return _ARM_MARKERS[idx % len(_ARM_MARKERS)]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {path}")


def _load_manifest() -> list[dict[str, object]]:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found: {MANIFEST_PATH}. Run compare_train.py first.")
        sys.exit(1)
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _load_arm_data(entry: dict[str, object]) -> dict[str, list[dict[str, str]]]:
    run_dir = Path(str(entry["run_dir"]))
    return {
        "metrics": _read_csv(run_dir / "metrics.csv"),
        "raw": _read_csv(run_dir / "results" / "raw.csv"),
        "buckets": _read_csv(run_dir / "results" / "density_bucket_summary.csv"),
        "cost": _read_csv(run_dir / "results" / "summary_cost.csv"),
        "quality": _read_csv(run_dir / "results" / "summary_quality.csv"),
    }


def _bucket_for_rho(rho: float) -> str:
    if rho < 1.0 / 3.0:
        return "sparse"
    if rho < 2.0 / 3.0:
        return "medium"
    return "dense"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_training_curves(manifest, data) -> None:
    """Learning curves, kept honest about a metrics.csv asymmetry.

    The old model's ``train.py`` logs no raw-lambda2 eval column at all -- only
    ``best_eval_metric``, which is normalized (lambda2/n). The new model logs
    both. So the top panel uses ``best_eval_metric`` (the one metric both arms
    log, same definition, directly comparable) and the middle panel shows raw
    eval lambda2 only for the arms that record it. Raw lambda2 for *both* arms
    lives in the evaluation figures, which run the shared evaluate.py.
    """
    if not any(d["metrics"] for d in data):
        print("No metrics.csv found across arms; skipping training_curves.png")
        return

    has_raw = [any(r.get("eval_terminal_lambda2_mean") for r in d["metrics"]) for d in data]
    missing_raw = [str(e["label"]) for e, ok in zip(manifest, has_raw) if not ok]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    for i, (entry, d) in enumerate(zip(manifest, data)):
        rows = d["metrics"]
        if not rows:
            continue
        color, marker = _arm_color(i), _arm_marker(i)
        label = str(entry["label"])
        steps = [int(float(r["global_env_steps"])) for r in rows]

        ax1.plot(steps, [float(r.get("best_eval_metric", 0) or 0) for r in rows],
                 color=color, marker=marker, markersize=5, linewidth=2, alpha=0.9, label=label)
        if has_raw[i]:
            ax2.plot(steps, [float(r.get("eval_terminal_lambda2_mean", 0) or 0) for r in rows],
                     color=color, marker=marker, markersize=5, linewidth=2, alpha=0.9, label=label)
        ax3.plot(steps, [float(r.get("mean_episode_return", 0) or 0) for r in rows],
                 color=color, marker=marker, markersize=5, linewidth=2, alpha=0.9, label=label)

    ax1.set_ylabel("Best eval lambda2 / n so far", color=_INK)
    ax1.set_title("Learning progress (normalized -- the one eval metric both code bases log)",
                  fontsize=11, color=_INK)
    ax1.grid(True, **_GRID_KW)
    ax1.legend(fontsize=9, loc="best")

    ax2.set_ylabel("Eval terminal lambda2 (raw)", color=_INK)
    subtitle = "Evaluation: raw terminal algebraic connectivity"
    if missing_raw:
        subtitle += f"  --  not logged by: {', '.join(missing_raw)}"
    ax2.set_title(subtitle, fontsize=11, color=_INK)
    ax2.grid(True, **_GRID_KW)
    if any(has_raw):
        ax2.legend(fontsize=9, loc="best")
    else:
        ax2.text(0.5, 0.5, "no arm logs raw eval lambda2", ha="center", va="center",
                 transform=ax2.transAxes, color=_MUTED, fontsize=10)

    ax3.set_xlabel("Environment steps", color=_INK)
    ax3.set_ylabel("Mean episode return", color=_MUTED)
    ax3.set_title("Training return -- NOT comparable across arms (different reward functions)",
                  fontsize=11, color=_MUTED)
    ax3.grid(True, **_GRID_KW)
    ax3.legend(fontsize=9, loc="best")

    fig.suptitle("Lite training curves: new analytic model vs old model", fontsize=14)
    fig.tight_layout()
    out = COMPARISON_DIR / "training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def _binned_mean_std(points, n_bins: int = 10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rhos = np.asarray([p[0] for p in points], dtype=np.float64)
    lams = np.asarray([p[1] for p in points], dtype=np.float64)
    centers, means, stds = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (rhos >= lo) & (rhos < hi) if hi < 1.0 else (rhos >= lo) & (rhos <= hi)
        if not np.any(mask):
            continue
        centers.append(0.5 * (lo + hi))
        means.append(float(np.mean(lams[mask])))
        stds.append(float(np.std(lams[mask])))
    return np.asarray(centers), np.asarray(means), np.asarray(stds)


def plot_lambda2_vs_density(manifest, data) -> None:
    n_values = sorted({int(float(r["n"])) for d in data for r in d["raw"]})
    if not n_values:
        print("No raw.csv data found; skipping lambda2_vs_density.png")
        return

    ncols = min(2, len(n_values))
    nrows = int(np.ceil(len(n_values) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.2 * nrows), squeeze=False)

    for panel, n_val in enumerate(n_values):
        ax = axes[panel // ncols][panel % ncols]
        for i, d in enumerate(data):
            color = _arm_color(i)
            by_method: dict[str, list[tuple[float, float]]] = {}
            for r in d["raw"]:
                if int(float(r["n"])) != n_val:
                    continue
                by_method.setdefault(r.get("method", "unknown"), []).append(
                    (float(r["rho_final"]), float(r["terminal_lambda2"]))
                )
            for method, pts in by_method.items():
                if len(pts) < 2:
                    continue
                centers, means, stds = _binned_mean_std(pts)
                if centers.size == 0:
                    continue
                ax.plot(centers, means, color=color,
                        linestyle=_METHOD_LINESTYLE.get(method, "-"),
                        linewidth=2, alpha=0.9)
                ax.fill_between(centers, means - stds, means + stds,
                                color=color, alpha=0.10, linewidth=0)
        ax.set_title(f"n = {n_val}", fontsize=11, color=_INK)
        ax.set_xlabel("Density (rho)", color=_MUTED)
        ax.set_ylabel("lambda2 (raw)", color=_MUTED)
        ax.grid(True, **_GRID_KW)

    for panel in range(len(n_values), nrows * ncols):
        axes[panel // ncols][panel % ncols].axis("off")

    handles = [Patch(color=_arm_color(i), label=str(e["label"])) for i, e in enumerate(manifest)]
    handles += [Line2D([0], [0], color=_INK, linestyle=ls, label=m)
                for m, ls in _METHOD_LINESTYLE.items()]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 3), fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Algebraic connectivity vs density (binned mean +/- std)", fontsize=14)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    out = COMPARISON_DIR / "lambda2_vs_density.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_density_bucket_bars(manifest, data) -> None:
    methods = sorted({r["method"] for d in data for r in d["buckets"]})
    n_values = sorted({int(float(r["n"])) for d in data for r in d["buckets"]})
    if not methods or not n_values:
        print("No density_bucket_summary.csv data; skipping density_bucket_bars.png")
        return

    nrows, ncols = len(n_values), len(methods)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.2 * nrows), squeeze=False)
    n_arms = len(manifest)
    bar_w = 0.8 / max(1, n_arms)
    xs = np.arange(len(_BUCKET_ORDER))

    for row, n_val in enumerate(n_values):
        for col, method in enumerate(methods):
            ax = axes[row][col]
            for i, d in enumerate(data):
                lookup = {
                    r["density_bucket"]: (float(r["lambda2_mean"]), float(r["lambda2_std"]))
                    for r in d["buckets"]
                    if r["method"] == method and int(float(r["n"])) == n_val
                }
                means = [lookup[b][0] if b in lookup else np.nan for b in _BUCKET_ORDER]
                stds = [lookup[b][1] if b in lookup else 0.0 for b in _BUCKET_ORDER]
                offs = xs + (i - (n_arms - 1) / 2.0) * bar_w
                # 2px surface gap between adjacent bars.
                ax.bar(offs, means, width=bar_w * 0.88, color=_arm_color(i),
                       yerr=stds, capsize=2,
                       error_kw={"linewidth": 0.8, "alpha": 0.6, "ecolor": _MUTED})
            ax.set_xticks(xs)
            ax.set_xticklabels(_BUCKET_ORDER, fontsize=9)
            ax.set_title(f"{method},  n={n_val}", fontsize=10, color=_INK)
            ax.set_ylabel("lambda2 (raw)", fontsize=9, color=_MUTED)
            ax.grid(True, axis="y", **_GRID_KW)

    handles = [Patch(color=_arm_color(i), label=str(e["label"])) for i, e in enumerate(manifest)]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 3),
               fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Algebraic connectivity averaged over density regimes", fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out = COMPARISON_DIR / "density_bucket_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_head_to_head(manifest, head_rows: list[dict[str, object]], baseline_tag: str) -> None:
    """Paired lambda_2 delta vs the baseline arm, per n and density regime."""
    rows = [r for r in head_rows if r["arm"] != baseline_tag]
    if not rows:
        print("No non-baseline arms; skipping head_to_head_delta.png")
        return

    methods = sorted({str(r["method"]) for r in rows})
    n_values = sorted({int(r["n"]) for r in rows})
    arm_tags = [str(e["tag"]) for e in manifest]
    arm_labels = {str(e["tag"]): str(e["label"]) for e in manifest}
    plotted = [t for t in arm_tags if t != baseline_tag]

    nrows, ncols = len(methods), len(n_values)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    bar_w = 0.8 / max(1, len(plotted))
    xs = np.arange(len(_BUCKET_ORDER))

    for row, method in enumerate(methods):
        for col, n_val in enumerate(n_values):
            ax = axes[row][col]
            for j, tag in enumerate(plotted):
                color = _arm_color(arm_tags.index(tag))
                lookup = {
                    str(r["density_bucket"]): float(r["delta_lambda2_mean"])
                    for r in rows
                    if r["arm"] == tag and str(r["method"]) == method and int(r["n"]) == n_val
                }
                vals = [lookup.get(b, np.nan) for b in _BUCKET_ORDER]
                offs = xs + (j - (len(plotted) - 1) / 2.0) * bar_w
                ax.bar(offs, vals, width=bar_w * 0.88, color=color)
            ax.axhline(0.0, color=_INK, linewidth=1.2, zorder=3)
            ax.set_xticks(xs)
            ax.set_xticklabels(_BUCKET_ORDER, fontsize=9)
            ax.set_title(f"{method},  n={n_val}", fontsize=10, color=_INK)
            if col == 0:
                ax.set_ylabel("delta lambda2 vs baseline", fontsize=9, color=_MUTED)
            ax.grid(True, axis="y", **_GRID_KW)

    handles = [Patch(color=_arm_color(arm_tags.index(t)), label=arm_labels[t]) for t in plotted]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 3),
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Paired lambda2 delta vs '{arm_labels.get(baseline_tag, baseline_tag)}'\n"
        "(same n, same target edge count, same eval seed; above zero = better)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    out = COMPARISON_DIR / "head_to_head_delta.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def _speedup_label(ax, xs, base, other) -> None:
    """Annotate the rightmost point with the baseline/other ratio."""
    finite = [k for k in range(len(xs)) if np.isfinite(base[k]) and np.isfinite(other[k]) and other[k] > 0]
    if not finite:
        return
    k = finite[-1]
    ratio = base[k] / other[k]
    ax.annotate(
        f"{ratio:.1f}x faster" if ratio >= 1 else f"{1 / ratio:.1f}x slower",
        (xs[k], other[k]), textcoords="offset points", xytext=(-8, -16),
        ha="right", fontsize=9, color=_INK,
    )


def plot_runtime(manifest, data, baseline_tag: str) -> None:
    """The headline cost comparison.

    Row 1 -- inference: mean rollout wall time per episode vs n, and the
    density-independent per-edge-decision cost. ``rl_runtime_sec`` is used, not
    ``runtime_total_sec``: backbone construction is byte-identical shared code
    and would dilute the difference being measured.

    Row 2 -- how cost scales with problem size within a fixed n (more edges to
    place = longer episodes), and training throughput from metrics.csv.
    """
    n_values = sorted({int(float(r["n"])) for d in data for r in d["cost"]})
    if not n_values:
        print("No summary_cost.csv data; skipping runtime_comparison.png")
        return
    methods = sorted({r["method"] for d in data for r in d["cost"]})

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) episode wall time vs n, one line per (arm, method)
    ax = axes[0][0]
    path_curves: dict[str, np.ndarray] = {}
    for i, (entry, d) in enumerate(zip(manifest, data)):
        for method in methods:
            lookup = {
                int(float(r["n"])): float(r["rl_runtime_mean_over_density_s"])
                for r in d["cost"] if r["method"] == method
            }
            ys = np.asarray([lookup.get(n, np.nan) for n in n_values], dtype=np.float64)
            ax.plot(n_values, ys, color=_arm_color(i), marker=_arm_marker(i), markersize=7,
                    linestyle=_METHOD_LINESTYLE.get(method, "-"), linewidth=2, alpha=0.9)
            if method == "RL+Path":
                path_curves[str(entry["tag"])] = ys
    # Annotate each non-baseline arm against the baseline, keyed by tag rather
    # than by position -- there are three arms now, and --arms can reorder them.
    base_curve = path_curves.get(baseline_tag)
    if base_curve is not None:
        for tag, ys in path_curves.items():
            if tag != baseline_tag:
                _speedup_label(ax, n_values, base_curve, ys)
    ax.set_xlabel("n", color=_MUTED)
    ax.set_ylabel("Rollout wall time per episode (s)", color=_MUTED)
    ax.set_title("Inference cost per completion episode", fontsize=11, color=_INK)
    ax.set_yscale("log")
    ax.set_xticks(n_values)
    ax.grid(True, **_GRID_KW)

    # (0,1) per-edge-decision cost vs n
    ax = axes[0][1]
    for i, d in enumerate(data):
        for method in methods:
            lookup = {
                int(float(r["n"])): float(r["rl_runtime_per_step_mean_s"])
                for r in d["cost"] if r["method"] == method
            }
            ys = [lookup.get(n, np.nan) for n in n_values]
            ax.plot(n_values, ys, color=_arm_color(i), marker=_arm_marker(i), markersize=7,
                    linestyle=_METHOD_LINESTYLE.get(method, "-"), linewidth=2, alpha=0.9)
    ax.set_xlabel("n", color=_MUTED)
    ax.set_ylabel("Wall time per edge decision (s)", color=_MUTED)
    ax.set_title("Per-step cost (episode length divided out)", fontsize=11, color=_INK)
    ax.set_yscale("log")
    ax.set_xticks(n_values)
    ax.grid(True, **_GRID_KW)

    # (1,0) runtime vs density at the largest n, from raw.csv
    ax = axes[1][0]
    n_big = n_values[-1]
    for i, d in enumerate(data):
        pts = [
            (float(r["rho_final"]), float(r["rl_runtime_sec"]))
            for r in d["raw"]
            if int(float(r["n"])) == n_big and r["method"] == "RL+Path"
        ]
        if len(pts) < 2:
            continue
        centers, means, stds = _binned_mean_std(pts)
        if centers.size == 0:
            continue
        ax.plot(centers, means, color=_arm_color(i), marker=_arm_marker(i),
                markersize=6, linewidth=2, alpha=0.9)
        ax.fill_between(centers, np.maximum(means - stds, 1e-6), means + stds,
                        color=_arm_color(i), alpha=0.10, linewidth=0)
    ax.set_xlabel("Density (rho)", color=_MUTED)
    ax.set_ylabel("Rollout wall time (s)", color=_MUTED)
    ax.set_title(f"Cost vs target density, n={n_big}, RL+Path", fontsize=11, color=_INK)
    ax.set_yscale("log")
    ax.grid(True, **_GRID_KW)

    # (1,1) training throughput
    ax = axes[1][1]
    any_sps = False
    for i, (entry, d) in enumerate(zip(manifest, data)):
        rows = [r for r in d["metrics"] if r.get("steps_per_sec")]
        if not rows:
            continue
        any_sps = True
        steps = [int(float(r["global_env_steps"])) for r in rows]
        sps = [float(r["steps_per_sec"]) for r in rows]
        ax.plot(steps, sps, color=_arm_color(i), marker=_arm_marker(i), markersize=5,
                linewidth=2, alpha=0.9, label=str(entry["label"]))
    if any_sps:
        ax.set_xlabel("Environment steps", color=_MUTED)
        ax.set_ylabel("Training throughput (env steps / s)", color=_MUTED)
        ax.set_title("Training speed", fontsize=11, color=_INK)
        ax.grid(True, **_GRID_KW)
        ax.legend(fontsize=9, loc="best")
    else:
        ax.axis("off")

    handles = [
        Line2D([0], [0], color=_arm_color(i), marker=_arm_marker(i), markersize=7,
               linewidth=2, label=str(e["label"]))
        for i, e in enumerate(manifest)
    ]
    handles += [Line2D([0], [0], color=_INK, linestyle=ls, label=m)
                for m, ls in _METHOD_LINESTYLE.items() if m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
               fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Runtime comparison: new analytic model vs old model", fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out = COMPARISON_DIR / "runtime_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def write_runtime_table(manifest, data, baseline_tag: str) -> None:
    """Per-(arm, method, n) cost table, with the baseline/arm ratio."""
    rows: list[dict[str, object]] = []
    base_lookup: dict[tuple[str, int], float] = {}
    # Keyed by tag, not position: --arms can reorder or drop arms.
    base_data = next(
        (d for e, d in zip(manifest, data) if str(e["tag"]) == baseline_tag), None
    )
    if base_data is not None:
        for r in base_data["cost"]:
            base_lookup[(str(r["method"]), int(float(r["n"])))] = float(
                r["rl_runtime_mean_over_density_s"]
            )
    for entry, d in zip(manifest, data):
        for r in d["cost"]:
            method, n = str(r["method"]), int(float(r["n"]))
            rl = float(r["rl_runtime_mean_over_density_s"])
            base = base_lookup.get((method, n))
            rows.append(
                {
                    "arm": str(entry["tag"]),
                    "label": str(entry["label"]),
                    "method": method,
                    "n": n,
                    "rl_runtime_mean_s": rl,
                    "rl_runtime_std_s": float(r["rl_runtime_std_over_density_s"]),
                    "rl_runtime_per_step_mean_s": float(r["rl_runtime_per_step_mean_s"]),
                    "init_runtime_mean_s": float(r["init_runtime_mean_over_density_s"]),
                    "total_runtime_mean_s": float(r["runtime_mean_over_density_s"]),
                    "speedup_vs_baseline": (base / rl) if base and rl > 0 else "",
                }
            )
    rows.sort(key=lambda r: (str(r["method"]), int(r["n"]), str(r["arm"])))
    _write_csv(
        COMPARISON_DIR / "runtime_summary.csv",
        rows,
        [
            "arm", "label", "method", "n",
            "rl_runtime_mean_s", "rl_runtime_std_s", "rl_runtime_per_step_mean_s",
            "init_runtime_mean_s", "total_runtime_mean_s", "speedup_vs_baseline",
        ],
    )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def build_head_to_head(manifest, data, baseline_tag: str) -> list[dict[str, object]]:
    """Pair every arm against the baseline on (method, n, m_target, seed)."""
    by_tag = {str(e["tag"]): d for e, d in zip(manifest, data)}
    if baseline_tag not in by_tag:
        raise ValueError(f"Baseline arm '{baseline_tag}' not in manifest.")

    def keyed(rows):
        return {
            (str(r["method"]), int(float(r["n"])), int(float(r["m_target"])), int(float(r["seed"]))):
            (float(r["terminal_lambda2"]), float(r["rho_final"]))
            for r in rows
        }

    base = keyed(by_tag[baseline_tag]["raw"])

    out: list[dict[str, object]] = []
    for entry, d in zip(manifest, data):
        tag = str(entry["tag"])
        arm_rows = keyed(d["raw"])
        acc: dict[tuple[str, int, str], list[tuple[float, float]]] = {}
        for key, (lam, rho) in arm_rows.items():
            if key not in base:
                continue
            base_lam, _ = base[key]
            method, n, _m, _s = key
            acc.setdefault((method, n, _bucket_for_rho(rho)), []).append((lam, base_lam))

        for (method, n, bucket), pairs in sorted(acc.items()):
            arr = np.asarray([p[0] for p in pairs], dtype=np.float64)
            barr = np.asarray([p[1] for p in pairs], dtype=np.float64)
            delta = arr - barr
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = np.where(np.abs(barr) > 1e-12, delta / np.maximum(np.abs(barr), 1e-12) * 100.0, np.nan)
            out.append(
                {
                    "arm": tag,
                    "label": str(entry["label"]),
                    "baseline": baseline_tag,
                    "method": method,
                    "n": int(n),
                    "density_bucket": bucket,
                    "lambda2_mean": float(np.mean(arr)),
                    "baseline_lambda2_mean": float(np.mean(barr)),
                    "delta_lambda2_mean": float(np.mean(delta)),
                    "delta_lambda2_std": float(np.std(delta)),
                    "delta_lambda2_pct_mean": float(np.nanmean(pct)) if np.any(np.isfinite(pct)) else float("nan"),
                    "win_pct_vs_baseline": float(100.0 * np.mean(delta > 1e-10)),
                    "paired_points": int(delta.size),
                }
            )
    return out


def build_overall_summary(manifest, data, head_rows, baseline_tag: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for entry, d in zip(manifest, data):
        tag = str(entry["tag"])
        path_l2 = [float(r["lambda2_mean"]) for r in d["buckets"] if r["method"] == "RL+Path"]
        bb_l2 = [float(r["lambda2_mean"]) for r in d["buckets"] if r["method"] == "RL+Backbone"]
        # Raw eval lambda2 is only logged by the new model's train.py.
        eval_vals = [float(v) for r in d["metrics"] if (v := r.get("eval_terminal_lambda2_mean"))]
        best_norm = [float(r["best_eval_metric"]) for r in d["metrics"] if r.get("best_eval_metric")]
        rt = [float(r["rl_runtime_mean_over_density_s"]) for r in d["cost"]]
        sps = [float(r["steps_per_sec"]) for r in d["metrics"] if r.get("steps_per_sec")]
        h2h = [r for r in head_rows if r["arm"] == tag and r["arm"] != baseline_tag]
        deltas = [float(r["delta_lambda2_mean"]) for r in h2h]
        wins = [float(r["win_pct_vs_baseline"]) for r in h2h]

        out.append(
            {
                "arm": tag,
                "label": str(entry["label"]),
                "package": str(entry["package"]),
                "reward": json.dumps(entry.get("env_overrides", {})),
                "train_wall_sec": entry.get("train_wall_sec", ""),
                "max_env_steps": entry.get("max_env_steps", ""),
                "best_eval_lambda2_raw_during_training": max(eval_vals) if eval_vals else "",
                "best_eval_lambda2_norm_during_training": max(best_norm) if best_norm else "",
                "mean_lambda2_rl_path": float(np.mean(path_l2)) if path_l2 else "",
                "mean_lambda2_rl_backbone": float(np.mean(bb_l2)) if bb_l2 else "",
                "mean_rollout_runtime_sec": float(np.mean(rt)) if rt else "",
                "mean_training_steps_per_sec": float(np.mean(sps)) if sps else "",
                "delta_lambda2_vs_baseline": float(np.mean(deltas)) if deltas else 0.0,
                "win_pct_vs_baseline": float(np.mean(wins)) if wins else "",
            }
        )
    return out


def main() -> None:
    args = parse_args()
    manifest = _load_manifest()
    if not manifest:
        print("Manifest is empty; nothing to visualize.")
        return
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    data = [_load_arm_data(e) for e in manifest]
    if not any(d["raw"] for d in data):
        print("No evaluation results found. Run compare_experiment.py first.")
        sys.exit(1)

    head_rows = build_head_to_head(manifest, data, args.baseline)
    _write_csv(
        COMPARISON_DIR / "head_to_head.csv",
        head_rows,
        [
            "arm", "label", "baseline", "method", "n", "density_bucket",
            "lambda2_mean", "baseline_lambda2_mean",
            "delta_lambda2_mean", "delta_lambda2_std", "delta_lambda2_pct_mean",
            "win_pct_vs_baseline", "paired_points",
        ],
    )
    _write_csv(
        COMPARISON_DIR / "comparison_summary.csv",
        build_overall_summary(manifest, data, head_rows, args.baseline),
        [
            "arm", "label", "package", "reward", "train_wall_sec", "max_env_steps",
            "best_eval_lambda2_raw_during_training", "best_eval_lambda2_norm_during_training",
            "mean_lambda2_rl_path", "mean_lambda2_rl_backbone",
            "mean_rollout_runtime_sec", "mean_training_steps_per_sec",
            "delta_lambda2_vs_baseline", "win_pct_vs_baseline",
        ],
    )

    write_runtime_table(manifest, data, args.baseline)

    plot_runtime(manifest, data, args.baseline)
    plot_training_curves(manifest, data)
    plot_lambda2_vs_density(manifest, data)
    plot_density_bucket_bars(manifest, data)
    plot_head_to_head(manifest, head_rows, args.baseline)

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
