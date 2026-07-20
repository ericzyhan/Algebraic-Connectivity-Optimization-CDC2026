"""Estimate η_R and η_P so that the three reward terms have comparable magnitude.

Run from the project root:
    python -m rl_train.estimate_eta

This simulates random edge additions on path graphs of various sizes
and computes the ratio of reward-term magnitudes to find good default
values for η_R and η_P.

    suggested_η_R = mean(|Δλ₂ / n|) / mean(|ΔR_G / n²|)
    suggested_η_P = mean(|Δλ₂ / n|) / mean(|ΔP_min|)

Each trajectory resets to a fresh path graph after every `traj_len`
edge additions, so the statistics cover the early-to-mid construction
regime that the RL agent operates in.

Outputs results per graph size so you can decide whether size-dependent
η values make sense.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure the project root is on the path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rl_train.graph_math import (
    build_path_adjacency,
    edge_count,
    non_edges,
    resistance_curvature,
    spectral_features,
)


def _random_non_edge(
    adj: np.ndarray,
    rng: np.random.RandomState,
) -> tuple[int, int]:
    """Pick a uniformly random non-edge from the graph."""
    ne = non_edges(adj)
    if not ne:
        msg = "No non-edges available — graph is complete."
        raise RuntimeError(msg)
    idx = rng.randint(len(ne))
    return ne[idx]


def estimate_eta(
    ns: list[int],
    total_samples: int,
    traj_len: int = 50,
    seed: int = 42,
) -> None:
    """Run the estimation for a list of graph sizes.

    Parameters
    ----------
    ns : list[int]
        Graph sizes to test.
    total_samples : int
        Total (Δλ₂, ΔR_G, ΔP_min) observations to collect for each n.
    traj_len : int
        Number of edge additions per trajectory before resetting to a
        fresh path graph.  Keeps the density regime realistic.
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.RandomState(seed)

    print("=" * 76)
    print("  Reward-term magnitude calibration  (η_R and η_P estimation)")
    print("=" * 76)
    print(f"  Total samples per n: {total_samples}")
    print(f"  Trajectory length:   {traj_len} edges per reset")
    print(f"  Seed:                {seed}")
    print()

    header = (
        f"{'n':>6}  {'samples':>8}  {'mean|Δλ₂/n|':>14}  "
        f"{'mean|ΔR_G/n²|':>16}  {'η_R':>10}  "
        f"{'mean|ΔP_min|':>14}  {'η_P':>10}"
    )
    print(header)
    print("-" * len(header))

    all_eta_r: list[float] = []
    all_eta_p: list[float] = []

    for n in ns:
        deltas_l2: list[float] = []
        deltas_rg: list[float] = []
        deltas_pmin: list[float] = []
        collected = 0

        while collected < total_samples:
            # Fresh path graph
            adj = build_path_adjacency(n).astype(np.uint8)

            for _ in range(traj_len):
                # Pick a random non-edge
                try:
                    i, j = _random_non_edge(adj, rng)
                except RuntimeError:
                    break  # graph became complete

                # Spectral info BEFORE addition
                adj_before = adj.copy()
                l2_old, _, _, _, _, _, rg_old, _, evecs_old, evals_old = spectral_features(adj)

                # Add the edge
                adj[i, j] = 1
                adj[j, i] = 1

                # Spectral info AFTER addition
                l2_new, _, _, _, _, _, rg_new, _, evecs_new, evals_new = spectral_features(adj)

                # Compute curvature BEFORE and AFTER
                _, P_min_old, _, _ = resistance_curvature(adj_before, evals_old, evecs_old)
                _, P_min_new, _, _ = resistance_curvature(adj, evals_new, evecs_new)

                # Record the terms
                delta_l2 = float(l2_new - l2_old)
                delta_rg = float(rg_old - rg_new)  # positive when resistance drops
                delta_pmin = float(P_min_new - P_min_old)

                deltas_l2.append(delta_l2 / float(max(1, n)))
                deltas_rg.append(delta_rg / float(max(1, n * n)))
                deltas_pmin.append(delta_pmin)
                collected += 1

                if collected >= total_samples:
                    break

        arr_l2 = np.asarray(deltas_l2[:total_samples], dtype=np.float64)
        arr_rg = np.asarray(deltas_rg[:total_samples], dtype=np.float64)
        arr_pmin = np.asarray(deltas_pmin[:total_samples], dtype=np.float64)

        mean_abs_l2 = float(np.mean(np.abs(arr_l2)))
        mean_abs_rg = float(np.mean(np.abs(arr_rg)))
        mean_abs_pmin = float(np.mean(np.abs(arr_pmin)))

        if mean_abs_rg > 1e-15:
            suggested_eta_r = mean_abs_l2 / mean_abs_rg
        else:
            suggested_eta_r = 1.0

        if mean_abs_pmin > 1e-15:
            suggested_eta_p = mean_abs_l2 / mean_abs_pmin
        else:
            suggested_eta_p = 1.0

        all_eta_r.append(suggested_eta_r)
        all_eta_p.append(suggested_eta_p)

        print(
            f"{n:>6}  {len(arr_l2):>8}  {mean_abs_l2:>14.6e}  "
            f"{mean_abs_rg:>16.6e}  {suggested_eta_r:>10.4f}  "
            f"{mean_abs_pmin:>14.6e}  {suggested_eta_p:>10.4f}"
        )

    print("-" * len(header))

    # Overall summary
    if len(ns) > 1:
        mean_eta_r = float(np.mean(all_eta_r))
        mean_eta_p = float(np.mean(all_eta_p))
        print(f"\n  Summary:")
        print(f"    Mean η_R across sizes:       {mean_eta_r:.4f}")
        print(f"    Mean η_P across sizes:       {mean_eta_p:.4f}")
        print()

    # Config recommendation
    print("  Recommended per-size config entries:")
    print()
    for n, eta_r, eta_p in zip(ns, all_eta_r, all_eta_p):
        print(f"    # n={n}:  reward_eta: {eta_r:.6f}  |  reward_eta_pmin: {eta_p:.6f}")
    print()
    print("  Set in config YAML under env.reward_eta and env.reward_eta_pmin")
    print("=" * 76)


if __name__ == "__main__":
    estimate_eta(
        ns=[8, 12, 16, 24, 32, 48, 64],
        total_samples=500,
        traj_len=50,
        seed=42,
    )
