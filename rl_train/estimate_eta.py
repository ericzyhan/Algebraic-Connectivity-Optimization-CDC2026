"""Estimate η so that |η · ΔR_G / n²| ≈ |Δλ₂ / n| on average.

Run from the project root:
    python -m rl_train.estimate_eta

This simulates random edge additions on path graphs of various sizes
and computes the ratio of the two reward-term magnitudes to find a good
default η.  The formula is:

    suggested_η = mean(|Δλ₂ / n|) / mean(|ΔR_G / n²|)

Each trajectory resets to a fresh path graph after every `traj_len`
edge additions, so the statistics cover the early-to-mid construction
regime that the RL agent operates in.

Outputs results per graph size so you can decide whether a size-dependent
η makes sense.
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
        Total (Δλ₂, ΔR_G) observations to collect for each n.
    traj_len : int
        Number of edge additions per trajectory before resetting to a
        fresh path graph.  Keeps the density regime realistic.
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.RandomState(seed)

    print("=" * 76)
    print("  Reward-term magnitude calibration  (η estimation)")
    print("=" * 76)
    print(f"  Total samples per n: {total_samples}")
    print(f"  Trajectory length:   {traj_len} edges per reset")
    print(f"  Seed:                {seed}")
    print()

    header = (
        f"{'n':>6}  {'samples':>8}  {'mean|Δλ₂/n|':>14}  "
        f"{'mean|ΔR_G/n²|':>16}  {'suggested η':>12}  "
        f"{'η·n':>10}"
    )
    print(header)
    print("-" * len(header))

    all_suggested: list[float] = []

    for n in ns:
        deltas_l2: list[float] = []
        deltas_rg: list[float] = []
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
                l2_old, _, _, _, _, _, rg_old, _, _ = spectral_features(adj)

                # Add the edge
                adj[i, j] = 1
                adj[j, i] = 1

                # Spectral info AFTER addition
                l2_new, _, _, _, _, _, rg_new, _, _ = spectral_features(adj)

                # Record the terms
                delta_l2 = float(l2_new - l2_old)
                delta_rg = float(rg_old - rg_new)  # positive when resistance drops

                deltas_l2.append(delta_l2 / float(max(1, n)))
                deltas_rg.append(delta_rg / float(max(1, n * n)))
                collected += 1

                if collected >= total_samples:
                    break

        arr_l2 = np.asarray(deltas_l2[:total_samples], dtype=np.float64)
        arr_rg = np.asarray(deltas_rg[:total_samples], dtype=np.float64)

        mean_abs_l2 = float(np.mean(np.abs(arr_l2)))
        mean_abs_rg = float(np.mean(np.abs(arr_rg)))

        if mean_abs_rg > 1e-15:
            suggested_eta = mean_abs_l2 / mean_abs_rg
        else:
            suggested_eta = 1.0

        all_suggested.append(suggested_eta)

        print(
            f"{n:>6}  {len(arr_l2):>8}  {mean_abs_l2:>14.6e}  "
            f"{mean_abs_rg:>16.6e}  {suggested_eta:>12.4f}  "
            f"{suggested_eta * float(n):>10.2f}"
        )

    print("-" * len(header))

    # Overall summary
    if len(ns) > 1:
        mean_eta = float(np.mean(all_suggested))
        median_eta = float(np.median(all_suggested))
        eta_n_products = [eta * float(n) for eta, n in zip(all_suggested, ns)]
        print(f"\n  Summary:")
        print(f"    Mean η across sizes:       {mean_eta:.4f}")
        print(f"    Median η across sizes:     {median_eta:.4f}")
        print(f"    Mean η·n (if η ∝ 1/n):     {float(np.mean(eta_n_products)):.2f}")
        print()

    # Config recommendation
    print("  Recommended per-size config entries:")
    print()
    for n, eta in zip(ns, all_suggested):
        print(f"    # n={n}:  reward_eta: {eta:.6f}")
    print()
    print("  Set in config YAML under env.reward_eta")
    print("=" * 76)


if __name__ == "__main__":
    estimate_eta(
        ns=[8, 12, 16, 24, 32, 48, 64],
        total_samples=500,
        traj_len=50,
        seed=42,
    )
