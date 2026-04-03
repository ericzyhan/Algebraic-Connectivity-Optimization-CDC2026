from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx

from graph_design.spectral import algebraic_connectivity
from graph_design.utils import normalized_density


@dataclass(frozen=True)
class SamplePoint:
    sample_id: int
    n: int
    m: int
    density: float
    lambda2: float


def _random_labeled_tree(n: int, rng: random.Random) -> nx.Graph:
    seed = rng.randint(0, 2**31 - 1)
    if hasattr(nx, "random_labeled_tree"):
        return nx.random_labeled_tree(n, seed=seed)
    if hasattr(nx, "random_tree"):
        return nx.random_tree(n, seed=seed)
    from networkx.generators.trees import random_labeled_tree

    return random_labeled_tree(n, seed=seed)


def random_connected_graph_with_m(
    *,
    n: int,
    m: int,
    rng: random.Random,
) -> nx.Graph:
    min_edges = n - 1
    max_edges = comb(n, 2)
    if m < min_edges or m > max_edges:
        raise ValueError(f"m out of range for connected graph: {m} not in [{min_edges}, {max_edges}]")

    graph = _random_labeled_tree(n, rng)
    extra_edges = m - graph.number_of_edges()
    if extra_edges <= 0:
        return graph

    missing = [
        (u, v)
        for u in range(n)
        for v in range(u + 1, n)
        if not graph.has_edge(u, v)
    ]
    chosen = rng.sample(missing, extra_edges)
    graph.add_edges_from(chosen)
    return graph


def sample_connected_points(
    *,
    n: int,
    samples: int,
    seed: int,
) -> list[SamplePoint]:
    if n < 2:
        raise ValueError("n must be >= 2")
    if samples <= 0:
        raise ValueError("samples must be > 0")

    rng = random.Random(seed)
    min_edges = n - 1
    max_edges = comb(n, 2)

    points: list[SamplePoint] = []
    for idx in range(samples):
        m = rng.randint(min_edges, max_edges)
        graph = random_connected_graph_with_m(n=n, m=m, rng=rng)
        l2 = float(algebraic_connectivity(graph))
        rho = float(normalized_density(n, m))
        points.append(SamplePoint(sample_id=idx, n=n, m=m, density=rho, lambda2=l2))
    return points


def upper_left_staircase(points: Sequence[SamplePoint], eps: float = 1e-12) -> list[SamplePoint]:
    ordered = sorted(points, key=lambda p: (p.density, -p.lambda2, p.m, p.sample_id))
    staircase: list[SamplePoint] = []
    best_lambda2 = -float("inf")
    for point in ordered:
        if point.lambda2 > best_lambda2 + eps:
            staircase.append(point)
            best_lambda2 = point.lambda2
    return staircase


def write_points_csv(
    *,
    points: Iterable[SamplePoint],
    staircase_ids: set[int],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "n", "m", "density", "lambda2", "on_staircase"],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "sample_id": point.sample_id,
                    "n": point.n,
                    "m": point.m,
                    "density": point.density,
                    "lambda2": point.lambda2,
                    "on_staircase": int(point.sample_id in staircase_ids),
                }
            )


def write_staircase_csv(*, staircase: Sequence[SamplePoint], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["staircase_idx", "sample_id", "n", "m", "density", "lambda2"],
        )
        writer.writeheader()
        for idx, point in enumerate(staircase):
            writer.writerow(
                {
                    "staircase_idx": idx,
                    "sample_id": point.sample_id,
                    "n": point.n,
                    "m": point.m,
                    "density": point.density,
                    "lambda2": point.lambda2,
                }
            )


def plot_points_with_staircase(
    *,
    points: Sequence[SamplePoint],
    staircase: Sequence[SamplePoint],
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it before running this script."
        ) from exc

    xs = [p.density for p in points]
    ys = [p.lambda2 for p in points]
    sx = [p.density for p in staircase]
    sy = [p.lambda2 for p in staircase]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    plt.scatter(
        xs,
        ys,
        s=12,
        alpha=0.35,
        color="#607d8b",
        label=f"Random connected graphs ({len(points)})",
    )
    plt.scatter(
        sx,
        sy,
        s=18,
        color="#d62728",
        zorder=3,
        label="Upper-left staircase points",
    )
    plt.xlabel("Density", fontsize=18)
    plt.ylabel("Algebraic Connectivity", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample random connected graphs for fixed n, compute (density, lambda_2), "
            "and plot the upper-left staircase (Pareto frontier)."
        )
    )
    parser.add_argument("--n", type=int, required=True, help="Number of nodes.")
    parser.add_argument("--samples", type=int, default=2000, help="Number of random connected graphs.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "experiments" / "random_connected",
        help="Output directory for CSV + PNG artifacts.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional filename tag. Defaults to n{n}_s{samples}_seed{seed}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.tag or f"n{args.n}_s{args.samples}_seed{args.seed}"
    out_dir: Path = args.out_dir.resolve()

    points = sample_connected_points(n=args.n, samples=args.samples, seed=args.seed)
    staircase = upper_left_staircase(points)
    staircase_ids = {p.sample_id for p in staircase}

    points_csv = out_dir / f"points_{tag}.csv"
    staircase_csv = out_dir / f"staircase_{tag}.csv"
    plot_png = out_dir / f"scatter_staircase_{tag}.png"

    write_points_csv(points=points, staircase_ids=staircase_ids, out_path=points_csv)
    write_staircase_csv(staircase=staircase, out_path=staircase_csv)
    plot_points_with_staircase(points=points, staircase=staircase, out_path=plot_png)

    print(f"[random_connected_frontier] wrote points: {points_csv}")
    print(f"[random_connected_frontier] wrote staircase: {staircase_csv}")
    print(f"[random_connected_frontier] wrote plot: {plot_png}")
    print(
        "[random_connected_frontier] "
        f"n={args.n} samples={args.samples} staircase_points={len(staircase)}"
    )


if __name__ == "__main__":
    main()
