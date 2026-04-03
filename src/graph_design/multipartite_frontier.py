from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Iterable, Sequence

from graph_design.utils import normalized_density


@dataclass(frozen=True)
class MultipartitePoint:
    point_id: int
    n: int
    k: int
    parts: tuple[int, ...]
    m: int
    density: float
    lambda2: float


def _partitions_fixed_k(total: int, k: int, max_part: int | None = None) -> list[tuple[int, ...]]:
    if k <= 0 or total <= 0:
        return []
    if max_part is None:
        max_part = total
    if k == 1:
        if 1 <= total <= max_part:
            return [(total,)]
        return []

    out: list[tuple[int, ...]] = []
    upper = min(max_part, total - (k - 1))
    lower = (total + k - 1) // k
    for first in range(upper, lower - 1, -1):
        for rest in _partitions_fixed_k(total - first, k - 1, first):
            out.append((first,) + rest)
    return out


def _complete_multipartite_edges(parts: Sequence[int]) -> int:
    n = sum(parts)
    return (n * n - sum(p * p for p in parts)) // 2


def generate_multipartite_points(n: int, part_counts: Sequence[int]) -> list[MultipartitePoint]:
    if n < 2:
        raise ValueError("n must be >= 2")

    seen: set[tuple[int, tuple[int, ...]]] = set()
    points: list[MultipartitePoint] = []
    point_id = 0

    for k in sorted(set(int(x) for x in part_counts)):
        if k < 2 or k > n:
            continue
        for parts in _partitions_fixed_k(total=n, k=k):
            key = (k, parts)
            if key in seen:
                continue
            seen.add(key)

            n_max = max(parts)
            lambda2 = float(n - n_max)  # complete multipartite lambda_2 formula
            m = int(_complete_multipartite_edges(parts))
            rho = float(normalized_density(n=n, m=m))

            points.append(
                MultipartitePoint(
                    point_id=point_id,
                    n=n,
                    k=k,
                    parts=parts,
                    m=m,
                    density=rho,
                    lambda2=lambda2,
                )
            )
            point_id += 1
    return points


def upper_left_staircase(
    points: Sequence[MultipartitePoint], eps: float = 1e-12
) -> list[MultipartitePoint]:
    ordered = sorted(points, key=lambda p: (p.density, -p.lambda2, p.m, p.k, p.parts))
    staircase: list[MultipartitePoint] = []
    best_lambda2 = -float("inf")
    for point in ordered:
        if point.lambda2 > best_lambda2 + eps:
            staircase.append(point)
            best_lambda2 = point.lambda2
    return staircase


def write_points_csv(
    *,
    points: Iterable[MultipartitePoint],
    staircase_ids: set[int],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "point_id",
                "n",
                "k",
                "parts",
                "m",
                "density",
                "lambda2",
                "on_staircase",
            ],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "point_id": point.point_id,
                    "n": point.n,
                    "k": point.k,
                    "parts": "-".join(str(x) for x in point.parts),
                    "m": point.m,
                    "density": point.density,
                    "lambda2": point.lambda2,
                    "on_staircase": int(point.point_id in staircase_ids),
                }
            )


def write_staircase_csv(*, staircase: Sequence[MultipartitePoint], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "staircase_idx",
                "point_id",
                "n",
                "k",
                "parts",
                "m",
                "density",
                "lambda2",
            ],
        )
        writer.writeheader()
        for idx, point in enumerate(staircase):
            writer.writerow(
                {
                    "staircase_idx": idx,
                    "point_id": point.point_id,
                    "n": point.n,
                    "k": point.k,
                    "parts": "-".join(str(x) for x in point.parts),
                    "m": point.m,
                    "density": point.density,
                    "lambda2": point.lambda2,
                }
            )


def plot_points_with_staircase(
    *,
    points: Sequence[MultipartitePoint],
    staircase: Sequence[MultipartitePoint],
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it before running this script."
        ) from exc

    sx = [p.density for p in staircase]
    sy = [p.lambda2 for p in staircase]
    color_by_k = {
        2: "#d62728",  # red
        3: "#2ca02c",  # green
        4: "#1f77b4",  # blue
        5: "#ffd700",  # yellow
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    for k in sorted({p.k for p in points}):
        sub = [p for p in points if p.k == k]
        xs = [p.density for p in sub]
        ys = [p.lambda2 for p in sub]
        plt.scatter(
            xs,
            ys,
            s=14,
            alpha=0.55,
            color=color_by_k.get(k, "#607d8b"),
            label=f"{k}-partite",
        )
    plt.scatter(
        sx,
        sy,
        s=18,
        facecolors="none",
        edgecolors="#111111",
        linewidths=0.9,
        zorder=3,
        label="_nolegend_",
    )
    plt.xlabel("Density", fontsize=18)
    plt.ylabel("Algebraic Connectivity", fontsize=18)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 36.0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate complete multipartite graphs (2/3/4/5-partite by default) for fixed n, "
            "compute (density, lambda_2), and plot points with upper-left staircase."
        )
    )
    parser.add_argument("--n", type=int, required=True, help="Number of nodes.")
    parser.add_argument(
        "--parts",
        type=str,
        default="2,3,4,5",
        help="Comma-separated part counts to include, e.g. 2,3,4,5.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "experiments" / "multipartite",
        help="Output directory for CSV + PNG artifacts.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional filename tag. Defaults to n{n}_k{parts}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    part_counts = _parse_int_list(args.parts)
    part_tag = "_".join(str(x) for x in sorted(set(part_counts)))
    tag = args.tag or f"n{args.n}_k{part_tag}"
    out_dir: Path = args.out_dir.resolve()

    points = generate_multipartite_points(n=args.n, part_counts=part_counts)
    if not points:
        raise RuntimeError("No points generated. Check n and --parts.")
    staircase = upper_left_staircase(points)
    staircase_ids = {p.point_id for p in staircase}

    points_csv = out_dir / f"multipartite_points_{tag}.csv"
    staircase_csv = out_dir / f"multipartite_staircase_{tag}.csv"
    plot_png = out_dir / f"multipartite_scatter_staircase_{tag}.png"

    write_points_csv(points=points, staircase_ids=staircase_ids, out_path=points_csv)
    write_staircase_csv(staircase=staircase, out_path=staircase_csv)
    plot_points_with_staircase(points=points, staircase=staircase, out_path=plot_png)

    print(f"[multipartite_frontier] wrote points: {points_csv}")
    print(f"[multipartite_frontier] wrote staircase: {staircase_csv}")
    print(f"[multipartite_frontier] wrote plot: {plot_png}")
    print(
        "[multipartite_frontier] "
        f"n={args.n} parts={sorted(set(part_counts))} total_points={len(points)} staircase_points={len(staircase)}"
    )


if __name__ == "__main__":
    main()
