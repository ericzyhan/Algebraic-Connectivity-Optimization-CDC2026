from __future__ import annotations

import random
from typing import Iterable

from graph_design.baselines.registry import create_baseline
from graph_design.config import DesignConfig
from graph_design.solver import HybridGraphDesigner
from graph_design.types import DesignProblem
from graph_design.utils import edge_budget_from_density


def build_problem_grid(
    n_values: Iterable[int],
    densities: Iterable[float],
    seeds: Iterable[int],
) -> list[DesignProblem]:
    problems: list[DesignProblem] = []
    for n in n_values:
        for rho in densities:
            m = edge_budget_from_density(n=n, rho=rho)
            for seed in seeds:
                problems.append(DesignProblem(n=n, m=m, seed=seed))
    return problems


def _hybrid_metadata_columns(metadata: dict) -> dict:
    backbone = metadata.get("backbone_metadata", {})
    if not isinstance(backbone, dict):
        backbone = {}

    out = {
        "selected_index": backbone.get("selected_index", ""),
        "j_selected": backbone.get("j_selected", ""),
        "n_lift": backbone.get("n_lift", ""),
        "k_max_selected": backbone.get("k_max_selected", ""),
        "k_feasible_cap": backbone.get("k_feasible_cap", ""),
        "m_proj_selected": backbone.get("m_proj_selected", ""),
        "e_selected": backbone.get("e_selected", ""),
        "multi_index_overlap": backbone.get("multi_index_overlap", ""),
        "index2_pocket_high": backbone.get("index2_pocket_high", ""),
        "evaluated_indices": backbone.get("evaluated_indices", ""),
        "m_lift": backbone.get("m_lift", ""),
        "m_after_delete": backbone.get("m_after_delete", ""),
        "deleted_edges": backbone.get("deleted_edges", ""),
    }
    for index in (2, 3, 4):
        out[f"j_{index}"] = backbone.get(f"j_{index}", "")
        out[f"k_max_{index}"] = backbone.get(f"k_max_{index}", "")
        out[f"m_proj_{index}"] = backbone.get(f"m_proj_{index}", "")
        out[f"e_{index}"] = backbone.get(f"e_{index}", "")
        out[f"in_pocket_{index}"] = backbone.get(f"in_pocket_{index}", "")
    return out


def run_suite(
    problems: Iterable[DesignProblem],
    config: DesignConfig,
    baseline_names: Iterable[str],
) -> list[dict]:
    rows: list[dict] = []
    hybrid = HybridGraphDesigner(config=config)
    baseline_names = list(baseline_names)

    for problem in problems:
        hybrid_result = hybrid.solve(problem)
        rows.append(
            {
                "method": "hybrid",
                "solver_name": "hybrid",
                "n": problem.n,
                "m": problem.m,
                "rho": problem.density,
                "seed": problem.seed,
                "route_label": hybrid_result.route_label,
                "selected_family": hybrid_result.selected_family,
                "final_lambda2": hybrid_result.final_lambda2,
                "backbone_lambda2": hybrid_result.backbone_lambda2,
                "completion_gain": hybrid_result.completion_gain,
                "runtime_sec": hybrid_result.runtime_sec,
                "fallback_used": hybrid_result.fallback_used,
                **_hybrid_metadata_columns(hybrid_result.metadata),
            }
        )

        for name in baseline_names:
            solver = create_baseline(name)
            result = solver.solve(problem, config, random.Random(problem.seed))
            rows.append(
                {
                    "method": "baseline",
                    "solver_name": result.name,
                    "n": problem.n,
                    "m": problem.m,
                    "rho": problem.density,
                    "seed": problem.seed,
                    "route_label": "baseline",
                    "selected_family": result.name,
                    "final_lambda2": result.final_lambda2,
                    "backbone_lambda2": "",
                    "completion_gain": "",
                    "runtime_sec": result.runtime_sec,
                    "fallback_used": False,
                    "selected_index": "",
                    "j_selected": "",
                    "n_lift": "",
                    "k_max_selected": "",
                    "k_feasible_cap": "",
                    "m_proj_selected": "",
                    "e_selected": "",
                    "multi_index_overlap": "",
                    "index2_pocket_high": "",
                    "evaluated_indices": "",
                    "m_lift": "",
                    "m_after_delete": "",
                    "deleted_edges": "",
                    "j_2": "",
                    "k_max_2": "",
                    "m_proj_2": "",
                    "e_2": "",
                    "in_pocket_2": "",
                    "j_3": "",
                    "k_max_3": "",
                    "m_proj_3": "",
                    "e_3": "",
                    "in_pocket_3": "",
                    "j_4": "",
                    "k_max_4": "",
                    "m_proj_4": "",
                    "e_4": "",
                    "in_pocket_4": "",
                }
            )

    return rows
