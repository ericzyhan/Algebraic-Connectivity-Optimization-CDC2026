import random

from graph_design.baselines.registry import create_baseline, list_baselines
from graph_design.baselines.fvg import FVGBaseSolver
from graph_design.config import DesignConfig
from graph_design.types import DesignProblem


def test_baseline_registry_contains_fvg():
    assert "fvg" in list_baselines()
    assert "reverse_fvg" in list_baselines()


def test_fvg_baseline_solves_to_exact_budget():
    solver = create_baseline("fvg")
    problem = DesignProblem(n=10, m=15, seed=7)
    result = solver.solve(problem, DesignConfig(), random.Random(7))

    assert result.name == "fvg"
    assert result.graph.number_of_edges() == problem.m
    assert result.final_lambda2 >= 0.0
    assert "initial_graph" in result.metadata


def test_reverse_fvg_baseline_solves_to_exact_budget():
    solver = create_baseline("reverse_fvg")
    problem = DesignProblem(n=10, m=15, seed=11)
    result = solver.solve(problem, DesignConfig(), random.Random(11))

    assert result.name == "reverse_fvg"
    assert result.graph.number_of_edges() == problem.m
    assert result.final_lambda2 >= 0.0
    assert "deletion_steps" in result.metadata


def test_fvg_cached_runtime_reports_cumulative_time_to_target():
    FVGBaseSolver._cache.clear()
    solver = create_baseline("fvg")
    config = DesignConfig(fvg_reuse_cache=True)

    p1 = DesignProblem(n=14, m=20, seed=0)
    p2 = DesignProblem(n=14, m=26, seed=0)
    p3 = DesignProblem(n=14, m=22, seed=0)

    r1 = solver.solve(p1, config, random.Random(0))
    r2 = solver.solve(p2, config, random.Random(0))
    r3 = solver.solve(p3, config, random.Random(0))

    assert r1.metadata["runtime_mode"] == "cumulative_to_m"
    assert r2.metadata["runtime_mode"] == "cumulative_to_m"
    assert r3.metadata["runtime_mode"] == "cumulative_to_m"
    assert r2.runtime_sec >= r1.runtime_sec
    assert r3.runtime_sec <= r2.runtime_sec
