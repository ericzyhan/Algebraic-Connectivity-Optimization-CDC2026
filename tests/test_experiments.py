import networkx as nx

from graph_design.baselines.base import BaselineSolver
from graph_design.baselines.registry import register_baseline
from graph_design.config import DesignConfig
from graph_design.experiments import build_problem_grid, run_suite
from graph_design.spectral import algebraic_connectivity
from graph_design.types import BaselineResult


class DummyBaseline(BaselineSolver):
    name = "dummy"

    def solve(self, problem, config, rng):
        del config, rng
        graph = nx.path_graph(problem.n)
        l2 = algebraic_connectivity(graph)
        return BaselineResult(
            name=self.name,
            problem=problem,
            graph=graph,
            final_lambda2=l2,
            runtime_sec=0.0,
            metadata={},
        )


def test_experiment_suite_accepts_multiple_baselines():
    register_baseline("dummy", DummyBaseline)
    problems = build_problem_grid(n_values=[8], densities=[0.55], seeds=[0])
    rows = run_suite(
        problems=problems,
        config=DesignConfig(cayley_samples=8),
        baseline_names=["fvg", "dummy"],
    )

    names = {(row["method"], row["solver_name"]) for row in rows}
    assert ("hybrid", "hybrid") in names
    assert ("baseline", "fvg") in names
    assert ("baseline", "dummy") in names
