from graph_design.config import DesignConfig
from graph_design.solver import HybridGraphDesigner
from graph_design.types import DesignProblem


def test_end_to_end_cayley_region():
    solver = HybridGraphDesigner(config=DesignConfig(cayley_samples=12))
    problem = DesignProblem(n=12, m=20, seed=1)  # normalized rho < 0.5
    result = solver.solve(problem)

    assert result.route_label == "cayley_only"
    assert result.graph.number_of_edges() == problem.m


def test_end_to_end_middle_region():
    solver = HybridGraphDesigner(config=DesignConfig(cayley_samples=12))
    problem = DesignProblem(n=12, m=39, seed=2)  # normalized rho ~= 0.509
    result = solver.solve(problem)

    assert result.route_label == "both"
    assert result.graph.number_of_edges() == problem.m


def test_end_to_end_envelope_region():
    solver = HybridGraphDesigner(config=DesignConfig(cayley_samples=12))
    problem = DesignProblem(n=12, m=55, seed=3)  # normalized rho = 0.8
    result = solver.solve(problem)

    assert result.route_label == "envelope_only"
    assert result.graph.number_of_edges() == problem.m
