from graph_design.types import DesignProblem
from graph_design.utils import edge_budget_from_density, normalized_density


def test_connected_range_density_definition():
    n = 12
    min_edges = n - 1
    max_edges = n * (n - 1) // 2

    assert normalized_density(n, min_edges) == 0.0
    assert normalized_density(n, max_edges) == 1.0


def test_edge_budget_mapping_matches_new_density():
    n = 12
    min_edges = n - 1
    max_edges = n * (n - 1) // 2

    assert edge_budget_from_density(n, 0.0) == min_edges
    assert edge_budget_from_density(n, 1.0) == max_edges


def test_design_problem_density_uses_connected_range():
    problem = DesignProblem(n=12, m=39)
    # (39 - 11) / (66 - 11) = 28 / 55
    assert abs(problem.density - (28 / 55)) < 1e-12
