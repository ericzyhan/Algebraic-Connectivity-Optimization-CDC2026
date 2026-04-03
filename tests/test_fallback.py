import networkx as nx

from graph_design.backbones.base import BackboneGenerator
from graph_design.config import DesignConfig
from graph_design.solver import HybridGraphDesigner
from graph_design.spectral import algebraic_connectivity
from graph_design.types import BackboneCandidate, DesignProblem


class NullGenerator(BackboneGenerator):
    family = "null"

    def generate(self, problem, config, rng):
        del problem, config, rng
        return None


class PathCandidateGenerator(BackboneGenerator):
    family = "path"

    def generate(self, problem, config, rng):
        del config, rng
        graph = nx.path_graph(problem.n)
        return BackboneCandidate(
            family=self.family,
            graph=graph,
            backbone_lambda2=algebraic_connectivity(graph),
            metadata={"source": "test_stub"},
        )


def test_fallback_to_other_family_when_primary_has_no_candidate():
    config = DesignConfig(routing_mode="window", rho_low=0.5, rho_high=0.75)
    solver = HybridGraphDesigner(
        config=config,
        cayley_generator=NullGenerator(),
        envelope_generator=PathCandidateGenerator(),
    )
    # rho < 0.5 routes to cayley_only, so envelope should be used by fallback.
    problem = DesignProblem(n=8, m=10, seed=3)

    result = solver.solve(problem)

    assert result.route_label == "cayley_only"
    assert result.selected_family == "envelope"
    assert result.fallback_used is True
    assert result.graph.number_of_edges() == problem.m
