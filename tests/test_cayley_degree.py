import random

from graph_design.backbones.cayley import (
    _PocketScore,
    _build_group,
    _choose_index_from_scores,
    _largest_feasible_degree_leq,
    _score_index_candidates,
    _in_open_interval,
    _InverseClosedSampler,
    _lift_nodes_needed,
    _score_index_candidate,
    _split_cosets,
    _valid_scores,
)
from graph_design.config import DesignConfig
from graph_design.backbones.cayley import (
    CayleyBackboneGenerator,
    RandomCayleyBackboneGenerator,
)
from graph_design.solver import HybridGraphDesigner
from graph_design.types import DesignProblem


def test_lift_nodes_needed_for_index3_example():
    # n=58 needs 2 lifted nodes to become divisible by 3.
    assert _lift_nodes_needed(58, 3) == 2


def test_k_max_uses_n_minus_j_denominator():
    score = _score_index_candidate(n=58, m=200, index=3)
    assert score.lift_nodes == 2
    assert score.k_max == (2 * 200) // (58 - 2)


def test_edge_ratio_uses_2m_over_n_squared():
    score = _score_index_candidate(n=50, m=400, index=2)
    expected = 2.0 * score.m_proj / (50 * 50)
    assert abs(score.edge_ratio - expected) < 1e-12


def test_open_interval_boundaries_are_excluded():
    assert _in_open_interval(0.5, 0.0, 0.5) is False
    assert _in_open_interval(2.0 / 3.0, 0.5, 2.0 / 3.0) is False
    assert _in_open_interval(0.75, 2.0 / 3.0, 0.75) is False
    assert _in_open_interval(0.74, 2.0 / 3.0, 0.75) is True


def test_choose_smallest_valid_index_when_multiple_are_valid():
    scores = {
        2: _PocketScore(2, 0, 64, 10, 320, 0.30, 0.0, 0.5, True),
        3: _PocketScore(3, 2, 66, 20, 640, 0.64, 0.5, 2.0 / 3.0, True),
        4: _PocketScore(4, 0, 64, 30, 960, 0.70, 2.0 / 3.0, 0.75, True),
    }
    selected = _choose_index_from_scores(scores)
    assert selected is not None
    assert selected.index == 2


def test_cayley_metadata_tracks_lift_delete_edges():
    problem = DesignProblem(n=12, m=20, seed=1)
    config = DesignConfig(cayley_samples=8, cayley_eval_batch_size=4)
    generator = CayleyBackboneGenerator()

    candidate = generator.generate(problem, config, random.Random(problem.seed))

    assert candidate is not None
    meta = candidate.metadata
    assert meta["m_after_delete"] == candidate.graph.number_of_edges()
    assert meta["deleted_edges"] == meta["m_lift"] - meta["m_after_delete"]
    assert meta["m_after_delete"] <= problem.m


def test_largest_feasible_degree_snaps_odd_kmax_to_even_for_index3_case():
    n = 34
    m = 340
    score = _score_index_candidate(n=n, m=m, index=3)
    assert score.k_max % 2 == 1

    group = _build_group(target_n=n, index=3, lift_nodes=score.lift_nodes)
    subgroup, nontrivial_union = _split_cosets(group, 3)
    nontrivial_sampler = _InverseClosedSampler(group, nontrivial_union)
    subgroup_sampler = _InverseClosedSampler(group, [g for g in subgroup if g != 0])

    snapped = _largest_feasible_degree_leq(
        max_degree=score.k_max,
        min_degree=1,
        nontrivial_size=len(nontrivial_union),
        nontrivial_sampler=nontrivial_sampler,
        subgroup_sampler=subgroup_sampler,
    )
    assert snapped == score.k_max - 1


def test_cayley_generator_uses_feasible_degree_cap_when_kmax_is_infeasible():
    problem = DesignProblem(n=34, m=340, seed=0)
    config = DesignConfig(cayley_samples=16, cayley_degree_slack=0, cayley_eval_batch_size=16)
    generator = CayleyBackboneGenerator()

    candidate = generator.generate(problem, config, random.Random(problem.seed))

    assert candidate is not None
    meta = candidate.metadata
    assert meta["k_max_selected"] == 21
    assert meta["k_feasible_cap"] == 20
    assert meta["generator_size"] == 20


def test_extended_index2_overlap_adds_index2_in_0p5_to_0p6_band():
    default_scores = _score_index_candidates(n=34, m=320, index2_upper=0.5)
    extended_scores = _score_index_candidates(n=34, m=320, index2_upper=0.6)

    default_valid = [score.index for score in _valid_scores(default_scores)]
    extended_valid = [score.index for score in _valid_scores(extended_scores)]

    assert default_valid == [3]
    assert extended_valid == [2, 3]


def test_multi_index_overlap_mode_evaluates_all_valid_indices():
    problem = DesignProblem(n=34, m=320, seed=0)
    config = DesignConfig(
        cayley_samples=24,
        cayley_degree_slack=0,
        cayley_eval_batch_size=24,
        cayley_multi_index_overlap=True,
        cayley_index2_overlap_high=0.6,
    )
    generator = CayleyBackboneGenerator()

    candidate = generator.generate(problem, config, random.Random(problem.seed))

    assert candidate is not None
    meta = candidate.metadata
    assert meta["multi_index_overlap"] is True
    assert abs(float(meta["index2_pocket_high"]) - 0.6) < 1e-12
    assert meta["evaluated_indices"] == "2,3"
    assert int(meta["selected_index"]) in (2, 3)


def test_random_cayley_generator_returns_candidate_with_random_mode_metadata():
    problem = DesignProblem(n=34, m=320, seed=0)
    config = DesignConfig(
        cayley_samples=24,
        cayley_degree_slack=0,
        cayley_eval_batch_size=24,
        cayley_generation_mode="random",
    )
    generator = RandomCayleyBackboneGenerator()

    candidate = generator.generate(problem, config, random.Random(problem.seed))

    assert candidate is not None
    assert candidate.graph.number_of_edges() <= problem.m
    assert candidate.metadata["generation_mode"] == "random"


def test_solver_uses_random_cayley_mode_when_configured():
    config = DesignConfig(
        routing_mode="always_cayley",
        cayley_samples=24,
        cayley_degree_slack=0,
        cayley_eval_batch_size=24,
        cayley_generation_mode="random",
    )
    solver = HybridGraphDesigner(config=config)
    problem = DesignProblem(n=34, m=320, seed=0)

    result = solver.solve(problem)

    assert result.selected_family == "cayley"
    bb_meta = result.metadata.get("backbone_metadata", {})
    assert bb_meta.get("generation_mode") == "random"
