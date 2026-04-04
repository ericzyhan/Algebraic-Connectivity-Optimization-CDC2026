from __future__ import annotations

import random
import time

from graph_design.backbones.cayley import CayleyBackboneGenerator
from graph_design.backbones.envelope import EnvelopeBackboneGenerator
from graph_design.baselines.registry import create_baseline
from graph_design.completion.base import CompletionPolicy
from graph_design.completion.fvg import FVGCompletionPolicy
from graph_design.config import DesignConfig
from graph_design.routing import DensityRoutingPolicy, RouteDecision
from graph_design.spectral import algebraic_connectivity
from graph_design.types import CandidateEvaluation, DesignProblem, DesignResult
from graph_design.utils import validate_problem


class HybridGraphDesigner:
    def __init__(
        self,
        config: DesignConfig | None = None,
        completion_policy: CompletionPolicy | None = None,
        envelope_generator: EnvelopeBackboneGenerator | None = None,
        cayley_generator: CayleyBackboneGenerator | None = None,
    ) -> None:
        self.config = config or DesignConfig()
        self.routing = DensityRoutingPolicy(self.config)
        self.completion_policy = completion_policy or FVGCompletionPolicy()
        self.envelope_generator = envelope_generator or EnvelopeBackboneGenerator()
        self.cayley_generator = cayley_generator or CayleyBackboneGenerator()

    def solve(self, problem: DesignProblem) -> DesignResult:
        validate_problem(problem)
        start_total = time.perf_counter()
        rng = random.Random(problem.seed)
        route = self.routing.route(problem.density)

        selected_evals: list[CandidateEvaluation] = []
        selected_errors: list[dict] = []
        for family in route.families:
            evaluation, error = self._evaluate_family(problem, family, rng, route)
            if evaluation is not None:
                selected_evals.append(evaluation)
            elif error is not None:
                selected_errors.append(error)

        fallback_used = False
        chosen_pool = selected_evals
        if not selected_evals:
            fallback_used = True
            fallback_families = tuple(
                f for f in ("cayley", "envelope") if f not in route.families
            )
            for family in fallback_families:
                evaluation, error = self._evaluate_family(problem, family, rng, route)
                if evaluation is not None:
                    chosen_pool.append(evaluation)
                elif error is not None:
                    selected_errors.append(error)

        if chosen_pool:
            chosen = self._select_best(chosen_pool)
            return DesignResult(
                problem=problem,
                route_label=route.label,
                selected_family=chosen.family,
                graph=chosen.final_graph,
                final_lambda2=chosen.final_lambda2,
                backbone_lambda2=chosen.backbone_lambda2,
                completion_gain=chosen.completion_gain,
                runtime_sec=time.perf_counter() - start_total,
                fallback_used=fallback_used,
                metadata={
                    "candidate_count": len(chosen_pool),
                    "route_families": route.families,
                    "candidate_summaries": [
                        {
                            "family": item.family,
                            "backbone_edges": item.backbone_edges,
                            "backbone_lambda2": item.backbone_lambda2,
                            "final_lambda2": item.final_lambda2,
                        }
                        for item in chosen_pool
                    ],
                    "errors": selected_errors,
                    **chosen.metadata,
                },
            )

        baseline_solver = create_baseline("fvg")
        baseline_result = baseline_solver.solve(problem, self.config, rng)
        return DesignResult(
            problem=problem,
            route_label=route.label,
            selected_family="fvg_fallback",
            graph=baseline_result.graph,
            final_lambda2=baseline_result.final_lambda2,
            backbone_lambda2=baseline_result.final_lambda2,
            completion_gain=0.0,
            runtime_sec=time.perf_counter() - start_total,
            fallback_used=True,
            metadata={
                "fallback_reason": "no feasible backbone candidate",
                "errors": selected_errors,
                "baseline_metadata": baseline_result.metadata,
            },
        )

    def _evaluate_family(
        self,
        problem: DesignProblem,
        family: str,
        rng: random.Random,
        route: RouteDecision,
    ) -> tuple[CandidateEvaluation | None, dict | None]:
        generator = {
            "cayley": self.cayley_generator,
            "envelope": self.envelope_generator,
        }.get(family)
        if generator is None:
            return None, {"family": family, "error": "unknown family"}

        start = time.perf_counter()
        try:
            candidate = generator.generate(problem, self.config, rng)
            if candidate is None:
                return None, {"family": family, "error": "no candidate"}
            if candidate.graph.number_of_edges() > problem.m:
                return None, {
                    "family": family,
                    "error": "candidate exceeds target edge budget",
                }

            completed = self.completion_policy.complete(
                graph=candidate.graph,
                target_edges=problem.m,
                rng=rng,
                max_steps=self.config.fvg_max_steps,
            )
            if completed.number_of_edges() != problem.m:
                return None, {"family": family, "error": "completion failed exact budget"}
            final_l2 = algebraic_connectivity(completed)
            runtime = time.perf_counter() - start

            return (
                CandidateEvaluation(
                    family=family,
                    route_label=route.label,
                    backbone_edges=candidate.graph.number_of_edges(),
                    backbone_lambda2=candidate.backbone_lambda2,
                    final_graph=completed,
                    final_lambda2=final_l2,
                    completion_gain=final_l2 - candidate.backbone_lambda2,
                    runtime_sec=runtime,
                    metadata={
                        "backbone_metadata": candidate.metadata,
                    },
                ),
                None,
            )
        except Exception as exc:  # pragma: no cover
            return None, {"family": family, "error": str(exc)}

    @staticmethod
    def _select_best(candidates: list[CandidateEvaluation]) -> CandidateEvaluation:
        return max(
            candidates,
            key=lambda item: (
                item.final_lambda2,
                item.backbone_lambda2,
                item.backbone_edges,
            ),
        )
