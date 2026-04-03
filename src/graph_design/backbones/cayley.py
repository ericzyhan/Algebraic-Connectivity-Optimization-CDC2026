from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random

import networkx as nx
import numpy as np

from graph_design.backbones.base import BackboneGenerator
from graph_design.config import DesignConfig
from graph_design.types import BackboneCandidate, DesignProblem
from graph_design.utils import relabel_to_integers

try:  # pragma: no cover - optional dependency.
    import cupy as cp
except Exception:  # pragma: no cover - CPU-only environments.
    cp = None


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1) == 0)


def _lift_nodes_needed(n: int, index: int) -> int:
    if index <= 0:
        raise ValueError("index must be positive.")
    return (index - (n % index)) % index


def _index_edge_interval(index: int, *, index2_upper: float = 0.5) -> tuple[float, float]:
    if index == 2:
        return (0.0, index2_upper)
    if index == 3:
        return (0.5, 2.0 / 3.0)
    if index == 4:
        return (2.0 / 3.0, 0.75)
    raise ValueError(f"Unsupported index: {index}")


def _in_open_interval(value: float, low: float, high: float) -> bool:
    return low < value < high


@dataclass(frozen=True)
class _PocketScore:
    index: int
    lift_nodes: int
    lifted_n: int
    k_max: int
    m_proj: int
    edge_ratio: float
    pocket_low: float
    pocket_high: float
    in_pocket: bool


def _score_index_candidate(
    n: int,
    m: int,
    index: int,
    *,
    index2_upper: float = 0.5,
) -> _PocketScore:
    j = _lift_nodes_needed(n, index)
    lifted_n = n + j
    denom = max(1, n - j)
    k_max = int((2 * m) // denom)
    k_max = max(1, min(lifted_n - 1, k_max))
    m_proj = int((k_max * denom) // 2)
    edge_ratio = (2.0 * float(m_proj) / float(n * n)) if n > 0 else 0.0
    pocket_low, pocket_high = _index_edge_interval(index, index2_upper=index2_upper)
    in_pocket = _in_open_interval(edge_ratio, pocket_low, pocket_high)
    return _PocketScore(
        index=index,
        lift_nodes=j,
        lifted_n=lifted_n,
        k_max=k_max,
        m_proj=m_proj,
        edge_ratio=edge_ratio,
        pocket_low=pocket_low,
        pocket_high=pocket_high,
        in_pocket=in_pocket,
    )


def _score_index_candidates(
    n: int,
    m: int,
    *,
    index2_upper: float = 0.5,
) -> dict[int, _PocketScore]:
    return {
        index: _score_index_candidate(
            n=n,
            m=m,
            index=index,
            index2_upper=index2_upper,
        )
        for index in (2, 3, 4)
    }


def _valid_scores(scores: dict[int, _PocketScore]) -> list[_PocketScore]:
    return [scores[idx] for idx in sorted(scores.keys()) if scores[idx].in_pocket]


def _choose_index_from_scores(scores: dict[int, _PocketScore]) -> _PocketScore | None:
    valid = _valid_scores(scores)
    if not valid:
        return None
    return valid[0]


@dataclass(frozen=True)
class _SemidirectGroup:
    order: int
    half: int
    twist: int
    kind: str

    def mul(self, g: int, h: int) -> int:
        a1, b1 = g % self.half, g // self.half
        a2, b2 = h % self.half, h // self.half
        acted = a2 if b1 == 0 else (self.twist * a2) % self.half
        a = (a1 + acted) % self.half
        b = b1 ^ b2
        return a + b * self.half

    def inv(self, g: int) -> int:
        a, b = g % self.half, g // self.half
        if b == 0:
            return (-a) % self.half
        inv_a = (-(self.twist * a)) % self.half
        return inv_a + self.half


@dataclass(frozen=True)
class _CyclicGroup:
    order: int
    kind: str = "cyclic"

    def mul(self, g: int, h: int) -> int:
        return (g + h) % self.order

    def inv(self, g: int) -> int:
        return (-g) % self.order


def _build_group(target_n: int, index: int, lift_nodes: int):
    lifted_n = target_n + lift_nodes
    if index == 2:
        if lifted_n < 2 or lifted_n % 2 != 0:
            raise ValueError(f"Unsupported lifted order {lifted_n} for index-2.")
        half = lifted_n // 2
        use_semidihedral = (
            lift_nodes == 0
            and _is_power_of_two(target_n)
            and target_n >= 16
        )
        if use_semidihedral:
            k = int(math.log2(lifted_n))
            twist = (2 ** (k - 2) - 1) % half
            kind = "semidihedral"
        else:
            twist = (half - 1) % half if half > 1 else 0
            kind = "dihedral"
        return _SemidirectGroup(order=lifted_n, half=half, twist=twist, kind=kind)
    if index in (3, 4):
        return _CyclicGroup(order=lifted_n)
    raise ValueError(f"Unsupported index: {index}")


def _split_cosets(group, index: int) -> tuple[list[int], list[int]]:
    subgroup: list[int] = []
    nontrivial_union: list[int] = []

    for g in range(group.order):
        if index == 2 and isinstance(group, _SemidirectGroup):
            residue = 0 if g < group.half else 1
        else:
            residue = g % index
        if residue == 0:
            subgroup.append(g)
        else:
            nontrivial_union.append(g)

    return subgroup, nontrivial_union


class _InverseClosedSampler:
    def __init__(self, group, pool: list[int]) -> None:
        self.group = group
        self.pool = list(pool)
        self.pool_set = set(pool)
        self.involutions: list[int] = []
        self.pairs: list[tuple[int, int]] = []
        self._build_parts()

    def _build_parts(self) -> None:
        visited: set[int] = set()
        for g in self.pool:
            if g in visited:
                continue
            h = self.group.inv(g)
            if h == g:
                self.involutions.append(g)
                visited.add(g)
                continue
            if h not in self.pool_set:
                continue
            visited.add(g)
            visited.add(h)
            self.pairs.append((g, h))

    def feasible(self, size: int) -> bool:
        return len(self._feasible_involution_counts(size)) > 0

    def sample(self, size: int, rng: random.Random) -> set[int]:
        if size < 0 or size > len(self.pool):
            raise ValueError(f"Invalid subset size={size}, pool={len(self.pool)}.")
        if size == 0:
            return set()

        feasible_t = self._feasible_involution_counts(size)
        if not feasible_t:
            raise ValueError(f"No inverse-closed subset of size {size}.")

        t = rng.choice(feasible_t)
        p = (size - t) // 2
        chosen_inv = rng.sample(self.involutions, t)
        chosen_pairs = rng.sample(self.pairs, p)

        out = set(chosen_inv)
        for a, b in chosen_pairs:
            out.add(a)
            out.add(b)
        return out

    def _feasible_involution_counts(self, size: int) -> list[int]:
        min_t = max(0, size - 2 * len(self.pairs))
        max_t = min(size, len(self.involutions))
        out: list[int] = []
        for t in range(min_t, max_t + 1):
            rem = size - t
            if rem % 2 != 0:
                continue
            if rem // 2 <= len(self.pairs):
                out.append(t)
        return out


def _sample_generators(
    *,
    nontrivial_union: list[int],
    subgroup_no_identity: list[int],
    desired_degree: int,
    rng: random.Random,
    nontrivial_sampler: _InverseClosedSampler,
    subgroup_sampler: _InverseClosedSampler,
) -> set[int] | None:
    if desired_degree <= 0:
        return None

    nontrivial_size = len(nontrivial_union)
    if desired_degree <= nontrivial_size:
        fixed: set[int] = set()
        sampler = nontrivial_sampler
        need = desired_degree
    else:
        fixed = set(nontrivial_union)
        sampler = subgroup_sampler
        need = desired_degree - nontrivial_size

    if not sampler.feasible(need):
        return None
    sampled = sampler.sample(need, rng)
    output = set(fixed)
    output.update(sampled)
    return output


def _degree_sampler_need(
    *,
    desired_degree: int,
    nontrivial_size: int,
):
    if desired_degree <= 0:
        return None
    if desired_degree <= nontrivial_size:
        return "nontrivial", desired_degree
    return "subgroup", desired_degree - nontrivial_size


def _is_degree_feasible(
    *,
    desired_degree: int,
    nontrivial_size: int,
    nontrivial_sampler: _InverseClosedSampler,
    subgroup_sampler: _InverseClosedSampler,
) -> bool:
    sampler_need = _degree_sampler_need(
        desired_degree=desired_degree,
        nontrivial_size=nontrivial_size,
    )
    if sampler_need is None:
        return False
    which, need = sampler_need
    sampler = nontrivial_sampler if which == "nontrivial" else subgroup_sampler
    return sampler.feasible(need)


def _largest_feasible_degree_leq(
    *,
    max_degree: int,
    min_degree: int,
    nontrivial_size: int,
    nontrivial_sampler: _InverseClosedSampler,
    subgroup_sampler: _InverseClosedSampler,
) -> int | None:
    for degree in range(max_degree, min_degree - 1, -1):
        if _is_degree_feasible(
            desired_degree=degree,
            nontrivial_size=nontrivial_size,
            nontrivial_sampler=nontrivial_sampler,
            subgroup_sampler=subgroup_sampler,
        ):
            return degree
    return None


class _LiftDeleteBatchEvaluator:
    def __init__(
        self,
        *,
        group,
        delete_count: int,
        backend: str,
    ) -> None:
        self.group = group
        self.n = group.order
        self.delete_count = delete_count
        self.keep = np.arange(delete_count, self.n, dtype=np.int32)
        if self.keep.size < 2:
            raise ValueError("Need at least 2 kept nodes to evaluate lambda_2.")

        if backend not in {"cpu", "cuda"}:
            raise ValueError(f"Unknown Cayley spectral backend: {backend}")
        if backend == "cuda" and cp is None:
            raise RuntimeError(
                "cayley_spectral_backend='cuda' requires CuPy, but it is unavailable."
            )
        self.backend = backend

        self.rows_np = np.arange(self.n, dtype=np.int32)
        self.diag_full_np = np.arange(self.n, dtype=np.int32)
        self.diag_keep_np = np.arange(self.keep.size, dtype=np.int32)
        self.right_action_np = np.empty((self.n, self.n), dtype=np.int32)

        for s in range(self.n):
            for g in range(self.n):
                self.right_action_np[s, g] = group.mul(g, s)

        if self.backend == "cuda":  # pragma: no cover - optional CUDA path.
            self.rows_cp = cp.asarray(self.rows_np)
            self.diag_full_cp = cp.asarray(self.diag_full_np)
            self.diag_keep_cp = cp.asarray(self.diag_keep_np)
            self.keep_cp = cp.asarray(self.keep)
            self.right_action_cp = cp.asarray(self.right_action_np)

    def lambda2_edges_from_index_matrix(
        self,
        generator_index_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(generator_index_matrix, dtype=np.int32)
        if idx.ndim != 2:
            raise ValueError("generator_index_matrix must have shape [batch, degree].")
        if idx.shape[0] == 0:
            empty = np.empty((0,), dtype=np.float64)
            empty_i = np.empty((0,), dtype=np.int64)
            return empty, empty_i
        if idx.shape[1] == 0:
            zeros = np.zeros((idx.shape[0],), dtype=np.float64)
            zeros_i = np.zeros((idx.shape[0],), dtype=np.int64)
            return zeros, zeros_i
        if self.backend == "cuda":  # pragma: no cover - optional CUDA path.
            return self._lambda2_edges_cuda_batch(idx)
        return self._lambda2_edges_cpu_batch(idx)

    def _lambda2_edges_cpu_batch(
        self,
        generator_index_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(generator_index_matrix, dtype=np.int32)
        b, d = idx.shape
        n = self.n

        A = np.zeros((b, n, n), dtype=np.float64)
        batch_idx = np.arange(b, dtype=np.int32)[:, None]
        rows = self.rows_np[None, :]
        for j in range(d):
            s = idx[:, j]
            cols = self.right_action_np[s]
            A[batch_idx, rows, cols] = 1.0
        A = np.maximum(A, np.transpose(A, (0, 2, 1)))
        A[:, self.diag_full_np, self.diag_full_np] = 0.0

        sub = A[:, self.keep][:, :, self.keep]
        deg = np.sum(sub, axis=2)
        edges = np.rint(np.sum(deg, axis=1) / 2.0).astype(np.int64)

        L = -sub
        L[:, self.diag_keep_np, self.diag_keep_np] = deg
        evals = np.linalg.eigvalsh(L)
        l2 = np.asarray(evals[:, 1], dtype=np.float64)
        return l2, edges

    def _lambda2_edges_cuda_batch(  # pragma: no cover - optional CUDA path.
        self,
        generator_index_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = cp.asarray(generator_index_matrix, dtype=cp.int32)
        b, d = idx.shape
        n = self.n

        A = cp.zeros((b, n, n), dtype=cp.float64)
        batch_idx = cp.arange(b, dtype=cp.int32)[:, None]
        rows = self.rows_cp[None, :]
        for j in range(d):
            s = idx[:, j]
            cols = self.right_action_cp[s]
            A[batch_idx, rows, cols] = 1.0
        A = cp.maximum(A, cp.transpose(A, (0, 2, 1)))
        A[:, self.diag_full_cp, self.diag_full_cp] = 0.0

        sub = A[:, self.keep_cp][:, :, self.keep_cp]
        deg = cp.sum(sub, axis=2)
        edges = cp.rint(cp.sum(deg, axis=1) / 2.0).astype(cp.int64)

        L = -sub
        L[:, self.diag_keep_cp, self.diag_keep_cp] = deg
        evals = cp.linalg.eigvalsh(L)
        l2 = cp.asnumpy(evals[:, 1])
        edges_np = cp.asnumpy(edges)
        return np.asarray(l2, dtype=np.float64), np.asarray(edges_np, dtype=np.int64)


def _build_cayley_graph(group, generators: set[int]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(group.order))
    generator_list = list(generators)
    for g in range(group.order):
        for s in generator_list:
            h = group.mul(g, s)
            if g != h:
                graph.add_edge(g, h)
    return graph


@dataclass
class CayleyBackboneGenerator(BackboneGenerator):
    family: str = "cayley"
    generation_mode: str = "coset"  # coset, random

    def generate(
        self,
        problem: DesignProblem,
        config: DesignConfig,
        rng: random.Random,
    ) -> BackboneCandidate | None:
        mode = self._normalized_mode()
        index2_upper = (
            config.cayley_index2_overlap_high
            if config.cayley_multi_index_overlap
            else 0.5
        )
        scores = _score_index_candidates(
            n=problem.n,
            m=problem.m,
            index2_upper=index2_upper,
        )
        valid_scores = _valid_scores(scores)
        if not valid_scores:
            return None

        if not config.cayley_multi_index_overlap:
            selected = _choose_index_from_scores(scores)
            if selected is None:
                return None
            return self._generate_for_score(
                problem=problem,
                config=config,
                rng=rng,
                mode=mode,
                selected=selected,
                scores=scores,
                index2_upper=index2_upper,
                evaluated_indices=[selected.index],
            )

        candidates: list[BackboneCandidate] = []
        evaluated_indices: list[int] = [score.index for score in valid_scores]
        for selected in valid_scores:
            candidate = self._generate_for_score(
                problem=problem,
                config=config,
                rng=rng,
                mode=mode,
                selected=selected,
                scores=scores,
                index2_upper=index2_upper,
                evaluated_indices=evaluated_indices,
            )
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return None
        return max(candidates, key=self._candidate_key)

    @staticmethod
    def _candidate_key(candidate: BackboneCandidate) -> tuple[float, int]:
        return (candidate.backbone_lambda2, candidate.graph.number_of_edges())

    def _normalized_mode(self) -> str:
        mode = str(self.generation_mode).strip().lower()
        if mode not in {"coset", "random"}:
            raise ValueError(
                f"Unsupported Cayley generation_mode={self.generation_mode!r}. "
                "Use 'coset' or 'random'."
            )
        return mode

    def _generate_for_score(
        self,
        *,
        problem: DesignProblem,
        config: DesignConfig,
        rng: random.Random,
        mode: str,
        selected: _PocketScore,
        scores: dict[int, _PocketScore],
        index2_upper: float,
        evaluated_indices: list[int],
    ) -> BackboneCandidate | None:
        group = _build_group(
            target_n=problem.n,
            index=selected.index,
            lift_nodes=selected.lift_nodes,
        )
        max_degree = min(group.order - 1, selected.k_max)

        random_pool_size: int | None = None
        if mode == "coset":
            subgroup, nontrivial_union = _split_cosets(group, selected.index)
            if not nontrivial_union:
                return None

            subgroup_no_identity = [g for g in subgroup if g != 0]
            nontrivial_sampler = _InverseClosedSampler(group, nontrivial_union)
            subgroup_sampler = _InverseClosedSampler(group, subgroup_no_identity)

            feasible_cap = _largest_feasible_degree_leq(
                max_degree=max_degree,
                min_degree=1,
                nontrivial_size=len(nontrivial_union),
                nontrivial_sampler=nontrivial_sampler,
                subgroup_sampler=subgroup_sampler,
            )

            def degree_feasible(degree: int) -> bool:
                return _is_degree_feasible(
                    desired_degree=degree,
                    nontrivial_size=len(nontrivial_union),
                    nontrivial_sampler=nontrivial_sampler,
                    subgroup_sampler=subgroup_sampler,
                )

            def sample_for_degree(degree: int) -> set[int] | None:
                return _sample_generators(
                    nontrivial_union=nontrivial_union,
                    subgroup_no_identity=subgroup_no_identity,
                    desired_degree=degree,
                    rng=rng,
                    nontrivial_sampler=nontrivial_sampler,
                    subgroup_sampler=subgroup_sampler,
                )

        else:
            full_pool = [g for g in range(group.order) if g != 0]
            if not full_pool:
                return None
            random_pool_size = len(full_pool)
            random_sampler = _InverseClosedSampler(group, full_pool)

            def degree_feasible(degree: int) -> bool:
                return random_sampler.feasible(degree)

            feasible_cap = None
            for degree in range(max_degree, 0, -1):
                if degree_feasible(degree):
                    feasible_cap = degree
                    break

            def sample_for_degree(degree: int) -> set[int] | None:
                if not degree_feasible(degree):
                    return None
                return random_sampler.sample(degree, rng)

        if feasible_cap is None:
            return None

        low_d = max(1, feasible_cap - config.cayley_degree_slack)
        feasible_degrees = [
            degree
            for degree in range(low_d, feasible_cap + 1)
            if degree_feasible(degree)
        ]
        if not feasible_degrees:
            return None

        sample_count = max(1, config.cayley_samples)
        samples_by_degree: dict[int, list[np.ndarray]] = defaultdict(list)
        for _ in range(sample_count):
            desired_degree = (
                rng.choice(feasible_degrees)
                if len(feasible_degrees) > 1
                else feasible_degrees[0]
            )
            try:
                generators = sample_for_degree(desired_degree)
            except ValueError:
                continue
            if not generators:
                continue
            idx = np.fromiter(
                sorted(generators),
                dtype=np.int32,
                count=len(generators),
            )
            samples_by_degree[len(idx)].append(idx)

        if not samples_by_degree:
            return None

        evaluator = _LiftDeleteBatchEvaluator(
            group=group,
            delete_count=selected.lift_nodes,
            backend=config.cayley_spectral_backend,
        )
        batch_size = max(1, config.cayley_eval_batch_size)

        best_key: tuple[float, int] | None = None
        best_indices: np.ndarray | None = None
        best_lambda2: float | None = None
        best_m_after: int | None = None
        best_m_lift: int | None = None
        best_deleted: int | None = None

        for degree, rows in samples_by_degree.items():
            for start in range(0, len(rows), batch_size):
                chunk = rows[start : start + batch_size]
                mat = np.stack(chunk, axis=0)
                l2_vals, edge_vals = evaluator.lambda2_edges_from_index_matrix(mat)
                m_lift = int((degree * group.order) // 2)
                for idx, generators_idx in enumerate(chunk):
                    l2 = float(l2_vals[idx])
                    m_after = int(edge_vals[idx])
                    deleted_edges = m_lift - m_after
                    if m_after > problem.m:
                        continue
                    if config.force_connected_backbone and l2 <= 1e-9:
                        continue
                    key = (l2, m_after)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_indices = generators_idx
                        best_lambda2 = l2
                        best_m_after = m_after
                        best_m_lift = m_lift
                        best_deleted = deleted_edges

        if best_indices is None or best_lambda2 is None:
            return None

        graph = _build_cayley_graph(group, set(int(x) for x in best_indices))
        if selected.lift_nodes > 0:
            graph = graph.copy()
            graph.remove_nodes_from(range(selected.lift_nodes))
            graph = relabel_to_integers(graph)

        if graph.number_of_nodes() != problem.n:
            return None
        if graph.number_of_edges() > problem.m:
            return None
        if config.force_connected_backbone and not nx.is_connected(graph):
            return None

        metadata: dict[str, float | int | bool | str] = {
            "group_kind": group.kind,
            "group_order": group.order,
            "selected_index": selected.index,
            "j_selected": selected.lift_nodes,
            "n_lift": selected.lifted_n,
            "k_max_selected": selected.k_max,
            "k_feasible_cap": int(feasible_cap),
            "m_proj_selected": selected.m_proj,
            "e_selected": selected.edge_ratio,
            "generator_size": int(best_indices.size),
            "m_lift": int(best_m_lift if best_m_lift is not None else 0),
            "m_after_delete": int(best_m_after if best_m_after is not None else 0),
            "deleted_edges": int(best_deleted if best_deleted is not None else 0),
            "multi_index_overlap": bool(config.cayley_multi_index_overlap),
            "index2_pocket_high": float(index2_upper),
            "evaluated_indices": ",".join(str(x) for x in evaluated_indices),
            "generation_mode": mode,
        }
        if random_pool_size is not None:
            metadata["random_pool_size"] = int(random_pool_size)
        for index, score in scores.items():
            metadata[f"j_{index}"] = score.lift_nodes
            metadata[f"k_max_{index}"] = score.k_max
            metadata[f"m_proj_{index}"] = score.m_proj
            metadata[f"e_{index}"] = score.edge_ratio
            metadata[f"in_pocket_{index}"] = score.in_pocket

        return BackboneCandidate(
            family=self.family,
            graph=graph,
            backbone_lambda2=best_lambda2,
            metadata=metadata,
        )


@dataclass
class RandomCayleyBackboneGenerator(CayleyBackboneGenerator):
    generation_mode: str = "random"
