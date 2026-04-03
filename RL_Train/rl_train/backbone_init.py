from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping

import numpy as np

from .config import BackboneInitConfig
from .graph_math import m_target_from_rho


INIT_METADATA_KEYS: tuple[str, ...] = (
    "init_mode",
    "init_route_label",
    "init_family",
    "init_edges",
    "init_lambda2",
    "init_build_runtime_sec",
    "init_cache_hit",
    "init_selected_index",
    "init_errors",
)


def default_path_init_metadata(n: int) -> dict[str, Any]:
    return {
        "init_mode": "path",
        "init_route_label": "path",
        "init_family": "path",
        "init_edges": int(max(0, n - 1)),
        "init_lambda2": 0.0,
        "init_build_runtime_sec": 0.0,
        "init_cache_hit": False,
        "init_selected_index": "",
        "init_errors": "",
    }


def normalize_init_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    n: int,
    init_mode: str,
) -> dict[str, Any]:
    base = default_path_init_metadata(n)
    if init_mode == "backbone":
        base.update(
            {
                "init_mode": "backbone",
                "init_route_label": "",
                "init_family": "",
                "init_edges": 0,
                "init_lambda2": 0.0,
                "init_build_runtime_sec": 0.0,
                "init_cache_hit": False,
                "init_selected_index": "",
                "init_errors": "",
            }
        )

    if metadata:
        for key in INIT_METADATA_KEYS:
            if key in metadata:
                base[key] = metadata[key]

    base["init_mode"] = str(base["init_mode"])
    base["init_route_label"] = str(base["init_route_label"])
    base["init_family"] = str(base["init_family"])
    base["init_edges"] = int(base["init_edges"])
    base["init_lambda2"] = float(base["init_lambda2"])
    base["init_build_runtime_sec"] = float(base["init_build_runtime_sec"])
    base["init_cache_hit"] = bool(base["init_cache_hit"])
    base["init_selected_index"] = str(base["init_selected_index"])
    base["init_errors"] = str(base["init_errors"])
    return base


class BackboneInitializer:
    def __init__(self, cfg: BackboneInitConfig):
        project_root = Path(__file__).resolve().parents[1]
        src_dir = project_root / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from graph_design.backbones.cayley import CayleyBackboneGenerator
        from graph_design.backbones.envelope import EnvelopeBackboneGenerator
        from graph_design.config import DesignConfig
        from graph_design.routing import DensityRoutingPolicy
        from graph_design.types import DesignProblem

        self._DesignProblem = DesignProblem
        self._cfg = cfg
        self._design_cfg = DesignConfig(
            rho_very_low=float(cfg.rho_very_low),
            rho_low_mix_end=float(cfg.rho_low_mix_end),
            rho_low=float(cfg.rho_low),
            rho_high=float(cfg.rho_high),
            routing_mode=str(cfg.routing_mode),
            cayley_samples=int(cfg.cayley_samples),
            cayley_degree_slack=int(cfg.cayley_degree_slack),
            cayley_multi_index_overlap=bool(cfg.cayley_multi_index_overlap),
            cayley_index2_overlap_high=float(cfg.cayley_index2_overlap_high),
            cayley_spectral_backend=str(cfg.cayley_spectral_backend),
            cayley_eval_batch_size=int(cfg.cayley_eval_batch_size),
            force_connected_backbone=bool(cfg.force_connected_backbone),
        )
        self._config_signature = (
            self._design_cfg.rho_very_low,
            self._design_cfg.rho_low_mix_end,
            self._design_cfg.rho_low,
            self._design_cfg.rho_high,
            self._design_cfg.routing_mode,
            self._design_cfg.cayley_samples,
            self._design_cfg.cayley_degree_slack,
            self._design_cfg.cayley_multi_index_overlap,
            self._design_cfg.cayley_index2_overlap_high,
            self._design_cfg.cayley_spectral_backend,
            self._design_cfg.cayley_eval_batch_size,
            self._design_cfg.force_connected_backbone,
        )
        self._routing = DensityRoutingPolicy(self._design_cfg)
        self._envelope = EnvelopeBackboneGenerator()
        self._cayley = CayleyBackboneGenerator()
        self._cache: dict[
            tuple[int, int, int, tuple[Any, ...]],
            tuple[np.ndarray, dict[str, Any]],
        ] = {}

    def _family_generator(self, family: str):
        if family == "cayley":
            return self._cayley
        if family == "envelope":
            return self._envelope
        raise ValueError(f"Unknown backbone family: {family}")

    @staticmethod
    def _candidate_key(candidate: Any) -> tuple[float, int]:
        return (float(candidate.backbone_lambda2), int(candidate.graph.number_of_edges()))

    def _build_candidate(
        self,
        *,
        family: str,
        problem: Any,
        seed: int,
    ):
        rng = random.Random(seed * 10007 + (1 if family == "cayley" else 2))
        return self._family_generator(family).generate(problem, self._design_cfg, rng)

    def _build_backbone(
        self,
        *,
        n: int,
        rho_target: float,
        seed: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        m_target = m_target_from_rho(n, rho_target)
        problem = self._DesignProblem(n=n, m=m_target, seed=seed)
        route = self._routing.route(problem.density)
        start = time.perf_counter()

        candidates: list[Any] = []
        errors: list[str] = []

        for family in route.families:
            candidate = self._build_candidate(family=family, problem=problem, seed=seed)
            if candidate is None:
                errors.append(f"{family}:no_candidate")
                continue
            m_backbone = int(candidate.graph.number_of_edges())
            if m_backbone > m_target:
                raise RuntimeError(
                    f"Backbone edge budget violation: family={family} "
                    f"m_backbone={m_backbone} > m_target={m_target}"
                )
            candidates.append(candidate)

        if not candidates:
            fallback = tuple(
                family for family in ("cayley", "envelope") if family not in route.families
            )
            for family in fallback:
                candidate = self._build_candidate(family=family, problem=problem, seed=seed)
                if candidate is None:
                    errors.append(f"{family}:no_candidate")
                    continue
                m_backbone = int(candidate.graph.number_of_edges())
                if m_backbone > m_target:
                    raise RuntimeError(
                        f"Backbone edge budget violation: family={family} "
                        f"m_backbone={m_backbone} > m_target={m_target}"
                    )
                candidates.append(candidate)

        if not candidates:
            raise RuntimeError(
                "No feasible backbone candidate for "
                f"(n={n}, m={m_target}, rho={rho_target:.6f}, seed={seed}); "
                f"errors={errors}"
            )

        selected = max(candidates, key=self._candidate_key)

        try:
            import networkx as nx
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("networkx is required for backbone initialization") from exc

        adj = nx.to_numpy_array(
            selected.graph,
            nodelist=list(range(n)),
            dtype=np.uint8,
        )
        np.fill_diagonal(adj, 0)
        m_init = int(np.sum(adj) // 2)
        if m_init > m_target:
            raise RuntimeError(
                "Backbone edge budget violation after adjacency conversion: "
                f"m_backbone={m_init} > m_target={m_target}"
            )

        meta = normalize_init_metadata(
            {
                "init_mode": "backbone",
                "init_route_label": route.label,
                "init_family": str(selected.family),
                "init_edges": int(m_init),
                "init_lambda2": float(selected.backbone_lambda2),
                "init_build_runtime_sec": float(time.perf_counter() - start),
                "init_selected_index": str(selected.metadata.get("selected_index", "")),
                "init_errors": ";".join(errors),
            },
            n=n,
            init_mode="backbone",
        )
        return adj, meta

    def get_initial_adj(
        self,
        *,
        n: int,
        rho_target: float,
        seed: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        m_target = m_target_from_rho(n, rho_target)
        key = (int(n), int(m_target), int(seed), self._config_signature)
        cached = self._cache.get(key)
        if cached is not None:
            adj, info = cached
            reused = dict(info)
            reused["init_cache_hit"] = True
            reused["init_build_runtime_sec"] = 0.0
            return adj.copy(), normalize_init_metadata(reused, n=n, init_mode="backbone")

        adj, info = self._build_backbone(n=n, rho_target=rho_target, seed=seed)
        info["init_cache_hit"] = False
        normalized = normalize_init_metadata(info, n=n, init_mode="backbone")
        self._cache[key] = (adj.copy(), dict(normalized))
        return adj.copy(), normalized

    def meta(self) -> dict[str, Any]:
        return asdict(self._cfg)

    def state_dict(self) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for key, (adj, info) in self._cache.items():
            items.append(
                {
                    "key": key,
                    "adj": adj.copy(),
                    "info": dict(info),
                }
            )
        return {
            "config_signature": self._config_signature,
            "cache_items": items,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        raw_sig = tuple(state.get("config_signature", ()))
        if raw_sig and raw_sig != self._config_signature:
            raise ValueError("Backbone initializer config signature mismatch on resume.")
        cache_items = state.get("cache_items", [])
        restored: dict[
            tuple[int, int, int, tuple[Any, ...]],
            tuple[np.ndarray, dict[str, Any]],
        ] = {}
        for item in cache_items:
            key = tuple(item["key"])
            if len(key) != 4:
                continue
            n, m_target, seed, cfg_sig = key
            normalized_key = (
                int(n),
                int(m_target),
                int(seed),
                tuple(cfg_sig),
            )
            adj = np.asarray(item["adj"], dtype=np.uint8).copy()
            info = normalize_init_metadata(item.get("info"), n=adj.shape[0], init_mode="backbone")
            restored[normalized_key] = (adj, info)
        self._cache = restored
