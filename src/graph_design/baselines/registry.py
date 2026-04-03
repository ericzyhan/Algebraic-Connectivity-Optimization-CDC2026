from __future__ import annotations

from graph_design.baselines.base import BaselineSolver
from graph_design.baselines.fvg import FVGBaseSolver
from graph_design.baselines.reverse_fvg import ReverseFVGBaseSolver


_REGISTRY: dict[str, type[BaselineSolver]] = {}


def register_baseline(name: str, solver_cls: type[BaselineSolver]) -> None:
    _REGISTRY[name] = solver_cls


def create_baseline(name: str) -> BaselineSolver:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown baseline solver: {name}")
    return _REGISTRY[name]()


def list_baselines() -> list[str]:
    return sorted(_REGISTRY.keys())


register_baseline("fvg", FVGBaseSolver)
register_baseline("reverse_fvg", ReverseFVGBaseSolver)
