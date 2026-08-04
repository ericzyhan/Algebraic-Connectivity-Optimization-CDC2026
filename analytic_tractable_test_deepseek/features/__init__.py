from __future__ import annotations

from .full import (
    build_candidate_features,
    build_edge_index,
    build_global_features,
    build_node_features,
)

try:  # pragma: no cover - lite features are optional in this package
    from .lite import (
        build_candidate_features as build_candidate_features_lite,
        build_edge_index as build_edge_index_lite,
        build_global_features as build_global_features_lite,
        build_node_features as build_node_features_lite,
    )

    _HAS_LITE = True
except ImportError:  # pragma: no cover - package has no features/lite.py
    build_candidate_features_lite = None
    build_edge_index_lite = None
    build_global_features_lite = None
    build_node_features_lite = None
    _HAS_LITE = False

__all__ = [
    "build_candidate_features",
    "build_edge_index",
    "build_global_features",
    "build_node_features",
    "_HAS_LITE",
]
if _HAS_LITE:  # pragma: no cover
    __all__ += [
        "build_candidate_features_lite",
        "build_edge_index_lite",
        "build_global_features_lite",
        "build_node_features_lite",
    ]
