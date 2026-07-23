"""Shared config for the lite alpha-sweep pipeline.

The 3-term reward blend is:
    r_t = alpha_1 * Delta(lambda_2)/n + alpha_2 * eta_R * Delta(R_G)/n^2 + alpha_3 * eta_P * Delta(P_min)

alpha_1 weights the greedy algebraic-connectivity term, alpha_2 the effective
graph resistance term, alpha_3 the P_min term. This module defines the set of
(alpha_1, alpha_2, alpha_3) combinations to sweep, and the density buckets used
to summarize algebraic connectivity at "specific densities" rather than only
pointwise across a continuous rho sweep.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlphaCombo:
    tag: str
    alpha_1: float
    alpha_2: float
    alpha_3: float


# Corners + interior points of the alpha simplex (alpha_1 + alpha_2 + alpha_3 == 1):
# the three pure single-heuristic ablations, the naive-balanced split, and the
# resistance-leaning split currently checked into lite_config.yaml.
DEFAULT_ALPHA_COMBOS: list[AlphaCombo] = [
    AlphaCombo("lambda2_only", 1.00, 0.00, 0.00),
    AlphaCombo("resistance_only", 0.00, 1.00, 0.00),
    AlphaCombo("pmin_only", 0.00, 0.00, 1.00),
    AlphaCombo("balanced", 0.34, 0.33, 0.33),
    AlphaCombo("resistance_leaning", 0.20, 0.30, 0.50),
]

# (bucket_name, rho_lo_inclusive, rho_hi_exclusive)
DENSITY_BUCKETS: list[tuple[str, float, float]] = [
    ("sparse", 0.0, 1.0 / 3.0),
    ("medium", 1.0 / 3.0, 2.0 / 3.0),
    ("dense", 2.0 / 3.0, 1.0 + 1e-9),
]


def parse_alpha_combos(spec: str | None) -> list[AlphaCombo]:
    """Parse a CLI spec of the form 'tag:a1,a2,a3;tag2:a1,a2,a3' into AlphaCombos.

    Returns DEFAULT_ALPHA_COMBOS when spec is None/empty.
    """
    if not spec:
        return list(DEFAULT_ALPHA_COMBOS)
    combos: list[AlphaCombo] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        tag, sep, triple = chunk.partition(":")
        if not sep:
            raise ValueError(f"Alpha combo '{chunk}' must be of the form tag:a1,a2,a3")
        parts = [p.strip() for p in triple.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Alpha combo '{chunk}' must have exactly 3 comma-separated values")
        a1, a2, a3 = (float(p) for p in parts)
        combos.append(AlphaCombo(tag.strip(), a1, a2, a3))
    if not combos:
        raise ValueError(f"No alpha combos parsed from spec: {spec!r}")
    return combos


def bucket_for_rho(rho: float) -> str:
    for name, lo, hi in DENSITY_BUCKETS:
        if lo <= rho < hi:
            return name
    return DENSITY_BUCKETS[-1][0]
