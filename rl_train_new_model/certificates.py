from __future__ import annotations

from typing import Dict

from .spectral import SpectralTracker


def log_diagnostics(tracker: SpectralTracker, *, k_remaining: int) -> Dict[str, float]:
    """Lightweight, logging-only certificates (docs/optimality-gap.md S1, S2).

    Neither bound gates training or candidate selection -- they are reported
    per step as a running budget / trajectory diagnostic, per the docs'
    explicit instruction to use the resistance sandwich as "a trajectory
    diagnostic and a reporting device, not the OPT certificate."
    """
    interlacing = tracker.interlacing_bound(k_remaining)
    lower, upper, loose_upper = tracker.resistance_sandwich()
    return {
        "cert_interlacing_bound": float(interlacing),
        "cert_resistance_lower": float(lower),
        "cert_resistance_upper": float(upper),
        "cert_resistance_loose_upper": float(loose_upper),
        "drift_residual": float(tracker.drift_residual),
        "multiplicity": int(tracker.multiplicity),
        "soft_degenerate": bool(tracker.soft_degenerate),
        "secular_bracket_width": (
            float(tracker.last_secular_bracket[1] - tracker.last_secular_bracket[0])
            if tracker.last_secular_bracket is not None
            else 0.0
        ),
    }
