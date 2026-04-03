from __future__ import annotations

from dataclasses import dataclass

from graph_design.config import DesignConfig


@dataclass(frozen=True)
class RouteDecision:
    label: str
    families: tuple[str, ...]


@dataclass
class DensityRoutingPolicy:
    config: DesignConfig

    def route(self, density: float) -> RouteDecision:
        mode = self.config.routing_mode
        if mode == "always_both":
            return RouteDecision(label="both", families=("cayley", "envelope"))
        if mode == "always_envelope":
            return RouteDecision(label="envelope_only", families=("envelope",))
        if mode == "always_cayley":
            return RouteDecision(label="cayley_only", families=("cayley",))
        if mode != "window":
            raise ValueError(f"Unknown routing_mode: {mode}")

        low_mix_end = min(
            max(self.config.rho_low_mix_end, self.config.rho_very_low),
            self.config.rho_low,
        )

        if density < self.config.rho_very_low:
            return RouteDecision(label="envelope_only", families=("envelope",))
        if density <= low_mix_end:
            return RouteDecision(label="both", families=("cayley", "envelope"))
        if density < self.config.rho_low:
            return RouteDecision(label="cayley_only", families=("cayley",))
        if density > self.config.rho_high:
            return RouteDecision(label="envelope_only", families=("envelope",))
        return RouteDecision(label="both", families=("cayley", "envelope"))
