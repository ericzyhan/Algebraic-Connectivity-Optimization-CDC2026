from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DesignConfig:
    rho_very_low: float = 0.05
    rho_low_mix_end: float = 0.10
    rho_low: float = 0.5
    rho_high: float = 0.75
    routing_mode: str = "window"  # window, always_both, always_envelope, always_cayley
    cayley_samples: int = 500  # Tunable via CLI flag --cayley-samples.
    cayley_degree_slack: int = 0
    cayley_multi_index_overlap: bool = False
    cayley_index2_overlap_high: float = 0.6
    cayley_eval_batch_size: int = 128
    cayley_spectral_backend: str = "cpu"  # cpu, cuda (optional)
    cayley_generation_mode: str = "coset"  # coset, random
    fvg_reuse_cache: bool = True
    fvg_max_steps: int | None = None
    force_connected_backbone: bool = True
