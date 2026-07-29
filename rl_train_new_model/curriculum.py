from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from .config import CurriculumConfig, CurriculumPhaseConfig


class CurriculumScheduler:
    def __init__(self, curriculum_config: CurriculumConfig):
        self.phases: List[CurriculumPhaseConfig] = sorted(
            curriculum_config.phases,
            key=lambda p: p.start_env_step,
        )
        if not self.phases:
            raise ValueError("Curriculum must contain at least one phase")

    def get_phase(self, global_env_step: int) -> CurriculumPhaseConfig:
        for phase in self.phases:
            if phase.start_env_step <= global_env_step < phase.end_env_step:
                return phase
        return self.phases[-1]

    def sample_n(self, global_env_step: int, rng) -> int:
        phase = self.get_phase(global_env_step)
        return rng.randint(phase.n_min, phase.n_max)

    def state_dict(self) -> Dict:
        return {"phases": [asdict(p) for p in self.phases]}

    def load_state_dict(self, state: Dict) -> None:
        del state  # phases are static from config

    def phase_name(self, global_env_step: int) -> str:
        return self.get_phase(global_env_step).name
