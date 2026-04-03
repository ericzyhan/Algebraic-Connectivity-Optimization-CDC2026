from __future__ import annotations

from abc import ABC, abstractmethod
import random

from graph_design.config import DesignConfig
from graph_design.types import BackboneCandidate, DesignProblem


class BackboneGenerator(ABC):
    family: str

    @abstractmethod
    def generate(
        self,
        problem: DesignProblem,
        config: DesignConfig,
        rng: random.Random,
    ) -> BackboneCandidate | None:
        raise NotImplementedError
