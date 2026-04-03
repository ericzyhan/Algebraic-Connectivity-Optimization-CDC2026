from __future__ import annotations

from abc import ABC, abstractmethod
import random

import networkx as nx


class CompletionPolicy(ABC):
    @abstractmethod
    def complete(
        self,
        graph: nx.Graph,
        target_edges: int,
        rng: random.Random,
        max_steps: int | None = None,
    ) -> nx.Graph:
        raise NotImplementedError
