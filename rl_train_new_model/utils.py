from __future__ import annotations

import hashlib
import io
from statistics import mean
from typing import Iterable, List

import torch


def state_dict_hash(state_dict) -> str:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def safe_mean(values: Iterable[float]) -> float:
    data: List[float] = list(values)
    if not data:
        return 0.0
    return float(mean(data))
