from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import torch

from .config import DeterminismConfig


def apply_determinism(cfg: DeterminismConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.num_threads)

    if cfg.strict:
        os.environ["PYTHONHASHSEED"] = str(cfg.seed)
        os.environ["OMP_NUM_THREADS"] = str(cfg.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(cfg.num_threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_rng_state() -> Dict:
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _to_cpu_byte_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        tensor = torch.as_tensor(value)
    if tensor.dtype != torch.uint8:
        tensor = tensor.to(dtype=torch.uint8)
    if tensor.device.type != "cpu":
        tensor = tensor.to(device="cpu")
    return tensor.contiguous()


def _to_cuda_rng_state_list(value) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [_to_cpu_byte_tensor(value)]
    if isinstance(value, (list, tuple)):
        return [_to_cpu_byte_tensor(v) for v in value]
    raise TypeError(f"Unsupported torch_cuda RNG state type: {type(value)!r}")


def set_rng_state(state: Dict) -> None:
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(_to_cpu_byte_tensor(state["torch_cpu"]))
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(_to_cuda_rng_state_list(state["torch_cuda"]))
