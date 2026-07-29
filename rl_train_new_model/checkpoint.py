from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch


def save_checkpoint(path: str, payload: Dict) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path_obj)


def load_checkpoint(path: str, map_location: Optional[str] = "cpu") -> Dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def prune_checkpoints(checkpoint_dir: str, keep_last_k: int) -> None:
    path = Path(checkpoint_dir)
    if not path.exists():
        return
    files = sorted(path.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    if len(files) <= keep_last_k:
        return
    for file_path in files[: len(files) - keep_last_k]:
        file_path.unlink(missing_ok=True)


def write_run_state(path: str, state: Dict) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
