from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore


class ProgressTracker:
    def __init__(
        self,
        max_env_steps: int,
        metrics_csv_path: str,
        tensorboard_dir: str,
        resumed_elapsed_sec: float = 0.0,
    ):
        self.max_env_steps = max_env_steps
        self.metrics_csv_path = Path(metrics_csv_path)
        self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.tb_writer = None
        if SummaryWriter is not None:
            self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)

        self._csv_file = self.metrics_csv_path.open("a", encoding="utf-8", newline="")
        self._csv_writer: Optional[csv.DictWriter] = None

        self.start_time = time.time()
        self.resumed_elapsed_sec = resumed_elapsed_sec

    def elapsed_sec(self) -> float:
        return self.resumed_elapsed_sec + (time.time() - self.start_time)

    def _init_csv(self, row: Dict) -> None:
        if self._csv_writer is None:
            fields = list(row.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fields)
            if self._csv_file.tell() == 0:
                self._csv_writer.writeheader()

    def log(self, row: Dict) -> None:
        self._init_csv(row)
        assert self._csv_writer is not None
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        step = int(row.get("global_env_steps", 0))
        if self.tb_writer is not None:
            for key, value in row.items():
                if key in {"phase", "elapsed_sec", "eta_sec"}:
                    continue
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)

    def print_console(self, row: Dict) -> None:
        step = int(row.get("global_env_steps", 0))
        pct = 100.0 * float(step) / float(max(1, self.max_env_steps))
        phase = str(row.get("phase", "unknown"))
        sps = float(row.get("steps_per_sec", 0.0))
        elapsed = float(row.get("elapsed_sec", 0.0))
        eta = float(row.get("eta_sec", 0.0))
        ep_ret = float(row.get("mean_episode_return", 0.0))
        ep_l2 = float(row.get("mean_terminal_lambda2_norm", 0.0))
        policy_loss = float(row.get("policy_loss", 0.0))
        value_loss = float(row.get("value_loss", 0.0))
        entropy = float(row.get("entropy", 0.0))
        lr = float(row.get("learning_rate", 0.0))
        selector_loss = float(row.get("selector_loss", 0.0))
        selector_recall = float(row.get("selector_recall_at_k", 0.0))
        selector_k = float(row.get("selector_k_mean", 0.0))

        print(
            f"[train] step={step}/{self.max_env_steps} ({pct:.2f}%) "
            f"phase={phase} sps={sps:.2f} elapsed={elapsed:.1f}s eta={eta:.1f}s "
            f"mean_ret={ep_ret:.6f} mean_l2={ep_l2:.6f} "
            f"policy_loss={policy_loss:.6f} value_loss={value_loss:.6f} "
            f"entropy={entropy:.6f} lr={lr:.6g} "
            f"sel_loss={selector_loss:.6f} sel_rec@k={selector_recall:.4f} sel_k={selector_k:.1f}",
            flush=True,
        )

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
        self._csv_file.close()
