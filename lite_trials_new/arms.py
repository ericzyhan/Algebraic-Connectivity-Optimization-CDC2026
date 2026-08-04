"""Arm definitions for the new-model vs old-model lite comparison.

An "arm" is one (code base, reward setting) pair that gets its own full — but
short — training loop, its own checkpoint, and its own path-vs-backbone
evaluation. Everything else (curriculum, network sizes, PPO hyperparameters,
seed, evaluation protocol) is held identical across arms so the only thing
varying is what we actually want to compare.

Three code bases are compared, one arm each:

* ``old_model/rl_train`` (a sibling checkout of this repo) — the pre-analytic
  agent. Its reward is the plain greedy one, r_t = Delta(lambda_2)/n, plus a
  terminal bonus; its ``EnvConfig`` has no blend weights at all.
* ``analytic_tractable_test_deepseek`` (this repo) — the analytic agent. Keeps a
  full n x n eigenbasis and solves the secular equation exactly, so lambda_2 is
  tracked to ~1e-15.
* ``rl_train_new_model`` (this repo) — tracks a bottom-q invariant subspace by
  block power iteration instead, with a certified two-sided bracket on
  Delta(lambda_2). Cheaper per step and flat in n; lambda_2's point estimate is
  approximate but bracketed.

Both analytic arms use the same 3-term blended reward
r_t = a1*Delta(lambda_2)/n + a2*eta_R*Delta(R_G)/n^2 + a3*eta_P*Delta(P_min).

This is deliberately not a sweep: alpha is fixed, and pinned to the *same*
values across both analytic arms so the reward blend does not confound the
comparison. Each arm is one full training loop, just short.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LITE_ROOT = Path(__file__).resolve().parent

#: Where the old model lives. Override with LITE_TRIALS_OLD_MODEL_ROOT if the
#: sibling checkout is somewhere else.
OLD_MODEL_ROOT = Path(
    os.environ.get("LITE_TRIALS_OLD_MODEL_ROOT", str(PROJECT_ROOT.parent / "old_model"))
).resolve()

ANALYTIC_PACKAGE = "analytic_tractable_test_deepseek"
SUBSPACE_PACKAGE = "rl_train_new_model"
OLD_PACKAGE = "rl_train"


@dataclass(frozen=True)
class Arm:
    tag: str
    label: str
    #: Directory used as both cwd and sys.path root when invoking ``package``.
    repo_root: Path
    package: str
    #: Extra ``env:`` keys merged into the shared base config. The old model's
    #: EnvConfig dataclass has no reward-blend fields, so its arm must leave
    #: this empty or config loading raises TypeError.
    env_overrides: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def run_dir(self) -> Path:
        return LITE_ROOT / "runs" / self.tag

    @property
    def config_path(self) -> Path:
        return LITE_ROOT / "configs" / f"{self.tag}.yaml"

    @property
    def best_checkpoint(self) -> Path:
        return self.run_dir / "checkpoints" / "best.pt"

    @property
    def last_checkpoint(self) -> Path:
        """Most recent training state, as opposed to the best-scoring one.

        This is what a resume continues from: it carries optimizer, scheduler,
        env-manager, rollout-buffer and RNG state, so training picks up exactly
        where it stopped. best.pt is a snapshot for evaluation only.
        """
        return self.run_dir / "checkpoints" / "last.pt"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.csv"

    @property
    def results_dir(self) -> Path:
        return self.run_dir / "results"


# Fixed reward blend, pinned to the SAME values on both analytic arms so the
# reward does not confound the comparison. These are rl_train_new_model's own
# defaults; the deepseek package defaults to (0.10, 0.20, 0.70) and is pinned
# here instead. Not swept.
FIXED_BLEND = {
    "reward_alpha_1": 0.15,
    "reward_alpha_2": 0.15,
    "reward_alpha_3": 0.70,
    "reward_eta": 0.0,    # 0 = auto-calibrate eta_R per n
    "reward_eta_p": 1.0,
}

# rl_train_new_model's EnvConfig carries tracker and candidacy knobs the other
# two packages do not have. They must ride in env_overrides, never in the shared
# base config -- old_model's EnvConfig dataclass will TypeError on them.
SUBSPACE_ENV = {
    "spectral_oversample": 3,
    "spectral_exact_reset_every": 50,
    "spectral_power_iters": 4,
    "spectral_drift_threshold": 1e-8,
    "spectral_soft_band_rel": 1e-2,
    "spectral_degeneracy_probe_rel": 1e-5,
    "tier1_topk_dr": 32,
    "tier1_topk_spectral": 32,
    "tier2_survivor_size": 32,
    "tier3_top": 24,
    "tier3_random": 8,
    "deflation_macro_action": True,
}


DEFAULT_ARMS: list[Arm] = [
    Arm(
        tag="old_model",
        label="Old model (rl_train)",
        repo_root=OLD_MODEL_ROOT,
        package=OLD_PACKAGE,
        env_overrides={},
        notes="Baseline. Reward is Delta(lambda_2)/n + terminal bonus; no blend terms exist.",
    ),
    Arm(
        tag="analytic",
        label="Analytic (exact secular)",
        repo_root=PROJECT_ROOT,
        package=ANALYTIC_PACKAGE,
        env_overrides=dict(FIXED_BLEND),
        notes="Full eigenbasis, exact secular root. lambda_2 tracked to ~1e-15.",
    ),
    Arm(
        tag="subspace",
        label="Subspace (certified bracket)",
        repo_root=PROJECT_ROOT,
        package=SUBSPACE_PACKAGE,
        env_overrides={**FIXED_BLEND, **SUBSPACE_ENV},
        notes="Bottom-q subspace + block power iteration; bracketed lambda_2, flat in n.",
    ),
]

ARMS_BY_TAG: dict[str, Arm] = {a.tag: a for a in DEFAULT_ARMS}


def select_arms(spec: str | None) -> list[Arm]:
    """Resolve a comma-separated list of arm tags (None/empty -> all arms)."""
    if not spec or not spec.strip():
        return list(DEFAULT_ARMS)
    tags = [t.strip() for t in spec.split(",") if t.strip()]
    unknown = [t for t in tags if t not in ARMS_BY_TAG]
    if unknown:
        raise ValueError(
            f"Unknown arm tag(s): {unknown}. Known: {sorted(ARMS_BY_TAG)}"
        )
    return [ARMS_BY_TAG[t] for t in tags]


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

#: Total env steps per arm. "Lite" only in episode count -- this is the full
#: training loop (PPO, curriculum, periodic eval, checkpointing), just short.
MAX_ENV_STEPS = 20_000

#: Density buckets for summarizing lambda_2 over a density regime rather than
#: pointwise. Mirrors lite_trials/alpha_sweep_config.py.
DENSITY_BUCKETS: list[tuple[str, float, float]] = [
    ("sparse", 0.0, 1.0 / 3.0),
    ("medium", 1.0 / 3.0, 2.0 / 3.0),
    ("dense", 2.0 / 3.0, 1.0 + 1e-9),
]


def bucket_for_rho(rho: float) -> str:
    for name, lo, hi in DENSITY_BUCKETS:
        if lo <= rho < hi:
            return name
    return DENSITY_BUCKETS[-1][0]


def base_config(arm: Arm, max_env_steps: int = MAX_ENV_STEPS) -> dict[str, Any]:
    """Config shared by every arm, with per-arm paths and env overrides applied.

    Paths are relative to ``lite_trials_new/`` because both code bases resolve
    relative config paths against ``config_file.parent.parent`` — which is this
    directory, since configs live in ``lite_trials_new/configs/``. That holds
    for the old model too, even though it runs with cwd=old_model, so every
    arm's artifacts land here next to each other.
    """
    run_rel = f"runs/{arm.tag}"
    cfg: dict[str, Any] = {
        "curriculum": {
            "phases": [
                {
                    "name": "phase_lite",
                    "start_env_step": 0,
                    "end_env_step": int(max_env_steps),
                    "n_min": 8,
                    "n_max": 12,
                }
            ]
        },
        "env": {
            "num_envs": 4,
            "top_k": 64,
            "dist_cap": 4,
            "terminal_bonus_coef": 0.2,
        },
        "model": {
            "gat_hidden_dim": 32,
            "gat_heads": 2,
            "gat_layers": 2,
            "edge_mlp_hidden_dim": 64,
            "value_mlp_hidden_dim": 64,
        },
        "trainer": {"algo": "ppo"},
        "ppo": {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_ratio": 0.2,
            "learning_rate": 0.0003,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 1.0,
            "update_epochs": 4,
            "minibatch_size": 64,
            "rollout_env_steps": 1024,
        },
        "logging": {
            "console_log_every_env_steps": 2000,
            "eval_every_env_steps": 1000,
            "metrics_path": f"{run_rel}/metrics.csv",
            "tensorboard_dir": f"{run_rel}/tensorboard",
            "run_state_path": f"{run_rel}/run_state.json",
        },
        "checkpoint": {
            "every_env_steps": 2000,
            "keep_last_k": 3,
            "dir": f"{run_rel}/checkpoints",
        },
        "determinism": {"strict": False, "seed": 1234, "num_threads": 2},
        "evaluation": {"episodes": 4},
        "train": {"max_env_steps": int(max_env_steps), "output_dir": run_rel},
        "variant": {"name": "full"},
        "init": {"mode": "path"},
    }
    cfg["env"].update(arm.env_overrides)
    return cfg
