# Algebraic Connectivity Optimization (CDC2026)

Curated source-code snapshot for the CDC2026 submission on density-dependent backbone construction and RL completion for algebraic-connectivity maximization under fixed `(n,m)` constraints.

## Included in this snapshot

- `src/graph_design/`: backbone construction, baselines, completion modules, solver utilities.
- `RL_Train/rl_train/`: RL training/evaluation framework (PPO/REINFORCE, backbone/path init modes, rollout/evaluate tools).
- `RL_Train/configs/`: runnable config files for path/backbone and lite variants.
- `tests/`: baseline and integration-style tests used during development.
- `pyproject.toml`: base Python project/dependency config.

## Not included (for now)

To keep this repository lightweight, large experiment artifacts are excluded (full run folders, checkpoints, generated plots, and paper-source Overleaf files).

## Quick start

### 1) Base solver environment

```bash
pip install -e .
pytest -q
```

### 2) RL training/evaluation environment

```bash
cd RL_Train
pip install -r requirements-rl.txt

# Example training
python -m rl_train.train --config ./configs/rl_cpu_backbone_v1.yaml

# Example evaluation
python -m rl_train.evaluate --config ./configs/rl_cpu_backbone_v1.yaml --checkpoint best=./runs/checkpoints/best.pt
```

## Reproducibility note

This is an initial publication snapshot. Additional scripts/artifacts (full experiment manifests, fixed commit tags, and extended outputs) can be added after submission.
