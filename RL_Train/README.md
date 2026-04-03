# RL Training Pipeline

## Install

```bash
cd /gpfs/home/zezhu/Algebraic_Connectivity
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-rl.txt
```

## Train

```bash
python -m rl_train.train \
  --config /gpfs/home/zezhu/Algebraic_Connectivity/configs/rl_cpu_standard.yaml
```

Resume:

```bash
python -m rl_train.train \
  --config /gpfs/home/zezhu/Algebraic_Connectivity/configs/rl_cpu_standard.yaml \
  --resume-from /gpfs/home/zezhu/Algebraic_Connectivity/runs/ppo_path/checkpoints/last.pt
```

Train REINFORCE + baseline:

```bash
python -m rl_train.train \
  --config /gpfs/home/zezhu/Algebraic_Connectivity/configs/rl_reinforce_cpu_standard.yaml
```

Resume REINFORCE + baseline:

```bash
python -m rl_train.train \
  --config /gpfs/home/zezhu/Algebraic_Connectivity/configs/rl_reinforce_cpu_standard.yaml \
  --resume-from /gpfs/home/zezhu/Algebraic_Connectivity/runs/reinforce_path/checkpoints/last.pt
```

Train backbone+RL V1 (same RL internals, backbone episode init):

```bash
python -m rl_train.train \
  --config /gpfs/home/zezhu/Algebraic_Connectivity/configs/rl_cpu_backbone_v1.yaml
```

Variant tagging:
- Configs include `variant.name` (currently `full`).
- Eval CSV / metadata and rollout JSON include this variant tag for future multi-variant comparisons.

## Rollout

```bash
python -m rl_train.rollout \
  --config /gpfs/home/zezhu/Algebraic_Connectivity/configs/rl_cpu_standard.yaml \
  --checkpoint /gpfs/home/zezhu/Algebraic_Connectivity/runs/ppo_path/checkpoints/last.pt \
  --n 32 \
  --rho 0.5 \
  --out /gpfs/home/zezhu/Algebraic_Connectivity/experiments/rl/data/eval/rollout_result.json
```

Backbone-init rollout:

```bash
python -m rl_train.rollout \
  --config /gpfs/home/zezhu/Algebraic_Connectivity/configs/rl_cpu_backbone_v1.yaml \
  --checkpoint /gpfs/home/zezhu/Algebraic_Connectivity/runs/ppo_backbone_v1/checkpoints/last.pt \
  --n 32 \
  --rho 0.5 \
  --out /gpfs/home/zezhu/Algebraic_Connectivity/experiments/rl/data/eval/rollout_backbone_result.json
```

## Grid Evaluation (CSV)

Evaluate one or multiple checkpoints on a density sweep without reloading the model each rollout:

```bash
python -m rl_train.evaluate \
  --config /gpfs/home/zezhu/Algebraic_Connectivity/configs/rl_cpu_standard.yaml \
  --checkpoint best=/gpfs/home/zezhu/Algebraic_Connectivity/runs/ppo_path/checkpoints/best.pt \
  --checkpoint last=/gpfs/home/zezhu/Algebraic_Connectivity/runs/ppo_path/checkpoints/last.pt \
  --n-values 32,64 \
  --density-count 100 \
  --density-min 0 \
  --density-max 1 \
  --seeds 1234 \
  --progress-every 20 \
  --out /gpfs/home/zezhu/Algebraic_Connectivity/experiments/rl/data/eval/eval_best_last.csv
```

Evaluate REINFORCE checkpoints (same script):

```bash
python -m rl_train.evaluate \
  --config /gpfs/home/zezhu/Algebraic_Connectivity/configs/rl_reinforce_cpu_standard.yaml \
  --checkpoint reinforce_last=/gpfs/home/zezhu/Algebraic_Connectivity/runs/reinforce_path/checkpoints/last.pt \
  --n-values 32,64 \
  --density-count 100 \
  --seeds 1234 \
  --out /gpfs/home/zezhu/Algebraic_Connectivity/experiments/rl/data/eval/eval_reinforce_last.csv
```

Evaluate RL checkpoints in backbone-initialized environment:

```bash
python -m rl_train.evaluate \
  --config /gpfs/home/zezhu/Algebraic_Connectivity/configs/rl_cpu_standard.yaml \
  --checkpoint best=/gpfs/home/zezhu/Algebraic_Connectivity/runs/ppo_path/checkpoints/best.pt \
  --checkpoint last=/gpfs/home/zezhu/Algebraic_Connectivity/runs/ppo_path/checkpoints/last.pt \
  --n-values 32,64 \
  --density-count 100 \
  --density-min 0 \
  --density-max 1 \
  --seeds 1234 \
  --init-mode backbone \
  --progress-every 20 \
  --out /gpfs/home/zezhu/Algebraic_Connectivity/experiments/rl/data/eval/eval_best_last_backbone_init.csv
```

Plot eval CSV vs FVG baseline:

```bash
python -m rl_train.plot_eval_vs_fvg \
  --eval-csv /gpfs/home/zezhu/Algebraic_Connectivity/experiments/rl/data/eval/eval_best_last_backbone_init.csv \
  --runtime-source auto
```

Default output layout after plotting/comparison:

```text
RL_Train/
  runs/
    ppo_path/                 # PPO + path-init training state
    ppo_backbone_v1/          # PPO + backbone-init training state
    reinforce_path/           # REINFORCE + path-init training state
    selfcheck/                # deterministic resume selfcheck outputs

model/
  experiments/
    rl/
      data/
        eval/                 # eval_*.csv + eval_*.meta.json
        compare/<tag>/        # combined/summary comparison CSVs
      plots/<tag>/            # figures
      logs/train/             # train stdout/stderr logs
```

## Deterministic Resume Selfcheck

```bash
python -m rl_train.selfcheck \
  --config /gpfs/home/zezhu/Algebraic_Connectivity/configs/rl_cpu_standard.yaml \
  --total-env-steps 4000 \
  --split-env-steps 2000 \
  --root-dir /gpfs/home/zezhu/Algebraic_Connectivity/runs/selfcheck/ppo
```

REINFORCE selfcheck:

```bash
python -m rl_train.selfcheck \
  --config /gpfs/home/zezhu/Algebraic_Connectivity/configs/rl_reinforce_cpu_standard.yaml \
  --total-env-steps 4000 \
  --split-env-steps 2000 \
  --root-dir /gpfs/home/zezhu/Algebraic_Connectivity/runs/selfcheck/reinforce
```

Run via sbatch with config selection:

```bash
CONFIG_PATH=/gpfs/home/zezhu/Algebraic_Connectivity/configs/rl_reinforce_cpu_standard.yaml \
sbatch /gpfs/home/zezhu/Algebraic_Connectivity/run_rl_train.sbatch
```
