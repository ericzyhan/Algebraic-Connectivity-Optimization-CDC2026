# Paper Companion Bundle (Baseline + RL)

This bundle contains the code and packaged outputs used in the paper.

## Layout

- `baseline/`, `configs/`, `experiments/`, `rl_train/`, `src/`: runnable code
- `results/`: packaged reference outputs (CSV/JSON) used in paper tables/figures

## Environment

- Python 3.11
- Baseline solvers: GUROBI and MOSEK available in CVXPY (with valid licenses)
- RL checkpoint in snapshot:
  - `experiments/rl/runs/ppo_path_init/checkpoints/best.pt`

## Minimal Local Re-run

Assume current directory is this bundle root.

### 1) Baseline comparison

```bash
cd baseline
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install gurobipy mosek
mkdir -p ../results/baseline
python experiments/baseline_comparison/run_manifest_hpc.py \
  --manifest experiments/baseline_comparison/manifests/baseline_comparison.json \
  --out_csv ../results/baseline/raw.csv \
  --out_meta ../results/baseline/raw.meta.json
python experiments/baseline_comparison/summarize_big_experiment.py \
  --in_csv ../results/baseline/raw.csv \
  --out_cost_csv ../results/baseline/summary_cost.csv \
  --out_quality_csv ../results/baseline/summary_quality.csv
```

### 2) RL path-vs-backbone

```bash
cd ..
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-rl.txt
mkdir -p results/rl
python -m rl_train.run_path_vs_backbone_experiment \
  --config configs/rl_cpu_standard.yaml \
  --checkpoint experiments/rl/runs/ppo_path_init/checkpoints/best.pt \
  --n-values 8,16,32,64,128 \
  --points-per-n 100 \
  --repeats 5 \
  --seed-start 0 \
  --device cpu \
  --progress-every 20 \
  --out-dir results/rl \
  --out-csv raw_rl_path_vs_backbone.csv \
  --out-meta raw_rl_path_vs_backbone.meta.json \
  --out-cost-csv summary_cost.csv \
  --out-quality-csv summary_quality.csv \
  --cayley-index2-overlap-high 0.6 \
  --cayley-index3-overlap-high 0.6666666666666666 \
  --cayley-index4-overlap-low 0.6666666666666666 \
  --cayley-multi-index-overlap \
  --cayley-enable-index4
```

## Reference Output Sizes

- Baseline: `raw=11700`, `summary_cost=29`, `summary_quality=29` rows
- RL: `raw=4200`, `summary_cost=10`, `summary_quality=10` rows


