#!/bin/bash
#SBATCH --job-name=rl_train
#SBATCH --partition=long-40core-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/home/zezhu/Algebraic_Connectivity/experiments/rl/logs/train/%x-%j.out
#SBATCH --error=/gpfs/home/zezhu/Algebraic_Connectivity/experiments/rl/logs/train/%x-%j.err

set -euo pipefail

PROJECT_DIR="/gpfs/home/zezhu/Algebraic_Connectivity"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/configs/rl_cpu_standard.yaml}"
VENV_PATH="${VENV_PATH:-$PROJECT_DIR/.venv}"
DEVICE="${DEVICE:-cpu}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

mkdir -p "$PROJECT_DIR/experiments/rl/logs/train"
cd "$PROJECT_DIR"

if ! type module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
fi

module purge
module load python/3.11.2 openssl/1.1.1

python --version
which python

echo "[sbatch] job_id=${SLURM_JOB_ID:-unknown} host=$(hostname)"
echo "[sbatch] config=$CONFIG_PATH"
echo "[sbatch] venv=$VENV_PATH"
echo "[sbatch] device=$DEVICE"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "[sbatch] creating virtualenv at $VENV_PATH"
  python -m venv "$VENV_PATH"
fi

# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  NEED_INSTALL=0
  python - <<'PY' || NEED_INSTALL=1
import importlib.util as u
mods = ["torch", "torch_geometric", "yaml", "numpy", "tensorboard"]
missing = [m for m in mods if u.find_spec(m) is None]
if missing:
    raise SystemExit(1)
PY
  if [[ "$NEED_INSTALL" == "1" ]]; then
    echo "[sbatch] installing dependencies from requirements-rl.txt"
    python -m pip install --upgrade pip
    python -m pip install -r "$PROJECT_DIR/requirements-rl.txt"
  fi
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CMD=(python -m rl_train.train --config "$CONFIG_PATH" --device "$DEVICE")

if [[ -n "${MAX_ENV_STEPS:-}" ]]; then
  CMD+=(--max-env-steps "$MAX_ENV_STEPS")
fi

if [[ -n "${CHECKPOINT_EVERY_ENV_STEPS:-}" ]]; then
  CMD+=(--checkpoint-every-env-steps "$CHECKPOINT_EVERY_ENV_STEPS")
fi

if [[ -n "${RESUME_FROM:-}" ]]; then
  if [[ -f "$RESUME_FROM" ]]; then
    CMD+=(--resume-from "$RESUME_FROM")
  else
    echo "[sbatch] RESUME_FROM is set but file not found: $RESUME_FROM" >&2
    exit 1
  fi
fi

echo "[sbatch] launching: ${CMD[*]}"
"${CMD[@]}"

echo "[sbatch] training finished"
