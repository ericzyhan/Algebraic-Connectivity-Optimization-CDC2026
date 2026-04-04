#!/bin/bash
#SBATCH --job-name=baseline_type123
#SBATCH --partition=long-40core-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/baseline_comparison/%x-%j.out
#SBATCH --error=logs/baseline_comparison/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="${BASELINE_DIR:-$SCRIPT_DIR}"
VENV_PATH="${VENV_PATH:-$BASELINE_DIR/.venv}"
MANIFEST="${MANIFEST:-experiments/baseline_comparison/manifests/baseline_comparison.json}"
OUT_CSV="${OUT_CSV:-}"
OUT_META="${OUT_META:-}"
RAW_CSV="${RAW_CSV:-logs/baseline_comparison/eval_big_type123_raw.csv}"
SUMMARY_COST="${SUMMARY_COST:-logs/baseline_comparison/summary_cost.csv}"
SUMMARY_QUALITY="${SUMMARY_QUALITY:-logs/baseline_comparison/summary_quality.csv}"

if [[ -n "$OUT_CSV" ]] && [[ "$RAW_CSV" == "logs/baseline_comparison/eval_big_type123_raw.csv" ]]; then
  RAW_CSV="$OUT_CSV"
fi

mkdir -p "$BASELINE_DIR/logs/baseline_comparison"
cd "$BASELINE_DIR"

if ! type module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
fi

module purge
module load python/3.11.2 openssl/1.1.1

python --version
which python
echo "[sbatch] job_id=${SLURM_JOB_ID:-unknown} host=$(hostname)"
echo "[sbatch] baseline_dir=$BASELINE_DIR"
echo "[sbatch] venv=$VENV_PATH"
echo "[sbatch] manifest=$MANIFEST"
echo "[sbatch] out_csv=${OUT_CSV:-<manifest-default>}"
echo "[sbatch] out_meta=${OUT_META:-<manifest-default>}"

if [[ -x "$VENV_PATH/bin/python" ]]; then
  VENV_PY_MM="$("$VENV_PATH/bin/python" - <<'PY'
import sys
print(f"{sys.version_info[0]}.{sys.version_info[1]}")
PY
)"
  if [[ "$VENV_PY_MM" != "3.11" ]]; then
    echo "[sbatch] existing venv uses Python $VENV_PY_MM; recreating for Python 3.11"
    rm -rf "$VENV_PATH"
    python -m venv "$VENV_PATH"
  fi
else
  echo "[sbatch] creating virtualenv at $VENV_PATH"
  python -m venv "$VENV_PATH"
fi

# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install gurobipy mosek

python - <<'PY'
import cvxpy as cp

solvers = cp.installed_solvers()
print("cvxpy.installed_solvers() =", solvers, flush=True)
missing = [name for name in ("GUROBI", "MOSEK") if name not in solvers]
if missing:
    raise SystemExit(f"Missing required cvxpy solvers: {missing}")

# License/usability smoke test to fail fast before the long run starts.
x = cp.Variable(nonneg=True)
prob = cp.Problem(cp.Minimize(x), [x >= 1.0])
for name in ("GUROBI", "MOSEK"):
    try:
        prob.solve(solver=getattr(cp, name), verbose=False)
    except Exception as exc:
        raise SystemExit(f"{name} solver smoke test failed (likely licensing/config): {exc}") from exc
    if str(prob.status).lower() not in {"optimal", "optimal_inaccurate"}:
        raise SystemExit(f"{name} solver smoke test returned unexpected status: {prob.status}")
    print(f"{name} smoke test status={prob.status} value={prob.value}", flush=True)
PY

export PYTHONUNBUFFERED=1

RUN_DRY_CMD=(python experiments/baseline_comparison/run_manifest_hpc.py --manifest "$MANIFEST" --dry_run)
RUN_FULL_CMD=(python experiments/baseline_comparison/run_manifest_hpc.py --manifest "$MANIFEST")
if [[ -n "$OUT_CSV" ]]; then
  RUN_DRY_CMD+=(--out_csv "$OUT_CSV")
  RUN_FULL_CMD+=(--out_csv "$OUT_CSV")
fi
if [[ -n "$OUT_META" ]]; then
  RUN_DRY_CMD+=(--out_meta "$OUT_META")
  RUN_FULL_CMD+=(--out_meta "$OUT_META")
fi

"${RUN_DRY_CMD[@]}"
"${RUN_FULL_CMD[@]}"

python experiments/baseline_comparison/summarize_big_experiment.py \
  --in_csv "$RAW_CSV" \
  --out_cost_csv "$SUMMARY_COST" \
  --out_quality_csv "$SUMMARY_QUALITY"

export BASELINE_DIR RAW_CSV SUMMARY_COST SUMMARY_QUALITY
python - <<'PY'
import csv
import os
from pathlib import Path

base = Path(os.environ["BASELINE_DIR"])
targets = [
    base / os.environ["RAW_CSV"],
    base / os.environ["SUMMARY_COST"],
    base / os.environ["SUMMARY_QUALITY"],
]
for path in targets:
    if not path.exists():
        print(f"[csv] missing: {path}", flush=True)
        continue
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = sum(1 for _ in csv.DictReader(f))
    print(f"[csv] {path} rows={rows}", flush=True)
PY

echo "[sbatch] baseline comparison complete"

