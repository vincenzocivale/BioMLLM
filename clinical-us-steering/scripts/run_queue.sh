#!/usr/bin/env bash
# Milestone queue on a single GPU (GPU 1 on DUNE). Usage: bash scripts/run_queue.sh [SEED]
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
PY=${PY:-/tmp/vcivale_envs/biomllm/bin/python}
SEED=${1:-17}
mkdir -p outputs/logs
step() {  # step <log name> <cmd...>
  local name=$1; shift
  echo "=== $(date '+%F %T') $name: $*" | tee -a outputs/logs/queue.log
  "$@" > "outputs/logs/${name}.log" 2>&1
}
for run in baseline_attributes_usfmae clinical_steering_usfmae; do
  step "train_${run}_s${SEED}" $PY scripts/train_steering.py --run $run --seed $SEED
done
echo "=== $(date '+%F %T') training done (seed $SEED)" | tee -a outputs/logs/queue.log
