#!/usr/bin/env bash
# Resume of run_post.sh after the embeddings dtype fix (evaluation already done).
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
PY=${PY:-/tmp/vcivale_envs/biomllm/bin/python}
SEED=${1:-17}
step() { local name=$1; shift; echo "=== $(date '+%F %T') $name: $*" | tee -a outputs/logs/queue.log; "$@" > "outputs/logs/${name}.log" 2>&1; }
for run in diagnosis_base_usfmae diagnosis_clinical_steering diagnosis_clinical_steering_all5 diagnosis_oracle_attributes; do
  step "diag_${run}_s${SEED}" $PY scripts/train_diagnosis.py --run $run --seed $SEED
done
step "diag_eval_s${SEED}" $PY scripts/evaluate_diagnosis.py --seed $SEED
echo "=== $(date '+%F %T') main milestone done (seed $SEED)" | tee -a outputs/logs/queue.log
for run in baseline_attributes_dinov2 clinical_steering_dinov2; do
  step "train_${run}_s${SEED}" $PY scripts/train_steering.py --run $run --seed $SEED
  step "eval_${run}_s${SEED}" $PY scripts/evaluate_steering.py --run $run --seed $SEED
done
echo "=== $(date '+%F %T') dinov2 done" | tee -a outputs/logs/queue.log
