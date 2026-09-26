#!/usr/bin/env bash
# Quick LR sweep on C0 (seg) to find a rate where train loss actually trends down,
# before committing GPU-hours to the full power-check queue.
set -euo pipefail
cd "$(dirname "$0")/.."
export BIOMLLM_ROOT=/raid/DATASETS/BioMLLMData
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=1
PY=/tmp/vcivale_envs/biomllm/bin/python

for lr in 3e-5 1e-4 3e-4; do
    echo "=== lr=$lr ==="
    "$PY" scripts/train_seg_c0_vs_c3.py mllm=qwen_vl condition=c0_none train=seg_smoke \
        train.lr="$lr" train.max_steps=300 +eval_every=300 \
        +run_name="lrsweep_c0_${lr}" +wandb_project=biomllm-lr-sweep
done
echo "=== sweep done ==="
