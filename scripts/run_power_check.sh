#!/usr/bin/env bash
# Option-1 validation queue: C0 vs C3 on seg and VQA, single seed, high step budget,
# sequential on one GPU (GPU 1, free at launch time). lr=1e-4: the seg_smoke default
# (3e-4) plateaus/oscillates instead of trending down -- see scripts/run_lr_sweep.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
export BIOMLLM_ROOT=/raid/DATASETS/BioMLLMData
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=1
PY=/tmp/vcivale_envs/biomllm/bin/python

run() {
    echo "=== $* ==="
    "$PY" "$@"
}

# Segmentation: 3025 train images, batch=4 -> ~756 steps/epoch. 20000 steps ~ 26 epochs.
run scripts/train_seg_c0_vs_c3.py mllm=qwen_vl condition=c0_none     train=seg_smoke \
    train.lr=1e-4 train.max_steps=20000 +eval_every=1000 +run_name=seg_c0_power +wandb_project=biomllm-power-check

run scripts/train_seg_c0_vs_c3.py mllm=qwen_vl condition=c3_rad_dino train=seg_smoke \
    train.lr=1e-4 train.max_steps=20000 +eval_every=1000 +run_name=seg_c3_power +wandb_project=biomllm-power-check

# VQA-RAD yes/no: 940 train pairs, batch=1 -> ~940 steps/epoch. 12000 steps ~ 12.8 epochs.
run scripts/train_vqa_c0_vs_c3.py mllm=qwen_vl task=vqa condition=c0_none     train=seg_smoke \
    train.lr=1e-4 train.max_steps=12000 +eval_every=600 +run_name=vqa_c0_power +wandb_project=biomllm-power-check

run scripts/train_vqa_c0_vs_c3.py mllm=qwen_vl task=vqa condition=c3_rad_dino train=seg_smoke \
    train.lr=1e-4 train.max_steps=12000 +eval_every=600 +run_name=vqa_c3_power +wandb_project=biomllm-power-check

echo "=== all done ==="
