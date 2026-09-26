#!/usr/bin/env bash
# Stage 1 queue: expert bias on the native visual tokens (injection=native / native_prepend),
# same budget as scripts/run_power_check.sh so the runs line up with its C0 / C3 (task-token)
# rows. Sequential on GPU 1. It starts with the two VQA runs of the power check (moved here
# when seg_c3_power was stopped at 7160 steps, already in plateau, to free the GPU).
# VQA first: it is where the LLM reading the expert signal should show up.
# After each run: H2 drift on MMStar (scripts/eval_general_drift.py) and, for `native`, linear
# probes on the LLM hidden states at the image positions (scripts/probe_llm_hidden.py).
set -euo pipefail
cd "$(dirname "$0")/.."
export BIOMLLM_ROOT=/raid/DATASETS/BioMLLMData
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1
# The workstation is shared and CPU-bound (load ~180 / 192 cores): one torch thread per core
# only adds contention.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/tmp/vcivale_envs/biomllm/bin/python
PROJECT=biomllm-native-bias
OUT=$BIOMLLM_RUNS/native_bias

run() {
    echo "=== $(date '+%F %T') $* ==="
    "$PY" "$@"
}
# job <seg|vqa> <run_name> <overrides...>: train, then evaluate the saved trainable.pt.
job() {
    local task=$1 name=$2; shift 2
    if [ "$task" = seg ]; then
        run scripts/train_seg_c0_vs_c3.py mllm=qwen_vl train=seg_smoke train.lr=1e-4 train.max_steps=20000 \
            +eval_every=1000 +wandb_project=$PROJECT +run_name="$name" hydra.run.dir="$OUT/$name" "$@"
    else
        run scripts/train_vqa_c0_vs_c3.py mllm=qwen_vl task=vqa train=seg_smoke train.lr=1e-4 train.max_steps=12000 \
            +eval_every=600 +wandb_project=$PROJECT +run_name="$name" hydra.run.dir="$OUT/$name" "$@"
    fi
    local ckpt="$OUT/$name/trainable.pt"
    run scripts/eval_general_drift.py mllm=qwen_vl train=seg_smoke "$@" +checkpoint="$ckpt" \
        +run_name="drift_$name" hydra.run.dir="$OUT/$name/drift"
    if [[ " $* " == *" injection=native "* ]]; then
        run scripts/probe_llm_hidden.py mllm=qwen_vl train=seg_smoke "$@" +checkpoint="$ckpt" \
            +run_name="probe_$name" hydra.run.dir="$OUT/$name/probe_llm"
    fi
}

# Power-check VQA baselines (same arguments as scripts/run_power_check.sh).
for c in c0_none c3_rad_dino; do
    run scripts/train_vqa_c0_vs_c3.py mllm=qwen_vl task=vqa condition=$c train=seg_smoke \
        train.lr=1e-4 train.max_steps=12000 +eval_every=600 +run_name=vqa_${c%%_*}_power \
        +wandb_project=biomllm-power-check hydra.run.dir="$BIOMLLM_RUNS/power_check/vqa_${c%%_*}_power"
done

# Reference for the LLM probes: the native V (C0; C3 on task tokens has the same image context).
run scripts/probe_llm_hidden.py mllm=qwen_vl train=seg_smoke condition=c0_none \
    +run_name=probe_c0 hydra.run.dir="$OUT/probe_c0"

job vqa vqa_c3_native         condition=c3_rad_dino injection=native
job seg seg_c3_native         condition=c3_rad_dino injection=native
job vqa vqa_c3_native_prepend condition=c3_rad_dino injection=native_prepend
job seg seg_c3_native_prepend condition=c3_rad_dino injection=native_prepend
# Controls for the additive variant: random-init specialist (architecture, no pretraining).
job vqa vqa_ctrl_random_native condition=ctrl_rad_dino_random injection=native
job seg seg_ctrl_random_native condition=ctrl_rad_dino_random injection=native

echo "=== $(date '+%F %T') all done ==="
