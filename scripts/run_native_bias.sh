#!/usr/bin/env bash
# Stage 1 queue: expert bias on the native visual tokens (injection=native / native_prepend),
# same budget as scripts/run_power_check.sh so the runs line up with its C0 / C3 (task-token)
# rows. Sequential on GPU 1; waits for a running power-check queue to finish first.
# VQA first: it is where the LLM reading the expert signal should show up.
set -euo pipefail
cd "$(dirname "$0")/.."
export BIOMLLM_ROOT=/raid/DATASETS/BioMLLMData
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=1
PY=/tmp/vcivale_envs/biomllm/bin/python
PROJECT=biomllm-native-bias

while pgrep -f run_power_check.sh >/dev/null; do sleep 300; done

run() {
    echo "=== $(date '+%F %T') $* ==="
    "$PY" "$@"
}
seg() {
    run scripts/train_seg_c0_vs_c3.py mllm=qwen_vl train=seg_smoke train.lr=1e-4 train.max_steps=20000 \
        +eval_every=1000 +wandb_project=$PROJECT "$@"
}
vqa() {
    run scripts/train_vqa_c0_vs_c3.py mllm=qwen_vl task=vqa train=seg_smoke train.lr=1e-4 train.max_steps=12000 \
        +eval_every=600 +wandb_project=$PROJECT "$@"
}

vqa condition=c3_rad_dino injection=native         +run_name=vqa_c3_native
seg condition=c3_rad_dino injection=native         +run_name=seg_c3_native
vqa condition=c3_rad_dino injection=native_prepend +run_name=vqa_c3_native_prepend
seg condition=c3_rad_dino injection=native_prepend +run_name=seg_c3_native_prepend
# Controls for the additive variant: random-init specialist (architecture, no pretraining).
vqa condition=ctrl_rad_dino_random injection=native +run_name=vqa_ctrl_random_native
seg condition=ctrl_rad_dino_random injection=native +run_name=seg_ctrl_random_native

echo "=== $(date '+%F %T') all done ==="
