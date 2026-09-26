#!/usr/bin/env bash
# BUV detection, H1: C0 (no expert) vs C3 (SonoBase) vs the random-SonoBase control, frozen
# Qwen3-VL-8B, sequential on one GPU (~1.6 s/step at batch 8 -> ~9 h per 20k-step run).
# The cross-attention projector is shrunk to hidden_dim=512 (~5M params, vs 101M at the
# default hidden_dim = 4096) so the conditioned runs are not a pure capacity gain over C0.
set -euo pipefail
cd "$(dirname "$0")/.."
export BIOMLLM_ROOT=/raid/DATASETS/BioMLLMData
source scripts/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
PY=/tmp/vcivale_envs/biomllm/bin/python
COMMON=(mllm=qwen_vl_8b task=det train=det_buv +eval_every=2000 +wandb_project=biomllm-det-buv)
XATTN=(projector=cross_attn projector.hidden_dim=512)

run() {
    echo "=== $* ==="
    "$PY" scripts/train_det_c0_vs_c3.py "${COMMON[@]}" "$@"
}

run condition=c0_none                         +run_name=det_c0
run condition=c3_sonobase          "${XATTN[@]}" +run_name=det_c3_sonobase
run condition=ctrl_sonobase_random "${XATTN[@]}" +run_name=det_ctrl_sonobase_random

echo "=== all done ==="
