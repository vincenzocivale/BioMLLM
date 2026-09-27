#!/usr/bin/env bash
# BUV detection, H1: does the ultrasound FM help, or does the frozen MLLM just need more
# trainable capacity? Frozen Qwen3-VL-8B, 20k steps each, sequential on one GPU.
#
#   c0_none               [DET] queries + heads only                            (baseline)
#   c3_sonobase           + alpha * P(SonoBase features)                         (method)
#   ctrl_noise            + alpha * P(Gaussian noise, resampled every forward)   random tokens
#   ctrl_static           + alpha * P(learned image-independent tokens)          random-init, learned
#   ctrl_sonobase_random  + alpha * P(SonoBase architecture, random weights)     no pretraining
#   c2_sam2               + alpha * P(generic SAM2.1 features)                   no ultrasound data
#
# Every conditioned run uses the same cross-attention projector (hidden_dim=512, ~5M params;
# the default hidden_dim = 4096 would be 101M) and the same 256-d input tokens, so they
# differ only in what the tokens carry. c3 > ctrl_noise / ctrl_static => the gain is image
# information from the FM, not capacity; c3 > c2_sam2 / ctrl_sonobase_random => it comes
# from the ultrasound pretraining. Conditioned runs also report test-time `shuffled`
# (another frame's features) and `alpha0` (correction off) in results.json.
#
# Timing (A100 40GB, batch 8): ~1.6 s/step with an expert (~9 h), ~0.8 s/step without (~5 h).
set -euo pipefail
cd "$(dirname "$0")/.."
export BIOMLLM_ROOT=/raid/DATASETS/BioMLLMData
source scripts/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
PY=/tmp/vcivale_envs/biomllm/bin/python
COMMON=(mllm=qwen_vl_8b task=det train=det_buv +eval_every=2000 +wandb_project=biomllm-det-buv)
XATTN=(projector=cross_attn projector.hidden_dim=512)

run() {
    echo "=== $* ==="
    "$PY" scripts/train_det_c0_vs_c3.py "${COMMON[@]}" "$@"
}

# Select a subset with RUNS="c0 noise" bash scripts/run_det_buv.sh (default: all, in order).
RUNS="${RUNS:-c0 c3 noise static sonobase_random sam2}"
for r in $RUNS; do
    case "$r" in
        c0)              run condition=c0_none +run_name=det_c0 ;;
        c3)              run condition=c3_sonobase "${XATTN[@]}" +run_name=det_c3_sonobase ;;
        noise)           run condition=ctrl_noise "${XATTN[@]}" conditioner.noise_dim=256 \
                             +run_name=det_ctrl_noise ;;
        static)          run condition=ctrl_static "${XATTN[@]}" conditioner.static_dim=256 \
                             "conditioner.static_grid=[32,32]" +run_name=det_ctrl_static ;;
        sonobase_random) run condition=ctrl_sonobase_random "${XATTN[@]}" \
                             +run_name=det_ctrl_sonobase_random ;;
        sam2)            run condition=c2_sam2 "${XATTN[@]}" +run_name=det_c2_sam2 ;;
        *) echo "unknown run '$r'" >&2; exit 1 ;;
    esac
done

echo "=== all done ==="
