#!/usr/bin/env bash
# After the first steering model is trained and evaluated: Qwen3-Omni grounding (MODE 0, forced
# protocol) on the us_bench test splits -> detection baselines + predicted boxes for the end-to-end
# segmentation; then the rest of queue 2 (remaining steering conditions, thinking-mode runs).
set -u
cd "$(dirname "$0")/../.."
export BIOMLLM_ROOT=${BIOMLLM_ROOT:-/raid/DATASETS/BioMLLMData}
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTHONPATH=src
PY=${PY:-/tmp/vcivale_envs/biomllm/bin/python}
R=results/omni
LOG_DIR=$R/logs
S=$BIOMLLM_ROOT/datasets/us_bench
GD=configs/omni/guidelines
G=$GD/breast_us_birads.txt
N=$GD/neutral_control.txt
W=$GD/thyroid_tirads_wrong.txt
RC=0
step() {
    local name=$1; shift
    echo "$(date '+%F %T') START $name" | tee -a "$LOG_DIR/queue.log"
    "$@" > "$LOG_DIR/$name.log" 2>&1
    RC=$?
    echo "$(date '+%F %T') END   $name rc=$RC" | tee -a "$LOG_DIR/queue.log"
}
F="$PY scripts/omni/run_grounding.py --forced-json"
TRAIN="$PY scripts/omni/train_steering.py"

# MODE 2 guideline steering (the queue-2 run stopped at step ~60 on a target-tokenization check,
# fixed by clipping GT coordinates to 0..1000): retrain from scratch, then evaluate with its three texts
rm -rf $R/steer_guideline $R/buv_steer_guideline_forced $R/buv_steer_guideline_wrongtext_forced \
    $R/buv_steer_guideline_neutraltext_forced
step train_steer_guideline $TRAIN --steer-text $G --out $R/steer_guideline
step forced_steer_guideline $F --buv --steering $R/steer_guideline/steering.pt --out $R/buv_steer_guideline_forced
step forced_steer_guideline_wrongtext $F --buv --steering $R/steer_guideline/steering.pt --steer-text $W \
    --out $R/buv_steer_guideline_wrongtext_forced
step forced_steer_guideline_neutraltext $F --buv --steering $R/steer_guideline/steering.pt --steer-text $N \
    --out $R/buv_steer_guideline_neutraltext_forced

# dataset:class_id:target
for u in breast_lesions:1:"breast lesion" bus_uclm:1:"breast lesion" busi:1:"breast lesion" \
         busbra:1:"breast lesion" hc18:1:"fetal head" psfhs:1:"pubic symphysis" psfhs:2:"fetal head" \
         mmotu:1:"ovarian tumor" tn3k:1:"thyroid nodule" camus:1:"left ventricle" camus:2:"myocardium" \
         camus:3:"left atrium"; do
    IFS=: read -r d k t <<< "$u"
    tag="${d}_c${k}"
    step suite_ground_$tag $F --data-root $S/$d --split test --class-id $k --target "$t" \
        --out $R/suite_ground/$tag
    # end-to-end segmentation with the Omni box (CPU, same decoder as the GT-box oracle), in the background
    if [ -f $R/seg/$d/decoder_${k}_gt.pt ]; then
        (CUDA_VISIBLE_DEVICES="" $PY scripts/omni/train_mask_decoder.py --dataset $d --threads 16 \
            --eval-pred $R/suite_ground/$tag/records.jsonl --class-id $k --out $R/seg/$d \
            > $LOG_DIR/seg_pred_$tag.log 2>&1 &)
    fi
done
step summary_suite_ground $PY scripts/omni/summarize_grounding.py $R/suite_ground/*

step train_steer_neutral $TRAIN --steer-text $N --out $R/steer_neutral
step forced_steer_neutral $F --buv --steering $R/steer_neutral/steering.pt --out $R/buv_steer_neutral_forced
step train_steer_prompt_guideline $TRAIN --steer-text $G --prompt-guideline $G --out $R/steer_prompt_guideline
step forced_steer_prompt_guideline $F --buv --guideline $G --steering $R/steer_prompt_guideline/steering.pt \
    --out $R/buv_steer_prompt_guideline_forced
step summary_forced $PY scripts/omni/summarize_grounding.py $R/buv_vanilla_forced $R/buv_guideline_forced \
    $R/buv_steer_guideline_forced $R/buv_steer_prompt_guideline_forced $R/buv_steer_neutral_forced \
    $R/buv_steer_guideline_wrongtext_forced $R/buv_steer_guideline_neutraltext_forced

step buv_vanilla_thinking $PY scripts/omni/run_grounding.py --buv --out $R/buv_vanilla_thinking
step buv_guideline_thinking $PY scripts/omni/run_grounding.py --buv --guideline $G --out $R/buv_guideline_thinking
step precision_thinking $PY scripts/omni/validate_precision.py --records $R/buv_vanilla_thinking/records.jsonl \
    --n 8 --out $R/precision_check_thinking
[ -f $R/steer_guideline/steering.pt ] && step buv_steer_guideline_thinking $PY scripts/omni/run_grounding.py \
    --buv --steering $R/steer_guideline/steering.pt --out $R/buv_steer_guideline_thinking
step buv_guideline_nothink $PY scripts/omni/run_grounding.py --buv --no-thinking --max-new-tokens 1024 \
    --guideline $G --out $R/buv_guideline_nothink
step summary_all $PY scripts/omni/summarize_grounding.py $R/buv_vanilla_forced $R/buv_guideline_forced \
    $R/buv_steer_guideline_forced $R/buv_steer_prompt_guideline_forced $R/buv_steer_neutral_forced \
    $R/buv_steer_guideline_wrongtext_forced $R/buv_steer_guideline_neutraltext_forced \
    $R/buv_vanilla_nothink $R/buv_guideline_nothink $R/buv_vanilla_thinking $R/buv_guideline_thinking \
    $R/buv_steer_guideline_thinking
echo "$(date '+%F %T') QUEUE3 DONE" | tee -a "$LOG_DIR/queue.log"
