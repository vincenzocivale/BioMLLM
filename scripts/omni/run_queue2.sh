#!/usr/bin/env bash
# Controlled detection matrix (format-controlled "forced" protocol) + steering, then the official
# thinking-mode runs. One GPU, sequential, all Qwen3-Omni weights frozen. Records resume by id.
set -u
cd "$(dirname "$0")/../.."
export BIOMLLM_ROOT=${BIOMLLM_ROOT:-/raid/DATASETS/BioMLLMData}
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTHONPATH=src
PY=${PY:-/tmp/vcivale_envs/biomllm/bin/python}
R=results/omni
LOG_DIR=$R/logs
mkdir -p "$LOG_DIR"
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
F="$PY scripts/omni/run_grounding.py --buv --forced-json"
TRAIN="$PY scripts/omni/train_steering.py"

# MODE 0 / MODE 1 under the controlled protocol (zero training)
step forced_vanilla $F --out $R/buv_vanilla_forced
step forced_guideline $F --guideline $G --out $R/buv_guideline_forced

# NF4 vs BF16 reference on the free no-thinking vanilla traces (moved from queue 1)
step precision_nothink $PY scripts/omni/validate_precision.py --records $R/buv_vanilla_nothink/records.jsonl \
    --n 20 --out $R/precision_check_nothink

# MODE 2 and controls
step steer_smoke $TRAIN --steer-text $G --steps 20 --eval-every 10 --out $R/steer_smoke
if [ $RC -eq 0 ]; then
    step train_steer_guideline $TRAIN --steer-text $G --out $R/steer_guideline
    step forced_steer_guideline $F --steering $R/steer_guideline/steering.pt --out $R/buv_steer_guideline_forced
    step forced_steer_guideline_wrongtext $F --steering $R/steer_guideline/steering.pt --steer-text $W \
        --out $R/buv_steer_guideline_wrongtext_forced
    step forced_steer_guideline_neutraltext $F --steering $R/steer_guideline/steering.pt --steer-text $N \
        --out $R/buv_steer_guideline_neutraltext_forced
    step train_steer_neutral $TRAIN --steer-text $N --out $R/steer_neutral
    step forced_steer_neutral $F --steering $R/steer_neutral/steering.pt --out $R/buv_steer_neutral_forced
    step train_steer_prompt_guideline $TRAIN --steer-text $G --prompt-guideline $G --out $R/steer_prompt_guideline
    step forced_steer_prompt_guideline $F --guideline $G --steering $R/steer_prompt_guideline/steering.pt \
        --out $R/buv_steer_prompt_guideline_forced
fi
step summary_forced $PY scripts/omni/summarize_grounding.py $R/buv_vanilla_forced $R/buv_guideline_forced \
    $R/buv_steer_guideline_forced $R/buv_steer_prompt_guideline_forced $R/buv_steer_neutral_forced \
    $R/buv_steer_guideline_wrongtext_forced $R/buv_steer_guideline_neutraltext_forced

# official default decoding (thinking): the "true baseline" of the spec, then steering transfer
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
echo "$(date '+%F %T') QUEUE2 DONE" | tee -a "$LOG_DIR/queue.log"
