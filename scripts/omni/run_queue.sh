#!/usr/bin/env bash
# Phase-0 / MODE 0-1 experiment queue for Qwen3-Omni on ONE GPU (default GPU 1), sequential.
# Each step logs to $LOG_DIR/<step>.log; a failing step is recorded and the queue continues.
#   bash scripts/omni/run_queue.sh            (GPU=1 by default)
set -u
cd "$(dirname "$0")/../.."
export BIOMLLM_ROOT=${BIOMLLM_ROOT:-/raid/DATASETS/BioMLLMData}
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTHONPATH=src
PY=${PY:-/tmp/vcivale_envs/biomllm/bin/python}
R=results/omni
LOG_DIR=$R/logs
mkdir -p "$LOG_DIR"
G=configs/omni/guidelines/breast_us_birads.txt

step() {  # step <name> <cmd...>
    local name=$1; shift
    echo "$(date '+%F %T') START $name" | tee -a "$LOG_DIR/queue.log"
    "$@" > "$LOG_DIR/$name.log" 2>&1
    echo "$(date '+%F %T') END   $name rc=$?" | tee -a "$LOG_DIR/queue.log"
}

step checkpoint_tests env BIOMLLM_OMNI_CHECKPOINT=1 $PY -m pytest tests/test_omni_checkpoint.py -q
step official_thinking $PY scripts/omni/run_grounding.py --official --out $R/official_nf4_fast
step buv_vanilla_nothink $PY scripts/omni/run_grounding.py --buv --no-thinking --max-new-tokens 1024 \
    --out $R/buv_vanilla_nothink
step buv_guideline_nothink $PY scripts/omni/run_grounding.py --buv --no-thinking --max-new-tokens 1024 \
    --guideline $G --out $R/buv_guideline_nothink
step precision_nothink $PY scripts/omni/validate_precision.py --records $R/buv_vanilla_nothink/records.jsonl \
    --n 20 --out $R/precision_check_nothink
step summary_nothink $PY scripts/omni/summarize_grounding.py $R/buv_vanilla_nothink $R/buv_guideline_nothink
step buv_vanilla_thinking $PY scripts/omni/run_grounding.py --buv --out $R/buv_vanilla_thinking
step buv_guideline_thinking $PY scripts/omni/run_grounding.py --buv --guideline $G --out $R/buv_guideline_thinking
step precision_thinking $PY scripts/omni/validate_precision.py --records $R/buv_vanilla_thinking/records.jsonl \
    --n 8 --out $R/precision_check_thinking
step summary_all $PY scripts/omni/summarize_grounding.py $R/buv_vanilla_nothink $R/buv_guideline_nothink \
    $R/buv_vanilla_thinking $R/buv_guideline_thinking
echo "$(date '+%F %T') QUEUE DONE" | tee -a "$LOG_DIR/queue.log"
