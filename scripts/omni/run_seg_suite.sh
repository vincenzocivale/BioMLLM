#!/usr/bin/env bash
# Segmentation readout of frozen Qwen3-Omni features on the us_bench suite, on CPU (the GPU is
# busy with the 30 B model). Three datasets at a time, 24 threads each. GT-box oracle + full-image
# lower bound per structure; the Omni-predicted-box condition is evaluated later (--eval-pred).
set -u
cd "$(dirname "$0")/../.."
export BIOMLLM_ROOT=${BIOMLLM_ROOT:-/raid/DATASETS/BioMLLMData}
source scripts/env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONPATH=src
PY=${PY:-/tmp/vcivale_envs/biomllm/bin/python}
R=results/omni/seg
LOG=$R/logs
mkdir -p $LOG
run() {
    echo "$(date '+%F %T') START $1" >> $LOG/seg_queue.log
    $PY scripts/omni/train_mask_decoder.py --dataset $1 --device cpu --threads 24 --out $R/$1 > $LOG/$1.log 2>&1
    echo "$(date '+%F %T') END   $1 rc=$?" >> $LOG/seg_queue.log
}
lane() { for d in "$@"; do run $d; done; }
lane breast_lesions busi hc18 &
lane bus_uclm psfhs camus &
lane mmotu busbra tn3k &
wait
echo "$(date '+%F %T') SEG SUITE DONE" >> $LOG/seg_queue.log
