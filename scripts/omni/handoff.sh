#!/usr/bin/env bash
# Wait until queue 1 finished its no-thinking steps, stop it (its thinking steps move to queue 2),
# then run queue 2 on the same GPU.
cd "$(dirname "$0")/../.."
Q=results/omni/logs/queue.log
until grep -q "END   summary_nothink" $Q; do sleep 20; done
pkill -f "scripts/omni/run_queue.sh"
sleep 2
pkill -f "run_grounding.py --buv --out results/omni/buv_vanilla_thinking"
while pgrep -f "run_grounding.py --buv --out results/omni/buv_vanilla_thinking" >/dev/null; do sleep 5; done
echo "$(date '+%F %T') HANDOFF queue1 -> queue2" >> $Q
exec bash scripts/omni/run_queue2.sh
