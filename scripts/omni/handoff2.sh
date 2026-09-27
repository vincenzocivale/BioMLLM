#!/usr/bin/env bash
# Wait for queue 1's vanilla no-thinking run, stop queue 1, run queue 2 (which carries the remaining
# queue-1 steps: precision check first, free no-thinking guideline run last).
cd "$(dirname "$0")/../.."
Q=results/omni/logs/queue.log
until grep -q "END   buv_vanilla_nothink" $Q; do sleep 20; done
pkill -f "scripts/omni/run_queue.sh"
sleep 2
pkill -f "run_grounding.py --buv --no-thinking --max-new-tokens 1024 --guideline"
while pgrep -f "run_grounding.py --buv --no-thinking --max-new-tokens 1024 --guideline" >/dev/null; do sleep 5; done
echo "$(date '+%F %T') HANDOFF queue1 -> queue2" >> $Q
exec bash scripts/omni/run_queue2.sh
