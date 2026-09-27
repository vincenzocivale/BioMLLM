#!/usr/bin/env bash
# When the first steering model has been evaluated, stop queue 2 (it would start the neutral-control
# training) and continue with queue 3 (suite grounding first, then the rest of queue 2).
cd "$(dirname "$0")/../.."
Q=results/omni/logs/queue.log
until grep -q "END   forced_steer_guideline_neutraltext" $Q; do sleep 20; done
pkill -f "scripts/omni/run_queue2.sh"
sleep 2
pkill -f "train_steering.py --steer-text configs/omni/guidelines/neutral_control.txt"
while pgrep -f "train_steering.py --steer-text configs/omni/guidelines/neutral_control.txt" >/dev/null; do sleep 5; done
rm -rf results/omni/steer_neutral  # partial run of queue 2, restarted from scratch in queue 3
echo "$(date '+%F %T') HANDOFF queue2 -> queue3" >> $Q
exec bash scripts/omni/run_queue3.sh
