#!/usr/bin/env bash
# Mount the project folder on DUNE without root (sshfs + passwordless ssh key).
# Unmount with:  fusermount -u ~/dune_data
set -euo pipefail
# IP instead of the hostname: name resolution of "dune" is intermittent on this network.
REMOTE="${DUNE_REMOTE:-vcivale@192.168.1.237:/raid/DATASETS/BioMLLMData}"
MOUNTPOINT="${BIOMLLM_ROOT:-$HOME/dune_data}"
mkdir -p "$MOUNTPOINT"
if findmnt "$MOUNTPOINT" >/dev/null 2>&1; then
    echo "already mounted: $MOUNTPOINT"
    exit 0
fi
sshfs "$REMOTE" "$MOUNTPOINT" -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,BatchMode=yes
echo "mounted $REMOTE on $MOUNTPOINT"
