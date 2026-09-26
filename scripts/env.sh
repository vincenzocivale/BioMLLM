# Source before running experiments:  source scripts/env.sh
# Data, model caches and run outputs live on DUNE (/raid/DATASETS/BioMLLMData), mounted via sshfs
# because the NFS mount needs root. Override any variable before sourcing to change locations.
export BIOMLLM_ROOT="${BIOMLLM_ROOT:-$HOME/dune_data}"
export BIOMLLM_DATA="${BIOMLLM_DATA:-$BIOMLLM_ROOT/datasets}"
export BIOMLLM_FEATURES="${BIOMLLM_FEATURES:-$BIOMLLM_ROOT/expert_cache}"
export BIOMLLM_RUNS="${BIOMLLM_RUNS:-$BIOMLLM_ROOT/runs}"
# Always use the project cache on DUNE: ~/.bashrc points HF_HOME to /scratch/$USER, which does
# not exist on this workstation. Set BIOMLLM_HF_HOME to use another location.
export HF_HOME="${BIOMLLM_HF_HOME:-$BIOMLLM_ROOT/hf_cache}"

# On DUNE itself the folder is local (export BIOMLLM_ROOT=/raid/DATASETS/BioMLLMData): no mount.
if [ ! -d "$BIOMLLM_ROOT/datasets" ] && ! findmnt "$BIOMLLM_ROOT" >/dev/null 2>&1; then
    echo "warning: $BIOMLLM_ROOT is not mounted; run scripts/mount_dune.sh" >&2
fi
