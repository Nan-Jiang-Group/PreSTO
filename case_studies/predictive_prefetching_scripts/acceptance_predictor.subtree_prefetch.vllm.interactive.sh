#!/bin/bash
# The acceptance_predictor.subtree_prefetch.vllm.sh grid on interactive. Runs in this shell on an already-allocated GPU with an active Python environment.
# Edit the settings below, then run:
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/predictive_prefetching_scripts/acceptance_predictor.subtree_prefetch.vllm.interactive.sh

# ---- Settings (edit here) ----
export FAMILY=${FAMILY:-uniform}          # uniform (likelihood only) | entropycut (likelihood + entropy)
export METHODS=presto                     # presto only: the features come from the prefetch tree
export DATASETS="lcb_v6"                  # space-separated, e.g. "math500 aime"
export MODELS="qwen3.5-9b"                 # keys of power_sharpening.tasks.constants.MODEL_MAP
export ALPHA=4.0                          # sharpening power p^alpha
export MCMC_STEPS=100                     # total MH steps
export NUM_BLOCKS=1
export MAX_SAMPLES=20                     # number of benchmark problems
export MAX_NEW_TOKENS=1024
export TEMPERATURE=-1                     # -1 uses proposal temperature 1/alpha
export RANKS="bfs_accept_first"           # presto arm only
export BUDGETS="10"                       # presto arm only
export PRINT_TREE=true                    # must stay true: the features are printed with the tree
export RESOURCE_PROBE=off                 # on | off | auto
export KV_CACHE_MODE=inherit              # hooks | inherit
export EXTRA_OVERRIDES=""                 # extra Python key=value settings, e.g. "dtype=bfloat16 max_model_len=8192"
export JOB_PER=cell                       # cell | model (one job runs every rank/budget)
export RUN_DATE="$(date +%F)"             # log folder date
export DRY_RUN=0                          # 1 prints the commands only
# ------------------------------

exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/acceptance_predictor.subtree_prefetch.vllm.sh" interactive "$@"
