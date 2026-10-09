#!/bin/bash
# The EntropyCut bfs_vs_predictive_dp.subtree_prefetch.vllm.sh grid on interactive. Runs in this shell on an already-allocated GPU with an active Python environment.
# Edit the settings below, then run:
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/predictive_prefetching_scripts/EntropyCut-family/bfs_vs_predictive_dp.subtree_prefetch.vllm.interactive.sh

# ---- Settings (edit here) ----
export METHODS=presto                     # presto only: both arms are traversal rules of the prefetch tree
export DATASETS="lcb_v6"                  # space-separated, e.g. "math500 aime"
export MODELS="qwen3.5-9b"                # keys of power_sharpening.tasks.constants.MODEL_MAP
export ALPHA=4.0                          # sharpening power p^alpha
export MCMC_STEPS=100                     # total MH steps
export NUM_BLOCKS=1
export MAX_SAMPLES=20                     # number of benchmark problems
export MAX_NEW_TOKENS=1024
export TEMPERATURE=-1                     # -1 uses proposal temperature 1/alpha
export CUT_POWER=4.0                      # EntropyCut exponent
export RANKS="bfs_accept_first predictive_dp" # traversal rules compared: default BFS vs the predictive DP
export BUDGETS="10"                       # presto arm only
export PRINT_TREE=true                    # keep true: the per-request features are printed with the tree
export RESOURCE_PROBE=off                 # on | off | auto
export KV_CACHE_MODE=inherit              # hooks | inherit
export EXTRA_OVERRIDES=""                 # extra Python key=value settings, e.g. "dtype=bfloat16 max_model_len=8192"
export JOB_PER=cell                       # cell | model (one job runs every rank/budget)
export LOG_ROOT=case_studies/logs_predictive_prefetching/EntropyCut-family # log tree, relative to the repo root
export RUN_DATE="$(date +%F)"             # log folder date
# ------------------------------

exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/bfs_vs_predictive_dp.subtree_prefetch.vllm.sh" interactive "$@"
