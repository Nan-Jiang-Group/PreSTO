#!/bin/bash
# The power_mh_and_subtree_prefetch.long_mh_steps+resource_prob_one-dataset.vllm.sh grid on interactive. Runs in this shell on an already-allocated GPU with an active Python environment.
# Edit the settings below, then run:
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/power_mh_and_subtree_prefetch.long_mh_steps+resource_prob_one-dataset.vllm.interactive.sh

# ---- Settings (edit here) ----
export METHODS="baseline presto"          # baseline and/or presto
export DATASETS=lcb_v6                    # space-separated, e.g. "math500 aime"
export MODELS=qwen3.5-9b                  # keys of power_sharpening.tasks.constants.MODEL_MAP, e.g. "qwen3.5-9b"
export ALPHA=4.0                          # sharpening power p^alpha
export MCMC_STEPS=100                     # total MH steps
export NUM_BLOCKS=1
export MAX_SAMPLES=20                     # number of benchmark problems
export MAX_NEW_TOKENS=1024
export TEMPERATURE=-1                     # -1 uses proposal temperature 1/alpha
export RANKS="accept_first reject_first longest_first smallest_cut_diff longest_path_first bfs_accept_first bfs_reject_first" # presto arm only
export BUDGETS="4 6 8 10 12 14 16 18 20"  # presto arm only
export PRINT_TREE=true
export RESOURCE_PROBE=on                  # on | off | auto
export KV_CACHE_MODE=hooks                # hooks | inherit
export EXTRA_OVERRIDES=""                 # extra Python key=value settings, e.g. "dtype=bfloat16 max_model_len=8192"
export JOB_PER=cell                       # cell | model (one job runs every rank/budget)
export RUN_DATE="$(date +%F)"             # log folder date
export DRY_RUN=0                          # 1 prints the commands only
# ------------------------------

exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/power_mh_and_subtree_prefetch.long_mh_steps+resource_prob_one-dataset.vllm.sh" interactive "$@"
