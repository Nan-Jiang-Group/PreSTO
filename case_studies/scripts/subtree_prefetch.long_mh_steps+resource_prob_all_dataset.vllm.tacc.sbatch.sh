#!/bin/bash
# The subtree_prefetch.long_mh_steps+resource_prob_all_dataset.vllm.sh grid on tacc. Submits its jobs with sbatch on TACC (gh partition, Conda env cuda130).
# Edit the settings below, then run:
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/subtree_prefetch.long_mh_steps+resource_prob_all_dataset.vllm.tacc.sbatch.sh

# ---- Settings (edit here) ----
export METHODS="baseline presto"          # baseline and/or presto
export DATASETS="math500 aime mbpp gpqa human_eval lcb_v6 mmlu" # space-separated, e.g. "math500 aime"
export MODELS="qwen3.5-4b qwen3-4b qwen3-8b gemma-12b-it" # keys of power_sharpening.tasks.constants.MODEL_MAP, e.g. "qwen3.5-9b"
export ALPHA=4.0                          # sharpening power p^alpha
export MCMC_STEPS=100                     # total MH steps
export NUM_BLOCKS=1
export MAX_SAMPLES=20                     # number of benchmark problems
export MAX_NEW_TOKENS=1024
export TEMPERATURE=-1                     # -1 uses proposal temperature 1/alpha
export RANKS=bfs_accept_first             # presto arm only
export BUDGETS="10 20"                    # presto arm only
export PRINT_TREE=true
export RESOURCE_PROBE=on                  # on | off | auto
export KV_CACHE_MODE=hooks                # hooks | inherit
export EXTRA_OVERRIDES=""                 # extra Python key=value settings, e.g. "dtype=bfloat16 max_model_len=8192"
export JOB_PER=cell                       # cell | model (one job runs every rank/budget)
export TACC_PARTITION=gh
export TACC_ACCOUNT=CCR25054
export TIME=12:00:00                      # Slurm time limit
export RUN_DATE="$(date +%F)"             # log folder date
# ------------------------------

exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/subtree_prefetch.long_mh_steps+resource_prob_all_dataset.vllm.sh" tacc "$@"
