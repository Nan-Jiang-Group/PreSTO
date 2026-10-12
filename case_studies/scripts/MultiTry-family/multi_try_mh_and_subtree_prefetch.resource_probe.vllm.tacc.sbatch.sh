#!/bin/bash
# The multi_try_mh_and_subtree_prefetch.resource_probe.vllm.sh grid on tacc. Submits its jobs with sbatch on TACC (gh partition, Conda env cuda130).
# Edit the settings below, then run:
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family/multi_try_mh_and_subtree_prefetch.resource_probe.vllm.tacc.sbatch.sh

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
export NUM_TRIES=4
export PROPOSAL_TEMPERATURES="[0.25,0.5,1.0]"
export SCORING_BATCH_SIZE=128
export RANKS=bfs_accept_first             # presto arm only
export BUDGETS="4 6 8 10 12 14 16 18 20"  # presto arm only; each must be >= NUM_TRIES
export PRINT_TREE=false
export RESOURCE_PROBE=on                  # on | off | auto
export KV_CACHE_MODE=hooks                # hooks | inherit
export EXTRA_OVERRIDES=""                 # extra Python key=value settings, e.g. "dtype=bfloat16 max_model_len=8192"
export JOB_PER=cell                       # cell | model (one job runs every rank/budget)
export TACC_PARTITION=gh
export TACC_ACCOUNT=CCR25054
export TIME=12:00:00                      # Slurm time limit
export RUN_DATE="$(date +%F)"             # log folder date
# ------------------------------

exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/multi_try_mh_and_subtree_prefetch.resource_probe.vllm.sh" tacc "$@"
