#!/bin/bash
# Smoke test for multi_try_mh_vs_subtree_prefetching_multi_try_mh_v2.resource_probe.tacc.sbatch.sh on TACC gh-dev.
# Submits one dataset/model cell per arm (MultiTryMH v1 baseline and PreSTO + MultiTryMH v2) with few problems, so
# each job fits the gh-dev limits (2 hours, 1 running job, 3 submitted jobs per user). When both finish cleanly,
# run the full sweep with the gh launcher (multi_try_mh_vs_subtree_prefetching_multi_try_mh_v2.resource_probe.tacc.sbatch.sh).
# Edit the settings below, then run:
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family/multi_try_mh_vs_subtree_prefetching_multi_try_mh_v2.resource_probe.tacc-dev.sbatch.sh

set -euo pipefail

# ---- Settings (edit here; shared by both arms) ----
ARMS="baseline presto"                    # baseline (v1) and/or presto (v2); at most 3 jobs may be submitted on gh-dev
export DATASETS="math500"                 # keep to one dataset on gh-dev
export MODELS="qwen3-4b"                  # keep to one model on gh-dev
export ALPHA=4.0                          # sharpening power p^alpha
export MCMC_STEPS=100                     # total MH steps
export NUM_BLOCKS=1
export MAX_SAMPLES=2                      # number of benchmark problems; small so the job fits in 2 hours
export MAX_NEW_TOKENS=1024
export TEMPERATURE=-1                     # -1 uses proposal temperature 1/alpha
export NUM_TRIES=4
export PROPOSAL_TEMPERATURES="[0.25,0.5,1.0]"
export SCORING_BATCH_SIZE=128             # v1 baseline only; v2 scores during generation
export RANKS=bfs_accept_first             # PreSTO v2 arm only; keep to one on gh-dev
export BUDGETS=10                         # PreSTO v2 arm only; each must be >= NUM_TRIES; keep to one on gh-dev
export PRINT_TREE=true                    # PreSTO v2 arm only
export RESOURCE_PROBE=on                  # on | off | auto
export KV_CACHE_MODE=hooks                # hooks | inherit
export EXTRA_OVERRIDES=""                 # extra Python key=value settings, e.g. "dtype=bfloat16 max_model_len=8192"
export JOB_PER=cell                       # cell | model (one job runs every rank/budget)
export TACC_PARTITION=gh-dev
export TACC_ACCOUNT=CCR25054
export TIME=02:00:00                      # Slurm time limit (gh-dev max 02:00:00)
export RUN_DATE="$(date +%F)"             # log folder date
export DRY_RUN=0                          # 1 prints the commands only
# ---------------------------------------------------

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ " $ARMS " == *" baseline "* ]]; then
    METHODS=baseline bash "$script_dir/multi_try_mh.resource_probe.sh" tacc
fi
if [[ " $ARMS " == *" presto "* ]]; then
    METHODS=presto bash "$script_dir/subtree_prefetching_multi_try_mh_v2.resource_probe.sh" tacc
fi
