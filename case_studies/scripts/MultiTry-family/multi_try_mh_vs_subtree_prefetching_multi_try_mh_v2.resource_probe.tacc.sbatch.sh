#!/bin/bash
# MultiTryMH (v1 baseline) vs PreSTO + MultiTryMH v2 on tacc, resource-probed, with one shared settings block.
# Runs subtree_prefetching_multi_try_mh_v2.resource_probe.sh (v2 PreSTO arm) over every DATASETS x MODELS cell, and
# multi_try_mh.resource_probe.sh (v1 baseline) first only when RUN_BASELINE=1. Submits the jobs with sbatch on TACC
# (gh partition, Conda env cuda130).
# Edit the settings below, then run:
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family/multi_try_mh_vs_subtree_prefetching_multi_try_mh_v2.resource_probe.tacc.sbatch.sh

set -euo pipefail

# ---- Settings (edit here; shared by both arms) ----
export DATASETS="math500 mbpp gpqa human_eval lcb_v6"  # space-separated
export MODELS="qwen3.5-4b  qwen3.5-9b gemma-12b-it"  # keys of power_sharpening.tasks.constants.MODEL_MAP, e.g. "qwen3.5-9b"
export ALPHA=4.0                          # sharpening power p^alpha
export MCMC_STEPS=100                     # total MH steps
export NUM_BLOCKS=1
export MAX_SAMPLES=20                     # number of benchmark problems
export MAX_NEW_TOKENS=1024
export TEMPERATURE=-1                     # -1 uses proposal temperature 1/alpha
export NUM_TRIES=4
export PROPOSAL_TEMPERATURES="[0.25,0.5,1.0]"
export SCORING_BATCH_SIZE=128             # v1 baseline only; v2 scores during generation
export RANKS=bfs_accept_first             # PreSTO v2 arm only
export BUDGETS=20                         # PreSTO v2 arm only; each must be >= NUM_TRIES
export PRINT_TREE=true                    # PreSTO v2 arm only
export RESOURCE_PROBE=on                  # on | off | auto
export KV_CACHE_MODE=hooks                # hooks | inherit
export EXTRA_OVERRIDES=""                 # extra Python key=value settings, e.g. "dtype=bfloat16 max_model_len=8192"
export JOB_PER=cell                       # cell | model (one job runs every rank/budget)
export TACC_PARTITION=gh
export TACC_ACCOUNT=CCR25054
export TIME=48:00:00                      # Slurm time limit
export RUN_DATE="$(date +%F)"             # log folder date
RUN_BASELINE=1                            # 1 also submits the v1 baseline arm (already run 2026-09-24)
# ---------------------------------------------------

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "$RUN_BASELINE" == 1 ]]; then
    METHODS=baseline bash "$script_dir/multi_try_mh.resource_probe.sh" tacc
fi
METHODS=presto bash "$script_dir/subtree_prefetching_multi_try_mh_v2.resource_probe.sh" tacc
