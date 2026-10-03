#!/bin/bash
# MultiTryMH (v1 baseline) vs PreSTO + MultiTryMH v2 on punakha, resource-probed, with one shared settings block.
# Runs multi_try_mh.resource_probe.sh (v1 baseline) and then subtree_prefetching_multi_try_mh_v2.resource_probe.sh
# (v2 PreSTO arm). Submits both arms' jobs with sbatch on Punakha (dgx partition, Conda env vllm-cuda130).
# Edit the settings below, then run:
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family/multi_try_mh_vs_subtree_prefetching_multi_try_mh_v2.resource_probe.punakha.sbatch.sh

set -euo pipefail

# ---- Settings (edit here; shared by both arms) ----
export DATASETS="lcb_v6 math500 aime mbpp gpqa human_eval mmlu"  # space-separated
export MODELS="qwen3.5-9b qwen3-4b qwen3.5-4b qwen3-8b gemma-12b-it"  # keys of power_sharpening.tasks.constants.MODEL_MAP, e.g. "qwen3.5-9b"
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
export BUDGETS="10 20"                    # PreSTO v2 arm only; each must be >= NUM_TRIES
export PRINT_TREE=true                    # PreSTO v2 arm only
export RESOURCE_PROBE=on                  # on | off | auto
export KV_CACHE_MODE=hooks                # hooks | inherit
export EXTRA_OVERRIDES=""                 # extra Python key=value settings, e.g. "dtype=bfloat16 max_model_len=8192"
export JOB_PER=cell                       # cell | model (one job runs every rank/budget)
export PUNAKHA_MACHINE=dgx
export TIME=1-00:00:00                    # Slurm time limit (QOS max 1-00:00:00)
export RUN_DATE="$(date +%F)"             # log folder date
export DRY_RUN=0                          # 1 prints the commands only
# ---------------------------------------------------

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
METHODS=baseline bash "$script_dir/multi_try_mh.resource_probe.sh" punakha
METHODS=presto bash "$script_dir/subtree_prefetching_multi_try_mh_v2.resource_probe.sh" punakha
