#!/bin/bash
# The entropycut_mh_and_subtree_prefetch.resource_probe.vllm.sh grid on punakha. Submits its jobs with sbatch on Punakha (dgx partition, Conda env vllm-cuda130).
# Edit the settings below, then run:
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/EntropyCut-family/entropycut_mh_and_subtree_prefetch.resource_probe.vllm.punakha.sbatch.sh

# ---- Settings (edit here) ----
export METHODS="baseline presto"          # baseline and/or presto
export DATASETS="math500 aime mbpp gpqa human_eval lcb_v6 mmlu" # space-separated, e.g. "math500 aime"
export MODELS="gemma-12b-it" # "qwen3.5-4b qwen3-4b qwen3-8b gemma-12b-it qwen3.5-9b" # keys of power_sharpening.tasks.constants.MODEL_MAP, e.g. "qwen3.5-9b"
export ALPHA=4.0                          # sharpening power p^alpha
export MCMC_STEPS=100                     # total MH steps
export NUM_BLOCKS=1
export MAX_SAMPLES=20                     # number of benchmark problems
export MAX_NEW_TOKENS=1024
export TEMPERATURE=-1                     # -1 uses proposal temperature 1/alpha
export CUT_POWER=4.0
export RANKS=accept_first                 # presto arm only
export BUDGETS=20                    # presto arm only
export PRINT_TREE=true
export RESOURCE_PROBE=on                  # on | off | auto
export KV_CACHE_MODE=hooks                # hooks | inherit
export EXTRA_OVERRIDES=""                 # extra Python key=value settings, e.g. "dtype=bfloat16 max_model_len=8192"
export JOB_PER=cell                       # cell | model (one job runs every rank/budget)
export PUNAKHA_MACHINE=dgx
export TIME=1-00:00:00                    # Slurm time limit (QOS max 1-00:00:00)
export RUN_DATE="$(date +%F)"             # log folder date
export DRY_RUN=0                          # 1 prints the commands only
# ------------------------------

exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/entropycut_mh_and_subtree_prefetch.resource_probe.vllm.sh" punakha "$@"
