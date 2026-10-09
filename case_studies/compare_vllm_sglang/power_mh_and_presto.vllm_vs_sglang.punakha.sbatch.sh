#!/bin/bash
# The power_mh_and_presto.vllm_vs_sglang.sh comparison on Punakha. Submits one sbatch job per backend/method/dataset/
# model/budget on Punakha (dgx partition; Conda envs vllm-cuda130 for vLLM and sgl-cuda129 for SGLang).
# Edit the settings below, then run:
#   bash /work/njiang/data/Subtree-Prefetching-Power-Sharpening/case_studies/compare_vllm_sglang/power_mh_and_presto.vllm_vs_sglang.punakha.sbatch.sh

# ---- Settings (edit here) ----
export BACKENDS="vllm sglang"             # vllm and/or sglang
export METHODS="baseline presto"          # baseline (PowerMH) and/or presto (PreSTO-PowerMH)
export DATASETS="lcb_v6"                  # space-separated, e.g. "math500 lcb_v6"
export MODELS="qwen3.5-9b"                # keys of power_sharpening.tasks.constants.MODEL_MAP
export ALPHA=4.0                          # sharpening power p^alpha; proposal temperature is 1/alpha
export MCMC_STEPS=100                     # total MH steps
export NUM_BLOCKS=1
export MAX_SAMPLES=20                     # number of benchmark problems
export MAX_NEW_TOKENS=1024
export BUDGETS="20"                       # PreSTO prefetch budgets
export RANK_FN=accept_first               # PreSTO subtree ranking (prefetch_rank.RANK_FNS key)
export RESOURCE_PROBE=off                 # vLLM only: on | off | auto
export VLLM_CONDA_ENV=vllm-cuda130
export SGLANG_CONDA_ENV=sgl-cuda129
export PUNAKHA_MACHINE=dgx                # partition; QOS is punakha_<machine>_general
export TIME=20:00:00                      # Slurm time limit (Punakha QOS max 1-00:00:00)
export RUN_DATE="$(date +%F)"             # log folder date
export DRY_RUN=${DRY_RUN:-0}             # 1 prints the commands only
# ------------------------------

exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/power_mh_and_presto.vllm_vs_sglang.sh" punakha "$@"
