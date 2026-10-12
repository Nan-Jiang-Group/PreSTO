#!/bin/bash
# PowerMH baseline over the full dataset x model grid, without resource probing. It pairs with
# subtree_prefetch.long_mh_steps.vllm.sh; submits one job per dataset/model.
# Usage: run the entry point for your cluster (power_mh.long_mh_steps.interactive.sh, power_mh.long_mh_steps.tacc.sbatch.sh, or power_mh.long_mh_steps.punakha.sbatch.sh), or pass
# the cluster here (interactive | punakha | tacc).
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/power_mh.long_mh_steps.punakha.sbatch.sh
# Environment variables such as DATASETS, MODELS, RANKS, or BUDGETS override the defaults below; see
# _launch_common.sh for the full list.

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)

d_methods=baseline
d_datasets="math500 aime mbpp gpqa human_eval lcb_v6 mmlu"
d_models="qwen qwen3-4b qwen3.5-4b qwen3-8b qwen3.5-9b qwen-math-medium tulu phi3.5"
d_resource_probe=auto
d_kv_cache_mode=inherit

source "$script_dir/_launch_common.sh" "$@"
