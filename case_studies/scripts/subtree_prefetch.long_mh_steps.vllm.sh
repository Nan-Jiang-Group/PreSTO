#!/bin/bash
# PreSTO (SubTreeMH) over the full dataset x model x rank x budget grid, without resource probing.
# Each dataset/model gets one allocation that runs its rank x budget cells back to back; narrow MODELS/BUDGETS so a
# Punakha job stays inside the 24 h QOS limit.
# Usage: run the entry point for your cluster (subtree_prefetch.long_mh_steps.vllm.interactive.sh, subtree_prefetch.long_mh_steps.vllm.tacc.sbatch.sh, or subtree_prefetch.long_mh_steps.vllm.punakha.sbatch.sh), or pass
# the cluster here (interactive | punakha | tacc). DRY_RUN=1 prints the commands only.
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/subtree_prefetch.long_mh_steps.vllm.punakha.sbatch.sh
# Environment variables such as DATASETS, MODELS, RANKS, or BUDGETS override the defaults below; see
# _launch_common.sh for the full list.

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)

d_methods=presto
d_datasets="math500 aime mbpp gpqa human_eval lcb_v6 mmlu"
d_models="qwen qwen3-4b qwen3.5-4b qwen3-8b qwen3.5-9b qwen-math-medium tulu phi3.5"
d_ranks="accept_first reject_first longest_first smallest_cut_diff longest_path_first bfs_accept_first bfs_reject_first"
d_budgets="4 6 8 10 12 14 16 18 20"
d_resource_probe=auto
d_kv_cache_mode=inherit
d_job_per=model

source "$script_dir/_launch_common.sh" "$@"
