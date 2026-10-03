#!/bin/bash
# Matched PowerMH baseline and PreSTO runs with resource probing, one job per cell, over every
# supported dataset. Focused rerun, e.g.: DATASETS=lcb_v6 MODELS=gemma-12b-it bash ... punakha
# Usage: run the entry point for your cluster (subtree_prefetch.long_mh_steps+resource_prob_all_dataset.vllm.interactive.sh, subtree_prefetch.long_mh_steps+resource_prob_all_dataset.vllm.tacc.sbatch.sh, or subtree_prefetch.long_mh_steps+resource_prob_all_dataset.vllm.punakha.sbatch.sh), or pass
# the cluster here (interactive | punakha | tacc). DRY_RUN=1 prints the commands only.
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/subtree_prefetch.long_mh_steps+resource_prob_all_dataset.vllm.punakha.sbatch.sh
# Environment variables such as DATASETS, MODELS, RANKS, or BUDGETS override the defaults below; see
# _launch_common.sh for the full list.

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)

d_methods="baseline presto"
d_datasets="math500 aime mbpp gpqa human_eval lcb_v6 mmlu"
d_models="qwen3.5-4b qwen3-4b qwen3-8b gemma-12b-it"
d_ranks=bfs_accept_first
d_budgets="10 20"

source "$script_dir/_launch_common.sh" "$@"
