#!/bin/bash
# One dataset/model: a PowerMH baseline plus PreSTO over every rank x budget, all resource-probed,
# one job per cell. Quick interactive variant: MCMC_STEPS=10 BUDGETS="8 12" bash ... interactive
# Usage: run the entry point for your cluster (power_mh_and_subtree_prefetch.long_mh_steps+resource_prob_one-dataset.vllm.interactive.sh, power_mh_and_subtree_prefetch.long_mh_steps+resource_prob_one-dataset.vllm.tacc.sbatch.sh, or power_mh_and_subtree_prefetch.long_mh_steps+resource_prob_one-dataset.vllm.punakha.sbatch.sh), or pass
# the cluster here (interactive | punakha | tacc).
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/power_mh_and_subtree_prefetch.long_mh_steps+resource_prob_one-dataset.vllm.punakha.sbatch.sh
# Environment variables such as DATASETS, MODELS, RANKS, or BUDGETS override the defaults below; see
# _launch_common.sh for the full list.

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)

d_methods="baseline presto"
d_datasets=lcb_v6
d_models=qwen3.5-9b
d_ranks="accept_first reject_first longest_first smallest_cut_diff longest_path_first bfs_accept_first bfs_reject_first"
d_budgets="4 6 8 10 12 14 16 18 20"

source "$script_dir/_launch_common.sh" "$@"
