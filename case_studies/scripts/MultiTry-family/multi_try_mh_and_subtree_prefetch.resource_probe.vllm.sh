#!/bin/bash
# MultiTryMH baseline and PreSTO + MultiTryMH over prefetch budgets 4-20 with bfs_accept_first, resource-probed,
# one job per cell. Four tries per transition over proposal temperatures [0.25,0.5,1.0]; each budget must be >= NUM_TRIES.
# Usage: run the entry point for your cluster (multi_try_mh_and_subtree_prefetch.resource_probe.vllm.interactive.sh, multi_try_mh_and_subtree_prefetch.resource_probe.vllm.tacc.sbatch.sh, or multi_try_mh_and_subtree_prefetch.resource_probe.vllm.punakha.sbatch.sh), or pass
# the cluster here (interactive | punakha | tacc).
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family/multi_try_mh_and_subtree_prefetch.resource_probe.vllm.punakha.sbatch.sh
# Environment variables such as METHODS, DATASETS, MODELS, RANKS, or BUDGETS override the defaults below; see
# ../_launch_common.sh for the full list.

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)

family=multitry
d_methods="baseline presto"
d_datasets=lcb_v6
d_models=qwen3.5-9b
d_ranks=bfs_accept_first
d_budgets="4 6 8 10 12 14 16 18 20"
d_print_tree=false
d_punakha_time=1-00:00:00

source "$script_dir/../_launch_common.sh" "$@"
