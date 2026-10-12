#!/bin/bash
# PreSTO + EntropyCutMH only (no baseline arm), resource-probed, on Math500 / Qwen with bfs_accept_first at budget 10
# and no tree logging. Other budgets, e.g.: BUDGETS="10 20" bash ... interactive
# Usage: run the entry point for your cluster (subtree_prefetching_mh.resource_probe.interactive.sh, subtree_prefetching_mh.resource_probe.tacc.sbatch.sh, or subtree_prefetching_mh.resource_probe.punakha.sbatch.sh), or pass
# the cluster here (interactive | punakha | tacc).
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/EntropyCut-family/subtree_prefetching_mh.resource_probe.interactive.sh
# Environment variables such as DATASETS, MODELS, RANKS, or BUDGETS override the defaults below; see
# ../_launch_common.sh for the full list.

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)

family=entropycut
d_methods=presto
d_datasets=math500
d_models=qwen
d_ranks=bfs_accept_first
d_budgets=10
d_print_tree=false
d_punakha_time=1-00:00:00

source "$script_dir/../_launch_common.sh" "$@"
