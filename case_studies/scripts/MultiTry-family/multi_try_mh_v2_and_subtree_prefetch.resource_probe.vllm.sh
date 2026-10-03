#!/bin/bash
# MultiTryMH v2 baseline and PreSTO + MultiTryMH v2, resource-probed. v2 scores each suffix at every mixture temperature
# during generation, so there are no extra scoring requests and SCORING_BATCH_SIZE does not apply. Baseline only:
#   METHODS=baseline bash ... interactive
# Usage: run the entry point for your cluster (multi_try_mh_v2_and_subtree_prefetch.resource_probe.vllm.interactive.sh, multi_try_mh_v2_and_subtree_prefetch.resource_probe.vllm.tacc.sbatch.sh, or multi_try_mh_v2_and_subtree_prefetch.resource_probe.vllm.punakha.sbatch.sh), or pass
# the cluster here (interactive | punakha | tacc). DRY_RUN=1 prints the commands only.
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family/multi_try_mh_v2_and_subtree_prefetch.resource_probe.vllm.punakha.sbatch.sh
# Environment variables such as METHODS, DATASETS, MODELS, RANKS, or BUDGETS override the defaults below; see
# ../_launch_common.sh for the full list.

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)

family=multitry-v2
d_methods="baseline presto"
d_datasets=math500
d_models=qwen
d_ranks=bfs_accept_first
d_budgets="4 8 12"
d_print_tree=false
d_punakha_time=1-00:00:00

source "$script_dir/../_launch_common.sh" "$@"
