#!/bin/bash
# PreSTO on Qwen3.5-27B on one 80 GB GPU. Capping max_model_len keeps vLLM from reserving KV cache for
# the full native context; 8,192 still covers the 1,024-token generations.
# Usage: run the entry point for your cluster (subtree_prefetch.large_baseLLM.vllm.interactive.sh, subtree_prefetch.large_baseLLM.vllm.tacc.sbatch.sh, or subtree_prefetch.large_baseLLM.vllm.punakha.sbatch.sh), or pass
# the cluster here (interactive | punakha | tacc). DRY_RUN=1 prints the commands only.
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/subtree_prefetch.large_baseLLM.vllm.punakha.sbatch.sh
# Environment variables such as DATASETS, MODELS, RANKS, or BUDGETS override the defaults below; see
# _launch_common.sh for the full list.

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)

d_methods=presto
d_datasets=lcb_v6
d_models=qwen3.5-27b
d_ranks=bfs_reject_first
d_budgets="8 10 12 14 16 18 20"
d_print_tree=false
d_resource_probe=off
d_extra_overrides="dtype=bfloat16 gpu_memory_utilization=0.95 max_model_len=8192"

source "$script_dir/_launch_common.sh" "$@"
