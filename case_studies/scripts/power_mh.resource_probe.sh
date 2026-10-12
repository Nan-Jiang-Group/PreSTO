#!/bin/bash
# PowerMH-only resource probe with eight blocks and 3,072 new tokens. Submits one job per
# dataset/model.
# Usage: run the entry point for your cluster (power_mh.resource_probe.interactive.sh, power_mh.resource_probe.tacc.sbatch.sh, or power_mh.resource_probe.punakha.sbatch.sh), or pass
# the cluster here (interactive | punakha | tacc).
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/power_mh.resource_probe.punakha.sbatch.sh
# Environment variables such as DATASETS, MODELS, RANKS, or BUDGETS override the defaults below; see
# _launch_common.sh for the full list.

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)

d_methods=baseline
d_datasets="math500 lcb_v6"
d_models="qwen qwen3-4b qwen3.5-4b qwen3-8b qwen3.5-9b"
d_num_blocks=8
d_max_new_tokens=3072
d_kv_cache_mode=inherit

source "$script_dir/_launch_common.sh" "$@"
