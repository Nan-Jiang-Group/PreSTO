#!/bin/bash
# MultiTryMH only (no PreSTO arm), resource-probed, on Math500 / Qwen. Four tries per transition over proposal
# temperatures [0.25,0.5,1.0], scoring batch size 128.
# Usage: run the entry point for your cluster (multi_try_mh.resource_probe.interactive.sh, multi_try_mh.resource_probe.tacc.sbatch.sh, or multi_try_mh.resource_probe.punakha.sbatch.sh), or pass
# the cluster here (interactive | punakha | tacc).
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family/multi_try_mh.resource_probe.interactive.sh
# Environment variables such as DATASETS, MODELS, RANKS, or BUDGETS override the defaults below; see
# ../_launch_common.sh for the full list.

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)

family=multitry
d_methods=baseline
d_datasets=math500
d_models=qwen
d_punakha_time=1-00:00:00

source "$script_dir/../_launch_common.sh" "$@"
