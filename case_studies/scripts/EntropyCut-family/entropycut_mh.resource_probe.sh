#!/bin/bash
# EntropyCut MH only (no PreSTO arm), resource-probed, on Math500 / Qwen. Entropy cuts with exponent
# CUT_POWER (default 4) and a constant 1/alpha proposal temperature.
# Usage: run the entry point for your cluster (entropycut_mh.resource_probe.interactive.sh, entropycut_mh.resource_probe.tacc.sbatch.sh, or entropycut_mh.resource_probe.punakha.sbatch.sh), or pass
# the cluster here (interactive | punakha | tacc).
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/EntropyCut-family/entropycut_mh.resource_probe.interactive.sh
# Environment variables such as DATASETS, MODELS, RANKS, or BUDGETS override the defaults below; see
# ../_launch_common.sh for the full list.

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)

family=entropycut
d_methods=baseline
d_datasets=math500
d_models=qwen
d_punakha_time=1-00:00:00

source "$script_dir/../_launch_common.sh" "$@"
