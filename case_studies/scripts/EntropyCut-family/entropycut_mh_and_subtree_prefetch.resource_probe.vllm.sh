#!/bin/bash
# EntropyCut MH baseline and PreSTO + EntropyCutMH with resource probing, one job per cell. Both arms use entropy
# cuts (exponent CUT_POWER, default 4) and a constant 1/alpha proposal temperature. Focused rerun, e.g.:
#   DATASETS="gpqa human_eval mmlu" bash ... punakha
# Usage: run the entry point for your cluster (entropycut_mh_and_subtree_prefetch.resource_probe.vllm.interactive.sh, entropycut_mh_and_subtree_prefetch.resource_probe.vllm.tacc.sbatch.sh, or entropycut_mh_and_subtree_prefetch.resource_probe.vllm.punakha.sbatch.sh), or pass
# the cluster here (interactive | punakha | tacc). DRY_RUN=1 prints the commands only.
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/EntropyCut-family/entropycut_mh_and_subtree_prefetch.resource_probe.vllm.punakha.sbatch.sh
# Environment variables such as METHODS, DATASETS, MODELS, RANKS, or BUDGETS override the defaults below; see
# ../_launch_common.sh for the full list.

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)

family=entropycut
d_methods="baseline presto"
d_datasets="math500 aime mbpp gpqa human_eval lcb_v6 mmlu"
d_models="qwen3.5-4b qwen3-4b qwen3-8b gemma-12b-it qwen3.5-9b"
d_ranks=accept_first
d_budgets="10 20"
d_punakha_time=1-00:00:00

source "$script_dir/../_launch_common.sh" "$@"
