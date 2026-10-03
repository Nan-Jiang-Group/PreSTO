#!/bin/bash
# PreSTO (SubTreeMH) runs that log the acceptance-predictor features: print_tree=true makes the sampler print, every MH
# step, the root's per-token log p / delta log p (and entropy / delta entropy under EntropyCut) before the cut draw, and
# a per-request feature table after the tree and sampled path. Analyze the logs with
# case_studies/draw/run_acceptance_predictors.sh.
# FAMILY=uniform (default) uses uniform cuts and logs only likelihood; FAMILY=entropycut uses the EntropyCut sampler,
# which also has the entropies, and writes to case_studies/logs-entropycut.
# Usage: run the entry point for your cluster (acceptance_predictor.subtree_prefetch.vllm.interactive.sh,
# acceptance_predictor.subtree_prefetch.vllm.tacc.sbatch.sh, or acceptance_predictor.subtree_prefetch.vllm.punakha.sbatch.sh),
# or pass the cluster here (interactive | punakha | tacc). DRY_RUN=1 prints the commands only.
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/predictive_prefetching_scripts/acceptance_predictor.subtree_prefetch.vllm.interactive.sh
#   FAMILY=entropycut bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/predictive_prefetching_scripts/acceptance_predictor.subtree_prefetch.vllm.tacc.sbatch.sh
# Environment variables such as DATASETS, MODELS, RANKS, or BUDGETS override the defaults below; see
# ../scripts/_launch_common.sh for the full list.

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)

family=${FAMILY:-uniform}
if [[ "$family" != uniform && "$family" != entropycut ]]; then
    echo "Invalid FAMILY=$family. Use uniform or entropycut." >&2
    exit 2
fi
d_methods=presto
d_datasets=lcb_v6
d_models=qwen3.5-9b
d_ranks=bfs_accept_first
d_budgets=10
# The features are printed only with the tree; the analysis needs them.
d_print_tree=true
d_resource_probe=off
d_kv_cache_mode=inherit
d_job_per=cell

source "$script_dir/../scripts/_launch_common.sh" "$@"
