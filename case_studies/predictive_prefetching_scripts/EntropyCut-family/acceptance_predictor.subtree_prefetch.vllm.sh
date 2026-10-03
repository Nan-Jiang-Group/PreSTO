#!/bin/bash
# PreSTO + EntropyCutMH runs that log the acceptance-predictor features: print_tree=true makes the sampler print, every
# MH step, the root's per-token log p / delta log p and entropy / delta entropy before the entropy-cut draw, and a
# per-request feature table after the tree and sampled path. Cuts use exponent CUT_POWER (default 4) with a constant
# 1/alpha proposal temperature. Logs go to case_studies/predictive_prefetching_logs/EntropyCut-family/<dataset>/<date>/vllm
# (LOG_ROOT overrides it), apart from the other EntropyCut runs. Analyze them with case_studies/draw/run_acceptance_predictors.sh.
# Usage: run the entry point for your cluster (acceptance_predictor.subtree_prefetch.vllm.interactive.sh), or pass the
# cluster here (interactive | punakha | tacc). DRY_RUN=1 prints the commands only.
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/predictive_prefetching_scripts/EntropyCut-family/acceptance_predictor.subtree_prefetch.vllm.interactive.sh
# Environment variables such as DATASETS, MODELS, RANKS, BUDGETS, or CUT_POWER override the defaults below; see
# ../../scripts/_launch_common.sh for the full list.

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)

family=entropycut
d_methods=presto
d_datasets=lcb_v6
d_models=qwen3.5-9b
d_ranks=bfs_accept_first
d_budgets=10
d_cut_power=4.0
# The features are printed only with the tree; the analysis needs them.
d_print_tree=true
d_resource_probe=off
d_kv_cache_mode=inherit
d_job_per=cell
# Keep these logs apart from the EntropyCut comparison runs in case_studies/logs-entropycut.
d_log_root=case_studies/predictive_prefetching_logs/EntropyCut-family

source "$script_dir/../../scripts/_launch_common.sh" "$@"
