#!/bin/bash
# PreSTO + EntropyCutMH runs that compare two traversal rules at the same prefetch budget: the default BFS rule
# (rank bfs_accept_first) and the predictive dynamic program (rank predictive_dp; Appendix C.2) with h_hat from the
# cut-fraction logistic regression fit on EntropyCut runs (Appendix E.2). Each rule runs as its own job with a
# rank-<rule> run name. The runs also log the acceptance-predictor features: print_tree=true makes the sampler print, every
# MH step, the root's per-token log p / delta log p and entropy / delta entropy before the entropy-cut draw, and a
# per-request feature table after the tree and sampled path. Cuts use exponent CUT_POWER (default 4) with a constant
# 1/alpha proposal temperature. Logs go to case_studies/logs_predictive_prefetching/EntropyCut-family/<dataset>/<date>/vllm
# (LOG_ROOT overrides it), apart from the other EntropyCut runs. Analyze them with case_studies/draw/run_acceptance_predictors.sh.
# Usage: run the entry point for your cluster (bfs_vs_predictive_dp.subtree_prefetch.vllm.interactive.sh), or pass the
# cluster here (interactive | punakha | tacc).
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/predictive_prefetching_scripts/EntropyCut-family/bfs_vs_predictive_dp.subtree_prefetch.vllm.interactive.sh
# Environment variables such as DATASETS, MODELS, RANKS, BUDGETS, or CUT_POWER override the defaults below; see
# ../../scripts/_launch_common.sh for the full list.

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)

family=entropycut
d_methods=presto
d_datasets=lcb_v6
d_models=qwen3.5-9b
d_ranks="bfs_accept_first predictive_dp"
d_budgets=10
d_cut_power=4.0
# The per-request features are printed only with the tree; the predictor analysis needs them.
d_print_tree=true
d_resource_probe=off
d_kv_cache_mode=inherit
d_job_per=cell
# Keep these logs apart from the EntropyCut comparison runs in case_studies/logs-entropycut.
d_log_root=case_studies/logs_predictive_prefetching/EntropyCut-family

source "$script_dir/../../scripts/_launch_common.sh" "$@"
