#!/bin/bash
# PreSTO (SubTreeMH) runs that compare two traversal rules at the same prefetch budget: the default BFS rule
# (rank bfs_accept_first) and the predictive dynamic program (rank predictive_dp; Appendix C.2), which picks the
# ancestor-closed request set maximizing the predicted transition count, with h_hat from the three-class logistic
# regression on the cut fraction c_j/T (Appendix E.2). Each rule runs as its own job with a rank-<rule> run name.
# The runs also log the acceptance-predictor features: print_tree=true makes the sampler print, every MH
# step, the root's per-token log p / delta log p (and entropy / delta entropy under EntropyCut) before the cut draw, and
# a per-request feature table after the tree and sampled path. Analyze the logs with
# case_studies/draw/run_acceptance_predictors.sh.
# FAMILY=uniform (default) uses uniform cuts and logs only likelihood; FAMILY=entropycut uses the EntropyCut sampler,
# which also has the entropies. Logs go to case_studies/logs_predictive_prefetching/{Uniform,EntropyCut}-family.
# Usage: run the entry point for your cluster (bfs_vs_predictive_dp.subtree_prefetch.vllm.interactive.sh,
# bfs_vs_predictive_dp.subtree_prefetch.vllm.tacc.sbatch.sh, or bfs_vs_predictive_dp.subtree_prefetch.vllm.punakha.sbatch.sh),
# or pass the cluster here (interactive | punakha | tacc).
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/predictive_prefetching_scripts/bfs_vs_predictive_dp.subtree_prefetch.vllm.interactive.sh
#   FAMILY=entropycut bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/predictive_prefetching_scripts/bfs_vs_predictive_dp.subtree_prefetch.vllm.tacc.sbatch.sh
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
d_ranks="bfs_accept_first predictive_dp"
d_budgets=10
# The per-request features are printed only with the tree; the predictor analysis needs them.
d_print_tree=true
d_resource_probe=off
d_kv_cache_mode=inherit
d_job_per=cell
# Keep these runs apart from the other runs (LOG_ROOT overrides it).
if [[ "$family" == entropycut ]]; then
    d_log_root=case_studies/logs_predictive_prefetching/EntropyCut-family
else
    d_log_root=case_studies/logs_predictive_prefetching/Uniform-family
fi

source "$script_dir/../scripts/_launch_common.sh" "$@"
