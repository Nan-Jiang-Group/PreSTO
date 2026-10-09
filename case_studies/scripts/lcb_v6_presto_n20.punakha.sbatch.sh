#!/bin/bash
# LCB V6 PreSTO-PowerMH runs at N_pf=20 (default BFS traversal) on Punakha (dgx partition, Conda env vllm-cuda130).
# These fill the LCB V6 column of the PreSTO epsilon-certainty grid (paper Figure 15) for the models whose only
# LCB V6 runs used N_pf=10 (2026-09-23); their N_pf=20 runs of 2026-09-16 read the 880-problem V5 file.
# Settings match the other cells of that grid. Edit the settings below, then run:
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/lcb_v6_presto_n20.punakha.sbatch.sh
# Logs go to case_studies/logs/lcb_v6/$RUN_DATE/vllm; then rebuild the grid with
#   LCB_RUN_DATE=$RUN_DATE bash case_studies/draw/rebuild_presto_certainty_grid.sh

# ---- Settings (edit here) ----
export METHODS=presto                     # PreSTO arm only
export DATASETS="lcb_v6"
export MODELS="qwen3-4b qwen3-8b qwen3.5-4b"
export ALPHA=4.0                          # sharpening power p^alpha
export MCMC_STEPS=100                     # total MH steps
export NUM_BLOCKS=1
export MAX_SAMPLES=20                     # first 20 problems
export MAX_NEW_TOKENS=1024
export TEMPERATURE=-1                     # -1 uses proposal temperature 1/alpha
export RANKS="bfs_accept_first"           # default BFS traversal
export BUDGETS="20"
export PRINT_TREE=true                    # the certainty grid is read from the printed trees
export RESOURCE_PROBE=on                  # on | off | auto
export KV_CACHE_MODE=hooks                # hooks | inherit
export EXTRA_OVERRIDES="prefix_cache=true gpu_memory_utilization=0.9 max_prompt_chars=3500"
export JOB_PER=model                      # one job per model
export PUNAKHA_MACHINE=dgx
export TIME=12:00:00                      # Slurm time limit (QOS max 1-00:00:00)
export RUN_DATE="$(date +%F)"             # log folder date
# ------------------------------

exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/subtree_prefetch.long_mh_steps.vllm.sh" punakha "$@"
