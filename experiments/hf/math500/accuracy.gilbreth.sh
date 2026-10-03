#!/bin/zsh
# NOT RUNNABLE: `math_experiments.eval_math` was dropped in the 2026-07 restructure and no CSV-folder accuracy evaluator
# exists in the package yet (the grader itself lives at power_sharpening.tasks.grader_utils.math_grader).
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"

# Gilbreth-specific scratch (overrides default.sh's ~/SCRATCH cache paths).
CACHE_DIR=/scratch/gilbreth/$USER
export HF_HOME="$CACHE_DIR/.cache/huggingface"
export HF_HUB_CACHE="$CACHE_DIR/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"


dump_dir=$1


python -m math_experiments.eval_math \
  --csv_folder $dump_dir
