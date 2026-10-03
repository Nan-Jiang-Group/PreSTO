#!/usr/bin/env bash
# Run the paper-default MH settings with the public-SGLang top-k entropy proxy.
# Usage:
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/experiments/sglang/math500/entropy_cut_mh.interactive.sh

source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"

algorithm=entropy_cut_mh
base_model=qwen-math-medium
alpha=4.0
mcmc_steps=10
num_blocks=16
cut_power=4.0
entropy_mode=topk
entropy_top_k=64

dump_dir="$basepath/result/MATH500/$(date +%F)/entropy_cut_mh"
mkdir -p "$dump_dir"

log_file="$dump_dir/dataset-MATH500.model-${base_model}.alpha-${alpha}.steps-${mcmc_steps}.blocks-${num_blocks}.beta-${cut_power}.${entropy_mode}-${entropy_top_k}.seed-${SEED}.sglang.log"

uv run --project "$basepath/src" --extra sglang \
  python -m power_sharpening.runners.sglang.run_entropy_cut_mh \
  --dataset math500 \
  --algorithm "$algorithm" \
  --model_str "$base_model" \
  --save_str "$dump_dir" \
  --override \
    seed="$SEED" \
    alpha="$alpha" \
    temperature=-1 \
    mcmc_steps="$mcmc_steps" \
    num_blocks="$num_blocks" \
    cut_power="$cut_power" \
    entropy_mode="$entropy_mode" \
    entropy_top_k="$entropy_top_k" \
  > "$log_file" 2>&1
