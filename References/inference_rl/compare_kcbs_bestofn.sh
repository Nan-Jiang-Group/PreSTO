#!/bin/bash

set -e
set -o pipefail

log() { echo "[$(date '+%H:%M:%S')] $1"; }

model="qwen_math"
beam_width=2        # k sequences
topk=20             # per-step top-k logprobs
temperature=0.8     # for best-of-N sampling
max_problems=25
seed=0
max_new_tokens=1024

cd reasoning-with-sampling/llm_experiments

log "Running Best-of-$beam_width sampling (temp=$temperature) on first $max_problems problems..."
python vllm_passk_math.py \
  --model "$model" \
  --temperature "$temperature" \
  --k "$beam_width" \
  --max_problems "$max_problems" \
  --save_str sampling_results/ \
  --seed "$seed"

bestof_csv="sampling_results/${model}/${model}_vllm_passk_${beam_width}_temp_${temperature}_seed_${seed}.csv"

log "Running KCBS (k=$beam_width, topk=$topk) on first $max_problems problems..."
python vllm_kcbs_math.py \
  --model "$model" \
  --k "$beam_width" \
  --topk "$topk" \
  --max_problems "$max_problems" \
  --max_new_tokens "$max_new_tokens" \
  --save_str kcbs_results/ \
  --seed "$seed" \
  --debug

kcbs_csv="kcbs_results/${model}/${model}_vllm_kcbs_bw_${beam_width}_topk_${topk}_seed_${seed}.csv"

log "Merging and exporting side-by-side CSV with decoded generations..."
out_dir="compare_results/${model}"
mkdir -p "$out_dir"
out_csv="$out_dir/kcbs_vs_bestof${beam_width}_subset${max_problems}.csv"

python compare_kcbs_bestofn.py \
  --bestof_csv "$bestof_csv" \
  --kcbs_csv "$kcbs_csv" \
  --out_csv "$out_csv" \
  --k "$beam_width"

log "Done. Output at: $out_csv"

