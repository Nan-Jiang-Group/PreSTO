#!/bin/bash

set -e
set -o pipefail

# Debug runner for KCBS vs Best-of-N on a tiny subset.
# Defaults: model=qwen_math, k=2, topk=20, max_problems=500, max_new_tokens=4096

log() { echo "[$(date '+%H:%M:%S')] $1"; }

export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd reasoning-with-sampling/llm_experiments

model="${MODEL:-qwen_math}"
beam_width="${K:-2}"
topk="${TOPK:-20}"
max_problems="${MAX_PROBLEMS:-500}"
max_new_tokens="${MAX_NEW_TOKENS:-4096}"
seed="${SEED:-0}"
# Don't read system TEMP (often a temp dir). Use BESTOF_TEMP if provided.
sampling_temp="${BESTOF_TEMP:-1.0}"

log "Debug run: model=$model, k=$beam_width, topk=$topk, max_problems=$max_problems"

# KCBS
out_root_kcbs="kcbs_debug_results"
mkdir -p "$out_root_kcbs"

if [[ "${DEBUG_STEPS:-0}" != "0" ]]; then
  kcbs_debug_flag="--debug"
  log "Running KCBS (k=$beam_width, topk=$topk) with inner-loop tqdm (DEBUG_STEPS=1)"
else
  kcbs_debug_flag=""
  log "Running KCBS (k=$beam_width, topk=$topk) with overall tqdm only (set DEBUG_STEPS=1 for step bar)"
fi
python vllm_kcbs_math.py \
  --model "$model" \
  --k "$beam_width" \
  --topk "$topk" \
  --max_problems "$max_problems" \
  --max_new_tokens "$max_new_tokens" \
  --save_str "$out_root_kcbs/" \
  --seed "$seed" \
  --quiet_vllm \
  ${kcbs_debug_flag}

kcbs_csv="$out_root_kcbs/${model}/${model}_vllm_kcbs_bw_${beam_width}_topk_${topk}_seed_${seed}.csv"
kcbs_plot="$out_root_kcbs/${model}/${model}_vllm_kcbs_bw_${beam_width}_topk_${topk}_plot.png"

if [[ -f "$kcbs_csv" ]]; then
  log "Plotting KCBS pass@k from $kcbs_csv"
  python eval_vllm_passk.py "$kcbs_csv" --plot --output_plot "$kcbs_plot"
  log "Saved KCBS plot to $kcbs_plot"
else
  log "KCBS CSV not found: $kcbs_csv"
fi

# Best-of-N sampling for side-by-side sanity check
out_root_bo="sampling_debug_results"
mkdir -p "$out_root_bo"

log "Running Best-of-$beam_width sampling (temp=$sampling_temp) on the same subset..."
python vllm_passk_math.py \
  --model "$model" \
  --temperature "$sampling_temp" \
  --k "$beam_width" \
  --max_problems "$max_problems" \
  --save_str "$out_root_bo/" \
  --seed "$seed" \
  --quiet_vllm

bestof_csv="$out_root_bo/${model}/${model}_vllm_passk_${beam_width}_temp_${sampling_temp}_seed_${seed}.csv"

# Side-by-side comparison CSV (fully quoted for commas/newlines)
compare_dir="compare_debug_results/${model}"
mkdir -p "$compare_dir"
compare_csv="$compare_dir/kcbs_vs_bestof${beam_width}_subset${max_problems}.csv"

if [[ -f "$kcbs_csv" && -f "$bestof_csv" ]]; then
  log "Building side-by-side CSV at $compare_csv"
  python compare_kcbs_bestofn.py \
    --bestof_csv "$bestof_csv" \
    --kcbs_csv "$kcbs_csv" \
    --out_csv "$compare_csv" \
    --k "$beam_width"
  log "Wrote: $compare_csv"
else
  log "Skipping compare; missing input CSV(s)."
fi

log "Debug run complete."
