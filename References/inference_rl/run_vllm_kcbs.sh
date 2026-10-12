#!/bin/bash

set -e
set -o pipefail

log_info() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"; }
log_error() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2; }
log_success() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $1"; }
log_warning() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1"; }
log_separator() { printf '=%.0s' {1..80}; echo; }

log_separator
log_info "VLLM TOP-K CONSTRAINED BEAM SEARCH (k-CBS) RUN STARTED"
log_separator

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MPLBACKEND=Agg
export DISPLAY=""
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING}
export VLLM_DISABLE_LOG_STATS=${VLLM_DISABLE_LOG_STATS:-1}
export DEBUG_STEPS=1

if nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | grep -E "(MiB|%)" || nvidia-smi
else
  log_warning "nvidia-smi unavailable; continuing without GPU stats"
fi

start_time=$(date +%s)

log_info "Changing to experiments directory..."
cd reasoning-with-sampling/llm_experiments
log_success "Working dir: $(pwd)"

models=("qwen_math")
k_values=(10)          # number of sequences (beam width)
topk_values=(20)       # per-step top-k next-token logprobs
length_penalties=(1.0)
max_new_tokens=3072    # match best-of-N script default

save_root="kcbs_results"
mkdir -p "$save_root"
log_success "Results directory ready: $(pwd)/$save_root"

total_configs=$((${#models[@]} * ${#k_values[@]} * ${#topk_values[@]} * ${#length_penalties[@]}))
config_count=0
successful_configs=0
failed_configs=0
declare -a config_results

for model in "${models[@]}"; do
  for k in "${k_values[@]}"; do
    for topk in "${topk_values[@]}"; do
      for lenpen in "${length_penalties[@]}"; do
        config_count=$((config_count + 1))
        log_separator
      log_info "CONFIGURATION $config_count/$total_configs"
      log_info "Model=$model, beam_width(k)=$k, topk=$topk, length_penalty=$lenpen"
        log_separator

      if python vllm_kcbs_math.py \
        --model "$model" \
        --k "$k" \
        --topk "$topk" \
        --length_penalty "$lenpen" \
        --max_new_tokens "$max_new_tokens" \
        --quiet_vllm \
        --save_str "$save_root/" \
        --seed 0; then
        python_exit=0
        log_success "k-CBS run completed"
      else
        python_exit=$?
        failed_configs=$((failed_configs + 1))
        config_results+=("FAILED: $model k=$k topk=$topk lenpen=$lenpen (exit $python_exit)")
        log_error "k-CBS run failed"
        continue
      fi

      csv_file="$save_root/${model}/${model}_vllm_kcbs_bw_${k}_topk_${topk}_seed_0.csv"
      plot_file="$save_root/${model}/${model}_vllm_kcbs_bw_${k}_topk_${topk}_plot.png"

      if [[ -f "$csv_file" ]]; then
        log_info "CSV located: $csv_file"
        if python eval_vllm_passk.py "$csv_file" --plot --output_plot "$plot_file"; then
          log_success "Plot saved: $plot_file"
        else
          log_warning "Plotting script failed"
        fi
      else
        log_warning "CSV missing: $csv_file"
      fi

      successful_configs=$((successful_configs + 1))
      config_results+=("SUCCESS: $model k=$k topk=$topk lenpen=$lenpen")
      log_info "Progress: $successful_configs success, $failed_configs failed"
      done
    done
  done
done

log_separator
log_info "RESULTS SUMMARY"
log_separator
log_info "Total configs: $total_configs"
log_info "Successful: $successful_configs"
log_info "Failed: $failed_configs"

if (( successful_configs > 0 )); then
  log_success "k-CBS results saved under $(pwd)/$save_root"
fi

if nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | grep -E "(MiB|%)" || nvidia-smi
fi

elapsed=$(( $(date +%s) - start_time ))
log_info "Total execution time: ${elapsed}s"

if (( failed_configs == 0 )); then
  log_success "ALL k-CBS RUNS COMPLETED"
else
  log_error "k-CBS RUNS COMPLETED WITH FAILURES"
fi
