#!/bin/bash

set -e
set -o pipefail

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $1"
}

log_warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1"
}

log_separator() {
    echo "$(printf '=%.0s' {1..80})"
}

log_separator
log_info "VLLM BEST-OF-N SAMPLING RUN STARTED"
log_separator

log_info "Setting up environment variables..."
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MPLBACKEND=Agg
export DISPLAY=""
log_success "Environment configured"
log_separator

if ! nvidia-smi >/dev/null 2>&1; then
    log_warning "nvidia-smi unavailable; continuing without GPU stats"
else
    nvidia-smi | grep -E "(MiB|%)" || nvidia-smi
fi
log_separator

start_time=$(date +%s)

log_info "Changing to experiments directory..."
cd reasoning-with-sampling/llm_experiments
log_success "Working dir: $(pwd)"

models=("qwen_math")
temperatures=(0.8)
k_values=(10)

log_separator
log_info "CONFIGURATION"
log_separator
log_info "Models: ${models[*]}"
log_info "Temperatures: ${temperatures[*]}"
log_info "Samples per problem (k): ${k_values[*]}"
log_separator

save_root="sampling_results"
mkdir -p "$save_root"
log_success "Results directory ready: $(pwd)/$save_root"

config_count=0
successful_configs=0
failed_configs=0
declare -a config_results

total_configs=$((${#models[@]} * ${#temperatures[@]} * ${#k_values[@]}))

for model in "${models[@]}"; do
    for temp in "${temperatures[@]}"; do
        for k in "${k_values[@]}"; do
            config_count=$((config_count + 1))
            config_start=$(date +%s)
            log_separator
            log_info "CONFIGURATION $config_count/$total_configs"
            log_info "Model=$model, Temp=$temp, K=$k"
            log_separator

            if python vllm_passk_math.py \
                --model "$model" \
                --temperature "$temp" \
                --k "$k" \
                --save_str "$save_root/" \
                --seed 0; then
                python_exit=0
                log_success "Sampling run completed"
            else
                python_exit=$?
                failed_configs=$((failed_configs + 1))
                config_results+=("FAILED: $model temp=$temp k=$k (exit $python_exit)")
                log_error "Sampling run failed"
                continue
            fi

            csv_file="$save_root/${model}/${model}_vllm_passk_${k}_temp_${temp}_seed_0.csv"
            plot_file="$save_root/${model}/${model}_vllm_passk_${k}_temp_${temp}_plot.png"

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

            config_elapsed=$(( $(date +%s) - config_start ))
            successful_configs=$((successful_configs + 1))
            config_results+=("SUCCESS: $model temp=$temp k=$k (${config_elapsed}s)")
            log_info "Config finished in ${config_elapsed}s"
            log_info "Progress: $successful_configs success, $failed_configs failed"
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
    log_success "Sampling results saved under $(pwd)/$save_root"
fi

log_separator
log_info "POST-RUN GPU MEMORY STATUS"
log_separator
if nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | grep -E "(MiB|%)" || nvidia-smi
fi

if (( failed_configs == 0 )); then
    log_success "ALL VLLM BEST-OF-N SAMPLING RUNS COMPLETED"
else
    log_error "BEST-OF-N SAMPLING RUNS COMPLETED WITH FAILURES"
fi

end_time=$(date +%s)
elapsed=$((end_time - start_time))
log_info "Total execution time: ${elapsed}s"
