#!/bin/bash

# Run vLLM beam search inference with Qwen2.5 models and generate plots
# Usage: ./run_vllm_beam.sh

set -e  # Exit on any error
set -o pipefail  # Fail if any command in a pipeline fails

# Logging functions
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
log_info "VLLM BEAM SEARCH RUN STARTED"
log_separator

# Set up debugging environment variables
log_info "Setting up debugging environment variables..."
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Set headless mode for matplotlib
export MPLBACKEND=Agg
export DISPLAY=""

log_separator
log_info "ENVIRONMENT VARIABLES CONFIGURED"
log_separator
log_info "CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
log_info "TORCH_USE_CUDA_DSA: $TORCH_USE_CUDA_DSA"
log_info "PYTHONUNBUFFERED: $PYTHONUNBUFFERED"
log_info "MPLBACKEND: $MPLBACKEND"
log_separator

# Check CUDA availability
log_info "Checking CUDA availability..."
if nvidia-smi > /dev/null 2>&1; then
    log_success "CUDA/nvidia-smi available"
    nvidia-smi | grep -E "(MiB|%)" || nvidia-smi
else
    log_error "nvidia-smi not available!"
fi
log_separator

# Record start time
start_time=$(date +%s)

# Change to the experiments directory
log_info "Changing to experiments directory..."
if cd reasoning-with-sampling/llm_experiments; then
    log_success "Changed to $(pwd)"
else
    log_error "Failed to change to experiments directory"
    exit 1
fi

# Models to test
models=("qwen_math")
temperatures=(1.0)
k_values=(10)  # Limited to 10 due to vLLM logprobs constraints

# Calculate total configurations
total_configs=$((${#models[@]} * ${#temperatures[@]} * ${#k_values[@]}))
config_count=0
successful_configs=0
failed_configs=0

log_separator
log_info "CONFIGURATION"
log_separator
log_info "Models: ${models[*]}"
log_info "Temperatures: ${temperatures[*]}"
log_info "Beam widths (k): ${k_values[*]}"
log_info "Total configurations to run: $total_configs"
log_separator

# Create results directory
log_info "Creating results directory..."
if mkdir -p beam_results; then
    log_success "Results directory ready: $(pwd)/beam_results"
else
    log_error "Failed to create results directory"
    exit 1
fi
log_separator

# Track configuration results
declare -a config_results

for model in "${models[@]}"; do
    for temp in "${temperatures[@]}"; do
        for k in "${k_values[@]}"; do
            config_count=$((config_count + 1))
            config_start_time=$(date +%s)
            
            log_separator
            log_info "CONFIGURATION $config_count/$total_configs"
            log_separator
            log_info "Model: $model"
            log_info "Temperature: $temp"
            log_info "Beam width (k): $k"
            log_separator
            
            # Run vLLM beam search inference
            log_info "Starting vLLM beam search inference..."
            if python vllm_beam_math.py \
                --model "$model" \
                --k "$k" \
                --temperature "$temp" \
                --save_str "beam_results/" \
                --seed 0; then
                
                python_exit_code=0
                log_success "Beam search inference completed successfully"
            else
                python_exit_code=$?
                log_error "Beam search inference failed with exit code: $python_exit_code"
                failed_configs=$((failed_configs + 1))
                config_results+=("FAILED: $model, temp=$temp, k=$k (exit code: $python_exit_code)")
                continue
            fi
            
            # Generate plot (adapt existing eval script for beam results)
            # Note: actual beam width may be limited to 10
            csv_file="beam_results/${model}/${model}_vllm_beam_10_temp_${temp}_seed_0.csv"
            plot_file="beam_results/${model}/${model}_vllm_beam_10_temp_${temp}_plot.png"
            
            if [ -f "$csv_file" ]; then
                log_info "CSV file found: $csv_file"
                log_info "Generating plot..."
                
                if python eval_vllm_passk.py "$csv_file" --plot --output_plot "$plot_file"; then
                    log_success "Plot saved to: $plot_file"
                else
                    log_error "Failed to generate plot"
                fi
            else
                log_warning "CSV file not found: $csv_file"
                log_warning "Skipping plot generation for this configuration"
            fi
            
            config_end_time=$(date +%s)
            config_elapsed=$((config_end_time - config_start_time))
            
            successful_configs=$((successful_configs + 1))
            config_results+=("SUCCESS: $model, temp=$temp, k=$k (${config_elapsed}s)")
            
            log_separator
            if [ $python_exit_code -eq 0 ]; then
                log_success "Configuration $config_count/$total_configs completed successfully in ${config_elapsed}s"
            else
                log_error "Configuration $config_count/$total_configs failed"
            fi
            log_separator
            
            log_info "Progress: $successful_configs successful, $failed_configs failed, $((total_configs - config_count)) remaining"
        done
    done
done

# Calculate and display elapsed time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
hours=$((elapsed_time / 3600))
minutes=$(((elapsed_time % 3600) / 60))
seconds=$((elapsed_time % 60))

log_separator
log_info "TIMING SUMMARY"
log_separator
if [ $hours -gt 0 ]; then
    log_info "Total execution time: ${hours}h ${minutes}m ${seconds}s"
elif [ $minutes -gt 0 ]; then
    log_info "Total execution time: ${minutes}m ${seconds}s"
else
    log_info "Total execution time: ${seconds}s"
fi
log_separator

# Results summary
log_separator
log_info "RESULTS SUMMARY"
log_separator
log_info "Total configurations: $total_configs"
log_info "Successful: $successful_configs"
log_info "Failed: $failed_configs"

if [ $successful_configs -gt 0 ]; then
    log_success "Results saved in: $(pwd)/beam_results/"
    
    # Show result files
    result_files=$(find beam_results -name "*.csv" -o -name "*.png" 2>/dev/null | wc -l)
    if [ $result_files -gt 0 ]; then
        log_info "Generated result files: $result_files"
        log_info "CSV files:"
        find beam_results -name "*.csv" 2>/dev/null | while read -r file; do
            log_info "  - $file"
        done
        log_info "Plot files:"
        find beam_results -name "*.png" 2>/dev/null | while read -r file; do
            log_info "  - $file"
        done
    fi
fi

if [ $failed_configs -gt 0 ]; then
    log_error "Failed configurations:"
    for result in "${config_results[@]}"; do
        if [[ $result == FAILED:* ]]; then
            log_error "  $result"
        fi
    done
fi
log_separator

# Post-run GPU memory status
log_separator
log_info "POST-RUN GPU MEMORY STATUS"
log_separator
if nvidia-smi > /dev/null 2>&1; then
    nvidia-smi | grep -E "(MiB|%)" || nvidia-smi
else
    log_error "nvidia-smi not available for final memory check"
fi
log_separator

# Final status
if [ $failed_configs -eq 0 ]; then
    log_success "ALL VLLM BEAM SEARCH EXPERIMENTS COMPLETED SUCCESSFULLY"
    exit 0
else
    log_error "VLLM BEAM SEARCH EXPERIMENTS COMPLETED WITH FAILURES"
    exit 1
fi