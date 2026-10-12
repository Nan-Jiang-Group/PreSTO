#!/bin/bash

# Debug script for beam search with comprehensive CUDA debugging
# Usage: ./beam_search_debug.sh [num_problems] [beam_width] [gpu_memory]

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

log_separator() {
    echo "$(printf '=%.0s' {1..80})"
}

# Parameters with defaults
NUM_PROBLEMS=${1:-500}
BEAM_WIDTH=${2:-5}
GPU_MEMORY=${3:-0.8}

log_separator
log_info "BEAM SEARCH DEBUG SCRIPT STARTED"
log_separator
log_info "Configuration:"
log_info "  Number of problems: $NUM_PROBLEMS"
log_info "  Beam width: $BEAM_WIDTH"
log_info "  GPU memory utilization: $GPU_MEMORY"
log_separator

# Set comprehensive debugging environment variables
log_info "Setting up debugging environment variables..."
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Memory debugging
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_MEMORY_FRACTION=0.7

# Additional vLLM debugging
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export NCCL_DEBUG=INFO

# Set headless mode for matplotlib
export MPLBACKEND=Agg
export DISPLAY=""

# Record start time
start_time=$(date +%s)

log_separator
log_info "ENVIRONMENT VARIABLES CONFIGURED"
log_separator
log_info "CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
log_info "TORCH_USE_CUDA_DSA: $TORCH_USE_CUDA_DSA"
log_info "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
log_info "CUDA_MEMORY_FRACTION: $CUDA_MEMORY_FRACTION"
log_info "VLLM_LOGGING_LEVEL: $VLLM_LOGGING_LEVEL"
log_separator

# Check CUDA availability
log_info "Checking CUDA availability..."
if nvidia-smi > /dev/null 2>&1; then
    log_success "CUDA/nvidia-smi available"
    nvidia-smi
else
    log_error "nvidia-smi not available!"
fi

# Change to the experiments directory
log_info "Changing to experiments directory..."
if cd reasoning-with-sampling/llm_experiments; then
    log_success "Changed to $(pwd)"
else
    log_error "Failed to change to experiments directory"
    exit 1
fi

# Create debug results directory
log_info "Creating debug results directory..."
if mkdir -p debug_results; then
    log_success "Debug results directory ready: $(pwd)/debug_results"
else
    log_error "Failed to create debug results directory"
    exit 1
fi

# Setup logging
LOG_FILE="debug_results/beam_debug_$(date +%Y%m%d_%H%M%S).log"
log_separator
log_info "STARTING BEAM SEARCH DEBUG RUN"
log_separator
log_info "Processing $NUM_PROBLEMS problems with beam width $BEAM_WIDTH"
log_info "All output will be logged to: $LOG_FILE"
log_separator

# Run the debug script with comprehensive logging
{
    echo "=== PYTHON SCRIPT OUTPUT BEGINS ==="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Python beam_search_debug.py"
    python beam_search_debug.py \
        --model "qwen_math" \
        --k "$BEAM_WIDTH" \
        --temperature 1.0 \
        --save_str "debug_results/" \
        --seed 0 \
        --num_problems "$NUM_PROBLEMS" \
        --gpu_memory "$GPU_MEMORY"
    python_exit_code=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Python script finished with exit code: $python_exit_code"
    echo "=== PYTHON SCRIPT OUTPUT ENDS ==="
    exit $python_exit_code
} 2>&1 | tee "$LOG_FILE"

exit_code=${PIPESTATUS[0]}

log_separator
if [ $exit_code -eq 0 ]; then
    log_success "DEBUG RUN COMPLETED SUCCESSFULLY"
else
    log_error "DEBUG RUN FAILED (exit code: $exit_code)"
fi
log_separator

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

# Show final results
log_separator
log_info "RESULTS SUMMARY"
log_separator

if [ $exit_code -eq 0 ]; then
    log_success "Beam search debug completed successfully!"
    log_info "Results location: $(pwd)/debug_results/"
    log_info "Log file: $LOG_FILE"
    
    # Show result files
    result_files=$(ls debug_results/ 2>/dev/null | grep -E "\.(csv|json)$" | wc -l)
    if [ $result_files -gt 0 ]; then
        log_info "Generated result files:"
        ls -la debug_results/ | grep -E "\.(csv|json)$" | while read -r line; do
            log_info "  $line"
        done
    else
        log_error "No result files (csv/json) found in debug_results/"
    fi
else
    log_error "Beam search debug FAILED with exit code: $exit_code"
    log_error "Check the log file for detailed error information: $LOG_FILE"
    log_error "Common issues to check:"
    log_error "  - CUDA/GPU memory issues"
    log_error "  - Model loading failures" 
    log_error "  - vLLM configuration problems"
    log_error "  - Python environment issues"
fi

log_separator
log_info "POST-RUN GPU MEMORY STATUS"
log_separator
if nvidia-smi > /dev/null 2>&1; then
    nvidia-smi | grep -E "(MiB|%)"
else
    log_error "nvidia-smi not available for final memory check"
fi
log_separator

# Final status
if [ $exit_code -eq 0 ]; then
    log_success "BEAM SEARCH DEBUG SESSION COMPLETED SUCCESSFULLY"
else
    log_error "BEAM SEARCH DEBUG SESSION FAILED - CHECK LOGS"
fi
log_separator

exit $exit_code