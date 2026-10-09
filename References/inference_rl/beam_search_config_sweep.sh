#!/bin/bash

# Beam search configuration sweep script
# This script launches separate Python processes for each config to avoid CUDA context corruption

set +e

# Default parameters - modify these arrays to change sweep ranges
BEAM_WIDTHS=(3 5)
MAX_TOKENS_LIST=(1024 2048)
GPU_MEMORY_UTILS=(0.6 0.7)
MAX_MODEL_LENS=(2048)
MAX_BATCHED_TOKENS=(1024)

# Fixed parameters
MODEL="qwen_math"
NUM_PROBLEMS=1
SEED=42
SAVE_DIR="sweep_results"
TIMEOUT=60

# Create results directory
mkdir -p $SAVE_DIR

# Log file for the sweep
LOG_FILE="$SAVE_DIR/beam_sweep_$(date +%Y%m%d_%H%M%S).log"

echo "=== BEAM SEARCH CONFIGURATION SWEEP ===" | tee $LOG_FILE
echo "Start time: $(date)" | tee -a $LOG_FILE
echo "Model: $MODEL" | tee -a $LOG_FILE
echo "Problems per config: $NUM_PROBLEMS" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "============================================" | tee -a $LOG_FILE

# Calculate total configurations
TOTAL_CONFIGS=$((${#BEAM_WIDTHS[@]} * ${#MAX_TOKENS_LIST[@]} * ${#GPU_MEMORY_UTILS[@]} * ${#MAX_MODEL_LENS[@]} * ${#MAX_BATCHED_TOKENS[@]}))
echo "Total configurations to test: $TOTAL_CONFIGS" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Initialize counters
CONFIG_NUM=0
SUCCESS_COUNT=0
FAIL_COUNT=0

# Results summary file
SUMMARY_FILE="$SAVE_DIR/beam_sweep_summary_$(date +%Y%m%d_%H%M%S).csv"
echo "config_id,beam_width,max_tokens,gpu_memory_util,max_model_len,max_batched_tokens,status,error_type,pass_at_k_accuracy,timestamp" > $SUMMARY_FILE

# Function to run a single configuration
run_config() {
    local beam_width=$1
    local max_tokens=$2
    local gpu_memory_util=$3
    local max_model_len=$4
    local max_batched_tokens=$5
    local config_id=$6
    
    local config_str="beam_width=${beam_width}, max_tokens=${max_tokens}, gpu_memory_util=${gpu_memory_util}, max_model_len=${max_model_len}, max_batched_tokens=${max_batched_tokens}"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] CONFIG $config_id/$TOTAL_CONFIGS: $config_str" | tee -a $LOG_FILE
    
    # Create unique output directory for this config
    local config_save_dir="$SAVE_DIR/config_${config_id}"
    mkdir -p $config_save_dir
    
    # Check if we can access the experiments directory
    if ! cd reasoning-with-sampling/llm_experiments 2>/dev/null; then
        echo "  ✗ FAILURE - Cannot find reasoning-with-sampling/llm_experiments directory" | tee -a $LOG_FILE
        return 1
    fi
    
    # Run the beam search debug script with this configuration
    local python_cmd="python beam_search_debug.py \
        --model $MODEL \
        --k $beam_width \
        --max_tokens $max_tokens \
        --gpu_memory $gpu_memory_util \
        --max_model_len $max_model_len \
        --max_num_batched_tokens $max_batched_tokens \
        --num_problems $NUM_PROBLEMS \
        --seed $SEED \
        --save_str ../../$config_save_dir"
    
    # Capture output and check for success/failure
    local output_file="../../$config_save_dir/output.log"
    local error_file="../../$config_save_dir/error.log"
    local status="UNKNOWN"
    local error_type=""
    local pass_at_k_accuracy=""
    
    echo "    Running: $python_cmd" | tee -a ../../$LOG_FILE
    echo "    Working directory: $(pwd)" | tee -a ../../$LOG_FILE
    
    if timeout $TIMEOUT bash -c "$python_cmd" > $output_file 2> $error_file; then
        # Check if the script completed successfully
        if grep -q "FINAL RESULTS" $output_file && grep -q "Pass@" $output_file; then
            status="SUCCESS"
            # Extract pass@k accuracy from output
            pass_at_k_accuracy=$(grep "Overall Pass@" $output_file | grep -oP 'Accuracy: \K[0-9.]+' || echo "")
            echo "  ✓ SUCCESS - Pass@k accuracy: $pass_at_k_accuracy" | tee -a ../../$LOG_FILE
            ((SUCCESS_COUNT++))
        else
            status="PARTIAL_FAILURE"
            error_type="INCOMPLETE_EXECUTION"
            echo "  ⚠ PARTIAL FAILURE - Script ran but didn't complete properly" | tee -a ../../$LOG_FILE
            ((FAIL_COUNT++))
        fi
    else
        status="FAILURE"
        # Determine error type from stderr
        if grep -qi "cuda.*out of memory\|CUDA error" $error_file; then
            error_type="CUDA_OOM"
        elif grep -qi "cuda.*error\|cuda.*exception" $error_file; then
            error_type="CUDA_ERROR"
        elif grep -qi "timeout\|killed" $error_file; then
            error_type="TIMEOUT"
        else
            error_type="OTHER_ERROR"
        fi
        
        echo "  ✗ FAILURE - Error type: $error_type" | tee -a ../../$LOG_FILE
        
        # Show first few lines of error for debugging
        echo "    Error preview:" | tee -a ../../$LOG_FILE
        if [ -f $error_file ]; then
            head -3 $error_file | sed 's/^/    /' | tee -a ../../$LOG_FILE
        fi
        ((FAIL_COUNT++))
    fi
    
    # Log to summary CSV
    echo "$config_id,$beam_width,$max_tokens,$gpu_memory_util,$max_model_len,$max_batched_tokens,$status,$error_type,$pass_at_k_accuracy,$timestamp" >> ../../$SUMMARY_FILE
    
    # Return to original directory
    cd - > /dev/null
    
    echo "  Memory cleanup..." | tee -a $LOG_FILE
    # Force memory cleanup between configs
    sleep 2
    
    return 0
}

# Main sweep loop
echo "Starting configuration sweep..." | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

for beam_width in "${BEAM_WIDTHS[@]}"; do
    for max_tokens in "${MAX_TOKENS_LIST[@]}"; do
        for gpu_memory_util in "${GPU_MEMORY_UTILS[@]}"; do
            for max_model_len in "${MAX_MODEL_LENS[@]}"; do
                for max_batched_tokens in "${MAX_BATCHED_TOKENS[@]}"; do
                    ((CONFIG_NUM++))
                    
                    # Run this configuration
                    run_config $beam_width $max_tokens $gpu_memory_util $max_model_len $max_batched_tokens $CONFIG_NUM
                    
                    # Show progress
                    echo "Progress: $CONFIG_NUM/$TOTAL_CONFIGS complete (Success: $SUCCESS_COUNT, Failed: $FAIL_COUNT)" | tee -a $LOG_FILE
                    echo "" | tee -a $LOG_FILE
                    
                done
            done
        done
    done
done

# Final summary
echo "============================================" | tee -a $LOG_FILE
echo "SWEEP COMPLETE" | tee -a $LOG_FILE
echo "End time: $(date)" | tee -a $LOG_FILE
echo "Total configurations tested: $CONFIG_NUM" | tee -a $LOG_FILE
echo "Successful configurations: $SUCCESS_COUNT" | tee -a $LOG_FILE
echo "Failed configurations: $FAIL_COUNT" | tee -a $LOG_FILE
echo "Success rate: $(echo "scale=2; $SUCCESS_COUNT * 100 / $CONFIG_NUM" | bc)%" | tee -a $LOG_FILE
echo "Results summary: $SUMMARY_FILE" | tee -a $LOG_FILE
echo "Full log: $LOG_FILE" | tee -a $LOG_FILE
echo "============================================" | tee -a $LOG_FILE

# Analysis of failure patterns
echo "" | tee -a $LOG_FILE
echo "FAILURE ANALYSIS:" | tee -a $LOG_FILE
if [ $FAIL_COUNT -gt 0 ]; then
    echo "Failure types:" | tee -a $LOG_FILE
    cut -d',' -f8 $SUMMARY_FILE | sort | uniq -c | grep -v "error_type" | while read count error_type; do
        echo "  $error_type: $count configurations" | tee -a $LOG_FILE
    done
else
    echo "No failures detected!" | tee -a $LOG_FILE
fi

# Find boundary conditions
if [ $SUCCESS_COUNT -gt 0 ] && [ $FAIL_COUNT -gt 0 ]; then
    echo "" | tee -a $LOG_FILE
    echo "BOUNDARY ANALYSIS:" | tee -a $LOG_FILE
    echo "Highest successful beam width: $(grep "SUCCESS" $SUMMARY_FILE | cut -d',' -f2 | sort -n | tail -1)" | tee -a $LOG_FILE
    echo "Highest successful max_tokens: $(grep "SUCCESS" $SUMMARY_FILE | cut -d',' -f3 | sort -n | tail -1)" | tee -a $LOG_FILE
    echo "Highest successful GPU memory: $(grep "SUCCESS" $SUMMARY_FILE | cut -d',' -f4 | sort -n | tail -1)" | tee -a $LOG_FILE
    echo "Highest successful max_model_len: $(grep "SUCCESS" $SUMMARY_FILE | cut -d',' -f5 | sort -n | tail -1)" | tee -a $LOG_FILE
    echo "Highest successful max_batched_tokens: $(grep "SUCCESS" $SUMMARY_FILE | cut -d',' -f6 | sort -n | tail -1)" | tee -a $LOG_FILE
fi

echo ""
echo "Sweep complete! Check $LOG_FILE for full details and $SUMMARY_FILE for structured results."