#!/bin/bash

# Test version of beam search config sweep - only tests a few configs

set -e

# Limited test parameters
BEAM_WIDTHS=(3 5)
MAX_TOKENS_LIST=(1024 2048)  
GPU_MEMORY_UTILS=(0.6 0.7)
MAX_MODEL_LENS=(2048)
MAX_BATCHED_TOKENS=(1024)

# Fixed parameters
MODEL="qwen_math"
NUM_PROBLEMS=2  # Very small for testing
SEED=42
SAVE_DIR="sweep_test_results"

# Create results directory
mkdir -p $SAVE_DIR

# Log file for the sweep
LOG_FILE="$SAVE_DIR/beam_sweep_test_$(date +%Y%m%d_%H%M%S).log"

echo "=== BEAM SEARCH CONFIG SWEEP TEST ===" | tee $LOG_FILE
echo "Start time: $(date)" | tee -a $LOG_FILE
echo "Model: $MODEL" | tee -a $LOG_FILE
echo "Problems per config: $NUM_PROBLEMS" | tee -a $LOG_FILE
echo "=======================================" | tee -a $LOG_FILE

# Calculate total configurations
TOTAL_CONFIGS=$((${#BEAM_WIDTHS[@]} * ${#MAX_TOKENS_LIST[@]} * ${#GPU_MEMORY_UTILS[@]} * ${#MAX_MODEL_LENS[@]} * ${#MAX_BATCHED_TOKENS[@]}))
echo "Total test configurations: $TOTAL_CONFIGS" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Initialize counters
CONFIG_NUM=0
SUCCESS_COUNT=0
FAIL_COUNT=0

# Results summary file
SUMMARY_FILE="$SAVE_DIR/beam_sweep_test_summary.csv"
echo "config_id,beam_width,max_tokens,gpu_memory_util,max_model_len,max_batched_tokens,status,error_type,timestamp" > $SUMMARY_FILE

# Test one configuration
run_test_config() {
    local beam_width=$1
    local max_tokens=$2
    local gpu_memory_util=$3
    local max_model_len=$4
    local max_batched_tokens=$5
    local config_id=$6
    
    local config_str="beam_width=${beam_width}, max_tokens=${max_tokens}, gpu_memory_util=${gpu_memory_util}, max_model_len=${max_model_len}, max_batched_tokens=${max_batched_tokens}"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] TEST CONFIG $config_id/$TOTAL_CONFIGS: $config_str" | tee -a $LOG_FILE
    
    # Create unique output directory for this config
    local config_save_dir="$SAVE_DIR/config_${config_id}"
    mkdir -p $config_save_dir
    
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
        --save_str $config_save_dir"
    
    # Run in reasoning-with-sampling/llm_experiments directory
    cd reasoning-with-sampling/llm_experiments
    
    # Capture output and check for success/failure
    local output_file="$config_save_dir/output.log"
    local error_file="$config_save_dir/error.log"
    local status="UNKNOWN"
    local error_type=""
    
    if timeout 300 $python_cmd > $output_file 2> $error_file; then
        if grep -q "FINAL RESULTS" $output_file; then
            status="SUCCESS"
            echo "  ✓ SUCCESS" | tee -a $LOG_FILE
            ((SUCCESS_COUNT++))
        else
            status="PARTIAL_FAILURE"
            error_type="INCOMPLETE_EXECUTION"
            echo "  ⚠ PARTIAL FAILURE" | tee -a $LOG_FILE
            ((FAIL_COUNT++))
        fi
    else
        status="FAILURE"
        if grep -qi "cuda.*out of memory\|CUDA error" $error_file; then
            error_type="CUDA_OOM"
        elif grep -qi "cuda.*error" $error_file; then
            error_type="CUDA_ERROR"
        else
            error_type="OTHER_ERROR"
        fi
        
        echo "  ✗ FAILURE - Error type: $error_type" | tee -a $LOG_FILE
        ((FAIL_COUNT++))
    fi
    
    # Log to summary CSV
    echo "$config_id,$beam_width,$max_tokens,$gpu_memory_util,$max_model_len,$max_batched_tokens,$status,$error_type,$timestamp" >> $SUMMARY_FILE
    
    # Return to original directory
    cd - > /dev/null
    
    return 0
}

# Main test loop
echo "Starting test sweep..." | tee -a $LOG_FILE

for beam_width in "${BEAM_WIDTHS[@]}"; do
    for max_tokens in "${MAX_TOKENS_LIST[@]}"; do
        for gpu_memory_util in "${GPU_MEMORY_UTILS[@]}"; do
            for max_model_len in "${MAX_MODEL_LENS[@]}"; do
                for max_batched_tokens in "${MAX_BATCHED_TOKENS[@]}"; do
                    ((CONFIG_NUM++))
                    
                    run_test_config $beam_width $max_tokens $gpu_memory_util $max_model_len $max_batched_tokens $CONFIG_NUM
                    
                    echo "Progress: $CONFIG_NUM/$TOTAL_CONFIGS complete" | tee -a $LOG_FILE
                    echo "" | tee -a $LOG_FILE
                    
                done
            done
        done
    done
done

echo "=======================================" | tee -a $LOG_FILE
echo "TEST COMPLETE" | tee -a $LOG_FILE
echo "End time: $(date)" | tee -a $LOG_FILE
echo "Success: $SUCCESS_COUNT, Failed: $FAIL_COUNT" | tee -a $LOG_FILE
echo "=======================================" | tee -a $LOG_FILE

echo ""
echo "Test complete! Check $LOG_FILE and $SUMMARY_FILE for results."