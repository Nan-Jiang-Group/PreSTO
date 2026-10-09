#!/bin/bash

# Run MCMC sampling experiments
# Usage: ./run_mcmc_sampling.sh

set -e  # Exit on any error

# Set headless mode for matplotlib
export MPLBACKEND=Agg
export DISPLAY=""

# Record start time
start_time=$(date +%s)

echo "Starting MCMC sampling experiments..."

# Change to the experiments directory
cd reasoning-with-sampling/llm_experiments

# MCMC experiment parameters  
models=("qwen_math")
temperatures=(1.0)  # Match vLLM temperature
seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)  # 16 seeds for pass@16 comparison
batch_indices=(0 1 2 3 4)  # 5 batches of 100 problems each = 500 total

# Create results directory
mkdir -p mcmc_results

for model in "${models[@]}"; do
    for temp in "${temperatures[@]}"; do
        echo "Running MCMC for $model with temperature=$temp (for pass@16 comparison)"
        
        for seed in "${seeds[@]}"; do
            echo "  Seed: $seed"
            
            for batch_idx in "${batch_indices[@]}"; do
                echo "    Batch: $batch_idx (problems $((batch_idx*100))-$((batch_idx*100+99)))"
                
                # Run MCMC sampling (generates std, naive_temp, and mcmc completions)
                python power_samp_math.py \
                    --model "$model" \
                    --temperature "$temp" \
                    --dataset "MATH" \
                    --save_str "mcmc_results/" \
                    --seed "$seed" \
                    --batch_idx "$batch_idx"
                
            done
            echo "    Completed all batches for seed $seed"
        done
        
        # After all seeds are done, evaluate pass@k
        results_folder="mcmc_results/${model}"
        if [ -d "$results_folder" ]; then
            echo "Evaluating pass@16 for $model..."
            
            # Run pass@k evaluation (using original passk script)
            python passk_math.py "$results_folder"
            
            echo "Pass@16 evaluation completed for $model"
        fi
        
        echo "Completed: $model with temp=$temp"
        echo "----------------------------------------"
    done
done

echo "All MCMC experiments completed!"
echo "Results saved in: $(pwd)/mcmc_results/"

# Calculate and display elapsed time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

hours=$((elapsed_time / 3600))
minutes=$(((elapsed_time % 3600) / 60))
seconds=$((elapsed_time % 60))

echo ""
echo "=== TIMING SUMMARY ==="
if [ $hours -gt 0 ]; then
    echo "Total time: ${hours}h ${minutes}m ${seconds}s"
elif [ $minutes -gt 0 ]; then
    echo "Total time: ${minutes}m ${seconds}s"
else
    echo "Total time: ${seconds}s"
fi
echo "======================="