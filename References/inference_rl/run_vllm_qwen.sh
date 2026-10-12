#!/bin/bash

# Run vLLM inference with Qwen2.5 models and generate plots
# Usage: ./run_vllm_qwen.sh

set -e  # Exit on any error

# Set headless mode for matplotlib
export MPLBACKEND=Agg
export DISPLAY=""

# Record start time
start_time=$(date +%s)

echo "Starting vLLM inference with Qwen2.5 models..."

# Change to the experiments directory
cd reasoning-with-sampling/llm_experiments

# Models to test
models=("qwen_math")
temperatures=(1.0)
k_values=(16)

# Create results directory
mkdir -p vllm_results

for model in "${models[@]}"; do
    for temp in "${temperatures[@]}"; do
        for k in "${k_values[@]}"; do
            echo "Running $model with temperature=$temp, k=$k"
            
            # Run vLLM inference
            python vllm_passk_math.py \
                --model "$model" \
                --k "$k" \
                --temperature "$temp" \
                --save_str "vllm_results/" \
                --seed 0
            
            # Generate plot
            csv_file="vllm_results/${model}/${model}_vllm_passk_${k}_temp_${temp}_seed_0.csv"
            plot_file="vllm_results/${model}/${model}_vllm_passk_${k}_temp_${temp}_plot.png"
            
            if [ -f "$csv_file" ]; then
                echo "Generating plot for $csv_file"
                python eval_vllm_passk.py "$csv_file" --plot --output_plot "$plot_file"
                echo "Plot saved to: $plot_file"
            else
                echo "Warning: CSV file not found: $csv_file"
            fi
            
            echo "Completed: $model, temp=$temp, k=$k"
            echo "----------------------------------------"
        done
    done
done

echo "All vLLM experiments completed!"
echo "Results saved in: $(pwd)/vllm_results/"

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