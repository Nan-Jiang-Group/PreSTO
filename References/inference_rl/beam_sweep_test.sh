#!/bin/bash

# Test script for beam search sweep
cd reasoning-with-sampling/llm_experiments

echo "Testing beam search sweep script..."

# Run a small test with limited configurations
python beam_search_sweep.py \
    --model qwen_math \
    --num_problems 2 \
    --beam_widths 3 5 \
    --max_tokens_list 1024 2048 \
    --gpu_memory_utils 0.6 0.7 \
    --max_model_lens 2048 \
    --max_batched_tokens 1024 \
    --seed 42 \
    2>&1 | tee beam_sweep_test.log

echo "Test complete. Check beam_sweep_test.log for output."