#!/bin/bash

# Quick test script for beam search implementation
# Runs a single problem to verify everything works

echo "Quick Beam Search Test"
echo "======================"
echo ""
echo "This will test beam search on 1 MATH problem with small parameters."
echo ""

MODEL="qwen"  # Use smallest model for quick test
TEMPERATURE=0.25
BEAM_WIDTH=3
TOKENS_PER_STEP=8

# Test basic functionality
echo "Step 1: Testing basic beam search functionality..."
python llm_experiments/test_beam_search.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Basic functionality test failed!"
    exit 1
fi

echo ""
echo "Step 2: Testing on 1 MATH problem..."

# Create temporary results directory
mkdir -p results/$MODEL/beam_search

# Run on just 1 problem (batch_idx=0 processes problems 0-99, but we can limit in code)
# For now, this will process the first batch but you can modify beam_search_math.py
# to add a --num_problems flag if needed

python llm_experiments/beam_search_math.py \
    --model $MODEL \
    --temperature $TEMPERATURE \
    --beam_width $BEAM_WIDTH \
    --tokens_per_step $TOKENS_PER_STEP \
    --batch_idx 0 \
    --seed 0 \
    --verbose

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All tests passed!"
    echo ""
    echo "Check results in: results/$MODEL/beam_search/"
    echo ""
    echo "To run full experiments, use:"
    echo "  bash llm_experiments/scripts/beam_search_math.sh"
else
    echo ""
    echo "❌ Beam search test failed!"
    exit 1
fi
