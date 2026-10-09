#!/bin/bash

# Beam search on MATH benchmark

MODEL="qwen_math"
TEMPERATURE=0.25  # α = 1/0.25 = 4
BEAM_WIDTH=5
TOKENS_PER_STEP=16
LENGTH_PENALTY=0.6
BATCH_IDX=0
SEED=0

echo "Running beam search on MATH benchmark"
echo "======================================"

# Basic run with default parameters
python llm_experiments/beam_search_math.py \
    --model $MODEL \
    --temperature $TEMPERATURE \
    --beam_width $BEAM_WIDTH \
    --tokens_per_step $TOKENS_PER_STEP \
    --length_penalty $LENGTH_PENALTY \
    --batch_idx $BATCH_IDX \
    --seed $SEED \
    --cot True

# Run with MCMC comparison
echo ""
echo "Running with MCMC comparison..."
python llm_experiments/beam_search_math.py \
    --model $MODEL \
    --temperature $TEMPERATURE \
    --beam_width $BEAM_WIDTH \
    --tokens_per_step $TOKENS_PER_STEP \
    --length_penalty $LENGTH_PENALTY \
    --batch_idx $BATCH_IDX \
    --seed $SEED \
    --cot True \
    --compare_mcmc

# Sweep: Run with different beam widths for comparison
echo ""
echo "Running beam width sweep..."
for BW in 3 5 10; do
    echo "Testing beam_width=$BW"
    python llm_experiments/beam_search_math.py \
        --model $MODEL \
        --temperature $TEMPERATURE \
        --beam_width $BW \
        --tokens_per_step $TOKENS_PER_STEP \
        --length_penalty $LENGTH_PENALTY \
        --batch_idx $BATCH_IDX \
        --seed $SEED \
        --cot True
done

# Sweep: Run with different tokens_per_step
echo ""
echo "Running tokens_per_step sweep..."
for TPS in 8 16 32; do
    echo "Testing tokens_per_step=$TPS"
    python llm_experiments/beam_search_math.py \
        --model $MODEL \
        --temperature $TEMPERATURE \
        --beam_width $BEAM_WIDTH \
        --tokens_per_step $TPS \
        --length_penalty $LENGTH_PENALTY \
        --batch_idx $BATCH_IDX \
        --seed $SEED \
        --cot True
done

# Sweep: Run with different temperatures
echo ""
echo "Running temperature sweep..."
for TEMP in 0.1 0.25 0.5; do
    echo "Testing temperature=$TEMP (α=$(python3 -c "print(1/$TEMP)"))"
    python llm_experiments/beam_search_math.py \
        --model $MODEL \
        --temperature $TEMP \
        --beam_width $BEAM_WIDTH \
        --tokens_per_step $TOKENS_PER_STEP \
        --length_penalty $LENGTH_PENALTY \
        --batch_idx $BATCH_IDX \
        --seed $SEED \
        --cot True
done

echo ""
echo "All experiments complete!"
echo "Results saved to: results/$MODEL/beam_search/"
