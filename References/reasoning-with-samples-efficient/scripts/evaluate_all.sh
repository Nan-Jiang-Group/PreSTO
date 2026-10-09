#!/bin/bash
# Evaluate all prediction files using official evaluators

set -e

PREDICTIONS_DIR="${1:-predictions}"
RESULTS_DIR="${2:-results}"

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Evaluating All Prediction Files"
echo "=========================================="
echo "Predictions directory: $PREDICTIONS_DIR"
echo "Results directory: $RESULTS_DIR"
echo ""

# Check if evaluators are installed
if ! command -v evaluate_functional_correctness &> /dev/null; then
    echo "⚠️  Warning: human-eval not installed"
    echo "   Install with: pip install human-eval"
    echo ""
fi

if ! command -v swebench &> /dev/null; then
    echo "⚠️  Warning: swebench not installed"
    echo "   Install with: pip install swebench"
    echo ""
fi

# Evaluate HumanEval predictions
echo "=========================================="
echo "Evaluating HumanEval Predictions"
echo "=========================================="
echo ""

humaneval_count=0
for file in "$PREDICTIONS_DIR"/humaneval_*.jsonl; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "Evaluating: $filename"
        
        if command -v evaluate_functional_correctness &> /dev/null; then
            result=$(evaluate_functional_correctness "$file" 2>&1)
            echo "$result"
            
            # Extract pass@1 score
            pass_at_1=$(echo "$result" | grep "pass@1" | awk '{print $2}')
            
            # Save to results file
            result_file="$RESULTS_DIR/${filename%.jsonl}_results.txt"
            echo "File: $filename" > "$result_file"
            echo "Pass@1: $pass_at_1" >> "$result_file"
            echo "$result" >> "$result_file"
            
            echo "✓ Saved results to: $result_file"
        else
            echo "❌ Skipping (human-eval not installed)"
        fi
        echo ""
        humaneval_count=$((humaneval_count + 1))
    fi
done

if [ $humaneval_count -eq 0 ]; then
    echo "No HumanEval predictions found"
    echo ""
fi

# Evaluate SWE-bench predictions
echo "=========================================="
echo "Evaluating SWE-bench Predictions"
echo "=========================================="
echo ""

swebench_count=0
for file in "$PREDICTIONS_DIR"/swebench_*.jsonl; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "Evaluating: $filename"
        
        if command -v swebench &> /dev/null; then
            # Determine split from filename
            split="lite"
            if [[ $filename == *"verified"* ]]; then
                split="verified"
            fi
            
            echo "Using split: $split"
            swebench eval \
                --predictions_path "$file" \
                --swe_bench_tasks "princeton-nlp/SWE-bench_Lite" \
                --log_dir "$RESULTS_DIR/${filename%.jsonl}_logs" \
                2>&1 | tee "$RESULTS_DIR/${filename%.jsonl}_eval.log"
            
            echo "✓ Saved results to: $RESULTS_DIR/${filename%.jsonl}_eval.log"
        else
            echo "❌ Skipping (swebench not installed)"
        fi
        echo ""
        swebench_count=$((swebench_count + 1))
    fi
done

if [ $swebench_count -eq 0 ]; then
    echo "No SWE-bench predictions found"
    echo ""
fi

# Summary
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
echo "HumanEval predictions evaluated: $humaneval_count"
echo "SWE-bench predictions evaluated: $swebench_count"
echo "Results saved to: $RESULTS_DIR"
echo ""

# Display all results
if [ -f "$RESULTS_DIR"/*.txt ] || [ -f "$RESULTS_DIR"/*.log ]; then
    echo "=========================================="
    echo "Results Summary"
    echo "=========================================="
    echo ""
    
    for result_file in "$RESULTS_DIR"/*.txt; do
        if [ -f "$result_file" ]; then
            echo "$(basename "$result_file"):"
            grep "Pass@1" "$result_file" || echo "  (no pass@1 found)"
            echo ""
        fi
    done
fi

echo "Done!"
