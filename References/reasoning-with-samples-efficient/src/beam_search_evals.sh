#!/bin/bash

# Detect OS and install GNU parallel if needed
echo "Checking for GNU parallel..."

# Check if parallel is already installed
if command -v parallel &> /dev/null; then
    echo "✓ GNU parallel is already installed"
    parallel --version | head -n 1
else
    echo "GNU parallel not found. Installing..."
    
    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "Detected macOS"
        if command -v brew &> /dev/null; then
            echo "Installing via Homebrew..."
            brew install parallel
        else
            echo "Error: Homebrew not found. Please install Homebrew first:"
            echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "Detected Linux"
        if command -v apt-get &> /dev/null; then
            echo "Installing via apt-get..."
            sudo apt-get update && sudo apt-get install -y parallel
        elif command -v yum &> /dev/null; then
            echo "Installing via yum..."
            sudo yum install -y parallel
        elif command -v dnf &> /dev/null; then
            echo "Installing via dnf..."
            sudo dnf install -y parallel
        else
            echo "Error: No supported package manager found (apt-get, yum, dnf)"
            exit 1
        fi
    else
        echo "Error: Unsupported OS: $OSTYPE"
        echo "Please install GNU parallel manually:"
        echo "  - macOS: brew install parallel"
        echo "  - Linux: sudo apt-get install parallel"
        exit 1
    fi
    
    # Verify installation
    if command -v parallel &> /dev/null; then
        echo "✓ GNU parallel installed successfully"
        parallel --version | head -n 1
    else
        echo "✗ Failed to install GNU parallel"
        exit 1
    fi
fi

echo ""

# Create output directory for logs
mkdir -p eval_logs
rm -f eval_logs/*.log

# Create parameter file
# cat > params.txt << 'EOF'
# beam_width=2 n_per_beam=2 alpha=4.0 length_penalty=0.6 tokens_per_step=16
# beam_width=2 n_per_beam=2 alpha=4.0 length_penalty=0.8 tokens_per_step=16
# beam_width=2 n_per_beam=2 alpha=4.0 length_penalty=1.0 tokens_per_step=16
# beam_width=3 n_per_beam=3 alpha=4.0 length_penalty=0.6 tokens_per_step=16
# beam_width=3 n_per_beam=3 alpha=4.0 length_penalty=0.8 tokens_per_step=16
# beam_width=3 n_per_beam=3 alpha=4.0 length_penalty=1.0 tokens_per_step=16
# beam_width=5 n_per_beam=5 alpha=4.0 length_penalty=0.6 tokens_per_step=16
# beam_width=5 n_per_beam=5 alpha=4.0 length_penalty=0.8 tokens_per_step=16
# beam_width=5 n_per_beam=5 alpha=4.0 length_penalty=1.0 tokens_per_step=16
# EOF

cat > params.txt << 'EOF'
beam_width=2 n_per_beam=2 alpha=4.0 length_penalty=0.8 tokens_per_step=256
beam_width=5 n_per_beam=5 alpha=4.0 length_penalty=0.8 tokens_per_step=256
EOF

echo "Running 6 experiments in parallel (max 4 at once)..."
echo "Logs will be saved to eval_logs/"
echo ""

# Run in parallel with max 4 jobs at once, saving output to separate log files
parallel -j 4 --colsep ' ' \
	'uv run --python 3.12 python run_benchmark.py \
		benchmark.name=gsm8k \
		benchmark.num_problems=30 \
		beam_search.enabled=true \
		beam_search.{1} \
		beam_search.{2} \
		beam_search.{3} \
		beam_search.{4} \
		beam_search.{5} \
		greedy.enabled=true \
		mcmc.enabled=false \
		temperature_sampling.enabled=false \
		> eval_logs/exp_{1}_{2}_{3}_{4}_{5}.log 2>&1' \
	:::: params.txt

echo ""
echo "All experiments completed!"
echo ""

# Aggregate and display results using Python script
if [ -f "aggregate_results.py" ]; then
	echo "Aggregating results..."
	echo ""
	uv run --python 3.12 python aggregate_results.py eval_logs
else
	# Fallback: simple text extraction
	echo "======================================================================================================"
	echo "AGGREGATED RESULTS FROM ALL EXPERIMENTS"
	echo "======================================================================================================"
	echo ""
	
	# Extract and display results from each log file
	for logfile in eval_logs/*.log; do
		if [ -f "$logfile" ]; then
			# Extract the benchmark results table
			echo "Results from $(basename $logfile):"
			echo ""
			
			# Find the table section
			awk '/^GSM8K BENCHMARK RESULTS$/,/^\+.*\+$/{
				if (/^GSM8K BENCHMARK RESULTS$/) {
					table_started=1
					next
				}
				if (table_started) {
					print
				}
			}' "$logfile"
			
			echo ""
			echo "------------------------------------------------------------------------------------------------------"
			echo ""
		fi
	done
fi

echo ""
echo "✅ Individual logs saved in eval_logs/"
echo "✅ To re-aggregate results: python aggregate_results.py eval_logs"
echo ""
