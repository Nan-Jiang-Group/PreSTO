# SWE-bench Evaluation Pipeline

Complete guide for generating predictions and evaluating them on SWE-bench using Modal.

---

## Overview

**The Pipeline:**
1. **Generate Predictions** - Use LLMs to generate patches for SWE-bench problems
2. **Evaluate on Modal** - Run predictions against actual test suites in the cloud

**Why Modal?** SWE-bench evaluation requires:
- Cloning GitHub repos
- Applying patches
- Running full test suites
- Docker isolation

Modal handles all this infrastructure automatically in the cloud.

---

## Prerequisites

### 1. Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -e ".[eval]"
```

### 2. API Keys

Create `.env` file:
```bash
XAI_API_KEY=your_xai_api_key_here
```

### 3. Modal Setup (One-time)

```bash
# Install Modal
pip install modal

# Authenticate (opens browser)
modal token new
```

This creates `~/.modal.toml` with your credentials.

---

## Step 1: Generate Predictions

Generate patches for SWE-bench problems using your LLM.

### Basic Usage

```bash
python src/run_benchmark.py \
  model.name=grok-4-1-fast-reasoning \
  benchmark.name=swebench \
  benchmark.num_problems=5 \
  mcmc.enabled=false \
  temperature_sampling.enabled=false
```

**Output:**
```
predictions/swe_bench_lite_grok_4_1_fast_reasoning_Greedy.jsonl
```

### Parameters (Hydra Config)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.name` | Model to use (e.g., grok-4-1-fast-reasoning) | grok-2-1212 |
| `benchmark.name` | Benchmark: `swebench`, `swebench-verified`, `humaneval` | humaneval |
| `benchmark.num_problems` | Number of problems to test | 10 |
| `benchmark.max_tokens` | Max tokens per completion | 3072 |
| `greedy.enabled` | Enable greedy sampling | true |
| `mcmc.enabled` | Enable MCMC sampling | true |
| `mcmc.alpha` | MCMC power factor | 1.67 |
| `mcmc.steps` | MCMC steps per block | 10 |
| `temperature_sampling.enabled` | Enable temperature sampling | true |
| `temperature_sampling.temperature` | Temperature | 0.8 |

### Example: Multiple Strategies

```bash
# Run all strategies (greedy, mcmc, temperature)
python src/run_benchmark.py \
  model.name=grok-4-1-fast-reasoning \
  benchmark.name=swebench \
  benchmark.num_problems=10 \
  mcmc.steps=5 \
  mcmc.alpha=1.67
```

### Example: Only MCMC

```bash
python src/run_benchmark.py \
  model.name=grok-4-1-fast-reasoning \
  benchmark.name=swebench \
  benchmark.num_problems=10 \
  greedy.enabled=false \
  temperature_sampling.enabled=false \
  mcmc.steps=5
```

### Batch Runs with Hydra Multirun

Run multiple configurations in one command:

```bash
# Compare models
python src/run_benchmark.py -m \
  model.name=grok-4-1-fast-reasoning,grok-2-1212 \
  benchmark.name=swebench \
  benchmark.num_problems=10

# Sweep MCMC alpha
python src/run_benchmark.py -m \
  mcmc.alpha=1.0,1.67,4.0 \
  benchmark.name=swebench
```

### Prediction File Format

Each line is a JSON object:
```json
{
  "instance_id": "astropy__astropy-12907",
  "model_patch": "diff --git a/file.py b/file.py\n...",
  "model_name_or_path": "custom"
}
```

**Important:** `model_patch` must be a valid unified diff with:
- Real line numbers (not `@@ -XXX,YYY +XXX,YYY @@`)
- Proper diff format (`---`, `+++`, `@@`)
- Context lines around changes

### Cost Estimation (API Calls)

| Problems | Estimated Cost | Time |
|----------|---------------|------|
| 5 | ~$0.05-0.10 | 2-5 min |
| 10 | ~$0.10-0.20 | 5-10 min |
| 50 | ~$0.50-1.00 | 30-60 min |
| 300 (full) | ~$3-6 | 3-5 hours |

---

## Step 2: Evaluate on Modal

Run predictions against actual SWE-bench test suites in the cloud.

### Basic Usage

```bash
modal run scripts/modal_evaluate.py \
  --prediction predictions/swe_bench_lite_grok_4_1_fast_reasoning_Greedy.jsonl
```

### Evaluate All Predictions

```bash
modal run scripts/modal_evaluate.py \
  --predictions-dir predictions/
```

Evaluates all `swe_bench*.jsonl` and `swebench*.jsonl` files.

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--prediction` | Single prediction file to evaluate | None |
| `--predictions-dir` | Directory of predictions to evaluate | None |
| `--dataset` | Dataset: `princeton-nlp/SWE-bench_Lite`, `princeton-nlp/SWE-bench_Verified` | `SWE-bench_Lite` |
| `--max-workers` | Parallel evaluation workers | 16 |
| `--list-only` | List files in Modal storage | False |
| `--download` | Download specific results | None |

### What Happens

1. **Upload** - Prediction file uploaded to Modal volume
2. **Configure** - Modal credentials configured in container
3. **Evaluate** - SWE-bench spawns Modal tasks for each problem:
   - Clones the repository
   - Checks out base commit
   - Applies your patch
   - Runs test suite
   - Records pass/fail
4. **Download** - Results saved to `results/` directory

### Cost Estimation (Modal)

| Problems | Workers | Time | Estimated Cost |
|----------|---------|------|----------------|
| 2 | 16 | 10-15 min | ~$0.35-0.50 |
| 5 | 16 | 15-20 min | ~$0.60-0.80 |
| 10 | 16 | 20-30 min | ~$1.50-2.00 |
| 50 | 16 | 1-2 hrs | ~$6-8 |
| 300 (full) | 16 | 4-6 hrs | ~$30-40 |

**Note:** Modal free tier includes $30/month credits.

### Speed vs Cost

**Faster (default):**
```bash
modal run scripts/modal_evaluate.py --prediction predictions/file.jsonl
# 16 workers: fastest, ~$0.35 per 2 problems
```

**Balanced:**
```bash
modal run scripts/modal_evaluate.py --prediction predictions/file.jsonl --max-workers 8
# 8 workers: good speed, ~$0.30 per 2 problems
```

**Budget:**
```bash
modal run scripts/modal_evaluate.py --prediction predictions/file.jsonl --max-workers 4
# 4 workers: slower, ~$0.25 per 2 problems
```

---

## Step 3: View Results

Results are saved to `results/` directory.

### Result File Format

```bash
cat results/swe_bench_lite_grok_4_1_fast_reasoning_Greedy_results.json
```

```json
{
  "prediction_file": "/data/predictions/...",
  "dataset": "princeton-nlp/SWE-bench_Lite",
  "status": "success",
  "stdout": "...",
  "report_dir": "/data/logs/..."
}
```

### Extract Pass Rate

The evaluation output shows:
```
Instances resolved: X
Instances unresolved: Y
```

**Pass Rate = X / (X + Y)**

### Example Output

```bash
modal run scripts/modal_evaluate.py --prediction predictions/file.jsonl
```

```
Reading Modal credentials from ~/.modal.toml
Found active Modal profile: shivaperi47
Starting SWE-bench evaluation on Modal
Found 1 prediction file(s)
Dataset: princeton-nlp/SWE-bench_Lite
Workers: 16

[1/1] Processing: swe_bench_lite_grok_4_1_fast_reasoning_Greedy.jsonl
Uploaded: ...
Starting evaluation for: ...
Configuring Modal credentials...
Modal credentials configured

[Wait 15-30 minutes...]

Instances resolved: 3
Instances unresolved: 7

============================================================
EVALUATION SUMMARY
============================================================
Total evaluations: 1
Successful: 1
Failed: 0

Saving results locally...
Downloaded results for: swe_bench_lite_grok_4_1_fast_reasoning_Greedy
   Saved: results/swe_bench_lite_grok_4_1_fast_reasoning_Greedy_results.json

Done! Results saved to results/ directory
```

**Result:** 3/10 problems solved = 30% pass rate

---

## Complete Example Workflow

### Scenario: Compare Greedy vs MCMC on 10 problems

#### 1. Generate Predictions

```bash
# Run with both greedy and MCMC
python src/run_benchmark.py \
  model.name=grok-4-1-fast-reasoning \
  benchmark.name=swebench \
  benchmark.num_problems=10 \
  mcmc.steps=5 \
  temperature_sampling.enabled=false
```

**Output:**
```
predictions/swe_bench_lite_grok_4_1_fast_reasoning_Greedy.jsonl
predictions/swe_bench_lite_grok_4_1_fast_reasoning_MCMC(...).jsonl
```

**Cost:** ~$0.20 (API calls)
**Time:** ~10 minutes

#### 2. Evaluate on Modal

```bash
modal run scripts/modal_evaluate.py --predictions-dir predictions/
```

**Cost:** ~$2-3 (Modal compute)
**Time:** ~30-40 minutes

#### 3. Compare Results

```bash
cat results/*_results.json | grep "resolved"
```

**Output:**
```
Greedy: 3/10 resolved (30%)
MCMC: 4/10 resolved (40%)
```

**Conclusion:** MCMC performed better (+10% improvement)

**Total Cost:** ~$2.50
**Total Time:** ~50 minutes

---

## Troubleshooting

### Issue: "Patch Apply Failed"

**Symptom:**
```
EvaluationError: >>>>> Patch Apply Failed:
patch: **** missing line number at line 5: @@ -XXX,YYY +XXX,YYY @@
```

**Cause:** Your model generated invalid patches with placeholder line numbers.

**Fix:**
1. Check prediction file: `cat predictions/file.jsonl | head -1 | jq -r '.model_patch'`
2. Verify diff has real line numbers: `@@ -10,5 +10,8 @@` (not `@@ -XXX,YYY +XXX,YYY @@`)
3. If invalid, the model needs better prompting or different model

**Note:** Even with invalid patches, evaluation "succeeds" (no crash) but shows 0 resolved.

### Issue: "Modal credentials not found"

**Symptom:**
```
RuntimeError: ~/.modal.toml not found
```

**Fix:**
```bash
modal token new
```

This creates credentials at `~/.modal.toml`.

### Issue: "Evaluation is slow"

**Solution 1 - Increase workers:**
```bash
modal run scripts/modal_evaluate.py \
  --prediction predictions/file.jsonl \
  --max-workers 32  # Max parallelism
```

**Solution 2 - Test with fewer problems first:**
```bash
python src/run_benchmark.py \
  benchmark.num_problems=2 \
  benchmark.name=swebench \
  mcmc.enabled=false \
  temperature_sampling.enabled=false
```

### Issue: "Out of Modal credits"

**Check usage:**
```bash
open https://modal.com/apps/shivaperi47
```

**Options:**
- Free tier: $30/month
- Pay-as-you-go: Add payment method
- Reduce workers: `--max-workers 4` (cheaper)

### Issue: "ModuleNotFoundError"

**Symptom:**
```
ModuleNotFoundError: No module named 'modal'
```

**Fix:**
```bash
source .venv/bin/activate
pip install -e ".[eval]"
```

---

## Best Practices

### 1. Start Small

```bash
# Test with 2 problems first
python src/run_benchmark.py benchmark.num_problems=2 benchmark.name=swebench \
  mcmc.enabled=false temperature_sampling.enabled=false
modal run scripts/modal_evaluate.py --predictions-dir predictions/

# If successful, scale up
python src/run_benchmark.py benchmark.num_problems=10 benchmark.name=swebench \
  mcmc.enabled=false temperature_sampling.enabled=false
modal run scripts/modal_evaluate.py --predictions-dir predictions/
```

### 2. Use HumanEval for Iteration

SWE-bench is expensive and slow. Use HumanEval for quick testing:

```bash
# Fast and cheap
python src/run_benchmark.py benchmark.name=humaneval benchmark.num_problems=20

# Evaluate locally (no Modal needed)
# HumanEval has built-in evaluation
```

Then switch to SWE-bench for final validation.

### 3. Monitor Costs

- Check Modal dashboard regularly
- Start with `--max-workers 8` (balanced)
- Use `benchmark.num_problems=5` or `10` for testing
- Full runs (300 problems) should be final validation only

### 4. Validate Predictions Before Evaluating

```bash
# Check first prediction
cat predictions/file.jsonl | head -1 | jq -r '.model_patch'

# Should see valid diff format with real line numbers:
# @@ -10,5 +10,8 @@  Good
# @@ -XXX,YYY +XXX,YYY @@  Bad
```

Don't waste Modal credits on invalid predictions!

---

## Quick Reference

### Generate Predictions
```bash
python src/run_benchmark.py \
  model.name=grok-4-1-fast-reasoning \
  benchmark.name=swebench \
  benchmark.num_problems=10 \
  mcmc.enabled=false \
  temperature_sampling.enabled=false
```

### Evaluate on Modal
```bash
modal run scripts/modal_evaluate.py \
  --predictions-dir predictions/
```

### Check Results
```bash
cat results/*_results.json | grep "resolved"
```

### Estimated Costs
- **5 problems:** ~$0.80 total ($0.10 API + $0.70 Modal)
- **10 problems:** ~$2.20 total ($0.20 API + $2.00 Modal)
- **50 problems:** ~$7 total ($1 API + $6 Modal)

---

## Advanced Usage

### Custom Dataset

```bash
modal run scripts/modal_evaluate.py \
  --prediction predictions/file.jsonl \
  --dataset princeton-nlp/SWE-bench_Verified
```

### Download Previous Results

```bash
# List files in Modal storage
modal run scripts/modal_evaluate.py --list-only

# Download specific result
modal run scripts/modal_evaluate.py --download swe_bench_lite_model_strategy
```

### Batch Multiple Models

```bash
# Generate for multiple models using Hydra multirun
python src/run_benchmark.py -m \
  model.name=grok-4-1-fast-reasoning,grok-4-1-fast-non-reasoning \
  benchmark.name=swebench \
  benchmark.num_problems=10 \
  mcmc.enabled=false \
  temperature_sampling.enabled=false

# Evaluate all at once
modal run scripts/modal_evaluate.py --predictions-dir predictions/
```

---

## Support

- **Modal Dashboard:** https://modal.com/apps/shivaperi47
- **SWE-bench:** https://github.com/princeton-nlp/SWE-bench
- **Modal Docs:** https://modal.com/docs
