#!/bin/bash
# Submit V6-only matched PowerMH and PreSTO-PowerMH runs on TACC.
# Submit from the cluster checkout by running its absolute path:
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/lcb_v6_power_mh_vs_presto.tacc.sbatch.sh
# Each model's two methods run sequentially in the same GPU allocation.

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)
repo_root=${REPO_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
export REPO_ROOT="$repo_root"

if [[ "${1:-}" == --run ]]; then
    model=$2
    run_date=$3
    source "$repo_root/experiments/env.sh"
    set +x
    source "${CONDA_INIT:-$HOME/WORK/miniconda3/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV:-cuda130}"
    cd "$repo_root"
    export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

    # Verify the downloaded V6-only shard, not just its filename or release label.
    python - <<'PY'
import json
import os
from pathlib import Path
from power_sharpening.tasks.lcb_benchmark import LiveCodeBenchBenchmark

expected = Path(os.environ['REPO_ROOT']) / 'data/LiveCodeBench_v6.jsonl'
benchmark = LiveCodeBenchBenchmark(1, version='release_v6')
if benchmark.source_path.resolve() != expected.resolve():
    raise SystemExit('LCB V6 loader did not select the V6 file')
rows = [json.loads(line) for line in expected.open() if line.strip()]
expected_sha256 = 'e92fbf266ad18eaa6995a03104792d494b342f1a3c2524bbd99d67afc35b6f21'
if len(rows) != 175 or benchmark.source_sha256 != expected_sha256:
    raise SystemExit('Expected the verified 175-problem V6-only shard; omit V5 and cumulative releases')
print(f'Verified V6-only dataset: {len(rows)} problems, sha256={benchmark.source_sha256}', flush=True)
PY
    dump_dir="$repo_root/case_studies/logs/lcb_v6/$run_date/vllm"
    prefix="dataset-lcb_v6.model-${model}.alpha4.0.steps100.blocks-1"
    suffix=samples20.maxnew1024.seed10086
    for log in \
        "$dump_dir/$prefix.$suffix.powerMH.vllm.log" \
        "$dump_dir/$prefix.prefetch-budget-20.rank-bfs_accept_first.$suffix.subtreePrefetch.vllm.log"; do
        [[ ! -e "$log" ]] || { echo "Already exists: $log; choose another RUN_DATE." >&2; exit 1; }
    done
    common=(--datasets lcb_v6 --models "$model" --run-date "$run_date"
        --resource-probe on --kv-cache-mode hooks)
    overrides=(seed=10086 alpha=4.0 mcmc_steps=100 num_blocks=1 max_samples=20
        max_new_tokens=1024 batch_size=1 temperature=-1 temperature_schedule_type=const
        cut_dist_type=uniform prefix_cache=true gpu_memory_utilization=0.9
        max_prompt_chars=3500)
    bash "$repo_root/case_studies/scripts/run_power_mh_case_study.sh" \
        "${common[@]}" --override "${overrides[@]}"
    bash "$repo_root/case_studies/scripts/run_subtree_prefetching_mh_case_study.sh" \
        "${common[@]}" --prefetch-budgets 20 --rank-fns bfs_accept_first \
        --override "${overrides[@]}" print_tree=true
    exit
fi

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 2
fi
read -r -a models <<< "${MODELS:-qwen qwen3-4b qwen3-8b qwen3.5-4b qwen3.5-9b gemma-12b-it}"
run_date=${RUN_DATE:-$(date +%F)-v6-only}
slurm_dir="$repo_root/case_studies/logs/lcb_v6/$run_date/vllm/slurm"
[[ -s "$repo_root/data/LiveCodeBench_v6.jsonl" ]] || {
    echo "Fetch the V6-only dataset first; see case_studies/lcb-v6-matched-comparison-runs.md." >&2
    exit 1
}
mkdir -p "$slurm_dir"
for model in "${models[@]}"; do
    command=(sbatch -p gh -A CCR25054 -N 1 -n 1 -t "${TACC_TIME:-12:00:00}"
        --job-name="lcb-v6-matched-$model"
        --output="$slurm_dir/$model.powerMH-vs-presto.out"
        "$repo_root/case_studies/scripts/lcb_v6_power_mh_vs_presto.tacc.sbatch.sh"
        --run "$model" "$run_date")
    "${command[@]}"
done
