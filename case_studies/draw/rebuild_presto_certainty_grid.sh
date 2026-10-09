#!/bin/bash
# Rebuild the PreSTO-PowerMH epsilon-certainty grid (paper Figure 15: five base LLMs x seven datasets, N_pf=20,
# default BFS traversal, all edges, hatched) from one fixed log per cell.
# The draw script finds logs by <log-root>/<dataset>/<run-date>/vllm/<run name>, so the chosen logs are symlinked
# into a staging tree under one run date. The LCB V6 cells of Qwen3-4B, Qwen3-8B, and Qwen3.5-4B use the N_pf=20
# runs of lcb_v6_presto_n20.punakha.sbatch.sh, dated LCB_RUN_DATE.
#   LCB_RUN_DATE=2026-10-09 bash case_studies/draw/rebuild_presto_certainty_grid.sh

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)
repo_root=${REPO_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
cases="$repo_root/case_studies"
: "${LCB_RUN_DATE:?set LCB_RUN_DATE to the run date of the LCB V6 N_pf=20 runs}"
paper_dir=${PAPER_DIR:-"/Users/jiangnanhugo/overleaf papers/power_sampling paper/prefetching"}
output="$paper_dir/exps/epsilon-certainty/dataset-all.model-all.prefetch-budget-20.rank-bfs_accept_first.all-edges.proposal-certainty.hatched.pdf"
stage_root=$(mktemp -d "${TMPDIR:-/tmp}/presto-certainty.XXXXXX")
trap 'rm -rf "$stage_root"' EXIT

suffix=alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-bfs_accept_first.samples20.maxnew1024.seed10086.subtreePrefetch.vllm.log
# dataset model directory-of-the-log (relative to case_studies/)
cells=(
    "math500 qwen3-4b logs/math500/2026-09-16" "math500 qwen3-8b logs/math500/2026-09-16"
    "math500 qwen3.5-4b logs/math500/2026-09-16" "math500 qwen3.5-9b logs/math500/2026-09-04"
    "math500 gemma-12b-it logs/math500/2026-09-24"
    "aime qwen3-4b logs/aime/2026-09-16" "aime qwen3-8b logs/aime/2026-09-16"
    "aime qwen3.5-4b logs/aime/2026-09-16" "aime qwen3.5-9b logs/aime/2026-09-04"
    "aime gemma-12b-it logs/aime/2026-09-24"
    "mbpp qwen3-4b logs/mbpp/2026-09-16" "mbpp qwen3-8b logs/mbpp/2026-09-16"
    "mbpp qwen3.5-4b logs/mbpp/2026-09-16" "mbpp qwen3.5-9b logs/mbpp/2026-09-04"
    "mbpp gemma-12b-it logs/mbpp/2026-09-24"
    "gpqa qwen3-4b logs/gpqa/2026-09-16" "gpqa qwen3-8b logs/gpqa/2026-09-16"
    "gpqa qwen3.5-4b logs/gpqa/2026-09-16" "gpqa qwen3.5-9b logs/gpqa/2026-09-05"
    "gpqa gemma-12b-it logs/gpqa/2026-09-24"
    "human_eval qwen3-4b logs/human_eval/2026-09-16" "human_eval qwen3-8b logs/human_eval/2026-09-16"
    "human_eval qwen3.5-4b logs/human_eval/2026-09-16" "human_eval qwen3.5-9b logs/human_eval/2026-09-04"
    "human_eval gemma-12b-it logs/human_eval/2026-09-24"
    "mmlu qwen3-4b logs/mmlu/2026-09-16" "mmlu qwen3-8b logs/mmlu/2026-09-16"
    "mmlu qwen3.5-4b logs/mmlu/2026-09-16" "mmlu qwen3.5-9b logs/mmlu/2026-09-04"
    "mmlu gemma-12b-it logs/mmlu/2026-09-24"
    "lcb_v6 qwen3-4b logs/lcb_v6/$LCB_RUN_DATE" "lcb_v6 qwen3-8b logs/lcb_v6/$LCB_RUN_DATE"
    "lcb_v6 qwen3.5-4b logs/lcb_v6/$LCB_RUN_DATE"
    "lcb_v6 qwen3.5-9b logs_predictive_prefetching/Uniform-family/lcb_v6/2026-10-07"
    "lcb_v6 gemma-12b-it logs/lcb_v6/2026-09-24"
)
for cell in "${cells[@]}"; do
    read -r dataset model dir <<< "$cell"
    name="dataset-$dataset.model-$model.$suffix"
    source="$cases/$dir/vllm/$name"
    [[ -s "$source" ]] || { echo "Missing log: $source" >&2; exit 1; }
    # Every LCB V6 cell must come from the 175-problem V6 file, not the 880-problem V5 file.
    if [[ "$dataset" == lcb_v6 ]] && ! grep -q -m1 "benchmark_size: 175" "$source"; then
        echo "Not an LCB V6 run (benchmark_size is not 175): $source" >&2; exit 1
    fi
    mkdir -p "$stage_root/$dataset/STAGE/vllm"
    ln -s "$source" "$stage_root/$dataset/STAGE/vllm/$name"
done

cd "$repo_root"
MPLCONFIGDIR="$stage_root/mpl" "$repo_root/src/.venv/bin/python" \
    "$cases/draw/draw_multi_model_proposal_certainty.py" \
    --log-root "$stage_root" --run-date STAGE --rank-fn bfs_accept_first --prefetch-budget 20 \
    --hatch --all-edges \
    --models qwen3-4b qwen3-8b qwen3.5-4b qwen3.5-9b gemma-12b-it \
    --datasets math500 aime mbpp gpqa human_eval mmlu lcb_v6 \
    --output "$output" --csv-output "${output%.pdf}.csv"
