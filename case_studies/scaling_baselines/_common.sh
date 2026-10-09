# Shared launcher helper for the scaling baselines. Sourced by the three scripts next to it; not run directly.
#
# Each script takes optional KEY=value (or --key value) settings. The cluster is the script's CLUSTER default (the
# .punakha.sh scripts set punakha); a leading interactive|punakha|tacc argument overrides it:
#
#   bash <script>.punakha.sh [interactive|tacc] [DATASETS="math500 gpqa"] [MODELS=qwen3-8b] ...
#
# Common settings (each script adds its own scaling axis):
#   DATASETS  MODELS         grids; default is the paper grid, 7 datasets x 5 base LLMs, full datasets
#   MAX_NEW_TOKENS           generation budget; unset keeps the per-dataset value in dataset.yaml
#   SEED (10086)  RUN_DATE (today)  PYTHON (interpreter for `interactive`)
#
# Every run writes <run_name>.csv (one row per prompt) and <run_name>.<algorithm>.vllm.log under
# case_studies/scaling_baselines/logs/<dataset>/<RUN_DATE>/. A run whose CSV already exists is skipped, so
# re-launching the same command resumes a partly finished sweep. A script may set log_root (instead of logs/) and
# log_leaf (a subfolder such as /vllm) before sourcing this file.

scaling_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$scaling_dir/../.." && pwd)

cluster=${CLUSTER:-}
case "${1:-}" in
    interactive|punakha|tacc) cluster=$1; shift ;;
esac
case "$cluster" in
    interactive|punakha|tacc) ;;
    *) echo "Usage: $0 [interactive|punakha|tacc] [KEY=value ...] (or set CLUSTER)" >&2; exit 2 ;;
esac

while (( $# )); do
    case "$1" in
        --*=*) key=${1%%=*}; value=${1#*=}; shift ;;
        --*) key=$1; value=${2:?missing value for $1}; shift 2 ;;
        *=*) key=${1%%=*}; value=${1#*=}; shift ;;
        *) echo "Unexpected argument: $1. Use KEY=value or --key value." >&2; exit 2 ;;
    esac
    key=${key#--}; key=${key//-/_}; key=$(printf '%s' "$key" | tr '[:lower:]' '[:upper:]')
    printf -v "$key" '%s' "$value"
done

datasets=${DATASETS:-math500 aime mbpp gpqa human_eval mmlu lcb_v6}
models=${MODELS:-qwen3-4b qwen3-8b qwen3.5-4b qwen3.5-9b gemma-12b-it}
seed=${SEED:-10086}
run_date=${RUN_DATE:-$(date +%F)}
max_new_tokens=${MAX_NEW_TOKENS:-}
py=${PYTHON:-python}

# launch DATASET ALGORITHM RUN_NAME MODULE [OVERRIDE ...]
# Submits (or runs) one run of the runner module with the given config overrides.
launch() {
    local dataset=$1 algorithm=$2 run_name=$3 module=$4
    shift 4
    local dump_dir=${log_root:-$scaling_dir/logs}/$dataset/$run_date${log_leaf:-}
    mkdir -p "$dump_dir"
    if [[ -f $dump_dir/$run_name.csv ]]; then
        echo "skip (done): $run_name"
        return 0
    fi
    local overrides=("seed=$seed" "$@")
    [[ -n $max_new_tokens ]] && overrides+=("max_new_tokens=$max_new_tokens")
    local args=(--dataset "$dataset" --algorithm "$algorithm" --save_str "$dump_dir" --run_name "$run_name"
                --override "${overrides[@]}")
    if [[ $cluster == interactive ]]; then
        "$py" -m "$module" "${args[@]}" --model_str "$model" 2>&1 | tee "$dump_dir/$run_name.$algorithm.vllm.log"
    else
        zsh "$repo_root/experiments/clusters/$cluster.sbatch.sh" "$dump_dir" "$algorithm" "$run_name" \
            -m "$module" "${args[@]}" --model_str "$model"
    fi
}
