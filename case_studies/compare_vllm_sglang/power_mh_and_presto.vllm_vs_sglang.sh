#!/bin/bash
# PowerMH and PreSTO-PowerMH on vLLM and SGLang with matched settings, so the two backends can be compared on the same
# model, dataset, and sampler configuration. Every (backend, method, dataset, model) cell is one fresh Python process.
# Edit the settings below (or pass KEY=value after the cluster), then run:
#   bash /work/njiang/data/Subtree-Prefetching-Power-Sharpening/case_studies/compare_vllm_sglang/power_mh_and_presto.vllm_vs_sglang.sh interactive
#   bash /work/njiang/data/Subtree-Prefetching-Power-Sharpening/case_studies/compare_vllm_sglang/power_mh_and_presto.vllm_vs_sglang.sh punakha MODELS=qwen3-4b
#   bash /work/njiang/data/Subtree-Prefetching-Power-Sharpening/case_studies/compare_vllm_sglang/power_mh_and_presto.vllm_vs_sglang.punakha.sbatch.sh
#   bash /work/njiang/data/Subtree-Prefetching-Power-Sharpening/case_studies/compare_vllm_sglang/power_mh_and_presto.vllm_vs_sglang.tacc.sbatch.sh
# interactive runs every cell in this shell on an already-allocated GPU; punakha and tacc submit one sbatch job per cell.
# DRY_RUN=1 prints the commands only.
#
# Matched across backends: alpha, MH steps, blocks, samples, max new tokens, seed, proposal temperature 1/alpha with a
# const schedule, uniform cut distribution, and prefetch budget. The SGLang PreSTO sampler always ranks prefetch nodes
# with longest_path_first, so the vLLM PreSTO arm uses that rank too.
#
# Both backends write their run logs, CSVs, and stats under the same run name:
#   $LOG_ROOT/<dataset>/<RUN_DATE>/{vllm,sglang}/<run name>.{powerMH,subtreePrefetch}.{vllm,sglang}.log

set -euo pipefail

# ---- Settings (edit here; KEY=value on the command line overrides) ----
BACKENDS=${BACKENDS:-"vllm sglang"}            # vllm and/or sglang
METHODS=${METHODS:-"baseline presto"}          # baseline (PowerMH) and/or presto (PreSTO-PowerMH)
DATASETS=${DATASETS:-"lcb_v6"}                 # math500 aime mbpp gpqa human_eval lcb_v6 mmlu
MODELS=${MODELS:-"qwen3.5-9b"}                 # keys of power_sharpening.tasks.constants.MODEL_MAP
ALPHA=${ALPHA:-4.0}                            # sharpening power p^alpha; proposal temperature is 1/alpha
MCMC_STEPS=${MCMC_STEPS:-100}
NUM_BLOCKS=${NUM_BLOCKS:-1}
MAX_SAMPLES=${MAX_SAMPLES:-20}                 # number of benchmark problems
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
BUDGETS=${BUDGETS:-"20"}                       # PreSTO prefetch budgets
RANK_FN=${RANK_FN:-accept_first}              # PreSTO subtree ranking; a RANK_FNS key of power_sharpening.common.prefetch_rank
RESOURCE_PROBE=${RESOURCE_PROBE:-off}          # vLLM only: on | off | auto; off keeps timing comparable
VLLM_CONDA_ENV=${VLLM_CONDA_ENV:-}            # empty: vllm-cuda130 (Punakha) or cuda130 (TACC)
SGLANG_CONDA_ENV=${SGLANG_CONDA_ENV:-}        # empty: sgl-cuda129
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-}  # SGLang PreSTO --mem_fraction_static; empty: SGLang default
SGLANG_PRINT_TREE=${SGLANG_PRINT_TREE:-0}     # 1 passes --print_tree to the SGLang PreSTO run
CONDA_INIT=${CONDA_INIT:-$HOME/WORK/miniconda3/etc/profile.d/conda.sh}
PUNAKHA_MACHINE=${PUNAKHA_MACHINE:-dgx}
TACC_PARTITION=${TACC_PARTITION:-gh}
TACC_ACCOUNT=${TACC_ACCOUNT:-CCR25054}
TIME=${TIME:-}                                 # Slurm time limit; empty: 20:00:00 (Punakha QOS max 1-00:00:00) or 12:00:00 (TACC)
RUN_DATE=${RUN_DATE:-$(date +%F)}
DRY_RUN=${DRY_RUN:-0}
# -----------------------------------------------------------------------

repo_root=${REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
LOG_ROOT=${LOG_ROOT:-$repo_root/case_studies/logs-compare-vllm-sglang}
# env.sh supplies SEED and cache paths; it also turns on xtrace, which is noise here.
source "$repo_root/experiments/env.sh"
set +x
export REPO_ROOT="$repo_root"

cluster=${1:-}
(( $# == 0 )) || shift
for setting in "$@"; do
    case "$setting" in
        BACKENDS=*|METHODS=*|DATASETS=*|MODELS=*|ALPHA=*|MCMC_STEPS=*|NUM_BLOCKS=*|MAX_SAMPLES=*|MAX_NEW_TOKENS=*|\
        BUDGETS=*|RANK_FN=*|RESOURCE_PROBE=*|VLLM_CONDA_ENV=*|SGLANG_CONDA_ENV=*|CONDA_INIT=*|PUNAKHA_MACHINE=*|TIME=*|\
        TACC_PARTITION=*|TACC_ACCOUNT=*|RUN_DATE=*|DRY_RUN=*|LOG_ROOT=*|SGLANG_MEM_FRACTION=*|SGLANG_PRINT_TREE=*)
            printf -v "${setting%%=*}" '%s' "${setting#*=}" ;;
        *)
            echo "Unknown setting: $setting" >&2
            exit 2
            ;;
    esac
done
# Cluster-specific sbatch options and Conda environments. Submitted jobs rerun this script as interactive with the
# environments resolved here.
case "$cluster" in
    interactive|punakha)
        VLLM_CONDA_ENV=${VLLM_CONDA_ENV:-vllm-cuda130}
        sbatch_opts=(--account=punakha_general --qos="punakha_${PUNAKHA_MACHINE}_general"
            --partition="$PUNAKHA_MACHINE" --gres=gpu:1 --nodes=1 --ntasks=1 --time="${TIME:-20:00:00}")
        ;;
    tacc)
        VLLM_CONDA_ENV=${VLLM_CONDA_ENV:-cuda130}
        sbatch_opts=(-p "$TACC_PARTITION" -A "$TACC_ACCOUNT" -N 1 -n 1 -t "${TIME:-12:00:00}")
        ;;
    *)
        echo "Usage: bash $(basename -- "$0") <interactive|punakha|tacc> [KEY=value ...]" >&2
        exit 2
        ;;
esac
SGLANG_CONDA_ENV=${SGLANG_CONDA_ENV:-sgl-cuda129}
[[ "$LOG_ROOT" == /* ]] || LOG_ROOT=$repo_root/$LOG_ROOT

temperature=$(awk -v a="$ALPHA" 'BEGIN { print 1 / a }')
# SGLang runners take --task plus task options; vLLM runners take the dataset.yaml key.
sglang_task_args() {
    case "$1" in
        lcb_v6) echo "--task lcb --lcb_version release_v6 --difficulty all" ;;
        aime) echo "--task aime --aime_dataset aime2024-2025" ;;
        *) echo "--task $1" ;;
    esac
}

# Print one cell's Python command, one argument per line. Arguments: backend, arm, dataset, model, budget, run name,
# output directory.
python_command() {
    local backend=$1 arm=$2 dataset=$3 model=$4 budget=$5 run_name=$6 out_dir=$7
    local -a cmd
    if [[ "$backend" == vllm ]]; then
        local -a overrides=(seed="$SEED" alpha="$ALPHA" mcmc_steps="$MCMC_STEPS" num_blocks="$NUM_BLOCKS"
            max_samples="$MAX_SAMPLES" max_new_tokens="$MAX_NEW_TOKENS" batch_size=1 temperature=-1
            temperature_schedule_type=const cut_dist_type=uniform)
        if [[ "$arm" == baseline ]]; then
            cmd=(python -m power_sharpening.runners.vllm.run_power_mh --dataset "$dataset" --algorithm power_mcmc)
        else
            cmd=(python -m power_sharpening.runners.vllm.run_subtree_prefetching_mh --dataset "$dataset"
                --algorithm subtree_prefetching_mh)
            overrides+=(print_tree=false prefetch_budget="$budget" rank_fn="$RANK_FN")
        fi
        cmd+=(--model_str="$model" --save_str "$out_dir" --run_name "$run_name")
        case "$RESOURCE_PROBE" in
            on) cmd+=(--resource_probe) ;;
            off) cmd+=(--no-resource_probe) ;;
        esac
        cmd+=(--override "${overrides[@]}")
    else
        local -a task_args
        read -r -a task_args <<< "$(sglang_task_args "$dataset")"
        if [[ "$arm" == baseline ]]; then
            cmd=(python -m power_sharpening.runners.sglang.run_power_sample_mh --algorithm power_mcmc
                --temperature "$temperature" --temperature_schedule_type const --cut_dist_type uniform)
        else
            cmd=(python -m power_sharpening.runners.sglang.run_subtree_prefetching_mh --prefetch_budget "$budget"
                --rank_fn "$RANK_FN")
            [[ -z "$SGLANG_MEM_FRACTION" ]] || cmd+=(--mem_fraction_static "$SGLANG_MEM_FRACTION")
            [[ "$SGLANG_PRINT_TREE" != 1 ]] || cmd+=(--print_tree)
        fi
        cmd+=("${task_args[@]}" --model_str "$model" --save_str "$out_dir" --run_name "$run_name"
            --alpha "$ALPHA" --mcmc_steps "$MCMC_STEPS" --num_blocks "$NUM_BLOCKS" --seed "$SEED"
            --max_samples "$MAX_SAMPLES" --max_new_tokens "$MAX_NEW_TOKENS")
    fi
    printf '%s\n' "${cmd[@]}"
}

# Run one cell in a subshell with that backend's Conda environment.
run_cell() {
    local backend=$1 arm=$2 dataset=$3 model=$4 budget=$5
    local run_name="dataset-${dataset}.model-${model}.alpha${ALPHA}.steps${MCMC_STEPS}.blocks-${NUM_BLOCKS}"
    local log_tag=powerMH
    if [[ "$arm" == presto ]]; then
        run_name+=".prefetch-budget-${budget}.rank-${RANK_FN}"
        log_tag=subtreePrefetch
    fi
    run_name+=".samples${MAX_SAMPLES}.maxnew${MAX_NEW_TOKENS}.seed${SEED}"
    local out_dir="$LOG_ROOT/$dataset/$RUN_DATE/$backend"
    local run_log="$out_dir/${run_name}.${log_tag}.${backend}.log"
    local conda_env=$VLLM_CONDA_ENV
    [[ "$backend" == vllm ]] || conda_env=$SGLANG_CONDA_ENV
    local -a cmd
    mapfile -t cmd < <(python_command "$backend" "$arm" "$dataset" "$model" "$budget" "$run_name" "$out_dir")

    echo "[$backend/$arm] $run_name"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '  conda activate %s && ' "$conda_env"
        printf '%q ' "${cmd[@]}"
        printf '> %q\n' "$run_log"
        return 0
    fi
    mkdir -p "$out_dir"
    # set -e is off inside the caller's `if !`, so every step that can fail exits the subshell explicitly.
    (
        set +u
        source "$CONDA_INIT" || exit 1
        conda activate "$conda_env" || exit 1
        set -u
        export PYTHONUNBUFFERED=1
        # Lets PyTorch reuse fragmented reserved memory for the large one-off logits tensors of batched scoring.
        [[ "$backend" != sglang ]] || export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export VLLM_WORKER_MULTIPROC_METHOD=spawn
        cd "$repo_root" || exit 1
        hostname
        local start=$SECONDS
        "${cmd[@]}" || exit $?
        echo "wall time: $((SECONDS - start)) sec"
    ) 2>&1 | tee "$run_log"
}

# Submit one cell as its own Slurm job that reruns this script interactively on the narrowed grid.
submit_cell() {
    local backend=$1 arm=$2 dataset=$3 model=$4 budget=$5
    local slurm_dir="$LOG_ROOT/$dataset/$RUN_DATE/slurm"
    local job_name="${arm}-${backend}-${dataset}-${model}"
    [[ "$arm" == baseline ]] || job_name+="-p${budget}"
    local -a cmd=(sbatch "${sbatch_opts[@]}" --job-name="$job_name" --output="$slurm_dir/$job_name.out" --export=ALL
        "$repo_root/case_studies/compare_vllm_sglang/$(basename -- "$0")" interactive
        BACKENDS="$backend" METHODS="$arm" DATASETS="$dataset" MODELS="$model" BUDGETS="$budget" RANK_FN="$RANK_FN"
        ALPHA="$ALPHA" MCMC_STEPS="$MCMC_STEPS" NUM_BLOCKS="$NUM_BLOCKS" MAX_SAMPLES="$MAX_SAMPLES"
        MAX_NEW_TOKENS="$MAX_NEW_TOKENS" RESOURCE_PROBE="$RESOURCE_PROBE" RUN_DATE="$RUN_DATE" LOG_ROOT="$LOG_ROOT"
        VLLM_CONDA_ENV="$VLLM_CONDA_ENV" SGLANG_CONDA_ENV="$SGLANG_CONDA_ENV" CONDA_INIT="$CONDA_INIT"
        SGLANG_MEM_FRACTION="$SGLANG_MEM_FRACTION" SGLANG_PRINT_TREE="$SGLANG_PRINT_TREE")
    echo "Submitting $job_name"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        mkdir -p "$slurm_dir"
        "${cmd[@]}"
    fi
}

echo "backends=($BACKENDS) methods=($METHODS) datasets=($DATASETS) models=($MODELS) budgets=($BUDGETS)" \
    "alpha=$ALPHA steps=$MCMC_STEPS samples=$MAX_SAMPLES maxnew=$MAX_NEW_TOKENS seed=$SEED run_date=$RUN_DATE"

failed=0
for dataset in $DATASETS; do
    for model in $MODELS; do
        for arm in $METHODS; do
            case "$arm" in
                baseline|power_mh) arm=baseline budget_list="-" ;;
                presto|subtree_prefetch) arm=presto budget_list=$BUDGETS ;;
                *) echo "Invalid method: $arm. Use baseline and/or presto." >&2; exit 2 ;;
            esac
            for budget in $budget_list; do
                for backend in $BACKENDS; do
                    [[ "$backend" == vllm || "$backend" == sglang ]] || {
                        echo "Invalid backend: $backend. Use vllm and/or sglang." >&2
                        exit 2
                    }
                    if [[ "$cluster" != interactive ]]; then
                        submit_cell "$backend" "$arm" "$dataset" "$model" "$budget"
                    elif ! run_cell "$backend" "$arm" "$dataset" "$model" "$budget"; then
                        echo "Run failed: $backend/$arm $dataset $model budget=$budget" >&2
                        failed=$((failed + 1))
                    fi
                done
            done
        done
    done
done
if (( failed > 0 )); then
    echo "$failed run(s) failed; see the per-run logs under $LOG_ROOT." >&2
    exit 1
fi
