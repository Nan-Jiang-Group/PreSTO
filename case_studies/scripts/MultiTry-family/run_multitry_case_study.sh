#!/bin/bash
# Shared vLLM MultiTryMH job body. Run through a caller in this directory.
# --method selects multitry-mh, presto-multitry-mh, or their engine-scored -v2 variants; --override forwards settings.
# Pass --job-name and --output to sbatch: #SBATCH does not expand shell variables.

set -eo pipefail

# Slurm runs a spooled copy of this script, so its location cannot identify the checkout: the launchers export
# REPO_ROOT, and a direct run outside Slurm falls back to the checkout holding this script.
repo_root=${REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}

# The launcher supplies the experiment grid through named options.
method=""
num_tries=4
proposal_temperatures="[0.25,0.5,1.0]"
scoring_batch_size=128
datasets=""
models=""
prefetch_budgets=""
rank_fns=""
run_date=""
# An empty environment name uses the Python environment already active.
conda_env=""
conda_init="$HOME/WORK/miniconda3/etc/profile.d/conda.sh"
# inherit preserves the vLLM process setting; hooks enables exact KV-cache hooks.
kv_cache_mode=inherit
# auto uses the Python runner default; on/off passes an explicit measurement flag.
resource_probe=auto
# file writes the run log; tee also displays it in the terminal or Slurm output.
log_mode=file
continue_on_error=false
flashinfer_skip_version_check=false
check_support=false

# Read one option at a time. Boolean flags have no value; --override ends launcher options and leaves the remaining
# arguments for the Python runner.
while (( $# > 0 )); do
    option="$1"
    shift

    case "$option" in
        --check-support)
            check_support=true
            continue
            ;;
        --continue-on-error)
            continue_on_error=true
            continue
            ;;
        --flashinfer-skip-version-check)
            flashinfer_skip_version_check=true
            continue
            ;;
        --override)
            break
            ;;
    esac

    # All other options need a value, for example: --datasets "math500 lcb_v6".
    if [[ $# -eq 0 || -z "$1" || "$1" == --* ]]; then
        echo "Missing value for $option" >&2
        exit 2
    fi
    value="$1"
    shift

    case "$option" in
        --method)
            method="$value"
            ;;
        --num-tries)
            num_tries="$value"
            ;;
        --proposal-temperatures)
            proposal_temperatures="$value"
            ;;
        --scoring-batch-size)
            scoring_batch_size="$value"
            ;;
        --datasets)
            datasets="$value"
            ;;
        --models)
            models="$value"
            ;;
        --prefetch-budgets)
            prefetch_budgets="$value"
            ;;
        --rank-fns)
            rank_fns="$value"
            ;;
        --run-date)
            run_date="$value"
            ;;
        --conda-env)
            conda_env="$value"
            ;;
        --conda-init)
            conda_init="$value"
            ;;
        --kv-cache-mode)
            kv_cache_mode="$value"
            ;;
        --resource-probe)
            resource_probe="$value"
            ;;
        --log-mode)
            log_mode="$value"
            ;;
        *)
            echo "Unknown job argument: $option" >&2
            exit 2
            ;;
    esac
done

# Check required inputs and mode names before loading Conda or starting a run.
# presto marks the subtree-prefetching arms; v2 marks engine-cached proposal scores (no scoring requests).
presto=false
v2=false
case "$method" in
    multitry-mh)
        runner_name=run_multi_try_mh
        algorithm=multi_try
        ;;
    # step-multitry-mh is the former name, still accepted so jobs spooled before the rename keep dispatching.
    presto-multitry-mh|step-multitry-mh)
        method=presto-multitry-mh
        runner_name=run_subtree_prefetching_multi_try_mh
        algorithm=subtree_prefetching_multi_try_mh
        presto=true
        ;;
    multitry-mh-v2)
        runner_name=run_multi_try_mh_v2
        algorithm=multi_try_v2
        v2=true
        ;;
    presto-multitry-mh-v2)
        runner_name=run_subtree_prefetching_multi_try_mh_v2
        algorithm=subtree_prefetching_multi_try_mh_v2
        presto=true
        v2=true
        ;;
    *)
        echo "Supply --method multitry-mh, presto-multitry-mh, multitry-mh-v2, or presto-multitry-mh-v2." >&2
        exit 2
        ;;
esac
if [[ ! -f "$repo_root/src/power_sharpening/runners/vllm/$runner_name.py" ]]; then
    echo "Missing vLLM runner: $repo_root/src/power_sharpening/runners/vllm/$runner_name.py" >&2
    exit 2
fi
if [[ "$check_support" == true ]]; then
    exit 0
fi
if [[ ! "$num_tries" =~ ^[1-9][0-9]*$ || ! "$scoring_batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-tries and --scoring-batch-size must be positive integers." >&2
    exit 2
fi
# Use the same temperature tag in run artifacts and Slurm output filenames.
temperature_tag=$(printf '%s' "$proposal_temperatures" | tr -d '[][:space:]' | tr ',' '-')
if [[ -z "$temperature_tag" || "$temperature_tag" == *[!0-9.eE+-]* ]]; then
    echo "--proposal-temperatures must be an inline numeric list, e.g. [0.25,0.5,1.0]." >&2
    exit 2
fi
if [[ -z "$datasets" || -z "$models" || -z "$run_date" ]]; then
    echo "Supply --datasets, --models, and --run-date." >&2
    exit 2
fi
if [[ ! -d "$repo_root/src/power_sharpening" ]]; then
    echo "Not a repository checkout (set REPO_ROOT): $repo_root" >&2
    exit 2
fi

if [[ "$kv_cache_mode" != hooks && "$kv_cache_mode" != inherit ]]; then
    echo "Invalid KV-cache mode: $kv_cache_mode. Use hooks or inherit." >&2
    exit 2
fi
if [[ "$resource_probe" != auto && "$resource_probe" != on && "$resource_probe" != off ]]; then
    echo "Invalid resource-probe mode: $resource_probe. Use auto or on or off." >&2
    exit 2
fi
if [[ "$log_mode" != file && "$log_mode" != tee ]]; then
    echo "Invalid log mode: $log_mode. Use file or tee." >&2
    exit 2
fi

# Forward everything after --override as key=value settings, such as alpha=4.0. Read the fields used in output paths;
# ${setting#*=} removes the "key=" prefix.
overrides=("$@")
seed="" alpha="" mcmc_steps="" num_blocks="" max_samples="" max_new_tokens=""
dtype="" max_model_len=""
for setting in "${overrides[@]}"; do
    case "$setting" in
        seed=*) seed=${setting#*=} ;;
        alpha=*) alpha=${setting#*=} ;;
        mcmc_steps=*) mcmc_steps=${setting#*=} ;;
        num_blocks=*) num_blocks=${setting#*=} ;;
        max_samples=*) max_samples=${setting#*=} ;;
        max_new_tokens=*) max_new_tokens=${setting#*=} ;;
        dtype=*) dtype=${setting#*=} ;;
        max_model_len=*) max_model_len=${setting#*=} ;;
        num_tries=*|proposal_temperatures=*|scoring_batch_size=*)
            echo "Use --num-tries, --proposal-temperatures, and --scoring-batch-size before --override." >&2
            exit 2
            ;;
        cut_dist_type=*)
            if [[ "$setting" != cut_dist_type=uniform ]]; then
                echo "MultiTryMH uses cut_dist_type=uniform." >&2
                exit 2
            fi
            ;;
        temperature_schedule_type=*)
            if [[ "$setting" != temperature_schedule_type=const ]]; then
                echo "MultiTry comparisons require temperature_schedule_type=const." >&2
                exit 2
            fi
            ;;
        prefetch_budget=*|rank_fn=*)
            echo "Use --prefetch-budgets and --rank-fns before --override." >&2
            exit 2
            ;;
        algorithm=*|method=*|save_str=*|run_name=*)
            echo "The launcher controls method selection and output paths: $setting" >&2
            exit 2
            ;;
    esac
done
[[ -n "$seed" && -n "$alpha" && -n "$mcmc_steps" && -n "$num_blocks" &&
   -n "$max_samples" && -n "$max_new_tokens" ]] || {
    echo "Overrides must include seed, alpha, mcmc_steps, num_blocks, max_samples, and max_new_tokens." >&2
    exit 2
}
if [[ "$presto" == true ]]; then
    if [[ -z "$prefetch_budgets" || -z "$rank_fns" ]]; then
        echo "Supply --prefetch-budgets and --rank-fns." >&2
        exit 2
    fi
elif [[ -n "$prefetch_budgets" || -n "$rank_fns" ]]; then
    echo "--prefetch-budgets and --rank-fns apply only to the presto-multitry-mh methods." >&2
    exit 2
fi
overrides+=(num_tries="$num_tries" proposal_temperatures="$proposal_temperatures"
    cut_dist_type=uniform temperature_schedule_type=const)
# v2 scores every mixture temperature during generation, so it has no scoring batches to size.
[[ "$v2" == true ]] || overrides+=(scoring_batch_size="$scoring_batch_size")
# Turn quoted, space-separated lists into arrays for the sweep loops below.
read -r -a dataset_list <<< "$datasets"
read -r -a model_list <<< "$models"
read -r -a budget_list <<< "$prefetch_budgets"
read -r -a rank_list <<< "$rank_fns"
if (( ${#dataset_list[@]} == 0 || ${#model_list[@]} == 0 )); then
    echo "--datasets and --models must each contain at least one entry." >&2
    exit 2
fi
if [[ "$presto" == true ]]; then
    if (( ${#budget_list[@]} == 0 || ${#rank_list[@]} == 0 )); then
        echo "--prefetch-budgets and --rank-fns must each contain at least one entry." >&2
        exit 2
    fi
    for budget in "${budget_list[@]}"; do
        if [[ ! "$budget" =~ ^[1-9][0-9]*$ ]] || (( budget < num_tries )); then
            echo "Each prefetch budget counts suffix requests and must be >= num_tries ($num_tries)." >&2
            exit 2
        fi
    done
fi

hostname
if [[ -n "$conda_env" ]]; then
    source "$conda_init"
    conda activate "$conda_env"
fi

# Avoid glibc TLS-loader races and preserve early startup diagnostics.
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
if [[ "$kv_cache_mode" == hooks ]]; then
    # In-process EngineCore exposes exact scheduler/block-pool measurements.
    export VLLM_ENABLE_V1_MULTIPROCESSING=0
fi
if [[ "$flashinfer_skip_version_check" == true ]]; then
    export FLASHINFER_DISABLE_VERSION_CHECK=1
fi
# Enable unset-variable checks after Conda initialization.
set -u
cd "$repo_root"

# Run the current loop combination in a fresh Python process. The run name is shared by its log and the artifacts
# written by the Python runner.
run_one() {
    local run_name="dataset-${dataset_key}.model-${base_model}"
    [[ -z "$dtype" ]] || run_name+=".dtype-${dtype}"
    run_name+=".alpha${alpha}.steps${mcmc_steps}.blocks-${num_blocks}"
    local log_suffix=multiTryMH
    if [[ "$presto" == true ]]; then
        run_name+=".prefetch-budget-${prefetch_budget}.rank-${rank_fn}"
        log_suffix=subtreePrefetchMultiTryMH
    fi
    run_name+=".tries${num_tries}.temps${temperature_tag}"
    if [[ "$v2" == true ]]; then
        run_name+=".v2"
        log_suffix+=v2
    else
        run_name+=".scorebatch${scoring_batch_size}"
    fi
    log_suffix+=.vllm
    run_name+=".samples${max_samples}.maxnew${max_new_tokens}"
    [[ -z "$max_model_len" ]] || run_name+=".maxmodel${max_model_len}"
    run_name+=".seed${seed}"
    local run_log="$dump_dir/${run_name}.${log_suffix}.log"
    local command=(python -m "power_sharpening.runners.vllm.$runner_name"
        --algorithm "$algorithm" --dataset "$dataset_key"
        --model_str="$base_model" --save_str "$dump_dir" --run_name "$run_name")
    # In auto mode, omit the flag. Python overrides can still set resource_probe.
    case "$resource_probe" in
        on) command+=(--resource_probe) ;;
        off) command+=(--no-resource_probe) ;;
    esac
    command+=(--override "${overrides[@]}")
    # These values come from the current budget/rank combination in the sweep.
    if [[ "$presto" == true ]]; then
        command+=(prefetch_budget="$prefetch_budget" rank_fn="$rank_fn")
    fi
    echo "Running $run_name"
    # pipefail makes a Python failure visible even when tee itself succeeds.
    if [[ "$log_mode" == tee ]]; then
        if ! "${command[@]}" 2>&1 | tee "$run_log"; then
            echo "Run failed; see $run_log" >&2
            return 1
        fi
    elif ! "${command[@]}" > "$run_log" 2>&1; then
        echo "Run failed; see $run_log" >&2
        return 1
    fi
}

# Every method writes its logs and Python artifacts into the MultiTry tree.
log_root="$repo_root/case_studies/logs-MultiTry"
# A baseline has no subtree grid; one sentinel pair executes it once per model.
if [[ "$presto" == false ]]; then
    budget_list=(none)
    rank_list=(none)
fi

# Stop at the first failure by default. --continue-on-error finishes the sweep and still returns a nonzero exit status
# if any run failed.
failed=0
for dataset_key in "${dataset_list[@]}"; do
    dump_dir="$log_root/$dataset_key/$run_date/vllm"
    mkdir -p "$dump_dir"
    for base_model in "${model_list[@]}"; do
        for prefetch_budget in "${budget_list[@]}"; do
            for rank_fn in "${rank_list[@]}"; do
                if ! run_one; then
                    failed=$((failed + 1))
                    [[ "$continue_on_error" == true ]] || exit 1
                fi
            done
        done
    done
done
if [[ "$failed" -gt 0 ]]; then
    echo "$failed run(s) failed; see the per-run logs." >&2
    exit 1
fi
