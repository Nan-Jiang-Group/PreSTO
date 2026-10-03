#!/bin/bash
# PowerMH job body for the launchers in this directory. Run it through a sibling interactive or sbatch launcher, which
# supplies the experiment arguments. Pass --job-name and --output to sbatch: #SBATCH does not expand shell variables.

set -eo pipefail

# Slurm runs a spooled copy of this script, so its location cannot identify the checkout: the launchers export
# REPO_ROOT, and a direct run outside Slurm falls back to the checkout holding this script.
repo_root=${REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}

# The launcher supplies the experiment grid through named options.
datasets=""
models=""
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
# An empty log root keeps the default tree (logs, or logs-entropycut for entropy cuts).
log_root_override=""
continue_on_error=false
flashinfer_skip_version_check=false

# Read one option at a time. Boolean flags have no value; --override ends launcher options and leaves the remaining
# arguments for the Python runner.
while (( $# > 0 )); do
    option="$1"
    shift

    case "$option" in
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
        --datasets)
            datasets="$value"
            ;;
        --models)
            models="$value"
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
        --log-root)
            log_root_override="$value"
            ;;
        *)
            echo "Unknown job argument: $option" >&2
            exit 2
            ;;
    esac
done

# Check required inputs and mode names before loading Conda or starting a run.
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
cut_dist_type=uniform cut_power=4.0
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
        cut_dist_type=*) cut_dist_type=${setting#*=} ;;
        cut_power=*) cut_power=${setting#*=} ;;
    esac
done
[[ -n "$seed" && -n "$alpha" && -n "$mcmc_steps" && -n "$num_blocks" &&
   -n "$max_samples" && -n "$max_new_tokens" ]] || {
    echo "Overrides must include seed, alpha, mcmc_steps, num_blocks, max_samples, and max_new_tokens." >&2
    exit 2
}
# Turn quoted, space-separated lists into arrays for the sweep loops below.
read -r -a dataset_list <<< "$datasets"
read -r -a model_list <<< "$models"

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
    # Keep EntropyCut logs/resources separate from otherwise identical uniform runs.
    if [[ "$cut_dist_type" == entropy ]]; then
        run_name+=".cut-entropy${cut_power}"
    fi
    run_name+=".samples${max_samples}.maxnew${max_new_tokens}"
    [[ -z "$max_model_len" ]] || run_name+=".maxmodel${max_model_len}"
    run_name+=".seed${seed}"
    local run_log="$dump_dir/${run_name}.powerMH.vllm.log"
    local command=(python -m power_sharpening.runners.vllm.run_power_mh
        --dataset "$dataset_key" --algorithm power_mcmc
        --model_str="$base_model" --save_str "$dump_dir" --run_name "$run_name")
    # In auto mode, omit the flag. Python overrides can still set resource_probe.
    case "$resource_probe" in
        on) command+=(--resource_probe) ;;
        off) command+=(--no-resource_probe) ;;
    esac
    command+=(--override "${overrides[@]}")
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

# Entropy-cut runs keep their logs and Python artifacts in a separate tree.
log_root="$repo_root/case_studies/logs"
if [[ "$cut_dist_type" == entropy ]]; then
    log_root="$repo_root/case_studies/logs-entropycut"
fi
# --log-root sends a launcher's runs to its own tree; relative paths start at the repository root.
if [[ -n "$log_root_override" ]]; then
    [[ "$log_root_override" == /* ]] || log_root_override="$repo_root/$log_root_override"
    log_root="$log_root_override"
fi

# Stop at the first failure by default. --continue-on-error finishes the sweep and still returns a nonzero exit status
# if any run failed.
failed=0
for dataset_key in "${dataset_list[@]}"; do
    dump_dir="$log_root/$dataset_key/$run_date/vllm"
    mkdir -p "$dump_dir"
    for base_model in "${model_list[@]}"; do
        if ! run_one; then
            failed=$((failed + 1))
            [[ "$continue_on_error" == true ]] || exit 1
        fi
    done
done
if [[ "$failed" -gt 0 ]]; then
    echo "$failed run(s) failed; see the per-run logs." >&2
    exit 1
fi
