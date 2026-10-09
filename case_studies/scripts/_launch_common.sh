#!/bin/bash
# Shared body of the case-study launchers in this directory and its *-family/ folders; do not run it directly. Each
# launcher sets its grid as d_* defaults (and optionally family=...) and then sources this file with its arguments:
#   source "$script_dir/_launch_common.sh" "$@"
# The first argument is the cluster: interactive | punakha | tacc. interactive runs the job bodies in this
# shell on an allocated GPU; the others sbatch them with that cluster's account, partition, and Conda environment.
#
# family selects the method pair and job body:
#   uniform     PowerMH vs PreSTO (run_power_mh_case_study.sh / run_subtree_prefetching_mh_case_study.sh)
#   entropycut  EntropyCut MH vs PreSTO + EntropyCutMH (EntropyCut-family/run_entropycut_case_study.sh)
#   multitry    MultiTryMH vs PreSTO + MultiTryMH (MultiTry-family/run_multitry_case_study.sh)
#   multitry-v2 the engine-scored v2 variants of the MultiTry pair
#
# Any of these environment variables overrides the launcher's default:
#   METHODS         "baseline presto" (either or both; power_mh and subtree_prefetch are accepted aliases)
#   DATASETS MODELS RANKS BUDGETS   space-separated lists; RANKS/BUDGETS apply to the presto arm only
#   ALPHA MCMC_STEPS NUM_BLOCKS MAX_SAMPLES MAX_NEW_TOKENS TEMPERATURE BATCH_SIZE PRINT_TREE
#   RESOURCE_PROBE  on | off | auto           KV_CACHE_MODE  hooks | inherit
#   EXTRA_OVERRIDES extra Python key=value settings, e.g. "dtype=bfloat16 max_model_len=8192"
#   JOB_PER         cell  - one job per dataset/model (baseline) and per dataset/model/rank/budget (presto)
#                   model - one job per dataset/model/arm that runs every rank/budget back to back
#   CUT_POWER                                        entropycut only (default 4.0)
#   LOG_ROOT        uniform/entropycut only: log tree instead of the family default, absolute or relative to the
#                   repository root (e.g. case_studies/logs_predictive_prefetching/EntropyCut-family)
#   NUM_TRIES PROPOSAL_TEMPERATURES SCORING_BATCH_SIZE   multitry only (scoring batch size is ignored by v2)
#   RUN_DATE (default today)  TIME (Slurm time limit)
#   PUNAKHA_MACHINE  TACC_PARTITION  TACC_ACCOUNT
#
# Per-run logs go to case_studies/<log tree>/<dataset>/<RUN_DATE>/vllm, where the log tree is logs, logs-entropycut, or
# logs-MultiTry, or to LOG_ROOT/<dataset>/<RUN_DATE>/vllm when it is set. Slurm output goes to case_studies/logs/all-datasets/<RUN_DATE>/vllm for the uniform family and to the
# per-dataset vllm/slurm folder for the others.

set -euo pipefail

repo_root=${REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
# env.sh supplies SEED and cache paths; it also turns on xtrace, which is noise here.
source "$repo_root/experiments/env.sh"
set +x
export REPO_ROOT="$repo_root"
scripts_dir="$repo_root/case_studies/scripts"
launcher_name=$(basename -- "$0")

# After the cluster, every override above may also be given on the command line, as KEY=value or --key value
# (--mcmc-steps 50 sets MCMC_STEPS=50). Unknown names stop the launch instead of being silently ignored.
allowed_settings=" METHODS DATASETS MODELS RANKS BUDGETS ALPHA MCMC_STEPS NUM_BLOCKS MAX_SAMPLES MAX_NEW_TOKENS
    TEMPERATURE BATCH_SIZE PRINT_TREE RESOURCE_PROBE KV_CACHE_MODE EXTRA_OVERRIDES JOB_PER LOG_ROOT CUT_POWER NUM_TRIES
    PROPOSAL_TEMPERATURES SCORING_BATCH_SIZE RUN_DATE TIME PUNAKHA_MACHINE TACC_PARTITION TACC_ACCOUNT "
allowed_settings=$(printf '%s' "$allowed_settings" | tr -s '[:space:]' ' ')
cluster=${1:-}
(( $# == 0 )) || shift
while (( $# > 0 )); do
    case "$1" in
        --*=*) key=${1%%=*} key=${key#--} value=${1#*=}; shift ;;
        --*)
            key=${1#--}
            if (( $# < 2 )); then
                echo "Missing value for $1" >&2
                exit 2
            fi
            value=$2
            shift 2
            ;;
        *=*) key=${1%%=*} value=${1#*=}; shift ;;
        *)
            echo "Unexpected argument: $1. Use KEY=value or --key value, e.g. MCMC_STEPS=50 or --models qwen3.5-9b." >&2
            exit 2
            ;;
    esac
    key=${key//-/_}
    key=${key^^}
    if [[ "$allowed_settings" != *" $key "* ]]; then
        echo "Unknown setting: $key. Known settings:$allowed_settings" >&2
        exit 2
    fi
    printf -v "$key" '%s' "$value"
done

# Defaults shared by every launcher unless it sets its own: alpha 4, 100 MH steps, one block, 20 samples, 1,024 new
# tokens. -1 selects proposal temperature 1/alpha; every family keeps the schedule constant.
family=${family:-uniform}
methods=${METHODS:-${d_methods:-baseline presto}}
datasets=${DATASETS:-${d_datasets:-lcb_v6}}
models=${MODELS:-${d_models:-qwen3.5-9b}}
ranks=${RANKS:-${d_ranks:-bfs_accept_first}}
budgets=${BUDGETS:-${d_budgets:-20}}
alpha=${ALPHA:-${d_alpha:-4.0}}
mcmc_steps=${MCMC_STEPS:-${d_mcmc_steps:-100}}
num_blocks=${NUM_BLOCKS:-${d_num_blocks:-1}}
max_samples=${MAX_SAMPLES:-${d_max_samples:-20}}
max_new_tokens=${MAX_NEW_TOKENS:-${d_max_new_tokens:-1024}}
temperature=${TEMPERATURE:--1}
batch_size=${BATCH_SIZE:-1}
print_tree=${PRINT_TREE:-${d_print_tree:-true}}
resource_probe=${RESOURCE_PROBE:-${d_resource_probe:-on}}
kv_cache_mode=${KV_CACHE_MODE:-${d_kv_cache_mode:-hooks}}
extra_overrides=${EXTRA_OVERRIDES:-${d_extra_overrides:-}}
job_per=${JOB_PER:-${d_job_per:-cell}}
log_root=${LOG_ROOT:-${d_log_root:-}}
run_date=${RUN_DATE:-$(date +%F)}

# Normalize the arm names so METHODS reads the same in every family.
arms=""
for method in $methods; do
    case "$method" in
        baseline|power_mh) arms+=" baseline" ;;
        presto|subtree_prefetch) arms+=" presto" ;;
        *)
            echo "Invalid method in METHODS: $method. Use baseline and/or presto." >&2
            exit 2
            ;;
    esac
done

# Family-specific job bodies, method selectors, extra job options, name tags, and output locations. The uniform family
# pins the cut distribution and schedule itself; the family entry points add and enforce their own.
name_suffix=samples${max_samples}.maxnew${max_new_tokens}.seed${SEED}
case "$family" in
    uniform)
        baseline_job="$scripts_dir/run_power_mh_case_study.sh"
        presto_job="$scripts_dir/run_subtree_prefetching_mh_case_study.sh"
        baseline_args=() presto_args=() family_args=()
        family_overrides=(temperature_schedule_type=const cut_dist_type=uniform)
        baseline_prefix=powermh presto_prefix=presto-powermh
        baseline_log=powerMH.vllm presto_log=subtreePrefetch.vllm
        log_tree=logs
        ;;
    entropycut)
        cut_power=${CUT_POWER:-${d_cut_power:-4.0}}
        baseline_job="$scripts_dir/EntropyCut-family/run_entropycut_case_study.sh"
        presto_job=$baseline_job
        baseline_args=(--method entropycut-mh) presto_args=(--method presto-entropycut-mh)
        family_args=(--cut-power "$cut_power")
        family_overrides=()
        baseline_prefix=entropycut-mh presto_prefix=presto-entropycut-mh
        baseline_log=powerMH.vllm presto_log=subtreePrefetch.vllm
        name_suffix=cut-entropy${cut_power}.${name_suffix}
        log_tree=logs-entropycut
        ;;
    multitry|multitry-v2)
        num_tries=${NUM_TRIES:-${d_num_tries:-4}}
        proposal_temperatures=${PROPOSAL_TEMPERATURES:-${d_proposal_temperatures:-[0.25,0.5,1.0]}}
        temperature_tag=$(printf '%s' "$proposal_temperatures" | tr -d '[][:space:]' | tr ',' '-')
        baseline_job="$scripts_dir/MultiTry-family/run_multitry_case_study.sh"
        presto_job=$baseline_job
        family_args=(--num-tries "$num_tries" --proposal-temperatures "$proposal_temperatures")
        family_overrides=()
        log_tree=logs-MultiTry
        if [[ "$family" == multitry ]]; then
            scoring_batch_size=${SCORING_BATCH_SIZE:-${d_scoring_batch_size:-128}}
            family_args+=(--scoring-batch-size "$scoring_batch_size")
            baseline_args=(--method multitry-mh) presto_args=(--method presto-multitry-mh)
            baseline_prefix=multitry-mh presto_prefix=presto-multitry-mh
            baseline_log=multiTryMH.vllm presto_log=subtreePrefetchMultiTryMH.vllm
            name_suffix=tries${num_tries}.temps${temperature_tag}.scorebatch${scoring_batch_size}.${name_suffix}
        else
            # v2 scores every mixture temperature during generation, so it has no scoring batch size.
            baseline_args=(--method multitry-mh-v2) presto_args=(--method presto-multitry-mh-v2)
            baseline_prefix=multitry-mh-v2 presto_prefix=presto-multitry-mh-v2
            baseline_log=multiTryMHv2.vllm presto_log=subtreePrefetchMultiTryMHv2.vllm
            name_suffix=tries${num_tries}.temps${temperature_tag}.v2.${name_suffix}
        fi
        ;;
    *)
        echo "Invalid family=$family. Use uniform, entropycut, multitry, or multitry-v2." >&2
        exit 2
        ;;
esac

# Cluster-specific sbatch options and Conda environment.
case "$cluster" in
    interactive)
        ;;
    punakha)
        # The punakha_<machine>_general QOS caps MaxWall at 1-00:00:00; longer requests are never scheduled.
        machine=${PUNAKHA_MACHINE:-dgx}
        sbatch_opts=(--account=punakha_general --qos="punakha_${machine}_general" --partition="$machine"
            --gres=gpu:1 --nodes=1 --ntasks=1 --time="${TIME:-${d_punakha_time:-20:00:00}}")
        conda_env=vllm-cuda130
        conda_init="$HOME/WORK/miniconda3/etc/profile.d/conda.sh"
        ;;
    tacc)
        sbatch_opts=(-p "${TACC_PARTITION:-gh}" -A "${TACC_ACCOUNT:-CCR25054}" -N 1 -n 1 -t "${TIME:-12:00:00}")
        conda_env=cuda130
        conda_init="$HOME/WORK/miniconda3/etc/profile.d/conda.sh"
        ;;
    *)
        echo "Usage: [VAR=value ...] bash $launcher_name <interactive|punakha|tacc>" >&2
        exit 2
        ;;
esac
if [[ -n "$log_root" && "$family" == multitry* ]]; then
    echo "LOG_ROOT is supported by the uniform and entropycut families only." >&2
    exit 2
fi
if [[ "$job_per" != cell && "$job_per" != model ]]; then
    echo "Invalid JOB_PER=$job_per. Use cell or model." >&2
    exit 2
fi

# MultiTry checks that each selected runner exists before anything is submitted.
if [[ "$family" == multitry* ]]; then
    [[ "$arms" != *baseline* ]] || bash "$baseline_job" "${baseline_args[@]}" --check-support
    [[ "$arms" != *presto* ]] || bash "$presto_job" "${presto_args[@]}" --check-support
fi

echo "$launcher_name family=$family cluster=$cluster arms=(${arms# }) datasets=($datasets) models=($models)" \
    "ranks=($ranks) budgets=($budgets) run_date=$run_date${log_root:+ log_root=$log_root}"

read -r -a extra_list <<< "$extra_overrides"
common_overrides=(seed="$SEED" alpha="$alpha" mcmc_steps="$mcmc_steps" num_blocks="$num_blocks"
    max_samples="$max_samples" max_new_tokens="$max_new_tokens" batch_size="$batch_size"
    temperature="$temperature" "${family_overrides[@]}" "${extra_list[@]}")

# Run or submit one job body. Arguments: job name, Slurm output stem, job script, then the job's own arguments, which
# must end with --override and the Python settings.
launch() {
    local job_name=$1 out_stem=$2 job_script=$3
    shift 3
    local command
    if [[ "$cluster" == interactive ]]; then
        command=(bash "$job_script" --log-mode tee --flashinfer-skip-version-check "$@")
    else
        mkdir -p "$slurm_dump_dir"
        command=(sbatch "${sbatch_opts[@]}" --job-name="$job_name" --output="$slurm_dump_dir/$out_stem.out"
            "$job_script" --continue-on-error --conda-env "$conda_env" --conda-init "$conda_init" "$@")
    fi
    echo "Launching $job_name"
    "${command[@]}"
}

for dataset_key in $datasets; do
    if [[ -n "$log_root" ]]; then
        [[ "$log_root" == /* ]] && log_dir=$log_root || log_dir=$repo_root/$log_root
        slurm_dump_dir="$log_dir/$dataset_key/$run_date/vllm/slurm"
    elif [[ "$family" == uniform ]]; then
        slurm_dump_dir="$repo_root/case_studies/logs/all-datasets/$run_date/vllm"
    else
        slurm_dump_dir="$repo_root/case_studies/$log_tree/$dataset_key/$run_date/vllm/slurm"
    fi
    for base_model in $models; do
        cell_args=("${family_args[@]}" --datasets "$dataset_key" --models "$base_model" --run-date "$run_date"
            --resource-probe "$resource_probe" --kv-cache-mode "$kv_cache_mode")
        [[ -z "$log_root" ]] || cell_args+=(--log-root "$log_root")
        name_prefix=dataset-${dataset_key}.model-${base_model}.alpha${alpha}.steps${mcmc_steps}.blocks-${num_blocks}

        if [[ "$arms" == *baseline* ]]; then
            launch "${baseline_prefix}-${dataset_key}-${base_model}" "${name_prefix}.${name_suffix}.${baseline_log}" \
                "$baseline_job" "${baseline_args[@]}" "${cell_args[@]}" --override "${common_overrides[@]}"
        fi

        [[ "$arms" == *presto* ]] || continue
        presto_overrides=("${common_overrides[@]}" print_tree="$print_tree")
        if [[ "$job_per" == model ]]; then
            launch "${presto_prefix}-${dataset_key}-${base_model}" \
                "${name_prefix}.prefetch-budget-sweep.rank-sweep.${name_suffix}.${presto_log}" \
                "$presto_job" "${presto_args[@]}" "${cell_args[@]}" --prefetch-budgets "$budgets" \
                --rank-fns "$ranks" --override "${presto_overrides[@]}"
            continue
        fi
        for rank_fn in $ranks; do
            for prefetch_budget in $budgets; do
                launch "${presto_prefix}-${dataset_key}-${base_model}-${rank_fn}-p${prefetch_budget}" \
                    "${name_prefix}.prefetch-budget-${prefetch_budget}.rank-${rank_fn}.${name_suffix}.${presto_log}" \
                    "$presto_job" "${presto_args[@]}" "${cell_args[@]}" --prefetch-budgets "$prefetch_budget" \
                    --rank-fns "$rank_fn" --override "${presto_overrides[@]}"
            done
        done
    done
done
