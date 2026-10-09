# Total sampling time of the low-temperature baseline, best-of-N at low temperature, PowerMH, and PreSTO-PowerMH, all on
# the standard configuration: 1,024 new tokens, 8 blocks, 10 MH steps per block, alpha 4, proposal temperature 0.25
# (= 1/alpha), one prompt at a time, the same first MAX_SAMPLES prompts. One run per dataset x model x method (best-of-N:
# one run per N, so each run's time is the cost of that N alone). METHODS=power_smc adds Power-SMC, one run per particle
# count in N_PARTICLES (1 2 4 8 16 32 64 96 128); power_smc_standard runs the same grid on stock vLLM
# (run_standard_smc.py, base-model log p from a prompt-logprob rescoring pass). Neither is in the default METHODS.
#
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scaling_baselines/total_time.standard_config.punakha.sh \
#       [interactive|tacc] [METHODS="low_temp best_of_n power_mh presto"] [BEST_OF_N="4 8 16 32"] [N_PARTICLES=...] \
#       [DATASETS=lcb_v6] [MODELS=qwen3.5-9b] [MAX_SAMPLES=20] [RANKS=bfs_accept_first] [BUDGETS=10]
#
# Other settings: MAX_NEW_TOKENS (1024), MCMC_STEPS (10), NUM_BLOCKS (8), ALPHA (4.0), TEMPERATURE (0.25), BATCH_SIZE
# (1), LOG_ROOT. PowerMH and PreSTO go through case_studies/scripts/_launch_common.sh with tree printing off.
#
# KV-cache usage: RESOURCE_PROBE (on | off, default on) and KV_CACHE_MODE (hooks | inherit, default hooks) apply to all
# arms alike, so every arm pays the same measurement cost. hooks runs the vLLM engine in-process
# (VLLM_ENABLE_V1_MULTIPROCESSING=0) for exact peak KV occupancy and eviction counts; inherit keeps the engine in a
# child process, where occupancy is polled and eviction is unavailable.
#
# Every run writes <run_name>.csv, <run_name>.resources.json (KV cache, prefix cache, GPU memory), and its .log under
# case_studies/scaling_baselines/logs_total_time/<dataset>/<date>/vllm.
# Tabulate the times with:
#   python -m case_studies.scaling_baselines.collect_total_time
DATASETS=${DATASETS:-lcb_v6}
MODELS=${MODELS:-qwen3.5-9b}
CLUSTER=${CLUSTER:-punakha}
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

methods=${METHODS:-low_temp best_of_n power_mh presto}
best_of_n=${BEST_OF_N:-4 8 16 32}
n_particles=${N_PARTICLES:-1 2 4 8 16 32 64 96 128}
max_samples=${MAX_SAMPLES:-20}
max_new_tokens=${MAX_NEW_TOKENS:-1024}
mcmc_steps=${MCMC_STEPS:-10}
num_blocks=${NUM_BLOCKS:-8}
alpha=${ALPHA:-4.0}
temperature=${TEMPERATURE:-0.25}
batch_size=${BATCH_SIZE:-1}
resource_probe=${RESOURCE_PROBE:-on}
kv_cache_mode=${KV_CACHE_MODE:-hooks}
log_root=${LOG_ROOT:-$scaling_dir/logs_total_time}
log_leaf=/vllm
tag=samples$max_samples.maxnew$max_new_tokens.seed$seed

for method in $methods; do
    case $method in
        low_temp|best_of_n|power_smc|power_smc_standard|power_mh|presto) ;;
        *) echo "Unknown method: $method. Use low_temp, best_of_n, power_smc, power_smc_standard, power_mh, presto." >&2
           exit 2 ;;
    esac
done

case $resource_probe in
    on) probe_overrides=() ;;
    off) probe_overrides=(resource_probe=false) ;;
    *) echo "Invalid RESOURCE_PROBE=$resource_probe. Use on or off." >&2; exit 2 ;;
esac
case $kv_cache_mode in
    # The same engine environment the MH job scripts set; sbatch passes the exported environment to the job.
    hooks) export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1 ;;
    inherit) ;;
    *) echo "Invalid KV_CACHE_MODE=$kv_cache_mode. Use hooks or inherit." >&2; exit 2 ;;
esac

for dataset in $datasets; do
    for model in $models; do
        if [[ " $methods " == *" low_temp "* ]]; then
            launch "$dataset" low_temp "dataset-$dataset.model-$model.low_temp.temp$temperature.$tag" \
                power_sharpening.runners.vllm.run_low_temp \
                "temperature=$temperature" "max_samples=$max_samples" "batch_size=$batch_size" \
                "${probe_overrides[@]}"
        fi
        if [[ " $methods " == *" best_of_n "* ]]; then
            for n in $best_of_n; do
                launch "$dataset" best_of_n "dataset-$dataset.model-$model.best_of_n.temp$temperature.n$n.$tag" \
                    power_sharpening.runners.vllm.run_low_temp \
                    "temperature=$temperature" "best_of_N=[$n]" "max_samples=$max_samples" "batch_size=$batch_size" \
                    "${probe_overrides[@]}"
            done
        fi
        if [[ " $methods " == *" power_smc "* ]]; then
            for particles in $n_particles; do
                launch "$dataset" smc "dataset-$dataset.model-$model.power_smc.nparticles$particles.$tag" \
                    power_sharpening.runners.vllm.run_power_smc \
                    "n_particles=$particles" "alpha=$alpha" "temperature=$temperature" "max_samples=$max_samples" \
                    "batch_size=$batch_size" "${probe_overrides[@]}"
            done
        fi
        if [[ " $methods " == *" power_smc_standard "* ]]; then
            for particles in $n_particles; do
                launch "$dataset" smc "dataset-$dataset.model-$model.power_smc_standard.nparticles$particles.$tag" \
                    power_sharpening.runners.vllm.run_standard_smc \
                    "n_particles=$particles" "alpha=$alpha" "temperature=$temperature" "max_samples=$max_samples" \
                    "batch_size=$batch_size" "${probe_overrides[@]}"
            done
        fi
    done
done

# PowerMH (baseline arm) and PreSTO (presto arm) through the case-study launcher. It runs in a subshell because it sets
# its own shell options and variables, and it reads METHODS with its own arm names.
mh_arms=""
[[ " $methods " == *" power_mh "* ]] && mh_arms+=" baseline"
[[ " $methods " == *" presto "* ]] && mh_arms+=" presto"
if [[ -n $mh_arms ]]; then
    (
        unset METHODS
        export LOG_ROOT=$log_root DATASETS=$datasets MODELS=$models MAX_SAMPLES=$max_samples
        export MAX_NEW_TOKENS=$max_new_tokens MCMC_STEPS=$mcmc_steps NUM_BLOCKS=$num_blocks ALPHA=$alpha
        export TEMPERATURE=$temperature BATCH_SIZE=$batch_size RESOURCE_PROBE=$resource_probe
        export KV_CACHE_MODE=$kv_cache_mode
        d_methods=$mh_arms
        d_ranks=bfs_accept_first
        d_budgets=10
        d_print_tree=false
        d_job_per=cell
        source "$repo_root/case_studies/scripts/_launch_common.sh" "$cluster"
    )
fi
