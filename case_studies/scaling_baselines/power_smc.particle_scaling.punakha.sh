# PowerSMC: accuracy against the number of SMC particles per prompt (alpha, block size, and ESS threshold from
# algorithms.yaml). Scaling axis: N_PARTICLES. One run per dataset x model x particle count, first MAX_SAMPLES prompts,
# one prompt at a time (BATCH_SIZE=1) so the timing is comparable across methods.
#
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scaling_baselines/power_smc.particle_scaling.punakha.sh \
#       [interactive|tacc] [N_PARTICLES="1 2 4 8 16 32"] [DATASETS=lcb_v6] [MODELS=qwen3.5-9b] [MAX_SAMPLES=20] [BATCH_SIZE=1] [MAX_NEW_TOKENS=1024]
DATASETS=${DATASETS:-lcb_v6}
MODELS=${MODELS:-qwen3.5-9b}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
CLUSTER=${CLUSTER:-punakha}
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

n_particles=${N_PARTICLES:-1 2 4 8 16 32}
max_samples=${MAX_SAMPLES:-20}
batch_size=${BATCH_SIZE:-1}

for dataset in $datasets; do
    for model in $models; do
        for particles in $n_particles; do
            launch "$dataset" smc \
                "dataset-$dataset.model-$model.power_smc.nparticles$particles.samples$max_samples.maxnew$max_new_tokens.seed$seed" \
                power_sharpening.runners.vllm.run_power_smc \
                "n_particles=$particles" "max_samples=$max_samples" "batch_size=$batch_size"
        done
    done
done
