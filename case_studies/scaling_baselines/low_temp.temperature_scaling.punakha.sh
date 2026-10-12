# Low-temperature baseline: accuracy against sampling temperature (one sample per prompt, no search).
# Scaling axis: TEMPERATURES. One run per dataset x model x temperature, first MAX_SAMPLES prompts,
# one prompt at a time (BATCH_SIZE=1) so the timing is comparable across methods.
#
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scaling_baselines/low_temp.temperature_scaling.punakha.sh \
#       [interactive|tacc] [TEMPERATURES="0.1 0.25 0.5 0.75 1.0"] [DATASETS=lcb_v6] [MODELS=qwen3.5-9b] [MAX_SAMPLES=20] [BATCH_SIZE=1] [MAX_NEW_TOKENS=1024]
DATASETS=${DATASETS:-lcb_v6}
MODELS=${MODELS:-qwen3.5-9b}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
CLUSTER=${CLUSTER:-punakha}
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

temperatures=${TEMPERATURES:-0.1 0.25 0.5 0.75 1.0}
max_samples=${MAX_SAMPLES:-20}
batch_size=${BATCH_SIZE:-1}

for dataset in $datasets; do
    for model in $models; do
        for temperature in $temperatures; do
            launch "$dataset" low_temp \
                "dataset-$dataset.model-$model.low_temp.temp$temperature.samples$max_samples.maxnew$max_new_tokens.seed$seed" \
                power_sharpening.runners.vllm.run_low_temp \
                "temperature=$temperature" "max_samples=$max_samples" "batch_size=$batch_size"
        done
    done
done
