# Best-of-N baseline: accuracy against the number of samples N, keeping the candidate with the highest sequence
# log-probability. Scaling axis: BEST_OF_N. The runner draws max(BEST_OF_N) samples once and reuses the first n of
# them for every n, so one run covers the whole ladder. One run per dataset x model x temperature, first MAX_SAMPLES prompts,
# one prompt at a time (BATCH_SIZE=1) so the timing is comparable across methods.
#
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scaling_baselines/best_of_n.sample_scaling.punakha.sh \
#       [interactive|tacc] [BEST_OF_N="1 2 4 8 16 32 64"] [TEMPERATURES="1.0"] [DATASETS=lcb_v6] [MODELS=qwen3.5-9b] [MAX_SAMPLES=20] [BATCH_SIZE=1] [MAX_NEW_TOKENS=1024]
#
# The sampling temperature sets how different the candidates are; at 0.25 (= 1/alpha) they are nearly identical, so the
# default is 1.0. Pass TEMPERATURES="0.25 1.0" to see both.
DATASETS=${DATASETS:-lcb_v6}
MODELS=${MODELS:-qwen3.5-9b}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
CLUSTER=${CLUSTER:-punakha}
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

best_of_n=${BEST_OF_N:-1 2 4 8 16 32 64}
temperatures=${TEMPERATURES:-1.0}
max_samples=${MAX_SAMPLES:-20}
batch_size=${BATCH_SIZE:-1}
ladder="[${best_of_n// /,}]"

for dataset in $datasets; do
    for model in $models; do
        for temperature in $temperatures; do
            launch "$dataset" best_of_n \
                "dataset-$dataset.model-$model.best_of_n.temp$temperature.nmax${best_of_n##* }.samples$max_samples.maxnew$max_new_tokens.seed$seed" \
                power_sharpening.runners.vllm.run_low_temp \
                "temperature=$temperature" "best_of_N=$ladder" "max_samples=$max_samples" "batch_size=$batch_size"
        done
    done
done
