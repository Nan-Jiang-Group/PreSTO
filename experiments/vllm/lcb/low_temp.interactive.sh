source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"

# Sampler hyperparameters live in src/power_sharpening/config/{dataset,algorithms}.yaml; this script selects the dataset
# + model and overrides only what it varies.

algorithm=best_of_n
base_model=qwen-coder-instruct

batch_size=880

lcb_version=release_v6
difficulty=all   # one of: all, easy, medium, hard

dump_dir=$basepath/result/livecode_bench/${lcb_version}_${difficulty}/$(date +%F)
mkdir -p $dump_dir
echo "livecodebench dataset (${lcb_version}, difficulty=${difficulty})..."
best_of_N="4 2 1"

#for temp in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
#do
#  echo temp is $temp
temp=0.6
python -m power_sharpening.runners.vllm.run_low_temp \
  --dataset lcb \
  --algorithm $algorithm \
  --model_str=$base_model \
  --save_str $dump_dir \
  --override seed=$SEED batch_size=${batch_size} temperature=${temp} "best_of_N=[${best_of_N// /,}]" lcb_version=${lcb_version} difficulty=${difficulty} \
  > $dump_dir/model-${base_model}.algo-${algorithm}.lcb-${lcb_version}.diff-${difficulty}.InitTemp-${temp}.best-of-N${best_of_N// /_}.seed-${SEED}.low_temp.vllm.log 2>&1

#done
