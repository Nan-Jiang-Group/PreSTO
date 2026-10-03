source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"

# Sampler hyperparameters live in src/power_sharpening/config/{dataset,algorithms}.yaml; this script selects the dataset
# + model and overrides only what it varies.
#
# NOTE: dataset.yaml defines a `swebench` entry, but no SWEBench benchmark class
# is implemented/registered in power_sharpening.tasks yet — the runner exits with "unsupported task" until one is added.

algorithm=best_of_n
base_model=qwen-coder

batch_size=500

swebench_version=lite   # one of: lite, verified, test

dump_dir=$basepath/result/swebench_${swebench_version}/$(date +%F)
mkdir -p $dump_dir
echo "SWEBench dataset (version=${swebench_version})..."
for best_of_N in "4 2 1" "128 64 32 16 8 4 2 1" \
"192 128 64 32 16 8 2 1" ;
do
for temp in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
do
  echo temp is $temp

  python -m power_sharpening.runners.vllm.run_low_temp \
  --dataset swebench \
  --algorithm $algorithm \
  --model_str=$base_model \
  --save_str $dump_dir \
  --override seed=$SEED batch_size=${batch_size} temperature=${temp} "best_of_N=[${best_of_N// /,}]" swebench_version=${swebench_version} #> $dump_dir/model-${base_model}.algo-${algorithm}.swebench-${swebench_version}.InitTemp-${temp}.best-of-N${best_of_N%% *}.seed-${SEED}.low_temp.vllm.log 2>&1
done
done
