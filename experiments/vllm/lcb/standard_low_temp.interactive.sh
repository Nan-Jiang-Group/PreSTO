source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"

# Sampler hyperparameters live in src/power_sharpening/config/{dataset,algorithms}.yaml; this script selects the dataset
# + model and overrides only what it varies.

algorithm=low_temp
temperature=1.0
difficulty=all   # one of: all, easy, medium, hard

for lcb_version in release_v6;
do
dump_dir=$basepath/result/livecode_bench/${lcb_version}_${difficulty}/$(date +%F)
mkdir -p $dump_dir
echo "livecodebench dataset (${lcb_version}, difficulty=${difficulty})..."

for base_model in qwen3-4b qwen3-8b tulu phi3.5;
do
max_new_tokens=8192
run_name=model-${base_model}.algo-standard_low_temp.lcb-${lcb_version}.diff-${difficulty}.Temp-${temperature}.seed-${SEED}
python -m power_sharpening.runners.vllm.run_standard_low_temp \
  --dataset lcb \
  --algorithm $algorithm \
  --model_str=$base_model \
  --save_str $dump_dir \
  --run_name "$run_name" \
  --override seed=$SEED temperature=${temperature} max_new_tokens=${max_new_tokens}  lcb_version=${lcb_version} difficulty=${difficulty} \
  > $dump_dir/${run_name}.standard_low_temp.vllm.log 2>&1
done
done
