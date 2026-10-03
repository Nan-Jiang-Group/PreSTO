source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"



algorithm=power_mcmc
base_model=qwen-coder-instruct

batch_size=200

lcb_version=release_v6
difficulty=all   # one of: all, easy, medium, hard

dump_dir=$basepath/result/livecode_bench/${lcb_version}_${difficulty}/$(date +%F)
mkdir -p $dump_dir
echo "livecodebench dataset (${lcb_version}, difficulty=${difficulty})..."

alpha=4.0
mcmc_steps=10
num_blocks=16
schedule=const
temp=None

for alpha in 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5;
do
python -m power_sharpening.runners.vllm.run_power_mh \
  --dataset lcb \
  --algorithm $algorithm \
  --model_str=$base_model \
  --save_str $dump_dir \
  --override seed=$SEED batch_size=${batch_size} alpha=${alpha} mcmc_steps=${mcmc_steps} num_blocks=${num_blocks} temperature_schedule_type=${schedule} lcb_version=${lcb_version} difficulty=${difficulty} \
  > $dump_dir/model-${base_model}.algo-${algorithm}.lcb-${lcb_version}.diff-${difficulty}.InitTemp-${temp}.alpha-${alpha}.mcmc-${mcmc_steps}.blocks-${num_blocks}.sched-${schedule}.seed-${SEED}.mh.vllm.log 2>&1

done
