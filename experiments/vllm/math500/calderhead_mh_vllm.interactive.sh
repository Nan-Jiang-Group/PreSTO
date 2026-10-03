source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"


base_model=qwen-math-medium
mcmc_steps=10
alpha=4.0

num_blocks=16
num_proposals=4
temperature_schedule_type=cosine
batch_size=500

dump_dir=$basepath/result/MATH500/$(date +%F)
mkdir -p $dump_dir
for temperature_schedule_type in const cosine linear step;
do
echo temperature_schedule_type $temperature_schedule_type
for temp in 0.1 0.15 0.2 0.25 0.3 0.35 0.4;
do
echo temp is $temp
python -m power_sharpening.runners.vllm.run_calderhead_mh \
  --dataset math500 \
  --model_str=${base_model} \
  --save_str $dump_dir \
  --override seed=${SEED} mcmc_steps=${mcmc_steps} alpha=${alpha} num_blocks=${num_blocks} num_proposals=${num_proposals} batch_size=${batch_size} temperature=${temp} temperature_schedule_type=${temperature_schedule_type} #> $dump_dir/model-${base_model}.algo-calderhead_power_mcmc.InitTemp-${temp}.alpha-${alpha}.steps-${mcmc_steps}.blocks-${num_blocks}.num_proposals-${num_proposals}.seed-${SEED}.temp_sched_type-${temperature_schedule_type}.powerSharp.vllm.log 2>&1
done
done
