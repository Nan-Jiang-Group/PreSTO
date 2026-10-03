source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"


algorithm=power_mcmc
base_model=qwen-math-small
mcmc_steps=10
alpha=4.0

num_blocks=8
temperature_schedule_type=const
temp=0.25

dump_dir=$basepath/result/MATH500/$(date +%F)
mkdir -p $dump_dir
for temperature_schedule_type in cosine linear step const;
do
echo temperature_schedule_type $temperature_schedule_type
for mcmc_steps in 5 10 15 20 25 30;
do
echo temp is $temp
python -m power_sharpening.runners.hf.run_power_mh \
  --task math500 \
  --mcmc_steps=${mcmc_steps} \
  --algorithm ${algorithm} \
  --alpha $alpha \
  --num_blocks=${num_blocks} \
  --seed=${SEED} \
  --model_str=${base_model} \
  --temperature=${temp} \
  --save_str ${dump_dir} \
  --temperature_schedule_type ${temperature_schedule_type} > $dump_dir/model-${base_model}.algo-${algorithm}.InitTemp-${temp}.alpha-${alpha}.steps-${mcmc_steps}.blocks-${num_blocks}.seed-${SEED}.temp_sched_type-${temperature_schedule_type}.powerSharp.hf.log 2>&1
done
done
