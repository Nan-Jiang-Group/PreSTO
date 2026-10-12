#!/bin/zsh
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"

algorithm=power_mcmc
base_model=qwen-math-small
mcmc_steps=10
alpha=4.0
temp=0.25
num_blocks=8
temperature_schedule_type=const

dump_dir=$basepath/result/MATH500/$(date +%F)
mkdir -p $dump_dir

python -m power_sharpening.runners.hf.run_power_mh \
  --task math500 \
  --mcmc_steps=${mcmc_steps} \
  --alpha=${alpha} \
  --temperature=${temp} \
  --num_blocks=${num_blocks} \
  --seed=${SEED} \
  --model_str=${base_model} \
  --algorithm ${algorithm} \
  --temperature_schedule_type ${temperature_schedule_type} \
  --save_str ${dump_dir} \
  > $dump_dir/dataset-MATH500.model-${base_model}.algo-${algorithm}.temp${temp}.steps${mcmc_steps}.blocks${num_blocks}.seed${SEED}.power-sharp.hf.log
