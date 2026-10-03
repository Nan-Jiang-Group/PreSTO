#!/bin/zsh
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"
py=~/WORK/miniconda3/envs/cuda129/bin/python


algorithm=power_mcmc
base_model=qwen-math-medium
mcmc_steps=10
alpha=3.0

num_blocks=16
temperature_schedule_type=step
batch_size=164

dump_dir=$basepath/result/HumanEval/$(date +%F)
mkdir -p $dump_dir

for temperature_schedule_type in cosine step const linear;
do
for temp in 0.1 0.15 0.2 0.25 0.3 0.35 0.4;
do
sbatch -p gh -N 1 -n 1 -t 02:00:00  <<EOT
#!/bin/bash
#SBATCH --job-name="model-${base_model}.algo-${algorithm}.InitTemp-${temp}.alpha-${alpha}.steps-${mcmc_steps}.blocks-${num_blocks}.seed-${SEED}.temp_sched_type-${temperature_schedule_type}"
#SBATCH --output=$dump_dir/model-${base_model}.algo-${algorithm}.InitTemp-${temp}.alpha-${alpha}.steps-${mcmc_steps}.blocks-${num_blocks}.seed-${SEED}.temp_sched_type-${temperature_schedule_type}.powerSharp.vllm.out
hostname

$py -m power_sharpening.runners.vllm.run_power_mh \
  --dataset human_eval \
  --algorithm ${algorithm} \
  --model_str=${base_model} \
  --save_str $dump_dir \
  --override seed=${SEED} mcmc_steps=${mcmc_steps} alpha=${alpha} num_blocks=${num_blocks} batch_size=${batch_size} temperature=${temp} temperature_schedule_type=${temperature_schedule_type} \
  > $dump_dir/model-${base_model}.algo-${algorithm}.InitTemp-${temp}.alpha-${alpha}.steps-${mcmc_steps}.blocks-${num_blocks}.seed-${SEED}.temp_sched_type-${temperature_schedule_type}.powerSharp.vllm.log 2>&1
EOT
done
done
