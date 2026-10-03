#!/bin/zsh
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"
py=~/WORK/miniconda3/envs/cuda129/bin/python


base_model=qwen-math-medium
mcmc_steps=10
alpha=3.5

#num_blocks=16
#num_proposals=4
#temperature_schedule_type=step
batch_size=500

dump_dir=$basepath/result/MATH500/$(date +%F)
mkdir -p $dump_dir

for temperature_schedule_type in const cosine;
do
for num_blocks in 4 8;
do
for temp in 0.15 0.2 0.25;
do
for num_proposals in 4 8 12;
do
sbatch -p gh -N 1 -n 1 -t 02:00:00  <<EOT
#!/bin/bash
#SBATCH --job-name="model-${base_model}.algo-calderhead_power_mcmc.InitTemp-${temp}.alpha-${alpha}.steps-${mcmc_steps}.blocks-${num_blocks}.num_proposals-${num_proposals}.seed-${SEED}.temp_sched_type-${temperature_schedule_type}"
#SBATCH --output=$dump_dir/model-${base_model}.algo-calderhead_power_mcmc.InitTemp-${temp}.alpha-${alpha}.steps-${mcmc_steps}.blocks-${num_blocks}.num_proposals-${num_proposals}.seed-${SEED}.temp_sched_type-${temperature_schedule_type}.powerSharp.vllm.out
hostname

$py -m power_sharpening.runners.vllm.run_calderhead_mh \
  --dataset math500 \
  --model_str=${base_model} \
  --save_str $dump_dir \
  --override seed=${SEED} mcmc_steps=${mcmc_steps} alpha=${alpha} num_blocks=${num_blocks} num_proposals=${num_proposals} batch_size=${batch_size} temperature=${temp} temperature_schedule_type=${temperature_schedule_type} \
  > $dump_dir/model-${base_model}.algo-calderhead_power_mcmc.InitTemp-${temp}.alpha-${alpha}.steps-${mcmc_steps}.blocks-${num_blocks}.num_proposals-${num_proposals}.seed-${SEED}.temp_sched_type-${temperature_schedule_type}.powerSharp.vllm.log 2>&1
EOT
done
done
done
done
