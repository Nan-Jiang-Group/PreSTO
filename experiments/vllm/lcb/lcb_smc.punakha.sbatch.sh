source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"
py=~/WORK/miniconda3/envs/vllm-cuda130/bin/python

# Sampler hyperparameters live in src/power_sharpening/config/{dataset,algorithms}.yaml; this script selects the LCB
# dataset/model and overrides only what it varies.

dataset=lcb
algorithm=smc
machine=dgx
difficulty=all   # one of: all, easy, medium, hard

for lcb_version in release_v6;
do
dump_dir=$basepath/result/livecode_bench/${lcb_version}_${difficulty}/$(date +%F)
mkdir -p $dump_dir
echo "livecodebench dataset (${lcb_version}, difficulty=${difficulty})..."

for base_model in qwen3-4b qwen3-8b;
do
batch_size=880
n_particles=16
max_new_tokens=8192
run_name=model-${base_model}.algo-${algorithm}.lcb-${lcb_version}.diff-${difficulty}.nparticles-${n_particles}.seed-${SEED}
sbatch -t 40:00:00 <<EOT
#!/bin/bash
#SBATCH --account=punakha_general
#SBATCH --qos=punakha_${machine}_general
#SBATCH --partition=${machine}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name="vllm-${run_name}"
#SBATCH --output=$dump_dir/${run_name}.smc.vllm.out
hostname

$py -m power_sharpening.runners.vllm.run_standard_smc \
  --dataset $dataset \
  --algorithm $algorithm \
  --model_str=$base_model \
  --save_str $dump_dir \
  --run_name "$run_name" \
  --override seed=$SEED n_particles=${n_particles} batch_size=${batch_size} max_new_tokens=${max_new_tokens} lcb_version=${lcb_version} difficulty=${difficulty} gpu_memory_utilization=0.7 \
  > $dump_dir/${run_name}.smc.vllm.log 2>&1
EOT
done
done
