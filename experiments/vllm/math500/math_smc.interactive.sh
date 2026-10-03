source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"

# Sampler hyperparameters live in src/power_sharpening/config/{dataset,algorithms}.yaml; this script selects the dataset
# + model and overrides only what it varies.

dump_dir=$basepath/result/MATH500/$(date +%F)
mkdir -p $dump_dir
algorithm=smc
n_particles=16

for base_model in qwen qwen-math-medium qwen3-4b qwen3-8b tulu phi3.5;
do

run_name=model-${base_model}.algo-${algorithm}.nparticles-${n_particles}.seed-${SEED}
python -m power_sharpening.runners.vllm.run_standard_smc \
  --dataset math500 \
  --algorithm $algorithm \
  --model_str=$base_model \
  --save_str $dump_dir \
  --run_name "$run_name" \
  --override seed=$SEED n_particles=${n_particles} \
  > $dump_dir/${run_name}.smc.vllm.log 2>&1
done
