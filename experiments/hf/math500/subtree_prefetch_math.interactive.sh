source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"


base_model=qwen
alpha=4.0
mcmc_steps=10
prefetch_budget=10
num_blocks=8

dump_dir=$basepath/result/MATH500/$(date +%F)
mkdir -p $dump_dir

python -m power_sharpening.runners.hf.run_subtree_prefetching_mh \
  --dataset math500 \
  --algorithm subtree_prefetching_mh \
  --model_str=${base_model} \
  --save_str ${dump_dir} \
  --override seed=${SEED} alpha=${alpha} mcmc_steps=${mcmc_steps} prefetch_budget=${prefetch_budget} num_blocks=${num_blocks} \
  > $dump_dir/dataset-MATH500.model-${base_model}.alpha${alpha}.steps${mcmc_steps}.blocks-${num_blocks}.prefetch-budget-${prefetch_budget}.seed${SEED}.subtreePrefetch.hf.log 2>&1
