source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"

# MATH500 accuracy sweep for subtree-prefetching MH on vLLM: temperature schedule x prefetch budget, with the traversal
# rank function held fixed.
#
# Unlike run_power_mh, this sampler walks one prompt at a time (a subtree is single-prompt), so a full 500-problem run
# per configuration is slow. Lower max_samples below for a shorter pass.

algorithm=subtree_prefetching_mh
base_model=qwen-math-medium
mcmc_steps=10
alpha=4.0

num_blocks=8
rank_fn=bfs_accept_first
batch_size=500
temp=0.25          # = 1/alpha, the natural proposal temperature for p(x)^alpha
# The shared config block ships max_samples=10 (a short HF smoke run); an accuracy sweep has to clear it or only the
# first 10 problems are graded.
max_samples=null
print_tree=false   # one tree per MH step would swamp a 500-problem log

dump_dir=$basepath/result/MATH500/$(date +%F)
mkdir -p $dump_dir
for temperature_schedule_type in cosine linear step const;
do
echo temperature_schedule_type $temperature_schedule_type
for prefetch_budget in 2 4 8 12 16;
do
echo prefetch_budget $prefetch_budget
python -m power_sharpening.runners.vllm.run_subtree_prefetching_mh \
  --dataset math500 \
  --algorithm ${algorithm} \
  --model_str=${base_model} \
  --save_str $dump_dir \
  --override seed=${SEED} mcmc_steps=${mcmc_steps} alpha=${alpha} num_blocks=${num_blocks} prefetch_budget=${prefetch_budget} rank_fn=${rank_fn} batch_size=${batch_size} max_samples=${max_samples} print_tree=${print_tree} temperature=${temp} temperature_schedule_type=${temperature_schedule_type} \
  > $dump_dir/model-${base_model}.algo-${algorithm}.InitTemp-${temp}.alpha-${alpha}.steps-${mcmc_steps}.blocks-${num_blocks}.prefetch-budget-${prefetch_budget}.rank-${rank_fn}.seed-${SEED}.temp_sched_type-${temperature_schedule_type}.subtreePrefetch.vllm.log 2>&1
done
done
