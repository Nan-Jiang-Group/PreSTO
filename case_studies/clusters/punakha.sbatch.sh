CACHE_DIR=~/SCRATCH/

export HF_HOME="$HOME/SCRATCH/huggingface"
export HF_HUB_CACHE="$CACHE_DIR/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"

time_limit=${PUNAKHA_TIME:-10:00:00}
machine=${PUNAKHA_MACHINE:-dgx}

if (( $# < 4 )); then
  echo "Usage: $0 DUMP_DIR ALGORITHM RUN_NAME PYTHON_SCRIPT [PYTHON_ARGS...]"
  exit 2
fi

dump_dir=$1
algorithm=$2
run_name=$3
script_path=$4
shift 4

mkdir -p "$dump_dir"
output_file=$dump_dir/${run_name}.${algorithm}.vllm.out
log_file=$dump_dir/${run_name}.${algorithm}.vllm.log

command_parts=("python" "$script_path" "$@")
command_line="${(j: :)${(q)command_parts[@]}}"

sbatch -A punakha_general -q punakha_${machine}_general -p ${machine} --gres=gpu:1 -t "$time_limit" <<EOT
#!/bin/bash
#SBATCH --job-name="${run_name}"
#SBATCH --output=${output_file}
hostname

source "$HOME/WORK/miniconda3/etc/profile.d/conda.sh" || exit 1
conda activate vllm-cuda130 || exit 1
${command_line} > ${log_file} 2>&1
EOT
