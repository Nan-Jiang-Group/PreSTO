import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
from grader_utils.he_grader import entry_point
import itertools
import re
from typing import List, Dict, Tuple

_LAST_NUM_RE = re.compile(r"_(\d+)(?=\.[^.]+$)")



def group_fnames_by_seed(fnames: List[str]) -> Tuple[List[List[str]], List[int]]:
    seed_to_files: Dict[int, List[str]] = {}
    for f in fnames:
        m = _LAST_NUM_RE.search(f)
        if not m:
            continue
        seed = int(m.group(1))
        seed_to_files.setdefault(seed, []).append(f)

    if not seed_to_files:
        return [], []

    max_seed = max(seed_to_files.keys())
    groups: List[List[str]] = [[] for _ in range(max_seed + 1)]
    for s, files in seed_to_files.items():
        groups[s] = sorted(files)

    seeds_sorted = sorted(seed_to_files.keys())
    return groups, seeds_sorted

def fnames_to_json(grouped_fnames, output_fname, tag, data_file='data/HumanEval.jsonl'):
    with open(data_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    output_file = output_fname + "_full_" + tag + ".jsonl"
    with open(output_file, "a") as fout:
        for seed in range(len(grouped_fnames)):
            fnames = grouped_fnames[seed]
            for idx in range(len(fnames)):
                fname = fnames[idx]
                print(fname)
                df = pd.read_csv(fname)
                for i in range(len(df)):
                    mult = len(df)
                    task_id = df["id"][i]
                    assert task_id == dataset[i + mult*idx]["task_id"]
                    entry_point = dataset[i + mult*idx]["entry_point"]
                    prompt = dataset[i + mult*idx]["prompt"]
    
                    if tag=="mcmc":
                        response = df["mcmc_completion"][i]
                    elif tag=="std":
                        response = prompt + df["std_completion"][i]
                    elif tag=="naive":
                        response = prompt + df["naive_completion"][i]
    
                    code_completion = extract_code(response, entry_point)
          
                    line = {
                        "task_id": df["id"][i],
                        "completion": code_completion,
                    }
          
                    fout.write(json.dumps(line) + "\n")
    return output_file


def plot_passk(fnames, output_fname):
    grouped_fnames, SEEDS = group_fnames_by_seed(fnames)
    tag = "mcmc"
    # tag = "std"
    # tag = "naive"
    output_file = fnames_to_json(grouped_fnames, output_fname, tag)
    entry_point(output_file, k="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16")
    
    
      
          
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("output_fname", type=str)
    args = parser.parse_args()

    folder = Path(args.folder)
    fnames = sorted(str(p) for p in folder.glob("*.csv"))
    plot_passk(fnames, args.output_fname)

