import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from grader_utils.math_grader import grade_answer
import itertools
import re
from typing import List, Dict, Tuple

_LAST_NUM_RE = re.compile(r"_(\d+)(?=\.[^.]+$)")



def safe_grade_math(ans, correct_ans):
    try:
        return int(grade_answer(ans, correct_ans))
    except Exception:
        return 0

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


def plot_passk(fnames):
    grouped_fnames, SEEDS = group_fnames_by_seed(fnames)

    shard_lengths = []
    for fname in grouped_fnames[0]:
        shard_lengths.append(len(pd.read_csv(fname)))
    TOTAL = int(sum(shard_lengths))
    
    correct_by_seed = np.zeros((max(SEEDS) + 1, TOTAL), dtype=np.uint8)
    
    for seed in SEEDS:
        for idx, fname in enumerate(grouped_fnames[seed]):
          df = pd.read_csv(fname)
          for i in range(len(df)):
              prob_id = idx*(len(df)) + i
              correct_by_seed[seed, prob_id] = safe_grade(df["mcmc_answer"][i], df["correct_answer"][i])
              # correct_by_seed[seed, prob_id] = safe_grade(df["std_answer"][i], df["correct_answer"][i])
              # correct_by_seed[seed, prob_id] = safe_grade(df["naive_answer"][i], df["correct_answer"][i])
    
    
    best_of_N_acc = []
    
    num_seeds = len(SEEDS)
    
    for N in range(1, num_seeds + 1):
      accs = []
      if num_seeds <= num_seeds-1 and N <= 5:
          for subset in itertools.combinations(SEEDS, N):
              subset_correct = correct_by_seed[list(subset), :].max(axis=0)
              accs.append(subset_correct.mean())
      else: 
          n_samples = 200
          rng = np.random.default_rng(0)
          for _ in range(n_samples):
              subset = rng.choice(SEEDS, size=N, replace=False)
              subset_correct = correct_by_seed[subset, :].max(axis=0)
              accs.append(subset_correct.mean())
    
      best_of_N_acc.append((N, np.mean(accs)))
    
    for N, mean_acc in best_of_N_acc:
      print(f"Best-of-{N}: {mean_acc:.4f}")
    
    plt.figure(figsize=(6,4))
    plt.plot(
      [N for N, _ in best_of_N_acc],
      [mean for _, mean in best_of_N_acc],
      "o-"
    )
    
    plt.xlabel("k")
    plt.ylabel("Pass@k Accuracy")
    plt.title("MATH500")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    args = parser.parse_args()

    folder = Path(args.folder)
    fnames = sorted(str(p) for p in folder.glob("*.csv"))
    plot_passk(fnames)
