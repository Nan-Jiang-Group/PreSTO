import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
from grader_utils.he_grader import entry_point



def fnames_to_json(fnames, output_fname, tag, data_file='data/HumanEval.jsonl'):
    with open(data_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    output_file = output_fname + "_" + tag + ".jsonl"
    with open(output_file, "w") as fout:
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
      


def he_results(fnames, output_fname):
    tags = ["std", "naive", "mcmc"]
    for tag in tags:
        output_file = fnames_to_json(fnames, output_fname, tag)
        entry_point(output_file)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("output_fname", type=str)
    args = parser.parse_args()

    folder = Path(args.folder)
    fnames = sorted(str(p) for p in folder.glob("*.csv"))
    he_results(fnames, args.output_fname)
    
    

