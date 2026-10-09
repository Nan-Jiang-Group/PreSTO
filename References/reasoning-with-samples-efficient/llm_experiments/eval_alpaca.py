import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any


def fnames_to_json(fnames, output_fname, tag):
    output_file = output_fname + "_" + tag + ".jsonl"
    with open(output_file, "w") as fout:
        for idx in range(len(fnames)):
            fname = fnames[idx]
            print(fname)
            df = pd.read_csv(fname)
            for i in range(len(df)):
                prompt = df["instruction"][i]
                
                if tag=="mcmc":
                    response =  df["mcmc_completion"][i][len(prompt):]
                elif tag=="std":
                    response = df["std_completion"][i]
                elif tag=="naive":
                    response = df["naive_completion"][i]
                  
                line = {
                    "instruction": prompt,
                    "output": response,
                    "generator": output_fname + "_" + tag
                }

                fout.write(json.dumps(line) + "\n")

    return output_file

def jsonl_to_json(fname):
    data = []
    jsonl_path = fname
    json_path = jsonl_path[:-6] + ".json"
    
    with open(jsonl_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    # Save as JSON array
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

def collate_alpaca(fnames, output_fname):
    tags = ["std", "naive", "mcmc"]
    for tag in tags:
        output_file = fnames_to_json(fnames, output_fname, tag)
        jsonl_to_json(output_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("output_fname", type=str)
    args = parser.parse_args()

    folder = Path(args.folder)
    fnames = sorted(str(p) for p in folder.glob("*.csv"))
    collate_alpaca(fnames, args.output_fname)


    
