import os

from huggingface_hub import constants
import re

from contextlib import nullcontext
from glob import glob
import json
import random
from tqdm import tqdm
import argparse

import pandas as pd
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from datasets import Dataset, load_dataset, concatenate_datasets


import torch
import torch.nn as nn
from torch.nn import functional as F
import transformers

from grader_utils.parse_utils import parse_answer
from constants import *
from power_samp_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", action = "store", type = str, default = "results/",  dest = "save_str")
    add_model_argument(parser, default="qwen")
    parser.add_argument("--temperature", action = "store", default = 0.5, type = float, dest = "temperature")
    parser.add_argument("--dataset", action = "store", default = "HUMANEVAL", type = str)
    parser.add_argument("--cot", action = "store", type = bool, default = True)
    parser.add_argument("--type", action = "store", type = str, default = "chat", choices = ["chat"])
    parser.add_argument("--mcmc_steps", action = "store", type = int, default = 10)
    parser.add_argument("--device", action = "store", type = str, dest = "device", default = "cuda" if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--batch_idx", action = "store", type = int, default = 0)
    parser.add_argument("--seed", action = "store", type = int, default = 0)
    args = parser.parse_args()

    random.seed(0)


    model = args.model
    device = args.device
    dataset_name = args.dataset
    cot = args.cot
    temp = args.temperature
    mcmc_steps = args.mcmc_steps

    save_str = os.path.join(args.save_str, model)
    os.makedirs(save_str, exist_ok=True)


    print(model)
    print(device)
    print(mcmc_steps)

    if dataset_name == "HUMANEVAL":
        dataset = load_dataset("openai/openai_humaneval", split="test")

    print("dataset done")
    hf_model, tokenizer, autoreg_sampler = load_model_and_tokenizer(model, device, local_files_only=False, trust_remote_code=False)
    print("loaded models")
    results = []

    start = 41*args.batch_idx
    end = 41*(args.batch_idx+1)

    for problem, data in tqdm(enumerate(dataset.select(range(start, min(end, len(dataset))))), desc = "Benchmark on HumanEval"):
        prompt = data["prompt"]
        task_id = data["task_id"]

        if model == "phi" or model == "phi_grpo":
            signature = re.search(
                rf"def\s+({data['entry_point']}.*?):\s*\n", data["prompt"]
            ).group(1)
            description = "\n".join(
                [
                    line.strip()
                    for line in re.search(
                        rf"(?:\"\"\"|''')(.*?)(?:\"\"\"|''')", data["prompt"], re.DOTALL
                    )
                    .group(1)
                    .split("\n")
                ]
            )
            input_text = (
                f"Write a Python function `{signature}` to solve the following problem:\n"
                f"{description}\n"
                f"{data['prompt']}"
            )

        else:
            input_text = prompt

        print(input_text)


        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        prefx = [idx.item() for idx in input_ids[0]]


        naive_temp_output = hf_model.generate(input_ids, max_new_tokens=3072, 
                                return_dict_in_generate=True, output_scores=True, do_sample = True, temperature = temp)
        
        print(tokenizer.decode(naive_temp_output[0][:, len(input_ids[0]):].squeeze().to("cpu"), skip_special_tokens=True))
        print("naive done")
        
        
        std_output = hf_model.generate(input_ids, max_new_tokens=3072, 
                                return_dict_in_generate=True, output_scores=True, do_sample = True)
        
        print(tokenizer.decode(std_output[0][:, len(input_ids[0]):].squeeze().to("cpu"), skip_special_tokens=True))
        print("std done")

        mcmc_temp_output, _, _, acceptance_ratio = mcmc_power_samp(autoreg_sampler, prefx, temp, mcmc_steps, max_new_tokens=3072)

        print(len(std_output))
        print(len(naive_temp_output))
        print(len(mcmc_temp_output))
        print(tokenizer.decode(torch.tensor([mcmc_temp_output], dtype=torch.long, device=device).squeeze().to("cpu"), skip_special_tokens=True))
        print("mcmc done")

        naive_generated_ids = naive_temp_output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        std_generated_ids = std_output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        mcmc_temp_ids = torch.tensor([mcmc_temp_output], dtype=torch.long, device=device).squeeze().to("cpu")

        naive_completion = tokenizer.decode(naive_generated_ids, skip_special_tokens=True)
        std_completion = tokenizer.decode(std_generated_ids, skip_special_tokens=True)
        mcmc_completion = tokenizer.decode(mcmc_temp_ids, skip_special_tokens=True)

        naive_answer = parse_answer(naive_completion)
        std_answer = parse_answer(std_completion)
        mcmc_answer = parse_answer(mcmc_completion)
        
        print(f'Acceptance: {acceptance_ratio}')


        results.append({
            "question": prompt,
            "id": task_id,
            "naive_completion": naive_completion,
            "std_completion": std_completion,
            "mcmc_completion": mcmc_completion,
        })

    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_str, model+"_he_base_power_samp_results_" + str(mcmc_steps) + "_" + str(temp) + "_" + str(args.batch_idx)  + "_" + str(args.seed) + ".csv"), index=False)
    












        













