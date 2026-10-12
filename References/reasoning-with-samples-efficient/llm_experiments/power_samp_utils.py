import os

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

# Model configuration registry
MODEL_REGISTRY = {
    "qwen": "Qwen/Qwen2.5-7B",
    "qwen_math": "Qwen/Qwen2.5-Math-7B",
    "qwen_math_grpo": "stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150",
    "phi": "microsoft/Phi-3.5-mini-instruct",
    "tulu": "allenai/Llama-3.1-Tulu-3-8B-DPO",
}

ALL_MODEL_CHOICES = list(MODEL_REGISTRY.keys())

### DESCRIPTION ###
# power sampling to sample from p^{alpha}, where p is the base model
# takes in 1/alpha (temperature) as an argument (default 0.25), and mcmc_power_samp implements sampling from p^{alpha} 


class AutoregressiveSampler:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = self.model.config.max_position_embeddings

    # returns log probs
    @torch.no_grad()
    def next_token(self, prefix):
        device = self.device
        torch_prefix = torch.tensor([prefix], dtype=torch.long, device=device)
        prefix_cond = torch_prefix if torch_prefix.size(1) <= self.block_size else torch_prefix[:, -self.block_size:]
        output = self.model(prefix_cond)
        logits = output.logits
        logits = logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        return torch.log(probs)



# returns probabilities (normed)
def normalize(dist):
    probs = F.softmax(dist, dim=-1)
    return probs

# returns sum of logits (product of distributions p*q)
def dist_product(logit_p, logit_q):
    return logit_p+logit_q

# returns logit scaled by temp (temperature scaling p^(1/tau))
def dist_temp_scale(logit_p, temp):
    return logit_p * torch.tensor(1 / temp, dtype=logit_p.dtype, device=logit_p.device)

# low-temperature sampling proposal distribution
def naive_temp(p : AutoregressiveSampler, context, temp, seq_len):
    c = len(context)
    device = p.device
    tokenizer = p.tokenizer
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    output = p.model.generate(
        input_ids=input_ids,
        max_new_tokens=seq_len - c,
        do_sample=True,
        temperature=temp,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )
    unscaled_logits = torch.stack(output.logits, dim=0)
    scaled_logits = torch.stack(output.scores, dim=0)
    tokens = output.sequences[0][c:]
    prop = output.sequences[0].tolist()

    assert len(tokens) == unscaled_logits.shape[0] == scaled_logits.shape[0]


    idx = tokens.view(unscaled_logits.shape[0], 1, 1)

    log_probs_unnorm = (1/temp * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)).view(-1).tolist()
    log_probs_norm = torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()

    assert len(tokens) == len(log_probs_unnorm) == len(log_probs_norm)

    return prop, log_probs_norm, log_probs_unnorm


# alpha = infty power sampling; temp is for proposal distribution
def max_swap(p : AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16):
    c = len(context)
    print(f'Temp: {temp}')
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []


    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    attempts = 0
    acceptances = 0


    for _ in tqdm(range(block_num)):
        gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=jump_size+len(gen))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts+=1
            t = len(gen)
            idx = random.randint(c, t-1)
            # llm query takes the burden of time
            prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            s = len(prop)
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)
            log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]
            log_r = sum(target_log_prob_prop) - sum(target_log_prob_cur)

            if log_r > 0:
                acceptances+=1
                gen = prop.copy()
                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances/attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances/attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

# power sampling with autoregressive mcmc
def mcmc_power_samp(p : AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16):
    c = len(context)
    print(f'alpha: {1/temp}')
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []


    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    attempts = 0
    acceptances = 0


    for _ in tqdm(range(block_num)):
        gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=jump_size+len(gen))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts+=1
            t = len(gen)
            idx = random.randint(c, t-1)
            # llm query takes the burden of time
            prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            s = len(prop)
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)
            log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]
            log_r = sum(target_log_prob_prop) + sum(log_prob_cur) - sum(target_log_prob_cur) - sum(log_prob_prop)

            if np.random.rand() < np.exp(log_r):
                acceptances+=1
                gen = prop.copy()
                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances/attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances/attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio


def format_prompt(question, model, tokenizer, cot=True):
    if model == "qwen":
        format_str = PROMPT + question
        if cot:
            format_str+=COT
        else:
            format_str+=BASE

    elif model == "qwen_math":
        format_str = PROMPT + question
        if cot:
            format_str+=COT
        else:
            format_str+=BASE

    elif model == "qwen_math_grpo":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "phi_grpo":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "phi":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "tulu":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    return format_str


def get_model_path(model_name):
    """Get the HuggingFace path for a model name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]


def load_model_and_tokenizer(model_name, device, local_files_only=False, trust_remote_code=True):
    """Load model, tokenizer, and create AutoregressiveSampler."""
    model_str = get_model_path(model_name)

    if local_files_only:
        config = transformers.AutoConfig.from_pretrained(model_str, trust_remote_code=trust_remote_code, local_files_only=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code=trust_remote_code, local_files_only=True)
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_str, config=config, torch_dtype="auto", device_map="auto", trust_remote_code=trust_remote_code, local_files_only=True)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code=trust_remote_code)
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_str, torch_dtype="auto", device_map="auto", trust_remote_code=trust_remote_code)

    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, device)
    return hf_model, tokenizer, autoreg_sampler


def add_model_argument(parser, default="qwen"):
    """Add standardized --model argument to an argument parser."""
    parser.add_argument("--model", action="store", default=default, type=str, choices=ALL_MODEL_CHOICES)
