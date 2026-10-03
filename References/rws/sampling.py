import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
from .models import ModelWrapper


def sample_suffix(
    model_wrapper: ModelWrapper,
    prefix_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    eos_token_id: Optional[int] = None,
    top_p: float = 1.0,
    top_k: int = 0,
) -> Tuple[torch.Tensor, float, int]:
    if eos_token_id is None:
        eos_token_id = model_wrapper.tokenizer.eos_token_id
        
    if prefix_ids.dim() == 1:
        prefix_ids = prefix_ids.unsqueeze(0)
    prefix_ids = prefix_ids.to(model_wrapper.device)
    
    current_ids = prefix_ids.clone()
    suffix_ids_list = []
    logq_forward = 0.0
    n_gen_tokens = 0
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            attention_mask = torch.ones_like(current_ids)
            outputs = model_wrapper.model(current_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)
            
            scaled_logits = logits / temperature if temperature != 1.0 else logits
            
            log_probs = F.log_softmax(scaled_logits, dim=-1)
            probs = torch.exp(log_probs)
            
            if top_k > 0:
                indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
                probs = probs.masked_fill(indices_to_remove, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                probs = probs.masked_fill(indices_to_remove, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            
            token_logp = log_probs[0, next_token[0, 0]].item()
            logq_forward += token_logp
            
            suffix_ids_list.append(next_token[0, 0].item())
            n_gen_tokens += 1
            
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            if next_token[0, 0].item() == eos_token_id:
                break
                
    suffix_ids = torch.tensor(suffix_ids_list, dtype=torch.long, device=model_wrapper.device)
    
    return suffix_ids, logq_forward, n_gen_tokens


def sample_suffix_fast(
    model_wrapper: ModelWrapper,
    prefix_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    eos_token_id: Optional[int] = None,
    top_p: float = 1.0,
) -> Tuple[torch.Tensor, float, int]:
    if eos_token_id is None:
        eos_token_id = model_wrapper.tokenizer.eos_token_id
        
    if prefix_ids.dim() == 1:
        prefix_ids = prefix_ids.unsqueeze(0)
    prefix_ids = prefix_ids.to(model_wrapper.device)
    prefix_len = prefix_ids.shape[1]
    
    output_ids = model_wrapper.generate(
        prefix_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=top_p,
        eos_token_id=eos_token_id,
    )
    
    suffix_ids = output_ids[0, prefix_len:]
    n_gen_tokens = len(suffix_ids)
    
    if n_gen_tokens == 0:
        return torch.tensor([], dtype=torch.long, device=model_wrapper.device), 0.0, 0
    
    logq_forward = logprob_suffix_given_prefix(
        model_wrapper,
        prefix_ids.squeeze(0),
        suffix_ids,
        temperature=temperature,
    )
    
    return suffix_ids, logq_forward, n_gen_tokens


def logprob_suffix_given_prefix(
    model_wrapper: ModelWrapper,
    prefix_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    temperature: float = 1.0,
) -> float:
    if len(suffix_ids) == 0:
        return 0.0
    
    device = model_wrapper.device
    
    if prefix_ids.dim() > 1:
        prefix_ids = prefix_ids.squeeze(0)
    if suffix_ids.dim() > 1:
        suffix_ids = suffix_ids.squeeze(0)
    
    prefix_ids = prefix_ids.to(device)
    suffix_ids = suffix_ids.to(device)
    
    suffix_len = len(suffix_ids)
    
    current_ids = prefix_ids.unsqueeze(0)  
    
    total_logp = 0.0
    
    with torch.no_grad():
        for i in range(suffix_len):
            attention_mask = torch.ones_like(current_ids)
            outputs = model_wrapper.model(current_ids, attention_mask=attention_mask)
            
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)
            
            if temperature != 1.0:
                logits = logits / temperature
            
            log_probs = F.log_softmax(logits, dim=-1)  # (1, vocab_size)
            
            target_token = suffix_ids[i]
            token_logp = log_probs[0, target_token].item()
            total_logp += token_logp
            
            next_token = suffix_ids[i:i+1].unsqueeze(0)  # (1, 1)
            current_ids = torch.cat([current_ids, next_token], dim=1)
    
    return total_logp


def compute_base_logp(
    model_wrapper: ModelWrapper,
    token_ids: torch.Tensor,
    start_pos: int = 0,
) -> float:
    return model_wrapper.teacher_forced_logp(token_ids, start_pos=start_pos)


def compute_proposal_logp(
    model_wrapper: ModelWrapper,
    token_ids: torch.Tensor,
    temperature: float,
    start_pos: int = 0,
) -> float:
    return model_wrapper.teacher_forced_logp_with_temperature(
        token_ids,
        temperature=temperature,
        start_pos=start_pos,
    )


def extend_sequence(
    model_wrapper: ModelWrapper,
    current_ids: torch.Tensor,
    target_length: int,
    temperature: float = 1.0,
    eos_token_id: Optional[int] = None,
    top_p: float = 1.0,
) -> Tuple[torch.Tensor, float, int, bool]:
    current_len = len(current_ids)
    
    if current_len >= target_length:
        return current_ids, 0.0, 0, False
        
    max_new_tokens = target_length - current_len
    
    suffix_ids, logq, n_gen = sample_suffix(
        model_wrapper,
        current_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_token_id=eos_token_id,
        top_p=top_p,
    )
    
    if len(suffix_ids) > 0:
        extended_ids = torch.cat([current_ids.to(suffix_ids.device), suffix_ids], dim=0)
    else:
        extended_ids = current_ids
        
    hit_eos = (len(suffix_ids) > 0 and 
               suffix_ids[-1].item() == (eos_token_id or model_wrapper.tokenizer.eos_token_id))
    
    return extended_ids, logq, n_gen, hit_eos
