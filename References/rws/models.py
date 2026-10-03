import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class ModelWrapper:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None,
    ):
        
        self.model = model
        self.tokenizer = tokenizer
        
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)
            
        self.model.eval()
        
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
        **kwargs
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
        print(f"Loading model {model_name_or_path} on {device} with dtype {torch_dtype}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device if device != "cpu" else None,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        
        if device == "cpu":
            model = model.to(device)
            
        return cls(model, tokenizer, device)
    
    @torch.no_grad()
    def logprobs_next_tokens(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        input_ids = input_ids.to(self.device)
        
        attention_mask = torch.ones_like(input_ids)
        
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
        
        if temperature != 1.0:
            logits = logits / temperature
            
        log_probs = F.log_softmax(logits, dim=-1)
        
        if squeeze_output:
            log_probs = log_probs.squeeze(0)
            
        return log_probs
    
    @torch.no_grad()
    def teacher_forced_logp(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> float:
        return self.teacher_forced_logp_with_temperature(input_ids, temperature=1.0, start_pos=start_pos)
    
    @torch.no_grad()
    def teacher_forced_logp_with_temperature(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        start_pos: int = 0,
    ) -> float:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        input_ids = input_ids.to(self.device)
        seq_len = input_ids.shape[1]
        
        if seq_len <= start_pos + 1:
            return 0.0
        
        attention_mask = torch.ones_like(input_ids)
            
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  
        
        if temperature != 1.0:
            logits = logits / temperature
            
        log_probs = F.log_softmax(logits, dim=-1)  
        
        target_tokens = input_ids[:, start_pos + 1:] 
        prediction_log_probs = log_probs[:, start_pos:-1, :]  
        
        token_log_probs = prediction_log_probs.gather(
            dim=-1, 
            index=target_tokens.unsqueeze(-1)
        ).squeeze(-1)  

        total_logp = token_log_probs.sum().item()
        
        return total_logp
    
    @torch.no_grad()
    def get_sequence_logprobs(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, float]:

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        input_ids = input_ids.to(self.device)
        seq_len = input_ids.shape[1]
        
        if seq_len <= 1:
            return torch.tensor([]), 0.0
        
        attention_mask = torch.ones_like(input_ids)
            
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        if temperature != 1.0:
            logits = logits / temperature
            
        log_probs = F.log_softmax(logits, dim=-1)
        
        target_tokens = input_ids[:, 1:]
        prediction_log_probs = log_probs[:, :-1, :]
        
        token_log_probs = prediction_log_probs.gather(
            dim=-1,
            index=target_tokens.unsqueeze(-1)
        ).squeeze(-1).squeeze(0)  # (seq_len - 1,)
        
        total_logp = token_log_probs.sum().item()
        
        return token_log_probs, total_logp
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 1.0,
        top_k: int = 0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        input_ids = input_ids.to(self.device)
        
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
            
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
            "top_k": top_k if (do_sample and top_k > 0) else None,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
        }
        generation_config = {k: v for k, v in generation_config.items() if v is not None}
        generation_config.update(kwargs)
        
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, attention_mask=attention_mask, **generation_config)
            
        return output_ids
    
    def encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        return self.tokenizer.encode(
            text, 
            return_tensors="pt", 
            add_special_tokens=add_special_tokens
        ).squeeze(0)
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze(0)
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def load_model(
    model_name_or_path: str,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    **kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    wrapper = ModelWrapper.from_pretrained(
        model_name_or_path,
        device=device,
        torch_dtype=torch_dtype,
        **kwargs
    )
    return wrapper.model, wrapper.tokenizer
