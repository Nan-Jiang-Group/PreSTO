import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Callable


class CustomSamplerLLM:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.device = device or self.model.device
        self.block_size = self.model.config.max_position_embeddings

    def sample_token(
        self, 
        logits: torch.Tensor, 
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        power: float = 1.0
    ) -> torch.Tensor:
        """
        Custom sampling function with multiple sampling strategies.
        
        Args:
            logits: Raw logits from the model
            temperature: Temperature for scaling (1.0 = no scaling)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            power: Power to raise probabilities to (for power sampling p^alpha)
        
        Returns:
            Sampled token id
        """
        # Power scaling (sample from p^alpha)
        if power != 1.0:
            logits = logits * power
        
        # Temperature scaling
        logits = logits / temperature
        
        # Top-k filtering
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('inf')
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least the first token
            sorted_indices_to_remove[..., 0] = False
            
            # Scatter back to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            probs[indices_to_remove] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Sample from the distribution
        return torch.multinomial(probs, num_samples=1)

    def sample_token_custom(
        self, 
        logits: torch.Tensor, 
        sampling_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply a fully custom sampling function.
        
        Args:
            logits: Raw logits from the model
            sampling_fn: Custom function that takes logits and returns sampled token
        
        Returns:
            Sampled token id
        """
        return sampling_fn(logits)

    @torch.no_grad()
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        power: float = 1.0,
        custom_sampling_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> str:
        """
        Generate text using custom sampling.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            power: Power for power sampling
            custom_sampling_fn: Optional custom sampling function
        
        Returns:
            Generated text including the prompt
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        for _ in range(max_new_tokens):
            # Truncate if exceeds block size
            input_ids_cond = (
                input_ids if input_ids.size(1) <= self.block_size 
                else input_ids[:, -self.block_size:]
            )
            
            # Forward pass
            outputs = self.model(input_ids_cond)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Sample next token
            if custom_sampling_fn is not None:
                next_token = self.sample_token_custom(next_token_logits, custom_sampling_fn)
            else:
                next_token = self.sample_token(
                    next_token_logits, 
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    power=power
                )
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop at EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def get_next_token_logits(self, prefix: list[int]) -> torch.Tensor:
        """
        Get logits for the next token given a prefix.
        
        Args:
            prefix: List of token ids
        
        Returns:
            Logits for next token
        """
        torch_prefix = torch.tensor([prefix], dtype=torch.long, device=self.device)
        prefix_cond = (
            torch_prefix if torch_prefix.size(1) <= self.block_size 
            else torch_prefix[:, -self.block_size:]
        )
        output = self.model(prefix_cond)
        return output.logits[0, -1, :]

    @torch.no_grad()
    def get_next_token_log_probs(self, prefix: list[int]) -> torch.Tensor:
        """
        Get log probabilities for the next token given a prefix.
        
        Args:
            prefix: List of token ids
        
        Returns:
            Log probabilities for next token
        """
        logits = self.get_next_token_logits(prefix)
        return F.log_softmax(logits, dim=-1)


# Utility functions for distribution manipulation
def normalize(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to normalized probabilities."""
    return F.softmax(logits, dim=-1)


def dist_product(logit_p: torch.Tensor, logit_q: torch.Tensor) -> torch.Tensor:
    """Product of two distributions (sum of log probabilities)."""
    return logit_p + logit_q


def dist_temp_scale(logits: torch.Tensor, temp: float) -> torch.Tensor:
    """Temperature scaling: p^(1/tau)."""
    return logits / temp


def dist_power_scale(logits: torch.Tensor, power: float) -> torch.Tensor:
    """Power scaling: p^alpha."""
    return logits * power
