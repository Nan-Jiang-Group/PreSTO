import torch
from llm_wrapper import CustomSamplerLLM


# Example 1: Basic usage with default sampling
def basic_example():
    llm = CustomSamplerLLM("microsoft/Phi-3-mini-4k-instruct")
    output = llm.generate("Once upon a time,", max_new_tokens=50)
    print("Basic generation:")
    print(output)
    print()


# Example 2: Custom temperature and top-p
def custom_params_example():
    llm = CustomSamplerLLM("microsoft/Phi-3-mini-4k-instruct")
    output = llm.generate(
        "Once upon a time,",
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9
    )
    print("With custom temperature and top-p:")
    print(output)
    print()


# Example 3: Power sampling
def power_sampling_example():
    llm = CustomSamplerLLM("microsoft/Phi-3-mini-4k-instruct")
    output = llm.generate(
        "Once upon a time,",
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.95,
        power=2.0  # Sample from p^2
    )
    print("With power sampling (p^2):")
    print(output)
    print()


# Example 4: Fully custom sampling function
def custom_sampling_example():
    llm = CustomSamplerLLM("microsoft/Phi-3-mini-4k-instruct")
    
    # Define a custom sampling function
    def my_custom_sampler(logits):
        # Apply temperature
        logits = logits / 0.8
        
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Top-p (nucleus) sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens outside nucleus
        nucleus_mask = cumulative_probs <= 0.9
        nucleus_mask[0] = True  # Always keep the best token
        
        # Zero out probabilities outside nucleus
        filtered_probs = torch.zeros_like(probs)
        filtered_probs[sorted_indices[nucleus_mask]] = sorted_probs[nucleus_mask]
        filtered_probs = filtered_probs / filtered_probs.sum()
        
        # Sample
        return torch.multinomial(filtered_probs, num_samples=1)
    
    output = llm.generate(
        "Once upon a time,",
        max_new_tokens=50,
        custom_sampling_fn=my_custom_sampler
    )
    print("With fully custom sampling function:")
    print(output)
    print()


# Example 5: Get log probabilities for analysis
def log_prob_example():
    llm = CustomSamplerLLM("microsoft/Phi-3-mini-4k-instruct")
    
    # Encode a prompt
    prompt = "The capital of France is"
    input_ids = llm.tokenizer.encode(prompt)
    
    # Get log probs for next token
    log_probs = llm.get_next_token_log_probs(input_ids)
    
    # Get top 5 most likely tokens
    top_log_probs, top_indices = torch.topk(log_probs, 5)
    
    print("Top 5 most likely next tokens:")
    for log_prob, idx in zip(top_log_probs, top_indices):
        token = llm.tokenizer.decode([idx.item()])
        prob = torch.exp(log_prob).item()
        print(f"  '{token}' - probability: {prob:.4f}")
    print()


if __name__ == "__main__":
    # Run examples
    basic_example()
    custom_params_example()
    power_sampling_example()
    custom_sampling_example()
    log_prob_example()
