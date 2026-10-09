"""HuggingFace model wrapper for power-sharpening samplers.

Run the HF subtree sampler with:
    python -m power_sharpening.runners.hf.run_subtree_prefetching_mh \
      --dataset math500 --algorithm subtree_prefetching_mh
"""

import torch
import transformers
from power_sharpening.common.temp_scheduler import set_schedule


class _BatchTemperatureProcessor(transformers.LogitsProcessor):
    """Apply one fixed proposal temperature to each generation batch row."""

    def __init__(self, temperatures, device):
        self.temperatures = torch.as_tensor(
            temperatures, dtype=torch.float32, device=device
        ).reshape(-1, 1)
        if not torch.isfinite(self.temperatures).all() or (
            self.temperatures <= 0
        ).any():
            raise ValueError("proposal temperatures must be finite and positive")

    def __call__(self, input_ids, scores):
        return scores.float() / self.temperatures


class HF_LLM_Wrapper(object):
    """Wrapper of a causal LM, tokenizer, propsal and target hyper-parameters.
     used to construct proposal samplers.

    Attributes:
        base_model: a HuggingFace causal LM (e.g. GPT-2, LLaMA).
        tokenizer: the matching tokenizer; its ``eos_token_id`` is used as the pad / stop token during generation.
        device: torch device (e.g. ``"cuda"``, ``"cpu"``) for tensor allocation.
        block_size: maximum context window size for blockwise MH sampling.
        temperature: softmax temperature applied to logits, i.e. softmax(logit / T). Used for the low-temperature
            proposal distribution.
        alpha: Target-distribution exponent.  The target is pi(x) ~ p(x)^alpha.
    """

    def __init__(self, base_model, tokenizer, device, block_size=1024, temperature: float = 1.0, alpha: float = 1.0):
        # base model
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        # used for blockwise suffix resampling
        self.block_size = block_size

        
        # used for proposal sampler
        self.temperature = self.init_temperature = temperature
        # used for target sampler
        self.alpha = alpha
        self.schedule_handle = None
        # When False, generation is greedy (argmax). Defaults to stochastic sampling; tests flip this to make the scalar
        # and batched paths produce identical tokens.
        self.do_sample = True

    def init_schedule(self, temperature_schedule_type, length):
        self.temperature=self.init_temperature
        self.schedule_handle = set_schedule(self.temperature, temperature_schedule_type, total_steps=length)

    def temperature_step(self):
        new_temp = self.schedule_handle.step()
        self.temperature = new_temp
        self.alpha = 1.0 / new_temp

    def generate(
        self,
        input_ids,
        max_new_tokens,
        output_logits=False,
        use_cache=True,
        attention_mask=None,
        ignore_eos=False,
        row_temperatures=None,
    ):
        """Generate tokens, optionally using one temperature per batch row."""
        kwargs = dict(
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=None if ignore_eos else self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=output_logits,
            use_cache=use_cache,
        )
        # Explicitly neutralize inherited truncation settings so the proposal is exactly softmax(logits / T), as assumed
        # by the MH calculation.
        if self.do_sample:
            kwargs.update(
                temperature=self.temperature,
                top_k=0,
                top_p=1.0,
            )
        if row_temperatures is not None:
            if not self.do_sample:
                raise ValueError("per-row proposal temperatures require sampling")
            if len(row_temperatures) != input_ids.shape[0]:
                raise ValueError("expected one proposal temperature per batch row")
            kwargs.update(
                temperature=1.0,
                logits_processor=transformers.LogitsProcessorList([
                    _BatchTemperatureProcessor(row_temperatures, input_ids.device)
                ]),
            )
        out = self.base_model.generate(input_ids, **kwargs)
        return out

    def __repr__(self):
        return (
            f"HF_LLM_Wrapper("
            f"model={self.base_model.__class__.__name__}, "
            f"device={self.device}, "
            f"block_size={self.block_size}, "
            f"temperature={self.temperature}, "
            f"alpha={self.alpha})"
        )



def hf_load_model_and_tokenizer(model_str: str, device_name: str | None = None):
    """Load a causal LM and its tokenizer.

    Args:
        model_str: HuggingFace model identifier.
        device_name: Device string (e.g. ``"cuda"``).  When *None* the best available backend is chosen automatically.

    Returns:
        (device, tokenizer, hf_model)
    """
    device = _auto_device(device_name)
    print(f"Loading model: {model_str} on {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_str,
        dtype="auto",
        trust_remote_code=True,
        attn_implementation ="flex_attention"
    ).to(device)

    hf_model.eval()
    hf_model = torch.compile(hf_model, mode="reduce-overhead")

    print(f"Model loaded: {hf_model.__class__.__name__}")
    print("-" * 80)

    return device, tokenizer, hf_model





def _auto_device(device_name: str | None = None) -> torch.device:
    """Return a torch.device, auto-detecting the best backend when *device_name* is None."""
    if device_name is None:
        if torch.cuda.is_available():
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    return torch.device(device_name)
