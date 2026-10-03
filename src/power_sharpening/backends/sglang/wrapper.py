"""SGLang-backed wrapper, mirroring ``power_sharpening.backends.hf.wrapper``.

Exposes the same surface the sampling algorithms rely on — ``tokenizer``, ``alpha``, ``temperature``, ``block_size`` and
the temperature-schedule hooks — but holds an SGLang ``Engine`` instead of a HuggingFace model. The actual forward
passes happen in ``low_temp_proposal_sampler`` via ``self.engine``.
"""

import transformers
import sglang as sgl
from contextlib import nullcontext
from power_sharpening.common.temp_scheduler import set_schedule


class SGL_LLM_Wrapper(object):
    """Wrapper of an SGLang engine, tokenizer, and proposal/target hyper-parameters.

    Attributes:
        engine: an ``sglang.Engine`` serving the base LM.
        tokenizer: the matching HuggingFace tokenizer; its ``eos_token_id`` is used as the stop token and to truncate
            completions.
        device: kept for API parity with the HF wrapper (unused by SGLang, which manages devices internally).
        block_size: maximum context window size for blockwise MH sampling.
        temperature: softmax temperature applied to logits, i.e. softmax(logit / T). Defines the low-temperature
            proposal q (T = 1/alpha).
        alpha: target-distribution exponent. The target is pi(x) ~ p(x)^alpha.
    """

    def __init__(self, engine, tokenizer, device="cuda", block_size=1024,
                 temperature: float = 1.0, alpha: float = 1.0):
        self.engine = engine
        self.tokenizer = tokenizer
        self.device = device
        # used for blockwise suffix resampling
        self.block_size = block_size

        # used for proposal sampler
        self.temperature = self.init_temperature = temperature
        # used for target sampler
        self.alpha = alpha
        self.schedule_handle = None

    def init_schedule(self, temperature_schedule_type, length):
        self.temperature = self.init_temperature
        self.schedule_handle = set_schedule(
            self.temperature,
            temperature_schedule_type=temperature_schedule_type,
            total_steps=length,
        )

    def temperature_step(self):
        new_temp = self.schedule_handle.step()
        self.temperature = new_temp
        self.alpha = 1.0 / new_temp

    def close(self) -> None:
        """Shut down the underlying SGLang engine (releases GPU memory)."""
        shutdown = getattr(self.engine, "shutdown", None)
        if shutdown is not None:
            shutdown()

    def evict_subtree_cache(self, sequences, keep_token_ids):
        """Prune discarded full token sequences while preserving keep_token_ids.

        Returns counts from the scheduler; both arguments include prompt tokens.
        """
        if not getattr(self.engine, "subtree_cache_eviction_enabled", False):
            raise ValueError("load SGLang with enable_subtree_cache_eviction=True")
        from power_sharpening.backends.sglang.cache_eviction import call_cache_eviction

        return call_cache_eviction(self.engine, sequences, keep_token_ids)

    def __enter__(self) -> "SGL_LLM_Wrapper":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __repr__(self):
        return (
            f"SGLang-LLM-Wrapper("
            f"engine={self.engine.__class__.__name__}, "
            f"device={self.device}, "
            f"block_size={self.block_size}, "
            f"temperature={self.temperature}, "
            f"alpha={self.alpha})"
        )


def sglang_load_model_and_tokenizer(
    model_str: str,
    *,
    skip_tokenizer_init: bool = True,
    enable_subtree_cache_eviction: bool = False,
    **engine_kwargs,
):
    """Load an SGLang engine and a HuggingFace tokenizer.

    Args:
        model_str: HuggingFace model identifier or local path.
        device_name: kept for API parity with the HF loader (SGLang chooses the device itself; pass engine options via
            ``engine_kwargs`` instead).
        skip_tokenizer_init: run the engine in token-id mode (we feed/read token ids directly, so SGLang need not build
            its own tokenizer).
        enable_subtree_cache_eviction: install selective cleanup in the scheduler subprocesses.
        **engine_kwargs: forwarded to ``sglang.Engine`` (e.g. ``tp_size``,
            ``mem_fraction_static``, ``trust_remote_code``, ``dtype``).

    Returns:
        (engine, tokenizer)
    """
    

    launch_context = nullcontext()
    if enable_subtree_cache_eviction:
        from importlib.metadata import version
        from power_sharpening.backends.sglang.cache_eviction import (
            cache_eviction_launch, call_cache_eviction,
        )

        if version("sglang").split("+")[0] != "0.5.2":
            raise ValueError("subtree cache eviction currently supports SGLang 0.5.2")
        if any(engine_kwargs.get(key, 1) != 1 for key in ("dp_size", "pp_size", "nnodes")):
            raise ValueError("subtree cache eviction requires dp_size=pp_size=nnodes=1")
        if engine_kwargs.get("disable_radix_cache", False):
            raise ValueError("subtree cache eviction requires the radix cache")
        launch_context = cache_eviction_launch()

    print(f"Loading SGLang engine: {model_str}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_str, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with launch_context:
        engine = sgl.Engine(
            model_path=model_str,
            skip_tokenizer_init=skip_tokenizer_init,
            **engine_kwargs,
        )
    if enable_subtree_cache_eviction:
        try:
            # Validate the actual cache implementation before any sampling.
            call_cache_eviction(engine, [], [])
        except Exception:
            engine.shutdown()
            raise
    engine.subtree_cache_eviction_enabled = enable_subtree_cache_eviction

    print("SGLang engine loaded.")
    print("-" * 80)
    return engine, tokenizer
