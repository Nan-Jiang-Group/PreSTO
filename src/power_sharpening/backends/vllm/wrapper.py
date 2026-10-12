"""Load vLLM engines; use the subtree runner's --evict_subtree_cache for pruning."""

from transformers import AutoTokenizer





class vLLM_Wrapper(object):

    def __init__(
        self,
        model: str,
        trust_remote_code: bool = True,
        enable_prefix_caching: bool = False,
        engine_type :str="standard",
        dtype: str = "auto",
        verbose: bool = False,
        enable_subtree_cache_eviction: bool = False,
        log_z_temperatures=None,
        **vllm_kwargs,
    ):
        self.model_name = model
        self.vllm_kwargs = vllm_kwargs
        self.trust_remote_code = trust_remote_code
        self.enable_prefix_caching = enable_prefix_caching
        self.dtype = dtype
        self.verbose = verbose
        self.enable_subtree_cache_eviction = enable_subtree_cache_eviction
        if enable_subtree_cache_eviction:
            from importlib.metadata import version

            if version("vllm").split("+")[0] != "0.27.1":
                raise ValueError("subtree cache eviction currently supports vLLM 0.27.1")
            if engine_type == "standard" or not enable_prefix_caching:
                raise ValueError("subtree cache eviction requires the custom engine and prefix caching")
            if vllm_kwargs.get("data_parallel_size", 1) != 1:
                raise ValueError("subtree cache eviction currently requires data_parallel_size=1")
            additional_config = dict(vllm_kwargs.get("additional_config") or {})
            additional_config["subtree_cache_eviction"] = True
            vllm_kwargs["additional_config"] = additional_config
        # The custom sampler returns logsumexp(log p / T) for each listed T with every scored token, so proposal
        # scores at all of these temperatures need no extra engine requests (see samplers/multi_try_mh_v2.py).
        self.log_z_temperatures = (
            None if log_z_temperatures is None else tuple(float(value) for value in log_z_temperatures)
        )
        if self.log_z_temperatures is not None:
            if engine_type == "standard":
                raise ValueError("log_z_temperatures requires the custom engine")
            # Matches engine_patch.log_normalizers.LOG_Z_TEMPERATURES_KEY; importing it here would load vLLM before
            # VLLM_USE_V2_MODEL_RUNNER is set below.
            additional_config = dict(vllm_kwargs.get("additional_config") or {})
            additional_config["log_z_temperatures"] = list(self.log_z_temperatures)
            vllm_kwargs["additional_config"] = additional_config

        # Set env var so EngineCore subprocess modules can read it.
        if verbose:
            import os
            os.environ["VERBOSE"] = "1"
            os.environ["MH_LLM_VERBOSE"] = "1"

        # Defensive: drop any aliases of the prefix-cache flag that a caller
        # may have passed via **kwargs, so they can't leak into vLLM's EngineArgs (which only accepts
        # `enable_prefix_caching`).

        vllm_kwargs.pop("enable_prefix_caching", None)

        gpu_memory_utilization = vllm_kwargs.pop("gpu_memory_utilization", 0.95)

        # Initialize vLLM
        if engine_type == 'standard':
            from vllm import LLM
        else:
            import os
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
            from power_sharpening.backends.vllm.engine_patch import LLM
        self.llm = LLM(
                model,
                enable_prefix_caching=enable_prefix_caching,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                **vllm_kwargs,
            )
        # Initialize tokenizer Gets a tokenizer for the given model name via HuggingFace or ModelScope.
        self.tokenizer = self.llm.get_tokenizer()

        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     model,
        #     trust_remote_code=trust_remote_code,
        # )
        # if self.tokenizer.pad_token_id is None:
        #     self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self._closed = False



    def close(self) -> None:
        """Shut down the vLLM engine core, releasing GPU memory and processes.

        Safe to call multiple times; only the first call performs cleanup.
        """
        if self._closed:
            return

        # Navigate vLLM's internal structure to reach the engine core. Uses getattr chains so this stays resilient
        # across vLLM versions.
        engine_core = getattr(
            getattr(self.llm, "llm_engine", None), "engine_core", None
        )
        if engine_core is not None:
            engine_core.shutdown()
        self._closed = True

    def __del__(self) -> None:
        """Destructor safety net — ensures cleanup if close() was never called.

        During interpreter shutdown built-in globals (e.g. getattr) may already be None, so all cleanup here is wrapped
        in a bare except.
        """
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "vLLM_Wrapper":
        """Enter the context manager — returns self for use in ``with`` blocks."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context manager — always calls close(), even on exceptions."""
        self.close()

    def generate(self, *args, **kwargs):
        """Thin passthrough to the underlying vLLM engine's generate method."""
        return self.llm.generate(*args, **kwargs)

    def evict_subtree_cache(self, sequences, keep_token_ids):
        """Evict idle cached prefixes in sequences, preserving keep_token_ids.

        Both arguments contain full token sequences including the prompt.
        Returns scheduler-reported eviction counts after cleanup completes.
        """
        if not self.enable_subtree_cache_eviction:
            raise ValueError("construct the wrapper with enable_subtree_cache_eviction=True")
        from power_sharpening.backends.vllm.cache_eviction import call_cache_eviction

        return call_cache_eviction(self.llm, sequences, keep_token_ids)

    def _build_prompt(self, prompt: str, tokens: list[int]) -> str:
        """Concatenate *prompt* with the decoded representation of *tokens*."""
        if not tokens:
            return prompt
        return prompt + self.tokenizer.decode(tokens, skip_special_tokens=True)

    def __repr__(self):
        return (
            f"vLLM-Wrapper("
            f"model={self.model_name!r}, "
            f"llm={self.llm.__class__.__name__}, "
            f"tokenizer={self.tokenizer.__class__.__name__}, "
            f"enable_prefix_caching={self.enable_prefix_caching}, "
            f"trust_remote_code={self.trust_remote_code}, "
            f"verbose={self.verbose})"
        )
