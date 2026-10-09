"""KV-cache and prefix-cache logging for SGLang runs, the counterpart of the vLLM runners' resource probe.

Used by the SGLang runners (``python -m power_sharpening.runners.sglang.run_power_sample_mh`` and
``run_subtree_prefetching_mh``); not run directly.

SGLang's scheduler lives in a child process, so the main process sees the KV cache only through two windows:

  * ``engine.get_server_info()`` -- the KV pool capacity in tokens and GiB. Logged once as ``GPU KV cache size: N
    tokens`` (vLLM's wording, so the case-study parsers read both backends).
  * each request's ``meta_info`` -- ``prompt_tokens`` and ``cached_tokens`` (prompt tokens served from the radix cache).
    Summed over every ``engine.generate`` call, including the scoring passes, they give an exact token-level prefix-cache
    hit rate per sample and per run.

Live pool occupancy is only visible inside the scheduler; run the engine with ``log_level="info"`` and the scheduler logs
it itself as ``token usage`` on its ``Prefill batch`` / ``Decode batch`` lines.
"""

import logging

logger = logging.getLogger("[sglang kv cache]")

_TOKEN_COUNTS = ("prompt_tokens", "cached_tokens")


class SglangKvCacheStats:
    """Count prompt and cached tokens across an SGLang engine's ``generate`` calls."""

    def __init__(self, engine):
        self.engine = engine
        self.kv_capacity_tokens = None
        self.kv_pool_gib = None
        self.weight_gib = None
        self._run = dict.fromkeys(_TOKEN_COUNTS, 0)
        self._window = dict.fromkeys(_TOKEN_COUNTS, 0)
        self._run_calls = 0
        self._window_calls = 0
        self._wrap_generate()

    def _wrap_generate(self):
        original = self.engine.generate

        def generate(*args, **kwargs):
            outputs = original(*args, **kwargs)
            self._record(outputs)
            return outputs

        # An instance attribute shadows the bound method, so every sampler's engine.generate call is counted.
        self.engine.generate = generate

    def _record(self, outputs):
        self._run_calls += 1
        self._window_calls += 1
        for output in outputs if isinstance(outputs, list) else [outputs]:
            meta = output.get("meta_info", {}) if isinstance(output, dict) else {}
            for key in _TOKEN_COUNTS:
                count = int(meta.get(key) or 0)
                self._run[key] += count
                self._window[key] += count

    def log_capacity(self):
        """Log the KV pool size once the engine is up and idle."""
        try:
            info = self.engine.get_server_info()
            memory = info["internal_states"][0]["memory_usage"]
            self.kv_capacity_tokens = int(memory["token_capacity"])
            self.kv_pool_gib = float(memory["kvcache"])
            self.weight_gib = float(memory["weight"])
        except Exception as exc:  # A renamed SGLang field costs the line, not the run.
            logger.warning("could not read the SGLang KV pool size (%s)", exc)
            return
        logger.info(
            "GPU KV cache size: %s tokens, KV pool %.2f GiB, weights %.2f GiB",
            f"{self.kv_capacity_tokens:,}",
            self.kv_pool_gib,
            self.weight_gib,
        )

    @staticmethod
    def _rates(counts: dict, calls: int, prefix: str = "") -> dict:
        queried, hits = counts["prompt_tokens"], counts["cached_tokens"]
        return {
            f"{prefix}prefix_cache_hit_rate": hits / queried if queried else None,
            f"{prefix}prefix_cache_queried_tokens": queried,
            f"{prefix}prefix_cache_hit_tokens": hits,
            f"{prefix}engine_generate_calls": calls,
        }

    def mark(self, sample_idx: int) -> dict:
        """Close the per-sample window, log it, and return its metrics for the result row."""
        row = self._rates(self._window, self._window_calls)
        logger.info(
            "sample %d prefix cache hit rate: %s (%d/%d prompt tokens) over %d engine calls",
            sample_idx,
            "n/a" if row["prefix_cache_hit_rate"] is None else f"{row['prefix_cache_hit_rate']:.6%}",
            row["prefix_cache_hit_tokens"],
            row["prefix_cache_queried_tokens"],
            row["engine_generate_calls"],
        )
        self._window = dict.fromkeys(_TOKEN_COUNTS, 0)
        self._window_calls = 0
        return row

    def summary(self) -> dict:
        return {
            "enabled": True,
            "kv_capacity_tokens": self.kv_capacity_tokens,
            "kv_pool_gib": self.kv_pool_gib,
            "weight_gib": self.weight_gib,
            **self._rates(self._run, self._run_calls),
            "kv_source": "sglang-meta-info",
        }

    def summary_line(self) -> str:
        stats = self.summary()
        rate = stats["prefix_cache_hit_rate"]
        return (
            f"kv_capacity={stats['kv_capacity_tokens']} tokens "
            f"({'n/a' if self.kv_pool_gib is None else format(self.kv_pool_gib, '.2f')} GiB pool), "
            f"prefix_hit_rate={'n/a' if rate is None else format(rate, '.6%')} "
            f"({stats['prefix_cache_hit_tokens']}/{stats['prefix_cache_queried_tokens']} tokens), "
            f"engine_generate_calls={stats['engine_generate_calls']}, kv_source={stats['kv_source']}"
        )
