"""SGLang reimplementation of top-level ``base_lm_sampler`` (plain sampling)."""


def _generate_completion(sampler_wrapper, input_ids, max_new_tokens, temperature):
    """Decode a completion from token ids and return the decoded text."""
    out = sampler_wrapper.engine.generate(
        input_ids=list(input_ids),
        sampling_params={"temperature": temperature, "max_new_tokens": max_new_tokens,
                         "top_p": 1.0, "top_k": -1},
    )
    gen_ids = out["output_ids"] if "output_ids" in out else [
        tok for (_, tok, *_ ) in out["meta_info"]["output_token_logprobs"]
    ]
    return sampler_wrapper.tokenizer.decode(gen_ids, skip_special_tokens=True)


def base_LLM_sampling(sampler_wrapper, input_ids, MAX_NEW_TOKENS):
    """Greedy/base-temperature sampling (temperature = 1.0)."""
    return _generate_completion(sampler_wrapper, input_ids, MAX_NEW_TOKENS, temperature=1.0)


def low_temperature_sampling(sampler_wrapper, input_ids, MAX_NEW_TOKENS):
    """Low-temperature sampling at the wrapper's configured proposal temperature."""
    return _generate_completion(
        sampler_wrapper, input_ids, MAX_NEW_TOKENS, temperature=sampler_wrapper.temperature
    )
