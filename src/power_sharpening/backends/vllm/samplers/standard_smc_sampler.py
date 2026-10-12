"""Standard-vLLM Sequential Monte Carlo (SMC) power sampler.

Run with the installed package:
    python -m power_sharpening.runners.vllm.run_standard_smc --dataset lcb_v6 --override max_samples=20

Power-SMC weights each generated token by the base-model log p and the proposal log q. The custom engine returns both
from one forward pass (``out.power_logprobs`` and ``out.logprobs``); stock vLLM returns only one. This sampler keeps
stock vLLM and gets log p from a second pass: every chunk's tokens are rescored as a prompt with ``prompt_logprobs``,
which vLLM always computes from the raw logits. The engine must run with ``logprobs_mode="processed_logprobs"`` so the
generation pass reports log q after temperature and top-k/top-p.

The SMC loop itself is ``smc_power_sampler`` unchanged (alpha ramp, ESS resampling, min_new_tokens, boxed stop), so
standard and custom Power-SMC differ only in where log p comes from.
"""

import logging

from vllm import SamplingParams as StockSamplingParams
from vllm.inputs import TokensPrompt
from vllm.logprobs import Logprob

from power_sharpening.backends.vllm.samplers.smc_sampler import smc_power_sampler

logger = logging.getLogger("[Standard-vLLM SMC power sampler]")


def _to_stock_sampling_params(sampling_params) -> StockSamplingParams:
    """Copy power-sharpening SamplingParams (which carry ``alpha``) into stock vLLM SamplingParams."""
    stock = StockSamplingParams.__new__(StockSamplingParams)
    for field in StockSamplingParams.__struct_fields__:
        setattr(stock, field, getattr(sampling_params, field))
    return stock


def _extract_prompt_logprobs(prompt_logprobs, token_ids: list[int], start_idx: int, length: int) -> list[float]:
    """Exact-token prompt logprobs for token_ids[start:start+length]."""
    if not prompt_logprobs:
        return []
    extracted: list[float] = []
    for pos in range(start_idx, start_idx + length):
        if pos >= len(prompt_logprobs) or pos >= len(token_ids):
            break
        lp = prompt_logprobs[pos]
        if not lp:
            break
        token_lp = lp.get(token_ids[pos])
        if token_lp is None:
            break
        extracted.append(token_lp.logprob)
    return extracted


class _StandardLLMScorer:
    """Stock ``vllm.LLM`` whose ``generate`` also attaches base-model ``power_logprobs`` to every completion."""

    def __init__(self, llm, score_batch_size: int):
        self._llm = llm
        self.score_batch_size = max(int(score_batch_size), 1)

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def generate(self, prompts, sampling_params, use_tqdm=False):
        if isinstance(sampling_params, list):
            stock_params = [_to_stock_sampling_params(params) for params in sampling_params]
        else:
            stock_params = _to_stock_sampling_params(sampling_params)
        outputs = self._llm.generate(prompts, sampling_params=stock_params, use_tqdm=use_tqdm)

        # Rescore prompt + generated tokens. vLLM requires max_tokens >= 1 even for prompt_logprobs-only scoring, so
        # one greedy dummy token is generated and ignored.
        score_prompts: list[TokensPrompt] = []
        score_meta: list[tuple[int, list[int], int, int]] = []
        for index, request_output in enumerate(outputs):
            completion = request_output.outputs[0]
            completion.power_logprobs = []
            generated = list(completion.token_ids)
            if not generated:
                continue
            prompt_token_ids = list(request_output.prompt_token_ids or [])
            full_token_ids = prompt_token_ids + generated
            score_prompts.append(TokensPrompt(prompt_token_ids=full_token_ids))
            score_meta.append((index, full_token_ids, len(prompt_token_ids), len(generated)))
        if not score_prompts:
            return outputs

        score_params = StockSamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0, detokenize=False)
        for start in range(0, len(score_prompts), self.score_batch_size):
            end = start + self.score_batch_size
            scored = self._llm.generate(score_prompts[start:end], sampling_params=score_params, use_tqdm=False)
            for score_output, (index, full_token_ids, start_idx, length) in zip(scored, score_meta[start:end]):
                base_logp = _extract_prompt_logprobs(score_output.prompt_logprobs, full_token_ids, start_idx, length)
                generated = full_token_ids[start_idx:]
                # Same shape smc_power_sampler reads from the custom engine: one {token_id: Logprob} per position.
                outputs[index].outputs[0].power_logprobs = [
                    {token_id: Logprob(logprob=value)} for token_id, value in zip(generated, base_logp)
                ]
        return outputs


class _StandardModel:
    """The ``.llm`` / ``.tokenizer`` pair ``smc_power_sampler`` drives, backed by stock vLLM plus rescoring."""

    def __init__(self, vllm_wrapper, score_batch_size: int):
        self.llm = _StandardLLMScorer(vllm_wrapper.llm, score_batch_size)
        self.tokenizer = vllm_wrapper.tokenizer


def standard_vllm_smc_power_sampler(
    vllm_wrapper,
    prompts,
    sampling_params,
    score_batch_size: int = 8,
    **smc_kwargs,
) -> list[str]:
    """Run ``smc_power_sampler`` on a stock-vLLM wrapper, scoring log p with a prompt-logprob pass per chunk.

    Args:
        vllm_wrapper:     ``vLLM_Wrapper`` built with ``engine_type="standard"`` and
                          ``logprobs_mode="processed_logprobs"``.
        prompts:          One prompt or a list of prompts.
        sampling_params:  power-sharpening ``SamplingParams`` (carries ``alpha``).
        score_batch_size: Particles rescored per ``prompt_logprobs`` call; prompt logprobs materialize full-vocabulary
                          logits for every context token, so large values can run out of GPU memory.
        **smc_kwargs:     Passed to ``smc_power_sampler`` (max_new_tokens, block_size, n_particles, ...).
    """
    model = _StandardModel(vllm_wrapper, score_batch_size)
    return smc_power_sampler(model, prompts, sampling_params=sampling_params, **smc_kwargs)


__all__ = ["standard_vllm_smc_power_sampler"]
