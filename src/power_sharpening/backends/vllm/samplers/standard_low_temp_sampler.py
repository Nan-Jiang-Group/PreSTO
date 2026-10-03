"""Standard-vLLM low-temperature proposal sampler.
"""

import logging

from vllm import SamplingParams
from vllm.inputs import TokensPrompt


logger = logging.getLogger("[standard vLLM low-temperature sampler]")




def _copy_sampling_params(
    sampling_params: SamplingParams,
    **kwargs,
) -> SamplingParams:
    """Clone stock vLLM SamplingParams and apply overrides."""
    new_params = sampling_params.clone()
    for key, value in kwargs.items():
        setattr(new_params, key, value)
    return new_params


def _make_sampling_params(
    model_or_wrapper,
    sampling_params: SamplingParams | None,
    max_tokens: int,
    temperature: float,
) -> SamplingParams:
    """Build decode params from explicit values, params, or wrapper attrs."""

    return _copy_sampling_params(
        sampling_params,
        n=1,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def low_temp_proposal_sampling(
    vllm_wrapper,
    context: list[int],
    seq_len: int,
    sampling_params: SamplingParams,
    temperature: float =1.0,
    verbose: bool = False,
) -> list[int]:
    """Sample a suffix from low-temperature q and return the full token sequence.

    Returns:
        proposal_seq: full token sequence, context plus generated suffix.
    """
    
    context = list(context)
    max_new_tokens = seq_len - len(context)
    assert max_new_tokens > 0, "seq_len must exceed the context length"

    decode_params = _make_sampling_params(
        vllm_wrapper,
        sampling_params,
        max_tokens=max_new_tokens,
        temperature=temperature,
    )

    decode_outputs = vllm_wrapper.llm.generate(
        [TokensPrompt(prompt_token_ids=context)],
        sampling_params=decode_params,
        use_tqdm=True,
    )
    out = decode_outputs[0].outputs[0]
    proposal_seq = context + list(out.token_ids)

    if verbose:
        logger.info(
            "low-temp proposal: context_len=%d suffix_len=%d",
            len(context), len(out.token_ids),
        )

    return proposal_seq


def batched_low_temp_sampling(
    llm_wrapper,
    contexts: list[list[int]],
    sampling_params: SamplingParams,
    verbose: bool = False,
) -> list[list[int]]:
    """Batched version of :func:`low_temp_proposal_sampling`.

    Returns full proposal sequences: each input context plus its generated suffix.
    """
    contexts = [list(context) for context in contexts]
    context_lens = [len(context) for context in contexts]
    decode_params = _copy_sampling_params(sampling_params, n=1)

    decode_outputs = llm_wrapper.llm.generate(
        [TokensPrompt(prompt_token_ids=context) for context in contexts],
        sampling_params=decode_params,
        use_tqdm=True,
    )

    proposal_seqs: list[list[int]] = []
    for context, req_out in zip(contexts, decode_outputs):
        out = req_out.outputs[0]
        proposal_seqs.append(context + list(out.token_ids))

    if verbose:
        logger.info(
            "batched low-temp proposals: batch=%d suffix_lens=%s",
            len(contexts),
            [len(seq) - context_len for seq, context_len in zip(
                proposal_seqs, context_lens
            )],
        )

    return proposal_seqs


__all__ = [
    "low_temp_proposal_sampling",
    "batched_low_temp_sampling",
]
