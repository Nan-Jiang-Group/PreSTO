import torch
import transformers

from power_sharpening.tasks import MATHBenchmark
from power_sharpening.tasks.constants import MODEL_MAP
from power_sharpening.backends.hf.wrapper import HF_LLM_Wrapper, _auto_device
from power_sharpening.backends.hf.samplers.low_temp_sampler import (
    batched_low_temp_proposal_sampling,
    low_temp_sampling,
)

# Constants match the original HF power-sampling runner defaults.
ALPHA = 4.0
MODEL = "qwen-math-small"
SUBTREE_BATCH_SIZE = 10

# Suffix tokens to generate beyond the longest context. Kept small so the per-sequence suffixes stay short (every
# context is filled up to the same seq_len, so shorter contexts already generate more tokens than this).
NEW_TOKENS = 4

# Size of the candidate pool drawn from the dataset before picking the minimum-spread window of contexts.
CONTEXT_POOL_SIZE = 6 * SUBTREE_BATCH_SIZE


def _load_fp32_model_and_tokenizer(model_str):
    """Load the model in float32 with eager attention.

    The equivalence between the batched (left-padded) and scalar paths is exact only in float32: in bf16 the batch
    dimension alone perturbs the logits enough to flip a borderline argmax, so the two paths sample different tokens and
    diverge. float32 + eager removes that noise (log-probs agree to ~1e-5), which is why this test does not use
    ``hf_load_model_and_tokenizer`` (it forces bf16 + flex_attention + torch.compile).
    """
    device = _auto_device()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_str, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_str,
        dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return device, tokenizer, model


def _make_contexts(benchmark, tokenizer, batch_size):
    """Return ``batch_size`` true, variable-length MATH500 prompt contexts.

    The full (untruncated) tokenized prompts are used so the contexts are real model inputs of naturally varying length.
    Of a candidate pool, the contiguous window with the smallest length spread is chosen: this still exercises the
    left-padding code path (the lengths differ) while keeping the longest suffix short, so generation stays fast and
    deterministic.
    """
    pool = []
    benchmark.seen = 0
    for batch in benchmark.dataset_loader:
        prompts, _ = benchmark.get_question_and_answer(batch)
        for prompt in prompts:
            pool.append(tokenizer.encode(prompt))
        if len(pool) >= CONTEXT_POOL_SIZE:
            break
    if len(pool) < batch_size:
        raise AssertionError(f"MATHBenchmark only yielded {len(pool)} prompts")

    pool.sort(key=len)
    start = min(
        range(len(pool) - batch_size + 1),
        key=lambda s: len(pool[s + batch_size - 1]) - len(pool[s]),
    )
    return pool[start : start + batch_size]


def _assert_nested_close(actual, expected, rtol=1e-4, atol=1e-4):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        assert len(actual_row) == len(expected_row)
        torch.testing.assert_close(
            torch.tensor(actual_row),
            torch.tensor(expected_row),
            rtol=rtol,
            atol=atol,
        )


def test_batched_low_temp_proposal_sampling_matches_scalar_variable_contexts():
    model_str = MODEL_MAP[MODEL]
    device, tokenizer, hf_model = _load_fp32_model_and_tokenizer(model_str)
    sampler_wrapper = HF_LLM_Wrapper(
        hf_model,
        tokenizer,
        device,
        temperature=1.0 / ALPHA,
        alpha=ALPHA,
    )
    # Greedy decoding so the scalar loop and the single batched call sample the same tokens; stochastic sampling
    # consumes RNG differently across the two paths and cannot be expected to agree.
    sampler_wrapper.do_sample = False

    benchmark = MATHBenchmark(batch_size=1)
    contexts = _make_contexts(
        benchmark,
        tokenizer=sampler_wrapper.tokenizer,
        batch_size=SUBTREE_BATCH_SIZE,
    )
    context_lens = [len(c) for c in contexts]
    assert min(context_lens) != max(context_lens), "contexts must be variable length"
    seq_len = max(context_lens) + NEW_TOKENS

    scalar_outputs = [
        low_temp_sampling(
            sampler_wrapper,
            context=context,
            seq_len=seq_len,
            use_cache=False,
        )
        for context in contexts
    ]

    batch_proposals, batch_proposal_lps, batch_target_lss = (
        batched_low_temp_proposal_sampling(
            sampler_wrapper,
            contexts=contexts,
            seq_len=seq_len,
            use_cache=False,
        )
    )

    scalar_proposals = [output[0] for output in scalar_outputs]
    scalar_proposal_lps = [output[1] for output in scalar_outputs]
    scalar_target_lss = [output[2] for output in scalar_outputs]

    assert batch_proposals == scalar_proposals
    _assert_nested_close(batch_proposal_lps, scalar_proposal_lps)
    _assert_nested_close(batch_target_lss, scalar_target_lss)


if __name__ == "__main__":
    test_batched_low_temp_proposal_sampling_matches_scalar_variable_contexts()
    print("batched low-temp sampler equivalence test passed")
