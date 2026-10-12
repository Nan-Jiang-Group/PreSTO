"""CPU tests for base-model likelihood/confidence diagnostics.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/tests/test_base_diagnostics.py
"""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from power_sharpening.backends.hf.compute_log_likelihood_and_confidence import (
    compute_log_likelihood_and_confidence,
)
from power_sharpening.common.sample_stats import (
    BaseModelDiagnostics,
    SamplingStats,
)

VOCAB_SIZE = 5


class _PositionalLogitsModel:
    """Emit position-dependent logits so misalignment changes the result."""

    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, use_cache=False):
        assert use_cache is False
        seq_len = input_ids.shape[1]
        self.calls.append(seq_len)
        # Row t favours token (t + 1) % VOCAB_SIZE, so shifting the scored rows by one position moves probability mass
        # to a different token.
        logits = torch.zeros((1, seq_len, VOCAB_SIZE), dtype=torch.float32)
        for position in range(seq_len):
            logits[0, position, (position + 1) % VOCAB_SIZE] = 2.0
            logits[0, position, position % VOCAB_SIZE] = 1.0
        return SimpleNamespace(logits=logits)


def _wrapper(model):
    return SimpleNamespace(base_model=model, device=torch.device("cpu"))


def _expected_sums(model, tokens, prompt_len):
    """Reference reduction over the response positions, computed per token."""
    logits = model(input_ids=torch.tensor([tokens]), use_cache=False).logits[0]
    logprob_sum = 0.0
    neg_entropy_sum = 0.0
    for position in range(prompt_len, len(tokens)):
        logprobs = F.log_softmax(logits[position - 1], dim=-1)
        logprob_sum += float(logprobs[tokens[position]])
        neg_entropy_sum += float((logprobs.exp() * logprobs).sum())
    return logprob_sum, neg_entropy_sum


def test_diagnostics_score_response_tokens_under_the_base_model():
    model = _PositionalLogitsModel()
    tokens = [1, 4, 2, 0, 3, 1]
    prompt_len = 2

    diagnostics = compute_log_likelihood_and_confidence(_wrapper(model), tokens, prompt_len)

    expected_logprob_sum, expected_neg_entropy_sum = _expected_sums(
        model, tokens, prompt_len
    )
    num_response_tokens = len(tokens) - prompt_len
    assert diagnostics.num_tokens == num_response_tokens
    assert diagnostics.logprob_sum == pytest.approx(expected_logprob_sum, abs=1e-6)
    assert diagnostics.neg_entropy_sum == pytest.approx(
        expected_neg_entropy_sum, abs=1e-6
    )
    assert diagnostics.log_likelihood == pytest.approx(
        expected_logprob_sum / num_response_tokens, abs=1e-6
    )
    assert diagnostics.confidence == pytest.approx(
        expected_neg_entropy_sum / num_response_tokens, abs=1e-6
    )


def test_diagnostics_use_a_single_forward_pass_over_the_terminal_state():
    model = _PositionalLogitsModel()
    tokens = [1, 4, 2, 0, 3, 1]

    compute_log_likelihood_and_confidence(_wrapper(model), tokens, prompt_len=2)

    assert model.calls == [len(tokens)]


def test_diagnostics_are_negative_and_bounded_by_a_uniform_distribution():
    diagnostics = compute_log_likelihood_and_confidence(
        _wrapper(_PositionalLogitsModel()),
        [1, 4, 2, 0, 3, 1],
        prompt_len=2,
    )

    uniform_neg_entropy = -torch.tensor(float(VOCAB_SIZE)).log().item()
    assert diagnostics.log_likelihood < 0.0
    assert uniform_neg_entropy < diagnostics.confidence < 0.0


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 64])
def test_chunked_reduction_matches_single_chunk(chunk_size):
    model = _PositionalLogitsModel()
    tokens = [1, 4, 2, 0, 3, 1, 2, 2, 4]

    chunked = compute_log_likelihood_and_confidence(
        _wrapper(model), tokens, prompt_len=3, chunk_size=chunk_size
    )
    whole = compute_log_likelihood_and_confidence(
        _wrapper(model), tokens, prompt_len=3, chunk_size=len(tokens)
    )

    assert chunked.logprob_sum == pytest.approx(whole.logprob_sum, abs=1e-6)
    assert chunked.neg_entropy_sum == pytest.approx(whole.neg_entropy_sum, abs=1e-6)


class _NonFiniteLogitsModel:
    """Return an infinite logit at the row scoring the last response token."""

    def __call__(self, input_ids, use_cache=False):
        logits = torch.zeros(
            (1, input_ids.shape[1], VOCAB_SIZE), dtype=torch.float32
        )
        logits[0, -2, 3] = float("inf")
        return SimpleNamespace(logits=logits)


def test_non_finite_logits_are_rejected():
    with pytest.raises(RuntimeError, match="non-finite"):
        compute_log_likelihood_and_confidence(
            _wrapper(_NonFiniteLogitsModel()), [1, 2, 3, 4], prompt_len=2
        )


def test_response_with_no_tokens_is_rejected():
    with pytest.raises(ValueError, match="no response tokens"):
        compute_log_likelihood_and_confidence(
            _wrapper(_PositionalLogitsModel()), [1, 2], prompt_len=2
        )


def test_empty_prompt_is_rejected():
    with pytest.raises(ValueError, match="at least one prompt token"):
        compute_log_likelihood_and_confidence(
            _wrapper(_PositionalLogitsModel()), [1, 2, 3], prompt_len=0
        )


def test_sampling_stats_serializes_diagnostics_when_available():
    stats = SamplingStats(
        base_diagnostics=BaseModelDiagnostics(
            num_tokens=4,
            logprob_sum=-8.0,
            neg_entropy_sum=-2.0,
        )
    )

    data = stats.to_json()

    assert data["num_response_tokens"] == 4
    assert data["log_likelihood"] == pytest.approx(-2.0)
    assert data["confidence"] == pytest.approx(-0.5)


def test_sampling_stats_omits_unavailable_diagnostics():
    data = SamplingStats().to_json()

    assert "log_likelihood" not in data
    assert "confidence" not in data


def test_diagnostics_reject_empty_response_length():
    with pytest.raises(ValueError, match="at least one scored response token"):
        BaseModelDiagnostics(num_tokens=0, logprob_sum=0.0, neg_entropy_sum=0.0)
