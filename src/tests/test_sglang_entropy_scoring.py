"""Test SGLang entropy metadata without loading a model.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/tests/test_sglang_entropy_scoring.py
"""

import importlib.util
import math
import sys
import types

import pytest

if (
    "sglang" not in sys.modules
    and importlib.util.find_spec("sglang") is None
):
    sys.modules["sglang"] = types.ModuleType("sglang")

from power_sharpening.backends.sglang.samplers.low_temp_proposal_sampler import (
    low_temp_proposal_sampling,
    require_suffix_entropies,
)


def _entry(probability, token_id):
    return [math.log(probability), token_id, None]


class _FakeEngine:
    def __init__(self, include_top_logprobs=True):
        self.calls = []
        self.include_top_logprobs = include_top_logprobs

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return {
                "meta_info": {
                    "output_token_logprobs": [
                        _entry(0.7, 11),
                        _entry(0.6, 12),
                    ]
                }
            }

        meta = {
            "input_token_logprobs": [
                _entry(0.9, 2),
                _entry(0.5, 11),
                _entry(0.8, 12),
            ]
        }
        if self.include_top_logprobs:
            meta["input_top_logprobs"] = [
                [_entry(0.9, 2), _entry(0.05, 3)],
                [_entry(0.5, 11), _entry(0.25, 21)],
                [_entry(0.8, 12), _entry(0.1, 22)],
            ]
        return {"meta_info": meta}


class _FakeWrapper:
    def __init__(self, engine):
        self.engine = engine
        self.temperature = 0.25
        self.alpha = 4.0


def _tail_bucket_entropy(*probabilities):
    tail = 1.0 - sum(probabilities)
    masses = [*probabilities, tail]
    return -sum(probability * math.log(probability) for probability in masses)


def test_entropy_metadata_uses_existing_base_scoring_pass():
    engine = _FakeEngine()
    wrapper = _FakeWrapper(engine)

    proposal, proposal_logprobs, target_scores, meta = low_temp_proposal_sampling(
        wrapper,
        context=[1, 2],
        seq_len=4,
        use_cache=True,
        entropy_top_k=2,
        ignore_eos=True,
    )

    assert proposal == [1, 2, 11, 12]
    assert proposal_logprobs == pytest.approx([math.log(0.7), math.log(0.6)])
    assert target_scores == pytest.approx(
        [4.0 * math.log(0.5), 4.0 * math.log(0.8)]
    )
    assert require_suffix_entropies(meta) == pytest.approx(
        [
            _tail_bucket_entropy(0.5, 0.25),
            _tail_bucket_entropy(0.8, 0.1),
        ]
    )
    assert meta["entropy_mode"] == "topk_tail_bucket"
    assert meta["entropy_top_k"] == 2

    # Entropy is obtained from pass 2; there is no third entropy-only call.
    assert len(engine.calls) == 2
    assert "top_logprobs_num" not in engine.calls[0]
    assert engine.calls[0]["sampling_params"]["ignore_eos"] is True
    assert engine.calls[1]["top_logprobs_num"] == 2
    assert engine.calls[1]["sampling_params"]["temperature"] == 1.0
    assert engine.calls[1]["sampling_params"]["ignore_eos"] is True


def test_default_call_preserves_two_pass_contract_without_entropy_payload():
    engine = _FakeEngine(include_top_logprobs=False)
    wrapper = _FakeWrapper(engine)

    _, _, _, meta = low_temp_proposal_sampling(
        wrapper,
        context=[1, 2],
        seq_len=4,
    )

    assert len(engine.calls) == 2
    assert "top_logprobs_num" not in engine.calls[1]
    assert meta["suffix_entropies"] is None
    assert meta["entropy_mode"] is None
    with pytest.raises(RuntimeError, match="Pass entropy_top_k"):
        require_suffix_entropies(meta)


def test_missing_top_logprobs_fails_at_scoring_boundary():
    engine = _FakeEngine(include_top_logprobs=False)
    wrapper = _FakeWrapper(engine)

    with pytest.raises(RuntimeError, match="did not return input_top_logprobs"):
        low_temp_proposal_sampling(
            wrapper,
            context=[1, 2],
            seq_len=4,
            entropy_top_k=2,
        )


@pytest.mark.parametrize("entropy_top_k", [0, -1])
def test_entropy_top_k_must_be_positive(entropy_top_k):
    with pytest.raises(ValueError, match="must be positive"):
        low_temp_proposal_sampling(
            _FakeWrapper(_FakeEngine()),
            context=[1],
            seq_len=2,
            entropy_top_k=entropy_top_k,
        )
