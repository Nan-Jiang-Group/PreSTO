"""CPU checks for entropy computation and transport through the patched vLLM engine.

Run with:
    python -m pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_engine_entropy.py

These tests execute the patched methods with vLLM dependencies stubbed; CUDA integration is separate.
"""

from contextlib import nullcontext
from dataclasses import dataclass
import logging
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest
import torch

from test_vllm_prefill_stats import (
    PACKAGE, PATCH_DIR, _Request, _load_file, _load_scheduler, _make_scheduler, _stub_module,
)


class _Logprobs(NamedTuple):
    logprob_token_ids: np.ndarray
    logprobs: np.ndarray
    sampled_token_ranks: np.ndarray
    cu_num_generated_tokens: list[int] | None = None

    def slice_request(self, index, count):
        start = self.cu_num_generated_tokens[index] if self.cu_num_generated_tokens is not None else index
        return _Logprobs(*(values[start:start + count] for values in self[:3]))


def _scores(values, offsets=None):
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    return _Logprobs(np.full(values.shape, 7), values, np.ones(len(values), dtype=int), offsets)


def _load_gpu_sampler(monkeypatch):
    _stub_module(monkeypatch, PACKAGE, __path__=[str(PATCH_DIR)])
    _stub_module(monkeypatch, f"{PACKAGE}.outputs", SamplerOutput=SimpleNamespace)
    _stub_module(monkeypatch, "vllm.config.model", PROCESSED_LOGPROBS_MODES={"processed_logprobs"})
    _stub_module(
        monkeypatch, "vllm.v1.worker.gpu.input_batch", InputBatch=object,
        get_num_sampled_and_rejected=lambda *args: (torch.ones(2), torch.zeros(2)),
    )
    _stub_module(monkeypatch, "vllm.v1.worker.gpu.metrics.logits", get_num_nans=lambda logits: None)

    def topk_scores(logits, count, sampled, offsets, **kwargs):
        logprobs = logits.log_softmax(-1).gather(-1, sampled[:, None])
        return SimpleNamespace(logprobs=logprobs, cu_num_generated_tokens=offsets)

    _stub_module(monkeypatch, "vllm.v1.worker.gpu.sample.logprob", compute_topk_scores=topk_scores)
    _stub_module(monkeypatch, "vllm.v1.worker.gpu.sample.sampler", Sampler=object)
    _stub_module(monkeypatch, "vllm.v1.worker.gpu.sample.states", NO_LOGPROBS=-1)
    return _load_file(monkeypatch, f"{PACKAGE}.sampler", "sampler.py")


@pytest.mark.parametrize("request_scores", [False, True])
def test_gpu_sampler_computes_entropy_from_full_unscaled_logits(monkeypatch, request_scores):
    module = _load_gpu_sampler(monkeypatch)
    sampler = module.Sampler.__new__(module.Sampler)
    sampler.compute_nans = False
    sampler.logprobs_mode = "processed_logprobs"
    sampler.sampling_states = SimpleNamespace(max_num_logprobs=lambda indices: 1 if request_scores else -1)
    sampler.logprob_token_ids_state = SimpleNamespace(max_num_token_ids=lambda indices: 0)
    sampler.req_states = SimpleNamespace(prefill_len=SimpleNamespace(gpu=None))

    def sample(logits, *args, **kwargs):
        # Simulate vLLM modifying logits in place for a low-temperature proposal.
        logits.mul_(4.0)
        return logits.argmax(-1), logits

    sampler.sample = sample
    probabilities = torch.tensor([[0.5, 0.25, 0.125, 0.125], [0.25] * 4, [0.9, 0.1, 0.0, 0.0]])
    logits = probabilities.log()
    batch = SimpleNamespace(
        expanded_idx_mapping=torch.tensor([0, 0, 1]), idx_mapping_np=np.array([0, 1]),
        cu_num_logits_np=np.array([0, 2, 3]), expanded_local_pos=torch.tensor([0, 1, 0]),
        positions=torch.arange(3), logits_indices=torch.arange(3), input_ids=torch.tensor([7, 8, 9]),
        seq_lens=torch.tensor([2, 1]), num_reqs=2, cu_num_logits=torch.tensor([0, 2, 3]),
        idx_mapping=torch.tensor([0, 1]),
    )
    output = sampler(logits, batch)

    if not request_scores:
        assert output.entropies_tensors is None
        assert output.logprobs_tensors is None
        assert output.power_logprobs_tensors is None
        return

    expected_entropy = [
        -sum(float(p) * np.log(float(p)) for p in row if p > 0) for row in probabilities
    ]
    assert output.entropies_tensors.tolist() == pytest.approx(expected_entropy, abs=1e-6)
    torch.testing.assert_close(output.power_logprobs_tensors.logprobs[:, 0], probabilities[:, 0].log())
    torch.testing.assert_close(output.logprobs_tensors.logprobs[:, 0], logits.log_softmax(-1)[:, 0])
    assert output.power_logprobs_tensors.cu_num_generated_tokens == [0, 2, 3]


def _base_output():
    return SimpleNamespace(
        req_ids=["a", "b"], req_id_to_index={"a": 0, "b": 1}, sampled_token_ids=[[7, 8], [9]],
        logprobs=_scores([-0.1, -0.2, -0.3], [0, 2, 3]), prompt_logprobs_dict={},
        pooler_output=None, kv_connector_output=None, num_nans_in_logits=None, cudagraph_stats=None,
    )


@pytest.mark.parametrize("output_kind", ["sync", "wrapped", "async", "fallback_async", "empty"])
def test_model_runner_preserves_entropy_with_power_scores(monkeypatch, output_kind):
    class BaseAsyncOutput:
        def __init__(self, output, sampler_output):
            self.model_runner_output = output
            self.sampler_output = sampler_output

        def get_output(self):
            return self.model_runner_output

    class BaseRunner:
        def sample(self, *args):
            return self.test_sampler_output, None, None

        def sample_tokens(self, grammar):
            return self.test_result

    _stub_module(monkeypatch, PACKAGE, __path__=[str(PATCH_DIR)])
    _stub_module(monkeypatch, "vllm.v1.outputs", AsyncModelRunnerOutput=object, ModelRunnerOutput=SimpleNamespace)
    _stub_module(monkeypatch, "vllm.v1.worker.gpu.async_utils", AsyncOutput=BaseAsyncOutput)
    _stub_module(monkeypatch, "vllm.v1.worker.gpu.model_runner", GPUModelRunner=BaseRunner)
    _stub_module(monkeypatch, f"{PACKAGE}.outputs", ModelRunnerOutput=SimpleNamespace)
    _stub_module(monkeypatch, f"{PACKAGE}.sampler", Sampler=object)
    _stub_module(monkeypatch, f"{PACKAGE}.utils", patch=lambda *args: nullcontext())
    module = _load_file(monkeypatch, f"{PACKAGE}.model_runner", "model_runner.py")

    scores = _scores([-0.4, -0.5, -0.6], [0, 2, 3])
    sampler_output = SimpleNamespace(
        power_logprobs_tensors=SimpleNamespace(tolists=lambda: scores),
        entropies_tensors=torch.tensor([0.7, 0.8, 0.9]),
    )
    runner = module.GPUModelRunner.__new__(module.GPUModelRunner)
    runner.test_sampler_output = sampler_output
    runner.sample(None, None, None)
    result = _base_output()
    if output_kind == "async":
        result = module.PowerAsyncOutput(result, sampler_output)
    elif output_kind == "fallback_async":
        result = BaseAsyncOutput(result, SimpleNamespace())
    elif output_kind == "wrapped":
        result = SimpleNamespace(model_runner_output=result)
    elif output_kind == "empty":
        result = None
    runner.test_result = result

    actual = runner.sample_tokens(None)

    assert runner._entropies_tensors is None
    assert runner._power_logprobs_tensors is None
    if output_kind == "empty":
        assert actual is None
    else:
        if output_kind == "wrapped":
            actual = actual.model_runner_output
        assert actual.entropies == pytest.approx([0.7, 0.8, 0.9])
        assert actual.power_logprobs is scores
        assert actual.sampled_token_ids == [[7, 8], [9]]


@pytest.mark.parametrize("packed", [False, True])
def test_scheduler_routes_entropies_using_logprob_offsets(monkeypatch, packed):
    module = _load_scheduler(monkeypatch)
    first, second = _Request(None), _Request(None)
    first.request_id, second.request_id = "a", "b"
    first.sampling_params = second.sampling_params = SimpleNamespace(logprobs=1)
    scheduler = _make_scheduler(module, first)
    scheduler.requests["b"] = second
    scheduler.running.append(second)
    scheduled = SimpleNamespace(num_scheduled_tokens={"b": 1, "a": 1}, scheduled_spec_decode_tokens={})
    runner_output = _base_output()
    if not packed:
        runner_output.sampled_token_ids = [[7], [9]]
        runner_output.logprobs = _scores([-0.1, -0.3])
    runner_output.power_logprobs = runner_output.logprobs
    runner_output.entropies = [0.7, 0.8, 0.9] if packed else [0.7, 0.9]

    outputs = scheduler.update_from_output(scheduled, runner_output)[0].outputs

    assert [output.request_id for output in outputs] == ["b", "a"]
    assert outputs[0].new_entropies == [0.9]
    assert outputs[1].new_entropies == ([0.7, 0.8] if packed else [0.7])
    assert all(len(output.new_entropies) == len(output.new_logprobs.logprobs) for output in outputs)


def _load_processors(monkeypatch):
    @dataclass
    class BaseLogprobsProcessor:
        num_logprobs: int = 1
        logprobs: list | None = None
        cumulative_logprob: float = 0.0
        tokenizer: object = None

        @classmethod
        def from_new_request(cls, tokenizer, request):
            return cls(num_logprobs=request.sampling_params.logprobs, logprobs=[], tokenizer=tokenizer)

    _stub_module(monkeypatch, PACKAGE, __path__=[str(PATCH_DIR)])
    _stub_module(monkeypatch, f"{PACKAGE}.outputs", EngineCoreOutput=SimpleNamespace, CompletionOutput=SimpleNamespace)
    _stub_module(monkeypatch, f"{PACKAGE}.utils", patch=lambda *args: nullcontext())
    _stub_module(monkeypatch, "vllm.logger", init_logger=logging.getLogger)
    _stub_module(monkeypatch, "vllm.logprobs", SampleLogprobs=list, create_sample_logprobs=lambda flat: [])
    _stub_module(monkeypatch, "vllm.tokenizers", TokenizerLike=object)
    _stub_module(monkeypatch, "vllm.tokenizers.detokenizer_utils", convert_ids_list_to_tokens=lambda *args: [])
    _stub_module(monkeypatch, "vllm.v1.engine", EngineCoreRequest=object, FinishReason=object)
    _stub_module(monkeypatch, "vllm.v1.outputs", LogprobsLists=_Logprobs)
    _stub_module(
        monkeypatch, "vllm.v1.engine.logprobs", LogprobsProcessor=BaseLogprobsProcessor,
        append_logprobs_for_next_position=lambda target, ids, values, *args: target.append(values),
    )
    logprobs = _load_file(monkeypatch, f"{PACKAGE}.logprobs", "logprobs.py")
    _stub_module(monkeypatch, "vllm.sampling_params", RequestOutputKind=SimpleNamespace(DELTA="delta"))
    _stub_module(
        monkeypatch, "vllm.v1.engine.output_processor", OutputProcessor=object,
        RequestOutputCollector=object, RequestState=object,
    )
    _stub_module(monkeypatch, "vllm.v1.engine.parallel_sampling", ParentRequest=object)
    output = _load_file(monkeypatch, f"{PACKAGE}.output_processor", "output_processor.py")
    return logprobs, output


@pytest.mark.parametrize("kind,tokens,expected", [
    ("cumulative", [9], [0.7, 0.8, 0.9]),
    ("delta", [8, 9], [0.8, 0.9]),
    ("delta", [], []),
])
def test_completion_accumulates_and_slices_entropy_with_tokens(monkeypatch, kind, tokens, expected):
    logprobs, output_module = _load_processors(monkeypatch)
    processor = logprobs.LogprobsProcessor.from_new_request(
        None, SimpleNamespace(sampling_params=SimpleNamespace(logprobs=1, flat_logprobs=False)),
    )
    for scores, entropies in [([-0.1, -0.2], [0.7, 0.8]), ([-0.3, -0.4], [0.9, 1.0])]:
        processor.update_from_output(SimpleNamespace(
            new_logprobs=_scores(scores), new_power_logprobs=_scores(scores),
            new_entropies=entropies, new_prompt_logprobs_tensors=None,
        ))
    assert processor.entropies == [0.7, 0.8, 0.9, 1.0]

    state = output_module.RequestState()
    state.detokenizer = SimpleNamespace(output_token_ids=[7, 8, 9], get_next_output_text=lambda *args: "text")
    state.logprobs_processor = processor
    state.output_kind = kind
    state.request_index = 0
    completion = state._new_completion_output(tokens, "stop", None)

    assert completion.entropies == expected
    assert len(completion.entropies) == len(completion.token_ids)
    assert processor.entropies == [0.7, 0.8, 0.9, 1.0]


@pytest.mark.parametrize("entropies", [None, []])
def test_processor_rejects_missing_or_misaligned_entropy(monkeypatch, entropies):
    logprobs, _ = _load_processors(monkeypatch)
    processor = logprobs.LogprobsProcessor(logprobs=[], power_logprobs=[], entropies=[])
    with pytest.raises(RuntimeError, match="one entropy per logprob position"):
        processor.update_from_output(SimpleNamespace(
            new_logprobs=_scores([-0.1]), new_power_logprobs=_scores([-0.2]),
            new_entropies=entropies, new_prompt_logprobs_tensors=None,
        ))
