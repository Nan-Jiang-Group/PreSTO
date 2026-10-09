"""CPU checks for MultiTryMH v2: engine-cached logZ_T replaces extra temperature-scoring requests.

Run with:
    python -m pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_multi_try_mh_v2.py

vLLM/CUDA are stubbed: a small normalized autoregressive model supplies exact probabilities, and the patched engine files
run with their vLLM imports replaced. These tests do not claim a real vLLM runtime or GPU benchmark.
"""

import copy
import importlib.util
from pathlib import Path
import random
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_vllm_engine_entropy import _base_output, _load_gpu_sampler, _load_processors, _scores
from test_vllm_prefill_stats import PACKAGE, PATCH_DIR, _Request, _load_scheduler, _make_scheduler, _stub_module

SAMPLERS_DIR = PATCH_DIR.parent / "samplers"
TEMPERATURES = [0.25, 0.6, 1.2]


def _load_path(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------------------------------------
# logZ_T identity
# --------------------------------------------------------------------------------------------------------------------


def test_cached_normalizers_reproduce_tempered_softmax_scores(monkeypatch):
    module = _load_path(monkeypatch, "_test_log_normalizers", PATCH_DIR / "log_normalizers.py")
    torch.manual_seed(0)
    logits = torch.randn(5, 11) * 3
    logits[2, 4] = -float("inf")  # A masked vocabulary entry must not break the normalizer.
    tokens = [0, 3, 5, 10, 7]
    log_z = module.temperature_log_normalizers(logits, TEMPERATURES)
    assert log_z.shape == (5, len(TEMPERATURES))

    base = logits.log_softmax(-1)[torch.arange(5), tokens].tolist()
    actual = module.tempered_logprobs(base, log_z.tolist(), TEMPERATURES)
    expected = np.array([
        (logits / temperature).log_softmax(-1)[torch.arange(5), tokens].numpy() for temperature in TEMPERATURES
    ])
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_additional_config_temperatures_are_validated(monkeypatch):
    module = _load_path(monkeypatch, "_test_log_normalizers", PATCH_DIR / "log_normalizers.py")
    assert module.resolve_log_z_temperatures(None) is None
    assert module.resolve_log_z_temperatures({"subtree_cache_eviction": True}) is None
    assert module.resolve_log_z_temperatures({"log_z_temperatures": [0.5, 1]}) == (0.5, 1.0)
    with pytest.raises(ValueError, match="positive finite"):
        module.resolve_log_z_temperatures({"log_z_temperatures": [0.5, 0.0]})


# --------------------------------------------------------------------------------------------------------------------
# Engine transport: sampler -> model runner -> scheduler -> logprobs processor -> CompletionOutput
# --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("configured", [False, True])
def test_gpu_sampler_returns_log_z_from_unscaled_logits(monkeypatch, configured):
    module = _load_gpu_sampler(monkeypatch)
    sampler = module.Sampler.__new__(module.Sampler)
    sampler.compute_nans = False
    sampler.logprobs_mode = "processed_logprobs"
    sampler.sampling_states = SimpleNamespace(max_num_logprobs=lambda indices: 1)
    sampler.logprob_token_ids_state = SimpleNamespace(max_num_token_ids=lambda indices: 0)
    sampler.req_states = SimpleNamespace(prefill_len=SimpleNamespace(gpu=None))
    sampler.log_z_temperatures = tuple(TEMPERATURES) if configured else None

    def sample(logits, *args, **kwargs):
        logits.mul_(4.0)  # vLLM may scale logits in place for a low-temperature proposal.
        return logits.argmax(-1), logits

    sampler.sample = sample
    logits = torch.tensor([[0.5, 0.25, 0.125, 0.125], [0.25] * 4, [0.9, 0.1, 1e-6, 1e-6]]).log()
    batch = SimpleNamespace(
        expanded_idx_mapping=torch.tensor([0, 0, 1]), idx_mapping_np=np.array([0, 1]),
        cu_num_logits_np=np.array([0, 2, 3]), expanded_local_pos=torch.tensor([0, 1, 0]),
        positions=torch.arange(3), logits_indices=torch.arange(3), input_ids=torch.tensor([7, 8, 9]),
        seq_lens=torch.tensor([2, 1]), num_reqs=2, cu_num_logits=torch.tensor([0, 2, 3]),
        idx_mapping=torch.tensor([0, 1]),
    )
    output = sampler(logits.clone(), batch)

    if not configured:
        assert output.log_z_tensors is None
        return
    expected = torch.stack(
        [torch.logsumexp(logits.log_softmax(-1) / t, dim=-1) for t in TEMPERATURES], dim=-1,
    )
    torch.testing.assert_close(output.log_z_tensors, expected)


def test_model_runner_preserves_log_z(monkeypatch):
    class BaseRunner:
        def sample(self, *args):
            return self.test_sampler_output, None, None

        def sample_tokens(self, grammar):
            return self.test_result

    _stub_module(monkeypatch, PACKAGE, __path__=[str(PATCH_DIR)])
    _stub_module(monkeypatch, "vllm.v1.outputs", AsyncModelRunnerOutput=object, ModelRunnerOutput=SimpleNamespace)
    _stub_module(monkeypatch, "vllm.v1.worker.gpu.async_utils", AsyncOutput=type("AsyncOutput", (), {}))
    _stub_module(monkeypatch, "vllm.v1.worker.gpu.model_runner", GPUModelRunner=BaseRunner)
    _stub_module(monkeypatch, f"{PACKAGE}.outputs", ModelRunnerOutput=SimpleNamespace)
    _stub_module(monkeypatch, f"{PACKAGE}.sampler", Sampler=object)
    _stub_module(monkeypatch, f"{PACKAGE}.utils", patch=lambda *args: __import__("contextlib").nullcontext())
    module = _load_path(monkeypatch, f"{PACKAGE}.model_runner", PATCH_DIR / "model_runner.py")

    scores = _scores([-0.4, -0.5, -0.6], [0, 2, 3])
    rows = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    runner = module.GPUModelRunner.__new__(module.GPUModelRunner)
    runner.test_sampler_output = SimpleNamespace(
        power_logprobs_tensors=SimpleNamespace(tolists=lambda: scores),
        entropies_tensors=torch.tensor([0.7, 0.8, 0.9]), log_z_tensors=torch.tensor(rows),
    )
    runner.sample(None, None, None)
    runner.test_result = _base_output()
    actual = runner.sample_tokens(None)
    assert actual.log_z_by_temperature == rows
    assert runner._log_z_tensors is None


def test_scheduler_routes_log_z_rows_using_logprob_offsets(monkeypatch):
    module = _load_scheduler(monkeypatch)
    first, second = _Request(None), _Request(None)
    first.request_id, second.request_id = "a", "b"
    first.sampling_params = second.sampling_params = SimpleNamespace(logprobs=1)
    scheduler = _make_scheduler(module, first)
    scheduler.requests["b"] = second
    scheduler.running.append(second)
    scheduled = SimpleNamespace(num_scheduled_tokens={"b": 1, "a": 1}, scheduled_spec_decode_tokens={})
    runner_output = _base_output()
    runner_output.power_logprobs = runner_output.logprobs
    runner_output.entropies = [0.7, 0.8, 0.9]
    runner_output.log_z_by_temperature = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    outputs = scheduler.update_from_output(scheduled, runner_output)[0].outputs

    assert [output.request_id for output in outputs] == ["b", "a"]
    assert outputs[0].new_log_z_by_temperature == [[5.0, 6.0]]
    assert outputs[1].new_log_z_by_temperature == [[1.0, 2.0], [3.0, 4.0]]


@pytest.mark.parametrize("kind,tokens,expected", [
    ("cumulative", [9], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    ("delta", [8, 9], [[3.0, 4.0], [5.0, 6.0]]),
])
def test_completion_accumulates_and_slices_log_z_with_tokens(monkeypatch, kind, tokens, expected):
    logprobs, output_module = _load_processors(monkeypatch)
    processor = logprobs.LogprobsProcessor.from_new_request(
        None, SimpleNamespace(sampling_params=SimpleNamespace(logprobs=1, flat_logprobs=False)),
    )
    chunks = [([-0.1, -0.2], [[1.0, 2.0], [3.0, 4.0]]), ([-0.3, -0.4], [[5.0, 6.0], [7.0, 8.0]])]
    for scores, log_z in chunks:
        processor.update_from_output(SimpleNamespace(
            new_logprobs=_scores(scores), new_power_logprobs=_scores(scores),
            new_entropies=[0.0] * len(scores), new_log_z_by_temperature=log_z,
            new_prompt_logprobs_tensors=None,
        ))
    assert processor.log_z_by_temperature == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]

    state = output_module.RequestState()
    state.detokenizer = SimpleNamespace(output_token_ids=[7, 8, 9], get_next_output_text=lambda *args: "text")
    state.logprobs_processor = processor
    state.output_kind = kind
    state.request_index = 0
    completion = state._new_completion_output(tokens, "stop", None)
    assert completion.log_z_by_temperature == expected


def test_processor_without_log_z_keeps_none_and_rejects_gaps(monkeypatch):
    logprobs, _ = _load_processors(monkeypatch)
    processor = logprobs.LogprobsProcessor(logprobs=[], power_logprobs=[], entropies=[])
    update = dict(new_logprobs=_scores([-0.1]), new_power_logprobs=_scores([-0.2]),
                  new_entropies=[0.5], new_prompt_logprobs_tensors=None)
    processor.update_from_output(SimpleNamespace(**update))
    assert processor.log_z_by_temperature is None
    # A later chunk cannot start logZ rows after earlier positions were missed.
    with pytest.raises(RuntimeError, match="earlier positions"):
        processor.update_from_output(SimpleNamespace(**update, new_log_z_by_temperature=[[1.0]]))


# --------------------------------------------------------------------------------------------------------------------
# Sampler: v2 reproduces v1's MTM weights without scoring requests
# --------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def samplers(monkeypatch):
    """Load v1 and v2 with only the vLLM import surface stubbed."""
    vllm = types.ModuleType("vllm")
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = dict
    engine_name = "power_sharpening.backends.vllm.engine_patch"
    engine = types.ModuleType(engine_name)
    engine.__path__ = [str(PATCH_DIR)]
    engine.SamplingParams = SimpleNamespace
    params = types.ModuleType(engine_name + ".sampling_params")

    def clone(source, **overrides):
        result = copy.deepcopy(source)
        for key, value in overrides.items():
            setattr(result, key, value)
        return result

    params._copy_sampling_params = clone
    package_name = "power_sharpening.backends.vllm.samplers"
    package = types.ModuleType(package_name)
    package.__path__ = [str(SAMPLERS_DIR)]
    for name, module in [("vllm", vllm), ("vllm.inputs", inputs), (engine_name, engine),
                         (params.__name__, params), (package_name, package)]:
        monkeypatch.setitem(sys.modules, name, module)
    v1 = _load_path(monkeypatch, package_name + ".multi_try_mh", SAMPLERS_DIR / "multi_try_mh.py")
    v2 = _load_path(monkeypatch, package_name + ".multi_try_mh_v2", SAMPLERS_DIR / "multi_try_mh_v2.py")
    return v1, v2


def _distribution(prefix, temperature=1.0):
    logits = np.array([-1.8, -0.7, 0.1, -0.2, 0.9, 0.2, -0.1, -0.3])
    logits += ((np.arange(8) + sum(prefix)) % 3) * 0.2
    logits /= temperature
    return logits - np.logaddexp.reduce(logits)


class _Llm:
    """Deterministic fake engine; ``log_z_temperatures`` enables engine-cached normalizers."""

    def __init__(self, log_z_temperatures=None):
        self.log_z_temperatures = log_z_temperatures
        self.generations, self.scoring = [], []

    def generate(self, prompts, sampling_params, use_tqdm):
        assert len(prompts) == len(sampling_params)
        is_scoring = bool(getattr(sampling_params[0], "logprob_token_ids", None))
        (self.scoring if is_scoring else self.generations).append(len(prompts))
        outputs = []
        for prompt, sp in zip(prompts, sampling_params):
            assert (sp.top_k, sp.top_p, sp.min_p) == (-1, 1.0, 0.0) and sp.ignore_eos
            prefix = list(prompt["prompt_token_ids"])
            tokens, lps, base_lps, log_z = [], [], [], []
            for step in range(sp.max_tokens):
                # A token depending on the prefix and temperature, but not on RNG state shared with the sampler.
                token = (sp.logprob_token_ids[0] if is_scoring
                         else (3 * sum(prefix) + len(prefix) + int(sp.temperature * 10)) % 7 + 1)
                q, p = _distribution(prefix, sp.temperature), _distribution(prefix)
                lps.append({token: SimpleNamespace(logprob=q[token])})
                base_lps.append({token: SimpleNamespace(logprob=p[token])})
                if self.log_z_temperatures:
                    log_z.append([float(np.logaddexp.reduce(p / t)) for t in self.log_z_temperatures])
                tokens.append(token)
                prefix.append(token)
            completion = SimpleNamespace(
                token_ids=tokens, logprobs=lps, power_logprobs=base_lps,
                log_z_by_temperature=log_z if self.log_z_temperatures else None,
            )
            outputs.append(SimpleNamespace(outputs=[completion]))
        return outputs


class _Tokenizer:
    eos_token_id = 0

    def encode(self, prompt):
        return {"A": [7, 2], "B": [6]}[prompt]

    def decode(self, tokens, skip_special_tokens):
        return " ".join(map(str, tokens))


def _run(sampler_fn, v1_module, monkeypatch, llm, original, **kwargs):
    weights = []
    monkeypatch.setattr(v1_module, "_categorical_from_log_weights",
                        lambda values: weights.append(list(values)) or original(values))
    random.seed(3)
    np.random.seed(3)
    model = SimpleNamespace(tokenizer=_Tokenizer(), llm=llm, verbose=False,
                            log_z_temperatures=llm.log_z_temperatures)
    outputs = sampler_fn(model, ["A", "B"], SimpleNamespace(alpha=3.0, temperature=0.5, seed=11),
                         num_of_blocks=2, max_new_tokens=6, mcmc_steps=3, num_tries=3,
                         proposal_temperatures=TEMPERATURES, **kwargs)
    return outputs, weights


def test_v2_matches_v1_transitions_without_scoring_requests(samplers, monkeypatch):
    v1, v2 = samplers
    v1_llm, v2_llm = _Llm(), _Llm(log_z_temperatures=(1.2, 0.6, 0.25))  # Column order must not matter.
    choose = v1._categorical_from_log_weights
    expected, expected_weights = _run(v1.multi_try_mcmc_power_sampler, v1, monkeypatch, v1_llm, choose)
    actual, actual_weights = _run(v2.multi_try_mcmc_power_sampler_v2, v1, monkeypatch, v2_llm, choose)

    assert actual == expected
    assert len(actual_weights) == 2 * 2 * 3  # prompts * blocks * steps
    np.testing.assert_allclose(actual_weights, expected_weights, atol=1e-10)
    assert v1_llm.scoring and not v2_llm.scoring
    assert v2_llm.generations == v1_llm.generations  # Same batched generation calls.


def test_v2_draw_scores_every_component(samplers):
    _, v2 = samplers
    llm = _Llm(log_z_temperatures=tuple(TEMPERATURES))
    model = SimpleNamespace(llm=llm, log_z_temperatures=llm.log_z_temperatures)
    params = SimpleNamespace(alpha=3.0, temperature=0.5, seed=None, n=1, top_k=-1, top_p=1.0, min_p=0.0,
                             ignore_eos=True)
    tokens, base, components = v2._draw_proposals_v2(
        model, [[7, 2], [6]], [3, 2], TEMPERATURES, params, 128, None, row_temperatures=[0.6, 1.2],
    )
    for prefix, suffix, row_base, row in zip([[7, 2], [6]], tokens, base, components):
        assert row.shape == (len(TEMPERATURES), len(suffix))
        for component, temperature in enumerate(TEMPERATURES):
            context, expected = list(prefix), []
            for token in suffix:
                expected.append(_distribution(context, temperature)[token])
                context.append(token)
            np.testing.assert_allclose(row[component], expected, atol=1e-12)
    assert not llm.scoring


def test_v2_requires_engine_temperatures(samplers):
    _, v2 = samplers
    model = SimpleNamespace(tokenizer=_Tokenizer(), llm=_Llm(log_z_temperatures=(0.25, 1.2)),
                            log_z_temperatures=(0.25, 1.2))
    params = SimpleNamespace(alpha=3.0, temperature=0.5, seed=None)
    with pytest.raises(ValueError, match=r"\[0.6\]"):
        v2.multi_try_mcmc_power_sampler_v2(model, "A", params, proposal_temperatures=TEMPERATURES)
    with pytest.raises(ValueError, match="explicit proposal_temperatures"):
        v2.multi_try_mcmc_power_sampler_v2(model, "A", params, proposal_temperatures=None)
    model.log_z_temperatures = None
    with pytest.raises(ValueError, match="log_z_temperatures"):
        v2.multi_try_mcmc_power_sampler_v2(model, "A", params, proposal_temperatures=TEMPERATURES)
