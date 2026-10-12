"""Thread custom MH log-probabilities, entropies, and logZ_T through vLLM's model runner.

Run through a custom-vLLM entry point such as:
    python -m power_sharpening.runners.vllm.run_subtree_prefetching_mh --help
"""

import logging
from typing import TYPE_CHECKING

from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    ModelRunnerOutput as BaseModelRunnerOutput,
)
from vllm.v1.worker.gpu.async_utils import AsyncOutput as BaseAsyncOutput
from vllm.v1.worker.gpu.model_runner import (
    GPUModelRunner as BaseGPUModelRunner,
)

from .log_normalizers import resolve_log_z_temperatures
from .outputs import ModelRunnerOutput
from .sampler import Sampler
from .utils import patch as mh_patch

logger = logging.getLogger("vllm_model_runner")

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput


def _to_list(tensor):
    return tensor.tolist() if tensor is not None else None


def _attach_power_logprobs(
    base_output: BaseModelRunnerOutput,
    power_logprobs,
    entropies=None,
    log_z_by_temperature=None,
) -> ModelRunnerOutput:
    """Return a ModelRunnerOutput carrying the custom MH metadata."""
    return ModelRunnerOutput(
        req_ids=base_output.req_ids,
        req_id_to_index=base_output.req_id_to_index,
        sampled_token_ids=base_output.sampled_token_ids,
        logprobs=base_output.logprobs,
        prompt_logprobs_dict=base_output.prompt_logprobs_dict,
        pooler_output=base_output.pooler_output,
        kv_connector_output=base_output.kv_connector_output,
        ec_connector_output=getattr(base_output, "ec_connector_output", None),
        num_nans_in_logits=base_output.num_nans_in_logits,
        cudagraph_stats=base_output.cudagraph_stats,
        routed_experts=getattr(base_output, "routed_experts", None),
        power_logprobs=power_logprobs,
        entropies=entropies,
        log_z_by_temperature=log_z_by_temperature,
    )


class PowerAsyncOutput(BaseAsyncOutput):
    """AsyncOutput variant that preserves custom MH tensors."""

    def get_output(self) -> ModelRunnerOutput:
        base_output = super().get_output()
        power_logprobs_tensors = getattr(
            self.sampler_output, "power_logprobs_tensors", None
        )
        power_logprobs = (
            power_logprobs_tensors.tolists()
            if power_logprobs_tensors is not None
            else None
        )
        entropies_tensors = getattr(self.sampler_output, "entropies_tensors", None)
        entropies = entropies_tensors.tolist() if entropies_tensors is not None else None
        log_z = _to_list(getattr(self.sampler_output, "log_z_tensors", None))
        return _attach_power_logprobs(base_output, power_logprobs, entropies, log_z)


class GPUModelRunner(BaseGPUModelRunner):

    def __init__(self, vllm_config, device, verbose: bool = False):
        self.verbose = verbose
        super().__init__(vllm_config, device)
        # vLLM <= 0.26 builds the sampler in ``__init__``; newer builds defer it
        # to ``load_model``. Swap in ours wherever it already exists.
        self._install_power_sampler()
        self._power_logprobs_tensors = None
        self._entropies_tensors = None
        self._log_z_tensors = None

    def load_model(self, *args, **kwargs) -> None:
        super().load_model(*args, **kwargs)
        # vLLM 0.27+ creates ``self.sampler`` here, after ``__init__`` has run.
        self._install_power_sampler()

    def _install_power_sampler(self) -> None:
        """Replace vLLM's sampler with the one that emits power logprobs.

        The custom sampler keeps the pre-temperature (base-model) logprobs and
        the full-vocabulary entropies that MH acceptance needs.
        """
        base_sampler = getattr(self, "sampler", None)
        if base_sampler is None or isinstance(base_sampler, Sampler):
            return

        kwargs = {}
        if hasattr(self.model_config, "use_fp64_gumbel"):
            kwargs["use_fp64_gumbel"] = self.model_config.use_fp64_gumbel
        self.sampler = Sampler(
            max_num_reqs=self.max_num_reqs,
            vocab_size=self.vocab_size,
            device=self.device,
            req_states=self.req_states,
            logprobs_mode=self.model_config.logprobs_mode,
            num_speculative_tokens=getattr(
                base_sampler,
                "num_speculative_tokens",
                self.num_speculative_steps + 1,
            ),
            verbose=self.verbose,
            log_z_temperatures=resolve_log_z_temperatures(
                getattr(self.vllm_config, "additional_config", None)
            ),
            **kwargs,
        )
        # Spec decoding verifies through the rejection sampler's inner sampler.
        if getattr(self, "rejection_sampler", None) is not None \
                and getattr(self.rejection_sampler, "sampler", None) is base_sampler:
            self.rejection_sampler.sampler = self.sampler

    def sample(self, hidden_states, input_batch, grammar_output):
        sampler_output, num_sampled, num_rejected = super().sample(
            hidden_states,
            input_batch,
            grammar_output,
        )
        self._power_logprobs_tensors = getattr(
            sampler_output, "power_logprobs_tensors", None
        )
        self._entropies_tensors = getattr(sampler_output, "entropies_tensors", None)
        self._log_z_tensors = getattr(sampler_output, "log_z_tensors", None)
        return sampler_output, num_sampled, num_rejected

    def sample_tokens(
        self,
        grammar_output: "GrammarOutput | None",
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        with mh_patch([{
            "module": "vllm.v1.worker.gpu.model_runner",
            "class": PowerAsyncOutput,
            "name": "AsyncOutput",
        }]):
            result = super().sample_tokens(grammar_output)

        if result is None:
            self._power_logprobs_tensors = None
            self._entropies_tensors = None
            self._log_z_tensors = None
            return result
        if isinstance(result, PowerAsyncOutput):
            self._power_logprobs_tensors = None
            self._entropies_tensors = None
            self._log_z_tensors = None
            return result.get_output()
        if isinstance(result, BaseAsyncOutput):
            power_logprobs_tensors = getattr(
                result.sampler_output, "power_logprobs_tensors", None
            )
            if power_logprobs_tensors is None:
                power_logprobs_tensors = self._power_logprobs_tensors
            power_logprobs = (
                power_logprobs_tensors.tolists()
                if power_logprobs_tensors is not None
                else None
            )
            base_output = result.get_output()
            entropies_tensors = getattr(result.sampler_output, "entropies_tensors", None)
            if entropies_tensors is None:
                entropies_tensors = self._entropies_tensors
            entropies = entropies_tensors.tolist() if entropies_tensors is not None else None
            log_z_tensors = getattr(result.sampler_output, "log_z_tensors", None)
            if log_z_tensors is None:
                log_z_tensors = self._log_z_tensors
            self._power_logprobs_tensors = None
            self._entropies_tensors = None
            self._log_z_tensors = None
            return _attach_power_logprobs(
                base_output,
                power_logprobs,
                entropies,
                _to_list(log_z_tensors),
            )

        # Determine the base ModelRunnerOutput to wrap.
        if hasattr(result, "model_runner_output"):
            base_output = result.model_runner_output
        elif hasattr(result, "_model_runner_output"):
            base_output = result._model_runner_output
        elif isinstance(result, BaseModelRunnerOutput):
            base_output = result
        else:
            # IntermediateTensors or other types — pass through.
            return result

        power_logprobs = getattr(base_output, "power_logprobs", None)
        if power_logprobs is None and self._power_logprobs_tensors is not None:
            power_logprobs = self._power_logprobs_tensors.tolists()
        entropies = getattr(base_output, "entropies", None)
        if entropies is None and self._entropies_tensors is not None:
            entropies = self._entropies_tensors.tolist()
        log_z = getattr(base_output, "log_z_by_temperature", None)
        if log_z is None:
            log_z = _to_list(self._log_z_tensors)
        self._power_logprobs_tensors = None
        self._entropies_tensors = None
        self._log_z_tensors = None

        custom_output = _attach_power_logprobs(
            base_output,
            power_logprobs,
            entropies,
            log_z,
        )

        logger.debug("sample_tokens: power_logprobs=%s, "
                     "custom_output.power_logprobs=%s",
                     "SET" if power_logprobs else None,
                     "SET" if custom_output.power_logprobs else None)

        if hasattr(result, "model_runner_output"):
            result.model_runner_output = custom_output
            return result
        if hasattr(result, "_model_runner_output"):
            result._model_runner_output = custom_output
            return result

        return custom_output
