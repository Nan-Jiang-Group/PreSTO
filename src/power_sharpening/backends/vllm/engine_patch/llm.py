from pathlib import Path
from typing import Any
import importlib
import os

from vllm.config import (
    AttentionConfig,
    CompilationConfig,
    PoolerConfig,
    ProfilerConfig,
    StructuredOutputsConfig,
)
from vllm.config.model import (
    ModelDType,
    TokenizerMode,
)
from vllm.engine.arg_utils import (
    ConvertOption,
    HfOverrides,
    RunnerOption,
)
# yapf: enable
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.v1.sample.logits_processor import LogitsProcessor

from vllm.entrypoints.llm import LLM as BaseLLM, logger

from .outputs import EngineCoreOutput, EngineCoreOutputs
from .logprobs import LogprobsProcessor
from .output_processor import OutputProcessor
from .scheduler import Scheduler
from .worker import Worker

from .utils import patch as mh_patch


def _patch_if_available(module: str, cls: type, name: str | None = None):
  """Return a vLLM monkey-patch spec only when the target exists."""
  try:
    imported = importlib.import_module(module)
  except ImportError:
    return None
  attr_name = name or cls.__name__
  if not hasattr(imported, attr_name):
    return None
  return {
      'module': module,
      'class': cls,
      **({'name': name} if name is not None else {}),
  }


def _runtime_patches():
  """Patches needed while vLLM builds and decodes custom outputs."""
  patches = [
      _patch_if_available(
          'vllm.v1.engine.output_processor',
          LogprobsProcessor,
      ),
      _patch_if_available(
          'vllm.v1.engine.core_client',
          EngineCoreOutput,
      ),
      _patch_if_available(
          'vllm.v1.engine.core_client',
          EngineCoreOutputs,
      ),
      _patch_if_available(
          'vllm.v1.engine',
          EngineCoreOutput,
      ),
      _patch_if_available(
          'vllm.v1.engine',
          EngineCoreOutputs,
      ),
      _patch_if_available(
          'vllm.v1.engine.core',
          EngineCoreOutput,
      ),
      _patch_if_available(
          'vllm.v1.engine.core',
          EngineCoreOutputs,
      ),
      _patch_if_available(
          'vllm.v1.engine.llm_engine',
          OutputProcessor,
      ),
  ]
  return [patch for patch in patches if patch is not None]


class LLM(BaseLLM):

  def __init__(
      self,
      model: str,
      *,
      runner: RunnerOption = "auto",
      convert: ConvertOption = "auto",
      tokenizer: str | None = None,
      tokenizer_mode: TokenizerMode | str = "auto",
      skip_tokenizer_init: bool = False,
      trust_remote_code: bool = False,
      allowed_local_media_path: str = "",
      allowed_media_domains: list[str] | None = None,
      tensor_parallel_size: int = 1,
      dtype: ModelDType = "auto",
      quantization: QuantizationMethods | None = None,
      revision: str | None = None,
      tokenizer_revision: str | None = None,
      chat_template: Path | str | None = None,
      seed: int = 0,
      gpu_memory_utilization: float = 0.9,
      cpu_offload_gb: float = 0,
      offload_group_size: int = 0,
      offload_num_in_group: int = 1,
      offload_prefetch_step: int = 1,
      offload_params: set[str] | None = None,
      enforce_eager: bool = False,
      enable_return_routed_experts: bool = False,
      disable_custom_all_reduce: bool = False,
      hf_token: bool | str | None = None,
      hf_overrides: HfOverrides | None = None,
      mm_processor_kwargs: dict[str, Any] | None = None,
      pooler_config: PoolerConfig | None = None,
      structured_outputs_config: dict[str, Any] | StructuredOutputsConfig |
      None = None,
      profiler_config: dict[str, Any] | ProfilerConfig | None = None,
      attention_config: dict[str, Any] | AttentionConfig | None = None,
      kv_cache_memory_bytes: int | None = None,
      compilation_config: int | dict[str, Any] | CompilationConfig |
      None = None,
      logits_processors: list[str | type[LogitsProcessor]] | None = None,
      **kwargs: Any,
  ) -> None:
    """LLM constructor."""
    # The custom worker installs the v2 GPU model runner, whose
    # add_requests() path requires SchedulerOutput.NewRequestData to include
    # prefill_token_ids. Force vLLM's scheduler into the matching output mode.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    kwargs[
        'worker_cls'] = f'{Worker.__module__}.{Worker.__name__}'  # use our own worker
    logger.info('Using worker_cls: %s', kwargs['worker_cls'])
    kwargs['logprobs_mode'] = 'processed_logprobs'  # force processed logprobs
    kwargs['scheduler_cls'] = Scheduler  # use our own scheduler

    with mh_patch(_runtime_patches()):
      super().__init__(
          model=model,
          runner=runner,
          convert=convert,
          tokenizer=tokenizer,
          tokenizer_mode=tokenizer_mode,
          skip_tokenizer_init=skip_tokenizer_init,
          trust_remote_code=trust_remote_code,
          allowed_local_media_path=allowed_local_media_path,
          allowed_media_domains=allowed_media_domains,
          tensor_parallel_size=tensor_parallel_size,
          dtype=dtype,
          quantization=quantization,
          revision=revision,
          tokenizer_revision=tokenizer_revision,
          chat_template=chat_template,
          seed=seed,
          gpu_memory_utilization=gpu_memory_utilization,
          cpu_offload_gb=cpu_offload_gb,
          offload_group_size=offload_group_size,
          offload_num_in_group=offload_num_in_group,
          offload_prefetch_step=offload_prefetch_step,
          offload_params=offload_params,
          enforce_eager=enforce_eager,
          enable_return_routed_experts=enable_return_routed_experts,
          disable_custom_all_reduce=disable_custom_all_reduce,
          hf_token=hf_token,
          hf_overrides=hf_overrides,
          mm_processor_kwargs=mm_processor_kwargs,
          pooler_config=pooler_config,
          structured_outputs_config=structured_outputs_config,
          profiler_config=profiler_config,
          attention_config=attention_config,
          kv_cache_memory_bytes=kv_cache_memory_bytes,
          compilation_config=compilation_config,
          logits_processors=logits_processors,
          **kwargs,
      )

  def generate(self, *args: Any, **kwargs: Any):
    """Generate while keeping custom EngineCore wire structs patched.

    MRV2 decodes async EngineCore outputs during ``generate()``, not only
    during construction, so the custom output classes must be visible for the
    whole call.
    """
    with mh_patch(_runtime_patches()):
      return super().generate(*args, **kwargs)
