"""Per-temperature log normalizers for exact proposal scoring during generation.

Enable through ``vLLM_Wrapper(engine_type="custom", log_z_temperatures=[0.25, 0.5, 1.0])``; see
``power_sharpening.backends.vllm.samplers.multi_try_mh_v2``. Run the CPU checks with:
    python -m pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_multi_try_mh_v2.py

Following the REPPS exact-stats cache, the sampler stores logZ_T = logsumexp(log p / T) over the full vocabulary for
every configured T at each generated position. For an untruncated, unpenalized proposal softmax(logits / T),

    log q_T(y) = log p(y) / T - logZ_T,

so a token's proposal score under every configured temperature follows from its base log probability without another
engine request.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

# ``vllm_config.additional_config`` key naming the temperatures whose normalizers the sampler returns.
LOG_Z_TEMPERATURES_KEY = "log_z_temperatures"


def resolve_log_z_temperatures(additional_config) -> tuple[float, ...] | None:
  """Read validated log-normalizer temperatures from vLLM's ``additional_config``."""
  if not isinstance(additional_config, Mapping):
    return None
  temperatures = additional_config.get(LOG_Z_TEMPERATURES_KEY)
  if not temperatures:
    return None
  temperatures = tuple(float(value) for value in temperatures)
  if any(not np.isfinite(value) or value <= 0 for value in temperatures):
    raise ValueError(f"{LOG_Z_TEMPERATURES_KEY} must be positive finite numbers, got {temperatures}")
  return temperatures


def temperature_log_normalizers(
    logits: torch.Tensor,
    temperatures: Sequence[float],
) -> torch.Tensor:
  """Return logsumexp(log p / T) for each decoding position and temperature.

  Args:
    logits: Base-model logits of shape ``[num_positions, vocab_size]``, before temperature scaling or sampling
      processors.
    temperatures: Proposal temperatures T_1..T_M.

  Returns:
    Float32 tensor of shape ``[num_positions, M]`` on the input device.
  """
  base_logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
  return torch.stack(
      [torch.logsumexp(base_logprobs / float(temperature), dim=-1) for temperature in temperatures],
      dim=-1,
  )


def tempered_logprobs(
    base_logprobs: Sequence[float],
    log_z_by_temperature: Sequence[Sequence[float]],
    temperatures: Sequence[float],
) -> np.ndarray:
  """Return per-token log q_T(y) with shape ``(M, L)`` from engine-cached normalizers.

  ``base_logprobs[t]`` is log p(y_t); ``log_z_by_temperature[t][m]`` is logZ_{T_m} at position t.
  """
  base = np.asarray(base_logprobs, dtype=np.float64)
  log_z = np.asarray(log_z_by_temperature, dtype=np.float64).reshape(len(base), len(temperatures))
  inverse = 1.0 / np.asarray(temperatures, dtype=np.float64)
  return inverse[:, None] * base[None, :] - log_z.T


__all__ = [
    "LOG_Z_TEMPERATURES_KEY",
    "resolve_log_z_temperatures",
    "temperature_log_normalizers",
    "tempered_logprobs",
]
