
"""Extend vLLM sampling parameters for power sharpening.

Use through a vLLM runner, for example
``python -m power_sharpening.runners.vllm.run_power_mh --help``.
"""

from vllm.sampling_params import SamplingParams as BaseSamplingParams

from .utils import patch as mh_patch


class SamplingParams(BaseSamplingParams):
  """Sampling parameters with MH LLM extensions.

  Extends vLLM's SamplingParams to add MH LLM specific parameters.
  """

  alpha: float = 1.0
  """Controls the sharpening of the power distribution."""

  @staticmethod
  def from_optional(
      alpha: float | None = None,
      **kwargs,
  ) -> "SamplingParams":
    """Create SamplingParams from optional parameters.

    Args:
      alpha: Controls the sharpening of the power distribution.
      **kwargs: Other SamplingParams parameters.
    """
    with mh_patch({'module': 'vllm.sampling_params', 'class': SamplingParams}):
      params = BaseSamplingParams.from_optional(**kwargs)

    params.alpha = alpha = 1.0 if alpha is None else alpha
    return params

  def _verify_args(self) -> None:
    """Verify the arguments."""
    super()._verify_args()
    if self.alpha < 0.0:
      raise ValueError(f'alpha must be >= 0.0, got {self.alpha}')

  def __repr__(self) -> str:
    """Return vLLM's version-compatible representation plus ``alpha``."""
    base_repr = super().__repr__()
    if base_repr.endswith(")"):
      return f"{base_repr[:-1]}, alpha={self.alpha})"
    return f"{base_repr} (alpha={self.alpha})"



def _copy_sampling_params(
    sampling_params: SamplingParams,
    **kwargs,
) -> SamplingParams:
    """Create a deep copy of the given SamplingParams.

    Args:
        sampling_params (SamplingParams): The sampling parameters to copy.
        **kwargs: keywords to override in the copied SamplingParams.

    Returns:
        SamplingParams: A deep copy of the given sampling parameters.
    """
    new_params = sampling_params.clone()
    for key, value in kwargs.items():
        setattr(new_params, key, value)

    return new_params
