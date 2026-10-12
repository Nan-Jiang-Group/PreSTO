import os

from vllm.v1.worker.gpu_worker import Worker as GPUWorker
from .model_runner import GPUModelRunner


class Worker(GPUWorker):

  def init_device(self):
    super().init_device()
    verbose = os.environ.get('MH_LLM_VERBOSE', '0') == '1'
    # Replace the base model runner with our custom one that supports
    # power logprobs.
    self.model_runner = GPUModelRunner(
        self.vllm_config, self.device, verbose=verbose)
